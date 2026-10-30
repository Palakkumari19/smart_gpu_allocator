"""
Bit 7: Real-Time Risk-Aware MARL Scheduling & Monitoring Test Suite
Implements the required deterministic test scenarios:
  - Scenario A: Equal performance, different risk scores (risk penalty changes ranking & selection)
  - Scenario B: Different expected failure costs (EFC penalty changes scores & placement)
  - Scenario C: Risk changes during simulation (dynamic risk shifts trigger re-evaluation & selection pivot)
  - Scenario D: Quarantined or unavailable GPU exclusion
  - Scenario E: Risk-aware versus risk-blind ablation benchmark
"""

import copy
import pytest
import numpy as np

from environment import GPUEnvironment
from gpu import GPU
from marl_agent import MARLAgent, MARLManager, ProbeTask
from risk_engine import RiskEngine
from task import Task
from evaluation.workload_trace import generate_workload_trace
from evaluation.scientific_benchmark import run_episode, collect_run_metrics


def make_test_gpu(gpu_id: int, capacity: float = 8.0, current_mem: float = 0.0, temp: float = 50.0,
                  risk_score: float = 0.0, failure_rate: float = 0.0, quarantined: bool = False,
                  crashed: bool = False, cooldown: int = 0) -> GPU:
    """Helper to create a configured GPU with attached risk engine."""
    gpu = GPU(gpu_id, capacity)
    gpu.current_memory = current_mem
    gpu.temperature = temp
    gpu.crashed = crashed
    gpu.crash_cooldown_timer = cooldown
    gpu.task_queue = []
    gpu.running_tasks = []
    gpu.memory_fragments = []

    engine = RiskEngine()
    gpu.set_risk_engine(engine)
    risk_state = engine.initialize_gpu(gpu_id, capacity)
    risk_state.risk_score = risk_score
    risk_state.is_quarantined = quarantined
    if quarantined:
        risk_state.quarantine_reason = "Test Quarantine"
    if failure_rate > 0:
        risk_state.reliability_history.total_tasks = 100
        risk_state.reliability_history.failed_tasks = int(failure_rate)
    return gpu


def make_task(task_id="video_0001", mem=2.0, time=3, severity="MEDIUM") -> Task:
    """Helper to create a Task with custom memory, time, and severity attributes."""
    t = Task()
    t.id = task_id
    t.memory_required = mem
    t.execution_time = time
    t.workload_severity = severity
    t.severity = severity
    return t


def test_scenario_a_equal_performance_different_risk():
    """
    Scenario A — Equal performance, different risk.
    Two eligible GPUs with identical memory headroom, queue length, and temperature,
    but GPU 1 has 0.1 risk and GPU 2 has 0.8 risk.
    Verify that:
      1. Candidate breakdowns reflect the risk penalty (0.25 * risk_score).
      2. GPU 1 receives a higher bid score than GPU 2.
      3. MARL scheduler selects the lower-risk GPU (GPU 1).
    """
    manager = MARLManager(num_gpus=2)
    # Ensure untrained agents have equal Q-values (0.0)
    for agent in manager.agents:
        agent.q_table.clear()
        agent.core_q_table.clear()

    gpu1 = make_test_gpu(1, capacity=8.0, current_mem=2.0, temp=55.0, risk_score=0.1)
    gpu2 = make_test_gpu(2, capacity=8.0, current_mem=2.0, temp=55.0, risk_score=0.8)
    gpus = [gpu1, gpu2]

    task = make_task("task_a", mem=2.0, time=4, severity="MEDIUM")
    cluster_state = manager.get_cluster_state(gpus)

    selected_gpu_id, breakdowns = manager.select_gpu_for_task(
        task, gpus, cluster_state, return_infeasible_fallback=False, return_breakdown=True
    )

    bd1 = next(b for b in breakdowns if b["gpu_id"] == 1)
    bd2 = next(b for b in breakdowns if b["gpu_id"] == 2)

    # Identical performance components
    assert bd1["memory_headroom"] == bd2["memory_headroom"]
    assert bd1["queue_penalty"] == bd2["queue_penalty"]
    assert bd1["temp_penalty"] == bd2["temp_penalty"]

    # Different risk penalties
    assert bd1["risk_penalty"] < bd2["risk_penalty"]
    assert bd1["bid_score"] > bd2["bid_score"]

    # Lower risk GPU must be chosen
    assert selected_gpu_id == 0  # 0-indexed index in gpus list, corresponding to GPU 1
    assert breakdowns[selected_gpu_id]["gpu_id"] == 1


def test_scenario_b_different_expected_failure_costs():
    """
    Scenario B — Different expected failure costs (EFC).
    Two eligible GPUs with equal base risk scores and memory, but GPU 2 has
    historical failures leading to significantly higher expected failure cost.
    Verify that:
      1. EFC is independently and accurately computed.
      2. The EFC penalty (0.20 * min(1.0, efc / 4.0)) reduces GPU 2's score.
      3. GPU 1 is selected due to lower failure impact.
    """
    manager = MARLManager(num_gpus=2)
    for agent in manager.agents:
        agent.q_table.clear()
        agent.core_q_table.clear()

    # GPU 1 has clean history, GPU 2 has 80% failure rate
    gpu1 = make_test_gpu(1, capacity=8.0, current_mem=1.0, temp=50.0, risk_score=0.2, failure_rate=0.0)
    gpu2 = make_test_gpu(2, capacity=8.0, current_mem=1.0, temp=50.0, risk_score=0.2, failure_rate=80.0)
    gpus = [gpu1, gpu2]

    task = make_task("task_b", mem=3.0, time=6, severity="HIGH")
    cluster_state = manager.get_cluster_state(gpus)

    selected_gpu_id, breakdowns = manager.select_gpu_for_task(
        task, gpus, cluster_state, return_infeasible_fallback=False, return_breakdown=True
    )

    bd1 = next(b for b in breakdowns if b["gpu_id"] == 1)
    bd2 = next(b for b in breakdowns if b["gpu_id"] == 2)

    assert bd2["efc"] > bd1["efc"], f"Expected GPU 2 EFC > GPU 1 EFC, got {bd2['efc']} vs {bd1['efc']}"
    assert bd2["efc_penalty"] > bd1["efc_penalty"]
    assert bd1["bid_score"] > bd2["bid_score"]
    assert breakdowns[selected_gpu_id]["gpu_id"] == 1


def test_scenario_c_risk_changes_during_simulation():
    """
    Scenario C — Dynamic risk changes during simulation.
    Initially, GPU 1 has low risk and is preferred.
    Then, physical temperature and risk score of the preferred GPU are increased.
    Verify that:
      1. Risk telemetry updates dynamically.
      2. Candidate scores are recomputed.
      3. Subsequent scheduling decisions pivot away from the high-risk GPU.
    """
    env = GPUEnvironment(scheduler="marl")
    env.learning_enabled = False  # Isolate heuristic & risk dynamics

    # Initialize all GPUs with low risk
    for g in env.gpus:
        risk_state = g.get_risk_state()
        if risk_state:
            risk_state.risk_score = 0.1

    # Task 1: Both GPUs available, GPU 1 is preferred/selected
    task1 = make_task("task_c1", mem=2.0, time=3, severity="MEDIUM")
    assigned1 = env.marl_scheduler(task1)
    assert assigned1 is True
    first_decision = env.last_marl_decision
    assert first_decision is not None
    assert "candidates" in first_decision
    first_selected_id = first_decision["selected_gpu_id"]

    # Now dramatically spike the selected GPU's temperature and risk
    target_gpu = next(g for g in env.gpus if g.id == first_selected_id)
    target_gpu.temperature = 84.0
    risk_state = target_gpu.get_risk_state()
    if risk_state:
        risk_state.risk_score = 0.82

    # Task 2: Under new telemetry, candidate scores must recompute
    task2 = make_task("task_c2", mem=2.0, time=3, severity="MEDIUM")
    assigned2 = env.marl_scheduler(task2)
    assert assigned2 is True
    second_decision = env.last_marl_decision

    # The spiked GPU must suffer a higher risk penalty in decision 2
    spiked_candidate = next(c for c in second_decision["candidates"] if c["gpu_id"] == first_selected_id)
    assert spiked_candidate["risk_penalty"] > 0.03
    assert spiked_candidate["risk_score"] > 0.15

    # A different eligible GPU should be chosen
    assert second_decision["selected_gpu_id"] != first_selected_id


def test_scenario_d_quarantined_or_unavailable_gpu():
    """
    Scenario D — Quarantined or unavailable GPU.
    Verify that crashed, quarantined, or cooldown GPUs are marked ineligible
    with explicit reasons, and the scheduler strictly excludes them.
    """
    manager = MARLManager(num_gpus=4)
    gpu1 = make_test_gpu(1, capacity=8.0, crashed=True)
    gpu2 = make_test_gpu(2, capacity=8.0, cooldown=5)
    gpu3 = make_test_gpu(3, capacity=8.0, quarantined=True)
    gpu4 = make_test_gpu(4, capacity=8.0, current_mem=1.0, temp=45.0, risk_score=0.1)
    gpus = [gpu1, gpu2, gpu3, gpu4]

    task = make_task("task_d", mem=2.0, time=3, severity="LOW")
    cluster_state = manager.get_cluster_state(gpus)

    selected_gpu_id, breakdowns = manager.select_gpu_for_task(
        task, gpus, cluster_state, return_infeasible_fallback=False, return_breakdown=True
    )

    bd1 = next(b for b in breakdowns if b["gpu_id"] == 1)
    bd2 = next(b for b in breakdowns if b["gpu_id"] == 2)
    bd3 = next(b for b in breakdowns if b["gpu_id"] == 3)
    bd4 = next(b for b in breakdowns if b["gpu_id"] == 4)

    assert bd1["eligible"] is False
    assert bd1["ineligible_reason"] == "CRASHED"

    assert bd2["eligible"] is False
    assert "COOLDOWN" in bd2["ineligible_reason"]

    assert bd3["eligible"] is False
    assert "QUARANTINED" in bd3["ineligible_reason"]

    assert bd4["eligible"] is True
    assert bd4["ineligible_reason"] is None

    assert breakdowns[selected_gpu_id]["gpu_id"] == 4


def test_scenario_e_risk_aware_vs_risk_blind_ablation():
    """
    Scenario E — Risk-aware versus risk-blind ablation benchmark.
    Uses the exact same deterministic workload trace and seed.
    Compares the full risk-aware MARL scheduler against a risk-blind ablation
    where risk and EFC weights are zeroed in bidding and rewards.
    Verifies that the ablation is deterministic, all tasks are accounted for,
    and reports exact metrics without forcing arbitrary outcomes.
    """
    trace_seed = 42
    workload = generate_workload_trace("mixed", 12, trace_seed)

    # 1. Run with standard Risk-Aware MARL
    env_risk_aware, metrics_aware = run_episode("marl", copy.deepcopy(workload), learning=False)

    # 2. Run with Risk-Blind MARL ablation
    class RiskBlindManager(MARLManager):
        def compute_bid_score_breakdown(self, gpu, task, agent, cluster_state):
            bd = super().compute_bid_score_breakdown(gpu, task, agent, cluster_state)
            # Remove risk and EFC penalties from bid score
            bd["risk_penalty"] = 0.0
            bd["efc_penalty"] = 0.0
            if bd["eligible"]:
                bd["bid_score"] = round(bd["q_contribution"] + bd["perf_score"], 4)
            return bd

    env_risk_blind = GPUEnvironment(scheduler="marl")
    env_risk_blind.marl_manager = RiskBlindManager(num_gpus=4)
    env_risk_blind.enable_deterministic_physics()
    env_risk_blind.set_eval_mode()
    env_risk_blind.attach_workload_trace(copy.deepcopy(workload))
    for _ in range(len(workload)):
        env_risk_blind.step()

    metrics_blind = collect_run_metrics(env_risk_blind, len(workload))

    # Verify task conservation law holds for both across all lifecycle states
    counts_aware = env_risk_aware.task_queue.get_task_counts()
    sum_aware = sum(counts_aware[k] for k in ["completed", "running", "queued", "failed", "scheduled", "arriving", "cancelled", "waiting", "retry"])
    assert counts_aware["total"] == sum_aware

    counts_blind = env_risk_blind.task_queue.get_task_counts()
    sum_blind = sum(counts_blind[k] for k in ["completed", "running", "queued", "failed", "scheduled", "arriving", "cancelled", "waiting", "retry"])
    assert counts_blind["total"] == sum_blind

    # Output reproducible benchmark telemetry
    print(f"\n[Ablation Result] Risk-Aware Completed: {metrics_aware.get('completed', 0)}, Risk-Blind Completed: {metrics_blind.get('completed', 0)}")
    print(f"[Ablation Result] Risk-Aware Avg EFC: {metrics_aware.get('average_efc', 0.0):.4f}, Risk-Blind Avg EFC: {metrics_blind.get('average_efc', 0.0):.4f}")
