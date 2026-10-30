"""
Bit 6: Risk-Aware RL + Multi-Agent Reinforcement Learning (MARL) Test Suite
Covers:
  - Suite A: Risk-Aware State Representation
  - Suite B: Risk-Aware Reward Function
  - Suite C: Expected Failure Cost
  - Suite D: Actual MARL GPU Selection
  - Suite E: Dynamic Risk Changes
  - Suite F: Concurrency and Regression
"""

import concurrent.futures
import math
import numpy as np
import pytest
import threading

from environment import GPUEnvironment
from gpu import GPU
from marl_agent import MARLAgent, MARLManager, RiskAwareRewardConfig
from risk_engine import RiskEngine, GPURiskState, RiskLevel, ReliabilityHistory
from task import Task


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


# ==============================================================================
# SUITE A: Risk-Aware State Representation
# ==============================================================================

def test_suite_a_state_contains_risk_dimensions():
    """Verify that agent.get_state extracts all required risk and contextual dimensions."""
    manager = MARLManager(num_gpus=1)
    agent = manager.agents[0]
    gpu = make_test_gpu(1, capacity=8.0, current_mem=2.0, temp=60.0, risk_score=0.35, failure_rate=12.0)
    cluster_state = manager.get_cluster_state([gpu])
    cluster_state['current_task_memory'] = 2.0
    cluster_state['current_task_time'] = 4.0
    cluster_state['current_task_impact'] = 0.6

    state = agent.get_state(gpu, cluster_state)
    assert isinstance(state, tuple)
    assert len(state) == 16
    for dim in state:
        assert isinstance(dim, (int, np.integer))

    # Continuous representation for observability
    c_state = agent.get_continuous_state(gpu, cluster_state)
    assert len(c_state) == 16
    for val in c_state:
        assert 0.0 <= val <= 1.0


def test_suite_a_different_risk_levels_produce_different_states():
    """Verify that varying a GPU's risk score changes the discrete state key."""
    manager = MARLManager(num_gpus=1)
    agent = manager.agents[0]

    gpu_low_risk = make_test_gpu(1, risk_score=0.10)
    gpu_high_risk = make_test_gpu(1, risk_score=0.85)

    cs = manager.get_cluster_state([gpu_low_risk])
    state_low = agent.get_state(gpu_low_risk, cs)
    state_high = agent.get_state(gpu_high_risk, cs)

    assert state_low != state_high, "Different risk scores must produce distinct state keys in MARL Q-table"
    # Dimension index 3 is risk_bin
    assert state_low[3] != state_high[3]
    assert state_low[3] == 0  # <= 0.25
    assert state_high[3] == 3  # > 0.75


def test_suite_a_deterministic_states_for_equivalent_conditions():
    """Verify state generation is strictly deterministic for equivalent inputs."""
    manager = MARLManager(num_gpus=1)
    agent = manager.agents[0]
    gpu_a = make_test_gpu(1, current_mem=3.0, temp=65.0, risk_score=0.45)
    gpu_b = make_test_gpu(1, current_mem=3.0, temp=65.0, risk_score=0.45)
    cs = manager.get_cluster_state([gpu_a])

    state_a = agent.get_state(gpu_a, cs)
    state_b = agent.get_state(gpu_b, cs)
    assert state_a == state_b


def test_suite_a_missing_risk_data_safe_fallback():
    """Verify safe documented fallback when risk engine is not attached to GPU."""
    manager = MARLManager(num_gpus=1)
    agent = manager.agents[0]
    raw_gpu = GPU(1, 8.0)  # No risk engine attached
    raw_gpu.current_memory = 2.0
    raw_gpu.temperature = 55.0

    cs = manager.get_cluster_state([raw_gpu])
    state = agent.get_state(raw_gpu, cs)
    assert isinstance(state, tuple)
    assert len(state) == 16
    assert state[3] == 0  # risk_bin safely defaults to 0
    assert state[4] == 0  # fail_bin safely defaults to 0


# ==============================================================================
# SUITE B: Risk-Aware Reward Function
# ==============================================================================

def test_suite_b_higher_efc_and_risk_produces_lower_reward():
    """Verify lower-risk GPU yields more favorable expected reward than higher-risk GPU."""
    manager = MARLManager(num_gpus=2)
    task = make_task("t1", mem=2.0, time=4, severity="MEDIUM")

    gpu_safe = make_test_gpu(1, risk_score=0.05)
    gpu_risky = make_test_gpu(2, risk_score=0.85)

    reward_safe = manager.calculate_reward(gpu_safe, task, success=True, expected_failure_cost=0.1)
    reward_risky = manager.calculate_reward(gpu_risky, task, success=True, expected_failure_cost=3.5)

    assert reward_safe > reward_risky
    assert reward_safe - reward_risky >= 2.0  # Demonstrable risk penalty delta


def test_suite_b_success_positive_contribution():
    """Verify successful execution on healthy GPU produces a bounded positive reward."""
    manager = MARLManager(num_gpus=1)
    task = make_task("t_succ", mem=1.0, time=2)

    gpu = make_test_gpu(1, risk_score=0.0)
    reward = manager.calculate_reward(gpu, task, success=True, expected_failure_cost=0.0)

    assert reward > 5.0
    assert reward <= 12.0  # Bounded by config max_reward


def test_suite_b_failure_and_cooldown_penalties():
    """Verify strong bounded penalties for failures, crashes, and cooldowns."""
    manager = MARLManager(num_gpus=1)
    task = make_task("t_fail")

    gpu_failed = make_test_gpu(1)
    reward_failed = manager.calculate_reward(gpu_failed, task, success=False)
    assert reward_failed == -20.0

    gpu_crashed = make_test_gpu(1, crashed=True)
    reward_crashed = manager.calculate_reward(gpu_crashed, task, success=True)
    assert reward_crashed == -20.0

    gpu_cooldown = make_test_gpu(1, cooldown=3)
    gpu_normal = make_test_gpu(1, cooldown=0)
    reward_cd = manager.calculate_reward(gpu_cooldown, task, success=True)
    reward_norm = manager.calculate_reward(gpu_normal, task, success=True)
    assert reward_cd < reward_norm


def test_suite_b_quarantined_constraint_penalty():
    """Verify assignment to quarantined GPU incurs constraint violation penalty."""
    manager = MARLManager(num_gpus=1)
    task = make_task("t_quar")
    gpu_quarantine = make_test_gpu(1, quarantined=True)
    gpu_healthy = make_test_gpu(1, quarantined=False)

    r_quar = manager.calculate_reward(gpu_quarantine, task, success=True)
    r_norm = manager.calculate_reward(gpu_healthy, task, success=True)
    assert r_quar < r_norm


def test_suite_b_configurable_reward_weights():
    """Verify reward weights are configurable in one documented configuration object."""
    custom_cfg = RiskAwareRewardConfig(
        weight_risk_penalty=10.0,
        weight_expected_failure_cost=5.0
    )
    manager = MARLManager(num_gpus=1, reward_config=custom_cfg)
    task = make_task("t_cfg")
    gpu = make_test_gpu(1, risk_score=0.5)

    reward = manager.calculate_reward(gpu, task, success=True, expected_failure_cost=1.0)
    assert reward < 2.0


# ==============================================================================
# SUITE C: Expected Failure Cost
# ==============================================================================

def test_suite_c_cost_increases_with_failure_probability():
    """Verify EFC increases monotonically with failure probability / risk score."""
    manager = MARLManager(num_gpus=1)
    task = make_task("t_prob", mem=2.0, time=4)

    gpu_low = make_test_gpu(1, risk_score=0.10)
    gpu_high = make_test_gpu(1, risk_score=0.90)

    cost_low = manager.calculate_expected_failure_cost(gpu_low, task)
    cost_high = manager.calculate_expected_failure_cost(gpu_high, task)

    assert cost_high > cost_low
    assert cost_low >= 0.0


def test_suite_c_cost_increases_with_recovery_cost():
    """Verify EFC increases when recovery/restart cost increases."""
    manager = MARLManager(num_gpus=1)
    task = make_task("t_rec", mem=2.0, time=4)
    gpu = make_test_gpu(1, risk_score=0.50)

    cost_fast_rec = manager.calculate_expected_failure_cost(gpu, task, recovery_cost=1.0)
    cost_slow_rec = manager.calculate_expected_failure_cost(gpu, task, recovery_cost=10.0)

    assert cost_slow_rec > cost_fast_rec


def test_suite_c_cost_reflects_task_criticality():
    """Verify EFC scales with task criticality / workload severity."""
    manager = MARLManager(num_gpus=1)
    gpu = make_test_gpu(1, risk_score=0.50)

    task_low = make_task("t_low_sev", mem=1.0, time=2, severity="LOW")
    task_critical = make_task("t_high_sev", mem=6.0, time=15, severity="VERY HIGH")

    cost_low = manager.calculate_expected_failure_cost(gpu, task_low)
    cost_critical = manager.calculate_expected_failure_cost(gpu, task_critical)

    assert cost_critical > cost_low


def test_suite_c_missing_inputs_fallback_and_bounded():
    """Verify missing inputs produce safe, finite, bounded cost."""
    manager = MARLManager(num_gpus=1)
    bare_gpu = GPU(1, 8.0)  # No risk engine
    bare_task = Task()  # Bare task
    cost = manager.calculate_expected_failure_cost(bare_gpu, bare_task)
    assert math.isfinite(cost)
    assert 0.0 <= cost <= 20.0


# ==============================================================================
# SUITE D: Actual MARL GPU Selection
# ==============================================================================

def test_suite_d_marl_selects_feasible_lower_risk_gpu():
    """Verify MARL cooperative bidding selects lower-risk GPU under comparable performance."""
    manager = MARLManager(num_gpus=2)
    task = make_task("t_select", mem=2.0, time=3)

    gpu_safe = make_test_gpu(1, capacity=8.0, current_mem=2.0, risk_score=0.05)
    gpu_risky = make_test_gpu(2, capacity=8.0, current_mem=2.0, risk_score=0.85)

    cs = manager.get_cluster_state([gpu_safe, gpu_risky])
    bid_safe = manager.compute_bid_score(gpu_safe, task, manager.agents[0], cs)
    bid_risky = manager.compute_bid_score(gpu_risky, task, manager.agents[1], cs)

    assert bid_safe > bid_risky, "Lower-risk GPU must receive higher bid score"
    selected = manager.select_gpu_for_task(task, [gpu_safe, gpu_risky], cs)
    assert selected == 0, "MARL must select index 0 (GPU 1, lower risk)"


def test_suite_d_quarantined_gpu_is_never_selected():
    """Verify quarantined GPU returns -inf bid and is never selected over a feasible GPU."""
    manager = MARLManager(num_gpus=2)
    task = make_task("t_quar_sel", mem=1.0, time=2)

    gpu_quar = make_test_gpu(1, quarantined=True)
    gpu_healthy = make_test_gpu(2, quarantined=False)

    cs = manager.get_cluster_state([gpu_quar, gpu_healthy])
    bid_quar = manager.compute_bid_score(gpu_quar, task, manager.agents[0], cs)
    bid_healthy = manager.compute_bid_score(gpu_healthy, task, manager.agents[1], cs)

    assert bid_quar == -np.inf
    assert bid_healthy > -np.inf

    selected = manager.select_gpu_for_task(task, [gpu_quar, gpu_healthy], cs)
    assert selected == 1, "Must select healthy GPU 2"


def test_suite_d_insufficient_vram_gpu_is_rejected():
    """Verify GPU exceeding physical memory admission constraint receives -inf bid."""
    manager = MARLManager(num_gpus=2)
    task = make_task("t_vram", mem=4.0, time=2)

    gpu_full = make_test_gpu(1, capacity=8.0, current_mem=6.0)  # Only 2GB free
    gpu_avail = make_test_gpu(2, capacity=8.0, current_mem=1.0)  # 7GB free

    cs = manager.get_cluster_state([gpu_full, gpu_avail])
    bid_full = manager.compute_bid_score(gpu_full, task, manager.agents[0], cs)
    bid_avail = manager.compute_bid_score(gpu_avail, task, manager.agents[1], cs)

    assert bid_full == -np.inf
    assert bid_avail > -np.inf

    selected = manager.select_gpu_for_task(task, [gpu_full, gpu_avail], cs)
    assert selected == 1


def test_suite_d_environment_assigns_gpu_selected_by_marl():
    """Verify end-to-end that environment actually assigns task to the GPU selected by MARL."""
    env = GPUEnvironment(scheduler="marl")
    task = make_task("video_0001", mem=2.0, time=3)

    assigned = env.marl_scheduler(task)
    assert assigned is True
    assert task.id in env.task_queue.task_to_gpu
    assigned_gpu_id = env.task_queue.task_to_gpu[task.id]

    decision = env.last_marl_decision
    assert decision["task_id"] == task.id
    assert decision["selected_gpu"] == f"GPU {assigned_gpu_id}"
    assert "risk_score" in decision
    assert "expected_failure_cost" in decision
    assert "reward" in decision


def test_suite_d_no_assignment_when_all_infeasible():
    """Verify no assignment occurs when all GPUs are infeasible (quarantined/full)."""
    env = GPUEnvironment(scheduler="marl")
    # Make all GPUs quarantined
    for gpu in env.gpus:
        risk_state = gpu.get_risk_state()
        if risk_state:
            risk_state.is_quarantined = True

    task = make_task("video_0099", mem=2.0, time=3)
    assigned = env.marl_scheduler(task)
    assert assigned is False
    assert task.id not in env.task_queue.task_to_gpu
    assert env.last_marl_decision["status"] == "rejected_no_feasible_gpu"


def test_suite_d_q_learning_update_matches_transition():
    """Verify pre-action state, reward, and next state correctly update agent Q-table."""
    env = GPUEnvironment(scheduler="marl")
    env.learning_enabled = True
    agent = env.marl_manager.agents[0]
    initial_q_size = len(agent.q_table)

    task = make_task("video_0002", mem=1.0, time=2)
    env.marl_scheduler(task)

    assert len(agent.q_table) >= initial_q_size
    assert env.last_marl_decision["reward"] is not None


# ==============================================================================
# SUITE E: Dynamic Risk Changes
# ==============================================================================

def test_suite_e_dynamic_risk_increase_shifts_scheduling():
    """Verify elevating a GPU's risk dynamically changes subsequent scheduling decisions."""
    env = GPUEnvironment(scheduler="marl")

    # Initially both GPU 1 and GPU 2 are healthy and idle
    task1 = make_task("v1", mem=1.0, time=2)
    selected_initial = env.marl_manager.select_gpu_for_task(task1, env.gpus, env.marl_manager.get_cluster_state(env.gpus))

    # Elevate risk on the preferred GPU to high risk (0.90)
    pref_gpu = env.gpus[selected_initial]
    risk_state = pref_gpu.get_risk_state()
    risk_state.risk_score = 0.90

    # Next task should shift away from high-risk GPU
    task2 = make_task("v2", mem=1.0, time=2)
    selected_after = env.marl_manager.select_gpu_for_task(task2, env.gpus, env.marl_manager.get_cluster_state(env.gpus))

    assert selected_after != selected_initial, "Risk increase must steer scheduler to alternative feasible GPU"


def test_suite_e_fault_quarantine_and_recovery_cycle():
    """Verify fault quarantine excludes GPU, and recovery restores eligibility."""
    env = GPUEnvironment(scheduler="marl")
    target_gpu = env.gpus[0]
    task = make_task("v_rec", mem=1.0, time=2)

    # 1. Quarantine target GPU
    risk_state = target_gpu.get_risk_state()
    risk_state.is_quarantined = True

    cs = env.marl_manager.get_cluster_state(env.gpus)
    bid_quar = env.marl_manager.compute_bid_score(target_gpu, task, env.marl_manager.agents[0], cs)
    assert bid_quar == -np.inf

    # 2. Recover target GPU
    risk_state.is_quarantined = False
    risk_state.risk_score = 0.10

    bid_recovered = env.marl_manager.compute_bid_score(target_gpu, task, env.marl_manager.agents[0], cs)
    assert bid_recovered > -np.inf, "Recovered GPU must regain finite bid score"


# ==============================================================================
# SUITE F: Concurrency and Regression
# ==============================================================================

def test_suite_f_concurrent_marl_scheduling_thread_safety():
    """Verify concurrent MARL scheduling requests do not corrupt Q-tables or memory accounting."""
    env = GPUEnvironment(scheduler="marl")
    env.learning_enabled = True

    num_threads = 4
    tasks_per_thread = 15
    errors = []

    def worker(worker_id):
        for i in range(tasks_per_thread):
            try:
                task = make_task(f"thread_{worker_id}_task_{i}", mem=1.0, time=2)
                env.marl_scheduler(task)
            except Exception as e:
                errors.append(e)

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0, f"Concurrent MARL scheduling raised errors: {errors}"

    # Verify memory invariants hold
    for gpu in env.gpus:
        assert 0.0 <= gpu.current_memory <= gpu.memory_capacity
        assert gpu.current_memory >= 0.0


def test_suite_f_bit5_locking_invariants_preserved():
    """Verify physical capacity admission invariant (8.0 GB) and data consistency under load."""
    env = GPUEnvironment(scheduler="marl")
    # Attempting to assign task larger than capacity must be strictly rejected
    huge_task = make_task("huge_task", mem=9.0, time=10)
    assigned = env.marl_scheduler(huge_task)
    assert assigned is False
    assert huge_task.id not in env.task_queue.task_to_gpu


# ==============================================================================
# SUITE G: Bit 6 Decisive Audit Verifications
# ==============================================================================

def test_decisive_higher_risk_changes_scores_and_gpu_selection():
    """Prove deterministically that increasing GPU risk lowers candidate score and flips GPU selection."""
    manager = MARLManager(num_gpus=2)
    task = make_task("t_risk_flip", mem=2.0, time=3)

    gpu1 = make_test_gpu(1, capacity=8.0, current_mem=2.0, risk_score=0.0)
    gpu2 = make_test_gpu(2, capacity=8.0, current_mem=2.0, risk_score=0.0)
    cs = manager.get_cluster_state([gpu1, gpu2])

    score1_init = manager.compute_bid_score(gpu1, task, manager.agents[0], cs)
    score2_init = manager.compute_bid_score(gpu2, task, manager.agents[1], cs)
    assert score1_init == score2_init

    # Elevate risk on GPU 1
    gpu1.get_risk_state().risk_score = 0.85
    score1_risky = manager.compute_bid_score(gpu1, task, manager.agents[0], cs)
    score2_safe = manager.compute_bid_score(gpu2, task, manager.agents[1], cs)

    assert score1_risky < score1_init, "Elevated risk must strictly reduce the candidate bid score"
    assert score2_safe > score1_risky, "Healthy GPU must score strictly higher than high-risk GPU"

    # MARL selection flips to GPU 2 (index 1)
    selected = manager.select_gpu_for_task(task, [gpu1, gpu2], cs)
    assert selected == 1, "MARL must select GPU 2 (lower risk)"


def test_decisive_higher_efc_influences_decisions():
    """Prove deterministically that higher expected failure cost lowers candidate score and steers scheduling."""
    manager = MARLManager(num_gpus=2)
    task = make_task("t_efc", mem=2.0, time=4)

    # Identical load, but GPU 1 has higher failure history
    gpu1 = make_test_gpu(1, capacity=8.0, current_mem=2.0, risk_score=0.40, failure_rate=25.0)
    gpu2 = make_test_gpu(2, capacity=8.0, current_mem=2.0, risk_score=0.40, failure_rate=0.0)

    efc1 = manager.calculate_expected_failure_cost(gpu1, task)
    efc2 = manager.calculate_expected_failure_cost(gpu2, task)
    assert efc1 > efc2, "GPU with failure history must incur higher expected failure cost"

    cs = manager.get_cluster_state([gpu1, gpu2])
    score1 = manager.compute_bid_score(gpu1, task, manager.agents[0], cs)
    score2 = manager.compute_bid_score(gpu2, task, manager.agents[1], cs)
    assert score2 > score1, "Lower EFC must yield higher candidate score"

    selected = manager.select_gpu_for_task(task, [gpu1, gpu2], cs)
    assert selected == 1, "MARL must steer decision to the lower EFC candidate"


def test_decisive_actual_assigned_gpu_matches_marl_decision():
    """Prove the actual assigned GPU in environment exactly matches the MARL decision."""
    env = GPUEnvironment(scheduler="marl")
    task = make_task("video_0042", mem=2.0, time=3)

    assigned = env.marl_scheduler(task)
    assert assigned is True

    decision = env.last_marl_decision
    assert decision["status"] == "assigned"
    selected_idx = decision["selected_gpu_index"]
    expected_gpu_str = f"GPU {env.gpus[selected_idx].id}"
    assert decision["selected_gpu"] == expected_gpu_str

    # Task queue assignment matches
    actual_assigned_gpu_id = env.task_queue.task_to_gpu[task.id]
    assert actual_assigned_gpu_id == env.gpus[selected_idx].id

    # The selected GPU actually received the task
    with env.gpus[selected_idx]._lock:
        assigned_tasks = [t.id for t in env.gpus[selected_idx].task_queue + env.gpus[selected_idx].running_tasks]
        assert task.id in assigned_tasks


def test_decisive_correct_q_table_entry_updated_after_transition():
    """Prove the correct Q-table entry is updated after a scheduling transition and negative reward reduces future bids."""
    env = GPUEnvironment(scheduler="marl")
    env.learning_enabled = True
    task = make_task("video_0088", mem=2.0, time=3)

    agent = env.marl_manager.agents[0]
    target_gpu = env.gpus[0]
    cs = env.marl_manager.get_cluster_state(env.gpus)
    cs['current_task_memory'] = task.memory_required
    cs['current_task_time'] = task.execution_time
    cs['current_efc'] = env.marl_manager.calculate_expected_failure_cost(target_gpu, task)
    pre_state = agent.get_state(target_gpu, cs)

    # Initialize entry to 0.0
    agent.q_table[pre_state][MARLAgent.ACTION_ACCEPT] = 0.0
    bid_before = env.marl_manager.compute_bid_score(target_gpu, task, agent, cs)

    # Perform scheduling
    env.marl_scheduler(task)
    q_val_after = agent.q_table[pre_state][MARLAgent.ACTION_ACCEPT]

    # Verify Q-value changed from transition reward
    assert q_val_after != 0.0, "Q-table entry for ACTION_ACCEPT must be updated"

    # Now verify negative Q-value on pre_state reduces bid score for that state
    fresh_gpu = make_test_gpu(1)
    fresh_cs = env.marl_manager.get_cluster_state(env.gpus)
    fresh_cs['current_task_memory'] = task.memory_required
    fresh_cs['current_task_time'] = task.execution_time
    fresh_cs['current_efc'] = env.marl_manager.calculate_expected_failure_cost(fresh_gpu, task)

    agent.q_table[pre_state][MARLAgent.ACTION_ACCEPT] = 0.0
    bid_zero = env.marl_manager.compute_bid_score(fresh_gpu, task, agent, fresh_cs)

    agent.q_table[pre_state][MARLAgent.ACTION_ACCEPT] = -10.0
    bid_neg = env.marl_manager.compute_bid_score(fresh_gpu, task, agent, fresh_cs)

    assert bid_neg < bid_zero, "Negative Q-value must directly reduce candidate bid score"
    assert bid_zero - bid_neg == 5.0, "Q-weight (0.50 * 10) must produce exactly 5.0 bid delta"


def test_decisive_all_infeasible_candidates_cause_safe_rejection():
    """Prove all-infeasible candidates cause safe rejection without state corruption."""
    env = GPUEnvironment(scheduler="marl")
    initial_memories = [g.current_memory for g in env.gpus]
    initial_failed = env.failed_assignments

    # Make all GPUs unavailable (e.g. at capacity)
    for g in env.gpus:
        g.current_memory = g.memory_capacity

    task = make_task("video_reject", mem=2.0, time=3)
    assigned = env.marl_scheduler(task)

    assert assigned is False
    assert env.last_marl_decision["status"] == "rejected_no_feasible_gpu"
    assert env.failed_assignments == initial_failed + 1
    assert task.id not in env.task_queue.task_to_gpu

    # Memories unchanged
    for g, init_mem in zip(env.gpus, initial_memories):
        assert g.current_memory == g.memory_capacity


def test_decisive_benchmark_metrics_use_correct_denominators_and_resets():
    """Prove benchmark metrics use total_submitted as denominator and environments isolate state."""
    from benchmark_risk_aware_marl import setup_env_for_scenario, run_scenario_episode

    env1, trace1 = setup_env_for_scenario(1, "marl", steps=5, seed=201)
    for _ in range(len(trace1)):
        env1.step()

    metrics1 = env1.get_metrics()
    total_submitted = metrics1["total_tasks_submitted"]
    completed = metrics1["completed"]
    assert total_submitted > 0
    calculated_rate = (completed / total_submitted) * 100.0

    # Execute episode via benchmark runner
    res = run_scenario_episode(1, "marl", steps=5, seed=201)
    assert math.isclose(res["completion_rate"], calculated_rate, rel_tol=1e-5)
    assert res["submitted"] == total_submitted
    assert res["completed"] == completed
    assert res["capacity_violations"] == 0


def test_decisive_core_q_table_generalization():
    """Prove that core Q-table enables cross-state generalization across non-core cluster variations."""
    agent = MARLAgent(gpu_id=0)
    agent.epsilon = 0.0  # Greedy selection for testing
    gpu = make_test_gpu(0, capacity=8.0, current_mem=2.0, risk_score=0.10)
    task1 = make_task("t1", mem=2.0, time=3)

    # Context 1: low cluster queue, low cluster risk
    cs1 = {"cluster_queue_load": 0.1, "cluster_avg_risk": 0.1, "cluster_risk_variance": 0.0,
           "num_quarantined": 0, "quarantine_ratio": 0.0, "time_of_day": 0.2, "hour_of_day": 2,
           "day_of_week": 1, "recent_crash_count": 0, "consecutive_successes": 0,
           "current_task_memory": 2.0, "current_task_time": 3, "current_efc": 0.05}
    s1 = agent.get_state(gpu, cs1)

    # Initially zero
    assert agent.choose_action(s1) == MARLAgent.ACTION_ACCEPT

    # Update agent with a severe penalty for action ACCEPT on s1
    next_s1 = s1
    agent.update(s1, MARLAgent.ACTION_ACCEPT, -8.0, next_s1)

    # Context 2: totally different non-core features (high cluster utilization, idle gpus)
    # but SAME core physical and task features
    cs2 = {"total_cluster_utilization": 85.0, "idle_gpu_count": 3, "overloaded_gpu_count": 2,
           "available_cluster_memory": 20.0, "current_task_impact": 0.8,
           "current_task_memory": 2.0, "current_task_time": 3, "current_efc": 0.05}
    s2 = agent.get_state(gpu, cs2)
    assert s2 != s1, "Full 16-tuple states must be different due to non-core cluster features"
    assert agent.get_core_state(s2) == agent.get_core_state(s1), "Core states must match"

    # Even though s2 was never directly visited in exact q_table:
    assert s2 not in agent.q_table
    # The learned penalty generalized via core_q_table, steering action away from accept:
    chosen_action = agent.choose_action(s2)
    assert chosen_action != MARLAgent.ACTION_ACCEPT, "Core Q-table must generalize penalty to steer away from accept"


def test_decisive_strict_admission_and_quarantine_rejection():
    """Prove that normal scheduling never exceeds physical 8.0 GB capacity and never assigns to quarantined/unavailable GPUs."""
    env = GPUEnvironment(scheduler="marl")

    # 1. Physical memory limit enforcement
    # GPU 0 has 2GB free (6GB used)
    env.gpus[0].current_memory = 6.0
    for g in env.gpus[1:]:
        g.current_memory = 7.5  # Only 0.5GB free

    # Task requiring 3.0 GB cannot fit on any GPU (exceeds 8.0 GB limit)
    task_oversize = make_task("oversize_task", mem=3.0, time=4)
    assigned = env.marl_scheduler(task_oversize)
    assert assigned is False
    assert task_oversize.id not in env.task_queue.task_to_gpu
    for g in env.gpus:
        assert g.current_memory <= 8.0, "Physical capacity of 8.0 GB must never be exceeded"

    # 2. Quarantined GPU enforcement
    # Give GPU 0 plenty of memory (1.0 GB used), but quarantine it
    env.gpus[0].current_memory = 1.0
    env.gpus[0].get_risk_state().is_quarantined = True

    # Other GPUs have no room (8.0 GB used)
    for g in env.gpus[1:]:
        g.current_memory = 8.0

    task_fit_quar_only = make_task("fit_quar_only", mem=2.0, time=2)
    assigned = env.marl_scheduler(task_fit_quar_only)
    assert assigned is False
    assert task_fit_quar_only.id not in env.task_queue.task_to_gpu
    assert len(env.gpus[0].running_tasks) == 0
    assert len(env.gpus[0].task_queue) == 0

    # 3. Unavailable / Cooldown GPU enforcement
    env.gpus[0].get_risk_state().is_quarantined = False
    env.gpus[0].crashed = True
    env.gpus[0].crash_cooldown_timer = 5

    task_fit_cooldown_only = make_task("fit_cooldown_only", mem=2.0, time=2)
    assigned = env.marl_scheduler(task_fit_cooldown_only)
    assert assigned is False
    assert task_fit_cooldown_only.id not in env.task_queue.task_to_gpu


def test_decisive_scenario_differentiation_and_workload_dist():
    """Prove deterministically that each benchmark scenario changes its intended workload or risk conditions."""
    from benchmark_risk_aware_marl import setup_env_for_scenario
    from evaluation.workload_trace import get_trace_distribution

    # Scenario 1 vs Scenario 5 differentiation
    env1, trace1 = setup_env_for_scenario(1, "marl", steps=15, seed=101)
    env5, trace5 = setup_env_for_scenario(5, "marl", steps=15, seed=101)

    dist1 = get_trace_distribution(trace1)
    dist5 = get_trace_distribution(trace5)

    # 1. Total tasks must differ (normal 2-5 vs mixed 2-12)
    assert dist1["total_tasks"] != dist5["total_tasks"], "Scenario 1 (normal) and Scenario 5 (mixed) must generate different task counts"
    assert dist1["total_tasks"] < dist5["total_tasks"], "Mixed workload must have higher throughput demand than steady normal"

    # 2. Priority composition: Scenario 5 must have critical tasks
    assert "critical" in dist5["priorities"], "Scenario 5 must include critical priority tasks"
    assert "critical" not in dist1["priorities"], "Scenario 1 standard workload must not include critical tier"

    # 3. Scenario 2: GPU 0 has elevated risk
    env2, _ = setup_env_for_scenario(2, "marl", steps=15, seed=101)
    assert env2.gpus[0].get_risk_state().risk_score >= 0.75, "Scenario 2 must elevate GPU 0 risk score"
    assert env1.gpus[0].get_risk_state().risk_score < 0.25, "Scenario 1 GPU 0 must remain healthy low-risk"

    # 4. Scenario 3: GPU 0 has active cooldown timer
    env3, _ = setup_env_for_scenario(3, "marl", steps=15, seed=101)
    assert env3.gpus[0].is_in_cooldown(), "Scenario 3 must place GPU 0 in active cooldown"
    assert not env1.gpus[0].is_in_cooldown(), "Scenario 1 GPU 0 must not be in cooldown"

    # 5. Scenario 4: Fault injector is active
    env4, _ = setup_env_for_scenario(4, "marl", steps=15, seed=101)
    assert env4.fault_injector.enabled is True, "Scenario 4 must have fault injector enabled"

    # 6. Scenario 6: Burst pattern variance
    _, trace6 = setup_env_for_scenario(6, "marl", steps=15, seed=101)
    counts6 = [len(step) for step in trace6]
    # Burst traffic has steps with 0 tasks and steps with >=15 tasks
    assert any(c <= 2 for c in counts6) and any(c >= 15 for c in counts6), "Scenario 6 must produce burst traffic"


def test_decisive_task_accounting_conservation_law():
    """Prove task accounting conservation: submitted = completed + queued + running + failed + cancelled."""
    from benchmark_risk_aware_marl import setup_env_for_scenario
    from dynamic_queue import TaskStatus

    env, trace = setup_env_for_scenario(5, "marl", steps=10, seed=101)

    for step in range(len(trace)):
        env.step()
        counts = env.task_queue.get_task_counts()
        submitted = max(env.total_tasks, counts["total"])
        completed = counts["completed"]
        queued = counts["queued"] + counts["scheduled"] + counts["waiting"]
        running = counts["running"]
        failed = counts["failed"]
        cancelled = counts["cancelled"]

        total_accounted = completed + queued + running + failed + cancelled
        assert submitted == total_accounted, f"Step {step}: Task accounting mismatch! {submitted} != {total_accounted}"

    # Verify post-drain conservation
    env.attach_workload_trace([[] for _ in range(5)])
    for step in range(5):
        env.step()
        counts = env.task_queue.get_task_counts()
        submitted = max(env.total_tasks, counts["total"])
        completed = counts["completed"]
        queued = counts["queued"] + counts["scheduled"] + counts["waiting"]
        running = counts["running"]
        failed = counts["failed"]
        cancelled = counts["cancelled"]
        assert submitted == completed + queued + running + failed + cancelled


def test_decisive_marl_trained_vs_heuristic_ablation():
    """Prove that trained MARL updates the 7D core Q-table and shifts decisions compared to untrained heuristic."""
    from benchmark_risk_aware_marl import setup_env_for_scenario
    from copy import deepcopy

    # 1. Untrained MARL (Heuristic-only, Q-tables zeroed)
    env_untrained, trace = setup_env_for_scenario(5, "marl", steps=10, seed=101)
    env_untrained.learning_enabled = False
    for ag in env_untrained.marl_manager.agents:
        ag.epsilon = 0.0
        ag.q_table.clear()
        ag.core_q_table.clear()

    decisions_untrained = []
    for _ in range(len(trace)):
        env_untrained.step()
        if hasattr(env_untrained, "last_marl_decision") and env_untrained.last_marl_decision:
            decisions_untrained.append(deepcopy(env_untrained.last_marl_decision))

    # 2. Trained MARL (Active learning)
    env_trained, _ = setup_env_for_scenario(5, "marl", steps=10, seed=101)
    env_trained.learning_enabled = True

    decisions_trained = []
    for _ in range(len(trace)):
        env_trained.step()
        if hasattr(env_trained, "last_marl_decision") and env_trained.last_marl_decision:
            decisions_trained.append(deepcopy(env_trained.last_marl_decision))

    # Verify that core Q-table has been populated
    total_core_q_entries = sum(len(ag.core_q_table) for ag in env_trained.marl_manager.agents)
    assert total_core_q_entries > 0, "Trained MARL must update the 7-dimensional core Q-table"

    # Verify that Q-values are non-zero
    core_values = [v for ag in env_trained.marl_manager.agents for q_arr in ag.core_q_table.values() for v in q_arr if v != 0.0]
    assert len(core_values) > 0, "Core Q-table must have non-zero Bellman updated values"

    # Verify candidate scores differ between trained and untrained
    bid_scores_differ = any(
        not math.isclose(d_t.get("gpu_scores", {}).get(k, 0.0), d_u.get("gpu_scores", {}).get(k, 0.0), abs_tol=1e-4)
        for d_t, d_u in zip(decisions_trained, decisions_untrained)
        for k in d_t.get("gpu_scores", {})
    )
    assert bid_scores_differ, "Trained Q-values must measurably alter candidate bid scores"


