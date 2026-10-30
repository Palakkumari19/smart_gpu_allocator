"""
Bit 6: Scientific Benchmark for Risk-Aware Cooperative MARL vs Traditional Baselines
Evaluates the 7 required experimental scenarios:
  1. Normal workload with low GPU risk (2-5 tasks/step steady normal arrival)
  2. One GPU with elevated risk (GPU 1 risk maintained elevated at 0.85)
  3. One GPU with high expected recovery cost (GPU 1 in active crash cooldown)
  4. Dynamic GPU fault and quarantine (mid-run fault injection and quarantine)
  5. Mixed workload with diverse task sizes (1-7 GB) and priorities (low/med/high/critical)
  6. Heavy burst workload (burst arrival spikes)
  7. Concurrent scheduling workload (multi-threaded concurrent arrivals)

Compares Risk-Aware MARL against:
  - Least Loaded
  - Round Robin
  - First-Come First-Served (FCFS)
"""

import argparse
from copy import deepcopy
import json
import math
import numpy as np
import random
import threading
import time

from environment import GPUEnvironment
from evaluation.workload_trace import generate_workload_trace, get_trace_distribution
from workload.task_generator import TaskGenerator, VideoProfile


SCENARIOS = [
    "1. Normal Workload (Low Risk)",
    "2. Elevated Risk on Single GPU",
    "3. High Recovery Cost on Single GPU",
    "4. Dynamic Fault & Quarantine",
    "5. Mixed Task Sizes & Priorities",
    "6. Heavy Burst Workload",
    "7. Concurrent Scheduling Workload"
]

SCHEDULERS = ["marl", "least_loaded", "round_robin", "fcfs"]


def setup_env_for_scenario(scenario_idx: int, scheduler: str, steps: int, seed: int):
    """Set up GPUEnvironment and initial conditions for a controlled scenario."""
    env = GPUEnvironment(scheduler=scheduler)
    env.enable_deterministic_physics()

    if scenario_idx == 1:
        # Scenario 1: Normal steady workload, low risk
        trace = generate_workload_trace("normal", steps, seed)
        env.attach_workload_trace(deepcopy(trace))

    elif scenario_idx == 2:
        # Scenario 2: One GPU with elevated risk (GPU 1 risk maintained elevated)
        trace = generate_workload_trace("normal", steps, seed)
        env.attach_workload_trace(deepcopy(trace))
        gpu = env.gpus[0]
        gpu.reliability_history.total_tasks = 100
        gpu.reliability_history.failed_tasks = 45
        gpu.reliability_history.anomalies_detected = 5
        gpu.reliability_history.crash_events = 5
        gpu.temperature = 80.0
        risk_state = gpu.get_risk_state()
        if risk_state:
            risk_state.risk_score = 0.85
            risk_state.reliability_history.total_tasks = 100
            risk_state.reliability_history.failed_tasks = 45
            risk_state.reliability_history.anomalies_detected = 5
            risk_state.reliability_history.crash_events = 5

    elif scenario_idx == 3:
        # Scenario 3: One GPU with high recovery cost (cooldown timer active)
        trace = generate_workload_trace("normal", steps, seed)
        env.attach_workload_trace(deepcopy(trace))
        gpu = env.gpus[0]
        gpu.crash_cooldown_timer = 5
        gpu.crashed = True
        gpu.reliability_history.total_tasks = 100
        gpu.reliability_history.failed_tasks = 25
        gpu.reliability_history.crash_events = 3
        risk_state = gpu.get_risk_state()
        if risk_state:
            risk_state.risk_score = 0.50
            risk_state.reliability_history.total_tasks = 100
            risk_state.reliability_history.failed_tasks = 25
            risk_state.reliability_history.crash_events = 3

    elif scenario_idx == 4:
        # Scenario 4: Dynamic GPU fault and quarantine mid-run
        trace = generate_workload_trace("normal", steps, seed)
        env.attach_workload_trace(deepcopy(trace))
        env.fault_injector.set_enabled(True)
        env.set_fault_injection_probability("HIGH", 0.3)

    elif scenario_idx == 5:
        # Scenario 5: Mixed workload with diverse task sizes (1-7 GB) and all priority tiers
        trace = generate_workload_trace("mixed", steps, seed, diverse_mix=True)
        env.attach_workload_trace(deepcopy(trace))

    elif scenario_idx == 6:
        # Scenario 6: Heavy burst workload
        trace = generate_workload_trace("burst", steps, seed)
        env.attach_workload_trace(deepcopy(trace))

    elif scenario_idx == 7:
        # Scenario 7: Handled via concurrent runner
        trace = []

    return env, trace


def log_scenario_conditions(scenario_idx: int, trace, seed: int):
    """Log deterministic workload distributions and initial environment conditions."""
    print(f"\n{'='*75}")
    print(f"  SCENARIO {SCENARIOS[scenario_idx - 1]} (Seed {seed})")
    print(f"{'='*75}")
    if scenario_idx == 7:
        print("  Workload Type       : Concurrent Arrival (4 threads x 10 tasks = 40 tasks)")
        print("  Task Composition    : Dynamic video generator mix across threads")
        print("  Initial Cluster Risk: All 4 GPUs Healthy (Risk ~0.10)")
    else:
        dist = get_trace_distribution(trace)
        print(f"  Workload Tasks      : {dist['total_tasks']} total tasks across {len(trace)} steps")
        print(f"  VRAM Range (GB)     : {dist['min_memory']:.1f} - {dist['max_memory']:.1f} GB (Avg: {dist['avg_memory']:.2f} GB)")
        print(f"  Execution Time      : {dist['min_time']} - {dist['max_time']} steps (Avg: {dist['avg_time']:.2f} steps)")
        print(f"  Priorities          : {dist['priorities']}")
        print(f"  Severities          : {dist['severities']}")
        if scenario_idx == 1:
            print("  Initial Cluster Risk: All 4 GPUs Healthy (Risk ~0.10, 0% failure history)")
        elif scenario_idx == 2:
            print("  Initial Cluster Risk: GPU 0 ELEVATED (Risk: 0.85, Failure Rate: 45%, Temp: 80°C)")
        elif scenario_idx == 3:
            print("  Initial Cluster Risk: GPU 0 CRASHED & COOLDOWN (Cooldown: 5 steps, Risk: 0.50)")
        elif scenario_idx == 4:
            print("  Initial Cluster Risk: Dynamic fault injected on GPU 1 at step 7 -> Quarantine")
        elif scenario_idx == 5:
            print("  Initial Cluster Risk: All 4 GPUs Healthy, Heterogeneous task demands & critical priority")
        elif scenario_idx == 6:
            print("  Initial Cluster Risk: Heavy Burst Traffic (70% idle, 30% massive spikes)")
    print(f"{'='*75}")


def run_scenario_episode(scenario_idx: int, scheduler: str, steps: int, seed: int, drain_steps: int = 0):
    """Execute one episode for a scenario and collect verified metrics across active and drain phases."""
    if scenario_idx == 7:
        return run_concurrent_scenario(scheduler, steps, seed, drain_steps=drain_steps)

    env, trace = setup_env_for_scenario(scenario_idx, scheduler, steps, seed)

    total_efc = 0.0
    efc_count = 0
    marl_rewards = []
    step_memory_usages = []

    # 1. Active Arrival Phase
    for step in range(len(trace)):
        # Mid-episode dynamic fault injection for Scenario 4
        if scenario_idx == 4 and step == steps // 2:
            target_gpu = env.gpus[1]
            target_gpu.reliability_history.anomalies_detected = 5
            target_gpu.reliability_history.crash_events = 5
            target_gpu.reliability_history.failed_tasks = 40
            target_gpu.temperature = 86.0
            target_gpu.update_risk()
            risk_state = target_gpu.get_risk_state()
            if risk_state:
                risk_state.is_quarantined = True
                risk_state.risk_score = max(0.95, risk_state.risk_score)
                risk_state.quarantine_reason = "Dynamic fault injection quarantine"

        env.step()
        step_memory_usages.append(sum(g.get_memory_usage_percentage() for g in env.gpus) / len(env.gpus))

        if scheduler == "marl" and hasattr(env, "last_marl_decision") and env.last_marl_decision:
            efc = env.last_marl_decision.get("expected_failure_cost", 0.0)
            rew = env.last_marl_decision.get("reward", None)
            total_efc += efc
            efc_count += 1
            if rew is not None:
                marl_rewards.append(rew)

    active_metrics = env.get_metrics()
    episode_completed = float(active_metrics["completed"])
    submitted = float(active_metrics.get("total_tasks_submitted", env.total_tasks))
    episode_completion_rate = (episode_completed / max(1.0, submitted)) * 100.0

    # 2. Drain Phase (if drain_steps > 0)
    if drain_steps > 0:
        env.attach_workload_trace([[] for _ in range(drain_steps)])
        for _ in range(drain_steps):
            env.step()
            step_memory_usages.append(sum(g.get_memory_usage_percentage() for g in env.gpus) / len(env.gpus))

    final_metrics = env.get_metrics()
    post_drain_completed = float(final_metrics["completed"])
    post_drain_completion_rate = (post_drain_completed / max(1.0, submitted)) * 100.0

    # 3. Task Accounting Verification
    counts = env.task_queue.get_task_counts()
    q_completed = counts["completed"]
    q_queued = counts["queued"] + counts["scheduled"] + counts["waiting"]
    q_running = counts["running"]
    q_failed = counts["failed"]
    q_cancelled = counts["cancelled"]
    accounted_sum = q_completed + q_queued + q_running + q_failed + q_cancelled
    accounting_valid = (int(submitted) == int(accounted_sum))

    crashes = float(final_metrics.get("total_crash_events", 0))
    failed_assignments = float(final_metrics.get("failed_assignments", 0))
    avg_efc = float(final_metrics.get("average_efc", (total_efc / max(1, efc_count)) if efc_count > 0 else 0.0))
    cap_violations = sum(1 for g in env.gpus if g.current_memory > g.memory_capacity)

    time_avg_utilization = float(np.mean(step_memory_usages)) if step_memory_usages else 0.0
    end_utilization = float(final_metrics["avg_memory_usage"])

    return {
        "seed": seed,
        "scheduler": scheduler,
        "submitted": submitted,
        "completed": post_drain_completed if drain_steps > 0 else episode_completed,
        "completion_rate": post_drain_completion_rate if drain_steps > 0 else episode_completion_rate,
        "episode_completed": episode_completed,
        "episode_completion_rate": episode_completion_rate,
        "post_drain_completed": post_drain_completed,
        "post_drain_completion_rate": post_drain_completion_rate,
        "failed_assignments": failed_assignments,
        "crashes": crashes,
        "latency": float(final_metrics["latency"]),
        "time_avg_utilization": time_avg_utilization,
        "end_utilization": end_utilization,
        "utilization": time_avg_utilization,  # Default for backward compatibility
        "throughput": (post_drain_completed if drain_steps > 0 else episode_completed) / max(1, steps + drain_steps),
        "avg_efc": avg_efc,
        "capacity_violations": cap_violations,
        "reward_mean": float(np.mean(marl_rewards)) if marl_rewards else 0.0,
        "accounting_valid": accounting_valid,
        "accounted_sum": accounted_sum,
        "q_queued": q_queued,
        "q_running": q_running,
    }


def run_concurrent_scenario(scheduler: str, steps: int, seed: int, drain_steps: int = 0):
    """Scenario 7: Concurrent scheduling workload across multiple threads with active & drain phases."""
    env = GPUEnvironment(scheduler=scheduler)
    env.enable_deterministic_physics()
    env.attach_workload_trace([[] for _ in range(steps)])
    num_threads = 4
    tasks_per_thread = 10
    total_efc = 0.0
    efc_count = 0
    marl_rewards = []
    lock = threading.Lock()
    step_memory_usages = []

    gen = TaskGenerator(seed=seed)
    thread_tasks = []
    for w in range(num_threads):
        tasks = []
        for i in range(tasks_per_thread):
            t = gen.generate_task()
            t.id = f"conc_{w}_{i}"
            tasks.append(t)
        thread_tasks.append(tasks)

    def worker(tid, task_list):
        nonlocal total_efc, efc_count
        for t in task_list:
            task_copy = deepcopy(t)
            env.add_task(task_copy)
            env._schedule_task(task_copy)
            if scheduler == "marl":
                with lock:
                    if hasattr(env, "last_marl_decision") and env.last_marl_decision:
                        total_efc += env.last_marl_decision.get("expected_failure_cost", 0.0)
                        rew = env.last_marl_decision.get("reward", None)
                        efc_count += 1
                        if rew is not None:
                            marl_rewards.append(rew)

    threads = [threading.Thread(target=worker, args=(w, thread_tasks[w])) for w in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 1. Active Steps
    for _ in range(steps):
        env.step()
        step_memory_usages.append(sum(g.get_memory_usage_percentage() for g in env.gpus) / len(env.gpus))

    active_metrics = env.get_metrics()
    episode_completed = float(active_metrics["completed"])
    submitted = float(active_metrics.get("total_tasks_submitted", env.task_queue.get_task_counts()["total"]))
    episode_completion_rate = (episode_completed / max(1.0, submitted)) * 100.0

    # 2. Drain Steps (if drain_steps > 0)
    if drain_steps > 0:
        env.attach_workload_trace([[] for _ in range(drain_steps)])
        for _ in range(drain_steps):
            env.step()
            step_memory_usages.append(sum(g.get_memory_usage_percentage() for g in env.gpus) / len(env.gpus))

    final_metrics = env.get_metrics()
    post_drain_completed = float(final_metrics["completed"])
    post_drain_completion_rate = (post_drain_completed / max(1.0, submitted)) * 100.0

    # 3. Task Accounting Verification
    counts = env.task_queue.get_task_counts()
    q_completed = counts["completed"]
    q_queued = counts["queued"] + counts["scheduled"] + counts["waiting"]
    q_running = counts["running"]
    q_failed = counts["failed"]
    q_cancelled = counts["cancelled"]
    accounted_sum = q_completed + q_queued + q_running + q_failed + q_cancelled
    accounting_valid = (int(submitted) == int(accounted_sum))

    time_avg_utilization = float(np.mean(step_memory_usages)) if step_memory_usages else 0.0
    end_utilization = float(final_metrics["avg_memory_usage"])
    cap_violations = sum(1 for g in env.gpus if g.current_memory > g.memory_capacity)

    return {
        "seed": seed,
        "scheduler": scheduler,
        "submitted": submitted,
        "completed": post_drain_completed if drain_steps > 0 else episode_completed,
        "completion_rate": post_drain_completion_rate if drain_steps > 0 else episode_completion_rate,
        "episode_completed": episode_completed,
        "episode_completion_rate": episode_completion_rate,
        "post_drain_completed": post_drain_completed,
        "post_drain_completion_rate": post_drain_completion_rate,
        "failed_assignments": float(final_metrics.get("failed_assignments", 0)),
        "crashes": float(final_metrics.get("total_crash_events", 0)),
        "latency": float(final_metrics["latency"]),
        "time_avg_utilization": time_avg_utilization,
        "end_utilization": end_utilization,
        "utilization": time_avg_utilization,
        "throughput": (post_drain_completed if drain_steps > 0 else episode_completed) / max(1, steps + drain_steps),
        "avg_efc": float(final_metrics.get("average_efc", (total_efc / max(1, efc_count)) if efc_count > 0 else 0.0)),
        "capacity_violations": cap_violations,
        "reward_mean": float(np.mean(marl_rewards)) if marl_rewards else 0.0,
        "accounting_valid": accounting_valid,
        "accounted_sum": accounted_sum,
        "q_queued": q_queued,
        "q_running": q_running,
    }


def run_marl_ablation(steps: int = 15, seed: int = 101):
    """Ablation: Compare Trained MARL against Untrained (Heuristic-Only) MARL on identical workloads."""
    print("\n" + "=" * 90)
    print("    MARL ABLATION EXPERIMENT: TRAINED RL (Q-TABLE) vs UNTRAINED (HEURISTIC-ONLY)")
    print(f"    Workload: Diverse Mixed (Scenario 5) | Steps: {steps} | Seed: {seed}")
    print("=" * 90)

    trace = generate_workload_trace("mixed", steps, seed, diverse_mix=True)

    # 1. Untrained MARL (Heuristic-only, Q=0.0)
    env_untrained = GPUEnvironment(scheduler="marl")
    env_untrained.enable_deterministic_physics()
    env_untrained.attach_workload_trace(deepcopy(trace))
    # Freeze Q-values to 0.0
    for ag in env_untrained.marl_manager.agents:
        ag.epsilon = 0.0
        ag.q_table.clear()
        ag.core_q_table.clear()
    env_untrained.learning_enabled = False

    decisions_untrained = []
    for _ in range(len(trace)):
        env_untrained.step()
        if hasattr(env_untrained, "last_marl_decision") and env_untrained.last_marl_decision:
            decisions_untrained.append(deepcopy(env_untrained.last_marl_decision))
    m_untrained = env_untrained.get_metrics()

    # 2. Trained MARL (Active Q-learning updates)
    env_trained = GPUEnvironment(scheduler="marl")
    env_trained.enable_deterministic_physics()
    env_trained.attach_workload_trace(deepcopy(trace))
    env_trained.learning_enabled = True

    decisions_trained = []
    for _ in range(len(trace)):
        env_trained.step()
        if hasattr(env_trained, "last_marl_decision") and env_trained.last_marl_decision:
            decisions_trained.append(deepcopy(env_trained.last_marl_decision))
    m_trained = env_trained.get_metrics()

    # Calculate Q-table statistics
    total_q_entries = sum(len(ag.q_table) for ag in env_trained.marl_manager.agents)
    total_core_q_entries = sum(len(ag.core_q_table) for ag in env_trained.marl_manager.agents)
    q_values = [v for ag in env_trained.marl_manager.agents for entry in ag.core_q_table.values() for v in entry if v != 0.0]
    avg_q = float(np.mean(q_values)) if q_values else 0.0
    min_q = float(np.min(q_values)) if q_values else 0.0
    max_q = float(np.max(q_values)) if q_values else 0.0

    # Decision divergence
    divergences = 0
    for d_u, d_t in zip(decisions_untrained, decisions_trained):
        if d_u.get("selected_gpu") != d_t.get("selected_gpu"):
            divergences += 1
    divergence_pct = (divergences / max(1, len(decisions_untrained))) * 100.0

    print(f"  Exact Tabular Q-Entries : {total_q_entries}")
    print(f"  Core Abstract Q-Entries : {total_core_q_entries}")
    print(f"  Core Q-Value Range      : [{min_q:.3f}, {max_q:.3f}] (Mean: {avg_q:.3f})")
    print(f"  Policy Decision Shifts  : {divergences}/{len(decisions_untrained)} ({divergence_pct:.1f}% divergence from pure heuristic)")
    print(f"  Completed (Untrained)   : {m_untrained['completed']} tasks")
    print(f"  Completed (Trained)     : {m_trained['completed']} tasks")
    print(f"  Latency (Untrained)     : {m_untrained['latency']:.2f}")
    print(f"  Latency (Trained)       : {m_trained['latency']:.2f}")
    print(f"  Avg EFC (Untrained)     : {m_untrained.get('average_efc', 0.0):.3f}")
    print(f"  Avg EFC (Trained)       : {m_trained.get('average_efc', 0.0):.3f}")
    print("=" * 90)


def run_benchmark(steps: int = 15, drain_steps: int = 10, num_seeds: int = 3):
    """Run full benchmark across all 7 scenarios with seed breakdown, mean, and std deviation."""
    seeds = [101 + i for i in range(num_seeds)]
    print("\n" + "=" * 115)
    print("    AETHERGRID BIT 6: RISK-AWARE MARL vs TRADITIONAL BASELINES BENCHMARK")
    print(f"    Active Window: {steps} steps | Drain Window: {drain_steps} steps | Seeds: {seeds}")
    print("    Metric Defs: Latency = Accumulated Queue Delay Score | Utilization = Time-Averaged % | Drain Rate = Total Post-Drain %")
    print("=" * 115)

    all_results = {}

    for s_idx, s_name in enumerate(SCENARIOS, 1):
        all_results[s_name] = {}
        # Log workload and risk conditions for this scenario using seed 101 as reference trace
        ref_env, ref_trace = setup_env_for_scenario(s_idx, "marl", steps, seeds[0])
        log_scenario_conditions(s_idx, ref_trace, seeds[0])

        # Collect runs across seeds
        scenario_seed_runs = {sched: [] for sched in SCHEDULERS}

        for sched in SCHEDULERS:
            for seed in seeds:
                res = run_scenario_episode(s_idx, sched, steps, seed, drain_steps=drain_steps)
                scenario_seed_runs[sched].append(res)

        # Print Per-Seed Breakdown
        print(f"\n---> Per-Seed Detailed Breakdown for {s_name}:")
        print(f"{'Seed':<6} | {'Scheduler':<13} | {'Ep Compl%':<10} | {'Drain Compl%':<12} | {'Compl':<7} | {'Failed':<8} | {'Lat Score':<10} | {'TimeUtil%':<10} | {'InstUtil%':<10} | {'Avg EFC':<8} | {'AccValid':<8}")
        print("-" * 115)
        for seed_idx, seed in enumerate(seeds):
            for sched in SCHEDULERS:
                r = scenario_seed_runs[sched][seed_idx]
                efc_str = f"{r['avg_efc']:.3f}"
                valid_str = "PASS" if r["accounting_valid"] else "FAIL"
                print(f"{seed:<6} | {sched:<13} | {r['episode_completion_rate']:>8.1f}% | {r['post_drain_completion_rate']:>10.1f}% | {r['completed']:>7.0f} | {r['failed_assignments']:>8.0f} | {r['latency']:>10.2f} | {r['time_avg_utilization']:>9.1f}% | {r['end_utilization']:>9.1f}% | {efc_str:>8} | {valid_str:>8}")

        # Compute Mean and Standard Deviation across seeds
        print(f"\n---> Summary Statistics (Mean ± Std Dev) for {s_name}:")
        print(f"{'Scheduler':<13} | {'Ep Compl Rate (%)':<18} | {'Drain Compl Rate (%)':<21} | {'Completed':<15} | {'Failed Retries':<16} | {'Lat Score':<15} | {'Time-Avg Util (%)':<18} | {'Avg EFC':<13} | {'Cap Viol'}")
        print("-" * 135)

        for sched in SCHEDULERS:
            runs = scenario_seed_runs[sched]
            ep_comp_rates = [r["episode_completion_rate"] for r in runs]
            drain_comp_rates = [r["post_drain_completion_rate"] for r in runs]
            compls = [r["completed"] for r in runs]
            fails = [r["failed_assignments"] for r in runs]
            lats = [r["latency"] for r in runs]
            utils = [r["time_avg_utilization"] for r in runs]
            efcs = [r["avg_efc"] for r in runs]
            caps = sum(r["capacity_violations"] for r in runs)

            ep_str = f"{np.mean(ep_comp_rates):.1f}% ± {np.std(ep_comp_rates):.1f}%"
            drain_str = f"{np.mean(drain_comp_rates):.1f}% ± {np.std(drain_comp_rates):.1f}%"
            compl_str = f"{np.mean(compls):.1f} ± {np.std(compls):.1f}"
            fail_str = f"{np.mean(fails):.1f} ± {np.std(fails):.1f}"
            lat_str = f"{np.mean(lats):.2f} ± {np.std(lats):.2f}"
            util_str = f"{np.mean(utils):.1f}% ± {np.std(utils):.1f}%"
            efc_mean_str = f"{np.mean(efcs):.3f} ± {np.std(efcs):.3f}"

            print(f"{sched:<13} | {ep_str:>18} | {drain_str:>21} | {compl_str:>15} | {fail_str:>16} | {lat_str:>15} | {util_str:>18} | {efc_mean_str:>13} | {caps:>8}")

            all_results[s_name][sched] = {
                "episode_completion_rate_mean": float(np.mean(ep_comp_rates)),
                "episode_completion_rate_std": float(np.std(ep_comp_rates)),
                "drain_completion_rate_mean": float(np.mean(drain_comp_rates)),
                "drain_completion_rate_std": float(np.std(drain_comp_rates)),
                "completed_mean": float(np.mean(compls)),
                "completed_std": float(np.std(compls)),
                "failed_mean": float(np.mean(fails)),
                "failed_std": float(np.std(fails)),
                "latency_mean": float(np.mean(lats)),
                "latency_std": float(np.std(lats)),
                "utilization_mean": float(np.mean(utils)),
                "utilization_std": float(np.std(utils)),
                "avg_efc_mean": float(np.mean(efcs)),
                "avg_efc_std": float(np.std(efcs)),
                "capacity_violations": caps
            }

    print("\n" + "=" * 115)
    print("BENCHMARK COMPLETE - All results recorded from real simulator execution.")
    print("=" * 115)
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=15, help="Active simulation steps per episode")
    parser.add_argument("--drain-steps", type=int, default=10, help="Post-arrival drain simulation steps")
    parser.add_argument("--trials", type=int, default=3, help="Number of random seed trials")
    parser.add_argument("--ablation", action="store_true", help="Run trained MARL vs heuristic-only ablation")
    args = parser.parse_args()

    if args.ablation:
        run_marl_ablation(steps=args.steps, seed=101)

    run_benchmark(steps=args.steps, drain_steps=args.drain_steps, num_seeds=args.trials)
