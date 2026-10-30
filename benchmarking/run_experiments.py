"""
Bit 9: Standalone Experiment Runner
Executes real, reproducible multi-algorithm benchmark and ablation study,
persisting all runs to benchmarking/results/
"""

import json
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarking.experiment_config import BenchmarkConfig, AblationConfig, ALL_SCHEDULERS
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.ablation_runner import AblationRunner
from benchmarking.results_store import ResultsStore


def run_all():
    store = ResultsStore()
    runner = BenchmarkRunner(results_store=store)
    ab_runner = AblationRunner(results_store=store)

    print("======================================================================")
    print("AETHERGRID BIT 9: RESEARCH METRICS & BENCHMARKING EXPERIMENTS")
    print("======================================================================")

    # 1. Multi-Algorithm Benchmark across all 6 Schedulers
    print("\n[Step 1/2] Executing 6-Algorithm Fair Comparison Benchmark...")
    print(f"Schedulers: {ALL_SCHEDULERS}")
    bench_cfg = BenchmarkConfig(
        name="Six-Algorithm Scientific Benchmark",
        schedulers=ALL_SCHEDULERS,
        num_tasks=35,
        num_repetitions=3,
        seeds=[42, 43, 44],
        workload_type="mixed",
        max_steps_per_run=60,
        description="Controlled comparison of FCFS, Round Robin, Least Loaded, RL, MARL, and Risk-Aware MARL under identical seeded workloads.",
    )

    bench_res = runner.run_benchmark(bench_cfg)
    exp_id = bench_res["experiment_id"]
    print(f"Benchmark completed successfully! Experiment ID: {exp_id}")
    print(f"Saved artifacts:")
    for k, v in bench_res["artifacts"].items():
        if v:
            print(f"  - {k}: {v}")

    # Print summary table
    print("\n----------------------------------------------------------------------")
    print("BENCHMARK RESULTS SUMMARY (Across 3 seeds)")
    print("----------------------------------------------------------------------")
    print(f"{'Scheduler':<25} | {'Throughput':<12} | {'Success Rate':<12} | {'Avg Turnaround':<15} | {'Avg Risk':<10} | {'Avg EFC':<10}")
    print("-" * 95)
    for s_id, s_summary in bench_res["summaries"].items():
        m = s_summary["metrics"]
        name = s_summary["scheduler_name"]
        tp = f"{m['throughput_tasks_per_sec']['mean']:.2f} t/s"
        succ = f"{m['success_rate_pct']['mean']:.1f}%"
        turn = f"{m['avg_turnaround_time_sec']['mean']:.2f}s"
        risk = f"{m['avg_risk_exposure']['mean']:.4f}"
        efc = f"{m['avg_efc']['mean']:.4f}"
        print(f"{name:<25} | {tp:<12} | {succ:<12} | {turn:<15} | {risk:<10} | {efc:<10}")

    # 2. Ablation Study
    print("\n[Step 2/2] Executing Risk-Aware MARL Ablation Study...")
    ab_cfg = AblationConfig(
        name="Risk & EFC Penalty Ablation Study",
        num_tasks=35,
        num_repetitions=3,
        seeds=[101, 102, 103],
        workload_type="mixed",
        max_steps_per_run=60,
        ablation_modes=["full_risk_aware", "no_risk_penalty", "no_efc_penalty", "risk_blind"],
    )

    ab_res = ab_runner.run_ablation_study(ab_cfg)
    study_id = ab_res["study_id"]
    print(f"Ablation study completed successfully! Study ID: {study_id}")
    print(f"Saved artifacts:")
    for k, v in ab_res["artifacts"].items():
        if v:
            print(f"  - {k}: {v}")

    print("\n----------------------------------------------------------------------")
    print("ABLATION STUDY SUMMARY (Across 3 seeds)")
    print("----------------------------------------------------------------------")
    print(f"{'Ablation Mode':<30} | {'Throughput':<12} | {'Success Rate':<12} | {'Avg Risk':<10} | {'Avg EFC':<10}")
    print("-" * 85)
    for mode, m_summary in ab_res["summaries"].items():
        m = m_summary["metrics"]
        name = m_summary["scheduler_name"]
        tp = f"{m['throughput_tasks_per_sec']['mean']:.2f} t/s"
        succ = f"{m['success_rate_pct']['mean']:.1f}%"
        risk = f"{m['avg_risk_exposure']['mean']:.4f}"
        efc = f"{m['avg_efc']['mean']:.4f}"
        print(f"{name:<30} | {tp:<12} | {succ:<12} | {risk:<10} | {efc:<10}")

    print("\n======================================================================")
    print("ALL EXPERIMENTS COMPLETED WITH REAL MEASUREMENTS PERSISTED!")
    print("======================================================================")
    return bench_res, ab_res


if __name__ == "__main__":
    run_all()
