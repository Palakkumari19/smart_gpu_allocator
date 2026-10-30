"""
Empirical Risk and Workload Benchmark Runner.

Executes scientifically rigorous, paired comparisons across 6 schedulers:
  - FCFS ('fcfs')
  - Round Robin ('round_robin')
  - Least Loaded ('least_loaded')
  - Centralized Single-Agent RL ('rl')
  - Risk-Blind MARL ('marl')
  - Risk-Aware MARL ('risk_aware_marl')

Key Scientific Protocol Rules:
1. Pre-registered hypotheses (PREREGISTRATION.md):
   Primary metrics: avg_risk_exposure, terminal_efc, crash_count, quarantine_time (lower is better).
   Secondary metric: completed_tasks (higher is better).
2. ADD-ONLY: Reuses existing interfaces (GPUEnvironment, EnhancedTask, scheduler_adapters.py,
   evaluation/stats.py) without touching original codebase.
3. Strict Pairing: For every condition (window x load_preset x seed), ONE master trace is built
   and evaluated across all 6 schedulers on fresh GPUEnvironment instances with identical seeds.
4. Clean Generalization: RL and MARL are trained exclusively on disjoint TRAIN windows (earlier days).
   Q-tables are snapshotted and frozen (set_eval_mode(), epsilon=0.0) during EVAL window runs.
5. Physics Determinism: enable_deterministic_physics() is invoked on every instance.
6. Empirical Hazard Overlay: Optionally applies calibrated empirical failure probabilities
   from real trace data (hazard.py and risk_overlay.py) identically across all schedulers.
7. Excludes failure_rate_pct and rejection_rate_pct per NOTES.md (queue buffering disparity).
"""

import argparse
import json
import random
import time
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd

from benchmarking.scheduler_adapters import get_scheduler_adapter
from environment import GPUEnvironment
from evaluation.stats import (
    paired_diff_ci,
    win_rate,
    ci_excludes_zero,
    compare_against_baseline,
)
from realdata.config import (
    BURSTGPT_WINDOWS_PATH,
    RESULTS_DIR,
    HAZARD_CURVES_PATH,
)
from realdata.hazard import HazardCatalog, load_hazard_curves
from realdata.loaders import load_burstgpt_data, load_uci_video_data
from realdata.risk_overlay import RiskHazardOverlay, step_with_hazard_overlay
from realdata.trace_builder import build_real_trace, compute_target_load_scale
from realdata.windows import TraceWindowsSummary, WindowMetadata


SCHEDULERS = [
    "fcfs",
    "round_robin",
    "least_loaded",
    "rl",
    "marl",
    "risk_aware_marl",
]

PRIMARY_METRICS = [
    "avg_risk_exposure",
    "terminal_efc",
    "crash_count",
    "quarantine_time",
]

SECONDARY_METRICS = [
    "completed_tasks",
]

HIGHER_IS_BETTER = {"completed_tasks", "throughput"}


def find_strongest_baseline(
    records: Union[List[Dict[str, Any]], pd.DataFrame], metric: str
) -> str:
    """Identify the best-performing baseline scheduler for a given metric."""
    if isinstance(records, list):
        df = pd.DataFrame(records)
    else:
        df = records

    higher_better = metric in HIGHER_IS_BETTER
    baseline_df = df[df["scheduler"] != "risk_aware_marl"]
    grouped = baseline_df.groupby("scheduler")[metric].mean()

    if higher_better:
        return str(grouped.idxmax())
    else:
        return str(grouped.idxmin())


def train_learners_on_train_window(
    train_window: WindowMetadata,
    train_steps: int = 100,
    seed: int = 42,
    df_burst: Optional[pd.DataFrame] = None,
    df_video: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Train MARL and Single-Agent RL agents exclusively on disjoint TRAIN window data.

    Snapshots learned Q-tables and returns them for frozen evaluation.
    """
    print(f"\n[Training] Synthesizing TRAIN trace from window '{train_window.window_id}' ({train_steps} steps, seed={seed}) ...")

    # Generate training trace with 0.9 target utilization
    load_scale = compute_target_load_scale(
        target_utilization=0.9,
        raw_requests_count=train_window.request_count,
        window_duration_seconds=train_window.duration_seconds,
        seconds_per_step=1.0,
    )

    train_trace = build_real_trace(
        window=train_window,
        seconds_per_step=1.0,
        load_scale=load_scale,
        video_fraction=0.2,
        seed=seed,
        df_burst=df_burst,
        df_video=df_video,
        total_steps=train_steps,
        partition="train",
    )

    # 1. Train MARL Agents
    print("[Training] Training multi-agent MARL on TRAIN trace ...")
    random.seed(seed)
    np.random.seed(seed)
    marl_env = GPUEnvironment(scheduler="marl")
    marl_env.enable_deterministic_physics()
    marl_env.attach_workload_trace(deepcopy(train_trace))
    for _ in range(len(train_trace)):
        marl_env.step()
    marl_q = marl_env.marl_manager.snapshot_q_tables()

    # 2. Train Centralized Single-Agent RL
    print("[Training] Training centralized single-agent RL on TRAIN trace ...")
    random.seed(seed)
    np.random.seed(seed)
    rl_env = GPUEnvironment(scheduler="rl")
    rl_env.enable_deterministic_physics()
    rl_env.attach_workload_trace(deepcopy(train_trace))
    for _ in range(len(train_trace)):
        rl_env.step()
    rl_q = rl_env.agent.snapshot_q_table()

    print("[Training] Q-tables successfully snapshotted. Frozen for evaluation.")
    return marl_q, rl_q


def run_single_episode(
    scheduler_id: str,
    trace: List[List[Any]],
    marl_q: Optional[Dict[str, Any]] = None,
    rl_q: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    overlay: Optional[RiskHazardOverlay] = None,
) -> Dict[str, float]:
    """
    Execute one evaluation episode on a fresh GPUEnvironment instance.
    Optionally applies external empirical hazard overlay identically.
    """
    # Strict per-episode seeding
    random.seed(seed)
    np.random.seed(seed)

    env = GPUEnvironment(scheduler="fcfs")
    env.enable_deterministic_physics()

    # Configure scheduler adapter
    adapter = get_scheduler_adapter(scheduler_id)
    adapter.configure_environment(env)
    adapter.reset_state(env)

    # Restore pre-trained Q-tables
    if scheduler_id in ("marl", "risk_aware_marl") and marl_q is not None:
        env.marl_manager.restore_q_tables(deepcopy(marl_q))
    if scheduler_id == "rl" and rl_q is not None:
        env.agent.restore_q_table(deepcopy(rl_q))

    # Freeze learning and exploratory epsilon
    env.set_eval_mode()

    # Attach deep copy of the condition trace
    env.attach_workload_trace(deepcopy(trace))

    # Run discrete simulation steps with optional empirical hazard overlay
    steps = len(trace)
    for _ in range(steps):
        step_with_hazard_overlay(env, overlay)

    # Extract metrics (excluding percentage rejection/failure rates per pre-registration)
    raw_m = env.get_metrics()
    return {
        "avg_risk_exposure": float(raw_m.get("average_gpu_risk", 0.0)),
        "terminal_efc": float(raw_m.get("total_assigned_efc", 0.0)),
        "crash_count": float(raw_m.get("total_crash_events", 0.0)),
        "quarantine_time": float(raw_m.get("quarantined_gpus", 0.0)),
        "completed_tasks": float(raw_m.get("completed", 0.0)),
        "latency": float(raw_m.get("latency", 0.0)),
        "throughput": float(raw_m.get("throughput", 0.0)),
        "gpu_utilization": float(raw_m.get("gpu_utilization", 0.0)),
        "avg_memory_usage": float(raw_m.get("avg_memory_usage", raw_m.get("gpu_utilization", 0.0))),
    }


def compute_paired_comparison_stats(
    treatment_values: List[float],
    baseline_values: List[float],
    higher_is_better: bool = False,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute paired difference bootstrap confidence interval and win rate.
    Reuses evaluation/stats.py compare_against_baseline directly.
    """
    return compare_against_baseline(
        treatment_values=treatment_values,
        baseline_values=baseline_values,
        higher_is_better=higher_is_better,
        seed=seed,
    )


def run_benchmark_experiment(
    windows: Optional[List[str]] = None,
    load_presets: Optional[List[float]] = None,
    seeds: Optional[List[int]] = None,
    steps_per_window: int = 120,
    train_steps: int = 80,
    train_seed: int = 42,
    smoke_test: bool = False,
    seconds_per_step: float = 1.0,
    use_hazard_overlay: bool = False,
    hazard_curve_name: str = "burstgpt_dispersion",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run empirical real-data benchmark across condition matrix.
    """
    print("\n" + "=" * 70)
    print("STARTING REAL-DATA RISK EVALUATION PIPELINE")
    print("=" * 70)

    # Load input trace datasets
    print("Loading empirical traces (BurstGPT and UCI Video) ...")
    df_burst = load_burstgpt_data()
    df_video = load_uci_video_data()

    # Load canonical windows summary
    windows_summary = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    eval_windows = windows_summary.eval_windows

    # Optional hazard curve overlay loading
    hazard_curve = None
    if use_hazard_overlay:
        if HAZARD_CURVES_PATH.exists():
            catalog = load_hazard_curves(HAZARD_CURVES_PATH)
            hazard_curve = catalog.get_curve(hazard_curve_name)
            print(f"[Overlay] Loaded empirical hazard curve '{hazard_curve.name}' from {HAZARD_CURVES_PATH}")
        else:
            print("[Overlay] Hazard curves JSON not found; generating on the fly...")
            from realdata.hazard import build_hazard_catalog, save_hazard_curves
            catalog = build_hazard_catalog(df_burst)
            save_hazard_curves(catalog)
            hazard_curve = catalog.get_curve(hazard_curve_name)

    # Configure evaluation grid
    if smoke_test:
        eval_window_keys = ["burstiest_window"]
        presets = [0.9]
        active_seeds = [42, 43]
        eval_steps = min(steps_per_window, 20)
        t_steps = min(train_steps, 25)
        print("\n" + "=" * 70)
        print("RUNNING 1-WINDOW REAL-DATA SMOKE TEST (Fast Verification)")
        print(f"Window: {eval_window_keys[0]} | Load: {presets[0]} | Seeds: {active_seeds} | Steps: {eval_steps}")
        print(f"Hazard Overlay: {'ENABLED (' + hazard_curve_name + ')' if use_hazard_overlay else 'DISABLED (Workload Only)'}")
        print("=" * 70 + "\n")
    else:
        eval_window_keys = windows or ["median_hour", "burstiest_window"]
        presets = load_presets or [0.6, 0.9, 1.2]
        active_seeds = seeds or [42, 43, 44, 45, 46]
        eval_steps = steps_per_window
        t_steps = train_steps
        print("\n" + "=" * 70)
        print("RUNNING FULL REAL-DATA RISK BENCHMARK")
        print(f"Windows: {eval_window_keys} | Loads: {presets} | Seeds: {len(active_seeds)} | Steps: {eval_steps}")
        print(f"Hazard Overlay: {'ENABLED (' + hazard_curve_name + ')' if use_hazard_overlay else 'DISABLED (Workload Only)'}")
        print("=" * 70 + "\n")

    # Train on disjoint TRAIN window
    marl_q, rl_q = train_learners_on_train_window(
        train_window=windows_summary.train_windows["median_hour"],
        train_steps=t_steps,
        seed=train_seed,
        df_burst=df_burst,
        df_video=df_video,
    )

    run_records: List[Dict[str, Any]] = []
    total_conditions = len(eval_window_keys) * len(presets) * len(active_seeds)
    condition_counter = 0

    start_time = time.perf_counter()

    for win_key in eval_window_keys:
        meta = eval_windows.get(win_key)
        if meta is None:
            continue

        for preset in presets:
            load_scale = compute_target_load_scale(
                target_utilization=preset,
                raw_requests_count=meta.request_count,
                window_duration_seconds=meta.duration_seconds,
                seconds_per_step=seconds_per_step,
            )

            for seed in active_seeds:
                condition_counter += 1
                cond_id = f"{win_key}_rho{int(preset * 100)}_s{seed}"
                print(f"[{condition_counter}/{total_conditions}] Generating master trace for {cond_id} (scale={load_scale:.4f}) ...")

                # Generate ONE master trace for this condition
                trace = build_real_trace(
                    window=meta,
                    seconds_per_step=seconds_per_step,
                    load_scale=load_scale,
                    video_fraction=0.2,
                    seed=seed,
                    df_burst=df_burst,
                    df_video=df_video,
                    total_steps=eval_steps,
                    partition="eval",
                )
                tasks_in_trace = sum(len(s) for s in trace)

                # Configure hazard overlay for this condition if enabled
                overlay = None
                if use_hazard_overlay and hazard_curve is not None:
                    overlay = RiskHazardOverlay(
                        hazard_curve=hazard_curve,
                        enabled=True,
                        seed=seed,
                        operating_metric_value=meta.dispersion_index,
                    )

                # Run each scheduler on a deep copy of the identical trace
                for scheduler in SCHEDULERS:
                    overlay_inst = deepcopy(overlay) if overlay is not None else None
                    metrics = run_single_episode(
                        scheduler_id=scheduler,
                        trace=trace,
                        marl_q=marl_q,
                        rl_q=rl_q,
                        seed=seed,
                        overlay=overlay_inst,
                    )

                    record = {
                        "condition_id": cond_id,
                        "window": win_key,
                        "target_utilization": preset,
                        "load_scale": load_scale,
                        "seed": seed,
                        "scheduler": scheduler,
                        "tasks_offered": tasks_in_trace,
                        "hazard_overlay": use_hazard_overlay,
                        **metrics,
                    }
                    run_records.append(record)

    elapsed_sec = time.perf_counter() - start_time
    runs_df = pd.DataFrame(run_records)

    print("\n" + "=" * 70)
    print(f"BENCHMARK COMPLETED: {len(runs_df)} episodes across {condition_counter} paired conditions in {elapsed_sec:.2f}s")
    print("=" * 70 + "\n")

    # Evaluate Pre-Registered Statistical Hypotheses
    print("=" * 70)
    print("PRE-REGISTERED HYPOTHESIS EVALUATION: Risk-Aware MARL vs Baselines")
    print("=" * 70)

    comparisons: Dict[str, Any] = {}
    metrics_to_test = PRIMARY_METRICS + SECONDARY_METRICS

    for metric in metrics_to_test:
        higher_better = metric in SECONDARY_METRICS

        # Find strongest baseline for this metric across all runs
        baseline_means = {}
        for s in SCHEDULERS:
            if s == "risk_aware_marl":
                continue
            vals = runs_df[runs_df["scheduler"] == s][metric].values
            baseline_means[s] = float(np.mean(vals)) if len(vals) > 0 else 0.0

        if not baseline_means:
            continue

        if higher_better:
            strongest = max(baseline_means.keys(), key=lambda k: baseline_means[k])
        else:
            strongest = min(baseline_means.keys(), key=lambda k: baseline_means[k])

        treatment_vals = runs_df[runs_df["scheduler"] == "risk_aware_marl"][metric].values
        baseline_vals = runs_df[runs_df["scheduler"] == strongest][metric].values

        stat_result = compute_paired_comparison_stats(
            treatment_values=treatment_vals.tolist(),
            baseline_values=baseline_vals.tolist(),
            higher_is_better=higher_better,
            confidence=0.95,
            n_resamples=2000,
            seed=0,
        )
        stat_result["strongest_baseline"] = strongest
        stat_result["treatment_mean"] = float(np.mean(treatment_vals))
        stat_result["baseline_mean"] = float(np.mean(baseline_vals))
        comparisons[metric] = stat_result

        diff_dir = "decrease" if not higher_better else "increase"
        sig_tag = " [SIGNIFICANT]" if stat_result["significant_better"] else (" [INFERIOR]" if stat_result["significant_worse"] else " [TIE / NEUTRAL]")
        print(f"\nMetric: {metric} ({'Lower is better' if not higher_better else 'Higher is better'}){sig_tag}")
        print(f"  Strongest Baseline: {strongest} (Mean: {stat_result['baseline_mean']:.4f})")
        print(f"  Risk-Aware MARL:     Mean: {stat_result['treatment_mean']:.4f}")
        print(f"  Paired Mean Diff:    {stat_result['mean_diff']:.4f} {diff_dir}")
        print(f"  Bootstrap 95% CI:    [{stat_result['ci_low']:.4f}, {stat_result['ci_high']:.4f}]")
        print(f"  Paired Win Rate:     {stat_result['win_rate'] * 100:.1f}% ({stat_result['wins']}W / {stat_result['ties']}T / {stat_result['losses']}L)")

    # Save artifact files to realdata/results/
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    mode_prefix = "overlay" if use_hazard_overlay else "workload"
    runs_csv_path = RESULTS_DIR / f"realdata_runs_{mode_prefix}_{timestamp}.csv"
    summary_json_path = RESULTS_DIR / f"realdata_summary_{mode_prefix}_{timestamp}.json"

    runs_df.to_csv(runs_csv_path, index=False)

    summary_data = {
        "timestamp_utc": timestamp,
        "smoke_test": smoke_test,
        "hazard_overlay_enabled": use_hazard_overlay,
        "hazard_curve_name": hazard_curve_name if use_hazard_overlay else None,
        "elapsed_seconds": round(elapsed_sec, 2),
        "total_episodes": len(runs_df),
        "windows_evaluated": eval_window_keys,
        "load_presets": presets,
        "seeds": active_seeds,
        "eval_steps_per_episode": eval_steps,
        "schedulers_evaluated": SCHEDULERS,
        "comparisons_vs_strongest_baseline": comparisons,
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print("\n" + "=" * 70)
    print("RESULTS SUCCESSFULLY SAVED")
    print(f"  Per-run CSV:     {runs_csv_path}")
    print(f"  Summary JSON:    {summary_json_path}")
    print(f"  Total Duration:  {elapsed_sec:.2f} seconds")
    print("=" * 70 + "\n")

    return runs_df, summary_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Data Risk Benchmark Runner")
    parser.add_argument("--smoke", action="store_true", help="Run fast 1-window smoke test")
    parser.add_argument("--steps", type=int, default=120, help="Steps per evaluation episode")
    parser.add_argument("--train-steps", type=int, default=80, help="Steps for TRAIN window Q-learning")
    parser.add_argument("--with-hazard-overlay", action="store_true", help="Enable empirical hazard overlay")
    parser.add_argument("--hazard-curve", type=str, default="burstgpt_dispersion", help="Hazard curve to use")
    args = parser.parse_args()

    run_benchmark_experiment(
        smoke_test=args.smoke,
        steps_per_window=args.steps,
        train_steps=args.train_steps,
        use_hazard_overlay=args.with_hazard_overlay,
        hazard_curve_name=args.hazard_curve,
    )
