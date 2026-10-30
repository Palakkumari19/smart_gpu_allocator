"""Fair, paired evaluation of traditional, RL, and MARL schedulers.

This is the experiment that can support (or reject) a superiority claim.
It does not add MARL-only score bonuses, does not use wall-clock throughput,
and replays identical arrivals for every scheduler.
"""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime, timezone

import numpy as np

from environment import GPUEnvironment
from evaluation.stats import compare_against_baseline, marl_consistency_verdict, mean_std
from evaluation.workload_trace import generate_workload_trace, trace_task_count


SCHEDULERS = ["random", "round_robin", "fcfs", "least_loaded", "rl", "marl"]
PRIMARY_METRIC = "completed"
HIGHER_IS_BETTER = {
    "completed": True,
    "sim_throughput": True,
    "successful_assignments": True,
    "crash_events": False,
    "failed_assignments": False,
    "latency": False,
    "preemptions": False,
}


def collect_run_metrics(env, steps):
    metrics = env.get_metrics()
    completed = float(metrics["completed"])
    return {
        "completed": completed,
        "crash_events": float(metrics.get("total_crash_events", 0)),
        "failed_assignments": float(metrics.get("failed_assignments", 0)),
        "successful_assignments": float(metrics.get("successful_assignments", 0)),
        "latency": float(metrics["latency"]),
        "preemptions": float(metrics["total_preemptions"]),
        "avg_memory_usage": float(metrics["avg_memory_usage"]),
        "avg_temperature": float(metrics["avg_temperature"]),
        "avg_fragmentation": float(metrics["avg_fragmentation"]),
        "queued_remaining": float(metrics["total_queued_tasks"]),
        "tasks_submitted": float(metrics.get("total_tasks_submitted", env.total_tasks)),
        "sim_throughput": completed / max(1, steps),
        "completion_ratio": completed / max(1.0, float(metrics.get("total_tasks_submitted", env.total_tasks))),
    }


def run_episode(scheduler, trace, learning=True, marl_q=None, rl_q=None):
    env = GPUEnvironment(scheduler=scheduler)
    env.enable_deterministic_physics()
    env.attach_workload_trace(deepcopy(trace))
    if marl_q is not None:
        env.marl_manager.restore_q_tables(deepcopy(marl_q))
    if rl_q is not None:
        env.agent.restore_q_table(deepcopy(rl_q))
    if not learning:
        env.set_eval_mode()
    for _ in range(len(trace)):
        env.step()
    return env, collect_run_metrics(env, len(trace))


def train_learners(train_steps, train_seed):
    trace = generate_workload_trace("mixed", train_steps, train_seed)
    marl_env, _ = run_episode("marl", trace, learning=True)
    rl_env, _ = run_episode("rl", trace, learning=True)
    return marl_env.marl_manager.snapshot_q_tables(), rl_env.agent.snapshot_q_table()


def aggregate_runs(run_rows):
    metrics = [key for key in HIGHER_IS_BETTER if key in run_rows[0]]
    summary = {}
    for metric in metrics:
        values = [row[metric] for row in run_rows]
        mean, std = mean_std(values)
        summary[metric] = {"mean": mean, "std": std, "values": values}
    return summary


def strongest_baseline(completed_by_scheduler, candidates):
    return max(candidates, key=lambda name: float(np.mean(completed_by_scheduler[name])))


def analyze_protocol(results_by_scenario, baseline_candidates):
    """results_by_scenario[scenario][scheduler] = list of metric dicts, aligned by seed."""
    scenario_comparisons = {}
    all_marl = []
    all_baseline = []
    strongest_names = {}

    for scenario, by_scheduler in results_by_scenario.items():
        completed = {name: [row["completed"] for row in rows] for name, rows in by_scheduler.items()}
        strongest = strongest_baseline(completed, baseline_candidates)
        strongest_names[scenario] = strongest
        comparison = compare_against_baseline(completed["marl"], completed[strongest], higher_is_better=True)
        comparison["baseline"] = strongest
        scenario_comparisons[scenario] = comparison
        all_marl.extend(completed["marl"])
        all_baseline.extend(completed[strongest])

    overall = compare_against_baseline(all_marl, all_baseline, higher_is_better=True)
    overall["baseline"] = "per-scenario-strongest"
    scenario_comparisons["_overall"] = overall
    verdict = marl_consistency_verdict(scenario_comparisons, overall["win_rate"])
    verdict["strongest_baseline_by_scenario"] = strongest_names
    return scenario_comparisons, verdict


def format_mean(summary, metric):
    stats = summary[metric]
    return f"{stats['mean']:.2f} ± {stats['std']:.2f}"


def print_report(protocol_name, summaries, comparisons, verdict):
    print("\n" + "=" * 88)
    print(f"PROTOCOL: {protocol_name}")
    print("=" * 88)
    print(f"Primary metric: {PRIMARY_METRIC} (higher is better)")
    print("No MARL-only bonus. Throughput is completed tasks / simulation steps.")
    print()
    for scenario, by_scheduler in summaries.items():
        print(f"--- {scenario} ---")
        header = f"{'scheduler':<16}{'completed':>16}{'crashes':>16}{'fail_assign':>16}{'sim_tput':>14}"
        print(header)
        for scheduler, summary in by_scheduler.items():
            print(
                f"{scheduler:<16}"
                f"{format_mean(summary, 'completed'):>16}"
                f"{format_mean(summary, 'crash_events'):>16}"
                f"{format_mean(summary, 'failed_assignments'):>16}"
                f"{format_mean(summary, 'sim_throughput'):>14}"
            )
        print()

    print("Paired MARL vs strongest baseline (completed tasks)")
    for scenario, row in comparisons.items():
        print(
            f"  {scenario:<12} baseline={row['baseline']:<16} "
            f"Δ={row['mean_diff']:+.2f}  CI[{row['ci_low']:+.2f}, {row['ci_high']:+.2f}]  "
            f"win_rate={row['win_rate']:.1%}  "
            f"{'SIG+' if row['significant_better'] else 'SIG-' if row['significant_worse'] else 'n.s.'}"
        )
    print()
    print(f"VERDICT: {verdict['status']}")
    print(verdict["summary"])


def run_scientific_evaluation(
    steps=60,
    runs=12,
    seed=42,
    train_steps=400,
    scenarios=("light", "mixed", "heavy", "burst"),
    schedulers=None,
):
    schedulers = list(schedulers or SCHEDULERS)
    baseline_candidates = [name for name in schedulers if name not in ("marl", "rl")]
    if not baseline_candidates:
        baseline_candidates = [name for name in schedulers if name != "marl"]

    print("ATHERGRID SCIENTIFIC EVALUATION")
    print("Paired traces, deterministic GPU physics, freeze-eval for learned policies")
    print(f"scenarios={list(scenarios)}  steps={steps}  runs={runs}  train_steps={train_steps}")

    traces = {}
    for scenario in scenarios:
        traces[scenario] = [
            generate_workload_trace(scenario, steps, seed + run_idx)
            for run_idx in range(runs)
        ]

    print("\nTraining RL and MARL on a held-out mixed trace...")
    marl_q, rl_q = train_learners(train_steps, seed + 10_000)

    protocols = {
        "online_learning": {"learning": True, "marl_q": None, "rl_q": None},
        "pretrained_eval": {"learning": False, "marl_q": marl_q, "rl_q": rl_q},
    }

    full_results = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "steps": steps,
            "runs": runs,
            "seed": seed,
            "train_steps": train_steps,
            "scenarios": list(scenarios),
            "schedulers": schedulers,
            "primary_metric": PRIMARY_METRIC,
        },
        "protocols": {},
    }

    for protocol_name, protocol in protocols.items():
        results_by_scenario = {scenario: {name: [] for name in schedulers} for scenario in scenarios}
        summaries = {scenario: {} for scenario in scenarios}
        for scenario in scenarios:
            for run_idx, trace in enumerate(traces[scenario]):
                submitted = trace_task_count(trace)
                for scheduler in schedulers:
                    _, metrics = run_episode(
                        scheduler,
                        trace,
                        learning=protocol["learning"] if scheduler in ("rl", "marl") else False,
                        marl_q=protocol["marl_q"] if scheduler == "marl" else None,
                        rl_q=protocol["rl_q"] if scheduler == "rl" else None,
                    )
                    metrics["tasks_submitted"] = float(submitted)
                    metrics["completion_ratio"] = metrics["completed"] / max(1.0, submitted)
                    results_by_scenario[scenario][scheduler].append(metrics)
            for scheduler in schedulers:
                summaries[scenario][scheduler] = aggregate_runs(results_by_scenario[scenario][scheduler])
        comparisons, verdict = analyze_protocol(results_by_scenario, baseline_candidates)
        print_report(protocol_name, summaries, comparisons, verdict)
        full_results["protocols"][protocol_name] = {
            "summaries": {
                scenario: {
                    scheduler: {metric: {"mean": vals["mean"], "std": vals["std"]} for metric, vals in sched_summary.items()}
                    for scheduler, sched_summary in scenario_summary.items()
                }
                for scenario, scenario_summary in summaries.items()
            },
            "comparisons": comparisons,
            "verdict": verdict,
            "raw": results_by_scenario,
        }

    return full_results


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    def _convert(obj):
        if isinstance(obj, dict):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(_convert(results), handle, indent=2)
    print(f"\nWrote {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run the AtherGrid scientific scheduler evaluation")
    parser.add_argument("--quick", action="store_true", help="Short smoke evaluation")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quick:
        steps = args.steps or 20
        runs = args.runs or 4
        train_steps = args.train_steps or 40
        scenarios = ("mixed", "heavy")
    else:
        steps = args.steps or 60
        runs = args.runs or 12
        train_steps = args.train_steps or 400
        scenarios = ("light", "mixed", "heavy", "burst")

    results = run_scientific_evaluation(
        steps=steps,
        runs=runs,
        seed=args.seed,
        train_steps=train_steps,
        scenarios=scenarios,
    )
    out_path = os.path.join("evaluation", "results", "scientific_evaluation.json")
    save_results(results, out_path)


if __name__ == "__main__":
    main()
