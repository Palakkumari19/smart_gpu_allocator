"""
Bit 9: Research Metrics Calculator & Statistical Analyzer
Computes performance, reliability, safety, and risk-aware metrics with
repetition-level statistics and 95% confidence intervals.
"""

from dataclasses import dataclass, field, asdict
import math
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class RunMetrics:
    """Metrics recorded for a single benchmark run."""
    run_id: str
    scheduler_id: str
    scheduler_name: str
    seed: int
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    rejected_tasks: int
    unfinished_tasks: int
    success_rate_pct: float
    failure_rate_pct: float
    rejection_rate_pct: float
    makespan_steps: int
    elapsed_seconds: float
    throughput_tasks_per_sec: float
    avg_completion_time_sec: float
    avg_waiting_time_sec: float
    avg_turnaround_time_sec: float
    avg_gpu_utilization_pct: float
    avg_gpu_memory_pct: float
    load_balance_std: float
    jains_fairness_index: float
    capacity_violations: int
    scheduler_exceptions: int
    cooldown_events: int
    quarantine_events: int
    avg_risk_exposure: float
    max_risk_exposure: float
    avg_efc: float
    total_efc: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AggregateMetric:
    """Statistical summary of a metric across multiple seeded repetitions."""
    mean: float
    std: float
    ci_95_lower: float
    ci_95_upper: float
    min_val: float
    max_val: float
    sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark metrics across all repetitions for a scheduler."""
    scheduler_id: str
    scheduler_name: str
    repetitions: int
    seeds: List[int]
    total_tasks_evaluated: int
    metrics: Dict[str, AggregateMetric] = field(default_factory=dict)
    improvements_vs_baselines: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scheduler_id": self.scheduler_id,
            "scheduler_name": self.scheduler_name,
            "repetitions": self.repetitions,
            "seeds": self.seeds,
            "total_tasks_evaluated": self.total_tasks_evaluated,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "improvements_vs_baselines": self.improvements_vs_baselines,
        }


class MetricsCalculator:
    """Calculates granular run metrics and statistical summaries across runs."""

    @staticmethod
    def calculate_run_metrics(
        run_id: str,
        scheduler_id: str,
        scheduler_name: str,
        seed: int,
        env,
        total_stream_tasks: int,
        elapsed_seconds: float,
        step_count: int,
        capacity_violations: int = 0,
        scheduler_exceptions: int = 0,
    ) -> RunMetrics:
        """Calculate complete metrics for one simulation run."""
        # Task accounting
        completed = sum(gpu.completed_tasks for gpu in env.gpus)
        failed = getattr(env, "failed_assignments", 0)
        rejected = getattr(env, "rejected_tasks", 0)
        unfinished = max(0, total_stream_tasks - (completed + failed + rejected))

        # Rates
        total_denom = max(1, total_stream_tasks)
        success_rate = round((completed / total_denom) * 100.0, 2)
        failure_rate = round((failed / total_denom) * 100.0, 2)
        rejection_rate = round((rejected / total_denom) * 100.0, 2)

        # Durations from task queue metadata
        completion_times = []
        waiting_times = []
        turnaround_times = []

        all_tasks = env.task_queue.get_all_tasks()
        for task_id, (task, meta) in all_tasks.items():
            if meta.completion_time and meta.start_time:
                dur = max(0.0, meta.completion_time - meta.start_time)
                completion_times.append(dur)
            if meta.start_time and meta.arrival_time:
                wait = max(0.0, meta.start_time - meta.arrival_time)
                waiting_times.append(wait)
            if meta.completion_time and meta.arrival_time:
                turn = max(0.0, meta.completion_time - meta.arrival_time)
                turnaround_times.append(turn)

        avg_comp = float(np.mean(completion_times)) if completion_times else 0.0
        avg_wait = float(np.mean(waiting_times)) if waiting_times else 0.0
        avg_turn = float(np.mean(turnaround_times)) if turnaround_times else 0.0

        # Throughput
        safe_elapsed = max(0.001, elapsed_seconds)
        throughput = round(completed / safe_elapsed, 2)

        # GPU utilization & memory
        gpu_utils = [float(g.utilization) for g in env.gpus]
        gpu_mems = [float(g.get_memory_usage_percentage()) for g in env.gpus]
        avg_util = float(np.mean(gpu_utils)) if gpu_utils else 0.0
        avg_mem = float(np.mean(gpu_mems)) if gpu_mems else 0.0

        # Load balance across GPUs
        tasks_per_gpu = [gpu.completed_tasks for gpu in env.gpus]
        load_balance_std = float(np.std(tasks_per_gpu)) if tasks_per_gpu else 0.0

        # Jain's fairness index: (sum(x_i))^2 / (n * sum(x_i^2))
        sum_x = sum(tasks_per_gpu)
        sum_sq_x = sum(x ** 2 for x in tasks_per_gpu)
        n_gpus = len(tasks_per_gpu)
        if sum_sq_x > 0:
            jains = float((sum_x ** 2) / (n_gpus * sum_sq_x))
        else:
            jains = 1.0  # Perfectly fair if 0 tasks on all

        # Reliability & Risk metrics
        cooldown_events = sum(1 for g in env.gpus if g.is_in_cooldown())
        quarantine_events = sum(
            1 for g in env.gpus
            if (hasattr(g, "is_quarantined") and g.is_quarantined())
            or (g.get_risk_state() and g.get_risk_state().is_quarantined)
        )

        risk_scores = [g.get_risk_state().risk_score if g.get_risk_state() else 0.0 for g in env.gpus]
        avg_risk = float(np.mean(risk_scores)) if risk_scores else 0.0
        max_risk = float(np.max(risk_scores)) if risk_scores else 0.0

        efc_vals = [
            float(env.marl_manager.calculate_expected_failure_cost(g))
            for g in env.gpus
            if hasattr(env, "marl_manager") and env.marl_manager
        ]
        avg_efc = float(np.mean(efc_vals)) if efc_vals else 0.0
        total_efc = float(sum(efc_vals)) if efc_vals else 0.0

        return RunMetrics(
            run_id=run_id,
            scheduler_id=scheduler_id,
            scheduler_name=scheduler_name,
            seed=seed,
            total_tasks=total_stream_tasks,
            completed_tasks=completed,
            failed_tasks=failed,
            rejected_tasks=rejected,
            unfinished_tasks=unfinished,
            success_rate_pct=round(success_rate, 2),
            failure_rate_pct=round(failure_rate, 2),
            rejection_rate_pct=round(rejection_rate, 2),
            makespan_steps=step_count,
            elapsed_seconds=round(elapsed_seconds, 3),
            throughput_tasks_per_sec=throughput,
            avg_completion_time_sec=round(avg_comp, 3),
            avg_waiting_time_sec=round(avg_wait, 3),
            avg_turnaround_time_sec=round(avg_turn, 3),
            avg_gpu_utilization_pct=round(avg_util, 2),
            avg_gpu_memory_pct=round(avg_mem, 2),
            load_balance_std=round(load_balance_std, 2),
            jains_fairness_index=round(jains, 4),
            capacity_violations=capacity_violations,
            scheduler_exceptions=scheduler_exceptions,
            cooldown_events=cooldown_events,
            quarantine_events=quarantine_events,
            avg_risk_exposure=round(avg_risk, 4),
            max_risk_exposure=round(max_risk, 4),
            avg_efc=round(avg_efc, 4),
            total_efc=round(total_efc, 4),
        )

    @staticmethod
    def aggregate_runs(scheduler_id: str, scheduler_name: str, runs: List[RunMetrics]) -> BenchmarkSummary:
        """Compute statistical mean, standard deviation, and 95% CI across repetitions."""
        if not runs:
            return BenchmarkSummary(
                scheduler_id=scheduler_id,
                scheduler_name=scheduler_name,
                repetitions=0,
                seeds=[],
                total_tasks_evaluated=0,
            )

        n = len(runs)
        seeds = [r.seed for r in runs]
        total_tasks = sum(r.total_tasks for r in runs)

        metric_keys = [
            "success_rate_pct",
            "failure_rate_pct",
            "rejection_rate_pct",
            "throughput_tasks_per_sec",
            "avg_turnaround_time_sec",
            "avg_waiting_time_sec",
            "avg_completion_time_sec",
            "avg_gpu_utilization_pct",
            "avg_gpu_memory_pct",
            "load_balance_std",
            "jains_fairness_index",
            "capacity_violations",
            "cooldown_events",
            "quarantine_events",
            "avg_risk_exposure",
            "max_risk_exposure",
            "avg_efc",
            "total_efc",
        ]

        aggregates = {}
        for key in metric_keys:
            values = [getattr(r, key, 0.0) for r in runs]
            mean_val = float(np.mean(values))
            std_val = float(np.std(values, ddof=1)) if n > 1 else 0.0

            # 95% Confidence Interval: critical z ~ 1.96 (or student-t for n < 30)
            margin = 1.96 * (std_val / math.sqrt(n)) if n > 1 else 0.0
            ci_lower = max(0.0, mean_val - margin)
            ci_upper = mean_val + margin

            aggregates[key] = AggregateMetric(
                mean=round(mean_val, 4),
                std=round(std_val, 4),
                ci_95_lower=round(ci_lower, 4),
                ci_95_upper=round(ci_upper, 4),
                min_val=round(float(np.min(values)), 4),
                max_val=round(float(np.max(values)), 4),
                sample_size=n,
            )

        return BenchmarkSummary(
            scheduler_id=scheduler_id,
            scheduler_name=scheduler_name,
            repetitions=n,
            seeds=seeds,
            total_tasks_evaluated=total_tasks,
            metrics=aggregates,
        )

    @staticmethod
    def compute_paired_improvements(
        target_summary: BenchmarkSummary,
        baseline_summaries: Dict[str, BenchmarkSummary],
    ) -> None:
        """
        Compute percentage improvement of target scheduler against baseline algorithms:
        Formula: ((target_mean - baseline_mean) / baseline_mean) * 100%
        Positive indicates superior performance for 'higher is better' metrics;
        for latency/risk/failure metrics, lower is better.
        """
        target_metrics = target_summary.metrics
        improvements = {}

        for b_id, b_summary in baseline_summaries.items():
            if b_id == target_summary.scheduler_id:
                continue
            improvements[b_id] = {}
            for m_key, m_agg in target_metrics.items():
                if m_key in b_summary.metrics:
                    b_val = b_summary.metrics[m_key].mean
                    t_val = m_agg.mean
                    if abs(b_val) > 1e-6:
                        pct = ((t_val - b_val) / abs(b_val)) * 100.0
                    else:
                        pct = 0.0
                    improvements[b_id][m_key] = round(pct, 2)

        target_summary.improvements_vs_baselines = improvements
