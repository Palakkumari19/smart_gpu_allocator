"""
Bit 9: Benchmark Runner
Orchestrates fair, reproducible multi-algorithm experiments.
Ensures identical task streams, fresh isolated environment state per run,
capacity invariant verification, and metrics aggregation.
"""

from copy import deepcopy
import os
import time
import uuid
from typing import Dict, Any, List, Optional
import numpy as np

from environment import GPUEnvironment
from .experiment_config import BenchmarkConfig, ALL_SCHEDULERS
from .workload_generator import WorkloadGenerator
from .scheduler_adapters import get_scheduler_adapter
from .metrics import MetricsCalculator, RunMetrics, BenchmarkSummary
from .results_store import ResultsStore


class BenchmarkRunner:
    """Executes controlled, reproducible benchmark experiments."""

    def __init__(self, results_store: Optional[ResultsStore] = None):
        self.results_store = results_store or ResultsStore()
        self.is_running = False
        self.current_progress: Dict[str, Any] = {
            "status": "idle",
            "percent": 0.0,
            "current_scheduler": None,
            "current_run": 0,
            "total_runs": 0,
            "experiment_id": None,
        }

    def run_benchmark(self, config: BenchmarkConfig) -> Dict[str, Any]:
        """
        Execute a full benchmark experiment according to config.
        Guarantees:
        1. Identical workload stream for all schedulers within a repetition.
        2. Isolated fresh simulator instances with no shared mutable state.
        3. Strict monitoring of the 8.0 GB physical capacity invariant.
        """
        config.validate()
        experiment_id = f"bench_{int(time.time())}_{uuid.uuid4().hex[:6]}"

        self.is_running = True
        total_runs = len(config.schedulers) * config.num_repetitions
        completed_runs_count = 0

        self.current_progress = {
            "status": "running",
            "percent": 0.0,
            "current_scheduler": None,
            "current_run": 0,
            "total_runs": total_runs,
            "experiment_id": experiment_id,
            "start_time": time.time(),
        }

        all_run_metrics: List[RunMetrics] = []
        task_records: List[Dict[str, Any]] = []

        try:
            # Loop over seeded repetitions
            for rep_idx in range(config.num_repetitions):
                seed = config.seeds[rep_idx]

                # 1. Generate the deterministic task stream once per seed
                task_stream = WorkloadGenerator.generate_stream(
                    seed=seed,
                    num_tasks=config.num_tasks,
                    workload_type=config.workload_type,
                )

                # Convert stream into a step-by-step workload trace (1 to 2 tasks per step)
                tasks_list = task_stream.tasks
                steps_trace = []
                chunk_size = 2  # Feed 2 tasks per step
                for i in range(0, len(tasks_list), chunk_size):
                    steps_trace.append(tasks_list[i : i + chunk_size])

                # 2. Evaluate each scheduler against the identical workload trace
                for scheduler_id in config.schedulers:
                    completed_runs_count += 1
                    self.current_progress.update({
                        "current_scheduler": scheduler_id,
                        "current_run": completed_runs_count,
                        "percent": round((completed_runs_count / total_runs) * 100.0, 1),
                    })

                    adapter = get_scheduler_adapter(scheduler_id)
                    if not adapter.is_available:
                        # Record unavailable run safely
                        continue

                    # Execute single run with isolated environment
                    run_id = f"{experiment_id}_{scheduler_id}_rep{rep_idx}_s{seed}"
                    run_metrics, run_tasks = self._execute_single_run(
                        run_id=run_id,
                        scheduler_id=scheduler_id,
                        adapter=adapter,
                        seed=seed,
                        steps_trace=steps_trace,
                        total_tasks_count=len(tasks_list),
                        max_steps=config.max_steps_per_run,
                    )

                    all_run_metrics.append(run_metrics)
                    if config.record_task_telemetry:
                        task_records.extend(run_tasks)

            # 3. Compute summaries and paired improvements
            summaries: Dict[str, BenchmarkSummary] = {}
            for scheduler_id in config.schedulers:
                sched_runs = [r for r in all_run_metrics if r.scheduler_id == scheduler_id]
                adapter = get_scheduler_adapter(scheduler_id)
                summary = MetricsCalculator.aggregate_runs(
                    scheduler_id=scheduler_id,
                    scheduler_name=adapter.name,
                    runs=sched_runs,
                )
                summaries[scheduler_id] = summary

            # Compute paired comparisons vs baselines
            for s_id, summary in summaries.items():
                MetricsCalculator.compute_paired_improvements(summary, summaries)

            # 4. Save results to disk
            saved_paths = self.results_store.save_experiment(
                experiment_id=experiment_id,
                config={
                    "name": config.name,
                    "schedulers": config.schedulers,
                    "num_tasks": config.num_tasks,
                    "num_repetitions": config.num_repetitions,
                    "seeds": config.seeds[: config.num_repetitions],
                    "workload_type": config.workload_type,
                    "max_steps_per_run": config.max_steps_per_run,
                    "description": config.description,
                },
                runs=all_run_metrics,
                summaries=summaries,
                task_records=task_records if config.record_task_telemetry else None,
            )

            self.current_progress.update({
                "status": "completed",
                "percent": 100.0,
                "end_time": time.time(),
            })

            return {
                "experiment_id": experiment_id,
                "status": "success",
                "config": config,
                "runs": [r.to_dict() for r in all_run_metrics],
                "summaries": {k: v.to_dict() for k, v in summaries.items()},
                "artifacts": saved_paths,
            }

        except Exception as e:
            self.current_progress.update({"status": "failed", "error": str(e)})
            raise
        finally:
            self.is_running = False

    def _execute_single_run(
        self,
        run_id: str,
        scheduler_id: str,
        adapter,
        seed: int,
        steps_trace: List[List[Any]],
        total_tasks_count: int,
        max_steps: int,
    ) -> (RunMetrics, List[Dict[str, Any]]):
        """Execute one isolated simulation run with strict capacity and safety monitoring."""
        # Deepcopy trace to prevent inter-scheduler contamination
        cloned_trace = []
        for step_tasks in steps_trace:
            cloned_step = []
            for t in step_tasks:
                clone = deepcopy(t)
                clone.execution_progress = 0
                clone.preempted = False
                clone.preemption_count = 0
                cloned_step.append(clone)
            cloned_trace.append(cloned_step)

        # Create isolated GPUEnvironment instance
        env = GPUEnvironment(scheduler="fcfs")  # initial default
        env.enable_deterministic_physics()
        env.set_eval_mode()
        adapter.configure_environment(env)
        adapter.reset_state(env)

        env.attach_workload_trace(cloned_trace)

        capacity_violations = 0
        scheduler_exceptions = 0
        step_count = 0
        start_time = time.perf_counter()

        # Step until all tasks are finished or step limit reached
        for _ in range(max_steps):
            step_count += 1
            try:
                env.step()
            except Exception:
                scheduler_exceptions += 1

            # Check strict physical GPU capacity invariant (8.0 GB per GPU)
            for gpu in env.gpus:
                if round(float(gpu.current_memory), 2) > 8.00:
                    capacity_violations += 1

            # Termination condition: no more incoming trace, no queued tasks, no running tasks
            queued_cnt = len(env.task_queue.get_tasks_by_status(env.task_queue.status_enum.QUEUED if hasattr(env.task_queue, "status_enum") else "QUEUED"))
            running_cnt = sum(gpu.get_running_tasks_count() for gpu in env.gpus)
            trace_exhausted = (env.time_step >= len(cloned_trace))

            if trace_exhausted and queued_cnt == 0 and running_cnt == 0:
                break

        elapsed_sec = time.perf_counter() - start_time

        # Calculate run metrics
        run_metrics = MetricsCalculator.calculate_run_metrics(
            run_id=run_id,
            scheduler_id=scheduler_id,
            scheduler_name=adapter.name,
            seed=seed,
            env=env,
            total_stream_tasks=total_tasks_count,
            elapsed_seconds=elapsed_sec,
            step_count=step_count,
            capacity_violations=capacity_violations,
            scheduler_exceptions=scheduler_exceptions,
        )

        # Collect per-task records
        task_records = []
        all_tasks = env.task_queue.get_all_tasks()
        for task_id, (task, meta) in all_tasks.items():
            dur = (meta.completion_time - meta.start_time) if (meta.completion_time and meta.start_time) else None
            wait = (meta.start_time - meta.arrival_time) if (meta.start_time and meta.arrival_time) else None
            task_records.append({
                "run_id": run_id,
                "scheduler_id": scheduler_id,
                "seed": seed,
                "task_id": task_id,
                "profile": getattr(task, "profile_name", "Unknown"),
                "memory_gb": getattr(task, "memory_required", 0.0),
                "duration_steps": getattr(task, "execution_time", 0),
                "assigned_gpu": meta.assigned_gpu,
                "status": meta.status.value if hasattr(meta.status, "value") else str(meta.status),
                "waiting_time_sec": wait,
                "execution_duration_sec": dur,
                "retry_count": meta.retry_count,
            })

        return run_metrics, task_records
