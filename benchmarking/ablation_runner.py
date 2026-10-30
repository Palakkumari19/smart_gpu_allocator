"""
Bit 9: Ablation Runner
Executes controlled ablation studies to isolate and measure the individual
contributions of Risk Penalty and Expected Failure Cost (EFC) in MARL scheduling.
"""

from copy import deepcopy
import time
import uuid
from typing import Dict, Any, List, Optional
import numpy as np

from environment import GPUEnvironment
from .experiment_config import AblationConfig
from .workload_generator import WorkloadGenerator
from .metrics import MetricsCalculator, RunMetrics, BenchmarkSummary
from .results_store import ResultsStore


class AblationRunner:
    """Evaluates Risk-Aware MARL with isolated penalty components."""

    def __init__(self, results_store: Optional[ResultsStore] = None):
        self.results_store = results_store or ResultsStore()

    def run_ablation_study(self, config: Optional[AblationConfig] = None) -> Dict[str, Any]:
        """
        Execute an ablation study across:
        1. Full Risk-Aware MARL (risk_penalty=0.25*risk, efc_penalty=0.20*efc)
        2. MARL without Risk Penalty (risk_penalty=0.0, efc_penalty=0.20*efc)
        3. MARL without EFC Penalty (risk_penalty=0.25*risk, efc_penalty=0.0)
        4. Standard MARL (risk_blind: risk_penalty=0.0, efc_penalty=0.0)
        """
        config = config or AblationConfig()
        config.validate()

        study_id = f"ablation_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        modes = config.ablation_modes
        all_runs: List[RunMetrics] = []

        for rep_idx in range(config.num_repetitions):
            seed = config.seeds[rep_idx]
            task_stream = WorkloadGenerator.generate_stream(
                seed=seed,
                num_tasks=config.num_tasks,
                workload_type=config.workload_type,
            )

            # Chunk into trace
            tasks_list = task_stream.tasks
            chunk_size = 2
            steps_trace = [tasks_list[i : i + chunk_size] for i in range(0, len(tasks_list), chunk_size)]

            for mode in modes:
                run_id = f"{study_id}_{mode}_rep{rep_idx}_s{seed}"
                run_metric = self._execute_ablation_run(
                    run_id=run_id,
                    mode=mode,
                    seed=seed,
                    steps_trace=steps_trace,
                    total_tasks_count=len(tasks_list),
                    max_steps=config.max_steps_per_run,
                )
                all_runs.append(run_metric)

        # Aggregate summaries per mode
        mode_summaries: Dict[str, BenchmarkSummary] = {}
        mode_names = {
            "full_risk_aware": "Full Risk-Aware MARL",
            "no_risk_penalty": "MARL (No Risk Penalty)",
            "no_efc_penalty": "MARL (No EFC Penalty)",
            "risk_blind": "Standard MARL (Risk-Blind)",
        }

        for mode in modes:
            runs_for_mode = [r for r in all_runs if r.scheduler_id == mode]
            summary = MetricsCalculator.aggregate_runs(
                scheduler_id=mode,
                scheduler_name=mode_names.get(mode, mode),
                runs=runs_for_mode,
            )
            mode_summaries[mode] = summary

        # Compute delta vs Full Risk-Aware MARL
        full_baseline = mode_summaries.get("full_risk_aware")
        deltas = {}
        if full_baseline:
            for mode, s in mode_summaries.items():
                deltas[mode] = {}
                for m_key, agg in s.metrics.items():
                    if m_key in full_baseline.metrics:
                        base_val = full_baseline.metrics[m_key].mean
                        cur_val = agg.mean
                        diff = round(cur_val - base_val, 4)
                        pct = round(((cur_val - base_val) / abs(base_val)) * 100.0, 2) if abs(base_val) > 1e-6 else 0.0
                        deltas[mode][m_key] = {"difference": diff, "pct_change": pct}

        # Save to disk
        saved_paths = self.results_store.save_experiment(
            experiment_id=study_id,
            config={
                "name": config.name,
                "modes": modes,
                "num_tasks": config.num_tasks,
                "num_repetitions": config.num_repetitions,
                "seeds": config.seeds,
                "workload_type": config.workload_type,
            },
            runs=all_runs,
            summaries=mode_summaries,
            ablations={"deltas_vs_full": deltas},
        )

        return {
            "study_id": study_id,
            "status": "success",
            "modes": modes,
            "runs": [r.to_dict() for r in all_runs],
            "summaries": {k: v.to_dict() for k, v in mode_summaries.items()},
            "deltas_vs_full": deltas,
            "artifacts": saved_paths,
        }

    def _execute_ablation_run(
        self,
        run_id: str,
        mode: str,
        seed: int,
        steps_trace: List[List[Any]],
        total_tasks_count: int,
        max_steps: int,
    ) -> RunMetrics:
        """Run an isolated GPUEnvironment with the specified ablation bid calculation."""
        # Clone trace
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

        env = GPUEnvironment(scheduler="marl")
        env.enable_deterministic_physics()
        env.set_eval_mode()

        # Patch compute_bid_score_breakdown according to ablation mode
        orig_bid = env.marl_manager.compute_bid_score_breakdown

        def ablation_bid(gpu, task, agent, cluster_state) -> dict:
            breakdown = orig_bid(gpu, task, agent, cluster_state)
            if breakdown.get("eligible", False):
                q_contrib = breakdown.get("q_contribution", 0.0)
                perf = breakdown.get("perf_score", 0.0)
                risk_pen = breakdown.get("risk_penalty", 0.0)
                efc_pen = breakdown.get("efc_penalty", 0.0)

                if mode == "no_risk_penalty":
                    risk_pen = 0.0
                elif mode == "no_efc_penalty":
                    efc_pen = 0.0
                elif mode == "risk_blind":
                    risk_pen = 0.0
                    efc_pen = 0.0

                score = q_contrib + perf - risk_pen - efc_pen
                breakdown["risk_penalty"] = round(risk_pen, 4)
                breakdown["efc_penalty"] = round(efc_pen, 4)
                breakdown["bid_score"] = round(score, 4)
            return breakdown

        env.marl_manager.compute_bid_score_breakdown = ablation_bid
        env.attach_workload_trace(cloned_trace)

        capacity_violations = 0
        exceptions = 0
        step_count = 0
        start_time = time.perf_counter()

        for _ in range(max_steps):
            step_count += 1
            try:
                env.step()
            except Exception:
                exceptions += 1

            for gpu in env.gpus:
                if round(float(gpu.current_memory), 2) > 8.00:
                    capacity_violations += 1

            queued = len(env.task_queue.get_tasks_by_status(env.task_queue.status_enum.QUEUED if hasattr(env.task_queue, "status_enum") else "QUEUED"))
            running = sum(gpu.get_running_tasks_count() for gpu in env.gpus)
            if env.time_step >= len(cloned_trace) and queued == 0 and running == 0:
                break

        elapsed = time.perf_counter() - start_time

        mode_labels = {
            "full_risk_aware": "Full Risk-Aware MARL",
            "no_risk_penalty": "MARL (No Risk Penalty)",
            "no_efc_penalty": "MARL (No EFC Penalty)",
            "risk_blind": "Standard MARL (Risk-Blind)",
        }

        return MetricsCalculator.calculate_run_metrics(
            run_id=run_id,
            scheduler_id=mode,
            scheduler_name=mode_labels.get(mode, mode),
            seed=seed,
            env=env,
            total_stream_tasks=total_tasks_count,
            elapsed_seconds=elapsed,
            step_count=step_count,
            capacity_violations=capacity_violations,
            scheduler_exceptions=exceptions,
        )
