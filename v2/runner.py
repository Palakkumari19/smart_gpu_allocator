"""
v2 episode runner.

Supports the three required conditions on identical seeds (condition 1):
  "v1"            as run: buggy load scale, v1 training, v1 state buckets
  "v1_corrected"  fixed load scale ONLY; v1 training and v1 buckets kept
  "v2"            fixed load + v2 training + refined buckets

Also records metrics v1 could not: true time-averaged risk, peak per-GPU risk,
quarantine at an exploratory threshold, and the admission funnel.
"""

from __future__ import annotations

import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from benchmarking.scheduler_adapters import get_scheduler_adapter
from environment import GPUEnvironment
from realdata.risk_overlay import step_with_hazard_overlay
from v2.improvements import no_patch, refined_risk_buckets
from v2.load_model import build_calibrated_trace, build_legacy_trace, measure_offered_rho

SCHEDULERS = ["fcfs", "round_robin", "least_loaded", "rl", "marl", "risk_aware_marl"]


def build_trace_for_condition(condition, window, episode_steps, target_rho,
                              df_burst, df_video, seed, partition="eval"):
    if condition == "v1":
        trace, scale = build_legacy_trace(window, episode_steps, target_rho,
                                          df_burst, df_video, seed, partition=partition)
        return trace, {"load_scale": scale, "delivered_rho": measure_offered_rho(trace)}
    trace, cal = build_calibrated_trace(window, episode_steps, target_rho,
                                        df_burst, df_video, seed, partition=partition)
    return trace, {"load_scale": cal.load_scale, "delivered_rho": cal.delivered_rho}


def run_episode(
    scheduler_id: str,
    trace: List[List[Any]],
    q_snapshots: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    overlay=None,
    risk_edges: Optional[Sequence[float]] = None,
    quarantine_thresholds: Sequence[float] = (0.85,),
) -> Dict[str, float]:
    """
    One evaluation episode. Q-tables are restored and frozen, exactly as in v1.

    quarantine_thresholds: risk levels at which to COUNT quarantine events for
    reporting. 0.85 is the shipped mechanism; lower values are exploratory and
    are counted observationally (they do not change scheduler behaviour).
    """
    random.seed(seed); np.random.seed(seed)

    ctx = refined_risk_buckets(risk_edges) if risk_edges else no_patch()
    with ctx:
        env = GPUEnvironment(scheduler="fcfs")
        env.enable_deterministic_physics()
        adapter = get_scheduler_adapter(scheduler_id)
        adapter.configure_environment(env)
        adapter.reset_state(env)

        if q_snapshots:
            snap = q_snapshots.get(scheduler_id)
            if snap is not None:
                if scheduler_id == "rl":
                    env.agent.restore_q_table(deepcopy(snap))
                elif scheduler_id in ("marl", "risk_aware_marl"):
                    env.marl_manager.restore_q_tables(deepcopy(snap))

        env.set_eval_mode()
        env.attach_workload_trace(deepcopy(trace))

        step_means: List[float] = []
        per_gpu_peak: Dict[int, float] = {}
        quarantine_steps = {t: 0 for t in quarantine_thresholds}
        contested = 0
        total_decisions = 0

        mgr = env.marl_manager
        orig_select = mgr.select_gpu_for_task

        def wrapped(task, gpus, cluster_state, *a, **kw):
            nonlocal contested, total_decisions
            out = orig_select(task, gpus, cluster_state, *a, **kw)
            bds = mgr.last_candidate_breakdowns or []
            n_elig = sum(1 for b in bds if b.get("eligible"))
            total_decisions += 1
            if n_elig >= 2:
                contested += 1
            return out

        if scheduler_id in ("marl", "risk_aware_marl"):
            mgr.select_gpu_for_task = wrapped

        for _ in range(len(trace)):
            step_with_hazard_overlay(env, overlay)
            rs = []
            for g in env.gpus:
                st = g.get_risk_state()
                r = float(st.risk_score) if st else 0.0
                rs.append(r)
                per_gpu_peak[g.id] = max(per_gpu_peak.get(g.id, 0.0), r)
            step_means.append(float(np.mean(rs)))
            for t in quarantine_thresholds:
                if any(r >= t for r in rs):
                    quarantine_steps[t] += 1

        if scheduler_id in ("marl", "risk_aware_marl"):
            mgr.select_gpu_for_task = orig_select

        m = env.get_metrics()

    offered = sum(len(s) for s in trace)
    completed = float(m.get("completed", 0.0))
    crashes = float(m.get("total_crash_events", 0.0))
    mean_risk = float(np.mean(step_means)) if step_means else 0.0

    out = {
        # v1-comparable metrics (same definitions, so v1 vs v2 is like-for-like)
        "avg_risk_exposure": float(m.get("average_gpu_risk", 0.0)),   # terminal snapshot, as v1
        "terminal_efc": float(m.get("total_assigned_efc", 0.0)),
        "crash_count": crashes,
        "quarantine_time": float(m.get("quarantined_gpus", 0.0)),
        "completed_tasks": completed,
        # v2 additions
        "mean_risk_over_time": mean_risk,
        "peak_gpu_risk": float(max(per_gpu_peak.values())) if per_gpu_peak else 0.0,
        "tasks_offered": float(offered),
        "successful_assignments": float(m.get("successful_assignments", 0.0)),
        "admitted_pct": 100.0 * float(m.get("successful_assignments", 0.0)) / max(1.0, offered),
        "contested_decisions": float(contested),
        "total_decisions": float(total_decisions),
        "contested_pct": 100.0 * contested / max(1, total_decisions),
        "avg_temperature": float(m.get("avg_temperature", 0.0)),
        # combined objectives (condition 5 / V2-H1, V2-H2)
        "risk_per_completed_task": mean_risk / max(1.0, completed),
        "crashes_per_completed_task": crashes / max(1.0, completed),
    }
    for t, n in quarantine_steps.items():
        out[f"quarantine_steps_at_{t}"] = float(n)
        out[f"quarantine_triggered_at_{t}"] = 1.0 if n > 0 else 0.0
    return out
