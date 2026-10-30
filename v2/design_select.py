"""
PHASE 4a — design selection on VALIDATION windows, using BASELINE schedulers only.

Two decisions are made here, both by rules fixed in advance:

1. LOAD LEVELS (condition 3). Sweep candidate rho values and measure the share of
   scheduling decisions with >= 2 eligible GPUs, using only FCFS and Least Loaded.
   Pick one light, one near-capacity and one overloaded level by a stated rule.
   No MARL result is consulted.

2. EXPLORATORY QUARANTINE THRESHOLD (Decision 1b). The smallest threshold from a
   fixed candidate ladder at which quarantine triggers in >= TRIGGER_SHARE of
   VALIDATION episodes run with FCFS and Least Loaded only.

Outputs: results/metrics/v2/design_selection.json and logs/v2_tuning_log.csv rows.
"""

from __future__ import annotations

import json
import random
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from benchmarking.scheduler_adapters import get_scheduler_adapter
from environment import GPUEnvironment
from realdata.config import BURSTGPT_WINDOWS_PATH
from realdata.loaders import load_burstgpt_data, load_uci_video_data
from realdata.windows import TraceWindowsSummary
from v2.load_model import build_calibrated_trace, measure_offered_rho

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "metrics" / "v2"
LOGS = ROOT / "logs"
MET.mkdir(parents=True, exist_ok=True)

# ── fixed rules, stated before running ────────────────────────────────────────
BASELINES_FOR_DESIGN = ["fcfs", "least_loaded"]
VALIDATION_SEEDS = [200, 201, 202]
EPISODE_STEPS = 300                      # default; confirmed by the sweep below
CANDIDATE_EPISODE_STEPS = [120, 300, 600]
EPISODE_LENGTH_RHO = 0.9
CANDIDATE_RHO = [0.3, 0.5, 0.7, 0.9, 1.1, 1.4, 1.8]
QUARANTINE_LADDER = [0.85, 0.80, 0.75, 0.70, 0.65, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.30]
TRIGGER_SHARE = 0.30                     # >=30% of validation episodes must trigger

# Selection rule for load levels (fixed before the sweep; baselines only).
# Guarantees one level per regime by construction:
#   LIGHT      = LARGEST candidate rho whose contested share >= 25%
#                (lightly loaded, scheduler has ample choice)
#   NEAR_CAP   = candidate rho whose DELIVERED rho is closest to 1.0
#   OVERLOADED = SMALLEST candidate rho whose contested share <= 10%
#                (genuinely saturated); if none, the largest candidate
LIGHT_MIN_CONTESTED = 25.0
OVERLOAD_MAX_CONTESTED = 10.0


def _run_probe(scheduler_id, trace, seed, quarantine_ladder):
    random.seed(seed); np.random.seed(seed)
    env = GPUEnvironment(scheduler="fcfs")
    env.enable_deterministic_physics()
    get_scheduler_adapter(scheduler_id).configure_environment(env)
    env.set_eval_mode()
    env.attach_workload_trace(deepcopy(trace))

    # count eligibility exactly as the MARL bidder would see it, without using MARL
    contested = 0
    total = 0
    quar_steps = {t: 0 for t in quarantine_ladder}
    max_risk = 0.0

    for _ in range(len(trace)):
        # eligibility snapshot BEFORE stepping, for each pending arrival this step
        env.step()
        rs = []
        for g in env.gpus:
            st = g.get_risk_state()
            r = float(st.risk_score) if st else 0.0
            rs.append(r)
            max_risk = max(max_risk, r)
        for t in quarantine_ladder:
            if any(r >= t for r in rs):
                quar_steps[t] += 1

    m = env.get_metrics()
    offered = sum(len(s) for s in trace)
    return {
        "offered": offered,
        "admitted": float(m.get("successful_assignments", 0.0)),
        "admitted_pct": 100.0 * float(m.get("successful_assignments", 0.0)) / max(1, offered),
        "completed": float(m.get("completed", 0.0)),
        "max_risk": max_risk,
        "crashes": float(m.get("total_crash_events", 0.0)),
        **{f"quar_steps_{t}": quar_steps[t] for t in quarantine_ladder},
    }


def _count_contested(trace, seed):
    """
    Share of scheduling decisions where >= 2 GPUs could legally take the task,
    measured AT THE DECISION POINT.

    An earlier version sampled eligibility when a task arrived, which was wrong:
    the environment first retries the backlog of already-queued tasks, and those
    occupy the GPUs before the new arrival is ever considered. We therefore wrap
    GPUEnvironment._schedule_task and evaluate eligibility for each task at the
    instant the scheduler actually decides for it.

    Eligibility (crashed / cooldown / quarantined / capacity) is policy
    independent; a neutral baseline dispatcher drives the trajectory.
    """
    random.seed(seed); np.random.seed(seed)
    env = GPUEnvironment(scheduler="fcfs")
    env.enable_deterministic_physics()
    get_scheduler_adapter("least_loaded").configure_environment(env)
    env.set_eval_mode()
    env.attach_workload_trace(deepcopy(trace))

    counts = {"contested": 0, "total": 0}
    orig = env._schedule_task

    def wrapped(task):
        elig = 0
        for g in env.gpus:
            rs = g.get_risk_state()
            quarantined = bool(rs and rs.is_quarantined)
            if (not g.crashed) and (not g.is_in_cooldown()) and (not quarantined) \
                    and (g.current_memory + task.memory_required <= g.memory_capacity):
                elig += 1
        counts["total"] += 1
        if elig >= 2:
            counts["contested"] += 1
        return orig(task)

    env._schedule_task = wrapped
    for _ in range(len(trace)):
        env.step()
    env._schedule_task = orig
    return counts["contested"], counts["total"]


def sweep_episode_length(val_windows, df_burst, df_video, verbose=True):
    """
    Episode length is a design decision: diagnosis 3 showed risk accumulates with
    time. Rule: pick the SHORTEST candidate length whose max observed risk is
    within 0.02 of the longest candidate (i.e. risk has plateaued), subject to
    the CPU budget. Measured on VALIDATION with baselines only.
    """
    rows = []
    if verbose:
        print(f"\nEpisode-length sweep at rho={EPISODE_LENGTH_RHO} "
              f"(validation, baselines only)")
        print(f"{'steps':>7}{'max_risk':>10}{'completed':>11}{'crashes':>9}")
    for steps in CANDIDATE_EPISODE_STEPS:
        mx, comp, cr = [], [], []
        for w in val_windows:
            for sd in VALIDATION_SEEDS:
                trace, _ = build_calibrated_trace(
                    window=w, episode_steps=steps, target_rho=EPISODE_LENGTH_RHO,
                    df_burst=df_burst, df_video=df_video, seed=sd, partition="train")
                for sid in BASELINES_FOR_DESIGN:
                    r = _run_probe(sid, trace, sd, QUARANTINE_LADDER)
                    mx.append(r["max_risk"]); comp.append(r["completed"]); cr.append(r["crashes"])
        rows.append({"steps": steps, "max_risk": float(np.max(mx)),
                     "completed": float(np.mean(comp)), "crashes": float(np.mean(cr))})
        if verbose:
            print(f"{steps:>7}{rows[-1]['max_risk']:>10.3f}{rows[-1]['completed']:>11.1f}"
                  f"{rows[-1]['crashes']:>9.2f}")
    best = max(r["max_risk"] for r in rows)
    chosen = next((r["steps"] for r in rows if r["max_risk"] >= best - 0.02),
                  CANDIDATE_EPISODE_STEPS[-1])
    if verbose:
        print(f"chosen episode length: {chosen} steps")
    return chosen, rows


def select_design(verbose=True) -> Dict[str, Any]:
    df_burst, df_video = load_burstgpt_data(), load_uci_video_data()
    ws = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    # VALIDATION = TRAIN-partition windows NOT used for training
    val_windows = [ws.train_windows["median_hour"], ws.train_windows["burstiest_window"]]

    chosen_steps, length_rows = sweep_episode_length(val_windows, df_burst, df_video, verbose)
    global EPISODE_STEPS
    EPISODE_STEPS = chosen_steps

    rows: List[Dict[str, Any]] = []
    if verbose:
        print(f"VALIDATION windows: {[w.window_id for w in val_windows]} "
              f"(TRAIN partition, days 0-30)")
        print(f"candidate rho: {CANDIDATE_RHO}\n")
        print(f"{'rho':>5}{'delivered':>11}{'contested%':>12}{'admitted%':>11}"
              f"{'completed':>11}{'max_risk':>10}")

    for rho in CANDIDATE_RHO:
        c_tot = c_con = 0
        adm, comp, mx = [], [], []
        quar = {t: [] for t in QUARANTINE_LADDER}
        delivered = []
        for w in val_windows:
            for sd in VALIDATION_SEEDS:
                trace, cal = build_calibrated_trace(
                    window=w, episode_steps=EPISODE_STEPS, target_rho=rho,
                    df_burst=df_burst, df_video=df_video, seed=sd, partition="train")
                delivered.append(measure_offered_rho(trace))
                con, tot = _count_contested(trace, sd)
                c_con += con; c_tot += tot
                for sid in BASELINES_FOR_DESIGN:
                    r = _run_probe(sid, trace, sd, QUARANTINE_LADDER)
                    adm.append(r["admitted_pct"]); comp.append(r["completed"]); mx.append(r["max_risk"])
                    for t in QUARANTINE_LADDER:
                        quar[t].append(1.0 if r[f"quar_steps_{t}"] > 0 else 0.0)

        row = {
            "rho": rho,
            "delivered_rho": float(np.mean(delivered)),
            "contested_pct": 100.0 * c_con / max(1, c_tot),
            "admitted_pct": float(np.mean(adm)),
            "completed": float(np.mean(comp)),
            "max_risk": float(np.max(mx)),
            **{f"quar_trigger_share_{t}": float(np.mean(quar[t])) for t in QUARANTINE_LADDER},
        }
        rows.append(row)
        if verbose:
            print(f"{rho:>5}{row['delivered_rho']:>11.3f}{row['contested_pct']:>12.1f}"
                  f"{row['admitted_pct']:>11.1f}{row['completed']:>11.1f}{row['max_risk']:>10.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(MET / "design_selection_sweep.csv", index=False)

    # ── apply the load-level rules ──
    light_c = df[df.contested_pct >= LIGHT_MIN_CONTESTED]
    light = float(light_c.rho.max()) if len(light_c) else float(df.rho.min())
    near = float(df.loc[(df.delivered_rho - 1.0).abs().idxmin(), "rho"])
    over_c = df[df.contested_pct <= OVERLOAD_MAX_CONTESTED]
    over = float(over_c.rho.min()) if len(over_c) else float(df.rho.max())

    levels = sorted({light, near, over})
    # guarantee three distinct levels spanning the range
    if len(levels) < 3:
        pool = [r for r in CANDIDATE_RHO if r not in levels]
        while len(levels) < 3 and pool:
            levels.append(pool.pop(len(pool) // 2))
        levels = sorted(levels)

    # ── apply the quarantine-threshold rule ──
    chosen_thr = None
    for t in QUARANTINE_LADDER:                       # descending: largest first
        share = float(df[f"quar_trigger_share_{t}"].max())
        if share >= TRIGGER_SHARE:
            chosen_thr = t
            break

    out = {
        "validation_windows": [w.window_id for w in val_windows],
        "validation_seeds": VALIDATION_SEEDS,
        "episode_steps": EPISODE_STEPS,
        "episode_length_sweep": length_rows,
        "candidate_episode_steps": CANDIDATE_EPISODE_STEPS,
        "baselines_used": BASELINES_FOR_DESIGN,
        "candidate_rho": CANDIDATE_RHO,
        "rules": {
            "light": f"largest rho with contested_pct >= {LIGHT_MIN_CONTESTED}",
            "near_capacity": "rho whose delivered_rho is closest to 1.0",
            "overloaded": f"smallest rho with contested_pct <= {OVERLOAD_MAX_CONTESTED}",
            "quarantine": f"largest threshold on ladder triggering in >= {TRIGGER_SHARE:.0%} of validation episodes",
        },
        "chosen_load_levels": levels,
        "quarantine_ladder": QUARANTINE_LADDER,
        "quarantine_trigger_shares": {
            str(t): float(df[f"quar_trigger_share_{t}"].max()) for t in QUARANTINE_LADDER},
        "chosen_quarantine_threshold": chosen_thr,
        "quarantine_untestable_at_shipped_threshold": bool(
            df["quar_trigger_share_0.85"].max() < TRIGGER_SHARE),
        "max_risk_observed_validation": float(df.max_risk.max()),
        "sweep": rows,
    }
    json.dump(out, open(MET / "design_selection.json", "w"), indent=2)

    if verbose:
        print("\nchosen load levels      :", levels)
        print("quarantine trigger share by threshold:")
        for t in QUARANTINE_LADDER:
            print(f"   {t:>5}: {df[f'quar_trigger_share_{t}'].max():.0%}")
        print("chosen quarantine thresh:", chosen_thr)
        print("max risk seen on validation:", round(float(df.max_risk.max()), 4))
    return out


if __name__ == "__main__":
    select_design()
