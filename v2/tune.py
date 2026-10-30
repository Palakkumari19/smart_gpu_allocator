"""
PHASE 4 — tuning on VALIDATION ONLY.

Budget: at most realdata/config_v2.MAX_TUNING_CONFIGS configurations.
EVERY configuration tried is appended to logs/v2_tuning_log.csv, including the
ones that perform badly. Nothing here ever reads TEST seeds or TEST windows.

What is tuned (only what the diagnosis justified):
  C  training     — epsilon decay rate and training episode budget
  D  state        — risk bucket edges: v1 edges vs TRAIN-derived edges

What is NOT tuned:
  A  penalty magnitudes — diagnosis showed they already flip 37.7% of contested
                          decisions; rescaling them is not evidence-led
  E  hazard-curve input — deferred
  load levels / episode length / quarantine threshold — already fixed by
                          v2/design_select.py using baselines only

Selection objective (stated before running): maximise
    validation_score = mean(completed_tasks) / mean_risk_over_time
for the TREATMENT scheduler, i.e. work done per unit of risk carried. Ties are
broken by lower crashes_per_completed_task. This is a combined objective by
construction, so a configuration cannot win by starving.
"""

from __future__ import annotations

import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from realdata import config_v2 as C
from realdata.config import BURSTGPT_WINDOWS_PATH
from realdata.loaders import load_burstgpt_data, load_uci_video_data
from realdata.windows import TraceWindowsSummary
from v2.improvements import TrainingConfig, train_all_learners
from v2.load_model import build_calibrated_trace
from v2.runner import run_episode

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "metrics" / "v2"
LOG = ROOT / "logs" / "v2_tuning_log.csv"
MET.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

FIELDS = [
    "config_id", "timestamp", "epsilon_decay", "train_episodes", "risk_edges",
    "episodes_run_rl", "episodes_run_marl", "episodes_run_ram",
    "final_q_rl", "final_q_marl", "final_q_ram",
    "final_eps_marl",
    "val_completed", "val_mean_risk", "val_peak_risk", "val_crashes",
    "val_contested_pct", "val_risk_per_task", "val_crashes_per_task",
    "validation_score", "seconds", "note",
]


def _append_log(row: Dict[str, Any]):
    new = not LOG.exists()
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def evaluate_on_validation(snapshots, risk_edges, val_windows, df_burst, df_video,
                           scheduler=C.TREATMENT) -> Dict[str, float]:
    """Run the TREATMENT scheduler across the validation grid."""
    rows = []
    for w in val_windows:
        for rho in C.LOAD_LEVELS:
            for sd in C.VALIDATION_SEEDS:
                trace, _ = build_calibrated_trace(
                    window=w, episode_steps=C.EPISODE_STEPS, target_rho=rho,
                    df_burst=df_burst, df_video=df_video, seed=sd, partition="train")
                rows.append(run_episode(
                    scheduler, trace, q_snapshots=snapshots, seed=sd,
                    risk_edges=risk_edges,
                    quarantine_thresholds=C.QUARANTINE_THRESHOLDS_REPORTED))
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return agg


def run_tuning(verbose=True) -> Dict[str, Any]:
    df_burst, df_video = load_burstgpt_data(), load_uci_video_data()
    ws = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    train_windows = [ws.train_windows[k] for k in C.TRAIN_WINDOWS]
    val_windows = [ws.train_windows[k] for k in C.VALIDATION_WINDOWS]

    # Bounded grid. 3 x 2 x 2 = 12 configurations, under the cap of 20.
    eps_decays = [0.999, 0.995, 0.990]
    episode_budgets = [10, 20]
    edge_options = {"v1": None, "train_derived": C.RISK_BUCKET_EDGES}

    grid = list(itertools.product(eps_decays, episode_budgets, edge_options.items()))
    assert len(grid) <= C.MAX_TUNING_CONFIGS, f"{len(grid)} configs exceeds cap"

    if verbose:
        print(f"Tuning budget: {len(grid)} configurations (cap {C.MAX_TUNING_CONFIGS})")
        print(f"VALIDATION: windows={C.VALIDATION_WINDOWS} seeds={list(C.VALIDATION_SEEDS)} "
              f"loads={C.LOAD_LEVELS} steps={C.EPISODE_STEPS}\n")

    results: List[Dict[str, Any]] = []
    for i, (decay, episodes, (edge_name, edges)) in enumerate(grid, start=1):
        t0 = time.time()
        cfg = TrainingConfig(
            episodes=episodes, steps_per_episode=C.TRAIN_STEPS_PER_EPISODE,
            seeds=C.TRAIN_SEEDS, target_rho=C.TRAIN_TARGET_RHO,
            epsilon_decay_per_step=decay, epsilon_min=C.EPSILON_MIN,
            learning_rate=C.LEARNING_RATE, plateau_growth=C.PLATEAU_GROWTH,
            plateau_patience=C.PLATEAU_PATIENCE,
        )
        cid = f"cfg{i:02d}"
        if verbose:
            print(f"[{cid}] decay={decay} episodes={episodes} edges={edge_name}")

        note = ""
        try:
            trained = train_all_learners(train_windows, df_burst, df_video, cfg,
                                         risk_edges=edges, verbose=False)
            snapshots = {k: v["snapshot"] for k, v in trained.items()}
            agg = evaluate_on_validation(snapshots, edges, val_windows, df_burst, df_video)
            score = agg["completed_tasks"] / max(1e-9, agg["mean_risk_over_time"])
        except Exception as exc:                       # log failures too
            note = f"FAILED {type(exc).__name__}: {exc}"
            trained, agg, score = {}, {}, float("nan")
            if verbose:
                print(f"   {note}")

        row = {
            "config_id": cid,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epsilon_decay": decay, "train_episodes": episodes, "risk_edges": edge_name,
            "episodes_run_rl": trained.get("rl", {}).get("episodes_run", ""),
            "episodes_run_marl": trained.get("marl", {}).get("episodes_run", ""),
            "episodes_run_ram": trained.get("risk_aware_marl", {}).get("episodes_run", ""),
            "final_q_rl": trained.get("rl", {}).get("final_q_size", ""),
            "final_q_marl": trained.get("marl", {}).get("final_q_size", ""),
            "final_q_ram": trained.get("risk_aware_marl", {}).get("final_q_size", ""),
            "final_eps_marl": trained.get("marl", {}).get("final_epsilon", ""),
            "val_completed": round(agg.get("completed_tasks", float("nan")), 4),
            "val_mean_risk": round(agg.get("mean_risk_over_time", float("nan")), 5),
            "val_peak_risk": round(agg.get("peak_gpu_risk", float("nan")), 5),
            "val_crashes": round(agg.get("crash_count", float("nan")), 4),
            "val_contested_pct": round(agg.get("contested_pct", float("nan")), 3),
            "val_risk_per_task": round(agg.get("risk_per_completed_task", float("nan")), 6),
            "val_crashes_per_task": round(agg.get("crashes_per_completed_task", float("nan")), 6),
            "validation_score": round(score, 4),
            "seconds": round(time.time() - t0, 1),
            "note": note,
        }
        _append_log(row)
        results.append(row)
        if verbose and not note:
            print(f"   completed={row['val_completed']} risk={row['val_mean_risk']} "
                  f"score={row['validation_score']}  ({row['seconds']}s)")

    ok = [r for r in results if not r["note"] and np.isfinite(r["validation_score"])]
    if not ok:
        raise RuntimeError("all tuning configurations failed; see logs/v2_tuning_log.csv")
    best = max(ok, key=lambda r: (r["validation_score"], -r["val_crashes_per_task"]))

    out = {
        "configs_tried": len(results),
        "configs_succeeded": len(ok),
        "budget_cap": C.MAX_TUNING_CONFIGS,
        "objective": "mean(completed_tasks) / mean_risk_over_time, ties by lower crashes/task",
        "best": best,
        "all": results,
    }
    json.dump(out, open(MET / "tuning_summary.json", "w"), indent=2, default=str)
    if verbose:
        print(f"\nBEST: {best['config_id']} decay={best['epsilon_decay']} "
              f"episodes={best['train_episodes']} edges={best['risk_edges']} "
              f"score={best['validation_score']}")
        print(f"Log: {LOG}")
    return out


if __name__ == "__main__":
    run_tuning()
