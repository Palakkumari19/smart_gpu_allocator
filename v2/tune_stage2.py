"""
PHASE 4, stage 2 — remaining tuning budget (8 of the 20-configuration cap).

Stage 1 (v2/tune.py, 12 configs) found no signal: the plateau stopping rule
fired at episode 6 in every configuration and epsilon hit its 0.05 floor
regardless of decay rate, so neither the decay rate nor the episode budget was
ever a binding constraint. The only difference came from the risk-bucket edges,
and it was 0.08% — noise.

Stage 2 therefore tests the question stage 1 could not reach:
**does more training help at all?** It relaxes or removes the early-stopping
rule and lengthens the training episodes.

Appends to the same logs/v2_tuning_log.csv. Total across both stages: 20.
"""

from __future__ import annotations

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
from v2.tune import _append_log, evaluate_on_validation

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "metrics" / "v2"


def run_stage2(verbose=True) -> Dict[str, Any]:
    df_burst, df_video = load_burstgpt_data(), load_uci_video_data()
    ws = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    train_windows = [ws.train_windows[k] for k in C.TRAIN_WINDOWS]
    val_windows = [ws.train_windows[k] for k in C.VALIDATION_WINDOWS]

    # (plateau_growth, plateau_patience, steps_per_episode) x edges  = 4 x 2 = 8
    stop_rules = [
        (1, 6, 300),          # relaxed early stop
        (1, 6, 600),          # relaxed + longer episodes
        (0, 10 ** 9, 300),    # no early stop: full 20 episodes
        (0, 10 ** 9, 600),    # no early stop + longer episodes
    ]
    edge_options = {"v1": None, "train_derived": C.RISK_BUCKET_EDGES}
    grid = list(itertools.product(stop_rules, edge_options.items()))

    if verbose:
        print(f"Stage 2: {len(grid)} configurations (12 already used; cap {C.MAX_TUNING_CONFIGS})")

    results: List[Dict[str, Any]] = []
    for i, ((growth, patience, steps), (edge_name, edges)) in enumerate(grid, start=13):
        t0 = time.time()
        cid = f"cfg{i:02d}"
        cfg = TrainingConfig(
            episodes=20, steps_per_episode=steps, seeds=C.TRAIN_SEEDS,
            target_rho=C.TRAIN_TARGET_RHO, epsilon_decay_per_step=0.999,
            epsilon_min=C.EPSILON_MIN, learning_rate=C.LEARNING_RATE,
            plateau_growth=growth, plateau_patience=patience,
        )
        if verbose:
            stop = "none" if patience > 1000 else f"g<{growth} x{patience}"
            print(f"[{cid}] stop={stop} steps={steps} edges={edge_name}", flush=True)

        note = ""
        try:
            trained = train_all_learners(train_windows, df_burst, df_video, cfg,
                                         risk_edges=edges, verbose=False)
            snapshots = {k: v["snapshot"] for k, v in trained.items()}
            agg = evaluate_on_validation(snapshots, edges, val_windows, df_burst, df_video)
            score = agg["completed_tasks"] / max(1e-9, agg["mean_risk_over_time"])
        except Exception as exc:
            note = f"FAILED {type(exc).__name__}: {exc}"
            trained, agg, score = {}, {}, float("nan")
            if verbose:
                print(f"   {note}")

        row = {
            "config_id": cid, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epsilon_decay": 0.999, "train_episodes": 20, "risk_edges": edge_name,
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
            "note": note or f"stage2 stop_growth={growth} patience={patience} steps={steps}",
        }
        _append_log(row)
        results.append(row)
        if verbose and not note:
            print(f"   episodes_run={row['episodes_run_marl']} q={row['final_q_marl']} "
                  f"completed={row['val_completed']} risk={row['val_mean_risk']} "
                  f"score={row['validation_score']}  ({row['seconds']}s)", flush=True)

    json.dump(results, open(MET / "tuning_stage2.json", "w"), indent=2, default=str)
    return {"results": results}


if __name__ == "__main__":
    run_stage2()
