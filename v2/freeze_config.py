"""
PHASE 4c — apply the selection rules mechanically and freeze the config.

Rules, applied in this order and exactly as written:
  1. THROUGHPUT FLOOR on config selection: a configuration whose mean
     val_completed is below 95% of the best configuration's is INELIGIBLE.
     (The score is a ratio and is NOT structurally immune to starving.)
  2. BEST = highest validation_score among eligible configurations.
  3. TIE BAND: configurations within 1% of the best score are ties.
     The band is applied mechanically and is not adjusted.
  4. Among ties, choose the SIMPLEST/CHEAPEST: fewest training steps
     (episodes x steps_per_episode), then fewest episodes, then lowest seconds.
  5. Risk-bucket edges are set to the TRAIN-derived edges on a pre-stated
     principle (84% of risk readings fell in a single v1 bucket), independently
     of the validation score, which did not favour them.

Also reports the standard error of validation_score (bootstrap over validation
episodes) so the reader can judge whether the selection is within noise.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

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

THROUGHPUT_FLOOR_CONFIG = 0.95
TIE_BAND = 0.01


def _steps_of(row) -> int:
    """Total training steps implied by a logged configuration."""
    m = re.search(r"steps=(\d+)", str(row.get("note", "")))
    steps = int(m.group(1)) if m else C.TRAIN_STEPS_PER_EPISODE
    eps = int(row.get("episodes_run_marl") or row.get("train_episodes") or 0)
    return eps * steps, steps


def select() -> Dict[str, Any]:
    df = pd.read_csv(LOG)
    df = df[df["note"].isna() | ~df["note"].astype(str).str.startswith("FAILED")]
    df = df.dropna(subset=["validation_score"])

    best_completed = float(df["val_completed"].max())
    cutoff = THROUGHPUT_FLOOR_CONFIG * best_completed
    df["eligible"] = df["val_completed"] >= cutoff

    elig = df[df.eligible]
    best_score = float(elig["validation_score"].max())
    band = best_score * (1.0 - TIE_BAND)
    ties = elig[elig["validation_score"] >= band].copy()

    tot = ties.apply(lambda r: _steps_of(r)[0], axis=1)
    per_ep = ties.apply(lambda r: _steps_of(r)[1], axis=1)
    ties["total_train_steps"] = tot
    ties["steps_per_episode"] = per_ep
    ties = ties.sort_values(["total_train_steps", "episodes_run_marl", "seconds"])
    chosen = ties.iloc[0]

    return {
        "n_configs_logged": int(len(df)),
        "throughput_floor_config": THROUGHPUT_FLOOR_CONFIG,
        "best_val_completed": best_completed,
        "completed_cutoff": cutoff,
        "n_ineligible_by_floor": int((~df.eligible).sum()),
        "best_score": best_score,
        "tie_band_lower_bound": band,
        "n_ties": int(len(ties)),
        "tie_config_ids": sorted(ties.config_id.tolist()),
        "chosen_config_id": str(chosen.config_id),
        "chosen": {k: (None if pd.isna(v) else v) for k, v in chosen.items()},
    }


def bootstrap_score_se(epsilon_decay, episodes, steps_per_episode, edges,
                       n_boot=400, verbose=True) -> Dict[str, Any]:
    """
    Standard error of validation_score = mean(completed) / mean(risk),
    bootstrapped over the validation episodes.
    """
    df_burst, df_video = load_burstgpt_data(), load_uci_video_data()
    ws = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    train_windows = [ws.train_windows[k] for k in C.TRAIN_WINDOWS]
    val_windows = [ws.train_windows[k] for k in C.VALIDATION_WINDOWS]

    cfg = TrainingConfig(
        episodes=episodes, steps_per_episode=steps_per_episode, seeds=C.TRAIN_SEEDS,
        target_rho=C.TRAIN_TARGET_RHO, epsilon_decay_per_step=epsilon_decay,
        epsilon_min=C.EPSILON_MIN, learning_rate=C.LEARNING_RATE,
        plateau_growth=1, plateau_patience=6,
    )
    if verbose:
        print("  retraining chosen config to collect per-episode validation rows ...")
    trained = train_all_learners(train_windows, df_burst, df_video, cfg,
                                 risk_edges=edges, verbose=False)
    snaps = {k: v["snapshot"] for k, v in trained.items()}

    rows = []
    for w in val_windows:
        for rho in C.LOAD_LEVELS:
            for sd in C.VALIDATION_SEEDS:
                trace, _ = build_calibrated_trace(
                    window=w, episode_steps=C.EPISODE_STEPS, target_rho=rho,
                    df_burst=df_burst, df_video=df_video, seed=sd, partition="train")
                rows.append(run_episode(C.TREATMENT, trace, q_snapshots=snaps, seed=sd,
                                        risk_edges=edges,
                                        quarantine_thresholds=C.QUARANTINE_THRESHOLDS_REPORTED))
    comp = np.array([r["completed_tasks"] for r in rows])
    risk = np.array([r["mean_risk_over_time"] for r in rows])
    point = comp.mean() / risk.mean()

    rng = np.random.default_rng(0)
    n = len(comp)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        boots.append(comp[idx].mean() / max(1e-9, risk[idx].mean()))
    boots = np.array(boots)
    se = float(boots.std(ddof=1))
    return {
        "n_validation_episodes": int(n),
        "validation_score_point": float(point),
        "validation_score_se": se,
        "se_pct_of_score": float(100.0 * se / point),
        "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
        "n_boot": n_boot,
    }


def main():
    sel = select()
    print(json.dumps({k: v for k, v in sel.items() if k != "chosen"}, indent=2, default=str))
    ch = sel["chosen"]
    print(f"\nCHOSEN: {sel['chosen_config_id']}  score={ch['validation_score']} "
          f"edges={ch['risk_edges']} note={ch.get('note')}")

    _, steps_per_ep = _steps_of(ch)
    # Principle override (pre-stated): always use TRAIN-derived edges.
    edges = C.RISK_BUCKET_EDGES
    se = bootstrap_score_se(
        epsilon_decay=float(ch["epsilon_decay"]),
        episodes=int(ch["episodes_run_marl"]),
        steps_per_episode=steps_per_ep,
        edges=edges,
    )
    print("\nvalidation_score standard error:")
    print(json.dumps(se, indent=2))

    frozen = {
        "selection": sel,
        "score_uncertainty": se,
        "selection_within_noise": se["se_pct_of_score"] > 100 * TIE_BAND,
        "epsilon_decay_per_step": float(ch["epsilon_decay"]),
        "episodes": int(ch["episodes_run_marl"]),
        "steps_per_episode": steps_per_ep,
        "risk_edges": edges,
        "risk_edges_rationale": (
            "TRAIN-derived edges kept on a pre-stated principle (84% of risk "
            "readings fell in one v1 bucket). Validation scores did NOT favour "
            "them (181.74 vs 182.85 for v1 edges at 600-step training), but the "
            "difference lies inside the 1% tie band."),
        "applies_to_all_learners": ["rl", "marl", "risk_aware_marl"],
    }
    json.dump(frozen, open(MET / "frozen_training.json", "w"), indent=2, default=str)
    print(f"\nwrote {MET / 'frozen_training.json'}")


if __name__ == "__main__":
    main()
