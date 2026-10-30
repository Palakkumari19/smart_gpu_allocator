"""
PHASE 5 — THE FINAL TEST. RUNS ONCE.

Reads the frozen configuration from realdata/config_v2.py and the frozen
training settings chosen during validation tuning. Evaluates all six schedulers
across three conditions (v1 / v1_corrected / v2), both regimes (workload-only
and hazard overlay), all load levels, on TEST seeds 100-129.

This script must not be run before realdata/PREREGISTRATION_V2.md is committed.
It refuses to start if that file is missing or uncommitted.

Outputs
  realdata/results/realdata_v2_runs_<regime>_<ts>.csv
  realdata/results/realdata_v2_summary_<regime>_<ts>.json
  results/metrics/v2/...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from realdata import config_v2 as C
from realdata.config import BURSTGPT_WINDOWS_PATH, HAZARD_CURVES_PATH, RESULTS_DIR
from realdata.hazard import load_hazard_curves
from realdata.loaders import load_burstgpt_data, load_uci_video_data
from realdata.risk_overlay import RiskHazardOverlay
from realdata.windows import TraceWindowsSummary
from v2.improvements import TrainingConfig, train_all_learners
from v2.runner import SCHEDULERS, build_trace_for_condition, run_episode

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "metrics" / "v2"
PREREG = ROOT / "realdata" / "PREREGISTRATION_V2.md"


def _git(*args) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    except Exception:
        return ""


GUARDED_FILES = [
    ROOT / "realdata" / "PREREGISTRATION_V2.md",
    ROOT / "realdata" / "config_v2.py",
    ROOT / "results" / "metrics" / "v2" / "frozen_training.json",
]


def guard_preregistration(allow_dirty: bool = False) -> str:
    """
    Refuse to run unless the pre-registration AND the frozen configuration are
    committed and clean. Returns the pre-registration's commit hash, which is
    recorded in every output file.
    """
    for f in GUARDED_FILES:
        rel = str(f.relative_to(ROOT))
        if not f.exists():
            raise SystemExit(f"{rel} is missing. Create and commit it before the final test.")
        if not _git("ls-files", "--error-unmatch", rel):
            raise SystemExit(f"{rel} is not committed. Commit it before the final test.")
        if _git("status", "--porcelain", rel) and not allow_dirty:
            raise SystemExit(
                f"{rel} has uncommitted changes. Commit before the final test "
                "(the pre-registration must describe exactly what is run).")
    commit = _git("log", "-1", "--format=%H", "--", str(PREREG.relative_to(ROOT)))
    print("Guard passed. Committed and clean:")
    for f in GUARDED_FILES:
        print(f"   {f.relative_to(ROOT)}")
    return commit


def frozen_training_config() -> TrainingConfig:
    """Training settings chosen during validation tuning."""
    path = MET / "frozen_training.json"
    if path.exists():
        d = json.loads(path.read_text())
        return TrainingConfig(
            episodes=d["episodes"], steps_per_episode=C.TRAIN_STEPS_PER_EPISODE,
            seeds=tuple(C.TRAIN_SEEDS), target_rho=C.TRAIN_TARGET_RHO,
            epsilon_decay_per_step=d["epsilon_decay_per_step"],
            epsilon_min=C.EPSILON_MIN, learning_rate=C.LEARNING_RATE,
            plateau_growth=C.PLATEAU_GROWTH, plateau_patience=C.PLATEAU_PATIENCE,
        )
    return C.training_config()


def frozen_risk_edges():
    path = MET / "frozen_training.json"
    if path.exists():
        d = json.loads(path.read_text())
        return d.get("risk_edges")
    return C.RISK_BUCKET_EDGES


def v1_training_config() -> TrainingConfig:
    """Reproduces v1's training budget: 80 steps, one episode, shipped decay."""
    return TrainingConfig(
        episodes=1, steps_per_episode=80, seeds=(42,), target_rho=0.9,
        epsilon_decay_per_step=0.997, epsilon_min=0.05, learning_rate=0.01,
        plateau_growth=0, plateau_patience=10 ** 9,
    )


def run_regime(regime: str, conditions: List[str], df_burst, df_video, ws,
               snapshots_by_condition, overlay, verbose=True) -> pd.DataFrame:
    test_windows = {k: ws.eval_windows[k] for k in C.TEST_WINDOWS}
    rows: List[Dict[str, Any]] = []
    total = len(conditions) * len(test_windows) * len(C.LOAD_LEVELS) * len(C.TEST_SEEDS)
    done = 0
    t0 = time.time()

    for cond in conditions:
        snaps = snapshots_by_condition[cond]
        edges = frozen_risk_edges() if cond == "v2" else None
        steps = C.EPISODE_STEPS
        for wkey, window in test_windows.items():
            for rho in C.LOAD_LEVELS:
                for seed in C.TEST_SEEDS:
                    trace, meta = build_trace_for_condition(
                        cond, window, steps, rho, df_burst, df_video, seed,
                        partition="eval")
                    for sid in SCHEDULERS:
                        m = run_episode(
                            sid, trace, q_snapshots=snaps, seed=seed, overlay=overlay,
                            risk_edges=edges,
                            quarantine_thresholds=C.QUARANTINE_THRESHOLDS_REPORTED)
                        rows.append({
                            "condition": cond, "regime": regime, "window": wkey,
                            "target_rho": rho, "delivered_rho": meta["delivered_rho"],
                            "seed": seed, "scheduler": sid, "episode_steps": steps,
                            **m,
                        })
                    done += 1
                    if verbose and done % 10 == 0:
                        el = time.time() - t0
                        print(f"   {done}/{total} cells  ({el:.0f}s elapsed, "
                              f"~{el/done*(total-done):.0f}s left)", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regimes", nargs="+", default=["workload", "overlay"])
    ap.add_argument("--conditions", nargs="+", default=C.CONDITIONS)
    ap.add_argument("--allow-dirty-prereg", action="store_true")
    args = ap.parse_args()

    commit = guard_preregistration(args.allow_dirty_prereg)
    print(f"PREREGISTRATION_V2.md commit: {commit}")
    print(f"Config: steps={C.EPISODE_STEPS} loads={C.LOAD_LEVELS} "
          f"seeds={C.TEST_SEEDS[0]}-{C.TEST_SEEDS[-1]} windows={C.TEST_WINDOWS}")

    df_burst, df_video = load_burstgpt_data(), load_uci_video_data()
    ws = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    train_windows = [ws.train_windows[k] for k in C.TRAIN_WINDOWS]

    # Train once per condition family. v1 and v1_corrected share v1's training.
    print("\nTraining learners (identical budget within each condition) ...")
    v1_cfg, v2_cfg = v1_training_config(), frozen_training_config()
    t_v1 = train_all_learners(train_windows, df_burst, df_video, v1_cfg,
                              risk_edges=None, verbose=False)
    t_v2 = train_all_learners(train_windows, df_burst, df_video, v2_cfg,
                              risk_edges=frozen_risk_edges(), verbose=False)
    snaps_v1 = {k: v["snapshot"] for k, v in t_v1.items()}
    snaps_v2 = {k: v["snapshot"] for k, v in t_v2.items()}
    for name, t in (("v1", t_v1), ("v2", t_v2)):
        for sid, r in t.items():
            print(f"   {name:<4} {sid:<16} episodes={r['episodes_run']:<3} "
                  f"q={r['final_q_size']:<6} eps={r['final_epsilon']}")
    json.dump(
        {n: {k: {kk: vv for kk, vv in v.items() if kk != "snapshot"} for k, v in t.items()}
         for n, t in (("v1", t_v1), ("v2", t_v2))},
        open(MET / "final_training_report.json", "w"), indent=2, default=str)

    snapshots_by_condition = {"v1": snaps_v1, "v1_corrected": snaps_v1, "v2": snaps_v2}

    hazard_curve = None
    if "overlay" in args.regimes:
        catalog = load_hazard_curves(HAZARD_CURVES_PATH)
        hazard_curve = catalog.get_curve("burstgpt_dispersion")

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for regime in args.regimes:
        overlay = RiskHazardOverlay(hazard_curve) if regime == "overlay" else None
        print(f"\n=== REGIME: {regime} ===", flush=True)
        df = run_regime(regime, args.conditions, df_burst, df_video, ws,
                        snapshots_by_condition, overlay)
        out = RESULTS_DIR / f"realdata_v2_runs_{regime}_{ts}.csv"
        df.to_csv(out, index=False)
        print(f"saved {out}  ({len(df)} rows)")
        json.dump({"preregistration_commit": commit, "config": C.summary(),
                   "regime": regime, "rows": len(df), "timestamp_utc": ts},
                  open(RESULTS_DIR / f"realdata_v2_summary_{regime}_{ts}.json", "w"),
                  indent=2, default=str)

    print("\nFINAL TEST COMPLETE. Do not re-run or re-tune.")


if __name__ == "__main__":
    main()
