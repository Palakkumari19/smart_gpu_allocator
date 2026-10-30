"""
Measure the real per-cell cost of the final test, so the time estimate given to
the user is measured rather than guessed.

A "cell" = one (condition, window, load, seed): build one trace, then evaluate
all six schedulers on it.
"""

from __future__ import annotations

import time

import numpy as np

from realdata import config_v2 as C
from realdata.config import BURSTGPT_WINDOWS_PATH
from realdata.loaders import load_burstgpt_data, load_uci_video_data
from realdata.windows import TraceWindowsSummary
from v2.runner import SCHEDULERS, build_trace_for_condition, run_episode


def time_cell(condition, window, rho, seed, df_burst, df_video, edges):
    t0 = time.time()
    trace, meta = build_trace_for_condition(
        condition, window, C.EPISODE_STEPS, rho, df_burst, df_video, seed, partition="eval")
    t_build = time.time() - t0
    t1 = time.time()
    for sid in SCHEDULERS:
        run_episode(sid, trace, q_snapshots=None, seed=seed, overlay=None,
                    risk_edges=edges,
                    quarantine_thresholds=C.QUARANTINE_THRESHOLDS_REPORTED)
    return t_build, time.time() - t1, sum(len(s) for s in trace)


def main():
    df_burst, df_video = load_burstgpt_data(), load_uci_video_data()
    ws = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    wins = {k: ws.eval_windows[k] for k in C.TEST_WINDOWS}

    n_cells = len(C.TEST_WINDOWS) * len(C.LOAD_LEVELS) * len(C.TEST_SEEDS)
    print(f"cells per condition per regime = {len(C.TEST_WINDOWS)} windows x "
          f"{len(C.LOAD_LEVELS)} loads x {len(C.TEST_SEEDS)} seeds = {n_cells}")
    print(f"episode steps = {C.EPISODE_STEPS}, schedulers = {len(SCHEDULERS)}\n")

    per_condition = {}
    for cond in C.CONDITIONS:
        edges = C.RISK_BUCKET_EDGES if cond == "v2" else None
        samples = []
        # sample the corners of the grid so the average is representative
        for wkey, w in wins.items():
            for rho in (C.LOAD_LEVELS[0], C.LOAD_LEVELS[-1]):
                b, e, n = time_cell(cond, w, rho, 100, df_burst, df_video, edges)
                samples.append((b + e, n))
                print(f"  {cond:<13} {wkey:<17} rho={rho:<4} tasks={n:<6} "
                      f"build={b:.2f}s eval={e:.2f}s total={b+e:.2f}s")
        mean_cell = float(np.mean([s[0] for s in samples]))
        per_condition[cond] = mean_cell
        print(f"  -> {cond}: mean {mean_cell:.2f}s/cell, "
              f"{mean_cell * n_cells / 60:.1f} min per regime\n")

    print("=" * 64)
    both = {c: v * n_cells * 2 / 3600 for c, v in per_condition.items()}
    for c, h in both.items():
        print(f"  {c:<14} both regimes: {h:.2f} h")
    opt_b = sum(both.values())
    opt_a = both["v1_corrected"] + both["v2"]
    print(f"\n  OPTION (a) reuse existing v1, run v1_corrected + v2 : {opt_a:.2f} h")
    print(f"  OPTION (b) run all three conditions                 : {opt_b:.2f} h")
    print("  (plus ~2 min of training, done once)")


if __name__ == "__main__":
    main()
