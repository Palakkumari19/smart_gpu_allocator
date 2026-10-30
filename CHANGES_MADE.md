# CHANGES_MADE.md

Every modification made to existing project source files during the "run the project and document it"
pass of **30 October 2026**, with the reason for each. The policy followed was: **change nothing unless
it actually prevents the code from running on this machine.**

---

## Summary

**Two files were changed: one code fallback, and the README (rewritten on request).**
No scheduler logic, no simulator physics, no reward function, no metric calculation and no test was
touched. No results were altered by this change.

---

## Change 1 — `realdata/generate_visualizations.py` (lines 21–22)

### What was wrong

The script contained an **absolute file path belonging to the original author's Linux computer**:

```python
ARTIFACT_DIR = Path("/home/bmsit/.gemini/antigravity/brain/3e352e17-fa78-444c-b888-445471b73dc2")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
```

This runs at **import time**, before any plotting happens. On macOS, `/home` is a special
system-managed directory that cannot be written to, so the script crashed immediately with:

```
OSError: [Errno 45] Operation not supported: '/home/bmsit'
```

The result: **none of the four report figures could be produced on this machine.** This was a hard
blocker, not a cosmetic issue.

### What was changed

```python
ARTIFACT_DIR = Path("/home/bmsit/.gemini/antigravity/brain/3e352e17-fa78-444c-b888-445471b73dc2")
try:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
except OSError:
    # Original absolute path exists only on the author's Linux machine.
    # Fall back to the in-repo figures directory so the script is portable.
    ARTIFACT_DIR = FIGURES_DIR
```

### Why this is safe

- The original path and the original `mkdir` call are **kept exactly as they were**. On the author's
  original Linux machine the `try` block succeeds and behaviour is byte-for-byte identical.
- The fallback only activates when the directory genuinely cannot be created.
- The fallback target, `FIGURES_DIR`, is `realdata/reports/figures/` — the directory the script already
  writes its primary copy of each figure to. The script saves every figure twice (once to each
  directory); after the fallback both copies simply land in the same place.
- **No plotting code, no data loading, no axis, no number and no colour was modified.** Only the
  destination folder of the duplicate copy changed.

---

## Change 2 — `README.md` (rewritten on request)

### Why

The user explicitly asked for the README to be updated and finalised after the measurement pass.

### What was wrong with the original

1. **Performance claims that today's runs do not reproduce.** It stated "36% improvement in learning
   performance", "MARL consistently outperforms traditional methods" and "+1-5% improvement over
   traditional methods". The measured results show parity, and the pre-registered hypotheses H1-H3
   were rejected.
2. **Seven broken links** of the form `file:///d:/BMSIT/7th%20sem%20-%20proj/...` — absolute Windows
   paths from the original author's machine.
3. **Incorrect run instructions.** It listed `python main.py` as "Basic Demo"; that file has no
   `__main__` block and produces no output.
4. **Incomplete install instructions** (`pip install numpy` only, versus the seven packages in
   `requirements.txt`).
5. **No mention** of `realdata/`, `benchmarking/`, `evaluation/` or `execution/` in the project
   structure, despite `realdata/` being the scientifically strongest part of the project.

### What the rewrite does

- Reports the **measured** results from 30 October 2026, with source tables for both real-data regimes,
  the seven-scenario benchmark and the ablation study.
- States the pre-registered verdicts plainly (H1-H3 rejected, H4 inconclusive, H5 partial).
- Keeps and corrects the detailed concurrency / locking documentation, which was accurate.
- Fixes all broken links to repository-relative paths.
- Adds a "Known issues and limitations" section listing the items above.
- Adds a documentation index pointing at `docs/01`-`07`.
- Flags that `MARL_SUPERIORITY_REPORT.md`, `MARL_IMPROVEMENTS_IMPLEMENTED.md` and
  `AETHERGRID_IMPLEMENTATION_REPORT.md` contain unreproduced claims.

**No code behaviour is affected by this change** — `README.md` is documentation only.

---

## Change 3 — v2 evaluation fix: corrected offered-load calibration (NEW FILES ONLY)

### What was wrong

`realdata/trace_builder.py` computes the arrival scale over the **window** duration
(`window_duration_seconds / seconds_per_step` = 3600 steps, line 77) but
`build_real_trace()` emits those arrivals into the **episode** length (`total_steps` = 120,
line 187). Arrivals are therefore delivered **exactly 30x too fast**. It also uses hardcoded
constants (`DEFAULT_AVG_MEMORY_GB = 4.54`, `DEFAULT_AVG_DURATION_STEPS = 10.61`) that do not
match the per-window calibrated task mappings.

Measured effect: the "rho = 0.6 / 0.9 / 1.2" presets actually delivered **11.3x to 64.5x**
cluster capacity. Evidence: `docs/07_diagnosis.md` Finding 2.

### What was changed

**No existing file was edited.** `realdata/trace_builder.py` is byte-for-byte unchanged, so the
v1 results remain exactly reproducible. The correction lives in a new module:

- `v2/load_model.py` — `calibrate_load_scale()` / `build_calibrated_trace()` compute the scale
  against the **same step count the trace is built with**, and measure the mean task footprint
  empirically from a probe trace instead of assuming constants. A short fixed-point refinement
  (`scale *= target / delivered`) corrects the residual sampling drift.
  `build_legacy_trace()` reproduces the v1 behaviour deliberately, so the v1 condition can be
  re-run unchanged on the new test seeds.

### Why this is an evaluation fix, not a scheduler change

It alters the offered workload **identically for all six schedulers**. No scheduler, reward,
risk, EFC or physics code is touched. Verification used offered-load numbers only
(delivered rho vs target); no scheduler comparison was inspected while verifying.

### Verification

New regression test `tests/v2/test_load_scale_v2.py` (11 tests, all passing) asserts that
delivered/intended is ~1.0 for every window and preset, that the legacy path still reproduces
the ~30x defect, and that delivered load increases monotonically with the target.

Measured after the fix (mean of seeds 100-102):

| Window | rho | v1 delivered | v2 delivered | v2 ratio |
|---|---|---|---|---|
| median_hour | 0.6 | 32.51 | 0.599 | 1.00x |
| median_hour | 0.9 | 48.56 | 0.902 | 1.00x |
| median_hour | 1.2 | 64.42 | 1.211 | 1.01x |
| burstiest_window | 0.6 | 11.34 | 0.619 | 1.03x |
| burstiest_window | 0.9 | 17.08 | 0.917 | 1.02x |
| burstiest_window | 1.2 | 22.86 | 1.196 | 1.00x |

---

## Things that were deliberately NOT changed

These are genuine problems that were found, but each was **recorded instead of fixed**, because none of
them blocked the project from running:

| File | Problem | Why it was left alone |
|---|---|---|
| `main_marl_demo.py` line 140 | Crashes with `KeyError: 'llm_tasks_generated'`. `environment.get_metrics()` actually returns `light_tasks_generated`, `medium_tasks_generated`, `heavy_tasks_generated` and `extreme_tasks_generated` — there is no key with `llm` in the name. | The crash happens in the **final optional verification section**, after the whole demonstration has already printed its results. Nothing downstream depends on it, and the same functionality is covered by `verify_phase3.py` and `verify_all_features.py`, which both run cleanly. |
| `main.py` | Running `python main.py` produces no output at all, because the file is an older FastAPI app with no `if __name__ == "__main__":` block. The README lists it as "Basic Demo". | Not broken — just mis-documented. The live server that actually works is `backend/server.py`. Correct instructions are given in `docs/02_how_to_run.md`. |
| `requirements.txt` | Does not list `httpx2` (required by 5 test files) or `torch` (required by 1 optional test file). | Adding packages to `requirements.txt` would be a source change, and installing extra packages was blocked in this environment. Recorded in `docs/05_results_and_findings.md` instead. |
| Everything listed in `realdata/NOTES.md` | Eight pre-existing quirks the original authors documented rather than fixed (float truncation on preemption, duplicated if/else branches, task-ID renaming, linear temperature model, etc.). | The project's own guardrails say to leave these alone so the measured baseline stays honest. That decision was respected. |

---

## Files added (nothing overwritten)

These are new and do not affect how the project runs:

- `docs/` — six documentation files
- `logs/` — console output of every run
- `results/graphs/`, `results/metrics/` — collected figures and metrics
- `sample_videos/sample.mp4` — a **copy** of the existing `data/sample.mp4`, placed where
  `phase4_real_video_demo.py` expects to find its input (`VIDEO_DIR = "sample_videos"`). This supplies
  missing input data; it is not a code change.
- `realdata/data/raw/BurstGPT_1.csv`, `realdata/data/raw/transcoding_mesurment.tsv` and the UCI zip —
  the two public datasets, downloaded from the exact URLs in `realdata/config.py`.
- `.venv/` — the Python virtual environment (already in `.gitignore`).

## Output files the project overwrote by re-running itself

These were regenerated by running the project's own scripts, exactly as intended. They are results,
not source code:

- `realdata/reports/figures/fig1..fig4*.png` — redrawn by `realdata/generate_visualizations.py`
- `realdata/reports/trace_summary.md` — rewritten by `realdata/trace_builder.py`
- `benchmarking/results/experiments_index.json` + five new `experiment_*` files — written by
  `benchmarking/run_experiments.py`
- `realdata/results/realdata_*_20261030_13*.csv|json` — written by `realdata/run_realdata_benchmark.py`
  (new timestamped files; the earlier runs were left untouched)
- `marl_training_results.json` — written by `simple_marl_training.py`

Note that `realdata/reports/data_profile.md` regenerated **byte-identically**, which independently
confirms the re-downloaded datasets match the ones originally used.
