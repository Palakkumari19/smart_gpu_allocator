# CHANGES_MADE.md

Every modification made to existing project source files during the "run the project and document it"
pass of **30 October 2026**, with the reason for each. The policy followed was: **change nothing unless
it actually prevents the code from running on this machine.**

---

## Summary

**One source file was changed. One line of behaviour was added as a fallback.**
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
