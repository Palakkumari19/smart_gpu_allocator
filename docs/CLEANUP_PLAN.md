# Cleanup plan — final state of branch `v2-improvements`

Written 31 October 2026, before any files were touched. Every candidate was checked for
references with `git grep` across the whole repository first.

**Nothing cited as evidence by `docs/`, `README.md`, `CHANGES_MADE.md` or either
pre-registration is deleted.** Superseded material is moved to `archive/` with a banner,
never removed, so the links in those documents keep resolving.

---

## DELETE — regenerable junk only

| Item | Count | Why it is safe |
|---|---|---|
| `__pycache__/` directories | 13 | Python bytecode, regenerated on import. Untracked; already in `.gitignore` |
| `*.pyc` files | 77 | Same. Untracked |
| `uvicorn.log` | 1 | 2-line error log from the **original author's Linux machine** (`/home/bmsit/…`, "Address already in use"). Not referenced anywhere. Not our evidence |

Nothing else is deleted. No tracked source, result, log or figure is removed.

## ARCHIVE — `git mv` to `archive/`, with a SUPERSEDED banner

These contain performance claims that today's runs do **not** reproduce. `README.md` and
`CHANGES_MADE.md` already warn about three of them; those links are updated to the new paths.

| File | Superseded claim it contains |
|---|---|
| `MARL_SUPERIORITY_REPORT.md` | *"MARL Victory: Demonstrated superior coordination"*, 60 % win rate |
| `MARL_IMPROVEMENTS_IMPLEMENTED.md` | Improvement claims not reproduced in the v1 or v2 runs |
| `AETHERGRID_IMPLEMENTATION_REPORT.md` | Performance figures not reproduced |
| `MARL_IMPLEMENTATION_SUMMARY.md` | "IMPLEMENTATION COMPLETE" status claims; unreferenced anywhere |
| `benchmark_clean.txt` | Saved BIT-6 era benchmark output, superseded by the pre-registered runs |

## KEEP — everything else

| Item | Why |
|---|---|
| `logs/` (38 tracked files) | **Cited as evidence** throughout `docs/05`, `README.md`, `CHANGES_MADE.md` |
| `logs/run_main.log` (0 bytes) | Its **emptiness is the evidence** for the documented "`main.py` produces no output" finding (README §Known issues, docs/05 §10) |
| `logs/v2/`, `logs/v2_tuning_log.csv` | Final-test console output and all 20 tuning configurations |
| `results/`, `realdata/results/` | All figures, metrics and the 6,480-episode result CSVs |
| `realdata/PREREGISTRATION*.md` | Must stay untouched and visible by instruction |
| `BIT6`–`BIT9`, `MARL_TRAINING_GUIDE.md` | Architecture/design docs linked from README's doc table; describe the system, not superseded results |
| `marl_training_results.json` | Written by `simple_marl_training.py`; cited in `CHANGES_MADE.md` |
| `docs/`, `v2/`, `tests/`, all source | Current work |
| `realdata/data/raw/*` | Datasets — already untracked and ignored; `realdata/fetch_data.py` downloads them |

## Not touched

- No scheduler, config, pre-registration, result file, summary or PPT is modified.
- No experiment is re-run and no tuning is repeated.
- `.venv/` and `node_modules/` are untracked and ignored; they are left on disk.

## Verification after every batch

`.venv/bin/python -m pytest tests/ -q` must stay at **181 passed**.
