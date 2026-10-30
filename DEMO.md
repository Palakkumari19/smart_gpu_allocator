# Demo cheat sheet — panel presentation

**Every command below was actually run on this machine on 31 October 2026 and its output checked.**
Times are measured, not estimated. Nothing here needs a GPU or the internet.

Run everything from the project root unless a command says otherwise:

```bash
cd /Users/palakkumari/Documents/projects/smart_gpu_allocator
```

> **Longest command is ~10 seconds.** Nothing here is a long-running experiment. The full v2
> experiment takes 3.3 hours and is **not** part of the demo — show its saved results instead (§6).

---

## 1. Environment and sanity check

### 1a. Check Python — *instant*
```bash
.venv/bin/python --version
```
**Panel sees:** `Python 3.14.6`

### 1b. Fast test subset — *0.6 s*
```bash
.venv/bin/python -m pytest tests/test_marl_bidding.py tests/test_marl_reward.py tests/test_gpu_memory_accounting.py -q
```
**Panel sees:** `9 passed in 0.07s` — bidding logic, reward function and memory accounting.

### 1c. Full test suite — *9.6 s*
```bash
.venv/bin/python -m pytest tests/ -q
```
**Panel sees:** `181 passed in 8.96s`, zero skipped. This is the headline quality number.

> You do not need to "activate" anything — calling `.venv/bin/python` directly uses the virtual
> environment. If you prefer activating: `source .venv/bin/activate`, then `python` works plainly.

---

## 2. Simplest demo — the schedulers working — *under 1 s*

```bash
PYTHONPATH=. .venv/bin/python main_baseline_demo.py
```

**What it does:** runs FCFS, Round Robin and Least Loaded against the same tasks and shows which GPU
each one picks.

**Panel sees:** 125 lines ending in `PHASE 3 STEP 5 COMPLETE`. Point at the three
`--- <name> Scheduler ---` blocks and the differing GPU choices — that is scheduling policy visible
in one screen.

---

## 3. Small benchmark — Risk-Aware vs Risk-Blind vs baselines — *under 1 s*

```bash
PYTHONPATH=. .venv/bin/python benchmarking/run_experiments.py
```

**What was reduced:** nothing was reduced for the demo — this script is *already* small by design
(**3 seeds, 60 steps per run**, set in `benchmarking/run_experiments.py:37,39`). That is why it
finishes instantly. It is a smoke-scale comparison, **not** the pre-registered experiment.

**Panel sees:** two tables. The first compares all six schedulers; the second is the ablation:

```
Full Risk-Aware MARL       | 4990.05 t/s | 100.0% | Avg Risk 0.0720 | Avg EFC 0.0275
Standard MARL (Risk-Blind) | 4932.85 t/s | 100.0% | Avg Risk 0.0473 | Avg EFC 0.0181
```

> ⚠️ **Say this out loud:** these are 3-seed smoke numbers with no confidence intervals. The real
> comparison is the 6,480-episode pre-registered run in §6. Do **not** present these as the result.

**Note:** this writes ~5 new files into `benchmarking/results/`. That is normal and harmless; the
committed results are separate, timestamped files.

---

## 4. Concurrency stress test — *3 s*

```bash
PYTHONPATH=. .venv/bin/python concurrency_stress_test.py --workers 8 --duration 2.5 --scheduler marl
```

**What it does:** hammers the scheduler with 8 threads at once and checks nothing corrupts.

**Panel sees:**
```
Operations Attempted  : 939 (365.4 ops/sec)
Duplicates Prevented  : 8 (Atomic CAS Guards)
Invariant Violations  : 0
[OK] ALL CONCURRENCY INVARIANTS SATISFIED (Zero Violations)
```
The exact counts vary run to run (it is timing-dependent); **`Invariant Violations : 0` is the line
that matters** and it is deterministic.

---

## 5. Backend + dashboard

`npm install` has **already been run** (139 packages installed), so the demo needs no internet.

### 5a. Terminal 1 — backend — *ready in ~6 s*
```bash
cd backend && PYTHONPATH=.. ../.venv/bin/python -m uvicorn server:app --host 127.0.0.1 --port 8000
```
**Wait for:** `INFO: Application startup complete.`

Verify in another terminal — *instant*:
```bash
curl -s http://127.0.0.1:8000/api/status
```
**Panel sees:** live JSON — `"total_gpus":4, "healthy_gpus":4, "scheduler_mode":"marl"`.

### 5b. Terminal 2 — dashboard — *ready in ~1 s*
```bash
cd frontend && npm run dev
```
**Open:** **http://localhost:3000/**
(Vite prints a harmless CommonJS config warning above the URL — ignore it.)

**Panel sees:** the live dashboard — four GPU cards with temperature and memory, the task queue, the
risk panel, and the scheduler selector, all updating over a WebSocket.

### 5c. Stop both
Press **Ctrl-C** in each terminal. If a port stays stuck:
```bash
lsof -ti:8000 | xargs kill
lsof -ti:3000 | xargs kill
```

---

## 6. The result figures — *instant*

```bash
open results/graphs/v2/v2_01_choice_availability.png
open results/graphs/v2/v2_07_load_fix_validity.png
open results/graphs/v2/v2_08_primary_hypothesis.png
open results/graphs/v2/v2_09_mean_vs_peak.png
open results/graphs/v2/v2_11_v1_vs_v2.png
open results/graphs/v2/v2_12_quarantine.png
open results/graphs/03_hazard_curves.png
open results/graphs/00_architecture.png
```

| Figure | One line to say |
|---|---|
| `v2_01_choice_availability` | 99.28 % of v1's decisions had **no GPU to choose from** — why v1 proved nothing |
| `v2_07_load_fix_validity` | After the fix, delivered load matches target and ~22 % of decisions are real choices |
| `v2_08_primary_hypothesis` | **The pre-registered test is REJECTED** — better in only 6 of 60 episodes |
| `v2_09_mean_vs_peak` | Worst on *mean* risk, **best on *peak* risk** — the metric rewarded the wrong thing |
| `v2_11_v1_vs_v2` | Almost the whole improvement came from the **bug fix**, not our clever changes |
| `v2_12_quarantine` | The shipped safety threshold fired **0 times in 3,240 episodes** |
| `03_hazard_curves` | Real data: failures rise from ~1 % to **26.25 %** under bursty traffic |
| `00_architecture` | System architecture, if the panel asks how it fits together |

Open the whole v2 set at once:
```bash
open results/graphs/v2/
```

---

## 7. The documents to show

```bash
open realdata/PREREGISTRATION_V2.md     # what we committed to BEFORE running the test
open realdata/PREREGISTRATION.md        # the original v1 pre-registration
open docs/05_results_and_findings.md    # Part I = v1, Part II = v2 (start at §11 and §14)
open docs/SUMMARY_SIMPLE.md             # plain-language summary
open CHANGES_MADE.md                    # every change to the original code, and why
```

**The one thing to show if you only have 30 seconds:** `realdata/PREREGISTRATION_V2.md`, then
`v2_08_primary_hypothesis.png`. Pre-registered claim, then the rejection. That pairing is the project.

Prove the pre-registration predates the run — *instant*:
```bash
git log -1 --format="%H %ci" -- realdata/PREREGISTRATION_V2.md
```
**Panel sees:** commit `3a0bcc3…` dated **before** the final test ran. That is the integrity proof.

---

## 8. Troubleshooting

| Problem | Symptom | Exact fix |
|---|---|---|
| **Port 8000 in use** | `ERROR: [Errno 48] Address already in use` | `lsof -ti:8000 \| xargs kill` then restart. Or use another port: `--port 8001` (then start Vite with `VITE_API=http://127.0.0.1:8001`) |
| **Port 3000 in use** | Vite says "Port 3000 is in use, trying another" | Let it pick the next port and **read the URL it prints** — it will say `http://localhost:3001/` |
| **venv not active / wrong Python** | `ModuleNotFoundError`, or `python` is 3.9 | Always call `.venv/bin/python` explicitly, never bare `python`. Check with `.venv/bin/python --version` → must say 3.14.6 |
| **`ModuleNotFoundError: No module named 'environment'`** | Running a script from the wrong folder | Add `PYTHONPATH=.` and run from the project root |
| **`npm: command not found`** | nvm not loaded in a fresh terminal | `source ~/.nvm/nvm.sh && nvm use 20` — node must be v20.20.0 |
| **npm wants to install** | `node_modules` missing | Already installed. If it vanished: `cd frontend && npm install` (needs internet — do it **tonight**, not at the demo) |
| **Missing dataset** | `FileNotFoundError: BurstGPT_1.csv` | Only the real-data pipeline needs it, and **no demo command above does**. If asked: `PYTHONPATH=. .venv/bin/python realdata/fetch_data.py` (needs internet, ~57 MB). The saved results in `results/` and `realdata/results/` do **not** need the dataset |
| **A figure won't open** | `open` does nothing | Use Finder: `open results/graphs/v2/` — or show the copies in `docs/demo_backup/figures/` |

---

## 9. BACKUP PLAN — if anything fails live

Every demo command's real output is saved in **`docs/demo_backup/`**. Open the file instead of
running the command and say *"here's the output from last night's run"*.

```bash
open docs/demo_backup/
```

| If this fails | Open this instead |
|---|---|
| §1c full test suite | `docs/demo_backup/01_pytest_full_suite.txt` |
| §2 scheduler demo | `docs/demo_backup/02_baseline_schedulers_demo.txt` |
| §3 six-scheduler benchmark | `docs/demo_backup/03_six_scheduler_benchmark.txt` |
| §4 concurrency stress test | `docs/demo_backup/04_concurrency_stress_test.txt` |
| §5a backend | `docs/demo_backup/05_backend_api_status.txt` |
| §5b dashboard | `docs/demo_backup/05_frontend_vite_startup.txt` |
| §6 any figure | `docs/demo_backup/figures/` |

**Total fallback:** the slide deck `~/Downloads/project_updated_v2.pptx` contains every figure and
number already. If the laptop misbehaves entirely, present from the deck.

---

## Suggested order on the day

1. `pytest -q` (§1c) — "181 tests, zero skipped"
2. `main_baseline_demo.py` (§2) — "here are schedulers making decisions"
3. Dashboard (§5) — "here it is running live"
4. `v2_01` then `v2_07` (§6) — "our first experiment was broken, here's the proof and the fix"
5. `PREREGISTRATION_V2.md` + `v2_08` (§7, §6) — "we committed the claim, then it was rejected"
6. `v2_09` (§6) — "but here's the interesting part: best on peak risk, worst on mean"
7. `concurrency_stress_test.py` (§4) — "and the engineering is solid: zero violations"
