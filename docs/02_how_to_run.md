# 02 — How to Run This Project

Every command below was **actually executed** on 30 October 2026 on macOS 14 (Apple Silicon) with
Python 3.14.6. The console output of each one is saved in `logs/`.

---

## 1. Prerequisites

### Software

| What | Version used | Notes |
|---|---|---|
| Python | **3.14.6** | 3.9 or newer should work. Check with `python3 --version`. |
| pip | latest | Comes with Python. |
| Node.js + npm | optional | Only needed for the React dashboard. |
| `curl` and `unzip` | preinstalled | Used to download the datasets. |

### Hardware

**No physical GPU is needed.** The GPUs in this project are simulated in software, so any ordinary
laptop works. A real GPU is only used by the optional PyTorch execution path in `execution/`.

Disk space: about **200 MB** — roughly 60 MB of downloaded datasets and 140 MB for the Python packages.

Time: the full set of runs below takes roughly **25 minutes**, almost all of it in the two real-data
benchmarks (about 7 minutes each).

### Datasets

Two public datasets are downloaded automatically. Both are Creative Commons Attribution 4.0.

| Dataset | Size | What it is |
|---|---|---|
| `BurstGPT_1.csv` | ~48.5 MB | 1,429,737 real AI inference requests over 61 days |
| `transcoding_mesurment.tsv` | ~8.5 MB | 168,286 real video-transcoding measurements |

---

## 2. Setup

```bash
# 1. Go to the project folder
cd /Users/palakkumari/Documents/projects/smart_gpu_allocator

# 2. Create an isolated Python environment so project packages
#    don't interfere with anything else on your computer
python3 -m venv .venv

# 3. Upgrade pip, then install everything the project needs
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt pytest
```

**Expected output:** a line ending with
`Successfully installed ... fastapi ... matplotlib ... numpy ... pandas ... opencv-python-headless ...`

**Two extra packages are needed for the full test suite but are missing from `requirements.txt`:**

```bash
.venv/bin/pip install httpx2   # needed by 5 test files that use FastAPI's TestClient
.venv/bin/pip install torch    # needed by 1 optional test (real GPU execution)
```

> **Note.** `PYTHONPATH=.` appears in front of most commands below. It tells Python to look in the
> project folder for its own modules. Without it you will see `ModuleNotFoundError: No module named
> 'environment'`.

---

## 3. Download the datasets

```bash
PYTHONPATH=. .venv/bin/python realdata/fetch_data.py
```

**Expected output:** progress messages, then both files appearing in `realdata/data/raw/`.

**Common error — and the fix that was needed on this machine:**

```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: unable to get local issuer certificate>
```

This happens on macOS when Python was installed from python.org and has no root certificates. Either
run `/Applications/Python\ 3.14/Install\ Certificates.command`, or download the files with `curl`,
which uses the system certificates:

```bash
cd realdata/data/raw

curl -sSL -o BurstGPT_1.csv \
  "https://raw.githubusercontent.com/HPMLL/BurstGPT/main/data/BurstGPT_1.csv"

curl -sSL -o uci335.zip \
  "https://archive.ics.uci.edu/static/public/335/online+video+characteristics+and+transcoding+time+dataset.zip"
unzip -o -q uci335.zip -d uci335_extract
cp uci335_extract/transcoding_mesurment.tsv .

cd ../../..
```

---

## 4. Run the tests

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

**Expected output (with `httpx2` and `torch` installed):** all tests passing.

**What actually happened here (without them):** 5 test files could not even be loaded. Skipping those:

```bash
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q \
  --ignore=tests/test_bit7_backend_websocket.py \
  --ignore=tests/test_bit8_dashboard_visualization.py \
  --ignore=tests/test_concurrency_locks.py \
  --ignore=tests/test_real_gpu_execution.py \
  --ignore=tests/test_scheduler_selection.py
```

**Result obtained:** `2 failed, 121 passed in 1.50s` — and both failures were only the same missing
`httpx2`, not a real defect. Log: `logs/01_pytest_collectable.log`.

---

## 5. Run the demos (fast — seconds each)

Run these in order. Each prints a report to the screen.

```bash
# Shows how jobs are generated: LLM, image and video workloads
PYTHONPATH=. .venv/bin/python main_workload_demo.py

# Shows the traffic patterns: light, heavy, burst, peak/off-peak
PYTHONPATH=. .venv/bin/python main_traffic_demo.py

# Runs the simple traditional schedulers (FCFS, Round Robin, Least Loaded)
PYTHONPATH=. .venv/bin/python main_baseline_demo.py

# Shows the improved MARL reward function and learning parameters
PYTHONPATH=. .venv/bin/python marl_improvements_summary.py

# Checks that every Phase 1–3 feature works
PYTHONPATH=. .venv/bin/python verify_phase3.py

# Broader feature verification across all phases
PYTHONPATH=. .venv/bin/python verify_all_features.py

# A short MARL training demonstration on several scenarios
PYTHONPATH=. .venv/bin/python simple_marl_training.py
```

All seven exit cleanly. Logs: `logs/run_<name>.log`.

### Two commands from the README that do NOT work as described

```bash
python main.py          # produces NO output
```
`main.py` is an **older, superseded FastAPI app** with no `if __name__ == "__main__":` block, so running
it does nothing at all. The README calls it "Basic Demo", which is incorrect. The working server is
`backend/server.py` (see §9).

```bash
PYTHONPATH=. .venv/bin/python main_marl_demo.py     # crashes at the very end
```
This produces its full demonstration correctly, then **crashes on the final optional check** with
`KeyError: 'llm_tasks_generated'` (line 140). The metrics dictionary uses
`light_/medium_/heavy_/extreme_tasks_generated`; there is no key containing `llm`. This is a
pre-existing bug and was deliberately left unfixed. The same ground is covered by `verify_phase3.py`.

---

## 6. Run the benchmarks (fast — under a minute each)

```bash
# Compares all 6 schedulers over 10 runs each with fixed seeds
PYTHONPATH=. .venv/bin/python benchmark_schedulers.py

# Head-to-head MARL vs traditional across scenarios and stress tests
PYTHONPATH=. .venv/bin/python marl_vs_traditional.py

# Detailed per-agent MARL performance analysis
PYTHONPATH=. .venv/bin/python marl_performance_analysis.py

# The 7-scenario scientific benchmark for risk-aware MARL (3 seeds each)
PYTHONPATH=. .venv/bin/python benchmark_risk_aware_marl.py

# The formal 6-algorithm comparison + the ablation study.
# Writes CSV and JSON into benchmarking/results/
PYTHONPATH=. .venv/bin/python benchmarking/run_experiments.py
```

**Expected output of the last one:** two summary tables, and file paths under `benchmarking/results/`.

---

## 7. Run the concurrency stress test

```bash
PYTHONPATH=. .venv/bin/python concurrency_stress_test.py --workers 8 --duration 2.5 --scheduler marl
```

**Expected output:** a results block ending in
`[OK] ALL CONCURRENCY INVARIANTS SATISFIED (Zero Violations)`.

Measured here: 902 operations in 2.531 s (356.4 ops/sec), 607 legitimate capacity rejections,
6 duplicate completions blocked, **0 invariant violations**.

---

## 8. Run the video execution demos

```bash
# Simulated video rendering at four intensity levels
PYTHONPATH=. .venv/bin/python phase4_execution_demo.py

# Real video files analysed with OpenCV and turned into jobs.
# This script looks for a folder named sample_videos/, which is not in the repo.
mkdir -p sample_videos && cp data/sample.mp4 sample_videos/
PYTHONPATH=. .venv/bin/python phase4_real_video_demo.py
```

Without that `mkdir` step the second script stops with
`FileNotFoundError: Video directory not found: sample_videos`.

---

## 9. Run the real-data pipeline (the main experiment)

Run these **in order** — each one writes a file the next one reads.

```bash
# 1. Statistical profile of both datasets -> realdata/reports/data_profile.md
PYTHONPATH=. .venv/bin/python realdata/profile_data.py

# 2. Split the 61-day trace into TRAIN (days 0-30) and EVAL (days 31-61),
#    and pick the representative hours to replay
PYTHONPATH=. .venv/bin/python realdata/windows.py

# 3. Fit the percentile mappings from real measurements to simulator job sizes
PYTHONPATH=. .venv/bin/python realdata/video_mapper.py
PYTHONPATH=. .venv/bin/python realdata/llm_mapper.py

# 4. Measure the real-world failure rate as a function of traffic burstiness
PYTHONPATH=. .venv/bin/python realdata/hazard.py

# 5. Build the replay trace (80% LLM, 20% video) at three load levels
PYTHONPATH=. .venv/bin/python realdata/trace_builder.py

# 6. THE MAIN EXPERIMENT — 180 episodes, workload replay only  (~7 minutes)
PYTHONPATH=. .venv/bin/python realdata/run_realdata_benchmark.py

# 7. THE MAIN EXPERIMENT — 180 episodes, with the real failure overlay  (~7 minutes)
PYTHONPATH=. .venv/bin/python realdata/run_realdata_benchmark.py --with-hazard-overlay

# 8. Draw the four report figures
PYTHONPATH=. .venv/bin/python realdata/generate_visualizations.py
```

**Expected output of step 4:** a table showing the failure rate rising from about 1 % at low burstiness
to **26.25 %** at high burstiness.

**Expected output of steps 6 and 7:** per-scheduler tables, the pre-registered hypothesis tests, and
`RESULTS SUCCESSFULLY SAVED` with paths under `realdata/results/`.

**Quick version** if you only want to check it works — about 30 seconds instead of 7 minutes:

```bash
PYTHONPATH=. .venv/bin/python realdata/run_realdata_benchmark.py --smoke
```

**Common error:** `OSError: [Errno 45] Operation not supported: '/home/bmsit'` in step 8. The script
contained a hardcoded path from the original author's Linux machine. This was fixed with a fallback —
see `CHANGES_MADE.md`.

---

## 10. Generate the summary graphs

```bash
PYTHONPATH=. .venv/bin/python results/make_architecture_diagram.py
PYTHONPATH=. .venv/bin/python results/make_graphs.py
```

Writes eight charts into `results/graphs/` and the underlying numbers into `results/metrics/`.
These scripts only read result files the project already produced — they invent nothing.

---

## 11. Run the live dashboard (optional)

```bash
# Terminal 1 — the backend
PYTHONPATH=. .venv/bin/uvicorn backend.server:app --reload --port 8000

# Terminal 2 — the frontend
cd frontend
npm install
npm run dev
```

Then open the URL that Vite prints (usually `http://localhost:5173`).

---

## 12. Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'environment'` | Python can't find the project's own modules | Put `PYTHONPATH=.` in front of the command and run from the project root |
| `RuntimeError: The starlette.testclient module requires the httpx2 package` | Test-only dependency missing from `requirements.txt` | `.venv/bin/pip install httpx2` |
| `ModuleNotFoundError: No module named 'torch'` | Optional dependency for the real-GPU test | `.venv/bin/pip install torch`, or skip that one test file |
| `SSL: CERTIFICATE_VERIFY_FAILED` during download | macOS Python has no root certificates | Run `Install Certificates.command`, or download with `curl` as shown in §3 |
| `OSError: [Errno 45] Operation not supported: '/home/bmsit'` | Hardcoded Linux path in the figure script | Already fixed — see `CHANGES_MADE.md` |
| `FileNotFoundError: Video directory not found: sample_videos` | Input folder missing | `mkdir -p sample_videos && cp data/sample.mp4 sample_videos/` |
| `KeyError: 'llm_tasks_generated'` | Pre-existing bug at the end of `main_marl_demo.py` | Harmless — the demo already printed its results. Use `verify_phase3.py` instead |
| `python main.py` prints nothing | It is a superseded file with no entry point | Use `backend/server.py` for the server |
| Real-data scripts fail with a missing-file error | Datasets not downloaded yet | Run §3 first |
