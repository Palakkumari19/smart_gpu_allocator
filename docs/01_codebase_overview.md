# 01 — Codebase Overview

**Project:** SmartGPU Allocator — An Autonomous Multi-Agent Reinforcement Learning Framework for Crash-Aware GPU Orchestration in AI-Native Cloud Systems
**Repository folder:** `smart_gpu_allocator` (internal codename used throughout the code: *AetherGrid*)
**Date this overview was written:** 30 October 2026
**Size:** ~19,000 lines of Python across ~60 modules, plus a React dashboard

---

## 1. What this project is, in one paragraph

This is a **simulator** of a small GPU cluster (4 GPUs, 8 GB of video memory each) plus a set of
**schedulers** — programs that decide which GPU each incoming job should run on. The project's goal is
to test whether a scheduler built from **reinforcement learning** that is also aware of **how likely each
GPU is to crash** can do better than simple, traditional rules. It then replays **real production data**
(real AI inference requests and real video-transcoding measurements) through the simulator to check
whether the answer holds up outside of made-up workloads.

**Jargon, defined once:**

| Term | Plain meaning |
|---|---|
| **GPU** | Graphics Processing Unit — the chip that runs AI workloads. Here it is simulated, not real hardware. |
| **Scheduler** | The decision-maker: given a job, which GPU should run it? |
| **Reinforcement Learning (RL)** | A program that learns by trial and error. It takes an action, gets a numeric reward, and adjusts to earn more reward next time. |
| **Q-learning** | A specific RL method. It keeps a lookup table ("Q-table") of *situation → action → how good that was*. |
| **Multi-Agent RL (MARL)** | Instead of one brain deciding for the whole cluster, each GPU gets its own learning agent, and they bid against each other for jobs. |
| **Risk score** | A number from 0 to 1 saying how dangerous a GPU currently looks (hot, full, recently crashed, etc.). |
| **EFC (Expected Failure Cost)** | A forecast of how much a placement will cost if it goes wrong: *chance of failure × how bad the failure is × how long recovery takes*. |
| **Quarantine** | Taking a GPU out of service because its risk score got too high. |
| **Episode** | One complete simulation run of a fixed number of time steps. |
| **Trace** | A recorded list of real jobs with their arrival times and sizes, replayed into the simulator. |

---

## 2. Folder structure and what each part does

### 2.1 The simulated hardware (the "physics")

| File | What it does |
|---|---|
| `gpu.py` | One simulated GPU. Tracks memory used, temperature, running jobs, queue, memory fragmentation, crash state and cooldown. Crashes above 85 °C. Capacity is 8 GB. |
| `task.py` | The simplest job object (used by the oldest demos). Random 1–4 GB, 2–5 steps. |
| `workload/task_generator.py` | The richer job object, `EnhancedTask`, with job type (LLM / Image / Video), priority and severity. Also generates batches of jobs. |
| `workload/traffic_patterns.py` | Controls how *many* jobs arrive per step — light, heavy, burst, peak/off-peak. |
| `environment.py` | The main engine (~1,600 lines). Holds the 4 GPUs, advances the clock one step at a time, routes jobs through whichever scheduler is selected, and collects all statistics. This is the single most important file. |

### 2.2 The decision-makers (schedulers)

| File | What it does |
|---|---|
| `schedulers/traditional.py` | The simple baselines: FCFS (first come first served), Round Robin (take turns), Least Loaded (pick the emptiest), Random, Best Fit, Priority. |
| `agent.py` | Single-agent Q-learning: one brain for the whole cluster. |
| `marl_agent.py` | The core contribution (~850 lines). `MARLAgent` = one learning agent per GPU. `MARLManager` = runs the auction between them, computes the risk and EFC penalties, and assembles the reward. |

### 2.3 Safety and failure modelling

| File | What it does |
|---|---|
| `risk_engine.py` | Turns a GPU's live condition into a single risk score in [0, 1] using eight weighted ingredients (see §4). Also decides quarantine. |
| `fault_injection.py` | Deliberately injects simulated faults (memory errors, thermal events, etc.) so the schedulers can be tested against failures. Also contains the failure-impact and EFC cost calculators. |
| `dynamic_queue.py` | A thread-safe job queue tracking every job's lifecycle: QUEUED → SCHEDULED → RUNNING → COMPLETED / FAILED / CANCELLED, with retries. |

### 2.4 Measurement and experiments

| File / folder | What it does |
|---|---|
| `benchmarking/` | A reusable experiment harness: `workload_generator.py` builds job streams, `scheduler_adapters.py` wraps all 6 schedulers behind one interface, `metrics.py` computes throughput/utilisation/fairness/risk, `benchmark_runner.py` and `ablation_runner.py` run the experiments, `results_store.py` saves CSV and JSON. `run_experiments.py` is the entry point. |
| `evaluation/stats.py` | The statistics: bootstrap confidence intervals, paired differences, win rates. This is what makes the claims defensible rather than anecdotal. |
| `evaluation/scientific_benchmark.py`, `evaluation/workload_trace.py` | The 7-scenario scientific benchmark and the trace-replay plumbing. |
| `benchmark_schedulers.py`, `marl_vs_traditional.py`, `marl_performance_analysis.py`, `benchmark_risk_aware_marl.py` | Standalone comparison scripts, each producing a printed report. |

### 2.5 The real-data pipeline (`realdata/`) — the scientifically strongest part

This folder replays **genuine production data** instead of invented jobs.

| File | What it does |
|---|---|
| `config.py` | All paths, dataset URLs and expected column names. |
| `fetch_data.py` | Downloads the two public datasets. |
| `loaders.py` | Reads and validates them. |
| `profile_data.py` | Writes a statistical profile of both datasets to `reports/data_profile.md`. |
| `windows.py` | Splits the 61-day trace into a TRAIN half (days 0–30) and an EVAL half (days 31–61) so the learning agents are never tested on data they trained on. Picks representative hours. |
| `llm_mapper.py`, `video_mapper.py` | Convert real measurements into simulator job sizes using **percentile mapping** (preserves the ranking of small vs large jobs without inventing a scale). |
| `hazard.py` | Measures, from the real data, how the real-world failure rate changes with how "bursty" traffic is. |
| `trace_builder.py` | Assembles the final replay trace: 80 % LLM jobs, 20 % video jobs, rescaled to three load levels. |
| `risk_overlay.py` | Applies the measured real-world failure probability on top of the simulation, without editing the simulator. |
| `run_realdata_benchmark.py` | The main experiment: 6 schedulers × 2 time windows × 3 load levels × 5 random seeds = 180 episodes per regime. |
| `generate_visualizations.py` | Draws the four report figures. |
| `PREREGISTRATION.md` | The hypotheses and pass/fail rules, written **before** the results. This is why the project can report a negative result honestly. |
| `NOTES.md` | A list of pre-existing quirks in the original code that were deliberately *not* fixed, so the baseline stays untouched. |

### 2.6 Live demo (web app)

| File | What it does |
|---|---|
| `backend/server.py` | FastAPI server (~1,400 lines). REST endpoints plus a WebSocket that streams the live cluster state. Can accept an uploaded video and turn it into real jobs. |
| `backend/video_analyzer.py` | Reads a video file with OpenCV to estimate how much GPU memory and time it would need. |
| `execution/` | Optional path for running *real* GPU work via PyTorch instead of simulating it. |
| `frontend/` | React 19 + Vite + Tailwind + Recharts dashboard with 7 panels (cluster view, MARL decisions, risk, task graph, execution, benchmarking). |
| `main.py` | **An older, superseded FastAPI app.** It has no `if __name__ == "__main__"` block, so `python main.py` does nothing. The README incorrectly calls this "the basic demo". |

### 2.7 Tests

`tests/` holds 17 test files; `tests/realdata/` holds 9 more covering the real-data pipeline.

---

## 3. How the pieces connect

```
Job source                Decision                 Hardware             Measurement
─────────────────────────────────────────────────────────────────────────────────────
task_generator.py  ─┐
traffic_patterns.py ─┼──► environment.py ──► schedulers/traditional.py ─┐
realdata trace     ─┘        (the clock)      agent.py (RL)             ├──► gpu.py ──► metrics
                                              marl_agent.py (MARL)     ─┘      ▲
                                                      ▲                        │
                                        risk_engine.py + fault_injection.py ───┘
```

Every simulation step, `environment.step()` does roughly this:

1. Ask the workload source how many new jobs arrived.
2. For each job, ask the selected scheduler which GPU should take it.
3. Try to place it (`_try_assign`) — this is transactional: if anything fails, the memory is given back.
4. Advance every GPU by one tick: run jobs, raise temperature, maybe crash, maybe recover.
5. Update each GPU's risk score.
6. If a learning scheduler is active, hand out rewards and update the Q-tables.
7. Record statistics and append to history.

---

## 4. The two formulas at the heart of the project

**Risk score** (`risk_engine.py`) — a weighted blend, clamped to [0, 1]:

| Ingredient | Weight |
|---|---|
| Past anomalies | 0.25 |
| Past task failures | 0.20 |
| Temperature | 0.15 |
| Crash history | 0.15 |
| Memory pressure | 0.10 |
| Memory fragmentation | 0.05 |
| Performance degradation | 0.05 |
| Utilisation | 0.05 |

Bands: healthy < 0.25, low risk < 0.50, suspicious < 0.75, high risk ≥ 0.75. **Quarantine at ≥ 0.85.**

**Expected Failure Cost** (`marl_agent.py`):

```
EFC = failure_probability × failure_impact × recovery_cost      (clamped to [0, 20])

failure_probability = 0.70 × risk_score + 0.30 × observed_failure_rate
failure_impact      = 0.6 × severity + 0.4 × resource_intensity
recovery_cost       = max(1, 0.3 × runtime + 0.1 × memory) + cooldown_remaining
```

The MARL reward is bounded to **[−20.0, +12.0]** by `RiskAwareRewardConfig`, which prevents one
catastrophic step from destroying the learned table.

---

## 5. Environment needed to run this

| Requirement | Detail |
|---|---|
| **Python** | 3.9+. Verified working on **3.14.6** (macOS, Apple Silicon). |
| **Hardware** | CPU only. **No physical GPU is required** — the GPUs are simulated. A real GPU is only used by the optional `execution/` PyTorch path. |
| **Core packages** | `numpy`, `pandas`, `matplotlib`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `opencv-python-headless` (all listed in `requirements.txt`), plus `pytest` for tests. |
| **Missing from requirements.txt** | `httpx2` (needed by 5 test files that use FastAPI's TestClient) and `torch` (needed by 1 optional test). See `docs/05_results_and_findings.md`. |
| **Datasets** | Not stored in the repo. Two public files are downloaded into `realdata/data/raw/`: `BurstGPT_1.csv` (~48.5 MB) and `transcoding_mesurment.tsv` (~8.5 MB). Both are CC-BY-4.0. |
| **Environment variables** | None required. |
| **Node.js** | Only for the optional React dashboard (`frontend/`). |

---

## 6. Where results are written

| Location | Contents |
|---|---|
| `benchmarking/results/` | CSV and JSON from the 6-scheduler comparison and the ablation study. |
| `realdata/results/` | CSV and JSON from the 180-episode real-data runs. |
| `realdata/reports/` | `REAL_DATA_REPORT.md`, `data_profile.md`, `trace_summary.md`, and `figures/`. |
| `realdata/calibration/` | The fitted percentile mappings, hazard curves and chosen time windows. |
| `logs/` *(added by this documentation pass)* | The console output of every run. |
| `results/graphs/`, `results/metrics/` *(added by this documentation pass)* | Collected figures and metrics. |
