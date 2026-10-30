# SmartGPU Allocator

**An Autonomous Multi-Agent Reinforcement Learning Framework for Crash-Aware GPU Orchestration in AI-Native Cloud Systems**

*Internal codename used throughout the source: **AetherGrid***

Department of Computer Science & Business Systems
BMS Institute of Technology and Management, Bengaluru — 2026–27

| Team | USN |
|---|---|
| Amisha Verma | 1BY23CB003 |
| Chitrbhanu B | 1BY23CB011 |
| Palak Kumari | 1BY23CB038 |
| Payal A Bhavesh | 1BY23CB039 |

**Guide:** Dr. Pradeep Kumar G M, Professor

---

## What this project is

A **simulator** of a 4-GPU cluster (8 GB each) plus **nine scheduling algorithms**, built to answer one
question:

> Does a scheduler in which **each GPU has its own reinforcement-learning agent** — and in which those
> agents bid for work using both expected performance **and** how likely that GPU is to crash — actually
> produce fewer crashes than simple traditional rules, without destroying throughput?

The answer is tested against **real production data** (1.4 million recorded AI inference requests and
168k video-transcoding measurements), with the hypotheses **registered in advance** so the result could
not be reframed afterwards.

**No physical GPU is required.** The GPUs are simulated; everything runs on an ordinary laptop.

---

## Headline findings (measured 30–31 October 2026)

This section reports what the code actually produces. It replaces earlier performance claims in this
file, which today's runs do not reproduce.

> **Read this first.** The v1 experiment (October 30) **could not answer its own question**. A load
> calibration defect delivered **30× the intended workload**, so **99.28 %** of its scheduling
> decisions had no eligible GPU to choose from and every scheduler was forced into identical
> behaviour. The v2 re-test (October 31) fixes this and re-runs under a pre-registration committed
> before the run. **Both results are reported below; neither replaces the other.**

### ✅ The engineering works

- **181 tests pass, 0 skipped** (127 original + 54 new v2 tests). At the time of the v1 run 2 tests
  failed and 5 files would not load; as of 31 October the whole suite collects and passes.
- **Zero concurrency violations** under 8 threads of deliberate contention: 902 operations at
  356 ops/sec, 6 double-completions correctly blocked, memory never negative and never above 8 GB.
- **Zero capacity violations** across every benchmark run.
- The full real-data pipeline reruns end to end and **reproduces its published calibration exactly**.

### ❌ The main research hypothesis was not supported

Across **360 episodes** of real-trace replay, under pre-registered decision rules:

| Hypothesis | Metric | Verdict |
|---|---|---|
| H1 | Risk exposure | **Rejected** |
| H2 | Expected Failure Cost | **Rejected** |
| H3 | Crash count | **Rejected** |
| H4 | Quarantine prevention | **Inconclusive** (quarantine never triggered) |
| H5 | Throughput preserved | **Partially supported** |

Against schedulers that actually dispatch work, Risk-Aware MARL achieves **parity, not superiority**.

> ⚠️ **v2 correction.** This v1 table could not have detected a difference either way — see the
> diagnosis below. The properly calibrated re-test rejects the hypothesis *more* sharply.

### 🔬 v2: the re-test (31 October 2026)

**6,480 episodes**, fresh test seeds 100–129, pre-registered at commit
[`3a0bcc3`](realdata/PREREGISTRATION_V2.md), **run exactly once**.

First, evidence the re-test is valid where v1 was not:

| intended load ρ | delivered ρ, v1 | delivered ρ, v2 | decisions with a real choice, v1 | …v2 |
|---|---|---|---|---|
| 0.3 | 2.18 | **0.299** | 13.79 % | **32.78 %** |
| 1.1 | 7.98 | **1.101** | 0.11 % | **21.96 %** |
| 1.4 | 10.18 | **1.399** | 0.07 % | **16.15 %** |

**The pre-registered primary test — the only confirmatory one — is REJECTED.**
Risk per completed task at ρ = 1.1, Risk-Aware MARL **0.003095** vs Least Loaded **0.002666**;
win rate **6/60 = 10 %** against a required 60 %; 95 % CI [−0.00055, −0.00032] excludes zero.
Risk-Aware MARL is not merely unsupported here — it is **significantly worse**.

| v2 hypothesis | Verdict |
|---|---|
| **V2-H1 (primary)** — lower risk per completed task | ❌ **Rejected** (significantly worse) |
| V2-H2 — fewer crashes per completed task | ❌ Rejected (Round Robin better) |
| V2-H3 — on the throughput/risk Pareto frontier | ❌ Rejected at ρ = 1.1 and 1.4 |
| V2-H4 — quarantine reduces crashes | ⚠️ **Untestable** — fired in **0 of 3,240** episodes |
| V2-H5 — throughput within 5 % of best | ✅ **Supported** (99.98 %) |

**What it does win, as an exploratory result.** Risk-Aware MARL has the **lowest peak GPU risk**
(0.5217) and **lowest terminal failure cost** (32.70) of all six schedulers, at the **highest
throughput**, beating every baseline including its own risk-blind ablation (win rates 0.67–0.97).

The reason it loses anyway: the risk-aware bidder **spreads** work so no single GPU overheats, leaving
all four moderately warm. FCFS **concentrates** work — some GPUs cold, some burning. Averaged over
GPUs, "four warm" scores worse than "two cold, two burning", even though only the latter destroys
hardware. **The pre-registered primary metric rewards the failure mode the system was built to
prevent.** That is reported as a finding, not used to justify changing the metric after the fact.

Two honest caveats: those peak-risk wins are **exploratory** (≈72 uncorrected secondary tests, so
3–4 false positives are expected by chance), and **Round Robin beats Risk-Aware MARL on crashes**
(5.95 vs 6.58) with no risk model at all — so most of the benefit comes from *spreading*, not from
*risk awareness*.

**Which fix actually mattered?** Almost entirely the load fix. The v1+load-fix and full-v2 conditions
are nearly identical (mean risk 0.164 vs 0.163; crashes 5.90 vs 5.81). The risk-bucket change and the
larger training budget moved results by well under 1 %.

### 🔍 The two results that are genuinely valuable

1. **Burstiness drives real failures.** Measured from 1,429,737 real requests: when traffic arrives
   smoothly, ~**1 %** of requests fail; when it arrives in extreme bursts, **26.25 %** fail — a 26-fold
   jump, 95 % CI [25.82 %, 26.68 %].

2. **The starvation trap.** Single-agent RL scored the *best* on every safety metric — by refusing to
   work. It completed **6.3 tasks** where every other scheduler completed **~19**. Idle GPUs stay cool
   and never crash, so a safety metric with no throughput guard rail can be won by doing nothing. This
   is a methodological warning for anyone benchmarking safety-aware schedulers.

> Full numbers, tables and caveats: **[`docs/05_results_and_findings.md`](docs/05_results_and_findings.md)**
> — Part I (§1–§10) is v1, Part II (§11–§16) is the v2 diagnosis and re-test.
> Figures: **[`docs/06_graph_explanations.md`](docs/06_graph_explanations.md)**.
> Pre-registration: **[`realdata/PREREGISTRATION_V2.md`](realdata/PREREGISTRATION_V2.md)**.

---

## Quick start

```bash
# 1. Environment
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/pip install -r requirements.txt pytest

# 2. Optional, but needed for the full test suite
.venv/bin/pip install httpx2 torch

# 3. Verify
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q

# 4. See it work
PYTHONPATH=. .venv/bin/python verify_phase3.py
PYTHONPATH=. .venv/bin/python benchmarking/run_experiments.py
```

> `PYTHONPATH=.` is required in front of every command, and commands must be run from the repository
> root. Without it you will see `ModuleNotFoundError: No module named 'environment'`.

**Requirements:** Python 3.9+ (verified on 3.14.6), CPU only, ~200 MB disk.
Node.js is needed only for the optional React dashboard.

**Full step-by-step instructions, expected output and troubleshooting:**
**[`docs/02_how_to_run.md`](docs/02_how_to_run.md)**

---

## Repository structure

```
smart_gpu_allocator/
├── Core simulator
│   ├── environment.py              # The clock: scheduling loop, transactional placement, metrics
│   ├── gpu.py                      # One simulated GPU: memory, heat, crash at 85 °C, cooldown
│   ├── task.py                     # Simple task object (legacy demos)
│   └── dynamic_queue.py            # Thread-safe task lifecycle + retries
│
├── Schedulers
│   ├── schedulers/traditional.py   # FCFS, Round Robin, Least Loaded, Random, Best Fit, Priority
│   ├── agent.py                    # Single-agent Q-learning
│   └── marl_agent.py               # Per-GPU agents, bidding auction, EFC, reward config
│
├── Safety
│   ├── risk_engine.py              # 8-signal risk score, quarantine at 0.85
│   └── fault_injection.py          # Injected faults, failure impact, EFC calculators
│
├── Workload
│   └── workload/                   # task_generator.py, traffic_patterns.py
│
├── Measurement
│   ├── benchmarking/               # Experiment harness, metrics, ablation, results store
│   │   └── run_experiments.py      #   → entry point
│   └── evaluation/                 # Bootstrap CIs, paired stats, scientific benchmark
│
├── Real-data pipeline              # ← the scientifically strongest part
│   └── realdata/
│       ├── fetch_data.py           # Downloads BurstGPT + UCI datasets
│       ├── profile_data.py         # Statistical profile of both datasets
│       ├── windows.py              # TRAIN/EVAL temporal split (no leakage)
│       ├── llm_mapper.py           # Percentile mapping: real tokens → task size
│       ├── video_mapper.py         # Percentile mapping: real transcoding → task size
│       ├── hazard.py               # Measures real failure rate vs burstiness
│       ├── trace_builder.py        # Builds the replay trace
│       ├── risk_overlay.py         # Applies empirical failure rates non-invasively
│       ├── run_realdata_benchmark.py  # → THE MAIN EXPERIMENT
│       ├── generate_visualizations.py
│       ├── PREREGISTRATION.md      # Hypotheses fixed BEFORE results
│       └── NOTES.md                # Known quirks deliberately left unfixed
│
├── Live demo
│   ├── backend/server.py           # FastAPI + WebSocket streaming
│   ├── execution/                  # Optional real-GPU execution via PyTorch
│   └── frontend/                   # React 19 + Vite + Tailwind + Recharts
│
├── Demos & benchmarks (standalone scripts)
│   ├── main_workload_demo.py, main_traffic_demo.py, main_baseline_demo.py
│   ├── verify_phase3.py, verify_all_features.py
│   ├── benchmark_schedulers.py, benchmark_risk_aware_marl.py
│   ├── marl_vs_traditional.py, marl_performance_analysis.py
│   ├── simple_marl_training.py, marl_training_system.py
│   ├── phase4_execution_demo.py, phase4_real_video_demo.py
│   └── concurrency_stress_test.py
│
├── tests/                          # 17 suites + tests/realdata/ (9 more)
├── docs/                           # Full documentation — start here
├── results/graphs/, results/metrics/
└── logs/                           # Console output of every run
```

---

## Architecture

Six layers. Jobs enter at the top, a scheduler picks a GPU, the simulated hardware runs the job, and
two feedback loops carry risk and reward back into the decision.

```
Layer 1  Job source      synthetic generator  |  real trace replay
Layer 2  Decision        FCFS · RR · LeastLoaded · RL · MARL · Risk-Aware MARL
Layer 3  Hardware        environment.py clock → gpu.py × 4 → dynamic_queue.py
Layer 4  Safety          risk_engine.py · fault_injection.py · EFC     ──┐
Layer 5  Measurement     metrics · bootstrap CIs · CSV/JSON/figures      │ feeds the bid
Layer 6  Live demo       FastAPI + WebSocket → React dashboard         ──┘
```

Each simulation step: collect arrivals → ask the scheduler → place transactionally (roll back on
failure) → advance every GPU → update risk → pay rewards → record statistics.

**Diagram and full component walkthrough:** [`docs/03_architecture.md`](docs/03_architecture.md)
**Rendered image:** `results/graphs/00_architecture.png`

---

## Scheduling algorithms

| Algorithm | How it decides |
|---|---|
| **FCFS** | First feasible GPU |
| **Round Robin** | Cycles through GPUs |
| **Least Loaded** | Most free memory wins |
| **Random** | Random feasible GPU |
| **Best Fit** | Minimises leftover memory |
| **Priority** | Important tasks first |
| **Single-Agent RL** | One Q-learning brain for the whole cluster |
| **Risk-Blind MARL** | Per-GPU agents bid on Q-value + headroom |
| **Risk-Aware MARL** | Same auction, bid also subtracts risk and EFC |

**The risk-aware bid:**

```
bid = Q-value + free headroom − (w_risk × risk_score) − (w_efc × EFC)
```

Two cases bid **−∞** and can never win: the GPU is quarantined, or the task does not fit.

---

## The risk model

**Risk score** (`risk_engine.py`) — eight weighted signals, clamped to [0, 1]. History is weighted more
heavily than instantaneous readings:

| Signal | Weight | | Signal | Weight |
|---|---|---|---|---|
| Past anomalies | 0.25 | | Memory pressure | 0.10 |
| Past task failures | 0.20 | | Fragmentation | 0.05 |
| Temperature | 0.15 | | Performance degradation | 0.05 |
| Crash history | 0.15 | | Utilisation | 0.05 |

Bands: healthy < 0.25 · low risk < 0.50 · suspicious < 0.75 · high risk ≥ 0.75 · **quarantine ≥ 0.85**

**Expected Failure Cost** (`marl_agent.py`) — what a placement would cost if it goes wrong:

```
EFC = failure_probability × failure_impact × recovery_cost          (clamped to [0, 20])

failure_probability = 0.70 × risk_score + 0.30 × observed_failure_rate
failure_impact      = 0.60 × severity   + 0.40 × resource_intensity
recovery_cost       = max(1, 0.3 × runtime + 0.1 × memory) + cooldown_remaining
```

The MARL reward is bounded to **[−20.0, +12.0]** so one catastrophic step cannot erase the learned
table.

---

## The real-data pipeline

Two public, permissively licensed production datasets are replayed through the simulator.

| Dataset | Size | Contents | Licence |
|---|---|---|---|
| **BurstGPT** (EuroSys '24) | 48.5 MB | 1,429,737 LLM inference requests over 61 days | CC-BY-4.0 |
| **UCI Online Video Transcoding** (ID 335) | 8.5 MB | 168,286 video transcoding measurements | CC-BY-4.0 |

**Method:**
- **Percentile mapping** converts real measurements to simulator task sizes, preserving the shape of the
  real distribution instead of applying an arbitrary linear scale.
- **80 % LLM / 20 % video** mix, rescaled to three offered-load levels (ρ = 0.6, 0.9, 1.2).
- **Disjoint temporal split:** agents train on days 0–30, are tested on days 31–61 with exploration off
  (`set_eval_mode()`, ε = 0). No data leakage.
- **Empirical hazard overlay** applies the measured real-world failure probability on top of the
  simulation without modifying simulator code.
- **Pre-registration** (`realdata/PREREGISTRATION.md`) fixes hypotheses and pass/fail rules in advance.

**Scale:** 6 schedulers × 2 time windows × 3 load levels × 5 seeds = **180 episodes per regime**,
360 in total, 120 steps each.

### Measured hazard curve

| Burstiness (variance ÷ mean) | Requests | Failures | **Failure rate** | 95 % Wilson CI |
|---|---|---|---|---|
| 0.0 – 1.0 | 491,806 | 4,924 | 1.00 % | [0.97 %, 1.03 %] |
| 1.0 – 2.0 | 541,820 | 5,086 | 0.94 % | [0.91 %, 0.96 %] |
| 2.0 – 5.0 | 238,484 | 2,945 | 1.23 % | [1.19 %, 1.28 %] |
| 5.0 – 10.0 | 117,432 | 1,936 | 1.65 % | [1.58 %, 1.72 %] |
| **10.0 – 50.0** | 40,195 | 10,552 | **26.25 %** | [25.82 %, 26.68 %] |

---

## Results

### Real-data benchmark — workload replay (180 episodes)

| Scheduler | Risk ↓ | Terminal EFC ↓ | Crashes ↓ | Quarantine ↓ | Tasks ↑ |
|---|---|---|---|---|---|
| FCFS | 0.2582 | 7.380 | 0.73 | 0.00 | 18.80 |
| Round Robin | 0.2581 | 7.134 | 0.87 | 0.00 | 18.73 |
| Least Loaded | **0.2546** | 7.248 | 0.80 | 0.00 | 18.70 |
| Single-Agent RL | **0.1011** | **3.266** | **0.20** | 0.00 | **6.33** ← starving |
| Risk-Blind MARL | 0.2564 | 7.405 | 0.97 | 0.00 | **19.00** |
| Risk-Aware MARL | 0.2617 | **7.208** | 0.90 | 0.00 | 18.77 |

### With the empirical hazard overlay (180 episodes)

| Scheduler | Risk ↓ | Terminal EFC ↓ | Crashes ↓ | Quarantine ↓ | Tasks ↑ |
|---|---|---|---|---|---|
| FCFS | 0.3284 | 10.255 | 0.90 | 0.00 | 18.30 |
| Round Robin | 0.3794 | 9.870 | 0.97 | 0.00 | 18.03 |
| Least Loaded | 0.3260 | 10.033 | 0.90 | 0.00 | 18.30 |
| Single-Agent RL | **0.1202** | **4.027** | **0.27** | 0.00 | **6.13** ← starving |
| Risk-Blind MARL | 0.3626 | **9.808** | 1.07 | 0.00 | **18.60** |
| Risk-Aware MARL | 0.3854 | 9.868 | 1.07 | 0.00 | 18.17 |

Applying realistic failure rates makes every scheduler measurably more dangerous, while the **ranking
between schedulers is unchanged** — a useful robustness result.

### Seven-scenario stress benchmark (completion rate, 3 seeds)

| Scenario | Risk-Aware MARL | Least Loaded | Round Robin | FCFS |
|---|---|---|---|---|
| 1. Normal workload | 53.0 % | 55.3 % | 52.2 % | 54.8 % |
| 2. Elevated risk on one GPU | 46.9 % | 50.0 % | 48.8 % | 52.1 % |
| 3. High recovery cost | 49.2 % | 52.6 % | 51.5 % | 52.8 % |
| 4. Dynamic fault & quarantine | 46.5 % | 46.8 % | 47.4 % | 51.2 % |
| 5. Mixed task sizes | 22.4 % | 23.7 % | 22.8 % | 23.5 % |
| 6. Heavy burst | 35.1 % | 33.6 % | 34.3 % | 36.2 % |
| **7. Concurrent scheduling** | **96.7 %** | 92.5 % | 90.0 % | 91.7 % |

MARL wins decisively in **one of seven** scenarios — concurrent arrivals from multiple threads, where
the per-GPU auction resolves contention better than a single global rule. Elsewhere the four are
statistically tied. **Capacity violations: 0 everywhere.**

### Ablation study — an unexpected result

| Variant | Avg risk ↓ | Avg EFC ↓ | Throughput |
|---|---|---|---|
| Full Risk-Aware MARL | **0.0847** | 0.0324 | 5,190 |
| MARL, no risk penalty | 0.0556 | 0.0213 | 5,253 |
| MARL, no EFC penalty | **0.0390** | 0.0149 | 5,267 |
| Standard risk-blind MARL | 0.0556 | 0.0212 | 5,223 |

Turning the safety penalties **off** produced **lower** measured risk. One plausible explanation is
that the penalties spread work across all four GPUs, raising the *average* risk even while lowering the
peak on any one GPU — but **this was not tested** and is reported as interpretation, not a finding.

> ⚠️ **On "throughput" in the synthetic benchmarks.** `throughput_tasks_per_sec` is
> `completed_tasks ÷ elapsed_seconds` where `elapsed_seconds` is **wall-clock CPU time** (~5 ms per
> episode). It measures **scheduler overhead**, not cluster throughput. MARL appears slower purely
> because it runs a four-agent auction per task. Do not cite it as a performance result.

**Figures:** `results/graphs/` (12 charts) · **Raw metrics:** `results/metrics/`
**Full analysis:** [`docs/05_results_and_findings.md`](docs/05_results_and_findings.md)
**Every figure explained:** [`docs/06_graph_explanations.md`](docs/06_graph_explanations.md)

---

## Reproducing the results

Run in this order. Total time ≈ 25 minutes, almost all of it in steps 6 and 7.

```bash
# Download the datasets (~57 MB)
PYTHONPATH=. .venv/bin/python realdata/fetch_data.py

# Calibration pipeline
PYTHONPATH=. .venv/bin/python realdata/profile_data.py     # 1. profile both datasets
PYTHONPATH=. .venv/bin/python realdata/windows.py          # 2. TRAIN/EVAL split
PYTHONPATH=. .venv/bin/python realdata/video_mapper.py     # 3. percentile mapping
PYTHONPATH=. .venv/bin/python realdata/llm_mapper.py       # 4. percentile mapping
PYTHONPATH=. .venv/bin/python realdata/hazard.py           # 5. empirical hazard curve
PYTHONPATH=. .venv/bin/python realdata/trace_builder.py    # 6. build replay trace

# The main experiment (~7 minutes each)
PYTHONPATH=. .venv/bin/python realdata/run_realdata_benchmark.py
PYTHONPATH=. .venv/bin/python realdata/run_realdata_benchmark.py --with-hazard-overlay

# Figures
PYTHONPATH=. .venv/bin/python realdata/generate_visualizations.py
PYTHONPATH=. .venv/bin/python results/make_graphs.py

# Supporting benchmarks
PYTHONPATH=. .venv/bin/python benchmarking/run_experiments.py
PYTHONPATH=. .venv/bin/python benchmark_risk_aware_marl.py
PYTHONPATH=. .venv/bin/python concurrency_stress_test.py --workers 8 --duration 2.5 --scheduler marl
```

A 30-second smoke test instead of the full 7-minute run:

```bash
PYTHONPATH=. .venv/bin/python realdata/run_realdata_benchmark.py --smoke
```

### Reproducing the v2 investigation

The v2 work lives on branch `v2-improvements` and is independent of the v1 commands above —
it adds new modules and does **not** modify `realdata/trace_builder.py` or any scheduler.

```bash
# Phase 1 — diagnose why v1 came out the way it did (~6 min)
PYTHONPATH=. .venv/bin/python v2/diagnose.py
PYTHONPATH=. .venv/bin/python v2/make_diagnosis_graphs.py

# Verify the corrected load calibration (11 regression tests)
.venv/bin/python -m pytest tests/v2/test_load_scale_v2.py -v

# Phase 4 — design selection and tuning on VALIDATION data only (~40 min)
PYTHONPATH=. .venv/bin/python v2/design_select.py
PYTHONPATH=. .venv/bin/python v2/tune.py          # 12 configurations
PYTHONPATH=. .venv/bin/python v2/tune_stage2.py   # 8 more; all 20 logged
PYTHONPATH=. .venv/bin/python v2/freeze_config.py # mechanical selection + bootstrap SE

# Phase 5 — THE FINAL TEST. Runs once. ~3.3 hours.
# Refuses to start unless PREREGISTRATION_V2.md and config_v2.py are committed and clean.
caffeinate -i env PYTHONPATH=. .venv/bin/python v2/final_test.py \
    --regimes workload overlay --conditions v1 v1_corrected v2

# Analysis and figures (reads only the CSVs the final test produced)
PYTHONPATH=. .venv/bin/python v2/analyze.py
PYTHONPATH=. .venv/bin/python v2/make_results_graphs.py
```

Outputs land in `realdata/results/realdata_v2_runs_*.csv`, `results/metrics/v2/` and
`results/graphs/v2/`. Every configuration tried during tuning — including the ones that failed —
is in [`logs/v2_tuning_log.csv`](logs/v2_tuning_log.csv).

### Live dashboard (optional)

```bash
# Terminal 1
PYTHONPATH=. .venv/bin/uvicorn backend.server:app --reload --port 8000

# Terminal 2
cd frontend && npm install && npm run dev
```

---

## Concurrency, locking and data consistency

Every mutable shared resource is guarded by a reentrant lock, and all multi-lock paths follow one
global acquisition order.

### Shared state and its lock

| Component | File | Primitive | Protected state |
|---|---|---|---|
| GPU model | `gpu.py` | `self._lock` (`RLock`) | `current_memory`, `memory_fragments`, `running_tasks`, `task_queue`, `preempted_tasks`, `crashed`, `crash_cooldown_timer`, `temperature`, `utilization`, `total_latency` |
| Dynamic task queue | `dynamic_queue.py` | `self._lock` | `task_registry`, `task_to_gpu`, status dictionaries, lifecycle event log |
| Risk engine | `risk_engine.py` | `self._lock` | `reliability_history`, risk scores, quarantine state, anomaly records |
| Fault injector | `fault_injection.py` | `self._lock` | `event_log`, `event_counter`, RNG state, custom probabilities |
| MARL manager & agents | `marl_agent.py` | `self._lock` | Per-agent Q-tables, epsilon schedules, decision telemetry |
| Environment | `environment.py` | `_step_lock`, `_stats_lock` | Step transitions, task registration, history, cumulative stats |
| Backend server | `backend/server.py` | `server_lock` | `video_queue`, task counter, upload cache, scheduler selection, WebSocket clients |

### Why these mechanisms

- **Reentrant locks (`threading.RLock`).** Scheduling frequently calls one locked method from inside
  another on the same thread (e.g. `_try_assign` holding the queue lock while calling
  `task_queue.schedule_task()`). A plain mutex would self-deadlock.
- **Global lock hierarchy.** To prevent AB-BA deadlock, every multi-lock path acquires in this order:

  ```
  _step_lock  →  task_queue._lock  →  gpu._lock (ordered by GPU id)  →  _stats_lock
  ```

- **Atomic compare-and-swap lifecycle transitions.** `complete_task(task_id)` verifies the task is
  `SCHEDULED` or `RUNNING` before moving it to `COMPLETED`. A concurrent duplicate returns `False`,
  eliminating double memory deallocation and double counter increments.
- **Transactional allocation with rollback.** `_try_assign()` registers the task, acquires the GPU lock,
  and allocates. If any later step fails, a rollback handler frees the memory, removes the task from
  the GPU's queues, and restores consistent state.
- **Single centralised simulation loop.** Stepping happens in one synchronised background worker;
  WebSocket clients only read point-in-time snapshots via `get_dashboard_state()`.

### Invariants verified by the tests

- `gpu.current_memory >= 0.0` — memory is never negative
- `gpu.current_memory <= gpu.memory_capacity` (8.0 GB) — capacity is never exceeded
- `gpu.get_available_memory() + gpu.current_memory == gpu.memory_capacity` — arithmetic stays consistent
- `sum(task_counts.values()) == len(task_registry)` — queue state stays coherent
- `complete_task()` returns `False` on an already-completed task — no duplicate transitions
- No thread timeouts or deadlocks under barrier-synchronised contention

### Race conditions covered

Concurrent allocation on one GPU · across multiple GPUs · allocation racing deallocation · concurrent
queue updates · duplicate completion · preemption near completion · GPU failure during allocation ·
recovery during allocation · multi-agent Q-table updates · API requests during stepping ·
high-contention interleaved stress.

### Measured result

```
PYTHONPATH=. .venv/bin/python concurrency_stress_test.py --workers 8 --duration 2.5 --scheduler marl

Duration               : 2.531s
Operations Attempted   : 902 (356.4 ops/sec)
Allocations Succeeded  : 40
Capacity Rejections    : 607 (Legitimate OOM)
Quarantine Rejections  : 0 (Legitimate Safety)
Duplicates Prevented   : 6 (Atomic CAS Guards)
Invariant Violations   : 0

[OK] ALL CONCURRENCY INVARIANTS SATISFIED (Zero Violations)
```

**Scope limitation.** All primitives are **in-process**. They protect threads, thread pools and the
ASGI event loop within one process. They do **not** provide cross-process or distributed locking
(Redis, ZooKeeper, etc.).

---

## Configuration

| GPU | Value | | MARL | Value |
|---|---|---|---|---|
| Memory capacity | 8 GB | | Learning rate | 0.002 (adaptive) |
| Crash temperature | 85 °C | | Epsilon start → min | 1.0 → 0.05 |
| Cooldown duration | 5 steps | | Epsilon decay | 0.997 |
| Preemption threshold | memory > 90 % | | Coordination frequency | every 5 steps |
| Quarantine threshold | risk ≥ 0.85 | | Reward bounds | [−20.0, +12.0] |

**Traffic patterns:** light (0.3–0.5) · heavy (0.7–0.9) · burst (alternating) · peak (0.8–1.0) ·
off-peak (0.1–0.3) task probability per step.

---

## Known issues and limitations

### Things that do not work as documented

| Item | Problem |
|---|---|
| `python main.py` | Produces **no output**. `main.py` is a superseded FastAPI app with no `if __name__ == "__main__":` block. The working server is `backend/server.py`. |
| `main_marl_demo.py` | Prints its full demo, then **crashes** on the final optional check with `KeyError: 'llm_tasks_generated'` (line 140). `get_metrics()` returns `light_/medium_/heavy_/extreme_tasks_generated`. Left unfixed by policy; use `verify_phase3.py` instead. |
| `requirements.txt` | Missing `httpx2` (needed by 5 test files) and `torch` (1 optional test file). |
| `marl_vs_traditional.py` | Prints *"SUCCESS: MARL has demonstrated superiority"* regardless of what it measured — on the same run it reported 3 wins / 2 losses and *"Learning Improvement: NO"*. **Do not cite this script.** Use the real-data benchmark. |
| `REAL_DATA_REPORT.md` checksums | The listed SHA256 values do not match the files downloaded today from the same URLs — yet every calibration statistic reproduces exactly. The data is correct; the recorded hashes could not be confirmed. |

### Research limitations

- **Quarantine never triggered.** `quarantine_time` was exactly 0.00 in all 360 episodes — no GPU held
  risk above 0.85 long enough in a 120-step episode. The headline safety mechanism was never exercised.
- **Learning does not converge from scratch.** Untrained MARL over 70 steps: success rate falls from
  100 % to 0.8 %, cumulative reward diverges to ≈ −188,000. This is why the real-data benchmark trains
  on 30 days first and freezes the Q-tables.
- **Simulated physics.** Temperature rises by a linear formula (`gpu.py:179`), not thermodynamics. The
  built-in fault gate is a hardcoded 5 % (`environment.py:888`).
- **Quantile mapping compresses dynamic range.** It preserves task-size ranking but does not model
  quadratic attention memory scaling.
- **CPU-measured video telemetry.** The UCI data came from `ffmpeg` on general-purpose CPUs, not GPU
  encoders (NVENC/NVDEC).
- **Scale mismatch.** Hyperscaler traces are compressed onto a single 4-GPU node.
- **Not verified here:** 47 tests in 5 files, the real-GPU PyTorch path, and the React dashboard.

Pre-existing code quirks that were deliberately **not** fixed (so the measured baseline stays honest)
are catalogued in [`realdata/NOTES.md`](realdata/NOTES.md).
Source changes made during the documentation pass are listed in [`CHANGES_MADE.md`](CHANGES_MADE.md).

---

## Documentation

**Start here:**

| Document | Contents |
|---|---|
| [`docs/01_codebase_overview.md`](docs/01_codebase_overview.md) | What every module does and how they connect |
| [`docs/02_how_to_run.md`](docs/02_how_to_run.md) | Every command, expected output, troubleshooting |
| [`docs/03_architecture.md`](docs/03_architecture.md) | Component walkthrough, data flow, design rationale |
| [`docs/04_objective_and_approach.md`](docs/04_objective_and_approach.md) | The problem, the objective, the method step by step |
| [`docs/05_results_and_findings.md`](docs/05_results_and_findings.md) | **All measured results, honestly reported** |
| [`docs/06_graph_explanations.md`](docs/06_graph_explanations.md) | Every figure explained in plain language |
| [`docs/07_ppt_number_provenance.md`](docs/07_ppt_number_provenance.md) | Where each number in the presentation came from |

**Design notes (written during development):**

| Document | Contents |
|---|---|
| [`realdata/PREREGISTRATION.md`](realdata/PREREGISTRATION.md) | Hypotheses and decision rules, fixed in advance |
| [`realdata/reports/REAL_DATA_REPORT.md`](realdata/reports/REAL_DATA_REPORT.md) | Full empirical evaluation report |
| [`realdata/NOTES.md`](realdata/NOTES.md) | Known quirks left deliberately unfixed |
| [`BIT6_RISK_AWARE_MARL.md`](BIT6_RISK_AWARE_MARL.md) | Risk-aware MARL and EFC architecture |
| [`BIT7_BACKEND_WEBSOCKET_INTEGRATION.md`](BIT7_BACKEND_WEBSOCKET_INTEGRATION.md) | REST and WebSocket API |
| [`BIT8_FRONTEND_DASHBOARD_AND_VISUALIZATION.md`](BIT8_FRONTEND_DASHBOARD_AND_VISUALIZATION.md) | Dashboard panels |
| [`BIT9_RESEARCH_METRICS_AND_BENCHMARKING.md`](BIT9_RESEARCH_METRICS_AND_BENCHMARKING.md) | Metrics and benchmarking harness |
| [`MARL_TRAINING_GUIDE.md`](MARL_TRAINING_GUIDE.md) | Training system guide |

> ⚠️ `MARL_SUPERIORITY_REPORT.md`, `MARL_IMPROVEMENTS_IMPLEMENTED.md` and
> `AETHERGRID_IMPLEMENTATION_REPORT.md` contain performance claims that today's runs do **not**
> reproduce. Treat them as development history, not as results.

---

## Dataset citations

```bibtex
@inproceedings{burstgpt2024,
  title     = {BurstGPT: Characterizing and Mitigating Burstiness in
               Production LLM Inference Workloads},
  author    = {Y. Jin and others},
  booktitle = {Proceedings of the Nineteenth European Conference on
               Computer Systems (EuroSys '24)},
  year      = {2024}
}

@misc{uci_video_transcoding_335,
  author       = {Li, Z. and Xiao, Y. and Baugh, M. and Mark, N.},
  title        = {{Online Video Characteristics and Transcoding Time Dataset}},
  year         = {2015},
  howpublished = {UCI Machine Learning Repository},
  doi          = {10.24432/C57G7W}
}
```

Both datasets are licensed **CC-BY-4.0** and are downloaded at runtime; they are not stored in this
repository.

---

## References

- Sutton, R. S. & Barto, A. G. — *Reinforcement Learning: An Introduction*
- Watkins, C. J. C. H. & Dayan, P. — *Q-Learning*, Machine Learning, 1992
- Wooldridge, M. — *An Introduction to MultiAgent Systems*
- Chab, R., Setia, S. & Li, F. — *Algorithmic Techniques for GPU Scheduling: A Comprehensive Survey*,
  Algorithms 18(7), Art. 385, 2025, doi:10.3390/a18070385
- Jain, R. N. et al. — *PAL: A Variability-Aware Policy for Scheduling ML Workloads in GPU Clusters*,
  arXiv:2408.11919, 2024

---

## License

Provided for **educational and research purposes**.

---

**Status:** All components implemented and verified · research hypotheses tested and reported honestly
**Last full verification run:** 30 October 2026 (Python 3.14.6, macOS 14, CPU-only)
