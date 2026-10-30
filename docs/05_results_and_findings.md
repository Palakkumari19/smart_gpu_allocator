# 05 — Results and Findings

Everything in this document was produced by running the project on **30 October 2026** on macOS 14
(Apple Silicon, CPU only) with Python 3.14.6. Every number can be traced to a file in `logs/`,
`results/metrics/`, `benchmarking/results/` or `realdata/results/`. Nothing is estimated or carried
over from the repository's older write-ups.

---

## 1. The short version

The project **works as engineered** and **does not support its own main hypothesis**.

- Everything that was built runs: the simulator, nine scheduling algorithms, the risk engine, the
  failure-cost model, the thread-safe concurrency layer, the web dashboard and the real-data replay
  pipeline. 121 tests pass and the concurrency stress test records **zero invariant violations**.
- The headline research claim — that Risk-Aware MARL produces measurably fewer crashes and lower risk
  than the alternatives — **was tested properly and was not supported**. Against schedulers that
  actually dispatch work, it reaches **parity, not superiority**.
- The project's most valuable results turned out to be two things it did not originally set out to
  prove: a **measured real-world failure curve** (failures jump from ~1 % to **26.25 %** under bursty
  traffic), and a clear demonstration of the **"starvation trap"** — a safety metric with no throughput
  guard rewards a scheduler that simply refuses to work.

---

## 2. Did the code run?

| What was run | Result | Evidence |
|---|---|---|
| Install dependencies | Clean, no conflicts | `logs/00_env_setup.log` |
| Test suite (collectable part) | **121 passed, 2 failed** | `logs/01_pytest_collectable.log` |
| 7 demo / verification scripts | All exit 0 | `logs/run_*.log` |
| 5 benchmark scripts | All exit 0 | `logs/run_*.log` |
| Concurrency stress test | **0 invariant violations** | `logs/run_concurrency_stress_test.log` |
| Real-data pipeline (8 stages) | All exit 0 | `logs/run_realdata_*.log` |
| Full real-data benchmark | 180 + 180 episodes | `realdata/results/*_20261030_13*.csv` |

**The 2 test failures are not defects.** Both are caused by the package `httpx2` being missing — it is
required by FastAPI's test client but is not listed in `requirements.txt`. Five further test files
could not even be loaded for the same reason (plus one needing `torch`). Installing extra packages was
blocked in this environment, so **18 of 139 tests remain unverified**. This is stated plainly rather
than papered over.

**Concurrency stress test, measured:** 902 operations in 2.531 s (**356.4 ops/sec**) across 8 threads;
607 legitimate capacity rejections; **6 duplicate completions blocked** by the atomic guards;
**0 invariant violations**. Memory never went negative and never exceeded 8 GB on any GPU.

---

## 3. The main experiment: real production data

**Setup:** 6 schedulers × 2 time windows × 3 load levels × 5 seeds = **180 episodes per regime**,
120 simulation steps each. Every scheduler in a condition replays the *identical* trace.

### 3.1 Workload-only replay (180 episodes)

Source: `realdata/results/realdata_runs_workload_20261030_133854.csv`

| Scheduler | Risk exposure ↓ | Terminal EFC ↓ | Crashes ↓ | Quarantine ↓ | Tasks done ↑ |
|---|---|---|---|---|---|
| FCFS | 0.2582 | 7.380 | 0.73 | 0.00 | 18.80 |
| Round Robin | 0.2581 | 7.134 | 0.87 | 0.00 | 18.73 |
| Least Loaded | **0.2546** | 7.248 | 0.80 | 0.00 | 18.70 |
| **Single-Agent RL** | **0.1011** | **3.266** | **0.20** | 0.00 | **6.33** ← starving |
| Risk-Blind MARL | 0.2564 | 7.405 | 0.97 | 0.00 | **19.00** |
| **Risk-Aware MARL** | 0.2617 | **7.208** | 0.90 | 0.00 | 18.77 |

### 3.2 With the real failure overlay applied (180 episodes)

Source: `realdata/results/realdata_runs_overlay_20261030_134603.csv`

| Scheduler | Risk exposure ↓ | Terminal EFC ↓ | Crashes ↓ | Quarantine ↓ | Tasks done ↑ |
|---|---|---|---|---|---|
| FCFS | 0.3284 | 10.255 | 0.90 | 0.00 | 18.30 |
| Round Robin | 0.3794 | 9.870 | 0.97 | 0.00 | 18.03 |
| Least Loaded | 0.3260 | 10.033 | 0.90 | 0.00 | 18.30 |
| **Single-Agent RL** | **0.1202** | **4.027** | **0.27** | 0.00 | **6.13** ← starving |
| Risk-Blind MARL | 0.3626 | **9.808** | 1.07 | 0.00 | **18.60** |
| **Risk-Aware MARL** | 0.3854 | 9.868 | 1.07 | 0.00 | 18.17 |

Applying the real-world failure rate makes the cluster measurably more dangerous for every scheduler
(risk rises by roughly 0.07–0.12, crashes rise), while the **ranking between schedulers barely
changes**. That is a useful robustness result.

### 3.3 The pre-registered hypothesis tests

The rule, fixed in advance in `realdata/PREREGISTRATION.md`: a hypothesis counts as supported only if
the bootstrap 95 % confidence interval excludes zero in the favourable direction **and** the win rate
against the strongest baseline is at least 60 %.

| Hypothesis | Metric | Strongest baseline | Baseline | Risk-Aware MARL | 95 % CI | Win rate | **Verdict** |
|---|---|---|---|---|---|---|---|
| H1 | Risk exposure | `rl` | 0.1011 | 0.2617 | [−0.196, −0.127] | 0.0 % | **Rejected** |
| H2 | Terminal EFC | `rl` | 3.266 | 7.208 | [−5.72, −2.40] | 23.3 % | **Rejected** |
| H3 | Crash count | `rl` | 0.200 | 0.900 | [−1.10, −0.33] | 3.3 % | **Rejected** |
| H4 | Quarantine time | `fcfs` | 0.000 | 0.000 | [0, 0] | 0.0 % (30 ties) | **Inconclusive** |
| H5 | Tasks completed | `marl` | 19.00 | 18.77 | [−0.57, +0.13] | 13.3 % | **Tie / partial** |

(Overlay regime gives the same verdicts; `completed_tasks` there is marginally worse,
CI [−0.867, −0.067].)

**Conclusion against the pre-registered rule: H1–H3 rejected, H4 inconclusive, H5 partially supported.**

---

## 4. Why the hypotheses failed — the starvation trap

This is the most important finding in the project, and it is a *methodological* one.

The pre-registration says "compare against the strongest baseline **on each metric independently**".
On risk, EFC and crashes, the strongest baseline turned out to be Single-Agent RL — which scored so
well **because it stopped doing the job**. It completed **6.33 tasks** where everything else completed
about **19**.

The mechanism is visible in the code. In `gpu.py:179`, temperature rises with queue length:

```python
temp_increase = self._randint(1, 4) + len(self.task_queue) // 2
```

A GPU running almost nothing stays cool, never approaches the 85 °C crash threshold, and accumulates
no failure history — so its risk score stays low. **An algorithm that admits zero tasks scores a
perfect zero on every safety metric.**

**The lesson:** a safety metric without a throughput guard rail can be won by doing nothing. Any future
benchmark of safety-aware scheduling needs a combined objective, not independent per-metric baselines.

Restricting the comparison to the five schedulers that actually dispatch work gives a much fairer
picture, and the answer there is **statistical parity**: risk exposure spans 0.2546–0.2617 and crashes
span 0.73–0.97, differences that are small next to a standard deviation of roughly ±0.12 and ±1.1.

---

## 5. The genuinely strong result: burstiness drives real failures

Measured from the full BurstGPT dataset — **1,429,737 real requests, 25,443 observed failures**
(global rate 1.78 %). Reproduced exactly today. Source: `logs/run_realdata_hazard.log`.

| Burstiness (variance ÷ mean) | Requests | Failures | **Failure rate** | 95 % Wilson CI |
|---|---|---|---|---|
| 0.0 – 1.0 (very smooth) | 491,806 | 4,924 | 1.00 % | [0.97 %, 1.03 %] |
| 1.0 – 2.0 (normal) | 541,820 | 5,086 | 0.94 % | [0.91 %, 0.96 %] |
| 2.0 – 5.0 (mildly bursty) | 238,484 | 2,945 | 1.23 % | [1.19 %, 1.28 %] |
| 5.0 – 10.0 (heavily bursty) | 117,432 | 1,936 | 1.65 % | [1.58 %, 1.72 %] |
| **10.0 – 50.0 (extreme)** | 40,195 | 10,552 | **26.25 %** | [25.82 %, 26.68 %] |

A **26-fold jump**. The confidence intervals are extremely tight because the sample sizes are large.
This stands on its own as an empirical contribution, independent of whether the scheduler works.

---

## 6. Supporting experiments

### 6.1 Seven stress scenarios (`benchmark_risk_aware_marl.py`, 3 seeds each)

Completion rate, Risk-Aware MARL vs the three heuristics:

| Scenario | MARL | Least Loaded | Round Robin | FCFS | Verdict |
|---|---|---|---|---|---|
| 1. Normal workload | 53.0 % | 55.3 % | 52.2 % | 54.8 % | tie |
| 2. Elevated risk on one GPU | 46.9 % | 50.0 % | 48.8 % | 52.1 % | tie |
| 3. High recovery cost | 49.2 % | 52.6 % | 51.5 % | 52.8 % | tie |
| 4. Dynamic fault & quarantine | 46.5 % | 46.8 % | 47.4 % | 51.2 % | tie |
| 5. Mixed task sizes | 22.4 % | 23.7 % | 22.8 % | 23.5 % | tie |
| 6. Heavy burst | 35.1 % | 33.6 % | 34.3 % | 36.2 % | tie |
| **7. Concurrent scheduling** | **96.7 %** | 92.5 % | 90.0 % | 91.7 % | **MARL wins** |

**Capacity violations across all scenarios: 0.** MARL wins decisively in exactly one of seven
scenarios — concurrent arrivals from multiple threads, where the per-GPU bidding auction resolves
contention better than a single global rule. This is a real but narrow advantage.

### 6.2 Six-scheduler synthetic benchmark (`benchmarking/run_experiments.py`, 3 seeds)

| Scheduler | Success rate | Avg risk ↓ | Avg EFC ↓ |
|---|---|---|---|
| FCFS | 100.0 % | 0.0663 | 0.0253 |
| Round Robin | 100.0 % | 0.0553 | 0.0212 |
| Least Loaded | 100.0 % | 0.0594 | 0.0227 |
| Single-Agent RL | **81.9 %** | 0.0280 | 0.0107 |
| Risk-Blind MARL | 100.0 % | 0.0471 | 0.0180 |
| Risk-Aware MARL | 100.0 % | 0.0721 | 0.0276 |

> **Important caveat about the "throughput" column in this benchmark.** The CSV reports
> `throughput_tasks_per_sec = completed_tasks ÷ elapsed_seconds`, where `elapsed_seconds` is
> **wall-clock CPU time** (about 0.005 s per episode). It therefore measures **how expensive the
> scheduler's own decision-making is**, not how much work the cluster did. MARL appears "slower"
> (≈5,200 vs ≈8,900) purely because it runs a four-agent auction for every task. On real hardware,
> where a task takes seconds, that overhead would be negligible. **Do not present this as cluster
> throughput.**

### 6.3 Ablation study — a surprise

| Variant | Avg risk ↓ | Avg EFC ↓ | Throughput |
|---|---|---|---|
| Full Risk-Aware MARL | **0.0847** | 0.0324 | 5,190 |
| MARL, no risk penalty | 0.0556 | 0.0213 | 5,253 |
| MARL, no EFC penalty | **0.0390** | 0.0149 | 5,267 |
| Standard risk-blind MARL | 0.0556 | 0.0212 | 5,223 |

**This is the opposite of what was expected.** Turning the safety penalties *off* produced *lower*
measured risk. The full model recorded the highest risk exposure of the four variants.

A plausible explanation is that the risk and EFC penalties push work away from whichever GPU is
currently loaded and spread it across the cluster, raising the *average* risk across four GPUs even
while reducing the peak on any one. But this is **interpretation, not measurement** — the experiment
to confirm it (comparing peak vs mean per-GPU risk) was not run. It is reported as-is.

### 6.4 MARL learning from scratch — does not converge

From `logs/run_marl_vs_traditional.log`:

| Step | Success rate | Cumulative reward | Q-table size |
|---|---|---|---|
| 0 | 100.00 % | +83.6 | 20 |
| 20 | 10.71 % | −10,137.8 | 166 |
| 40 | 2.68 % | −53,353.2 | 178 |
| 70 | **0.80 %** | **−188,322.2** | 182 |

The script itself prints **"MARL Learning Improvement: NO"**. Seventy steps is far too short to learn a
four-way placement problem, and the Q-table stops growing at ~182 states while performance keeps
falling. This is exactly why the serious experiment trains on 30 days of real traffic first and
**freezes the Q-tables** before testing.

---

## 7. What worked well

1. **The engineering.** Every component runs. Concurrency is genuinely safe: 0 violations under
   8 threads of deliberate contention, 6 double-completions correctly blocked.
2. **Scientific method.** Pre-registration, paired identical task streams, bootstrap confidence
   intervals, disjoint train/test windows, and a documented "do not fix" list of known quirks. This is
   unusually rigorous for a student project and is why the negative result is trustworthy.
3. **The hazard measurement.** The 1 % → 26.25 % burstiness curve is a solid, reusable empirical
   finding drawn from 1.4 million real requests.
4. **Reproducibility.** The published pipeline reran end to end and reproduced the previous
   calibration figures exactly, and the benchmark means to within ~0.002.
5. **One real algorithmic win.** Concurrent arrivals: 96.7 % vs 90.0–92.5 %, with a tighter spread.

---

## 8. What did not work, and surprises

1. **Risk-Aware MARL does not beat the baselines.** H1–H3 rejected. Against working schedulers it is a
   tie; against the starving RL scheduler it is significantly worse on paper.
2. **The quarantine mechanism never fired.** Quarantine time was **exactly 0.00 in all 360 episodes**.
   No GPU held a risk score above 0.85 long enough during a 120-step episode. The project's headline
   safety feature was never actually exercised, which is why H4 is inconclusive rather than answered.
3. **The ablation ran backwards** (§6.3) — removing safety penalties lowered measured risk.
4. **Learning does not converge from scratch** (§6.4).
5. **The repository's own documentation overstates the results.** The README claims "36 % improvement
   in learning performance" and "MARL consistently outperforms traditional methods". Today's runs do
   not reproduce those claims. Worse, `marl_vs_traditional.py` prints
   *"SUCCESS: MARL has demonstrated superiority"* on the same run where it measured a 60 % win rate
   (3 wins / 2 losses) and printed *"Learning Improvement: NO"* — the concluding text is hardcoded and
   does not depend on the measurements above it. **Use the real-data benchmark, not these scripts, for
   any claim about performance.**

---

## 9. Did the project achieve its objective?

**Partly — and the honest split matters.**

| Objective | Achieved? | Evidence |
|---|---|---|
| Build a GPU scheduling simulator with multiple algorithms | ✅ Yes | 9 algorithms, all run |
| Implement RL and coordinated MARL | ✅ Yes | `agent.py`, `marl_agent.py`, auction verified |
| Integrate risk-aware scheduling | ✅ Yes | Risk engine + EFC wired into bidding |
| Concurrency control / no race conditions | ✅ Yes | 0 violations under stress |
| Evaluate against benchmarks | ✅ Yes | 360 real-data episodes + 3 benchmark suites |
| **Show risk-aware MARL is safer** | ❌ **No** | H1–H3 rejected; parity with baselines |
| Validate on real production data | ✅ Yes | BurstGPT + UCI replayed, pre-registered |

The **engineering objective was met in full**. The **scientific hypothesis was tested properly and not
confirmed**. A correctly executed experiment that returns a negative result is a legitimate outcome —
and because the hypotheses were registered in advance, this project can say so credibly rather than
quietly reframing the claim.

---

## 10. Honest notes on what could not be run or verified

| Item | Status | Reason |
|---|---|---|
| 5 test files + 1 optional test (18 tests) | **Not run** | `httpx2` and `torch` unavailable; installing extra packages was blocked in this environment |
| Real GPU execution (`execution/`, PyTorch) | **Not run** | Requires `torch` and a physical NVIDIA GPU; this machine is Apple Silicon |
| React dashboard | **Not launched** | Requires `npm install`; the FastAPI backend it depends on was verified only through its Python modules |
| `python main.py` | **Produces no output** | Superseded FastAPI app with no entry point; README mislabels it as the basic demo |
| `main_marl_demo.py` final section | **Crashes** | Pre-existing `KeyError: 'llm_tasks_generated'`; left unfixed by policy |
| `quarantine_time` as a discriminating metric | **Uninformative** | Exactly 0.00 in all 360 episodes |
| Dataset checksums | **Do not match** | `REAL_DATA_REPORT.md` lists SHA256 values that differ from the files downloaded today from the same URLs — yet every calibration figure reproduced exactly (1,429,737 requests; 25,443 failures; 26.25 % peak). The data is right; **the recorded hashes appear not to have been computed from these files.** |
| Ablation explanation (§6.3) | **Unverified** | The proposed mechanism is interpretation; the confirming experiment was not run |
