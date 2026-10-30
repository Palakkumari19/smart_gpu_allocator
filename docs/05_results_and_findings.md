# 05 — Results and Findings

> **This document has two parts.** **Part I (§1–§10)** is the original v1 result and is **unchanged**.
> **Part II (§11–§16)** is the v2 investigation: why v1 came out the way it did, what was fixed, and a
> re-test under a pre-registration committed before the run. v2 is reported *alongside* v1, never
> instead of it. If you only read one thing, read **§11** — it explains why the v1 experiment could
> not answer its own question.

Everything in this document was produced by running the project on **30–31 October 2026** on macOS 14
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
  than the alternatives — **was tested and was not supported**. Against schedulers that actually
  dispatch work, it reaches **parity, not superiority**.
  **(v2 correction — see §11: this v1 experiment could not have detected a difference either way,
  because 99.28 % of its scheduling decisions had no eligible GPU to choose from. The properly
  calibrated re-test in §14 rejects the hypothesis more sharply: Risk-Aware MARL is *significantly
  worse* on the pre-registered primary metric, while being the best of all six schedulers on *peak*
  GPU risk and terminal failure cost.)**
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
| **Show risk-aware MARL is safer** | ❌ **No** | v1: H1–H3 rejected, parity. v2: primary hypothesis rejected, *significantly worse* on risk per completed task (§14.1) |
| *(v2)* Show the v1 experiment was valid | ❌ **No** | The 30× load defect meant v1 tested nothing (§11) |
| *(v2)* Re-test the hypothesis properly | ✅ Yes | 6,480 episodes, pre-registered at `3a0bcc3`, run once (§14) |
| Validate on real production data | ✅ Yes | BurstGPT + UCI replayed, pre-registered |

The **engineering objective was met in full**. The **scientific hypothesis was tested and not
confirmed**. A correctly executed experiment that returns a negative result is a legitimate outcome —
and because the hypotheses were registered in advance, this project can say so credibly rather than
quietly reframing the claim.

**v2 addendum.** The v1 experiment above was *not* correctly executed: §11 shows it delivered 30× the
intended load, so almost none of its decisions were real choices. The v2 re-test (§14) is the one that
earns this paragraph. Its verdict is the same in direction and sharper in degree, and it also finds a
genuine exploratory advantage for the risk machinery on *peak* risk that v1 was structurally incapable
of detecting.

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

---

# Part II — v2: diagnosis, fixes and re-test

> **Everything above this line is the original v1 result and is unchanged.** The v2 work does not
> replace it. It asks *why* v1 came out the way it did, fixes what was genuinely broken, and re-runs
> the test under a pre-registration written before the run. Where v2 disagrees with v1, both are shown.

The v2 work was done on branch `v2-improvements` and run on **30–31 October 2026**, same machine and
Python version as v1. The pre-registration is `realdata/PREREGISTRATION_V2.md`, committed at
**`3a0bcc355d06b75618f8a958ee3eeb53fe2438d7`** *before* the final test was started; the final-test
script refuses to run if that file or the frozen config has uncommitted changes.

---

## 11. The diagnosis: v1 did not test the question it asked

The single most important v2 finding is not about scheduling at all. It is about the **experiment**.

`compute_target_load_scale()` in `realdata/trace_builder.py:77` normalises task arrivals over
`window_duration_seconds / seconds_per_step` = **3,600 steps**, but `build_real_trace()` at line 187
emits those arrivals into an episode of **120 steps**. The arrival rate is therefore multiplied by
exactly **30.0×** in every window × preset combination.

| intended tasks | delivered tasks | ratio |
|---|---|---|
| 47.8 | 1,435 | 30.0× |
| 71.7 | 2,153 | 30.0× |
| 95.7 | 2,870 | 30.0× |

The consequence, measured in `results/metrics/v2/d6_offered_load_pressure.csv`:

- Real utilisation was **11.5× to 64.5×** cluster capacity, not the 0.6–1.2 the experiment intended.
- Only **1.19 %** of offered tasks were ever admitted.
- **99.28 %** of 65,271 scheduling decisions had **zero eligible GPUs**.

When almost every decision has no choice available, every dispatching scheduler is forced into
identical behaviour. v1's "parity" was not a finding about Risk-Aware MARL. **It was the experiment
reporting that it had nothing to measure.**

This is confirmed directly in the final test (`v2_07_load_fix_validity.png`):

| intended ρ | delivered ρ, v1 | delivered ρ, v2 | contested decisions, v1 | contested decisions, v2 |
|---|---|---|---|---|
| 0.3 | 2.18 | **0.299** | 13.79 % | **32.78 %** |
| 1.1 | 7.98 | **1.101** | 0.11 % | **21.96 %** |
| 1.4 | 10.18 | **1.399** | 0.07 % | **16.15 %** |

In v1 at the near-capacity load, roughly **1 decision in 900** was a real choice. In v2 it is about
**1 in 5**. The v2 experiment can answer the question. v1's could not.

### Other diagnosis findings

| Finding | Evidence |
|---|---|
| The risk penalties **do work** — bids diverge on 37.74 % of decisions; penalty SD is 0.85× reward SD | `d1_divergence_summary.json` |
| The agents were **essentially untrained** — ε still 0.852, 49 of 57,395,628 states visited (0.00009 %) | `d5_training_adequacy.json` |
| Quarantine was **unreachable** — max risk 0.72 vs a 0.85 threshold | `d3_quarantine_summary.csv` |
| Risk buckets **collapsed** — 84 % of readings in bucket 0; buckets 2 and 3 never used | `risk_bucket_edges.json` |
| v1's "ablation reversal" (§6.3) **does not replicate** — effects are 1–6 % of one SD | `d2_ablation_peak_mean.csv` |
| `avg_risk_exposure` is a **terminal snapshot**, not a mean (0.2758 vs true mean 0.1451) — but r = 0.996, so v1's rankings still hold | `d1_bid_components.csv` |

---

## 12. What was changed, and what was deliberately not

| # | Change | Why | Kind |
|---|---|---|---|
| A | ~~Strengthen risk penalties~~ | **Rejected on evidence.** The penalties already move bids on 37.74 % of decisions. The problem was never penalty strength. | not made |
| B | Risk buckets from TRAIN data (percentiles 50/75/95 → `[0.0614, 0.1165, 0.3047]`) | v1's fixed `[0.25, 0.50, 0.75]` put 84 % of observations in one bucket, so the agent could not tell risk levels apart | scheduler input |
| C | **Load calibration fix** | The 30× defect above. This is an **evaluation fix**, not a scheduler change — it corrects how the workload is generated, identically for all nine schedulers | evaluation |
| D | Equal, larger training budget (9 × 600 steps vs v1's 1 × 80) given to **all three learners** | Fairness: Single-Agent RL and Risk-Blind MARL get exactly the same budget as Risk-Aware MARL | all learners |
| E | ~~Lower the quarantine threshold~~ | **Deferred.** 0.85 is the shipped value; changing it would be tuning the system to pass its own test. Reported at 0.85 *and* at an exploratory 0.60. | not made |
| F | Fault-injection / hazard intensity | **Explicitly not touched.** The hazard curves are calibrated to real BurstGPT data. | not made |

**The baseline schedulers' logic was not modified.** They received only the same training budget and
the same evaluation conditions.

### Three conditions, reported side by side

So that the load fix can be separated from everything else, every result is reported three ways:

- **v1** — exactly as shipped, including the 30× defect
- **v1_corrected** — v1 plus the load fix only
- **v2** — all fixes (load fix + TRAIN-derived risk buckets + equal larger training)

---

## 13. Tuning: an honest negative

20 of a permitted 20 configurations were tried; **all 20 are logged in `logs/v2_tuning_log.csv`,
including failures.** Tuning used VALIDATION windows and seeds only — never test data.

| What was varied | Effect on the validation score |
|---|---|
| ε-decay (0.997 / 0.999 / 0.9995) | **none** |
| Episode count (3 / 6 / 9 / 20) | **none** — the Q-table converges and the stopping rule fires at episode 9 |
| Steps per episode (300 → 600) | **+1.09 %** — the only setting that mattered (Q-table 484 → 787 entries) |

20 configurations produced only **4 distinct scores**. The bootstrap standard error of the validation
score is **7.61 = 4.19 % of the score**, 95 % CI **[164.99, 195.72]** — wider than the 1 % tie band
*and wider than the entire 1.09 % spread between configurations*.

**The configuration choice is within noise.** It is a deterministic tie-break (cheapest configuration
among those within 1 % of the best), not a demonstrated optimum. This is stated in the
pre-registration, not discovered afterwards.

---

## 14. The v2 final test

Run **once**, on fresh TEST seeds 100–129 never used in v1 or in tuning.
**6,480 episodes** = 3 conditions × 2 regimes × 2 windows × 3 loads × 30 seeds × 6 schedulers.

### 14.1 The PRIMARY hypothesis — the only confirmatory test

> **V2-H1 (PRIMARY):** Risk-Aware MARL achieves lower `risk_per_completed_task` than the strongest
> eligible baseline, at ρ = 1.1, workload-only, v2 condition.

| | value |
|---|---|
| Risk-Aware MARL | 0.003095 |
| Least Loaded (strongest eligible baseline) | 0.002666 |
| Paired win rate | **6 / 60 = 10.0 %** (threshold: ≥ 60 %) |
| 95 % CI on the difference | [−0.00055, −0.00032] — excludes zero |

### ❌ **REJECTED.** Not merely unsupported — Risk-Aware MARL is *significantly worse* on this metric.

Single-Agent RL scores a nominally better 0.0022, but it is **INFEASIBLE**: it completes 108.5 tasks
against the best scheduler's 151.3, below the pre-registered 80 % throughput floor. It is excluded
from baseline selection and shown hatched in the figures. This is the starvation trap from §4,
caught this time by a guard that was fixed in advance.

### 14.2 Full v2 results, ρ = 1.1, workload-only

| scheduler | completed | mean risk | **peak GPU risk** | crashes | terminal EFC | risk / task |
|---|---|---|---|---|---|---|
| FCFS | 150.7 | 0.1736 | 0.5842 | 8.23 | 38.94 | 0.0027 |
| Round Robin | 151.2 | 0.1863 | 0.5440 | **5.95** | 33.38 | 0.0030 |
| Least Loaded | 151.2 | 0.1737 | 0.5744 | 7.32 | 36.94 | 0.0027 |
| Single-Agent RL *(infeasible)* | 108.5 | 0.1095 | 0.6356 | 6.50 | 32.34 | 0.0022 |
| Risk-Blind MARL | **151.3** | 0.1879 | 0.5492 | 6.68 | 33.72 | 0.0030 |
| **Risk-Aware MARL** | 151.3 | 0.1916 | **0.5217** | 6.58 | **32.70** | 0.0031 |

### 14.3 Why it loses — and what it actually wins

Risk-Aware MARL has the **lowest peak GPU risk and the lowest terminal EFC of all six schedulers**,
at the **highest throughput**, with **20 % fewer crashes than FCFS** — and still loses the primary
test, because the primary metric is built on *mean* risk across GPUs.

The mechanism (`v2_09_mean_vs_peak.png`): the risk-aware bidder **spreads** work so that no single
GPU gets dangerously hot, which leaves all four GPUs moderately warm. FCFS and Least Loaded
**concentrate** work, leaving some GPUs cold and others overheating. Averaged across GPUs and time,
*"four warm"* scores worse than *"two cold, two burning"* — even though the second arrangement is
what actually destroys hardware.

**The pre-registered primary metric rewards the failure mode the system was built to prevent.**

This is reported as a finding, **not as grounds to change the metric**. The metric was fixed in
advance; the test stands as rejected.

### 14.4 Exploratory results — reported as exploratory

The pre-registration commits to roughly **72 secondary tests, uncorrected for multiple comparisons**,
all exploratory. At a 5 % level roughly 3–4 false positives are expected by chance, so no single
result below is confirmatory. They are listed because the pre-registration requires reporting all of
them, and because the pattern is consistent.

Risk-Aware MARL vs each eligible baseline, v2 / workload / ρ = 1.1, 60 paired episodes:

| metric | vs FCFS | vs Least Loaded | vs Round Robin | vs Risk-Blind MARL |
|---|---|---|---|---|
| **peak GPU risk** | ✅ win 0.97 | ✅ win 0.87 | ✅ win 0.78 | ✅ win 0.80 |
| **terminal EFC** | ✅ win 0.97 | ✅ win 0.77 | ✅ win 0.68 | ✅ win 0.67 |
| crashes | ✅ win 0.85 | ✅ win 0.53 | ❌ **worse** (0.13) | – win 0.32 |
| completed tasks | – win 0.47 | – win 0.33 | – win 0.43 | – win 0.35 |

Two observations worth carrying forward:

1. **The risk machinery does something measurable.** Risk-Aware MARL beats Risk-Blind MARL — its own
   ablation, same training budget, same auction, differing only in the risk and EFC bid terms — on
   peak risk (0.80) and terminal EFC (0.67). v1 could not detect this because v1 had no real choices.
2. **Round Robin beats it on crashes** (5.95 vs 6.58). Round Robin spreads load perfectly for free,
   with no risk model, no learning and no state. Most of the crash benefit is explained by
   *spreading*, not by *risk awareness*.

### 14.5 Hazard-overlay regime

With the real-data hazard overlay enabled (dispersion-calibrated, 26.25 % band on the burstiest
window), every scheduler gets worse and the ordering is unchanged (`v2_13_overlay_effect.png`).
Risk-Aware MARL still has the lowest peak risk (0.6135) among feasible schedulers, and the primary
metric still rejects.

### 14.6 All hypothesis verdicts

| ID | Hypothesis | Verdict |
|---|---|---|
| **V2-H1** | Lower risk per completed task than strongest eligible baseline | ❌ **REJECTED** (significantly worse) |
| V2-H2 | Fewer crashes per completed task | ❌ REJECTED (Round Robin better) |
| V2-H3 | On the throughput/risk Pareto frontier | ❌ REJECTED at ρ = 1.1 and 1.4 (✅ supported in `v1_corrected`/workload only) |
| V2-H4 | Quarantine reduces crashes | ⚠️ **UNTESTABLE** — the shipped 0.85 threshold fired in **0 of 3,240 episodes** in both regimes |
| V2-H4′ | Same, at exploratory threshold 0.60 | ⚠️ exploratory only; fires in 47.3 % (workload) / 68.3 % (overlay) of episodes |
| V2-H5 | Throughput within 5 % of the best dispatcher | ✅ **SUPPORTED** (151.27 vs 151.30 = 99.98 %) |

### 14.7 v1 vs v2 for Risk-Aware MARL

| | v1 (as shipped) | v1 + load fix | v2 (all fixes) |
|---|---|---|---|
| completed tasks | 302 | 128 | 128 |
| mean risk | 0.354 | 0.164 | 0.163 |
| peak GPU risk | 0.666 | 0.471 | 0.469 |
| crashes | 27.7 | 5.90 | 5.81 |

Two things to read carefully here:

- v1's higher task count is **not a better result**. It reflects ~7× more work being offered than
  intended by the broken load scale.
- **`v1_corrected` and `v2` are nearly identical.** Almost the entire change comes from the load fix.
  The risk-bucket change and the larger training budget moved the numbers by well under 1 % —
  consistent with §13, where tuning showed no detectable effect either.

---

## 15. What v2 changes about the conclusion

**It does not rescue the hypothesis. It makes the negative result trustworthy.**

v1's conclusion was *"Risk-Aware MARL reaches parity with baselines."* That conclusion was not
earned — the experiment that produced it had no real decisions in it. The correct v1 statement is:

> **v1 did not test its own hypothesis.**

v2 does test it, on an experiment that demonstrably contains real choices, with the primary
comparison fixed in advance and a single run on untouched seeds. The answer is:

> **Risk-Aware MARL does not reduce risk per completed task relative to simple baselines. On the
> pre-registered primary metric it is significantly worse. Its genuine, exploratory advantage is in
> *peak* risk and terminal failure cost rather than *mean* risk — and a large part of even that is
> explained by load spreading, which Round Robin achieves with no risk model at all.**

That is a more useful finding than the one the project set out to produce, and it is defensible
because the test could have come out the other way and was not permitted to be re-run.

---

## 16. v2 limitations and what was not done

| Item | Status |
|---|---|
| The final test was run **exactly once** | Pre-registered; no re-running or re-tuning after seeing results |
| ~72 secondary tests, **uncorrected** | All exploratory by pre-registration; ~3–4 false positives expected by chance |
| Configuration selection is **within noise** | SE 4.19 % vs a 1 % tie band; tie-break is deterministic, not optimal |
| V2-H4 (quarantine) | Structurally untestable — the shipped threshold is unreachable in this simulator |
| v2 ablation | Reported from **validation** data (`d2_ablation_peak_mean.csv`) plus the Risk-Aware vs Risk-Blind test-set comparison in §14.4. No new post-hoc run was performed, because the pre-registration forbids it |
| Hazard curves / fault injection | **Untouched** by instruction |
| `PREREGISTRATION_V2.md` committed early | It was inadvertently included in commit `eadf79c` via `git add -A` before being finalised. Disclosed in the file's own revision log rather than silently amended. The frozen content used for the run is commit `3a0bcc3`. |
