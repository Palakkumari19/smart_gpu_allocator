# PRE-REGISTRATION v2

**Written and committed BEFORE the single final test run.**
**Date:** 30 October 2026 · **Branch:** `v2-improvements`

This document supersedes nothing. `realdata/PREREGISTRATION.md` (v1) and all v1 results remain
untouched and are reported alongside v2 throughout.

---

## 0. Why there is a v2

v1 reported that Risk-Aware MARL failed to beat its baselines (H1–H3 rejected, H4 inconclusive,
H5 partial). The Phase 1 diagnosis (`docs/07_diagnosis.md`) found that **v1's experiment could not
discriminate between any schedulers**, for a measurable reason:

| Evidence | Measurement | Source |
|---|---|---|
| Offered load was **exactly 30.0×** the intended target in all 6 window×preset cells | `trace_builder.py:77` normalises over the window's 3600 steps; `:187` emits into the episode's 120 | `d6_offered_load_pressure.csv` |
| **99.28 %** of 65,271 scheduling decisions had **zero** eligible GPUs | only 0.39 % offered a real choice | `d1_divergence_summary.json` |
| The risk penalty **does** work: it changed the pick **37.74 %** of the time when consulted | penalty SD ÷ reward SD = 0.85 | `d1_bid_components.csv` |
| Agents were essentially **untrained**: ε ended at **0.852**, 49 of 57,395,628 states (0.00009 %) | shipped decay needs thousands of steps; it got 80 | `d5_training_adequacy.json` |
| Quarantine threshold **0.85 unreachable**: max risk 0.72 even at 1000 steps and ρ=2.0 | 0 of 18 episodes triggered | `d3_quarantine_reachability.csv` |
| Risk state dimension collapses: **84 %** of observations in bucket 0, buckets 2–3 never used | edges 0.25/0.50/0.75 vs observed max 0.489 | `d1_bid_components.csv` |
| v1's "ablation reversal" **does not replicate** — effects are 1–6 % of one SD | it was a 3-seed artefact | `d2_ablation_peak_mean.csv` |

**Therefore the honest reading of v1 is not "risk-aware MARL fails" but "the v1 experiment did not
test the question."** v2 re-tests it under conditions where the scheduler can actually make choices.

---

## 1. What changed since v1, and the evidence for each change

| ID | Change | Evidence | Status |
|---|---|---|---|
| **F1** | **Offered-load calibration fixed.** Scale computed against the episode's step count; mean task footprint measured empirically. New module `v2/load_model.py`; `trace_builder.py` untouched so v1 stays reproducible. | 30.0× over-delivery; 99.28 % of decisions had no choice | **Implemented** |
| **F2** | **Episode length 600 steps** (was 120). | risk accumulates with episode length; chosen by rule on VALIDATION | **Implemented** |
| **F3** | **Load levels re-derived** to span light / near-capacity / overloaded. | v1's ρ presets were indistinguishable (all ≥30× saturated) | **Implemented** |
| **C1** | **Working ε decay** and a real training budget; **identical for all three learners**. | ε ended at 0.852 after v1 training | **Implemented** |
| **C2** | **Each learner trained under its own bid rule.** | v1 trained one table with the risk-aware bid and gave it to risk-blind MARL too | **Implemented** |
| **D1** | **Risk-bucket edges re-derived from the TRAIN risk distribution.** | 84 % of observations fell in one bucket; 2 of 4 buckets unused | **Implemented** |
| **B1** | **Throughput floor** in the decision rule, plus combined objectives. | single-agent RL won v1's safety metrics by completing 6.0 vs 19.0 tasks | **Implemented** |
| **A** | ~~Rescale risk/EFC penalty magnitudes~~ | **NOT DONE.** Penalties already flip 37.74 % of contested decisions; penalty SD ÷ reward SD = 0.85. Rescaling a working component would be unjustified tuning. | **Rejected on evidence** |
| **E** | ~~Hazard curve as a decision-time input~~ | **DEFERRED.** Speculative until the scheduler actually has choices to make. | **Deferred** |

**Not touched:** every scheduler's logic (FCFS, Round Robin, Least Loaded, Single-Agent RL,
Risk-Blind MARL, Risk-Aware MARL), the reward function, the risk engine, the EFC model, the GPU
physics, **the fault injector, and the empirically calibrated hazard curves**. Fault-injection and
hazard intensity are **explicitly unchanged** — the hazard curves are calibrated to real BurstGPT
data and must stay as measured. Changes C1/C2 alter the **training budget and protocol**, applied
**identically to all three learners**.

---

## 2. Data splits — strictly disjoint, TEST used exactly once

| Split | Partition | Windows | Seeds | Used for |
|---|---|---|---|---|
| **TRAIN** | days 0–30 | `busiest_hour` (33,344 requests) | 1, 2, 3 | agent learning only |
| **VALIDATION** | days 0–30 | `median_hour` (193), `burstiest_window` (567) | 200, 201, 202 | **all** tuning and design decisions |
| **TEST** | days 31–61 | `median_hour` (263), `burstiest_window` (16,902) | **100–129** (30 fresh seeds) | the single final run |

TEST windows are the same two v1 evaluated on, so v1 and v2 are directly comparable. TEST seeds are
fresh (v1 used 42–46). **No TEST data was inspected while making any decision in this document.**

---

## 3. The exact frozen configuration

Frozen in `realdata/config_v2.py` and `results/metrics/v2/frozen_training.json`. Selected on
VALIDATION using **baseline schedulers only** (FCFS, Least Loaded) — no MARL result influenced the
design decisions below.

| Setting | Value | Selection rule |
|---|---|---|
| Episode length | **600 steps** | shortest candidate of {120, 300, 600} whose max observed risk is within 0.02 of the longest. Measured: 0.347 / 0.497 / 0.553 → 600 |
| Load levels | **ρ ∈ {0.3, 1.1, 1.4}** | *light* = largest ρ with contested ≥ 25 % (0.3, 25.3 %); *near-capacity* = ρ whose **delivered** ρ is closest to 1.0 (1.1, delivered 1.102); *overloaded* = smallest ρ with contested ≤ 10 % (1.4, 9.2 %) |
| Risk-bucket edges | **[0.0614, 0.1165, 0.3047]** | 50/75/95th percentiles of the per-step per-GPU risk distribution on **TRAIN** windows under a neutral dispatcher. Non-positive/duplicate edges dropped (>25 % of readings are exactly 0.0). v1 edges were [0.25, 0.50, 0.75]. |
| Shipped quarantine threshold | **0.85** | unchanged; always reported |
| Exploratory quarantine threshold | **0.60** | largest value on the fixed ladder {0.85…0.30} triggering in ≥ 30 % of VALIDATION episodes under FCFS/Least Loaded only. Measured: 0.85→0 %, 0.70→0 %, **0.60→58 %**, 0.50→100 % |
| Throughput floor (results) | **0.80** | a scheduler completing < 80 % of the best scheduler's mean completed tasks is **INFEASIBLE** |
| **Training (all 3 learners)** | **9 episodes × 600 steps**, ρ=0.9, ε-decay **0.999**, ε-floor 0.05, lr 0.01 | tuning, §8 |
| **Training stopping rule** | stop at 20 episodes, or early if Q-table growth < **1 state** for **6** consecutive episodes (fires at episode 9) | **revised after stage-1 evidence — see §11** |
| Hazard overlay | both regimes run: workload-only and overlay | unchanged, calibrated curves untouched |

**Equal budget is enforced in code.** `v2/improvements.train_all_learners()` applies one
`TrainingConfig` object to `rl`, `marl` and `risk_aware_marl` in turn. Each is trained under **its
own bid rule** via its own scheduler adapter.

---

## 4. Experimental conditions — reported side by side

All three run on the **same TEST seeds**, same windows, same load levels, same episode length, so
the contribution of each fix is separable.

| Condition | Load scale | Training | Risk buckets |
|---|---|---|---|
| **v1** | legacy (30× defect) | v1 (80 steps, one window, shipped decay) | v1 [0.25, 0.50, 0.75] |
| **v1_corrected** | **fixed** | v1 | v1 |
| **v2** | **fixed** | **v2 (9×600)** | **v2 (TRAIN-derived)** |

`v1` vs `v1_corrected` isolates the load fix. `v1_corrected` vs `v2` isolates training + state.

---

## 5. Metrics

**Primary (lower is better):** `avg_risk_exposure`, `terminal_efc`, `crash_count`, `quarantine_time`
— identical definitions to v1, so v1↔v2 is like-for-like.

**Secondary (higher is better):** `completed_tasks`.

**Combined objectives (lower is better)** — cannot be gamed by starving:
- `risk_per_completed_task` = mean risk over time ÷ completed tasks
- `crashes_per_completed_task` = crashes ÷ completed tasks

**Diagnostic (reported, not hypothesis-tested):** `mean_risk_over_time` (a true time-average, unlike
v1's terminal snapshot), `peak_gpu_risk`, `contested_pct`, `admitted_pct`.

> **Known metric defect, carried forward deliberately.** `avg_risk_exposure` maps to
> `average_gpu_risk`, a terminal snapshot rather than a time average. Measured bias: 0.2758 vs a true
> mean of 0.1451 (~1.9× inflated), but correlation r = 0.996, so rankings are unaffected. Kept
> unchanged for v1 comparability; `mean_risk_over_time` is reported alongside it.

---

## 6. Hypotheses

"Eligible baselines" = schedulers passing the 0.80 throughput floor (§3).

### 6.1 PRIMARY comparison — the single confirmatory test

> **PRIMARY:** Risk-Aware MARL has **lower `risk_per_completed_task`** than the strongest eligible
> baseline, at the **near-capacity load level ρ = 1.1**, in the **workload-only regime**, in the
> **v2 condition**.

This is **the** result. If it fails, **the outcome is reported as "not supported"**, regardless of
what any other test shows. One test, named in advance, so it cannot be swapped for a luckier one.

### 6.2 Secondary and exploratory hypotheses

| ID | Hypothesis | Class |
|---|---|---|
| **V2-H1** | Lower `risk_per_completed_task` than every eligible baseline (all loads, both regimes) | Secondary |
| **V2-H2** | Lower `crashes_per_completed_task` than every eligible baseline | Secondary |
| **V2-H3** | On the **Pareto frontier** of (completed tasks ↑, risk exposure ↓), not dominated by any baseline. A dominates B iff A completes ≥ B's tasks **and** has ≤ B's risk, with at least one strict | Secondary |
| **V2-H4** | **Quarantine.** At the shipped **0.85** threshold: reported as **UNTESTABLE** (max risk observed on validation was 0.644; 0 % of episodes trigger). At the exploratory **0.60** threshold: triggers in ≥ 30 % of episodes **and** triggering episodes show fewer crashes | **Exploratory** |
| **V2-H5** | `completed_tasks` within **5 %** of the best dispatching scheduler | Secondary |

**V2-H4 is exploratory and is not the shipped mechanism.** The 0.60 threshold is an observational
relabelling for analysis only: it does not change scheduler behaviour, and **no fault-injection or
hazard-intensity parameter is altered**.

### 6.3 Multiple comparisons

The secondary tests span 3 conditions × 2 regimes × 3 load levels × 4 hypotheses ≈ **72 tests**.

**They are reported UNCORRECTED and are EXPLORATORY.** No correction is applied and none is claimed.
Only the §6.1 primary comparison carries confirmatory weight. Readers should treat any isolated
secondary "SUPPORTED" at this multiplicity as hypothesis-generating, not as evidence.

---

## 7. Decision rule — unchanged from v1

A hypothesis is **SUPPORTED** only if **both** hold against the strongest **eligible** baseline:

1. the **bootstrap 95 % CI** (2,000 resamples) of the paired difference **excludes zero** in the
   favourable direction, **and**
2. the **paired win rate ≥ 60 %**.

Otherwise **REJECTED**, or **INCONCLUSIVE** where the metric cannot discriminate (e.g. all values
identical, as quarantine was in v1).

**Throughput floor.** A scheduler whose mean `completed_tasks` is below **80 %** of the best
scheduler's is labelled **INFEASIBLE** and excluded from "strongest baseline" selection — preventing
a starving scheduler from becoming the benchmark, which is what happened in v1.

**Single-agent RL is reported BOTH ways** — included and excluded by the floor — so the starvation
effect is visible rather than hidden by the rule.

---

## 8. Tuning: budget, rules, and the honest result

- **Hard cap: 20 configurations. Exactly 20 were run**, all logged to `logs/v2_tuning_log.csv`
  (including those that changed nothing).
  - Stage 1 (12): ε-decay {0.999, 0.995, 0.990} × episodes {10, 20} × edges {v1, TRAIN-derived}
  - Stage 2 (8): stopping rule {g<1 ×6, none} × train steps/episode {300, 600} × edges {v1, TRAIN-derived}
- Tuning ran on **VALIDATION only**.
- **Selection objective, fixed in advance:**
  `validation_score = mean(completed_tasks) / mean(mean_risk_over_time)` for the treatment
  scheduler — work done per unit of risk carried. Ties broken by lower `crashes_per_completed_task`.
- **Config-selection throughput floor: 0.95.** A configuration completing < 95 % of the best
  configuration's mean `completed_tasks` is ineligible. *The score is a ratio and is **not**
  structurally immune to starving*, so this floor is required rather than optional. (Measured: no
  configuration was excluded; all were within 0.3 % of the best.)
- **Tie band: 1 %, applied mechanically and not adjusted.** Configurations within 1 % of the best
  score are ties; among ties the **simplest/cheapest** is chosen (fewest total training steps, then
  fewest episodes, then lowest wall-clock).

### 8.1 The selection is within noise — stated plainly

| Quantity | Value |
|---|---|
| Best score | 182.8544 (cfg15 / cfg19) |
| Worst score | 180.8787 |
| Total spread across all 20 configs | **1.09 %** |
| Tie band lower bound | 181.0259 |
| Ties | cfg15, cfg16, cfg19, cfg20 |
| Chosen mechanically | **cfg15** (fewest training steps among ties, 9 × 600) |
| **Bootstrap SE of validation_score** | **7.61 = 4.19 % of the score** |
| 95 % CI of validation_score | [164.99, 195.72] |

**The standard error (4.19 %) is more than four times the tie band (1 %), and wider than the entire
spread between the best and worst configuration (1.09 %).** Therefore **the tuning did not
discriminate between configurations.** The 1 % tie-break rule is retained only as a *deterministic
way to choose*, not as evidence that the chosen configuration is better. This is stated here so it
cannot be presented later as a tuning success.

### 8.2 Risk-bucket edges: chosen on principle, against the score

Mechanical selection picked **cfg15**, which uses the **v1** edges. The **TRAIN-derived edges are
used anyway**, on the pre-stated principle that 84 % of risk readings fell into a single v1 bucket
and two of the four buckets were never used — a state representation that cannot express the
distinction the project is about.

**The validation scores did not favour this choice**: at 600-step training, v1 edges scored
**182.85** and TRAIN-derived scored **181.74**. Both lie inside the 1 % tie band, and both lie well
inside the 4.19 % standard error. The effective frozen configuration therefore equals **cfg16**.

### 8.3 What the tuning actually showed

- ε-decay rate (0.999 / 0.995 / 0.990): **no effect whatsoever** — ε reaches its 0.05 floor under
  all three.
- Training episode count (6 / 9 / 20): **no effect** — Q-tables converge; cfg17/18 (20 episodes)
  reproduce cfg01/02 (6 episodes) exactly.
- Training **steps per episode** (300 → 600): the only lever with any effect. Q-table 484 → 787
  states, score 180.88 → 182.85 (**+1.09 %**, inside the noise).

**Honest summary: improvement C (better training) produced no measurable validation gain, and
improvement D (risk buckets) produced none either.** If v2 beats v1, the three-condition design
(§4) will show the credit belongs to the load fix (F1).

Q-tables are **snapshotted and frozen** (`set_eval_mode()`, ε = 0) before any evaluation, as in v1.

---

## 9. Commitments

1. **The final test runs once.** No re-running or re-tuning after seeing TEST results. If a run
   fails for a technical reason it is fixed and re-run **with the same configuration**, and that is
   reported.
2. **All three conditions, both regimes and all three load levels are reported in full**, regardless
   of outcome.
3. **If the §6.1 primary comparison fails, the result is reported as "not supported"** — plainly,
   and as the headline, exactly as v1's negative result was.
4. **A negative result will be reported as a negative result.**
5. **v1 results stay visible** and are never replaced.
6. **20 of 20 tuning configurations are reported**, including the 18 that changed nothing.
7. The final-test script records the commit hash of this file in every output, and **refuses to run**
   if this file, `realdata/config_v2.py` or `results/metrics/v2/frozen_training.json` has
   uncommitted changes.

---

## 10. What would falsify the v2 premise

If, after the load fix, Risk-Aware MARL remains statistically tied with Least Loaded and Risk-Blind
MARL on the combined objectives across all three load levels, the conclusion is that **risk-aware
bidding offers no measurable benefit in this simulator** — the diagnosis will have corrected the
*method* without rescuing the *hypothesis*. That is an acceptable outcome, and this document is
designed to be able to report it.

---

## 11. Revision log

Full transparency about what changed in this document and when.

| # | Revision | Reason | Evidence |
|---|---|---|---|
| 1 | **Committed early and inadvertently** in `eadf79c` ("Phase 5 scaffolding") via `git add -A`, before tuning had completed. | Disclosure: this amendment is therefore a tracked revision, not a silent edit. The operative requirement — committed and clean **before the final test** — is met. | `git log eadf79c` |
| 2 | **Stopping-rule clause revised** from "stop early if Q-growth < 5 states for 3 consecutive episodes" to "< 1 state for 6 consecutive episodes". | The original rule **fired at episode 6 in all 12 stage-1 configurations**, so the training budget never bound and the stage-1 grid could not test whether more training helps. ε also reached its 0.05 floor regardless of decay rate, making that axis inert. | `logs/v2_tuning_log.csv` cfg01–cfg12; `results/metrics/v2/tuning_summary.json` |
| 3 | **Added** §6.1 primary comparison, §6.3 multiple-comparison statement, §8 config-selection throughput floor (0.95), §8.1 standard-error statement, §8.2 bucket-edge rationale, §9.7 guard requirement. | Requested before the final run, to prevent outcome-shopping across ~72 secondary tests. | this document |

**All revisions were made BEFORE the final test run and before any TEST data was generated or
inspected.**

---

## 12. The exact final command

```bash
caffeinate -i env PYTHONPATH=. .venv/bin/python -u v2/final_test.py \
  --regimes workload overlay \
  --conditions v1 v1_corrected v2 \
  2>&1 | tee logs/v2/final_test.log
```

Measured cost: **2.80 hours** (v1 condition 2.47 h, v1_corrected 0.17 h, v2 0.17 h; both regimes),
plus ~2 minutes of training.
