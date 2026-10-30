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
| **F1** | **Offered-load calibration fixed.** Scale computed against the episode's step count; mean task footprint measured empirically, not assumed. New module `v2/load_model.py`; `trace_builder.py` untouched so v1 stays reproducible. | 30.0× over-delivery; 99.28 % of decisions had no choice | **Implemented** |
| **F2** | **Episode length 600 steps** (was 120). | Risk accumulates with episode length; chosen by rule on VALIDATION | **Implemented** |
| **F3** | **Load levels re-derived** to span light / near-capacity / overloaded. | v1's ρ presets were indistinguishable (all ≥30× saturated) | **Implemented** |
| **C1** | **Working ε decay** and a real training budget; **identical for all three learners**. | ε ended at 0.852 after v1 training | **Implemented** |
| **C2** | **Each learner trained under its own bid rule.** | v1 trained one table with the risk-aware bid and gave it to risk-blind MARL too | **Implemented** |
| **D1** | **Risk-bucket edges re-derived from the TRAIN risk distribution.** | 84 % of observations fell in one bucket; 2 of 4 buckets unused | **Implemented** |
| **B1** | **Throughput floor** in the decision rule, plus combined objectives. | Single-agent RL won v1's safety metrics by completing 6.0 vs 19.0 tasks | **Implemented** |
| **A** | ~~Rescale risk/EFC penalty magnitudes~~ | **NOT DONE.** Penalties already flip 37.74 % of contested decisions; penalty SD ÷ reward SD = 0.85. Rescaling a working component would be unjustified tuning. | **Rejected on evidence** |
| **E** | ~~Hazard curve as a decision-time input~~ | **DEFERRED.** Speculative until the scheduler actually has choices to make. | **Deferred** |

**Not touched:** every scheduler's logic (FCFS, Round Robin, Least Loaded, Single-Agent RL,
Risk-Blind MARL, Risk-Aware MARL), the reward function, the risk engine, the EFC model, the GPU
physics, the fault injector, and the empirically calibrated hazard curves. Changes C1/C2 alter the
**training budget and protocol**, applied **identically to all three learners**.

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

## 3. The exact configuration being tested

Frozen in `realdata/config_v2.py`. Selected on VALIDATION using **baseline schedulers only**
(FCFS, Least Loaded) — no MARL result influenced any of it.

| Setting | Value | Selection rule |
|---|---|---|
| Episode length | **600 steps** | Shortest candidate of {120, 300, 600} whose max observed risk is within 0.02 of the longest. Measured: 0.347 / 0.497 / 0.553 → 600. |
| Load levels | **ρ ∈ {0.3, 1.1, 1.4}** | *light* = largest ρ with contested ≥ 25 % (0.3, 25.3 %); *near-capacity* = ρ whose **delivered** ρ is closest to 1.0 (1.1, delivered 1.102); *overloaded* = smallest ρ with contested ≤ 10 % (1.4, 9.2 %) |
| Risk-bucket edges | **[0.0614, 0.1165, 0.3047]** | 50th/75th/95th percentiles of the per-step per-GPU risk distribution on **TRAIN** windows under a neutral dispatcher. Non-positive or duplicate edges dropped (>25 % of observations are exactly 0.0, so equal-frequency quartiles are degenerate). v1 edges were [0.25, 0.50, 0.75]. |
| Shipped quarantine threshold | **0.85** | unchanged; always reported |
| Exploratory quarantine threshold | **0.60** | largest value on the fixed ladder {0.85…0.30} that triggers in ≥ 30 % of VALIDATION episodes under FCFS/Least Loaded only. Measured: 0.85→0 %, 0.70→0 %, **0.60→58 %**, 0.50→100 % |
| Throughput floor | **0.80** | a scheduler completing < 80 % of the best scheduler's mean completed tasks is **INFEASIBLE** |
| Training | 20 episodes × 300 steps, ρ=0.9, ε-decay and seeds per tuning | identical for `rl`, `marl`, `risk_aware_marl` |
| **Training stopping rule** | stop at **20 episodes**, or earlier if Q-table growth per episode is **< 5 states for 3 consecutive episodes** | fixed here, before the final run; identical for all learners |
| Hazard overlay | both regimes run: workload-only and overlay | unchanged, calibrated curves untouched |

*(Any value above that the tuning phase alters is recorded in §8 before the final run.)*

---

## 4. Experimental conditions — reported side by side

All three run on the **same TEST seeds**, same windows, same load levels, same episode length, so the
contribution of each fix is separable.

| Condition | Load scale | Training | Risk buckets |
|---|---|---|---|
| **v1** | legacy (30× defect) | v1 (80 steps, one window) | v1 [0.25, 0.50, 0.75] |
| **v1_corrected** | **fixed** | v1 | v1 |
| **v2** | **fixed** | **v2** | **v2 (TRAIN-derived)** |

`v1` vs `v1_corrected` isolates the load fix. `v1_corrected` vs `v2` isolates training + state.

---

## 5. Metrics

**Primary (lower is better):** `avg_risk_exposure`, `terminal_efc`, `crash_count`, `quarantine_time`
— identical definitions to v1, so v1↔v2 comparison is like-for-like.

**Secondary (higher is better):** `completed_tasks`.

**Combined objectives (lower is better)** — new in v2, and the ones that cannot be gamed by starving:
- `risk_per_completed_task` = mean risk over time ÷ completed tasks
- `crashes_per_completed_task` = crashes ÷ completed tasks

**Diagnostic (reported, not hypothesis-tested):** `mean_risk_over_time` (a true time-average, unlike
v1's terminal snapshot), `peak_gpu_risk`, `contested_pct`, `admitted_pct`.

> **Known metric defect, carried forward deliberately.** `avg_risk_exposure` maps to
> `average_gpu_risk`, a terminal snapshot rather than a time average. Measured bias: 0.2758 vs a true
> mean of 0.1451 (~1.9× inflated), but correlation r = 0.996, so rankings are unaffected. It is kept
> unchanged for v1 comparability, and `mean_risk_over_time` is reported alongside it.

---

## 6. Hypotheses

Stated before the final run. "Eligible baselines" = schedulers passing the throughput floor (§3).

| ID | Hypothesis |
|---|---|
| **V2-H1** | Risk-Aware MARL has **lower `risk_per_completed_task`** than every eligible baseline. |
| **V2-H2** | Risk-Aware MARL has **lower `crashes_per_completed_task`** than every eligible baseline. |
| **V2-H3** | Risk-Aware MARL lies on the **Pareto frontier** of (completed tasks ↑, risk exposure ↓) and is **not dominated** by any baseline. A dominates B iff A completes ≥ B's tasks **and** has ≤ B's risk, with at least one strict. |
| **V2-H4** | **Quarantine.** At the shipped 0.85 threshold, reported as a finding (expected: untestable — max observed risk on validation was 0.644). At the exploratory 0.60 threshold, quarantine triggers in ≥ 30 % of episodes **and** episodes in which it triggers show **fewer crashes** than matched episodes in which it does not. Exploratory, clearly labelled, **not the shipped mechanism**. |
| **V2-H5** | Risk-Aware MARL's `completed_tasks` is **within 5 %** of the best dispatching scheduler. |

Also reported (not hypothesis-tested): the v1 primary metrics, so v1↔v2 can be compared directly.

---

## 7. Decision rule — unchanged from v1

A hypothesis is **SUPPORTED** only if **both** hold against the strongest **eligible** baseline:

1. the **bootstrap 95 % CI** (2,000 resamples) of the paired difference **excludes zero** in the
   favourable direction, **and**
2. the **paired win rate ≥ 60 %**.

Otherwise **REJECTED**, or **INCONCLUSIVE** where the metric cannot discriminate (e.g. all values
identical, as quarantine was in v1).

**Throughput floor.** A scheduler whose mean `completed_tasks` is below **80 %** of the best
scheduler's is labelled **INFEASIBLE** and is excluded from "strongest baseline" selection — this
prevents a starving scheduler from being treated as the benchmark, which is what happened in v1.

**Single-agent RL is reported BOTH ways** — included and excluded by the floor — so the reader can
see the starvation effect directly rather than having it hidden by the rule.

---

## 8. Tuning budget and protocol

- **Hard cap: 20 configurations.** The grid is **12**: ε-decay {0.999, 0.995, 0.990} × training
  episodes {10, 20} × risk edges {v1, TRAIN-derived}.
- **Every configuration tried is logged** to `logs/v2_tuning_log.csv`, including failures.
- Tuning runs on **VALIDATION only**.
- **Selection objective, fixed in advance:** maximise `mean(completed_tasks) / mean_risk_over_time`
  for the treatment scheduler — work done per unit of risk carried. Ties broken by lower
  `crashes_per_completed_task`. Being a ratio, it cannot be won by starving.
- Load levels, episode length and the quarantine threshold are **not** tuned here; they were fixed by
  `v2/design_select.py` using baselines only.
- Q-tables are **snapshotted and frozen** (`set_eval_mode()`, ε = 0) before any evaluation, as in v1.

---

## 9. Commitments

1. **The final test runs once.** No re-running or re-tuning after seeing TEST results. If a run fails
   for a technical reason it is fixed and re-run **with the same configuration**, and that is
   reported.
2. **All three conditions and all load levels are reported**, whatever they show.
3. **A negative result will be reported as a negative result.** If Risk-Aware MARL still does not win
   under corrected conditions, that is the finding, and it will be stated as plainly as v1's was.
4. **v1 results stay visible** and are never replaced.
5. The number of tuning configurations tried is reported in the results.

---

## 10. What would falsify the v2 premise

If, after the load fix, Risk-Aware MARL remains statistically tied with Least Loaded and Risk-Blind
MARL on the combined objectives across all three load levels, then the conclusion is that
**risk-aware bidding offers no measurable benefit in this simulator** — and the diagnosis will have
corrected the *method* without rescuing the *hypothesis*. That is an acceptable and publishable
outcome, and it is the outcome this document is designed to be able to report.
