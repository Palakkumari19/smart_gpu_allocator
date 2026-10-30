# 07 — Phase 1 Diagnosis: Why Risk-Aware MARL Did Not Win in v1

**Date:** 30 October 2026 · **Branch:** `v2-improvements` · **Harness:** `v2/diagnose.py`
**Raw metrics:** `results/metrics/v2/` · **Figures:** `results/graphs/v2/` · **Log:** `logs/v2/diagnose_full.log`

Nothing in the algorithm, reward, physics or baselines was changed to produce this document. The
diagnostic harness only *wraps* existing functions to observe them. All v1 results and
`realdata/PREREGISTRATION.md` remain untouched.

**Scope note.** The diagnosis reproduces v1's exact conditions: `seconds_per_step=1.0`, 80 training
steps on the `median_hour` TRAIN window with seed 42, 120-step evaluation episodes, frozen Q-tables.
Sweeps use seeds 42–44 and ρ ∈ {0.6, 0.9, 1.2} over both EVAL windows unless stated.

---

## The one-sentence answer

**v1 did not measure what it intended to measure.** A load-scaling defect delivered **exactly 30× the
intended workload**, which saturated the cluster so completely that **99.3 % of scheduling decisions
had zero eligible GPUs** — so five of the six schedulers were forced into identical behaviour and
parity was the only possible outcome.

The risk-aware mechanism itself is **not** broken. When it was actually consulted it changed the
decision 37.7 % of the time.

---

## Finding 1 — The scheduler almost never gets to choose

> Question asked: *how often does Risk-Aware MARL choose a different GPU than Risk-Blind MARL? If under
> ~5 %, the risk penalty is not influencing decisions.*

**Method.** A counterfactual test. The real risk-aware policy runs. At every decision the same four bid
breakdowns are scored twice — once with the risk and EFC penalties, once without. State is identical,
so any difference is attributable purely to the penalties.

**Result — 65,271 scheduling decisions:**

| Eligible GPUs | Decisions | Share |
|---|---|---|
| **0** (task fits nowhere) | 64,808 | **99.28 %** |
| 1 (forced choice) | 206 | 0.32 % |
| 2 | 92 | 0.14 % |
| 3 | 77 | 0.12 % |
| 4 | 88 | 0.13 % |
| **≥ 2 — a genuine choice** | **257** | **0.39 %** |

Of those 257 genuine choices, the penalties changed the selected GPU **97 times = 37.74 %**.

**Verdict: the premise of the question is refuted.** Divergence is 37.7 %, far above the 5 % threshold.
The risk penalty is a strong influence on decisions — it just gets consulted in 0.39 % of them.

### Are the penalty terms big enough? Yes.

Measured over the 767 candidate evaluations in contested decisions:

| Bid term | Mean | Median | p95 | Max |
|---|---|---|---|---|
| Q-value contribution (learned) | 0.0201 | **0.0000** | 0.0438 | 0.1206 |
| Performance score (hand-written) | 0.2161 | 0.2096 | 0.2744 | 0.2866 |
| Risk penalty | 0.0316 | 0.0254 | 0.0940 | 0.1221 |
| EFC penalty | 0.0077 | 0.0013 | 0.0324 | 0.0637 |

- Reward side (Q + performance): mean **0.2362**, SD **0.0425**
- Penalty side (risk + EFC): mean **0.0393**, SD **0.0361**
- **Penalty SD ÷ reward SD = 0.85**

What decides an `argmax` is *spread across candidates*, not absolute magnitude. The penalties vary
almost as much as the reward side, which is exactly why they flip 37.7 % of contested picks.

> **This rules out proposed improvement A (rescaling the penalty magnitudes).** The penalties are
> already correctly scaled. Rescaling them would be tuning a component the evidence shows is working.

### A separate problem this exposes

The **median learned Q-contribution is 0.0000**. In most states the Q-table has never been visited, so
`q_value = 0` and the bid reduces to `performance_score − penalties`. The "learned" part of this MARL
contributes a mean of 0.0201 against a hand-written heuristic worth 0.2161 — **the heuristic is ~11×
more influential than the learning.** See Finding 5.

*Figures:* `v2_01_choice_availability.png`, `v2_02_bid_term_magnitudes.png`
*Data:* `d1_divergence_summary.json`, `d1_bid_components.csv`

---

## Finding 2 — Root cause: the offered load is exactly 30× too high

> This was not on the diagnostic list. It is the single most important finding.

`compute_target_load_scale()` ([`trace_builder.py:77`](../realdata/trace_builder.py#L77)) computes the
arrival rate over the **window duration**:

```python
total_steps = max(1.0, window_duration_seconds / seconds_per_step)   # 3600 / 1.0 = 3600 steps
raw_arrival_rate = raw_requests_count / total_steps
load_scale = target_utilization / raw_rho
```

But `build_real_trace()` ([`trace_builder.py:187`](../realdata/trace_builder.py#L187)) packs that hour
into the episode's step count:

```python
num_steps = total_steps or int(math.ceil(duration_sec / max(0.1, seconds_per_step)))   # = 120
```

The scale is computed for **3600 steps** and applied over **120 steps**. Every arrival is therefore
delivered **3600 / 120 = 30×** too fast.

**Measured, per 120-step episode:**

| Window | ρ target | Tasks intended | Tasks delivered | Over-delivery |
|---|---|---|---|---|
| median_hour | 0.6 | 47.8 | 1,435 | **30.0×** |
| median_hour | 0.9 | 71.7 | 2,153 | **30.0×** |
| median_hour | 1.2 | 95.7 | 2,870 | **30.0×** |
| burstiest_window | 0.6 | 47.8 | 1,436 | **30.0×** |
| burstiest_window | 0.9 | 71.7 | 2,155 | **30.0×** |
| burstiest_window | 1.2 | 95.7 | 2,873 | **30.0×** |

Exactly 30.0× in all six cells — the signature of a pure step-count mismatch.

**Resulting actual utilisation** (offered GB·steps per step ÷ 32 GB cluster):

| Window | ρ = 0.6 | ρ = 0.9 | ρ = 1.2 |
|---|---|---|---|
| median_hour | 32.4× capacity | 48.4× | 64.5× |
| burstiest_window | 11.5× | 17.1× | 22.9× |

A secondary contributor: the hardcoded constants `DEFAULT_AVG_MEMORY_GB = 4.54` and
`DEFAULT_AVG_DURATION_STEPS = 10.61` do not match the per-window calibrated means (median_hour is
5.45 GB / 14.7 steps; burstiest_window is 2.90 GB / 6.5 steps), which is why the two windows land at
different multiples of capacity despite the identical 30× task over-delivery.

**Consequences, all confirmed by measurement:**

1. Only **1.19 %** of offered tasks are ever admitted (25.6 of 2,154).
2. The cluster is permanently full → 99.3 % of decisions have no eligible GPU (Finding 1).
3. **The ρ presets are not distinguishable.** At 1000 steps, max risk was 0.7056 / 0.7050 / 0.7153 at
   ρ = 0.9 / 1.2 / 2.0 — all three are far past saturation, so "load" was never a real experimental
   variable in v1.

*Data:* `d6_offered_load_pressure.csv`

---

## Finding 3 — The ablation "mystery" was noise

> Question asked: *re-run the ablation recording BOTH peak and mean per-GPU risk, to test whether risk
> penalties spread load (raising mean, lowering peak). Report the result either way.*

v1's ablation reported only a terminal mean and appeared to show that *removing* safety penalties
*lowered* risk. Re-run here with risk sampled **every step** on all 4 GPUs, 18 episodes per variant:

| Variant | Mean risk (over time) | Peak GPU risk | Mean of per-GPU peaks | Crashes | Tasks |
|---|---|---|---|---|---|
| Full Risk-Aware | 0.1365 | **0.3778** | 0.2838 | **0.889** | 19.00 |
| No risk penalty | 0.1365 | 0.3772 | 0.2822 | 1.000 | 19.06 |
| No EFC penalty | 0.1366 | **0.3729** | 0.2841 | 0.944 | 18.94 |
| Risk-blind | **0.1355** | 0.3845 | 0.2788 | 1.000 | 19.00 |

Standard deviations: mean risk ≈ 0.098, peak risk ≈ 0.11, crashes ≈ 1.2, tasks ≈ 10.1.

**Full risk-aware vs risk-blind:** peak −0.0067, mean +0.0010, crashes −0.111, tasks ±0.000.

**Verdict.** The spreading hypothesis is *directionally* supported — the full model does have the lower
peak and a marginally higher mean — but the effects are **6 % and 1 % of one standard deviation**. They
are indistinguishable from noise. **v1's apparent reversal was a 3-seed artefact on a different
(synthetic) harness, not a real property of the algorithm.** The honest statement is that all four
variants perform identically under v1's conditions, which is exactly what Finding 1 predicts.

*Figure:* `v2_03_ablation_peak_vs_mean.png` · *Data:* `d2_ablation_peak_mean.csv`

---

## Finding 4 — Quarantine is unreachable, at any episode length or load

> Question asked: *what threshold triggers quarantine, what was the max risk reached, and do longer
> episodes (500, 1000 steps) or higher load let it trigger at all?*

**Threshold:** `RiskEngine.QUARANTINE_THRESHOLD = 0.85`.

| Episode length | ρ = 0.9 | ρ = 1.2 | ρ = 2.0 | Episodes quarantined |
|---|---|---|---|---|
| 120 steps (v1) | 0.331 | 0.331 | 0.331 | **0 / 6** |
| 500 steps | 0.651 | 0.624 | 0.652 | **0 / 6** |
| 1000 steps | 0.706 | 0.705 | 0.715 | **0 / 6** |

(values = highest risk any GPU ever reached, max over seeds)

**Verdict: quarantine never triggered in any of the 18 episodes.** Risk grows with episode length but
**asymptotes around 0.70–0.72**, well below 0.85. Raising load from 0.9 to 2.0 changes nothing, because
of Finding 2 — all loads are already saturated.

This is why v1's H4 came back inconclusive, and it would come back inconclusive again at 1000 steps.
The risk score is a weighted blend in which reaching 0.85 requires nearly all eight components to be
near maximum simultaneously; with the current physics and the hardcoded 5 % fault gate, that
combination does not occur.

> **Implication for v2:** lengthening episodes alone will *not* make H4 answerable. Either the stress
> must be raised in a way the simulator can actually express (more fault injection / hazard overlay),
> or the threshold must be treated as a tunable parameter and pre-registered as such — or H4 must be
> declared untestable and dropped. This must be decided *before* the final run.

*Figure:* `v2_04_quarantine_reachability.png` · *Data:* `d3_quarantine_reachability.csv`

---

## Finding 5 — Starvation confirmed, and training is grossly inadequate

### 5a. The starvation mechanism (v1's headline oddity)

Per 120-step episode, averaged over 18 conditions:

| Scheduler | Offered | Admitted | Admitted % | Completed | Mean temp | Mean risk | Crashes |
|---|---|---|---|---|---|---|---|
| Least Loaded | 2,154 | 25.6 | 1.19 % | 18.7 | 60.9 °C | 0.272 | 0.89 |
| Risk-Blind MARL | 2,154 | 25.6 | 1.19 % | 19.0 | 60.7 °C | 0.273 | 1.00 |
| Risk-Aware MARL | 2,154 | 25.7 | 1.19 % | 19.0 | 61.0 °C | 0.280 | 0.89 |
| **Single-Agent RL** | 2,154 | **7.7** | **0.36 %** | **6.0** | **39.6 °C** | **0.104** | **0.17** |

The mechanism is exactly as suspected: RL admits **one third** as many tasks, so its GPUs run **21 °C
cooler**, never approach the 85 °C crash threshold, accumulate no failure history, and therefore score
the lowest risk. Its "safety" is produced by not working.

*Figure:* `v2_05_starvation_mechanism.png` · *Data:* `d4_starvation.csv`

### 5b. Training adequacy — the learning barely happens

v1 trained for **80 steps** on the `median_hour` TRAIN window, which contains **193 requests**, with a
single seed.

| Measure | Value |
|---|---|
| Training steps | 80 |
| Distinct states learned (Q-table size) | **49** |
| Q-table growth in the final quarter | +14 (**still rising — not converged**) |
| Final exploration rate ε | **0.852** (intended floor: 0.05) |
| States the representation can express | **57,395,628** |
| **Coverage** | **0.0000854 %** |

Two independent failures:
1. **ε never decays.** Training ends with agents taking random actions **85 %** of the time. The decay
   schedule (0.997) needs thousands of steps; it got 80.
2. **Coverage is effectively zero.** 49 of 57.4 million states. This is why the median Q-contribution
   in Finding 1 is exactly 0.0 — nearly every state encountered at evaluation time was never seen in
   training.

**Verdict: the agents are essentially untrained.** What v1 evaluated as "MARL" was, in the large
majority of decisions, the hand-written `perf_score` heuristic plus the risk penalties.

*Figure:* `v2_06_training_adequacy.png` · *Data:* `d5_training_adequacy.json`, `d5_training_curve.csv`

---

## Finding 6 — Other defects found

### 6a. The state's risk dimension is effectively constant

`risk_bin` uses 4 buckets: ≤0.25, 0.25–0.50, 0.50–0.75, >0.75. Observed across all decisions:

| Bucket | Share of observations |
|---|---|
| bin 0 (≤ 0.25) | **83.96 %** |
| bin 1 (0.25–0.50) | 16.04 % |
| bin 2 (0.50–0.75) | **0.00 %** |
| bin 3 (> 0.75) | **0.00 %** |

Observed risk: mean 0.126, p95 0.376, max 0.489. **Half the risk buckets are never used, and 84 % of
observations collapse into a single bucket.** The agent therefore cannot distinguish "slightly risky"
from "safe" — the dimension the whole project is about is nearly invisible in its own state.

### 6b. `avg_risk_exposure` is a terminal snapshot, not an average

v1's primary metric maps to `average_gpu_risk`, which `RiskEngine.get_global_risk_metrics()` computes
as the mean across the 4 GPUs **at the moment it is called** — i.e. the final step. It is a spatial
average at one instant, not an average over the episode.

Measured bias: terminal snapshot **0.2758** vs true time-average **0.1451** — inflated ~1.9×.
**However**, correlation between the two is **r = 0.996**, so *rankings* from v1 remain valid. The
metric's absolute values are wrong; its comparative conclusions are not. This is a naming/reporting
defect, not an invalidation of v1.

### 6c. Training/evaluation mismatch between the two MARL variants

[`run_realdata_benchmark.py:138`](../realdata/run_realdata_benchmark.py#L138) trains a single Q-table
using `GPUEnvironment(scheduler="marl")` with **no adapter applied** — so training uses the *default*,
i.e. the **risk-aware** bid rule. That one table is then restored into both `marl` (risk-blind) and
`risk_aware_marl` at evaluation. Risk-blind MARL is therefore evaluated on a policy learned under a
different decision rule. The effect is small (Q contributes ~0.02 of a ~0.24 bid) but it is a genuine
confound.

### 6d. Checked and found clean

- `set_eval_mode()` correctly sets `learning_enabled = False` and ε = 0, and `learning_enabled` is
  honoured at all four update sites in `environment.py` (lines 631, 735, 769, 945). **v1's frozen
  Q-table protocol was sound.**
- `enable_deterministic_physics()` is applied to every environment in both training and evaluation.
- Trace pairing is correct: one master trace per condition, deep-copied to all six schedulers.

---

## Summary: what explains v1's parity

| # | Finding | Severity | Explains parity? |
|---|---|---|---|
| 2 | Offered load is exactly 30× too high | **Critical** | **Yes — primary cause** |
| 1 | 99.3 % of decisions have no eligible GPU | **Critical** | **Yes — direct consequence of #2** |
| 5b | Agents essentially untrained (ε = 0.85, 49 states) | **High** | Yes — contributory |
| 4 | Quarantine threshold unreachable (max 0.72 vs 0.85) | **High** | Explains H4 inconclusive |
| 6a | Risk dimension of state collapses to ~1 bucket | Medium | Contributory |
| 6c | Risk-blind MARL evaluated on risk-aware-trained table | Low | Minor confound |
| 6b | Primary metric is a terminal snapshot | Low (reporting) | No — rankings still valid |
| 3 | Ablation reversal | — | **Not real; it was noise** |

**The headline conclusion:** v1's negative result is *not* evidence that risk-aware MARL fails. It is
evidence that **v1's experiment could not discriminate between any schedulers**, because the cluster
was saturated ~30× beyond the intended operating point. The honest v1 claim should be weakened from
"risk-aware MARL does not beat the baselines" to **"under v1's conditions no scheduler could have
beaten any other, so the question was not answered."**

That is a meaningfully different — and more useful — statement, and it is what the v2 experiment must
now test properly.
