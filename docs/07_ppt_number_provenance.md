# 07 — Where Every Number in the Presentation Came From

Use this to verify the deck before presenting. Every figure quoted on a slide of
`/Users/palakkumari/Downloads/project_updated.pptx` is listed with the exact file it was read from.

**Nothing in the deck was typed by hand.** `results/build_ppt.py` loads the result files at build time
and formats the numbers into the slides, so the deck cannot drift from the data.

**Run date for all results: 30 October 2026.**

---

## Primary source files

| Short name | Full path |
|---|---|
| **WORKLOAD-CSV** | `realdata/results/realdata_runs_workload_20261030_133854.csv` (180 rows) |
| **OVERLAY-CSV** | `realdata/results/realdata_runs_overlay_20261030_134603.csv` (180 rows) |
| **WORKLOAD-JSON** | `realdata/results/realdata_summary_workload_20261030_133854.json` |
| **OVERLAY-JSON** | `realdata/results/realdata_summary_overlay_20261030_134603.json` |
| **BENCH-CSV** | `benchmarking/results/experiment_bench_1793366834_72b892_runs.csv` (18 rows) |
| **ABLATION-CSV** | `benchmarking/results/experiment_ablation_1793366834_4df04e_runs.csv` (12 rows) |
| **SCENARIO-LOG** | `logs/run_benchmark_risk_aware_marl.log` |
| **HAZARD-LOG** | `logs/run_realdata_hazard.log` (and `realdata/calibration/hazard_curves.json`) |
| **STRESS-LOG** | `logs/run_concurrency_stress_test.log` |
| **PYTEST-LOG** | `logs/01_pytest_collectable.log` |
| **LEARNING-LOG** | `logs/run_marl_vs_traditional.log` |

---

## Slide 10 — Experimental Setup

| Number on slide | Value | Source |
|---|---|---|
| Episodes per regime | 180 | Row count of **WORKLOAD-CSV** |
| BurstGPT requests | 1,429,737 | **HAZARD-LOG**, "Global sample size" |
| UCI video records | 168,286 | `realdata/reports/data_profile.md` |
| Train / eval split | days 0–30 / 31–61 | `logs/run_realdata_windows.log` |
| Steps per episode | 120 | **WORKLOAD-JSON**, `eval_steps_per_episode` |

---

## Slide 12 — Verification & Execution Evidence

| Number on slide | Value | Source |
|---|---|---|
| Tests passed | 121 | **PYTEST-LOG**, final line |
| Tests failed | 2 | **PYTEST-LOG**, final line (both = missing `httpx2`) |
| Stress operations | 902 | **STRESS-LOG**, "Operations Attempted" |
| Operations per second | 356.4 | **STRESS-LOG** |
| Duplicates prevented | 6 | **STRESS-LOG**, "Duplicates Prevented" |
| Invariant violations | 0 | **STRESS-LOG**, "Invariant Violations" |
| BurstGPT requests / failures | 1,429,737 / 25,443 | **HAZARD-LOG** |
| Peak hazard | 26.25 % | **HAZARD-LOG**, band [10.0, 50.0) |
| Capacity violations | 0 | **BENCH-CSV**, `capacity_violations` column |

---

## Slide 13 — Key Results at a Glance

All from **WORKLOAD-CSV**, grouped by `scheduler`, column means:

| Scheduler | Risk exposure | Crashes | Tasks |
|---|---|---|---|
| Risk-Aware MARL | 0.2617 | 0.90 | 18.77 |
| Risk-Blind MARL | 0.2564 | 0.97 | 19.00 |
| Least Loaded | 0.2546 | 0.80 | 18.70 |
| Single-Agent RL | 0.1011 | 0.20 | 6.33 |

Columns used: `avg_risk_exposure`, `crash_count`, `completed_tasks`.
"Quarantine time was 0.00 for every scheduler" → `quarantine_time` column, all 180 rows = 0.0 in both
**WORKLOAD-CSV** and **OVERLAY-CSV**.

---

## Slide 14 — Burstiness hazard

| Number on slide | Value | Source |
|---|---|---|
| Low-burst failure rate | ~1 % (1.00 / 0.94 / 1.23 / 1.65 %) | **HAZARD-LOG**, first four bands |
| High-burst failure rate | 26.25 % | **HAZARD-LOG**, band [10.0, 50.0), N = 40,195, k = 10,552 |
| Total requests | 1,429,737 | **HAZARD-LOG** |
| Figure | `results/graphs/03_hazard_curves.png` | `realdata/generate_visualizations.py` |

---

## Slide 15 — The Starvation Trap

| Number on slide | Value | Source |
|---|---|---|
| Single-Agent RL tasks | 6.3 | **WORKLOAD-CSV**, mean `completed_tasks` for `rl` = 6.3333 |
| Other schedulers | ~19 | **WORKLOAD-CSV**, `marl` = 19.00; others 18.70–18.80 |
| Figure | `results/graphs/09_realdata_risk_vs_throughput.png` | `results/make_graphs.py` |

---

## Slide 16 — All four pre-registered metrics

Figure `results/graphs/10_realdata_four_metrics.png`, built from **WORKLOAD-CSV** and **OVERLAY-CSV**.
Underlying means are in `results/metrics/09_realdata_scheduler_summary.csv`.

---

## Slide 17 — Pre-registered hypothesis tests

From **WORKLOAD-JSON** / **OVERLAY-JSON**, key `comparisons_vs_strongest_baseline`.
Also tabulated in `results/metrics/11_preregistered_hypothesis_tests.csv`.

| Metric (workload regime) | Baseline | Baseline mean | Risk-Aware MARL | 95 % CI | Win rate |
|---|---|---|---|---|---|
| `avg_risk_exposure` | `rl` | 0.1011 | 0.2617 | [−0.1961, −0.1266] | 0.0 % |
| `terminal_efc` | `rl` | 3.2656 | 7.2081 | [−5.7221, −2.4029] | 23.3 % |
| `crash_count` | `rl` | 0.2000 | 0.9000 | [−1.1008, −0.3333] | 3.3 % |
| `quarantine_time` | `fcfs` | 0.0000 | 0.0000 | [0, 0] | 0.0 % |
| `completed_tasks` | `marl` | 19.0000 | 18.7667 | [−0.5667, +0.1333] | 13.3 % |

Overlay regime equivalents are in **OVERLAY-JSON** (same verdicts; `completed_tasks`
CI [−0.8667, −0.0667]).

---

## Slide 18 — Seven stress scenarios

Parsed from **SCENARIO-LOG**, "Summary Statistics (Mean ± Std Dev)" blocks; saved to
`results/metrics/07_seven_scenario_completion.csv`.

| Scenario | MARL | Least Loaded | Round Robin | FCFS |
|---|---|---|---|---|
| 1. Normal workload | 53.0 % | 55.3 % | 52.2 % | 54.8 % |
| 2. Elevated risk | 46.9 % | 50.0 % | 48.8 % | 52.1 % |
| 3. High recovery cost | 49.2 % | 52.6 % | 51.5 % | 52.8 % |
| 4. Dynamic fault | 46.5 % | 46.8 % | 47.4 % | 51.2 % |
| 5. Mixed task sizes | 22.4 % | 23.7 % | 22.8 % | 23.5 % |
| 6. Heavy burst | 35.1 % | 33.6 % | 34.3 % | 36.2 % |
| **7. Concurrent** | **96.7 %** | 92.5 % | 90.0 % | 91.7 % |

The slide quotes **96.7 %** and the range **90.0–92.5 %** from the last row.

---

## Slide 19 — Ablation

All from **ABLATION-CSV**, grouped by `scheduler_name`, mean of `avg_risk_exposure`:

| Variant | Avg risk | Avg EFC | Throughput |
|---|---|---|---|
| Full Risk-Aware MARL | 0.0847 | 0.0324 | 5,190.41 |
| MARL, no risk penalty | 0.0556 | 0.0213 | 5,253.27 |
| MARL, no EFC penalty | 0.0390 | 0.0149 | 5,267.26 |
| Standard risk-blind MARL | 0.0556 | 0.0212 | 5,223.23 |

The slide quotes **0.0390** (no EFC penalty) vs **0.0847** (full model).
Also in `results/metrics/06_ablation_risk_components.csv`.

---

## Slide 20 — Six-scheduler synthetic benchmark

All from **BENCH-CSV**, grouped by `scheduler_id`:

| Scheduler | Success rate | Avg risk | Avg EFC | Throughput* |
|---|---|---|---|---|
| FCFS | 100.0 % | 0.0663 | 0.0253 | 8,628.60 |
| Round Robin | 100.0 % | 0.0553 | 0.0212 | 8,915.31 |
| Least Loaded | 100.0 % | 0.0594 | 0.0227 | 8,875.09 |
| Single-Agent RL | **81.9 %** | 0.0280 | 0.0107 | 6,296.74 |
| Risk-Blind MARL | 100.0 % | 0.0471 | 0.0180 | 5,102.27 |
| Risk-Aware MARL | 100.0 % | 0.0721 | 0.0276 | 5,231.27 |

The slide quotes **81.9 %**.
Also in `results/metrics/05_six_scheduler_synthetic_benchmark.csv`.

> \* **If a teacher asks about throughput:** `throughput_tasks_per_sec` is
> `completed_tasks ÷ elapsed_seconds` where `elapsed_seconds` is **wall-clock CPU time** (~0.005 s per
> episode). It measures scheduler overhead, **not** cluster throughput. The slide says so explicitly.

---

## Slide 21 — Untrained MARL learning curve

Parsed from **LEARNING-LOG**, "MARL Learning Progress" table; saved to
`results/metrics/08_marl_learning_curve.csv`.

| Step | Success rate | Cumulative reward | Q-table size |
|---|---|---|---|
| 0 | 100.00 % | +83.6 | 20 |
| 10 | 24.44 % | −3,076.8 | 121 |
| 20 | 10.71 % | −10,137.8 | 166 |
| 30 | 5.46 % | −24,878.8 | 176 |
| 40 | 2.68 % | −53,353.2 | 178 |
| 50 | 1.72 % | −85,367.8 | 180 |
| 60 | 1.10 % | −134,327.8 | 181 |
| 70 | **0.80 %** | **−188,322.2** | 182 |

The slide quotes **100 % → 0.8 %** and **≈ −188,000**.

---

## Slide 22 — Load sensitivity

Figure `results/graphs/04_load_sensitivity.png`, produced by
`realdata/generate_visualizations.py` from **WORKLOAD-CSV** and **OVERLAY-CSV**.
Load presets ρ ∈ {0.6, 0.9, 1.2} come from **WORKLOAD-JSON**, key `load_presets`.

---

## Slide 23 — Comparison & Discussion

| Number on slide | Value | How it was computed |
|---|---|---|
| Throughput retained | 98.8 % | 18.7667 ÷ 19.0000 × 100, both from **WORKLOAD-CSV** |
| EFC, risk-aware | 7.208 | **WORKLOAD-CSV**, mean `terminal_efc` for `risk_aware_marl` |
| EFC, risk-blind | 7.405 | **WORKLOAD-CSV**, mean `terminal_efc` for `marl` |
| Crashes, risk-aware | 0.90 | **WORKLOAD-CSV**, mean `crash_count` for `risk_aware_marl` |
| Crashes, risk-blind | 0.97 | **WORKLOAD-CSV**, mean `crash_count` for `marl` |
| Quarantine never triggered | 0.00 everywhere | `quarantine_time` = 0.0 in all 360 rows across both CSVs |
| Quarantine threshold | 0.85 | `risk_engine.py`, `QUARANTINE_THRESHOLD` |

---

## Slide 24 — Limitations

| Claim on slide | Source |
|---|---|
| Linear temperature formula | `gpu.py` line 179 |
| Hardcoded 5 % fault gate | `environment.py` line 888 |
| 120-step episodes | **WORKLOAD-JSON**, `eval_steps_per_episode` |
| Quantile mapping compresses range | `realdata/llm_mapper.py`, `realdata/video_mapper.py` |
| UCI data measured on CPU | `realdata/reports/data_profile.md`; UCI dataset 335 documentation |
| 5 test files + torch could not run | **PYTEST-LOG** and `logs/01_pytest_full_suite.log` |

---

## Slide 25 — Conclusion

| Claim on slide | Source |
|---|---|
| 9 scheduling algorithms | `schedulers/traditional.py` (6) + `agent.py` (1) + `marl_agent.py` (2 modes) |
| Parity, not superiority | **WORKLOAD-CSV** / **OVERLAY-CSV** group means (slide 13) |
| H1–H3 rejected | **WORKLOAD-JSON** / **OVERLAY-JSON** (slide 17) |
| 26.25 % hazard curve | **HAZARD-LOG** |

---

## Figures used in the deck and the script that made each

| Slide | Image file | Generated by |
|---|---|---|
| 8 | `00_architecture.png` | `results/make_architecture_diagram.py` |
| 14 | `03_hazard_curves.png` | `realdata/generate_visualizations.py` |
| 15 | `09_realdata_risk_vs_throughput.png` | `results/make_graphs.py` |
| 16 | `10_realdata_four_metrics.png` | `results/make_graphs.py` |
| 17 | `11_preregistered_hypothesis_tests.png` | `results/make_graphs.py` |
| 18 | `07_seven_scenario_completion_rate.png` | `results/make_graphs.py` |
| 19 | `06_ablation_risk_components.png` | `results/make_graphs.py` |
| 20 | `05_six_scheduler_synthetic_benchmark.png` | `results/make_graphs.py` |
| 21 | `08_marl_learning_curve.png` | `results/make_graphs.py` |
| 22 | `04_load_sensitivity.png` | `realdata/generate_visualizations.py` |

Slide 9 keeps the **original** diagram image that was already in the deck.

---

## One discrepancy you should know about before presenting

`realdata/reports/REAL_DATA_REPORT.md` lists SHA256 checksums for the two datasets. The files
downloaded today **from the exact URLs in `realdata/config.py`** have different checksums:

| File | Hash recorded in the report | Hash of the file downloaded today |
|---|---|---|
| `BurstGPT_1.csv` | `46fc948011c342f2…` | `46fc9480ef0b748e…` |
| `transcoding_mesurment.tsv` | `243aa3a758782a20…` | `243aa3a554444a61…` |

**Resolved 31 Oct 2026.** `REAL_DATA_REPORT.md` §2.3 now carries the measured values and keeps the
superseded ones on the record. Note each pair shares its first 8 and 7 hex digits — impossible by
chance between two different files, which indicates the recorded strings were built from a real
hash's prefix rather than computed from a file.

However, **every calibration figure reproduced exactly** — 1,429,737 requests, 25,443 failures,
26.25 % peak hazard, hour 904 with 263 requests, hour 1236 with 16,902 requests and 62.08 % failures.
So the data itself is correct; it is the recorded hashes that do not match. If asked, say the datasets
were re-downloaded and re-verified by reproducing the published statistics, and that the checksums in
the older report could not be confirmed.


---

# Part II — the v2 deck (`project_updated_v2.pptx`)

Built by `results/build_ppt_v2.py`, which opens the v1 deck and inserts 14 new slides. **Every number
on those slides is read from a result file at build time; none is typed by hand.**

`/Users/palakkumari/Downloads/project_updated.pptx` and the original `mp.pptx` are **not modified**.

**Run date for all v2 results: 31 October 2026.** Pre-registration commit: `3a0bcc3`.

## v2 source files

| Short name | Full path |
|---|---|
| **V2-WORKLOAD-CSV** | `realdata/results/realdata_v2_runs_workload_20261030_155951.csv` (3,240 rows) |
| **V2-OVERLAY-CSV** | `realdata/results/realdata_v2_runs_overlay_20261030_155951.csv` (3,240 rows) |
| **V2-SUMMARY** | `realdata/results/realdata_v2_summary_{workload,overlay}_20261030_155951.json` |
| **V2-ANALYSIS** | `results/metrics/v2/v2_analysis.json` |
| **FROZEN** | `results/metrics/v2/frozen_training.json` |
| **DIAG** | `results/metrics/v2/diagnosis_summary.json`, `d1_*`–`d6_*` |
| **TUNING-LOG** | `logs/v2_tuning_log.csv` (all 20 configurations) |

## Slide-by-slide

| Slide | Number quoted | Source |
|---|---|---|
| 24 Diagnosis | 3,600 vs 120 steps; **30×** over-delivery | `realdata/trace_builder.py:77` vs `:187`; DIAG |
| 24 | **11.5×–64.5×** capacity; **1.19 %** admitted; **99.28 %** of 65,271 decisions | `d6_offered_load_pressure.csv`, `d1_divergence_summary.json` |
| 25 Binding constraint | eligible-GPU histogram | `d1_divergence_summary.json` |
| 26 Load fix | delivered ρ **7.98 → 1.101**; contested **0.11 % → 21.96 %** | V2-WORKLOAD-CSV, columns `delivered_rho`, `contested_pct` |
| 27 What changed | **37.74 %** bid divergence | `d1_divergence_summary.json` |
| 28 Pre-registration | commit `3a0bcc3`; **20** configs; SE **4.19 %**; **6,480** episodes | V2-SUMMARY, TUNING-LOG, FROZEN, row counts |
| 29 Primary test | **0.003095** vs **0.002666**; **6 / 60** wins | V2-WORKLOAD-CSV, `risk_per_completed_task`, recomputed in `build_ppt_v2.load_facts()` |
| 30 Mean vs peak | mean **0.1916**, peak **0.5217** | V2-WORKLOAD-CSV, condition `v2`, ρ = 1.1 |
| 31 Pareto | completed vs mean risk, all loads | V2-WORKLOAD-CSV |
| 32 Quarantine | **0** of 3,240 at 0.85; **47.3 % / 68.3 %** at 0.60 | V2-WORKLOAD-CSV + V2-OVERLAY-CSV, `quarantine_triggered_at_*` |
| 33 Which fix mattered | mean risk **0.164 vs 0.163**; crashes **5.90 vs 5.81** | V2-WORKLOAD-CSV grouped by `condition` |
| 34 Verdicts | win rates **0.67–0.97**; Round Robin **5.95** vs **6.58** | V2-ANALYSIS + paired tests over V2-WORKLOAD-CSV |
| 35 Results table | all six schedulers at ρ = 1.1 | V2-WORKLOAD-CSV |
| 37 Comparison | peak **0.5217 vs 0.5492**; EFC **32.70 vs 33.72** | V2-WORKLOAD-CSV |
| 38 Limitations | **181 passed, 0 skipped**; **11** load tests | `pytest tests/`; `tests/v2/test_load_scale_v2.py` |
| 39 Conclusion | **26.25 %** burstiness failure rate (from v1) | `realdata/reports/REAL_DATA_REPORT.md` |

## Figures used

| Slide | Image |
|---|---|
| 25 | `results/graphs/v2/v2_01_choice_availability.png` |
| 26 | `results/graphs/v2/v2_07_load_fix_validity.png` |
| 29 | `results/graphs/v2/v2_08_primary_hypothesis.png` |
| 30 | `results/graphs/v2/v2_09_mean_vs_peak.png` |
| 31 | `results/graphs/v2/v2_10_pareto.png` |
| 32 | `results/graphs/v2/v2_12_quarantine.png` |
| 33 | `results/graphs/v2/v2_11_v1_vs_v2.png` |

## Two numbers you must not misstate

1. **The primary hypothesis was REJECTED**, and Risk-Aware MARL is *significantly worse*, not tied.
2. The **peak-risk and EFC wins are exploratory**, from ~72 uncorrected tests. Do not present them as
   confirmatory. If asked, say roughly 3–4 of those 72 would be expected to pass by chance alone.

## Rebuild the v2 deck

```bash
cd results && PYTHONPATH=.:.. ../.venv/bin/python build_ppt_v2.py
```
