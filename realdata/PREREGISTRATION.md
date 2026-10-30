# Pre-Registration Plan: Real-Data Risk Evaluation

> **Status**: **PRE-REGISTERED**  
> **Repository Branch**: `realdata-risk`  
> **Commitment**: All hypotheses, metric designations, and statistical decision rules documented herein are committed **PRIOR** to running benchmark evaluations on real-world workload traces.

---

## 1. Experimental Motivation & Scope

This benchmark evaluates the scheduling efficacy of **Risk-Aware Multi-Agent Reinforcement Learning (`risk_aware_marl`)** against five competing scheduling paradigms:
1. `fcfs`: First-Come, First-Served baseline.
2. `round_robin`: Sequential round-robin dispatcher.
3. `least_loaded`: Minimum queue and memory occupancy heuristic.
4. `rl`: Centralized single-agent Q-learning (`agent.RLAgent`).
5. `marl`: Decentralized risk-blind MARL bidding agent.

Workload traces are generated from empirical telemetry:
- **BurstGPT Production Trace**: LLM request arrivals, KV-cache memory requirements, and burst-induced failure events.
- **UCI Video Transcoding Dataset**: Empirical video duration and CPU memory relative distributions.
- **Offered Load Presets**: $\rho \in \{0.60, 0.90, 1.20\}$ (under-subscribed, near-capacity, and over-subscribed stress regimes on a $4 \times 8\text{ GB}$ cluster).

---

## 2. Hypotheses

### Primary Hypotheses (Safety & Risk Mitigation — Lower is Better)
- **$H_{1}$ (Risk Exposure)**: Under real-world burst workloads, `risk_aware_marl` maintains a statistically significantly lower mean GPU risk exposure (`avg_risk_exposure`) than the strongest baseline.
- **$H_{2}$ (Expected Failure Cost)**: `risk_aware_marl` incurs a statistically significantly lower cumulative Expected Failure Cost (`terminal_efc`) than the strongest baseline.
- **$H_{3}$ (Cluster Reliability & Crashes)**: `risk_aware_marl` triggers significantly fewer hardware crash events (`crash_count`) under high load and burst arrival conditions than risk-blind schedulers.
- **$H_{4}$ (Quarantine Prevention)**: `risk_aware_marl` minimizes GPU quarantine time (`quarantine_time`), preventing cascading GPU isolation.

### Secondary Hypothesis (Task Throughput — Higher is Better)
- **$H_{5}$ (Non-Destructive Safety)**: `risk_aware_marl` achieves completed tasks (`completed_tasks`) competitive with traditional schedulers, ensuring safety penalties do not cause artificial task starvation.

---

## 3. Metrics Specification

| Metric Identifier | Role | Direction | Mathematical Definition / Source | Unit |
|---|---|---|---|---|
| `avg_risk_exposure` | **Primary** | **Lower is Better** | Mean risk score across all GPUs over the episode: `env.get_metrics()["average_gpu_risk"]` | Score $[0.0, 1.0]$ |
| `terminal_efc` | **Primary** | **Lower is Better** | Cumulative assigned Expected Failure Cost: `env.get_metrics()["total_assigned_efc"]` | Cost Units |
| `crash_count` | **Primary** | **Lower is Better** | Cumulative hardware crash events: `env.get_metrics()["total_crash_events"]` | Count |
| `quarantine_time` | **Primary** | **Lower is Better** | Total GPU quarantine episodes/steps: `env.get_metrics()["quarantined_gpus"]` | Count / Steps |
| `completed_tasks` | **Secondary** | **Higher is Better** | Total successfully finished tasks: `env.get_metrics()["completed"]` | Task Count |

---

## 4. Pre-Registered Decision Rules

All comparisons strictly reuse the non-parametric statistical functions from [`evaluation/stats.py`](file:///home/bmsit/Downloads/gpu-draft-main/evaluation/stats.py):
- `paired_diff_ci(treatment, baseline, n_boot=2000, seed=0)`
- `win_rate(treatment, baseline, higher_is_better)`
- `ci_excludes_zero(low, high)`

### Strongest Baseline Definition
For any metric $M$, the **strongest baseline** candidate among the five non-treatment schedulers (`fcfs`, `round_robin`, `least_loaded`, `rl`, `marl`) is the candidate with the best sample mean on metric $M$:
$$\text{Baseline}^*(M) = \begin{cases} 
\arg\min_{s} \text{Mean}(s, M) & \text{if } M \text{ is lower-is-better} \\ 
\arg\max_{s} \text{Mean}(s, M) & \text{if } M \text{ is higher-is-better} 
\end{cases}$$

### Pre-Registered Superiority Criterion
`risk_aware_marl` will be declared **Statistically Superior** on primary metric $M$ if and only if all three conditions hold:
1. **Bootstrap 95% Confidence Interval Excludes Zero**:
   The 95% CI for paired differences $\Delta = \text{treatment} - \text{baseline}^*$ strictly excludes zero in the favorable direction (`significant_better` is `True`).
2. **Paired Win Rate Threshold**:
   The paired win rate $\text{win\_rate}(\text{treatment}, \text{baseline}^*) \ge 0.60$ (wins in at least 60% of paired seed/condition comparisons).
3. **Throughput Non-Inferiority**:
   `risk_aware_marl` is **not significantly worse** than $\text{baseline}^*$ on the secondary throughput metric (`completed_tasks`).

---

## 5. Rationale for Excluding Percentage Failure / Rejection Rates

As formally documented in [`realdata/NOTES.md`](file:///home/bmsit/Downloads/gpu-draft-main/realdata/NOTES.md):
1. **Scheduler Queuing Asymmetry**: Traditional schedulers (`fcfs`, `round_robin`, `least_loaded`) queue tasks in internal queues indefinitely rather than emitting immediate HTTP-style task rejections, while specialized queue dropping policies vary across modules.
2. **Rate Distortion Under Fixed Steps**: Percentage rates can be distorted when denominators vary due to arrival scaling or incomplete trace draining within a fixed horizon.
3. **Objective Ground-Truth**: Raw crash counts, empirical risk exposure, and completed tasks provide unambiguous, unconfounded physical measurements of reliability.

---

## 6. Experimental Protocol & Verification

- **Evaluation Windows**: Canonical disjoint EVAL windows (`median_hour`, `busiest_hour`, `burstiest_window`).
- **Load Presets**: $\rho \in \{0.60, 0.90, 1.20\}$.
- **Repetitions**: Minimum 5 random seeds (`[42, 43, 44, 45, 46]`).
- **Trace Pairing**: For each window $\times$ preset $\times$ seed, exactly **one** master trace is generated and deep-copied to each scheduler.
- **Training Isolation**: RL and MARL are trained exclusively on disjoint TRAIN windows (Days 0–30); Q-tables are frozen (`set_eval_mode()`) before evaluation.
- **Seeding Invariant**: Explicit `random.seed(seed)` and `numpy.random.seed(seed)` calls precede every episode.
