# SmartGPU Allocator — a plain-language summary

*For anyone who has never heard of reinforcement learning or GPU scheduling.*
Every number comes from a file this project produced; sources are in brackets.

---

## 1. The project in one sentence

We built a simulated computer cluster to test whether a safety-aware AI scheduler crashes fewer
machines than simple traditional rules — and found that it does not.

---

## 2. What we built

- **A simulator of four GPUs** (the chips that run AI work): they heat up, run out of memory and
  crash at 85 °C — like four ovens in a small kitchen.
- **Nine schedulers.** A scheduler decides which GPU each job goes to, like a head waiter assigning
  tables. Some are simple (take turns; pick the emptiest); some learn from experience.
- **A risk engine** turning each GPU's temperature, memory and error history into one "how worried
  should we be?" score from 0 to 1.
- **The idea under test — risk-aware scheduling.** Our scheduler bids for jobs like an auction, but
  subtracts points for risky GPUs. A waiter who avoids the wobbly table.
- Plus a dashboard, a thread-safety layer, and a replay of **1.4 million real requests**.

---

## 3. What we tested

We claimed our scheduler would accumulate **less risk per job completed** than the best simple rule.
We wrote that claim down with its exact pass mark and **committed it to version control before
running the final test** — this is **pre-registration** (deciding what counts as success before you
can see the answer).

It matters because it stops us quietly moving the goalposts afterwards.

---

## 4. What we found — stated plainly

**We were wrong. The main test failed.**

**Rejected.** Risk-Aware MARL scored **0.003095** risk per job; the simple "Least Loaded" rule scored
**0.002666** (lower is better). Ours won only **6 of 60** matched comparisons, needing 36. Not a tie —
**significantly worse**. *(docs/05 §14.1)*

**Our first experiment never tested anything.** A bug in our measurement code computed arrival speed
on a 3,600-step clock, then played jobs into a 120-step episode — so jobs arrived **30× too fast**.
Imagine testing waiters by sending a year of customers through the door in one afternoon: every
waiter looks identical, because all anyone can do is turn people away. **99.28%** of decisions had no
free GPU to choose from. The honest statement about v1 is that **it did not test its own hypothesis.**
*(docs/05 §11)*

**After the fix, the test was valid.** Delivered load matched the target (1.101 vs 1.1, against 7.98
before) and about **22%** of decisions were real choices. It still said no. *(docs/05 §11)*

**One feature couldn't be tested.** "Quarantine" pulls a dangerous GPU out of service at risk 0.85.
Risk never got that high — it fired in **0 of 3,240 episodes**, twice over. **Untestable**, not
disproved. *(docs/05 §14.6)*

**Our clever improvements did almost nothing.** "Bug fix only" vs "everything": mean risk **0.164 vs
0.163**, crashes **5.90 vs 5.81** — **under 1%**. The bug fix was the whole story. *(docs/05 §14.7)*

---

## 5. What went well

### a. A solid, working system
All nine schedulers, the simulator, risk engine, dashboard and data pipeline run; **181 tests pass,
none skipped.** Under 8 competing threads: **902 operations at 356/sec, 0 invariant violations**, with
6 duplicate assignments and 607 over-capacity requests correctly blocked.
*(`logs/run_concurrency_stress_test.log`)*

### b. Scientific rigour
Pre-registered hypotheses; **identical job streams for every scheduler**; separate training/tuning/
final data; **bootstrap** confidence intervals (re-sampling data to see how much a result wobbles); a
disclosed tuning budget; a revision log; the final test run **once**. That is why a negative result
here beats an unverified win — we couldn't have faked it.

### c. We found our own bugs
Besides the 30× bug, we caught **three hazard-overlay bugs before the final run** — one left the
overlay switched off, silently making half our results meaningless. We fixed a script printing
*"SUCCESS: MARL has demonstrated superiority"* whenever it won over 40% of comparisons — below a coin
flip. And the recorded dataset checksums didn't match the files. Catching these ourselves is why the
final number can be believed. *(CHANGES_MADE.md, Changes 3–5)*

### d. A real-data finding
From **1,429,737 real requests**: smooth traffic fails ~**0.94%** of the time, extreme bursts
**26.25%** (95% CI **[25.82%, 26.68%]**) — a **26-fold jump**. Measured from real logs, so it is
**independent of our simulator and our bug**. *(docs/05 §5)*

### e. A methodological contribution
The **"starvation trap"**: a safety score with no productivity requirement is won by doing no work —
idle GPUs stay cool. One scheduler scored best on safety while completing 108 jobs to everyone else's
151. Subtler version: a metric built on **average** risk can prefer the wrong arrangement of work. We
addressed both with a **throughput floor** (ignore anything below 80% of the best output) and by
measuring risk *per job completed*. *(docs/05 §4, §14.3)*

### f. The risk mechanism does work
Where a real choice existed, the risk penalties changed which GPU was picked in **37.7%** of contested
decisions (97 of 257). Not inert — just not producing the outcome we predicted.
*(`results/metrics/v2/d1_divergence_summary.json`)*

### g. One scenario where it won
Under **concurrent arrivals**, Risk-Aware MARL completed **96.7% ± 1.2** of jobs vs Least Loaded
**92.5% ± 2.0**, FCFS **91.7% ± 5.1**, Round Robin **90.0% ± 2.0**. From
`benchmark_risk_aware_marl.py`, which uses its own workload generator and is **not affected by the 30×
bug**. Caveats: **3 seeds, no confidence intervals**; the other six scenarios were ties.
*(`results/metrics/07_seven_scenario_completion.csv`)*

### h. Exploratory observations — not pre-registered
Risk-Aware MARL had the **lowest peak GPU risk (0.5217)** and **lowest EFC** (expected failure cost —
roughly "how costly would a failure here be") at **32.70**, while completing the most jobs. But these
come from ~**72 uncorrected comparisons**, so 3–4 would look impressive by chance, and **Round Robin
still beat it on crashes** (5.95 vs 6.58) with no risk model at all. Most of the benefit likely comes
from **spreading work out**, not the risk model. **Not a claim of superiority.** *(docs/05 §14.4)*

### i. Reproducibility
The pipeline re-runs end to end, every command is in `docs/02_how_to_run.md`, and the work sits on a
branch with **17 commits**. All 20 tuning configurations are logged, failures included.

---

## 6. What we can and cannot claim

**We can claim:** the engineering works and is thread-safe under stress; real-traffic failures rise
from ~1% to **26.25%** with burstiness; our pre-registered hypothesis was **rejected** and our
scheduler was **worse**; v1 was invalid and we diagnosed why; throughput is preserved at **99.98%** of
the best scheduler (*pre-registered and supported*); the starvation trap is real and guardable.

**We cannot claim:** that Risk-Aware MARL is safer — it isn't, on our measure; that its peak-risk and
EFC results show superiority — those are **exploratory**, uncorrected and partly explained by load
spreading; anything about quarantine, which never fired; that our tuning found the best settings — the
differences were **smaller than the measurement noise**; anything about physical GPUs.

---

## 7. What this demonstrates about my skills

- **Multi-agent reinforcement learning:** Q-learning agents (a **Q-table** stores "how good was this
  action in this situation") coordinating via a bidding auction, with risk and failure-cost models.
- **Simulation and benchmarking:** a 4-GPU simulator with thermal, memory and crash behaviour; a
  6,480-episode controlled comparison of nine algorithms.
- **Experiment design and statistics:** pre-registration, disjoint data splits, paired comparisons,
  bootstrap intervals, multiple-comparison awareness, a throughput floor against degenerate winners.
- **Concurrency and testing:** ordered reentrant locks and atomic guards, verified by 181 tests and a
  stress test with zero violations.
- **Full-stack and honest practice:** a FastAPI/WebSocket backend with a React dashboard — and I
  diagnosed a bug that invalidated my own earlier results, then published the negative result.

---

## 8. What I would do next

- **A v3 experiment on fresh seeds with a new pre-registration — NOT yet run.** v2 suggests we chose
  the wrong primary metric, so v3 would pre-register **peak GPU risk** and **crashes per job**, which
  reflect what actually destroys hardware. It needs new seeds — reusing v2's would mean testing on
  data I've already seen.
- **A quarantine study at a reachable threshold**, with far longer episodes, so the safety valve can
  fire and be tested rather than remaining untestable.
- **Test the load-spreading explanation directly:** build a scheduler that only spreads load, with no
  risk model, and see whether it matches Risk-Aware MARL's peak-risk advantage. If it does, the risk
  model adds nothing — and I should say so.

---

## 9. 30-second spoken version

> We built a simulated GPU cluster and a scheduler that avoids risky hardware, then pre-registered our
> main hypothesis before testing it. It was rejected — our scheduler was actually worse than a simple
> rule on the measure we picked. Along the way we found a bug in our own benchmark that made the first
> experiment meaningless, fixed it, and re-ran properly. So what we've got is a working, fully tested
> system, a real finding that failures jump from one percent to twenty-six percent under bursty
> traffic, and an honest negative result we can actually defend.
