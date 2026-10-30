# 04 — Objective and Approach

---

## 1. The problem, in plain words

Modern AI runs on GPUs, and GPUs are expensive and scarce. In a cloud data centre, thousands of jobs
arrive every minute — someone asking a chatbot a question, someone encoding a video — and something has
to decide **which GPU each job runs on**.

Most real systems decide this with very simple rules. "Give it to whoever is free." "Take turns."
"Give it to the emptiest one." These rules are fast and predictable, but they are **blind to danger**.
They do not know that GPU 3 has crashed twice in the last ten minutes, or that GPU 1 is running hot and
is about to shut down. So they keep feeding jobs to a machine that is about to fail.

When a GPU fails mid-job, the cost is not just the lost GPU. Everything running on it dies, those jobs
must restart from the beginning, the GPU goes into a cooldown period where it accepts nothing, and the
remaining GPUs must absorb the extra load — which makes *them* more likely to fail too.

**Why it matters:** GPU time is one of the largest costs in AI infrastructure. A scheduler that avoids
even a modest number of crashes saves real money and gives users more reliable service. And as clusters
grow, hand-written rules become harder and harder to tune by hand — which is exactly the situation
where a system that learns from experience should help.

---

## 2. The objective

> Build a GPU cluster scheduler in which **each GPU has its own reinforcement-learning agent**, and in
> which those agents bid for incoming jobs using not only expected performance but also **how likely
> that GPU is to crash** and **how expensive that crash would be**.
>
> Then test honestly — against simple traditional schedulers and against a risk-blind version of the
> same learning system — whether this crash-awareness actually produces fewer crashes and lower risk
> **without destroying throughput**.
>
> Finally, validate the result on **real recorded production data** rather than only on invented
> workloads, with the hypotheses written down in advance so the answer cannot be adjusted afterwards.

---

## 3. The approach, step by step

### Step 1 — Build a GPU cluster that can actually fail

A faithful comparison needs something to compare *on*. Four simulated GPUs were built with 8 GB of
memory each, and with the failure behaviour that matters: temperature that rises as jobs pile up, a
crash above 85 °C, a cooldown period afterwards, memory that fragments over time, and job preemption.
Without crashes in the simulation, a crash-aware scheduler has nothing to be aware of.

### Step 2 — Implement the simple baselines first

FCFS, Round Robin and Least Loaded were implemented before anything clever. They are the honest
yardstick. If the learning system cannot beat these, it is not worth the complexity.

### Step 3 — Build a risk score

`risk_engine.py` turns each GPU's live condition into one number between 0 and 1. Eight signals are
blended, weighted so that **history counts more than instantaneous readings**: past anomalies (0.25) and
past task failures (0.20) outweigh current temperature (0.15) and memory pressure (0.10). A GPU above
0.85 is quarantined — taken out of service entirely.

### Step 4 — Turn risk into a forward-looking cost

A risk score says "this GPU looks dangerous". It does not say how much you would *lose*. So the
Expected Failure Cost was added:

```
EFC = chance of failure  ×  how damaging the failure would be  ×  how long recovery takes
```

This is computed for a specific job on a specific GPU *before* placing it. A small job on a risky GPU
may be fine; a large, long, high-priority job on the same GPU is not.

### Step 5 — Give every GPU its own learning agent

Instead of one brain for the cluster, each GPU gets a Q-learning agent. Each agent observes its own
situation as a 16-element discrete state (memory pressure, load, temperature, risk, failure rate,
status, job size, cluster contention) and learns how good it is to accept jobs in that situation.

### Step 6 — Make them bid

When a job arrives, every agent submits a bid:

```
bid = Q-value estimate + free headroom − (risk penalty) − (EFC penalty)
```

The highest bid wins. Two situations bid **−∞** and can never win: the GPU is quarantined, or the job
does not fit. The reward is clamped to **[−20, +12]** so a single disaster cannot erase everything the
agent has learned.

### Step 7 — Measure properly, not anecdotally

Running one simulation and reporting the winner proves nothing — the result could be luck. So:
every scheduler sees the **identical** job stream with the **identical** random seed, every experiment
is repeated across multiple seeds, and the comparison is reported with **bootstrap 95 % confidence
intervals** and **paired win rates** rather than single numbers.

### Step 8 — Replay real production data

Invented workloads can accidentally be shaped to flatter the method. So two genuine public datasets
were brought in:

- **BurstGPT** — 1,429,737 real AI inference requests across 61 days (EuroSys '24, CC-BY-4.0).
- **UCI video transcoding** — 168,286 real video-encoding measurements (CC-BY-4.0).

Real measurements are converted to simulator job sizes by **percentile mapping**, which preserves the
shape of the real distribution. The trace is split so the learning agents **train on days 0–30 and are
tested on days 31–61**, with exploration switched off during testing.

### Step 9 — Measure how often reality actually fails

From the real data, the project measured how the real-world failure rate changes with how bursty
traffic is. The answer is striking and is a genuine empirical finding: when arrivals are smooth, about
**1 %** of requests fail; when arrivals become extremely bursty, **26.25 %** fail. That measured curve
is then applied to the simulation, so failures happen at realistic rates instead of invented ones.

### Step 10 — Write down the hypotheses *before* looking

`realdata/PREREGISTRATION.md` states five hypotheses and the exact rule for calling one supported:
the 95 % confidence interval must exclude zero in the favourable direction **and** the win rate against
the strongest baseline must be at least 60 %.

This step is what makes the project's conclusion trustworthy — and it is why the honest answer,
reported in `docs/05_results_and_findings.md`, is that **the main hypotheses were not supported**.

---

## 4. What success was defined as

| Metric | Direction | Meaning |
|---|---|---|
| `avg_risk_exposure` | lower is better | How dangerous the cluster was, averaged over the run |
| `terminal_efc` | lower is better | Expected failure cost still outstanding at the end |
| `crash_count` | lower is better | How many GPU crashes happened |
| `quarantine_time` | lower is better | How long GPUs were out of service |
| `completed_tasks` | **higher** is better | How much work actually got done |

The last one is the guard rail. Without it, a scheduler could score perfectly on safety by simply
refusing to run anything — which, as the results show, is very nearly what the single-agent RL
scheduler learned to do.
