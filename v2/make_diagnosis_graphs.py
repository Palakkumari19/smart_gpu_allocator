"""
Phase 1 diagnosis figures. Reads only files produced by v2/diagnose.py.
Output: results/graphs/v2/
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "metrics" / "v2"
OUT = ROOT / "results" / "graphs" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

C = {"fcfs": "#4A5568", "round_robin": "#718096", "least_loaded": "#319795",
     "rl": "#E53E3E", "marl": "#3182CE", "risk_aware_marl": "#805AD5"}
L = {"fcfs": "FCFS", "round_robin": "Round Robin", "least_loaded": "Least Loaded",
     "rl": "Single-Agent RL", "marl": "Risk-Blind MARL", "risk_aware_marl": "Risk-Aware MARL"}


def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
    ax.grid(axis="y", alpha=0.3, ls="--"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=200); plt.close(fig)
    print(f"  saved {name}")


# ───────────────────────────── v2_01 the binding constraint
def g1_choice_availability():
    s = json.load(open(MET / "d1_divergence_summary.json"))
    hist = {int(k): v for k, v in s["eligible_gpu_count_histogram"].items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ks = sorted(hist)
    vals = [hist[k] for k in ks]
    total = sum(vals)
    bars = axes[0].bar([str(k) for k in ks], vals,
                       color=["#C53030" if k <= 1 else "#38A169" for k in ks], alpha=0.9)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, b.get_height(),
                     f"{v:,}\n({100*v/total:.1f}%)", ha="center", va="bottom", fontsize=9)
    style(axes[0], "How many GPUs could legally take the task?\n(red = the scheduler had no real choice)",
          "Number of eligible GPUs", "Number of scheduling decisions")
    axes[0].set_ylim(0, max(vals) * 1.22)

    d = s["decisions_diverged"]; nd = s["decisions_with_choice"] - d
    axes[1].bar(["Same GPU\nchosen", "Different GPU\nchosen"], [nd, d],
                color=["#A0AEC0", "#805AD5"], alpha=0.9)
    for i, v in enumerate([nd, d]):
        axes[1].text(i, v, f"{v}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    style(axes[1], f"When there WAS a choice ({s['decisions_with_choice']} times),\n"
                   f"risk penalties changed the pick {s['divergence_pct']:.1f}% of the time",
          "", "Decisions")
    fig.suptitle("Diagnosis 1 — the risk penalty works, but it is almost never consulted",
                 fontsize=13.5, fontweight="bold")
    save(fig, "v2_01_choice_availability.png")


# ───────────────────────────── v2_02 bid term magnitudes
def g2_bid_magnitudes():
    df = pd.read_csv(MET / "d1_bid_components.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    terms = ["q_contribution", "perf_score", "risk_penalty", "efc_penalty"]
    names = ["Q-value\n(learned)", "Performance\n(hand-written)", "Risk penalty", "EFC penalty"]
    cols = ["#3182CE", "#38A169", "#C53030", "#DD6B20"]
    data = [df[t].to_numpy() for t in terms]
    bp = axes[0].boxplot(data, labels=names, patch_artist=True, showfliers=False, widths=0.6)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c); patch.set_alpha(0.65)
    for i, t in enumerate(terms):
        axes[0].text(i + 1, df[t].max() * 1.02 + 0.005, f"mean\n{df[t].mean():.4f}",
                     ha="center", fontsize=8)
    style(axes[0], "Size of each term in the bid", "", "Contribution to the bid score")

    rew = (df.q_contribution + df.perf_score)
    pen = (df.risk_penalty + df.efc_penalty)
    axes[1].bar(["Reward side\n(Q + performance)", "Penalty side\n(risk + EFC)"],
                [rew.mean(), pen.mean()], color=["#38A169", "#C53030"], alpha=0.9)
    for i, v in enumerate([rew.mean(), pen.mean()]):
        axes[1].text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
    style(axes[1], f"Penalties are {100*pen.mean()/rew.mean():.0f}% the size of the reward side",
          "", "Mean contribution")
    fig.suptitle("Diagnosis 1b — the learned Q-value is the smallest term; 'performance' dominates",
                 fontsize=13.5, fontweight="bold")
    save(fig, "v2_02_bid_term_magnitudes.png")


# ───────────────────────────── v2_03 ablation peak vs mean
def g3_ablation_peak_mean():
    df = pd.read_csv(MET / "d2_ablation_peak_mean.csv")
    order = ["full_risk_aware", "no_risk_penalty", "no_efc_penalty", "risk_blind"]
    nice = ["Full\nRisk-Aware", "No risk\npenalty", "No EFC\npenalty", "Risk-blind"]
    g = df.groupby("variant")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, col, title in zip(
            axes,
            ["mean_risk_over_time", "peak_gpu_risk", "completed_tasks"],
            ["MEAN risk across the run\n(lower better)", "PEAK risk on any one GPU\n(lower better)",
             "Tasks completed\n(higher better)"]):
        m = [g[col].mean()[v] for v in order]
        e = [g[col].std()[v] for v in order]
        bars = ax.bar(nice, m, yerr=e, capsize=4,
                      color=["#805AD5", "#ED8936", "#38A169", "#3182CE"], alpha=0.9)
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width()/2, b.get_height(), f"{v:.4f}",
                    ha="center", va="bottom", fontsize=8.5)
        style(ax, title, "", "")
    fig.suptitle("Diagnosis 2 — do risk penalties lower the PEAK while raising the MEAN?",
                 fontsize=13.5, fontweight="bold")
    save(fig, "v2_03_ablation_peak_vs_mean.png")


# ───────────────────────────── v2_04 quarantine reachability
def g4_quarantine():
    df = pd.read_csv(MET / "d3_quarantine_reachability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    g = df.groupby(["steps", "load"])["max_risk_observed"].mean().reset_index()
    loads = sorted(g.load.unique()); steps = sorted(g.steps.unique())
    x = np.arange(len(steps)); w = 0.8 / max(1, len(loads))
    for i, ld in enumerate(loads):
        sub = g[g.load == ld].set_index("steps")
        axes[0].bar(x + (i - (len(loads)-1)/2) * w, [sub.max_risk_observed.get(s, 0) for s in steps],
                    w, label=f"load ρ={ld}", alpha=0.9)
    axes[0].axhline(0.85, color="#C53030", ls="--", lw=2)
    axes[0].text(len(steps)-0.5, 0.86, "quarantine threshold 0.85",
                 ha="right", color="#C53030", fontsize=9, fontweight="bold")
    axes[0].set_xticks(x); axes[0].set_xticklabels([f"{s} steps" for s in steps])
    axes[0].set_ylim(0, 1.0); axes[0].legend(frameon=False, fontsize=9)
    style(axes[0], "Highest risk any GPU ever reached", "Episode length", "Max risk score")

    q = df.groupby("steps")["ever_quarantined"].mean().mul(100)
    axes[1].bar([f"{s} steps" for s in q.index], q.values, color="#C53030", alpha=0.85)
    for i, v in enumerate(q.values):
        axes[1].text(i, v, f"{v:.0f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].set_ylim(0, 100)
    style(axes[1], "Episodes in which quarantine ever triggered", "Episode length", "% of episodes")
    fig.suptitle("Diagnosis 3 — the quarantine threshold is out of reach at v1's episode length",
                 fontsize=13.5, fontweight="bold")
    save(fig, "v2_04_quarantine_reachability.png")


# ───────────────────────────── v2_05 starvation
def g5_starvation():
    df = pd.read_csv(MET / "d4_starvation.csv")
    tr = pd.read_csv(MET / "d4_temperature_traces.csv")
    g = df.groupby("scheduler")
    order = [s for s in ["rl", "marl", "risk_aware_marl", "least_loaded"] if s in g.groups]

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    x = np.arange(len(order))
    off = [g["tasks_offered"].mean()[s] for s in order]
    asg = [g["successful_assignments"].mean()[s] for s in order]
    cmp_ = [g["completed_tasks"].mean()[s] for s in order]
    axes[0].bar(x - 0.22, asg, 0.42, label="Admitted to a GPU", color="#3182CE", alpha=0.9)
    axes[0].bar(x + 0.22, cmp_, 0.42, label="Completed", color="#38A169", alpha=0.9)
    axes[0].set_xticks(x); axes[0].set_xticklabels([L[s] for s in order], rotation=20, ha="right", fontsize=8.5)
    axes[0].legend(frameon=False, fontsize=9)
    style(axes[0], f"Of ~{off[0]:,.0f} tasks offered per episode", "", "Tasks")

    for s in order:
        sub = tr[tr.scheduler == s]
        if len(sub):
            axes[1].plot(sub.step, sub.avg_temperature, lw=2, color=C[s], label=L[s])
    axes[1].axhline(85, color="#C53030", ls="--", lw=1.5)
    axes[1].text(2, 86, "crash threshold 85 °C", color="#C53030", fontsize=8.5, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=8)
    style(axes[1], "Cluster temperature over one episode", "Simulation step", "Mean GPU temperature (°C)")

    r = [g["avg_risk"].mean()[s] for s in order]
    bars = axes[2].bar([L[s] for s in order], r, color=[C[s] for s in order], alpha=0.9)
    for b, v in zip(bars, r):
        axes[2].text(b.get_x()+b.get_width()/2, b.get_height(), f"{v:.3f}",
                     ha="center", va="bottom", fontsize=9)
    axes[2].set_xticklabels([L[s] for s in order], rotation=20, ha="right", fontsize=8.5)
    style(axes[2], "Resulting average risk", "", "Risk score")
    fig.suptitle("Diagnosis 4 — single-agent RL stays cool and low-risk by admitting almost nothing",
                 fontsize=13.5, fontweight="bold")
    save(fig, "v2_05_starvation_mechanism.png")


# ───────────────────────────── v2_06 training adequacy
def g6_training():
    c = pd.read_csv(MET / "d5_training_curve.csv")
    info = json.load(open(MET / "d5_training_adequacy.json"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    axes[0].plot(c.step, c.q_table_size, "o-", color="#38A169", lw=2, ms=3)
    style(axes[0], f"Q-table growth (ends at {info['final_q_table_size']} states)",
          "Training step", "Distinct states learned")

    axes[1].plot(c.step, c.avg_epsilon, "o-", color="#C53030", lw=2, ms=3)
    axes[1].set_ylim(0, 1.05)
    axes[1].axhline(0.05, color="#2D3748", ls="--", lw=1.2)
    axes[1].text(2, 0.08, "intended floor ε=0.05", fontsize=8.5)
    style(axes[1], f"Exploration rate (ends at ε={info['final_avg_epsilon']:.3f})",
          "Training step", "ε — fraction of random actions")

    total_states = 4 * (3 ** 15)
    visited = info["final_q_table_size"]
    axes[2].bar(["States visited\nin training", "States the\nrepresentation allows"],
                [visited, total_states], color=["#3182CE", "#A0AEC0"], alpha=0.9)
    axes[2].set_yscale("log")
    axes[2].text(0, visited, f"{visited:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    axes[2].text(1, total_states, f"{total_states:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    style(axes[2], f"Coverage = {100*visited/total_states:.7f}%", "", "Number of states (log scale)")
    fig.suptitle("Diagnosis 5 — training is far too short: ε barely decays and coverage is ~0",
                 fontsize=13.5, fontweight="bold")
    save(fig, "v2_06_training_adequacy.png")


if __name__ == "__main__":
    print("Generating diagnosis figures ...")
    for fn in (g1_choice_availability, g2_bid_magnitudes, g3_ablation_peak_mean,
               g4_quarantine, g5_starvation, g6_training):
        try:
            fn()
        except Exception as e:
            print(f"  [FAILED] {fn.__name__}: {type(e).__name__}: {e}")
    print("Done.")
