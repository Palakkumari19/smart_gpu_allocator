"""
Phase 5 result figures. Reads only the CSVs the final test produced and
results/metrics/v2/v2_analysis.json. Output: results/graphs/v2/
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
RES = ROOT / "realdata" / "results"
OUT = ROOT / "results" / "graphs" / "v2"
OUT.mkdir(parents=True, exist_ok=True)

C = {"fcfs": "#4A5568", "round_robin": "#718096", "least_loaded": "#319795",
     "rl": "#E53E3E", "marl": "#3182CE", "risk_aware_marl": "#805AD5"}
L = {"fcfs": "FCFS", "round_robin": "Round Robin", "least_loaded": "Least Loaded",
     "rl": "Single-Agent RL", "marl": "Risk-Blind MARL", "risk_aware_marl": "Risk-Aware MARL"}
ORDER = ["fcfs", "round_robin", "least_loaded", "rl", "marl", "risk_aware_marl"]
TREAT = "risk_aware_marl"


def style(ax, title, xl, yl):
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel(xl, fontsize=10); ax.set_ylabel(yl, fontsize=10)
    ax.grid(axis="y", alpha=0.3, ls="--"); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    fig.tight_layout(); fig.savefig(OUT / name, dpi=200, bbox_inches="tight"); plt.close(fig)
    print(f"  saved {name}")


def infeasible(d, floor=0.80):
    """Schedulers below `floor` x the best mean completed_tasks (pre-registered)."""
    m = d.groupby("scheduler")["completed_tasks"].mean()
    return set(m[m < floor * m.max()].index)


def load(regime):
    return pd.read_csv(sorted(RES.glob(f"realdata_v2_runs_{regime}_*.csv"))[-1])


# ───────────────────── v2_07 the load fix: intended vs delivered
def g7(w):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    g = w.groupby(["condition", "target_rho"])["delivered_rho"].mean().reset_index()
    conds = ["v1", "v1_corrected", "v2"]
    names = {"v1": "v1 (as shipped)", "v1_corrected": "v1 + load fix", "v2": "v2 (all fixes)"}
    cols = {"v1": "#E53E3E", "v1_corrected": "#319795", "v2": "#805AD5"}
    rhos = sorted(w.target_rho.unique())
    xs = np.arange(len(rhos)); width = 0.26

    ax = axes[0]
    for i, c in enumerate(conds):
        v = [g[(g.condition == c) & (g.target_rho == r)]["delivered_rho"].iloc[0] for r in rhos]
        ax.bar(xs + (i - 1) * width, v, width, label=names[c], color=cols[c])
    ax.plot(xs, rhos, "k--o", lw=1.6, ms=5, label="intended", zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels([str(r) for r in rhos])
    ax.set_yscale("log")
    style(ax, "The load fix: delivered vs intended utilisation",
          "intended utilisation $\\rho$", "delivered $\\rho$ (log scale)")
    ax.legend(fontsize=8)

    ax = axes[1]
    t = w[w.scheduler == TREAT].groupby(["condition", "target_rho"])["contested_pct"].mean().reset_index()
    for c in conds:
        v = [t[(t.condition == c) & (t.target_rho == r)]["contested_pct"].iloc[0] for r in rhos]
        ax.plot(rhos, v, "-o", color=cols[c], lw=2, ms=7, label=names[c])
    style(ax, "Decisions where the scheduler had a real choice",
          "intended utilisation $\\rho$", "contested decisions (%)")
    ax.legend(fontsize=8)
    fig.suptitle("Why v1 could not answer the question, and v2 can",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "v2_07_load_fix_validity.png")


# ───────────────────── v2_08 primary hypothesis outcome
def g8(w):
    d = w[(w.condition == "v2") & (w.target_rho == 1.1)]
    piv = d.pivot_table(index=["window", "seed"], columns="scheduler",
                        values="risk_per_completed_task").dropna()
    base = "least_loaded"
    diff = piv[TREAT] - piv[base]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    means = [d[d.scheduler == s]["risk_per_completed_task"].mean() for s in ORDER]
    bars = ax.bar([L[s] for s in ORDER], means, color=[C[s] for s in ORDER])
    bars[ORDER.index(TREAT)].set_edgecolor("black"); bars[ORDER.index(TREAT)].set_linewidth(2)
    infeas = infeasible(d)
    for s_, b, v in zip(ORDER, bars, means):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=8)
        if s_ in infeas:
            b.set_hatch("///"); b.set_alpha(0.55)
            ax.text(b.get_x() + b.get_width() / 2, v / 2, "INFEASIBLE\n(below\nthroughput\nfloor)",
                    ha="center", va="center", fontsize=7, fontweight="bold", color="white")
    ax.tick_params(axis="x", rotation=30)
    for lb in ax.get_xticklabels():
        lb.set_ha("right")
    style(ax, "PRIMARY metric at $\\rho$=1.1 (lower is better)", "", "risk per completed task")

    ax = axes[1]
    colors = ["#319795" if v < 0 else "#E53E3E" for v in sorted(diff)]
    ax.bar(range(len(diff)), sorted(diff), color=colors)
    ax.axhline(0, color="black", lw=1)
    wins = int((diff < 0).sum())
    ax.text(0.03, 0.95, f"Risk-Aware MARL better in {wins}/{len(diff)} paired episodes\n"
                        f"(pre-registered threshold: 36/60)",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="#FFF5F5", ec="#E53E3E"))
    style(ax, f"Paired difference vs {L[base]}", "paired episode (sorted)",
          "Risk-Aware MARL $-$ baseline")
    fig.suptitle("PRIMARY pre-registered test: REJECTED", fontsize=13,
                 fontweight="bold", y=1.02, color="#C53030")
    fig.text(0.5, -0.08, "Hatched = INFEASIBLE: completes under 80% of the best scheduler's "
             "tasks, so it is excluded from baseline selection (pre-registered throughput floor).",
             ha="center", fontsize=8, style="italic", color="#C53030")
    save(fig, "v2_08_primary_hypothesis.png")


# ───────────────────── v2_09 mean vs peak: the mechanism
def g9(w):
    d = w[(w.condition == "v2") & (w.target_rho == 1.1)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, metric, title, better in (
        (axes[0], "mean_risk_over_time", "Mean risk across GPUs\n(primary metric is built on this)", "lower"),
        (axes[1], "peak_gpu_risk", "Peak risk on the hottest GPU\n(what actually crashes a GPU)", "lower"),
        (axes[2], "crash_count", "Crashes per episode", "lower"),
    ):
        means = [d[d.scheduler == s][metric].mean() for s in ORDER]
        bars = ax.bar([L[s] for s in ORDER], means, color=[C[s] for s in ORDER])
        bars[ORDER.index(TREAT)].set_edgecolor("black"); bars[ORDER.index(TREAT)].set_linewidth(2)
        infeas = infeasible(d)
        for s_, b in zip(ORDER, bars):
            if s_ in infeas:
                b.set_hatch("///"); b.set_alpha(0.55)
        means = [m if s_ not in infeas else np.inf for s_, m in zip(ORDER, means)]
        bi = int(np.argmin(means))
        means = [d[d.scheduler == s][metric].mean() for s in ORDER]
        ax.text(bars[bi].get_x() + bars[bi].get_width() / 2, means[bi],
                " best ", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#2F855A")
        ax.tick_params(axis="x", rotation=35)
        for lb in ax.get_xticklabels():
            lb.set_ha("right"); lb.set_fontsize(8)
        style(ax, title, "", f"{metric} ({better} is better)")
    fig.suptitle("Risk-Aware MARL wins on peak risk and loses on mean risk  —  $\\rho$=1.1, v2",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.text(0.5, -0.06, "Hatched = INFEASIBLE: completes under 80% of the best scheduler's "
             "tasks, so it is excluded from baseline selection (pre-registered throughput floor).",
             ha="center", fontsize=8, style="italic", color="#C53030")
    save(fig, "v2_09_mean_vs_peak.png")


# ───────────────────── v2_10 Pareto
def g10(w):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    an = json.load(open(MET / "v2_analysis.json"))
    for ax, rho in zip(axes, sorted(w.target_rho.unique())):
        d = w[(w.condition == "v2") & (w.target_rho == rho)]
        pts = {}
        for s in ORDER:
            x = d[d.scheduler == s]["completed_tasks"].mean()
            y = d[d.scheduler == s]["mean_risk_over_time"].mean()
            pts[s] = (x, y)
            ax.scatter(x, y, s=210 if s == TREAT else 130, color=C[s],
                       edgecolor="black" if s == TREAT else "none",
                       linewidth=2 if s == TREAT else 0, zorder=5 if s == TREAT else 3)
        # The dispatchers cluster tightly, so labels are fanned out with leader
        # lines instead of sitting on top of one another.
        fan = {"risk_aware_marl": (34, 26), "marl": (34, 6), "round_robin": (34, -14),
               "least_loaded": (-18, -26), "fcfs": (-18, -46), "rl": (16, 8)}
        for s, (x, y) in pts.items():
            dx, dy = fan[s]
            ax.annotate(L[s], (x, y), textcoords="offset points", xytext=(dx, dy),
                        fontsize=8, color=C[s], fontweight="bold" if s == TREAT else "normal",
                        ha="left" if dx > 0 else "right",
                        arrowprops=dict(arrowstyle="-", color=C[s], lw=0.7,
                                        shrinkA=0, shrinkB=3, alpha=0.7))
        style(ax, f"$\\rho$ = {rho}", "completed tasks (higher better)",
              "mean risk (lower better)")
        ax.margins(0.40)
    fig.suptitle("Throughput vs risk, v2 condition  —  bottom-right is best",
                 fontsize=13, fontweight="bold", y=1.03)
    save(fig, "v2_10_pareto.png")


# ───────────────────── v2_11 v1 vs v2 side by side
def g11(w):
    t = w[w.scheduler == TREAT]
    mets = [("completed_tasks", "completed tasks"), ("mean_risk_over_time", "mean risk"),
            ("peak_gpu_risk", "peak GPU risk"), ("crash_count", "crashes")]
    conds = ["v1", "v1_corrected", "v2"]
    names = {"v1": "v1\n(as shipped)", "v1_corrected": "v1 + load fix", "v2": "v2\n(all fixes)"}
    cols = {"v1": "#E53E3E", "v1_corrected": "#319795", "v2": "#805AD5"}
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    for ax, (m, lab) in zip(axes, mets):
        v = [t[t.condition == c][m].mean() for c in conds]
        bars = ax.bar([names[c] for c in conds], v, color=[cols[c] for c in conds])
        for b, x in zip(bars, v):
            ax.text(b.get_x() + b.get_width() / 2, x, f"{x:.3g}", ha="center",
                    va="bottom", fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        style(ax, lab, "", lab)
        ax.margins(y=0.18)
    fig.suptitle("Risk-Aware MARL under each condition (all loads pooled, workload regime)",
                 fontsize=13, fontweight="bold", y=1.04)
    fig.text(0.5, -0.10,
             "v1 completes more tasks only because its broken load scale offered ~7x more work than "
             "intended; it is not a better result.\nNote v1+load fix and v2 are nearly identical: "
             "almost the entire change comes from the load fix, not from the risk-bucket or training changes.",
             ha="center", fontsize=8.5, style="italic", color="#4A5568")
    save(fig, "v2_11_v1_vs_v2.png")


# ───────────────────── v2_12 quarantine
def g12(w, o):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, df, rg in ((axes[0], w, "workload only"), (axes[1], o, "hazard overlay")):
        d = df[df.condition == "v2"]
        thrs = [0.85, 0.6]
        xs = np.arange(len(ORDER)); width = 0.38
        for i, thr in enumerate(thrs):
            col = f"quarantine_triggered_at_{thr}"
            v = [100 * d[d.scheduler == s][col].mean() for s in ORDER]
            ax.bar(xs + (i - 0.5) * width, v, width,
                   label=f"threshold {thr}" + (" (shipped)" if thr == 0.85 else " (exploratory)"),
                   color="#4A5568" if thr == 0.85 else "#805AD5")
            if thr == 0.85:
                for x, vv in zip(xs + (i - 0.5) * width, v):
                    ax.text(x, 1.5, "0", ha="center", va="bottom", fontsize=8,
                            fontweight="bold", color="#4A5568")
        ax.axhline(30, color="#E53E3E", ls="--", lw=1.4)
        ax.text(len(ORDER) - 0.45, 31.5, "30% testability floor", color="#E53E3E",
                fontsize=8, ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))
        n0 = int(d[f"quarantine_triggered_at_0.85"].sum())
        ax.text(0.02, 0.86, f"shipped threshold 0.85 fired in\n{n0} of {len(d)} episodes",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round", fc="#F7FAFC", ec="#4A5568"))
        ax.set_xticks(xs); ax.set_xticklabels([L[s] for s in ORDER], rotation=35, ha="right", fontsize=8)
        ax.set_ylim(0, 105)
        style(ax, f"Quarantine trigger rate — {rg}", "", "episodes where quarantine fired (%)")
        ax.legend(fontsize=8)
    fig.suptitle("The shipped quarantine threshold never fires  —  H4 remains untestable",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "v2_12_quarantine.png")


# ───────────────────── v2_13 workload vs overlay
def g13(w, o):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, metric, lab in ((axes[0], "crash_count", "crashes per episode"),
                            (axes[1], "mean_risk_over_time", "mean risk")):
        xs = np.arange(len(ORDER)); width = 0.38
        for i, (df, rg, cc) in enumerate(((w, "workload only", "#319795"),
                                          (o, "hazard overlay", "#DD6B20"))):
            d = df[(df.condition == "v2") & (df.target_rho == 1.1)]
            v = [d[d.scheduler == s][metric].mean() for s in ORDER]
            ax.bar(xs + (i - 0.5) * width, v, width, label=rg, color=cc)
        ax.set_xticks(xs); ax.set_xticklabels([L[s] for s in ORDER], rotation=35, ha="right", fontsize=8)
        style(ax, lab, "", lab)
        ax.legend(fontsize=8)
    fig.suptitle("Adding the real-data hazard overlay raises risk for every scheduler  ($\\rho$=1.1, v2)",
                 fontsize=13, fontweight="bold", y=1.02)
    save(fig, "v2_13_overlay_effect.png")


def main():
    w, o = load("workload"), load("overlay")
    print("building v2 result figures ...")
    g7(w); g8(w); g9(w); g10(w); g11(w); g12(w, o); g13(w, o)
    print("done")


if __name__ == "__main__":
    main()
