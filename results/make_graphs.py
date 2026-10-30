"""
Graph generator for the documentation pass (30 Oct 2026).

This script is NOT part of the original project. It reads result files that the project
itself produced during this run and draws summary charts into results/graphs/.
It invents no numbers: every value plotted is read from a CSV or JSON on disk.
"""

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
GRAPHS = ROOT / "results" / "graphs"
METRICS = ROOT / "results" / "metrics"
GRAPHS.mkdir(parents=True, exist_ok=True)
METRICS.mkdir(parents=True, exist_ok=True)

COLORS = {
    "fcfs": "#4A5568",
    "round_robin": "#718096",
    "least_loaded": "#319795",
    "rl": "#E53E3E",
    "marl": "#3182CE",
    "risk_aware_marl": "#805AD5",
}
LABELS = {
    "fcfs": "FCFS",
    "round_robin": "Round Robin",
    "least_loaded": "Least Loaded",
    "rl": "Single-Agent RL",
    "marl": "Risk-Blind MARL",
    "risk_aware_marl": "Risk-Aware MARL",
}
ORDER = ["fcfs", "round_robin", "least_loaded", "rl", "marl", "risk_aware_marl"]


def latest(pattern, folder):
    hits = sorted(folder.glob(pattern))
    return hits[-1] if hits else None


def style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def save(fig, name):
    path = GRAPHS / name
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"  saved {path.name}")


# ---------------------------------------------------------------- graph 05
def graph_six_scheduler_benchmark():
    """6-scheduler synthetic benchmark from benchmarking/run_experiments.py."""
    folder = ROOT / "benchmarking" / "results"
    csv = latest("experiment_bench_*_runs.csv", folder)
    if csv is None:
        print("  [skip] no benchmark runs csv")
        return
    df = pd.read_csv(csv)
    g = df.groupby("scheduler_id")

    metrics = [
        ("throughput_tasks_per_sec", "Throughput (tasks/sec)", "higher is better"),
        ("success_rate_pct", "Success rate (%)", "higher is better"),
        ("avg_risk_exposure", "Average risk exposure", "lower is better"),
        ("avg_efc", "Average Expected Failure Cost", "lower is better"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (col, title, direction) in zip(axes.ravel(), metrics):
        means = [g[col].mean().get(s, np.nan) for s in ORDER]
        stds = [g[col].std().get(s, 0.0) for s in ORDER]
        bars = ax.bar(range(len(ORDER)), means, yerr=stds, capsize=4,
                      color=[COLORS[s] for s in ORDER], alpha=0.9)
        ax.set_xticks(range(len(ORDER)))
        ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=25, ha="right", fontsize=9)
        style(ax, f"{title}\n({direction})", "", title)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{m:,.3g}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Six-scheduler comparison on the synthetic benchmark (3 seeds, mean ± SD)",
                 fontsize=14, fontweight="bold")
    save(fig, "05_six_scheduler_synthetic_benchmark.png")

    out = g[[m[0] for m in metrics] + ["completed_tasks", "jains_fairness_index",
                                       "avg_gpu_utilization_pct", "capacity_violations"]].agg(["mean", "std"])
    out.to_csv(METRICS / "05_six_scheduler_synthetic_benchmark.csv")


# ---------------------------------------------------------------- graph 06
def graph_ablation():
    """Ablation study: which part of the risk-aware design does the work?"""
    folder = ROOT / "benchmarking" / "results"
    csv = latest("experiment_ablation_*_runs.csv", folder)
    if csv is None:
        print("  [skip] no ablation csv")
        return
    df = pd.read_csv(csv)
    g = df.groupby("scheduler_name")
    names = list(g.groups.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (col, title) in zip(axes, [("avg_risk_exposure", "Average risk exposure\n(lower is better)"),
                                       ("throughput_tasks_per_sec", "Throughput (tasks/sec)\n(higher is better)")]):
        means = [g[col].mean()[n] for n in names]
        stds = [g[col].std()[n] for n in names]
        bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                      color=["#805AD5", "#ED8936", "#38A169", "#3182CE"][:len(names)], alpha=0.9)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels([n.replace(" (", "\n(") for n in names], rotation=0, fontsize=8)
        style(ax, title, "", "")
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{m:,.4g}",
                    ha="center", va="bottom", fontsize=8)
    fig.suptitle("Ablation study: turning off each risk component (3 seeds, mean ± SD)",
                 fontsize=14, fontweight="bold")
    save(fig, "06_ablation_risk_components.png")
    g[["avg_risk_exposure", "avg_efc", "throughput_tasks_per_sec",
       "success_rate_pct"]].agg(["mean", "std"]).to_csv(METRICS / "06_ablation_risk_components.csv")


# ---------------------------------------------------------------- graph 07
def graph_seven_scenarios():
    """7-scenario risk-aware benchmark, parsed from its console log."""
    log = ROOT / "logs" / "run_benchmark_risk_aware_marl.log"
    if not log.exists():
        print("  [skip] no risk-aware benchmark log")
        return
    text = log.read_text()

    scenarios, current, rows = [], None, []
    for line in text.splitlines():
        m = re.search(r"Summary Statistics \(Mean ± Std Dev\) for (.+):", line)
        if m:
            current = m.group(1).strip()
            scenarios.append(current)
            continue
        if current:
            m2 = re.match(r"\s*(marl|least_loaded|round_robin|fcfs)\s*\|\s*([\d.]+)% ± ([\d.]+)%", line)
            if m2:
                rows.append({"scenario": current, "scheduler": m2.group(1),
                             "ep_completion_pct": float(m2.group(2)),
                             "ep_completion_sd": float(m2.group(3))})
    if not rows:
        print("  [skip] could not parse scenario table")
        return
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "07_seven_scenario_completion.csv", index=False)

    scen = [s for s in scenarios if s in set(df.scenario)]
    sched = ["marl", "least_loaded", "round_robin", "fcfs"]
    colour = {"marl": "#3182CE", "least_loaded": "#319795",
              "round_robin": "#718096", "fcfs": "#4A5568"}
    name = {"marl": "Risk-Aware MARL", "least_loaded": "Least Loaded",
            "round_robin": "Round Robin", "fcfs": "FCFS"}

    fig, ax = plt.subplots(figsize=(14, 6))
    width = 0.2
    x = np.arange(len(scen))
    for i, s in enumerate(sched):
        sub = df[df.scheduler == s].set_index("scenario")
        vals = [sub.ep_completion_pct.get(c, np.nan) for c in scen]
        errs = [sub.ep_completion_sd.get(c, 0.0) for c in scen]
        ax.bar(x + (i - 1.5) * width, vals, width, yerr=errs, capsize=3,
               label=name[s], color=colour[s], alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels([c.split(". ", 1)[-1].replace(" ", "\n", 1) for c in scen], fontsize=8)
    style(ax, "Episode completion rate across the seven stress scenarios (3 seeds, mean ± SD)",
          "", "Tasks completed within the episode (%)")
    ax.legend(frameon=False, fontsize=9)
    save(fig, "07_seven_scenario_completion_rate.png")


# ---------------------------------------------------------------- graph 08
def graph_marl_learning_curve():
    """MARL learning progress, parsed from marl_vs_traditional console log."""
    log = ROOT / "logs" / "run_marl_vs_traditional.log"
    if not log.exists():
        print("  [skip] no marl_vs_traditional log")
        return
    rows = []
    for line in log.read_text().splitlines():
        m = re.match(r"\s*(\d+)\s*\|\s*([\d.]+)%\s*\|\s*(-?[\d.]+)\s*\|\s*(\d+)\s*$", line)
        if m:
            rows.append({"step": int(m.group(1)), "success_rate_pct": float(m.group(2)),
                         "total_reward": float(m.group(3)), "q_table_size": int(m.group(4))})
    if not rows:
        print("  [skip] could not parse learning curve")
        return
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "08_marl_learning_curve.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].plot(df.step, df.success_rate_pct, "o-", color="#3182CE", lw=2)
    style(axes[0], "Success rate during training", "Training step", "Success rate (%)")
    axes[1].plot(df.step, df.total_reward, "o-", color="#E53E3E", lw=2)
    style(axes[1], "Cumulative reward during training", "Training step", "Total reward")
    axes[2].plot(df.step, df.q_table_size, "o-", color="#38A169", lw=2)
    style(axes[2], "Q-table growth", "Training step", "Number of learned states")
    fig.suptitle("Untrained MARL learning progress — success falls and reward diverges",
                 fontsize=14, fontweight="bold")
    save(fig, "08_marl_learning_curve.png")


# ---------------------------------------------------------------- graph 09
def graph_realdata_summary():
    """Risk vs throughput for the real-data benchmark, both regimes."""
    folder = ROOT / "realdata" / "results"
    w = latest("realdata_runs_workload_*.csv", folder)
    o = latest("realdata_runs_overlay_*.csv", folder)
    if w is None or o is None:
        print("  [skip] real-data run CSVs not found")
        return
    dfw, dfo = pd.read_csv(w), pd.read_csv(o)
    print(f"  using {w.name} and {o.name}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, df, title in [(axes[0], dfw, "Workload-only replay"),
                          (axes[1], dfo, "Workload + real hazard overlay")]:
        g = df.groupby("scheduler")
        # Fan the labels out: the five active schedulers sit almost on top of
        # one another, so each gets its own offset and a leader line.
        offsets = {"fcfs": (-88, 18), "round_robin": (-34, -50), "least_loaded": (-86, -22),
                   "rl": (62, 14), "marl": (14, 46), "risk_aware_marl": (76, -20)}
        for s in ORDER:
            if s not in g.groups:
                continue
            x = g["avg_risk_exposure"].mean()[s]
            y = g["completed_tasks"].mean()[s]
            ax.scatter(x, y, s=230, color=COLORS[s], alpha=0.9, edgecolors="white",
                       linewidths=1.6, zorder=4)
            ax.annotate(f"{LABELS[s]}\n({x:.3f}, {y:.1f})", (x, y),
                        textcoords="offset points", xytext=offsets[s],
                        ha="center", fontsize=7.5, zorder=5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                  ec=COLORS[s], lw=1.0, alpha=0.95),
                        arrowprops=dict(arrowstyle="-", color=COLORS[s], lw=0.9,
                                        shrinkA=0, shrinkB=4))
        ax.margins(x=0.40, y=0.26)
        style(ax, title, "← Average risk exposure (lower is better)",
              "Tasks completed (higher is better) →")
    fig.suptitle("The starvation trap: Single-Agent RL has the lowest risk only because it barely runs anything",
                 fontsize=13, fontweight="bold")
    save(fig, "09_realdata_risk_vs_throughput.png")

    rows = []
    for regime, df in [("workload_only", dfw), ("hazard_overlay", dfo)]:
        g = df.groupby("scheduler")
        for s in ORDER:
            if s not in g.groups:
                continue
            rows.append({
                "regime": regime, "scheduler": s,
                "avg_risk_exposure_mean": g["avg_risk_exposure"].mean()[s],
                "avg_risk_exposure_sd": g["avg_risk_exposure"].std()[s],
                "terminal_efc_mean": g["terminal_efc"].mean()[s],
                "terminal_efc_sd": g["terminal_efc"].std()[s],
                "crash_count_mean": g["crash_count"].mean()[s],
                "crash_count_sd": g["crash_count"].std()[s],
                "quarantine_time_mean": g["quarantine_time"].mean()[s],
                "completed_tasks_mean": g["completed_tasks"].mean()[s],
                "completed_tasks_sd": g["completed_tasks"].std()[s],
            })
    pd.DataFrame(rows).to_csv(METRICS / "09_realdata_scheduler_summary.csv", index=False)


# ---------------------------------------------------------------- graph 10
def graph_realdata_four_metrics():
    folder = ROOT / "realdata" / "results"
    w = latest("realdata_runs_workload_*.csv", folder)
    o = latest("realdata_runs_overlay_*.csv", folder)
    if w is None or o is None:
        print("  [skip] real-data run CSVs not found")
        return
    dfw, dfo = pd.read_csv(w), pd.read_csv(o)

    metrics = [("avg_risk_exposure", "Average risk exposure", "lower is better"),
               ("terminal_efc", "Terminal Expected Failure Cost", "lower is better"),
               ("crash_count", "Crashes per episode", "lower is better"),
               ("completed_tasks", "Tasks completed", "higher is better")]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    x = np.arange(len(ORDER))
    for ax, (col, title, direction) in zip(axes.ravel(), metrics):
        gw, go = dfw.groupby("scheduler"), dfo.groupby("scheduler")
        mw = [gw[col].mean().get(s, np.nan) for s in ORDER]
        mo = [go[col].mean().get(s, np.nan) for s in ORDER]
        sw = [gw[col].std().get(s, 0.0) for s in ORDER]
        so = [go[col].std().get(s, 0.0) for s in ORDER]
        ax.bar(x - 0.2, mw, 0.4, yerr=sw, capsize=3, label="Workload only",
               color="#90CDF4", edgecolor="#2B6CB0")
        ax.bar(x + 0.2, mo, 0.4, yerr=so, capsize=3, label="With real hazard overlay",
               color="#FEB2B2", edgecolor="#C53030")
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[s] for s in ORDER], rotation=25, ha="right", fontsize=8)
        style(ax, f"{title}  ({direction})", "", title)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Real-data benchmark: all four pre-registered metrics, both regimes (30 conditions × 5 seeds)",
                 fontsize=14, fontweight="bold")
    save(fig, "10_realdata_four_metrics.png")


# ---------------------------------------------------------------- graph 11
def graph_hypothesis_verdicts():
    folder = ROOT / "realdata" / "results"
    sw = latest("realdata_summary_workload_*.json", folder)
    so = latest("realdata_summary_overlay_*.json", folder)
    if sw is None or so is None:
        print("  [skip] real-data summary JSONs not found")
        return
    rows = []
    for regime, p in [("Workload only", sw), ("Hazard overlay", so)]:
        data = json.loads(p.read_text())
        comps = data.get("comparisons_vs_strongest_baseline", {})
        for metric, c in comps.items():
            rows.append({"regime": regime, "metric": metric,
                         "baseline": c.get("strongest_baseline"),
                         "baseline_mean": c.get("baseline_mean"),
                         "treatment_mean": c.get("treatment_mean"),
                         "mean_diff": c.get("mean_diff"),
                         "ci_low": c.get("ci_low"), "ci_high": c.get("ci_high"),
                         "win_rate": c.get("win_rate"),
                         "significant_better": c.get("significant_better"),
                         "significant_worse": c.get("significant_worse")})
    if not rows:
        print("  [skip] no comparisons in summary JSON")
        return
    df = pd.DataFrame(rows)
    df.to_csv(METRICS / "11_preregistered_hypothesis_tests.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    labels, lows, highs, mids, cols = [], [], [], [], []
    for _, r in df.iterrows():
        if r.ci_low is None or r.mean_diff is None:
            continue
        labels.append(f"{r.metric}\n({r.regime})")
        lows.append(r.ci_low); highs.append(r.ci_high); mids.append(r.mean_diff)
        cols.append("#C53030" if r.significant_worse else ("#38A169" if r.significant_better else "#718096"))
    y = np.arange(len(labels))
    for i in range(len(labels)):
        ax.plot([lows[i], highs[i]], [y[i], y[i]], color=cols[i], lw=3, solid_capstyle="round")
        ax.scatter(mids[i], y[i], color=cols[i], s=70, zorder=3)
    ax.axvline(0, color="#1A202C", ls="--", lw=1.4)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    style(ax, "Pre-registered tests: Risk-Aware MARL minus strongest baseline\n(bars are bootstrap 95% confidence intervals)",
          "Paired mean difference  (note: metrics have different units and scales)", "")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], color="#C53030", lw=3, label="Significantly WORSE than the best baseline"),
        Line2D([0], [0], color="#38A169", lw=3, label="Significantly BETTER than the best baseline"),
        Line2D([0], [0], color="#718096", lw=3, label="No significant difference (bar crosses zero)"),
    ], loc="lower left", frameon=False, fontsize=8.5)
    save(fig, "11_preregistered_hypothesis_tests.png")


if __name__ == "__main__":
    print("Generating graphs into results/graphs/ ...")
    for fn in (graph_six_scheduler_benchmark, graph_ablation, graph_seven_scenarios,
               graph_marl_learning_curve, graph_realdata_summary,
               graph_realdata_four_metrics, graph_hypothesis_verdicts):
        try:
            fn()
        except Exception as exc:  # keep going; report honestly
            print(f"  [FAILED] {fn.__name__}: {type(exc).__name__}: {exc}")
    print("Done.")
