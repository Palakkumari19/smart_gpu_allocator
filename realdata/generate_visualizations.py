"""
Visualization generator for Real-Data Risk Benchmark.
Generates publication-quality charts for reports and artifacts.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from realdata.config import (
    RESULTS_DIR,
    HAZARD_CURVES_PATH,
    REPORTS_DIR,
)

# Target output directories
FIGURES_DIR = REPORTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR = Path("/home/bmsit/.gemini/antigravity/brain/3e352e17-fa78-444c-b888-445471b73dc2")
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

# Colors and styling
SCHEDULER_COLORS = {
    "fcfs": "#4A5568",            # Slate
    "round_robin": "#718096",     # Cool Grey
    "least_loaded": "#319795",    # Teal
    "rl": "#E53E3E",              # Crimson / Coral (Starvation outlier)
    "marl": "#3182CE",            # Blue (Risk-blind baseline)
    "risk_aware_marl": "#805AD5", # Purple (Treatment)
}

SCHEDULER_LABELS = {
    "fcfs": "FCFS",
    "round_robin": "Round Robin",
    "least_loaded": "Least Loaded",
    "rl": "Centralized RL",
    "marl": "Risk-Blind MARL",
    "risk_aware_marl": "Risk-Aware MARL",
}

SCHEDULER_ORDER = [
    "fcfs",
    "round_robin",
    "least_loaded",
    "rl",
    "marl",
    "risk_aware_marl",
]


def load_latest_csvs():
    workload_csv = sorted(RESULTS_DIR.glob("realdata_runs_workload_*.csv"))[-1]
    overlay_csv = sorted(RESULTS_DIR.glob("realdata_runs_overlay_*.csv"))[-1]
    df_w = pd.read_csv(workload_csv)
    df_o = pd.read_csv(overlay_csv)
    return df_w, df_o


def plot_fig1_tradeoff(df_w, df_o):
    """Figure 1: Safety vs Throughput Pareto Frontier (Highlighting the Starvation Trap)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for idx, (df, title, ax) in enumerate([
        (df_w, "Workload-Only Replay", axes[0]),
        (df_o, "Workload + Hazard Overlay", axes[1]),
    ]):
        for s in SCHEDULER_ORDER:
            sub = df[df["scheduler"] == s]
            x = sub["completed_tasks"]
            y = sub["avg_risk_exposure"]

            # Scatter points
            ax.scatter(
                x,
                y,
                alpha=0.35,
                color=SCHEDULER_COLORS[s],
                s=40,
                edgecolors="none",
            )
            # Centroid
            mean_x = x.mean()
            mean_y = y.mean()
            ax.scatter(
                mean_x,
                mean_y,
                color=SCHEDULER_COLORS[s],
                s=180,
                edgecolors="black",
                linewidths=1.5,
                label=SCHEDULER_LABELS[s],
                zorder=5,
            )

        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("Completed Tasks (Higher is Better)", fontsize=11)
        if idx == 0:
            ax.set_ylabel("Average GPU Risk Exposure (Lower is Better)", fontsize=11)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Highlight Centralized RL Starvation Trap
        rl_sub = df[df["scheduler"] == "rl"]
        ax.annotate(
            "Starvation Trap\n(Low Risk via Severe Under-Serving)",
            xy=(rl_sub["completed_tasks"].mean(), rl_sub["avg_risk_exposure"].mean()),
            xytext=(rl_sub["completed_tasks"].mean() + 1.5, rl_sub["avg_risk_exposure"].mean() + 0.08),
            arrowprops=dict(facecolor="#E53E3E", shrink=0.08, width=1.5, headwidth=7),
            fontsize=9.5,
            color="#C53030",
            fontweight="bold",
        )

        # Highlight Active Cluster
        marl_sub = df[df["scheduler"] == "risk_aware_marl"]
        ax.annotate(
            "Active Dispatchers\n(98% Throughput Preserved)",
            xy=(marl_sub["completed_tasks"].mean(), marl_sub["avg_risk_exposure"].mean()),
            xytext=(marl_sub["completed_tasks"].mean() - 8.5, marl_sub["avg_risk_exposure"].mean() + 0.12),
            arrowprops=dict(facecolor="#805AD5", shrink=0.08, width=1.5, headwidth=7),
            fontsize=9.5,
            color="#553C9A",
            fontweight="bold",
        )

    axes[0].legend(frameon=True, facecolor="white", edgecolor="#CBD5E0", loc="upper left", fontsize=9.5)
    plt.suptitle("Figure 1: Safety vs Throughput Tradeoff Across Schedulers (N=360 episodes)", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_p1 = FIGURES_DIR / "fig1_risk_vs_throughput.png"
    out_p2 = ARTIFACT_DIR / "fig1_risk_vs_throughput.png"
    plt.savefig(out_p1, dpi=300)
    plt.savefig(out_p2, dpi=300)
    plt.close()
    print(f"Saved Figure 1 to {out_p1} and {out_p2}")


def plot_fig2_grouped_metrics(df_w, df_o):
    """Figure 2: Grouped Bar Chart of Primary and Secondary Metrics with 95% CIs."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    metrics = [
        ("avg_risk_exposure", "Average GPU Risk Exposure", "Risk Score [0.0 - 1.0]", axes[0, 0]),
        ("terminal_efc", "Cumulative Expected Failure Cost (EFC)", "Cost Units", axes[0, 1]),
        ("crash_count", "Hardware Crash Events", "Crash Count", axes[1, 0]),
        ("completed_tasks", "Successfully Completed Tasks", "Task Count", axes[1, 1]),
    ]

    width = 0.35
    x_indices = np.arange(len(SCHEDULER_ORDER))

    for m_key, m_title, m_unit, ax in metrics:
        means_w = [df_w[df_w["scheduler"] == s][m_key].mean() for s in SCHEDULER_ORDER]
        stds_w = [df_w[df_w["scheduler"] == s][m_key].std() / np.sqrt(30) * 1.96 for s in SCHEDULER_ORDER]

        means_o = [df_o[df_o["scheduler"] == s][m_key].mean() for s in SCHEDULER_ORDER]
        stds_o = [df_o[df_o["scheduler"] == s][m_key].std() / np.sqrt(30) * 1.96 for s in SCHEDULER_ORDER]

        rects1 = ax.bar(x_indices - width/2, means_w, width, yerr=stds_w, label="Workload Only", color="#4299E1", capsize=4, alpha=0.85, edgecolor="#2B6CB0")
        rects2 = ax.bar(x_indices + width/2, means_o, width, yerr=stds_o, label="Workload + Hazard Overlay", color="#ED8936", capsize=4, alpha=0.85, edgecolor="#C05621")

        ax.set_title(m_title, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(m_unit, fontsize=10)
        ax.set_xticks(x_indices)
        ax.set_xticklabels([SCHEDULER_LABELS[s] for s in SCHEDULER_ORDER], rotation=20, ha="right", fontsize=9.5)
        ax.grid(True, linestyle="--", alpha=0.4, axis="y")

        if m_key == "avg_risk_exposure":
            ax.legend(frameon=True, facecolor="white", fontsize=9.5)

    plt.suptitle("Figure 2: Empirical Performance Across Schedulers with 95% Confidence Intervals", fontsize=15, fontweight="bold", y=0.99)
    plt.tight_layout()

    out_p1 = FIGURES_DIR / "fig2_metrics_comparison.png"
    out_p2 = ARTIFACT_DIR / "fig2_metrics_comparison.png"
    plt.savefig(out_p1, dpi=300)
    plt.savefig(out_p2, dpi=300)
    plt.close()
    print(f"Saved Figure 2 to {out_p1} and {out_p2}")


def plot_fig3_hazard_curve():
    """Figure 3: Empirical Hazard Calibration Curve from BurstGPT."""
    if not HAZARD_CURVES_PATH.exists():
        return

    with open(HAZARD_CURVES_PATH, "r") as f:
        data = json.load(f)

    curve = data["curves"]["burstgpt_dispersion"]
    bins = curve["bins"]

    labels = [b["bin_label"] for b in bins]
    probs = [b["failure_probability"] * 100 for b in bins]
    ci_lows = [b["ci_lower"] * 100 for b in bins]
    ci_highs = [b["ci_upper"] * 100 for b in bins]
    sample_sizes = [b["sample_size"] for b in bins]

    yerr_lower = [p - l for p, l in zip(probs, ci_lows)]
    yerr_upper = [h - p for p, h in zip(probs, ci_highs)]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_bars = "#3182CE"
    ax1.bar(labels, probs, yerr=[yerr_lower, yerr_upper], color=color_bars, alpha=0.75, edgecolor="#2B6CB0", capsize=6, width=0.55, label="Empirical Failure Rate (%)")
    ax1.set_xlabel("Arrival Dispersion Index Interval: Var/Mean [dispersion]", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Empirical Failure Rate (%) [Response tokens == 0]", color="#2B6CB0", fontsize=11, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor="#2B6CB0")
    ax1.grid(True, linestyle="--", alpha=0.4, axis="y")

    # Annotate peak failure jump
    ax1.annotate(
        f"26.25% Failure Spike\n(26x Jump under Burst Spikes)\nN={sample_sizes[-1]:,} requests",
        xy=(4, probs[-1]),
        xytext=(2.2, 22),
        arrowprops=dict(facecolor="#E53E3E", shrink=0.08, width=1.5, headwidth=7),
        fontsize=10,
        fontweight="bold",
        color="#C53030",
    )

    # Secondary axis for sample size
    ax2 = ax1.twinx()
    ax2.plot(labels, [s / 1000 for s in sample_sizes], color="#718096", marker="o", linewidth=2, linestyle=":", label="Sample Volume (Thousands)")
    ax2.set_ylabel("Request Volume (Thousands)", color="#4A5568", fontsize=11)
    ax2.tick_params(axis="y", labelcolor="#4A5568")
    ax2.grid(False)

    plt.title("Figure 3: BurstGPT Calibrated Hazard Curve (N=1,429,737 requests, 95% Wilson CIs)", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()

    out_p1 = FIGURES_DIR / "fig3_hazard_curves.png"
    out_p2 = ARTIFACT_DIR / "fig3_hazard_curves.png"
    plt.savefig(out_p1, dpi=300)
    plt.savefig(out_p2, dpi=300)
    plt.close()
    print(f"Saved Figure 3 to {out_p1} and {out_p2}")


def plot_fig4_load_sensitivity(df_w, df_o):
    """Figure 4: Sensitivity to Offered Load Target Utilization (rho = 0.6, 0.9, 1.2)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    loads = [0.6, 0.9, 1.2]

    # Panel A: Risk Exposure vs Load
    ax = axes[0]
    for s in ["rl", "least_loaded", "marl", "risk_aware_marl"]:
        y_w = [df_w[(df_w["scheduler"] == s) & (df_w["target_utilization"] == l)]["avg_risk_exposure"].mean() for l in loads]
        y_o = [df_o[(df_o["scheduler"] == s) & (df_o["target_utilization"] == l)]["avg_risk_exposure"].mean() for l in loads]
        ax.plot(loads, y_w, marker="o", linestyle="--", color=SCHEDULER_COLORS[s], label=f"{SCHEDULER_LABELS[s]} (Workload)")
        ax.plot(loads, y_o, marker="s", linestyle="-", color=SCHEDULER_COLORS[s], label=f"{SCHEDULER_LABELS[s]} (Overlay)")

    ax.set_title("Average Risk Exposure vs Offered Load", fontsize=12, fontweight="bold")
    ax.set_xlabel("Target Cluster Utilization (\u03c1)", fontsize=11)
    ax.set_ylabel("Average GPU Risk Exposure", fontsize=11)
    ax.set_xticks(loads)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True, facecolor="white", fontsize=8.5, ncol=2)

    # Panel B: Crash Count vs Load
    ax = axes[1]
    for s in ["rl", "least_loaded", "marl", "risk_aware_marl"]:
        y_w = [df_w[(df_w["scheduler"] == s) & (df_w["target_utilization"] == l)]["crash_count"].mean() for l in loads]
        y_o = [df_o[(df_o["scheduler"] == s) & (df_o["target_utilization"] == l)]["crash_count"].mean() for l in loads]
        ax.plot(loads, y_w, marker="o", linestyle="--", color=SCHEDULER_COLORS[s], label=f"{SCHEDULER_LABELS[s]} (Workload)")
        ax.plot(loads, y_o, marker="s", linestyle="-", color=SCHEDULER_COLORS[s], label=f"{SCHEDULER_LABELS[s]} (Overlay)")

    ax.set_title("Cluster Crash Events vs Offered Load", fontsize=12, fontweight="bold")
    ax.set_xlabel("Target Cluster Utilization (\u03c1)", fontsize=11)
    ax.set_ylabel("Crash Events (Mean Count)", fontsize=11)
    ax.set_xticks(loads)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.suptitle("Figure 4: Stress Sensitivity Across Offered Load Presets (\u03c1 \u2208 {0.6, 0.9, 1.2})", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()

    out_p1 = FIGURES_DIR / "fig4_load_sensitivity.png"
    out_p2 = ARTIFACT_DIR / "fig4_load_sensitivity.png"
    plt.savefig(out_p1, dpi=300)
    plt.savefig(out_p2, dpi=300)
    plt.close()
    print(f"Saved Figure 4 to {out_p1} and {out_p2}")


if __name__ == "__main__":
    df_w, df_o = load_latest_csvs()
    plot_fig1_tradeoff(df_w, df_o)
    plot_fig2_grouped_metrics(df_w, df_o)
    plot_fig3_hazard_curve()
    plot_fig4_load_sensitivity(df_w, df_o)
    print("All figures generated successfully.")
