"""
Renders the system-architecture diagram as a PNG for the slide deck.
Part of the documentation pass (30 Oct 2026); not original project code.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent.parent / "results" / "graphs" / "00_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

BAND = "#F7FAFC"
BANDEDGE = "#CBD5E0"

LAYERS = [
    # (y, height, band title, [(x, w, label, facecolor, textcolor)])
    (8.55, 1.15, "1 · WHERE JOBS COME FROM", [
        (0.45, 2.75, "task_generator.py\nsynthetic LLM / image / video", "#BEE3F8", "#1A365D"),
        (3.45, 2.75, "traffic_patterns.py\nlight · heavy · burst · peak", "#BEE3F8", "#1A365D"),
        (6.45, 3.10, "realdata/trace_builder.py\nREAL traces: BurstGPT 1.43M reqs\n+ UCI video 168k records",
         "#9AE6B4", "#1C4532"),
    ]),
    (6.85, 1.15, "2 · WHO DECIDES  (6 schedulers, one shared interface)", [
        (0.45, 2.45, "schedulers/traditional.py\nFCFS · Round Robin\n· Least Loaded", "#E2E8F0", "#1A202C"),
        (3.15, 2.45, "agent.py\nSingle-Agent RL\n(one central brain)", "#FED7D7", "#742A2A"),
        (5.85, 3.70, "marl_agent.py\nMARL — one agent per GPU, they BID\nbid = Q-value + headroom\n− risk penalty − EFC penalty",
         "#E9D8FD", "#44337A"),
    ]),
    (5.15, 1.15, "3 · SIMULATED HARDWARE", [
        (0.45, 3.10, "environment.py\nthe clock · transactional\nplacement with rollback", "#FEEBC8", "#7B341E"),
        (3.80, 3.10, "gpu.py × 4\n8 GB each · temperature\n· crash at 85 °C · cooldown", "#FEEBC8", "#7B341E"),
        (7.15, 2.40, "dynamic_queue.py\njob lifecycle\n+ retries", "#FEEBC8", "#7B341E"),
    ]),
    (3.45, 1.15, "4 · SAFETY BRAIN", [
        (0.45, 3.10, "risk_engine.py\nrisk score 0–1 from 8 signals\nquarantine at ≥ 0.85", "#FED7D7", "#742A2A"),
        (3.80, 3.10, "fault_injection.py\ninjects simulated\nhardware failures", "#FED7D7", "#742A2A"),
        (7.15, 2.40, "Expected Failure Cost\nP(fail) × impact\n× recovery", "#FED7D7", "#742A2A"),
    ]),
    (1.75, 1.15, "5 · MEASUREMENT", [
        (0.45, 3.10, "benchmarking/metrics.py\nthroughput · utilisation\n· fairness · risk", "#C6F6D5", "#22543D"),
        (3.80, 3.10, "evaluation/stats.py\nbootstrap 95% CIs\n· paired win rates", "#C6F6D5", "#22543D"),
        (7.15, 2.40, "results\nCSV · JSON\n· figures", "#C6F6D5", "#22543D"),
    ]),
    (0.30, 1.00, "6 · LIVE DEMO", [
        (0.45, 4.40, "backend/server.py — FastAPI + WebSocket", "#BEE3F8", "#1A365D"),
        (5.10, 4.45, "frontend/ — React 19 dashboard, 7 live panels", "#BEE3F8", "#1A365D"),
    ]),
]


def main():
    fig, ax = plt.subplots(figsize=(13.5, 10.5))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 10.3)
    ax.axis("off")

    ax.text(5.1, 10.0, "SmartGPU Allocator — System Architecture",
            ha="center", fontsize=18, fontweight="bold", color="#1A202C")
    ax.text(5.1, 9.72, "Crash-aware GPU orchestration with multi-agent reinforcement learning",
            ha="center", fontsize=10.5, color="#4A5568", style="italic")

    for y, h, band_title, boxes in LAYERS:
        ax.add_patch(FancyBboxPatch((0.25, y - 0.12), 9.7, h + 0.42,
                                    boxstyle="round,pad=0.02,rounding_size=0.06",
                                    facecolor=BAND, edgecolor=BANDEDGE, linewidth=1.1))
        ax.text(0.38, y + h + 0.17, band_title, fontsize=9.5, fontweight="bold", color="#2D3748")
        for x, w, label, fc, tc in boxes:
            ax.add_patch(FancyBboxPatch((x, y), w, h,
                                        boxstyle="round,pad=0.02,rounding_size=0.05",
                                        facecolor=fc, edgecolor="#4A5568", linewidth=1.0))
            ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                    fontsize=8.2, color=tc, linespacing=1.45)

    # Vertical flow arrows down the left, plus the learning feedback loop on the right.
    def arrow(x1, y1, x2, y2, color, style="-|>", lw=2.0, rad=0.0, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=16, linewidth=lw, color=color,
                                     linestyle=ls,
                                     connectionstyle=f"arc3,rad={rad}"))

    for y_from, y_to in [(8.55, 8.00), (6.85, 6.30), (5.15, 4.60), (3.45, 2.90), (1.75, 1.30)]:
        arrow(0.15, y_from, 0.15, y_to, "#2D3748", rad=0.0)

    # risk + EFC feed the MARL bid (layer 4 -> layer 2)
    arrow(9.95, 4.03, 9.95, 7.43, "#C53030", rad=-0.30, ls="--")
    ax.text(10.1, 5.75, "risk + EFC\nfeed the bid", fontsize=8, color="#C53030",
            rotation=90, va="center", ha="center", fontweight="bold")

    # reward feedback (layer 3 -> layer 2)
    arrow(9.60, 5.73, 9.60, 7.43, "#2B6CB0", rad=-0.35, ls="--")
    ax.text(9.40, 6.60, "reward\n(clamped −20…+12)", fontsize=8, color="#2B6CB0",
            rotation=90, va="center", ha="center", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
