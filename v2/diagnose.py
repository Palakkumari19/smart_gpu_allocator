"""
PHASE 1 DIAGNOSIS — why Risk-Aware MARL did not beat the baselines in v1.

Read-only with respect to the algorithm: this script does not change any scheduler,
reward or physics. It instruments existing code paths (by wrapping, never editing)
and measures. Every number it prints is written to results/metrics/v2/.

Run:  PYTHONPATH=. .venv/bin/python v2/diagnose.py [--quick]
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from benchmarking.scheduler_adapters import get_scheduler_adapter
from environment import GPUEnvironment
from realdata.config import BURSTGPT_WINDOWS_PATH
from realdata.loaders import load_burstgpt_data, load_uci_video_data
from realdata.trace_builder import build_real_trace, compute_target_load_scale
from realdata.windows import TraceWindowsSummary

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "metrics" / "v2"
MET.mkdir(parents=True, exist_ok=True)

# v1 settings, reused so the diagnosis describes the system AS TESTED in v1.
V1_TRAIN_STEPS = 80
V1_EVAL_STEPS = 120
V1_TRAIN_SEED = 42


# ────────────────────────────────────────────── shared trace plumbing
def load_sources():
    return load_burstgpt_data(), load_uci_video_data()


def make_trace(window, steps, seed, utilization, df_burst, df_video, partition):
    scale = compute_target_load_scale(
        target_utilization=utilization,
        raw_requests_count=window.request_count,
        window_duration_seconds=window.duration_seconds,
        seconds_per_step=1.0,
    )
    return build_real_trace(
        window=window, seconds_per_step=1.0, load_scale=scale, video_fraction=0.2,
        seed=seed, df_burst=df_burst, df_video=df_video,
        total_steps=steps, partition=partition,
    )


def train_v1_style(train_window, df_burst, df_video, steps=V1_TRAIN_STEPS, seed=V1_TRAIN_SEED):
    """Reproduce v1's training exactly, and also report what it learned."""
    trace = make_trace(train_window, steps, seed, 0.9, df_burst, df_video, "train")

    random.seed(seed); np.random.seed(seed)
    marl_env = GPUEnvironment(scheduler="marl")
    marl_env.enable_deterministic_physics()
    marl_env.attach_workload_trace(deepcopy(trace))
    curve = []
    for i in range(len(trace)):
        marl_env.step()
        st = marl_env.marl_manager.get_global_stats()
        curve.append({"step": i + 1,
                      "q_table_size": st["total_q_table_size"],
                      "total_reward": st["total_reward"],
                      "success_rate": st["global_success_rate"],
                      "avg_epsilon": st["avg_epsilon"]})
    marl_q = marl_env.marl_manager.snapshot_q_tables()

    random.seed(seed); np.random.seed(seed)
    rl_env = GPUEnvironment(scheduler="rl")
    rl_env.enable_deterministic_physics()
    rl_env.attach_workload_trace(deepcopy(trace))
    for _ in range(len(trace)):
        rl_env.step()
    rl_q = rl_env.agent.snapshot_q_table()

    return marl_q, rl_q, pd.DataFrame(curve), trace


def build_eval_env(scheduler_id, marl_q, rl_q, seed):
    random.seed(seed); np.random.seed(seed)
    env = GPUEnvironment(scheduler="fcfs")
    env.enable_deterministic_physics()
    adapter = get_scheduler_adapter(scheduler_id)
    adapter.configure_environment(env)
    adapter.reset_state(env)
    if scheduler_id in ("marl", "risk_aware_marl") and marl_q is not None:
        env.marl_manager.restore_q_tables(deepcopy(marl_q))
    if scheduler_id == "rl" and rl_q is not None:
        env.agent.restore_q_table(deepcopy(rl_q))
    env.set_eval_mode()
    return env


# ────────────────────────────── D1: decision divergence + penalty magnitudes
def d1_decision_divergence(windows, df_burst, df_video, marl_q, rl_q, seeds, presets):
    """
    Counterfactual test. Run the REAL risk-aware policy. At every decision, take the
    same four bid breakdowns and ask: which GPU would the risk-BLIND rule have picked?
    Identical state, only the decision rule differs — so divergence is attributable
    purely to the risk + EFC penalties.
    """
    print("\n[D1] Decision divergence: risk-aware vs risk-blind on identical state")
    decisions, diverged, rows = 0, 0, []
    infeasible_decisions = 0
    eligible_hist = Counter()
    offered_vs_capacity = []

    for wkey, window in windows.items():
        for preset in presets:
            for seed in seeds:
                trace = make_trace(window, V1_EVAL_STEPS, seed, preset, df_burst, df_video, "eval")
                env = build_eval_env("risk_aware_marl", marl_q, rl_q, seed)
                mgr = env.marl_manager
                orig_select = mgr.select_gpu_for_task

                def wrapped(task, gpus, cluster_state, *a, **kw):
                    nonlocal decisions, diverged, infeasible_decisions
                    out = orig_select(task, gpus, cluster_state, *a, **kw)
                    bds = mgr.last_candidate_breakdowns or []
                    elig = [b for b in bds if b.get("eligible")]
                    eligible_hist[len(elig)] += 1
                    if len(elig) >= 2:
                        aware = max(elig, key=lambda b: b["q_contribution"] + b["perf_score"]
                                    - b["risk_penalty"] - b["efc_penalty"])
                        blind = max(elig, key=lambda b: b["q_contribution"] + b["perf_score"])
                        decisions += 1
                        if aware["gpu_id"] != blind["gpu_id"]:
                            diverged += 1
                        for b in elig:
                            rows.append({
                                "window": wkey, "preset": preset, "seed": seed,
                                "n_eligible": len(elig),
                                "q_contribution": b["q_contribution"],
                                "perf_score": b["perf_score"],
                                "risk_penalty": b["risk_penalty"],
                                "efc_penalty": b["efc_penalty"],
                                "risk_score": b["risk_score"],
                                "efc": b["efc"],
                                "diverged": aware["gpu_id"] != blind["gpu_id"],
                            })
                    elif len(elig) <= 1:
                        infeasible_decisions += 1
                    return out

                mgr.select_gpu_for_task = wrapped
                env.attach_workload_trace(deepcopy(trace))
                for _ in range(len(trace)):
                    env.step()
                mgr.select_gpu_for_task = orig_select

                offered = sum(len(st) for st in trace)
                mem = [t.memory_required for st in trace for t in st]
                dur = [t.execution_time for st in trace for t in st]
                cap = 4 * 8.0
                offered_vs_capacity.append({
                    "window": wkey, "preset": preset, "seed": seed,
                    "tasks_offered": offered,
                    "steps": len(trace),
                    "mean_task_memory_gb": float(np.mean(mem)) if mem else 0.0,
                    "mean_task_duration": float(np.mean(dur)) if dur else 0.0,
                    "offered_gb_steps_per_step": float(np.sum(np.array(mem) * np.array(dur)) / len(trace)) if mem else 0.0,
                    "cluster_capacity_gb": cap,
                    "offered_over_capacity": float(np.sum(np.array(mem) * np.array(dur)) / len(trace) / cap) if mem else 0.0,
                })

    pd.DataFrame(offered_vs_capacity).to_csv(MET / "d6_offered_load_pressure.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(MET / "d1_bid_components.csv", index=False)
    pct = 100.0 * diverged / max(1, decisions)

    # How big are the penalties next to the terms that reward throughput?
    summary = {
        "decisions_with_choice": decisions,
        "decisions_diverged": diverged,
        "divergence_pct": round(pct, 3),
        "decisions_with_<=1_eligible_gpu": infeasible_decisions,
        "total_selection_calls": int(sum(eligible_hist.values())),
        "pct_decisions_with_real_choice": round(
            100.0 * decisions / max(1, sum(eligible_hist.values())), 3),
        "eligible_gpu_count_histogram": {str(k): int(v) for k, v in sorted(eligible_hist.items())},
    }
    for col in ("q_contribution", "perf_score", "risk_penalty", "efc_penalty", "risk_score", "efc"):
        s = df[col]
        summary[f"{col}_mean"] = round(float(s.mean()), 5)
        summary[f"{col}_p50"] = round(float(s.median()), 5)
        summary[f"{col}_p95"] = round(float(s.quantile(0.95)), 5)
        summary[f"{col}_max"] = round(float(s.max()), 5)
    # spread across the 4 candidate GPUs is what actually decides the argmax
    summary["reward_term_mean"] = round(float((df.q_contribution + df.perf_score).mean()), 5)
    summary["penalty_term_mean"] = round(float((df.risk_penalty + df.efc_penalty).mean()), 5)
    summary["penalty_to_reward_ratio"] = round(
        float((df.risk_penalty + df.efc_penalty).mean() / max(1e-9, (df.q_contribution + df.perf_score).abs().mean())), 5)

    print(f"   decisions with a real choice : {decisions}")
    print(f"   diverged                      : {diverged}  ({pct:.2f}%)")
    print(f"   mean reward term (Q+perf)     : {summary['reward_term_mean']}")
    print(f"   mean penalty term (risk+EFC)  : {summary['penalty_term_mean']}")
    print(f"   total selection calls         : {summary['total_selection_calls']}")
    print(f"   eligible-GPU histogram        : {summary['eligible_gpu_count_histogram']}")
    print(f"   %% of decisions with a choice  : {summary['pct_decisions_with_real_choice']}%%")
    ovc = pd.DataFrame(offered_vs_capacity)
    print(f"   offered load / cluster capacity: mean {ovc.offered_over_capacity.mean():.2f}x "
          f"(range {ovc.offered_over_capacity.min():.2f}-{ovc.offered_over_capacity.max():.2f})")
    json.dump(summary, open(MET / "d1_divergence_summary.json", "w"), indent=2)
    return summary, df


# ────────────────────────────── D2: ablation with PEAK and MEAN per-GPU risk
def d2_ablation_peak_vs_mean(windows, df_burst, df_video, marl_q, rl_q, seeds, presets):
    """
    v1's ablation reported only mean risk. Hypothesis to test: the risk penalties
    SPREAD load, raising the cluster MEAN while lowering the per-GPU PEAK.
    We therefore track both, sampled every step.
    """
    print("\n[D2] Ablation: peak vs mean per-GPU risk")

    def make_variant(env, use_risk, use_efc):
        orig = env.marl_manager.compute_bid_score_breakdown

        def patched(gpu, task, agent, cluster_state):
            b = orig(gpu, task, agent, cluster_state)
            if b.get("eligible", False):
                rp = b["risk_penalty"] if use_risk else 0.0
                ep = b["efc_penalty"] if use_efc else 0.0
                b["risk_penalty"], b["efc_penalty"] = rp, ep
                b["bid_score"] = round(b["q_contribution"] + b["perf_score"] - rp - ep, 4)
            return b

        env.marl_manager.compute_bid_score_breakdown = patched

    variants = {
        "full_risk_aware": (True, True),
        "no_risk_penalty": (False, True),
        "no_efc_penalty": (True, False),
        "risk_blind": (False, False),
    }

    rows = []
    for name, (ur, ue) in variants.items():
        for wkey, window in windows.items():
            for preset in presets:
                for seed in seeds:
                    trace = make_trace(window, V1_EVAL_STEPS, seed, preset, df_burst, df_video, "eval")
                    env = build_eval_env("risk_aware_marl", marl_q, rl_q, seed)
                    make_variant(env, ur, ue)
                    env.attach_workload_trace(deepcopy(trace))
                    per_gpu_peak = defaultdict(float)
                    step_means = []
                    for _ in range(len(trace)):
                        env.step()
                        rs = []
                        for g in env.gpus:
                            st = g.get_risk_state()
                            r = float(st.risk_score) if st else 0.0
                            rs.append(r)
                            per_gpu_peak[g.id] = max(per_gpu_peak[g.id], r)
                        step_means.append(float(np.mean(rs)))
                    m = env.get_metrics()
                    rows.append({
                        "variant": name, "window": wkey, "preset": preset, "seed": seed,
                        "mean_risk_over_time": float(np.mean(step_means)),
                        "peak_gpu_risk": float(max(per_gpu_peak.values())),
                        "mean_of_per_gpu_peaks": float(np.mean(list(per_gpu_peak.values()))),
                        "max_step_mean_risk": float(np.max(step_means)),
                        "completed_tasks": float(m.get("completed", 0)),
                        "crash_count": float(m.get("total_crash_events", 0)),
                        "terminal_avg_risk": float(m.get("average_gpu_risk", 0.0)),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(MET / "d2_ablation_peak_mean.csv", index=False)
    agg = df.groupby("variant")[["mean_risk_over_time", "peak_gpu_risk", "mean_of_per_gpu_peaks",
                                 "completed_tasks", "crash_count"]].mean().round(4)
    print(agg.to_string())
    agg.to_csv(MET / "d2_ablation_summary.csv")
    return agg, df


# ────────────────────────────── D3: quarantine reachability
def d3_quarantine_reachability(windows, df_burst, df_video, marl_q, rl_q, seeds, lengths, loads):
    """
    v1 never saw quarantine. Is the 0.85 threshold reachable at all? Sweep episode
    length and offered load, and record the MAX risk ever observed (not just terminal).
    Also records whether the v1 metric (a terminal snapshot) would have caught it.
    """
    print("\n[D3] Quarantine reachability across episode length and load")
    from risk_engine import RiskEngine
    threshold = RiskEngine.QUARANTINE_THRESHOLD
    print(f"   quarantine threshold = {threshold}")

    rows = []
    window = windows["burstiest_window"]
    for steps in lengths:
        for load in loads:
            for seed in seeds:
                trace = make_trace(window, steps, seed, load, df_burst, df_video, "eval")
                env = build_eval_env("risk_aware_marl", marl_q, rl_q, seed)
                env.attach_workload_trace(deepcopy(trace))
                max_risk, ever_quar, quar_steps = 0.0, False, 0
                for _ in range(len(trace)):
                    env.step()
                    q_now = 0
                    for g in env.gpus:
                        st = g.get_risk_state()
                        if st:
                            max_risk = max(max_risk, float(st.risk_score))
                            if st.is_quarantined:
                                q_now += 1
                    if q_now:
                        ever_quar = True
                        quar_steps += 1
                m = env.get_metrics()
                rows.append({
                    "steps": steps, "load": load, "seed": seed,
                    "max_risk_observed": round(max_risk, 4),
                    "ever_quarantined": ever_quar,
                    "steps_with_quarantine": quar_steps,
                    "v1_terminal_quarantine_metric": float(m.get("quarantined_gpus", 0)),
                    "crash_count": float(m.get("total_crash_events", 0)),
                    "completed_tasks": float(m.get("completed", 0)),
                })
                print(f"   steps={steps:<5} load={load:<4} seed={seed}  max_risk={max_risk:.3f}  "
                      f"quarantined={'YES' if ever_quar else 'no'}")

    df = pd.DataFrame(rows)
    df.to_csv(MET / "d3_quarantine_reachability.csv", index=False)
    agg = df.groupby(["steps", "load"]).agg(
        episodes=("seed", "count"),
        pct_quarantine=("ever_quarantined", lambda s: round(100.0 * s.mean(), 1)),
        max_risk=("max_risk_observed", "max"),
        mean_max_risk=("max_risk_observed", "mean"),
    ).round(4)
    print(agg.to_string())
    agg.to_csv(MET / "d3_quarantine_summary.csv")
    return agg, df, threshold


# ────────────────────────────── D4: starvation mechanism
def d4_starvation(windows, df_burst, df_video, marl_q, rl_q, seeds, presets):
    """Confirm HOW single-agent RL gets its low risk: admitted vs completed, and heat."""
    print("\n[D4] Starvation mechanism for single-agent RL")
    rows, traces = [], []
    for sid in ("rl", "risk_aware_marl", "marl", "least_loaded"):
        for wkey, window in windows.items():
            for preset in presets:
                for seed in seeds:
                    trace = make_trace(window, V1_EVAL_STEPS, seed, preset, df_burst, df_video, "eval")
                    env = build_eval_env(sid, marl_q, rl_q, seed)
                    env.attach_workload_trace(deepcopy(trace))
                    temp_trace = []
                    for _ in range(len(trace)):
                        env.step()
                        temp_trace.append(float(np.mean([g.temperature for g in env.gpus])))
                    m = env.get_metrics()
                    offered = sum(len(s) for s in trace)
                    rows.append({
                        "scheduler": sid, "window": wkey, "preset": preset, "seed": seed,
                        "tasks_offered": offered,
                        "successful_assignments": float(m.get("successful_assignments", 0)),
                        "failed_assignments": float(m.get("failed_assignments", 0)),
                        "completed_tasks": float(m.get("completed", 0)),
                        "queued_at_end": float(m.get("total_queued_tasks", 0)),
                        "running_at_end": float(m.get("total_running_tasks", 0)),
                        "avg_temperature": float(m.get("avg_temperature", 0)),
                        "avg_risk": float(m.get("average_gpu_risk", 0)),
                        "crash_count": float(m.get("total_crash_events", 0)),
                    })
                    if preset == presets[0] and seed == seeds[0] and wkey == "burstiest_window":
                        for i, t in enumerate(temp_trace):
                            traces.append({"scheduler": sid, "step": i, "avg_temperature": t})

    df = pd.DataFrame(rows)
    df.to_csv(MET / "d4_starvation.csv", index=False)
    pd.DataFrame(traces).to_csv(MET / "d4_temperature_traces.csv", index=False)
    agg = df.groupby("scheduler")[["tasks_offered", "successful_assignments", "completed_tasks",
                                   "avg_temperature", "avg_risk", "crash_count"]].mean().round(3)
    print(agg.to_string())
    agg.to_csv(MET / "d4_starvation_summary.csv")
    return agg, df, pd.DataFrame(traces)


# ────────────────────────────── D5: training adequacy
def d5_training_adequacy(train_curve, marl_q, windows, df_burst, df_video):
    """Is 80 steps on one small window enough? Measure Q-table coverage."""
    print("\n[D5] Training adequacy")
    train_curve.to_csv(MET / "d5_training_curve.csv", index=False)

    # How many distinct states did training actually visit, per agent?
    sizes = []
    for i, tbl in enumerate(marl_q.get("q_tables", marl_q) if isinstance(marl_q, dict) else []):
        try:
            sizes.append(len(tbl))
        except TypeError:
            pass

    # Count states reachable in principle from the discretisation in get_state().
    # The state is a tuple of discrete buckets; we count the product of bucket counts
    # by inspecting the agent implementation's documented ranges.
    info = {
        "train_steps": int(train_curve["step"].max()),
        "final_q_table_size": int(train_curve["q_table_size"].iloc[-1]),
        "q_table_size_at_half": int(train_curve["q_table_size"].iloc[len(train_curve) // 2]),
        "final_total_reward": float(train_curve["total_reward"].iloc[-1]),
        "final_success_rate": float(train_curve["success_rate"].iloc[-1]),
        "final_avg_epsilon": float(train_curve["avg_epsilon"].iloc[-1]),
        "per_agent_q_sizes": sizes,
    }
    # growth in the last quarter of training tells us whether it had converged
    q = train_curve["q_table_size"].to_numpy()
    info["q_growth_last_quarter"] = int(q[-1] - q[int(len(q) * 0.75)])
    info["train_window_request_count"] = int(windows["_train_window_requests"])
    print(json.dumps(info, indent=2))
    json.dump(info, open(MET / "d5_training_adequacy.json", "w"), indent=2)
    return info


# ────────────────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller sweeps")
    args = ap.parse_args()

    print("Loading empirical datasets ...")
    df_burst, df_video = load_sources()
    ws = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    eval_windows = {k: ws.eval_windows[k] for k in ("median_hour", "burstiest_window")}
    train_window = ws.train_windows["median_hour"]

    print("Reproducing v1 training (80 steps, median_hour, seed 42) ...")
    marl_q, rl_q, curve, _ = train_v1_style(train_window, df_burst, df_video)

    seeds = [42, 43] if args.quick else [42, 43, 44]
    presets = [0.9] if args.quick else [0.6, 0.9, 1.2]

    out = {}
    out["d1"], d1df = d1_decision_divergence(eval_windows, df_burst, df_video, marl_q, rl_q, seeds, presets)
    d2agg, d2df = d2_ablation_peak_vs_mean(eval_windows, df_burst, df_video, marl_q, rl_q, seeds, presets)
    lengths = [120, 500] if args.quick else [120, 500, 1000]
    loads = [0.9, 1.2] if args.quick else [0.9, 1.2, 2.0]
    d3agg, d3df, thr = d3_quarantine_reachability(eval_windows, df_burst, df_video, marl_q, rl_q,
                                                  seeds[:2], lengths, loads)
    d4agg, d4df, d4tr = d4_starvation(eval_windows, df_burst, df_video, marl_q, rl_q, seeds, presets)
    wmeta = dict(eval_windows); wmeta["_train_window_requests"] = train_window.request_count
    out["d5"] = d5_training_adequacy(curve, marl_q, wmeta, df_burst, df_video)
    out["d3_threshold"] = thr

    json.dump({k: v for k, v in out.items() if k != "d1"} | {"d1": out["d1"]},
              open(MET / "diagnosis_summary.json", "w"), indent=2, default=str)
    print(f"\nAll diagnostic metrics written to {MET}")


if __name__ == "__main__":
    main()
