"""
v2 candidate improvements, each behind a flag so it can be switched off and ablated.

Implemented (justified by docs/07_diagnosis.md):
  C  better training   — real epsilon decay, longer budget, multiple seeds,
                         each MARL variant trained under its OWN bid rule
  D  state buckets     — refined risk_bin edges, derived from the TRAIN-window
                         risk distribution only
  F  evaluation fixes  — see v2/load_model.py (load calibration) and the
                         episode-length / load-level settings in config_v2

NOT implemented, by evidence:
  A  penalty rescaling — diagnosis showed penalties already flip 37.7% of
                         contested decisions (penalty SD / reward SD = 0.85)
  E  hazard-curve input — deferred; speculative until real choices exist

Everything here patches at RUNTIME by wrapping. No baseline source file is edited.
"""

from __future__ import annotations

import random
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import marl_agent
from benchmarking.scheduler_adapters import get_scheduler_adapter
from environment import GPUEnvironment

# The shipped bucket edges, for reference and for the "off" setting.
V1_RISK_EDGES = (0.25, 0.50, 0.75)


# ───────────────────────────────────────── D: refined risk bucketing
@contextmanager
def refined_risk_buckets(edges: Sequence[float]):
    """
    Replace MARLAgent.get_state's risk_bin edges.

    Diagnosis 6a: the shipped edges (0.25/0.50/0.75) put 84% of observations in
    bucket 0 and never use buckets 2 or 3, so the agent cannot tell "slightly
    risky" from "safe" — the very distinction the project is about.

    Implemented by wrapping get_state and re-deriving element 3 of the state
    tuple from the GPU's live risk score. The rest of the state is untouched.
    """
    orig_get_state = marl_agent.MARLAgent.get_state
    e = tuple(sorted(float(x) for x in edges))

    def patched(self, gpu, cluster_state):
        state = list(orig_get_state(self, gpu, cluster_state))
        rs_obj = gpu.get_risk_state() if hasattr(gpu, "get_risk_state") else None
        r = float(rs_obj.risk_score) if rs_obj is not None else 0.0
        b = 0
        for i, edge in enumerate(e):
            if r > edge:
                b = i + 1
        if len(state) > 3:
            state[3] = b
        return tuple(state)

    marl_agent.MARLAgent.get_state = patched
    try:
        yield
    finally:
        marl_agent.MARLAgent.get_state = orig_get_state


@contextmanager
def no_patch():
    yield


# ───────────────────────────────────────── C: training
@dataclass
class TrainingConfig:
    """Identical budget is applied to every learner. Pre-registered before the final run."""
    episodes: int = 20                 # training episodes per learner
    steps_per_episode: int = 120
    seeds: tuple = (1, 2, 3)           # training seeds (NOT test seeds)
    target_rho: float = 0.9
    epsilon_decay_per_step: float = 0.9970
    epsilon_min: float = 0.05
    learning_rate: float = 0.01
    # stopping rule, pre-registered: stop at `episodes`, or earlier if Q-table
    # growth over a whole episode falls below `plateau_growth` for
    # `plateau_patience` consecutive episodes.
    plateau_growth: int = 5
    plateau_patience: int = 3

    def as_dict(self):
        return dict(self.__dict__)


def _decay_epsilon_inplace(manager_or_agent, cfg: TrainingConfig):
    """Apply the configured decay. The shipped schedule barely moves in 80 steps."""
    agents = getattr(manager_or_agent, "agents", None)
    if agents is None:
        a = manager_or_agent
        a.epsilon = max(cfg.epsilon_min, a.epsilon * cfg.epsilon_decay_per_step)
    else:
        for a in agents:
            a.epsilon = max(cfg.epsilon_min, a.epsilon * cfg.epsilon_decay_per_step)


def train_learner(
    scheduler_id: str,
    windows: Sequence[Any],
    df_burst,
    df_video,
    cfg: TrainingConfig,
    risk_edges: Optional[Sequence[float]] = None,
    trace_builder=None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Train ONE learner under ITS OWN bid rule.

    Fixes diagnosis 6c: v1 trained a single Q-table with the default
    (risk-aware) bid and handed it to risk-blind MARL as well.

    Returns the frozen Q-table snapshot plus a training curve.
    """
    from v2.load_model import build_calibrated_trace
    builder = trace_builder or build_calibrated_trace

    ctx = refined_risk_buckets(risk_edges) if risk_edges else no_patch()
    curve: List[Dict[str, Any]] = []
    snapshot = None
    plateau_hits = 0
    prev_q = 0

    with ctx:
        # One long-lived environment so learning accumulates across episodes.
        random.seed(cfg.seeds[0]); np.random.seed(cfg.seeds[0])
        env = GPUEnvironment(scheduler="fcfs")
        env.enable_deterministic_physics()
        adapter = get_scheduler_adapter(scheduler_id)
        adapter.configure_environment(env)
        adapter.reset_state(env)

        ep = 0
        for episode in range(cfg.episodes):
            window = windows[episode % len(windows)]
            seed = cfg.seeds[episode % len(cfg.seeds)] * 1000 + episode
            trace, _cal = builder(
                window=window, episode_steps=cfg.steps_per_episode,
                target_rho=cfg.target_rho, df_burst=df_burst, df_video=df_video,
                seed=seed, partition="train",
            )
            env.attach_workload_trace(deepcopy(trace))
            for _ in range(len(trace)):
                env.step()
                if scheduler_id == "rl":
                    _decay_epsilon_inplace(env.agent, cfg)
                else:
                    _decay_epsilon_inplace(env.marl_manager, cfg)

            if scheduler_id == "rl":
                qsize = len(getattr(env.agent, "q_table", {}))
                eps = float(getattr(env.agent, "epsilon", 0.0))
            else:
                st = env.marl_manager.get_global_stats()
                qsize = int(st["total_q_table_size"])
                eps = float(st["avg_epsilon"])

            growth = qsize - prev_q
            prev_q = qsize
            curve.append({"episode": episode + 1, "window": window.window_id,
                          "q_table_size": qsize, "epsilon": round(eps, 4),
                          "growth": growth})
            if verbose:
                print(f"    ep{episode+1:>3} {window.window_id:<18} q={qsize:<6} eps={eps:.3f} (+{growth})")

            ep = episode + 1
            plateau_hits = plateau_hits + 1 if growth < cfg.plateau_growth else 0
            if plateau_hits >= cfg.plateau_patience:
                if verbose:
                    print(f"    stopping rule: Q-table plateaued for {cfg.plateau_patience} episodes")
                break

        snapshot = (env.agent.snapshot_q_table() if scheduler_id == "rl"
                    else env.marl_manager.snapshot_q_tables())

    return {"scheduler_id": scheduler_id, "snapshot": snapshot, "curve": curve,
            "episodes_run": ep, "final_q_size": prev_q,
            "final_epsilon": curve[-1]["epsilon"] if curve else None}


def train_all_learners(windows, df_burst, df_video, cfg: TrainingConfig,
                       risk_edges=None, verbose: bool = True) -> Dict[str, Any]:
    """Identical budget for every learner: rl, marl (risk-blind), risk_aware_marl."""
    out = {}
    for sid in ("rl", "marl", "risk_aware_marl"):
        if verbose:
            print(f"  training {sid} ...")
        out[sid] = train_learner(sid, windows, df_burst, df_video, cfg,
                                 risk_edges=risk_edges, verbose=verbose)
    return out


# ───────────────────────────────────────── bucket-edge derivation (TRAIN only)
def derive_risk_edges_from_train(
    windows, df_burst, df_video, n_buckets: int = 4,
    steps: int = 120, seeds: Sequence[int] = (1, 2, 3), target_rho: float = 0.9,
    trace_builder=None,
) -> Dict[str, Any]:
    """
    Choose risk_bin edges as equal-frequency quantiles of the risk distribution
    observed on TRAIN windows only (condition 4).

    Uses a neutral dispatcher (least_loaded) so the edges are not chosen using
    any MARL behaviour.
    """
    from v2.load_model import build_calibrated_trace
    builder = trace_builder or build_calibrated_trace

    samples: List[float] = []
    for w in windows:
        for sd in seeds:
            trace, _ = builder(window=w, episode_steps=steps, target_rho=target_rho,
                               df_burst=df_burst, df_video=df_video, seed=sd,
                               partition="train")
            random.seed(sd); np.random.seed(sd)
            env = GPUEnvironment(scheduler="fcfs")
            env.enable_deterministic_physics()
            get_scheduler_adapter("least_loaded").configure_environment(env)
            env.attach_workload_trace(deepcopy(trace))
            for _ in range(len(trace)):
                env.step()
                for g in env.gpus:
                    st = g.get_risk_state()
                    if st:
                        samples.append(float(st.risk_score))

    arr = np.asarray(samples)
    # Rule (fixed in PREREGISTRATION_V2.md): edges are the 50th, 75th and 95th
    # percentiles of the TRAIN risk distribution. Equal-frequency quartiles are
    # degenerate here because >25% of observations are exactly 0.0, which would
    # place an edge at 0. Any non-positive or duplicated edge is dropped and the
    # next higher percentile substituted, keeping edges strictly increasing.
    pct_ladder = [50, 75, 95, 97, 99]
    edges: list = []
    for q in pct_ladder:
        if len(edges) >= n_buckets - 1:
            break
        v = float(round(np.percentile(arr, q), 4))
        if v > 0 and (not edges or v > edges[-1]):
            edges.append(v)
    return {
        "edges": edges,
        "n_samples": int(arr.size),
        "observed_mean": float(round(arr.mean(), 4)),
        "observed_p50": float(round(np.percentile(arr, 50), 4)),
        "observed_p95": float(round(np.percentile(arr, 95), 4)),
        "observed_max": float(round(arr.max(), 4)),
        "v1_edges": list(V1_RISK_EDGES),
    }
