"""
v2 EVALUATION FIX — corrected offered-load calibration.

Diagnosis (docs/07_diagnosis.md, Finding 2): the shipped
`compute_target_load_scale()` normalises the arrival rate over the WINDOW
duration (3600 steps) while `build_real_trace()` emits those arrivals into the
EPISODE length (120 steps), delivering exactly 30x the intended load. It also
uses hardcoded mean task size/duration constants that do not match the
per-window calibrated mappings.

This module computes the scale against the SAME step count the trace is built
with, and measures the mean task footprint empirically from a probe trace
instead of assuming constants.

This is an EVALUATION fix: it changes the offered workload identically for all
six schedulers and touches no scheduler, reward or physics logic. The original
`compute_target_load_scale()` is left untouched so v1 stays reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from realdata.trace_builder import build_real_trace

CLUSTER_NUM_GPUS = 4
GPU_VRAM_GB = 8.0
CLUSTER_VRAM_GB = CLUSTER_NUM_GPUS * GPU_VRAM_GB  # 32.0


@dataclass
class LoadCalibration:
    """Result of calibrating one (window, episode_steps, target_rho) cell."""
    target_rho: float
    load_scale: float
    probe_tasks: int
    mean_task_memory_gb: float
    mean_task_duration_steps: float
    intended_tasks_per_episode: float
    delivered_rho: float = 0.0
    refine_iterations: int = 0

    def as_dict(self):
        return dict(self.__dict__)


def _trace_stats(trace: List[List[Any]]):
    mems = [float(t.memory_required) for step in trace for t in step]
    durs = [float(t.execution_time) for step in trace for t in step]
    return mems, durs


def measure_offered_rho(trace: List[List[Any]], cluster_vram_gb: float = CLUSTER_VRAM_GB) -> float:
    """
    Offered load actually present in a built trace.

        rho = (sum over tasks of memory_gb * duration_steps) / (episode_steps * cluster_vram)

    This is the same definition the shipped docstring uses
    (rho = lambda * E[mem] * E[dur] / capacity), just measured directly.
    """
    steps = max(1, len(trace))
    mems, durs = _trace_stats(trace)
    if not mems:
        return 0.0
    demand = float(np.sum(np.asarray(mems) * np.asarray(durs)))
    return demand / (steps * cluster_vram_gb)


def calibrate_load_scale(
    window,
    episode_steps: int,
    target_rho: float,
    df_burst,
    df_video,
    seed: int = 0,
    video_fraction: float = 0.2,
    seconds_per_step: float = 1.0,
    cluster_vram_gb: float = CLUSTER_VRAM_GB,
    partition: str = "eval",
) -> LoadCalibration:
    """
    Find the load_scale that makes the DELIVERED offered load equal target_rho
    over an episode of `episode_steps` steps.

    Two-pass and self-calibrating:
      1. Build a probe trace at load_scale = 1.0 and measure the real mean task
         memory and duration produced by the calibrated mappers for THIS window.
      2. Solve for the scale analytically.

    No hardcoded task-size constants are used.
    """
    probe = build_real_trace(
        window=window, seconds_per_step=seconds_per_step, load_scale=1.0,
        video_fraction=video_fraction, seed=seed, df_burst=df_burst, df_video=df_video,
        total_steps=episode_steps, partition=partition,
    )
    mems, durs = _trace_stats(probe)
    n_probe = len(mems)
    if n_probe == 0:
        return LoadCalibration(target_rho, 1.0, 0, 0.0, 0.0, 0.0)

    e_mem = float(np.mean(mems))
    e_dur = float(np.mean(durs))

    # Tasks per step the builder emits at scale 1.0, for THIS episode length.
    raw_rate = n_probe / float(episode_steps)

    # Arrival rate that achieves the target offered load.
    lambda_target = (target_rho * cluster_vram_gb) / max(1e-9, e_mem * e_dur)

    scale = lambda_target / max(1e-9, raw_rate)
    return LoadCalibration(
        target_rho=target_rho,
        load_scale=float(scale),
        probe_tasks=n_probe,
        mean_task_memory_gb=e_mem,
        mean_task_duration_steps=e_dur,
        intended_tasks_per_episode=float(lambda_target * episode_steps),
    )


def build_calibrated_trace(
    window,
    episode_steps: int,
    target_rho: float,
    df_burst,
    df_video,
    seed: int,
    video_fraction: float = 0.2,
    seconds_per_step: float = 1.0,
    partition: str = "eval",
    calibration: Optional[LoadCalibration] = None,
    max_refine: int = 6,
    tol: float = 0.03,
):
    """
    Build a trace whose DELIVERED offered load matches target_rho.

    The analytic estimate is only approximate: changing the scale changes how
    many rows are sampled, which shifts the realised mean task size. We therefore
    refine with a short fixed-point iteration (scale *= target / delivered) on
    the actual seed that will be used, and keep the closest iterate.
    """
    if calibration is None:
        calibration = calibrate_load_scale(
            window=window, episode_steps=episode_steps, target_rho=target_rho,
            df_burst=df_burst, df_video=df_video, seed=seed,
            video_fraction=video_fraction, seconds_per_step=seconds_per_step,
            partition=partition,
        )

    def build(scale):
        return build_real_trace(
            window=window, seconds_per_step=seconds_per_step, load_scale=scale,
            video_fraction=video_fraction, seed=seed, df_burst=df_burst,
            df_video=df_video, total_steps=episode_steps, partition=partition,
        )

    scale = calibration.load_scale
    best = None
    for i in range(max_refine):
        trace = build(scale)
        delivered = measure_offered_rho(trace)
        err = abs(delivered - target_rho) / max(1e-9, target_rho)
        if best is None or err < best[0]:
            best = (err, scale, trace, delivered, i + 1)
        if err <= tol or delivered <= 0:
            break
        scale = scale * (target_rho / delivered)

    err, scale, trace, delivered, iters = best
    calibration.load_scale = float(scale)
    calibration.delivered_rho = float(delivered)
    calibration.refine_iterations = int(iters)
    return trace, calibration


def build_legacy_trace(
    window, episode_steps, target_rho, df_burst, df_video, seed,
    video_fraction: float = 0.2, seconds_per_step: float = 1.0, partition: str = "eval",
):
    """
    Build a trace exactly the way v1 did (with the 30x defect), so the v1
    condition can be re-run unchanged on the new test seeds.
    """
    from realdata.trace_builder import compute_target_load_scale
    scale = compute_target_load_scale(
        target_utilization=target_rho,
        raw_requests_count=window.request_count,
        window_duration_seconds=window.duration_seconds,
        seconds_per_step=seconds_per_step,
    )
    trace = build_real_trace(
        window=window, seconds_per_step=seconds_per_step, load_scale=scale,
        video_fraction=video_fraction, seed=seed, df_burst=df_burst, df_video=df_video,
        total_steps=episode_steps, partition=partition,
    )
    return trace, scale
