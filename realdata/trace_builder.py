"""
Real-Data Trace Builder for AetherGrid GPU Scheduling Simulations.

Assembles realistic, replayable multi-modal workload traces:
  - Step arrivals binned from empirical BurstGPT timestamps by seconds_per_step
  - Calibrated load_scale rescaling offered load to target cluster utilization presets (0.6, 0.9, 1.2)
  - Video tasks mapped via VideoTaskMapper at video_fraction; remaining tasks are LLM via LLMTaskMapper
  - Deterministic task IDs: real_{window}_s{seed}_t{index:04d}
  - Generates trace summary reports in realdata/reports/trace_summary.md
"""

import math
import random
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd

from realdata.config import (
    TRACE_SUMMARY_REPORT_PATH,
    BURSTGPT_WINDOWS_PATH,
)
from realdata.loaders import load_burstgpt_data, load_uci_video_data
from realdata.video_mapper import VideoTaskMapper
from realdata.llm_mapper import LLMTaskMapper
from realdata.windows import WindowMetadata, TraceWindowsSummary
from workload.task_generator import EnhancedTask, TaskType


# 4x8GB Cluster Physical Constants
CLUSTER_NUM_GPUS = 4
GPU_VRAM_GB = 8.0
CLUSTER_TOTAL_VRAM_GB = CLUSTER_NUM_GPUS * GPU_VRAM_GB  # 32.0 GB
MAX_PARALLEL_TASKS_PER_GPU = 3
CLUSTER_MAX_PARALLEL_SLOTS = CLUSTER_NUM_GPUS * MAX_PARALLEL_TASKS_PER_GPU  # 12 slots

# Baseline Empirical Resource Means (from Video & BurstGPT profiling)
DEFAULT_AVG_MEMORY_GB = 4.54      # Blended average VRAM per task
DEFAULT_AVG_DURATION_STEPS = 10.61  # Blended average execution steps per task


def compute_target_load_scale(
    target_utilization: float,
    raw_requests_count: int,
    window_duration_seconds: float = 3600.0,
    seconds_per_step: float = 10.0,
    avg_memory_gb: float = DEFAULT_AVG_MEMORY_GB,
    avg_duration_steps: float = DEFAULT_AVG_DURATION_STEPS,
    cluster_vram_gb: float = CLUSTER_TOTAL_VRAM_GB,
) -> float:
    """
    Calculate the load_scale multiplier to achieve a target cluster utilization preset.

    Queueing Theory Formulation:
      Offered Load rho = (lambda * avg_memory * avg_duration) / cluster_vram
      lambda_raw = raw_requests / total_simulation_steps
      rho_raw = (lambda_raw * avg_memory * avg_duration) / cluster_vram
      load_scale = target_utilization / rho_raw

    Target Presets:
      0.6: Under-subscribed (steady state, ~60% memory load)
      0.9: Near-capacity (queue backlog builds, ~90% load)
      1.2: Overload / Stress test (>100% capacity, high preemption and memory pressure)

    Args:
        target_utilization: Desired cluster utilization (e.g. 0.6, 0.9, 1.2).
        raw_requests_count: Raw request count in window.
        window_duration_seconds: Duration of the temporal window in seconds (default: 3600s).
        seconds_per_step: Seconds mapped per discrete simulation step (default: 10s).
        avg_memory_gb: Blended mean task memory footprint (default: 4.54 GB).
        avg_duration_steps: Blended mean task duration (default: 10.61 steps).
        cluster_vram_gb: Total physical cluster VRAM (default: 32.0 GB).

    Returns:
        float: Exact load_scale factor.
    """
    total_steps = max(1.0, window_duration_seconds / seconds_per_step)
    raw_arrival_rate = raw_requests_count / total_steps  # tasks per step

    # Raw offered memory load
    raw_rho = (raw_arrival_rate * avg_memory_gb * avg_duration_steps) / cluster_vram_gb
    if raw_rho <= 0:
        return 1.0

    load_scale = target_utilization / raw_rho
    return float(round(load_scale, 6))


def resolve_window_data(
    window: Union[str, WindowMetadata, int],
    df_burst: Optional[pd.DataFrame] = None,
    partition: str = "eval",
) -> Tuple[str, float, float, pd.DataFrame]:
    """
    Resolve a window identifier (string, metadata, or int hour) to start/end timestamps and slice DataFrame.

    Returns:
        (window_label, start_timestamp, end_timestamp, slice_dataframe)
    """
    burst_df = df_burst if df_burst is not None else load_burstgpt_data()

    if isinstance(window, WindowMetadata):
        start_ts = window.start_timestamp
        end_ts = window.end_timestamp
        label = f"{window.partition}_{window.window_id}"
        slice_df = burst_df[
            (burst_df["Timestamp"] >= start_ts) & (burst_df["Timestamp"] < end_ts)
        ].copy()
        return label, start_ts, end_ts, slice_df

    elif isinstance(window, int):
        hour = window
        start_ts = float(hour * 3600)
        end_ts = float((hour + 1) * 3600)
        label = f"hour_{hour}"
        slice_df = burst_df[
            (burst_df["Timestamp"] >= start_ts) & (burst_df["Timestamp"] < end_ts)
        ].copy()
        return label, start_ts, end_ts, slice_df

    elif isinstance(window, str):
        # Named canonical window (e.g. 'median_hour', 'busiest_hour', 'burstiest_window')
        if BURSTGPT_WINDOWS_PATH.exists():
            summary = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
            partition_dict = summary.eval_windows if partition == "eval" else summary.train_windows
            if window in partition_dict:
                meta = partition_dict[window]
                return resolve_window_data(meta, df_burst=burst_df)
        
        # Fallback to computing on the fly
        hourly_counts = burst_df.groupby("hour").size()
        if window in ("busiest_hour", "busiest"):
            hour = int(hourly_counts.idxmax())
        elif window in ("median_hour", "median"):
            hour = int((hourly_counts - hourly_counts.median()).abs().idxmin())
        else:
            hour = int(hourly_counts.idxmax())
        
        start_ts = float(hour * 3600)
        end_ts = float((hour + 1) * 3600)
        label = f"{window}_h{hour}"
        slice_df = burst_df[
            (burst_df["Timestamp"] >= start_ts) & (burst_df["Timestamp"] < end_ts)
        ].copy()
        return label, start_ts, end_ts, slice_df

    else:
        raise ValueError(f"Unsupported window specification: {window}")


def build_real_trace(
    window: Union[str, WindowMetadata, int] = "busiest_hour",
    seconds_per_step: float = 10.0,
    load_scale: float = 1.0,
    video_fraction: float = 0.2,
    seed: int = 42,
    df_burst: Optional[pd.DataFrame] = None,
    df_video: Optional[pd.DataFrame] = None,
    total_steps: Optional[int] = None,
    partition: str = "eval",
) -> List[List[EnhancedTask]]:
    """
    Build a discrete-time multi-modal workload trace directly from empirical datasets.

    Args:
        window: Window identifier (WindowMetadata, string name, or int hour).
        seconds_per_step: Seconds represented by each simulation step (default: 10s).
        load_scale: Multiplier on offered arrival intensity (default: 1.0).
        video_fraction: Probability [0.0, 1.0] that an arriving task is video transcoding (rest LLM).
        seed: Random seed for deterministic arrivals, sampling, and task IDs.
        df_burst: Optional pre-loaded BurstGPT DataFrame.
        df_video: Optional pre-loaded UCI Video DataFrame.
        total_steps: Optional override on trace step count. If None, derived from window duration.
        partition: Partition name ('eval' or 'train') when resolving string window names.

    Returns:
        List[List[EnhancedTask]]: Exact trace structure expected by GPUEnvironment.attach_workload_trace().
    """
    rng = random.Random(seed)

    # Resolve temporal window
    window_label, start_ts, end_ts, slice_df = resolve_window_data(
        window, df_burst=df_burst, partition=partition
    )

    duration_sec = end_ts - start_ts
    num_steps = total_steps or int(math.ceil(duration_sec / max(0.1, seconds_per_step)))

    # Initialize mappers with dedicated RNG
    video_df = df_video if df_video is not None else load_uci_video_data()
    video_mapper = VideoTaskMapper(seed=seed)
    llm_mapper = LLMTaskMapper(seed=seed)

    # Bin real request timestamps into simulation steps
    slice_df = slice_df.sort_values(by="Timestamp")
    rel_times = slice_df["Timestamp"].values - start_ts
    step_indices = np.clip((rel_times / seconds_per_step).astype(int), 0, num_steps - 1)

    # Group request rows by step index
    requests_by_step: Dict[int, List[pd.Series]] = {s: [] for s in range(num_steps)}
    for step_idx, (_, row) in zip(step_indices, slice_df.iterrows()):
        requests_by_step[step_idx].append(row)

    trace: List[List[EnhancedTask]] = []
    global_task_counter = 0

    for step in range(num_steps):
        raw_rows = requests_by_step[step]
        raw_count = len(raw_rows)

        # Rescale arrivals via load_scale preserving fractional expectations
        if raw_count > 0:
            expected_tasks = raw_count * load_scale
            int_tasks = int(expected_tasks)
            extra_prob = expected_tasks - int_tasks
            task_count = int_tasks + (1 if rng.random() < extra_prob else 0)
        else:
            task_count = 0

        step_tasks: List[EnhancedTask] = []

        for _ in range(task_count):
            global_task_counter += 1
            task_id = f"real_{window_label}_s{seed}_t{global_task_counter:04d}"

            # Multi-modal routing: video vs LLM
            is_video = (rng.random() < video_fraction)

            if is_video:
                # Deterministically sample from UCI video dataset
                vid_idx = rng.randint(0, len(video_df) - 1)
                vid_row = video_df.iloc[vid_idx]
                task = video_mapper.map_row(vid_row, task_id=task_id)
            else:
                # Sample from available BurstGPT rows in this step (or cycle)
                if raw_rows:
                    burst_row = raw_rows[rng.randint(0, len(raw_rows) - 1)]
                else:
                    burst_row = slice_df.iloc[rng.randint(0, len(slice_df) - 1)]
                task = llm_mapper.map_row(burst_row, task_id=task_id)

            step_tasks.append(task)

        trace.append(step_tasks)

    return trace


def summarize_trace(
    trace: List[List[EnhancedTask]],
    window_label: str,
    seconds_per_step: float,
    load_scale: float,
    video_fraction: float,
    seed: int,
) -> Dict[str, Any]:
    """
    Compute comprehensive statistics for a built trace.
    """
    total_steps = len(trace)
    step_arrivals = [len(step) for step in trace]
    all_tasks = [t for step in trace for t in step]
    total_tasks = len(all_tasks)

    if total_tasks == 0:
        return {
            "window_label": window_label,
            "total_tasks": 0,
            "total_steps": total_steps,
            "seconds_per_step": seconds_per_step,
            "load_scale": load_scale,
            "video_fraction": video_fraction,
            "seed": seed,
        }

    mems = [float(t.memory_required) for t in all_tasks]
    times = [int(t.execution_time) for t in all_tasks]

    video_tasks = sum(1 for t in all_tasks if getattr(t, "task_type", None) in (TaskType.VIDEO, "Video"))
    llm_tasks = total_tasks - video_tasks
    real_failed_tasks = sum(1 for t in all_tasks if getattr(t, "real_failed", False))

    empty_steps = sum(1 for count in step_arrivals if count == 0)

    return {
        "window_label": window_label,
        "total_tasks": total_tasks,
        "total_steps": total_steps,
        "seconds_per_step": seconds_per_step,
        "load_scale": load_scale,
        "video_fraction": video_fraction,
        "seed": seed,
        "video_tasks": video_tasks,
        "video_pct": round(video_tasks / total_tasks * 100, 2),
        "llm_tasks": llm_tasks,
        "llm_pct": round(llm_tasks / total_tasks * 100, 2),
        "real_failed_tasks": real_failed_tasks,
        "real_failed_pct": round(real_failed_tasks / total_tasks * 100, 2),
        "step_arrivals_max": int(np.max(step_arrivals)),
        "step_arrivals_mean": round(float(np.mean(step_arrivals)), 3),
        "step_arrivals_var": round(float(np.var(step_arrivals)), 3),
        "empty_steps_count": empty_steps,
        "empty_steps_pct": round(empty_steps / total_steps * 100, 2),
        "memory_min": round(float(np.min(mems)), 2),
        "memory_mean": round(float(np.mean(mems)), 2),
        "memory_median": round(float(np.median(mems)), 2),
        "memory_p95": round(float(np.percentile(mems, 95)), 2),
        "memory_max": round(float(np.max(mems)), 2),
        "duration_min": int(np.min(times)),
        "duration_mean": round(float(np.mean(times)), 2),
        "duration_median": int(np.median(times)),
        "duration_p95": int(np.percentile(times, 95)),
        "duration_max": int(np.max(times)),
    }


def write_trace_summary_report(
    summaries: List[Dict[str, Any]],
    output_path: Optional[Path] = None,
) -> Path:
    """
    Format trace summaries into trace_summary.md report.
    """
    dest = output_path or TRACE_SUMMARY_REPORT_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Real-Data Workload Trace Summary",
        "",
        "> **AetherGrid Trace Builder**: Empirical multi-modal workload traces calibrated to 4x8GB GPU cluster target utilization presets (0.6, 0.9, 1.2).",
        "",
        "## 1. Trace Generation Overview",
        "",
        "Workload traces are generated directly from real production traces:",
        "- **Arrival Timing**: Exact request timestamps from BurstGPT binned into simulation steps.",
        "- **Multi-Modal Mix**: Blended mix combining BurstGPT LLM inference requests with UCI Video transcoding jobs.",
        "- **Load Rescaling Formula**: Offered load is calibrated via queueing theory to reach target presets on the 4x8GB cluster:",
        "  $$\\rho = \\frac{\\lambda \\cdot \\bar{M} \\cdot \\bar{S}}{32.0 \\text{ GB}}, \\quad \\text{load\\_scale} = \\frac{\\rho_{\\text{target}}}{\\rho_{\\text{raw}}}$$",
        "",
        "---",
        "",
        "## 2. Calibrated Trace Catalog",
        "",
        "| Trace / Window | Target Load | Scale | Steps | Total Tasks | Video % | Real Fail % | Memory Mean (GB) | Dur Mean (Steps) | Peak Arrival/Step |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for s in summaries:
        target_str = f"rho={s.get('load_scale', 1.0)}"
        lines.append(
            f"| `{s['window_label']}` | `{target_str}` | {s['load_scale']:.4f} | "
            f"{s['total_steps']} | **{s['total_tasks']}** | {s.get('video_pct', 0.0)}% | "
            f"{s.get('real_failed_pct', 0.0)}% | {s.get('memory_mean', 0.0):.2f} | "
            f"{s.get('duration_mean', 0.0):.2f} | {s.get('step_arrivals_max', 0)} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 3. Detailed Trace Profiles",
        "",
    ])

    for s in summaries:
        lines.extend([
            f"### Trace: `{s['window_label']}` (Scale: {s['load_scale']:.4f}, Seed: {s['seed']})",
            "",
            f"- **Step Resolution**: {s['seconds_per_step']}s/step across **{s['total_steps']} steps** (Empty steps: {s.get('empty_steps_count', 0)} / {s.get('empty_steps_pct', 0.0)}%).",
            f"- **Arrival Volatility**: Mean {s.get('step_arrivals_mean', 0.0)} tasks/step, Variance {s.get('step_arrivals_var', 0.0)}, Max {s.get('step_arrivals_max', 0)} tasks/step.",
            f"- **Workload Composition**: {s.get('video_tasks', 0)} Video tasks ({s.get('video_pct', 0.0)}%) + {s.get('llm_tasks', 0)} LLM tasks ({s.get('llm_pct', 0.0)}%).",
            f"- **Real Fault Signal**: {s.get('real_failed_tasks', 0)} tasks ({s.get('real_failed_pct', 0.0)}%) originated from real BurstGPT requests aborted under server burst pressure.",
            f"- **VRAM Footprint**: Min {s.get('memory_min', 0.0)} GB, Mean **{s.get('memory_mean', 0.0)} GB**, Median {s.get('memory_median', 0.0)} GB, p95 {s.get('memory_p95', 0.0)} GB, Max **{s.get('memory_max', 0.0)} GB**.",
            f"- **Duration**: Min {s.get('duration_min', 0)} steps, Mean **{s.get('duration_mean', 0.0)} steps**, Median {s.get('duration_median', 0)} steps, p95 {s.get('duration_p95', 0)} steps, Max **{s.get('duration_max', 0)} steps**.",
            "",
        ])

    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[TraceBuilder] Summary report written to {dest}")
    return dest


def build_canonical_trace_catalog(
    output_report: bool = True,
    seconds_per_step: float = 10.0,
    seed: int = 42,
) -> Dict[str, List[List[EnhancedTask]]]:
    """
    Build canonical traces covering presets across canonical windows.
    """
    df_burst = load_burstgpt_data()
    df_video = load_uci_video_data()

    windows_summary = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    eval_windows = windows_summary.eval_windows

    traces: Dict[str, List[List[EnhancedTask]]] = {}
    summaries: List[Dict[str, Any]] = []

    # Build across targets 0.6, 0.9, 1.2
    for win_key, meta in [
        ("median_hour", eval_windows["median_hour"]),
        ("burstiest_window", eval_windows["burstiest_window"]),
    ]:
        for target_util in [0.6, 0.9, 1.2]:
            scale = compute_target_load_scale(
                target_utilization=target_util,
                raw_requests_count=meta.request_count,
                window_duration_seconds=meta.duration_seconds,
                seconds_per_step=seconds_per_step,
            )
            trace_id = f"eval_{win_key}_rho{int(target_util * 100)}"
            trace = build_real_trace(
                window=meta,
                seconds_per_step=seconds_per_step,
                load_scale=scale,
                video_fraction=0.2,
                seed=seed,
                df_burst=df_burst,
                df_video=df_video,
            )
            traces[trace_id] = trace
            summary = summarize_trace(
                trace=trace,
                window_label=trace_id,
                seconds_per_step=seconds_per_step,
                load_scale=scale,
                video_fraction=0.2,
                seed=seed,
            )
            summaries.append(summary)

    if output_report:
        write_trace_summary_report(summaries)

    return traces


if __name__ == "__main__":
    build_canonical_trace_catalog()
