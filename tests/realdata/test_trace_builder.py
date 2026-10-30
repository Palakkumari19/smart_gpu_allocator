"""
Unit tests for realdata.trace_builder.
Verifies determinism, 8.0 GB memory invariant, load scale computation,
deterministic ID formatting, and integration with GPUEnvironment.attach_workload_trace().
"""

import pytest
from pathlib import Path

from realdata.trace_builder import (
    build_real_trace,
    compute_target_load_scale,
    summarize_trace,
    write_trace_summary_report,
)
from environment import GPUEnvironment
from workload.task_generator import EnhancedTask, TaskType


@pytest.fixture
def tiny_synthetic_data(tmp_path: Path):
    """Create tiny mock BurstGPT and UCI Video DataFrames for fast isolated testing."""
    import pandas as pd

    # Mock BurstGPT (10 requests spanning 100 seconds)
    burst_df = pd.DataFrame({
        "Timestamp": [i * 10 for i in range(10)],
        "Model": ["ChatGPT"] * 8 + ["GPT-4"] * 2,
        "Request tokens": [100 + i * 20 for i in range(10)],
        "Response tokens": [50 if i != 3 else 0 for i in range(10)],  # 1 failure
        "Total tokens": [150 + i * 20 for i in range(10)],
        "is_failed": [i == 3 for i in range(10)],
        "hour": [0] * 10,
    })

    # Mock UCI Video (5 rows)
    video_df = pd.DataFrame({
        "id": [f"vid_{i}" for i in range(5)],
        "duration": [60.0 + i * 10 for i in range(5)],
        "codec": ["h264"] * 3 + ["vp8"] * 2,
        "width": [1920] * 3 + [1280] * 2,
        "height": [1080] * 3 + [720] * 2,
        "bitrate": [5000000.0] * 5,
        "framerate": [30.0] * 5,
        "umem": [200000 + i * 10000 for i in range(5)],
        "utime": [4.0 + i for i in range(5)],
        "resolution": ["1920x1080"] * 3 + ["1280x720"] * 2,
        "resolution_label": ["1080p"] * 3 + ["720p"] * 2,
    })

    return burst_df, video_df


def test_trace_builder_determinism(tiny_synthetic_data):
    """Verify that calling build_real_trace with the same seed yields identical traces."""
    burst_df, video_df = tiny_synthetic_data

    trace1 = build_real_trace(
        window=0,
        seconds_per_step=10.0,
        load_scale=1.0,
        video_fraction=0.3,
        seed=42,
        df_burst=burst_df,
        df_video=video_df,
        total_steps=10,
    )

    trace2 = build_real_trace(
        window=0,
        seconds_per_step=10.0,
        load_scale=1.0,
        video_fraction=0.3,
        seed=42,
        df_burst=burst_df,
        df_video=video_df,
        total_steps=10,
    )

    assert len(trace1) == len(trace2) == 10

    for step_idx in range(len(trace1)):
        tasks1 = trace1[step_idx]
        tasks2 = trace2[step_idx]
        assert len(tasks1) == len(tasks2)

        for t1, t2 in zip(tasks1, tasks2):
            assert t1.id == t2.id
            assert t1.memory_required == t2.memory_required
            assert t1.execution_time == t2.execution_time
            assert t1.task_type == t2.task_type
            assert t1.priority == t2.priority


def test_no_task_above_8gb(tiny_synthetic_data):
    """Verify that every task in the built trace adheres strictly to the 8.0 GB bound."""
    burst_df, video_df = tiny_synthetic_data

    trace = build_real_trace(
        window=0,
        seconds_per_step=5.0,
        load_scale=2.0,  # Scaled up
        video_fraction=0.5,
        seed=999,
        df_burst=burst_df,
        df_video=video_df,
        total_steps=20,
    )

    all_tasks = [t for step in trace for t in step]
    assert len(all_tasks) > 0

    for task in all_tasks:
        assert isinstance(task, EnhancedTask)
        assert 1.0 <= task.memory_required <= 8.0, f"Memory {task.memory_required} GB violates [1.0, 8.0]"
        assert 1 <= task.execution_time <= 20, f"Steps {task.execution_time} violates [1, 20]"


def test_load_scale_computation():
    """Verify compute_target_load_scale scales proportionally to target utilization."""
    scale_60 = compute_target_load_scale(target_utilization=0.6, raw_requests_count=1000)
    scale_120 = compute_target_load_scale(target_utilization=1.2, raw_requests_count=1000)

    # Doubling target utilization must exactly double the load_scale factor
    assert pytest.approx(scale_120, rel=1e-4) == 2.0 * scale_60

    # Inversely, doubling raw requests must halve the load_scale factor
    scale_double_requests = compute_target_load_scale(target_utilization=0.6, raw_requests_count=2000)
    assert pytest.approx(scale_double_requests, rel=1e-4) == 0.5 * scale_60


def test_deterministic_task_id_format(tiny_synthetic_data):
    """Verify generated task IDs conform to the real_{window}_s{seed}_t{index:04d} convention."""
    burst_df, video_df = tiny_synthetic_data

    trace = build_real_trace(
        window=0,
        seconds_per_step=10.0,
        load_scale=1.0,
        seed=77,
        df_burst=burst_df,
        df_video=video_df,
        total_steps=5,
    )

    all_tasks = [t for step in trace for t in step]
    assert len(all_tasks) > 0

    for idx, task in enumerate(all_tasks, start=1):
        assert task.id.startswith("real_")
        assert "_s77_t" in task.id
        assert task.id.endswith(f"_t{idx:04d}")


def test_structure_accepted_by_gpu_environment_smoke_run(tiny_synthetic_data):
    """
    Smoke test: verify that GPUEnvironment.attach_workload_trace() accepts
    the built real trace and steps cleanly through simulation.
    """
    burst_df, video_df = tiny_synthetic_data

    # Build a 15-step trace
    trace = build_real_trace(
        window=0,
        seconds_per_step=10.0,
        load_scale=1.0,
        video_fraction=0.3,
        seed=101,
        df_burst=burst_df,
        df_video=video_df,
        total_steps=15,
    )

    env = GPUEnvironment(scheduler="fcfs")
    env.enable_deterministic_physics()
    env.set_eval_mode()

    # Must accept structure with zero errors or schema mismatches
    env.attach_workload_trace(trace)

    # Step through simulation
    for _ in range(len(trace)):
        env.step()

    assert env.time_step == len(trace) == 15
    metrics = env.get_metrics()
    assert "completed" in metrics
    assert "latency" in metrics
    assert "average_gpu_risk" in metrics


def test_summarize_trace_and_report_writing(tiny_synthetic_data, tmp_path: Path):
    """Verify trace summarizer and report generation."""
    burst_df, video_df = tiny_synthetic_data

    trace = build_real_trace(
        window=0,
        seconds_per_step=10.0,
        load_scale=1.0,
        video_fraction=0.4,
        seed=42,
        df_burst=burst_df,
        df_video=video_df,
        total_steps=10,
    )

    summary = summarize_trace(
        trace=trace,
        window_label="test_window",
        seconds_per_step=10.0,
        load_scale=1.0,
        video_fraction=0.4,
        seed=42,
    )

    assert summary["total_steps"] == 10
    assert summary["total_tasks"] > 0
    assert "video_pct" in summary
    assert "memory_mean" in summary
    assert "duration_mean" in summary

    out_file = tmp_path / "test_trace_summary.md"
    written_path = write_trace_summary_report([summary], output_path=out_file)
    assert written_path.exists()
    content = written_path.read_text()
    assert "# Real-Data Workload Trace Summary" in content
    assert "test_window" in content
