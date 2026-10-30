"""
Unit tests for realdata.video_mapper.
Verifies quantile mapping, determinism, memory bounds, rank-order preservation, and calibration round-trip.
"""

import pytest
import pandas as pd
from pathlib import Path

from realdata.video_mapper import (
    VideoCalibration,
    VideoTaskMapper,
)
from workload.task_generator import EnhancedTask, TaskType


@pytest.fixture
def mock_calibration() -> VideoCalibration:
    """Create a calibrated VideoCalibration instance from synthetic points."""
    # Synthetic quantile grid of 11 points (0.0, 0.1, ..., 1.0)
    quantiles = [i / 10.0 for i in range(11)]
    # log(umem) range from 10.0 (approx 22,000 KB) to 14.0 (approx 1,200,000 KB)
    log_umem = [10.0 + i * 0.4 for i in range(11)]
    # utime range from 0.5s to 100.0s
    utime = [0.5 + i * 9.95 for i in range(11)]

    return VideoCalibration(
        dataset_name="Test Calibration",
        sample_count=1000,
        num_points=11,
        quantiles=quantiles,
        log_umem_thresholds=log_umem,
        utime_thresholds=utime,
        memory_min_gb=1.0,
        memory_max_gb=8.0,
        steps_min=1,
        steps_max=20,
    )


@pytest.fixture
def sample_rows():
    """Sample video rows with varying memory and time characteristics."""
    return [
        {
            "id": "vid_light",
            "umem": 25000,       # Low memory (~10.1 log)
            "utime": 1.0,        # Fast transcode
            "duration": 30.0,
            "codec": "h264",
            "width": 640,
            "height": 480,
            "framerate": 24.0,
            "bitrate": 800000.0,
        },
        {
            "id": "vid_medium",
            "umem": 200000,      # Medium memory (~12.2 log)
            "utime": 25.0,       # Moderate transcode
            "duration": 120.0,
            "codec": "vp8",
            "width": 1280,
            "height": 720,
            "framerate": 30.0,
            "bitrate": 2500000.0,
        },
        {
            "id": "vid_heavy",
            "umem": 900000,      # High memory (~13.7 log)
            "utime": 80.0,       # Slow transcode
            "duration": 300.0,
            "codec": "mpeg4",
            "width": 1920,
            "height": 1080,
            "framerate": 60.0,
            "bitrate": 8000000.0,
        },
    ]


def test_determinism_with_seed(mock_calibration: VideoCalibration, sample_rows):
    """Verify that mapping with the same seed produces identical EnhancedTask outputs."""
    mapper1 = VideoTaskMapper(calibration=mock_calibration, seed=42)
    mapper2 = VideoTaskMapper(calibration=mock_calibration, seed=42)

    for row in sample_rows:
        task1 = mapper1.map_row(row)
        task2 = mapper2.map_row(row)

        assert task1.id == task2.id
        assert task1.memory_required == task2.memory_required
        assert task1.execution_time == task2.execution_time
        assert task1.priority == task2.priority
        assert task1.workload_severity == task2.workload_severity
        assert task1.resolution == task2.resolution
        assert task1.codec == task2.codec


def test_memory_never_exceeds_8gb(mock_calibration: VideoCalibration):
    """Verify that mapped memory_required is strictly clamped to [1.0, 8.0] GB."""
    mapper = VideoTaskMapper(calibration=mock_calibration, seed=123)

    extreme_cases = [
        {"umem": 0.001, "utime": 0.01, "codec": "h264"},        # Extreme low
        {"umem": 10.0, "utime": 0.1, "codec": "h264"},          # Very low
        {"umem": 220000, "utime": 10.0, "codec": "h264"},       # Normal
        {"umem": 5000000, "utime": 500.0, "codec": "h264"},     # Very high
        {"umem": 1e12, "utime": 1e6, "codec": "h264"},          # Huge outlier
    ]

    for row in extreme_cases:
        task = mapper.map_row(row)
        assert isinstance(task, EnhancedTask)
        assert task.task_type == TaskType.VIDEO
        assert 1.0 <= task.memory_required <= 8.0, f"Memory {task.memory_required} GB exceeded [1.0, 8.0]"
        assert 1 <= task.execution_time <= 20, f"Steps {task.execution_time} exceeded [1, 20]"


def test_rank_order_preserved(mock_calibration: VideoCalibration):
    """
    Verify quantile mapping preserves strict monotonicity (rank order):
    higher umem -> higher or equal memory_required,
    higher utime -> higher or equal execution_steps.
    """
    mapper = VideoTaskMapper(calibration=mock_calibration, seed=999)

    # Monotonically increasing sequence of umem and utime
    umem_values = [5000, 20000, 50000, 100000, 250000, 500000, 1000000]
    utime_values = [0.2, 1.0, 5.0, 15.0, 30.0, 60.0, 120.0]

    memory_outputs = []
    step_outputs = []

    for u, t in zip(umem_values, utime_values):
        task = mapper.map_row({"umem": u, "utime": t, "codec": "h264"})
        memory_outputs.append(task.memory_required)
        step_outputs.append(task.execution_time)

    # Verify non-decreasing order
    for i in range(len(memory_outputs) - 1):
        assert memory_outputs[i] <= memory_outputs[i + 1], (
            f"Memory rank violated at index {i}: {memory_outputs[i]} > {memory_outputs[i + 1]}"
        )
        assert step_outputs[i] <= step_outputs[i + 1], (
            f"Step rank violated at index {i}: {step_outputs[i]} > {step_outputs[i + 1]}"
        )


def test_calibration_file_round_trips(mock_calibration: VideoCalibration, tmp_path: Path, sample_rows):
    """Verify VideoCalibration serializes to JSON and deserializes identically."""
    json_path = tmp_path / "test_calibration.json"
    saved_path = mock_calibration.to_json(json_path)
    assert saved_path.exists()

    loaded_cal = VideoCalibration.from_json(json_path)

    # Dictionary representation must be identical
    assert loaded_cal.to_dict() == mock_calibration.to_dict()

    # Mappings using loaded calibration must match original exactly
    mapper_orig = VideoTaskMapper(calibration=mock_calibration, seed=77)
    mapper_loaded = VideoTaskMapper(calibration=loaded_cal, seed=77)

    for row in sample_rows:
        t_orig = mapper_orig.map_row(row)
        t_loaded = mapper_loaded.map_row(row)
        assert t_orig.memory_required == t_loaded.memory_required
        assert t_orig.execution_time == t_loaded.execution_time
        assert t_orig.priority == t_loaded.priority
        assert t_orig.workload_severity == t_loaded.workload_severity


def test_enhanced_task_attributes_and_constructor(mock_calibration: VideoCalibration, sample_rows):
    """Verify tasks are built as EnhancedTask using its constructor with video attributes attached."""
    mapper = VideoTaskMapper(calibration=mock_calibration, seed=42)
    row = sample_rows[0]
    task = mapper.map_row(row, task_id="custom_task_001")

    assert task.id == "custom_task_001"
    assert task.task_type == TaskType.VIDEO
    assert task.resolution == "640x480"
    assert task.width == 640
    assert task.height == 480
    assert task.codec == "h264"
    assert task.fps == 24.0
    assert task.bitrate == "800.0 kbps"
    assert task.duration == 30.0
    assert task.characteristics["interruptible"] is True
