"""
Unit tests for realdata.llm_mapper using tiny fixtures.
Verifies token quantile mapping, side-table tracking of real_failed, bounds, and calibration round-trip.
"""

import pytest
import pandas as pd
from pathlib import Path

from realdata.llm_mapper import (
    LLMCalibration,
    LLMTaskMapper,
)
from workload.task_generator import EnhancedTask, TaskType


@pytest.fixture
def mock_llm_calibration() -> LLMCalibration:
    """Create a synthetic LLMCalibration instance."""
    quantiles = [i / 10.0 for i in range(11)]
    # Total tokens from 50 to 4000
    tot_tokens = [50 + i * 395 for i in range(11)]
    # Response tokens from 0 to 1500
    resp_tokens = [i * 150 for i in range(11)]

    return LLMCalibration(
        dataset_name="Mock LLM Calibration",
        sample_count=500,
        num_points=11,
        quantiles=quantiles,
        total_tokens_thresholds=tot_tokens,
        response_tokens_thresholds=resp_tokens,
        memory_min_gb=1.0,
        memory_max_gb=8.0,
        steps_min=1,
        steps_max=20,
    )


@pytest.fixture
def mock_llm_rows():
    """Sample BurstGPT rows including failed (0 response tokens) and normal requests."""
    return [
        {
            "Timestamp": 100,
            "Model": "ChatGPT",
            "Request tokens": 80,
            "Response tokens": 0,    # Failed request
            "Total tokens": 80,
        },
        {
            "Timestamp": 250,
            "Model": "ChatGPT",
            "Request tokens": 200,
            "Response tokens": 50,   # Normal request
            "Total tokens": 250,
        },
        {
            "Timestamp": 500,
            "Model": "GPT-4",
            "Request tokens": 1500,
            "Response tokens": 800,  # Heavy request
            "Total tokens": 2300,
        },
    ]


def test_llm_mapper_real_failed_and_sidetable(mock_llm_calibration: LLMCalibration, mock_llm_rows):
    """Verify real_failed is tracked accurately in side-table and attached to EnhancedTask."""
    mapper = LLMTaskMapper(calibration=mock_llm_calibration, seed=42)

    task_failed = mapper.map_row(mock_llm_rows[0], task_id="task_fail_01")
    task_success = mapper.map_row(mock_llm_rows[1], task_id="task_succ_01")

    # Verify side-table entries
    side_table = mapper.get_failure_side_table()
    assert side_table["task_fail_01"] is True
    assert side_table["task_succ_01"] is False

    # Verify task attributes attached directly to EnhancedTask
    assert task_failed.real_failed is True
    assert task_success.real_failed is False
    assert task_failed.task_type == TaskType.LLM
    assert task_success.task_type == TaskType.LLM


def test_llm_mapper_determinism(mock_llm_calibration: LLMCalibration, mock_llm_rows):
    """Verify identical seeds produce identical tasks and side-tables."""
    mapper1 = LLMTaskMapper(calibration=mock_llm_calibration, seed=123)
    mapper2 = LLMTaskMapper(calibration=mock_llm_calibration, seed=123)

    for row in mock_llm_rows:
        t1 = mapper1.map_row(row)
        t2 = mapper2.map_row(row)

        assert t1.id == t2.id
        assert t1.memory_required == t2.memory_required
        assert t1.execution_time == t2.execution_time
        assert t1.real_failed == t2.real_failed
        assert t1.priority == t2.priority

    assert mapper1.get_failure_side_table() == mapper2.get_failure_side_table()


def test_llm_mapper_bounds_and_rank_order(mock_llm_calibration: LLMCalibration):
    """Verify memory is bounded to [1.0, 8.0] GB, steps to [1, 20], and rank order holds."""
    mapper = LLMTaskMapper(calibration=mock_llm_calibration, seed=99)

    token_levels = [10, 100, 500, 1500, 3000, 50000]
    mem_outputs = []
    step_outputs = []

    for tokens in token_levels:
        row = {
            "Timestamp": 1000,
            "Model": "ChatGPT",
            "Request tokens": tokens // 2,
            "Response tokens": tokens // 2,
            "Total tokens": tokens,
        }
        task = mapper.map_row(row)
        assert 1.0 <= task.memory_required <= 8.0
        assert 1 <= task.execution_time <= 20
        mem_outputs.append(task.memory_required)
        step_outputs.append(task.execution_time)

    # Monotonicity check
    for i in range(len(mem_outputs) - 1):
        assert mem_outputs[i] <= mem_outputs[i + 1]
        assert step_outputs[i] <= step_outputs[i + 1]


def test_llm_calibration_round_trip(mock_llm_calibration: LLMCalibration, tmp_path: Path, mock_llm_rows):
    """Verify LLMCalibration serializes to JSON and round-trips identically."""
    json_path = tmp_path / "test_llm_cal.json"
    mock_llm_calibration.to_json(json_path)
    assert json_path.exists()

    loaded_cal = LLMCalibration.from_json(json_path)
    assert loaded_cal.to_dict() == mock_llm_calibration.to_dict()

    m_orig = LLMTaskMapper(calibration=mock_llm_calibration, seed=77)
    m_loaded = LLMTaskMapper(calibration=loaded_cal, seed=77)

    for row in mock_llm_rows:
        t1 = m_orig.map_row(row)
        t2 = m_loaded.map_row(row)
        assert t1.memory_required == t2.memory_required
        assert t1.execution_time == t2.execution_time
        assert t1.real_failed == t2.real_failed
