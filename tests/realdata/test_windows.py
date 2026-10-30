"""
Unit tests for realdata.windows using tiny fixtures.
Verifies window extraction, index of dispersion calculation, disjoint train/eval splits, and JSON serialization.
"""

import pytest
import pandas as pd
from pathlib import Path

from realdata.windows import (
    compute_hourly_dispersion,
    select_partition_windows,
    extract_burstgpt_windows,
    extract_window_dataframe,
    TraceWindowsSummary,
)


@pytest.fixture
def mock_burst_df() -> pd.DataFrame:
    """
    Create a synthetic multi-hour BurstGPT DataFrame with distinct patterns:
      Hour 0: Low steady traffic (10 reqs)
      Hour 1: Medium traffic (30 reqs evenly spaced)
      Hour 2: Burst traffic (50 reqs all clustered in 2 seconds -> high dispersion)
      Hour 3: Eval low (15 reqs)
      Hour 4: Eval busy (100 reqs)
      Hour 5: Eval bursty (60 reqs clustered in 3 seconds -> high dispersion, with 50% failures)
    """
    records = []
    # Hour 0
    for i in range(10):
        records.append({"Timestamp": 0 * 3600 + i * 300, "Model": "ChatGPT", "Request tokens": 100, "Response tokens": 50, "is_failed": False, "hour": 0})
    # Hour 1
    for i in range(30):
        records.append({"Timestamp": 1 * 3600 + i * 100, "Model": "ChatGPT", "Request tokens": 120, "Response tokens": 60, "is_failed": False, "hour": 1})
    # Hour 2 (Burst: 50 requests all at timestamps 7205 and 7206)
    for i in range(50):
        records.append({"Timestamp": 2 * 3600 + (5 if i < 25 else 6), "Model": "GPT-4", "Request tokens": 200, "Response tokens": 80, "is_failed": False, "hour": 2})
    # Hour 3
    for i in range(15):
        records.append({"Timestamp": 3 * 3600 + i * 200, "Model": "ChatGPT", "Request tokens": 100, "Response tokens": 40, "is_failed": False, "hour": 3})
    # Hour 4 (Busiest eval: 100 requests)
    for i in range(100):
        records.append({"Timestamp": 4 * 3600 + i * 30, "Model": "ChatGPT", "Request tokens": 150, "Response tokens": 50, "is_failed": (i % 10 == 0), "hour": 4})
    # Hour 5 (Burst eval: 60 requests clustered at 18010, half failed)
    for i in range(60):
        records.append({"Timestamp": 5 * 3600 + 10, "Model": "GPT-4", "Request tokens": 300, "Response tokens": (0 if i < 30 else 100), "is_failed": (i < 30), "hour": 5})

    return pd.DataFrame(records)


def test_compute_hourly_dispersion(mock_burst_df: pd.DataFrame):
    """Verify that clustered burst arrivals produce significantly higher dispersion."""
    disp = compute_hourly_dispersion(mock_burst_df)

    # Clustered Hour 2 must have much higher dispersion than spread-out Hour 1
    assert disp[2] > disp[1]
    # Clustered Hour 5 must have higher dispersion than spread-out Hour 4
    assert disp[5] > disp[4]


def test_select_partition_windows(mock_burst_df: pd.DataFrame):
    """Verify selection of median, busiest, and burstiest windows in a partition."""
    train_df = mock_burst_df[mock_burst_df["hour"] <= 2]
    windows = select_partition_windows(train_df, partition_name="train")

    assert set(windows.keys()) == {"median_hour", "busiest_hour", "burstiest_window"}

    # Busiest hour in train is Hour 2 (50 requests)
    assert windows["busiest_hour"].hour == 2
    assert windows["busiest_hour"].request_count == 50

    # Burstiest window in train is Hour 2 (all clustered in 2 seconds)
    assert windows["burstiest_window"].hour == 2
    assert windows["burstiest_window"].dispersion_index > 1.0


def test_disjoint_train_eval_windows(mock_burst_df: pd.DataFrame):
    """Verify that TRAIN and EVAL windows are strictly non-overlapping in hours and timestamps."""
    summary = extract_burstgpt_windows(mock_burst_df, train_split_ratio=0.5)

    train_h_range = summary.train_hours_range
    eval_h_range = summary.eval_hours_range

    # Disjoint check: train max hour < eval min hour
    assert train_h_range[1] < eval_h_range[0]

    for train_win in summary.train_windows.values():
        for eval_win in summary.eval_windows.values():
            # Timestamps must never overlap
            assert train_win.end_timestamp <= eval_win.start_timestamp
            assert train_win.hour < eval_win.hour


def test_extract_window_dataframe(mock_burst_df: pd.DataFrame):
    """Verify extracting a slice of the DataFrame matching a window."""
    summary = extract_burstgpt_windows(mock_burst_df, train_split_ratio=0.5)
    busiest_eval_win = summary.eval_windows["busiest_hour"]

    slice_df = extract_window_dataframe(mock_burst_df, busiest_eval_win)
    assert len(slice_df) == busiest_eval_win.request_count
    assert (slice_df["hour"] == busiest_eval_win.hour).all()


def test_trace_windows_summary_round_trip(mock_burst_df: pd.DataFrame, tmp_path: Path):
    """Verify TraceWindowsSummary JSON serialization and deserialization."""
    summary = extract_burstgpt_windows(mock_burst_df, train_split_ratio=0.5)
    json_path = tmp_path / "test_windows.json"
    summary.to_json(json_path)
    assert json_path.exists()

    loaded = TraceWindowsSummary.from_json(json_path)
    assert loaded.train_hours_range == summary.train_hours_range
    assert loaded.eval_hours_range == summary.eval_hours_range
    assert loaded.train_windows["busiest_hour"].hour == summary.train_windows["busiest_hour"].hour
    assert loaded.eval_windows["burstiest_window"].failure_rate_pct == summary.eval_windows["burstiest_window"].failure_rate_pct
