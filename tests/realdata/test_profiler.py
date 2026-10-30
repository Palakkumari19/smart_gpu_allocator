"""
Unit tests for realdata.profile_data on tiny fixtures.
"""

import pytest
import pandas as pd
from pathlib import Path

from realdata.profile_data import (
    compute_percentiles,
    profile_uci_video,
    profile_burstgpt,
    generate_data_profile_report,
)


def test_compute_percentiles():
    """Verify compute_percentiles computes standard cuts accurately."""
    # Series with values 1 to 100
    s = pd.Series(range(1, 101))
    pcts = compute_percentiles(s)
    assert set(pcts.keys()) == {"p5", "p25", "p50", "p75", "p95", "p99"}
    assert pytest.approx(pcts["p50"], 0.5) == 50.5
    assert pytest.approx(pcts["p25"], 0.5) == 25.75
    assert pytest.approx(pcts["p75"], 0.5) == 75.25


def test_compute_percentiles_empty():
    """Verify compute_percentiles returns zeros for empty series."""
    s = pd.Series([], dtype=float)
    pcts = compute_percentiles(s)
    assert all(val == 0.0 for val in pcts.values())


def test_profile_uci_video():
    """Verify profile_uci_video produces correct metrics on mock dataframe."""
    df = pd.DataFrame({
        "id": [f"vid_{i}" for i in range(10)],
        "duration": [10.0 * (i + 1) for i in range(10)],
        "codec": ["h264"] * 6 + ["vp8"] * 4,
        "width": [1920] * 5 + [1280] * 5,
        "height": [1080] * 5 + [720] * 5,
        "resolution": ["1920x1080"] * 5 + ["1280x720"] * 5,
        "umem": [100000 + i * 10000 for i in range(10)],
        "utime": [1.0 + i * 0.5 for i in range(10)],
    })
    res = profile_uci_video(df)
    assert res["total_rows"] == 10
    assert len(res["resolutions"]) == 2
    assert res["codecs"] == {"h264": 6, "vp8": 4}
    assert "p50" in res["umem_percentiles_kb"]
    assert "p50" in res["utime_percentiles_sec"]


def test_profile_burstgpt():
    """Verify profile_burstgpt computes failure rates and hourly metrics accurately."""
    df = pd.DataFrame({
        "Timestamp": [100, 200, 3700, 3800],
        "Model": ["ChatGPT", "ChatGPT", "GPT-4", "GPT-4"],
        "Request tokens": [100, 200, 300, 400],
        "Response tokens": [50, 0, 80, 0],  # 2 failed out of 4 (50%)
        "Total tokens": [150, 200, 380, 400],
        "is_failed": [False, True, False, True],
        "hour": [0, 0, 1, 1],
    })
    res = profile_burstgpt(df)
    assert res["total_rows"] == 4
    assert res["failed_rows"] == 2
    assert res["failure_rate_pct"] == 50.0
    assert res["hourly_summary"]["total_hours"] == 2
    assert res["models"] == {"ChatGPT": 2, "GPT-4": 2}


def test_generate_data_profile_report(tmp_path: Path, monkeypatch):
    """Verify report generation writes markdown document to specified destination."""
    mock_uci = pd.DataFrame({
        "id": ["vid_1"],
        "duration": [60.0],
        "codec": ["h264"],
        "width": [1920],
        "height": [1080],
        "bitrate": [5000000],
        "framerate": [30],
        "umem": [200000],
        "utime": [4.0],
        "resolution": ["1920x1080"],
        "resolution_label": ["1080p"],
    })
    mock_burst = pd.DataFrame({
        "Timestamp": [100],
        "Model": ["ChatGPT"],
        "Request tokens": [150],
        "Response tokens": [50],
        "Total tokens": [200],
        "is_failed": [False],
        "hour": [0],
    })

    monkeypatch.setattr("realdata.profile_data.load_uci_video_data", lambda: mock_uci)
    monkeypatch.setattr("realdata.profile_data.load_burstgpt_data", lambda: mock_burst)

    report_out = tmp_path / "test_report.md"
    generated_path = generate_data_profile_report(output_path=report_out)
    assert generated_path.exists()
    content = generated_path.read_text()
    assert "# Empirical Trace Data Profile" in content
    assert "UCI Transcoding Dataset" in content
    assert "BurstGPT Trace Analysis" in content
    assert "gaming_gpu_dataset_3000" in content
