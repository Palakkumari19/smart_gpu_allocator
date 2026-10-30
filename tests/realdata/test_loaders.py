"""
Unit tests for realdata.loaders on tiny fixture files.
Does not load or depend on real production datasets.
"""

import pytest
import pandas as pd
from pathlib import Path

from realdata.loaders import load_uci_video_data, load_burstgpt_data


@pytest.fixture
def mock_uci_tsv(tmp_path: Path) -> Path:
    """Create a tiny valid UCI Video TSV fixture."""
    data = {
        "id": ["vid_001", "vid_002", "vid_003"],
        "duration": [120.5, 45.0, 300.2],
        "codec": ["h264", "vp8", "mpeg4"],
        "width": [1920, 1280, 640],
        "height": [1080, 720, 480],
        "bitrate": [5000000, 2500000, 1000000],
        "framerate": [30, 60, 24],
        "i": [10, 5, 20],
        "p": [100, 50, 200],
        "b": [0, 0, 0],
        "frames": [110, 55, 220],
        "i_size": [5000, 2500, 1000],
        "p_size": [10000, 5000, 2000],
        "b_size": [0, 0, 0],
        "size": [15000, 7500, 3000],
        "o_codec": ["h264", "vp8", "mpeg4"],
        "o_bitrate": [4000000, 2000000, 800000],
        "o_framerate": [30, 60, 24],
        "o_width": [1920, 1280, 640],
        "o_height": [1080, 720, 480],
        "umem": [220000, 180000, 140000],
        "utime": [5.5, 2.3, 12.1],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "mock_uci.tsv"
    df.to_csv(file_path, sep="\t", index=False)
    return file_path


@pytest.fixture
def mock_burstgpt_csv(tmp_path: Path) -> Path:
    """Create a tiny valid BurstGPT CSV fixture."""
    data = {
        "Timestamp": [10, 15, 3610, 3620, 7200],
        "Model": ["ChatGPT", "ChatGPT", "GPT-4", "ChatGPT", "GPT-4"],
        "Request tokens": [100, 250, 500, 50, 1200],
        "Response tokens": [50, 0, 120, 25, 0],  # 2 failed (0 tokens)
        "Total tokens": [150, 250, 620, 75, 1200],
        "Log Type": ["Conversation", "Conversation", "API", "Conversation", "API"],
    }
    df = pd.DataFrame(data)
    file_path = tmp_path / "mock_burstgpt.csv"
    df.to_csv(file_path, index=False)
    return file_path


def test_load_uci_video_data_success(mock_uci_tsv: Path):
    """Verify loading valid mock UCI TSV adds derived resolution columns."""
    df = load_uci_video_data(mock_uci_tsv)
    assert len(df) == 3
    assert "resolution" in df.columns
    assert "resolution_label" in df.columns
    assert df["resolution"].tolist() == ["1920x1080", "1280x720", "640x480"]
    assert df["resolution_label"].tolist() == ["1080p", "720p", "480p"]
    assert df["umem"].iloc[0] == 220000
    assert df["utime"].iloc[0] == 5.5


def test_load_uci_video_data_schema_mismatch(tmp_path: Path):
    """Verify ValueError is raised when required columns are missing in UCI dataset."""
    bad_df = pd.DataFrame({
        "id": ["vid_001"],
        "duration": [60.0],
        # Missing width, height, umem, utime, etc.
    })
    bad_path = tmp_path / "bad_uci.tsv"
    bad_df.to_csv(bad_path, sep="\t", index=False)

    with pytest.raises(ValueError) as excinfo:
        load_uci_video_data(bad_path)
    assert "Schema mismatch for UCI Video Transcoding" in str(excinfo.value)
    assert "Missing required columns" in str(excinfo.value)
    assert "umem" in str(excinfo.value)


def test_load_uci_video_data_file_not_found(tmp_path: Path):
    """Verify FileNotFoundError is raised when file does not exist."""
    non_existent = tmp_path / "does_not_exist.tsv"
    with pytest.raises(FileNotFoundError):
        load_uci_video_data(non_existent)


def test_load_burstgpt_data_success(mock_burstgpt_csv: Path):
    """Verify loading valid mock BurstGPT CSV adds is_failed and hour columns."""
    df = load_burstgpt_data(mock_burstgpt_csv)
    assert len(df) == 5
    assert "is_failed" in df.columns
    assert "hour" in df.columns
    # Check failure mapping: index 1 and index 4 have 0 response tokens
    assert df["is_failed"].tolist() == [False, True, False, False, True]
    # Check hour mapping (Timestamp // 3600)
    assert df["hour"].tolist() == [0, 0, 1, 1, 2]


def test_load_burstgpt_data_schema_mismatch(tmp_path: Path):
    """Verify ValueError is raised when required columns are missing in BurstGPT dataset."""
    bad_df = pd.DataFrame({
        "Timestamp": [100, 200],
        "Model": ["ChatGPT", "ChatGPT"],
        # Missing "Request tokens", "Response tokens", "Total tokens"
    })
    bad_path = tmp_path / "bad_burstgpt.csv"
    bad_df.to_csv(bad_path, index=False)

    with pytest.raises(ValueError) as excinfo:
        load_burstgpt_data(bad_path)
    assert "Schema mismatch for BurstGPT LLM Trace" in str(excinfo.value)
    assert "Response tokens" in str(excinfo.value)


def test_load_burstgpt_data_file_not_found(tmp_path: Path):
    """Verify FileNotFoundError is raised when file does not exist."""
    non_existent = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        load_burstgpt_data(non_existent)
