"""
Data loaders with schema validation for empirical trace datasets.
"""

from pathlib import Path
from typing import Optional, List, Union
import pandas as pd

from realdata.config import (
    UCI_VIDEO_RAW_PATH,
    UCI_VIDEO_REQUIRED_COLUMNS,
    BURSTGPT_RAW_PATH,
    BURSTGPT_REQUIRED_COLUMNS,
)


def _validate_schema(
    df: pd.DataFrame,
    required_columns: List[str],
    dataset_name: str,
    source_path: Path,
) -> None:
    """
    Validate that all required columns are present in the DataFrame.

    Raises:
        ValueError: If any required column is missing.
    """
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Schema mismatch for {dataset_name} at {source_path}.\n"
            f"  Missing required columns ({len(missing)}): {missing}\n"
            f"  Available columns ({len(df.columns)}): {list(df.columns)}"
        )


def load_uci_video_data(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load and validate the UCI Video Characteristics and Transcoding Time dataset.

    Args:
        path: Path to transcoding_mesurment.tsv. Defaults to UCI_VIDEO_RAW_PATH.

    Returns:
        pd.DataFrame with validated columns and added 'resolution' column.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If schema validation fails.
    """
    file_path = Path(path) if path else UCI_VIDEO_RAW_PATH

    if not file_path.exists():
        raise FileNotFoundError(
            f"UCI Video dataset not found at {file_path}. "
            f"Run 'python -m realdata.fetch_data' first to acquire it."
        )

    df = pd.read_csv(file_path, sep="\t")

    # Validate schema
    _validate_schema(
        df=df,
        required_columns=UCI_VIDEO_REQUIRED_COLUMNS,
        dataset_name="UCI Video Transcoding",
        source_path=file_path,
    )

    # Add normalized resolution fields
    df["resolution"] = df["width"].astype(str) + "x" + df["height"].astype(str)
    df["resolution_label"] = df["height"].astype(str) + "p"

    return df


def load_burstgpt_data(
    path: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """
    Load and validate the BurstGPT LLM trace dataset.

    Args:
        path: Path to BurstGPT_1.csv. Defaults to BURSTGPT_RAW_PATH.

    Returns:
        pd.DataFrame with validated columns and added 'is_failed' and 'hour' columns.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If schema validation fails.
    """
    file_path = Path(path) if path else BURSTGPT_RAW_PATH

    if not file_path.exists():
        raise FileNotFoundError(
            f"BurstGPT dataset not found at {file_path}. "
            f"Run 'python -m realdata.fetch_data' first to acquire it."
        )

    df = pd.read_csv(file_path)

    # Validate schema
    _validate_schema(
        df=df,
        required_columns=BURSTGPT_REQUIRED_COLUMNS,
        dataset_name="BurstGPT LLM Trace",
        source_path=file_path,
    )

    # Add failure flag and temporal binning
    # In BurstGPT, requests with 0 response tokens are failed requests under burst/memory pressure
    df["is_failed"] = df["Response tokens"] == 0
    df["hour"] = (df["Timestamp"] // 3600).astype(int)

    return df
