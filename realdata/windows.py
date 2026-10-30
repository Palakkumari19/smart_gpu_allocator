"""
Time window extraction and workload characterization for BurstGPT.

Identifies canonical evaluation windows:
  - median_hour: Typical steady-state request volume
  - busiest_hour: Peak volume stress test (maximum arrival volume)
  - burstiest_window: Maximum arrival volatility measured by index of dispersion (variance / mean)

Provides strictly disjoint TRAIN (earlier days) and EVAL (later days) splits.
Persists metadata to burstgpt_windows.json.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd

from realdata.config import BURSTGPT_WINDOWS_PATH
from realdata.loaders import load_burstgpt_data


@dataclass
class WindowMetadata:
    """Metadata describing a selected temporal window."""
    window_id: str
    partition: str
    hour: int
    duration_seconds: float
    start_timestamp: float
    end_timestamp: float
    request_count: int
    failed_request_count: int
    failure_rate_pct: float
    arrival_mean_per_sec: float
    arrival_variance: float
    dispersion_index: float
    models_breakdown: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WindowMetadata":
        return cls(**data)


@dataclass
class TraceWindowsSummary:
    """Summary of all extracted windows across train, eval, and full partitions."""
    train_hours_range: List[int]
    eval_hours_range: List[int]
    train_windows: Dict[str, WindowMetadata]
    eval_windows: Dict[str, WindowMetadata]
    full_windows: Dict[str, WindowMetadata]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "train_hours_range": self.train_hours_range,
            "eval_hours_range": self.eval_hours_range,
            "train_windows": {k: v.to_dict() for k, v in self.train_windows.items()},
            "eval_windows": {k: v.to_dict() for k, v in self.eval_windows.items()},
            "full_windows": {k: v.to_dict() for k, v in self.full_windows.items()},
        }

    def to_json(self, path: Optional[Union[str, Path]] = None) -> Path:
        target_path = Path(path) if path else BURSTGPT_WINDOWS_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target_path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TraceWindowsSummary":
        return cls(
            train_hours_range=list(data["train_hours_range"]),
            eval_hours_range=list(data["eval_hours_range"]),
            train_windows={
                k: WindowMetadata.from_dict(v) for k, v in data["train_windows"].items()
            },
            eval_windows={
                k: WindowMetadata.from_dict(v) for k, v in data["eval_windows"].items()
            },
            full_windows={
                k: WindowMetadata.from_dict(v) for k, v in data["full_windows"].items()
            },
        )

    @classmethod
    def from_json(cls, path: Optional[Union[str, Path]] = None) -> "TraceWindowsSummary":
        target_path = Path(path) if path else BURSTGPT_WINDOWS_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Windows file not found at {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)


def compute_hourly_dispersion(df: pd.DataFrame, step_seconds: float = 1.0) -> pd.Series:
    """
    Compute index of dispersion (variance / mean) of per-step arrivals for each hour.

    Args:
        df: DataFrame with 'Timestamp' and 'hour'.
        step_seconds: Discretization step width in seconds (default: 1.0s).

    Returns:
        pd.Series indexed by hour containing dispersion index values.
    """
    if len(df) == 0:
        return pd.Series([], dtype=float)

    # Bin timestamps within each hour
    bin_ids = ((df["Timestamp"] % 3600) // step_seconds).astype(int)
    total_bins_per_hour = max(1.0, 3600.0 / step_seconds)

    # Group by hour and step bin
    temp_df = pd.DataFrame({"hour": df["hour"], "bin": bin_ids})
    counts = temp_df.groupby(["hour", "bin"]).size()

    # Sum of squared arrivals and total arrivals per hour
    sum_sq = (counts ** 2).groupby("hour").sum()
    total_n = df.groupby("hour").size()

    # Dispersion: var / mean = (sum(c^2)/N) - (N / Bins)
    dispersion = (sum_sq / total_n) - (total_n / total_bins_per_hour)
    return dispersion.fillna(0.0).clip(lower=0.0)


def extract_window_metadata(
    df: pd.DataFrame,
    hour: int,
    window_id: str,
    partition: str,
    step_seconds: float = 1.0,
) -> WindowMetadata:
    """Extract descriptive statistics for a specific 1-hour window."""
    start_ts = float(hour * 3600)
    end_ts = float((hour + 1) * 3600)

    sub = df[(df["Timestamp"] >= start_ts) & (df["Timestamp"] < end_ts)]
    req_cnt = len(sub)
    failed_cnt = int(sub["is_failed"].sum()) if "is_failed" in sub.columns else int((sub["Response tokens"] == 0).sum())
    fail_rate = round((failed_cnt / max(1, req_cnt)) * 100, 2)

    # Calculate exact arrival variance and mean
    bins_per_hour = int(round(3600.0 / step_seconds))
    bin_counts = np.zeros(bins_per_hour, dtype=float)
    if req_cnt > 0:
        rel_sec = (sub["Timestamp"].values - start_ts) / step_seconds
        valid_indices = np.clip(rel_sec.astype(int), 0, bins_per_hour - 1)
        np.add.at(bin_counts, valid_indices, 1.0)

    arrival_mean = float(np.mean(bin_counts))
    arrival_var = float(np.var(bin_counts))
    dispersion = float(arrival_var / arrival_mean) if arrival_mean > 0 else 0.0

    models = sub["Model"].value_counts().to_dict() if "Model" in sub.columns else {}

    return WindowMetadata(
        window_id=window_id,
        partition=partition,
        hour=hour,
        duration_seconds=3600.0,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        request_count=req_cnt,
        failed_request_count=failed_cnt,
        failure_rate_pct=fail_rate,
        arrival_mean_per_sec=round(arrival_mean, 4),
        arrival_variance=round(arrival_variance if (arrival_variance := arrival_var) else 0.0, 4),
        dispersion_index=round(dispersion, 4),
        models_breakdown=models,
    )


def select_partition_windows(
    df: pd.DataFrame,
    partition_name: str,
    step_seconds: float = 1.0,
) -> Dict[str, WindowMetadata]:
    """
    Select median_hour, busiest_hour, and burstiest_window for a partition.
    """
    if len(df) == 0:
        raise ValueError(f"Partition {partition_name} contains no records.")

    hourly_counts = df.groupby("hour").size()

    # 1. median_hour: Hour whose request count is closest to partition median
    target_median = hourly_counts.median()
    median_h = int((hourly_counts - target_median).abs().idxmin())

    # 2. busiest_hour: Hour with maximum total requests
    busiest_h = int(hourly_counts.idxmax())

    # 3. burstiest_window: Hour with highest index of dispersion (variance / mean)
    dispersion_series = compute_hourly_dispersion(df, step_seconds=step_seconds)
    # Filter hours with at least 10 requests to avoid trivial 1-request variance
    valid_dispersion = dispersion_series[hourly_counts >= 10]
    burstiest_h = int(valid_dispersion.idxmax()) if len(valid_dispersion) > 0 else busiest_h

    return {
        "median_hour": extract_window_metadata(
            df, median_h, "median_hour", partition_name, step_seconds
        ),
        "busiest_hour": extract_window_metadata(
            df, busiest_h, "busiest_hour", partition_name, step_seconds
        ),
        "burstiest_window": extract_window_metadata(
            df, burstiest_h, "burstiest_window", partition_name, step_seconds
        ),
    }


def extract_burstgpt_windows(
    df: Optional[pd.DataFrame] = None,
    train_split_ratio: float = 0.5,
    step_seconds: float = 1.0,
) -> TraceWindowsSummary:
    """
    Extract canonical windows across disjoint TRAIN and EVAL partitions.

    Args:
        df: BurstGPT DataFrame. Loaded automatically if None.
        train_split_ratio: Fraction of total hours allocated to earlier TRAIN period.
        step_seconds: Step size for dispersion calculation.

    Returns:
        TraceWindowsSummary containing disjoint train, eval, and full window metadata.
    """
    source_df = df if df is not None else load_burstgpt_data()

    min_hour = int(source_df["hour"].min())
    max_hour = int(source_df["hour"].max())
    total_hours_span = max_hour - min_hour + 1

    split_hour = int(min_hour + total_hours_span * train_split_ratio)

    train_df = source_df[source_df["hour"] <= split_hour]
    eval_df = source_df[source_df["hour"] > split_hour]

    train_windows = select_partition_windows(
        train_df, partition_name="train", step_seconds=step_seconds
    )
    eval_windows = select_partition_windows(
        eval_df, partition_name="eval", step_seconds=step_seconds
    )
    full_windows = select_partition_windows(
        source_df, partition_name="full", step_seconds=step_seconds
    )

    return TraceWindowsSummary(
        train_hours_range=[int(train_df["hour"].min()), int(train_df["hour"].max())],
        eval_hours_range=[int(eval_df["hour"].min()), int(eval_df["hour"].max())],
        train_windows=train_windows,
        eval_windows=eval_windows,
        full_windows=full_windows,
    )


def extract_window_dataframe(
    df: pd.DataFrame, window: WindowMetadata
) -> pd.DataFrame:
    """Slice DataFrame for a given WindowMetadata."""
    return df[
        (df["Timestamp"] >= window.start_timestamp)
        & (df["Timestamp"] < window.end_timestamp)
    ].copy()


def generate_and_save_burstgpt_windows(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
) -> TraceWindowsSummary:
    """Extract and save BurstGPT windows summary to JSON."""
    summary = extract_burstgpt_windows(df=df)
    dest = output_path or BURSTGPT_WINDOWS_PATH
    summary.to_json(dest)
    print(f"[Windows] Saved windows summary to {dest}")
    print(f"  TRAIN range: hours {summary.train_hours_range[0]} - {summary.train_hours_range[1]}")
    print(f"    - median_hour: {summary.train_windows['median_hour'].hour} ({summary.train_windows['median_hour'].request_count} reqs)")
    print(f"    - busiest_hour: {summary.train_windows['busiest_hour'].hour} ({summary.train_windows['busiest_hour'].request_count} reqs)")
    print(f"    - burstiest_window: {summary.train_windows['burstiest_window'].hour} (dispersion: {summary.train_windows['burstiest_window'].dispersion_index})")
    print(f"  EVAL range: hours {summary.eval_hours_range[0]} - {summary.eval_hours_range[1]}")
    print(f"    - median_hour: {summary.eval_windows['median_hour'].hour} ({summary.eval_windows['median_hour'].request_count} reqs)")
    print(f"    - busiest_hour: {summary.eval_windows['busiest_hour'].hour} ({summary.eval_windows['busiest_hour'].request_count} reqs)")
    print(f"    - burstiest_window: {summary.eval_windows['burstiest_window'].hour} (dispersion: {summary.eval_windows['burstiest_window'].dispersion_index}, fail_rate: {summary.eval_windows['burstiest_window'].failure_rate_pct}%)")
    return summary


if __name__ == "__main__":
    generate_and_save_burstgpt_windows()
