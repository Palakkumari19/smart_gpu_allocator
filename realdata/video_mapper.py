"""
Video Workload Mapper for UCI Online Video Characteristics & Transcoding Dataset.

Maps real-world transcoding jobs into simulator-native EnhancedTask instances
using non-linear QUANTILE (empirical percentile) mapping:
  - log(umem) quantile -> memory_required in [1.0, 8.0] GB
  - utime quantile -> execution_steps in [1, 20] steps
  - Quantile-based priority and workload_severity derivation
  - Attaches codec, resolution, framerate, and bitrate metadata
  - Persists and round-trips calibration tables to video_calibration.json
"""

import json
import math
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd

from realdata.config import VIDEO_CALIBRATION_PATH, CALIBRATION_DIR
from realdata.loaders import load_uci_video_data
from workload.task_generator import EnhancedTask, TaskType


@dataclass
class VideoCalibration:
    """
    Serializable calibration table holding empirical quantile thresholds
    for reproducing exact non-linear mappings from real traces to simulation units.
    """
    dataset_name: str
    sample_count: int
    num_points: int
    quantiles: List[float]
    log_umem_thresholds: List[float]
    utime_thresholds: List[float]
    memory_min_gb: float = 1.0
    memory_max_gb: float = 8.0
    steps_min: int = 1
    steps_max: int = 20
    severity_cutoffs: Dict[str, float] = field(
        default_factory=lambda: {
            "low": 0.35,
            "medium": 0.70,
            "high": 0.90,
            "critical": 1.00,
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        """Convert calibration to dictionary."""
        return asdict(self)

    def to_json(self, path: Optional[Union[str, Path]] = None) -> Path:
        """Serialize calibration to JSON file."""
        target_path = Path(path) if path else VIDEO_CALIBRATION_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target_path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VideoCalibration":
        """Reconstruct calibration instance from dictionary."""
        return cls(
            dataset_name=data["dataset_name"],
            sample_count=data["sample_count"],
            num_points=data["num_points"],
            quantiles=list(data["quantiles"]),
            log_umem_thresholds=list(data["log_umem_thresholds"]),
            utime_thresholds=list(data["utime_thresholds"]),
            memory_min_gb=float(data.get("memory_min_gb", 1.0)),
            memory_max_gb=float(data.get("memory_max_gb", 8.0)),
            steps_min=int(data.get("steps_min", 1)),
            steps_max=int(data.get("steps_max", 20)),
            severity_cutoffs=dict(data.get("severity_cutoffs", {
                "low": 0.35, "medium": 0.70, "high": 0.90, "critical": 1.00
            })),
        )

    @classmethod
    def from_json(cls, path: Optional[Union[str, Path]] = None) -> "VideoCalibration":
        """Load calibration instance from JSON file."""
        target_path = Path(path) if path else VIDEO_CALIBRATION_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Calibration file not found at {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    @classmethod
    def calibrate_from_dataframe(
        cls,
        df: pd.DataFrame,
        num_points: int = 101,
        dataset_name: str = "UCI Online Video Characteristics and Transcoding Time",
    ) -> "VideoCalibration":
        """
        Build empirical quantile calibration table from a DataFrame containing umem and utime.
        """
        umem_clean = df["umem"].dropna().values.astype(float)
        utime_clean = df["utime"].dropna().values.astype(float)

        # Enforce positive values for log transformation
        valid_mask = (umem_clean > 0) & (utime_clean > 0)
        umem_valid = umem_clean[valid_mask]
        utime_valid = utime_clean[valid_mask]

        if len(umem_valid) == 0:
            raise ValueError("DataFrame contains no valid positive umem/utime values.")

        log_umem = np.log(umem_valid)

        quantiles = np.linspace(0.0, 1.0, num_points)
        log_umem_thresh = np.quantile(log_umem, quantiles).tolist()
        utime_thresh = np.quantile(utime_valid, quantiles).tolist()

        return cls(
            dataset_name=dataset_name,
            sample_count=len(umem_valid),
            num_points=num_points,
            quantiles=quantiles.tolist(),
            log_umem_thresholds=log_umem_thresh,
            utime_thresholds=utime_thresh,
        )

    def value_to_quantile_log_umem(self, umem_kb: float) -> float:
        """Map raw umem in KB to empirical quantile in [0.0, 1.0] using log(umem)."""
        safe_umem = max(1e-6, float(umem_kb))
        log_val = math.log(safe_umem)
        # Monotonic linear interpolation against calibration grid
        q = float(np.interp(log_val, self.log_umem_thresholds, self.quantiles))
        return max(0.0, min(1.0, q))

    def value_to_quantile_utime(self, utime_sec: float) -> float:
        """Map raw utime in seconds to empirical quantile in [0.0, 1.0]."""
        safe_time = max(0.0, float(utime_sec))
        q = float(np.interp(safe_time, self.utime_thresholds, self.quantiles))
        return max(0.0, min(1.0, q))

    def quantile_to_memory(self, q: float) -> float:
        """Map quantile in [0.0, 1.0] to memory_required in [memory_min_gb, memory_max_gb]."""
        q_clamped = max(0.0, min(1.0, q))
        mem = self.memory_min_gb + q_clamped * (self.memory_max_gb - self.memory_min_gb)
        # Round to 2 decimals, strictly bounded
        return round(float(np.clip(mem, self.memory_min_gb, self.memory_max_gb)), 2)

    def quantile_to_steps(self, q: float) -> int:
        """Map quantile in [0.0, 1.0] to integer execution steps in [steps_min, steps_max]."""
        q_clamped = max(0.0, min(1.0, q))
        steps = self.steps_min + q_clamped * (self.steps_max - self.steps_min)
        return int(np.clip(round(steps), self.steps_min, self.steps_max))

    def derive_priority_and_severity(
        self, q_mem: float, q_time: float
    ) -> Tuple[str, str]:
        """
        Derive task priority and workload_severity from composite quantile score.
        Returns:
            (priority, workload_severity)
        """
        # Composite intensity score
        q_composite = 0.5 * q_mem + 0.5 * q_time

        cutoffs = self.severity_cutoffs
        if q_composite < cutoffs["low"]:
            return "low", "LOW"
        elif q_composite < cutoffs["medium"]:
            return "medium", "MEDIUM"
        elif q_composite < cutoffs["high"]:
            return "high", "HIGH"
        else:
            return "critical", "VERY HIGH"


class VideoTaskMapper:
    """
    Transforms raw UCI video transcoding dataset rows into simulator-ready EnhancedTask instances.
    Guarantees:
      - Memory strictly bounded to [1.0, 8.0] GB
      - Steps strictly bounded to [1, 20]
      - Monotonic rank-order preservation
      - Determinism when seed is provided
    """

    def __init__(
        self,
        calibration: Optional[VideoCalibration] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize VideoTaskMapper.

        Args:
            calibration: Optional VideoCalibration instance.
            calibration_path: Optional path to saved calibration JSON.
            seed: Optional integer seed for deterministic task IDs and sampling.
        """
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()

        if calibration is not None:
            self.calibration = calibration
        elif calibration_path and Path(calibration_path).exists():
            self.calibration = VideoCalibration.from_json(calibration_path)
        elif VIDEO_CALIBRATION_PATH.exists():
            self.calibration = VideoCalibration.from_json(VIDEO_CALIBRATION_PATH)
        else:
            # Calibrate on-the-fly and save to default calibration path
            df = load_uci_video_data()
            self.calibration = VideoCalibration.calibrate_from_dataframe(df)
            self.calibration.to_json(VIDEO_CALIBRATION_PATH)

    def map_row(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        task_id: Optional[str] = None,
    ) -> EnhancedTask:
        """
        Map a single UCI video dataset row to an EnhancedTask.

        Args:
            row: pandas Series or dict containing required columns.
            task_id: Optional explicit string ID. If None, generated deterministically.

        Returns:
            EnhancedTask configured with empirical memory, duration, and video attributes.
        """
        umem = float(row["umem"])
        utime = float(row["utime"])
        duration = float(row.get("duration", 60.0))
        codec = str(row.get("codec", "h264")).strip("\"'")
        width = int(row.get("width", 1920))
        height = int(row.get("height", 1080))
        framerate = float(row.get("framerate", 30.0))
        bitrate = float(row.get("bitrate", 5000000.0))

        # Quantile transformations
        q_mem = self.calibration.value_to_quantile_log_umem(umem)
        q_time = self.calibration.value_to_quantile_utime(utime)

        memory_required = self.calibration.quantile_to_memory(q_mem)
        execution_steps = self.calibration.quantile_to_steps(q_time)
        priority, severity = self.calibration.derive_priority_and_severity(q_mem, q_time)

        # Build task using only the existing constructor
        task = EnhancedTask(
            task_type=TaskType.VIDEO,
            memory_required=memory_required,
            execution_time=execution_steps,
            description=f"UCI Video Transcode ({width}x{height} {codec})",
            priority=priority,
        )

        # Assign deterministic ID if requested
        if task_id is not None:
            task.id = task_id
        elif self.seed is not None:
            task.id = f"video_{self.rng.randint(10000, 99999)}"

        # Attach real-trace video characteristics to EnhancedTask (allowed: no __slots__)
        task.resolution = f"{width}x{height}"
        task.width = width
        task.height = height
        task.codec = codec
        task.fps = framerate
        task.bitrate = f"{bitrate / 1000.0:.1f} kbps"
        task.duration = duration
        task.workload_severity = severity
        task.severity = severity
        task.raw_umem_kb = umem
        task.raw_utime_sec = utime
        task.umem_quantile = round(q_mem, 4)
        task.utime_quantile = round(q_time, 4)

        return task

    def map_dataframe(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None,
    ) -> List[EnhancedTask]:
        """Map an entire DataFrame of UCI video jobs to a list of EnhancedTasks."""
        sub_df = df.head(limit) if limit else df
        tasks: List[EnhancedTask] = []
        for idx, (_, row) in enumerate(sub_df.iterrows()):
            t_id = f"uci_video_{idx:05d}"
            tasks.append(self.map_row(row, task_id=t_id))
        return tasks

    def sample_tasks(
        self,
        count: int,
        df: Optional[pd.DataFrame] = None,
    ) -> List[EnhancedTask]:
        """
        Sample rows deterministically from the UCI dataset and map to EnhancedTasks.
        """
        source_df = df if df is not None else load_uci_video_data()
        indices = self.rng.sample(range(len(source_df)), min(count, len(source_df)))
        tasks = []
        for i, idx in enumerate(indices):
            row = source_df.iloc[idx]
            tasks.append(self.map_row(row, task_id=f"sample_vid_{i:04d}"))
        return tasks


def calibrate_and_save_video_data(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    num_points: int = 101,
) -> VideoCalibration:
    """Convenience function to calibrate from UCI dataset and save to JSON."""
    source_df = df if df is not None else load_uci_video_data()
    calibration = VideoCalibration.calibrate_from_dataframe(
        source_df, num_points=num_points
    )
    dest = output_path or VIDEO_CALIBRATION_PATH
    calibration.to_json(dest)
    print(f"[VideoMapper] Calibration saved to {dest} ({calibration.sample_count:,} samples, {num_points} quantiles)")
    return calibration


if __name__ == "__main__":
    calibrate_and_save_video_data()
