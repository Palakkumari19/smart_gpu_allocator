"""
LLM Workload Mapper for BurstGPT production traces.

Maps real LLM requests into simulator-native EnhancedTask instances using
QUANTILE (empirical percentile) mapping:
  - Total tokens (request + response) quantile -> memory_required in [1.0, 8.0] GB (KV-cache proxy)
  - Response tokens quantile -> execution_steps in [1, 20] steps
  - Tracks real_failed (response tokens == 0) in an auditable side-table and attaches to task
  - Persists and round-trips calibration tables to llm_calibration.json
"""

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Union, Tuple
import numpy as np
import pandas as pd

from realdata.config import LLM_CALIBRATION_PATH
from realdata.loaders import load_burstgpt_data
from workload.task_generator import EnhancedTask, TaskType


@dataclass
class LLMCalibration:
    """
    Empirical quantile calibration table for mapping BurstGPT tokens
    to simulator VRAM memory and execution duration steps.
    """
    dataset_name: str
    sample_count: int
    num_points: int
    quantiles: List[float]
    total_tokens_thresholds: List[float]
    response_tokens_thresholds: List[float]
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
        target_path = Path(path) if path else LLM_CALIBRATION_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target_path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMCalibration":
        """Reconstruct calibration instance from dictionary."""
        return cls(
            dataset_name=data["dataset_name"],
            sample_count=data["sample_count"],
            num_points=data["num_points"],
            quantiles=list(data["quantiles"]),
            total_tokens_thresholds=list(data["total_tokens_thresholds"]),
            response_tokens_thresholds=list(data["response_tokens_thresholds"]),
            memory_min_gb=float(data.get("memory_min_gb", 1.0)),
            memory_max_gb=float(data.get("memory_max_gb", 8.0)),
            steps_min=int(data.get("steps_min", 1)),
            steps_max=int(data.get("steps_max", 20)),
            severity_cutoffs=dict(data.get("severity_cutoffs", {
                "low": 0.35, "medium": 0.70, "high": 0.90, "critical": 1.00
            })),
        )

    @classmethod
    def from_json(cls, path: Optional[Union[str, Path]] = None) -> "LLMCalibration":
        """Load calibration instance from JSON file."""
        target_path = Path(path) if path else LLM_CALIBRATION_PATH
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
        dataset_name: str = "BurstGPT Production LLM Trace",
    ) -> "LLMCalibration":
        """
        Build empirical quantile calibration table from BurstGPT DataFrame.
        Uses Total tokens (request + response) and Response tokens.
        """
        if "Total tokens" in df.columns:
            tot_tokens = df["Total tokens"].dropna().values.astype(float)
        else:
            tot_tokens = (df["Request tokens"] + df["Response tokens"]).dropna().values.astype(float)

        resp_tokens = df["Response tokens"].dropna().values.astype(float)

        if len(tot_tokens) == 0:
            raise ValueError("DataFrame contains no valid token records.")

        quantiles = np.linspace(0.0, 1.0, num_points)
        tot_thresh = np.quantile(tot_tokens, quantiles).tolist()
        resp_thresh = np.quantile(resp_tokens, quantiles).tolist()

        return cls(
            dataset_name=dataset_name,
            sample_count=len(tot_tokens),
            num_points=num_points,
            quantiles=quantiles.tolist(),
            total_tokens_thresholds=tot_thresh,
            response_tokens_thresholds=resp_thresh,
        )

    def value_to_quantile_total_tokens(self, total_tokens: float) -> float:
        """Map total tokens to quantile in [0.0, 1.0]."""
        safe_val = max(0.0, float(total_tokens))
        q = float(np.interp(safe_val, self.total_tokens_thresholds, self.quantiles))
        return max(0.0, min(1.0, q))

    def value_to_quantile_response_tokens(self, response_tokens: float) -> float:
        """Map response tokens to quantile in [0.0, 1.0]."""
        safe_val = max(0.0, float(response_tokens))
        q = float(np.interp(safe_val, self.response_tokens_thresholds, self.quantiles))
        return max(0.0, min(1.0, q))

    def quantile_to_memory(self, q: float) -> float:
        """Map quantile to memory_required in [memory_min_gb, memory_max_gb]."""
        q_clamped = max(0.0, min(1.0, q))
        mem = self.memory_min_gb + q_clamped * (self.memory_max_gb - self.memory_min_gb)
        return round(float(np.clip(mem, self.memory_min_gb, self.memory_max_gb)), 2)

    def quantile_to_steps(self, q: float) -> int:
        """Map quantile to execution_steps in [steps_min, steps_max]."""
        q_clamped = max(0.0, min(1.0, q))
        steps = self.steps_min + q_clamped * (self.steps_max - self.steps_min)
        return int(np.clip(round(steps), self.steps_min, self.steps_max))

    def derive_priority_and_severity(
        self, q_mem: float, q_time: float
    ) -> Tuple[str, str]:
        """Derive priority and workload_severity label."""
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


class LLMTaskMapper:
    """
    Transforms BurstGPT rows into EnhancedTask instances.
    Maintains a side-table of real_failed status keyed by task id.
    """

    def __init__(
        self,
        calibration: Optional[LLMCalibration] = None,
        calibration_path: Optional[Union[str, Path]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize LLMTaskMapper.

        Args:
            calibration: Optional LLMCalibration instance.
            calibration_path: Optional path to saved calibration JSON.
            seed: Optional integer seed for reproducibility.
        """
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.failure_side_table: Dict[str, bool] = {}

        if calibration is not None:
            self.calibration = calibration
        elif calibration_path and Path(calibration_path).exists():
            self.calibration = LLMCalibration.from_json(calibration_path)
        elif LLM_CALIBRATION_PATH.exists():
            self.calibration = LLMCalibration.from_json(LLM_CALIBRATION_PATH)
        else:
            df = load_burstgpt_data()
            self.calibration = LLMCalibration.calibrate_from_dataframe(df)
            self.calibration.to_json(LLM_CALIBRATION_PATH)

    def map_row(
        self,
        row: Union[pd.Series, Dict[str, Any]],
        task_id: Optional[str] = None,
    ) -> EnhancedTask:
        """
        Map a single BurstGPT row to an EnhancedTask.

        Args:
            row: pandas Series or dict containing BurstGPT columns.
            task_id: Optional explicit string ID. If None, generated deterministically.

        Returns:
            EnhancedTask configured with empirical memory, duration, and LLM metadata.
        """
        req_tokens = float(row.get("Request tokens", 0))
        resp_tokens = float(row.get("Response tokens", 0))
        tot_tokens = float(row.get("Total tokens", req_tokens + resp_tokens))
        model = str(row.get("Model", "ChatGPT")).strip()
        timestamp = float(row.get("Timestamp", 0))

        # Real failure signal: response tokens == 0 signifies aborted / dropped under burst
        real_failed = bool(resp_tokens == 0)

        # Quantile transformations
        q_mem = self.calibration.value_to_quantile_total_tokens(tot_tokens)
        q_time = self.calibration.value_to_quantile_response_tokens(resp_tokens)

        memory_required = self.calibration.quantile_to_memory(q_mem)
        execution_steps = self.calibration.quantile_to_steps(q_time)
        priority, severity = self.calibration.derive_priority_and_severity(q_mem, q_time)

        # Build task using only the standard EnhancedTask constructor
        task = EnhancedTask(
            task_type=TaskType.LLM,
            memory_required=memory_required,
            execution_time=execution_steps,
            description=f"BurstGPT LLM ({model})",
            priority=priority,
        )

        # Deterministic task id
        if task_id is not None:
            task.id = task_id
        elif self.seed is not None:
            task.id = f"llm_{self.rng.randint(10000, 99999)}"

        # Store in failure side-table keyed by task id
        self.failure_side_table[task.id] = real_failed

        # Attach real_failed and LLM metadata (RECON.md confirms no __slots__ and safe)
        task.real_failed = real_failed
        task.model = model
        task.request_tokens = int(req_tokens)
        task.response_tokens = int(resp_tokens)
        task.total_tokens = int(tot_tokens)
        task.timestamp = timestamp
        task.workload_severity = severity
        task.severity = severity
        task.total_tokens_quantile = round(q_mem, 4)
        task.response_tokens_quantile = round(q_time, 4)

        return task

    def map_dataframe(
        self,
        df: pd.DataFrame,
        limit: Optional[int] = None,
    ) -> Tuple[List[EnhancedTask], Dict[str, bool]]:
        """
        Map a DataFrame of BurstGPT requests to EnhancedTasks and side-table.

        Returns:
            Tuple of (list_of_tasks, failure_side_table).
        """
        sub_df = df.head(limit) if limit else df
        tasks: List[EnhancedTask] = []
        for idx, (_, row) in enumerate(sub_df.iterrows()):
            t_id = f"burstgpt_{idx:06d}"
            task = self.map_row(row, task_id=t_id)
            tasks.append(task)
        return tasks, self.failure_side_table

    def get_failure_side_table(self) -> Dict[str, bool]:
        """Return the dictionary mapping task_id to boolean real_failed."""
        return dict(self.failure_side_table)


def calibrate_and_save_llm_data(
    df: Optional[pd.DataFrame] = None,
    output_path: Optional[Path] = None,
    num_points: int = 101,
) -> LLMCalibration:
    """Convenience function to calibrate from BurstGPT and save to JSON."""
    source_df = df if df is not None else load_burstgpt_data()
    calibration = LLMCalibration.calibrate_from_dataframe(
        source_df, num_points=num_points
    )
    dest = output_path or LLM_CALIBRATION_PATH
    calibration.to_json(dest)
    print(f"[LLMMapper] Calibration saved to {dest} ({calibration.sample_count:,} samples, {num_points} quantiles)")
    return calibration


if __name__ == "__main__":
    calibrate_and_save_llm_data()
