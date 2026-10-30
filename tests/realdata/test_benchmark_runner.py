"""
Unit tests for realdata.run_realdata_benchmark.
Verifies episode execution, strongest baseline detection, metric keys, and PREREGISTRATION file integrity.
"""

import pytest
from pathlib import Path
import numpy as np
import pandas as pd

from realdata.run_realdata_benchmark import (
    run_single_episode,
    find_strongest_baseline,
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    HIGHER_IS_BETTER,
)
from workload.task_generator import EnhancedTask, TaskType
from realdata.config import PACKAGE_ROOT


@pytest.fixture
def mini_trace():
    """Create a minimal 5-step trace with 1 task per step."""
    trace = []
    for step in range(5):
        task = EnhancedTask(
            task_type=TaskType.LLM,
            memory_required=2.0,
            execution_time=2,
            description=f"Task {step}",
            priority="medium",
        )
        task.id = f"mini_task_{step}"
        trace.append([task])
    return trace


def test_preregistration_file_exists():
    """Verify PREREGISTRATION.md exists and documents primary/secondary metrics and decision rules."""
    prereg_path = PACKAGE_ROOT / "PREREGISTRATION.md"
    assert prereg_path.exists(), "PREREGISTRATION.md must exist before running evaluations"
    content = prereg_path.read_text(encoding="utf-8")
    assert "avg_risk_exposure" in content
    assert "terminal_efc" in content
    assert "crash_count" in content
    assert "quarantine_time" in content
    assert "completed_tasks" in content
    assert "paired_diff_ci" in content
    assert "win_rate" in content


def test_run_single_episode_metrics_structure(mini_trace):
    """Verify run_single_episode returns all required primary and secondary metrics."""
    metrics_fcfs = run_single_episode("fcfs", mini_trace, seed=42)
    metrics_risk_marl = run_single_episode("risk_aware_marl", mini_trace, seed=42)

    for metric in PRIMARY_METRICS + SECONDARY_METRICS:
        assert metric in metrics_fcfs
        assert isinstance(metrics_fcfs[metric], (int, float))
        assert metric in metrics_risk_marl
        assert isinstance(metrics_risk_marl[metric], (int, float))

    # Memory bound invariant
    assert metrics_fcfs["avg_memory_usage"] >= 0.0
    assert metrics_risk_marl["avg_memory_usage"] >= 0.0


def test_find_strongest_baseline():
    """Verify find_strongest_baseline correctly picks min for lower-is-better and max for higher-is-better."""
    mock_records = [
        {"scheduler": "fcfs", "avg_risk_exposure": 0.45, "completed_tasks": 100},
        {"scheduler": "round_robin", "avg_risk_exposure": 0.35, "completed_tasks": 110},
        {"scheduler": "least_loaded", "avg_risk_exposure": 0.25, "completed_tasks": 120},  # Strongest for risk
        {"scheduler": "rl", "avg_risk_exposure": 0.30, "completed_tasks": 105},
        {"scheduler": "marl", "avg_risk_exposure": 0.28, "completed_tasks": 130},          # Strongest for tasks
    ]

    # Lower is better: least_loaded has minimum risk (0.25)
    best_risk = find_strongest_baseline(mock_records, "avg_risk_exposure")
    assert best_risk == "least_loaded"

    # Higher is better: marl has maximum completions (130)
    best_tasks = find_strongest_baseline(mock_records, "completed_tasks")
    assert best_tasks == "marl"
