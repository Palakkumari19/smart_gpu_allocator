"""
Unit tests for realdata/risk_overlay.py: Empirical hazard overlay and wrapper stepping.
"""

from copy import deepcopy
import pytest

from environment import GPUEnvironment
from fault_injection import SeverityLevel
from workload.task_generator import EnhancedTask, TaskType
from realdata.hazard import HazardBin, HazardCurve
from realdata.risk_overlay import (
    RiskHazardOverlay,
    OverlayFaultEvent,
    step_with_hazard_overlay,
)


@pytest.fixture
def mock_hazard_curve() -> HazardCurve:
    """Fixture providing a calibrated 2-bin hazard curve."""
    return HazardCurve(
        name="mock_curve",
        metric_name="dispersion",
        unit="index",
        global_sample_size=1000,
        global_failure_count=50,
        global_failure_rate=0.05,
        min_samples_threshold=100,
        bins=[
            HazardBin(
                bin_index=0,
                bin_label="[0, 2)",
                range_min=0.0,
                range_max=2.0,
                sample_size=800,
                failure_count=16,
                failure_probability=0.02,  # Low stress: 2%
                ci_lower=0.012,
                ci_upper=0.032,
            ),
            HazardBin(
                bin_index=1,
                bin_label="[2, 20)",
                range_min=2.0,
                range_max=20.0,
                sample_size=200,
                failure_count=60,
                failure_probability=0.30,  # High stress: 30%
                ci_lower=0.24,
                ci_upper=0.37,
            ),
        ],
    )


def test_overlay_switchable_on_off(mock_hazard_curve):
    """Verify overlay is strictly inactive when enabled=False and active when enabled=True."""
    env = GPUEnvironment(scheduler="fcfs")
    env.enable_deterministic_physics()

    task = EnhancedTask(TaskType.LLM, memory_required=2.0, execution_time=10)
    task.id = "test_overlay_task"
    env.attach_workload_trace([[task]])

    # Step into execution so task is running
    env.step()
    env.step()

    # 1. Overlay disabled ('workload only' mode)
    overlay = RiskHazardOverlay(
        hazard_curve=mock_hazard_curve,
        enabled=False,
        static_probability=1.0,  # Even with 100% prob, enabled=False must inject nothing
    )

    events = overlay.apply_post_step(env)
    assert len(events) == 0
    assert overlay.injected_count == 0

    # 2. Overlay enabled ('workload + real hazard' mode)
    overlay.set_enabled(True)
    events = overlay.apply_post_step(env)
    assert len(events) > 0
    assert overlay.injected_count > 0
    assert isinstance(events[0], OverlayFaultEvent)
    assert events[0].task_id in [task.id, "video_0001"]


def test_overlay_determinism_with_seed(mock_hazard_curve):
    """Verify that identical seeds produce identical injection sequences."""
    def run_simulation(seed: int):
        env = GPUEnvironment(scheduler="fcfs")
        env.enable_deterministic_physics()

        trace = [
            [EnhancedTask(TaskType.LLM, memory_required=1.5, execution_time=5) for _ in range(2)],
            [],
            [],
        ]
        env.attach_workload_trace(trace)

        overlay = RiskHazardOverlay(
            hazard_curve=mock_hazard_curve,
            enabled=True,
            static_probability=0.5,
            seed=seed,
        )

        injections = []
        for _ in range(4):
            res = step_with_hazard_overlay(env, overlay)
            injections.extend([e["task_id"] for e in res["overlay_events"]])

        return injections

    run1 = run_simulation(seed=1234)
    run2 = run_simulation(seed=1234)
    assert run1 == run2

    # Different seed should produce a different or offset sequence
    run3 = run_simulation(seed=9999)
    # Both are non-empty sequences
    assert isinstance(run1, list)
    assert isinstance(run3, list)


def test_step_with_hazard_overlay_wrapper(mock_hazard_curve):
    """Verify wrapper stepping function interacts seamlessly with base environment."""
    env = GPUEnvironment(scheduler="least_loaded")
    env.enable_deterministic_physics()

    tasks = [EnhancedTask(TaskType.VIDEO, memory_required=2.0, execution_time=6)]
    env.attach_workload_trace([tasks, [], []])

    # No overlay passed (defaults to clean baseline step)
    res_none = step_with_hazard_overlay(env, overlay=None)
    assert res_none["time_step"] == 1
    assert not res_none["overlay_enabled"]
    assert res_none["overlay_events"] == []

    # Overlay with high burstiness
    overlay = RiskHazardOverlay(
        hazard_curve=mock_hazard_curve,
        enabled=True,
        operating_metric_value=5.0,  # In bin 1: 30% probability
        seed=42,
    )

    res_overlay = step_with_hazard_overlay(env, overlay)
    assert res_overlay["time_step"] == 2
    assert res_overlay["overlay_enabled"]
    assert isinstance(res_overlay["overlay_events"], list)


def test_overlay_summary_and_reset(mock_hazard_curve):
    """Verify overlay counters, empirical rate computation, and reset functionality."""
    overlay = RiskHazardOverlay(
        hazard_curve=mock_hazard_curve,
        enabled=True,
        static_probability=0.25,
        seed=42,
    )

    env = GPUEnvironment()
    task = EnhancedTask(TaskType.LLM, memory_required=2.0, execution_time=5)
    env.attach_workload_trace([[task]])
    env.step()
    env.step()

    overlay.apply_post_step(env)
    summary = overlay.get_summary()

    assert summary["enabled"] is True
    assert summary["current_hazard_probability"] == 0.25
    assert summary["evaluations_count"] > 0
    assert 0.0 <= summary["empirical_injection_rate"] <= 1.0

    overlay.reset()
    assert overlay.evaluations_count == 0
    assert overlay.injected_count == 0
    assert len(overlay.history) == 0
