"""
Empirical Risk and Fault Overlay for AetherGrid.

Applies a post-step empirical hazard overlay to the GPU cluster:
1. Calls env.step() to advance discrete simulation physics and task progress.
2. In 'workload + real hazard' mode (enabled=True), evaluates each running task
   against the empirical failure probability calibrated from trace data (hazard.py).
3. If an anomaly is triggered, invokes gpu.inject_fault(task_id, severity) to
   consistently update GPU reliability counters, risk scores, cooldown timers,
   and task queue states.
4. Identical execution across all schedulers (fcfs, round_robin, least_loaded,
   rl, marl, risk_aware_marl) ensures unbiased paired evaluations.
5. Can be completely disabled (enabled=False) for pure 'workload only' runs.
"""

import random
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
from copy import deepcopy

from fault_injection import SeverityLevel, AnomalyResult
from realdata.hazard import HazardCurve, HazardCatalog, load_hazard_curves


@dataclass
class OverlayFaultEvent:
    """Record of an extra empirical fault injected by the risk overlay."""
    time_step: int
    gpu_id: int
    task_id: str
    severity: str
    anomaly_type: str
    gpu_crashed: bool
    task_failed: bool
    hazard_probability: float
    operating_metric_value: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_step": self.time_step,
            "gpu_id": self.gpu_id,
            "task_id": self.task_id,
            "severity": self.severity,
            "anomaly_type": self.anomaly_type,
            "gpu_crashed": self.gpu_crashed,
            "task_failed": self.task_failed,
            "hazard_probability": self.hazard_probability,
            "operating_metric_value": self.operating_metric_value,
        }


class RiskHazardOverlay:
    """
    Stateful overlay manager applying empirical failure probabilities from hazard curves.
    """

    def __init__(
        self,
        hazard_curve: Optional[HazardCurve] = None,
        enabled: bool = False,
        severity: Optional[SeverityLevel] = SeverityLevel.HIGH,
        seed: Optional[int] = 42,
        static_probability: Optional[float] = None,
        operating_metric_value: float = 1.0,
    ):
        """
        Args:
            hazard_curve: Calibrated HazardCurve instance (or loaded by name).
            enabled: Whether overlay is active (False = 'workload only', True = 'workload + real hazard').
            severity: Anomaly severity level passed to gpu.inject_fault().
            seed: Seed for deterministic pseudo-random hazard evaluation.
            static_probability: Optional fixed probability override (e.g. for testing).
            operating_metric_value: Current metric value (e.g. dispersion index or arrival rate).
        """
        self.hazard_curve = hazard_curve
        self.enabled = enabled
        self.severity = severity
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()
        self.static_probability = static_probability
        self.operating_metric_value = operating_metric_value

        self.evaluations_count: int = 0
        self.injected_count: int = 0
        self.crashes_count: int = 0
        self.task_failures_count: int = 0
        self.history: List[OverlayFaultEvent] = []

    def set_enabled(self, enabled: bool):
        """Switch the overlay on or off."""
        self.enabled = enabled

    def set_seed(self, seed: int):
        """Reset pseudo-random generator with a deterministic seed."""
        self.seed = seed
        self.rng = random.Random(seed)

    def set_operating_metric_value(self, value: float):
        """Update current operational stress value (e.g., dispersion index of window)."""
        self.operating_metric_value = float(value)

    def set_static_probability(self, probability: Optional[float]):
        """Set or clear a static probability override."""
        self.static_probability = probability

    def get_current_hazard_probability(self) -> float:
        """
        Compute failure probability for the current state.
        Uses static_probability if set; otherwise queries calibrated hazard curve.
        """
        if self.static_probability is not None:
            return max(0.0, min(1.0, float(self.static_probability)))

        if self.hazard_curve is not None:
            return self.hazard_curve.lookup_probability(self.operating_metric_value)

        # Baseline fallback if no curve or static rate provided
        return 0.01

    def apply_post_step(self, env: Any) -> List[OverlayFaultEvent]:
        """
        Examine all running tasks across cluster GPUs and probabilistically inject faults.

        Args:
            env: Active GPUEnvironment instance.

        Returns:
            List of OverlayFaultEvent instances injected during this step.
        """
        if not self.enabled:
            return []

        prob = self.get_current_hazard_probability()
        step_events: List[OverlayFaultEvent] = []
        current_step = getattr(env, "time_step", 0)

        for gpu in env.gpus:
            # Only evaluate running tasks on active (non-crashed) GPUs
            running_tasks = list(gpu.running_tasks)
            if not running_tasks:
                continue

            for task in running_tasks:
                self.evaluations_count += 1
                roll = self.rng.random()

                if roll < prob:
                    # Target severity level
                    sev = self.severity or SeverityLevel.HIGH

                    # Ensure fault occurs deterministically upon injection
                    prev_prob = env.fault_injector.custom_probabilities.get(sev, 0.30)
                    prev_enabled = env.fault_injector.enabled
                    try:
                        env.fault_injector.enabled = True
                        env.fault_injector.set_probability(sev, 1.0)
                        fault_result: AnomalyResult = gpu.inject_fault(
                            task_id=task.id, severity=sev
                        )
                    finally:
                        env.fault_injector.set_probability(sev, prev_prob)
                        env.fault_injector.enabled = prev_enabled

                    if fault_result.occurred:
                        self.injected_count += 1
                        if fault_result.gpu_crashed:
                            self.crashes_count += 1

                        if fault_result.task_failed:
                            self.task_failures_count += 1
                            # Clean up failed task from GPU running tasks and memory
                            if task in gpu.running_tasks:
                                gpu.running_tasks.remove(task)
                                gpu.current_memory = max(
                                    0.0,
                                    min(
                                        gpu.memory_capacity,
                                        round(gpu.current_memory - task.memory_required, 4),
                                    ),
                                )
                            if hasattr(env, "task_queue"):
                                env.task_queue.fail_task(task.id, retry=True)

                        # Immediately refresh GPU risk state so scheduling decisions in next step see it
                        if hasattr(gpu, "update_risk"):
                            gpu.update_risk()

                        evt = OverlayFaultEvent(
                            time_step=current_step,
                            gpu_id=gpu.id,
                            task_id=task.id,
                            severity=str(sev.value if hasattr(sev, "value") else sev),
                            anomaly_type=str(
                                fault_result.anomaly_type.value
                                if hasattr(fault_result.anomaly_type, "value")
                                else fault_result.anomaly_type
                            ),
                            gpu_crashed=bool(fault_result.gpu_crashed),
                            task_failed=bool(fault_result.task_failed),
                            hazard_probability=round(prob, 6),
                            operating_metric_value=round(self.operating_metric_value, 4),
                        )
                        self.history.append(evt)
                        step_events.append(evt)

        return step_events

    def get_summary(self) -> Dict[str, Any]:
        """Return summary metrics for the overlay execution."""
        return {
            "enabled": self.enabled,
            "operating_metric_value": self.operating_metric_value,
            "current_hazard_probability": self.get_current_hazard_probability(),
            "evaluations_count": self.evaluations_count,
            "injected_count": self.injected_count,
            "crashes_count": self.crashes_count,
            "task_failures_count": self.task_failures_count,
            "empirical_injection_rate": (
                round(self.injected_count / max(1, self.evaluations_count), 6)
            ),
        }

    def reset(self):
        """Reset overlay counters and history."""
        self.evaluations_count = 0
        self.injected_count = 0
        self.crashes_count = 0
        self.task_failures_count = 0
        self.history.clear()
        if self.seed is not None:
            self.rng = random.Random(self.seed)


def step_with_hazard_overlay(
    env: Any,
    overlay: Optional[RiskHazardOverlay] = None,
) -> Dict[str, Any]:
    """
    Step the GPU cluster environment and optionally apply empirical hazard overlay.

    1. Calls env.step() to advance discrete simulation physics.
    2. If overlay is active and enabled, evaluates and injects empirical faults
       on running tasks via gpu.inject_fault().
    3. Returns dictionary of step output and overlay events.

    Args:
        env: Active GPUEnvironment.
        overlay: Optional RiskHazardOverlay. If None or enabled=False, executes
                 pure workload step.

    Returns:
        Dict containing time_step, overlay_enabled status, and list of overlay events.
    """
    # 1. Step the base environment
    env.step()

    # 2. Apply external empirical overlay if configured
    events: List[OverlayFaultEvent] = []
    if overlay is not None and overlay.enabled:
        events = overlay.apply_post_step(env)

    return {
        "time_step": getattr(env, "time_step", 0),
        "overlay_enabled": bool(overlay.enabled) if overlay else False,
        "overlay_events": [e.to_dict() for e in events],
    }
