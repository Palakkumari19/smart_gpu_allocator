"""
Bit 9: Experiment Configuration & Definitions
Defines benchmark scenarios, scheduler enumerations, and validation bounds.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any


class Schedulers(str, Enum):
    FCFS = "fcfs"
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    RL = "rl"
    MARL = "marl"
    RISK_AWARE_MARL = "risk_aware_marl"


ALL_SCHEDULERS = [
    Schedulers.FCFS.value,
    Schedulers.ROUND_ROBIN.value,
    Schedulers.LEAST_LOADED.value,
    Schedulers.RL.value,
    Schedulers.MARL.value,
    Schedulers.RISK_AWARE_MARL.value,
]


@dataclass
class BenchmarkConfig:
    """Configuration for a multi-algorithm benchmark experiment."""
    name: str = "Standard Research Benchmark"
    schedulers: List[str] = field(default_factory=lambda: list(ALL_SCHEDULERS))
    num_tasks: int = 50
    num_repetitions: int = 3
    seeds: List[int] = field(default_factory=lambda: [42, 43, 44])
    workload_type: str = "mixed"  # "mixed", "light_heavy", "extreme_heavy", "burst"
    max_steps_per_run: int = 80
    enforce_capacity_limit: bool = True
    record_task_telemetry: bool = True
    timeout_seconds: float = 120.0
    description: str = "Fair comparison of traditional, RL, and Risk-Aware MARL schedulers under identical workloads."

    def validate(self) -> None:
        """Validate input parameters to guarantee safe, bounded execution."""
        if not self.schedulers:
            raise ValueError("At least one scheduler must be selected for benchmarking.")
        for s in self.schedulers:
            if s not in ALL_SCHEDULERS:
                raise ValueError(f"Unknown scheduler '{s}'. Available: {ALL_SCHEDULERS}")

        if not (5 <= self.num_tasks <= 300):
            raise ValueError(f"num_tasks ({self.num_tasks}) must be between 5 and 300.")

        if not (1 <= self.num_repetitions <= 10):
            raise ValueError(f"num_repetitions ({self.num_repetitions}) must be between 1 and 10.")

        if len(self.seeds) < self.num_repetitions:
            base_seed = self.seeds[0] if self.seeds else 42
            self.seeds = [base_seed + i for i in range(self.num_repetitions)]

        if not (10 <= self.max_steps_per_run <= 300):
            raise ValueError(f"max_steps_per_run ({self.max_steps_per_run}) must be between 10 and 300.")


@dataclass
class AblationConfig:
    """Configuration for Risk-Aware MARL ablation experiments."""
    name: str = "Risk-Aware MARL Ablation Study"
    num_tasks: int = 50
    num_repetitions: int = 3
    seeds: List[int] = field(default_factory=lambda: [101, 102, 103])
    workload_type: str = "mixed"
    max_steps_per_run: int = 80
    ablation_modes: List[str] = field(default_factory=lambda: [
        "full_risk_aware",      # Risk penalty + EFC penalty active
        "no_risk_penalty",      # Risk penalty = 0.0, EFC penalty active
        "no_efc_penalty",       # Risk penalty active, EFC penalty = 0.0
        "risk_blind",           # Risk penalty = 0.0, EFC penalty = 0.0
    ])

    def validate(self) -> None:
        valid_modes = {"full_risk_aware", "no_risk_penalty", "no_efc_penalty", "risk_blind"}
        for m in self.ablation_modes:
            if m not in valid_modes:
                raise ValueError(f"Invalid ablation mode '{m}'. Allowed: {valid_modes}")
        if len(self.seeds) < self.num_repetitions:
            self.seeds = [100 + i for i in range(self.num_repetitions)]
