"""
Bit 9: Scheduler Adapters
Provides uniform adapter interfaces for the six evaluated scheduling strategies:
1. FCFS
2. Round Robin
3. Least Loaded
4. RL (Single-Agent Q-learning)
5. MARL (Risk-Blind Multi-Agent RL)
6. Risk-Aware MARL (Multi-Agent RL with Risk & EFC penalties)

Ensures every baseline strictly respects the 8.0 GB physical GPU capacity invariant
and concurrency locks without bypassing safety rules.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

from environment import GPUEnvironment
from marl_agent import MARLManager


class BaseSchedulerAdapter(ABC):
    """Abstract base adapter for all evaluated schedulers."""

    def __init__(self, name: str, algorithm_id: str, description: str):
        self.name = name
        self.algorithm_id = algorithm_id
        self.description = description
        self.is_available = True
        self.unavailability_reason: Optional[str] = None

    @abstractmethod
    def configure_environment(self, env: GPUEnvironment, **kwargs) -> None:
        """Configure the isolated environment instance for this scheduler."""
        pass

    def reset_state(self, env: GPUEnvironment) -> None:
        """Reset internal scheduler pointers or episode state."""
        pass


class FCFSAdapter(BaseSchedulerAdapter):
    """First-Come, First-Served Scheduler Adapter."""

    def __init__(self):
        super().__init__(
            name="First-Come, First-Served (FCFS)",
            algorithm_id="fcfs",
            description="Assigns submitted tasks to the first eligible GPU in order, enforcing 8.0 GB capacity.",
        )

    def configure_environment(self, env: GPUEnvironment, **kwargs) -> None:
        env.scheduler = "fcfs"


class RoundRobinAdapter(BaseSchedulerAdapter):
    """Round Robin Scheduler Adapter."""

    def __init__(self):
        super().__init__(
            name="Round Robin",
            algorithm_id="round_robin",
            description="Rotates task assignments across eligible GPUs sequentially with deterministic pointer.",
        )

    def configure_environment(self, env: GPUEnvironment, **kwargs) -> None:
        env.scheduler = "round_robin"
        env.rr_index = 0

    def reset_state(self, env: GPUEnvironment) -> None:
        env.rr_index = 0


class LeastLoadedAdapter(BaseSchedulerAdapter):
    """Least Loaded Scheduler Adapter."""

    def __init__(self):
        super().__init__(
            name="Least Loaded",
            algorithm_id="least_loaded",
            description="Selects the eligible GPU with minimum current load (queue + running tasks + memory usage).",
        )

    def configure_environment(self, env: GPUEnvironment, **kwargs) -> None:
        env.scheduler = "least_loaded"


class RLAdapter(BaseSchedulerAdapter):
    """Single-Agent Reinforcement Learning Scheduler Adapter."""

    def __init__(self):
        super().__init__(
            name="Single-Agent RL",
            algorithm_id="rl",
            description="Centralized Q-learning agent mapping cluster state vectors to GPU index decisions.",
        )
        # Verify single-agent RL presence in codebase
        try:
            from agent import RLAgent
            self.is_available = True
        except ImportError:
            self.is_available = False
            self.unavailability_reason = "agent.py RLAgent module not available"

    def configure_environment(self, env: GPUEnvironment, **kwargs) -> None:
        if not self.is_available:
            raise RuntimeError(f"RL scheduler is unavailable: {self.unavailability_reason}")
        env.scheduler = "rl"
        # In evaluation mode, disable exploratory actions (epsilon = 0.0)
        if hasattr(env, "agent") and env.agent is not None:
            env.agent.epsilon = 0.0


class MARLAdapter(BaseSchedulerAdapter):
    """Standard Risk-Blind Multi-Agent Reinforcement Learning Adapter."""

    def __init__(self):
        super().__init__(
            name="Standard MARL (Risk-Blind)",
            algorithm_id="marl",
            description="Decentralized cooperative MARL bidding based on learned Q-values & performance, ignoring risk & EFC penalties.",
        )

    def configure_environment(self, env: GPUEnvironment, **kwargs) -> None:
        env.scheduler = "marl"
        # Patch marl_manager on this isolated env to omit risk and EFC penalties
        orig_compute_bid = env.marl_manager.compute_bid_score_breakdown

        def risk_blind_compute_bid(gpu, task, agent, cluster_state) -> dict:
            breakdown = orig_compute_bid(gpu, task, agent, cluster_state)
            if breakdown.get("eligible", False):
                # Standard risk-blind bid: Q-contribution + perf_score (no risk or efc deductions)
                q_contrib = breakdown.get("q_contribution", 0.0)
                perf = breakdown.get("perf_score", 0.0)
                blind_score = q_contrib + perf
                breakdown["risk_penalty"] = 0.0
                breakdown["efc_penalty"] = 0.0
                breakdown["bid_score"] = round(blind_score, 4)
            return breakdown

        env.marl_manager.compute_bid_score_breakdown = risk_blind_compute_bid


class RiskAwareMARLAdapter(BaseSchedulerAdapter):
    """Full Risk-Aware Multi-Agent Reinforcement Learning Adapter."""

    def __init__(self):
        super().__init__(
            name="Risk-Aware MARL",
            algorithm_id="risk_aware_marl",
            description="Production Risk-Aware MARL: Combines learned Q-values, performance headroom, risk penalties, and Expected Failure Cost (EFC).",
        )

    def configure_environment(self, env: GPUEnvironment, **kwargs) -> None:
        env.scheduler = "marl"
        # Uses default production compute_bid_score_breakdown with active risk and EFC penalties


def get_scheduler_adapter(scheduler_id: str) -> BaseSchedulerAdapter:
    """Factory method to instantiate a scheduler adapter by identifier."""
    registry = {
        "fcfs": FCFSAdapter,
        "round_robin": RoundRobinAdapter,
        "least_loaded": LeastLoadedAdapter,
        "rl": RLAdapter,
        "marl": MARLAdapter,
        "risk_aware_marl": RiskAwareMARLAdapter,
    }
    if scheduler_id not in registry:
        raise ValueError(f"Unknown scheduler '{scheduler_id}'. Registered: {list(registry.keys())}")
    return registry[scheduler_id]()
