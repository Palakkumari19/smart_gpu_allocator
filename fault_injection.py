"""
Fault/Anomaly Injection System for GPU Reliability Simulation

This module implements a configurable fault-injection mechanism for simulating
GPU reliability anomalies and failures. It provides reproducible randomness
for benchmark experiments and integrates with the risk engine.

Key Concepts:
- Anomaly Types: abnormal execution, transient error, checksum mismatch, memory anomaly, etc.
- Severity Levels: LOW, MEDIUM, HIGH
- Fault Injection: Configurable probability-based anomaly injection
- Event Logging: Structured logging of all fault events
- Recovery Cost: Simulated checkpoint/recovery cost for task failures
"""

import threading
from enum import Enum
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import random


class AnomalyType(Enum):
    """Types of GPU anomalies/faults that can be simulated"""
    ABNORMAL_EXECUTION_TIME = "abnormal_execution_time"
    TRANSIENT_COMPUTATIONAL_ERROR = "transient_computational_error"
    CHECKSUM_MISMATCH = "checksum_mismatch"
    MEMORY_ANOMALY = "memory_anomaly"
    PERFORMANCE_DEGRADATION = "performance_degradation"
    TASK_FAILURE = "task_failure"
    GPU_CRASH = "gpu_crash"


class SeverityLevel(Enum):
    """Severity levels for anomalies"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class EventType(Enum):
    """Structured event types for logging"""
    ANOMALY_DETECTED = "anomaly_detected"
    TASK_FAILURE = "task_failure"
    ABNORMAL_EXECUTION = "abnormal_execution"
    GPU_RISK_INCREASED = "gpu_risk_increased"
    GPU_HIGH_RISK = "gpu_high_risk"
    GPU_QUARANTINED = "gpu_quarantined"
    GPU_RECOVERY = "gpu_recovery"
    TASK_RESTARTED = "task_restarted"
    TASK_RECOVERED = "task_recovered"


@dataclass
class FaultEvent:
    """Structured fault event record"""
    event_id: str
    event_type: EventType
    gpu_id: int
    task_id: Optional[str] = None
    anomaly_type: Optional[AnomalyType] = None
    severity: Optional[SeverityLevel] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    details: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for serialization"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "gpu_id": self.gpu_id,
            "task_id": self.task_id,
            "anomaly_type": self.anomaly_type.value if self.anomaly_type else None,
            "severity": self.severity.value if self.severity else None,
            "timestamp": self.timestamp,
            "details": self.details
        }


@dataclass
class AnomalyResult:
    """Result of an anomaly injection"""
    occurred: bool
    anomaly_type: Optional[AnomalyType] = None
    severity: Optional[SeverityLevel] = None
    execution_time_multiplier: float = 1.0
    task_failed: bool = False
    gpu_crashed: bool = False
    details: Dict[str, Any] = field(default_factory=dict)


class FaultInjector:
    """
    Configurable fault injection system for GPU reliability simulation
    
    Supports reproducible randomness and configurable anomaly probabilities.
    """
    
    # Default anomaly probabilities by severity
    DEFAULT_PROBABILITIES = {
        SeverityLevel.LOW: 0.05,    # 5% chance
        SeverityLevel.MEDIUM: 0.15,  # 15% chance
        SeverityLevel.HIGH: 0.30    # 30% chance
    }
    
    # Severity impact multipliers
    SEVERITY_MULTIPLIERS = {
        SeverityLevel.LOW: 1.2,      # 20% impact
        SeverityLevel.MEDIUM: 1.5,   # 50% impact
        SeverityLevel.HIGH: 2.0      # 100% impact
    }
    
    # Anomaly type weights (for random selection)
    ANOMALY_WEIGHTS = {
        AnomalyType.ABNORMAL_EXECUTION_TIME: 0.30,
        AnomalyType.TRANSIENT_COMPUTATIONAL_ERROR: 0.15,
        AnomalyType.CHECKSUM_MISMATCH: 0.10,
        AnomalyType.MEMORY_ANOMALY: 0.15,
        AnomalyType.PERFORMANCE_DEGRADATION: 0.20,
        AnomalyType.TASK_FAILURE: 0.08,
        AnomalyType.GPU_CRASH: 0.02
    }
    
    def __init__(self, seed: Optional[int] = None, enabled: bool = True):
        """
        Initialize fault injector
        
        Args:
            seed: Random seed for reproducibility (None for random)
            enabled: Whether fault injection is enabled
        """
        self._lock = threading.RLock()
        self.lock = self._lock
        self.enabled = enabled
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random
        self.custom_probabilities = self.DEFAULT_PROBABILITIES.copy()
        self.event_log: List[FaultEvent] = []
        self.event_counter = 0
    
    def set_enabled(self, enabled: bool):
        """Enable or disable fault injection (thread-safe)."""
        with self._lock:
            self.enabled = enabled
    
    def set_probability(self, severity: SeverityLevel, probability: float):
        """Set custom probability for a severity level (thread-safe)."""
        with self._lock:
            self.custom_probabilities[severity] = max(0.0, min(1.0, probability))
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility (thread-safe)."""
        with self._lock:
            self.seed = seed
            self.rng = random.Random(seed)
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID (thread-safe)."""
        with self._lock:
            self.event_counter += 1
            return f"evt_{self.event_counter:06d}"
    
    def log_event(self, event: FaultEvent):
        """Log a fault event (thread-safe)."""
        with self._lock:
            self.event_log.append(event)
            # Keep only last 1000 events
            if len(self.event_log) > 1000:
                self.event_log = self.event_log[-1000:]
    
    def get_event_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event log entries (thread-safe)."""
        with self._lock:
            recent_events = self.event_log[-limit:]
            return [event.to_dict() for event in recent_events]
    
    def clear_event_log(self):
        """Clear event log (thread-safe)."""
        with self._lock:
            self.event_log = []
            self.event_counter = 0
    
    def inject_anomaly(self, 
                      gpu_id: int,
                      task_id: Optional[str] = None,
                      severity: Optional[SeverityLevel] = None) -> AnomalyResult:
        """
        Inject an anomaly based on configured probabilities (thread-safe).
        
        Args:
            gpu_id: GPU identifier
            task_id: Task identifier (if applicable)
            severity: Specific severity to use (None for random based on config)
            
        Returns:
            AnomalyResult with details of what occurred
        """
        with self._lock:
            if not self.enabled:
                return AnomalyResult(occurred=False)
            
            # Determine severity
            if severity is None:
                # Random severity based on configured probabilities
                severity = self._random_severity()
            
            # Check if anomaly occurs based on probability
            probability = self.custom_probabilities.get(severity, 0.0)
            if self.rng.random() > probability:
                return AnomalyResult(occurred=False)
            
            # Select anomaly type
            anomaly_type = self._random_anomaly_type()
            
            # Calculate impact based on severity
            severity_multiplier = self.SEVERITY_MULTIPLIERS[severity]
            
            # Generate anomaly-specific effects
            result = self._generate_anomaly_effects(anomaly_type, severity_multiplier)
            result.occurred = True
            result.anomaly_type = anomaly_type
            result.severity = severity
            
            # Log the event
            event = FaultEvent(
                event_id=self._generate_event_id(),
                event_type=EventType.ANOMALY_DETECTED,
                gpu_id=gpu_id,
                task_id=task_id,
                anomaly_type=anomaly_type,
                severity=severity,
                details=result.details
            )
            self.log_event(event)
            
            return result
    
    def _random_severity(self) -> SeverityLevel:
        """Select random severity based on configured probabilities"""
        # Weighted random selection
        weights = [
            self.custom_probabilities[SeverityLevel.LOW],
            self.custom_probabilities[SeverityLevel.MEDIUM],
            self.custom_probabilities[SeverityLevel.HIGH]
        ]
        return self.rng.choices(
            list(SeverityLevel),
            weights=weights,
            k=1
        )[0]
    
    def _random_anomaly_type(self) -> AnomalyType:
        """Select random anomaly type based on weights"""
        return self.rng.choices(
            list(AnomalyType),
            weights=list(self.ANOMALY_WEIGHTS.values()),
            k=1
        )[0]
    
    def _generate_anomaly_effects(self, 
                                 anomaly_type: AnomalyType,
                                 severity_multiplier: float) -> AnomalyResult:
        """Generate effects for a specific anomaly type"""
        result = AnomalyResult(occurred=False)
        
        if anomaly_type == AnomalyType.ABNORMAL_EXECUTION_TIME:
            # Execution time increases
            result.execution_time_multiplier = 1.0 + (self.rng.uniform(0.5, 1.5) * severity_multiplier)
            result.details = {
                "description": "Abnormal execution time detected",
                "time_multiplier": result.execution_time_multiplier
            }
            
        elif anomaly_type == AnomalyType.TRANSIENT_COMPUTATIONAL_ERROR:
            # Transient error, may cause retry
            result.execution_time_multiplier = 1.0 + (self.rng.uniform(0.2, 0.5) * severity_multiplier)
            result.details = {
                "description": "Transient computational error detected",
                "retry_required": self.rng.random() > 0.5
            }
            
        elif anomaly_type == AnomalyType.CHECKSUM_MISMATCH:
            # Checksum mismatch, task may fail
            result.task_failed = self.rng.random() > 0.7  # 30% chance of failure
            result.execution_time_multiplier = 1.0 + (self.rng.uniform(0.1, 0.3) * severity_multiplier)
            result.details = {
                "description": "Checksum mismatch detected",
                "task_failed": result.task_failed
            }
            
        elif anomaly_type == AnomalyType.MEMORY_ANOMALY:
            # Memory anomaly, affects performance
            result.execution_time_multiplier = 1.0 + (self.rng.uniform(0.3, 0.8) * severity_multiplier)
            result.details = {
                "description": "Memory anomaly detected",
                "corruption_risk": self.rng.uniform(0.1, 0.4)
            }
            
        elif anomaly_type == AnomalyType.PERFORMANCE_DEGRADATION:
            # Performance degradation
            result.execution_time_multiplier = 1.0 + (self.rng.uniform(0.5, 1.0) * severity_multiplier)
            result.details = {
                "description": "Performance degradation detected",
                "degradation_factor": result.execution_time_multiplier
            }
            
        elif anomaly_type == AnomalyType.TASK_FAILURE:
            # Task failure
            result.task_failed = True
            result.execution_time_multiplier = 0.5  # Failed early
            result.details = {
                "description": "Task failure detected",
                "failure_point": self.rng.uniform(0.1, 0.9)  # Failed at this point
            }
            
        elif anomaly_type == AnomalyType.GPU_CRASH:
            # GPU crash
            result.gpu_crashed = True
            result.task_failed = True
            result.details = {
                "description": "GPU crash detected",
                "crash_type": self.rng.choice(["thermal", "memory", "hardware"])
            }
        
        return result
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get fault injection statistics (thread-safe)."""
        with self._lock:
            if not self.event_log:
                return {
                    "total_events": 0,
                    "by_type": {},
                    "by_severity": {},
                    "by_gpu": {}
                }
            
            by_type = {}
            by_severity = {}
            by_gpu = {}
            
            for event in self.event_log:
                # Count by type
                if event.event_type.value not in by_type:
                    by_type[event.event_type.value] = 0
                by_type[event.event_type.value] += 1
                
                # Count by severity
                if event.severity:
                    if event.severity.value not in by_severity:
                        by_severity[event.severity.value] = 0
                    by_severity[event.severity.value] += 1
                
                # Count by GPU
                if event.gpu_id not in by_gpu:
                    by_gpu[event.gpu_id] = 0
                by_gpu[event.gpu_id] += 1
            
            return {
                "total_events": len(self.event_log),
                "by_type": by_type,
                "by_severity": by_severity,
                "by_gpu": by_gpu
            }


class FailureImpactCalculator:
    """
    Calculate failure impact score for video tasks
    
    The failure impact score (0-1) represents the cost of a task failing.
    Short/light videos have lower impact than long/extreme videos.
    """
    
    def __init__(self):
        """Initialize failure impact calculator"""
        pass
    
    def calculate_failure_impact(self, task) -> float:
        """
        Calculate normalized failure impact score for a video task
        
        Args:
            task: VideoTask object
            
        Returns:
            Normalized impact score (0-1)
        """
        # Get task properties
        execution_time = getattr(task, 'execution_time', 1)
        memory_required = getattr(task, 'memory_required', 1)
        workload_severity = getattr(task, 'workload_severity', 'LOW')
        
        # Severity multiplier
        severity_multipliers = {
            'LOW': 0.3,
            'MEDIUM': 0.5,
            'HIGH': 0.8,
            'VERY HIGH': 1.0
        }
        severity_multiplier = severity_multipliers.get(workload_severity, 0.5)
        
        # Resource intensity (normalized to 0-1)
        # Max reference: 8GB memory, 20 steps execution time
        memory_intensity = min(1.0, memory_required / 8.0)
        time_intensity = min(1.0, execution_time / 20.0)
        resource_intensity = (memory_intensity + time_intensity) / 2.0
        
        # Calculate base impact
        base_impact = (severity_multiplier * 0.6) + (resource_intensity * 0.4)
        
        # Normalize to [0, 1]
        impact_score = max(0.0, min(1.0, base_impact))
        
        return impact_score
    
    def calculate_restart_cost(self, task) -> float:
        """
        Calculate simulated restart/recovery cost for a task
        
        Args:
            task: VideoTask object
            
        Returns:
            Restart cost in simulation steps
        """
        # Restart cost is proportional to task complexity
        execution_time = getattr(task, 'execution_time', 1)
        memory_required = getattr(task, 'memory_required', 1)
        
        # Base restart cost: 20-50% of original execution time
        base_restart_cost = execution_time * 0.3
        
        # Add checkpoint overhead (proportional to memory usage)
        checkpoint_overhead = memory_required * 0.1
        
        total_restart_cost = base_restart_cost + checkpoint_overhead
        
        return max(1.0, total_restart_cost)  # Minimum 1 step


class ExpectedFailureCostCalculator:
    """
    Calculate expected failure cost for scheduling decisions
    
    expected_failure_cost = GPU_risk_score * failure_impact * recovery_cost
    """
    
    def __init__(self):
        """Initialize expected failure cost calculator"""
        self.impact_calculator = FailureImpactCalculator()
    
    def calculate_expected_failure_cost(self, 
                                       gpu_risk_score: float,
                                       task,
                                       recovery_cost: Optional[float] = None) -> float:
        """
        Calculate expected failure cost for a task on a GPU
        
        Args:
            gpu_risk_score: GPU risk score (0-1)
            task: VideoTask object
            recovery_cost: Optional recovery cost (calculated if not provided)
            
        Returns:
            Expected failure cost
        """
        # Calculate failure impact
        failure_impact = self.impact_calculator.calculate_failure_impact(task)
        
        # Calculate recovery cost if not provided
        if recovery_cost is None:
            recovery_cost = self.impact_calculator.calculate_restart_cost(task)
        
        # Calculate expected failure cost
        expected_cost = gpu_risk_score * failure_impact * recovery_cost
        
        return expected_cost
