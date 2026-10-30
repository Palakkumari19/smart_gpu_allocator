"""
GPU Risk Engine for Proactive Reliability-Aware GPU Scheduling

This module implements a risk assessment system for GPU scheduling decisions.
It calculates dynamic risk scores based on observable GPU/system signals and
maintains reliability history for each GPU.

Key Concepts:
- Risk Score: Normalized 0.0-1.0 score indicating GPU reliability risk
- Risk Levels: HEALTHY, LOW_RISK, SUSPICIOUS, HIGH_RISK
- Effective Capacity: Reliability-adjusted scheduling metric
- Performance Degradation: Expected vs actual execution time comparison
"""

import threading
from enum import Enum
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field


class RiskLevel(Enum):
    """GPU risk level classification"""
    HEALTHY = "HEALTHY"
    LOW_RISK = "LOW_RISK"
    SUSPICIOUS = "SUSPICIOUS"
    HIGH_RISK = "HIGH_RISK"


@dataclass
class ReliabilityHistory:
    """Track historical reliability metrics for a GPU"""
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    crash_events: int = 0
    recovery_events: int = 0
    preemptions: int = 0
    anomalies_detected: int = 0
    abnormal_executions: int = 0
    total_execution_time: float = 0.0
    successful_execution_time: float = 0.0
    
    @property
    def failure_rate(self) -> float:
        """Calculate failure rate as percentage"""
        if self.total_tasks == 0:
            return 0.0
        return (self.failed_tasks / self.total_tasks) * 100.0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_tasks == 0:
            return 0.0
        return (self.completed_tasks / self.total_tasks) * 100.0


@dataclass
class RiskComponents:
    """Individual components that contribute to overall risk score"""
    temperature_risk: float = 0.0
    memory_pressure: float = 0.0
    fragmentation_risk: float = 0.0
    failure_history: float = 0.0
    crash_history: float = 0.0
    anomaly_score: float = 0.0
    performance_degradation: float = 0.0
    utilization_risk: float = 0.0


@dataclass
class GPURiskState:
    """Complete risk state for a GPU"""
    risk_score: float = 0.0
    risk_level: RiskLevel = RiskLevel.HEALTHY
    components: RiskComponents = field(default_factory=RiskComponents)
    reliability_history: ReliabilityHistory = field(default_factory=ReliabilityHistory)
    effective_capacity: float = 0.0
    raw_capacity: float = 0.0
    is_quarantined: bool = False
    quarantine_reason: Optional[str] = None


class RiskEngine:
    """
    GPU Risk Engine for calculating dynamic risk scores
    
    The risk score is calculated from observable GPU/system signals using
    a weighted formula:
    
    risk_score = w_temp * temperature_risk
               + w_memory * memory_pressure
               + w_fragmentation * fragmentation_risk
               + w_failure * failure_history
               + w_crash * crash_history
               + w_anomaly * anomaly_score
               + w_degradation * performance_degradation
               + w_utilization * utilization_risk
    """
    
    # Risk level thresholds (configurable)
    RISK_THRESHOLDS = {
        RiskLevel.HEALTHY: 0.25,
        RiskLevel.LOW_RISK: 0.50,
        RiskLevel.SUSPICIOUS: 0.75,
        RiskLevel.HIGH_RISK: 1.00
    }
    
    # Quarantine threshold (configurable)
    QUARANTINE_THRESHOLD = 0.85
    
    # Risk component weights (configurable)
    WEIGHTS = {
        'temperature': 0.15,
        'memory': 0.10,
        'fragmentation': 0.05,
        'failure': 0.20,
        'crash': 0.15,
        'anomaly': 0.25,  # Increased from 0.10 to 0.25 for Phase 2 fault injection
        'degradation': 0.05,
        'utilization': 0.05
    }
    
    # Temperature thresholds (Celsius)
    TEMP_SAFE = 50.0
    TEMP_WARNING = 70.0
    TEMP_CRITICAL = 85.0
    
    def __init__(self):
        """Initialize risk engine"""
        self._lock = threading.RLock()
        self.lock = self._lock
        self.gpu_risk_states: Dict[int, GPURiskState] = {}
    
    def initialize_gpu(self, gpu_id: int, memory_capacity: float, reliability_history=None) -> GPURiskState:
        """Initialize risk state for a GPU (thread-safe)."""
        with self._lock:
            risk_state = GPURiskState(
                risk_score=0.0,
                risk_level=RiskLevel.HEALTHY,
                raw_capacity=memory_capacity,
                effective_capacity=memory_capacity,
                reliability_history=reliability_history if reliability_history else ReliabilityHistory()
            )
            self.gpu_risk_states[gpu_id] = risk_state
            return risk_state
    
    def calculate_risk_score(self, gpu) -> RiskComponents:
        """
        Calculate individual risk components for a GPU
        
        Args:
            gpu: GPU object with current state
            
        Returns:
            RiskComponents object with individual component scores
        """
        components = RiskComponents()
        
        # 1. Temperature risk (normalized 0-1)
        # Risk increases as temperature approaches critical threshold
        if gpu.temperature < self.TEMP_SAFE:
            components.temperature_risk = 0.0
        elif gpu.temperature < self.TEMP_WARNING:
            components.temperature_risk = (gpu.temperature - self.TEMP_SAFE) / (self.TEMP_WARNING - self.TEMP_SAFE)
        elif gpu.temperature < self.TEMP_CRITICAL:
            components.temperature_risk = 0.5 + 0.5 * (gpu.temperature - self.TEMP_WARNING) / (self.TEMP_CRITICAL - self.TEMP_WARNING)
        else:
            components.temperature_risk = 1.0
        
        # 2. Memory pressure (normalized 0-1)
        # Risk increases with memory usage percentage
        memory_usage = gpu.get_memory_usage_percentage()
        components.memory_pressure = min(1.0, memory_usage / 100.0)
        
        # 3. Fragmentation risk (normalized 0-1)
        # Risk increases with memory fragmentation
        fragmentation = gpu.get_fragmentation_percentage()
        components.fragmentation_risk = min(1.0, fragmentation / 50.0)  # 50% fragmentation = max risk
        
        # 4. Utilization risk (normalized 0-1)
        # Risk increases with high utilization
        components.utilization_risk = min(1.0, gpu.utilization / 100.0)
        
        return components
    
    def calculate_history_risk(self, history: ReliabilityHistory) -> Tuple[float, float]:
        """
        Calculate risk from failure and crash history
        
        Args:
            history: ReliabilityHistory object
            
        Returns:
            Tuple of (failure_history_risk, crash_history_risk)
        """
        # Failure history risk based on failure rate
        failure_rate = history.failure_rate
        failure_history_risk = min(1.0, failure_rate / 50.0)  # 50% failure rate = max risk
        
        # Crash history risk based on crash events
        # More recent crashes contribute more to risk
        crash_history_risk = min(1.0, history.crash_events / 10.0)  # 10 crashes = max risk
        
        return failure_history_risk, crash_history_risk
    
    def calculate_performance_degradation(self, 
                                         expected_time: float, 
                                         actual_time: float,
                                         history: ReliabilityHistory) -> float:
        """
        Calculate performance degradation from expected vs actual execution time
        
        Args:
            expected_time: Expected execution time
            actual_time: Actual execution time
            history: ReliabilityHistory for tracking
            
        Returns:
            Normalized degradation score (0-1)
        """
        if expected_time <= 0:
            return 0.0
        
        # Calculate degradation ratio
        degradation_ratio = max(0.0, (actual_time - expected_time) / expected_time)
        
        # Normalize to 0-1 (2x expected time = max degradation)
        degradation_score = min(1.0, degradation_ratio / 2.0)
        
        # Track in history
        if degradation_ratio > 0.1:  # More than 10% slower is considered abnormal
            history.abnormal_executions += 1
        
        return degradation_score
    
    def calculate_anomaly_score(self, history: ReliabilityHistory) -> float:
        """
        Calculate anomaly score from detected anomalies
        
        Args:
            history: ReliabilityHistory object
            
        Returns:
            Normalized anomaly score (0-1)
        """
        # Anomaly score increases with number of detected anomalies
        # 5 anomalies = max risk (more aggressive for Phase 2 fault injection)
        return min(1.0, history.anomalies_detected / 5.0)
    
    def compute_risk_score(self, 
                          gpu_id: int,
                          gpu,
                          components: RiskComponents,
                          history: ReliabilityHistory) -> GPURiskState:
        """
        Compute complete risk score for a GPU
        
        Args:
            gpu_id: GPU identifier
            gpu: GPU object with current state
            components: RiskComponents with individual component scores
            history: ReliabilityHistory object
            
        Returns:
            Updated GPURiskState with calculated risk score
        """
        with self._lock:
            # Get history-based risks
            failure_history_risk, crash_history_risk = self.calculate_history_risk(history)
            
            # Get anomaly score
            anomaly_score = self.calculate_anomaly_score(history)
            
            # Update components with history-based values
            components.failure_history = failure_history_risk
            components.crash_history = crash_history_risk
            components.anomaly_score = anomaly_score
            
            # Calculate weighted risk score
            risk_score = (
                self.WEIGHTS['temperature'] * components.temperature_risk +
                self.WEIGHTS['memory'] * components.memory_pressure +
                self.WEIGHTS['fragmentation'] * components.fragmentation_risk +
                self.WEIGHTS['failure'] * components.failure_history +
                self.WEIGHTS['crash'] * components.crash_history +
                self.WEIGHTS['anomaly'] * components.anomaly_score +
                self.WEIGHTS['degradation'] * components.performance_degradation +
                self.WEIGHTS['utilization'] * components.utilization_risk
            )
            
            # Normalize to [0, 1]
            risk_score = max(0.0, min(1.0, risk_score))
            
            # Determine risk level
            risk_level = self._classify_risk_level(risk_score)
            
            # Calculate effective capacity
            raw_capacity = gpu.memory_capacity - gpu.current_memory
            effective_capacity = raw_capacity * (1.0 - risk_score)
            
            # Check quarantine status (preserve manual quarantine injection if previously set)
            existing_state = self.gpu_risk_states.get(gpu_id)
            was_quarantined = bool(existing_state and getattr(existing_state, 'is_quarantined', False))
            is_quarantined = (risk_score >= self.QUARANTINE_THRESHOLD) or was_quarantined
            quarantine_reason = None
            if is_quarantined:
                if risk_score >= self.QUARANTINE_THRESHOLD:
                    quarantine_reason = self._generate_quarantine_reason(components, history)
                elif was_quarantined:
                    quarantine_reason = getattr(existing_state, 'quarantine_reason', 'High Risk Quarantine Policy')
            
            # Update or create risk state
            risk_state = existing_state
            if risk_state is None:
                risk_state = self.initialize_gpu(gpu_id, gpu.memory_capacity, history)
            else:
                # Use the passed history instead of creating a new one
                risk_state.reliability_history = history
            
            risk_state.risk_score = risk_score
            risk_state.risk_level = risk_level
            risk_state.components = components
            risk_state.effective_capacity = effective_capacity
            risk_state.raw_capacity = raw_capacity
            risk_state.is_quarantined = is_quarantined
            risk_state.quarantine_reason = quarantine_reason
            
            self.gpu_risk_states[gpu_id] = risk_state
            
            return risk_state
    
    def _classify_risk_level(self, risk_score: float) -> RiskLevel:
        """Classify risk score into risk level"""
        if risk_score < self.RISK_THRESHOLDS[RiskLevel.HEALTHY]:
            return RiskLevel.HEALTHY
        elif risk_score < self.RISK_THRESHOLDS[RiskLevel.LOW_RISK]:
            return RiskLevel.LOW_RISK
        elif risk_score < self.RISK_THRESHOLDS[RiskLevel.SUSPICIOUS]:
            return RiskLevel.SUSPICIOUS
        else:
            return RiskLevel.HIGH_RISK
    
    def _generate_quarantine_reason(self, 
                                    components: RiskComponents,
                                    history: ReliabilityHistory) -> str:
        """Generate reason for quarantine based on risk components"""
        reasons = []
        
        if components.temperature_risk > 0.7:
            reasons.append("critical temperature")
        if components.memory_pressure > 0.8:
            reasons.append("severe memory pressure")
        if components.failure_history > 0.5:
            reasons.append("high failure rate")
        if components.crash_history > 0.5:
            reasons.append("frequent crashes")
        if components.anomaly_score > 0.5:
            reasons.append("repeated anomalies")
        if components.performance_degradation > 0.5:
            reasons.append("severe performance degradation")
        
        if not reasons:
            reasons.append("elevated overall risk")
        
        return ", ".join(reasons)
    
    def get_risk_state(self, gpu_id: int) -> Optional[GPURiskState]:
        """Get current risk state for a GPU (thread-safe)."""
        with self._lock:
            return self.gpu_risk_states.get(gpu_id)
    
    def update_task_completion(self, 
                              gpu_id: int,
                              expected_time: float,
                              actual_time: float,
                              success: bool):
        """Update reliability history after task completion (thread-safe)."""
        with self._lock:
            risk_state = self.gpu_risk_states.get(gpu_id)
            if risk_state is None:
                return
            
            history = risk_state.reliability_history
            history.total_tasks += 1
            history.total_execution_time += actual_time
            
            if success:
                history.completed_tasks += 1
                history.successful_execution_time += actual_time
            else:
                history.failed_tasks += 1
            
            # Calculate performance degradation
            degradation = self.calculate_performance_degradation(
                expected_time, actual_time, history
            )
            risk_state.components.performance_degradation = degradation
    
    def record_crash(self, gpu_id: int):
        """Record a crash event for a GPU (thread-safe)."""
        with self._lock:
            risk_state = self.gpu_risk_states.get(gpu_id)
            if risk_state is None:
                return
            risk_state.reliability_history.crash_events += 1
    
    def record_recovery(self, gpu_id: int):
        """Record a recovery event for a GPU (thread-safe)."""
        with self._lock:
            risk_state = self.gpu_risk_states.get(gpu_id)
            if risk_state is None:
                return
            risk_state.reliability_history.recovery_events += 1
    
    def record_preemption(self, gpu_id: int):
        """Record a preemption event for a GPU (thread-safe)."""
        with self._lock:
            risk_state = self.gpu_risk_states.get(gpu_id)
            if risk_state is None:
                return
            risk_state.reliability_history.preemptions += 1
    
    def record_anomaly(self, gpu_id: int):
        """Record an anomaly detection for a GPU (thread-safe)."""
        with self._lock:
            risk_state = self.gpu_risk_states.get(gpu_id)
            if risk_state is None:
                return
            risk_state.reliability_history.anomalies_detected += 1
    
    def check_quarantine_recovery(self, gpu_id: int) -> bool:
        """Check if a quarantined GPU can recover (thread-safe)."""
        with self._lock:
            risk_state = self.gpu_risk_states.get(gpu_id)
            if risk_state is None or not risk_state.is_quarantined:
                return False
            
            # Recovery condition: risk score dropped below threshold
            if risk_state.risk_score < self.QUARANTINE_THRESHOLD - 0.1:
                risk_state.is_quarantined = False
                risk_state.quarantine_reason = None
                risk_state.reliability_history.recovery_events += 1
                return True
            
            return False
    
    def reset_gpu(self, gpu_id: int):
        """Reset risk state for a GPU (thread-safe)."""
        with self._lock:
            if gpu_id in self.gpu_risk_states:
                risk_state = self.gpu_risk_states[gpu_id]
                risk_state.risk_score = 0.0
                risk_state.risk_level = RiskLevel.HEALTHY
                risk_state.components = RiskComponents()
                risk_state.reliability_history = ReliabilityHistory()
                risk_state.is_quarantined = False
                risk_state.quarantine_reason = None
    
    def get_global_risk_metrics(self) -> Dict:
        """Get global risk metrics across all GPUs (thread-safe)."""
        with self._lock:
            if not self.gpu_risk_states:
                return {
                    'average_risk': 0.0,
                    'maximum_risk': 0.0,
                    'risk_variance': 0.0,
                    'total_anomalies': 0,
                    'total_crashes': 0,
                    'quarantined_gpus': 0
                }
            
            risk_scores = [state.risk_score for state in self.gpu_risk_states.values()]
            avg_risk = sum(risk_scores) / len(risk_scores)
            max_risk = max(risk_scores)
            
            # Calculate variance
            variance = sum((r - avg_risk) ** 2 for r in risk_scores) / len(risk_scores)
            
            total_anomalies = sum(state.reliability_history.anomalies_detected 
                                for state in self.gpu_risk_states.values())
            total_crashes = sum(state.reliability_history.crash_events 
                              for state in self.gpu_risk_states.values())
            quarantined_count = sum(1 for state in self.gpu_risk_states.values() 
                                  if state.is_quarantined)
            
            return {
                'average_risk': avg_risk,
                'maximum_risk': max_risk,
                'risk_variance': variance,
                'total_anomalies': total_anomalies,
                'total_crashes': total_crashes,
                'quarantined_gpus': quarantined_count
            }
