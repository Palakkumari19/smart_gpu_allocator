import random
import threading
import time
from typing import Optional, Dict, Any, List
from risk_engine import RiskEngine, ReliabilityHistory
from fault_injection import FaultInjector, FaultEvent, EventType, AnomalyResult

class GPU:
    def __init__(self, gpu_id, memory_capacity=8):  # Default 8GB as specified
        self._lock = threading.RLock()
        self.lock = self._lock  # Public alias for lock coordination
        self.id = gpu_id
        self.memory_capacity = memory_capacity  # in GB
        self.current_memory = 0.0  # in GB
        self.utilization = 0  # percentage
        self.temperature = 30.0  # Celsius
        self.task_queue = []
        self.running_tasks = []  # For parallel execution
        self.crashed = False
        self.total_latency = 0.0
        self.completed_tasks = 0
        
        # New features
        self.warm_up_delay = 2  # time steps for warm-up
        self.time_steps_active = 0
        self.is_warmed_up = False
        self.max_parallel_tasks = 3  # Max tasks can run in parallel
        self.last_started_tasks = 0
        self.crash_threshold_temp = 85  # Crash at 85°C
        self.memory_overload_factor = 1.2  # Crash at 120% memory usage
        
        # Advanced features - Step 2
        self.memory_fragments = []  # List of memory fragments
        self.preempted_tasks = []  # Tasks that were paused
        self.crash_cooldown_timer = 0  # Cooldown after crash
        self.total_preemptions = 0  # Track preemption count
        self.high_realism_mode = True  # Enable high realism features
        self.deterministic_physics = False
        self.total_crash_events = 0
        
        # Phase 3: Reliability and Risk Engine integration
        self.reliability_history = ReliabilityHistory()
        self.risk_engine = None  # Will be set by environment
        self.expected_execution_times = {}  # Track expected vs actual execution times
        
        # Phase 4: Fault Injection System integration
        self.fault_injector = None  # Will be set by environment
        self.fault_events = []  # Track fault events for this GPU

    def enable_deterministic_physics(self):
        """Make GPU stochastic events deterministic so schedulers can be compared fairly."""
        with self._lock:
            self.deterministic_physics = True

    def _rand(self):
        if self.deterministic_physics:
            return 0.5
        return random.random()

    def _randint(self, low, high):
        if self.deterministic_physics:
            return (low + high) // 2
        return random.randint(low, high)

    def _uniform(self, low, high):
        if self.deterministic_physics:
            return (low + high) / 2.0
        return random.uniform(low, high)

    def _enter_crash(self):
        with self._lock:
            if not self.crashed:
                self.total_crash_events += 1
                self.record_crash_event()  # Phase 3: Record crash in risk engine
            self.crashed = True

    def crash(self):
        """Thread-safe public method to trigger a GPU crash."""
        with self._lock:
            self._enter_crash()

    def recover(self):
        """Thread-safe public method to recover GPU from crash."""
        with self._lock:
            self.crashed = False
            self.crash_cooldown_timer = 0
            self.temperature = max(30.0, self.temperature - 20.0)
            self.record_recovery_event()

    def preempt_task(self, task_identifier) -> bool:
        """
        Atomically preempt a task by ID or task instance.
        Thread-safe: guarded by self._lock.
        """
        with self._lock:
            target = None
            for t in self.running_tasks:
                if t == task_identifier or getattr(t, 'id', None) == task_identifier:
                    target = t
                    break
            if target is None:
                for t in self.task_queue:
                    if t == task_identifier or getattr(t, 'id', None) == task_identifier:
                        target = t
                        break

            if target is not None:
                if target in self.running_tasks:
                    self.running_tasks.remove(target)
                if target in self.task_queue:
                    self.task_queue.remove(target)
                target.preempted = True
                if hasattr(target, 'preemption_count'):
                    target.preemption_count += 1
                self.preempted_tasks.append(target)
                self.task_queue.append(target)
                self.total_preemptions += 1
                self.record_preemption_event()
                self.temperature = max(30.0, self.temperature - 5.0)
                return True
            return False

    def assign_task(self, task):
        """
        Atomically check resources and assign a task to this GPU.
        Thread-safe: guarded by self._lock.
        """
        with self._lock:
            if self.crashed:
                return False
            
            # Check crash cooldown
            if self.crash_cooldown_timer > 0:
                return False
            
            # Phase 4: Check if GPU is quarantined
            if self.is_quarantined():
                return False
            
            # Memory fragmentation simulation: only active after tasks have executed and completed
            if self.high_realism_mode and self.completed_tasks > 0 and self.memory_fragments:
                available_memory = max(0.0, self.memory_capacity - self.current_memory)
                total_fragmented = min(sum(self.memory_fragments), available_memory)
                available_contiguous = max(0.0, available_memory - total_fragmented)
                
                # Fragmentation can cause allocation failure when contiguous free block is too small
                if available_memory >= task.memory_required > available_contiguous:
                    # Allocation failed due to fragmentation
                    frag_to_add = min(task.memory_required * 0.3, max(0.0, available_memory - total_fragmented))
                    if frag_to_add > 0.01:
                        self.memory_fragments.append(round(frag_to_add, 4))
                    return False
            
            # Check memory capacity (with overload factor):
            # Physical capacity is memory_capacity (8 GB). Overload is permitted up to
            # memory_capacity * memory_overload_factor (9.6 GB). Exceeding overload threshold triggers crash.
            if self.current_memory + task.memory_required > self.memory_capacity * self.memory_overload_factor:
                self._enter_crash()
                return False
                
            self.task_queue.append(task)
            self.current_memory = max(0.0, round(self.current_memory + task.memory_required, 4))

            # Memory fragmentation - create fragments only after tasks have completed in workload
            if self.high_realism_mode and self.completed_tasks > 0 and self._rand() < 0.3:  # 30% chance
                available_memory = max(0.0, self.memory_capacity - self.current_memory)
                total_fragmented = min(sum(self.memory_fragments), available_memory)
                max_allowable_fragments = available_memory * 0.5
                if total_fragmented < max_allowable_fragments:
                    fragment_size = min(task.memory_required * self._uniform(0.1, 0.4), max_allowable_fragments - total_fragmented)
                    if fragment_size > 0.01:
                        self.memory_fragments.append(round(fragment_size, 4))

            # increase utilization based on current load
            load_factor = len(self.task_queue) / 10.0
            self.utilization = min(100, self.utilization + self._randint(5, 12) + int(load_factor * 10))

            # increase temperature more with load
            temp_increase = self._randint(1, 4) + len(self.task_queue) // 2
            self.temperature += temp_increase

            # latency increases with queue size and fragmentation
            fragmentation_latency = len(self.memory_fragments) * 0.5
            self.total_latency += len(self.task_queue) * (1 + len(self.running_tasks) * 0.5 + fragmentation_latency)
            
            return True

    def process_tasks(self):
        """
        Process GPU queue and running tasks for one simulation step.
        Thread-safe: guarded by self._lock.
        """
        with self._lock:
            # Check crash cooldown
            if self.crashed or self.crash_cooldown_timer > 0:
                self.crash_cooldown_timer -= 1
                if self.crash_cooldown_timer <= 0:
                    # Reset from crash with high realism boost
                    self.temperature = max(30.0, self.temperature - 20.0)  # Big cooldown
                    self.crashed = False
                    self.current_memory = max(0.0, min(self.memory_capacity, self.current_memory))
                    self.memory_fragments = []  # Defragmentation
                    self.record_recovery_event()  # Phase 3: Record recovery
                    # Resume preempted tasks
                    if self.high_realism_mode and self.preempted_tasks:
                        resume_count = min(2, len(self.preempted_tasks))  # Resume up to 2 tasks
                        for _ in range(resume_count):
                            task = self.preempted_tasks.pop(0)
                            self.task_queue.append(task)
                return
                
            # Handle warm-up delay
            self.time_steps_active += 1
            if not self.is_warmed_up and self.time_steps_active >= self.warm_up_delay:
                self.is_warmed_up = True
            elif not self.is_warmed_up:
                # Reduced processing during warm-up
                return
                
            # Check crash conditions
            if self.temperature >= self.crash_threshold_temp:
                self._enter_crash()
                self.crash_cooldown_timer = 5  # 5 step cooldown
                # Preempt running tasks
                if self.high_realism_mode:
                    for task in self.running_tasks:
                        task.preempted = True
                        self.preempted_tasks.append(task)
                    self.running_tasks = []
                return
                
            # Task preemption logic (high realism)
            if self.high_realism_mode and len(self.running_tasks) > 0:
                # Preempt tasks under high load or temperature
                if self.temperature > 75 or len(self.running_tasks) == self.max_parallel_tasks:
                    if self._rand() < 0.2:  # 20% chance of preemption
                        task_to_preempt = (
                            self.running_tasks[0]
                            if self.deterministic_physics
                            else random.choice(self.running_tasks)
                        )

                        self.running_tasks.remove(task_to_preempt)

                        # Preserve execution progress so the task can resume later.
                        task_to_preempt.preempted = True
                        self.preempted_tasks.append(task_to_preempt)

                        # Re-queue the preempted task instead of allowing it to
                        # disappear from the execution lifecycle.
                        self.task_queue.append(task_to_preempt)

                        self.total_preemptions += 1
                        self.record_preemption_event()  # Phase 3: Record preemption

                        # Immediate cooldown from preemption
                        self.temperature = max(30.0, self.temperature - 5.0)
                
            # Multi-task parallel execution
            available_slots = self.max_parallel_tasks - len(self.running_tasks)
            tasks_to_start = min(available_slots, len(self.task_queue))
            self.last_started_tasks = tasks_to_start
            
            # Start tasks in parallel
            for _ in range(tasks_to_start):
                if self.task_queue:
                    task = self.task_queue.pop(0)

                    # Reset progress only for a new task.
                    # Preempted tasks resume from their existing progress.
                    if not task.preempted:
                        task.execution_progress = 0

                    task.preempted = False

                    if task in self.preempted_tasks:
                        self.preempted_tasks.remove(task)

                    self.running_tasks.append(task)
            
            # Process running tasks
            completed_tasks = []
            for task in self.running_tasks:
                task.execution_progress += 1
                if task.execution_progress >= task.execution_time:
                    completed_tasks.append(task)
                    
            # Remove completed tasks atomically and release memory
            for task in completed_tasks:
                if task in self.running_tasks:
                    self.running_tasks.remove(task)
                # Ensure memory release is atomic and never negative
                self.current_memory = max(0.0, round(self.current_memory - task.memory_required, 4))
                self.completed_tasks += 1

                # Memory fragmentation cleanup
                if self.high_realism_mode and self.memory_fragments:
                    # Clean some fragments when tasks complete
                    if self._rand() < 0.4:  # 40% chance
                        self.memory_fragments.pop(0)
                    avail = max(0.0, self.memory_capacity - self.current_memory)
                    while self.memory_fragments and sum(self.memory_fragments) > avail:
                        self.memory_fragments.pop(0)

                # decrease utilization
                self.utilization = max(0, self.utilization - self._randint(4, 8))

                # cool down
                self.temperature = max(30.0, self.temperature - self._randint(2, 5))
                
            # Natural temperature increase with running tasks
            if self.running_tasks:
                self.temperature += len(self.running_tasks) * 0.5

    def get_queue_length(self):
        with self._lock:
            return len(self.task_queue)
        
    def get_running_tasks_count(self):
        with self._lock:
            return len(self.running_tasks)
        
    def get_total_load(self):
        with self._lock:
            return len(self.task_queue) + len(self.running_tasks)
        
    def get_memory_usage_percentage(self):
        with self._lock:
            return max(0.0, min(100.0, (self.current_memory / self.memory_capacity) * 100))

    def get_available_memory(self) -> float:
        """Return available memory in GB (thread-safe)."""
        with self._lock:
            return max(0.0, round(self.memory_capacity - self.current_memory, 4))
        
    def is_available_for_task(self, task_memory):
        with self._lock:
            return (not self.crashed and 
                    not self.is_quarantined() and
                    self.crash_cooldown_timer <= 0 and
                    self.current_memory + task_memory <= self.memory_capacity and
                    len(self.running_tasks) < self.max_parallel_tasks)
                
    def get_memory_fragments_count(self):
        """Return number of memory fragments"""
        with self._lock:
            return len(self.memory_fragments)
        
    def get_preempted_tasks_count(self):
        """Return number of preempted tasks"""
        with self._lock:
            return len(self.preempted_tasks)
        
    def get_fragmentation_percentage(self):
        """Calculate memory fragmentation percentage [0.0, 100.0]"""
        with self._lock:
            if not self.memory_fragments:
                return 0.0
            available_memory = max(0.0, self.memory_capacity - self.current_memory)
            if available_memory <= 0.05:
                return 0.0
            total_fragmented = min(sum(self.memory_fragments), available_memory)
            return max(0.0, min(100.0, round((total_fragmented / self.memory_capacity) * 100.0, 2)))
        
    def is_in_cooldown(self):
        """Check if GPU is in crash cooldown"""
        with self._lock:
            return self.crash_cooldown_timer > 0 or self.crashed
        
    def get_cooldown_remaining(self):
        """Get remaining cooldown steps"""
        with self._lock:
            return max(0, self.crash_cooldown_timer)
        
    def defragment_memory(self):
        """Manual memory defragmentation"""
        with self._lock:
            if self.memory_fragments:
                self.memory_fragments = []
                # Small performance penalty for defragmentation
                self.temperature += 2
                return True
            return False
        
    def reset(self):
        """Reset GPU state for new simulation"""
        with self._lock:
            self.current_memory = 0.0
            self.utilization = 0
            self.temperature = 30.0
            self.task_queue = []
            self.running_tasks = []
            self.crashed = False
            self.total_latency = 0.0
            self.completed_tasks = 0
            self.time_steps_active = 0
            self.is_warmed_up = False
            
            # Reset advanced features
            self.memory_fragments = []
            self.preempted_tasks = []
            self.crash_cooldown_timer = 0
            self.total_preemptions = 0
            self.total_crash_events = 0
            
            # Reset reliability history
            self.reliability_history = ReliabilityHistory()
            self.expected_execution_times = {}
            
            # Reset risk engine state if available
            if self.risk_engine:
                self.risk_engine.reset_gpu(self.id)

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Return a thread-safe point-in-time snapshot of GPU state."""
        with self._lock:
            return {
                "id": self.id,
                "memory_capacity": self.memory_capacity,
                "current_memory": self.current_memory,
                "memory_percentage": self.get_memory_usage_percentage(),
                "temperature": self.temperature,
                "utilization": self.utilization,
                "queue_length": len(self.task_queue),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": self.completed_tasks,
                "fragmentation": self.get_fragmentation_percentage(),
                "preemptions": self.total_preemptions,
                "crashed": self.crashed,
                "cooldown": max(0, self.crash_cooldown_timer),
                "is_quarantined": self.is_quarantined()
            }
    
    # ============================
    # Phase 3: Risk Engine Methods
    # ============================
    
    def set_risk_engine(self, risk_engine: RiskEngine):
        """Set the risk engine for this GPU"""
        with self._lock:
            self.risk_engine = risk_engine
            self.risk_engine.initialize_gpu(self.id, self.memory_capacity, self.reliability_history)
    
    def get_risk_state(self):
        """Get current risk state from risk engine"""
        with self._lock:
            if self.risk_engine is None:
                return None
            return self.risk_engine.get_risk_state(self.id)
    
    def update_risk(self):
        """Update risk score based on current GPU state"""
        with self._lock:
            if self.risk_engine is None:
                return None
            
            # Calculate risk components
            components = self.risk_engine.calculate_risk_score(self)
            
            # Compute complete risk score with GPU's reliability history
            risk_state = self.risk_engine.compute_risk_score(
                self.id, self, components, self.reliability_history
            )
            
            # Update GPU's reliability history from risk state
            self.reliability_history = risk_state.reliability_history
            
            return risk_state
    
    def track_task_execution(self, task_id: str, expected_time: float):
        """Track expected execution time for a task"""
        with self._lock:
            self.expected_execution_times[task_id] = expected_time
    
    def record_task_completion(self, task_id: str, actual_time: float, success: bool):
        """Record task completion and update reliability history"""
        with self._lock:
            if self.risk_engine is None:
                return
            
            expected_time = self.expected_execution_times.get(task_id, actual_time)
            
            # Update risk engine with task completion
            self.risk_engine.update_task_completion(
                self.id, expected_time, actual_time, success
            )
            
            # Clean up tracking
            if task_id in self.expected_execution_times:
                del self.expected_execution_times[task_id]
    
    def record_crash_event(self):
        """Record a crash event"""
        with self._lock:
            if self.risk_engine:
                self.risk_engine.record_crash(self.id)
    
    def record_recovery_event(self):
        """Record a recovery event"""
        with self._lock:
            if self.risk_engine:
                self.risk_engine.record_recovery(self.id)
    
    def record_preemption_event(self):
        """Record a preemption event"""
        with self._lock:
            if self.risk_engine:
                self.risk_engine.record_preemption(self.id)
    
    def record_anomaly_event(self):
        """Record an anomaly detection event"""
        with self._lock:
            if self.risk_engine:
                self.risk_engine.record_anomaly(self.id)
    
    def check_quarantine_recovery(self) -> bool:
        """Check if GPU can recover from quarantine"""
        with self._lock:
            if self.risk_engine is None:
                return False
            return self.risk_engine.check_quarantine_recovery(self.id)
    
    def get_effective_capacity(self) -> float:
        """Get reliability-adjusted effective capacity"""
        with self._lock:
            risk_state = self.get_risk_state()
            if risk_state is None:
                return self.memory_capacity - self.current_memory
            return risk_state.effective_capacity
    
    def is_quarantined(self) -> bool:
        """Check if GPU is quarantined"""
        with self._lock:
            risk_state = self.get_risk_state()
            if risk_state is None:
                return False
            return risk_state.is_quarantined
    
    # ============================
    # Phase 4: Fault Injection Methods
    # ============================
    
    def set_fault_injector(self, fault_injector: FaultInjector):
        """Set the fault injector for this GPU"""
        with self._lock:
            self.fault_injector = fault_injector
    
    def inject_fault(self, task_id: Optional[str] = None, severity=None) -> AnomalyResult:
        """
        Inject a fault/anomaly on this GPU
        
        Args:
            task_id: Task identifier (if applicable)
            severity: Specific severity to use (None for random)
            
        Returns:
            AnomalyResult with details of what occurred
        """
        with self._lock:
            if self.fault_injector is None:
                return AnomalyResult(occurred=False)
            
            result = self.fault_injector.inject_anomaly(self.id, task_id, severity)
            
            if result.occurred:
                # Record the event locally
                self.fault_events.append(result)
                
                # Update GPU's reliability history
                self.reliability_history.anomalies_detected += 1
                self.reliability_history.total_tasks += 1
                
                # Update risk engine
                if self.risk_engine:
                    self.risk_engine.record_anomaly(self.id)
                    
                    # If task failed, record failure in GPU's reliability history
                    if result.task_failed:
                        self.reliability_history.failed_tasks += 1
                    
                    # If GPU crashed, record crash
                    if result.gpu_crashed:
                        self.risk_engine.record_crash(self.id)
                        self._enter_crash()
                
                # Apply effects
                if result.gpu_crashed:
                    self.crashed = True
                    self.crash_cooldown_timer = 5
            
            return result
    
    def get_fault_events(self, limit: int = 50) -> list:
        """Get recent fault events for this GPU"""
        with self._lock:
            return list(self.fault_events[-limit:])
    
    def clear_fault_events(self):
        """Clear fault event history for this GPU"""
        with self._lock:
            self.fault_events = []

