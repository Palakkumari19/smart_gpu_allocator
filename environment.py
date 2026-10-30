import copy
import math
import random
import re
import threading
import uuid
from datetime import datetime, timezone
import numpy as np
from agent import RLAgent
from gpu import GPU
from task import Task
import task
from workload.task_generator import TaskGenerator, VideoTask
from workload.traffic_patterns import TrafficManager, TrafficPattern
from marl_agent import MARLManager
from schedulers.traditional import BaselineComparison, FCFSScheduler, RoundRobinScheduler, LeastLoadedScheduler
from risk_engine import RiskEngine
from fault_injection import FaultInjector, ExpectedFailureCostCalculator
from dynamic_queue import DynamicTaskQueue, TaskStatus

class GPUEnvironment:
    def _serialize_task(self, task, gpu_id=None):
        if task is None:
            return None
        return {
            "id": getattr(task, "id", f"task-{gpu_id}-{self.time_step}"),
            "gpu_id": gpu_id,
            "task_type": getattr(task, "task_type", "Unknown"),
            "profile_name": getattr(task, "profile_name", "Unknown"),
            "resolution": getattr(task, "resolution", "Unknown"),
            "fps": getattr(task, "fps", 0),
            "duration": getattr(task, "duration", 0),
            "frames": getattr(task, "frames", 0),
            "codec": getattr(task, "codec", "Unknown"),
            "bitrate": getattr(task, "bitrate", "Unknown"),
            "workload_severity": getattr(task, "workload_severity", "Unknown"),
            "memory_required": getattr(task, "memory_required", 0),
            "execution_time": getattr(task, "execution_time", 0),
            "priority": getattr(task, "priority", "Medium"),
            "preempted": getattr(task, "preempted", False),
            "preemption_count": getattr(task, "preemption_count", 0),
            "status": self.task_queue.get_task_status(getattr(task, "id", "")).value
                if getattr(task, "id", "") in self.task_queue.task_registry else None,
        }

    def _record_event(self, message, event_type="info"):
        lock = getattr(self, "_stats_lock", None)
        if lock:
            with lock:
                self.event_log.append({
                    "id": f"{self.time_step}-{len(self.event_log)}",
                    "time": f"{self.time_step:03d}",
                    "message": message,
                    "type": event_type,
                })
                if len(self.event_log) > 50:
                    self.event_log = self.event_log[-50:]
        else:
            self.event_log.append({
                "id": f"{self.time_step}-{len(self.event_log)}",
                "time": f"{self.time_step:03d}",
                "message": message,
                "type": event_type,
            })
            if len(self.event_log) > 50:
                self.event_log = self.event_log[-50:]
    
    def _rand(self):
        """Random number generator for fault injection"""
        return random.random()

    def _append_history(self):
        metrics = self.get_metrics()
        lock = getattr(self, "_stats_lock", None)
        if lock:
            with lock:
                self.history.append({
                    "time_step": self.time_step,
                    "completed": metrics["completed"],
                    "latency": metrics["latency"],
                    "reward": metrics.get("marl_global_reward", 0),
                    "avg_memory_usage": metrics["avg_memory_usage"],
                    "avg_temperature": metrics["avg_temperature"],
                    "total_queued_tasks": metrics["total_queued_tasks"],
                    "total_running_tasks": metrics["total_running_tasks"],
                })
                if len(self.history) > 120:
                    self.history = self.history[-120:]
        else:
            self.history.append({
                "time_step": self.time_step,
                "completed": metrics["completed"],
                "latency": metrics["latency"],
                "reward": metrics.get("marl_global_reward", 0),
                "avg_memory_usage": metrics["avg_memory_usage"],
                "avg_temperature": metrics["avg_temperature"],
                "total_queued_tasks": metrics["total_queued_tasks"],
                "total_running_tasks": metrics["total_running_tasks"],
            })
            if len(self.history) > 120:
                self.history = self.history[-120:]

    def get_dashboard_state(self):
        metrics = self.get_metrics()

        with self._stats_lock:
            recent_completed = [self._serialize_task(t, t.get("gpu_id") if isinstance(t, dict) else None) if not isinstance(t, dict) else t for t in self.recent_completed_tasks[-12:]]
            events = list(self.event_log[-20:])
            history_recent = list(self.history[-20:])
            history_all = list(self.history)

        task_queue = []
        running_tasks = []
        represented_task_ids = set()
        with self.task_queue._lock:
            for gpu in self.gpus:
                with gpu._lock:
                    for task in list(gpu.task_queue):
                        task_queue.append(self._serialize_task(task, gpu.id))
                        represented_task_ids.add(task.id)
                    for task in list(gpu.running_tasks):
                        running_tasks.append(self._serialize_task(task, gpu.id))
                        represented_task_ids.add(task.id)

            queued_ids = self.task_queue.get_tasks_by_status(TaskStatus.QUEUED)
            for task_id in queued_ids:
                if task_id not in represented_task_ids:
                    t_entry = self.task_queue.get_task(task_id)
                    if t_entry:
                        task_queue.append(self._serialize_task(t_entry[0]))
            all_tasks_dict = {
                task_id: metadata.to_dict()
                for task_id, (_, metadata) in self.task_queue.get_all_tasks().items()
            }
            task_counts_dict = self.task_queue.get_task_counts()
            task_events_list = self.task_queue.get_event_log(50)
            task_to_gpu_dict = self.task_queue.task_to_gpu.copy()

        # Phase 3: Include risk state in GPU data
        gpu_data = []
        for gpu in self.gpus:
            with gpu._lock:
                gpu_info = {
                    "id": gpu.id,
                    "memory": gpu.get_memory_usage_percentage(),
                    "temperature": gpu.temperature,
                    "utilization": gpu.utilization,
                    "queue_length": gpu.get_queue_length(),
                    "running_tasks": gpu.get_running_tasks_count(),
                    "completed_tasks": gpu.completed_tasks,
                    "fragmentation": gpu.get_fragmentation_percentage(),
                    "preemptions": gpu.total_preemptions,
                    "crashed": gpu.crashed,
                    "cooldown": gpu.get_cooldown_remaining()
                }
                
                # Add risk information
                risk_state = gpu.get_risk_state()
                if risk_state:
                    gpu_info["risk_score"] = risk_state.risk_score
                    gpu_info["risk_level"] = risk_state.risk_level.value
                    gpu_info["effective_capacity"] = risk_state.effective_capacity
                    gpu_info["raw_capacity"] = risk_state.raw_capacity
                    gpu_info["is_quarantined"] = risk_state.is_quarantined
                    gpu_info["quarantine_reason"] = risk_state.quarantine_reason
                    
                    # Add reliability history
                    history = risk_state.reliability_history
                    gpu_info["reliability_history"] = {
                        "total_tasks": history.total_tasks,
                        "completed_tasks": history.completed_tasks,
                        "failed_tasks": history.failed_tasks,
                        "crash_events": history.crash_events,
                        "recovery_events": history.recovery_events,
                        "preemptions": history.preemptions,
                        "anomalies_detected": history.anomalies_detected,
                        "abnormal_executions": history.abnormal_executions,
                        "failure_rate": history.failure_rate
                    }
                    
                    # Add risk components
                    gpu_info["risk_components"] = {
                        "temperature_risk": risk_state.components.temperature_risk,
                        "memory_pressure": risk_state.components.memory_pressure,
                        "fragmentation_risk": risk_state.components.fragmentation_risk,
                        "failure_history": risk_state.components.failure_history,
                        "crash_history": risk_state.components.crash_history,
                        "anomaly_score": risk_state.components.anomaly_score,
                        "performance_degradation": risk_state.components.performance_degradation,
                        "utilization_risk": risk_state.components.utilization_risk
                    }
                else:
                    # Default values if risk engine not initialized
                    gpu_info["risk_score"] = 0.0
                    gpu_info["risk_level"] = "HEALTHY"
                    gpu_info["effective_capacity"] = gpu.memory_capacity - gpu.current_memory
                    gpu_info["raw_capacity"] = gpu.memory_capacity - gpu.current_memory
                    gpu_info["is_quarantined"] = False
                    gpu_info["quarantine_reason"] = None
                    gpu_info["risk_components"] = {}
                    gpu_info["reliability_history"] = {}

                gpu_info["efc"] = round(float(self.marl_manager.calculate_expected_failure_cost(gpu)), 4)
                gpu_info["is_eligible"] = not (
                    gpu.crashed or 
                    gpu.is_in_cooldown() or 
                    (risk_state and risk_state.is_quarantined) or 
                    (hasattr(gpu, "is_quarantined") and gpu.is_quarantined())
                )
                
                gpu_data.append(gpu_info)

        # Phase 3: Add global risk metrics
        global_risk_metrics = self.risk_engine.get_global_risk_metrics()

        return {
            "time_step": self.time_step,
            "scheduler": self.scheduler,
            "metrics": metrics,
            "gpus": gpu_data,
            "task_queue": task_queue,
            "running_tasks": running_tasks,
            "recent_completed_tasks": recent_completed,
            "events": events,
            "task_registry": all_tasks_dict,
            "task_counts": task_counts_dict,
            "task_events": task_events_list,
            "task_to_gpu": task_to_gpu_dict,
            "chart_data": {
                "reward": [
                    {"time_step": entry["time_step"], "value": entry["reward"]}
                    for entry in history_all
                ],
                "memory": [
                    {"time_step": entry["time_step"], "value": entry["avg_memory_usage"]}
                    for entry in history_all
                ],
                "temperature": [
                    {"time_step": entry["time_step"], "value": entry["avg_temperature"]}
                    for entry in history_all
                ],
                "queue": [
                    {"time_step": entry["time_step"], "value": entry["total_queued_tasks"]}
                    for entry in history_all
                ],
                "scheduler_comparison": [
                    {
                        "name": self.scheduler,
                        "completed": metrics["completed"],
                        "latency": metrics["latency"],
                        "utilization": metrics["avg_memory_usage"],
                    },
                    *[
                        {
                            "name": name,
                            "completed": int(stats.get("assignment_rate", 0) * 100),
                            "latency": metrics["latency"],
                            "utilization": metrics["avg_memory_usage"],
                        }
                        for name, stats in self.get_traditional_scheduler_stats().items()
                    ],
                ],
            },
            "history": history_recent,
            # Phase 3: Add risk metrics
            "risk_metrics": global_risk_metrics,
            # Bit 6: Risk-Aware MARL Decision Metadata
            "last_marl_decision": dict(getattr(self, "last_marl_decision", {}))
        }

    def __init__(self, scheduler="random"):
        self.last_marl_decision = {}
        self.marl_decision_counter = 0

        # 🔥 MULTI GPU SETUP with 8GB memory capacity as specified
        self.gpus = [
            GPU(1, 8),   # 8GB
            GPU(2, 8),   # 8GB  
            GPU(3, 8),   # 8GB
            GPU(4, 8)    # 8GB
        ]

        self.time_step = 0
        self.total_tasks = 0

        self.scheduler = scheduler
        self.rr_index = 0

        self.agent = RLAgent()
        
        # Phase 2 Step 3: Task Generator
        self.task_generator = TaskGenerator()
        self.task_queue = DynamicTaskQueue()
        self.workload_mode = "mixed"  # mixed, light_heavy, extreme_heavy

        
        # Phase 2 Step 4: Traffic Patterns
        self.traffic_manager = TrafficManager()
        self.traffic_mode = "auto"  # auto, manual, peak_off_peak
        
        # MARL System
        self.marl_manager = MARLManager(num_gpus=len(self.gpus))
        
        # Phase 3 Step 5: Traditional Schedulers
        self.baseline_comparison = BaselineComparison()
        self.traditional_schedulers = {
            'fcfs': FCFSScheduler(),
            'round_robin': RoundRobinScheduler(),
            'least_loaded': LeastLoadedScheduler()
        }

        # Phase 3: Risk Engine for Proactive Reliability-Aware Scheduling
        self.risk_engine = RiskEngine()
        
        # Initialize risk engine for all GPUs
        for gpu in self.gpus:
            gpu.set_risk_engine(self.risk_engine)
        
        # Phase 4: Fault Injection System
        self.fault_injector = FaultInjector(seed=42, enabled=True)  # Reproducible by default
        self.expected_failure_cost_calculator = ExpectedFailureCostCalculator()
        
        # Initialize fault injector for all GPUs
        for gpu in self.gpus:
            gpu.set_fault_injector(self.fault_injector)

        self._step_lock = threading.RLock()
        self.lock = self._step_lock
        self._stats_lock = threading.RLock()

        self.event_log = []
        self.history = []
        self.recent_completed_tasks = []
        self.workload_trace = None
        self.learning_enabled = True
        self.successful_assignments = 0
        self.failed_assignments = 0
        self.total_assigned_efc = 0.0
        self.total_efc_assignments = 0
        self._record_event("Simulation initialized", "info")

    def attach_workload_trace(self, steps_of_tasks):
        """Replay a fixed per-step task arrival sequence."""
        self.workload_trace = [list(step_tasks) for step_tasks in steps_of_tasks]

    def enable_deterministic_physics(self):
        for gpu in self.gpus:
            gpu.enable_deterministic_physics()

    def set_eval_mode(self):
        """Freeze exploration and Q-learning updates for evaluation."""
        self.learning_enabled = False
        self.agent.epsilon = 0.0
        for agent in self.marl_manager.agents:
            agent.epsilon = 0.0

    def _try_assign(self, gpu, task):
        """
        Atomically assign a task to a GPU.
        Guarantees strict lock ordering: self.task_queue._lock -> gpu._lock.
        Rolls back partially updated state on any exception.
        """
        self._register_task(task)
        with self.task_queue._lock:
            metadata = self.task_queue.get_task_metadata(task.id)
            if not metadata or metadata.status not in (TaskStatus.QUEUED, TaskStatus.RETRY):
                return False
            assigned = self.task_queue.get_assigned_gpu(task.id)
            if assigned is not None and assigned != gpu.id:
                return False
            
            with gpu._lock:
                if gpu.is_quarantined():
                    with self._stats_lock:
                        self.failed_assignments += 1
                    self._record_event(f"Task {getattr(task, 'task_type', 'Task')} rejected - GPU {gpu.id} is quarantined", "warning")
                    return False
                
                # Enforce physical capacity invariant for environment scheduling
                if gpu.current_memory + task.memory_required > gpu.memory_capacity:
                    with self._stats_lock:
                        self.failed_assignments += 1
                    return False
                
                success = gpu.assign_task(task)
                if success:
                    try:
                        self.task_queue.schedule_task(task.id, gpu.id)
                        with self._stats_lock:
                            self.successful_assignments += 1
                            assignment_efc = self.marl_manager.calculate_expected_failure_cost(gpu, task)
                            self.total_assigned_efc += assignment_efc
                            self.total_efc_assignments += 1
                        # Phase 3: Track expected execution time for risk calculation
                        gpu.track_task_execution(task.id, task.execution_time)
                        return True
                    except Exception:
                        if task in gpu.task_queue:
                            gpu.task_queue.remove(task)
                        if task in gpu.running_tasks:
                            gpu.running_tasks.remove(task)
                        gpu.current_memory = max(0.0, round(gpu.current_memory - task.memory_required, 4))
                        # Comprehensive rollback of queue mappings, task metadata, and counters
                        self.task_queue.task_to_gpu.pop(task.id, None)
                        if gpu.id in self.task_queue.gpu_to_tasks and task.id in self.task_queue.gpu_to_tasks[gpu.id]:
                            self.task_queue.gpu_to_tasks[gpu.id].remove(task.id)
                        if task.id in self.task_queue.task_registry:
                            meta = self.task_queue.task_registry[task.id][1]
                            meta.status = TaskStatus.QUEUED
                            meta.assigned_gpu = None
                        if task.id not in self.task_queue.queued_tasks:
                            self.task_queue.queued_tasks.append(task.id)
                        with self._stats_lock:
                            self.failed_assignments += 1
                        raise
                else:
                    with self._stats_lock:
                        self.failed_assignments += 1
                    if gpu.is_quarantined():
                        self._record_event(f"Task {getattr(task, 'task_type', 'Task')} rejected - GPU {gpu.id} is quarantined", "warning")
                    return False

    def add_task(self, task) -> str:
        """Register and enqueue a task into the environment (thread-safe)."""
        with self._step_lock:
            tid = self._register_task(task)
            self.total_tasks = self.task_queue.get_task_counts()["total"]
            return tid

    def _register_task(self, task):
        """Register one arriving task before the existing scheduler sees it."""
        if task.id not in self.task_queue.task_registry:
            explicit_id = task.id if re.fullmatch(r"video_\d{4}", str(task.id)) else None
            self.task_queue.add_task(task, task_id=explicit_id)
            self.total_tasks = self.task_queue.get_task_counts()["total"]
        return task.id

    def _schedule_task(self, task):
        """Dispatch through the configured existing scheduler and report success."""
        before = set(self.task_queue.task_to_gpu)
        if self.scheduler == "random":
            self.random_scheduler(task)
        elif self.scheduler == "round_robin":
            self.round_robin_scheduler(task)
        elif self.scheduler == "fcfs":
            self.fcfs_scheduler(task)
        elif self.scheduler == "least_loaded":
            self.least_loaded_scheduler(task)
        elif self.scheduler == "rl":
            self.rl_scheduler(task)
        elif self.scheduler == "marl":
            self.marl_scheduler(task)
        elif self.scheduler == "traditional_fcfs":
            self.traditional_fcfs_scheduler(task)
        elif self.scheduler == "traditional_round_robin":
            self.traditional_round_robin_scheduler(task)
        elif self.scheduler == "traditional_least_loaded":
            self.traditional_least_loaded_scheduler(task)
        return task.id in self.task_queue.task_to_gpu and task.id not in before

    def generate_task(self, task_type=None):
        """Generate task using the new task generator"""
        return self.task_generator.generate_task(task_type)
    
    def generate_workload_burst(self, intensity="medium"):
        """Generate burst workload"""
        return self.task_generator.generate_burst_workload(intensity)
    
    def set_workload_mode(self, mode):
        """Set workload generation mode"""
        if mode in ["mixed", "light_heavy", "extreme_heavy"]:
            self.workload_mode = mode
            from workload.task_generator import VideoProfile
            # Adjust task generator weights
            if mode == "light_heavy":
                self.task_generator.profile_weights = {
                    VideoProfile.LIGHT: 0.7, VideoProfile.MEDIUM: 0.2, VideoProfile.HEAVY: 0.1, VideoProfile.EXTREME: 0.0
                }
            elif mode == "extreme_heavy":
                self.task_generator.profile_weights = {
                    VideoProfile.LIGHT: 0.1, VideoProfile.MEDIUM: 0.2, VideoProfile.HEAVY: 0.4, VideoProfile.EXTREME: 0.3
                }
            else:  # mixed
                self.task_generator.profile_weights = {
                    VideoProfile.LIGHT: 0.40, VideoProfile.MEDIUM: 0.35, VideoProfile.HEAVY: 0.20, VideoProfile.EXTREME: 0.05
                }
    
    def get_workload_statistics(self):
        """Get workload generation statistics"""
        return self.task_generator.get_workload_statistics()
    
    def set_traffic_pattern(self, pattern, duration=10):
        """Set traffic pattern manually"""
        self.traffic_manager.set_pattern(pattern, duration)
        self.traffic_mode = "manual"
    
    def set_traffic_mode(self, mode):
        """Set traffic management mode"""
        if mode in ["auto", "manual", "peak_off_peak"]:
            self.traffic_mode = mode
            
    def get_traffic_statistics(self):
        """Get traffic pattern statistics"""
        return self.traffic_manager.get_traffic_statistics()
    
    # ============================
    # Phase 4: Fault Injection Methods
    # ============================
    
    def set_fault_injection_enabled(self, enabled: bool):
        """Enable or disable fault injection"""
        self.fault_injector.set_enabled(enabled)
    
    def set_fault_injection_probability(self, severity, probability: float):
        """Set fault injection probability for a severity level"""
        from fault_injection import SeverityLevel
        if isinstance(severity, str):
            severity = SeverityLevel(severity.upper())
        self.fault_injector.set_probability(severity, probability)
    
    def get_fault_event_log(self, limit: int = 100):
        """Get fault event log"""
        return self.fault_injector.get_event_log(limit)
    
    def clear_fault_event_log(self):
        """Clear fault event log"""
        self.fault_injector.clear_event_log()
    
    def calculate_expected_failure_cost(self, gpu, task) -> float:
        """
        Calculate expected failure cost for a task on a GPU
        
        Args:
            gpu: GPU object
            task: VideoTask object
            
        Returns:
            Expected failure cost
        """
        risk_state = gpu.get_risk_state()
        if risk_state is None:
            return 0.0
        
        gpu_risk_score = risk_state.risk_score
        return self.expected_failure_cost_calculator.calculate_expected_failure_cost(
            gpu_risk_score, task
        )
    
    def get_fault_statistics(self):
        """Get fault injection statistics"""
        return self.fault_injector.get_statistics()
    
    def simulate_peak_off_peak(self, steps=24):
        """Simulate peak/off-peak traffic patterns"""
        self.traffic_mode = "peak_off_peak"
        return self.traffic_manager.simulate_day_cycle(steps)
    
    def get_current_traffic_load(self):
        """Get current traffic load information"""
        return {
            "pattern": self.traffic_manager.current_pattern.value,
            "hour": self.traffic_manager.current_hour,
            "requests_this_step": self.traffic_manager.get_current_requests(),
            "mode": self.traffic_mode
        }

    # -----------------------
    # SCHEDULERS
    # -----------------------

    def random_scheduler(self, task):
        available_gpus = [g for g in self.gpus if not g.crashed and not g.is_quarantined() and not g.is_in_cooldown() and g.current_memory + task.memory_required <= g.memory_capacity]
        if available_gpus:
            return self._try_assign(random.choice(available_gpus), task)
        else:
            with self._stats_lock:
                self.failed_assignments += 1
            return False

    def round_robin_scheduler(self, task):
        for i in range(len(self.gpus)):
            idx = (self.rr_index + i) % len(self.gpus)
            gpu = self.gpus[idx]
            if not gpu.crashed and not gpu.is_quarantined() and not gpu.is_in_cooldown() and gpu.current_memory + task.memory_required <= gpu.memory_capacity:
                self.rr_index = (idx + 1) % len(self.gpus)
                if self._try_assign(gpu, task):
                    return True
        self.rr_index = (self.rr_index + 1) % len(self.gpus)
        with self._stats_lock:
            self.failed_assignments += 1
        return False

    def fcfs_scheduler(self, task):
        for gpu in self.gpus:
            if not gpu.crashed and not gpu.is_quarantined() and not gpu.is_in_cooldown() and gpu.current_memory + task.memory_required <= gpu.memory_capacity:
                if self._try_assign(gpu, task):
                    return True
        with self._stats_lock:
            self.failed_assignments += 1
        return False

    def least_loaded_scheduler(self, task):
        # Consider both queue and running tasks for load balancing among GPUs with capacity
        available_gpus = [g for g in self.gpus if not g.crashed and not g.is_quarantined() and not g.is_in_cooldown() and g.current_memory + task.memory_required <= g.memory_capacity]
        if available_gpus:
            gpu = min(available_gpus, key=lambda g: g.get_total_load())
            return self._try_assign(gpu, task)
        else:
            with self._stats_lock:
                self.failed_assignments += 1
            return False

    def rl_scheduler(self, task):
        state = self.agent.get_state(self.gpus)
        action = self.agent.choose_action(state, len(self.gpus))
        selected_gpu = self.gpus[action]
        
        # Check if assignment is possible
        if selected_gpu.crashed:
            # Try next available GPU
            for gpu in self.gpus:
                if not gpu.crashed:
                    selected_gpu = gpu
                    break
            else:
                self.failed_assignments += 1
                return  # No available GPUs
        
        self._try_assign(selected_gpu, task)

        if not self.learning_enabled:
            return

        # 🔥 ENHANCED REWARD SYSTEM
        reward = 0

        # prefer less loaded GPU (considering both queue and running tasks)
        reward -= selected_gpu.get_total_load()

        # reward healthy GPU
        if not selected_gpu.crashed:
            reward += 5

        # penalize crash strongly
        if selected_gpu.crashed:
            reward -= 40
            
        # reward efficient memory usage
        memory_usage = selected_gpu.get_memory_usage_percentage()
        if memory_usage < 80:
            reward += 2
        elif memory_usage > 95:
            reward -= 10

        # encourage balance across all GPUs
        loads = [g.get_total_load() for g in self.gpus]
        reward -= (max(loads) - min(loads)) * 0.5
        
        # penalize high temperature
        if selected_gpu.temperature > 70:
            reward -= (selected_gpu.temperature - 70) * 0.2

        next_state = self.agent.get_state(self.gpus)
        self.agent.update(state, action, reward, next_state, len(self.gpus))
    
    def marl_scheduler(self, task):
        """
        Risk-Aware Cooperative MARL scheduler:
        - Evaluates candidate GPUs using cooperative bidding with risk and expected failure cost (EFC).
        - Enforces strict admission (crashed, quarantined, or capacity-violating GPUs are ineligible).
        - Executes atomic assignment via _try_assign.
        - Calculates risk-aware reward incorporating reliability, risk score, and recovery penalties.
        - Updates agent Q-table under concurrency locks and records decision metadata.
        """
        # Ensure candidate GPU risk scores and components reflect current physical state
        for g in self.gpus:
            g.update_risk()

        cluster_state = self.marl_manager.get_cluster_state(self.gpus)
        cluster_state['current_task_memory'] = task.memory_required
        cluster_state['current_task_time'] = task.execution_time

        # Select best feasible GPU using risk-aware MARL bidding with rich candidate breakdown
        selected_gpu_id, candidate_breakdowns = self.marl_manager.select_gpu_for_task(
            task, self.gpus, cluster_state, return_infeasible_fallback=False, return_breakdown=True
        )

        self.marl_decision_counter += 1
        now_iso = datetime.now(timezone.utc).isoformat()
        dec_id = f"dec-{uuid.uuid4().hex[:8]}"

        # If all GPUs are infeasible (e.g. OOM, quarantined, crashed), do not force invalid assignment
        if selected_gpu_id is None:
            self.failed_assignments += 1
            self.last_marl_decision = {
                "decision_id": dec_id,
                "timestamp": now_iso,
                "task_id": task.id,
                "task_memory": task.memory_required,
                "task_time": task.execution_time,
                "selected_gpu": None,
                "selected_gpu_index": None,
                "selected_gpu_id": None,
                "status": "rejected_no_feasible_gpu",
                "reason": "All GPUs are currently infeasible due to capacity, quarantine, or crash",
                "candidates": candidate_breakdowns,
                "decision_score": -999.0,
                "gpu_scores": {b["label"]: b["bid_score"] for b in candidate_breakdowns},
                "risk_score": 0.0,
                "expected_failure_cost": 0.0,
                "reward": None,
                "counter": self.marl_decision_counter,
            }
            return False

        selected_gpu = self.gpus[selected_gpu_id]
        agent = self.marl_manager.agents[selected_gpu_id]

        # Calculate Expected Failure Cost (EFC) and pre-action state
        efc = self.marl_manager.calculate_expected_failure_cost(selected_gpu, task)
        cluster_state['current_efc'] = efc
        state = agent.get_state(selected_gpu, cluster_state)

        # Assign task via atomic allocation with rollback support
        success = self._try_assign(selected_gpu, task)

        # Extract bid scores map
        bid_scores = {b["label"]: b["bid_score"] for b in candidate_breakdowns}

        risk_state = selected_gpu.get_risk_state() if hasattr(selected_gpu, 'get_risk_state') else None
        risk_score = risk_state.risk_score if risk_state else 0.0

        if not success:
            # If primary assignment failed, penalize and attempt fallback if allowed
            if self.learning_enabled:
                penalty_reward = self.marl_manager.calculate_reward(selected_gpu, task, False, expected_failure_cost=efc)
                next_cs = self.marl_manager.get_cluster_state(self.gpus)
                next_cs['current_task_memory'] = task.memory_required
                next_cs['current_task_time'] = task.execution_time
                next_cs['current_task_impact'] = cluster_state.get('current_task_impact', 0.5)
                next_cs['current_efc'] = efc
                next_s = agent.get_state(selected_gpu, next_cs)
                agent.update(state, 0, penalty_reward, next_s)
            return self.least_loaded_scheduler(task)

        # Calculate risk-aware reward
        reward = self.marl_manager.calculate_reward(selected_gpu, task, True, expected_failure_cost=efc)

        self.last_marl_decision = {
            "decision_id": dec_id,
            "timestamp": now_iso,
            "task_id": task.id,
            "task_memory": task.memory_required,
            "task_time": task.execution_time,
            "selected_gpu": f"GPU {selected_gpu.id}",
            "selected_gpu_index": selected_gpu_id,
            "selected_gpu_id": selected_gpu.id,
            "decision_score": bid_scores.get(f"GPU {selected_gpu.id}", 0.0),
            "gpu_scores": bid_scores,
            "candidates": candidate_breakdowns,
            "risk_score": round(float(risk_score), 4),
            "expected_failure_cost": round(float(efc), 4),
            "reward": round(float(reward), 4),
            "status": "assigned",
            "reason": f"Assigned to GPU {selected_gpu.id} via risk-aware MARL cooperative bidding",
            "counter": self.marl_decision_counter,
        }

        if not self.learning_enabled:
            return True

        # Post-action next state and Q-learning update (with consistent task context)
        next_cluster_state = self.marl_manager.get_cluster_state(self.gpus)
        next_cluster_state['current_task_memory'] = task.memory_required
        next_cluster_state['current_task_time'] = task.execution_time
        next_cluster_state['current_task_impact'] = cluster_state.get('current_task_impact', 0.5)
        next_cluster_state['current_efc'] = efc
        next_state = agent.get_state(selected_gpu, next_cluster_state)
        agent.update(state, 0, reward, next_state)
        return True
    
    def traditional_fcfs_scheduler(self, task):
        """Traditional FCFS scheduler"""
        scheduler = self.traditional_schedulers['fcfs']
        gpu_index = scheduler.select_gpu(self.gpus, task)
        if gpu_index is not None:
            self._try_assign(self.gpus[gpu_index], task)
        else:
            self.failed_assignments += 1
    
    def traditional_round_robin_scheduler(self, task):
        """Traditional Round Robin scheduler"""
        scheduler = self.traditional_schedulers['round_robin']
        gpu_index = scheduler.select_gpu(self.gpus, task)
        if gpu_index is not None:
            self._try_assign(self.gpus[gpu_index], task)
        else:
            self.failed_assignments += 1
    
    def traditional_least_loaded_scheduler(self, task):
        """Traditional Least Loaded scheduler"""
        scheduler = self.traditional_schedulers['least_loaded']
        gpu_index = scheduler.select_gpu(self.gpus, task)
        if gpu_index is not None:
            self._try_assign(self.gpus[gpu_index], task)
        else:
            self.failed_assignments += 1
    
    def get_traditional_scheduler_stats(self):
        """Get statistics for traditional schedulers"""
        stats = {}
        for name, scheduler in self.traditional_schedulers.items():
            stats[name] = scheduler.get_stats()
        return stats
    
    def compare_traditional_schedulers(self, tasks=None):
        """Compare all traditional schedulers"""
        if tasks is None:
            # Generate test tasks
            tasks = []
            for _ in range(20):
                tasks.append(self.task_generator.generate_task())
        
        return self.baseline_comparison.compare_all(self.gpus, tasks)

    # -----------------------
    # STEP FUNCTION
    # -----------------------

    def step(self):
        """Execute one simulation step (thread-safe: guarded by self._step_lock)."""
        with self._step_lock:
            self.time_step += 1

            if self.workload_trace is not None:
                trace_index = self.time_step - 1
                if trace_index < len(self.workload_trace):
                    incoming_tasks = [copy.deepcopy(task) for task in self.workload_trace[trace_index]]
                else:
                    incoming_tasks = []
            else:
                # Phase 2 Step 4: Traffic Patterns - Get requests from traffic manager
                if self.traffic_mode == "peak_off_peak":
                    # Use peak/off-peak simulation
                    num_requests = self.traffic_manager.step()
                else:
                    # Use traffic patterns
                    num_requests = self.traffic_manager.step()
                
                # Update task generator weights based on traffic pattern
                mixed_weights = self.traffic_manager.get_mixed_workload_profile()
                from workload.task_generator import VideoProfile
                self.task_generator.profile_weights = {
                    VideoProfile.LIGHT: mixed_weights.get("Image", 0.4),
                    VideoProfile.MEDIUM: mixed_weights.get("Video", 0.35),
                    VideoProfile.HEAVY: mixed_weights.get("LLM", 0.2) / 2,
                    VideoProfile.EXTREME: mixed_weights.get("LLM", 0.2) / 2
                }
                incoming_tasks = [self.generate_task() for _ in range(num_requests)]

            for task in incoming_tasks:
                self._register_task(task)
                self.total_tasks = self.task_queue.get_task_counts()["total"]

            # Retry waiting tasks in arrival order whenever a GPU is eligible.
            for task_id in self.task_queue.get_tasks_by_status(TaskStatus.QUEUED):
                task = self.task_queue.get_task(task_id)[0]
                assigned = self._schedule_task(task)
                if assigned:
                    self._record_event(f"Task {task.id} assigned to GPU {self.task_queue.get_assigned_gpu(task.id)}", "info")
                else:
                    self.task_queue.record_waiting(task.id)
                    if task_id in {incoming.id for incoming in incoming_tasks}:
                        self._record_event(f"Task {task.id} waiting for an available GPU", "warning")

            # Process tasks on each GPU atomically following global lock hierarchy:
            # task_queue._lock -> gpu._lock (ordered)
            failed_task_ids = set()
            with self.task_queue._lock:
                for gpu in self.gpus:
                    with gpu._lock:
                        crashed_before = gpu.crashed
                        quarantined_before = gpu.is_quarantined()
                        
                        # Phase 4: Inject faults during task execution
                        if gpu.running_tasks and self.fault_injector.enabled:
                            for task in list(gpu.running_tasks):
                                if self._rand() < 0.05:
                                    fault_result = gpu.inject_fault(task.id)
                                    if fault_result.occurred:
                                        if fault_result.gpu_crashed:
                                            self._record_event(f"GPU {gpu.id} crashed due to fault injection", "error")
                                        elif fault_result.task_failed:
                                            failed_task_ids.add(task.id)
                                            self.task_queue.fail_task(task.id, retry=True)
                                            self._record_event(f"Task {task.id} failed on GPU {gpu.id} due to anomaly", "error")
                                            if task in gpu.running_tasks:
                                                gpu.running_tasks.remove(task)
                                                gpu.current_memory = max(0.0, min(gpu.memory_capacity, round(gpu.current_memory - task.memory_required, 4)))
                                                gpu.reliability_history.failed_tasks += 1
                                        else:
                                            self._record_event(f"Anomaly detected on GPU {gpu.id} for task {task.id}", "warning")
                        
                        gpu.process_tasks()
                        for task in gpu.running_tasks:
                            self.task_queue.start_task(task.id)
                        active_ids = {task.id for task in gpu.task_queue + gpu.running_tasks}
                        completed_now = []
                        for task_id, gpu_id in list(self.task_queue.task_to_gpu.items()):
                            if gpu_id != gpu.id or task_id in active_ids or task_id in failed_task_ids:
                                continue
                            task_entry = self.task_queue.get_task(task_id)
                            if task_entry and self.task_queue.get_task_status(task_id) in (TaskStatus.SCHEDULED, TaskStatus.RUNNING):
                                completed_now.append(task_entry[0])
                        if completed_now:
                            for task in completed_now:
                                self.task_queue.complete_task(task.id)
                                with self._stats_lock:
                                    self.recent_completed_tasks.append(self._serialize_task(task, gpu.id))
                                    if len(self.recent_completed_tasks) > 25:
                                        self.recent_completed_tasks = self.recent_completed_tasks[-25:]
                                self._record_event(f"GPU {gpu.id} completed {task.task_type} task", "success")
                                gpu.record_task_completion(task.id, task.execution_time, success=True)
                    
                    if gpu.crashed and not crashed_before:
                        self._record_event(f"GPU {gpu.id} entered crash cooldown", "warning")
                    
                    old_risk_score = gpu.get_risk_state().risk_score if gpu.get_risk_state() else 0.0
                    gpu.update_risk()
                    new_risk_score = gpu.get_risk_state().risk_score if gpu.get_risk_state() else 0.0
                    
                    if new_risk_score > old_risk_score + 0.1:
                        self._record_event(f"GPU {gpu.id} risk increased from {old_risk_score:.2f} to {new_risk_score:.2f}", "warning")
                    
                    risk_state = gpu.get_risk_state()
                    if risk_state and risk_state.risk_level.value == "HIGH_RISK" and old_risk_score < 0.75:
                        self._record_event(f"GPU {gpu.id} entered HIGH_RISK state (risk: {new_risk_score:.2f})", "warning")
                    
                    if gpu.check_quarantine_recovery():
                        self._record_event(f"GPU {gpu.id} recovered from quarantine", "info")
                    
                    if gpu.is_quarantined() and not quarantined_before:
                        self._record_event(f"GPU {gpu.id} quarantined - {risk_state.quarantine_reason if risk_state else 'unknown'}", "error")

            if self.learning_enabled:
                self.agent.decay_epsilon()
                self.marl_manager.decay_all_epsilon()
            self._append_history()

    # -----------------------
    # METRICS
    # -----------------------

    def get_metrics(self):
        base_metrics = {
            "latency": sum(g.total_latency for g in self.gpus),
            "completed": sum(g.completed_tasks for g in self.gpus),
            "crashes": sum(int(g.crashed) for g in self.gpus),
            "avg_memory_usage": sum(g.get_memory_usage_percentage() for g in self.gpus) / len(self.gpus),
            "avg_temperature": sum(g.temperature for g in self.gpus) / len(self.gpus),
            "total_running_tasks": sum(g.get_running_tasks_count() for g in self.gpus),
            "total_queued_tasks": sum(g.get_queue_length() for g in self.gpus),
            # Advanced metrics - Step 2
            "total_memory_fragments": sum(g.get_memory_fragments_count() for g in self.gpus),
            "avg_fragmentation": sum(g.get_fragmentation_percentage() for g in self.gpus) / len(self.gpus),
            "total_preemptions": sum(g.total_preemptions for g in self.gpus),
            "total_preempted_tasks": sum(g.get_preempted_tasks_count() for g in self.gpus),
            "gpus_in_cooldown": sum(1 for g in self.gpus if g.is_in_cooldown()),
            "total_crash_events": sum(g.total_crash_events for g in self.gpus),
            "successful_assignments": self.successful_assignments,
            "failed_assignments": self.failed_assignments,
            "total_tasks_submitted": max(self.total_tasks, self.task_queue.get_task_counts()["total"]),
            "average_efc": (self.total_assigned_efc / max(1, self.total_efc_assignments)) if self.total_efc_assignments > 0 else 0.0,
            "total_assigned_efc": self.total_assigned_efc,
        }
        base_metrics.update({
            f"tasks_{name}": value
            for name, value in self.task_queue.get_task_counts().items()
        })
        
        # Add workload statistics - Phase 2 Step 3
        workload_stats = self.get_workload_statistics()
        base_metrics.update({
            "workload_stats": workload_stats,
            "light_tasks_generated": workload_stats["task_distribution"].get("Light", 0),
            "medium_tasks_generated": workload_stats["task_distribution"].get("Medium", 0),
            "heavy_tasks_generated": workload_stats["task_distribution"].get("Heavy", 0),
            "extreme_tasks_generated": workload_stats["task_distribution"].get("Extreme", 0)
        })
        
        # Add traffic statistics - Phase 2 Step 4
        traffic_stats = self.get_traffic_statistics()
        base_metrics.update({
            "traffic_stats": traffic_stats,
            "current_traffic_pattern": traffic_stats["current_pattern"],
            "current_hour": traffic_stats["current_hour"],
            "total_traffic_requests": traffic_stats["total_requests"],
            "traffic_pattern_switches": traffic_stats["pattern_switches"],
            "avg_requests_per_step": traffic_stats["avg_requests_per_step"]
        })
        
        # Add MARL statistics
        marl_stats = self.marl_manager.get_global_stats()
        base_metrics.update({
            "marl_stats": marl_stats,
            "marl_global_reward": marl_stats["total_reward"],
            "marl_success_rate": marl_stats["global_success_rate"],
            "marl_avg_epsilon": marl_stats["avg_epsilon"],
            "marl_total_q_table_size": marl_stats["total_q_table_size"]
        })
        
        # Add traditional scheduler statistics
        traditional_stats = self.get_traditional_scheduler_stats()
        base_metrics.update({
            "traditional_stats": traditional_stats,
            "fcfs_assignment_rate": traditional_stats.get("fcfs", {}).get("assignment_rate", 0),
            "round_robin_assignment_rate": traditional_stats.get("round_robin", {}).get("assignment_rate", 0),
            "least_loaded_assignment_rate": traditional_stats.get("least_loaded", {}).get("assignment_rate", 0)
        })
        
        # Phase 3: Add risk metrics
        risk_metrics = self.risk_engine.get_global_risk_metrics()
        base_metrics.update({
            "risk_metrics": risk_metrics,
            "average_gpu_risk": risk_metrics["average_risk"],
            "maximum_gpu_risk": risk_metrics["maximum_risk"],
            "risk_variance": risk_metrics["risk_variance"],
            "total_anomalies": risk_metrics["total_anomalies"],
            "total_crashes": risk_metrics["total_crashes"],
            "quarantined_gpus": risk_metrics["quarantined_gpus"]
        })
        
        return base_metrics
