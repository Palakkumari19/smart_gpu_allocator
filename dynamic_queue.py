"""
Bit 3: Dynamic Video Task Queue and Allocation Infrastructure

This module provides:
- Dynamic task queue with unique IDs
- Task lifecycle management
- Task-to-GPU mapping
- Task lifecycle events
- Dynamic task arrival simulation
"""

import threading
import time
from enum import Enum
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field


class TaskStatus(Enum):
    """Task lifecycle states"""
    ARRIVING = "ARRIVING"
    QUEUED = "QUEUED"
    SCHEDULED = "SCHEDULED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    WAITING = "WAITING"
    RETRY = "RETRY"


class TaskEventType(Enum):
    """Task lifecycle event types"""
    TASK_ARRIVED = "TASK_ARRIVED"
    TASK_QUEUED = "TASK_QUEUED"
    TASK_SCHEDULED = "TASK_SCHEDULED"
    TASK_STARTED = "TASK_STARTED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    TASK_RETRY = "TASK_RETRY"
    TASK_CANCELLED = "TASK_CANCELLED"
    TASK_WAITING = "TASK_WAITING"


@dataclass
class TaskEvent:
    """Structured task lifecycle event"""
    timestamp: float
    task_id: str
    gpu_id: Optional[int]
    event_type: TaskEventType
    event_data: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "timestamp": self.timestamp,
            "task_id": self.task_id,
            "gpu_id": self.gpu_id,
            "event_type": self.event_type.value,
            "event_data": self.event_data
        }


@dataclass
class TaskMetadata:
    """Extended metadata for dynamic task tracking"""
    task_id: str
    source: str  # "REAL_VIDEO" or "SIMULATED"
    arrival_time: float
    queue_time: Optional[float] = None
    scheduling_time: Optional[float] = None
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    assigned_gpu: Optional[int] = None
    status: TaskStatus = TaskStatus.ARRIVING
    retry_count: int = 0
    waiting_time: float = 0.0
    original_filename: Optional[str] = None
    execution_time: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "task_id": self.task_id,
            "source": self.source,
            "arrival_time": self.arrival_time,
            "queue_time": self.queue_time,
            "scheduling_time": self.scheduling_time,
            "start_time": self.start_time,
            "completion_time": self.completion_time,
            "assigned_gpu": self.assigned_gpu,
            "status": self.status.value,
            "retry_count": self.retry_count,
            "waiting_time": self.waiting_time,
            "original_filename": self.original_filename,
            "execution_time": self.execution_time,
            "queue_delay": self.queue_delay,
            "execution_duration": self.execution_duration,
            "total_task_latency": self.total_task_latency
        }

    @property
    def queue_delay(self) -> Optional[float]:
        if self.queue_time is None or self.scheduling_time is None:
            return None
        return max(0.0, self.scheduling_time - self.queue_time)

    @property
    def execution_duration(self) -> Optional[float]:
        if self.start_time is None or self.completion_time is None:
            return None
        return max(0.0, self.completion_time - self.start_time)

    @property
    def total_task_latency(self) -> Optional[float]:
        end_time = self.completion_time or time.time()
        return max(0.0, end_time - self.arrival_time)


class DynamicTaskQueue:
    """
    Dynamic task queue with lifecycle management and event logging.
    Thread-safe implementation with fine-grained RLock.
    """
    
    def __init__(self):
        self._lock = threading.RLock()
        self.lock = self._lock  # Public alias for lock coordination
        
        # Task registry: task_id -> (VideoTask, TaskMetadata)
        self.task_registry: Dict[str, tuple] = {}
        
        # Task-to-GPU mapping: task_id -> gpu_id
        self.task_to_gpu: Dict[str, int] = {}
        
        # GPU-to-tasks mapping: gpu_id -> list of task_ids
        self.gpu_to_tasks: Dict[int, List[str]] = {1: [], 2: [], 3: [], 4: []}
        
        # Event log
        self.event_log: List[TaskEvent] = []
        
        # Task ID counter
        self.task_counter = 0
        
        # Status-based task lists
        self.queued_tasks: List[str] = []
        self.running_tasks: List[str] = []
        self.completed_tasks: List[str] = []
        self.failed_tasks: List[str] = []
        
    def _next_task_id(self) -> str:
        """Generate unique task ID (thread-safe)."""
        with self._lock:
            self.task_counter += 1
            return f"video_{self.task_counter:04d}"
    
    def _log_event(self, task_id: str, event_type: TaskEventType, 
                   gpu_id: Optional[int] = None, event_data: Dict[str, Any] = None):
        """Log a task lifecycle event (thread-safe)."""
        with self._lock:
            if event_data is None:
                event_data = {}
            event = TaskEvent(
                timestamp=time.time(),
                task_id=task_id,
                gpu_id=gpu_id,
                event_type=event_type,
                event_data=event_data
            )
            self.event_log.append(event)
            
            # Keep event log manageable
            if len(self.event_log) > 1000:
                self.event_log = self.event_log[-1000:]
    
    def add_task(self, video_task, source: str = "SIMULATED",
                 original_filename: Optional[str] = None,
                 task_id: Optional[str] = None) -> str:
        """
        Add a new task to the queue atomically.
        Thread-safe: guarded by self._lock.
        """
        with self._lock:
            task_id = task_id or self._next_task_id()
            if task_id in self.task_registry:
                raise ValueError(f"Duplicate task ID: {task_id}")
            if task_id.startswith("video_"):
                try:
                    self.task_counter = max(self.task_counter, int(task_id.split("_")[-1]))
                except ValueError:
                    pass
            
            # Update video task ID
            video_task.id = task_id
            video_task.created_time = time.time()
            
            # Create metadata
            metadata = TaskMetadata(
                task_id=task_id,
                source=source,
                arrival_time=time.time(),
                original_filename=original_filename,
                execution_time=getattr(video_task, "execution_time", None),
                status=TaskStatus.ARRIVING
            )
            
            # Add to registry
            self.task_registry[task_id] = (video_task, metadata)
            
            # Log arrival event
            self._log_event(task_id, TaskEventType.TASK_ARRIVED, event_data={
                "source": source,
                "original_filename": original_filename,
                "profile_name": getattr(video_task, "profile_name", "Unknown"),
                "resolution": getattr(video_task, "resolution", "Unknown"),
                "memory_required": getattr(video_task, "memory_required", 0)
            })
            
            # Move to queued state
            self._update_task_status(task_id, TaskStatus.QUEUED)
            
            return task_id
    
    def _update_task_status(self, task_id: str, new_status: TaskStatus,
                         gpu_id: Optional[int] = None):
        """Update task status and log event (thread-safe: guarded by self._lock)."""
        with self._lock:
            if task_id not in self.task_registry:
                return False
            
            video_task, metadata = self.task_registry[task_id]
            old_status = metadata.status
            
            # Update metadata
            metadata.status = new_status
            
            # Update timestamp based on status
            current_time = time.time()
            if new_status == TaskStatus.QUEUED:
                metadata.queue_time = current_time
                if task_id not in self.queued_tasks:
                    self.queued_tasks.append(task_id)
            elif new_status == TaskStatus.SCHEDULED:
                metadata.scheduling_time = current_time
                metadata.assigned_gpu = gpu_id
                if task_id in self.queued_tasks:
                    self.queued_tasks.remove(task_id)
            elif new_status == TaskStatus.RUNNING:
                metadata.start_time = current_time
                if task_id in self.queued_tasks:
                    self.queued_tasks.remove(task_id)
                if task_id not in self.running_tasks:
                    self.running_tasks.append(task_id)
            elif new_status == TaskStatus.COMPLETED:
                metadata.completion_time = current_time
                if task_id in self.running_tasks:
                    self.running_tasks.remove(task_id)
                if task_id not in self.completed_tasks:
                    self.completed_tasks.append(task_id)
            elif new_status == TaskStatus.FAILED:
                if task_id in self.running_tasks:
                    self.running_tasks.remove(task_id)
                if task_id not in self.failed_tasks:
                    self.failed_tasks.append(task_id)
            elif new_status == TaskStatus.CANCELLED:
                for task_list in (self.queued_tasks, self.running_tasks):
                    if task_id in task_list:
                        task_list.remove(task_id)
            elif new_status == TaskStatus.WAITING:
                if task_id in self.queued_tasks:
                    self.queued_tasks.remove(task_id)
                metadata.waiting_time += (current_time - (metadata.queue_time or metadata.arrival_time))
            
            # Update GPU mappings
            if gpu_id is not None:
                self.task_to_gpu[task_id] = gpu_id
                if task_id not in self.gpu_to_tasks[gpu_id]:
                    self.gpu_to_tasks[gpu_id].append(task_id)
            
            # Log status change event
            event_type_map = {
                TaskStatus.QUEUED: TaskEventType.TASK_QUEUED,
                TaskStatus.SCHEDULED: TaskEventType.TASK_SCHEDULED,
                TaskStatus.RUNNING: TaskEventType.TASK_STARTED,
                TaskStatus.COMPLETED: TaskEventType.TASK_COMPLETED,
                TaskStatus.FAILED: TaskEventType.TASK_FAILED,
                TaskStatus.CANCELLED: TaskEventType.TASK_CANCELLED,
                TaskStatus.WAITING: TaskEventType.TASK_WAITING,
                TaskStatus.RETRY: TaskEventType.TASK_RETRY
            }
            
            if new_status in event_type_map:
                self._log_event(task_id, event_type_map[new_status], gpu_id, {
                    "old_status": old_status.value,
                    "new_status": new_status.value
                })
            return True
    
    def schedule_task(self, task_id: str, gpu_id: int) -> bool:
        """
        Schedule a task to a specific GPU atomically.
        Thread-safe: guarded by self._lock.
        """
        with self._lock:
            if task_id not in self.task_registry:
                return False
            
            if gpu_id not in self.gpu_to_tasks:
                return False
            
            metadata = self.task_registry[task_id][1]
            if metadata.status not in (TaskStatus.QUEUED, TaskStatus.RETRY, TaskStatus.SCHEDULED):
                return False
            if task_id in self.task_to_gpu and self.task_to_gpu[task_id] != gpu_id:
                return False
            return bool(self._update_task_status(task_id, TaskStatus.SCHEDULED, gpu_id))
    
    def start_task(self, task_id: str) -> bool:
        """
        Mark a task as started (running) atomically.
        Thread-safe: guarded by self._lock.
        """
        with self._lock:
            if task_id not in self.task_registry:
                return False
            
            metadata = self.task_registry[task_id][1]
            if metadata.status not in (TaskStatus.SCHEDULED, TaskStatus.RUNNING):
                return False
            return bool(self._update_task_status(task_id, TaskStatus.RUNNING, metadata.assigned_gpu))
    
    def complete_task(self, task_id: str) -> bool:
        """
        Mark a task as completed atomically.
        Thread-safe: guarded by self._lock. Prevents double-completion.
        """
        with self._lock:
            if task_id not in self.task_registry:
                return False

            if self.get_task_status(task_id) not in (TaskStatus.RUNNING, TaskStatus.SCHEDULED):
                return False
            
            # Remove from GPU mapping
            if task_id in self.task_to_gpu:
                gpu_id = self.task_to_gpu[task_id]
                if task_id in self.gpu_to_tasks[gpu_id]:
                    self.gpu_to_tasks[gpu_id].remove(task_id)
                del self.task_to_gpu[task_id]
            
            return bool(self._update_task_status(task_id, TaskStatus.COMPLETED))
    
    def fail_task(self, task_id: str, retry: bool = False) -> bool:
        """
        Mark a task as failed atomically.
        Thread-safe: guarded by self._lock.
        """
        with self._lock:
            if task_id not in self.task_registry:
                return False
            
            video_task, metadata = self.task_registry[task_id]
            
            if retry:
                metadata.retry_count += 1
                if task_id in self.running_tasks:
                    self.running_tasks.remove(task_id)
                if task_id in self.task_to_gpu:
                    gpu_id = self.task_to_gpu.pop(task_id)
                    if task_id in self.gpu_to_tasks[gpu_id]:
                        self.gpu_to_tasks[gpu_id].remove(task_id)
                self._update_task_status(task_id, TaskStatus.RETRY)
                # Re-queue the task
                self._update_task_status(task_id, TaskStatus.QUEUED)
            else:
                # Remove from GPU mapping
                if task_id in self.task_to_gpu:
                    gpu_id = self.task_to_gpu[task_id]
                    if task_id in self.gpu_to_tasks[gpu_id]:
                        self.gpu_to_tasks[gpu_id].remove(task_id)
                    del self.task_to_gpu[task_id]
                
                self._update_task_status(task_id, TaskStatus.FAILED)
            
            return True
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a task atomically.
        Thread-safe: guarded by self._lock.
        """
        with self._lock:
            if task_id not in self.task_registry:
                return False
            
            # Remove from GPU mapping
            if task_id in self.task_to_gpu:
                gpu_id = self.task_to_gpu[task_id]
                if task_id in self.gpu_to_tasks[gpu_id]:
                    self.gpu_to_tasks[gpu_id].remove(task_id)
                del self.task_to_gpu[task_id]
            
            # Remove from status lists
            if task_id in self.queued_tasks:
                self.queued_tasks.remove(task_id)
            if task_id in self.running_tasks:
                self.running_tasks.remove(task_id)
            
            return bool(self._update_task_status(task_id, TaskStatus.CANCELLED))

    def record_waiting(self, task_id: str) -> bool:
        """Record that a queued task remained waiting without changing its state (thread-safe)."""
        with self._lock:
            if task_id not in self.task_registry:
                return False
            metadata = self.task_registry[task_id][1]
            if metadata.status != TaskStatus.QUEUED:
                return False
            now = time.time()
            metadata.waiting_time = max(0.0, now - (metadata.queue_time or metadata.arrival_time))
            self._log_event(task_id, TaskEventType.TASK_WAITING, metadata.assigned_gpu, {
                "status": metadata.status.value,
                "waiting_time": metadata.waiting_time
            })
            return True
    
    def get_task(self, task_id: str) -> Optional[tuple]:
        """
        Get task and metadata by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            (VideoTask, TaskMetadata) or None
        """
        with self._lock:
            return self.task_registry.get(task_id)
    
    def get_task_metadata(self, task_id: str) -> Optional[TaskMetadata]:
        """
        Get task metadata by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            TaskMetadata or None
        """
        with self._lock:
            if task_id in self.task_registry:
                return self.task_registry[task_id][1]
            return None
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """
        Get task status by ID
        
        Args:
            task_id: Task ID
            
        Returns:
            TaskStatus or None
        """
        with self._lock:
            metadata = self.get_task_metadata(task_id)
            if metadata:
                return metadata.status
            return None
    
    def get_assigned_gpu(self, task_id: str) -> Optional[int]:
        """
        Get GPU assigned to a task
        
        Args:
            task_id: Task ID
            
        Returns:
            GPU ID or None
        """
        with self._lock:
            return self.task_to_gpu.get(task_id)
    
    def get_gpu_tasks(self, gpu_id: int) -> List[str]:
        """
        Get all tasks assigned to a GPU
        
        Args:
            gpu_id: GPU ID
            
        Returns:
            List of task IDs (safe copy)
        """
        with self._lock:
            return list(self.gpu_to_tasks.get(gpu_id, []))
    
    def get_all_tasks(self) -> Dict[str, tuple]:
        """
        Get all tasks in registry
        
        Returns:
            Dictionary of task_id -> (VideoTask, TaskMetadata) (safe copy)
        """
        with self._lock:
            return self.task_registry.copy()
    
    def get_tasks_by_status(self, status: TaskStatus) -> List[str]:
        """
        Get all tasks with a specific status
        
        Args:
            status: TaskStatus
            
        Returns:
            List of task IDs
        """
        with self._lock:
            return [
                task_id for task_id, (_, metadata) in self.task_registry.items()
                if metadata.status == status
            ]
    
    def get_task_counts(self) -> Dict[str, int]:
        """
        Get dynamic task counts by status
        
        Returns:
            Dictionary with counts for each status
        """
        with self._lock:
            counts = {
                "total": len(self.task_registry),
                "queued": len(self.get_tasks_by_status(TaskStatus.QUEUED)),
                "running": len(self.get_tasks_by_status(TaskStatus.RUNNING)),
                "completed": len(self.get_tasks_by_status(TaskStatus.COMPLETED)),
                "failed": len(self.get_tasks_by_status(TaskStatus.FAILED)),
                "arriving": len(self.get_tasks_by_status(TaskStatus.ARRIVING)),
                "scheduled": len(self.get_tasks_by_status(TaskStatus.SCHEDULED)),
                "cancelled": len(self.get_tasks_by_status(TaskStatus.CANCELLED)),
                "waiting": len(self.get_tasks_by_status(TaskStatus.WAITING)),
                "retry": len(self.get_tasks_by_status(TaskStatus.RETRY))
            }
            counts["retrying"] = counts["retry"]
            return counts

    def get_queue_statistics(self) -> Dict[str, Any]:
        """Return comprehensive queue statistics (thread-safe)."""
        with self._lock:
            counts = self.get_task_counts()
            counts["total_tasks"] = counts["total"]
            counts["queued_tasks"] = counts["queued"]
            counts["running_tasks"] = counts["running"]
            counts["completed_tasks"] = counts["completed"]
            counts["failed_tasks"] = counts["failed"]
            return counts
    
    def get_event_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent event log
        
        Args:
            limit: Maximum number of events to return
            
        Returns:
            List of event dictionaries
        """
        with self._lock:
            events = self.event_log[-limit:]
            return [event.to_dict() for event in events]
    
    def clear_event_log(self):
        """Clear event log"""
        with self._lock:
            self.event_log = []
    
    def reset(self):
        """Reset the queue (clear all tasks and events)"""
        with self._lock:
            self.task_registry.clear()
            self.task_to_gpu.clear()
            self.gpu_to_tasks = {1: [], 2: [], 3: [], 4: []}
            self.event_log.clear()
            self.task_counter = 0
            self.queued_tasks.clear()
            self.running_tasks.clear()
            self.completed_tasks.clear()
            self.failed_tasks.clear()

    def get_state_snapshot(self) -> Dict[str, Any]:
        """Atomically capture queue state snapshot for concurrency-safe monitoring."""
        with self._lock:
            return {
                "task_counts": self.get_task_counts(),
                "queued_tasks": list(self.queued_tasks),
                "running_tasks": list(self.running_tasks),
                "completed_tasks": list(self.completed_tasks),
                "failed_tasks": list(self.failed_tasks),
                "task_to_gpu": dict(self.task_to_gpu),
                "gpu_to_tasks": {gid: list(tasks) for gid, tasks in self.gpu_to_tasks.items()}
            }


class TaskArrivalSimulator:
    """
    Simulates dynamic task arrival patterns
    Thread-safe: guarded by self._lock.
    """
    
    def __init__(self, seed: int = 42):
        import random
        self._lock = threading.RLock()
        self.random = random.Random(seed)
        self.seed = seed
        self.mode = "manual"  # manual, fixed_rate, poisson, burst
        self.arrival_rate = 1.0  # tasks per second
        self.burst_size = 5
        self.burst_probability = 0.1
        self.current_step = 0
    
    def set_mode(self, mode: str, **kwargs):
        """
        Set arrival simulation mode
        
        Args:
            mode: "manual", "fixed_rate", "poisson", "burst"
            **kwargs: Mode-specific parameters
        """
        with self._lock:
            self.mode = mode
            if "arrival_rate" in kwargs:
                self.arrival_rate = kwargs["arrival_rate"]
            if "burst_size" in kwargs:
                self.burst_size = kwargs["burst_size"]
            if "burst_probability" in kwargs:
                self.burst_probability = kwargs["burst_probability"]
    
    def get_arrivals_for_step(self, step_duration: float = 1.0) -> int:
        """
        Get number of task arrivals for a simulation step
        
        Args:
            step_duration: Duration of the step in seconds
            
        Returns:
            Number of tasks to arrive
        """
        with self._lock:
            self.current_step += 1
            
            if self.mode == "manual":
                return 0  # Manual mode: no automatic arrivals
            elif self.mode == "fixed_rate":
                # Fixed rate: arrival_rate * step_duration
                expected = self.arrival_rate * step_duration
                return int(expected)
            elif self.mode == "poisson":
                # Poisson-like arrival
                import math
                expected = self.arrival_rate * step_duration
                # Use Poisson approximation
                if expected > 0:
                    # Simple Poisson approximation
                    k = 0
                    p = 1.0
                    l = math.exp(-expected)
                    while p > l:
                        k += 1
                        p *= self.random.random()
                    return k - 1
                return 0
            elif self.mode == "burst":
                # Burst mode: occasional bursts
                if self.random.random() < self.burst_probability:
                    return self.burst_size
                return 0
            else:
                return 0
    
    def reset(self):
        """Reset simulator state"""
        with self._lock:
            import random
            self.current_step = 0
            self.random = random.Random(self.seed)
