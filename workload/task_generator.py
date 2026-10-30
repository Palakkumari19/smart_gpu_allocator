import random
import threading
from typing import Optional, Dict, Any, List
from enum import Enum
from fault_injection import FailureImpactCalculator

class TaskType(Enum):
    LLM = "LLM"
    IMAGE = "Image"
    VIDEO = "Video"

class VideoProfile(Enum):
    LIGHT = "Light"
    MEDIUM = "Medium"
    HEAVY = "Heavy"
    EXTREME = "Extreme"

class TaskGenerator:
    def __init__(self, seed: Optional[int] = None):
        self._lock = threading.RLock()
        self.seed = seed
        # Define video workload characteristics
        self.video_profiles = {
            VideoProfile.LIGHT: {
                "resolution": "720p",
                "width": 1280,
                "height": 720,
                "fps_range": (24, 30),
                "duration_range": (5, 20),
                "description": "Light Video (720p)",
                "workload_severity": "LOW"
            },
            VideoProfile.MEDIUM: {
                "resolution": "1080p",
                "width": 1920,
                "height": 1080,
                "fps_range": (30, 60),
                "duration_range": (10, 60),
                "description": "Medium Video (1080p)",
                "workload_severity": "MEDIUM"
            },
            VideoProfile.HEAVY: {
                "resolution": "1440p",
                "width": 2560,
                "height": 1440,
                "fps_range": (60, 60),
                "duration_range": (30, 120),
                "description": "Heavy Video (1440p)",
                "workload_severity": "HIGH"
            },
            VideoProfile.EXTREME: {
                "resolution": "4K",
                "width": 3840,
                "height": 2160,
                "fps_range": (60, 60),
                "duration_range": (60, 180),
                "description": "Extreme Video (4K)",
                "workload_severity": "VERY HIGH"
            }
        }
        
        # Backward compatibility for scientific evaluation and task generator benchmarks
        self.task_weights = {
            TaskType.LLM: 0.40,
            TaskType.IMAGE: 0.35,
            TaskType.VIDEO: 0.25
        }
        self.workload_profiles = {
            TaskType.LLM: {
                "memory_range": (1, 4),
                "time_range": (2, 5),
                "description": "LLM inference request",
                "priority": "high"
            },
            TaskType.IMAGE: {
                "memory_range": (2, 6),
                "time_range": (1, 4),
                "description": "Image generation request",
                "priority": "medium"
            },
            TaskType.VIDEO: {
                "memory_range": (4, 8),
                "time_range": (4, 12),
                "description": "Video processing request",
                "priority": "low"
            }
        }
        
        # Phase 4: Failure impact calculator
        self.failure_impact_calculator = FailureImpactCalculator()
        
        # Profile weights for randomness
        self.profile_weights = {
            VideoProfile.LIGHT: 0.40,
            VideoProfile.MEDIUM: 0.35,
            VideoProfile.HEAVY: 0.20,
            VideoProfile.EXTREME: 0.05
        }
        
        # Statistics tracking
        self.generated_tasks = {profile: 0 for profile in VideoProfile}
        self.total_generated = 0

    def _calculate_resource_demand(self, width, height, fps, duration):
        """
        Calculate memory requirement and execution time based on video dimensions and duration.
        """
        # Base formula: width * height * fps * duration
        # Normalize to 1-8 GB (since GPU capacity is 8GB)
        # Normalize time to a reasonable number of simulation steps (e.g. 1-15 steps)
        
        pixels_per_frame = width * height
        frames = fps * duration
        total_pixels = pixels_per_frame * frames
        
        # Max theoretical: 4K (3840x2160) * 60fps * 180s = 89,579,520,000 pixels
        # Min theoretical: 720p (1280x720) * 24fps * 5s = 110,592,000 pixels
        max_pixels = 3840 * 2160 * 60 * 180
        
        # Memory estimation (VRAM scales strongly with resolution/pixels per frame, less with duration)
        # Let's say 4K requires ~7GB, 720p requires ~1GB
        memory_factor = pixels_per_frame / (3840 * 2160)
        memory_required = max(1.0, min(7.5, memory_factor * 7.5))
        
        # Add some random variance to memory
        memory_required += random.uniform(-0.5, 0.5)
        memory_required = max(1.0, min(8.0, round(memory_required, 1)))

        # Execution time estimation (scales with total pixels processed)
        time_factor = total_pixels / max_pixels
        execution_time = max(1, min(20, int(time_factor * 20)))
        
        # Add some variance to execution time
        execution_time += random.randint(-1, 2)
        execution_time = max(1, execution_time)

        return memory_required, execution_time, frames

    def generate_task(self, task_type=None):
        """
        Generate a single video task with specified or random profile
        """
        if task_type is None:
            # Random profile selection
            profile = random.choices(
                list(VideoProfile),
                weights=list(self.profile_weights.values())
            )[0]
        else:
            # For compatibility if someone passes a string
            if isinstance(task_type, str):
                try:
                    profile = VideoProfile(task_type.upper())
                except ValueError:
                    profile = VideoProfile.MEDIUM
            else:
                profile = task_type
        
        # Get profile characteristics
        prof_data = self.video_profiles[profile]
        
        fps = random.randint(*prof_data["fps_range"])
        duration = random.randint(*prof_data["duration_range"])
        width = prof_data["width"]
        height = prof_data["height"]
        
        memory_required, execution_time, frames = self._calculate_resource_demand(width, height, fps, duration)
        
        # Pick a random codec and bitrate for flavor
        codecs = ["H.264", "H.265", "AV1", "VP9"]
        codec = random.choice(codecs)
        
        # Bitrate approximation based on resolution
        if profile == VideoProfile.LIGHT: bitrate = f"{random.uniform(2.0, 5.0):.1f} Mbps"
        elif profile == VideoProfile.MEDIUM: bitrate = f"{random.uniform(5.0, 15.0):.1f} Mbps"
        elif profile == VideoProfile.HEAVY: bitrate = f"{random.uniform(15.0, 30.0):.1f} Mbps"
        else: bitrate = f"{random.uniform(30.0, 80.0):.1f} Mbps"

        # Create enhanced task object
        task = VideoTask(
            profile_name=profile.value,
            resolution=prof_data["resolution"],
            width=width,
            height=height,
            fps=fps,
            duration=duration,
            frames=frames,
            codec=codec,
            bitrate=bitrate,
            workload_severity=prof_data["workload_severity"],
            memory_required=memory_required,
            execution_time=execution_time,
            description=prof_data["description"]
        )
        
        # Update statistics
        self.generated_tasks[profile] += 1
        self.total_generated += 1
        
        return task
    
    def generate_batch(self, count, task_types=None):
        """
        Generate a batch of tasks
        """
        tasks = []
        for _ in range(count):
            task = self.generate_task()
            tasks.append(task)
        return tasks
    
    def generate_burst_workload(self, intensity="medium"):
        """
        Generate burst workload based on intensity
        """
        intensity_levels = {
            "low": (1, 3),
            "medium": (3, 8),
            "high": (5, 12),
            "extreme": (8, 20)
        }
        
        count_range = intensity_levels.get(intensity, (3, 8))
        task_count = random.randint(*count_range)
        
        return self.generate_batch(task_count)
    
    def get_workload_statistics(self):
        """
        Get statistics about generated tasks
        """
        stats = {
            "total_generated": self.total_generated,
            "task_distribution": {
                profile.value: count for profile, count in self.generated_tasks.items()
            },
            "percentages": {
                profile.value: (count / max(1, self.total_generated)) * 100
                for profile, count in self.generated_tasks.items()
            }
        }
        return stats
    
    def reset_statistics(self):
        """
        Reset generation statistics
        """
        self.generated_tasks = {profile: 0 for profile in VideoProfile}
        self.total_generated = 0

class VideoTask:
    """
    Video task class with workload-specific properties
    """
    def __init__(self, profile_name, resolution, width, height, fps, duration, frames, codec, bitrate, workload_severity, memory_required, execution_time, description):
        # Basic properties
        self.task_type = "Video"
        self.profile_name = profile_name
        self.resolution = resolution
        self.width = width
        self.height = height
        self.fps = fps
        self.duration = duration
        self.frames = frames
        self.codec = codec
        self.bitrate = bitrate
        self.workload_severity = workload_severity
        
        self.memory_required = memory_required
        self.execution_time = execution_time
        self.description = description
        self.priority = "medium"
        
        # Execution tracking
        self.execution_progress = 0
        self.preempted = False
        self.preemption_count = 0
        self.original_execution_time = None
        
        # Workload-specific properties
        self.id = f"video_{random.randint(1000, 9999)}.mp4"
        self.created_time = None  # Will be set by environment
        
        # Performance characteristics based on task type
        self._set_performance_characteristics()
    
    def _set_performance_characteristics(self):
        """
        Set performance characteristics
        """
        self.characteristics = {
            "cpu_intensive": True,
            "memory_bandwidth": "high",
            "parallelizable": False,
            "interruptible": False
        }
    
    def preempt(self):
        """
        Mark task as preempted with type-specific behavior
        """
        if self.characteristics["interruptible"]:
            self.preempted = True
            self.preemption_count += 1
            
            # Resume boost based on task type
            if self.original_execution_time is None:
                self.original_execution_time = self.execution_time
            
            self.execution_time = max(1, self.execution_time - 0.7)
            
            return True
        return False
    
    def resume(self):
        """
        Resume task after preemption
        """
        self.preempted = False
    
    def get_resource_intensity(self):
        """
        Get resource intensity score for scheduling decisions
        """
        memory_score = self.memory_required / 8.0  # Normalize to 8GB
        time_score = self.execution_time / 20.0  # Normalize to 20 steps
        
        return memory_score + time_score
    
    def get_failure_impact(self) -> float:
        """
        Get failure impact score for this task (0-1)
        
        Returns:
            Normalized failure impact score
        """
        # Import here to avoid circular dependency
        from fault_injection import FailureImpactCalculator
        calculator = FailureImpactCalculator()
        return calculator.calculate_failure_impact(self)
    
    def get_restart_cost(self) -> float:
        """
        Get restart/recovery cost for this task
        
        Returns:
            Restart cost in simulation steps
        """
        # Import here to avoid circular dependency
        from fault_injection import FailureImpactCalculator
        calculator = FailureImpactCalculator()
        return calculator.calculate_restart_cost(self)
    
    def __str__(self):
        return (f"VideoTask({self.id}): {self.resolution} {self.fps}FPS - "
                f"{self.memory_required}GB, {self.execution_time} steps")
    
    def __repr__(self):
        return self.__str__()


class EnhancedTask:
    """
    Enhanced task class for multi-modal workloads (LLM, Image, Video)
    providing full backward compatibility for scientific evaluation and baseline comparisons.
    """
    def __init__(self, task_type, memory_required, execution_time, description="Workload task", priority="medium"):
        self.task_type = task_type
        self.memory_required = memory_required
        self.execution_time = execution_time
        self.description = description
        self.priority = priority
        self.execution_progress = 0
        self.preempted = False
        self.preemption_count = 0
        self.original_execution_time = None
        self.id = f"task_{random.randint(1000, 9999)}"
        self.created_time = None
        
        # Compatibility properties for video-specific pipeline
        self.profile_name = "Medium"
        self.resolution = "1080p"
        self.width = 1920
        self.height = 1080
        self.fps = 30
        self.duration = 10
        self.frames = 300
        self.codec = "H.264"
        self.bitrate = "5.0 Mbps"
        severity_map = {"high": "HIGH", "medium": "MEDIUM", "low": "LOW", "critical": "VERY HIGH"}
        self.workload_severity = severity_map.get(str(priority).lower(), "MEDIUM")
        self.severity = self.workload_severity
        self.characteristics = {
            "cpu_intensive": True,
            "memory_bandwidth": "high",
            "parallelizable": False,
            "interruptible": True
        }

    def preempt(self):
        self.preempted = True
        self.preemption_count += 1
        if self.original_execution_time is None:
            self.original_execution_time = self.execution_time
        self.execution_time = max(1, self.execution_time - 0.7)
        return True

    def resume(self):
        self.preempted = False

    def get_resource_intensity(self):
        memory_score = self.memory_required / 8.0
        time_score = self.execution_time / 20.0
        return memory_score + time_score

    def get_failure_impact(self) -> float:
        from fault_injection import FailureImpactCalculator
        calculator = FailureImpactCalculator()
        return calculator.calculate_failure_impact(self)

    def get_restart_cost(self) -> float:
        from fault_injection import FailureImpactCalculator
        calculator = FailureImpactCalculator()
        return calculator.calculate_restart_cost(self)

    def __str__(self):
        return f"EnhancedTask({self.id}): {self.task_type} - {self.memory_required}GB, {self.execution_time} steps"

    def __repr__(self):
        return self.__str__()

