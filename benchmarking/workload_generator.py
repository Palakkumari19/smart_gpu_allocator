"""
Bit 9: Workload Generator & Task Stream
Generates deterministic, reproducible task streams to guarantee fair comparison
across all evaluated scheduling strategies.
"""

from copy import deepcopy
import random
from typing import List, Dict, Any, Optional
import numpy as np

from workload.task_generator import TaskGenerator, VideoProfile, EnhancedTask


class TaskStream:
    """An immutable, reusable container for a deterministic workload stream."""

    def __init__(self, seed: int, workload_type: str, tasks: List[EnhancedTask]):
        self.seed = seed
        self.workload_type = workload_type
        self.tasks = tasks

    def get_fresh_copy(self) -> List[EnhancedTask]:
        """Return a fresh deep-copy of the task stream with reset runtime state."""
        clones = []
        for t in self.tasks:
            clone = deepcopy(t)
            clone.execution_progress = 0
            clone.preempted = False
            clone.preemption_count = 0
            clones.append(clone)
        return clones

    def get_summary(self) -> Dict[str, Any]:
        """Return statistical summary of the generated task stream."""
        memories = [t.memory_required for t in self.tasks]
        durations = [t.execution_time for t in self.tasks]
        profiles = [getattr(t, "profile_name", "Unknown") for t in self.tasks]

        profile_counts = {}
        for p in profiles:
            profile_counts[p] = profile_counts.get(p, 0) + 1

        return {
            "seed": self.seed,
            "workload_type": self.workload_type,
            "total_tasks": len(self.tasks),
            "memory_mean_gb": round(float(np.mean(memories)), 2) if memories else 0.0,
            "memory_max_gb": round(float(np.max(memories)), 2) if memories else 0.0,
            "memory_min_gb": round(float(np.min(memories)), 2) if memories else 0.0,
            "duration_mean_steps": round(float(np.mean(durations)), 2) if durations else 0.0,
            "duration_max_steps": int(np.max(durations)) if durations else 0,
            "profile_distribution": profile_counts,
        }


class WorkloadGenerator:
    """Generates reproducible workload streams for benchmarking."""

    @staticmethod
    def generate_stream(seed: int, num_tasks: int, workload_type: str = "mixed") -> TaskStream:
        """
        Generate a deterministic task stream given a random seed and profile.
        Guarantees that two calls with identical seed produce identical task parameters.
        """
        # Save Python & Numpy RNG states to isolate workload generation
        py_state = random.getstate()
        np_state = np.random.get_state()

        try:
            random.seed(seed)
            np.random.seed(seed)

            gen = TaskGenerator(seed=seed)

            # Adjust profile weights according to workload type
            if workload_type == "light_heavy":
                gen.profile_weights = {
                    VideoProfile.LIGHT: 0.70,
                    VideoProfile.MEDIUM: 0.20,
                    VideoProfile.HEAVY: 0.10,
                    VideoProfile.EXTREME: 0.00,
                }
            elif workload_type == "extreme_heavy":
                gen.profile_weights = {
                    VideoProfile.LIGHT: 0.10,
                    VideoProfile.MEDIUM: 0.20,
                    VideoProfile.HEAVY: 0.40,
                    VideoProfile.EXTREME: 0.30,
                }
            elif workload_type == "burst":
                gen.profile_weights = {
                    VideoProfile.LIGHT: 0.20,
                    VideoProfile.MEDIUM: 0.40,
                    VideoProfile.HEAVY: 0.30,
                    VideoProfile.EXTREME: 0.10,
                }
            else:  # "mixed" default
                gen.profile_weights = {
                    VideoProfile.LIGHT: 0.40,
                    VideoProfile.MEDIUM: 0.35,
                    VideoProfile.HEAVY: 0.20,
                    VideoProfile.EXTREME: 0.05,
                }

            tasks = []
            for i in range(num_tasks):
                task = gen.generate_task()
                # Ensure deterministic unique ID based on seed and index
                task.id = f"bench_s{seed}_t{i:04d}"
                tasks.append(task)

            return TaskStream(seed=seed, workload_type=workload_type, tasks=tasks)

        finally:
            # Restore RNG states
            random.setstate(py_state)
            np.random.set_state(np_state)
