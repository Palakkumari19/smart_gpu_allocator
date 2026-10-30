"""Generate isolated, replayable workload traces.

The arrival process uses a dedicated RNG so RL/MARL exploration cannot
change how many tasks arrive or which tasks arrive.
"""

from __future__ import annotations

import random
import numpy as np

from workload.task_generator import EnhancedTask, TaskGenerator, TaskType


PATTERN_REQUESTS = {
    "light": (1, 3),
    "heavy": (8, 15),
    "burst": (0, 20),
    "peak": (12, 18),
    "off_peak": (0, 2),
    "mixed": (2, 12),
    "normal": (2, 5),
}


def _requests_for_step(pattern, rng):
    if pattern == "burst":
        if rng.random() < 0.7:
            return rng.randint(0, 2)
        return rng.randint(15, 20)
    low, high = PATTERN_REQUESTS[pattern]
    return rng.randint(low, high)


def _make_task(generator, rng, diverse_mix=False):
    if diverse_mix:
        task_types = list(TaskType)
        task_type = rng.choice(task_types)
        priorities = ["low", "medium", "high", "critical"]
        priority = rng.choice(priorities)
        mem = rng.randint(1, 7)
        exec_time = rng.randint(1, 8)
        return EnhancedTask(
            task_type=task_type.value,
            memory_required=mem,
            execution_time=exec_time,
            description=f"Diverse {task_type.value} Task ({priority})",
            priority=priority,
        )

    task_type = rng.choices(
        list(TaskType),
        weights=list(generator.task_weights.values()),
    )[0]
    profile = generator.workload_profiles[task_type]
    return EnhancedTask(
        task_type=task_type.value,
        memory_required=rng.randint(*profile["memory_range"]),
        execution_time=rng.randint(*profile["time_range"]),
        description=profile["description"],
        priority=profile["priority"],
    )


def generate_workload_trace(pattern, steps, seed, diverse_mix=False):
    """Return a list of length `steps`; each entry is a list of tasks."""
    if pattern not in PATTERN_REQUESTS:
        raise ValueError(f"Unknown traffic pattern: {pattern}")
    rng = random.Random(seed)
    generator = TaskGenerator()
    trace = []
    for _ in range(steps):
        count = _requests_for_step(pattern, rng)
        trace.append([_make_task(generator, rng, diverse_mix=diverse_mix) for _ in range(count)])
    return trace


def trace_task_count(trace):
    return sum(len(step_tasks) for step_tasks in trace)


def get_trace_distribution(trace):
    """Compute task size, time, priority and severity distribution from a trace."""
    all_tasks = [t for step in trace for t in step]
    if not all_tasks:
        return {"total_tasks": 0}
    mems = [t.memory_required for t in all_tasks]
    times = [t.execution_time for t in all_tasks]
    priorities = {}
    severities = {}
    types = {}
    for t in all_tasks:
        p = getattr(t, "priority", "unknown")
        priorities[p] = priorities.get(p, 0) + 1
        s = getattr(t, "workload_severity", getattr(t, "severity", "unknown"))
        severities[s] = severities.get(s, 0) + 1
        tt = getattr(t, "task_type", "unknown")
        types[tt] = types.get(tt, 0) + 1

    return {
        "total_tasks": len(all_tasks),
        "avg_memory": float(np.mean(mems)),
        "min_memory": min(mems),
        "max_memory": max(mems),
        "avg_time": float(np.mean(times)),
        "min_time": min(times),
        "max_time": max(times),
        "priorities": priorities,
        "severities": severities,
        "types": types,
    }
