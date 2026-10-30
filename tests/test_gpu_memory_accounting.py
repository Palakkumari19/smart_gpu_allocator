from gpu import GPU
from task import Task


def test_crash_recovery_does_not_drive_memory_negative():
    gpu = GPU(1, 8)
    for _ in range(4):
        task = Task()
        task.memory_required = 2
        assert gpu.assign_task(task)

    gpu.crashed = True
    gpu.crash_cooldown_timer = 1
    gpu.process_tasks()

    assert gpu.current_memory >= 0
    assert gpu.get_memory_usage_percentage() >= 0


def test_completed_tasks_release_memory_once():
    gpu = GPU(1, 8)
    task = Task()
    task.memory_required = 2
    task.execution_time = 1
    assert gpu.assign_task(task)

    gpu.is_warmed_up = True
    gpu.process_tasks()

    assert gpu.current_memory == 0


def test_memory_usage_percentage_never_exceeds_capacity():
    gpu = GPU(1, 8)
    task = Task()
    task.memory_required = 8
    assert gpu.assign_task(task)

    assert gpu.current_memory == 8
    assert gpu.get_memory_usage_percentage() == 100.0
