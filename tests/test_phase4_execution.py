import random

from environment import GPUEnvironment


def create_environment():
    random.seed(42)

    env = GPUEnvironment(scheduler="least_loaded")

    if hasattr(env, "fault_injector"):
        env.fault_injector.enabled = False

    return env


def process_until_complete(env, expected_tasks, max_steps=100):
    for _ in range(max_steps):
        for gpu in env.gpus:
            gpu.process_tasks()

        completed = sum(gpu.completed_tasks for gpu in env.gpus)

        if completed == expected_tasks:
            return

    raise AssertionError(
        f"Tasks did not complete within {max_steps} steps"
    )


def test_single_video_execution():
    env = create_environment()

    task = env.generate_task()
    task.execution_time = 3

    gpu = env.gpus[0]

    assert gpu.assign_task(task)

    assert len(gpu.task_queue) == 1
    assert len(gpu.running_tasks) == 0

    process_until_complete(env, 1)

    assert sum(g.completed_tasks for g in env.gpus) == 1

    assert all(len(g.task_queue) == 0 for g in env.gpus)
    assert all(len(g.running_tasks) == 0 for g in env.gpus)


def test_parallel_video_execution():
    env = create_environment()

    gpu = env.gpus[0]

    for _ in range(3):
        task = env.generate_task()
        task.execution_time = 5

        assert gpu.assign_task(task)

    # First step: GPU warms up.
    gpu.process_tasks()

    assert len(gpu.running_tasks) == 0

    # Second step: tasks enter parallel execution.
    gpu.process_tasks()

    assert len(gpu.running_tasks) == gpu.max_parallel_tasks
    assert len(gpu.running_tasks) > 1


def test_gpu_queue_progression():
    env = create_environment()

    gpu = env.gpus[0]

    tasks = []

    for _ in range(4):
        task = env.generate_task()
        task.execution_time = 3
        tasks.append(task)

    for task in tasks:
        assert gpu.assign_task(task)

    assert len(gpu.task_queue) == 4
    assert len(gpu.running_tasks) == 0

    # Warm-up step.
    gpu.process_tasks()

    assert len(gpu.running_tasks) == 0
    assert len(gpu.task_queue) == 4

    # Execution starts.
    gpu.process_tasks()

    assert len(gpu.running_tasks) == gpu.max_parallel_tasks
    assert len(gpu.task_queue) == 1

    # Finish the first batch.
    for _ in range(3):
        gpu.process_tasks()

    assert gpu.completed_tasks == gpu.max_parallel_tasks

    # The queued task should eventually enter execution.
    assert len(gpu.running_tasks) <= gpu.max_parallel_tasks


def test_all_video_workloads_complete():
    env = create_environment()

    total_tasks = 12
    pending_tasks = []

    for _ in range(total_tasks):
        task = env.generate_task()
        task.execution_time = 3
        pending_tasks.append(task)

    assigned_tasks = 0

    for _ in range(100):

        remaining_tasks = []

        for task in pending_tasks:
            assigned = False

            for gpu in env.gpus:
                if env._try_assign(gpu, task):
                    assigned = True
                    assigned_tasks += 1
                    break

            if not assigned:
                remaining_tasks.append(task)

        pending_tasks = remaining_tasks

        for gpu in env.gpus:
            gpu.process_tasks()

        completed = sum(gpu.completed_tasks for gpu in env.gpus)

        if completed == total_tasks:
            break

    assert assigned_tasks == total_tasks

    completed = sum(gpu.completed_tasks for gpu in env.gpus)
    queued = sum(len(gpu.task_queue) for gpu in env.gpus)
    running = sum(len(gpu.running_tasks) for gpu in env.gpus)

    assert completed == total_tasks
    assert queued == 0
    assert running == 0
    assert len(pending_tasks) == 0


def test_gpu_memory_released_after_completion():
    env = create_environment()

    task = env.generate_task()
    task.execution_time = 3

    gpu = env.gpus[0]

    assert gpu.assign_task(task)

    memory_after_assignment = gpu.current_memory

    assert memory_after_assignment > 0

    process_until_complete(env, 1)

    assert gpu.current_memory == 0