from gpu import GPU
from task import Task
from marl_agent import MARLManager


def make_gpu(**overrides):
    gpu = GPU(1, 8)
    gpu.current_memory = 2
    gpu.temperature = 60
    gpu.task_queue = []
    gpu.running_tasks = []
    gpu.memory_fragments = []
    gpu.crash_cooldown_timer = 0
    gpu.crashed = False
    for key, value in overrides.items():
        setattr(gpu, key, value)
    return gpu


def test_failure_penalty_is_stronger_and_bounded():
    manager = MARLManager(num_gpus=1)
    gpu = make_gpu()
    task = Task()
    reward = manager.calculate_reward(gpu, task, False)
    assert reward == -20.0


def test_success_reward_is_bounded_and_stable():
    manager = MARLManager(num_gpus=1)
    gpu = make_gpu()
    task = Task()
    reward = manager.calculate_reward(gpu, task, True)
    assert -20.0 <= reward <= 12.0


def test_severe_overload_is_penalized():
    manager = MARLManager(num_gpus=1)
    gpu = make_gpu(task_queue=[Task(), Task(), Task(), Task(), Task()])
    task = Task()
    reward = manager.calculate_reward(gpu, task, True)
    assert reward < 3.0
