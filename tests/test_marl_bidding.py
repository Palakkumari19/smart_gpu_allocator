import numpy as np

from gpu import GPU
from marl_agent import MARLManager
from task import Task


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


def test_compute_bid_score_prefers_favorable_agents():
    manager = MARLManager(num_gpus=1)
    agent = manager.agents[0]
    task = Task()
    task.memory_required = 1
    task.execution_time = 2

    healthy_gpu = make_gpu(current_memory=1, temperature=60, task_queue=[], running_tasks=[])
    stressed_gpu = make_gpu(current_memory=6, temperature=85, task_queue=[Task(), Task()], running_tasks=[Task(), Task()])

    cluster_state = manager.get_cluster_state([healthy_gpu, stressed_gpu])
    healthy_state = agent.get_state(healthy_gpu, cluster_state)
    stressed_state = agent.get_state(stressed_gpu, cluster_state)

    agent.q_table[healthy_state] = np.array([0.0, 0.0, 0.0, 0.0])
    agent.q_table[stressed_state] = np.array([0.0, 0.0, 0.0, 0.0])

    healthy_bid = manager.compute_bid_score(healthy_gpu, task, agent, cluster_state)
    stressed_bid = manager.compute_bid_score(stressed_gpu, task, agent, cluster_state)

    assert healthy_bid > stressed_bid


def test_invalid_gpu_is_rejected_during_bidding():
    manager = MARLManager(num_gpus=2)
    task = Task()
    task.memory_required = 1
    task.execution_time = 2

    healthy_gpu = make_gpu(gpu_id=1)
    invalid_gpu = make_gpu(gpu_id=2)
    invalid_gpu.crashed = True

    selected_gpu_id = manager.select_gpu_for_task(task, [healthy_gpu, invalid_gpu], manager.get_cluster_state([healthy_gpu, invalid_gpu]))

    assert selected_gpu_id == 0


def test_bidding_falls_back_to_least_loaded_gpu():
    manager = MARLManager(num_gpus=2)
    task = Task()
    task.memory_required = 1
    task.execution_time = 2

    gpu_a = make_gpu(gpu_id=1)
    gpu_b = make_gpu(gpu_id=2)
    gpu_a.crashed = True
    gpu_b.crashed = True
    gpu_b.task_queue = [Task(), Task()]

    selected_gpu_id = manager.select_gpu_for_task(task, [gpu_a, gpu_b], manager.get_cluster_state([gpu_a, gpu_b]))

    assert selected_gpu_id == 0
