from dynamic_queue import DynamicTaskQueue, TaskArrivalSimulator, TaskEventType, TaskStatus
from environment import GPUEnvironment
from workload.task_generator import TaskGenerator, VideoProfile


def make_task(profile=VideoProfile.LIGHT):
    return TaskGenerator().generate_task(profile)


def test_dynamic_tasks_have_unique_ids_and_start_queued():
    queue = DynamicTaskQueue()
    ids = [queue.add_task(make_task()) for _ in range(5)]

    assert ids == [f"video_{index:04d}" for index in range(1, 6)]
    assert len(queue.get_all_tasks()) == 5
    assert queue.get_task_counts()["queued"] == 5


def test_burst_backpressure_and_completion_release():
    environment = GPUEnvironment(scheduler="fcfs")
    environment.set_eval_mode()
    environment.set_fault_injection_enabled(False)
    environment.enable_deterministic_physics()
    tasks = [make_task(VideoProfile.MEDIUM) for _ in range(20)]
    environment.attach_workload_trace([tasks] + [[] for _ in range(15)])

    environment.step()
    first_counts = environment.task_queue.get_task_counts()
    assert first_counts["total"] == 20
    assert first_counts["queued"] > 0
    assert all(len(ids) == len(set(ids)) for ids in environment.task_queue.gpu_to_tasks.values())

    for _ in range(15):
        environment.step()
    counts = environment.task_queue.get_task_counts()
    assert counts["completed"] > 0
    assert sum(counts[key] for key in ("queued", "scheduled", "running", "completed", "failed", "cancelled")) == 20


def test_task_cannot_be_assigned_to_two_gpus_and_quarantine_is_respected():
    environment = GPUEnvironment(scheduler="fcfs")
    environment.set_fault_injection_enabled(False)
    task = make_task()
    environment._register_task(task)
    assert environment._try_assign(environment.gpus[0], task)
    assert not environment._try_assign(environment.gpus[1], task)
    assert environment.task_queue.get_assigned_gpu(task.id) == 1

    risk_state = environment.gpus[1].get_risk_state()
    risk_state.is_quarantined = True
    risk_state.quarantine_reason = "test"
    quarantined_task = make_task()
    environment._register_task(quarantined_task)
    assert not environment._try_assign(environment.gpus[1], quarantined_task)
    assert environment.task_queue.get_task_status(quarantined_task.id) is TaskStatus.QUEUED


def test_lifecycle_events_and_all_four_gpus_are_exposed():
    queue = DynamicTaskQueue()
    task = make_task()
    task_id = queue.add_task(task)
    queue.schedule_task(task_id, 4)
    queue.start_task(task_id)
    queue.complete_task(task_id)

    event_types = [event["event_type"] for event in queue.get_event_log()]
    assert event_types == [
        TaskEventType.TASK_ARRIVED.value,
        TaskEventType.TASK_QUEUED.value,
        TaskEventType.TASK_SCHEDULED.value,
        TaskEventType.TASK_STARTED.value,
        TaskEventType.TASK_COMPLETED.value,
    ]
    assert set(queue.gpu_to_tasks) == {1, 2, 3, 4}
    assert queue.get_task_counts()["completed"] == 1


def test_arrival_modes_are_reproducible():
    fixed = TaskArrivalSimulator(seed=7)
    fixed.set_mode("fixed_rate", arrival_rate=2)
    assert [fixed.get_arrivals_for_step() for _ in range(3)] == [2, 2, 2]

    first = TaskArrivalSimulator(seed=7)
    second = TaskArrivalSimulator(seed=7)
    first.set_mode("poisson", arrival_rate=2)
    second.set_mode("poisson", arrival_rate=2)
    assert [first.get_arrivals_for_step() for _ in range(10)] == [second.get_arrivals_for_step() for _ in range(10)]

    burst = TaskArrivalSimulator(seed=7)
    burst.set_mode("burst", burst_size=10, burst_probability=1.0)
    assert burst.get_arrivals_for_step() == 10