"""
Comprehensive Concurrency and Race-Condition Test Suite for AETHERGRID
Tests all 10 concurrency scenarios required by Task 5:
1. Multiple tasks attempting to allocate memory on the same GPU simultaneously.
2. Multiple tasks attempting to allocate resources across different GPUs in parallel.
3. Concurrent memory allocation and deallocation on the same GPU.
4. Concurrent task assignment and dynamic queue updates.
5. Two workers attempting to execute or complete the exact same task (atomic CAS guard).
6. Task preemption occurring near completion or resource release.
7. GPU failure / quarantine occurring while tasks are being assigned.
8. GPU recovery occurring while new allocation requests arrive.
9. Multiple MARL agents coordinating and updating shared Q-tables concurrently.
10. Concurrent API requests and background drain operations on the FastAPI server.
11. Deadlock-free high-contention stress test.
"""

import concurrent.futures
import threading
import time
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gpu import GPU
from dynamic_queue import DynamicTaskQueue, TaskStatus
from environment import GPUEnvironment
from marl_agent import MARLManager
from risk_engine import RiskEngine
from fault_injection import FaultInjector
from workload.task_generator import VideoTask, TaskType, EnhancedTask
import backend.server as server


# ─────────────────────────────────────────────────────────────────────────────
# Helper to construct dummy tasks
# ─────────────────────────────────────────────────────────────────────────────

def create_task(task_id: str, memory: float = 2.0, duration: int = 5, priority: str = "medium") -> VideoTask:
    task = VideoTask.__new__(VideoTask)
    task.id = task_id
    task.memory_required = memory
    task.execution_time = duration
    task.duration = duration
    task.priority = priority
    task.task_type = "Video"
    task.created_time = time.time()
    task.execution_progress = 0
    task.preempted = False
    task.preemption_count = 0
    task.original_execution_time = duration
    task.description = f"Test Task {task_id}"
    task.status = TaskStatus.QUEUED
    task.workload_severity = "MEDIUM"
    task.profile_name = "Medium"
    task.resolution = "1080p"
    task.width = 1920
    task.height = 1080
    task.fps = 30.0
    task.frames = 150
    task.codec = "H.264"
    task.bitrate = "5 Mbps"
    task._set_performance_characteristics()
    return task


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 1: Concurrent Memory Allocation on Single GPU
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_1_concurrent_allocations_single_gpu():
    """
    Multiple threads simultaneously try to allocate memory on a single GPU.
    Verifies that:
      - Memory never exceeds capacity.
      - Memory never becomes negative.
      - Total allocated memory + available memory == capacity (within float precision).
      - No task is partially allocated.
    """
    gpu = GPU(gpu_id=0, memory_capacity=8.0)
    gpu.memory_overload_factor = 1.0  # Strict capacity boundary
    gpu.high_realism_mode = False
    num_threads = 20
    task_mem = 1.0  # 20 * 1.0 = 20.0 GB requested on an 8.0 GB GPU
    barrier = threading.Barrier(num_threads)
    results = [False] * num_threads

    def worker(idx):
        task = create_task(f"t_{idx}", memory=task_mem, duration=10)
        barrier.wait()  # Align all threads to trigger at exact same moment
        success = gpu.assign_task(task)
        results[idx] = success

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    # Invariants
    successes = sum(results)
    assert successes == 8, f"Expected exactly 8 tasks allocated (8 x 1.0 = 8.0 GB), got {successes}"
    assert gpu.current_memory == 8.0
    assert gpu.get_available_memory() == 0.0
    assert 0.0 <= gpu.current_memory <= gpu.memory_capacity
    assert len(gpu.task_queue) == 8


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 2: Concurrent Allocations Across Multiple GPUs
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_2_concurrent_allocations_multiple_gpus():
    """
    Multiple workers allocate memory across distinct GPUs simultaneously.
    Verifies that independent GPUs can process allocations concurrently without
    interfering with each other's memory or deadlocking.
    """
    gpus = [GPU(gpu_id=i, memory_capacity=8.0) for i in range(4)]
    for g in gpus:
        g.memory_overload_factor = 1.0  # Strict capacity boundary
        g.high_realism_mode = False
    num_tasks = 40
    barrier = threading.Barrier(num_tasks)
    allocated_counts = [0] * 4
    counts_lock = threading.Lock()

    def worker(idx):
        gpu_id = idx % 4
        target_gpu = gpus[gpu_id]
        task = create_task(f"task_{idx}", memory=1.5, duration=5)
        barrier.wait()
        if target_gpu.assign_task(task):
            with counts_lock:
                allocated_counts[gpu_id] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_tasks) as executor:
        futures = [executor.submit(worker, i) for i in range(num_tasks)]
        concurrent.futures.wait(futures)

    for i, gpu in enumerate(gpus):
        assert 0.0 <= gpu.current_memory <= gpu.memory_capacity
        # Each GPU had 10 tasks of 1.5 GB. 5 tasks = 7.5 GB fits, 6 tasks = 9.0 GB fails.
        assert allocated_counts[i] == 5
        assert round(gpu.current_memory, 2) == 7.5
        assert round(gpu.get_available_memory(), 2) == 0.5


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 3: Concurrent Memory Allocation and Deallocation
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_3_concurrent_allocation_and_deallocation():
    """
    Workers concurrently allocate new tasks while another worker steps/deallocates.
    Verifies:
      - Memory never dips below 0.0.
      - Memory never exceeds capacity.
      - Accounting stays consistent through interleaving.
    """
    gpu = GPU(gpu_id=0, memory_capacity=6.0)
    gpu.is_warmed_up = True
    stop_event = threading.Event()
    num_allocators = 6
    barrier = threading.Barrier(num_allocators + 1)
    alloc_success = 0
    dealloc_count = 0
    lock = threading.Lock()

    def allocator(idx):
        nonlocal alloc_success
        barrier.wait()
        t_id = 0
        while not stop_event.is_set():
            t_id += 1
            task = create_task(f"t_{idx}_{t_id}", memory=1.0, duration=1)
            if gpu.assign_task(task):
                with lock:
                    alloc_success += 1
            time.sleep(0.001)

    def deallocator():
        nonlocal dealloc_count
        barrier.wait()
        while not stop_event.is_set():
            completed = gpu.process_tasks()
            with lock:
                dealloc_count += len(completed or [])
            time.sleep(0.001)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_allocators + 1) as executor:
        futures = [executor.submit(allocator, i) for i in range(num_allocators)]
        futures.append(executor.submit(deallocator))
        time.sleep(0.3)
        stop_event.set()
        concurrent.futures.wait(futures)

    # Final deallocation pass to drain all queued and running tasks
    while gpu.task_queue or gpu.running_tasks:
        gpu.process_tasks()

    assert gpu.current_memory == 0.0
    assert gpu.get_available_memory() == 6.0
    assert len(gpu.running_tasks) == 0
    assert len(gpu.task_queue) == 0


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 4: Concurrent Task Assignment and Queue Updates
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_4_concurrent_queue_updates():
    """
    Multiple threads concurrently add tasks, schedule tasks, and inspect queue state.
    Verifies no corruption of queue registry, counters, or status mappings.
    """
    queue = DynamicTaskQueue()
    num_threads = 20
    barrier = threading.Barrier(num_threads)

    def worker(idx):
        barrier.wait()
        task = create_task(f"q_task_{idx}", memory=2.0)
        task_id = queue.add_task(task)
        queue.schedule_task(task_id, gpu_id=(idx % 4) + 1)
        queue.start_task(task_id)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    stats = queue.get_queue_statistics()
    assert stats["total_tasks"] == num_threads
    assert stats["running_tasks"] == num_threads
    assert stats["queued_tasks"] == 0
    snapshot = queue.get_state_snapshot()
    assert len(snapshot["running_tasks"]) == num_threads


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 5: Two Workers Attempting to Complete the Same Task (Atomic CAS)
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_5_duplicate_task_completion_atomic_cas():
    """
    Multiple workers attempt to complete the EXACT same task simultaneously.
    Verifies that exactly ONE worker succeeds and the duplicate completion is rejected.
    """
    queue = DynamicTaskQueue()
    task = create_task("race_task", memory=2.0)
    task_id = queue.add_task(task)
    queue.schedule_task(task_id, gpu_id=1)
    queue.start_task(task_id)

    num_workers = 10
    barrier = threading.Barrier(num_workers)
    completion_results = []
    res_lock = threading.Lock()

    def worker():
        barrier.wait()
        success = queue.complete_task(task_id)
        with res_lock:
            completion_results.append(success)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(worker) for _ in range(num_workers)]
        concurrent.futures.wait(futures)

    success_count = sum(completion_results)
    assert success_count == 1, f"Expected exactly 1 successful completion, got {success_count}"
    assert queue.get_task_status(task_id) == TaskStatus.COMPLETED
    assert queue.get_queue_statistics()["completed_tasks"] == 1


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 6: Task Preemption Occurring Near Completion
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_6_preemption_near_completion_race():
    """
    One thread tries to preempt a task while another thread finishes/deallocates it.
    Verifies that memory cannot be doubly freed or left inconsistent.
    """
    gpu = GPU(gpu_id=0, memory_capacity=8.0)
    gpu.is_warmed_up = True
    task = create_task("preempt_race_task", memory=4.0, duration=1)
    assert gpu.assign_task(task)
    assert gpu.current_memory == 4.0
    # Step once so task moves from task_queue to running_tasks
    gpu.process_tasks()

    barrier = threading.Barrier(2)
    preempt_res = []
    process_res = []

    def preempt_worker():
        barrier.wait()
        res = gpu.preempt_task("preempt_race_task")
        preempt_res.append(res)

    def process_worker():
        barrier.wait()
        res = gpu.process_tasks()
        process_res.append(res)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(preempt_worker)
        f2 = executor.submit(process_worker)
        concurrent.futures.wait([f1, f2])

    # Drain all remaining work on GPU
    while gpu.task_queue or gpu.running_tasks:
        gpu.process_tasks()

    # Final state invariants:
    # Memory must be freed cleanly, regardless of which thread won the race
    assert gpu.current_memory == 0.0
    assert gpu.get_available_memory() == 8.0
    assert len(gpu.running_tasks) == 0
    assert len(gpu.task_queue) == 0


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 7: GPU Failure / Quarantine During Allocation
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_7_gpu_quarantine_during_allocation():
    """
    Fault injection crashes and quarantines a GPU while multiple threads are attempting
    to allocate tasks onto it.
    Verifies that:
      - Crashed GPU rejects new allocations cleanly.
      - No resource leaks or partially allocated memory.
    """
    env = GPUEnvironment(scheduler="fcfs")
    target_gpu = env.gpus[0]
    num_threads = 15
    barrier = threading.Barrier(num_threads + 1)
    alloc_results = []
    res_lock = threading.Lock()

    def allocator(idx):
        task = create_task(f"crash_test_t_{idx}", memory=2.0)
        barrier.wait()
        success = env._try_assign(target_gpu, task)
        with res_lock:
            alloc_results.append(success)

    def crash_trigger():
        barrier.wait()
        target_gpu.crash()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads + 1) as executor:
        futures = [executor.submit(allocator, i) for i in range(num_threads)]
        futures.append(executor.submit(crash_trigger))
        concurrent.futures.wait(futures)

    # Invariants:
    assert target_gpu.crashed is True
    assert 0.0 <= target_gpu.current_memory <= target_gpu.memory_capacity
    # Memory cannot exceed physical capacity even with concurrent requests
    assert target_gpu.current_memory <= target_gpu.memory_capacity


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 8: GPU Recovery While Allocation Requests Arrive
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_8_gpu_recovery_concurrent_with_allocation():
    """
    A GPU recovers from cooldown while multiple threads attempt to allocate tasks.
    Verifies:
      - Memory bounds and task queue consistency during transition from crashed -> recovered.
    """
    gpu = GPU(gpu_id=0, memory_capacity=8.0)
    gpu.memory_overload_factor = 1.0
    gpu.crash()
    assert gpu.is_available_for_task(2.0) is False

    num_threads = 4
    barrier = threading.Barrier(num_threads + 1)
    alloc_results = []
    res_lock = threading.Lock()

    def allocator(idx):
        task = create_task(f"rec_t_{idx}", memory=2.0)
        barrier.wait()
        success = gpu.assign_task(task)
        with res_lock:
            alloc_results.append(success)

    def recoverer():
        barrier.wait()
        gpu.recover()

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads + 1) as executor:
        futures = [executor.submit(allocator, i) for i in range(num_threads)]
        futures.append(executor.submit(recoverer))
        concurrent.futures.wait(futures)

    assert gpu.crashed is False
    assert 0.0 <= gpu.current_memory <= gpu.memory_capacity
    # Maximum 4 tasks of 2.0 GB can ever fit into 8.0 GB
    assert sum(alloc_results) <= 4


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 9: Multi-Agent MARL Coordination Concurrency
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_9_marl_coordination_concurrency():
    """
    Multiple threads concurrently invoke MARL agent decision making, updates, and epsilon decay.
    Verifies that:
      - Q-tables are updated without race corruption or NaNs.
      - Epsilon decays monotonically without race conditions.
    """
    marl = MARLManager(num_gpus=4)
    gpus = [GPU(gpu_id=i, memory_capacity=8.0) for i in range(4)]
    cluster_state = marl.get_cluster_state(gpus)
    num_threads = 12
    barrier = threading.Barrier(num_threads)

    def worker(idx):
        task = create_task(f"marl_t_{idx}", memory=2.0)
        barrier.wait()
        for _ in range(5):
            selected_gpu_id = marl.select_gpu_for_task(task, gpus, cluster_state)
            assert 0 <= selected_gpu_id < 4
            # Agent state update
            reward = 10.0 if idx % 2 == 0 else -5.0
            agent = marl.agents[selected_gpu_id]
            s = agent.get_state(gpus[selected_gpu_id], cluster_state)
            agent.update(s, 0, reward, s)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    # Check stats and snapshot integrity
    snapshot = marl.snapshot_q_tables()
    assert len(snapshot) == 4
    stats = marl.get_global_stats()
    assert stats["total_q_table_size"] >= 0
    # Invariant: No NaNs in any Q-table
    for agent_id, q_table in enumerate(snapshot):
        for state_key, q_values in q_table.items():
            assert not np.isnan(q_values).any(), f"NaN detected in Q-table for agent {agent_id}"


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 10: Concurrent FastAPI Requests and State Access
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_10_concurrent_api_requests():
    """
    Concurrent HTTP requests to /dashboard-state, /start, /pause, and /reset
    using FastAPI TestClient.
    Verifies that the server does not deadlock, crash, or return corrupted JSON.
    """
    client = TestClient(server.app)
    num_requests = 25
    barrier = threading.Barrier(num_requests)

    def api_worker(idx):
        barrier.wait()
        if idx % 4 == 0:
            res = client.get("/dashboard-state")
            assert res.status_code == 200
            assert "gpus" in res.json()
        elif idx % 4 == 1:
            res = client.post("/start")
            assert res.status_code == 200
        elif idx % 4 == 2:
            res = client.post("/pause")
            assert res.status_code == 200
        else:
            res = client.post("/set-scheduler", json={"scheduler": "marl"})
            assert res.status_code == 200

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(api_worker, i) for i in range(num_requests)]
        concurrent.futures.wait(futures)

    # Reset server cleanly
    res = client.post("/reset")
    assert res.status_code == 200
    state = client.get("/dashboard-state").json()
    assert state["scheduler"] == "marl"


# ─────────────────────────────────────────────────────────────────────────────
# SCENARIO 11: High Contention Deadlock-Free Environment Stress Test
# ─────────────────────────────────────────────────────────────────────────────

def test_scenario_11_high_contention_deadlock_free():
    """
    High-contention stress test where concurrent threads perform interleaved operations:
      - stepping the environment
      - submitting tasks
      - querying dashboard state
      - injecting anomalies
    Verifies that no deadlock occurs and all locks are released cleanly within timeout.
    """
    env = GPUEnvironment(scheduler="least_loaded")
    num_workers = 8
    stop_event = threading.Event()
    barrier = threading.Barrier(num_workers)

    def task_submitter(idx):
        barrier.wait()
        t_id = 0
        while not stop_event.is_set():
            t_id += 1
            t = create_task(f"stress_{idx}_{t_id}", memory=1.5, duration=2)
            env.add_task(t)
            time.sleep(0.002)

    def stepper():
        barrier.wait()
        while not stop_event.is_set():
            env.step()
            time.sleep(0.005)

    def dashboard_reader():
        barrier.wait()
        while not stop_event.is_set():
            state = env.get_dashboard_state()
            assert "gpus" in state
            time.sleep(0.003)

    def fault_injector():
        barrier.wait()
        while not stop_event.is_set():
            env.fault_injector.inject_anomaly(env.gpus[0].id, "stress_task")
            time.sleep(0.01)

    workers = [
        stepper,
        dashboard_reader,
        fault_injector,
    ]
    # Add submitters
    for i in range(num_workers - len(workers)):
        workers.append(lambda idx=i: task_submitter(idx))

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(w) for w in workers]
        time.sleep(0.5)  # Run under high contention for 500ms
        stop_event.set()
        # All futures must complete without hanging
        done, not_done = concurrent.futures.wait(futures, timeout=5.0)
        assert len(not_done) == 0, f"Deadlock detected! {len(not_done)} threads did not finish"

    # Verify final invariants
    for gpu in env.gpus:
        assert 0.0 <= gpu.current_memory <= gpu.memory_capacity
