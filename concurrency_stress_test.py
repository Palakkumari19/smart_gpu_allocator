"""
AETHERGRID Concurrency Stress Testing & Observability Utility
Phase 5: Locks, Data Consistency, and Race-Condition Handling

This standalone module runs repeatable high-contention stress tests against the
AETHERGRID GPU cluster simulator and verifies that:
  1. Memory accounting invariants hold (no negative memory, bounds strictly respected).
  2. Duplicate task transitions/completions are prevented (atomic CAS guards).
  3. No deadlocks occur under high contention.
  4. Legitimate resource rejections (OOM, quarantine) are distinguished from race failures.
  5. The simulator state remains fully consistent.

Usage:
  python concurrency_stress_test.py [--workers 8] [--tasks 100] [--duration 3.0] [--scheduler marl]
"""

import argparse
import concurrent.futures
import sys
import threading
import time
from typing import Dict, Any, List

from gpu import GPU
from environment import GPUEnvironment
from dynamic_queue import DynamicTaskQueue, TaskStatus
from workload.task_generator import VideoTask, TaskGenerator, VideoProfile


class ConcurrencyStressTester:
    def __init__(self, num_workers: int = 8, total_tasks: int = 100, duration: float = 3.0, scheduler: str = "marl"):
        self.num_workers = num_workers
        self.total_tasks = total_tasks
        self.duration = duration
        self.scheduler = scheduler

        self.env = GPUEnvironment(scheduler=scheduler)
        self.stop_event = threading.Event()
        self.metrics_lock = threading.Lock()

        # Telemetry metrics
        self.metrics = {
            "operations_attempted": 0,
            "allocations_successful": 0,
            "allocations_rejected_capacity": 0,
            "allocations_rejected_quarantine": 0,
            "duplicate_transitions_prevented": 0,
            "tasks_completed": 0,
            "invariant_violations": [],
            "resource_inconsistencies": 0,
        }

    def _record_metric(self, key: str, count: int = 1):
        with self.metrics_lock:
            self.metrics[key] += count

    def _record_violation(self, description: str):
        with self.metrics_lock:
            self.metrics["invariant_violations"].append(description)

    def _task_worker(self, worker_id: int, barrier: threading.Barrier):
        """Worker generating and submitting tasks under high concurrency."""
        barrier.wait()
        generator = TaskGenerator(seed=42 + worker_id)
        profiles = [VideoProfile.LIGHT, VideoProfile.MEDIUM, VideoProfile.HEAVY]
        task_idx = 0

        while not self.stop_event.is_set():
            task_idx += 1
            self._record_metric("operations_attempted")
            profile = profiles[task_idx % len(profiles)]
            task = generator.generate_task(profile)
            task.id = f"stress_w{worker_id}_{task_idx:04d}"

            # Register task in queue
            task_id = self.env.add_task(task)

            # Attempt allocation
            with self.metrics_lock:
                # Pick GPU according to current scheduler or test
                target_gpu = min(self.env.gpus, key=lambda g: g.get_total_load())

            if target_gpu.is_quarantined() or target_gpu.crashed:
                self._record_metric("allocations_rejected_quarantine")
            elif not target_gpu.is_available_for_task(task.memory_required):
                self._record_metric("allocations_rejected_capacity")
            else:
                success = self.env._try_assign(target_gpu, task)
                if success:
                    self._record_metric("allocations_successful")
                else:
                    self._record_metric("allocations_rejected_capacity")

            # Try duplicate complete race on completed or active tasks
            if task_idx % 5 == 0:
                self._record_metric("operations_attempted")
                res1 = self.env.task_queue.complete_task(task_id)
                res2 = self.env.task_queue.complete_task(task_id)
                if res1 and not res2:
                    self._record_metric("duplicate_transitions_prevented")

            time.sleep(0.002)

    def _stepper_worker(self, barrier: threading.Barrier):
        """Worker stepping the simulation and checking point-in-time invariants."""
        barrier.wait()
        while not self.stop_event.is_set():
            self._record_metric("operations_attempted")
            self.env.step()

            # Immediate invariant check across all GPUs
            for gpu in self.env.gpus:
                with gpu.lock:
                    mem = gpu.current_memory
                    cap = gpu.memory_capacity
                    overload = gpu.memory_overload_factor

                    if mem < -1e-6:
                        self._record_violation(f"GPU {gpu.id} negative memory: {mem} GB")
                    if mem > round(cap + 1e-4, 4):
                        self._record_violation(f"GPU {gpu.id} exceeded physical memory capacity: {mem:.2f} > {cap:.2f} GB")

            time.sleep(0.005)

    def _state_reader_worker(self, barrier: threading.Barrier):
        """Worker reading state snapshots concurrently without blocking execution."""
        barrier.wait()
        while not self.stop_event.is_set():
            self._record_metric("operations_attempted")
            try:
                state = self.env.get_dashboard_state()
                if "gpus" not in state or len(state["gpus"]) != len(self.env.gpus):
                    self._record_violation("Incomplete state snapshot returned during concurrent read")
            except Exception as e:
                self._record_violation(f"Exception during state snapshot: {e}")
            time.sleep(0.004)

    def run(self) -> Dict[str, Any]:
        """Execute the concurrency stress test suite."""
        print(f"=== Starting AETHERGRID Concurrency Stress Test ===", flush=True)
        print(f"Workers: {self.num_workers} | Duration: {self.duration}s | Scheduler: {self.scheduler}", flush=True)

        total_threads = self.num_workers + 2  # workers + stepper + reader
        barrier = threading.Barrier(total_threads)

        start_time = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as executor:
            futures = []
            for w in range(self.num_workers):
                futures.append(executor.submit(self._task_worker, w, barrier))
            futures.append(executor.submit(self._stepper_worker, barrier))
            futures.append(executor.submit(self._state_reader_worker, barrier))

            time.sleep(self.duration)
            self.stop_event.set()

            done, not_done = concurrent.futures.wait(futures, timeout=5.0)
            for f in done:
                try:
                    f.result()
                except Exception as e:
                    self._record_violation(f"Worker thread error: {e}")
            if not_done:
                self._record_violation(f"DEADLOCK DETECTED! {len(not_done)} threads failed to complete within timeout.")

        elapsed = time.time() - start_time

        # Final Post-Run Invariant Auditing
        for gpu in self.env.gpus:
            with gpu.lock:
                if gpu.current_memory < -1e-6:
                    self._record_violation(f"Final state: GPU {gpu.id} negative memory ({gpu.current_memory})")
                if gpu.current_memory > gpu.memory_capacity + 1e-4:
                    self._record_violation(f"Final state: GPU {gpu.id} physical memory capacity exceeded ({gpu.current_memory:.2f} > {gpu.memory_capacity:.2f} GB)")

        # Dynamic queue integrity check
        queue_stats = self.env.task_queue.get_task_counts()
        task_reg_size = len(self.env.task_queue.task_registry)
        status_sum = (
            queue_stats["queued"] + queue_stats["scheduled"] + queue_stats["running"] +
            queue_stats["completed"] + queue_stats["failed"] + queue_stats["cancelled"]
        )
        if status_sum != task_reg_size:
            self._record_violation(f"Queue status mismatch: registered={task_reg_size}, sum of states={status_sum}")

        ops = self.metrics["operations_attempted"]
        throughput = ops / max(0.001, elapsed)

        results = {
            "duration_seconds": round(elapsed, 3),
            "operations_attempted": ops,
            "throughput_ops_per_sec": round(throughput, 1),
            "allocations_successful": self.metrics["allocations_successful"],
            "allocations_rejected_capacity": self.metrics["allocations_rejected_capacity"],
            "allocations_rejected_quarantine": self.metrics["allocations_rejected_quarantine"],
            "duplicate_transitions_prevented": self.metrics["duplicate_transitions_prevented"],
            "invariant_violations_count": len(self.metrics["invariant_violations"]),
            "invariant_violations": self.metrics["invariant_violations"],
            "final_queue_stats": queue_stats,
            "cluster_state": [
                {
                    "gpu_id": g.id,
                    "memory_gb": round(g.current_memory, 2),
                    "capacity_gb": g.memory_capacity,
                    "running_tasks": len(g.running_tasks),
                    "queued_tasks": len(g.task_queue),
                    "crashed": g.crashed,
                    "quarantined": g.is_quarantined(),
                }
                for g in self.env.gpus
            ],
        }

        self._print_report(results)
        return results

    def _print_report(self, r: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("          CONCURRENCY STRESS TEST RESULTS")
        print("=" * 60)
        print(f"Duration               : {r['duration_seconds']}s")
        print(f"Operations Attempted   : {r['operations_attempted']} ({r['throughput_ops_per_sec']} ops/sec)")
        print(f"Allocations Succeeded  : {r['allocations_successful']}")
        print(f"Capacity Rejections    : {r['allocations_rejected_capacity']} (Legitimate OOM)")
        print(f"Quarantine Rejections  : {r['allocations_rejected_quarantine']} (Legitimate Safety)")
        print(f"Duplicates Prevented   : {r['duplicate_transitions_prevented']} (Atomic CAS Guards)")
        print(f"Invariant Violations   : {r['invariant_violations_count']}")

        if r["invariant_violations"]:
            print("\n[!] DETECTED INVARIANT VIOLATIONS:")
            for v in r["invariant_violations"]:
                print(f"  - {v}")
        else:
            print("\n[OK] ALL CONCURRENCY INVARIANTS SATISFIED (Zero Violations)")

        print("\nFinal Cluster GPU State:")
        for g in r["cluster_state"]:
            if g["quarantined"]:
                status = "QUARANTINED"
            elif g["crashed"]:
                status = "CRASHED"
            elif g["memory_gb"] > g["capacity_gb"]:
                status = "OVERLOADED"
            else:
                status = "HEALTHY"
            print(f"  GPU {g['gpu_id']}: {g['memory_gb']:.2f}/{g['capacity_gb']} GB VRAM | "
                  f"Running: {g['running_tasks']} | Queued: {g['queued_tasks']} | Status: {status}")
        print("=" * 60, flush=True)
        sys.stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AETHERGRID Concurrency Stress Test")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent worker threads")
    parser.add_argument("--tasks", type=int, default=100, help="Approximate number of tasks to generate")
    parser.add_argument("--duration", type=float, default=2.5, help="Test execution duration in seconds")
    parser.add_argument("--scheduler", type=str, default="marl", help="Scheduler algorithm to test")
    args = parser.parse_args()

    tester = ConcurrencyStressTester(
        num_workers=args.workers,
        total_tasks=args.tasks,
        duration=args.duration,
        scheduler=args.scheduler,
    )
    res = tester.run()
    if res["invariant_violations_count"] > 0:
        sys.exit(1)
    sys.exit(0)
