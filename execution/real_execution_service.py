"""
Real Physical GPU Execution Service.
Manages a serialized single-worker queue (max_workers=1) to execute real GPU video processing
jobs sequentially, preventing concurrent VRAM collisions on the physical hardware.
"""

import os
import time
import logging
import threading
import torch
from concurrent.futures import ThreadPoolExecutor, Future
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from enum import Enum

from .real_gpu_executor import RealGPUVideoExecutor

logger = logging.getLogger("aethergrid.real_execution_service")


class RealExecutionStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class RealExecutionService:
    """
    Thread-safe execution service managing real GPU execution workloads.
    Uses a ThreadPoolExecutor with max_workers=1 to strictly serialize access
    to the physical GPU, avoiding out-of-memory errors from concurrent execution.
    """

    def __init__(
        self,
        device: str = "cuda:0",
        default_batch_size: int = 16,
        filter_type: str = "gaussian_blur",
        fallback_to_cpu: bool = False,
    ):
        self._lock = threading.RLock()
        self.device = device
        self.default_batch_size = default_batch_size
        self.filter_type = filter_type
        self.fallback_to_cpu = fallback_to_cpu

        # Initialize the hardware executor
        self.executor = RealGPUVideoExecutor(
            device=device,
            default_batch_size=default_batch_size,
            filter_type=filter_type,
            fallback_to_cpu=fallback_to_cpu,
        )

        # Single worker queue ensures zero VRAM collision across concurrent requests
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="RealGPUWorker")
        self._tasks: Dict[str, Dict[str, Any]] = {}
        self._futures: Dict[str, Future] = {}
        self._is_shutdown = False

    def submit_task(
        self,
        task_id: str,
        input_path: str,
        output_path: Optional[str] = None,
        batch_size: Optional[int] = None,
        filter_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a real GPU video processing task to the serialized execution queue.

        Args:
            task_id: Unique task identifier
            input_path: Path to input video file
            output_path: Path for output video file (optional)
            batch_size: Frame batch size (optional)
            filter_type: Filter operation name (optional)

        Returns:
            Initial task snapshot dictionary
        """
        with self._lock:
            if self._is_shutdown:
                raise RuntimeError("RealExecutionService has been shut down.")

            if task_id in self._tasks:
                raise ValueError(f"Task with ID '{task_id}' already exists.")

            if not os.path.isfile(input_path):
                raise FileNotFoundError(f"Input video file not found: {input_path}")

            now_iso = datetime.now(timezone.utc).isoformat()
            task_record: Dict[str, Any] = {
                "task_id": task_id,
                "input_path": input_path,
                "output_path": output_path,
                "batch_size": batch_size or self.default_batch_size,
                "filter_type": filter_type or self.filter_type,
                "status": RealExecutionStatus.QUEUED.value,
                "progress_percent": 0.0,
                "frames_processed": 0,
                "total_frames": 0,
                "created_at": now_iso,
                "started_at": None,
                "completed_at": None,
                "elapsed_seconds": None,
                "result": None,
                "error": None,
            }
            self._tasks[task_id] = task_record

            # Schedule execution on the single-worker thread pool
            future = self._pool.submit(
                self._run_task,
                task_id=task_id,
                input_path=input_path,
                output_path=output_path,
                batch_size=task_record["batch_size"],
            )
            self._futures[task_id] = future

            return dict(task_record)

    def _update_progress(self, task_id: str, frames_processed: int, total_frames: int, elapsed_sec: float):
        """Thread-safe callback invoked by RealGPUVideoExecutor during processing."""
        with self._lock:
            if task_id in self._tasks:
                rec = self._tasks[task_id]
                rec["frames_processed"] = frames_processed
                rec["total_frames"] = total_frames
                if total_frames > 0:
                    rec["progress_percent"] = round((frames_processed / total_frames) * 100.0, 1)
                rec["elapsed_seconds"] = round(elapsed_sec, 2)

    def _run_task(
        self,
        task_id: str,
        input_path: str,
        output_path: Optional[str],
        batch_size: int,
    ):
        """Worker routine executed sequentially inside the single background thread."""
        with self._lock:
            if task_id not in self._tasks:
                return
            rec = self._tasks[task_id]
            if rec["status"] == RealExecutionStatus.CANCELLED.value:
                return
            rec["status"] = RealExecutionStatus.RUNNING.value
            rec["started_at"] = datetime.now(timezone.utc).isoformat()

        logger.info(f"Starting real GPU execution for task '{task_id}' on {self.executor.device}")

        try:
            result = self.executor.execute(
                input_path=input_path,
                output_path=output_path,
                batch_size=batch_size,
                progress_callback=lambda f_proc, f_tot, elap: self._update_progress(task_id, f_proc, f_tot, elap),
            )

            with self._lock:
                rec["status"] = RealExecutionStatus.COMPLETED.value
                rec["completed_at"] = datetime.now(timezone.utc).isoformat()
                rec["progress_percent"] = 100.0
                rec["frames_processed"] = result.get("frames_processed", rec["frames_processed"])
                rec["total_frames"] = result.get("total_frames", rec["total_frames"])
                rec["elapsed_seconds"] = result.get("elapsed_seconds")
                rec["output_path"] = result.get("output_path")
                rec["result"] = result
            logger.info(f"Real GPU execution for task '{task_id}' completed successfully in {result.get('elapsed_seconds')}s")

        except Exception as e:
            logger.error(f"Real GPU execution failed for task '{task_id}': {e}", exc_info=True)
            with self._lock:
                rec["status"] = RealExecutionStatus.FAILED.value
                rec["completed_at"] = datetime.now(timezone.utc).isoformat()
                rec["error"] = str(e)

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve point-in-time snapshot of a task by ID."""
        with self._lock:
            rec = self._tasks.get(task_id)
            return dict(rec) if rec else None

    def get_all_tasks(
        self,
        limit: int = 50,
        offset: int = 0,
        status_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve paginated list of all execution tasks."""
        with self._lock:
            all_records = list(self._tasks.values())
            # Sort newest first
            all_records.sort(key=lambda t: t.get("created_at") or "", reverse=True)

            if status_filter:
                filter_upper = status_filter.upper()
                all_records = [t for t in all_records if t.get("status") == filter_upper]

            paginated = all_records[offset : offset + limit]
            return [dict(t) for t in paginated]

    def get_queue_status(self) -> Dict[str, Any]:
        """Return summary statistics of the real GPU execution queue."""
        with self._lock:
            counts = {
                "total": len(self._tasks),
                "queued": 0,
                "running": 0,
                "completed": 0,
                "failed": 0,
                "cancelled": 0,
            }
            for t in self._tasks.values():
                st = t.get("status", "").lower()
                if st in counts:
                    counts[st] += 1

            return {
                "device": str(self.executor.device),
                "cuda_available": torch.cuda.is_available(),
                "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
                "counts": counts,
                "is_shutdown": self._is_shutdown,
            }

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued task before it starts executing."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return False
            if rec["status"] == RealExecutionStatus.QUEUED.value:
                fut = self._futures.get(task_id)
                if fut and not fut.running():
                    fut.cancel()
                rec["status"] = RealExecutionStatus.CANCELLED.value
                rec["completed_at"] = datetime.now(timezone.utc).isoformat()
                return True
            return False

    def shutdown(self, wait: bool = True):
        """Shut down the execution service and worker thread pool."""
        with self._lock:
            self._is_shutdown = True
        self._pool.shutdown(wait=wait)
        logger.info("RealExecutionService shut down.")
