"""
Unit and integration tests for Real Physical GPU Video Execution Pipeline.
Verifies RealGPUVideoExecutor, RealExecutionService, and backend server endpoints.
"""

import os
import time
import tempfile
import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from execution.real_gpu_executor import RealGPUVideoExecutor
from execution.real_execution_service import RealExecutionService, RealExecutionStatus


def _create_sample_video(path: str, num_frames: int = 30, width: int = 320, height: int = 240, fps: float = 24.0):
    """Helper to generate a playable test video file."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    for i in range(num_frames):
        # Generate distinct test frames
        frame = np.full((height, width, 3), (i * 8) % 256, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def test_real_gpu_executor_initialization():
    """Verify executor parameter validation and device initialization."""
    # Invalid batch size
    with pytest.raises(ValueError):
        RealGPUVideoExecutor(default_batch_size=0)

    with pytest.raises(ValueError):
        RealGPUVideoExecutor(default_batch_size=-5)

    # Valid initialization
    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"
    executor = RealGPUVideoExecutor(device=device_to_use, default_batch_size=8, filter_type="gaussian_blur")
    assert executor.default_batch_size == 8
    assert executor.device.type in ("cuda", "cpu")


def test_real_gpu_executor_filters_and_execution():
    """Verify end-to-end video decoding, tensor transformation on GPU, and re-encoding."""
    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"
    executor = RealGPUVideoExecutor(device=device_to_use, default_batch_size=8, filter_type="gaussian_blur")

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
        in_path = f_in.name
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_out:
        out_path = f_out.name

    try:
        _create_sample_video(in_path, num_frames=24, width=160, height=120, fps=24.0)

        progress_calls = []

        def on_progress(p, total, elapsed):
            progress_calls.append((p, total))

        res = executor.execute(in_path, out_path, batch_size=8, progress_callback=on_progress)

        assert res["status"] == "COMPLETED"
        assert res["frames_processed"] == 24
        assert res["total_frames"] == 24
        assert res["width"] == 160
        assert res["height"] == 120
        assert os.path.isfile(out_path)
        assert os.path.getsize(out_path) > 0
        assert len(progress_calls) > 0

    finally:
        if os.path.exists(in_path):
            os.unlink(in_path)
        if os.path.exists(out_path):
            os.unlink(out_path)


def test_real_execution_service_sequential_queue():
    """Verify that RealExecutionService serializes tasks sequentially using max_workers=1."""
    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"
    service = RealExecutionService(device=device_to_use, default_batch_size=8)

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in1:
        in_path1 = f_in1.name
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in2:
        in_path2 = f_in2.name

    try:
        _create_sample_video(in_path1, num_frames=20, width=160, height=120)
        _create_sample_video(in_path2, num_frames=20, width=160, height=120)

        t1 = service.submit_task("task_seq_1", in_path1)
        t2 = service.submit_task("task_seq_2", in_path2)

        assert t1["status"] == RealExecutionStatus.QUEUED.value
        assert t2["status"] == RealExecutionStatus.QUEUED.value

        # Poll until both complete
        for _ in range(60):
            st1 = service.get_task("task_seq_1")
            st2 = service.get_task("task_seq_2")
            if st1["status"] in ("COMPLETED", "FAILED") and st2["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.1)

        final1 = service.get_task("task_seq_1")
        final2 = service.get_task("task_seq_2")

        assert final1["status"] == RealExecutionStatus.COMPLETED.value
        assert final2["status"] == RealExecutionStatus.COMPLETED.value
        assert final1["frames_processed"] == 20
        assert final2["frames_processed"] == 20

        # Verify query APIs
        all_tasks = service.get_all_tasks(limit=10)
        task_ids = [t["task_id"] for t in all_tasks]
        assert "task_seq_1" in task_ids
        assert "task_seq_2" in task_ids

        q_status = service.get_queue_status()
        assert q_status["counts"]["completed"] >= 2

        # Clean up output files
        if final1.get("output_path") and os.path.exists(final1["output_path"]):
            os.unlink(final1["output_path"])
        if final2.get("output_path") and os.path.exists(final2["output_path"]):
            os.unlink(final2["output_path"])

    finally:
        service.shutdown(wait=True)
        if os.path.exists(in_path1):
            os.unlink(in_path1)
        if os.path.exists(in_path2):
            os.unlink(in_path2)


def test_real_execution_service_error_handling():
    """Verify error reporting for non-existent files or invalid parameters."""
    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"
    service = RealExecutionService(device=device_to_use)

    try:
        with pytest.raises(FileNotFoundError):
            service.submit_task("nonexistent_task", "/path/to/definitely_missing_video_file.mp4")

        # Duplicate ID rejection
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
            in_path = f_in.name
        _create_sample_video(in_path, num_frames=5, width=160, height=120)

        t1 = service.submit_task("dup_id_test", in_path)
        with pytest.raises(ValueError):
            service.submit_task("dup_id_test", in_path)

        for _ in range(30):
            st = service.get_task("dup_id_test")
            if st["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.05)

        fin = service.get_task("dup_id_test")
        if fin.get("output_path") and os.path.exists(fin["output_path"]):
            os.unlink(fin["output_path"])
        if os.path.exists(in_path):
            os.unlink(in_path)

    finally:
        service.shutdown(wait=True)


def test_backend_endpoints_in_simulation_mode():
    """Verify backend endpoints in default simulation mode."""
    from backend.server import app

    client = TestClient(app)

    # Status endpoint in default mode
    res = client.get("/real-execution/status")
    assert res.status_code == 200
    data = res.json()
    assert "enabled" in data
    assert "mode" in data

    # Tasks endpoint
    res_tasks = client.get("/real-execution/tasks")
    assert res_tasks.status_code == 200
    tasks_data = res_tasks.json()
    assert "tasks" in tasks_data


def test_backend_endpoints_in_real_gpu_mode(monkeypatch):
    """Verify backend endpoints and task submission when real_execution_service is active."""
    from backend import server

    # Create temporary service for test
    device_to_use = "cuda:0" if torch.cuda.is_available() else "cpu"
    test_service = RealExecutionService(device=device_to_use, default_batch_size=4)
    monkeypatch.setattr(server, "real_execution_service", test_service)
    monkeypatch.setattr(server, "AETHERGRID_EXECUTION_MODE", "real_gpu")

    client = TestClient(server.app)

    try:
        # 1. Status endpoint
        res = client.get("/real-execution/status")
        assert res.status_code == 200
        st = res.json()
        assert st["enabled"] is True
        assert st["mode"] == "real_gpu"
        assert "counts" in st

        # 2. Submit task via endpoint
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f_in:
            in_path = f_in.name
        _create_sample_video(in_path, num_frames=12, width=160, height=120)

        sub_res = client.post(
            "/real-execution/submit",
            json={"input_path": in_path, "batch_size": 4, "filter_type": "gaussian_blur"},
        )
        assert sub_res.status_code == 200
        sub_data = sub_res.json()
        task_id = sub_data["task_id"]
        assert sub_data["status"] == "submitted"

        # 3. Query specific task
        for _ in range(40):
            t_res = client.get(f"/real-execution/tasks/{task_id}")
            assert t_res.status_code == 200
            t_data = t_res.json()
            if t_data["task"]["status"] in ("COMPLETED", "FAILED"):
                break
            time.sleep(0.05)

        final_res = client.get(f"/real-execution/tasks/{task_id}")
        assert final_res.status_code == 200
        final_data = final_res.json()
        assert final_data["task"]["status"] == "COMPLETED"
        assert final_data["task"]["frames_processed"] == 12

        # 4. Query all tasks
        all_res = client.get("/real-execution/tasks")
        assert all_res.status_code == 200
        assert all_res.json()["total"] >= 1

        # Clean output
        out_p = final_data["task"].get("output_path")
        if out_p and os.path.exists(out_p):
            os.unlink(out_p)
        if os.path.exists(in_path):
            os.unlink(in_path)

    finally:
        test_service.shutdown(wait=True)
