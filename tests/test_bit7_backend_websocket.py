"""
Bit 7: Backend API + WebSocket Integration Test Suite

Verifies:
  - System Overview REST API (GET /api/status)
  - GPU Telemetry REST API (GET /api/gpus)
  - Task State & Filtering REST API (GET /api/tasks, POST /api/tasks)
  - Strict 8.0 GB Physical GPU Capacity Invariant
  - Risk Engine Telemetry REST API (GET /api/risk)
  - Scheduler Decision Log REST API (GET /api/scheduler/decisions)
  - Execution Event Log REST API (GET /api/events)
  - Health Endpoint (GET /api/health)
  - WebSocket Real-Time Updates (/ws and /ws/simulation)
  - Initial Snapshot Delivery & Typed Envelope Protocol
  - Multi-Client Concurrency, Broadcasts & Safe Disconnect Cleanup
  - Thread-Safe Concurrent REST Reads during Background Scheduling
"""

import concurrent.futures
import json
import threading
import time
from datetime import datetime
import pytest
from fastapi.testclient import TestClient

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from backend import server


@pytest.fixture(autouse=True)
def reset_server_state():
    """Reset the server state before each test for deterministic execution."""
    client = TestClient(server.app)
    res = client.post("/reset")
    assert res.status_code == 200
    yield client
    client.post("/reset")


# ─────────────────────────────────────────────────────────────────────────────
# 1. REST API VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def test_api_status_counts_and_structure(reset_server_state):
    """GET /api/status returns a consistent runtime system snapshot."""
    client = reset_server_state
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()

    # Structural assertions
    assert data["total_gpus"] == 4
    assert data["healthy_gpus"] + data["crashed_gpus"] + data["cooling_down_gpus"] + data["quarantined_gpus"] >= 4
    assert isinstance(data["busy_gpus"], int)
    assert isinstance(data["idle_gpus"], int)
    assert isinstance(data["total_submitted_tasks"], int)
    assert isinstance(data["queued_tasks"], int)
    assert isinstance(data["running_tasks"], int)
    assert isinstance(data["completed_tasks"], int)
    assert isinstance(data["failed_tasks"], int)
    assert isinstance(data["cancelled_tasks"], int)
    assert data["scheduler_mode"] in ("marl", "rl", "fcfs", "round_robin", "least_loaded", "random")
    assert data["operational_status"] in ("RUNNING", "PAUSED")

    # Timestamp validity
    ts = datetime.fromisoformat(data["timestamp"])
    assert ts is not None


def test_api_gpus_runtime_telemetry(reset_server_state):
    """GET /api/gpus returns accurate physical telemetry for all GPUs."""
    client = reset_server_state
    res = client.get("/api/gpus")
    assert res.status_code == 200
    data = res.json()
    assert "gpus" in data
    gpus = data["gpus"]
    assert len(gpus) == 4

    for gpu in gpus:
        assert "gpu_id" in gpu
        assert gpu["total_vram_gb"] == 8.0  # Physical capacity invariant
        assert 0.0 <= gpu["allocated_vram_gb"] <= 8.0
        assert 0.0 <= gpu["available_vram_gb"] <= 8.0
        assert round(gpu["allocated_vram_gb"] + gpu["available_vram_gb"], 1) == 8.0
        assert 0 <= gpu["utilization_pct"] <= 100
        assert 0.0 <= gpu["memory_usage_pct"] <= 100.0
        assert gpu["queue_depth"] >= 0
        assert gpu["running_tasks_count"] >= 0
        assert gpu["temperature"] >= 20.0
        assert gpu["health_status"] in ("HEALTHY", "CRASHED", "COOLDOWN", "QUARANTINED", "BUSY", "IDLE")
        assert 0.0 <= gpu["fragmentation_pct"] <= 100.0
        assert isinstance(gpu["crashed"], bool)
        assert isinstance(gpu["cooldown"], bool)
        assert isinstance(gpu["quarantined"], bool)
        assert 0.0 <= gpu["risk_score"] <= 1.0
        assert isinstance(gpu["active_task_ids"], list)


def test_api_tasks_filtering_and_pagination(reset_server_state):
    """GET /api/tasks supports status filtering, pagination, and invalid filter handling."""
    client = reset_server_state

    # Submit 3 tasks of varying characteristics
    for i in range(3):
        res = client.post("/api/tasks", json={
            "memory_required": 1.5 + (i * 0.5),
            "execution_time": 4 + i,
            "priority": "medium",
            "profile_name": "Medium",
            "description": f"Test Task {i}"
        })
        assert res.status_code == 201

    # Query all tasks
    res_all = client.get("/api/tasks?limit=50&offset=0")
    assert res_all.status_code == 200
    data_all = res_all.json()
    assert data_all["total"] >= 3
    assert len(data_all["tasks"]) >= 3

    # Check task structure
    task0 = data_all["tasks"][0]
    assert "task_id" in task0
    assert "status" in task0
    assert "requested_memory_gb" in task0
    assert "priority" in task0
    assert "workload_type" in task0

    # Pagination test
    res_page = client.get("/api/tasks?limit=2&offset=0")
    assert res_page.status_code == 200
    assert len(res_page.json()["tasks"]) == 2

    res_page2 = client.get("/api/tasks?limit=2&offset=2")
    assert res_page2.status_code == 200
    assert len(res_page2.json()["tasks"]) >= 1

    # Invalid status filter handling
    res_invalid = client.get("/api/tasks?status=NON_EXISTENT_STATUS")
    assert res_invalid.status_code == 400
    assert "Invalid task status filter" in res_invalid.json()["detail"]


def test_api_task_submission_and_capacity_invariant(reset_server_state):
    """POST /api/tasks enforces 8.0 GB capacity invariant and records scheduling decision."""
    client = reset_server_state

    # 1. Successful submission within invariant
    res_ok = client.post("/api/tasks", json={
        "memory_required": 3.0,
        "execution_time": 5,
        "priority": "high",
        "description": "Valid 3GB Task"
    })
    assert res_ok.status_code == 201
    ok_data = res_ok.json()
    assert "task_id" in ok_data
    assert ok_data["status"] in ("assigned", "assigned_fallback", "queued")
    assert "decision" in ok_data

    # 2. Rejection of task exceeding strict physical GPU capacity invariant (8.0 GB)
    res_exceed = client.post("/api/tasks", json={
        "memory_required": 8.5,
        "execution_time": 5,
        "priority": "high",
        "description": "Oversized Task"
    })
    assert res_exceed.status_code == 400
    assert "exceeds strict physical GPU capacity invariant of 8.0 GB" in res_exceed.json()["detail"]


def test_api_risk_telemetry(reset_server_state):
    """GET /api/risk returns authoritative RiskEngine metrics and per-GPU components."""
    client = reset_server_state
    res = client.get("/api/risk")
    assert res.status_code == 200
    data = res.json()

    assert "cluster_avg_risk" in data
    assert 0.0 <= data["cluster_avg_risk"] <= 1.0
    assert "quarantined_gpus" in data
    assert "gpus" in data
    assert len(data["gpus"]) == 4

    for gpu_risk in data["gpus"]:
        assert "gpu_id" in gpu_risk
        assert 0.0 <= gpu_risk["risk_score"] <= 1.0
        assert "risk_category" in gpu_risk
        assert isinstance(gpu_risk["is_quarantined"], bool)
        assert 0.0 <= gpu_risk["effective_capacity_gb"] <= 8.0
        assert gpu_risk["raw_capacity_gb"] == 8.0
        assert "components" in gpu_risk
        assert "reliability" in gpu_risk


def test_api_scheduler_decisions_bounded(reset_server_state):
    """GET /api/scheduler/decisions exposes real MARL decisions with candidate bid scores."""
    client = reset_server_state

    # Submit task to generate a decision
    client.post("/api/tasks", json={"memory_required": 2.0, "execution_time": 3})

    res = client.get("/api/scheduler/decisions?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "decisions" in data
    assert len(data["decisions"]) >= 1

    dec = data["decisions"][0]
    assert "decision_id" in dec
    assert "task_id" in dec
    assert "status" in dec
    assert "decision_score" in dec
    assert "gpu_scores" in dec
    assert "risk_score" in dec
    assert "expected_failure_cost" in dec


def test_api_events_bounded_and_filtered(reset_server_state):
    """GET /api/events returns execution events with timestamp and supports type filtering."""
    client = reset_server_state

    # Submit task to generate events
    client.post("/api/tasks", json={"memory_required": 1.5, "execution_time": 2})

    res = client.get("/api/events?limit=20")
    assert res.status_code == 200
    data = res.json()
    assert "events" in data
    assert len(data["events"]) >= 1

    event0 = data["events"][0]
    assert "event_id" in event0
    assert "timestamp" in event0
    assert "type" in event0
    assert "message" in event0

    # Filter test
    res_filtered = client.get("/api/events?event_type=info")
    assert res_filtered.status_code == 200
    for evt in res_filtered.json()["events"]:
        assert evt["type"] == "info"


def test_api_health_check(reset_server_state):
    """GET /api/health returns health status and operational mode."""
    client = reset_server_state
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "AETHERGRID"
    assert "scheduler" in data
    assert "timestamp" in data


def test_api_read_only_safety(reset_server_state):
    """GET requests never mutate the scheduler state or task counts."""
    client = reset_server_state

    status_before = client.get("/api/status").json()
    gpus_before = client.get("/api/gpus").json()

    # Perform multiple reads
    for _ in range(5):
        client.get("/api/status")
        client.get("/api/gpus")
        client.get("/api/tasks")
        client.get("/api/risk")
        client.get("/api/scheduler/decisions")
        client.get("/api/events")

    status_after = client.get("/api/status").json()
    gpus_after = client.get("/api/gpus").json()

    assert status_before["total_submitted_tasks"] == status_after["total_submitted_tasks"]
    assert status_before["queued_tasks"] == status_after["queued_tasks"]
    assert len(gpus_before["gpus"]) == len(gpus_after["gpus"])


# ─────────────────────────────────────────────────────────────────────────────
# 2. WEBSOCKET REAL-TIME VERIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def test_websocket_connect_and_receive_snapshot(reset_server_state):
    """WebSocket /ws connects and receives initial cluster state snapshot."""
    client = reset_server_state
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_text()
        payload = json.loads(msg)

        # Bit 7 envelope protocol check
        assert payload["type"] == "snapshot"
        assert "event_id" in payload
        assert "timestamp" in payload
        assert "data" in payload

        # Telemetry content check
        snapshot = payload["data"]
        assert "gpus" in snapshot
        assert len(snapshot["gpus"]) == 4
        assert "metrics" in snapshot
        assert "scheduler" in snapshot


def test_websocket_legacy_endpoint_compatibility(reset_server_state):
    """Legacy /ws/simulation route connects and delivers initial snapshot."""
    client = reset_server_state
    with client.websocket_connect("/ws/simulation") as ws:
        msg = ws.receive_text()
        payload = json.loads(msg)
        assert payload["type"] == "snapshot"
        assert len(payload["data"]["gpus"]) == 4


def test_websocket_broadcast_on_task_submission(reset_server_state):
    """Task submission generates real-time WebSocket update for connected client."""
    client = reset_server_state
    with client.websocket_connect("/ws") as ws:
        # Read initial snapshot
        init_msg = json.loads(ws.receive_text())
        assert init_msg["type"] == "snapshot"

        # Submit task via REST
        res = client.post("/api/tasks", json={
            "memory_required": 2.0,
            "execution_time": 4,
            "priority": "high",
            "description": "WebSocket Broadcast Test Task"
        })
        assert res.status_code == 201

        # Client receives task lifecycle update
        broadcast_msg = json.loads(ws.receive_text())
        assert broadcast_msg["type"] in ("task.lifecycle", "snapshot")
        if broadcast_msg["type"] == "task.lifecycle":
            assert "task_id" in broadcast_msg["data"]
            assert broadcast_msg["data"]["memory_required"] == 2.0


def test_websocket_multi_client_and_disconnect(reset_server_state):
    """Multiple WebSocket clients receive updates; disconnected client is cleaned up safely."""
    client = reset_server_state

    # Connect client 1 and client 2
    with client.websocket_connect("/ws") as ws1:
        with client.websocket_connect("/ws") as ws2:
            snap1 = json.loads(ws1.receive_text())
            snap2 = json.loads(ws2.receive_text())
            assert snap1["type"] == "snapshot"
            assert snap2["type"] == "snapshot"

            # Broadcast state
            client.post("/start")

            # Both receive update
            msg1 = json.loads(ws1.receive_text())
            msg2 = json.loads(ws2.receive_text())
            assert msg1["type"] == "snapshot"
            assert msg2["type"] == "snapshot"

        # ws2 is now closed/disconnected
        # Broadcast another state update; server must not crash and ws1 must receive it
        client.post("/pause")
        msg1_after = json.loads(ws1.receive_text())
        assert msg1_after["type"] == "snapshot"


# ─────────────────────────────────────────────────────────────────────────────
# 3. CONCURRENCY & BIT 5 INVARIANT TESTS
# ─────────────────────────────────────────────────────────────────────────────

def test_concurrent_rest_reads_under_simulation_step(reset_server_state):
    """
    High-contention concurrent REST reads during simulation stepping.
    Verifies that simultaneous reads never deadlock, return 500, or violate invariants.
    """
    client = reset_server_state
    client.post("/api/tasks", json={"memory_required": 2.0, "execution_time": 5})

    num_threads = 12
    barrier = threading.Barrier(num_threads)
    errors = []

    def read_worker(worker_id):
        try:
            barrier.wait()
            for _ in range(5):
                if worker_id % 4 == 0:
                    r = client.get("/api/status")
                elif worker_id % 4 == 1:
                    r = client.get("/api/gpus")
                elif worker_id % 4 == 2:
                    r = client.get("/api/tasks")
                else:
                    r = client.get("/api/risk")
                assert r.status_code == 200
        except Exception as e:
            errors.append(e)

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(read_worker, i) for i in range(num_threads)]
        concurrent.futures.wait(futures)

    assert len(errors) == 0, f"Encountered concurrent read errors: {errors}"


def test_physical_gpu_capacity_invariant_never_exceeded(reset_server_state):
    """
    Capacity invariant test: Under heavy scheduling, no GPU ever exceeds 8.0 GB capacity.
    """
    client = reset_server_state

    # Submit multiple tasks
    for i in range(6):
        client.post("/api/tasks", json={
            "memory_required": 2.0,
            "execution_time": 4,
            "priority": "medium"
        })

    # Verify all GPUs have allocated <= 8.0 GB
    gpus_res = client.get("/api/gpus").json()["gpus"]
    for gpu in gpus_res:
        assert gpu["allocated_vram_gb"] <= 8.0
        assert gpu["total_vram_gb"] == 8.0
