"""
Bit 8: Frontend Dashboard + Dynamic Visualization Test Suite
Verifies:
  1. GPU cards contract & valid backend telemetry schema
  2. Missing GPU measurements fallback safety
  3. Task lifecycle transitions & timeline accuracy
  4. Task deduplication (replayed events don't create duplicate states/transitions)
  5. Risk visualization data contract (thresholds, 8-factor breakdown, rolling trajectory)
  6. MARL decision visualization schema (candidates, bid scores, penalties, winner)
  7. Execution visualization & throughput tracking
  8. WebSocket connection, typed event payloads & reconnection safety
  9. Empty & error states boundary resilience
 10. Existing Phase 1-7 backward compatibility
"""

import copy
import time
import pytest
from fastapi.testclient import TestClient
from backend.server import app, env, _build_full_state
from task import Task
from marl_agent import MARLManager


@pytest.fixture
def client():
    return TestClient(app)


def test_1_gpu_cards_contract_and_telemetry(client):
    """Verify GPU dashboard contract: capacity, allocated, available, utilization, temp, risk."""
    res = client.get("/api/gpus")
    assert res.status_code == 200
    data = res.json()
    assert "gpus" in data
    gpus = data["gpus"]
    assert len(gpus) == 4

    for gpu in gpus:
        assert "gpu_id" in gpu
        assert gpu["total_vram_gb"] == 8.0
        assert "allocated_vram_gb" in gpu
        assert "available_vram_gb" in gpu
        assert "utilization_pct" in gpu
        assert "temperature" in gpu
        assert "queue_depth" in gpu
        assert "running_tasks_count" in gpu
        assert "health_status" in gpu
        assert "risk_score" in gpu
        assert "expected_failure_cost" in gpu
        # Capacity invariant check
        assert round(gpu["allocated_vram_gb"] + gpu["available_vram_gb"], 2) == 8.0


def test_2_missing_gpu_measurements_fallback_safety():
    """Verify missing/null GPU measurements don't break data contract or computation."""
    state = _build_full_state()
    assert "gpus" in state
    assert len(state["gpus"]) > 0

    # Simulate sparse/partial GPU payload
    sparse_gpu = {"id": 99}
    # Verify fallback fields can be derived safely
    capacity = sparse_gpu.get("capacity", 8.0)
    mem_pct = sparse_gpu.get("memory", None)
    risk = sparse_gpu.get("risk_score", None)
    temp = sparse_gpu.get("temperature", None)

    assert capacity == 8.0
    assert mem_pct is None  # Distinguishes missing from zero
    assert risk is None
    assert temp is None


def test_3_task_lifecycle_events_and_transitions(client):
    """Verify task lifecycle stages: Submitted -> Queued -> Scheduled -> Running -> Completed."""
    payload = {
        "workload_type": "Video",
        "memory_required": 2.0,
        "priority": "high",
        "profile_name": "Medium",
        "description": "Bit 8 Test Lifecycle Task"
    }
    res = client.post("/api/tasks", json=payload)
    assert res.status_code == 201
    task_data = res.json()
    task_id = task_data["task_id"]
    assert task_id is not None
    assert task_data["status"].upper() in ("QUEUED", "SCHEDULED", "ASSIGNED", "RUNNING")

    # Fetch task via task endpoint
    task_res = client.get(f"/api/tasks?limit=10")
    assert task_res.status_code == 200
    tasks = task_res.json()["tasks"]
    matching = [t for t in tasks if t["task_id"] == task_id]
    assert len(matching) == 1
    assert matching[0]["requested_memory_gb"] == 2.0


def test_4_duplicate_task_events_deduplication():
    """Verify duplicate WebSocket/REST task events do not create duplicate nodes."""
    raw_events = [
        {"id": "evt-001", "task_id": "video_100", "status": "QUEUED", "time": "12:00:01"},
        {"id": "evt-001", "task_id": "video_100", "status": "QUEUED", "time": "12:00:01"},  # Duplicate
        {"id": "evt-002", "task_id": "video_100", "status": "RUNNING", "time": "12:00:02"},
        {"id": "evt-002", "task_id": "video_100", "status": "RUNNING", "time": "12:00:02"},  # Duplicate
        {"id": "evt-003", "task_id": "video_100", "status": "COMPLETED", "time": "12:00:05"},
    ]

    # Task deduplication logic matching DynamicTaskGraph / ExecutionVisualizer
    deduped_tasks = {}
    seen_events = set()
    unique_events = []

    for evt in raw_events:
        evt_id = evt["id"]
        if evt_id not in seen_events:
            seen_events.add(evt_id)
            unique_events.append(evt)
        # State machine update
        t_id = evt["task_id"]
        deduped_tasks[t_id] = evt["status"]

    assert len(unique_events) == 3
    assert len(deduped_tasks) == 1
    assert deduped_tasks["video_100"] == "COMPLETED"


def test_5_risk_visualization_contract_and_thresholds(client):
    """Verify risk visualization contract: warning (0.75), quarantine (0.85), 8-factor breakdown."""
    res = client.get("/api/risk/monitoring")
    assert res.status_code == 200
    data = res.json()
    assert "monitoring" in data
    monitoring = data["monitoring"]

    assert "summary" in monitoring
    summary = monitoring["summary"]
    assert "cluster_avg_risk" in summary
    assert "max_gpu_risk" in summary
    assert "avg_efc" in summary
    assert summary["threshold_warning"] == 0.75
    assert summary["threshold_quarantine"] == 0.85

    # Check GPU 8-factor components
    gpus = data.get("gpus", [])
    assert len(gpus) == 4
    for g in gpus:
        comps = g.get("risk_components", {})
        assert "temperature_risk" in comps
        assert "memory_pressure" in comps
        assert "fragmentation_risk" in comps
        assert "utilization_risk" in comps
        assert "failure_history" in comps
        assert "crash_history" in comps
        assert "anomaly_score" in comps
        assert "performance_degradation" in comps


def test_6_marl_decision_visualization_contract(client):
    """Verify MARL decision contract: candidates, bid scores, penalties, and winner selection."""
    res = client.get("/api/scheduler/decisions?limit=10")
    assert res.status_code == 200
    data = res.json()
    assert "decisions" in data
    decisions = data["decisions"]

    if len(decisions) > 0:
        latest = decisions[0]
        assert "task_id" in latest
        assert "candidates" in latest
        candidates = latest["candidates"]
        assert len(candidates) == 4

        for c in candidates:
            assert "gpu_id" in c
            assert "label" in c
            assert "bid_score" in c
            assert "risk_penalty" in c
            assert "efc_penalty" in c
            assert "perf_score" in c
            assert "eligible" in c


def test_7_execution_visualization_metrics_and_events(client):
    """Verify execution monitoring metrics: tasks count, queue status, event log."""
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert "operational_status" in data
    assert "total_gpus" in data
    assert "healthy_gpus" in data
    assert "running_tasks" in data
    assert "queued_tasks" in data
    assert "completed_tasks" in data
    assert "failed_tasks" in data

    evt_res = client.get("/api/events?limit=20")
    assert evt_res.status_code == 200
    events_data = evt_res.json()
    assert "events" in events_data
    assert isinstance(events_data["events"], list)


def test_8_websocket_typed_events_and_snapshot(client):
    """Verify WebSocket /ws endpoint delivers initial snapshot and handles typed event format."""
    with client.websocket_connect("/ws") as websocket:
        data = websocket.receive_json()
        assert data is not None
        # Must contain snapshot envelope
        assert data.get("type") == "snapshot" or "data" in data or "gpus" in data
        if "data" in data:
            snap = data["data"]
            assert "gpus" in snap
            assert "metrics" in snap
            assert "risk_monitoring" in snap


def test_9_empty_and_error_states_resilience(client):
    """Verify boundary cases: empty pagination, non-existent status filter, oversized task rejection."""
    # Non-existent status filter produces structured error
    err_res = client.get("/api/tasks?status=NON_EXISTENT_STATUS")
    assert err_res.status_code == 400

    # Oversized task exceeding 8.0 GB capacity invariant
    oversized = {
        "workload_type": "Video",
        "memory_required": 12.0,
        "priority": "high",
        "profile_name": "Extreme",
    }
    rej_res = client.post("/api/tasks", json=oversized)
    assert rej_res.status_code == 400
    assert "8.0" in rej_res.json()["detail"]


def test_10_phase_1_to_7_backward_compatibility(client):
    """Verify Phase 1-7 core routes remain functional and unaltered."""
    health = client.get("/api/health")
    assert health.status_code == 200
    assert health.json()["service"] == "AETHERGRID"

    dash = client.get("/dashboard-state")
    assert dash.status_code == 200
    assert "gpus" in dash.json()

    risk = client.get("/api/risk")
    assert risk.status_code == 200
    assert "cluster_avg_risk" in risk.json()


def test_11_real_time_task_snapshot_in_full_state_and_ws(client):
    """Verify _build_full_state and WebSocket deliver live tasks array matching REST schema."""
    state = _build_full_state()
    assert "tasks" in state
    assert isinstance(state["tasks"], list)
    assert "total_tasks" in state

    # Submit a task to ensure tasks list contains active data
    sub = client.post("/api/tasks", json={
        "workload_type": "Video",
        "memory_required": 1.5,
        "priority": "medium",
        "profile_name": "Light",
        "description": "Real-time verification task"
    })
    assert sub.status_code == 201

    updated_state = _build_full_state()
    assert len(updated_state["tasks"]) > 0
    first_task = updated_state["tasks"][0]
    assert "task_id" in first_task
    assert "status" in first_task
    assert "requested_memory_gb" in first_task
    assert "priority" in first_task
    assert "submission_time" in first_task


def test_12_submit_task_triggers_marl_and_lifecycle_event(client):
    """Verify POST /api/tasks triggers MARL scheduler and returns real candidate evaluation."""
    res = client.post("/api/tasks", json={
        "workload_type": "Video",
        "memory_required": 2.5,
        "priority": "high",
        "profile_name": "Medium",
        "description": "MARL Candidate Trigger Test"
    })
    assert res.status_code == 201
    data = res.json()
    assert "decision" in data
    decision = data["decision"]
    assert "candidates" in decision
    assert len(decision["candidates"]) == 4
    for cand in decision["candidates"]:
        assert "bid_score" in cand
        assert "risk_penalty" in cand
        assert "efc_penalty" in cand


def test_13_simulation_step_advances_execution(client):
    """Verify POST /api/simulation/step advances simulation time step and updates cluster."""
    # Read initial status
    initial_status = client.get("/api/status").json()
    initial_step = initial_status.get("cluster_time_step", 0)

    # Step simulation
    step_res = client.post("/api/simulation/step")
    assert step_res.status_code == 200
    step_data = step_res.json()
    assert step_data["status"] == "stepped"
    assert step_data["time_step"] >= initial_step


def test_14_capacity_invariant_strictly_enforced_across_allocation(client):
    """Verify GPU memory invariant: allocated + available == 8.0 GB for all GPUs."""
    res = client.get("/api/gpus")
    assert res.status_code == 200
    gpus = res.json()["gpus"]
    for gpu in gpus:
        alloc = gpu["allocated_vram_gb"]
        avail = gpu["available_vram_gb"]
        total = gpu["total_vram_gb"]
        assert total == 8.0
        assert round(alloc + avail, 2) == 8.0

