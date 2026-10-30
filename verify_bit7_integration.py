"""
End-to-End Verification Script for Bit 7: Backend API + WebSocket Integration
Exercises live API calls and WebSocket streaming against FastAPI backend.
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from fastapi.testclient import TestClient
sys.path.append(os.path.dirname(__file__))
from backend import server


def run_e2e_verification():
    print("=" * 60)
    print("AETHERGRID BIT 7: END-TO-END VERIFICATION")
    print("=" * 60)

    client = TestClient(server.app)

    # 1. Reset state
    print("\n[Step 1] Initializing & Resetting cluster state...")
    r = client.post("/reset")
    assert r.status_code == 200, f"Reset failed: {r.text}"
    print("  -> State reset successfully.")

    # 2. Check Health
    print("\n[Step 2] Testing GET /api/health...")
    r = client.get("/api/health")
    assert r.status_code == 200
    h_data = r.json()
    print(f"  -> Service: {h_data['service']}, Status: {h_data['status']}, Scheduler: {h_data['scheduler']}")

    # 3. Check System Overview
    print("\n[Step 3] Testing GET /api/status...")
    r = client.get("/api/status")
    assert r.status_code == 200
    status = r.json()
    print(f"  -> Total GPUs: {status['total_gpus']}, Healthy: {status['healthy_gpus']}, Idle: {status['idle_gpus']}")
    print(f"  -> Operational Status: {status['operational_status']}, Mode: {status['scheduler_mode']}")

    # 4. Check GPU Telemetry
    print("\n[Step 4] Testing GET /api/gpus...")
    r = client.get("/api/gpus")
    assert r.status_code == 200
    gpus_data = r.json()
    assert len(gpus_data["gpus"]) == 4
    for g in gpus_data["gpus"]:
        print(f"  -> GPU {g['gpu_id']}: Capacity: {g['total_vram_gb']}GB, Allocated: {g['allocated_vram_gb']}GB, Temp: {g['temperature']}C, Health: {g['health_status']}")

    # 5. Check Risk Engine Telemetry
    print("\n[Step 5] Testing GET /api/risk...")
    r = client.get("/api/risk")
    assert r.status_code == 200
    risk = r.json()
    print(f"  -> Cluster Avg Risk: {risk['cluster_avg_risk']}, Quarantined: {risk['quarantined_gpus']}")

    # 6. WebSocket Connection and Snapshot Delivery
    print("\n[Step 6] Testing WebSocket /ws connection and initial snapshot...")
    with client.websocket_connect("/ws") as ws:
        msg = json.loads(ws.receive_text())
        assert msg["type"] == "snapshot"
        assert "data" in msg
        print(f"  -> Received initial snapshot envelope: type={msg['type']}, event_id={msg['event_id']}")
        print(f"  -> Cluster state verified with {len(msg['data']['gpus'])} GPUs.")

        # 7. Programmatic Task Submission via REST with Live WebSocket Broadcast
        print("\n[Step 7] Testing task submission POST /api/tasks & WebSocket broadcast...")
        sub_res = client.post("/api/tasks", json={
            "memory_required": 2.5,
            "execution_time": 5,
            "priority": "high",
            "profile_name": "Medium",
            "description": "E2E Verification Task"
        })
        assert sub_res.status_code == 201
        sub_data = sub_res.json()
        print(f"  -> Task created: {sub_data['task_id']}, Assigned: {sub_data['decision'].get('selected_gpu')}")
        print(f"  -> MARL Bid Decision Score: {sub_data['decision'].get('decision_score')}")

        # Receive WebSocket broadcast
        ws_event = json.loads(ws.receive_text())
        print(f"  -> WebSocket broadcast received: type={ws_event['type']}")
        if ws_event["type"] == "task.lifecycle":
            print(f"     Task {ws_event['data']['task_id']} transitioned to {ws_event['data']['status']}")

    # 8. Physical GPU Capacity Invariant Enforcement
    print("\n[Step 8] Testing physical GPU capacity invariant (8.0 GB)...")
    inv_res = client.post("/api/tasks", json={
        "memory_required": 9.0,
        "execution_time": 5,
        "priority": "high"
    })
    assert inv_res.status_code == 400
    print(f"  -> Correctly rejected oversized task: {inv_res.json()['detail']}")

    # 9. Scheduler Decisions & Execution Events Log
    print("\n[Step 9] Testing GET /api/scheduler/decisions and GET /api/events...")
    dec_res = client.get("/api/scheduler/decisions?limit=5")
    assert dec_res.status_code == 200
    print(f"  -> Recent scheduler decisions count: {dec_res.json()['total']}")

    evt_res = client.get("/api/events?limit=5")
    assert evt_res.status_code == 200
    print(f"  -> Recent execution events count: {evt_res.json()['total']}")

    # 10. Task Queries & Filtering
    print("\n[Step 10] Testing GET /api/tasks with pagination...")
    tasks_res = client.get("/api/tasks?limit=10&offset=0")
    assert tasks_res.status_code == 200
    print(f"  -> Total registered tasks: {tasks_res.json()['total']}")

    print("\n" + "=" * 60)
    print("ALL BIT 7 END-TO-END CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_e2e_verification()
