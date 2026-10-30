import asyncio
import json
import random
from datetime import datetime
from typing import List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# 1. CORS Configuration (Allows frontend on port 3000/3001 to communicate)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Payload Schema
class SimulationConfig(BaseModel):
    traffic_pattern: str = "Heavy"
    scheduler: str = "MARL"
    num_tasks: int = 24
    speed: str = "Medium"

# Global State
sim_state = {
    "running": False,
    "config": SimulationConfig().model_dump(),
    "completed_tasks": 0,
    "pending_tasks": 24,
}

# Connection Manager for WebSockets
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()

# 2. REST API Endpoints (Triggered by Start / Pause / Reset buttons)
@app.post("/api/simulation/start")
async def start_simulation(config: SimulationConfig):
    sim_state["running"] = True
    sim_state["config"] = config.model_dump()
    sim_state["pending_tasks"] = config.num_tasks
    return {"status": "started", "config": sim_state["config"]}

@app.post("/api/simulation/pause")
async def pause_simulation(config: SimulationConfig):
    sim_state["running"] = False
    return {"status": "paused"}

@app.post("/api/simulation/reset")
async def reset_simulation(config: SimulationConfig):
    sim_state["running"] = False
    sim_state["completed_tasks"] = 0
    sim_state["pending_tasks"] = config.num_tasks
    return {"status": "reset"}

# 3. WebSocket Endpoint (Live telemetry updates for the Dashboard)
@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(1)

            if sim_state["running"] and sim_state["pending_tasks"] > 0:
                sim_state["completed_tasks"] += 1
                sim_state["pending_tasks"] -= 1
            elif sim_state["pending_tasks"] == 0:
                sim_state["running"] = False

            is_running = sim_state["running"]
            
            telemetry = {
                "running": is_running,
                "gpus": [
                    {
                        "id": i,
                        "status": "Active" if is_running else "Idle",
                        "memory": random.randint(45, 95) if is_running else 0,
                        "utilization": random.randint(40, 98) if is_running else 0,
                        "temp": random.randint(50, 78) if is_running else 30,
                        "queue": random.randint(0, 4) if is_running else 0,
                        "running": random.randint(1, 3) if is_running else 0,
                        "fragmentation": random.randint(2, 15) if is_running else 0,
                        "current_task": f"Task-{sim_state['completed_tasks'] + i}" if is_running else "Idle"
                    } for i in range(1, 5)
                ],
                "metrics": {
                    "completed_tasks": sim_state["completed_tasks"],
                    "pending_tasks": sim_state["pending_tasks"],
                    "avg_latency": random.randint(12, 38) if is_running else 0,
                    "throughput": random.randint(85, 140) if is_running else 0,
                    "task_success_rate": 99.4 if is_running else 0,
                    "crash_count": 0,
                    "cluster_utilization": random.randint(65, 92) if is_running else 0,
                    "marl_reward": round(random.uniform(0.75, 0.98), 2) if is_running else 0
                },
                "marl_metrics": {
                    "global_reward": round(sim_state["completed_tasks"] * 12.5, 1),
                    "agent_success_rate": "98.4%",
                    "average_epsilon": 0.05,
                    "q_table_size": 1420,
                    "bid_details": {
                        "incoming_task": f"TASK_{random.randint(1000, 9999)}" if is_running else "N/A",
                        "candidate_gpus": "GPU-1, GPU-2, GPU-3" if is_running else "N/A",
                        "winning_gpu": f"GPU-{random.randint(1, 4)}" if is_running else "N/A",
                        "bid_score": str(round(random.uniform(0.81, 0.99), 3)) if is_running else "N/A",
                        "reason_for_selection": "Optimal memory clearance and low queue latency" if is_running else "N/A"
                    }
                },
                "task_stream": [
                    {"id": f"T-{100 + i}", "type": "ResNet-50", "assigned": f"GPU-{random.randint(1, 4)}"}
                    for i in range(3)
                ] if is_running else [],
                "benchmarks": {
                    "Random": 42,
                    "Round Robin": 68,
                    "FCFS": 55,
                    "Least Loaded": 81,
                    "RL": 89,
                    "MARL": 96
                },
                "logs": [
                    {
                        "id": str(random.randint(100, 999)),
                        "time": datetime.now().strftime("%H:%M:%S"),
                        "type": "INFO",
                        "message": f"Scheduled task via {sim_state['config']['scheduler']} algorithm"
                    }
                ] if is_running else []
            }

            await websocket.send_json(telemetry)

    except WebSocketDisconnect:
        manager.disconnect(websocket)