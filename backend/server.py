import asyncio
import io
import json
import os
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from environment import GPUEnvironment
from benchmark_schedulers import benchmark_all_schedulers
from workload.task_generator import VideoTask, VideoProfile
from dynamic_queue import TaskStatus


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO ANALYSIS UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

def _classify_profile(width: int, height: int, fps: float, duration: float) -> str:
    """
    Classify a video into Light / Medium / Heavy / Extreme based on actual parameters.

    Thresholds (documented):
      - LIGHT   : height ≤ 720  OR  (height ≤ 1080 and fps ≤ 30 and duration ≤ 30)
      - MEDIUM  : height ≤ 1080 and fps ≤ 60 and duration ≤ 60
      - HEAVY   : height ≤ 1440
      - EXTREME : height > 1440  (i.e. 4K and above)
    """
    if height <= 720:
        return "Light"
    if height <= 1080 and fps <= 30 and duration <= 60:
        return "Medium"
    if height <= 1080 and (fps > 30 or duration > 60):
        return "Heavy"
    if height <= 1440:
        return "Heavy"
    return "Extreme"


def _estimate_resources(width: int, height: int, fps: float, duration: float, frames: int):
    """
    Estimate VRAM (GB) and execution steps from video parameters.

    Formula:
        total_pixels = width × height × frames
        max_pixels   = 3840 × 2160 × 60 × 180  (4K, 60fps, 180s)

    VRAM scales with pixels-per-frame (resolution):
        memory_factor = (width * height) / (3840 * 2160)
        estimated_vram ∈ [1.0, 7.5] GB

    Execution time scales with total pixel count:
        time_factor = total_pixels / max_pixels
        execution_steps ∈ [1, 20]
    """
    MAX_PIXELS = 3840 * 2160 * 60 * 180
    pixels_per_frame = width * height
    total_pixels = pixels_per_frame * frames

    memory_factor = pixels_per_frame / (3840 * 2160)
    estimated_vram = max(1.0, min(7.5, round(memory_factor * 7.5, 2)))

    time_factor = total_pixels / MAX_PIXELS
    execution_steps = max(1, min(20, int(time_factor * 20) + 1))

    return estimated_vram, execution_steps


def analyze_video_opencv(file_bytes: bytes, filename: str) -> dict:
    """Extract video metadata using OpenCV. Falls back gracefully on failure."""
    try:
        import cv2
        import numpy as np

        # Write to temp file so OpenCV can open it
        suffix = os.path.splitext(filename)[-1].lower() or ".mp4"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        try:
            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                raise ValueError("OpenCV could not open video file")

            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            # Guard against broken metadata
            if fps <= 0:
                fps = 30.0
            if frame_count <= 0:
                frame_count = max(1, int(fps * 10))  # assume 10s fallback

            duration = frame_count / fps

            # Resolution label
            if height >= 2160:
                res_label = "4K"
            elif height >= 1440:
                res_label = "1440p"
            elif height >= 1080:
                res_label = "1080p"
            elif height >= 720:
                res_label = "720p"
            else:
                res_label = f"{height}p"

            # Bitrate estimate from file size
            file_size_bytes = len(file_bytes)
            bitrate_bps = (file_size_bytes * 8) / max(1.0, duration)
            bitrate_mbps = round(bitrate_bps / 1_000_000, 2)

            return {
                "filename": filename,
                "file_size_mb": round(file_size_bytes / (1024 * 1024), 2),
                "width": width,
                "height": height,
                "resolution": res_label,
                "fps": round(fps, 2),
                "duration": round(duration, 2),
                "frames": frame_count,
                "codec": "H.264",  # cv2 basic probe; actual codec needs ffprobe
                "bitrate": f"{bitrate_mbps} Mbps",
            }
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    except ImportError:
        # OpenCV not available – produce sensible fallback from file size heuristic
        return _fallback_analysis(file_bytes, filename)
    except Exception as e:
        return _fallback_analysis(file_bytes, filename, error=str(e))


def _fallback_analysis(file_bytes: bytes, filename: str, error: str = None) -> dict:
    """
    When OpenCV is unavailable or fails, generate a plausible analysis
    based on file size so the system still produces a real VideoTask.
    """
    size_mb = len(file_bytes) / (1024 * 1024)

    # Heuristic: assume ~5 Mbps average bitrate → derive duration
    bitrate_mbps = 5.0
    duration = max(5.0, (size_mb * 8) / bitrate_mbps)

    # Default to 1080p/30fps for fallback
    width, height, fps = 1920, 1080, 30.0
    frames = int(fps * duration)

    return {
        "filename": filename,
        "file_size_mb": round(size_mb, 2),
        "width": width,
        "height": height,
        "resolution": "1080p",
        "fps": fps,
        "duration": round(duration, 2),
        "frames": frames,
        "codec": "Unknown",
        "bitrate": f"{bitrate_mbps} Mbps",
        "fallback": True,
        "fallback_reason": error or "OpenCV not available",
    }


def build_video_task_from_meta(meta: dict, task_id: str) -> VideoTask:
    """
    Create a deterministic VideoTask from extracted video metadata.
    No random noise – same video always produces same workload.
    """
    width      = meta["width"]
    height     = meta["height"]
    fps        = meta["fps"]
    duration   = meta["duration"]
    frames     = meta["frames"]
    resolution = meta["resolution"]

    estimated_vram, execution_steps = _estimate_resources(width, height, fps, duration, frames)
    profile_name = _classify_profile(width, height, fps, duration)

    SEVERITY_MAP = {"Light": "LOW", "Medium": "MEDIUM", "Heavy": "HIGH", "Extreme": "VERY HIGH"}
    workload_severity = SEVERITY_MAP.get(profile_name, "MEDIUM")

    task = VideoTask.__new__(VideoTask)
    # Manually set all fields (bypass __init__ to avoid random noise)
    task.task_type         = "Video"
    task.profile_name      = profile_name
    task.resolution        = resolution
    task.width             = width
    task.height            = height
    task.fps               = fps
    task.duration          = duration
    task.frames            = frames
    task.codec             = meta.get("codec", "H.264")
    task.bitrate           = meta.get("bitrate", "N/A")
    task.workload_severity = workload_severity
    task.memory_required   = estimated_vram
    task.execution_time    = execution_steps
    task.description       = f"{profile_name} Video ({resolution})"
    task.priority          = "medium"
    task.execution_progress= 0
    task.preempted         = False
    task.preemption_count  = 0
    task.original_execution_time = None
    task.id                = task_id
    task.created_time      = None
    task._set_performance_characteristics()

    return task


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STATE & CONNECTION MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

class ConnectionManager:
    """Thread-safe WebSocket manager with timeout protection and graceful cleanup."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self._lock = threading.RLock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self._lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        payload = json.dumps(message, default=str)
        with self._lock:
            targets = list(self.active_connections)
        if not targets:
            return

        async def _safe_send(ws: WebSocket):
            try:
                await asyncio.wait_for(ws.send_text(payload), timeout=2.0)
                return None
            except Exception:
                return ws

        results = await asyncio.gather(*[_safe_send(conn) for conn in targets], return_exceptions=True)
        for res in results:
            if isinstance(res, WebSocket):
                self.disconnect(res)

    async def close_all(self):
        with self._lock:
            targets = list(self.active_connections)
            self.active_connections.clear()
        for conn in targets:
            try:
                await conn.close(code=1000)
            except Exception:
                pass


manager = ConnectionManager()
env = GPUEnvironment(scheduler="marl")
simulation_running = False
benchmark_results = None
benchmark_lock = asyncio.Lock()
server_lock = threading.RLock()
last_selected_scheduler = "marl"

# Bounded histories for Bit 7 REST & WebSocket observability
scheduler_decisions: deque = deque(maxlen=100)
execution_events: deque = deque(maxlen=200)
risk_history: deque = deque(maxlen=120)
last_handled_decision_counter: int = 0

# Video queue: list of VideoTask objects waiting to be assigned
video_queue: List[VideoTask] = []
# Counter for task IDs
video_task_counter = 0
# Most recent MARL decision metadata
last_marl_decision: dict = {}
# Most recent incoming video metadata (for frontend display)
last_video_meta: dict = {}
# Uploaded video bytes (for HTML5 preview – stored temporarily)
uploaded_video_cache: dict = {}   # task_id -> bytes

_sim_task: Optional[asyncio.Task] = None

# Real Physical GPU Execution Pipeline Mode (Additive Feature)
AETHERGRID_EXECUTION_MODE = os.environ.get("AETHERGRID_EXECUTION_MODE", "simulation").strip().lower()
real_execution_service = None

if AETHERGRID_EXECUTION_MODE == "real_gpu":
    try:
        from execution.real_execution_service import RealExecutionService
        real_execution_service = RealExecutionService(device="cuda:0")
        print("[AETHERGRID] Real physical GPU execution pipeline initialized on cuda:0")
    except Exception as _exec_init_err:
        print(f"[AETHERGRID] Warning: Real physical GPU execution failed to initialize: {_exec_init_err}")
        real_execution_service = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS & EVENT TRACKING
# ─────────────────────────────────────────────────────────────────────────────

def _record_decision(decision: dict) -> dict:
    """Record scheduling decision into bounded history with timestamp and ID."""
    record = {
        "decision_id": f"dec-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "scheduler": env.scheduler,
        **decision,
    }
    with server_lock:
        scheduler_decisions.append(record)
    return record


def _record_env_event(
    message: str,
    event_type: str = "info",
    task_id: Optional[str] = None,
    gpu_id: Optional[int] = None,
    details: Optional[dict] = None
) -> dict:
    """Record an execution event with unique ID, timestamp, and structured details."""
    evt = {
        "event_id": f"evt-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": event_type,
        "message": message,
        "task_id": task_id,
        "gpu_id": gpu_id,
        "details": details or {},
    }
    with server_lock:
        execution_events.append(evt)
    env._record_event(message, event_type if event_type in ("info", "warning", "error", "success") else "info")
    return evt


async def broadcast_dashboard_state():
    """Broadcast current cluster state snapshot to all connected WebSocket clients."""
    state = _build_full_state()
    envelope = {
        "type": "snapshot",
        "event_id": f"snap-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": state,
        **state,  # Top-level backwards compatibility for existing frontend
    }
    await manager.broadcast(envelope)


async def broadcast_event(evt: dict):
    """Broadcast an execution event to all connected WebSocket clients."""
    envelope = {
        "type": "event.created",
        "event_id": evt.get("event_id", f"evt-{uuid.uuid4().hex[:8]}"),
        "timestamp": evt.get("timestamp", datetime.now(timezone.utc).isoformat()),
        "data": evt,
    }
    await manager.broadcast(envelope)


async def broadcast_typed_event(event_type: str, data: dict):
    """Broadcast a typed event envelope to all connected WebSocket clients."""
    envelope = {
        "type": event_type,
        "event_id": f"evt-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": data,
    }
    await manager.broadcast(envelope)


def _serialize_task_list(limit: int = 50, status_filter: Optional[str] = None, offset: int = 0) -> tuple:
    """Return total count and formatted list of tasks from environment task queue."""
    tasks_output = []
    with env.task_queue._lock:
        all_tasks = env.task_queue.get_all_tasks()
        sorted_items = sorted(
            all_tasks.items(),
            key=lambda item: item[1][1].arrival_time if (item[1] and item[1][1]) else 0,
            reverse=True,
        )

        filtered = []
        for task_id, (task, meta) in sorted_items:
            task_status_str = meta.status.value if (meta and hasattr(meta, "status")) else "UNKNOWN"
            if status_filter and task_status_str != status_filter:
                continue
            filtered.append((task_id, task, meta, task_status_str))

        total_count = len(filtered)
        paginated = filtered[offset : offset + limit]

        now = time.time()
        for task_id, task, meta, task_status_str in paginated:
            start_t = meta.start_time if meta else None
            comp_t = meta.completion_time if meta else None
            duration = None
            if start_t and comp_t:
                duration = round(comp_t - start_t, 3)
            elif start_t:
                duration = round(now - start_t, 3)

            sub_iso = (
                datetime.fromtimestamp(meta.arrival_time, timezone.utc).isoformat()
                if (meta and meta.arrival_time)
                else None
            )
            start_iso = (
                datetime.fromtimestamp(meta.start_time, timezone.utc).isoformat()
                if (meta and meta.start_time)
                else None
            )
            comp_iso = (
                datetime.fromtimestamp(meta.completion_time, timezone.utc).isoformat()
                if (meta and meta.completion_time)
                else None
            )

            tasks_output.append({
                "task_id": task_id,
                "status": task_status_str,
                "requested_memory_gb": getattr(task, "memory_required", 0.0),
                "priority": getattr(task, "priority", "medium"),
                "workload_type": getattr(task, "task_type", "Video"),
                "profile_name": getattr(task, "profile_name", "Medium"),
                "description": getattr(task, "description", ""),
                "submission_time": sub_iso,
                "start_time": start_iso,
                "completion_time": comp_iso,
                "duration": duration,
                "assigned_gpu": meta.assigned_gpu if meta else None,
                "retry_count": meta.retry_count if meta else 0,
                "failure_reason": getattr(meta, "failure_reason", None) if hasattr(meta, "failure_reason") else None,
            })

    return total_count, tasks_output


def _build_full_state() -> dict:
    """Combine environment dashboard state with video-pipeline-specific fields and Bit 7 metrics."""
    base = env.get_dashboard_state()

    with server_lock:
        serialised_queue = []
        for t in list(video_queue):
            serialised_queue.append({
                "id": t.id,
                "resolution": getattr(t, "resolution", "1080p"),
                "fps": getattr(t, "fps", 30.0),
                "duration": getattr(t, "duration", 5.0),
                "workload_severity": getattr(t, "workload_severity", "MEDIUM"),
                "memory_required": getattr(t, "memory_required", 2.0),
            })
        marl_decision = dict(last_marl_decision)
        video_meta = dict(last_video_meta)
        recent_decisions = list(scheduler_decisions)[-20:]
        recent_events = list(execution_events)[-30:]
        recent_risk_history = list(risk_history)[-60:]

        # Compute summary metrics for Real-Time Risk & Scheduling Monitor
        total_risk = sum(g.get_risk_state().risk_score if g.get_risk_state() else 0.0 for g in env.gpus)
        avg_risk = round(total_risk / max(1, len(env.gpus)), 4)

        max_risk_gpu = max(env.gpus, key=lambda g: g.get_risk_state().risk_score if g.get_risk_state() else 0.0)
        max_risk_score = round(max_risk_gpu.get_risk_state().risk_score if max_risk_gpu.get_risk_state() else 0.0, 4)
        max_risk_id = max_risk_gpu.id

        total_efc = sum(env.marl_manager.calculate_expected_failure_cost(g) for g in env.gpus)
        avg_efc = round(total_efc / max(1, len(env.gpus)), 4)

        risk_affected_decisions = sum(
            1 for d in scheduler_decisions
            if (d.get("risk_score", 0.0) > 0.0 or d.get("expected_failure_cost", 0.0) > 0.0 or
                any(c.get("risk_penalty", 0.0) > 0 or c.get("efc_penalty", 0.0) > 0 for c in d.get("candidates", [])))
        )

        quarantined_count = sum(1 for g in env.gpus if (g.get_risk_state() and g.get_risk_state().is_quarantined) or (hasattr(g, 'is_quarantined') and g.is_quarantined()))
        cooldown_count = sum(1 for g in env.gpus if g.is_in_cooldown())

        completed_count = sum(g.completed_tasks for g in env.gpus)
        failed_count = getattr(env, 'failed_assignments', 0)
        total_registered = env.total_tasks if hasattr(env, 'total_tasks') and env.total_tasks > 0 else (completed_count + failed_count)
        success_rate = round((completed_count / max(1, total_registered)) * 100.0, 1)

        risk_monitoring = {
            "summary": {
                "cluster_avg_risk": avg_risk,
                "max_gpu_risk": max_risk_score,
                "max_risk_gpu_id": max_risk_id,
                "avg_efc": avg_efc,
                "risk_affected_decisions": risk_affected_decisions,
                "quarantined_count": quarantined_count,
                "cooldown_count": cooldown_count,
                "scheduling_success_rate": success_rate,
                "threshold_warning": 0.75,
                "threshold_quarantine": 0.85,
            },
            "history": recent_risk_history,
            "latest_decision": marl_decision,
            "recent_decisions": recent_decisions,
        }

    # Populate real tasks list for real-time visualization
    total_tasks_count, tasks_list = _serialize_task_list(limit=50)

    base["tasks"] = tasks_list
    base["total_tasks"] = total_tasks_count
    base["video_queue"] = serialised_queue
    base["last_marl_decision"] = marl_decision
    base["last_video_meta"] = video_meta
    base["scheduler_decisions"] = recent_decisions
    base["structured_events"] = recent_events
    base["risk_monitoring"] = risk_monitoring
    base["risk_history"] = recent_risk_history
    base["running"] = simulation_running
    base["scheduler"] = env.scheduler
    base["timestamp"] = datetime.now(timezone.utc).isoformat()
    return base


def _next_task_id() -> str:
    global video_task_counter
    with server_lock:
        video_task_counter += 1
        return f"video_{video_task_counter:04d}"


def _compute_gpu_scores(task: VideoTask) -> List[float]:
    """Return per-GPU bid scores from the MARL manager (for UI display)."""
    cluster_state = env.marl_manager.get_cluster_state(env.gpus)
    cluster_state["current_task_memory"] = task.memory_required
    cluster_state["current_task_time"] = task.execution_time

    scores = []
    for i, (gpu, agent) in enumerate(zip(env.gpus, env.marl_manager.agents)):
        score = env.marl_manager.compute_bid_score(gpu, task, agent, cluster_state)
        scores.append(float(score) if not (score != score) else -99.0)  # NaN guard
    return scores


def _assign_task_to_env(task: VideoTask) -> dict:
    """
    Run the MARL scheduling loop for one task:
    - Update risk telemetry for all candidate GPUs
    - Compute full candidate bid score breakdown
    - MARL selects best GPU via cooperative bidding
    - Assign task atomically
    - Record decision evidence and metadata
    Returns the decision dict.
    """
    env._register_task(task)

    for g in env.gpus:
        g.update_risk()

    cluster_state = env.marl_manager.get_cluster_state(env.gpus)
    cluster_state["current_task_memory"] = task.memory_required
    cluster_state["current_task_time"] = task.execution_time

    # Calculate rich candidate breakdowns
    candidate_breakdowns = [
        env.marl_manager.compute_bid_score_breakdown(gpu, task, agent, cluster_state)
        for gpu, agent in zip(env.gpus, env.marl_manager.agents)
    ]
    gpu_scores = {b["label"]: b["bid_score"] for b in candidate_breakdowns}

    # Select via MARL with rich breakdown
    selected_gpu_id, _ = env.marl_manager.select_gpu_for_task(
        task, env.gpus, cluster_state, return_infeasible_fallback=False, return_breakdown=True
    )

    if selected_gpu_id is None:
        decision = {
            "status": "queued",
            "reason": "All GPUs are currently infeasible due to capacity, quarantine, or crash. Task queued.",
            "task_id": task.id,
            "task_memory": task.memory_required,
            "task_time": task.execution_time,
            "selected_gpu": None,
            "selected_gpu_index": None,
            "decision_score": -999.0,
            "gpu_scores": gpu_scores,
            "candidates": candidate_breakdowns,
            "risk_score": 0.0,
            "expected_failure_cost": 0.0,
            "reward": None,
        }
        _record_decision(decision)
        return decision

    selected_gpu = env.gpus[selected_gpu_id]

    # Calculate expected failure cost
    efc = env.marl_manager.calculate_expected_failure_cost(selected_gpu, task)
    risk_state = selected_gpu.get_risk_state() if hasattr(selected_gpu, 'get_risk_state') else None
    risk_score = risk_state.risk_score if risk_state else 0.0

    success = env._try_assign(selected_gpu, task)

    if success:
        reward = env.marl_manager.calculate_reward(selected_gpu, task, True, expected_failure_cost=efc)
        agent = env.marl_manager.agents[selected_gpu_id]
        cs = env.marl_manager.get_cluster_state(env.gpus)
        cs['current_efc'] = efc
        state = agent.get_state(selected_gpu, cs)
        next_cs = env.marl_manager.get_cluster_state(env.gpus)
        next_cs['current_efc'] = efc
        next_state = agent.get_state(selected_gpu, next_cs)
        agent.update(state, 0, reward, next_state)

        score_val = gpu_scores.get(f"GPU {selected_gpu.id}", 0.0)

        decision = {
            "status": "assigned",
            "task_id": task.id,
            "task_memory": task.memory_required,
            "task_time": task.execution_time,
            "selected_gpu": f"GPU {selected_gpu.id}",
            "selected_gpu_index": selected_gpu_id,
            "decision_score": round(score_val, 4),
            "risk_score": round(risk_score, 4),
            "expected_failure_cost": round(efc, 4),
            "reward": round(reward, 4),
            "gpu_scores": gpu_scores,
            "candidates": candidate_breakdowns,
            "reason": f"Assigned to GPU {selected_gpu.id} via risk-aware MARL cooperative bidding",
            "gpu_state_at_decision": {
                "memory_pct": round(selected_gpu.get_memory_usage_percentage(), 1),
                "temperature": selected_gpu.temperature,
                "queue": selected_gpu.get_queue_length(),
                "running": selected_gpu.get_running_tasks_count(),
                "fragmentation": round(selected_gpu.get_fragmentation_percentage(), 1),
                "crashed": selected_gpu.crashed,
                "is_quarantined": selected_gpu.is_quarantined() if hasattr(selected_gpu, 'is_quarantined') else False,
                "risk_score": round(risk_score, 4),
                "expected_failure_cost": round(efc, 4),
                "status": "CRASHED" if selected_gpu.crashed else (
                    "QUARANTINED" if (hasattr(selected_gpu, 'is_quarantined') and selected_gpu.is_quarantined()) else (
                        "BUSY" if selected_gpu.get_total_load() > 2 else "ACTIVE"
                    )
                ),
            },
        }
        _record_decision(decision)
        return decision
    else:
        # Assignment failed – try least loaded fallback
        gpu = min(env.gpus, key=lambda g: g.get_total_load())
        if env._try_assign(gpu, task):
            decision = {
                "status": "assigned_fallback",
                "task_id": task.id,
                "task_memory": task.memory_required,
                "task_time": task.execution_time,
                "selected_gpu": f"GPU {gpu.id}",
                "selected_gpu_index": gpu.id - 1 if isinstance(gpu.id, int) and gpu.id > 0 else 0,
                "decision_score": 0.0,
                "reason": "Primary selection failed; assigned to least-loaded GPU.",
                "gpu_scores": gpu_scores,
                "candidates": candidate_breakdowns,
                "risk_score": round(float(gpu.get_risk_state().risk_score if gpu.get_risk_state() else 0.0), 4),
                "expected_failure_cost": 0.0,
                "reward": None,
            }
            _record_decision(decision)
            return decision

        decision = {
            "status": "queued",
            "task_id": task.id,
            "task_memory": task.memory_required,
            "task_time": task.execution_time,
            "selected_gpu": None,
            "selected_gpu_index": None,
            "decision_score": -999.0,
            "reason": "No GPU could accept this task right now. Task queued.",
            "gpu_scores": gpu_scores,
            "candidates": candidate_breakdowns,
            "risk_score": 0.0,
            "expected_failure_cost": 0.0,
            "reward": None,
        }
        _record_decision(decision)
        return decision


# ─────────────────────────────────────────────────────────────────────────────
# APP
# ─────────────────────────────────────────────────────────────────────────────

def _step_and_drain():
    global last_handled_decision_counter, last_marl_decision
    with server_lock:
        env.step()
        _try_drain_queue()

        # Capture decisions from env.marl_scheduler
        current_counter = getattr(env, "marl_decision_counter", 0)
        if current_counter > last_handled_decision_counter:
            last_handled_decision_counter = current_counter
            dec = getattr(env, "last_marl_decision", {})
            if dec:
                _record_decision(dec)
                last_marl_decision = dict(dec)
                _record_env_event(
                    f"MARL scheduled task {dec.get('task_id')} to {dec.get('selected_gpu')} (score: {dec.get('decision_score')})",
                    "info",
                    task_id=dec.get("task_id"),
                    gpu_id=dec.get("selected_gpu_index", 0) + 1 if dec.get("selected_gpu_index") is not None else None,
                    details=dec
                )

        # Collect risk telemetry point
        total_risk = 0.0
        max_r = 0.0
        max_r_gpu_id = None
        total_efc = 0.0
        quarantined_count = 0
        cooldown_count = 0

        for g in env.gpus:
            rs = g.get_risk_state()
            r_val = float(rs.risk_score) if rs else 0.0
            total_risk += r_val
            if r_val > max_r:
                max_r = r_val
                max_r_gpu_id = g.id
            if (rs and rs.is_quarantined) or (hasattr(g, 'is_quarantined') and g.is_quarantined()):
                quarantined_count += 1
            if g.is_in_cooldown():
                cooldown_count += 1
            efc_val = float(env.marl_manager.calculate_expected_failure_cost(g))
            total_efc += efc_val

        avg_risk = round(total_risk / max(1, len(env.gpus)), 4)
        avg_efc = round(total_efc / max(1, len(env.gpus)), 4)
        now_dt = datetime.now(timezone.utc)

        point = {
            "timestamp": now_dt.isoformat(),
            "time_str": now_dt.strftime("%H:%M:%S"),
            "step": env.time_step,
            "cluster_avg_risk": avg_risk,
            "max_gpu_risk": round(max_r, 4),
            "max_risk_gpu_id": max_r_gpu_id,
            "avg_efc": avg_efc,
            "quarantined_count": quarantined_count,
            "cooldown_count": cooldown_count,
            "threshold_warning": 0.75,
            "threshold_quarantine": 0.85,
        }
        risk_history.append(point)

        # Check threshold crossings
        if len(risk_history) >= 2:
            prev_max = risk_history[-2]["max_gpu_risk"]
            if prev_max < 0.75 <= max_r:
                _record_env_event(
                    f"Risk threshold WARNING crossed: GPU {max_r_gpu_id} risk reached {max_r:.2f} (threshold: 0.75)",
                    "warning",
                    gpu_id=max_r_gpu_id,
                    details={"threshold": 0.75, "risk": max_r, "type": "threshold_warning"}
                )
            if prev_max < 0.85 <= max_r:
                _record_env_event(
                    f"Risk threshold QUARANTINE crossed: GPU {max_r_gpu_id} risk reached {max_r:.2f} (threshold: 0.85)",
                    "error",
                    gpu_id=max_r_gpu_id,
                    details={"threshold": 0.85, "risk": max_r, "type": "threshold_quarantine"}
                )


async def _background_simulation_loop():
    while True:
        try:
            if simulation_running:
                await asyncio.to_thread(_step_and_drain)
                await broadcast_dashboard_state()
            await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            break
        except Exception:
            await asyncio.sleep(0.5)


def _ensure_simulation_loop():
    global _sim_task
    try:
        loop = asyncio.get_running_loop()
        if _sim_task is None or _sim_task.done():
            _sim_task = loop.create_task(_background_simulation_loop())
    except RuntimeError:
        pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    _ensure_simulation_loop()
    yield
    global _sim_task, real_execution_service
    if _sim_task and not _sim_task.done():
        _sim_task.cancel()
        try:
            await _sim_task
        except asyncio.CancelledError:
            pass
    if real_execution_service is not None:
        try:
            real_execution_service.shutdown(wait=False)
        except Exception:
            pass
    await manager.close_all()



app = FastAPI(title="AETHERGRID Video GPU Scheduler API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO UPLOAD & ANALYSIS  (STEP 2 & 3)
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    """
    Accept a video file upload, extract metadata via OpenCV, derive a VideoTask,
    and immediately schedule it via MARL onto one of the 4 simulated GPUs.
    """
    global last_marl_decision, last_video_meta

    # Validate extension
    fname = file.filename or "upload.mp4"
    ext = os.path.splitext(fname)[-1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        return {
            "error": f"Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        }

    # Read bytes
    try:
        file_bytes = await file.read()
    except Exception as e:
        return {"error": f"Failed to read uploaded file: {e}"}

    if len(file_bytes) == 0:
        return {"error": "Uploaded file is empty."}

    # Analyse
    meta = analyze_video_opencv(file_bytes, fname)
    last_video_meta = meta

    # Log
    _record_env_event(f"Video uploaded: {fname}", "info")
    _record_env_event(
        f"Video analyzed: {meta['width']}×{meta['height']}, {meta['fps']} FPS, {meta['duration']}s",
        "info",
    )

    # Build VideoTask
    task_id = _next_task_id()
    try:
        task = build_video_task_from_meta(meta, task_id)
    except Exception as e:
        return {"error": f"Failed to build VideoTask: {e}"}

    _record_env_event(f"Workload classified as {task.workload_severity} ({task.profile_name})", "info")
    _record_env_event(f"Estimated VRAM: {task.memory_required} GB", "info")

    # Schedule via MARL
    _record_env_event("MARL evaluating 4 GPUs…", "info")
    try:
        decision = _assign_task_to_env(task)
    except Exception as e:
        decision = {"status": "error", "reason": str(e), "task_id": task_id}

    with server_lock:
        if decision.get("status") in ("queued", "error"):
            video_queue.append(task)
            _record_env_event(f"Task {task_id} queued: {decision.get('reason', '')}", "warning")
        else:
            _record_env_event(f"MARL selected {decision.get('selected_gpu', '?')}", "success")
            _record_env_event(f"Video {task_id} assigned to {decision.get('selected_gpu', '?')}", "success")

        last_marl_decision = {**decision, "task_meta": meta}

        # Cache video bytes for preview serving
        uploaded_video_cache[task_id] = (file_bytes, ext, fname)

        # Real Physical GPU execution dispatch (when AETHERGRID_EXECUTION_MODE == 'real_gpu')
        real_exec_dispatch = None
        if real_execution_service is not None:
            try:
                temp_upload_dir = os.path.join(tempfile.gettempdir(), "aethergrid_uploads")
                os.makedirs(temp_upload_dir, exist_ok=True)
                temp_in_path = os.path.join(temp_upload_dir, f"{task_id}_{fname}")
                with open(temp_in_path, "wb") as f_in:
                    f_in.write(file_bytes)

                temp_out_dir = os.path.join(tempfile.gettempdir(), "aethergrid_outputs")
                os.makedirs(temp_out_dir, exist_ok=True)
                temp_out_path = os.path.join(temp_out_dir, f"{task_id}_processed_{fname}")

                real_task = real_execution_service.submit_task(
                    task_id=task_id,
                    input_path=temp_in_path,
                    output_path=temp_out_path,
                )
                real_exec_dispatch = {
                    "enabled": True,
                    "task_id": task_id,
                    "status": real_task["status"],
                }
                _record_env_event(f"Task {task_id} queued on physical GPU (cuda:0)", "info", task_id=task_id)
            except Exception as _re_err:
                _record_env_event(f"Real GPU dispatch failed for {task_id}: {_re_err}", "warning", task_id=task_id)
                real_exec_dispatch = {"enabled": True, "error": str(_re_err)}

    # Push real-time state to all WebSocket clients
    await broadcast_dashboard_state()

    response_data = {
        "task_id": task_id,
        "meta": meta,
        "task": {
            "id": task.id,
            "profile_name": task.profile_name,
            "workload_severity": task.workload_severity,
            "memory_required": task.memory_required,
            "execution_steps": task.execution_time,
            "resolution": task.resolution,
            "fps": task.fps,
            "duration": task.duration,
            "frames": task.frames,
            "codec": task.codec,
            "bitrate": task.bitrate,
        },
        "decision": decision,
    }
    if real_exec_dispatch is not None:
        response_data["real_execution"] = real_exec_dispatch
    return response_data


@app.get("/video-preview/{task_id}")
async def video_preview(task_id: str):
    """Serve uploaded video bytes for HTML5 preview in the frontend."""
    with server_lock:
        cached = uploaded_video_cache.get(task_id)
    if cached is None:
        from fastapi.responses import JSONResponse
        return JSONResponse({"error": "Video not found"}, status_code=404)

    file_bytes, ext, fname = cached
    ext_to_mime = {
        ".mp4": "video/mp4",
        ".avi": "video/x-msvideo",
        ".mov": "video/quicktime",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
    }
    mime = ext_to_mime.get(ext, "video/mp4")
    from fastapi.responses import Response
    return Response(content=file_bytes, media_type=mime)


# ─────────────────────────────────────────────────────────────────────────────
# REAL PHYSICAL GPU EXECUTION ENDPOINTS (Additive Feature)
# ─────────────────────────────────────────────────────────────────────────────

class RealExecutionSubmissionRequest(BaseModel):
    input_path: str = Field(..., description="Path to video file on disk")
    output_path: Optional[str] = Field(default=None, description="Path for output processed video")
    batch_size: Optional[int] = Field(default=16, ge=1, le=128, description="CUDA frame batch size")
    filter_type: Optional[str] = Field(default="gaussian_blur", description="Filter operation")


@app.get("/real-execution/status")
@app.get("/api/real-execution/status")
def get_real_execution_status():
    """Return real GPU execution service status, queue statistics, and hardware availability."""
    if real_execution_service is None:
        return {
            "mode": AETHERGRID_EXECUTION_MODE,
            "enabled": False,
            "message": (
                "Real physical GPU execution is disabled (operating in simulation mode). "
                "Set environment variable AETHERGRID_EXECUTION_MODE=real_gpu to enable."
            ),
        }
    return {
        "mode": AETHERGRID_EXECUTION_MODE,
        "enabled": True,
        **real_execution_service.get_queue_status(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/real-execution/tasks")
@app.get("/api/real-execution/tasks")
def get_real_execution_tasks(
    limit: int = Query(50, ge=1, le=200, description="Pagination limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    status: Optional[str] = Query(None, description="Filter by status: QUEUED, RUNNING, COMPLETED, FAILED, CANCELLED"),
):
    """Inspect all tasks in the real GPU execution queue."""
    if real_execution_service is None:
        return {
            "mode": AETHERGRID_EXECUTION_MODE,
            "enabled": False,
            "total": 0,
            "tasks": [],
            "message": "Real physical GPU execution is disabled (operating in simulation mode).",
        }
    tasks = real_execution_service.get_all_tasks(limit=limit, offset=offset, status_filter=status)
    return {
        "mode": AETHERGRID_EXECUTION_MODE,
        "enabled": True,
        "total": len(tasks),
        "limit": limit,
        "offset": offset,
        "tasks": tasks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/real-execution/tasks/{task_id}")
@app.get("/api/real-execution/tasks/{task_id}")
def get_real_execution_task(task_id: str):
    """Inspect state, progress, and output of a specific real GPU execution task."""
    if real_execution_service is None:
        raise HTTPException(
            status_code=503,
            detail="Real physical GPU execution pipeline is disabled (operating in simulation mode). Set AETHERGRID_EXECUTION_MODE=real_gpu to enable.",
        )
    task_info = real_execution_service.get_task(task_id)
    if not task_info:
        raise HTTPException(
            status_code=404,
            detail=f"Real execution task '{task_id}' not found.",
        )
    return {
        "mode": AETHERGRID_EXECUTION_MODE,
        "enabled": True,
        "task": task_info,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/real-execution/submit")
@app.post("/api/real-execution/submit")
def submit_real_execution_task(req: RealExecutionSubmissionRequest):
    """Directly submit a video file path to the real GPU execution queue."""
    if real_execution_service is None:
        raise HTTPException(
            status_code=503,
            detail="Real physical GPU execution pipeline is disabled (operating in simulation mode). Set AETHERGRID_EXECUTION_MODE=real_gpu to enable.",
        )
    task_id = f"real_vid_{uuid.uuid4().hex[:8]}"
    try:
        task_record = real_execution_service.submit_task(
            task_id=task_id,
            input_path=req.input_path,
            output_path=req.output_path,
            batch_size=req.batch_size,
            filter_type=req.filter_type,
        )
        return {
            "status": "submitted",
            "task_id": task_id,
            "task": task_record,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))



# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION CONTROL
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/start")
@app.post("/api/simulation/start")
async def start_simulation():
    global simulation_running
    with server_lock:
        simulation_running = True
        _ensure_simulation_loop()
    await broadcast_dashboard_state()
    return {"status": "started"}


@app.post("/pause")
@app.post("/api/simulation/pause")
async def pause_simulation():
    global simulation_running
    with server_lock:
        simulation_running = False
    await broadcast_dashboard_state()
    return {"status": "paused"}


@app.post("/reset")
@app.post("/api/simulation/reset")
async def reset_simulation():
    global env, simulation_running, last_selected_scheduler, video_queue
    global last_marl_decision, last_video_meta, video_task_counter
    with server_lock:
        env = GPUEnvironment(scheduler=last_selected_scheduler)
        simulation_running = False
        video_queue = []
        last_marl_decision = {}
        last_video_meta = {}
        video_task_counter = 0
    await broadcast_dashboard_state()
    return {"status": "reset"}


@app.post("/step")
@app.post("/api/simulation/step")
async def step_simulation():
    """Advance the simulation by a single step manually and broadcast state."""
    await asyncio.to_thread(_step_and_drain)
    await broadcast_dashboard_state()
    return {"status": "stepped", "time_step": env.time_step}


class SchedulerRequest(BaseModel):
    scheduler: str


@app.post("/set-scheduler")
@app.post("/api/simulation/set-scheduler")
async def set_scheduler(req: SchedulerRequest):
    global last_selected_scheduler
    scheduler = req.scheduler
    allowed = {
        "random", "round_robin", "fcfs", "least_loaded", "rl", "marl",
        "traditional_fcfs", "traditional_round_robin", "traditional_least_loaded",
    }
    if scheduler not in allowed:
        return {"status": "error", "message": "invalid scheduler"}
    with server_lock:
        last_selected_scheduler = scheduler
        env.scheduler = scheduler
    await broadcast_dashboard_state()
    return {"status": "ok", "scheduler": env.scheduler}


class TaskSubmissionRequest(BaseModel):
    memory_required: float = Field(..., gt=0, description="Memory requested in GB (physical GPU limit is 8.0 GB)")
    execution_time: int = Field(default=5, ge=1, le=100, description="Estimated execution steps")
    priority: str = Field(default="medium", description="Task priority: low, medium, high")
    workload_type: str = Field(default="Video", description="Workload type")
    profile_name: str = Field(default="Medium", description="Profile name: Light, Medium, Heavy, Extreme")
    description: Optional[str] = Field(default="Synthetic Task", description="Task description")


@app.get("/dashboard-state")
@app.get("/api/simulation/dashboard-state")
def get_dashboard_state():
    return _build_full_state()


# ─────────────────────────────────────────────────────────────────────────────
# BIT 7: REST APIS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/status")
def get_system_status():
    """Return a consistent system overview snapshot derived from actual runtime state."""
    with server_lock:
        total_gpus = len(env.gpus)
        crashed_gpus = 0
        cooling_down_gpus = 0
        quarantined_gpus = 0
        busy_gpus = 0
        idle_gpus = 0
        healthy_gpus = 0

        for gpu in env.gpus:
            with gpu._lock:
                is_crashed = bool(gpu.crashed)
                cooldown_rem = (
                    gpu.get_cooldown_remaining()
                    if hasattr(gpu, "get_cooldown_remaining")
                    else (gpu.crash_cooldown_timer > 0)
                )
                is_quarantined = (
                    gpu.is_quarantined()
                    if hasattr(gpu, "is_quarantined")
                    else False
                )
                running_cnt = gpu.get_running_tasks_count()
                queue_len = gpu.get_queue_length()

                if is_crashed:
                    crashed_gpus += 1
                elif cooldown_rem:
                    cooling_down_gpus += 1
                elif is_quarantined:
                    quarantined_gpus += 1
                elif running_cnt > 0 or queue_len > 0:
                    busy_gpus += 1
                else:
                    idle_gpus += 1

                if not is_crashed and not cooldown_rem and not is_quarantined:
                    healthy_gpus += 1

        with env.task_queue._lock:
            task_counts = env.task_queue.get_task_counts()

        queued_total = task_counts.get("queued", 0) + len(video_queue)
        running_total = task_counts.get("running", 0) + task_counts.get("scheduled", 0)

        return {
            "total_gpus": total_gpus,
            "healthy_gpus": healthy_gpus,
            "busy_gpus": busy_gpus,
            "idle_gpus": idle_gpus,
            "cooling_down_gpus": cooling_down_gpus,
            "crashed_gpus": crashed_gpus,
            "quarantined_gpus": quarantined_gpus,
            "total_submitted_tasks": task_counts.get("total", 0),
            "queued_tasks": queued_total,
            "running_tasks": running_total,
            "completed_tasks": task_counts.get("completed", 0),
            "failed_tasks": task_counts.get("failed", 0),
            "cancelled_tasks": task_counts.get("cancelled", 0),
            "scheduler_mode": env.scheduler,
            "operational_status": "RUNNING" if simulation_running else "PAUSED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


@app.get("/api/gpus")
def get_gpu_states():
    """Return each GPU's actual current state and health telemetry."""
    gpu_list = []
    with server_lock:
        for gpu in env.gpus:
            with gpu._lock:
                risk_state = gpu.get_risk_state() if hasattr(gpu, "get_risk_state") else None
                running_ids = [getattr(t, "id", str(t)) for t in gpu.running_tasks]
                queued_ids = [getattr(t, "id", str(t)) for t in gpu.task_queue]
                active_task_ids = running_ids + queued_ids

                cooldown_active = bool(
                    gpu.get_cooldown_remaining()
                    if hasattr(gpu, "get_cooldown_remaining")
                    else (gpu.crash_cooldown_timer > 0)
                )
                quarantined = bool(
                    gpu.is_quarantined()
                    if hasattr(gpu, "is_quarantined")
                    else False
                )

                if gpu.crashed:
                    health_status = "CRASHED"
                elif cooldown_active:
                    health_status = "COOLDOWN"
                elif quarantined:
                    health_status = "QUARANTINED"
                elif gpu.get_running_tasks_count() > 0 or gpu.get_queue_length() > 0:
                    health_status = "BUSY"
                else:
                    health_status = "HEALTHY"

                allocated_vram = round(gpu.current_memory, 2)
                available_vram = round(max(0.0, gpu.memory_capacity - gpu.current_memory), 2)
                efc_val = getattr(risk_state, "expected_failure_cost", 0.0) if risk_state else 0.0

                gpu_list.append({
                    "gpu_id": gpu.id,
                    "total_vram_gb": float(gpu.memory_capacity),
                    "allocated_vram_gb": allocated_vram,
                    "available_vram_gb": available_vram,
                    "utilization_pct": float(gpu.utilization),
                    "memory_usage_pct": round(gpu.get_memory_usage_percentage(), 1),
                    "queue_depth": gpu.get_queue_length(),
                    "running_tasks_count": gpu.get_running_tasks_count(),
                    "temperature": float(gpu.temperature),
                    "health_status": health_status,
                    "fragmentation_pct": round(gpu.get_fragmentation_percentage(), 1),
                    "crashed": bool(gpu.crashed),
                    "cooldown": cooldown_active,
                    "quarantined": quarantined,
                    "risk_score": round(float(risk_state.risk_score), 4) if risk_state else 0.0,
                    "risk_category": risk_state.risk_level.value if risk_state else "LOW",
                    "expected_failure_cost": round(float(efc_val), 4),
                    "active_task_ids": active_task_ids,
                    "completed_tasks": gpu.completed_tasks,
                    "total_preemptions": gpu.total_preemptions,
                })

    return {"gpus": gpu_list, "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/tasks")
def get_tasks(
    status: Optional[str] = Query(None, description="Filter by status: QUEUED, SCHEDULED, RUNNING, COMPLETED, FAILED, CANCELLED"),
    limit: int = Query(50, ge=1, le=200, description="Pagination limit"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
):
    """Return task state with filtering and pagination."""
    status_filter = status.upper() if status else None
    valid_statuses = {
        "ARRIVING", "QUEUED", "SCHEDULED", "RUNNING", "COMPLETED",
        "FAILED", "CANCELLED", "WAITING", "RETRY",
    }
    if status_filter and status_filter not in valid_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task status filter '{status}'. Valid options: {sorted(list(valid_statuses))}",
        )

    total_count, tasks_output = _serialize_task_list(limit=limit, status_filter=status_filter, offset=offset)

    return {
        "total": total_count,
        "limit": limit,
        "offset": offset,
        "tasks": tasks_output,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/tasks", status_code=201)
async def submit_task(req: TaskSubmissionRequest):
    """
    Programmatically submit a task.
    Enforces strict 8.0 GB physical GPU capacity invariant.
    """
    if req.memory_required > 8.0:
        raise HTTPException(
            status_code=400,
            detail=f"Requested memory ({req.memory_required:.1f} GB) exceeds strict physical GPU capacity invariant of 8.0 GB",
        )

    task_id = _next_task_id()
    task = VideoTask.__new__(VideoTask)
    task.task_type = req.workload_type
    task.profile_name = req.profile_name
    task.resolution = "1080p"
    task.width = 1920
    task.height = 1080
    task.fps = 30.0
    task.duration = 10.0
    task.frames = 300
    task.codec = "H.264"
    task.bitrate = "5 Mbps"
    if req.memory_required > 4.0:
        task.workload_severity = "HIGH"
    elif req.memory_required > 2.0:
        task.workload_severity = "MEDIUM"
    else:
        task.workload_severity = "LOW"
    task.memory_required = float(req.memory_required)
    task.execution_time = int(req.execution_time)
    task.description = req.description or f"{req.profile_name} Task ({task_id})"
    task.priority = req.priority
    task.execution_progress = 0
    task.preempted = False
    task.preemption_count = 0
    task.original_execution_time = None
    task.id = task_id
    task.created_time = None
    task._set_performance_characteristics()

    _record_env_event(
        f"Task {task.id} submitted via REST API ({task.memory_required:.1f} GB, prio={task.priority})",
        "info",
        task_id=task.id,
    )

    decision = _assign_task_to_env(task)
    with server_lock:
        if decision.get("status") in ("queued", "error"):
            video_queue.append(task)
            _record_env_event(f"Task {task.id} queued: {decision.get('reason', '')}", "warning", task_id=task.id)
        else:
            _record_env_event(f"Task {task.id} assigned to {decision.get('selected_gpu', '?')}", "success", task_id=task.id)
        global last_marl_decision
        last_marl_decision = dict(decision)

    # Publish lifecycle event and updated snapshot to WebSocket clients
    raw_status = decision.get("status", "QUEUED")
    norm_status = "SCHEDULED" if raw_status == "assigned" else ("QUEUED" if raw_status == "queued" else str(raw_status).upper())
    assigned_gpu_val = decision.get("selected_gpu_index", 0) + 1 if decision.get("selected_gpu_index") is not None else decision.get("selected_gpu")

    await manager.broadcast({
        "type": "task.lifecycle",
        "event_id": f"evt-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": {
            "task_id": task.id,
            "status": norm_status,
            "assigned_gpu": assigned_gpu_val,
            "memory_required": task.memory_required,
            "requested_memory_gb": task.memory_required,
            "priority": task.priority,
            "profile_name": task.profile_name,
            "workload_type": task.task_type,
            "description": task.description,
            "submission_time": datetime.now(timezone.utc).isoformat(),
        },
    })
    await broadcast_dashboard_state()

    return {
        "task_id": task_id,
        "status": decision.get("status"),
        "memory_required": task.memory_required,
        "execution_time": task.execution_time,
        "decision": decision,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/risk")
def get_risk_state():
    """Return current risk metrics, components, and quarantine status."""
    gpu_risks = []
    total_risk = 0.0
    total_efc = 0.0
    max_r = 0.0
    max_r_gpu_id = 1
    quarantined_count = 0

    with server_lock:
        for gpu in env.gpus:
            with gpu._lock:
                risk_state = gpu.get_risk_state() if hasattr(gpu, "get_risk_state") else None
                if risk_state:
                    r_score = float(risk_state.risk_score)
                    r_level = risk_state.risk_level.value
                    is_q = bool(risk_state.is_quarantined)
                    q_reason = risk_state.quarantine_reason
                    eff_cap = float(risk_state.effective_capacity)
                    raw_cap = float(risk_state.raw_capacity)
                    components = {
                        "temperature_risk": float(risk_state.components.temperature_risk),
                        "memory_pressure": float(risk_state.components.memory_pressure),
                        "fragmentation_risk": float(risk_state.components.fragmentation_risk),
                        "failure_history": float(risk_state.components.failure_history),
                        "crash_history": float(risk_state.components.crash_history),
                        "anomaly_score": float(risk_state.components.anomaly_score),
                        "performance_degradation": float(risk_state.components.performance_degradation),
                        "utilization_risk": float(risk_state.components.utilization_risk),
                    }
                    hist = risk_state.reliability_history
                    reliability = {
                        "total_tasks": hist.total_tasks,
                        "completed_tasks": hist.completed_tasks,
                        "failed_tasks": hist.failed_tasks,
                        "crash_events": hist.crash_events,
                        "recovery_events": hist.recovery_events,
                        "preemptions": hist.preemptions,
                        "anomalies_detected": hist.anomalies_detected,
                        "failure_rate": round(float(hist.failure_rate), 4),
                    }
                else:
                    r_score = 0.0
                    r_level = "HEALTHY"
                    is_q = False
                    q_reason = None
                    eff_cap = float(gpu.memory_capacity - gpu.current_memory)
                    raw_cap = float(gpu.memory_capacity)
                    components = {}
                    reliability = {}

                efc_val = float(env.marl_manager.calculate_expected_failure_cost(gpu))
                total_risk += r_score
                total_efc += efc_val
                if r_score > max_r:
                    max_r = r_score
                    max_r_gpu_id = gpu.id
                if is_q:
                    quarantined_count += 1

                gpu_risks.append({
                    "gpu_id": gpu.id,
                    "risk_score": round(r_score, 4),
                    "risk_category": r_level,
                    "is_quarantined": is_q,
                    "quarantine_reason": q_reason,
                    "effective_capacity_gb": round(eff_cap, 2),
                    "raw_capacity_gb": round(raw_cap, 2),
                    "temperature": float(gpu.temperature),
                    "crashed": bool(gpu.crashed),
                    "efc": round(efc_val, 4),
                    "components": components,
                    "reliability": reliability,
                })

        avg_risk = round(total_risk / max(1, len(env.gpus)), 4)
        avg_efc = round(total_efc / max(1, len(env.gpus)), 4)
        recent_history = list(risk_history)[-60:]

    return {
        "cluster_avg_risk": avg_risk,
        "max_gpu_risk": round(max_r, 4),
        "max_risk_gpu_id": max_r_gpu_id,
        "avg_efc": avg_efc,
        "quarantined_gpus": quarantined_count,
        "threshold_warning": 0.75,
        "threshold_quarantine": 0.85,
        "gpus": gpu_risks,
        "history": recent_history,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/risk/monitoring")
def get_risk_monitoring():
    """Return complete risk monitoring panel data snapshot."""
    state = _build_full_state()
    return {
        "status": "success",
        "monitoring": state.get("risk_monitoring", {}),
        "gpus": state.get("gpus", []),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/risk/history")
def get_risk_history(limit: int = Query(60, ge=1, le=120)):
    """Return rolling time-series risk telemetry samples."""
    with server_lock:
        recent = list(risk_history)[-limit:]
    return {
        "total": len(recent),
        "history": recent,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/scheduler/decisions")
def get_scheduler_decisions(limit: int = Query(20, ge=1, le=100)):
    """Return recent real scheduler decisions from bounded history."""
    with server_lock:
        recent = list(scheduler_decisions)[-limit:]
    return {
        "total": len(recent),
        "decisions": list(reversed(recent)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/events")
def get_events(
    limit: int = Query(50, ge=1, le=200),
    event_type: Optional[str] = Query(None, description="Filter by event type: info, warning, error, success"),
):
    """Return recent execution events from bounded history."""
    with server_lock:
        all_evts = list(execution_events)
        if event_type:
            all_evts = [e for e in all_evts if e.get("type") == event_type]
        recent = all_evts[-limit:]
    return {
        "total": len(recent),
        "events": list(reversed(recent)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/health")
def get_health():
    """Health and readiness check."""
    return {
        "status": "healthy",
        "service": "AETHERGRID",
        "scheduler": env.scheduler,
        "simulation_running": simulation_running,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# VIDEO QUEUE DRAIN (background step)
# ─────────────────────────────────────────────────────────────────────────────

def _try_drain_queue():
    """Attempt to assign queued videos to now-available GPUs."""
    with server_lock:
        if not video_queue:
            return

        assigned_indices = []
        for i, task in enumerate(video_queue):
            status = env.task_queue.get_task_status(task.id)
            if status is not None and status.value != "QUEUED":
                assigned_indices.append(i)
                continue
            for gpu in env.gpus:
                if gpu.is_available_for_task(task.memory_required):
                    success = env._try_assign(gpu, task)
                    if success:
                        _record_env_event(
                            f"Queued task {task.id} assigned to GPU {gpu.id} from queue", "success", task_id=task.id, gpu_id=gpu.id
                        )
                        assigned_indices.append(i)
                        break

        for i in sorted(assigned_indices, reverse=True):
            if i < len(video_queue):
                video_queue.pop(i)


# ─────────────────────────────────────────────────────────────────────────────
# WEBSOCKET
# ─────────────────────────────────────────────────────────────────────────────

@app.websocket("/ws")
@app.websocket("/ws/simulation")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time cluster telemetry and execution events.
    Supports both standard /ws and legacy /ws/simulation.
    Sends an initial snapshot immediately upon connection.
    """
    await manager.connect(websocket)
    _ensure_simulation_loop()
    try:
        snapshot = _build_full_state()
        envelope = {
            "type": "snapshot",
            "event_id": f"snap-{uuid.uuid4().hex[:8]}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": snapshot,
            **snapshot,
        }
        await websocket.send_text(json.dumps(envelope, default=str))
        while True:
            try:
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception:
        manager.disconnect(websocket)



# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARKS
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/run-benchmarks")
@app.post("/api/simulation/run-benchmarks")
async def run_benchmarks(steps: int = 50, runs: int = 10, seed: int = 42):
    global benchmark_results
    async with benchmark_lock:
        results = await asyncio.to_thread(benchmark_all_schedulers, steps, runs, seed)
        benchmark_results = results
    return {
        "status": "completed",
        "schedulers": [r.get("Scheduler") for r in (benchmark_results or [])],
    }


@app.get("/benchmarks")
@app.get("/api/simulation/benchmarks")
async def get_benchmarks():
    if benchmark_results is None:
        return {
            "available": False,
            "message": "No benchmark results. POST /run-benchmarks to run.",
        }
    return {"available": True, "results": benchmark_results}


# ─────────────────────────────────────────────────────────────────────────────
# BIT 9: RESEARCH METRICS & BENCHMARKING ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

from benchmarking.experiment_config import BenchmarkConfig, AblationConfig, ALL_SCHEDULERS
from benchmarking.benchmark_runner import BenchmarkRunner
from benchmarking.ablation_runner import AblationRunner
from benchmarking.results_store import ResultsStore

_bit9_store = ResultsStore()
_bit9_runner = BenchmarkRunner(results_store=_bit9_store)
_bit9_ablation_runner = AblationRunner(results_store=_bit9_store)
_bit9_lock = asyncio.Lock()


class BenchmarkRunRequest(BaseModel):
    name: str = Field(default="Research Benchmark Run", description="Experiment title")
    schedulers: List[str] = Field(default_factory=lambda: list(ALL_SCHEDULERS), description="List of schedulers to compare")
    num_tasks: int = Field(default=30, ge=5, le=150, description="Tasks per run")
    num_repetitions: int = Field(default=3, ge=1, le=5, description="Number of seeded repetitions")
    seed: int = Field(default=42, description="Initial random seed")
    workload_type: str = Field(default="mixed", description="Workload profile: mixed, light_heavy, extreme_heavy, burst")
    max_steps: int = Field(default=50, ge=10, le=120, description="Max simulation steps per run")


@app.get("/api/benchmarks/history")
def get_benchmark_history():
    """Return index of completed research benchmark experiments."""
    return {
        "status": "success",
        "experiments": _bit9_store.list_experiments(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/benchmarks/latest")
def get_latest_benchmark():
    """Return the most recently saved experiment summary."""
    experiments = _bit9_store.list_experiments()
    if not experiments:
        return {"status": "empty", "message": "No research benchmarks have been executed yet."}
    latest_id = experiments[-1]["experiment_id"]
    data = _bit9_store.load_experiment(latest_id)
    if not data:
        return {"status": "error", "message": f"Could not load experiment {latest_id}"}
    return {"status": "success", "data": data}


@app.get("/api/benchmarks/experiment/{experiment_id}")
def get_benchmark_experiment(experiment_id: str):
    """Retrieve full results and metrics for a specific experiment ID."""
    data = _bit9_store.load_experiment(experiment_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Experiment '{experiment_id}' not found.")
    return {"status": "success", "data": data}


@app.get("/api/benchmarks/status")
def get_benchmark_runner_status():
    """Return current benchmark runner execution status and progress."""
    return {
        "is_running": _bit9_runner.is_running,
        "progress": _bit9_runner.current_progress,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/benchmarks/run")
async def run_research_benchmark(req: BenchmarkRunRequest):
    """
    Launch an isolated multi-algorithm research benchmark asynchronously.
    Enforces capacity limit, isolated fresh environments, and bounded workloads.
    """
    if _bit9_runner.is_running:
        raise HTTPException(
            status_code=409,
            detail="A benchmark experiment is already running. Please wait for completion.",
        )

    cfg = BenchmarkConfig(
        name=req.name,
        schedulers=req.schedulers,
        num_tasks=req.num_tasks,
        num_repetitions=req.num_repetitions,
        seeds=[req.seed + i for i in range(req.num_repetitions)],
        workload_type=req.workload_type,
        max_steps_per_run=req.max_steps,
    )

    async with _bit9_lock:
        res = await asyncio.to_thread(_bit9_runner.run_benchmark, cfg)

    return {
        "status": "success",
        "experiment_id": res["experiment_id"],
        "artifacts": res["artifacts"],
        "summary": res["summaries"],
    }


@app.post("/api/benchmarks/ablation")
async def run_ablation_study(tasks: int = 30, repetitions: int = 3, seed: int = 101):
    """Launch an isolated Risk-Aware MARL ablation study."""
    if _bit9_runner.is_running:
        raise HTTPException(
            status_code=409,
            detail="A benchmark experiment is already running. Please wait for completion.",
        )

    cfg = AblationConfig(
        num_tasks=tasks,
        num_repetitions=repetitions,
        seeds=[seed + i for i in range(repetitions)],
    )

    async with _bit9_lock:
        res = await asyncio.to_thread(_bit9_ablation_runner.run_ablation_study, cfg)

    return {
        "status": "success",
        "study_id": res["study_id"],
        "modes": res["modes"],
        "artifacts": res["artifacts"],
        "deltas_vs_full": res["deltas_vs_full"],
    }


if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)