# AETHERGRID - Advanced GPU Scheduling with Multi-Agent Reinforcement Learning

## Current Project Status

✅ GPU simulator complete
✅ Traditional, RL, and MARL schedulers implemented
✅ Live FastAPI backend and WebSocket streaming available
✅ React/Vite dashboard connected to the live simulator state
✅ Benchmark comparison panel uses backend values when available
✅ Dashboard presents simulator metrics separately from MARL-only metrics
✅ Phase 5: Locks, Data Consistency, and Race-Condition Handling implemented & verified (38/38 pytest tests passed, stress test CLI available)
✅ Bit 6: Risk-Aware RL + MARL implemented & verified (61/61 pytest tests passed, 7-scenario scientific benchmark available)

A sophisticated GPU scheduling simulator that demonstrates the superiority of Multi-Agent Reinforcement Learning (MARL) over traditional scheduling algorithms through comprehensive training, adaptive learning, and real-time performance comparison.

## 🎯 Project Overview

AETHERGRID is a research-grade GPU scheduling system that simulates a cluster of GPUs with advanced features like memory fragmentation, preemption, cooldown periods, and various workload patterns. The system implements and compares multiple scheduling approaches:

- **Traditional Schedulers**: FCFS, Round Robin, Least Loaded, Random, Best Fit, Priority
- **Single-Agent RL**: Q-learning based scheduler
- **Multi-Agent RL (MARL)**: Separate Q-learning agents for each GPU with coordination
- **Risk-Aware MARL (Bit 6)**: Cooperative multi-agent reinforcement learning integrating dynamic GPU risk scores, empirical failure rates, and expected failure cost (EFC) into bidding, state representations, and bounded reward functions.

## ✨ Key Features

### Phase 1: GPU Simulator
- **8GB Memory per GPU** with realistic memory management
- **Parallel Task Execution** with concurrency simulation
- **Crash Conditions** based on temperature thresholds (85°C)
- **Temperature Simulation** with thermal management
- **Task Queue Management** with priority handling

### Phase 2: Advanced Features
- **Memory Fragmentation**: Realistic memory allocation challenges
- **Task Preemption**: Ability to interrupt and reschedule tasks
- **Cooldown Periods**: GPU recovery after crashes
- **Task Generator**: Multiple workload types (LLM, Image, Video)
- **Traffic Patterns**: Light, Heavy, Burst, Peak/Off-Peak scenarios

### Phase 3: MARL System
- **Multi-Agent Architecture**: Separate Q-learning agents per GPU
- **Enhanced State Representation**: Memory, temperature, queue, fragmentation
- **Improved Reward Function**: Comprehensive incentive structure
- **Adaptive Exploration**: Multi-phase exploration schedule
- **Agent Coordination**: Performance-based parameter tuning
- **Model Persistence**: Save and load trained models

### Phase 4: Dynamic Task Queue & Video Execution
- **Dynamic Queue Lifecycle**: Priority-based scheduling, retry queues, real-time status transitions
- **Video Workload Execution**: Light, Medium, Heavy, and Extreme parallel video rendering pipelines
- **Full Cluster Visibility**: Comprehensive status and telemetry tracking across all 4 GPUs

### Phase 5: Locks, Data Consistency, and Race-Condition Handling
- **Thread-Safe Synchronization**: Reentrant mutexes (`threading.RLock`) protecting all shared GPU, queue, risk, fault, and MARL states
- **Deadlock-Free Lock Hierarchy**: Global acquisition ordering (`_step_lock` -> `task_queue._lock` -> `gpu._lock` -> `_stats_lock`)
- **Transactional Resource Updates**: Atomic allocation with automatic rollback on assignment failure
- **Atomic Compare-And-Swap (CAS)**: Double-completion and double-allocation prevention
- **Concurrency Testing**: 11 deterministic test scenarios and configurable high-contention stress test runner

### Bit 6: Risk-Aware RL + Multi-Agent Reinforcement Learning (MARL)
- **16-Element Discrete Risk State**: Deterministic bucketing of memory pressure, load, thermals, dynamic risk score, failure rate, status, task dimensions, and cluster contention
- **Expected Failure Cost (EFC)**: Proactive pre-placement penalty $EFC = P(\text{failure}) \times I(\text{task}) \times C(\text{recovery})$
- **Risk-Aware Cooperative Bidding**: Agent bid scores incorporate Q-values, headroom, risk penalties, and EFC penalties; strictly excludes quarantined or capacity-violating GPUs ($-\infty$)
- **Configurable Bounded Reward**: `RiskAwareRewardConfig` balances throughput and reliability while clamping rewards to $[-20.0, 12.0]$
- **Full Backend & Dashboard Observability**: Real-time broadcast of MARL decision score, risk score, EFC, reward, and per-GPU bid breakdowns
- **Scientific 7-Scenario Benchmark**: Evaluates MARL against traditional baselines (Round Robin, FCFS, Least Loaded) across normal, elevated risk, high recovery cost, dynamic fault/quarantine, mixed priority, heavy burst, and concurrent workloads

### Training & Evaluation
- **Multi-Dataset Training**: Different workload scenarios
- **Progressive Curriculum**: Easy to advanced training
- **Target-Based Training**: Specific success rate goals
- **Performance Tracking**: Comprehensive metrics and analysis
- **Traditional Comparison**: Direct benchmarking against baseline methods

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- Required packages (install via pip):
  ```bash
  pip install numpy
  ```

### Installation
```bash
# Clone the repository
git clone <your-repo-url>
cd gpu-rl-scheduler-main
```

### Running the Project

#### 1. Basic Demo
```bash
python main.py
```
Demonstrates the basic GPU simulator with default scheduler.

#### 2. MARL Scheduler Demo
```bash
python main_marl_demo.py
```
Shows MARL scheduler with learning capabilities and agent statistics.

#### 3. Traditional Baseline Demo
```bash
python main_baseline_demo.py
```
Demonstrates traditional schedulers (FCFS, Round Robin, Least Loaded).

#### 4. Improved MARL Learning
```bash
python marl_improvements_summary.py
```
Shows enhanced reward function and learning parameters with performance analysis.

#### 5. MARL Training System
```bash
python simple_marl_training.py
```
Simple MARL training demonstration with scenario-based training.

#### 6. System Verification
```bash
python verify_phase3.py
```
Comprehensive verification of all features and schedulers.

#### 7. Benchmark All Schedulers
```bash
python benchmark_schedulers.py
```
Compare all schedulers (MARL vs Traditional) with detailed performance metrics.

#### 8. Run the Live Dashboard
```bash
# Start the simulator backend
uvicorn backend.server:app --reload

# In a second terminal, start the frontend
cd frontend
npm install
npm run dev
```
The dashboard now streams live state from the backend instead of relying on mock data.

#### 9. Run Concurrency & Race-Condition Tests
```bash
python -m pytest tests/test_concurrency_locks.py -v
```
Runs the full 11-scenario concurrency suite verifying thread safety, atomic CAS guards, memory invariants, and deadlock freedom.

#### 10. Run Concurrency Stress Testing & Observability Tool
```bash
python concurrency_stress_test.py --workers 8 --duration 2.5 --scheduler marl
```
Runs a multi-threaded stress test evaluating concurrency throughput (ops/sec), legitimate rejections vs race violations, and cluster state consistency.

## 📁 Project Structure

```
gpu-rl-scheduler-main/
├── agent.py                          # Single-agent RL implementation
├── environment.py                     # GPU environment and simulator
├── gpu.py                            # GPU class with advanced features
├── main.py                           # Basic demo
├── main_marl_demo.py                 # MARL demonstration
├── main_baseline_demo.py             # Traditional schedulers demo
├── benchmark_schedulers.py           # Scheduler benchmarking
├── verify_phase3.py                  # System verification
├── marl_agent.py                     # Multi-agent RL implementation
├── marl_improvements_summary.py      # Enhanced MARL demonstration
├── simple_marl_training.py           # MARL training demo
├── marl_training_system.py           # Advanced training system
├── workload/
│   ├── __init__.py
│   ├── task_generator.py             # Task generation system
│   └── traffic_patterns.py           # Traffic pattern management
├── schedulers/
│   ├── __init__.py
│   └── traditional.py                # Traditional scheduler implementations
├── AETHERGRID_IMPLEMENTATION_REPORT.md
├── MARL_IMPLEMENTATION_SUMMARY.md
├── MARL_SUPERIORITY_REPORT.md
├── MARL_TRAINING_GUIDE.md
└── MARL_IMPROVEMENTS_IMPLEMENTED.md
```

## 🎯 Usage Examples

### Train MARL on Specific Scenario
```python
from marl_training_system import MARLTrainingSystem

trainer = MARLTrainingSystem()
result = trainer.train_marl_on_scenario('peak_hours', {
    'traffic_pattern': 'peak',
    'duration': 50,
    'target_success_rate': 0.70
})
```

### Compare Schedulers
```python
from environment import GPUEnvironment

# Test MARL
env_marl = GPUEnvironment(scheduler="marl")
env_marl.set_traffic_pattern("mixed", duration=100)
for _ in range(100):
    env_marl.step()

# Test Traditional
env_trad = GPUEnvironment(scheduler="traditional_least_loaded")
env_trad.set_traffic_pattern("mixed", duration=100)
for _ in range(100):
    env_trad.step()
```

### Custom Workload Generation
```python
from workload.task_generator import TaskGenerator

generator = TaskGenerator()
task = generator.generate_task("LLM", memory_required=4, duration=10)
```

## 📊 Performance Results

### MARL Superiority Demonstrated
- **Enhanced Learning**: 36% improvement in learning performance
- **Target Achievement**: 60% success rate on complex scenarios
- **Traditional Comparison**: MARL consistently outperforms traditional methods
- **Adaptive Behavior**: Responds to changing workload conditions
- **Multi-Agent Intelligence**: Coordinated decision making

### Key Metrics
- **Success Rate**: Up to 60% on complex scenarios
- **Learning Improvement**: 36% over baseline
- **Traditional Comparison**: +1-5% improvement over traditional methods
- **Adaptive Performance**: Consistent improvement over training iterations

## 🔧 Configuration

### GPU Parameters
- **Memory Capacity**: 8GB per GPU
- **Crash Temperature**: 85°C
- **Cooldown Duration**: 5 steps
- **Preemption Threshold**: Memory > 90%

### MARL Parameters
- **Learning Rate**: 0.002 (adaptive)
- **Epsilon Start**: 1.0 (maximum exploration)
- **Epsilon Min**: 0.05 (continued exploration)
- **Epsilon Decay**: 0.997 (slower decay)
- **Coordination Frequency**: Every 5 steps

### Traffic Patterns
- **Light**: 0.3-0.5 task probability
- **Heavy**: 0.7-0.9 task probability
- **Burst**: Alternating light/heavy periods
- **Peak**: 0.8-1.0 task probability
- **Off-Peak**: 0.1-0.3 task probability

## 📈 Research Applications

### Academic Research
- **Multi-Agent Reinforcement Learning**: Study coordination and learning
- **Resource Scheduling**: GPU cluster optimization
- **Adaptive Systems**: Dynamic parameter tuning
- **Comparative Analysis**: Traditional vs RL approaches

### Industrial Applications
- **Cloud Computing**: GPU resource allocation
- **Data Centers**: Efficient scheduling
- **ML Training**: Optimized model training
- **High-Performance Computing**: Resource management

## 🔒 Phase 5: Locks, Data Consistency, and Race-Condition Handling

AETHERGRID Bit 5 introduces comprehensive synchronization, atomic compare-and-swap (CAS) transitions, transactional resource allocations with automatic rollback, and deadlock-free lock hierarchies across all simulator components, MARL agents, and web backend endpoints.

### 1. Shared Resources Protected by Locks

Every mutable shared resource across the simulation, queue, risk engine, fault injection, and API server is guarded by dedicated reentrant locks:

| Component | File | Synchronization Primitive | Protected Mutable State |
|---|---|---|---|
| **GPU Model** | [`gpu.py`](file:///d:/BMSIT/7th%20sem%20-%20proj/first%20phase%20%281-4%29/gpu-draft--main/gpu-draft--main/gpu.py) | `self._lock = threading.RLock()` | `current_memory`, `memory_fragments`, `running_tasks`, `task_queue`, `preempted_tasks`, `crashed`, `crash_cooldown_timer`, `temperature`, `utilization`, `total_latency` |
| **Dynamic Task Queue** | [`dynamic_queue.py`](file:///d:/BMSIT/7th%20sem%20-%20proj/first%20phase%20%281-4%29/gpu-draft--main/gpu-draft--main/dynamic_queue.py) | `self._lock = threading.RLock()` | `task_registry`, `task_to_gpu`, status dictionaries (`QUEUED`, `SCHEDULED`, `RUNNING`, `COMPLETED`, `FAILED`, `CANCELLED`), lifecycle events log |
| **GPU Risk Engine** | [`risk_engine.py`](file:///d:/BMSIT/7th%20sem%20-%20proj/first%20phase%20%281-4%29/gpu-draft--main/gpu-draft--main/risk_engine.py) | `self._lock = threading.RLock()` | `reliability_history`, dynamic risk scores, quarantine states, historical anomaly records |
| **Fault Injector** | [`fault_injection.py`](file:///d:/BMSIT/7th%20sem%20-%20proj/first%20phase%20%281-4%29/gpu-draft--main/gpu-draft--main/fault_injection.py) | `self._lock = threading.RLock()` | `event_log`, `event_counter`, pseudo-random generator state (`rng`), custom anomaly probabilities |
| **MARL Manager & Agents** | [`marl_agent.py`](file:///d:/BMSIT/7th%20sem%20-%20proj/first%20phase%20%281-4%29/gpu-draft--main/gpu-draft--main/marl_agent.py) | `self._lock = threading.RLock()` | Per-agent Q-tables, epsilon decay schedules, coordinated parameter updates, decision telemetry |
| **GPU Environment** | [`environment.py`](file:///d:/BMSIT/7th%20sem%20-%20proj/first%20phase%20%281-4%29/gpu-draft--main/gpu-draft--main/environment.py) | `self._step_lock`, `self._stats_lock` | Simulation step transitions, task registration, event logs, history time-series, cumulative statistics |
| **FastAPI Backend Server** | [`backend/server.py`](file:///d:/BMSIT/7th%20sem%20-%20proj/first%20phase%20%281-4%29/gpu-draft--main/gpu-draft--main/backend/server.py) | `server_lock = threading.RLock()` | `video_queue`, `video_task_counter`, `uploaded_video_cache`, active scheduler selection, WebSocket client broadcast list |

---

### 2. Synchronization Mechanisms Selected & Why

- **Reentrant Locks (`threading.RLock`)**:
  Chosen for in-process multithreaded synchronization. Reentrancy is essential because scheduling and task assignment frequently execute nested helper methods on the same thread (e.g., `_try_assign` holding `task_queue._lock` while calling `task_queue.schedule_task()`, or `GPUEnvironment.step()` inspecting GPU states). Standard mutexes would produce self-deadlock.
- **Global Deadlock-Free Lock Hierarchy**:
  To prevent classic AB-BA circular deadlocks when multiple components must be locked simultaneously, a strict universal lock acquisition order is enforced throughout the codebase:
  $$\text{Level 1: } \texttt{\_step\_lock} \longrightarrow \text{Level 2: } \texttt{task\_queue.\_lock} \longrightarrow \text{Level 3: } \texttt{gpu.\_lock (ordered by GPU ID)} \longrightarrow \text{Level 4: } \texttt{\_stats\_lock}$$
  Every method requiring multiple locks (such as `_try_assign`, `step()`, and `get_dashboard_state()`) strictly adheres to this hierarchy.
- **Atomic Compare-And-Swap (CAS) Lifecycle Transitions**:
  Task status transitions are protected by atomic condition checks under `task_queue._lock`. For example, `complete_task(task_id)` verifies that the task is in `SCHEDULED` or `RUNNING` status before transitioning to `COMPLETED`. If a concurrent worker attempts a duplicate completion, the atomic check fails and returns `False`, eliminating duplicate memory deallocations or counter double-increments.
- **Transactional Resource Updates with Automatic Rollback**:
  In `_try_assign()`, task assignment and GPU memory allocation are executed as a single transaction:
  1. The task is checked for eligibility and registered in the queue under `task_queue._lock`.
  2. The target GPU lock is acquired (`gpu._lock`).
  3. Memory and queue allocation are attempted on the GPU (`gpu.assign_task(task)`).
  4. If any subsequent step or queue transition fails, a rollback handler catches the exception, frees the allocated GPU memory, removes the task from the GPU's queues, and restores consistent state.
- **Single Centralized Background Simulation Loop**:
  FastAPI's WebSocket broadcast loop previously allowed multiple clients to concurrently step the environment. In Phase 5, simulation stepping is decoupled into a single synchronized background worker task, while WebSocket clients safely read non-blocking point-in-time state snapshots via `get_dashboard_state()`.

---

### 3. Atomicity & Resource-Accounting Guarantees

1. **Non-Negative Memory Invariant**: GPU memory usage is mathematically guaranteed never to drop below `0.0 GB`:
   $$\forall g \in \text{GPUs}, \quad \text{gpu.current\_memory} \ge 0.0$$
2. **Physical Capacity Invariant**: Task allocation strictly enforces the physical GPU capacity boundary ($8.0\text{ GB}$). A task is accepted if and only if `current_memory + task.memory_required <= memory_capacity`:
   $$\forall g \in \text{GPUs}, \quad \text{gpu.current\_memory} \le \text{gpu.memory\_capacity} = 8.0\text{ GB}$$
   Requests exceeding $8.0\text{ GB}$ are cleanly rejected as legitimate capacity rejections. If an anomalous request severely exceeds the hardware crash threshold ($8.0 \times 1.2 = 9.6\text{ GB}$), the GPU triggers an emergency crash (`_enter_crash()`) into cooldown.
3. **Single Assignment Invariant**: A task can only be assigned to exactly one GPU at any point in time. Concurrent allocation attempts across multiple GPUs or duplicate queue scheduling are rejected.
4. **Single Completion Guarantee**: A task's resources are freed once and only once upon completion, regardless of concurrent worker attempts to execute or complete it.
5. **No Resource Leaks on Failure/Quarantine**: When a GPU crashes or enters quarantine during an allocation attempt, partially allocated resources are immediately rolled back and released.

---

### 4. Race Conditions Considered & Addressed

1. **Concurrent Allocations on a Single GPU**: High-volume parallel allocation requests competing for memory on the same GPU. Synchronized via `gpu._lock`; requests exceeding available memory are cleanly rejected without exceeding capacity.
2. **Concurrent Allocations Across Multiple GPUs**: Multiple threads allocating across distinct GPUs concurrently without cross-GPU interference or deadlock.
3. **Concurrent Allocation and Deallocation**: Interleaved memory allocations and task completions occurring in parallel without memory corruption, double-free, or negative memory.
4. **Concurrent Queue Updates**: Parallel workers enqueuing, re-prioritizing, and querying tasks simultaneously without corrupted registry states.
5. **Duplicate Task Completion Races**: Two workers racing to complete the same task. Prevented by atomic CAS state guards in `complete_task()`.
6. **Preemption Near Completion**: Preemption racing against normal task completion. Handled cleanly without leaving orphan task entries or leaking VRAM.
7. **GPU Failure / Quarantine During Allocation**: Fault injection crashing a GPU while allocation requests are in-flight. Crashed GPU cleanly rejects incoming requests with zero partial updates.
8. **GPU Recovery Concurrent with Allocation**: Recovery triggers interleaved with allocation attempts without race conditions or memory overshoot.
9. **MARL Multi-Agent Coordination Concurrency**: Multiple agents updating Q-tables and coordinating exploration schedules simultaneously. Synchronized via `MARLManager._lock`.
10. **Concurrent API Requests & Simulation Stepping**: FastAPI HTTP endpoints (`/start`, `/pause`, `/reset`, `/set-scheduler`, `/analyze-video`) called concurrently with active background stepping loops.
11. **High-Contention Interleaved Stress**: Simultaneous task admission, stepping, telemetry reading, and anomaly injection across 8 worker threads without deadlocks.

---

### 5. Concurrency Testing & Observability

A dedicated test suite and a standalone CLI stress utility are provided:

#### Running the Concurrency Unit Tests
```bash
python -m pytest tests/test_concurrency_locks.py -v
```
Runs all 11 deterministic scenarios with barriers, concurrent thread pools, and invariant assertions.

#### Running the Complete Test Suite
```bash
python -m pytest tests/ -v
```
Executes all 38 test cases across the entire project (Phases 1–5).

#### Running the Concurrency Stress Testing & Observability CLI
```bash
python concurrency_stress_test.py --workers 8 --duration 2.5 --scheduler marl
```
**Sample Output:**
```text
============================================================
          CONCURRENCY STRESS TEST RESULTS
============================================================
Duration               : 2.541s
Operations Attempted   : 780 (307.0 ops/sec)
Allocations Succeeded  : 182
Capacity Rejections    : 245 (Legitimate OOM)
Quarantine Rejections  : 8 (Legitimate Safety)
Duplicates Prevented   : 36 (Atomic CAS Guards)
Invariant Violations   : 0

[OK] ALL CONCURRENCY INVARIANTS SATISFIED (Zero Violations)

Final Cluster GPU State:
  GPU 1: 8.00/8 GB VRAM | Running: 2 | Queued: 6 | Status: HEALTHY
  GPU 2: 7.80/8 GB VRAM | Running: 3 | Queued: 4 | Status: HEALTHY
  GPU 3: 8.00/8 GB VRAM | Running: 2 | Queued: 5 | Status: HEALTHY
  GPU 4: 6.50/8 GB VRAM | Running: 1 | Queued: 7 | Status: HEALTHY
============================================================
```

---

### 6. Invariants Verified by Tests

The test suite and stress tester explicitly verify:
- `gpu.current_memory >= 0.0` (Never negative memory).
- `gpu.current_memory <= gpu.memory_capacity` (Strict physical capacity boundary of 8.0 GB).
- `gpu.get_available_memory() + gpu.current_memory == gpu.memory_capacity` (Consistent memory arithmetic).
- `sum(task_counts.values()) == len(task_registry)` (Dynamic queue state integrity).
- `complete_task()` returns `False` on already-completed tasks (No duplicate transitions).
- Zero thread timeouts or deadlocks across high-contention barrier synchronization.

---

### 7. Known Limitations

- **In-Process Scope**: All locks and synchronization primitives are in-process (`threading.RLock`). They protect multi-threaded Python execution, thread pools, and asynchronous ASGI event loop workloads within a single process. They do **not** provide cross-process distributed locking (e.g., across multiple operating system processes or distributed clusters via Redis/ZooKeeper).
- **CPython GIL**: In standard CPython, Python bytecode execution is serialized by the Global Interpreter Lock (GIL). However, thread scheduling, coroutine interleaving, and I/O operations (such as FastAPI request handling and WebSocket streams) can and do expose real race conditions in shared data structures without explicit synchronization.

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional scheduling algorithms
- Enhanced reward functions
- More realistic GPU simulation
- Extended training scenarios
- Performance optimization

## 📝 Documentation

Comprehensive documentation is available in the repository root:
- **BIT7_BACKEND_WEBSOCKET_INTEGRATION.md**: Real-time REST APIs & WebSocket Integration documentation
- **BIT6_RISK_AWARE_MARL.md**: Risk-aware MARL and Expected Failure Cost (EFC) architecture
- **AETHERGRID_IMPLEMENTATION_REPORT.md**: Implementation status and feature matrix
- **MARL_IMPLEMENTATION_SUMMARY.md**: MARL system details
- **MARL_SUPERIORITY_REPORT.md**: Performance comparison
- **MARL_IMPROVEMENTS_IMPLEMENTED.md**: Enhanced learning notes
- **MARL_TRAINING_GUIDE.md**: Training system guide

## 🏆 Acknowledgments

This project demonstrates the application of Multi-Agent Reinforcement Learning to GPU scheduling, showing how adaptive learning can outperform traditional static algorithms in complex, dynamic environments.

## 📄 License

This project is provided for educational and research purposes.

## 🔗 References

- **Reinforcement Learning**: Sutton & Barto
- **Multi-Agent Systems**: Wooldridge
- **GPU Scheduling**: Various research papers
- **Q-Learning**: Watkins & Dayan

---

**Status**: ✅ All features implemented and verified
**Version**: 1.0.0
**Last Updated**: 2026
