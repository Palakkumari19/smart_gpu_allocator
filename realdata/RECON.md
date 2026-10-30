# Codebase Reconnaissance Report: Real-Data Risk Evaluation

This document answers the six structural questions regarding existing interfaces in `gpu-draft-main`, citing exact file paths and line numbers. It confirms how the new `realdata/` package can seamlessly build upon existing components in an **ADD-ONLY** fashion with zero modifications to existing code.

---

## 1. Exact Constructor & Fields of `EnhancedTask`

- **File**: [`workload/task_generator.py`](file:///home/bmsit/Downloads/gpu-draft-main/workload/task_generator.py#L362-L431)
- **Class**: `EnhancedTask` (line 362)

### Constructor Signature
```python
def __init__(self, task_type, memory_required, execution_time, description="Workload task", priority="medium"):
```
*(line 367)*

### Parameters & Default Values
- `task_type`: Task category (e.g., `TaskType.LLM`, `TaskType.IMAGE`, `TaskType.VIDEO` or string).
- `memory_required`: float/int representing required GPU VRAM in GB.
- `execution_time`: float/int representing task execution duration in simulation steps.
- `description`: str, defaults to `"Workload task"`.
- `priority`: str, defaults to `"medium"`.

### Exact Initialized Fields
- `self.task_type = task_type` (line 368)
- `self.memory_required = memory_required` (line 369)
- `self.execution_time = execution_time` (line 370)
- `self.description = description` (line 371)
- `self.priority = priority` (line 372)
- `self.execution_progress = 0` (line 373)
- `self.preempted = False` (line 374)
- `self.preemption_count = 0` (line 375)
- `self.original_execution_time = None` (line 376)
- `self.id = f"task_{random.randint(1000, 9999)}"` (line 377)
- `self.created_time = None` (line 378)
- `self.profile_name = "Medium"` (line 381)
- `self.resolution = "1080p"` (line 382)
- `self.width = 1920` (line 383)
- `self.height = 1080` (line 384)
- `self.fps = 30` (line 385)
- `self.duration = 10` (line 386)
- `self.frames = 300` (line 387)
- `self.codec = "H.264"` (line 388)
- `self.bitrate = "5.0 Mbps"` (line 389)
- `self.workload_severity = severity_map.get(str(priority).lower(), "MEDIUM")` (line 391)
- `self.severity = self.workload_severity` (line 392)
- `self.characteristics = {"cpu_intensive": True, "memory_bandwidth": "high", "parallelizable": False, "interruptible": True}` (lines 393-398)

### Interruptible Attribute
- Stored under `self.characteristics["interruptible"] = True` (line 397).

### Attaching Extra Attributes
- `EnhancedTask` does **not** define `__slots__`. It is a standard Python object with a normal `__dict__`.
- Extra metadata attributes (e.g. `task.job_id`, `task.user`, `task.raw_dataset_source`, `task.arrival_time`) can be attached directly at runtime without errors.
- `task.id` can also be directly overwritten (e.g. `task.id = f"alibaba_job_{job_id}"`).

---

## 2. What `GPUEnvironment.attach_workload_trace()` Expects

- **File**: [`environment.py`](file:///home/bmsit/Downloads/gpu-draft-main/environment.py#L342-L344) and [`environment.py`](file:///home/bmsit/Downloads/gpu-draft-main/environment.py#L835-L841)

### Definition
```python
def attach_workload_trace(self, steps_of_tasks):
    """Replay a fixed per-step task arrival sequence."""
    self.workload_trace = [list(step_tasks) for step_tasks in steps_of_tasks]
```
*(lines 342-344)*

### Expected Shape
- An outer iterable (e.g., `list`) of inner iterables (e.g., `list` of `EnhancedTask`):
  `Sequence[Sequence[Task]]` of shape `[num_steps, tasks_per_step]`.
- Outer index corresponds to step index `0, 1, 2, ..., N-1`.
- Each inner list contains 0 or more task objects arriving at that simulation step.

### Copying Semantics
- At attachment time (`attach_workload_trace()`, line 344), it performs a shallow list conversion: `[list(step_tasks) for step_tasks in steps_of_tasks]`.
- At execution time inside `step()` (lines 835-840):
  ```python
  if self.workload_trace is not None:
      trace_index = self.time_step - 1
      if trace_index < len(self.workload_trace):
          incoming_tasks = [copy.deepcopy(task) for task in self.workload_trace[trace_index]]
      else:
          incoming_tasks = []
  ```
- Because `step()` performs `copy.deepcopy(task)`, a master trace object can be reused across multiple scheduler runs or episode repetitions without mutation contamination.

### Time Step Indexing
- At the start of `env.step()`, `self.time_step += 1` (line 833). (Initial `time_step` on initialization is `0`).
- The lookup index is `trace_index = self.time_step - 1` (line 836).
- Step 1 reads `self.workload_trace[0]`.
- Step 2 reads `self.workload_trace[1]`.
- Step $K$ reads `self.workload_trace[K - 1]`.
- If `trace_index >= len(self.workload_trace)`, `incoming_tasks` evaluates to `[]` (line 840), allowing existing in-flight and queued tasks to drain.

---

## 3. GPU Physics: Temperature Rise, Crashes, Risk Inputs, and `gpu.inject_fault()`

- **File**: [`gpu.py`](file:///home/bmsit/Downloads/gpu-draft-main/gpu.py)
- **Related Files**: [`risk_engine.py`](file:///home/bmsit/Downloads/gpu-draft-main/risk_engine.py), [`fault_injection.py`](file:///home/bmsit/Downloads/gpu-draft-main/fault_injection.py)

### Temperature Dynamics
- **Baseline**: Starts at `self.temperature = 30.0` °C (`gpu.py:16`, reset in line 390).
- **Heating Factors**:
  1. Task Allocation: On `assign_task(task)`:
     ```python
     temp_increase = self._randint(1, 4) + len(self.task_queue) // 2
     self.temperature += temp_increase
     ```
     *(lines 179-180)*
  2. Active Task Execution: In `process_tasks()`:
     ```python
     if self.running_tasks:
         self.temperature += len(self.running_tasks) * 0.5
     ```
     *(lines 312-313)*
  3. Defragmentation: In `defragment_memory()`: `self.temperature += 2` *(line 381)*.
- **Cooling Factors**:
  1. Task Completion: In `process_tasks()`: `self.temperature = max(30.0, self.temperature - self._randint(2, 5))` *(line 309)*.
  2. Task Preemption: In `process_tasks()`: `self.temperature = max(30.0, self.temperature - 5.0)` *(line 257)*, and `preempt_task()` *(line 119)*.
  3. Crash Cooldown Recovery: In `process_tasks()`: `self.temperature = max(30.0, self.temperature - 20.0)` *(line 199)*.
  4. Manual Recovery: In `recover()`: `self.temperature = max(30.0, self.temperature - 20.0)` *(line 87)*.

### Crash Conditions
1. **Thermal Crash**: In `process_tasks()`:
   If `self.temperature >= self.crash_threshold_temp` (`crash_threshold_temp = 85` °C, line 29), calls `self._enter_crash()`, sets `self.crash_cooldown_timer = 5`, and preempts running tasks (`gpu.py:221-230`).
2. **Memory Overload Crash**: In `assign_task(task)`:
   If `self.current_memory + task.memory_required > self.memory_capacity * self.memory_overload_factor` (`memory_overload_factor = 1.2`, line 30, meaning > 9.6 GB on an 8 GB GPU), calls `self._enter_crash()` and rejects task (`gpu.py:157-159`).
3. **Injected Crash**: In `gpu.inject_fault()`:
   If `result.gpu_crashed`, sets `self.crashed = True`, sets `self.crash_cooldown_timer = 5`, and records crash in `risk_engine` (`gpu.py:581-589`).
4. **Explicit Trigger**: `gpu.crash()` calls `self._enter_crash()` (`gpu.py:78`).

In `self._enter_crash()` (`gpu.py:71-75`):
Increments `self.total_crash_events += 1`, calls `self.record_crash_event()` (which notifies `self.risk_engine.record_crash(self.id)`), and sets `self.crashed = True`.

### Risk Inputs
Computed via `gpu.update_risk()` (`gpu.py:451-468`) calling `risk_engine.py`:
1. `temperature_risk`: Piecewise linear mapping of `gpu.temperature` between `TEMP_SAFE` (65°C), `TEMP_WARNING` (75°C), and `TEMP_CRITICAL` (85°C) (`risk_engine.py:162-169`).
2. `memory_pressure`: `gpu.get_memory_usage_percentage() / 100.0` (`risk_engine.py:173-174`).
3. `fragmentation_risk`: `gpu.get_fragmentation_percentage() / 50.0` (`risk_engine.py:178-179`).
4. `utilization_risk`: `gpu.utilization / 100.0` (`risk_engine.py:183`).
5. `failure_history`: `reliability_history.failure_rate / 50.0` (`risk_engine.py:198-199`).
6. `crash_history`: `reliability_history.crash_events / 10.0` (`risk_engine.py:203`).
7. `anomaly_score`: `reliability_history.anomalies_detected / 5.0` (`risk_engine.py:249`).
8. `performance_degradation`: `(actual_time - expected_time) / expected_time / 2.0` (`risk_engine.py:226-229`).

Combined weighted risk score in `risk_engine.py:281-290` determines risk level (`HEALTHY`, `LOW_RISK`, `SUSPICIOUS`, `HIGH_RISK`) and triggers quarantine if `risk_score >= 0.85` (`risk_engine.py:305`).

### Signature of `gpu.inject_fault()`
```python
def inject_fault(self, task_id: Optional[str] = None, severity=None) -> AnomalyResult:
```
*(file: `gpu.py`, line 548)*
Returns an `AnomalyResult` dataclass with fields `occurred`, `anomaly_type`, `severity`, `execution_time_multiplier`, `task_failed`, `gpu_crashed`, and `details` (`fault_injection.py:82-90`).

---

## 4. Risk-Related Metric Keys in `env.get_metrics()`

- **File**: [`environment.py:954-1033`](file:///home/bmsit/Downloads/gpu-draft-main/environment.py#L954-L1033)
- **Risk Engine Metrics**: [`risk_engine.py:460-494`](file:///home/bmsit/Downloads/gpu-draft-main/risk_engine.py#L460-L494)

The specific dictionary keys relating to risk, safety, crashes, and quarantine are:

| Metric Key | Source / Calculation | Line in `environment.py` | Description |
|---|---|---|---|
| `"average_gpu_risk"` | `risk_metrics["average_risk"]` | line 1025 | Mean risk score across all GPUs $[0.0, 1.0]$. |
| `"maximum_gpu_risk"` | `risk_metrics["maximum_risk"]` | line 1026 | Peak risk score across cluster $[0.0, 1.0]$. |
| `"risk_variance"` | `risk_metrics["risk_variance"]` | line 1027 | Variance of risk scores across GPUs. |
| `"total_anomalies"` | `risk_metrics["total_anomalies"]` | line 1028 | Cumulative detected anomalies across cluster. |
| `"quarantined_gpus"` | `risk_metrics["quarantined_gpus"]` | line 1030 | Current count of quarantined GPUs. |
| `"risk_metrics"` | `self.risk_engine.get_global_risk_metrics()` | line 1024 | Nested dict with global risk details. |
| `"average_efc"` | `total_assigned_efc / max(1, total_efc_assignments)` | line 973 | Mean Expected Failure Cost per assigned task. |
| `"total_assigned_efc"` | `self.total_assigned_efc` | line 974 | Cumulative assigned Expected Failure Cost. |
| `"crashes"` | `sum(int(g.crashed) for g in self.gpus)` | line 958 | Current active crash count. |
| `"total_crash_events"` | `sum(g.total_crash_events for g in self.gpus)` | line 969 | Cumulative crash events since start. |
| `"total_crashes"` | `risk_metrics["total_crashes"]` | line 1029 | Cumulative crashes from risk engine history. |
| `"gpus_in_cooldown"` | `sum(1 for g in self.gpus if g.is_in_cooldown())` | line 968 | GPUs currently cooling down from a crash. |
| `"total_preemptions"` | `sum(g.total_preemptions for g in self.gpus)` | line 966 | Cumulative task preemptions triggered. |
| `"total_preempted_tasks"`| `sum(g.get_preempted_tasks_count() for g in self.gpus)` | line 967 | Count of currently paused/preempted tasks. |
| `"failed_assignments"` | `self.failed_assignments` | line 971 | Count of task dispatch failures/rejections. |

Additionally, [`benchmarking/metrics.py:162-180`](file:///home/bmsit/Downloads/gpu-draft-main/benchmarking/metrics.py#L162-L180) maps these into structured `RunMetrics`: `avg_risk_exposure`, `max_risk_exposure`, `avg_efc`, `total_efc`, `cooldown_events`, `quarantine_events`.

---

## 5. Episode Construction & Execution Pattern in Adapters and Runners

- **Files**:
  - [`benchmarking/scheduler_adapters.py:23-172`](file:///home/bmsit/Downloads/gpu-draft-main/benchmarking/scheduler_adapters.py#L23-L172)
  - [`evaluation/scientific_benchmark.py:56-69`](file:///home/bmsit/Downloads/gpu-draft-main/evaluation/scientific_benchmark.py#L56-L69)
  - [`benchmarking/benchmark_runner.py:180-250`](file:///home/bmsit/Downloads/gpu-draft-main/benchmarking/benchmark_runner.py#L180-L250)

Both runners follow an identical, non-invasive execution recipe that a new runner in `realdata/` can mirror:

1. **Environment Initialization**:
   ```python
   env = GPUEnvironment(scheduler="fcfs")
   ```
   Creates an isolated cluster instance with default parameters.

2. **Enable Determinism & Evaluation Mode**:
   ```python
   env.enable_deterministic_physics()  # Locks random jitter/stochastics
   env.set_eval_mode()                 # Sets learning_enabled = False, epsilon = 0.0
   ```
   Ensures fair, repeatable comparisons across schedulers without exploration noise.

3. **Configure Scheduler Adapter**:
   ```python
   adapter = get_scheduler_adapter(scheduler_id)
   adapter.configure_environment(env)
   adapter.reset_state(env)
   ```
   - For `fcfs`, `round_robin`, `least_loaded`: sets `env.scheduler` and pointers.
   - For `rl`: configures single-agent Q-learning and disables exploration.
   - For `marl` (risk-blind): dynamically replaces `env.marl_manager.compute_bid_score_breakdown` on the isolated instance to eliminate risk & EFC penalties (`scheduler_adapters.py:127-142`).
   - For `risk_aware_marl`: uses standard MARL with active risk engine and EFC bidding deductions.

4. **Attach Workload Trace**:
   ```python
   env.attach_workload_trace(cloned_trace)
   ```
   Deep-copies tasks per step so each scheduler receives identical input arrivals.

5. **Step Loop & Termination**:
   ```python
   for _ in range(max_steps):
       env.step()
       # Optional: check if trace is exhausted and queues + running tasks are empty
       queued_cnt = len(env.task_queue.get_tasks_by_status("QUEUED"))
       running_cnt = sum(gpu.get_running_tasks_count() for gpu in env.gpus)
       if env.time_step >= len(cloned_trace) and queued_cnt == 0 and running_cnt == 0:
           break
   ```

6. **Metric Extraction**:
   - Via `MetricsCalculator.calculate_run_metrics(...)` (`benchmarking/metrics.py:93-180`), or
   - Via `collect_run_metrics(env, steps)` (`evaluation/scientific_benchmark.py:36-53`).

---

## 6. External Fault Overlay via `gpu.inject_fault()`

### Can an external fault overlay be applied between `env.step()` calls without modifying existing files?
**YES.**

### Verification & Rationale:
1. **Existing Wiring**: In `GPUEnvironment.__init__` (`environment.py:316-325`):
   - Every GPU is already initialized with `gpu.set_risk_engine(self.risk_engine)` (line 317).
   - Every GPU is already initialized with `gpu.set_fault_injector(self.fault_injector)` (line 325).
2. **Interface Safety**: `gpu.inject_fault()` (`gpu.py:548-592`) is thread-safe and public. It accepts `task_id` (optional) and `severity` (optional).
3. **Execution Effects**:
   - Calling `env.gpus[gpu_id].inject_fault(severity=SeverityLevel.HIGH)` directly updates `self.reliability_history.anomalies_detected` and notifies `self.risk_engine.record_anomaly(self.id)`.
   - If the anomaly includes a crash (`result.gpu_crashed`), it sets `gpu.crashed = True`, starts the 5-step cooldown, and records the crash in the risk engine (`gpu.py:581-589`).
   - On the next `env.step()`, `gpu.update_risk()` (`gpu.py:451`) incorporates these events, increasing the risk score and potentially triggering quarantine if risk $\ge 0.85$.
4. **Deterministic Control**:
   - By default, `FaultInjector` has internal probabilistic gating (`SeverityLevel.LOW: 0.05`, `MEDIUM: 0.15`, `HIGH: 0.30`) (`fault_injection.py:101-105`).
   - An external overlay desiring deterministic faults can simply call `env.fault_injector.set_probability(severity, 1.0)` (`fault_injection.py:147-150`).
   - Alternatively, for direct hardware crash overlay, the runner can call `env.gpus[gpu_id].crash()` directly (`gpu.py:78`).

---

## 7. Baseline Test Suite Verification

- **Command Executed**: `PYTHONPATH=. .venv/bin/pytest -q`
- **Result**:
  ```text
  136 passed, 10 warnings in 6.53s
  ```
- **Test Suite Breakdown**:
  - `tests/test_benchmark_reporting.py`: 1 passed
  - `tests/test_bit6_risk_aware_marl.py`: 34 passed
  - `tests/test_bit7_backend_websocket.py`: 15 passed
  - `tests/test_bit7_risk_monitoring.py`: 5 passed
  - `tests/test_bit8_dashboard_visualization.py`: 14 passed
  - `tests/test_bit9_benchmarking.py`: 15 passed
  - `tests/test_concurrency_locks.py`: 11 passed
  - `tests/test_dynamic_task_queue.py`: 5 passed
  - `tests/test_gpu_memory_accounting.py`: 3 passed
  - `tests/test_marl_bidding.py`: 3 passed
  - `tests/test_marl_reward.py`: 3 passed
  - `tests/test_phase4_execution.py`: 5 passed
  - `tests/test_scheduler_selection.py`: 1 passed
  - `tests/test_scientific_evaluation.py`: 6 passed
  - *(Subtotal core unit tests: **121 passed**)*
  - `tests/test_real_gpu_execution.py`: 6 passed
  - `test_improved_marl.py`: 2 passed
  - `test_phase2_fault_system.py`: 7 passed
  - **Total**: **136 passed, 0 failed** (10 non-fatal warnings documented in `realdata/NOTES.md`).
