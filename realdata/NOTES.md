# Existing Bugs & Codebase Observations (Do Not Fix)

Per guardrails, existing bugs are documented here rather than being modified in the existing codebase:

1. **`test_phase2_fault_system.py` & `test_improved_marl.py` (Pytest Return Warnings)**
   - *File & Lines*: `test_phase2_fault_system.py:17-100+`, `test_improved_marl.py:15-80`
   - *Issue*: Test functions return boolean values (e.g., `return True` or `return result`) instead of using `assert <condition>`. This causes `PytestReturnNotNoneWarning` under pytest.
   - *Action*: Do not edit or touch these test files.

2. **Identical Branches in `GPUEnvironment.step()`**
   - *File & Lines*: `environment.py:843-848`
   - *Issue*:
     ```python
     if self.traffic_mode == "peak_off_peak":
         num_requests = self.traffic_manager.step()
     else:
         num_requests = self.traffic_manager.step()
     ```
     Both branches execute `num_requests = self.traffic_manager.step()`.
   - *Action*: Do not edit. When using attached traces via `attach_workload_trace`, this code path is bypassed anyway as `self.workload_trace` is checked first (`environment.py:835-840`).

3. **Float Preemption Time Reduction in `EnhancedTask.preempt()`**
   - *File & Lines*: `workload/task_generator.py:405`
   - *Issue*: `self.execution_time = max(1, self.execution_time - 0.7)` changes `execution_time` from integer to float (e.g. 3.3). In `gpu.py:284-285`, `task.execution_progress += 1` is an integer compared via `task.execution_progress >= task.execution_time`.
   - *Action*: Do not edit.

4. **Linear Temperature Accumulation in `GPU.assign_task()`**
   - *File & Lines*: `gpu.py:179-180`
   - *Issue*:
     ```python
     temp_increase = self._randint(1, 4) + len(self.task_queue) // 2
     self.temperature += temp_increase
     ```
     Fast arrival of tasks directly pushes `self.temperature` up per task, which can trigger thermal crash (>85°C) if tasks accumulate in `task_queue` faster than they are processed or cooled down.
   - *Action*: Do not edit.

5. **Jain's Fairness Index Calculation on Zero Tasks**
   - *File & Lines*: `benchmarking/metrics.py:157-160`
   - *Issue*: If all GPUs have completed 0 tasks, `sum_sq_x == 0`, and the calculation falls back to `jains = 1.0` (technically vacuously fair, but 0 tasks completed).
   - *Action*: Do not edit.

6. **Exclusion of `failure_rate_pct` and `rejection_rate_pct` from Benchmark Metrics**
   - *Files & Interfaces*: `environment.py:970-972`, `benchmarking/metrics.py:113-117`
   - *Reason for Exclusion*:
     - Traditional schedulers (`fcfs`, `round_robin`, `least_loaded`) hold admitted tasks indefinitely in queues rather than rejecting them with an explicit status, whereas other components record rejections only upon strict memory checks.
     - Comparing percentage failure/rejection rates across schedulers creates measurement asymmetry where queue buffering is conflated with rejection drops.
     - Instead, the benchmark pre-registers objective physical metrics: `avg_risk_exposure`, `terminal_efc`, `crash_count`, `quarantine_time` (all lower is better), and `completed_tasks` (higher is better).


7. **Empirical Fault Overlay and Hardcoded 5% Gate in `GPUEnvironment.step()`**
   - *File & Lines*: `environment.py:886-890`, `fault_injection.py:184-212`, `gpu.py:548-592`
   - *Observation*:
     - During standard `env.step()`, line 888 applies a hardcoded 5% synthetic gate: `if self._rand() < 0.05: fault_result = gpu.inject_fault(task.id)`.
     - When `gpu.inject_fault()` is invoked, `fault_injection.py:209` additionally tests against `self.custom_probabilities[severity]`.
     - In `realdata/risk_overlay.py`, the empirical hazard probability from real-trace data (e.g., BurstGPT high-dispersion failure rate of 26.25%) is applied non-invasively via `step_with_hazard_overlay(env, overlay)` after each step.
     - When an empirical fault is drawn, `gpu.inject_fault()` is invoked with temporary 100% execution certainty for that specific injection, ensuring the true calibrated empirical hazard rate is accurately applied while leaving simulation code completely unmodified.

8. **Task ID Formatting Requirement in `GPUEnvironment._register_task()`**
   - *File & Lines*: `environment.py:433-434`
   - *Issue*: `_register_task` checks `explicit_id = task.id if re.fullmatch(r"video_\d{4}", str(task.id)) else None`. If a task's ID does not match this regex pattern, `task_queue.add_task()` ignores `task.id` and assigns a newly minted sequential ID (e.g. `video_0001`).
   - *Action*: Do not edit `environment.py`. Note that attached tasks will be registered under `video_XXXX` unless matching the regex.
