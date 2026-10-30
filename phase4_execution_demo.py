import random
from collections import Counter

from environment import GPUEnvironment

NUM_VIDEOS = random.randint(15, 30)
MAX_STEPS = 100


def task_id(task):
    """Return a readable task identifier."""
    return str(
        getattr(
            task,
            "task_id",
            getattr(task, "id", "unknown")
        )
    )


def main():

    env = GPUEnvironment(scheduler="least_loaded")

    # Disable random fault injection so this demo focuses
    # on execution management and parallelism.
    if hasattr(env, "fault_injector"):
        env.fault_injector.enabled = False

    # ---------------------------------------------------------
    # Generate workload
    # ---------------------------------------------------------

    generated_tasks = []

    for _ in range(NUM_VIDEOS):
        task = env.generate_task()

        # Keep execution duration fixed for predictable
        # Phase 4 execution validation.

        generated_tasks.append(task)

    pending_tasks = list(generated_tasks)

    assignments = 0

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    peak_queued = 0
    peak_running = 0
    peak_parallelism = 0

    total_queued_samples = 0
    total_running_samples = 0
    samples = 0

    # ---------------------------------------------------------
    # Timeline
    # ---------------------------------------------------------

    timeline = []

    for step in range(1, MAX_STEPS + 1):

        step_events = []

        # =====================================================
        # 1. Assign pending workloads
        # =====================================================

        remaining_tasks = []

        for task in pending_tasks:
            assigned = False

            for gpu in env.gpus:

                active_work = (
                    len(gpu.task_queue)
                    + len(gpu.running_tasks)
                )

                # Keep the GPU execution pipeline within
                # its configured parallel capacity.
                if active_work >= gpu.max_parallel_tasks:
                    continue

                if env._try_assign(gpu, task):
                    assignments += 1
                    assigned = True

                    step_events.append(
                        f"{task_id(task)} assigned to GPU {gpu.id}"
                    )

                    break

            if not assigned:
                remaining_tasks.append(task)

        pending_tasks = remaining_tasks

        # =====================================================
        # 2. Snapshot state BEFORE processing
        # =====================================================

        before_state = {}

        for gpu in env.gpus:
            before_state[gpu.id] = {
                "running": list(gpu.running_tasks),
                "preempted": list(gpu.preempted_tasks),
            }

        # =====================================================
        # 3. Process all GPUs
        # =====================================================

        for gpu in env.gpus:
            gpu.process_tasks()

        # =====================================================
        # 4. Detect execution lifecycle events
        # =====================================================

        for gpu in env.gpus:

            before_running = {
                task_id(task)
                for task in before_state[gpu.id]["running"]
            }

            after_running = {
                task_id(task)
                for task in gpu.running_tasks
            }

            before_preempted = {
                task_id(task)
                for task in before_state[gpu.id]["preempted"]
            }

            after_preempted = {
                task_id(task)
                for task in gpu.preempted_tasks
            }

            # -------------------------------------------------
            # Tasks that started running
            # -------------------------------------------------

            started = [
                task_id(task)
                for task in gpu.running_tasks
                if task_id(task) not in before_running
            ]

            for task in started:
                step_events.append(
                    f"{task} started running on GPU {gpu.id}"
                )

            # -------------------------------------------------
            # Tasks that were preempted
            # -------------------------------------------------

            preempted = [
                task
                for task in before_running
                if task in after_preempted
            ]

            for task in preempted:
                step_events.append(
                    f"{task} preempted on GPU {gpu.id}"
                )

            # -------------------------------------------------
            # Tasks that resumed
            # -------------------------------------------------

            resumed = [
                task
                for task in after_running
                if task in before_preempted
            ]

            for task in resumed:
                step_events.append(
                    f"{task} resumed on GPU {gpu.id}"
                )

            # -------------------------------------------------
            # Tasks that completed
            # -------------------------------------------------

            completed_this_step = [
                task
                for task in before_running
                if (
                    task not in after_running
                    and task not in after_preempted
                )
            ]

            for task in completed_this_step:
                step_events.append(
                    f"{task} completed on GPU {gpu.id}"
                )

        # =====================================================
        # 5. Calculate current global state
        # =====================================================

        # Tasks not yet assigned to any GPU.
        pending = len(pending_tasks)

        # Tasks assigned to a GPU but waiting for execution.
        queued_on_gpus = sum(
            len(gpu.task_queue)
            for gpu in env.gpus
        )

        # Tasks currently executing.
        running = sum(
            len(gpu.running_tasks)
            for gpu in env.gpus
        )

        # Tasks that have completed.
        completed = sum(
            gpu.completed_tasks
            for gpu in env.gpus
        )

        # Everything that has not completed yet.
        total_outstanding = (
            pending
            + queued_on_gpus
            + running
        )

        # This is kept as a combined queue metric for the
        # overall summary.
        queued = pending + queued_on_gpus

        # =====================================================
        # 6. Update metrics
        # =====================================================

        peak_queued = max(
            peak_queued,
            queued
        )

        peak_running = max(
            peak_running,
            running
        )

        peak_parallelism = max(
            peak_parallelism,
            running
        )

        total_queued_samples += queued
        total_running_samples += running
        samples += 1

        # =====================================================
        # 7. Store timeline snapshot
        # =====================================================

        gpu_snapshots = []

        for gpu in env.gpus:
            gpu_snapshots.append({
                "id": gpu.id,
                "queued": len(gpu.task_queue),
                "running": len(gpu.running_tasks),
                "completed": gpu.completed_tasks,
                "memory": gpu.current_memory,
                "temperature": gpu.temperature,
            })

        timeline.append({
            "step": step,
            "events": step_events,
            "pending": pending,
            "gpu_queued": queued_on_gpus,
            "running": running,
            "completed": completed,
            "total_outstanding": total_outstanding,
            "gpus": gpu_snapshots,
        })

        # =====================================================
        # 8. Stop when everything is complete
        # =====================================================

        if (
            completed == NUM_VIDEOS
            and total_outstanding == 0
        ):
            break

    # ---------------------------------------------------------
    # Final workload statistics
    # ---------------------------------------------------------

    profiles = Counter(
        task.profile_name
        for task in generated_tasks
    )

    total_memory = sum(
        task.memory_required
        for task in generated_tasks
    )

    average_memory = (
        total_memory / len(generated_tasks)
    )

    average_execution_time = (
        sum(
            task.execution_time
            for task in generated_tasks
        )
        / len(generated_tasks)
    )

    # ---------------------------------------------------------
    # Final execution state
    # ---------------------------------------------------------

    completed = sum(
        gpu.completed_tasks
        for gpu in env.gpus
    )

    preemptions = sum(
        gpu.total_preemptions
        for gpu in env.gpus
    )

    queued_on_gpus = sum(
        len(gpu.task_queue)
        for gpu in env.gpus
    )

    running = sum(
        len(gpu.running_tasks)
        for gpu in env.gpus
    )

    pending = len(pending_tasks)

    total_outstanding = (
        pending
        + queued_on_gpus
        + running
    )

    unfinished = NUM_VIDEOS - completed

    # =========================================================
    # OUTPUT
    # =========================================================

    print("=" * 70)
    print("        AETHERGRID — PHASE 4 EXECUTION DEMO")
    print("=" * 70)

    # ---------------------------------------------------------
    # Workload
    # ---------------------------------------------------------

    print("\nWORKLOAD")
    print("-" * 70)

    print(f"Videos generated           : {NUM_VIDEOS}")
    print(
        f"Light videos               : "
        f"{profiles.get('Light', 0)}"
    )
    print(
        f"Medium videos              : "
        f"{profiles.get('Medium', 0)}"
    )
    print(
        f"Heavy videos               : "
        f"{profiles.get('Heavy', 0)}"
    )
    print(
        f"Extreme videos             : "
        f"{profiles.get('Extreme', 0)}"
    )

    # ---------------------------------------------------------
    # Resource demand
    # ---------------------------------------------------------

    print("\nVIDEO RESOURCE DEMAND")
    print("-" * 70)

    print(
        f"Total memory requested     : "
        f"{total_memory:.2f} GB"
    )

    print(
        f"Average memory / video     : "
        f"{average_memory:.2f} GB"
    )

    print(
        f"Average execution time     : "
        f"{average_execution_time:.2f} steps"
    )

    # =========================================================
    # Execution timeline
    # =========================================================

    print("\nEXECUTION TIMELINE")
    print("=" * 70)

    for entry in timeline:

        print(
            f"\nSTEP {entry['step']}"
        )

        print("-" * 70)

        # -----------------------------------------------------
        # Lifecycle events
        # -----------------------------------------------------

        if entry["events"]:
            for event in entry["events"]:
                print(f"  • {event}")
        else:
            print("  • No task lifecycle events")

        # -----------------------------------------------------
        # Global state
        # -----------------------------------------------------

        print(
            f"\n  Pending / unassigned : "
            f"{entry['pending']}"
        )

        print(
            f"  GPU queued           : "
            f"{entry['gpu_queued']}"
        )

        print(
            f"  Running              : "
            f"{entry['running']}"
        )

        print(
            f"  Completed            : "
            f"{entry['completed']}"
        )

        print(
            f"  Total outstanding    : "
            f"{entry['total_outstanding']}"
        )

        # -----------------------------------------------------
        # GPU state
        # -----------------------------------------------------

        print("\n  GPU state:")

        for gpu in entry["gpus"]:
            print(
                f"    GPU {gpu['id']}: "
                f"queued={gpu['queued']}, "
                f"running={gpu['running']}, "
                f"completed={gpu['completed']}, "
                f"memory={gpu['memory']:.2f} GB, "
                f"temp={gpu['temperature']:.1f}°C"
            )

    # =========================================================
    # Execution summary
    # =========================================================

    print("\nEXECUTION SUMMARY")
    print("-" * 70)

    print(
        f"Successfully assigned      : "
        f"{assignments}"
    )

    print(
        f"Completed                  : "
        f"{completed}"
    )

    print(
        f"Preemptions                : "
        f"{preemptions}"
    )

    print(
        f"Simulation steps           : "
        f"{samples}"
    )

    # =========================================================
    # Queue / parallelism
    # =========================================================

    print("\nQUEUE / PARALLELISM")
    print("-" * 70)

    print(
        f"Peak queued tasks          : "
        f"{peak_queued}"
    )

    print(
        f"Peak running tasks         : "
        f"{peak_running}"
    )

    print(
        f"Peak parallelism           : "
        f"{peak_parallelism}"
    )

    print(
        f"Average queued tasks       : "
        f"{total_queued_samples / samples:.2f}"
    )

    print(
        f"Average running tasks      : "
        f"{total_running_samples / samples:.2f}"
    )

    # =========================================================
    # Final state
    # =========================================================

    print("\nFINAL STATE")
    print("-" * 70)

    print(
        f"Pending / unassigned       : "
        f"{pending}"
    )

    print(
        f"GPU queued                 : "
        f"{queued_on_gpus}"
    )

    print(
        f"Running                    : "
        f"{running}"
    )

    print(
        f"Completed                  : "
        f"{completed}"
    )

    print(
        f"Unfinished                 : "
        f"{unfinished}"
    )

    # =========================================================
    # Final GPU state
    # =========================================================

    print("\nGPU FINAL STATE")
    print("-" * 70)

    for gpu in env.gpus:
        print(
            f"GPU {gpu.id}: "
            f"completed={gpu.completed_tasks}, "
            f"queued={len(gpu.task_queue)}, "
            f"running={len(gpu.running_tasks)}, "
            f"memory={gpu.current_memory:.2f}/"
            f"{gpu.memory_capacity:.2f} GB, "
            f"temperature={gpu.temperature:.1f}°C, "
            f"preemptions={gpu.total_preemptions}"
        )

    print("\n" + "=" * 70)

    # =========================================================
    # Final result
    # =========================================================

    if (
        completed == NUM_VIDEOS
        and total_outstanding == 0
    ):
        print(
            "RESULT: ALL VIDEO WORKLOADS COMPLETED"
        )
    else:
        print(
            "RESULT: EXECUTION INCOMPLETE"
        )

    print("=" * 70)


if __name__ == "__main__":
    main()