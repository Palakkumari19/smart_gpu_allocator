import os
from collections import Counter

from backend.video_analyzer import VideoAnalyzer
from environment import GPUEnvironment
from workload.task_generator import VideoTask, VideoProfile


VIDEO_DIR = "sample_videos"
MAX_STEPS = 500


def discover_videos():
    """Find supported real video files."""
    if not os.path.isdir(VIDEO_DIR):
        raise FileNotFoundError(
            f"Video directory not found: {VIDEO_DIR}"
        )

    videos = []

    for filename in sorted(os.listdir(VIDEO_DIR)):
        extension = os.path.splitext(filename)[1].lower()

        if extension in VideoAnalyzer.SUPPORTED_FORMATS:
            videos.append(os.path.join(VIDEO_DIR, filename))

    return videos


def create_real_video_task(video_path, video_info):
    """
    Convert analyzed real-video metadata into an AETHERGRID VideoTask.

    The task is still executed by the existing simulator/MARL system.
    The actual video file is only the workload source.
    """

    profile_name = VideoAnalyzer.classify_workload(video_info)
    memory_required, execution_time = VideoAnalyzer.estimate_resources(
        video_info
    )

    profile = VideoProfile[profile_name]

    # VideoTask normally gets created by TaskGenerator.
    # We create one directly here because this workload comes from
    # real video metadata rather than synthetic generation.
    task = VideoTask.__new__(VideoTask)

    # Core task fields
    task.task_type = "Video"
    task.profile = profile
    task.profile_name = profile.value

    # Real-video metadata
    task.filename = video_info["filename"]
    task.file_path = video_path
    task.width = video_info["width"]
    task.height = video_info["height"]
    task.resolution = video_info["resolution"]
    task.fps = video_info["fps"]
    task.duration = video_info["duration"]
    task.frames = video_info["frames"]
    task.codec = video_info["codec"]
    task.bitrate = video_info["bitrate"]

    # AETHERGRID execution properties
    task.memory_required = memory_required
    task.execution_time = execution_time

    # Use a deterministic ID derived from the filename.
    # DynamicTaskQueue will register the task and normalize its ID.
    task.id = f"real_video_{os.path.splitext(video_info['filename'])[0]}"

    task.description = (
        f"{profile.value} Real Video ({video_info['resolution']})"
    )

    # Fields used by GPU execution / queue management
    task.execution_progress = 0
    task.preempted = False
    task.created_time = None

    return task


def analyze_videos(video_paths):
    """Analyze all discovered videos and create AETHERGRID tasks."""

    analyzed = []

    for video_path in video_paths:
        try:
            info = VideoAnalyzer.get_video_info(video_path)

            task = create_real_video_task(video_path, info)

            analyzed.append(
                {
                    "path": video_path,
                    "info": info,
                    "task": task,
                }
            )

        except Exception as exc:
            print(
                f"WARNING: Could not analyze "
                f"{os.path.basename(video_path)}: {exc}"
            )

    return analyzed


def build_workload_trace(tasks):
    """
    Put all real-video tasks into the first workload-trace step.

    GPUEnvironment.step() will then register these tasks and send them
    through the configured MARL scheduler.
    """

    return [tasks]


def get_queue_counts(env):
    """Return current queue/execution counts."""

    counts = env.task_queue.get_task_counts()

    queued = len(
        env.task_queue.get_tasks_by_status(
            env.task_queue.TaskStatus.QUEUED
        )
    ) if hasattr(env.task_queue, "TaskStatus") else 0

    running = sum(
        len(gpu.running_tasks)
        for gpu in env.gpus
    )

    gpu_queued = sum(
        len(gpu.task_queue)
        for gpu in env.gpus
    )

    completed = sum(
        gpu.completed_tasks
        for gpu in env.gpus
    )

    return {
        "registered": counts.get("total", 0),
        "queued": gpu_queued,
        "running": running,
        "completed": completed,
    }


def print_workload_table(workloads):
    print("\nVIDEO WORKLOADS")
    print("-" * 95)

    print(
        f"{'Video':34} "
        f"{'Resolution':12} "
        f"{'FPS':6} "
        f"{'Profile':9} "
        f"{'Memory':8} "
        f"{'Exec':6}"
    )

    print("-" * 95)

    for item in workloads:
        info = item["info"]
        task = item["task"]

        print(
            f"{info['filename'][:34]:34} "
            f"{info['resolution'][:12]:12} "
            f"{info['fps']:6.1f} "
            f"{task.profile_name:9} "
            f"{task.memory_required:7.1f}GB "
            f"{task.execution_time:5}"
        )


def print_gpu_state(env):
    for gpu in env.gpus:
        print(
            f"GPU {gpu.id}: "
            f"completed={gpu.completed_tasks}, "
            f"running={len(gpu.running_tasks)}, "
            f"queued={len(gpu.task_queue)}, "
            f"memory={gpu.current_memory:.1f}/{gpu.memory_capacity:.1f}GB"
        )


def print_marl_state(env):
    """Print basic MARL state without assuming extra fields."""

    print("\nMARL STATE")
    print("-" * 60)

    if not hasattr(env, "marl_manager"):
        print("MARL manager not available.")
        return

    agents = getattr(env.marl_manager, "agents", [])

    print(f"Agents: {len(agents)}")

    for index, agent in enumerate(agents):
        epsilon = getattr(agent, "epsilon", None)
        q_table = getattr(agent, "q_table", None)

        if epsilon is not None:
            epsilon_text = f"{epsilon:.3f}"
        else:
            epsilon_text = "N/A"

        if q_table is not None:
            q_entries = len(q_table)
        else:
            q_entries = "N/A"

        print(
            f"Agent {index}: "
            f"epsilon={epsilon_text}, "
            f"Q-entries={q_entries}"
        )


def main():
    print("=" * 80)
    print("AETHERGRID — PHASE 4 REAL VIDEO + MARL EXECUTION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # 1. Discover real videos
    # ------------------------------------------------------------------

    video_paths = discover_videos()

    if not video_paths:
        print(f"\nNo supported videos found in '{VIDEO_DIR}'.")
        return

    print(f"\nVideos discovered : {len(video_paths)}")

    # ------------------------------------------------------------------
    # 2. Analyze real videos
    # ------------------------------------------------------------------

    workloads = analyze_videos(video_paths)

    print(f"Videos analyzed   : {len(workloads)}")

    if not workloads:
        print("\nNo videos could be analyzed.")
        return

    # ------------------------------------------------------------------
    # 3. Show real-video workload mapping
    # ------------------------------------------------------------------

    print_workload_table(workloads)

    distribution = Counter(
        item["task"].profile_name.upper()
        for item in workloads
    )

    print("\nWORKLOAD DISTRIBUTION")
    print("-" * 40)
    print(f"LIGHT     : {distribution.get('LIGHT', 0)}")
    print(f"MEDIUM    : {distribution.get('MEDIUM', 0)}")
    print(f"HEAVY     : {distribution.get('HEAVY', 0)}")
    print(f"EXTREME   : {distribution.get('EXTREME', 0)}")

    # ------------------------------------------------------------------
    # 4. Create the actual AETHERGRID environment
    # ------------------------------------------------------------------

    print("\n" + "=" * 80)
    print("INITIALIZING AETHERGRID MARL ENVIRONMENT")
    print("=" * 80)

    env = GPUEnvironment(scheduler="marl")

    # Disable random fault injection for this integration demo.
    # Phase 4 is validating execution management, not fault injection.
    if hasattr(env, "fault_injector"):
        env.fault_injector.enabled = False

    # Evaluation mode:
    # MARL still makes the scheduling decisions, but we don't change
    # its learned policy while running the demonstration.
    if hasattr(env, "set_eval_mode"):
        env.set_eval_mode()

    tasks = [
        item["task"]
        for item in workloads
    ]

    # ------------------------------------------------------------------
    # 5. Attach real video workload to the existing environment
    # ------------------------------------------------------------------

    workload_trace = build_workload_trace(tasks)

    env.attach_workload_trace(workload_trace)

    print(f"\nReal video tasks loaded : {len(tasks)}")
    print("Scheduler              : MARL")
    print("GPU count              :", len(env.gpus))

    # ------------------------------------------------------------------
    # 6. Execute through GPUEnvironment.step()
    # ------------------------------------------------------------------

    print("\nGPU EXECUTION")
    print("-" * 80)

    peak_parallel = 0
    completed_previous = 0

    for step in range(1, MAX_STEPS + 1):

        env.step()

        # Count tasks that actually STARTED during this GPU processing cycle.
        # This captures short workloads that may start and finish within
        # the same env.step() call.
        started_now = sum(
            getattr(gpu, "last_started_tasks", 0)
            for gpu in env.gpus
        )

        # This is the number still running after the step.
        running_now = sum(
            len(gpu.running_tasks)
            for gpu in env.gpus
        )

        completed_now = sum(
            gpu.completed_tasks
            for gpu in env.gpus
        )

        # Peak parallelism is based on actual simultaneous task starts,
        # not only tasks that remain in running_tasks after the step.
        peak_parallel = max(
            peak_parallel,
            started_now
        )

        # Print useful execution information.
        if (
            step <= 10
            or started_now > 0
            or running_now != 0
            or completed_now != completed_previous
        ):
            print(
                f"\nStep {step:3d} | "
                f"Started={started_now:2d} | "
                f"Running={running_now:2d} | "
                f"Completed={completed_now:2d}"
            )

            print_gpu_state(env)

        completed_previous = completed_now

        if completed_now >= len(tasks):
            break

    # ------------------------------------------------------------------
    # 7. Final execution results
    # ------------------------------------------------------------------

    completed = sum(
        gpu.completed_tasks
        for gpu in env.gpus
    )

    queued = sum(
        len(gpu.task_queue)
        for gpu in env.gpus
    )

    running = sum(
        len(gpu.running_tasks)
        for gpu in env.gpus
    )

    preemptions = sum(
        gpu.total_preemptions
        for gpu in env.gpus
    )

    assigned = getattr(
        env,
        "successful_assignments",
        "N/A"
    )

    print("\n" + "=" * 80)
    print("EXECUTION RESULTS")
    print("=" * 80)

    print(f"Total submitted : {len(tasks)}")
    print(f"Assigned        : {assigned}")
    print(f"Completed       : {completed}")
    print(f"Peak parallel   : {peak_parallel}")
    print(f"Parallel capacity: {len(env.gpus) * max(gpu.max_parallel_tasks for gpu in env.gpus)}")
    print(f"Preemptions     : {preemptions}")
    print(f"Simulation steps: {step}")

    print("\nFINAL GPU STATE")
    print("-" * 60)
    print_gpu_state(env)

    print("\nFINAL STATE")
    print("-" * 60)
    print(f"Pending         : {max(0, len(tasks) - completed - queued - running)}")
    print(f"Queued          : {queued}")
    print(f"Running         : {running}")
    print(f"Completed       : {completed}")

    print_marl_state(env)

    print("\n" + "=" * 80)

    if (
        completed == len(tasks)
        and queued == 0
        and running == 0
    ):
        print("RESULT: SUCCESS — all real video workloads completed.")
    else:
        print("RESULT: INCOMPLETE — some workloads remain.")

    print("=" * 80)


if __name__ == "__main__":
    main()