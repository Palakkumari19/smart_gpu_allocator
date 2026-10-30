from copy import deepcopy

from evaluation.stats import compare_against_baseline, marl_consistency_verdict
from evaluation.workload_trace import generate_workload_trace, trace_task_count
from environment import GPUEnvironment
from evaluation.scientific_benchmark import run_episode


def test_traces_are_deterministic_and_isolated():
    a = generate_workload_trace("mixed", 8, seed=7)
    b = generate_workload_trace("mixed", 8, seed=7)
    c = generate_workload_trace("mixed", 8, seed=8)

    assert trace_task_count(a) == trace_task_count(b)
    assert [len(step) for step in a] == [len(step) for step in b]
    assert [task.memory_required for step in a for task in step] == [
        task.memory_required for step in b for task in step
    ]
    assert [task.memory_required for step in a for task in step] != [
        task.memory_required for step in c for task in step
    ]


def test_identical_traces_yield_identical_least_loaded_results():
    trace = generate_workload_trace("heavy", 12, seed=3)
    _, first = run_episode("least_loaded", deepcopy(trace), learning=False)
    _, second = run_episode("least_loaded", deepcopy(trace), learning=False)
    assert first["completed"] == second["completed"]
    assert first["crash_events"] == second["crash_events"]
    assert first["failed_assignments"] == second["failed_assignments"]


def test_eval_mode_does_not_update_marl_q_values():
    trace = generate_workload_trace("mixed", 8, seed=11)
    train_env, _ = run_episode("marl", deepcopy(trace), learning=True)
    snapshot = train_env.marl_manager.snapshot_q_tables()
    eval_env = GPUEnvironment(scheduler="marl")
    eval_env.enable_deterministic_physics()
    eval_env.attach_workload_trace(deepcopy(trace))
    eval_env.marl_manager.restore_q_tables(deepcopy(snapshot))
    eval_env.set_eval_mode()
    before = eval_env.marl_manager.snapshot_q_tables()
    for _ in range(len(trace)):
        eval_env.step()
    after = eval_env.marl_manager.snapshot_q_tables()
    assert before[0].keys() == after[0].keys()
    for state in before[0]:
        assert list(before[0][state]) == list(after[0][state])


def test_metrics_include_crash_events_not_just_snapshot():
    env = GPUEnvironment(scheduler="least_loaded")
    env.enable_deterministic_physics()
    env.attach_workload_trace(generate_workload_trace("peak", 5, seed=1))
    for _ in range(5):
        env.step()
    metrics = env.get_metrics()
    assert "total_crash_events" in metrics
    assert "failed_assignments" in metrics
    assert "successful_assignments" in metrics


def test_verdict_does_not_claim_superiority_without_evidence():
    weak = marl_consistency_verdict(
        {
            "_overall": {"significant_better": False, "significant_worse": False, "win_rate": 0.4},
            "mixed": {"significant_better": False, "significant_worse": False},
        },
        overall_win_rate=0.4,
    )
    assert weak["status"] == "NOT_ESTABLISHED"

    strong = marl_consistency_verdict(
        {
            "_overall": {"significant_better": True, "significant_worse": False, "win_rate": 0.75},
            "mixed": {"significant_better": True, "significant_worse": False},
            "heavy": {"significant_better": True, "significant_worse": False},
        },
        overall_win_rate=0.75,
    )
    assert strong["status"] == "CONSISTENTLY_BETTER"


def test_comparison_uses_paired_differences():
    marl = [10, 12, 11, 13]
    baseline = [9, 9, 10, 10]
    result = compare_against_baseline(marl, baseline, higher_is_better=True)
    assert result["mean_diff"] > 0
    assert result["win_rate"] == 1.0
    assert result["significant_better"]
