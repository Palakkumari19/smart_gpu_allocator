from benchmark_schedulers import benchmark_scheduler


def test_non_marl_benchmark_does_not_report_marl_metrics():
    result = benchmark_scheduler("random", steps=1)

    assert "MARL_Global_Reward" not in result
    assert "MARL_Success_Rate" not in result
