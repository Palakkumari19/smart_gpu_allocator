"""
Regression test for the v2 offered-load calibration fix.

Asserts that the DELIVERED offered load matches the INTENDED target (ratio ~1.0),
and documents the v1 defect (ratio ~30x) so it cannot silently return.

These tests need the real datasets in realdata/data/raw/; they skip if absent.
"""

import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


@pytest.fixture(scope="module")
def sources():
    pd = pytest.importorskip("pandas")
    from realdata.config import BURSTGPT_WINDOWS_PATH, BURSTGPT_RAW_PATH, UCI_VIDEO_RAW_PATH
    if not (BURSTGPT_RAW_PATH.exists() and UCI_VIDEO_RAW_PATH.exists() and BURSTGPT_WINDOWS_PATH.exists()):
        pytest.skip("real datasets not downloaded; run realdata/fetch_data.py")
    from realdata.loaders import load_burstgpt_data, load_uci_video_data
    from realdata.windows import TraceWindowsSummary
    ws = TraceWindowsSummary.from_json(BURSTGPT_WINDOWS_PATH)
    return load_burstgpt_data(), load_uci_video_data(), ws


EPISODE_STEPS = 120
TARGETS = [0.6, 0.9, 1.2]
WINDOWS = ["median_hour", "burstiest_window"]


@pytest.mark.parametrize("win_key", WINDOWS)
@pytest.mark.parametrize("target_rho", TARGETS)
def test_delivered_load_matches_target(sources, win_key, target_rho):
    """The corrected calibration must deliver the offered load it promises."""
    from v2.load_model import build_calibrated_trace, measure_offered_rho

    df_burst, df_video, ws = sources
    window = ws.eval_windows[win_key]

    trace, cal = build_calibrated_trace(
        window=window, episode_steps=EPISODE_STEPS, target_rho=target_rho,
        df_burst=df_burst, df_video=df_video, seed=7,
    )
    delivered = measure_offered_rho(trace)
    ratio = delivered / target_rho

    assert 0.80 <= ratio <= 1.25, (
        f"{win_key} rho={target_rho}: delivered {delivered:.3f} "
        f"(ratio {ratio:.2f}x) — calibration is off"
    )


@pytest.mark.parametrize("win_key", WINDOWS)
def test_v1_defect_is_reproduced_and_large(sources, win_key):
    """
    Documents the v1 bug: the shipped scale over-delivers by ~30x because it is
    computed for the window's 3600 steps but applied over a 120-step episode.
    If this ever drops to ~1.0, the legacy path has been changed and the v1
    reproduction condition is no longer faithful.
    """
    from v2.load_model import build_legacy_trace, measure_offered_rho

    df_burst, df_video, ws = sources
    window = ws.eval_windows[win_key]

    trace, _scale = build_legacy_trace(
        window=window, episode_steps=EPISODE_STEPS, target_rho=0.9,
        df_burst=df_burst, df_video=df_video, seed=7,
    )
    delivered = measure_offered_rho(trace)
    assert delivered > 5.0, (
        f"{win_key}: legacy path delivered rho={delivered:.2f}; expected heavy "
        "over-delivery (v1 defect). Legacy behaviour appears to have changed."
    )


@pytest.mark.parametrize("win_key", WINDOWS)
def test_task_count_ratio_is_about_thirty(sources, win_key):
    """The defect is a pure step-count mismatch: 3600 assumed / 120 actual = 30x."""
    from v2.load_model import build_legacy_trace

    df_burst, df_video, ws = sources
    window = ws.eval_windows[win_key]

    trace, scale = build_legacy_trace(
        window=window, episode_steps=EPISODE_STEPS, target_rho=0.9,
        df_burst=df_burst, df_video=df_video, seed=7,
    )
    delivered_tasks = sum(len(s) for s in trace)
    intended_tasks = window.request_count * scale * (EPISODE_STEPS / window.duration_seconds)
    ratio = delivered_tasks / max(1e-9, intended_tasks)
    assert 25.0 <= ratio <= 35.0, f"{win_key}: task over-delivery {ratio:.1f}x, expected ~30x"


def test_calibration_scales_monotonically(sources):
    """Higher target load must produce a strictly higher delivered load."""
    from v2.load_model import build_calibrated_trace, measure_offered_rho

    df_burst, df_video, ws = sources
    window = ws.eval_windows["median_hour"]

    delivered = []
    for rho in TARGETS:
        trace, _ = build_calibrated_trace(
            window=window, episode_steps=EPISODE_STEPS, target_rho=rho,
            df_burst=df_burst, df_video=df_video, seed=11,
        )
        delivered.append(measure_offered_rho(trace))

    assert delivered[0] < delivered[1] < delivered[2], f"not monotonic: {delivered}"
