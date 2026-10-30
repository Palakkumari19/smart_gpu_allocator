"""Paired statistical tests without claiming superiority unless evidence holds."""

from __future__ import annotations

import numpy as np


def mean_std(values):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    return float(arr.mean()), float(arr.std(ddof=1) if arr.size > 1 else 0.0)


def paired_diff_ci(treatment, baseline, n_boot=2000, seed=0):
    """Bootstrap 95% CI for mean(treatment - baseline)."""
    diffs = np.asarray(treatment, dtype=float) - np.asarray(baseline, dtype=float)
    if diffs.size == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        np.mean(rng.choice(diffs, size=len(diffs), replace=True))
        for _ in range(n_boot)
    ])
    return float(np.mean(diffs)), float(np.percentile(boot_means, 2.5)), float(np.percentile(boot_means, 97.5))


def win_rate(treatment, baseline, higher_is_better=True):
    t = np.asarray(treatment, dtype=float)
    b = np.asarray(baseline, dtype=float)
    if higher_is_better:
        wins = np.sum(t > b)
    else:
        wins = np.sum(t < b)
    ties = np.sum(t == b)
    return float(wins / max(1, len(t))), int(wins), int(ties), int(len(t) - wins - ties)


def ci_excludes_zero(low, high):
    return (high < 0.0) or (low > 0.0)


def compare_against_baseline(treatment_values, baseline_values, higher_is_better=True, seed=0):
    mean_diff, low, high = paired_diff_ci(treatment_values, baseline_values, seed=seed)
    if not higher_is_better:
        mean_diff, low, high = -mean_diff, -high, -low
    wr, wins, ties, losses = win_rate(treatment_values, baseline_values, higher_is_better=higher_is_better)
    significant_better = low > 0.0
    significant_worse = high < 0.0
    return {
        "mean_diff": mean_diff,
        "ci_low": low,
        "ci_high": high,
        "win_rate": wr,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "significant_better": significant_better,
        "significant_worse": significant_worse,
        "ci_excludes_zero": ci_excludes_zero(low, high),
    }


def marl_consistency_verdict(scenario_comparisons, overall_win_rate, min_win_rate=0.6):
    """Pre-registered rule for 'consistently better'.

    MARL is consistently better only if:
    - overall completed-task difference vs strongest baseline is significantly positive
    - paired win rate is at least min_win_rate
    - MARL is not significantly worse on completed tasks in any scenario
    """
    overall = scenario_comparisons.get("_overall", {})
    scenario_rows = {k: v for k, v in scenario_comparisons.items() if k != "_overall"}
    any_scenario_worse = any(row.get("significant_worse") for row in scenario_rows.values())
    overall_better = bool(overall.get("significant_better"))
    win_ok = overall_win_rate >= min_win_rate

    if overall_better and win_ok and not any_scenario_worse:
        status = "CONSISTENTLY_BETTER"
        summary = (
            "MARL is consistently better on completed tasks vs the strongest baseline "
            "under the pre-registered statistical rule."
        )
    elif overall_better and not any_scenario_worse:
        status = "CONDITIONALLY_BETTER"
        summary = (
            "MARL is significantly better overall, but the paired win rate is too low "
            "to call the advantage consistent."
        )
    elif any_scenario_worse and overall_better:
        status = "MIXED"
        summary = (
            "MARL is better in aggregate but significantly worse in at least one scenario."
        )
    else:
        status = "NOT_ESTABLISHED"
        summary = (
            "The current experiments do not establish that MARL is consistently better "
            "than the strongest baseline."
        )
    return {
        "status": status,
        "summary": summary,
        "overall_significant_better": overall_better,
        "overall_win_rate": overall_win_rate,
        "any_scenario_significantly_worse": any_scenario_worse,
    }
