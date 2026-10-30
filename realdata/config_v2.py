"""
FROZEN v2 CONFIGURATION.

Values marked FROZEN were selected on VALIDATION data only (TRAIN-partition
windows, baseline schedulers, validation seeds) and are fixed before the single
final test run on TEST seeds 100-129.

Do not edit after realdata/PREREGISTRATION_V2.md is committed. Any change after
that point invalidates the pre-registration and must be reported as such.
"""

from __future__ import annotations

import json
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
METRICS_V2 = REPO_ROOT / "results" / "metrics" / "v2"
GRAPHS_V2 = REPO_ROOT / "results" / "graphs" / "v2"
RESULTS_DIR = PACKAGE_ROOT / "results"
TUNING_LOG = REPO_ROOT / "logs" / "v2_tuning_log.csv"

# ───────────────────────────────────────────── data splits (disjoint)
# TRAIN      : agents learn here only
# VALIDATION : every tuning and design decision is made here only
# TEST       : touched exactly once, at the end
TRAIN_PARTITION = "train"          # BurstGPT days 0-30
EVAL_PARTITION = "eval"            # BurstGPT days 31-61

TRAIN_WINDOWS = ["busiest_hour"]                       # TRAIN partition (33,344 requests)
VALIDATION_WINDOWS = ["median_hour", "burstiest_window"]  # TRAIN partition, never trained on
TEST_WINDOWS = ["median_hour", "burstiest_window"]     # EVAL partition — same as v1, comparable

TRAIN_SEEDS = (1, 2, 3)
VALIDATION_SEEDS = (200, 201, 202)
TEST_SEEDS = tuple(range(100, 130))                    # 30 fresh seeds, per instruction

# ───────────────────────────────────────────── schedulers
SCHEDULERS = ["fcfs", "round_robin", "least_loaded", "rl", "marl", "risk_aware_marl"]
LEARNERS = ["rl", "marl", "risk_aware_marl"]
TREATMENT = "risk_aware_marl"

# ───────────────────────────────────────────── experiment conditions
# Reported side by side so the load fix can be separated from the other fixes.
CONDITIONS = ["v1", "v1_corrected", "v2"]

# ───────────────────────────────────────────── decision rule (same as v1)
CI_CONFIDENCE = 0.95
BOOTSTRAP_RESAMPLES = 2000
MIN_WIN_RATE = 0.60

# Throughput floor: a scheduler completing less than this share of the best
# scheduler's mean completed tasks is labelled INFEASIBLE and excluded from
# "strongest baseline" selection. Single-agent RL is reported both ways.
THROUGHPUT_FLOOR = 0.80

PRIMARY_METRICS = ["avg_risk_exposure", "terminal_efc", "crash_count", "quarantine_time"]
SECONDARY_METRICS = ["completed_tasks"]
COMBINED_METRICS = ["risk_per_completed_task", "crashes_per_completed_task"]
HIGHER_IS_BETTER = {"completed_tasks", "throughput"}

# ───────────────────────────────────────────── FROZEN: selected on VALIDATION
_SEL_PATH = METRICS_V2 / "design_selection.json"
_EDGES_PATH = METRICS_V2 / "risk_bucket_edges.json"


def _load(path, key, default):
    try:
        return json.loads(Path(path).read_text())[key]
    except Exception:
        return default


# Episode length — rule: shortest candidate whose max observed risk is within
# 0.02 of the longest candidate (risk plateau), measured on VALIDATION baselines.
EPISODE_STEPS = int(_load(_SEL_PATH, "episode_steps", 600))

# Load levels — rule stated in v2/design_select.py and PREREGISTRATION_V2.md.
LOAD_LEVELS = [float(x) for x in _load(_SEL_PATH, "chosen_load_levels", [0.3, 0.9, 1.8])]

# Exploratory quarantine threshold — largest value on a fixed ladder that
# triggers in >= 30% of VALIDATION episodes under FCFS / Least Loaded only.
# The SHIPPED threshold (0.85) is also always reported.
SHIPPED_QUARANTINE_THRESHOLD = 0.85
EXPLORATORY_QUARANTINE_THRESHOLD = _load(_SEL_PATH, "chosen_quarantine_threshold", 0.60)
QUARANTINE_THRESHOLDS_REPORTED = [SHIPPED_QUARANTINE_THRESHOLD]
if EXPLORATORY_QUARANTINE_THRESHOLD:
    QUARANTINE_THRESHOLDS_REPORTED.append(float(EXPLORATORY_QUARANTINE_THRESHOLD))

# Risk-bucket edges — equal-ish frequency percentiles (50/75/95) of the risk
# distribution observed on TRAIN windows only, under a neutral dispatcher.
V1_RISK_EDGES = [0.25, 0.50, 0.75]
RISK_BUCKET_EDGES = [float(x) for x in _load(_EDGES_PATH, "edges", [0.0614, 0.1165, 0.3047])]

# ───────────────────────────────────────────── FROZEN: training
# Identical budget for every learner (rl, marl, risk_aware_marl).
# Stopping rule: fixed episode count, or early stop if Q-table growth per
# episode stays below PLATEAU_GROWTH for PLATEAU_PATIENCE consecutive episodes.
TRAIN_EPISODES = 20
TRAIN_STEPS_PER_EPISODE = 300
TRAIN_TARGET_RHO = 0.9
EPSILON_DECAY_PER_STEP = 0.995
EPSILON_MIN = 0.05
LEARNING_RATE = 0.01
PLATEAU_GROWTH = 5
PLATEAU_PATIENCE = 3

# ───────────────────────────────────────────── tuning budget
MAX_TUNING_CONFIGS = 20     # hard cap; every configuration tried is logged


def training_config():
    from v2.improvements import TrainingConfig
    return TrainingConfig(
        episodes=TRAIN_EPISODES,
        steps_per_episode=TRAIN_STEPS_PER_EPISODE,
        seeds=TRAIN_SEEDS,
        target_rho=TRAIN_TARGET_RHO,
        epsilon_decay_per_step=EPSILON_DECAY_PER_STEP,
        epsilon_min=EPSILON_MIN,
        learning_rate=LEARNING_RATE,
        plateau_growth=PLATEAU_GROWTH,
        plateau_patience=PLATEAU_PATIENCE,
    )


def summary() -> dict:
    return {
        "splits": {
            "train_windows": TRAIN_WINDOWS, "train_seeds": list(TRAIN_SEEDS),
            "validation_windows": VALIDATION_WINDOWS, "validation_seeds": list(VALIDATION_SEEDS),
            "test_windows": TEST_WINDOWS, "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]],
        },
        "episode_steps": EPISODE_STEPS,
        "load_levels": LOAD_LEVELS,
        "risk_bucket_edges": RISK_BUCKET_EDGES,
        "v1_risk_edges": V1_RISK_EDGES,
        "quarantine_thresholds_reported": QUARANTINE_THRESHOLDS_REPORTED,
        "throughput_floor": THROUGHPUT_FLOOR,
        "training": {
            "episodes": TRAIN_EPISODES, "steps_per_episode": TRAIN_STEPS_PER_EPISODE,
            "epsilon_decay_per_step": EPSILON_DECAY_PER_STEP, "epsilon_min": EPSILON_MIN,
            "learning_rate": LEARNING_RATE,
            "stopping_rule": f"stop at {TRAIN_EPISODES} episodes, or early if Q-growth "
                             f"< {PLATEAU_GROWTH} for {PLATEAU_PATIENCE} consecutive episodes",
        },
        "decision_rule": {
            "ci": CI_CONFIDENCE, "resamples": BOOTSTRAP_RESAMPLES,
            "min_win_rate": MIN_WIN_RATE,
        },
        "max_tuning_configs": MAX_TUNING_CONFIGS,
        "conditions": CONDITIONS,
    }


if __name__ == "__main__":
    print(json.dumps(summary(), indent=2))
