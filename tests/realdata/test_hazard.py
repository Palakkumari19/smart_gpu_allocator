"""
Unit tests for realdata/hazard.py: Empirical hazard curves and confidence intervals.
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from realdata.hazard import (
    wilson_score_interval,
    HazardBin,
    HazardCurve,
    HazardCatalog,
    estimate_burstgpt_dispersion_hazard,
    estimate_burstgpt_offered_load_hazard,
    estimate_alibaba_hazard,
    build_hazard_catalog,
)


def test_wilson_score_interval_properties():
    """Verify statistical properties and boundary guarantees of Wilson score interval."""
    # 0 trials edge case
    assert wilson_score_interval(0, 0) == (0.0, 0.0)

    # 0 successes out of 100 trials: lower bound should be 0.0, upper > 0
    ci_l, ci_u = wilson_score_interval(0, 100)
    assert ci_l == 0.0
    assert 0.0 < ci_u <= 0.05

    # 100 successes out of 100 trials: upper bound should be 1.0, lower < 1
    ci_l, ci_u = wilson_score_interval(100, 100)
    assert ci_u == 1.0
    assert 0.95 <= ci_l < 1.0

    # 50 out of 100 trials (p=0.5): symmetric around 0.5
    ci_l, ci_u = wilson_score_interval(50, 100)
    assert ci_l < 0.5 < ci_u
    assert abs((0.5 - ci_l) - (ci_u - 0.5)) < 1e-4

    # Monotonicity with sample size (larger N yields narrower interval)
    w_small = wilson_score_interval(10, 100)
    w_large = wilson_score_interval(100, 1000)
    assert (w_large[1] - w_large[0]) < (w_small[1] - w_small[0])


def test_hazard_curve_lookup_and_fallback():
    """Verify HazardCurve bin matching, out-of-bounds clamping, and fallback logic."""
    bins = [
        HazardBin(
            bin_index=0,
            bin_label="[0.0, 1.0)",
            range_min=0.0,
            range_max=1.0,
            sample_size=1000,
            failure_count=10,
            failure_probability=0.01,
            ci_lower=0.005,
            ci_upper=0.018,
            too_little_data=False,
            fallback_probability=0.02,
        ),
        HazardBin(
            bin_index=1,
            bin_label="[1.0, 5.0)",
            range_min=1.0,
            range_max=5.0,
            sample_size=10,  # Below min_samples threshold!
            failure_count=5,
            failure_probability=0.50,
            ci_lower=0.23,
            ci_upper=0.76,
            too_little_data=True,
            fallback_probability=0.02,
        ),
    ]

    curve = HazardCurve(
        name="test_curve",
        metric_name="dispersion",
        unit="index",
        global_sample_size=1010,
        global_failure_count=15,
        global_failure_rate=0.02,
        min_samples_threshold=100,
        bins=bins,
    )

    # In-bounds query for bin 0 (sufficient data)
    assert curve.lookup_probability(0.5) == 0.01

    # In-bounds query for bin 1 (insufficient data -> fallback)
    assert curve.lookup_probability(2.5, use_fallback=True) == 0.02
    assert curve.lookup_probability(2.5, use_fallback=False) == 0.50

    # Below range_min: clamps to first bin
    assert curve.lookup_bin(-1.0).bin_index == 0
    # Above range_max: clamps to last bin
    assert curve.lookup_bin(10.0).bin_index == 1


def test_estimate_burstgpt_dispersion_hazard_synthetic():
    """Test dispersion hazard curve estimation on a small synthetic DataFrame."""
    # Synthetic dataset with 2 hours:
    # Hour 0: 100 requests, 1 fail, uniform arrivals (low dispersion)
    # Hour 1: 100 requests, 20 fails, clustered arrivals (high dispersion)
    data = []
    # Hour 0: 100 requests at seconds 0, 1, 2...
    for i in range(100):
        data.append({
            "Timestamp": float(i * 30),  # Spread evenly
            "Model": "gpt-4",
            "Request tokens": 100,
            "Response tokens": 50 if i > 0 else 0,  # 1 failure
            "Total tokens": 150,
        })
    # Hour 1: 100 requests clustered in first 10 seconds (high dispersion)
    for i in range(100):
        data.append({
            "Timestamp": 3600.0 + float(i % 10),
            "Model": "gpt-4",
            "Request tokens": 100,
            "Response tokens": 0 if i < 20 else 50,  # 20 failures
            "Total tokens": 150,
        })

    df = pd.DataFrame(data)
    curve = estimate_burstgpt_dispersion_hazard(
        df, bins=[0.0, 1.0, 50.0], min_samples=50
    )

    assert curve.name == "burstgpt_dispersion"
    assert curve.global_sample_size == 200
    assert curve.global_failure_count == 21
    assert len(curve.bins) == 2

    for b in curve.bins:
        assert b.sample_size >= 0
        assert 0.0 <= b.failure_probability <= 1.0
        assert b.ci_lower <= b.failure_probability <= b.ci_upper
        assert not b.too_little_data  # Each bin has >= 100 samples >= min_samples 50


def test_estimate_alibaba_hazard_synthetic(tmp_path: Path):
    """Test Alibaba PAI sample estimator with synthetic CSV and missing file handling."""
    # 1. Non-existent file returns None
    assert estimate_alibaba_hazard(tmp_path / "nonexistent.csv") is None

    # 2. Synthetic Alibaba file
    alibaba_file = tmp_path / "alibaba_sample.csv"
    alibaba_df = pd.DataFrame({
        "job_name": [f"job_{i}" for i in range(100)],
        "plan_gpu": [1.0] * 50 + [4.0] * 50,
        "status": ["Terminated"] * 45 + ["Failed"] * 5 + ["Terminated"] * 30 + ["Failed"] * 20,
    })
    alibaba_df.to_csv(alibaba_file, index=False)

    curve = estimate_alibaba_hazard(
        sample_path=alibaba_file, bins=[1.0, 2.0, 8.0], min_samples=30
    )

    assert curve is not None
    assert curve.name == "alibaba_gpu_size"
    assert curve.global_sample_size == 100
    assert curve.global_failure_count == 25
    assert len(curve.bins) == 2

    # 1 GPU bin: 5 fails out of 50 = 10%
    b1 = curve.lookup_bin(1.0)
    assert b1.sample_size == 50
    assert b1.failure_count == 5
    assert abs(b1.failure_probability - 0.10) < 1e-4

    # 4 GPU bin: 20 fails out of 50 = 40%
    b4 = curve.lookup_bin(4.0)
    assert b4.sample_size == 50
    assert b4.failure_count == 20
    assert abs(b4.failure_probability - 0.40) < 1e-4


def test_hazard_catalog_json_roundtrip(tmp_path: Path):
    """Verify HazardCatalog serialization and deserialization via JSON."""
    b = HazardBin(
        bin_index=0,
        bin_label="[0, 10)",
        range_min=0.0,
        range_max=10.0,
        sample_size=500,
        failure_count=50,
        failure_probability=0.10,
        ci_lower=0.076,
        ci_upper=0.129,
        too_little_data=False,
        fallback_probability=0.10,
    )
    curve = HazardCurve(
        name="test_curve",
        metric_name="metric",
        unit="units",
        global_sample_size=500,
        global_failure_count=50,
        global_failure_rate=0.10,
        min_samples_threshold=100,
        bins=[b],
    )
    catalog = HazardCatalog(
        curves={"test_curve": curve},
        metadata={"created_by": "unit_test"},
    )

    json_path = tmp_path / "hazard_test.json"
    saved_path = catalog.to_json(json_path)
    assert saved_path.exists()

    loaded = HazardCatalog.from_json(json_path)
    assert "test_curve" in loaded.curves
    loaded_curve = loaded.get_curve("test_curve")
    assert loaded_curve.global_sample_size == 500
    assert loaded_curve.bins[0].failure_probability == 0.10
