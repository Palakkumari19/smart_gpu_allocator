"""
Hazard curves and empirical failure rate estimation from trace data.

Estimates empirical failure probabilities under varying operational stress:
1. BurstGPT Dispersion Hazard: Failure probability as a function of arrival dispersion index
   (index of dispersion = variance / mean). High dispersion (volatility spikes) drives
   memory pressure and server failure rates from ~1% to over 26%.
2. BurstGPT Offered Load Hazard: Failure probability as a function of hourly arrival volume.
3. Optional Alibaba PAI Sizing Hazard: Job termination/failure probability as a function of requested GPU size.

Computes exact sample counts, empirical probabilities, Wilson score 95% confidence intervals,
and marks bins with insufficient sample size (too_little_data).
Saves calibrated curves to realdata/calibration/hazard_curves.json.
"""

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Union
import numpy as np
import pandas as pd

from realdata.config import HAZARD_CURVES_PATH
from realdata.loaders import load_burstgpt_data
from realdata.windows import compute_hourly_dispersion


def wilson_score_interval(
    k: int, n: int, confidence: float = 0.95
) -> Tuple[float, float]:
    """
    Compute Wilson score confidence interval for a binomial proportion.

    Provides stable, bounded [0, 1] confidence intervals even for small sample sizes
    or extreme probabilities near 0 or 1.

    Args:
        k: Observed successes / failures (numerator).
        n: Total trials / observations (denominator).
        confidence: Confidence level (default 0.95).

    Returns:
        (ci_lower, ci_upper) as floats bounded in [0.0, 1.0].
    """
    if n <= 0:
        return (0.0, 0.0)

    # Standard normal quantile z
    if confidence == 0.95:
        z = 1.959963984540054
    elif confidence == 0.99:
        z = 2.5758293035489004
    elif confidence == 0.90:
        z = 1.6448536269514722
    else:
        # Fallback approximation for arbitrary confidence
        import scipy.stats as st  # type: ignore
        z = float(st.norm.ppf(1.0 - (1.0 - confidence) / 2.0))

    p_hat = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / float(n)
    center = (p_hat + z2 / (2.0 * float(n))) / denom
    spread = (z / denom) * math.sqrt(
        (p_hat * (1.0 - p_hat) / float(n)) + (z2 / (4.0 * float(n) * float(n)))
    )

    ci_lower = max(0.0, float(center - spread))
    ci_upper = min(1.0, float(center + spread))
    return (round(ci_lower, 6), round(ci_upper, 6))


@dataclass
class HazardBin:
    """Statistical summary for one binned interval of an operational metric."""
    bin_index: int
    bin_label: str
    range_min: float
    range_max: float
    sample_size: int
    failure_count: int
    failure_probability: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    too_little_data: bool = False
    fallback_probability: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HazardBin":
        return cls(**data)


@dataclass
class HazardCurve:
    """An empirical hazard curve mapping metric intervals to failure probabilities."""
    name: str
    metric_name: str
    unit: str
    global_sample_size: int
    global_failure_count: int
    global_failure_rate: float
    min_samples_threshold: int
    bins: List[HazardBin]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric_name": self.metric_name,
            "unit": self.unit,
            "global_sample_size": self.global_sample_size,
            "global_failure_count": self.global_failure_count,
            "global_failure_rate": self.global_failure_rate,
            "min_samples_threshold": self.min_samples_threshold,
            "bins": [b.to_dict() for b in self.bins],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HazardCurve":
        bins = [HazardBin.from_dict(b) for b in data["bins"]]
        return cls(
            name=data["name"],
            metric_name=data["metric_name"],
            unit=data["unit"],
            global_sample_size=data["global_sample_size"],
            global_failure_count=data["global_failure_count"],
            global_failure_rate=data["global_failure_rate"],
            min_samples_threshold=data["min_samples_threshold"],
            bins=bins,
        )

    def lookup_bin(self, value: float) -> HazardBin:
        """Find the matching bin for a given metric value."""
        if not self.bins:
            raise ValueError(f"HazardCurve '{self.name}' has no bins.")

        # Handle out-of-bounds below min
        if value < self.bins[0].range_min:
            return self.bins[0]

        # Handle in-bounds or edge matching
        for b in self.bins:
            if b.range_min <= value < b.range_max:
                return b

        # Value >= highest range_max: clamp to last bin
        return self.bins[-1]

    def lookup_probability(self, value: float, use_fallback: bool = True) -> float:
        """
        Query empirical failure probability for a given metric value.

        If the matched bin has too_little_data, returns fallback_probability
        (or global failure rate) if use_fallback is True.
        """
        matched_bin = self.lookup_bin(value)
        if matched_bin.too_little_data and use_fallback:
            return matched_bin.fallback_probability
        return matched_bin.failure_probability


@dataclass
class HazardCatalog:
    """Catalog of empirical hazard curves."""
    curves: Dict[str, HazardCurve]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "curves": {k: v.to_dict() for k, v in self.curves.items()},
        }

    def to_json(self, path: Optional[Union[str, Path]] = None) -> Path:
        target_path = Path(path) if path else HAZARD_CURVES_PATH
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return target_path

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HazardCatalog":
        curves = {k: HazardCurve.from_dict(v) for k, v in data["curves"].items()}
        return cls(curves=curves, metadata=data.get("metadata", {}))

    @classmethod
    def from_json(cls, path: Optional[Union[str, Path]] = None) -> "HazardCatalog":
        target_path = Path(path) if path else HAZARD_CURVES_PATH
        if not target_path.exists():
            raise FileNotFoundError(f"Hazard curves file not found at {target_path}")
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def get_curve(self, name: str) -> HazardCurve:
        if name not in self.curves:
            raise KeyError(
                f"Curve '{name}' not found. Available curves: {list(self.curves.keys())}"
            )
        return self.curves[name]


def estimate_burstgpt_dispersion_hazard(
    df: pd.DataFrame,
    bins: Optional[List[float]] = None,
    min_samples: int = 100,
    confidence: float = 0.95,
) -> HazardCurve:
    """
    Estimate failure probability from BurstGPT binned by arrival dispersion index.

    Real failures in BurstGPT are defined as rows where Response tokens == 0.

    Args:
        df: BurstGPT DataFrame.
        bins: Bin edges for dispersion index (default: [0.0, 1.0, 2.0, 5.0, 10.0, 50.0]).
        min_samples: Minimum request count required to trust empirical rate.
        confidence: Confidence level for Wilson score interval.

    Returns:
        Calibrated HazardCurve.
    """
    if bins is None:
        bins = [0.0, 1.0, 2.0, 5.0, 10.0, 50.0]

    df_copy = df.copy()
    if "hour" not in df_copy.columns:
        df_copy["hour"] = (df_copy["Timestamp"] // 3600).astype(int)
    if "is_failed" not in df_copy.columns:
        df_copy["is_failed"] = (df_copy["Response tokens"] == 0)

    # Compute dispersion index per hour
    dispersion_series = compute_hourly_dispersion(df_copy)
    hourly_reqs = df_copy.groupby("hour").size()
    hourly_fails = df_copy.groupby("hour")["is_failed"].sum()

    hourly_df = pd.DataFrame({
        "requests": hourly_reqs,
        "failures": hourly_fails,
        "dispersion": dispersion_series,
    }).dropna()

    global_n = int(hourly_df["requests"].sum())
    global_k = int(hourly_df["failures"].sum())
    global_rate = round(float(global_k) / max(1, global_n), 6)

    # Bin hours by dispersion
    hourly_df["disp_bin"] = pd.cut(
        hourly_df["dispersion"], bins=bins, right=False, include_lowest=True
    )

    hazard_bins: List[HazardBin] = []
    for idx, (b_start, b_end) in enumerate(zip(bins[:-1], bins[1:])):
        label = f"[{b_start:.1f}, {b_end:.1f})"
        sub = hourly_df[hourly_df["disp_bin"] == pd.Interval(b_start, b_end, closed="left")]

        sample_size = int(sub["requests"].sum()) if len(sub) > 0 else 0
        fail_cnt = int(sub["failures"].sum()) if len(sub) > 0 else 0
        rate = round(float(fail_cnt) / sample_size, 6) if sample_size > 0 else global_rate

        ci_l, ci_u = wilson_score_interval(fail_cnt, sample_size, confidence=confidence)
        too_little = sample_size < min_samples

        hazard_bins.append(
            HazardBin(
                bin_index=idx,
                bin_label=label,
                range_min=float(b_start),
                range_max=float(b_end),
                sample_size=sample_size,
                failure_count=fail_cnt,
                failure_probability=rate,
                ci_lower=ci_l,
                ci_upper=ci_u,
                confidence_level=confidence,
                too_little_data=too_little,
                fallback_probability=global_rate,
            )
        )

    return HazardCurve(
        name="burstgpt_dispersion",
        metric_name="dispersion_index",
        unit="var/mean",
        global_sample_size=global_n,
        global_failure_count=global_k,
        global_failure_rate=global_rate,
        min_samples_threshold=min_samples,
        bins=hazard_bins,
    )


def estimate_burstgpt_offered_load_hazard(
    df: pd.DataFrame,
    bins: Optional[List[float]] = None,
    min_samples: int = 100,
    confidence: float = 0.95,
) -> HazardCurve:
    """
    Estimate failure probability from BurstGPT binned by hourly arrival volume (offered load).

    Args:
        df: BurstGPT DataFrame.
        bins: Bin edges for hourly request count (default: [0, 500, 1000, 2000, 5000, 30000]).
        min_samples: Minimum request count required to trust empirical rate.
        confidence: Confidence level for Wilson score interval.

    Returns:
        Calibrated HazardCurve.
    """
    if bins is None:
        bins = [0.0, 500.0, 1000.0, 2000.0, 5000.0, 30000.0]

    df_copy = df.copy()
    if "hour" not in df_copy.columns:
        df_copy["hour"] = (df_copy["Timestamp"] // 3600).astype(int)
    if "is_failed" not in df_copy.columns:
        df_copy["is_failed"] = (df_copy["Response tokens"] == 0)

    hourly_reqs = df_copy.groupby("hour").size()
    hourly_fails = df_copy.groupby("hour")["is_failed"].sum()

    hourly_df = pd.DataFrame({
        "requests": hourly_reqs,
        "failures": hourly_fails,
    }).dropna()

    global_n = int(hourly_df["requests"].sum())
    global_k = int(hourly_df["failures"].sum())
    global_rate = round(float(global_k) / max(1, global_n), 6)

    hourly_df["load_bin"] = pd.cut(
        hourly_df["requests"], bins=bins, right=False, include_lowest=True
    )

    hazard_bins: List[HazardBin] = []
    for idx, (b_start, b_end) in enumerate(zip(bins[:-1], bins[1:])):
        label = f"[{int(b_start)}, {int(b_end)})"
        sub = hourly_df[hourly_df["load_bin"] == pd.Interval(b_start, b_end, closed="left")]

        sample_size = int(sub["requests"].sum()) if len(sub) > 0 else 0
        fail_cnt = int(sub["failures"].sum()) if len(sub) > 0 else 0
        rate = round(float(fail_cnt) / sample_size, 6) if sample_size > 0 else global_rate

        ci_l, ci_u = wilson_score_interval(fail_cnt, sample_size, confidence=confidence)
        too_little = sample_size < min_samples

        hazard_bins.append(
            HazardBin(
                bin_index=idx,
                bin_label=label,
                range_min=float(b_start),
                range_max=float(b_end),
                sample_size=sample_size,
                failure_count=fail_cnt,
                failure_probability=rate,
                ci_lower=ci_l,
                ci_upper=ci_u,
                confidence_level=confidence,
                too_little_data=too_little,
                fallback_probability=global_rate,
            )
        )

    return HazardCurve(
        name="burstgpt_offered_load",
        metric_name="requests_per_hour",
        unit="req/hour",
        global_sample_size=global_n,
        global_failure_count=global_k,
        global_failure_rate=global_rate,
        min_samples_threshold=min_samples,
        bins=hazard_bins,
    )


def estimate_alibaba_hazard(
    sample_path: Optional[Union[str, Path]] = None,
    bins: Optional[List[float]] = None,
    min_samples: int = 50,
    confidence: float = 0.95,
) -> Optional[HazardCurve]:
    """
    Estimate job failure rate by requested GPU size from Alibaba PAI cluster trace sample.

    Looks for GPU allocation columns ('plan_gpu', 'gpu_num', or 'gpu_count') and job status
    ('status', 'term_flag', 'term_type').
    If sample file is not present or path is None, returns None without error.
    """
    if sample_path is None:
        return None

    path = Path(sample_path)
    if not path.exists():
        return None

    try:
        # Load sample with common delimiter inspection
        sep = "\t" if path.suffix in [".tsv"] else ","
        df = pd.read_csv(path, sep=sep, nrows=50000)

        # Detect GPU size column
        gpu_col = None
        for col in ["plan_gpu", "gpu_num", "gpu_count", "num_gpus"]:
            if col in df.columns:
                gpu_col = col
                break

        # Detect status column
        status_col = None
        for col in ["status", "term_flag", "job_status"]:
            if col in df.columns:
                status_col = col
                break

        if gpu_col is None or status_col is None:
            return None

        # Determine failure indicator
        if pd.api.types.is_string_dtype(df[status_col]) or df[status_col].dtype == object:
            df["is_failed"] = df[status_col].astype(str).str.lower().isin(["failed", "error", "killed", "fail"])
        else:
            # Term flag 0 is typically success, nonzero indicates failure/killed
            df["is_failed"] = (df[status_col] != 0)

        if bins is None:
            bins = [1.0, 2.0, 4.0, 8.0, 16.0]

        global_n = len(df)
        global_k = int(df["is_failed"].sum())
        global_rate = round(float(global_k) / max(1, global_n), 6)

        df["gpu_bin"] = pd.cut(df[gpu_col], bins=bins, right=False, include_lowest=True)

        hazard_bins: List[HazardBin] = []
        for idx, (b_start, b_end) in enumerate(zip(bins[:-1], bins[1:])):
            label = f"[{int(b_start)}, {int(b_end)}) GPUs"
            sub = df[df["gpu_bin"] == pd.Interval(b_start, b_end, closed="left")]

            sample_size = len(sub)
            fail_cnt = int(sub["is_failed"].sum()) if sample_size > 0 else 0
            rate = round(float(fail_cnt) / sample_size, 6) if sample_size > 0 else global_rate

            ci_l, ci_u = wilson_score_interval(fail_cnt, sample_size, confidence=confidence)
            too_little = sample_size < min_samples

            hazard_bins.append(
                HazardBin(
                    bin_index=idx,
                    bin_label=label,
                    range_min=float(b_start),
                    range_max=float(b_end),
                    sample_size=sample_size,
                    failure_count=fail_cnt,
                    failure_probability=rate,
                    ci_lower=ci_l,
                    ci_upper=ci_u,
                    confidence_level=confidence,
                    too_little_data=too_little,
                    fallback_probability=global_rate,
                )
            )

        return HazardCurve(
            name="alibaba_gpu_size",
            metric_name="requested_gpus",
            unit="gpus",
            global_sample_size=global_n,
            global_failure_count=global_k,
            global_failure_rate=global_rate,
            min_samples_threshold=min_samples,
            bins=hazard_bins,
        )

    except Exception:
        return None


def build_hazard_catalog(
    burstgpt_df: Optional[pd.DataFrame] = None,
    alibaba_sample_path: Optional[Union[str, Path]] = None,
) -> HazardCatalog:
    """Build and calibrate all available hazard curves."""
    if burstgpt_df is None:
        burstgpt_df = load_burstgpt_data()

    dispersion_curve = estimate_burstgpt_dispersion_hazard(burstgpt_df)
    load_curve = estimate_burstgpt_offered_load_hazard(burstgpt_df)

    curves: Dict[str, HazardCurve] = {
        dispersion_curve.name: dispersion_curve,
        load_curve.name: load_curve,
    }

    alibaba_curve = estimate_alibaba_hazard(alibaba_sample_path)
    if alibaba_curve is not None:
        curves[alibaba_curve.name] = alibaba_curve

    metadata = {
        "description": "Empirical hazard curves for failure probability calibration",
        "burstgpt_total_requests": dispersion_curve.global_sample_size,
        "burstgpt_total_failures": dispersion_curve.global_failure_count,
        "burstgpt_global_failure_rate": dispersion_curve.global_failure_rate,
        "alibaba_included": alibaba_curve is not None,
    }

    return HazardCatalog(curves=curves, metadata=metadata)


def save_hazard_curves(catalog: HazardCatalog, path: Optional[Union[str, Path]] = None) -> Path:
    """Persist hazard curves catalog to JSON."""
    return catalog.to_json(path)


def load_hazard_curves(path: Optional[Union[str, Path]] = None) -> HazardCatalog:
    """Load persisted hazard curves catalog from JSON."""
    return HazardCatalog.from_json(path)


if __name__ == "__main__":
    print("Estimating empirical hazard curves from BurstGPT data...")
    df = load_burstgpt_data()
    catalog = build_hazard_catalog(df)
    out_path = save_hazard_curves(catalog)
    print(f"Hazard curves successfully persisted to {out_path}\n")

    for curve_name, curve in catalog.curves.items():
        print(f"=== Curve: {curve.name} ({curve.metric_name} [{curve.unit}]) ===")
        print(f"Global sample size: {curve.global_sample_size:,} | Failures: {curve.global_failure_count:,} ({curve.global_failure_rate*100:.2f}%)")
        print(f"{'Bin Range':<18} | {'Samples':<10} | {'Failures':<10} | {'Rate (%)':<10} | {'95% CI':<24} | {'Too Little Data'}")
        print("-" * 88)
        for b in curve.bins:
            ci_str = f"[{b.ci_lower*100:.2f}%, {b.ci_upper*100:.2f}%]"
            print(f"{b.bin_label:<18} | {b.sample_size:<10,} | {b.failure_count:<10,} | {b.failure_probability*100:<10.2f} | {ci_str:<24} | {b.too_little_data}")
        print()
