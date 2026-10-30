"""
PHASE 5 analysis — v2 results tables, Pareto, quarantine, hypothesis verdicts.

Reads only the CSVs the final test produced. Applies the decision rule exactly
as pre-registered in realdata/PREREGISTRATION_V2.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from evaluation.stats import compare_against_baseline
from realdata import config_v2 as C

ROOT = Path(__file__).resolve().parent.parent
MET = ROOT / "results" / "metrics" / "v2"
RESULTS = ROOT / "realdata" / "results"
MET.mkdir(parents=True, exist_ok=True)

LOWER_BETTER = ["avg_risk_exposure", "terminal_efc", "crash_count", "quarantine_time",
                "risk_per_completed_task", "crashes_per_completed_task",
                "mean_risk_over_time", "peak_gpu_risk"]
HIGHER_BETTER = ["completed_tasks"]


def latest(regime: str) -> Path:
    hits = sorted(RESULTS.glob(f"realdata_v2_runs_{regime}_*.csv"))
    if not hits:
        raise FileNotFoundError(f"no v2 results for regime '{regime}'")
    return hits[-1]


# ───────────────────────────────── throughput floor
def eligibility(df: pd.DataFrame, floor: float = C.THROUGHPUT_FLOOR) -> Dict[str, Any]:
    """A scheduler below `floor` x the best mean completed_tasks is INFEASIBLE."""
    means = df.groupby("scheduler")["completed_tasks"].mean()
    best = float(means.max())
    cutoff = floor * best
    return {
        "best_scheduler": str(means.idxmax()),
        "best_mean_completed": best,
        "cutoff": cutoff,
        "means": {k: float(v) for k, v in means.items()},
        "eligible": sorted([k for k, v in means.items() if v >= cutoff]),
        "infeasible": sorted([k for k, v in means.items() if v < cutoff]),
    }


# ───────────────────────────────── per-scheduler table
def results_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["avg_risk_exposure", "terminal_efc", "crash_count", "quarantine_time",
            "completed_tasks", "mean_risk_over_time", "peak_gpu_risk",
            "risk_per_completed_task", "crashes_per_completed_task",
            "contested_pct", "admitted_pct"]
    cols = [c for c in cols if c in df.columns]
    return df.groupby("scheduler")[cols].mean().round(4)


# ───────────────────────────────── paired hypothesis test
def paired_test(df: pd.DataFrame, metric: str, treatment: str,
                candidates: List[str], higher_is_better: bool) -> Dict[str, Any]:
    """
    Pair on (window, target_rho, seed). Compare the treatment against the
    strongest ELIGIBLE baseline on this metric.
    """
    key = ["window", "target_rho", "seed"]
    piv = df.pivot_table(index=key, columns="scheduler", values=metric, aggfunc="mean")
    piv = piv.dropna()
    cands = [c for c in candidates if c in piv.columns and c != treatment]
    if not cands or treatment not in piv.columns:
        return {"metric": metric, "verdict": "INCONCLUSIVE", "reason": "no eligible baseline"}

    means = piv[cands].mean()
    strongest = str(means.idxmax() if higher_is_better else means.idxmin())

    t = piv[treatment].to_numpy()
    b = piv[strongest].to_numpy()
    if np.allclose(t, b):
        return {"metric": metric, "strongest_baseline": strongest,
                "treatment_mean": float(t.mean()), "baseline_mean": float(b.mean()),
                "verdict": "INCONCLUSIVE", "reason": "values identical across all pairs"}

    res = compare_against_baseline(treatment_values=list(t), baseline_values=list(b),
                                   higher_is_better=higher_is_better, seed=0)
    supported = bool(res.get("significant_better")) and float(res.get("win_rate", 0)) >= C.MIN_WIN_RATE
    return {
        "metric": metric, "strongest_baseline": strongest,
        "treatment_mean": float(t.mean()), "baseline_mean": float(b.mean()),
        "mean_diff": float(res.get("mean_diff", np.nan)),
        "ci_low": float(res.get("ci_low", np.nan)), "ci_high": float(res.get("ci_high", np.nan)),
        "win_rate": float(res.get("win_rate", np.nan)),
        "significant_better": bool(res.get("significant_better")),
        "significant_worse": bool(res.get("significant_worse")),
        "n_pairs": int(len(t)),
        "verdict": "SUPPORTED" if supported else "REJECTED",
    }


# ───────────────────────────────── Pareto frontier
def pareto(df: pd.DataFrame, x="completed_tasks", y="mean_risk_over_time") -> Dict[str, Any]:
    """x higher better, y lower better. Returns frontier membership + dominators."""
    m = df.groupby("scheduler")[[x, y]].mean()
    pts = {s: (float(m.loc[s, x]), float(m.loc[s, y])) for s in m.index}
    frontier, dominated_by = [], {}
    for s, (sx, sy) in pts.items():
        doms = [o for o, (ox, oy) in pts.items()
                if o != s and ox >= sx and oy <= sy and (ox > sx or oy < sy)]
        if doms:
            dominated_by[s] = doms
        else:
            frontier.append(s)
    return {"points": pts, "frontier": sorted(frontier), "dominated_by": dominated_by}


# ───────────────────────────────── quarantine analysis
def quarantine_analysis(df: pd.DataFrame) -> Dict[str, Any]:
    out = {}
    for thr in C.QUARANTINE_THRESHOLDS_REPORTED:
        col = f"quarantine_triggered_at_{thr}"
        if col not in df.columns:
            continue
        share = float(df[col].mean())
        trig, not_trig = df[df[col] > 0], df[df[col] <= 0]
        entry = {
            "threshold": thr,
            "episodes": int(len(df)),
            "episodes_triggered": int(df[col].sum()),
            "trigger_share": share,
            "crashes_when_triggered": float(trig["crash_count"].mean()) if len(trig) else None,
            "crashes_when_not": float(not_trig["crash_count"].mean()) if len(not_trig) else None,
        }
        if entry["crashes_when_triggered"] is not None and entry["crashes_when_not"] is not None:
            entry["crash_delta"] = entry["crashes_when_triggered"] - entry["crashes_when_not"]
            entry["fewer_crashes_when_triggered"] = entry["crash_delta"] < 0
        entry["testable"] = share >= 0.30
        out[str(thr)] = entry
    return out


# ───────────────────────────────── top level
def analyze_regime(regime: str) -> Dict[str, Any]:
    path = latest(regime)
    df = pd.read_csv(path)
    out: Dict[str, Any] = {"regime": regime, "source": path.name, "conditions": {}}

    for cond in sorted(df.condition.unique()):
        d = df[df.condition == cond]
        elig = eligibility(d)
        all_scheds = sorted(d.scheduler.unique())

        verdicts = {}
        for hid, metric, hib in (
            ("V2-H1", "risk_per_completed_task", False),
            ("V2-H2", "crashes_per_completed_task", False),
            ("V2-H5", "completed_tasks", True),
        ):
            verdicts[hid] = {
                "with_floor": paired_test(d, metric, C.TREATMENT, elig["eligible"], hib),
                "without_floor": paired_test(d, metric, C.TREATMENT, all_scheds, hib),
            }

        par = pareto(d)
        verdicts["V2-H3"] = {
            "frontier": par["frontier"],
            "treatment_on_frontier": C.TREATMENT in par["frontier"],
            "dominated_by": par["dominated_by"].get(C.TREATMENT, []),
            "verdict": "SUPPORTED" if C.TREATMENT in par["frontier"] else "REJECTED",
        }

        q = quarantine_analysis(d)
        shipped = q.get(str(C.SHIPPED_QUARANTINE_THRESHOLD), {})
        explor = q.get(str(C.EXPLORATORY_QUARANTINE_THRESHOLD), {})
        v2h4 = {"shipped": shipped, "exploratory": explor}
        if not shipped.get("testable", False):
            v2h4["shipped_verdict"] = "INCONCLUSIVE (untestable at shipped threshold)"
        if explor:
            ok = explor.get("testable") and explor.get("fewer_crashes_when_triggered")
            v2h4["exploratory_verdict"] = "SUPPORTED" if ok else (
                "REJECTED" if explor.get("testable") else "INCONCLUSIVE")
        verdicts["V2-H4"] = v2h4

        # V2-H5 extra: within 5% of best dispatcher
        best = elig["best_mean_completed"]
        t_mean = float(d[d.scheduler == C.TREATMENT]["completed_tasks"].mean())
        verdicts["V2-H5"]["within_5pct_of_best"] = bool(t_mean >= 0.95 * best)
        verdicts["V2-H5"]["treatment_mean_completed"] = t_mean
        verdicts["V2-H5"]["best_mean_completed"] = best

        out["conditions"][cond] = {
            "eligibility": elig,
            "table": results_table(d).to_dict(orient="index"),
            "pareto": par,
            "quarantine": q,
            "verdicts": verdicts,
            # keys are (rho, scheduler) tuples; JSON needs strings
            "by_load": {f"{k[0]}|{k[1]}": v for k, v in
                        d.groupby(["target_rho", "scheduler"])[
                            ["completed_tasks", "mean_risk_over_time", "crash_count",
                             "risk_per_completed_task", "contested_pct"]
                        ].mean().round(4).to_dict(orient="index").items()},
        }
    return out


def main():
    all_out = {}
    for regime in ("workload", "overlay"):
        try:
            all_out[regime] = analyze_regime(regime)
        except FileNotFoundError as e:
            print(f"[skip] {e}")
    json.dump(all_out, open(MET / "v2_analysis.json", "w"), indent=2, default=str)
    print(f"wrote {MET / 'v2_analysis.json'}")

    for regime, r in all_out.items():
        for cond, c in r["conditions"].items():
            print(f"\n=== {regime} / {cond} ===")
            print("eligible:", c["eligibility"]["eligible"],
                  "| infeasible:", c["eligibility"]["infeasible"])
            for hid in ("V2-H1", "V2-H2", "V2-H3", "V2-H5"):
                v = c["verdicts"][hid]
                if hid == "V2-H3":
                    print(f"  {hid}: {v['verdict']} (frontier={v['frontier']})")
                else:
                    wf = v.get("with_floor", {})
                    print(f"  {hid}: {wf.get('verdict')} vs {wf.get('strongest_baseline')} "
                          f"(t={wf.get('treatment_mean'):.4g} b={wf.get('baseline_mean'):.4g} "
                          f"win={wf.get('win_rate')})" if wf.get("verdict") else f"  {hid}: n/a")


if __name__ == "__main__":
    main()
