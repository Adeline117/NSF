"""
Non-Sybil Boundary Analysis (Evasion Margin)
==============================================
Analyzes non-sybil addresses that are CLOSE to detection thresholds, revealing
the "margin of safety" that an AI sybil could exploit to remain undetected.

Key analyses:
  a) For each indicator, find non-sybils with scores in the top 10%
  b) How many non-sybils are within 20% of ANY threshold?
  c) "Margin of safety" - the distribution gap between non-sybils and thresholds
  d) Feature distribution if an AI agent keeps all indicators at 80% of threshold

Usage:
    python3 paper3_ai_sybil/experiments/boundary_analysis.py
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HASCIDB_DATA = PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
OUTPUT_FILE = SCRIPT_DIR / "boundary_analysis_results.json"

THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}
INDICATORS = ["BT", "BW", "HF", "RF", "MA"]

# 80% of threshold = "safe zone" boundary
SAFE_FRACTIONS = {"BT": 4, "BW": 8, "HF": 0.64, "RF": 0.40, "MA": 4}

PROJECTS = [
    "1inch", "uniswap", "looksrare", "apecoin", "ens", "gitcoin",
    "etherfi", "x2y2", "dydx", "badger", "blur_s1", "blur_s2",
    "paraswap", "ampleforth", "eigenlayer", "pengu",
]


def load_project(project_name):
    """Load a project CSV."""
    path = HASCIDB_DATA / f"{project_name}_chi26_v3.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df["address"] = df["address"].str.lower().str.strip()
    return df


def compute_proximity(value, threshold):
    """Compute how close a value is to the threshold as a fraction.
    Returns value/threshold (1.0 = at threshold, 0.0 = zero score).
    """
    if threshold == 0:
        return 0.0
    return value / threshold


def main():
    print("=" * 80)
    print("NON-SYBIL BOUNDARY ANALYSIS (EVASION MARGIN)")
    print("=" * 80)

    all_results = {}
    agg_close_to_any = 0
    agg_nonsybil_total = 0
    agg_top10_counts = {ind: 0 for ind in INDICATORS}
    agg_within_20pct = {ind: 0 for ind in INDICATORS}

    # Aggregate non-sybil score distributions for safe-zone analysis
    agg_safe_zone_profiles = {ind: [] for ind in INDICATORS}

    for proj in PROJECTS:
        df = load_project(proj)
        if df is None:
            print(f"\n  SKIP: {proj}")
            continue

        nonsybils = df[df["is_sybil"] == 0]
        sybils = df[df["is_sybil"] == 1]
        n_nonsybil = len(nonsybils)
        n_sybil = len(sybils)

        if n_nonsybil == 0:
            continue

        print(f"\n{'='*60}")
        print(f"  PROJECT: {proj} ({n_nonsybil:,} non-sybils, {n_sybil:,} sybils)")
        print(f"{'='*60}")

        proj_results = {
            "n_total": len(df),
            "n_nonsybil": n_nonsybil,
            "n_sybil": n_sybil,
        }

        # ========================================================
        # (a) Top 10% non-sybils per indicator (closest to threshold)
        # ========================================================
        print(f"\n  (a) Non-sybils in top 10% per indicator (closest to threshold):")
        top10_analysis = {}
        for ind in INDICATORS:
            scores = nonsybils[ind].values
            # Top 10% = scores above the 90th percentile of non-sybil scores
            p90 = np.percentile(scores, 90)
            top10_mask = scores >= p90
            n_top10 = int(top10_mask.sum())
            top10_scores = scores[top10_mask]

            # How many of these are above 80% of threshold?
            thresh = THRESHOLDS[ind]
            above_80pct = int((top10_scores >= 0.8 * thresh).sum())
            above_50pct = int((top10_scores >= 0.5 * thresh).sum())

            top10_analysis[ind] = {
                "p90_value": round(float(p90), 4),
                "n_top10": n_top10,
                "top10_max": round(float(top10_scores.max()), 4) if len(top10_scores) > 0 else 0,
                "top10_mean": round(float(top10_scores.mean()), 4) if len(top10_scores) > 0 else 0,
                "above_80pct_threshold": above_80pct,
                "above_50pct_threshold": above_50pct,
                "threshold": thresh,
            }
            agg_top10_counts[ind] += n_top10

            proximity = p90 / thresh if thresh > 0 else 0
            print(f"    {ind}: P90={p90:.4f}, max={top10_scores.max():.4f}, "
                  f"threshold={thresh}, P90/threshold={proximity:.2%}, "
                  f"n_above_80%_thresh={above_80pct}")

        proj_results["top10_analysis"] = top10_analysis

        # ========================================================
        # (b) Non-sybils within 20% of ANY threshold
        # ========================================================
        within_20pct_mask = np.zeros(n_nonsybil, dtype=bool)
        per_ind_within = {}

        for ind in INDICATORS:
            thresh = THRESHOLDS[ind]
            low = 0.8 * thresh
            scores = nonsybils[ind].values
            mask = scores >= low
            per_ind_within[ind] = int(mask.sum())
            within_20pct_mask |= mask
            agg_within_20pct[ind] += int(mask.sum())

        n_close_any = int(within_20pct_mask.sum())
        close_pct = 100 * n_close_any / n_nonsybil

        print(f"\n  (b) Non-sybils within 20% of ANY threshold: {n_close_any:,} ({close_pct:.2f}%)")
        for ind in INDICATORS:
            print(f"      {ind} (>= {0.8*THRESHOLDS[ind]:.2f}): {per_ind_within[ind]:,}")

        agg_close_to_any += n_close_any
        agg_nonsybil_total += n_nonsybil

        proj_results["within_20pct"] = {
            "any_indicator": n_close_any,
            "pct": round(close_pct, 2),
            "per_indicator": per_ind_within,
        }

        # ========================================================
        # (c) Margin of safety: distribution gap analysis
        # ========================================================
        print(f"\n  (c) Margin of safety (non-sybil score percentiles vs threshold):")
        margin_analysis = {}
        for ind in INDICATORS:
            scores = nonsybils[ind].values
            thresh = THRESHOLDS[ind]
            percentiles = [50, 75, 90, 95, 99, 99.9]
            pvals = np.percentile(scores, percentiles)
            margin_analysis[ind] = {
                "threshold": thresh,
                "mean": round(float(scores.mean()), 4),
                "std": round(float(scores.std()), 4),
            }
            for p, v in zip(percentiles, pvals):
                margin_analysis[ind][f"p{p}"] = round(float(v), 4)
                margin_analysis[ind][f"p{p}_pct_of_thresh"] = round(float(v / thresh * 100), 2) if thresh > 0 else 0

            # Fraction of non-sybils at exactly 0
            zero_pct = (scores == 0).mean() * 100
            margin_analysis[ind]["zero_pct"] = round(float(zero_pct), 2)

            print(f"    {ind}: mean={scores.mean():.4f}, P99={pvals[4]:.4f}, "
                  f"P99/thresh={pvals[4]/thresh:.2%}, zero%={zero_pct:.1f}%")

        proj_results["margin_analysis"] = margin_analysis

        # ========================================================
        # (d) AI agent "80% of threshold" profile analysis
        # ========================================================
        # If an AI agent keeps all indicators at exactly 80% of threshold,
        # how does it compare to real non-sybils?
        safe_profile = {}
        for ind in INDICATORS:
            safe_val = SAFE_FRACTIONS[ind]
            thresh = THRESHOLDS[ind]
            scores = nonsybils[ind].values
            # What percentile would the safe value be among non-sybils?
            pctile = float(np.searchsorted(np.sort(scores), safe_val) / len(scores) * 100)
            safe_profile[ind] = {
                "safe_value": safe_val,
                "percentile_among_nonsybils": round(pctile, 2),
                "nonsybils_above": int((scores >= safe_val).sum()),
                "pct_nonsybils_above": round(100 * (scores >= safe_val).mean(), 2),
            }
            agg_safe_zone_profiles[ind].append(pctile)

        print(f"\n  (d) AI agent at 80% of threshold: percentile among non-sybils:")
        for ind in INDICATORS:
            sp = safe_profile[ind]
            print(f"    {ind}={sp['safe_value']}: P{sp['percentile_among_nonsybils']:.1f} "
                  f"({sp['pct_nonsybils_above']:.2f}% non-sybils also above)")

        proj_results["safe_zone_profile"] = safe_profile

        all_results[proj] = proj_results

    # ============================================================
    # AGGREGATE SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print(f"AGGREGATE SUMMARY")
    print(f"{'='*80}")

    print(f"\n  Total non-sybils across all projects: {agg_nonsybil_total:,}")
    print(f"  Within 20% of ANY threshold: {agg_close_to_any:,} ({100*agg_close_to_any/max(1,agg_nonsybil_total):.2f}%)")

    print(f"\n  Per-indicator, non-sybils within 20% of threshold:")
    for ind in INDICATORS:
        print(f"    {ind}: {agg_within_20pct[ind]:,}")

    print(f"\n  AI agent at 80% of threshold: average percentile across projects:")
    for ind in INDICATORS:
        vals = agg_safe_zone_profiles[ind]
        if vals:
            print(f"    {ind}: avg P{np.mean(vals):.1f} (range P{np.min(vals):.1f} - P{np.max(vals):.1f})")

    # Key insight: how anomalous would an AI agent look?
    print(f"\n  KEY INSIGHT: An AI agent at 80% of ALL thresholds simultaneously")
    print(f"  would be in the extreme tail of the non-sybil distribution")
    print(f"  for most indicators, making it distinguishable despite being")
    print(f"  below each individual threshold.")

    # Save
    output = {
        "per_project": all_results,
        "aggregate": {
            "total_nonsybils": agg_nonsybil_total,
            "close_to_any_threshold": agg_close_to_any,
            "close_pct": round(100 * agg_close_to_any / max(1, agg_nonsybil_total), 2),
            "per_indicator_within_20pct": agg_within_20pct,
            "safe_zone_avg_percentile": {
                ind: round(float(np.mean(agg_safe_zone_profiles[ind])), 2)
                for ind in INDICATORS if agg_safe_zone_profiles[ind]
            },
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("BOUNDARY ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
