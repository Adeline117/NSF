"""
Indicator Co-occurrence Deep Dive
==================================
For each of the 16 airdrop projects, compute pairwise indicator co-occurrence
matrices, identify the most common indicator combinations, analyze single-indicator
sybils (most vulnerable to evasion), and compute robustness metrics.

Key analyses:
  a) Pairwise indicator trigger co-occurrence matrix: P(A AND B | sybil)
  b) Most common indicator combinations among sybils
  c) Single-indicator sybils (caught by only 1 indicator)
  d) Robustness: fraction of sybils that would evade if each indicator were removed

Usage:
    python3 paper3_ai_sybil/experiments/indicator_cooccurrence.py
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
from itertools import combinations

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HASCIDB_DATA = PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
OUTPUT_FILE = SCRIPT_DIR / "indicator_cooccurrence_results.json"

THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}
INDICATORS = ["BT", "BW", "HF", "RF", "MA"]

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


def compute_indicator_flags(df):
    """Add boolean trigger columns for each indicator."""
    df = df.copy()
    df["BT_trig"] = (df["BT"] >= THRESHOLDS["BT"]).astype(int)
    df["BW_trig"] = (df["BW"] >= THRESHOLDS["BW"]).astype(int)
    df["HF_trig"] = (df["HF"] >= THRESHOLDS["HF"]).astype(int)
    df["RF_trig"] = (df["RF"] >= THRESHOLDS["RF"]).astype(int)
    df["MA_trig"] = (df["MA"] >= THRESHOLDS["MA"]).astype(int)
    return df


def main():
    print("=" * 80)
    print("INDICATOR CO-OCCURRENCE DEEP DIVE")
    print("=" * 80)

    all_results = {}
    aggregate_combos = Counter()
    aggregate_single = Counter()  # which indicator catches single-indicator sybils
    aggregate_removal = {ind: 0 for ind in INDICATORS}
    aggregate_sybil_total = 0

    for proj in PROJECTS:
        df = load_project(proj)
        if df is None:
            print(f"\n  SKIP: {proj} (file not found)")
            continue

        df = compute_indicator_flags(df)
        sybils = df[df["is_sybil"] == 1]
        n_sybil = len(sybils)

        if n_sybil == 0:
            print(f"\n  SKIP: {proj} (no sybils)")
            continue

        print(f"\n{'='*60}")
        print(f"  PROJECT: {proj} ({n_sybil:,} sybils / {len(df):,} total)")
        print(f"{'='*60}")

        # ========================================================
        # (a) Pairwise co-occurrence matrix: P(A AND B | sybil)
        # ========================================================
        trig_cols = [f"{ind}_trig" for ind in INDICATORS]
        cooccurrence = {}

        print(f"\n  Pairwise P(A AND B | sybil):")
        header = f"  {'':>6}"
        for ind in INDICATORS:
            header += f" {ind:>8}"
        print(header)

        for i, ind_a in enumerate(INDICATORS):
            row_str = f"  {ind_a:>6}"
            cooccurrence[ind_a] = {}
            for j, ind_b in enumerate(INDICATORS):
                if i == j:
                    # Marginal probability
                    p = sybils[f"{ind_a}_trig"].mean()
                else:
                    p = ((sybils[f"{ind_a}_trig"] == 1) & (sybils[f"{ind_b}_trig"] == 1)).mean()
                cooccurrence[ind_a][ind_b] = round(float(p), 4)
                row_str += f" {p:>8.4f}"
            print(row_str)

        # ========================================================
        # (b) Most common indicator combinations (vectorized)
        # ========================================================
        # Build combo string vectorized
        combo_parts = []
        for ind in INDICATORS:
            col = sybils[f"{ind}_trig"].astype(str).replace({"1": ind, "0": ""})
            combo_parts.append(col)
        combo_series = combo_parts[0]
        for p in combo_parts[1:]:
            combo_series = combo_series + "|" + p
        combo_series = combo_series.apply(lambda x: tuple(sorted(filter(None, x.split("|")))))
        combo_counter = Counter(combo_series)
        for combo, count in combo_counter.items():
            aggregate_combos[combo] += count

        print(f"\n  Most common indicator combinations:")
        print(f"  {'Combination':<40} {'Count':>8} {'%':>8}")
        print(f"  {'-'*58}")
        for combo, count in combo_counter.most_common(10):
            combo_str = " + ".join(combo) if combo else "(none)"
            print(f"  {combo_str:<40} {count:>8,} {100*count/n_sybil:>7.1f}%")

        # ========================================================
        # (c) Single-indicator sybils
        # ========================================================
        single_ind_sybils = sybils[sybils["n_indicators"] == 1]
        n_single = len(single_ind_sybils)
        single_pct = n_single / n_sybil if n_sybil > 0 else 0

        single_by_ind = {}
        for ind in INDICATORS:
            count = ((single_ind_sybils[f"{ind}_trig"] == 1)).sum()
            single_by_ind[ind] = int(count)
            aggregate_single[ind] += count

        print(f"\n  Single-indicator sybils: {n_single:,} ({100*single_pct:.1f}% of sybils)")
        print(f"  Breakdown by indicator:")
        for ind in INDICATORS:
            c = single_by_ind[ind]
            pct = 100 * c / max(1, n_single)
            print(f"    {ind}: {c:,} ({pct:.1f}% of single-indicator)")

        # ========================================================
        # (d) Robustness: if each indicator were removed
        # ========================================================
        print(f"\n  Evasion analysis (sybils that would evade if indicator removed):")
        removal_results = {}
        for ind in INDICATORS:
            # Sybils caught ONLY by this indicator (no other triggers)
            only_this = sybils[
                (sybils[f"{ind}_trig"] == 1) &
                (sybils["n_indicators"] == 1)
            ]
            n_evade = len(only_this)
            removal_results[ind] = {
                "would_evade": n_evade,
                "pct_of_sybils": round(100 * n_evade / n_sybil, 2),
            }
            aggregate_removal[ind] += n_evade
            print(f"    Remove {ind}: {n_evade:,} sybils evade ({100*n_evade/n_sybil:.1f}%)")

        aggregate_sybil_total += n_sybil

        # Store results
        all_results[proj] = {
            "n_total": len(df),
            "n_sybil": n_sybil,
            "cooccurrence_matrix": cooccurrence,
            "top_combos": [
                {"combo": list(combo), "count": count, "pct": round(100*count/n_sybil, 2)}
                for combo, count in combo_counter.most_common(15)
            ],
            "single_indicator_sybils": {
                "total": n_single,
                "pct_of_sybils": round(100 * single_pct, 2),
                "by_indicator": single_by_ind,
            },
            "removal_robustness": removal_results,
        }

    # ============================================================
    # AGGREGATE RESULTS ACROSS ALL PROJECTS
    # ============================================================
    print(f"\n{'='*80}")
    print(f"AGGREGATE ACROSS ALL PROJECTS")
    print(f"{'='*80}")

    print(f"\n  Total sybils across all projects: {aggregate_sybil_total:,}")

    print(f"\n  Most common indicator combinations (all projects):")
    print(f"  {'Combination':<40} {'Count':>8} {'%':>8}")
    print(f"  {'-'*58}")
    for combo, count in aggregate_combos.most_common(15):
        combo_str = " + ".join(combo) if combo else "(none)"
        print(f"  {combo_str:<40} {count:>8,} {100*count/aggregate_sybil_total:>7.1f}%")

    print(f"\n  Single-indicator sybils by indicator (all projects):")
    total_single = sum(aggregate_single.values())
    for ind in INDICATORS:
        c = aggregate_single[ind]
        print(f"    {ind}: {c:,} ({100*c/max(1,total_single):.1f}% of single-indicator sybils)")

    print(f"\n  Evasion impact of removing each indicator (all projects):")
    for ind in INDICATORS:
        n = aggregate_removal[ind]
        print(f"    Remove {ind}: {n:,} sybils evade ({100*n/max(1,aggregate_sybil_total):.1f}% of all sybils)")

    # Save
    output = {
        "per_project": all_results,
        "aggregate": {
            "total_sybils": aggregate_sybil_total,
            "top_combos": [
                {"combo": list(combo), "count": int(count), "pct": round(100*count/aggregate_sybil_total, 2)}
                for combo, count in aggregate_combos.most_common(20)
            ],
            "single_indicator_by_type": {k: int(v) for k, v in aggregate_single.items()},
            "removal_evasion": {
                ind: {"would_evade": int(aggregate_removal[ind]),
                      "pct": round(100*aggregate_removal[ind]/max(1,aggregate_sybil_total), 2)}
                for ind in INDICATORS
            },
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("INDICATOR CO-OCCURRENCE ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
