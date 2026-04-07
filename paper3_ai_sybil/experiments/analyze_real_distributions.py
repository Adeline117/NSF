"""
Analyze Real HasciDB Indicator Distributions from 16 Projects
==============================================================
Loads the chi26_v3 CSV files from the cloned HasciDB repository and computes
comprehensive distribution statistics for each of the 5 indicators (BT, BW,
HF, RF, MA) across all 16 Ethereum L1 airdrop projects.

For each project, computes:
  - Total eligible addresses and sybil rate
  - Per-indicator trigger rates (BT>=5, BW>=10, HF>=0.80, RF>=0.50, MA>=5)
  - Distribution statistics (mean, std, median, 25th/75th percentiles)
    for both sybil and non-sybil populations
  - Indicator co-occurrence patterns among sybils

Outputs:
  - Console summary table
  - JSON results file at paper3_ai_sybil/experiments/real_distribution_results.json

Usage:
    python3 paper3_ai_sybil/experiments/analyze_real_distributions.py
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HASCIDB_DATA = PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
OUTPUT_FILE = SCRIPT_DIR / "real_distribution_results.json"

# HasciDB thresholds (CHI'26 Table 13, Delphi Round 2)
THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}

# The 16 Ethereum L1 airdrop projects (excluding pengu_solana which is Solana-side)
PROJECTS = [
    "1inch", "uniswap", "looksrare", "apecoin", "ens", "gitcoin",
    "etherfi", "x2y2", "dydx", "badger", "blur_s1", "blur_s2",
    "paraswap", "ampleforth", "eigenlayer", "pengu",
]

# Column name mapping: ETH CSVs use BT/BW/HF/RF/MA
# PENGU Solana CSV uses bt_score/bw_score etc. (we skip Solana)
INDICATOR_COLS = ["BT", "BW", "HF", "RF", "MA"]


# ============================================================
# DATA LOADING
# ============================================================

def load_project(project: str) -> pd.DataFrame:
    """Load a single project's chi26_v3 CSV file."""
    csv_path = HASCIDB_DATA / f"{project}_chi26_v3.csv"
    if not csv_path.exists():
        print(f"  WARNING: {csv_path} not found, skipping.")
        return pd.DataFrame()

    df = pd.read_csv(csv_path)
    df["project"] = project

    # Normalize column names (pengu ETH uses same schema as others)
    # Ensure indicator columns exist
    for col in INDICATOR_COLS:
        if col not in df.columns:
            # Check for lowercase variants
            lower = col.lower()
            score_variant = f"{lower}_score"
            if score_variant in df.columns:
                df[col] = df[score_variant]
            elif lower in df.columns:
                df[col] = df[lower]
            else:
                print(f"  WARNING: Column {col} not found in {project}")
                df[col] = 0

    return df


# ============================================================
# STATISTICS COMPUTATION
# ============================================================

def compute_distribution_stats(values: pd.Series) -> dict:
    """Compute distribution statistics for a numeric series."""
    if len(values) == 0:
        return {
            "count": 0, "mean": 0, "std": 0, "min": 0,
            "p25": 0, "median": 0, "p75": 0, "p90": 0, "p95": 0, "p99": 0, "max": 0,
            "nonzero_count": 0, "nonzero_pct": 0,
        }
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "max": float(values.max()),
        "nonzero_count": int((values > 0).sum()),
        "nonzero_pct": float((values > 0).mean()),
    }


def compute_trigger_rates(df: pd.DataFrame) -> dict:
    """Compute per-indicator trigger rates for a dataframe."""
    n = len(df)
    if n == 0:
        return {ind: 0.0 for ind in INDICATOR_COLS}
    return {
        "BT": float((df["BT"] >= THRESHOLDS["BT"]).mean()),
        "BW": float((df["BW"] >= THRESHOLDS["BW"]).mean()),
        "HF": float((df["HF"] >= THRESHOLDS["HF"]).mean()),
        "RF": float((df["RF"] >= THRESHOLDS["RF"]).mean()),
        "MA": float((df["MA"] >= THRESHOLDS["MA"]).mean()),
    }


def compute_cooccurrence(sybils: pd.DataFrame) -> dict:
    """Compute indicator co-occurrence among sybils."""
    if len(sybils) == 0:
        return {}

    n = len(sybils)
    triggered = {
        "BT": sybils["BT"] >= THRESHOLDS["BT"],
        "BW": sybils["BW"] >= THRESHOLDS["BW"],
        "HF": sybils["HF"] >= THRESHOLDS["HF"],
        "RF": sybils["RF"] >= THRESHOLDS["RF"],
        "MA": sybils["MA"] >= THRESHOLDS["MA"],
    }

    # Count how many indicators each sybil triggers
    n_triggered = sum(triggered[ind].astype(int) for ind in INDICATOR_COLS)
    trigger_count_dist = {}
    for k in range(1, 6):
        count = int((n_triggered == k).sum())
        trigger_count_dist[str(k)] = {"count": count, "pct": round(count / n * 100, 2)}

    # Pairwise co-occurrence rates
    pairwise = {}
    for i, ind1 in enumerate(INDICATOR_COLS):
        for ind2 in INDICATOR_COLS[i + 1:]:
            both = (triggered[ind1] & triggered[ind2]).sum()
            pairwise[f"{ind1}+{ind2}"] = {
                "count": int(both),
                "pct_of_sybils": round(float(both) / n * 100, 2),
            }

    # Single-indicator sybils (caught by exactly 1)
    single_indicator = int((n_triggered == 1).sum())

    # Axis breakdown
    ops_only = int(((sybils["ops_flag"] == 1) & (sybils["fund_flag"] == 0)).sum())
    fund_only = int(((sybils["ops_flag"] == 0) & (sybils["fund_flag"] == 1)).sum())
    both_axes = int(((sybils["ops_flag"] == 1) & (sybils["fund_flag"] == 1)).sum())

    return {
        "trigger_count_distribution": trigger_count_dist,
        "pairwise_cooccurrence": pairwise,
        "single_indicator_sybils": {
            "count": single_indicator,
            "pct": round(single_indicator / n * 100, 2),
        },
        "axis_breakdown": {
            "ops_only": {"count": ops_only, "pct": round(ops_only / n * 100, 2)},
            "fund_only": {"count": fund_only, "pct": round(fund_only / n * 100, 2)},
            "both_axes": {"count": both_axes, "pct": round(both_axes / n * 100, 2)},
        },
    }


def analyze_project(df: pd.DataFrame, project: str) -> dict:
    """Compute full analysis for a single project."""
    n_total = len(df)
    sybils = df[df["is_sybil"] == 1]
    non_sybils = df[df["is_sybil"] == 0]
    n_sybils = len(sybils)
    n_non_sybils = len(non_sybils)

    result = {
        "project": project,
        "n_eligible": n_total,
        "n_sybils": n_sybils,
        "n_non_sybils": n_non_sybils,
        "sybil_rate": round(n_sybils / n_total * 100, 2) if n_total > 0 else 0,
    }

    # Per-indicator trigger rates (among ALL eligible)
    result["trigger_rates_all"] = compute_trigger_rates(df)

    # Per-indicator trigger rates among sybils only
    result["trigger_rates_sybils"] = compute_trigger_rates(sybils)

    # Per-indicator trigger rates among non-sybils (should be ~0 by definition)
    result["trigger_rates_non_sybils"] = compute_trigger_rates(non_sybils)

    # Distribution stats per indicator, split by sybil/non-sybil
    result["distributions"] = {}
    for ind in INDICATOR_COLS:
        result["distributions"][ind] = {
            "all": compute_distribution_stats(df[ind]),
            "sybil": compute_distribution_stats(sybils[ind]),
            "non_sybil": compute_distribution_stats(non_sybils[ind]),
        }

    # Co-occurrence analysis among sybils
    result["cooccurrence"] = compute_cooccurrence(sybils)

    return result


# ============================================================
# CROSS-PROJECT AGGREGATION
# ============================================================

def compute_aggregate(all_results: list[dict]) -> dict:
    """Compute aggregate statistics across all projects."""
    total_eligible = sum(r["n_eligible"] for r in all_results)
    total_sybils = sum(r["n_sybils"] for r in all_results)

    # Weighted average trigger rates (weighted by n_sybils)
    weighted_trigger = {ind: 0.0 for ind in INDICATOR_COLS}
    for r in all_results:
        w = r["n_sybils"]
        for ind in INDICATOR_COLS:
            weighted_trigger[ind] += r["trigger_rates_sybils"][ind] * w
    if total_sybils > 0:
        weighted_trigger = {k: round(v / total_sybils, 4) for k, v in weighted_trigger.items()}

    # Range of sybil rates across projects
    sybil_rates = [r["sybil_rate"] for r in all_results]

    # Range of trigger rates
    trigger_ranges = {}
    for ind in INDICATOR_COLS:
        rates = [r["trigger_rates_sybils"][ind] for r in all_results]
        trigger_ranges[ind] = {
            "min": round(min(rates), 4),
            "max": round(max(rates), 4),
            "mean": round(np.mean(rates), 4),
            "std": round(np.std(rates), 4),
        }

    return {
        "total_eligible": total_eligible,
        "total_sybils": total_sybils,
        "overall_sybil_rate": round(total_sybils / total_eligible * 100, 2),
        "sybil_rate_range": {
            "min": round(min(sybil_rates), 2),
            "max": round(max(sybil_rates), 2),
            "mean": round(np.mean(sybil_rates), 2),
        },
        "weighted_avg_trigger_rates_among_sybils": weighted_trigger,
        "trigger_rate_ranges_among_sybils": trigger_ranges,
    }


# ============================================================
# COMPARISON WITH PILOT ASSUMPTIONS
# ============================================================

def compare_with_pilot(aggregate: dict) -> dict:
    """Compare real distributions with the pilot's INDICATOR_PREVALENCE assumptions."""
    # From ai_sybil_generator.py line 66
    pilot_prevalence = {"BT": 0.08, "BW": 0.15, "HF": 0.45, "RF": 0.12, "MA": 0.20}
    real_weighted = aggregate["weighted_avg_trigger_rates_among_sybils"]

    comparison = {}
    for ind in INDICATOR_COLS:
        pilot_val = pilot_prevalence[ind]
        real_val = real_weighted[ind]
        diff = real_val - pilot_val
        comparison[ind] = {
            "pilot_assumption": pilot_val,
            "real_weighted_avg": round(real_val, 4),
            "absolute_difference": round(diff, 4),
            "relative_difference_pct": round(diff / pilot_val * 100, 1) if pilot_val > 0 else None,
            "direction": "HIGHER than assumed" if diff > 0 else "LOWER than assumed",
        }

    return comparison


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_summary_table(all_results: list[dict]):
    """Print a formatted summary table to console."""
    print("\n" + "=" * 120)
    print("HASCIDB REAL INDICATOR DISTRIBUTIONS — SUMMARY TABLE")
    print("=" * 120)

    # Header
    header = f"{'Project':<14} {'Eligible':>10} {'Sybils':>10} {'Rate%':>6} | " \
             f"{'BT%':>6} {'BW%':>6} {'HF%':>6} {'RF%':>6} {'MA%':>6} | " \
             f"{'1-ind%':>7} {'OpsOnly%':>9} {'FundOnly%':>10}"
    print(header)
    print("-" * 120)

    for r in all_results:
        tr = r["trigger_rates_sybils"]
        co = r["cooccurrence"]
        single_pct = co["single_indicator_sybils"]["pct"] if co else 0
        ops_pct = co["axis_breakdown"]["ops_only"]["pct"] if co else 0
        fund_pct = co["axis_breakdown"]["fund_only"]["pct"] if co else 0

        row = f"{r['project']:<14} {r['n_eligible']:>10,} {r['n_sybils']:>10,} {r['sybil_rate']:>5.1f}% | " \
              f"{tr['BT']*100:>5.1f}% {tr['BW']*100:>5.1f}% {tr['HF']*100:>5.1f}% " \
              f"{tr['RF']*100:>5.1f}% {tr['MA']*100:>5.1f}% | " \
              f"{single_pct:>6.1f}% {ops_pct:>8.1f}% {fund_pct:>9.1f}%"
        print(row)

    print("-" * 120)


def print_distribution_details(all_results: list[dict]):
    """Print detailed distribution statistics per indicator."""
    print("\n" + "=" * 120)
    print("DETAILED INDICATOR DISTRIBUTIONS (SYBIL POPULATION)")
    print("=" * 120)

    for ind in INDICATOR_COLS:
        print(f"\n--- {ind} (threshold >= {THRESHOLDS[ind]}) ---")
        header = f"{'Project':<14} {'Mean':>8} {'Std':>8} {'Median':>8} " \
                 f"{'P25':>8} {'P75':>8} {'P90':>8} {'P95':>8} {'Max':>10} {'NonZero%':>9}"
        print(header)
        print("-" * 100)
        for r in all_results:
            s = r["distributions"][ind]["sybil"]
            if s["count"] == 0:
                continue
            row = f"{r['project']:<14} {s['mean']:>8.3f} {s['std']:>8.3f} {s['median']:>8.3f} " \
                  f"{s['p25']:>8.3f} {s['p75']:>8.3f} {s['p90']:>8.3f} {s['p95']:>8.3f} " \
                  f"{s['max']:>10.1f} {s['nonzero_pct']*100:>8.1f}%"
            print(row)


def print_pilot_comparison(comparison: dict):
    """Print comparison with pilot assumptions."""
    print("\n" + "=" * 120)
    print("COMPARISON: PILOT ASSUMPTIONS vs REAL DATA")
    print("=" * 120)
    print(f"{'Indicator':<6} {'Pilot':>10} {'Real':>10} {'Diff':>10} {'Rel%':>10} {'Direction'}")
    print("-" * 70)
    for ind, comp in comparison.items():
        rel = f"{comp['relative_difference_pct']:>+.1f}%" if comp['relative_difference_pct'] is not None else "N/A"
        print(f"{ind:<6} {comp['pilot_assumption']:>10.4f} {comp['real_weighted_avg']:>10.4f} "
              f"{comp['absolute_difference']:>+10.4f} {rel:>10} {comp['direction']}")


def print_non_sybil_distributions(all_results: list[dict]):
    """Print non-sybil indicator distributions for generator calibration."""
    print("\n" + "=" * 120)
    print("NON-SYBIL (CLEAN) INDICATOR DISTRIBUTIONS — FOR GENERATOR CALIBRATION")
    print("=" * 120)

    for ind in INDICATOR_COLS:
        print(f"\n--- {ind} (non-sybils) ---")
        header = f"{'Project':<14} {'Mean':>8} {'Std':>8} {'Median':>8} " \
                 f"{'P75':>8} {'P90':>8} {'P95':>8} {'NonZero%':>9}"
        print(header)
        print("-" * 85)
        for r in all_results:
            s = r["distributions"][ind]["non_sybil"]
            if s["count"] == 0:
                continue
            row = f"{r['project']:<14} {s['mean']:>8.3f} {s['std']:>8.3f} {s['median']:>8.3f} " \
                  f"{s['p75']:>8.3f} {s['p90']:>8.3f} {s['p95']:>8.3f} {s['nonzero_pct']*100:>8.1f}%"
            print(row)


# ============================================================
# MAIN
# ============================================================

def main():
    print("HasciDB Real Distribution Analysis")
    print(f"Data source: {HASCIDB_DATA}")
    print(f"Projects: {len(PROJECTS)}")
    print()

    if not HASCIDB_DATA.exists():
        print(f"ERROR: HasciDB data directory not found at {HASCIDB_DATA}")
        print("Run: git clone https://github.com/UW-Decentralized-Computing-Lab/HasciDB.git "
              "paper3_ai_sybil/data/HasciDB")
        sys.exit(1)

    # Load and analyze each project
    all_results = []
    for project in PROJECTS:
        print(f"  Loading {project}...", end=" ")
        df = load_project(project)
        if df.empty:
            print("SKIPPED")
            continue
        result = analyze_project(df, project)
        all_results.append(result)
        print(f"OK ({result['n_eligible']:,} eligible, {result['n_sybils']:,} sybils, "
              f"{result['sybil_rate']}%)")

    if not all_results:
        print("ERROR: No data loaded.")
        sys.exit(1)

    # Aggregate statistics
    aggregate = compute_aggregate(all_results)

    # Comparison with pilot assumptions
    comparison = compare_with_pilot(aggregate)

    # Console output
    print_summary_table(all_results)
    print_distribution_details(all_results)
    print_non_sybil_distributions(all_results)
    print_pilot_comparison(comparison)

    # Save full results to JSON
    output = {
        "metadata": {
            "data_source": "HasciDB chi26_v3 CSVs",
            "n_projects": len(all_results),
            "thresholds": THRESHOLDS,
            "detection_logic": {
                "ops_flag": "(BT >= 5) OR (BW >= 10) OR (HF >= 0.80)",
                "fund_flag": "(RF >= 0.50) OR (MA >= 5)",
                "is_sybil": "ops_flag OR fund_flag",
            },
        },
        "aggregate": aggregate,
        "pilot_comparison": comparison,
        "per_project": all_results,
    }

    # Ensure output directory exists
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to {OUTPUT_FILE}")
    print(f"Total: {aggregate['total_eligible']:,} eligible, {aggregate['total_sybils']:,} sybils "
          f"({aggregate['overall_sybil_rate']}%)")

    # Print key findings for updating the generator
    print("\n" + "=" * 120)
    print("KEY FINDINGS FOR AI SYBIL GENERATOR UPDATE")
    print("=" * 120)
    print("\nCurrent INDICATOR_PREVALENCE in ai_sybil_generator.py:")
    pilot = {"BT": 0.08, "BW": 0.15, "HF": 0.45, "RF": 0.12, "MA": 0.20}
    for ind, val in pilot.items():
        real = aggregate["weighted_avg_trigger_rates_among_sybils"][ind]
        status = "OK" if abs(real - val) < 0.05 else "NEEDS UPDATE"
        print(f"  {ind}: pilot={val:.2f}, real={real:.4f} [{status}]")

    print("\nRecommended updates based on real data:")
    print("  INDICATOR_PREVALENCE = {")
    for ind in INDICATOR_COLS:
        real = aggregate["weighted_avg_trigger_rates_among_sybils"][ind]
        print(f'      "{ind}": {real:.2f},')
    print("  }")


if __name__ == "__main__":
    main()
