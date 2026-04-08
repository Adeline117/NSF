"""
Serial Sybil Analysis Across 16 Projects
=========================================
Identifies addresses flagged as sybil in multiple airdrop projects and analyzes
serial sybil behavior patterns, cross-project indicator correlation, and
temporal evolution of sybil sophistication.

Key questions:
  1. How many addresses are sybil in 2+, 3+, 5+ projects?
  2. Which indicator combinations catch serial sybils?
  3. Do serial sybils get caught by the same indicator or different ones?
  4. Are serial sybils harder or easier to detect?
  5. Cross-project indicator score correlation
  6. Temporal evolution of sybil rates and sophistication

Usage:
    python3 paper3_ai_sybil/experiments/serial_sybil_analysis.py
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HASCIDB_DATA = PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
OUTPUT_FILE = SCRIPT_DIR / "serial_sybil_results.json"

THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}
INDICATORS = ["BT", "BW", "HF", "RF", "MA"]

# 16 Ethereum L1 airdrop projects (excluding pengu_solana)
PROJECTS = [
    "1inch", "uniswap", "looksrare", "apecoin", "ens", "gitcoin",
    "etherfi", "x2y2", "dydx", "badger", "blur_s1", "blur_s2",
    "paraswap", "ampleforth", "eigenlayer", "pengu",
]

# Approximate chronological order by airdrop snapshot/claim date
PROJECT_DATES = {
    "uniswap":    "2020-09",
    "badger":     "2020-12",
    "1inch":      "2020-12",
    "ampleforth": "2021-04",
    "gitcoin":    "2021-05",
    "ens":        "2021-11",
    "dydx":       "2021-09",
    "paraswap":   "2021-11",
    "looksrare":  "2022-01",
    "x2y2":       "2022-02",
    "apecoin":    "2022-03",
    "blur_s1":    "2023-02",
    "blur_s2":    "2023-06",
    "eigenlayer": "2024-05",
    "etherfi":    "2024-03",
    "pengu":      "2024-12",
}


def load_project(project_name):
    """Load a project CSV and return the DataFrame."""
    path = HASCIDB_DATA / f"{project_name}_chi26_v3.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found, skipping")
        return None
    df = pd.read_csv(path)
    # Standardize address to lowercase
    df["address"] = df["address"].str.lower().str.strip()
    return df


def add_trigger_columns(df):
    """Add boolean trigger columns for each indicator (vectorized)."""
    df = df.copy()
    df["BT_trig"] = (df["BT"] >= THRESHOLDS["BT"]).astype(int)
    df["BW_trig"] = (df["BW"] >= THRESHOLDS["BW"]).astype(int)
    df["HF_trig"] = (df["HF"] >= THRESHOLDS["HF"]).astype(int)
    df["RF_trig"] = (df["RF"] >= THRESHOLDS["RF"]).astype(int)
    df["MA_trig"] = (df["MA"] >= THRESHOLDS["MA"]).astype(int)
    # Create a string combo column: sorted indicator names joined by "+"
    parts = []
    for ind in INDICATORS:
        parts.append(df[f"{ind}_trig"].astype(str).replace({"1": ind, "0": ""}))
    df["_combo"] = parts[0]
    for p in parts[1:]:
        df["_combo"] = df["_combo"] + "|" + p
    # Clean up combo string
    df["_combo"] = df["_combo"].apply(lambda x: "+".join(sorted(filter(None, x.split("|")))))
    return df


def main():
    print("=" * 80)
    print("SERIAL SYBIL ANALYSIS ACROSS 16 PROJECTS")
    print("=" * 80)
    sys.stdout.flush()

    # ============================================================
    # 1. LOAD ALL DATA
    # ============================================================
    print("\n[1] Loading all project data...")
    sys.stdout.flush()
    project_data = {}
    # Keep only sybil rows for cross-project analysis (saves memory)
    project_sybils = {}

    for proj in PROJECTS:
        df = load_project(proj)
        if df is not None:
            df = add_trigger_columns(df)
            project_data[proj] = df
            sybils = df[df["is_sybil"] == 1].copy()
            sybils["project"] = proj
            project_sybils[proj] = sybils
            n_sybil = len(sybils)
            print(f"  {proj}: {len(df):,} addresses, {n_sybil:,} sybils ({100*n_sybil/len(df):.1f}%)")
            sys.stdout.flush()

    # ============================================================
    # 2. FIND SERIAL SYBILS (addresses sybil in 2+ projects)
    # ============================================================
    print("\n[2] Finding serial sybils (addresses flagged in multiple projects)...")
    sys.stdout.flush()

    # Concatenate all sybils into one DataFrame
    all_sybils = pd.concat(project_sybils.values(), ignore_index=True)
    print(f"  Total sybil records across all projects: {len(all_sybils):,}")

    # Count projects per address
    addr_project_count = all_sybils.groupby("address")["project"].nunique()
    total_sybil_addrs = len(addr_project_count)
    serial_2plus = int((addr_project_count >= 2).sum())
    serial_3plus = int((addr_project_count >= 3).sum())
    serial_5plus = int((addr_project_count >= 5).sum())

    # Distribution of project counts
    n_projects_counts = addr_project_count.value_counts().sort_index()

    print(f"\n  Total unique sybil addresses (any project): {total_sybil_addrs:,}")
    print(f"  Sybil in 2+ projects (serial sybils):       {serial_2plus:,} ({100*serial_2plus/total_sybil_addrs:.2f}%)")
    print(f"  Sybil in 3+ projects:                       {serial_3plus:,} ({100*serial_3plus/total_sybil_addrs:.2f}%)")
    print(f"  Sybil in 5+ projects:                       {serial_5plus:,} ({100*serial_5plus/total_sybil_addrs:.2f}%)")

    print("\n  Distribution of # projects per sybil address:")
    for k, v in n_projects_counts.items():
        print(f"    {k} project(s): {v:,}")
    sys.stdout.flush()

    # ============================================================
    # 3. SERIAL SYBIL INDICATOR ANALYSIS
    # ============================================================
    print("\n[3] Analyzing indicator patterns for serial sybils...")
    sys.stdout.flush()

    # Get serial sybil addresses (2+ projects)
    serial_addrs = set(addr_project_count[addr_project_count >= 2].index)
    serial_records = all_sybils[all_sybils["address"].isin(serial_addrs)]

    # 3a: Which indicators catch serial sybils?
    indicator_catch_counts = {}
    for ind in INDICATORS:
        indicator_catch_counts[ind] = int(serial_records[f"{ind}_trig"].sum())

    print("\n  Indicator catch frequency for serial sybils (total flags across projects):")
    for ind in INDICATORS:
        print(f"    {ind}: {indicator_catch_counts[ind]:,}")

    # 3b: Do serial sybils get caught by SAME indicator across projects?
    # Group serial records by address
    same_indicator_count = 0
    different_indicator_count = 0
    mixed_count = 0

    serial_grouped = serial_records.groupby("address")
    for addr, group in serial_grouped:
        if len(group) < 2:
            continue
        # For each row, get set of triggered indicators
        sets_per_project = []
        for _, row in group.iterrows():
            triggered = set()
            for ind in INDICATORS:
                if row[f"{ind}_trig"] == 1:
                    triggered.add(ind)
            sets_per_project.append(triggered)

        intersection = sets_per_project[0].copy()
        union = sets_per_project[0].copy()
        for s in sets_per_project[1:]:
            intersection &= s
            union |= s

        if intersection == union and len(intersection) > 0:
            same_indicator_count += 1
        elif len(intersection) > 0:
            mixed_count += 1
        else:
            different_indicator_count += 1

    total_serial = same_indicator_count + mixed_count + different_indicator_count
    print(f"\n  Serial sybil indicator consistency (n={total_serial:,}):")
    print(f"    Same indicator(s) in ALL projects:      {same_indicator_count:,} ({100*same_indicator_count/max(1,total_serial):.1f}%)")
    print(f"    Mixed (some shared, some different):     {mixed_count:,} ({100*mixed_count/max(1,total_serial):.1f}%)")
    print(f"    Completely different indicators:         {different_indicator_count:,} ({100*different_indicator_count/max(1,total_serial):.1f}%)")
    sys.stdout.flush()

    # 3c: Are serial sybils harder or easier to detect?
    serial_n_ind = serial_records["n_indicators"].values
    nonserial_addrs = set(addr_project_count[addr_project_count == 1].index)
    nonserial_records = all_sybils[all_sybils["address"].isin(nonserial_addrs)]
    nonserial_n_ind = nonserial_records["n_indicators"].values

    serial_mean = float(np.mean(serial_n_ind)) if len(serial_n_ind) > 0 else 0
    nonserial_mean = float(np.mean(nonserial_n_ind)) if len(nonserial_n_ind) > 0 else 0

    print(f"\n  Detection difficulty (avg n_indicators triggered):")
    print(f"    Serial sybils (2+ projects):   {serial_mean:.2f} (n={len(serial_n_ind):,})")
    print(f"    Non-serial sybils (1 project): {nonserial_mean:.2f} (n={len(nonserial_n_ind):,})")
    if serial_mean > nonserial_mean:
        print(f"    -> Serial sybils trigger MORE indicators (easier to detect)")
    else:
        print(f"    -> Serial sybils trigger FEWER indicators (harder to detect)")
    sys.stdout.flush()

    # ============================================================
    # 4. CROSS-PROJECT INDICATOR CORRELATION
    # ============================================================
    print("\n[4] Cross-project indicator score correlation for addresses in 2+ projects...")
    sys.stdout.flush()

    # For pairs of projects, correlate scores of shared serial sybils
    overall_a = {ind: [] for ind in INDICATORS}
    overall_b = {ind: [] for ind in INDICATORS}

    project_list = sorted(project_sybils.keys())
    for i, pa in enumerate(project_list):
        for j, pb in enumerate(project_list):
            if i >= j:
                continue
            sa = project_sybils[pa].set_index("address")
            sb = project_sybils[pb].set_index("address")
            shared = sa.index.intersection(sb.index)
            if len(shared) < 5:
                continue
            for ind in INDICATORS:
                vals_a = sa.loc[shared, ind].values
                vals_b = sb.loc[shared, ind].values
                overall_a[ind].extend(vals_a)
                overall_b[ind].extend(vals_b)

    overall_correlations = {}
    for ind in INDICATORS:
        a = np.array(overall_a[ind])
        b = np.array(overall_b[ind])
        if len(a) >= 10 and np.std(a) > 0 and np.std(b) > 0:
            corr = float(np.corrcoef(a, b)[0, 1])
            overall_correlations[ind] = round(corr, 4)
            print(f"  {ind}: correlation = {corr:.4f} (n={len(a):,} pairs)")
        else:
            overall_correlations[ind] = None
            print(f"  {ind}: insufficient data or zero variance (n={len(a)})")
    sys.stdout.flush()

    # ============================================================
    # 5. TEMPORAL EVOLUTION
    # ============================================================
    print("\n[5] Temporal evolution of sybil rates and sophistication...")
    sys.stdout.flush()

    sorted_projects = sorted(
        [p for p in PROJECTS if p in project_data],
        key=lambda p: PROJECT_DATES.get(p, "2099-01")
    )

    temporal_data = []
    print(f"\n  {'Project':<15} {'Date':<10} {'Total':>10} {'Sybils':>10} {'Rate':>8} {'Avg n_ind':>10} {'1-ind%':>8}")
    print("  " + "-" * 75)

    for proj in sorted_projects:
        df = project_data[proj]
        sybils = df[df["is_sybil"] == 1]
        n_total = len(df)
        n_sybil = len(sybils)
        sybil_rate = n_sybil / n_total if n_total > 0 else 0
        avg_n_ind = float(sybils["n_indicators"].mean()) if n_sybil > 0 else 0
        single_ind_pct = float((sybils["n_indicators"] == 1).sum() / n_sybil) if n_sybil > 0 else 0

        date_str = PROJECT_DATES.get(proj, "unknown")
        print(f"  {proj:<15} {date_str:<10} {n_total:>10,} {n_sybil:>10,} {100*sybil_rate:>7.1f}% {avg_n_ind:>10.2f} {100*single_ind_pct:>7.1f}%")

        temporal_data.append({
            "project": proj,
            "date": date_str,
            "total": n_total,
            "sybils": n_sybil,
            "sybil_rate": round(sybil_rate, 4),
            "avg_n_indicators": round(avg_n_ind, 3),
            "single_indicator_pct": round(single_ind_pct, 4),
        })

    orders = list(range(len(temporal_data)))
    rates = [d["sybil_rate"] for d in temporal_data]
    avg_n_inds = [d["avg_n_indicators"] for d in temporal_data]
    single_pcts = [d["single_indicator_pct"] for d in temporal_data]

    rate_corr = float(np.corrcoef(orders, rates)[0, 1]) if len(orders) >= 3 else 0
    n_ind_corr = float(np.corrcoef(orders, avg_n_inds)[0, 1]) if len(orders) >= 3 else 0
    single_corr = float(np.corrcoef(orders, single_pcts)[0, 1]) if len(orders) >= 3 else 0

    print(f"\n  Temporal correlations (positive = increasing over time):")
    print(f"    Sybil rate vs. time:            r = {rate_corr:.4f}")
    print(f"    Avg n_indicators vs. time:      r = {n_ind_corr:.4f}")
    print(f"    Single-indicator sybil% vs time: r = {single_corr:.4f}")

    if single_corr > 0:
        print(f"    -> Sybils becoming MORE SOPHISTICATED (fewer indicators triggered)")
    else:
        print(f"    -> No clear trend toward more sophisticated sybils")
    sys.stdout.flush()

    # ============================================================
    # 6. TOP SERIAL SYBILS
    # ============================================================
    print("\n[6] Top serial sybils (most projects)...")
    sys.stdout.flush()

    # Get top addresses by project count
    top_addrs = addr_project_count.nlargest(20)
    print(f"\n  {'Address':<45} {'#Proj':>5} {'Projects'}")
    print("  " + "-" * 90)
    top_serial_list = []

    for addr, n_proj in top_addrs.items():
        projs_for_addr = all_sybils[all_sybils["address"] == addr]["project"].tolist()
        projs_str = ", ".join(sorted(projs_for_addr))
        print(f"  {addr:<45} {n_proj:>5} {projs_str}")

        # Get indicator info per project
        addr_rows = all_sybils[all_sybils["address"] == addr]
        indicators_per_proj = {}
        for _, row in addr_rows.iterrows():
            triggered = [ind for ind in INDICATORS if row[f"{ind}_trig"] == 1]
            indicators_per_proj[row["project"]] = triggered

        top_serial_list.append({
            "address": addr,
            "n_projects": int(n_proj),
            "projects": sorted(projs_for_addr),
            "indicators": indicators_per_proj,
        })

    # ============================================================
    # 7. MOST COMMON INDICATOR COMBOS FOR SERIAL SYBILS
    # ============================================================
    print("\n[7] Most common indicator combos for serial sybils across all their appearances...")
    sys.stdout.flush()

    combo_counter = serial_records["_combo"].value_counts()

    print(f"\n  {'Indicator Combination':<40} {'Count':>8} {'%':>8}")
    print("  " + "-" * 60)
    total_combos = combo_counter.sum()
    for combo_str, count in combo_counter.head(15).items():
        display = combo_str if combo_str else "(none)"
        print(f"  {display:<40} {count:>8,} {100*count/total_combos:>7.1f}%")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    results = {
        "summary": {
            "total_unique_sybil_addresses": total_sybil_addrs,
            "serial_2plus": serial_2plus,
            "serial_3plus": serial_3plus,
            "serial_5plus": serial_5plus,
            "serial_2plus_pct": round(100 * serial_2plus / max(1, total_sybil_addrs), 2),
            "serial_3plus_pct": round(100 * serial_3plus / max(1, total_sybil_addrs), 2),
            "serial_5plus_pct": round(100 * serial_5plus / max(1, total_sybil_addrs), 2),
        },
        "n_projects_distribution": {str(k): int(v) for k, v in n_projects_counts.items()},
        "indicator_consistency": {
            "same_all_projects": same_indicator_count,
            "mixed": mixed_count,
            "completely_different": different_indicator_count,
        },
        "detection_difficulty": {
            "serial_avg_n_indicators": round(serial_mean, 3),
            "nonserial_avg_n_indicators": round(nonserial_mean, 3),
            "serial_easier": serial_mean > nonserial_mean,
        },
        "cross_project_correlations": overall_correlations,
        "temporal_evolution": temporal_data,
        "temporal_trends": {
            "sybil_rate_vs_time_corr": round(rate_corr, 4),
            "avg_n_indicators_vs_time_corr": round(n_ind_corr, 4),
            "single_indicator_pct_vs_time_corr": round(single_corr, 4),
        },
        "top_serial_sybils": top_serial_list[:10],
        "serial_indicator_combos": [
            {"combo": combo_str, "count": int(count)}
            for combo_str, count in combo_counter.head(20).items()
        ],
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("SERIAL SYBIL ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
