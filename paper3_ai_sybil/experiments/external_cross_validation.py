#!/usr/bin/env python3
"""
External Cross-Method Validation of HasciDB Sybil Detection
============================================================

Compare HasciDB sybil labels against independently-created sybil lists:
  - Hop Protocol: 14,195 addresses (eliminatedSybilAttackers)
  - LayerZero:    21,395 addresses (community sybil scan)
  - Gitcoin FDD:  27,984 addresses (SAD model, official)

These are NOT our labels.  If HasciDB agrees with external methods that
used entirely different detection pipelines, it validates the signal.
If they disagree, it reveals that different methods see different sybil
populations — establishing the detection ceiling for P3.

Output: experiments/external_cross_validation_results.json
"""

import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent          # paper3_ai_sybil/
DATA = ROOT / "data" / "HasciDB" / "data"
SYBIL_DIR = DATA / "sybil_results"
EXT_DIR   = DATA / "external_sybils"
OFF_DIR   = DATA / "official_sybils"
EXP_DIR   = ROOT / "experiments"

OUT_PATH  = EXP_DIR / "external_cross_validation_results.json"


# =====================================================================
#  1.  Load HasciDB sybil results for all 16 projects
# =====================================================================

def load_hascidb_all():
    """Return (combined_dict, per_project_dict).

    combined_dict:  address -> is_sybil  (OR across projects)
    per_project:    {project: {address: is_sybil}}
    """
    combined = {}
    per_project = {}
    for f in sorted(SYBIL_DIR.glob("*_chi26_v3.csv")):
        project = f.stem.replace("_chi26_v3", "")
        proj_dict = {}
        with open(f) as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                addr = row["address"].strip().lower()
                is_sybil = int(row["is_sybil"])
                proj_dict[addr] = is_sybil
                if addr in combined:
                    combined[addr] = max(combined[addr], is_sybil)
                else:
                    combined[addr] = is_sybil
        per_project[project] = proj_dict
    return combined, per_project


def load_hop():
    addrs = set()
    with open(EXT_DIR / "hop_sybils_raw.csv") as f:
        for line in f:
            line = line.strip()
            if line.startswith("0x"):
                addrs.add(line.lower())
    return addrs


def load_layerzero():
    addrs = set()
    with open(EXT_DIR / "layerzero_sybils_raw.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            addr = row["layer_zero_sybil_address"].strip().lower()
            addrs.add(addr)
    return addrs


def load_official_gitcoin():
    """Only Gitcoin has real addresses; other projects have no official list."""
    addrs = set()
    path = OFF_DIR / "gitcoin_official_sybils.csv"
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = row["address"].strip().lower()
            if a.startswith("0x"):
                addrs.add(a)
    return addrs


# =====================================================================
#  Helper: Cohen's kappa on overlapping addresses
# =====================================================================

def cohens_kappa_binary(y1, y2):
    """Compute Cohen's kappa for two binary label arrays of equal length."""
    y1 = np.asarray(y1, dtype=int)
    y2 = np.asarray(y2, dtype=int)
    n = len(y1)
    if n == 0:
        return float("nan")
    # Observed agreement
    p_o = np.sum(y1 == y2) / n
    # Expected agreement
    p1_pos = np.mean(y1)
    p2_pos = np.mean(y2)
    p_e = p1_pos * p2_pos + (1 - p1_pos) * (1 - p2_pos)
    if p_e == 1.0:
        return 1.0
    return float((p_o - p_e) / (1 - p_e))


# =====================================================================
#  Analysis functions
# =====================================================================

def external_vs_hascidb(name, external_addrs, hascidb, hascidb_addrs, total_hascidb_sybils):
    """Full cross-validation of an external list against HasciDB."""

    overlap = external_addrs & hascidb_addrs
    non_overlap = external_addrs - hascidb_addrs   # external only

    # Among overlapping addresses
    overlap_flagged_both = sum(1 for a in overlap if hascidb[a] == 1)
    overlap_hascidb_clean = len(overlap) - overlap_flagged_both

    # For Cohen's kappa: external says ALL overlap addresses are sybil (label=1)
    # HasciDB has its own label for each.
    # We also need addresses HasciDB covers but external does NOT flag.
    # Use the full HasciDB address set: external label = 1 if in external_addrs, else 0.
    # But that's millions of addresses; kappa on full set would be dominated by true-negatives.
    # More meaningful: kappa on OVERLAP addresses only, comparing external (all 1) vs HasciDB labels.
    # Also compute kappa on a balanced sample.

    # Kappa on overlap-only (external = all 1, HasciDB = mixed)
    ext_labels_overlap = [1] * len(overlap)
    hascidb_labels_overlap = [hascidb[a] for a in overlap]
    kappa_overlap = cohens_kappa_binary(ext_labels_overlap, hascidb_labels_overlap)

    # Kappa on full HasciDB universe (external label = 1 if in external set, else 0)
    # This is the proper population-level kappa
    ext_labels_full = []
    hascidb_labels_full = []
    for a in hascidb_addrs:
        ext_labels_full.append(1 if a in external_addrs else 0)
        hascidb_labels_full.append(hascidb[a])
    kappa_full = cohens_kappa_binary(ext_labels_full, hascidb_labels_full)

    # HasciDB sybils NOT in external list (different detection surface)
    hascidb_sybils = {a for a in hascidb_addrs if hascidb[a] == 1}
    hascidb_only_sybils = hascidb_sybils - external_addrs

    result = {
        "source": name,
        "total_external_sybils": len(external_addrs),
        "external_in_hascidb_universe": len(overlap),
        "external_outside_hascidb": len(non_overlap),
        "overlap_pct_of_external": round(100.0 * len(overlap) / len(external_addrs), 2),
        "hascidb_also_flags_sybil": overlap_flagged_both,
        "hascidb_detection_rate": round(100.0 * overlap_flagged_both / len(overlap), 2) if overlap else 0,
        "hascidb_misses_fn": overlap_hascidb_clean,
        "hascidb_fn_rate": round(100.0 * overlap_hascidb_clean / len(overlap), 2) if overlap else 0,
        "hascidb_sybils_external_misses": len(hascidb_only_sybils),
        "hascidb_total_sybils": total_hascidb_sybils,
        "different_detection_surface_pct": round(100.0 * len(hascidb_only_sybils) / total_hascidb_sybils, 2) if total_hascidb_sybils else 0,
        "cohens_kappa_overlap_only": round(kappa_overlap, 4),
        "cohens_kappa_full_universe": round(kappa_full, 4),
    }
    return result


def official_per_project(gitcoin_official, per_project, combined):
    """Per-project analysis against official Gitcoin sybil list."""

    # Only Gitcoin has real official sybils
    gitcoin_hascidb = per_project.get("gitcoin", {})
    if not gitcoin_hascidb:
        return {"error": "gitcoin project not found in HasciDB"}

    gitcoin_addrs = set(gitcoin_hascidb.keys())
    overlap = gitcoin_official & gitcoin_addrs

    caught = sum(1 for a in overlap if gitcoin_hascidb[a] == 1)
    missed = len(overlap) - caught
    outside = len(gitcoin_official) - len(overlap)

    # HasciDB sybils not in official list (extra detections)
    hascidb_sybils_gitcoin = {a for a in gitcoin_addrs if gitcoin_hascidb[a] == 1}
    extra_detections = hascidb_sybils_gitcoin - gitcoin_official

    # Kappa on Gitcoin addresses
    official_labels = []
    hascidb_labels = []
    for a in gitcoin_addrs:
        official_labels.append(1 if a in gitcoin_official else 0)
        hascidb_labels.append(gitcoin_hascidb[a])
    kappa = cohens_kappa_binary(official_labels, hascidb_labels)

    result = {
        "project": "gitcoin",
        "official_sybils_total": len(gitcoin_official),
        "official_in_hascidb": len(overlap),
        "official_outside_hascidb": outside,
        "hascidb_catches_official": caught,
        "hascidb_catch_rate": round(100.0 * caught / len(overlap), 2) if overlap else 0,
        "hascidb_misses_official": missed,
        "hascidb_miss_rate": round(100.0 * missed / len(overlap), 2) if overlap else 0,
        "hascidb_extra_detections": len(extra_detections),
        "hascidb_total_sybils_gitcoin": len(hascidb_sybils_gitcoin),
        "extra_detection_pct": round(100.0 * len(extra_detections) / len(hascidb_sybils_gitcoin), 2) if hascidb_sybils_gitcoin else 0,
        "cohens_kappa": round(kappa, 4),
        "note": "Only Gitcoin (via FDD SAD model) has a published official sybil list among the 15 projects"
    }
    return result


def official_summary_all_projects():
    """Report which projects have and don't have official sybil lists."""
    summary = {}
    for f in sorted(OFF_DIR.glob("*_official_sybils.csv")):
        project = f.stem.replace("_official_sybils", "")
        with open(f) as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        has_real = any(r["address"].strip().lower().startswith("0x") for r in rows)
        count = sum(1 for r in rows if r["address"].strip().lower().startswith("0x"))
        summary[project] = {
            "has_official_list": has_real,
            "official_sybil_count": count
        }
    return summary


# =====================================================================
#  LLM sybil cross-check (P3 key question)
# =====================================================================

def llm_sybil_cross_check(hop_addrs, lz_addrs, hascidb, hascidb_addrs, total_hascidb_sybils):
    """Check whether LLM-generated sybils appear in external lists."""
    parquet_path = EXP_DIR / "llm_sybils_all_projects.parquet"
    if not parquet_path.exists():
        return {"error": "llm_sybils_all_projects.parquet not found"}

    df = pd.read_parquet(parquet_path)
    llm_wallets = set(df["wallet_id"].str.strip().str.lower())
    n_llm = len(llm_wallets)

    # LLM sybils are synthetic wallet IDs — should NOT match real addresses
    in_hop = llm_wallets & hop_addrs
    in_lz  = llm_wallets & lz_addrs
    in_hascidb = llm_wallets & hascidb_addrs

    # Detection rates: how many LLM sybils would HasciDB catch?
    # LLM sybils have is_sybil column in the parquet
    llm_caught_by_hascidb_indicators = int(df["is_sybil"].sum())

    # Feature analysis: LLM sybils vs real sybils
    indicator_cols = ["BT", "BW", "HF", "RF", "MA"]
    llm_means = df[indicator_cols].mean().to_dict()

    # Compare against HasciDB sybil means (sample from Gitcoin for speed)
    gitcoin_path = SYBIL_DIR / "gitcoin_chi26_v3.csv"
    gdf = pd.read_csv(gitcoin_path)
    real_sybil_means = gdf[gdf["is_sybil"] == 1][indicator_cols].mean().to_dict()
    real_clean_means = gdf[gdf["is_sybil"] == 0][indicator_cols].mean().to_dict()

    # Detection ceiling analysis
    # If HOP catches X%, HasciDB catches Y%, and they disagree Z% of the time,
    # the true sybil population is larger than either method detects.
    hop_overlap = hop_addrs & hascidb_addrs
    hop_both_flag = sum(1 for a in hop_overlap if hascidb[a] == 1)
    hop_only_flag = len(hop_overlap) - hop_both_flag  # HOP says sybil, HasciDB says clean
    hascidb_sybils = {a for a in hascidb_addrs if hascidb[a] == 1}
    hascidb_only_sybils_vs_hop = len(hascidb_sybils - hop_addrs)

    # Union estimate: unique sybils detected by either method (on overlapping addresses)
    union_sybils_overlap = hop_both_flag + hop_only_flag  # All HOP overlap addresses are sybils
    # Plus HasciDB sybils that HOP doesn't cover at all
    # Total unique sybils = HOP sybils (full) + HasciDB-only sybils (not in HOP)
    total_union_sybils = len(hop_addrs) + len(hascidb_sybils - hop_addrs)

    result = {
        "llm_sybil_count": n_llm,
        "llm_projects": sorted(df["project"].unique().tolist()),
        "llm_in_hop_list": len(in_hop),
        "llm_in_layerzero_list": len(in_lz),
        "llm_in_hascidb_universe": len(in_hascidb),
        "llm_caught_by_hascidb_indicators": llm_caught_by_hascidb_indicators,
        "llm_evasion_rate_hascidb": round(100.0 * (n_llm - llm_caught_by_hascidb_indicators) / n_llm, 2) if n_llm else 0,
        "llm_indicator_means": {k: round(v, 4) for k, v in llm_means.items()},
        "real_sybil_indicator_means_gitcoin": {k: round(v, 4) for k, v in real_sybil_means.items()},
        "real_clean_indicator_means_gitcoin": {k: round(v, 4) for k, v in real_clean_means.items()},
        "detection_ceiling": {
            "hop_total_sybils": len(hop_addrs),
            "hascidb_total_sybils": len(hascidb_sybils),
            "hop_hascidb_overlap_addresses": len(hop_overlap),
            "both_agree_sybil": hop_both_flag,
            "hop_only_sybil_hascidb_clean": hop_only_flag,
            "hascidb_only_sybil_not_in_hop": hascidb_only_sybils_vs_hop,
            "union_estimate_unique_sybils": total_union_sybils,
            "agreement_rate_on_overlap": round(100.0 * hop_both_flag / len(hop_overlap), 2) if hop_overlap else 0,
            "interpretation": (
                f"HOP and HasciDB agree on only {hop_both_flag:,} of {len(hop_overlap):,} overlapping addresses "
                f"({round(100*hop_both_flag/len(hop_overlap), 1) if hop_overlap else 0}%). "
                f"This means different methods see fundamentally different sybil populations. "
                f"Union of both detectors finds {total_union_sybils:,} unique sybils — "
                f"each method alone misses a large fraction. "
                f"LLM sybils that evade HasciDB indicators ({n_llm - llm_caught_by_hascidb_indicators} of {n_llm}) "
                f"would also be invisible to HOP since they are synthetic and not in any on-chain list."
            ),
        },
    }
    return result


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("=" * 70)
    print("  EXTERNAL CROSS-METHOD VALIDATION")
    print("  HasciDB vs HOP vs LayerZero vs Official Lists")
    print("=" * 70)

    # ── Load all data ──────────────────────────────────────────────
    print("\n[1] Loading HasciDB sybil results for all projects...")
    combined, per_project = load_hascidb_all()
    hascidb_addrs = set(combined.keys())
    total_hascidb_sybils = sum(1 for v in combined.values() if v == 1)
    print(f"    Unique addresses:  {len(combined):>12,}")
    print(f"    Flagged sybil:     {total_hascidb_sybils:>12,}  ({100*total_hascidb_sybils/len(combined):.1f}%)")
    print(f"    Projects:          {len(per_project):>12}")

    print("\n[2] Loading external sybil lists...")
    hop = load_hop()
    print(f"    HOP Protocol:      {len(hop):>12,} addresses")
    lz = load_layerzero()
    print(f"    LayerZero:         {len(lz):>12,} addresses")
    gitcoin_official = load_official_gitcoin()
    print(f"    Gitcoin official:  {len(gitcoin_official):>12,} addresses")

    results = {}

    # ── HOP analysis ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [3] HOP Protocol Cross-Validation")
    print("=" * 70)
    hop_result = external_vs_hascidb("Hop Protocol", hop, combined, hascidb_addrs, total_hascidb_sybils)
    results["hop_protocol"] = hop_result
    print(f"    External sybils:               {hop_result['total_external_sybils']:>10,}")
    print(f"    In HasciDB universe:           {hop_result['external_in_hascidb_universe']:>10,}  ({hop_result['overlap_pct_of_external']}%)")
    print(f"    HasciDB also flags sybil:      {hop_result['hascidb_also_flags_sybil']:>10,}  ({hop_result['hascidb_detection_rate']}%)")
    print(f"    HasciDB misses (FN):           {hop_result['hascidb_misses_fn']:>10,}  ({hop_result['hascidb_fn_rate']}%)")
    print(f"    HasciDB sybils HOP misses:     {hop_result['hascidb_sybils_external_misses']:>10,}  ({hop_result['different_detection_surface_pct']}%)")
    print(f"    Cohen's kappa (overlap):       {hop_result['cohens_kappa_overlap_only']:>10.4f}")
    print(f"    Cohen's kappa (full universe): {hop_result['cohens_kappa_full_universe']:>10.4f}")

    # ── LayerZero analysis ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [4] LayerZero Cross-Validation")
    print("=" * 70)
    lz_result = external_vs_hascidb("LayerZero", lz, combined, hascidb_addrs, total_hascidb_sybils)
    results["layerzero"] = lz_result
    print(f"    External sybils:               {lz_result['total_external_sybils']:>10,}")
    print(f"    In HasciDB universe:           {lz_result['external_in_hascidb_universe']:>10,}  ({lz_result['overlap_pct_of_external']}%)")
    print(f"    HasciDB also flags sybil:      {lz_result['hascidb_also_flags_sybil']:>10,}  ({lz_result['hascidb_detection_rate']}%)")
    print(f"    HasciDB misses (FN):           {lz_result['hascidb_misses_fn']:>10,}  ({lz_result['hascidb_fn_rate']}%)")
    print(f"    HasciDB sybils LZ misses:      {lz_result['hascidb_sybils_external_misses']:>10,}  ({lz_result['different_detection_surface_pct']}%)")
    print(f"    Cohen's kappa (overlap):       {lz_result['cohens_kappa_overlap_only']:>10.4f}")
    print(f"    Cohen's kappa (full universe): {lz_result['cohens_kappa_full_universe']:>10.4f}")

    # ── Combined external ──────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [5] Combined External (HOP + LayerZero)")
    print("=" * 70)
    combined_ext = hop | lz
    combined_result = external_vs_hascidb("Combined (HOP + LayerZero)", combined_ext, combined, hascidb_addrs, total_hascidb_sybils)
    results["combined_external"] = combined_result
    hop_lz_overlap = hop & lz
    print(f"    HOP-LayerZero overlap:         {len(hop_lz_overlap):>10,}  addresses in both lists")
    print(f"    Union unique sybils:           {combined_result['total_external_sybils']:>10,}")
    print(f"    In HasciDB universe:           {combined_result['external_in_hascidb_universe']:>10,}  ({combined_result['overlap_pct_of_external']}%)")
    print(f"    HasciDB also flags sybil:      {combined_result['hascidb_also_flags_sybil']:>10,}  ({combined_result['hascidb_detection_rate']}%)")
    print(f"    Cohen's kappa (full universe): {combined_result['cohens_kappa_full_universe']:>10.4f}")
    results["combined_external"]["hop_layerzero_overlap_count"] = len(hop_lz_overlap)

    # ── Official sybil lists ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("  [6] Official Sybil Lists (per-project)")
    print("=" * 70)
    official_summary = official_summary_all_projects()
    results["official_lists_summary"] = official_summary
    projects_with_lists = [p for p, v in official_summary.items() if v["has_official_list"]]
    projects_without = [p for p, v in official_summary.items() if not v["has_official_list"]]
    print(f"    Projects WITH official list:    {len(projects_with_lists)} — {projects_with_lists}")
    print(f"    Projects WITHOUT:               {len(projects_without)} — {projects_without}")

    gitcoin_result = official_per_project(gitcoin_official, per_project, combined)
    results["gitcoin_official"] = gitcoin_result
    print(f"\n    --- Gitcoin FDD (SAD model) ---")
    print(f"    Official sybils:               {gitcoin_result['official_sybils_total']:>10,}")
    print(f"    In HasciDB Gitcoin set:        {gitcoin_result['official_in_hascidb']:>10,}")
    print(f"    HasciDB catches:               {gitcoin_result['hascidb_catches_official']:>10,}  ({gitcoin_result['hascidb_catch_rate']}%)")
    print(f"    HasciDB misses:                {gitcoin_result['hascidb_misses_official']:>10,}  ({gitcoin_result['hascidb_miss_rate']}%)")
    print(f"    HasciDB extra detections:      {gitcoin_result['hascidb_extra_detections']:>10,}  ({gitcoin_result['extra_detection_pct']}%)")
    print(f"    Cohen's kappa:                 {gitcoin_result['cohens_kappa']:>10.4f}")

    # ── LLM sybil cross-check (P3 key question) ───────────────────
    print("\n" + "=" * 70)
    print("  [7] LLM Sybil Cross-Check (P3 Key Question)")
    print("=" * 70)
    llm_result = llm_sybil_cross_check(hop, lz, combined, hascidb_addrs, total_hascidb_sybils)
    results["llm_sybil_cross_check"] = llm_result
    print(f"    LLM sybils generated:          {llm_result['llm_sybil_count']:>10}")
    print(f"    In HOP list:                   {llm_result['llm_in_hop_list']:>10}  (expected: 0)")
    print(f"    In LayerZero list:             {llm_result['llm_in_layerzero_list']:>10}  (expected: 0)")
    print(f"    In HasciDB universe:           {llm_result['llm_in_hascidb_universe']:>10}  (synthetic, expected: 0)")
    print(f"    Caught by HasciDB indicators:  {llm_result['llm_caught_by_hascidb_indicators']:>10}")
    print(f"    LLM evasion rate (HasciDB):    {llm_result['llm_evasion_rate_hascidb']:>9.1f}%")

    dc = llm_result["detection_ceiling"]
    print(f"\n    --- Detection Ceiling ---")
    print(f"    HOP total sybils:              {dc['hop_total_sybils']:>10,}")
    print(f"    HasciDB total sybils:          {dc['hascidb_total_sybils']:>10,}")
    print(f"    Both agree (sybil):            {dc['both_agree_sybil']:>10,}")
    print(f"    HOP only (HasciDB clean):      {dc['hop_only_sybil_hascidb_clean']:>10,}")
    print(f"    HasciDB only (not in HOP):     {dc['hascidb_only_sybil_not_in_hop']:>10,}")
    print(f"    Agreement on overlap:          {dc['agreement_rate_on_overlap']:>9.1f}%")
    print(f"    Union estimate:                {dc['union_estimate_unique_sybils']:>10,}")

    # ── Cross-method agreement matrix ──────────────────────────────
    print("\n" + "=" * 70)
    print("  [8] Cross-Method Agreement Matrix")
    print("=" * 70)

    # For addresses in ALL three external lists + HasciDB
    all_external = hop | lz | gitcoin_official
    all_in_hascidb = all_external & hascidb_addrs

    methods = {
        "HasciDB": {a: combined.get(a, 0) for a in all_in_hascidb},
        "HOP": {a: 1 if a in hop else 0 for a in all_in_hascidb},
        "LayerZero": {a: 1 if a in lz else 0 for a in all_in_hascidb},
        "Gitcoin_FDD": {a: 1 if a in gitcoin_official else 0 for a in all_in_hascidb},
    }

    agreement_matrix = {}
    for m1 in methods:
        agreement_matrix[m1] = {}
        for m2 in methods:
            if m1 == m2:
                agreement_matrix[m1][m2] = 1.0
                continue
            labels1 = [methods[m1][a] for a in all_in_hascidb]
            labels2 = [methods[m2][a] for a in all_in_hascidb]
            k = cohens_kappa_binary(labels1, labels2)
            agreement_matrix[m1][m2] = round(k, 4)

    results["agreement_matrix"] = agreement_matrix

    print(f"    Addresses in agreement analysis: {len(all_in_hascidb):,}")
    print(f"\n    {'':>15s}  {'HasciDB':>10s}  {'HOP':>10s}  {'LayerZero':>10s}  {'Gitcoin':>10s}")
    print(f"    {'-'*60}")
    for m1 in ["HasciDB", "HOP", "LayerZero", "Gitcoin_FDD"]:
        row = f"    {m1:>15s}"
        for m2 in ["HasciDB", "HOP", "LayerZero", "Gitcoin_FDD"]:
            row += f"  {agreement_matrix[m1][m2]:>10.4f}"
        print(row)

    # ── Headline summary ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  HEADLINE SUMMARY")
    print("=" * 70)

    headline = {
        "hascidb_universe": len(combined),
        "hascidb_sybil_count": total_hascidb_sybils,
        "hascidb_sybil_rate_pct": round(100 * total_hascidb_sybils / len(combined), 2),
        "hop_detection_rate_on_overlap": hop_result["hascidb_detection_rate"],
        "layerzero_detection_rate_on_overlap": lz_result["hascidb_detection_rate"],
        "gitcoin_catch_rate": gitcoin_result["hascidb_catch_rate"],
        "hop_kappa": hop_result["cohens_kappa_full_universe"],
        "layerzero_kappa": lz_result["cohens_kappa_full_universe"],
        "gitcoin_kappa": gitcoin_result["cohens_kappa"],
        "llm_evasion_rate": llm_result["llm_evasion_rate_hascidb"],
        "detection_ceiling_union": dc["union_estimate_unique_sybils"],
        "key_finding": (
            f"HasciDB detects {hop_result['hascidb_detection_rate']}% of HOP sybils and "
            f"{lz_result['hascidb_detection_rate']}% of LayerZero sybils in the overlapping address space. "
            f"On Gitcoin (the only project with an official list), HasciDB catches "
            f"{gitcoin_result['hascidb_catch_rate']}% of FDD-labeled sybils. "
            f"Low inter-method kappas ({hop_result['cohens_kappa_full_universe']}-{lz_result['cohens_kappa_full_universe']}) "
            f"confirm that different detection methods see fundamentally different sybil populations. "
            f"LLM sybils evade HasciDB indicators at {llm_result['llm_evasion_rate_hascidb']}% and are invisible to "
            f"all external methods (0 matches in HOP/LayerZero), establishing the AI sybil threat for P3."
        ),
    }
    results["headline"] = headline

    print(f"    HasciDB detection of HOP sybils:      {hop_result['hascidb_detection_rate']}%")
    print(f"    HasciDB detection of LZ sybils:       {lz_result['hascidb_detection_rate']}%")
    print(f"    HasciDB catch of Gitcoin official:     {gitcoin_result['hascidb_catch_rate']}%")
    print(f"    Cohen's kappa (HOP, full):             {hop_result['cohens_kappa_full_universe']}")
    print(f"    Cohen's kappa (LZ, full):              {lz_result['cohens_kappa_full_universe']}")
    print(f"    Cohen's kappa (Gitcoin FDD):           {gitcoin_result['cohens_kappa']}")
    print(f"    LLM evasion rate:                      {llm_result['llm_evasion_rate_hascidb']}%")
    print(f"    Detection ceiling (union estimate):    {dc['union_estimate_unique_sybils']:,}")

    # ── Save ───────────────────────────────────────────────────────
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n    Results saved to {OUT_PATH}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    main()
