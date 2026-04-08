"""
Project-Level Diversity and Transferability Analysis
=====================================================
Analyzes how different sybil behaviors are across projects by computing
indicator distributions, pairwise Jensen-Shannon divergence, and clustering
projects by sybil behavior similarity.

Key questions:
  a) For each project, compute the indicator distribution for sybils
  b) Compute pairwise Jensen-Shannon divergence between project sybil distributions
  c) Cluster projects by sybil behavior similarity
  d) Can a detector trained on one project transfer to another?

Usage:
    python3 paper3_ai_sybil/experiments/project_diversity.py
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from itertools import combinations
from scipy.spatial.distance import jensenshannon
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HASCIDB_DATA = PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
OUTPUT_FILE = SCRIPT_DIR / "project_diversity_results.json"

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


def compute_indicator_trigger_distribution(sybils):
    """Compute the trigger rate distribution for sybils.
    Returns a vector of 5 trigger rates (one per indicator)."""
    rates = []
    for ind in INDICATORS:
        thresh = THRESHOLDS[ind]
        rate = (sybils[ind] >= thresh).mean()
        rates.append(rate)
    return np.array(rates)


def compute_combo_distribution(sybils, top_k=32):
    """Compute the indicator combination distribution for sybils (vectorized).
    Returns a dictionary mapping combo tuples to their frequencies."""
    from collections import Counter
    # Build combo string vectorized
    parts = []
    for ind in INDICATORS:
        col = (sybils[ind] >= THRESHOLDS[ind]).astype(str).replace({"True": ind, "False": ""})
        parts.append(col)
    combo_series = parts[0]
    for p in parts[1:]:
        combo_series = combo_series + "|" + p
    combo_series = combo_series.apply(lambda x: tuple(sorted(filter(None, x.split("|")))))
    combo_counter = Counter(combo_series)

    # Normalize
    total = sum(combo_counter.values())
    return {k: v / total for k, v in combo_counter.items()}, total


def compute_binned_distribution(sybils, indicator, n_bins=20):
    """Compute a binned distribution of an indicator's raw scores for sybils."""
    scores = sybils[indicator].values
    if len(scores) == 0:
        return np.zeros(n_bins)

    # Use fixed bins based on indicator range
    if indicator in ("HF", "RF"):
        bins = np.linspace(0, 1, n_bins + 1)
    else:
        max_val = max(scores.max(), 1)
        bins = np.linspace(0, max_val, n_bins + 1)

    hist, _ = np.histogram(scores, bins=bins)
    # Add small epsilon to avoid zero probabilities for JSD
    hist = hist.astype(float) + 1e-10
    hist /= hist.sum()
    return hist


def main():
    print("=" * 80)
    print("PROJECT-LEVEL DIVERSITY AND TRANSFERABILITY ANALYSIS")
    print("=" * 80)

    # ============================================================
    # 1. LOAD DATA AND COMPUTE PER-PROJECT DISTRIBUTIONS
    # ============================================================
    print("\n[1] Loading data and computing per-project indicator distributions...")

    project_data = {}
    trigger_distributions = {}
    combo_distributions = {}

    for proj in PROJECTS:
        df = load_project(proj)
        if df is None:
            continue
        sybils = df[df["is_sybil"] == 1]
        if len(sybils) == 0:
            print(f"  SKIP: {proj} (no sybils)")
            continue

        project_data[proj] = df
        trigger_distributions[proj] = compute_indicator_trigger_distribution(sybils)
        combo_distributions[proj], _ = compute_combo_distribution(sybils)

    active_projects = list(trigger_distributions.keys())
    n_active = len(active_projects)

    # ============================================================
    # (a) Per-project indicator trigger rates
    # ============================================================
    print(f"\n  {'Project':<15}", end="")
    for ind in INDICATORS:
        print(f" {ind:>8}", end="")
    print(f" {'Sybil%':>8}")
    print("  " + "-" * 60)

    project_profiles = {}
    for proj in active_projects:
        dist = trigger_distributions[proj]
        df = project_data[proj]
        sybil_rate = df["is_sybil"].mean()
        print(f"  {proj:<15}", end="")
        profile = {}
        for i, ind in enumerate(INDICATORS):
            print(f" {dist[i]:>8.3f}", end="")
            profile[ind] = round(float(dist[i]), 4)
        print(f" {100*sybil_rate:>7.1f}%")
        profile["sybil_rate"] = round(float(sybil_rate), 4)
        project_profiles[proj] = profile

    # ============================================================
    # (b) Pairwise Jensen-Shannon divergence
    # ============================================================
    print(f"\n[2] Computing pairwise Jensen-Shannon divergence (trigger rate vectors)...")

    # Method 1: JSD on 5-dimensional trigger rate vectors
    # We treat the trigger rates as probability distributions by normalizing
    jsd_trigger = np.zeros((n_active, n_active))
    for i, pa in enumerate(active_projects):
        for j, pb in enumerate(active_projects):
            if i >= j:
                continue
            da = trigger_distributions[pa]
            db = trigger_distributions[pb]
            # Normalize to probability distributions (add epsilon for zero handling)
            da_norm = (da + 1e-10) / (da + 1e-10).sum()
            db_norm = (db + 1e-10) / (db + 1e-10).sum()
            jsd = float(jensenshannon(da_norm, db_norm))
            jsd_trigger[i, j] = jsd
            jsd_trigger[j, i] = jsd

    # Method 2: JSD on indicator combination distributions
    print(f"\n  Computing JSD on indicator combination distributions...")

    # Create a unified set of all combos
    all_combos = set()
    for dist in combo_distributions.values():
        all_combos.update(dist.keys())
    all_combos = sorted(all_combos)
    combo_to_idx = {c: i for i, c in enumerate(all_combos)}

    jsd_combo = np.zeros((n_active, n_active))
    for i, pa in enumerate(active_projects):
        for j, pb in enumerate(active_projects):
            if i >= j:
                continue
            # Build probability vectors over the unified combo space
            va = np.zeros(len(all_combos)) + 1e-10
            vb = np.zeros(len(all_combos)) + 1e-10
            for combo, prob in combo_distributions[pa].items():
                va[combo_to_idx[combo]] += prob
            for combo, prob in combo_distributions[pb].items():
                vb[combo_to_idx[combo]] += prob
            va /= va.sum()
            vb /= vb.sum()
            jsd = float(jensenshannon(va, vb))
            jsd_combo[i, j] = jsd
            jsd_combo[j, i] = jsd

    # Print JSD matrix (trigger rates)
    print(f"\n  JSD matrix (trigger rate distributions):")
    print(f"  {'':>15}", end="")
    for p in active_projects:
        print(f" {p[:8]:>8}", end="")
    print()
    jsd_matrix_dict = {}
    for i, pa in enumerate(active_projects):
        print(f"  {pa:>15}", end="")
        jsd_matrix_dict[pa] = {}
        for j, pb in enumerate(active_projects):
            print(f" {jsd_trigger[i,j]:>8.4f}", end="")
            jsd_matrix_dict[pa][pb] = round(float(jsd_trigger[i, j]), 4)
        print()

    # ============================================================
    # (c) Cluster projects by sybil behavior similarity
    # ============================================================
    print(f"\n[3] Clustering projects by sybil behavior similarity...")

    # Use the combo-based JSD for clustering (more informative)
    # Convert to condensed distance matrix
    condensed = []
    for i in range(n_active):
        for j in range(i + 1, n_active):
            condensed.append(jsd_combo[i, j])
    condensed = np.array(condensed)

    # Hierarchical clustering
    Z = linkage(condensed, method="ward")

    # Cut at different numbers of clusters
    cluster_results = {}
    for n_clusters in [2, 3, 4, 5]:
        labels = fcluster(Z, t=n_clusters, criterion="maxclust")
        clusters = {}
        for proj, label in zip(active_projects, labels):
            label_str = str(int(label))
            if label_str not in clusters:
                clusters[label_str] = []
            clusters[label_str].append(proj)

        cluster_results[f"{n_clusters}_clusters"] = clusters
        print(f"\n  {n_clusters} clusters:")
        for label, members in sorted(clusters.items()):
            print(f"    Cluster {label}: {', '.join(members)}")

    # ============================================================
    # (d) Transferability analysis
    # ============================================================
    print(f"\n[4] Transferability analysis...")

    # For each pair, compute the overlap in dominant indicators
    print(f"\n  Indicator overlap (Jaccard similarity of top-triggered indicators):")
    # Top-triggered = indicators triggered for >10% of sybils
    project_dominant = {}
    for proj in active_projects:
        dist = trigger_distributions[proj]
        dominant = set()
        for i, ind in enumerate(INDICATORS):
            if dist[i] > 0.10:
                dominant.add(ind)
        project_dominant[proj] = dominant

    # Compute pairwise Jaccard
    jaccard_matrix = np.zeros((n_active, n_active))
    for i, pa in enumerate(active_projects):
        for j, pb in enumerate(active_projects):
            if i == j:
                jaccard_matrix[i, j] = 1.0
                continue
            sa = project_dominant[pa]
            sb = project_dominant[pb]
            if len(sa | sb) == 0:
                jaccard_matrix[i, j] = 0
            else:
                jaccard_matrix[i, j] = len(sa & sb) / len(sa | sb)

    # Average transferability per project
    print(f"\n  {'Project':<15} {'Dominant':<25} {'Avg Jaccard':>12}")
    print("  " + "-" * 55)
    transferability_scores = {}
    for i, proj in enumerate(active_projects):
        dom = project_dominant[proj]
        dom_str = ", ".join(sorted(dom)) if dom else "(none)"
        avg_jaccard = np.mean([jaccard_matrix[i, j] for j in range(n_active) if j != i])
        print(f"  {proj:<15} {dom_str:<25} {avg_jaccard:>12.3f}")
        transferability_scores[proj] = {
            "dominant_indicators": sorted(dom),
            "avg_jaccard": round(float(avg_jaccard), 4),
        }

    # Most and least similar pairs
    print(f"\n  Most similar project pairs (by JSD, lower = more similar):")
    pairs = []
    for i in range(n_active):
        for j in range(i + 1, n_active):
            pairs.append((active_projects[i], active_projects[j], jsd_combo[i, j]))
    pairs.sort(key=lambda x: x[2])

    for pa, pb, jsd in pairs[:10]:
        print(f"    {pa:>15} <-> {pb:<15}: JSD={jsd:.4f}")

    print(f"\n  Most different project pairs:")
    for pa, pb, jsd in pairs[-10:]:
        print(f"    {pa:>15} <-> {pb:<15}: JSD={jsd:.4f}")

    # Overall transferability metric
    avg_jsd = np.mean(condensed)
    print(f"\n  Average pairwise JSD (combo-based): {avg_jsd:.4f}")
    if avg_jsd < 0.3:
        print(f"  -> Low diversity: detectors likely transfer well across projects")
    elif avg_jsd < 0.5:
        print(f"  -> Moderate diversity: some transfer expected, but adaptation needed")
    else:
        print(f"  -> High diversity: project-specific detectors recommended")

    # ============================================================
    # (e) Per-indicator distribution comparison across projects
    # ============================================================
    print(f"\n[5] Per-indicator JSD across projects (are indicators stable?)...")

    per_indicator_jsd = {}
    for ind in INDICATORS:
        # Compute binned distribution for each project
        binned = {}
        for proj in active_projects:
            sybils = project_data[proj][project_data[proj]["is_sybil"] == 1]
            binned[proj] = compute_binned_distribution(sybils, ind, n_bins=20)

        # Pairwise JSD for this indicator
        jsds = []
        for i, pa in enumerate(active_projects):
            for j, pb in enumerate(active_projects):
                if i >= j:
                    continue
                jsd = float(jensenshannon(binned[pa], binned[pb]))
                jsds.append(jsd)

        avg = np.mean(jsds)
        per_indicator_jsd[ind] = {
            "avg_jsd": round(float(avg), 4),
            "min_jsd": round(float(np.min(jsds)), 4),
            "max_jsd": round(float(np.max(jsds)), 4),
        }
        print(f"  {ind}: avg JSD = {avg:.4f} (min={np.min(jsds):.4f}, max={np.max(jsds):.4f})")

    print(f"\n  -> Indicators with LOW JSD are stable across projects (good for transfer)")
    print(f"  -> Indicators with HIGH JSD vary by project (need adaptation)")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        "project_profiles": project_profiles,
        "jsd_trigger_matrix": jsd_matrix_dict,
        "jsd_combo_matrix": {
            pa: {pb: round(float(jsd_combo[i, j]), 4) for j, pb in enumerate(active_projects)}
            for i, pa in enumerate(active_projects)
        },
        "clusters": cluster_results,
        "transferability": transferability_scores,
        "most_similar_pairs": [
            {"a": pa, "b": pb, "jsd": round(jsd, 4)} for pa, pb, jsd in pairs[:10]
        ],
        "most_different_pairs": [
            {"a": pa, "b": pb, "jsd": round(jsd, 4)} for pa, pb, jsd in pairs[-10:]
        ],
        "avg_pairwise_jsd": round(float(avg_jsd), 4),
        "per_indicator_jsd": per_indicator_jsd,
        "project_dominant_indicators": {
            p: sorted(d) for p, d in project_dominant.items()
        },
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    print("\n" + "=" * 80)
    print("PROJECT DIVERSITY ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
