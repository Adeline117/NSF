"""
Paper 3: Minimal exp4 leakage verification on 4 projects
==========================================================
Runs the FIXED augment_with_ai_features on a subset of projects
(Uniswap, 1inch, ENS, Blur S2) to verify the leakage fix.

Full 16-project rerun takes ~80 minutes — this focused version
is ~15 min and demonstrates the same result for the 4 largest
or most-representative projects.

Compares HONEST (fixed) vs LEAKY (original) numbers by reading
the saved exp4_enhanced_detector from experiment_large_scale_results.json.
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_large_scale import (
    load_project,
    load_real_ai_calibration,
    augment_with_ai_features,
    AI_FEATURE_NAMES,
    OPS_COLS,
    GBM_PARAMS,
)

OUT_PATH = SCRIPT_DIR / "exp4_leakage_fix_results.json"

# Focused subset for fast verification
PROJECTS_SUBSET = ["1inch", "uniswap", "ens", "blur_s2"]


def run_lopo_fixed(projects_data, ai_calibration, rng):
    """Run exp4 enhanced-detector LOPO with the FIXED augmentation."""
    augmented = {}
    for name, df in projects_data.items():
        augmented[name] = augment_with_ai_features(df, ai_calibration, rng)

    base_features = OPS_COLS
    enhanced_features = OPS_COLS + AI_FEATURE_NAMES
    ai_only_features = AI_FEATURE_NAMES

    results = {}
    for test_proj in projects_data.keys():
        print(f"\n{test_proj}:")
        test_df = augmented[test_proj]
        y_test = test_df["fund_flag"].values.astype(int)
        if y_test.sum() < 10 or (len(y_test) - y_test.sum()) < 10:
            print(f"  insufficient class balance, skip")
            continue

        train_dfs = [augmented[p] for p in projects_data if p != test_proj]
        train_df = pd.concat(train_dfs, ignore_index=True)
        y_train = train_df["fund_flag"].values.astype(int)

        # Baseline: OPS only
        clf_b = GradientBoostingClassifier(**GBM_PARAMS)
        clf_b.fit(train_df[base_features].values, y_train)
        probs_b = clf_b.predict_proba(test_df[base_features].values)[:, 1]
        auc_b = roc_auc_score(y_test, probs_b)
        print(f"  baseline (OPS)      AUC={auc_b:.4f}", flush=True)

        # Enhanced: OPS + AI
        clf_e = GradientBoostingClassifier(**GBM_PARAMS)
        clf_e.fit(train_df[enhanced_features].values, y_train)
        probs_e = clf_e.predict_proba(test_df[enhanced_features].values)[:, 1]
        auc_e = roc_auc_score(y_test, probs_e)
        print(f"  enhanced (OPS+AI)   AUC={auc_e:.4f}", flush=True)

        # AI-only
        clf_a = GradientBoostingClassifier(**GBM_PARAMS)
        clf_a.fit(train_df[ai_only_features].values, y_train)
        probs_a = clf_a.predict_proba(test_df[ai_only_features].values)[:, 1]
        auc_a = roc_auc_score(y_test, probs_a)
        print(f"  ai_only             AUC={auc_a:.4f}", flush=True)

        results[test_proj] = {
            "baseline_auc": round(float(auc_b), 4),
            "enhanced_auc": round(float(auc_e), 4),
            "ai_only_auc": round(float(auc_a), 4),
            "improvement": round(float(auc_e - auc_b), 4),
        }
        print(f"  --- delta={auc_e - auc_b:+.4f}", flush=True)

    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("Paper 3: Minimal exp4 Leakage Verification")
    print("=" * 70)

    # Load subset
    projects_data = {}
    for proj in PROJECTS_SUBSET:
        df = load_project(proj)
        if not df.empty:
            projects_data[proj] = df
            print(f"  {proj}: {len(df)} rows ({int(df['is_sybil'].sum())} sybils)")

    ai_calibration = load_real_ai_calibration()

    rng = np.random.RandomState(42)
    honest_results = run_lopo_fixed(projects_data, ai_calibration, rng)

    # Load leaky results for comparison
    leaky_path = SCRIPT_DIR / "experiment_large_scale_results.json"
    leaky = {}
    if leaky_path.exists():
        with open(leaky_path) as f:
            d = json.load(f)
        leaky = d.get("exp4_enhanced_detector", {})

    # Build comparison
    comparison = {}
    for proj, honest in honest_results.items():
        leaky_proj = leaky.get(proj, {})
        comparison[proj] = {
            "honest_base_auc": honest["baseline_auc"],
            "leaky_base_auc": leaky_proj.get("baseline_auc"),
            "honest_enh_auc": honest["enhanced_auc"],
            "leaky_enh_auc": leaky_proj.get("enhanced_auc"),
            "honest_ai_only_auc": honest["ai_only_auc"],
            "leaky_ai_only_auc": leaky_proj.get("ai_only_auc"),
            "honest_improvement": honest["improvement"],
            "leaky_improvement": leaky_proj.get("improvement"),
            "inflation_of_improvement": round(
                (leaky_proj.get("improvement", 0) or 0)
                - honest["improvement"],
                4,
            ),
        }

    print("\n" + "=" * 90)
    print("HONEST vs LEAKY comparison")
    print("=" * 90)
    print(f"{'Project':<12} {'Base':<10} {'EnhHonest':<12} {'EnhLeaky':<12} "
          f"{'AI-onlyH':<10} {'AI-onlyL':<10} {'Inflation':<12}")
    print("-" * 90)
    for proj, c in comparison.items():
        print(f"{proj:<12} {c['honest_base_auc']:<10.4f} "
              f"{c['honest_enh_auc']:<12.4f} {c['leaky_enh_auc']:<12} "
              f"{c['honest_ai_only_auc']:<10.4f} {c['leaky_ai_only_auc']:<10} "
              f"{c['inflation_of_improvement']:+.4f}")

    honest_mean_enh = float(np.mean([
        c["honest_enh_auc"] for c in comparison.values()
        if c["honest_enh_auc"] is not None
    ]))
    leaky_mean_enh = float(np.mean([
        c["leaky_enh_auc"] for c in comparison.values()
        if c["leaky_enh_auc"] is not None
    ]))
    honest_mean_ai = float(np.mean([
        c["honest_ai_only_auc"] for c in comparison.values()
        if c["honest_ai_only_auc"] is not None
    ]))
    leaky_mean_ai = float(np.mean([
        c["leaky_ai_only_auc"] for c in comparison.values()
        if c["leaky_ai_only_auc"] is not None
    ]))

    summary = {
        "honest_mean_enhanced_auc": round(honest_mean_enh, 4),
        "leaky_mean_enhanced_auc": round(leaky_mean_enh, 4),
        "enhancement_inflation": round(leaky_mean_enh - honest_mean_enh, 4),
        "honest_mean_ai_only_auc": round(honest_mean_ai, 4),
        "leaky_mean_ai_only_auc": round(leaky_mean_ai, 4),
        "ai_only_inflation": round(leaky_mean_ai - honest_mean_ai, 4),
    }

    print("\n" + "=" * 70)
    print("Summary (4-project subset):")
    print(f"  Enhanced AUC: honest {honest_mean_enh:.4f} vs leaky {leaky_mean_enh:.4f}  "
          f"(inflation {leaky_mean_enh - honest_mean_enh:+.4f})")
    print(f"  AI-only AUC:  honest {honest_mean_ai:.4f} vs leaky {leaky_mean_ai:.4f}  "
          f"(inflation {leaky_mean_ai - honest_mean_ai:+.4f})")

    out = {
        "honest_results": honest_results,
        "comparison": comparison,
        "summary": summary,
        "elapsed_seconds": round(time.time() - t0, 2),
        "note": (
            "4-project subset (1inch, uniswap, ens, blur_s2) used to "
            "verify the leakage fix without waiting 80 min for full "
            "16-project rerun. Same pattern expected on all 16 projects."
        ),
    }
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
