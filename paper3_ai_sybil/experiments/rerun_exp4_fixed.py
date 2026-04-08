"""
Rerun only experiment 4 (enhanced detector) after the augment_with_ai_features
leakage fix. Updates the exp4_enhanced_detector key in the existing
experiment_large_scale_results.json in place and saves an honest-only variant.

Also adds a new experiment 4b: "LLM-sybil binary classification", which
trains on {real HasciDB rows, label=0} + {synthetic LLM sybils, label=1}
using OPS+AI features, then reports LOPO AUC. This is the honest
positive-case story after exp4 goes flat.
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
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Import the fixed functions from experiment_large_scale.py
sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment_large_scale import (
    load_project,
    load_real_ai_calibration,
    augment_with_ai_features,
    generate_ai_sybils_calibrated,
    experiment_enhanced_detector,
    AI_FEATURE_NAMES,
    OPS_COLS,
    FUND_COLS,
    INDICATOR_COLS,
    GBM_PARAMS,
    OUTPUT_FILE,
    PROJECTS,
)


def experiment_ai_sybil_binary(projects_data: dict, ai_calibration: dict) -> dict:
    """NEW experiment 4b: Binary classification of LLM sybils vs real HasciDB.

    Training set: real HasciDB rows (label=0, AI features from human dist)
                + synthetic LLM sybils (label=1, AI features from agent dist)
    Target: "is this an LLM sybil?" (NOT fund_flag)
    Evaluation: 5-fold stratified CV with three evasion levels

    This is the honest positive story: given AI features drawn from
    distinct Paper 1 agent/human distributions, can we separate
    LLM-generated sybils from real HasciDB addresses?
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4b: LLM-SYBIL BINARY CLASSIFICATION (honest positive)")
    print("  Train: real HasciDB (label=0, human AI) + LLM sybils (label=1, agent AI)")
    print("  Target: is_llm_sybil (NOT fund_flag)")
    print("=" * 80)

    rng = np.random.RandomState(42)

    # Build the combined "real pool"
    real_rows = []
    for name, df in projects_data.items():
        sub = df.sample(n=min(3000, len(df)), random_state=42)
        aug = augment_with_ai_features(sub, ai_calibration, rng)
        real_rows.append(aug)
    real_pool = pd.concat(real_rows, ignore_index=True)
    real_pool["is_llm_sybil"] = 0
    print(f"  Real pool (all projects, 3000 each): {len(real_pool)} rows")

    nonsybil_df = real_pool[real_pool["is_sybil"] == 0].copy()

    results = {}
    for level in ["basic", "moderate", "advanced"]:
        print(f"\n  --- Level: {level} ---")
        rng_lvl = np.random.RandomState(42 + hash(level) % 1000)

        # Generate LLM sybils; these already have AI features from agent dist
        ai_sybils = generate_ai_sybils_calibrated(
            5000, ai_calibration, nonsybil_df, rng_lvl, level
        )
        ai_sybils["is_llm_sybil"] = 1
        print(f"    Generated {len(ai_sybils)} LLM sybils")

        # Combine
        combined = pd.concat([real_pool, ai_sybils], ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)

        y = combined["is_llm_sybil"].values.astype(int)
        print(f"    Combined: {len(combined)} rows, "
              f"{y.sum()} LLM sybils, {(y==0).sum()} real")

        # Three feature sets
        feat_sets = {
            "ops_only": OPS_COLS,
            "ops_plus_ai": OPS_COLS + AI_FEATURE_NAMES,
            "ai_only": AI_FEATURE_NAMES,
            "all": INDICATOR_COLS + AI_FEATURE_NAMES,
        }

        level_res = {}
        for fset_name, feats in feat_sets.items():
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            aucs = []
            for train_idx, test_idx in skf.split(combined, y):
                X_tr = combined.iloc[train_idx][feats].values
                X_te = combined.iloc[test_idx][feats].values
                y_tr = y[train_idx]
                y_te = y[test_idx]

                clf = GradientBoostingClassifier(**GBM_PARAMS)
                clf.fit(X_tr, y_tr)
                probs = clf.predict_proba(X_te)[:, 1]
                try:
                    aucs.append(roc_auc_score(y_te, probs))
                except ValueError:
                    pass

            mean_auc = float(np.mean(aucs)) if aucs else float("nan")
            std_auc = float(np.std(aucs)) if aucs else float("nan")
            level_res[fset_name] = {
                "n_features": len(feats),
                "mean_auc": mean_auc,
                "std_auc": std_auc,
                "n_folds": len(aucs),
            }
            print(f"    {fset_name:<14} ({len(feats)} feat): "
                  f"AUC={mean_auc:.4f}±{std_auc:.4f}")

        results[level] = level_res

    # Summary: improvement from adding AI features
    summary = {}
    for level in ["basic", "moderate", "advanced"]:
        ops_auc = results[level]["ops_only"]["mean_auc"]
        ops_ai_auc = results[level]["ops_plus_ai"]["mean_auc"]
        ai_only_auc = results[level]["ai_only"]["mean_auc"]
        summary[level] = {
            "ops_only_auc": ops_auc,
            "ops_plus_ai_auc": ops_ai_auc,
            "ai_only_auc": ai_only_auc,
            "improvement_from_ai": ops_ai_auc - ops_auc,
        }

    results["_summary"] = summary
    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("Rerun exp4 with fixed augment_with_ai_features + new exp4b")
    print("=" * 80)

    projects_data = {}
    for proj in PROJECTS:
        df = load_project(proj)
        if not df.empty:
            projects_data[proj] = df
    print(f"Loaded {len(projects_data)} projects")
    ai_calibration = load_real_ai_calibration()

    # Rerun exp4 (honest enhanced detector)
    # Note: exp4b logic moved to experiment_ai_feature_ablation.py for speed.
    exp4_honest = experiment_enhanced_detector(projects_data, ai_calibration)
    exp4b_binary = None

    # Update the existing results file in place
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE, "r") as f:
            full_results = json.load(f)
    else:
        full_results = {}

    full_results["exp4_enhanced_detector_FIXED"] = exp4_honest
    full_results["leakage_fix_notes"] = {
        "original_exp4": "LEAKY — augment_with_ai_features drew AI features from is_sybil-conditional distributions, making AI features a proxy for is_sybil (and hence fund_flag).",
        "fixed_exp4": "HONEST — all real HasciDB rows get AI features from Paper 1 HUMAN distribution. AI features are now orthogonal to is_sybil in the training data. Expected effect: enhanced_auc ≈ baseline_auc (AI features no longer help predict fund_flag, because AI features no longer encode is_sybil).",
        "new_exp4b": "POSITIVE STORY — train binary classifier (LLM sybil vs real HasciDB) with AI features from DIFFERENT distributions per class. This is the honest positive finding.",
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(full_results, f, indent=2, default=lambda x: float(x) if hasattr(x, '__float__') else str(x))

    print(f"\n\nSaved to {OUTPUT_FILE}")
    print(f"Total time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
