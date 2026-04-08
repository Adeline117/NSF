"""
Paper 3: AI Feature Ablation Study
====================================
Three ablation protocols on the FIXED exp4b pipeline
(LLM-sybil vs real-HasciDB binary classification):

  1. Leave-one-feature-out (LOFO):
     Remove each of the 8 AI features one at a time.
     Report AUC delta per feature.

  2. Single-feature:
     Train with ONLY one AI feature.
     Rank features by isolated discriminative power.

  3. Significance-gated subset:
     Train with only the top-N features by Cohen's d
     (from real_ai_features.json).

Answers: "Is this a 1-feature, 3-feature, or 8-feature result?"
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_large_scale import (
    load_project,
    load_real_ai_calibration,
    augment_with_ai_features,
    generate_ai_sybils_calibrated,
    AI_FEATURE_NAMES,
    OPS_COLS,
    INDICATOR_COLS,
    GBM_PARAMS,
    PROJECTS,
)

OUT_PATH = SCRIPT_DIR / "experiment_ai_feature_ablation_results.json"


def build_mixed_pool(projects_data, ai_calibration, rng, level="advanced", n_per_proj=3000, n_sybils=5000):
    """Build {real + LLM-sybil} mixed pool."""
    real_rows = []
    for name, df in projects_data.items():
        sub = df.sample(n=min(n_per_proj, len(df)), random_state=42)
        aug = augment_with_ai_features(sub, ai_calibration, rng)
        real_rows.append(aug)
    real_pool = pd.concat(real_rows, ignore_index=True)
    real_pool["is_llm_sybil"] = 0

    nonsybil_df = real_pool[real_pool["is_sybil"] == 0].copy()
    ai_sybils = generate_ai_sybils_calibrated(
        n_sybils, ai_calibration, nonsybil_df, rng, level,
    )
    ai_sybils["is_llm_sybil"] = 1

    combined = pd.concat([real_pool, ai_sybils], ignore_index=True)
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return combined


def cv_auc(X, y, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    for tr_idx, te_idx in skf.split(X, y):
        clf = GradientBoostingClassifier(**GBM_PARAMS)
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[te_idx])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te_idx], probs))
        except ValueError:
            pass
    return float(np.mean(aucs)) if aucs else float("nan"), float(np.std(aucs)) if aucs else 0.0


def main():
    t0 = time.time()
    print("=" * 80)
    print("Paper 3: AI Feature Ablation Study")
    print("=" * 80)

    # Load data
    projects_data = {}
    for proj in PROJECTS:
        df = load_project(proj)
        if not df.empty:
            projects_data[proj] = df
    print(f"Loaded {len(projects_data)} projects")

    ai_calibration = load_real_ai_calibration()

    # Load real AI feature Cohen's d ordering
    real_ai_path = SCRIPT_DIR / "real_ai_features.json"
    effect_sizes = {}
    if real_ai_path.exists():
        with open(real_ai_path) as f:
            rai = json.load(f)
        for feat, stats in rai.get("statistical_tests", {}).items():
            if "cohens_d" in stats:
                effect_sizes[feat] = stats["cohens_d"]
    sorted_feats = sorted(effect_sizes.items(), key=lambda x: -x[1])
    print("\nFeatures by Cohen's d:")
    for name, d in sorted_feats:
        print(f"  {name:<30} d={d:.3f}")

    rng = np.random.RandomState(42)
    results = {
        "timestamp": datetime.now().isoformat(),
        "feature_effect_sizes": effect_sizes,
        "levels": {},
    }

    for level in ["basic", "moderate", "advanced"]:
        print(f"\n{'=' * 60}")
        print(f"Level: {level}")
        print(f"{'=' * 60}")

        combined = build_mixed_pool(projects_data, ai_calibration, rng, level=level)
        y = combined["is_llm_sybil"].values.astype(int)
        print(f"N={len(combined)}, LLM sybils={y.sum()}, real={(y==0).sum()}")

        level_res = {}

        # Baseline: all 8 AI features
        X_all = combined[AI_FEATURE_NAMES].values
        auc_all, std_all = cv_auc(X_all, y)
        level_res["all_8_features"] = {"auc": round(auc_all, 4), "std": round(std_all, 4)}
        print(f"\nAll 8 AI features: AUC = {auc_all:.4f} ± {std_all:.4f}")

        # LOFO
        print("\n--- LOFO (leave-one-feature-out) ---")
        lofo_res = {}
        for feat in AI_FEATURE_NAMES:
            remaining = [f for f in AI_FEATURE_NAMES if f != feat]
            X = combined[remaining].values
            auc, std = cv_auc(X, y)
            delta = auc_all - auc
            lofo_res[feat] = {
                "auc_without": round(auc, 4),
                "std": round(std, 4),
                "delta_from_all": round(delta, 4),
            }
            print(f"  w/o {feat:<30} AUC={auc:.4f} delta={delta:+.4f}")
        level_res["lofo"] = lofo_res

        # Single-feature
        print("\n--- Single feature ---")
        single_res = {}
        for feat in AI_FEATURE_NAMES:
            X = combined[[feat]].values
            auc, std = cv_auc(X, y)
            single_res[feat] = {
                "auc": round(auc, 4),
                "std": round(std, 4),
            }
            print(f"  {feat:<30} AUC={auc:.4f}")
        level_res["single_feature"] = single_res

        # Top-N by Cohen's d
        print("\n--- Top-N by Cohen's d ---")
        topn_res = {}
        for n in [1, 2, 3, 4, 5]:
            top_feats = [name for name, _ in sorted_feats[:n]]
            X = combined[top_feats].values
            auc, std = cv_auc(X, y)
            topn_res[f"top_{n}"] = {
                "features": top_feats,
                "auc": round(auc, 4),
                "std": round(std, 4),
            }
            print(f"  top_{n} = {top_feats}: AUC={auc:.4f}")
        level_res["topn_by_cohens_d"] = topn_res

        # OPS baseline for reference
        X_ops = combined[OPS_COLS].values
        auc_ops, std_ops = cv_auc(X_ops, y)
        level_res["ops_only_baseline"] = {
            "auc": round(auc_ops, 4), "std": round(std_ops, 4),
        }
        print(f"\nOPS only (BT,BW,HF) baseline: AUC={auc_ops:.4f}")

        results["levels"][level] = level_res

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}")
    print(f"Total elapsed: {results['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
