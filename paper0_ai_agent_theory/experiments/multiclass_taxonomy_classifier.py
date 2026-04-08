"""
Paper 0: Multi-Class Taxonomy Classifier
==========================================
Trains a multi-class classifier to predict taxonomy category from the
23 behavioral features alone (NO source/name leakage). Answers the
key CHI question: "is each taxonomy category individually predictable
from on-chain behavior?"

Models:
  - GradientBoosting
  - RandomForest
  - LogisticRegression (OvR)

Evaluation:
  - 5-fold stratified CV
  - Per-class precision / recall / F1
  - Confusion matrix
  - Feature importance (GBM)

Outputs:
  - multiclass_classifier_results.json
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PARQUET = (
    PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
)
RESULTS_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "multiclass_classifier_results.json"
)

FEATURE_COLS = [
    "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
    "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
    "burst_frequency",
    "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
    "gas_limit_precision", "gas_price_cv",
    "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
    "unique_contracts_ratio", "top_contract_concentration",
    "method_id_diversity", "contract_to_eoa_ratio",
    "sequential_pattern_score",
    "unlimited_approve_ratio", "approve_revoke_ratio",
    "unverified_contract_approve_ratio",
    "multi_protocol_interaction_count", "flash_loan_usage",
]

TAXONOMY_NAMES = {
    0: "SimpleTradingBot",
    1: "MEVSearcher",
    2: "DeFiManagementAgent",
    3: "LLMPoweredAgent",
    4: "AutonomousDAOAgent",
    5: "CrossChainBridgeAgent",
    6: "DeterministicScript",
    7: "RLTradingAgent",
}


def main():
    print("=" * 80)
    print("Paper 0: Multi-class Taxonomy Classifier (behavioral features only)")
    print("=" * 80)

    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["label"] == 1].copy()  # agents only
    print(f"Agents: {len(df)}")

    X = df[FEATURE_COLS].values.astype(float)
    y = df["taxonomy_index"].values.astype(int)

    # Impute/clip
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u} ({TAXONOMY_NAMES.get(int(u), '?'):<25}) {c}")

    # Drop tiny classes (<20 samples) — can't CV with too few
    mask = np.ones_like(y, dtype=bool)
    for u, c in zip(unique, counts):
        if c < 20:
            print(f"  Dropping class {u} (n={c})")
            mask &= y != u
    X = X[mask]
    y = y[mask]
    print(f"After filter: N={len(y)}, classes={sorted(set(y))}")

    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=5, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=2000, random_state=42,
        ),
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(y)),
        "n_features": len(FEATURE_COLS),
        "feature_names": FEATURE_COLS,
        "classes_present": sorted([int(c) for c in set(y)]),
        "class_counts": {int(u): int(c) for u, c in zip(*np.unique(y, return_counts=True))},
        "models": {},
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model_template in models.items():
        print(f"\nTraining {model_name} ...")
        fold_accs = []
        fold_f1_macro = []
        fold_f1_weighted = []
        all_y_true = []
        all_y_pred = []

        for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[tr_idx])
            X_te = scaler.transform(X[te_idx])

            clf = clone(model_template)
            clf.fit(X_tr, y[tr_idx])
            y_pred = clf.predict(X_te)

            fold_accs.append(accuracy_score(y[te_idx], y_pred))
            fold_f1_macro.append(f1_score(y[te_idx], y_pred, average="macro"))
            fold_f1_weighted.append(
                f1_score(y[te_idx], y_pred, average="weighted"),
            )
            all_y_true.extend(y[te_idx].tolist())
            all_y_pred.extend(y_pred.tolist())

        y_true = np.array(all_y_true)
        y_pred = np.array(all_y_pred)

        cm = confusion_matrix(
            y_true, y_pred,
            labels=sorted(set(y.tolist())),
        ).tolist()
        report = classification_report(
            y_true, y_pred,
            target_names=[TAXONOMY_NAMES.get(int(c), f"C{c}")
                          for c in sorted(set(y.tolist()))],
            output_dict=True,
            zero_division=0,
        )

        results["models"][model_name] = {
            "accuracy_mean": round(float(np.mean(fold_accs)), 4),
            "accuracy_std": round(float(np.std(fold_accs)), 4),
            "f1_macro_mean": round(float(np.mean(fold_f1_macro)), 4),
            "f1_macro_std": round(float(np.std(fold_f1_macro)), 4),
            "f1_weighted_mean": round(float(np.mean(fold_f1_weighted)), 4),
            "f1_weighted_std": round(float(np.std(fold_f1_weighted)), 4),
            "confusion_matrix": cm,
            "confusion_labels": [int(c) for c in sorted(set(y.tolist()))],
            "per_class_report": report,
        }

        print(f"  Accuracy: {np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        print(f"  F1-macro: {np.mean(fold_f1_macro):.4f} ± {np.std(fold_f1_macro):.4f}")
        print(f"  F1-weighted: {np.mean(fold_f1_weighted):.4f}")
        print("  Per-class F1:")
        for cls_name, stats in report.items():
            if isinstance(stats, dict) and "f1-score" in stats:
                print(f"    {cls_name:<25} F1={stats['f1-score']:.4f} "
                      f"P={stats['precision']:.4f} R={stats['recall']:.4f} "
                      f"(n={stats['support']})")

    # GBM feature importance on full data
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gbm = clone(models["GradientBoosting"])
    gbm.fit(Xs, y)
    sorted_idx = np.argsort(gbm.feature_importances_)[::-1]
    results["gbm_feature_importance"] = {
        FEATURE_COLS[i]: round(float(gbm.feature_importances_[i]), 4)
        for i in sorted_idx
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
