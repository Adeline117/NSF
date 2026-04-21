#!/usr/bin/env python
"""
Paper 0: Class-Balanced Accuracy Analysis
==========================================
Addresses reviewer concern that 95.3% overall accuracy with 60.8% majority class
is barely above majority-class voting. Computes:
  - Overall accuracy (should match ~95.3%)
  - Balanced accuracy (sklearn.metrics.balanced_accuracy_score)
  - Per-class recall
  - Majority-class baseline accuracy

Uses the same 31-feature set and GBM 10-fold CV as multiclass_31features.py.
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────
FEATURES_PARQUET = (
    PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
)
AI_FEATURES_JSON = (
    PROJECT_ROOT / "paper3_ai_sybil" / "experiments" / "real_ai_features.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "class_balanced_accuracy_results.json"
)

# ── Feature columns (same as multiclass_31features.py) ────────────────
ORIGINAL_23 = [
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

AI_8 = [
    "gas_price_precision", "hour_entropy", "behavioral_consistency",
    "action_sequence_perplexity", "error_recovery_pattern",
    "response_latency_variance", "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]

ALL_31 = ORIGINAL_23 + AI_8

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

# ── Config ────────────────────────────────────────────────────────────
N_FOLDS = 10
SEED = 42


def load_and_merge():
    """Load Paper 0's 23-feature dataset and merge with Paper 3's 8 AI features."""
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["label"] == 1].copy()
    print(f"Paper 0 agents: {len(df)}")

    with open(AI_FEATURES_JSON) as f:
        ai_raw = json.load(f)

    ai_rows = []
    for addr, feats in ai_raw["per_address"].items():
        row = {"address": addr}
        for col in AI_8:
            row[col] = feats.get(col, np.nan)
        ai_rows.append(row)

    df_ai = pd.DataFrame(ai_rows).set_index("address")
    df = df.join(df_ai[AI_8], how="left")
    return df


def impute_and_clip(X):
    """Median-impute NaNs and clip to [1st, 99th] percentile."""
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)
    return X


def main():
    print("=" * 70)
    print("Paper 0: Class-Balanced Accuracy Analysis")
    print("=" * 70)

    # Load and merge
    df = load_and_merge()
    X = df[ALL_31].values.astype(float)
    y = df["taxonomy_index"].values.astype(int)
    X = impute_and_clip(X)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    majority_class_idx = unique[counts.argmax()]
    majority_class_count = counts.max()
    majority_class_baseline = majority_class_count / total

    print(f"\nN = {total}, K = {len(unique)} classes")
    print(f"Majority class: {TAXONOMY_NAMES[majority_class_idx]} "
          f"(n={majority_class_count}, {majority_class_baseline*100:.1f}%)")
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {TAXONOMY_NAMES.get(int(u), '?'):<25} n={c:5d} ({c/total*100:.1f}%)")

    # GBM 10-fold CV
    gbm = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=SEED,
    )

    min_count = counts.min()
    actual_folds = min(N_FOLDS, min_count)
    print(f"\nUsing {actual_folds}-fold stratified CV")

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)

    all_y_true = []
    all_y_pred = []
    fold_accs = []
    fold_bal_accs = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        clf = clone(gbm)
        clf.fit(X_tr, y[tr_idx])
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y[te_idx], y_pred)
        bal_acc = balanced_accuracy_score(y[te_idx], y_pred)

        fold_accs.append(acc)
        fold_bal_accs.append(bal_acc)
        all_y_true.extend(y[te_idx].tolist())
        all_y_pred.extend(y_pred.tolist())

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)

    # Aggregate metrics
    overall_accuracy = accuracy_score(y_true_arr, y_pred_arr)
    overall_balanced_accuracy = balanced_accuracy_score(y_true_arr, y_pred_arr)

    # Per-class recall (= balanced accuracy components)
    per_class_recall = recall_score(
        y_true_arr, y_pred_arr, labels=sorted(unique), average=None, zero_division=0
    )

    print(f"\n{'─' * 70}")
    print("RESULTS")
    print(f"{'─' * 70}")
    print(f"  Overall accuracy:          {overall_accuracy*100:.1f}%")
    print(f"  Balanced accuracy:         {overall_balanced_accuracy*100:.1f}%")
    print(f"  Majority-class baseline:   {majority_class_baseline*100:.1f}%")
    print(f"  Balanced acc. mean (folds): {np.mean(fold_bal_accs)*100:.1f}% "
          f"+/- {np.std(fold_bal_accs)*100:.1f}%")
    print(f"\n  Per-class recall:")
    for i, cls_idx in enumerate(sorted(unique)):
        name = TAXONOMY_NAMES.get(int(cls_idx), f"C{cls_idx}")
        print(f"    {name:<25} recall = {per_class_recall[i]*100:.1f}%")

    # Save results
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": "Class-balanced accuracy analysis for reviewer response",
        "n_samples": int(total),
        "n_classes": int(len(unique)),
        "n_features": len(ALL_31),
        "n_folds": actual_folds,
        "overall_accuracy": round(float(overall_accuracy), 4),
        "balanced_accuracy": round(float(overall_balanced_accuracy), 4),
        "balanced_accuracy_fold_mean": round(float(np.mean(fold_bal_accs)), 4),
        "balanced_accuracy_fold_std": round(float(np.std(fold_bal_accs)), 4),
        "majority_class": TAXONOMY_NAMES[majority_class_idx],
        "majority_class_baseline": round(float(majority_class_baseline), 4),
        "per_class_recall": {
            TAXONOMY_NAMES.get(int(cls_idx), f"C{cls_idx}"): round(float(per_class_recall[i]), 4)
            for i, cls_idx in enumerate(sorted(unique))
        },
        "fold_accuracies": [round(float(a), 4) for a in fold_accs],
        "fold_balanced_accuracies": [round(float(a), 4) for a in fold_bal_accs],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
