"""
Paper 1: Baseline Comparisons and Ablation Study
==================================================
Compares the full 23-feature classifier against multiple baselines:

  a) Heuristic baseline: tx_count > 1000 AND tx_interval_mean < 60s -> agent
  b) Single-feature baselines: best single-feature AUC
  c) Feature group ablation: each of the 4 groups alone
  d) Full model: all 23 features combined
  e) Reports AUC, Precision@90%Recall, F1 for each configuration

Evaluation protocol:
  - Repeated Stratified 5-Fold CV (10 repeats) for stable estimates
  - Leave-One-Out CV for unbiased point estimates on small N
  - All results saved to baseline_comparison_results.json
"""

import json
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent
FEATURES_PATH = DATA_DIR / "features.parquet"
RESULTS_PATH = OUTPUT_DIR / "baseline_comparison_results.json"


# ==================================================================
# FEATURE GROUP DEFINITIONS
# ==================================================================

TEMPORAL_FEATURES = [
    "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
    "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
    "burst_frequency",
]

GAS_FEATURES = [
    "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
    "gas_limit_precision", "gas_price_cv",
    "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
]

INTERACTION_FEATURES = [
    "unique_contracts_ratio", "top_contract_concentration",
    "method_id_diversity", "contract_to_eoa_ratio",
    "sequential_pattern_score",
]

APPROVAL_FEATURES = [
    "unlimited_approve_ratio", "approve_revoke_ratio",
    "unverified_contract_approve_ratio",
    "multi_protocol_interaction_count", "flash_loan_usage",
]

ALL_FEATURES = TEMPORAL_FEATURES + GAS_FEATURES + INTERACTION_FEATURES + APPROVAL_FEATURES

FEATURE_GROUPS = {
    "temporal_only": TEMPORAL_FEATURES,
    "gas_only": GAS_FEATURES,
    "interaction_only": INTERACTION_FEATURES,
    "approval_only": APPROVAL_FEATURES,
    "all_features": ALL_FEATURES,
}


# ==================================================================
# PRECISION AT RECALL THRESHOLD
# ==================================================================

def precision_at_recall_threshold(y_true, y_prob, recall_threshold=0.90):
    """Compute the precision at a given recall threshold.

    Finds the threshold that achieves at least `recall_threshold` recall
    and returns the corresponding precision. If no threshold achieves
    the required recall, returns 0.0.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns precision/recall sorted by decreasing threshold
    # Find the highest precision where recall >= threshold
    valid = recalls >= recall_threshold
    if valid.any():
        return float(precisions[valid].max())
    return 0.0


# ==================================================================
# HEURISTIC BASELINE
# ==================================================================

def heuristic_baseline(df: pd.DataFrame, y: np.ndarray) -> dict:
    """Heuristic baseline: tx_count-proxy > threshold AND short intervals.

    Since we don't have tx_count as a feature, we use burst_frequency
    (high burst = high tx rate) and tx_interval_mean < 60s as proxy.

    Rule: tx_interval_mean < 60 AND burst_frequency > 0.1 -> AGENT
    """
    tx_interval = df["tx_interval_mean"].values
    burst_freq = df["burst_frequency"].values

    # Heuristic prediction
    y_pred_heuristic = ((tx_interval < 60) & (burst_freq > 0.1)).astype(int)

    # For AUC, use a simple score: lower interval + higher burst = more agent-like
    # Normalize and combine
    if tx_interval.std() > 0:
        interval_score = 1 - (tx_interval - tx_interval.min()) / (tx_interval.max() - tx_interval.min() + 1e-10)
    else:
        interval_score = np.zeros_like(tx_interval)
    if burst_freq.std() > 0:
        burst_score = (burst_freq - burst_freq.min()) / (burst_freq.max() - burst_freq.min() + 1e-10)
    else:
        burst_score = np.zeros_like(burst_freq)

    heuristic_score = 0.5 * interval_score + 0.5 * burst_score

    try:
        auc = roc_auc_score(y, heuristic_score)
    except ValueError:
        auc = 0.5

    prec_at_90 = precision_at_recall_threshold(y, heuristic_score, 0.90)

    return {
        "method": "Heuristic (interval<60s + burst>0.1)",
        "auc": round(float(auc), 4),
        "precision": round(float(precision_score(y, y_pred_heuristic, zero_division=0)), 4),
        "recall": round(float(recall_score(y, y_pred_heuristic, zero_division=0)), 4),
        "f1": round(float(f1_score(y, y_pred_heuristic, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, y_pred_heuristic)), 4),
        "precision_at_90recall": round(prec_at_90, 4),
    }


# ==================================================================
# SINGLE-FEATURE BASELINES
# ==================================================================

def single_feature_baselines(X: np.ndarray, y: np.ndarray,
                              feature_names: list[str]) -> list[dict]:
    """Evaluate each feature as a single predictor (univariate AUC)."""
    results = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if np.std(col) == 0:
            results.append({
                "feature": name,
                "auc": 0.5,
                "auc_adjusted": 0.5,
                "direction": "none",
            })
            continue

        auc = roc_auc_score(y, col)
        # If AUC < 0.5, the feature is negatively correlated with agent
        direction = "positive" if auc >= 0.5 else "negative"
        auc_adjusted = max(auc, 1 - auc)

        # Precision@90%Recall using the raw feature score
        score = col if auc >= 0.5 else -col
        prec_at_90 = precision_at_recall_threshold(y, score, 0.90)

        results.append({
            "feature": name,
            "auc": round(float(auc), 4),
            "auc_adjusted": round(float(auc_adjusted), 4),
            "direction": direction,
            "precision_at_90recall": round(prec_at_90, 4),
        })

    results.sort(key=lambda x: x["auc_adjusted"], reverse=True)
    return results


# ==================================================================
# MODEL-BASED EVALUATION
# ==================================================================

def evaluate_feature_set(
    X_set: np.ndarray,
    y: np.ndarray,
    set_name: str,
    feature_names: list[str],
    n_repeats: int = 10,
    n_splits: int = 5,
) -> dict:
    """Evaluate a feature set using Repeated Stratified K-Fold and LOO CV.

    Uses GradientBoosting as the primary classifier.

    Returns:
        Dictionary with CV metrics (mean +/- std) and LOO metrics.
    """
    model_template = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=3,
        random_state=42,
    )

    # --- Repeated Stratified K-Fold ---
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )

    fold_metrics = defaultdict(list)

    for train_idx, test_idx in rskf.split(X_set, y):
        X_tr, X_te = X_set[train_idx], X_set[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        try:
            fold_metrics["auc"].append(roc_auc_score(y_te, y_prob))
        except ValueError:
            fold_metrics["auc"].append(0.5)
        fold_metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
        fold_metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
        fold_metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
        fold_metrics["accuracy"].append(accuracy_score(y_te, y_pred))

    cv_mean = {k: round(float(np.mean(v)), 4) for k, v in fold_metrics.items()}
    cv_std = {k: round(float(np.std(v)), 4) for k, v in fold_metrics.items()}

    # --- Leave-One-Out CV ---
    loo = LeaveOneOut()
    loo_preds = np.zeros(len(y))
    loo_probs = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X_set):
        X_tr, X_te = X_set[train_idx], X_set[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)

        loo_preds[test_idx[0]] = clf.predict(X_te)[0]
        loo_probs[test_idx[0]] = clf.predict_proba(X_te)[0, 1]

    try:
        loo_auc = roc_auc_score(y, loo_probs)
    except ValueError:
        loo_auc = 0.5

    loo_prec_at_90 = precision_at_recall_threshold(y, loo_probs, 0.90)

    loo_metrics = {
        "auc": round(float(loo_auc), 4),
        "precision": round(float(precision_score(y, loo_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, loo_preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, loo_preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, loo_preds)), 4),
        "precision_at_90recall": round(loo_prec_at_90, 4),
    }

    return {
        "set_name": set_name,
        "n_features": int(X_set.shape[1]),
        "feature_names": feature_names,
        "repeated_cv": {"mean": cv_mean, "std": cv_std, "n_repeats": n_repeats, "n_splits": n_splits},
        "loo_cv": loo_metrics,
    }


# ==================================================================
# MAIN
# ==================================================================

def main():
    """Run all baseline comparisons and save results."""
    print("=" * 70)
    print("Paper 1: Baseline Comparisons and Ablation Study")
    print("=" * 70)

    # --- Load data ---
    df = pd.read_parquet(FEATURES_PATH)
    print(f"\nLoaded {len(df)} samples "
          f"({int(df['label'].sum())} agents, "
          f"{int((df['label'] == 0).sum())} humans)")

    feature_cols = [c for c in df.columns if c not in ("label", "name")]
    X_all = df[feature_cols].values.astype(float)
    y = df["label"].values

    # Handle NaN
    nan_mask = np.isnan(X_all)
    if nan_mask.any():
        col_medians = np.nanmedian(X_all, axis=0)
        for j in range(X_all.shape[1]):
            X_all[nan_mask[:, j], j] = col_medians[j]
        print(f"Filled {nan_mask.sum()} NaN values with column medians")

    results = {
        "dataset": {
            "n_samples": int(len(df)),
            "n_agents": int(df["label"].sum()),
            "n_humans": int((df["label"] == 0).sum()),
            "n_features": len(feature_cols),
        },
    }

    # ================================================================
    # A) Heuristic Baseline
    # ================================================================
    print(f"\n{'='*70}")
    print("A) Heuristic Baseline")
    print("=" * 70)

    heuristic = heuristic_baseline(df[feature_cols], y)
    results["heuristic_baseline"] = heuristic

    print(f"  Rule: tx_interval_mean < 60s AND burst_frequency > 0.1")
    print(f"  AUC:                {heuristic['auc']:.4f}")
    print(f"  Precision:          {heuristic['precision']:.4f}")
    print(f"  Recall:             {heuristic['recall']:.4f}")
    print(f"  F1:                 {heuristic['f1']:.4f}")
    print(f"  Precision@90%Rec:   {heuristic['precision_at_90recall']:.4f}")

    # ================================================================
    # B) Single-Feature Baselines
    # ================================================================
    print(f"\n{'='*70}")
    print("B) Single-Feature Baselines (univariate AUC)")
    print("=" * 70)

    single_feat_results = single_feature_baselines(X_all, y, feature_cols)
    results["single_feature_baselines"] = single_feat_results

    print(f"\n  {'Feature':<40} {'AUC':>8} {'Dir':>10} {'P@90R':>8}")
    print("  " + "-" * 70)
    for entry in single_feat_results:
        print(f"  {entry['feature']:<40} "
              f"{entry['auc_adjusted']:8.4f} "
              f"{entry['direction']:>10} "
              f"{entry.get('precision_at_90recall', 0):8.4f}")

    best_feat = single_feat_results[0]
    print(f"\n  Best single feature: {best_feat['feature']} "
          f"(AUC={best_feat['auc_adjusted']:.4f})")

    # ================================================================
    # C) Feature Group Ablation
    # ================================================================
    print(f"\n{'='*70}")
    print("C) Feature Group Ablation")
    print("=" * 70)

    ablation_results = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        # Get column indices for this feature group
        col_indices = [
            feature_cols.index(f) for f in group_features
            if f in feature_cols
        ]
        if not col_indices:
            print(f"\n  {group_name}: No matching features found, skipping")
            continue

        actual_features = [feature_cols[i] for i in col_indices]
        X_group = X_all[:, col_indices]

        print(f"\n  --- {group_name} ({len(col_indices)} features) ---")

        group_result = evaluate_feature_set(
            X_group, y, group_name, actual_features
        )
        ablation_results[group_name] = group_result

        cv = group_result["repeated_cv"]["mean"]
        loo = group_result["loo_cv"]
        print(f"    Repeated 5-Fold CV (10 repeats):")
        print(f"      AUC:       {cv['auc']:.4f} +/- {group_result['repeated_cv']['std']['auc']:.4f}")
        print(f"      F1:        {cv['f1']:.4f} +/- {group_result['repeated_cv']['std']['f1']:.4f}")
        print(f"      Precision: {cv['precision']:.4f}")
        print(f"      Recall:    {cv['recall']:.4f}")
        print(f"    Leave-One-Out CV:")
        print(f"      AUC:              {loo['auc']:.4f}")
        print(f"      F1:               {loo['f1']:.4f}")
        print(f"      Precision@90%Rec: {loo['precision_at_90recall']:.4f}")

    results["feature_group_ablation"] = ablation_results

    # ================================================================
    # D) Summary Comparison Table
    # ================================================================
    print(f"\n{'='*70}")
    print("D) Summary Comparison")
    print("=" * 70)

    print(f"\n  {'Method':<35} {'AUC':>8} {'F1':>8} {'P@90R':>8}")
    print("  " + "-" * 62)

    # Heuristic
    print(f"  {'Heuristic baseline':<35} "
          f"{heuristic['auc']:8.4f} "
          f"{heuristic['f1']:8.4f} "
          f"{heuristic['precision_at_90recall']:8.4f}")

    # Best single feature
    print(f"  {'Best single feature':35s} "
          f"{best_feat['auc_adjusted']:8.4f} "
          f"{'--':>8} "
          f"{best_feat.get('precision_at_90recall', 0):8.4f}")

    # Feature groups
    for group_name in ["temporal_only", "gas_only", "interaction_only",
                       "approval_only", "all_features"]:
        if group_name in ablation_results:
            loo = ablation_results[group_name]["loo_cv"]
            label = group_name.replace("_", " ").title()
            print(f"  {label:<35} "
                  f"{loo['auc']:8.4f} "
                  f"{loo['f1']:8.4f} "
                  f"{loo['precision_at_90recall']:8.4f}")

    # ================================================================
    # E) Additional: Multi-model comparison on full features
    # ================================================================
    print(f"\n{'='*70}")
    print("E) Multi-Model Comparison (all features, LOO CV)")
    print("=" * 70)

    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=3, random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=3,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0, max_iter=1000, random_state=42,
        ),
    }

    multi_model_results = {}
    for model_name, model_template in models.items():
        loo = LeaveOneOut()
        loo_preds = np.zeros(len(y))
        loo_probs = np.zeros(len(y))

        for train_idx, test_idx in loo.split(X_all):
            X_tr, X_te = X_all[train_idx], X_all[test_idx]
            y_tr = y[train_idx]

            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

            clf = clone(model_template)
            clf.fit(X_tr, y_tr)

            loo_preds[test_idx[0]] = clf.predict(X_te)[0]
            if hasattr(clf, "predict_proba"):
                loo_probs[test_idx[0]] = clf.predict_proba(X_te)[0, 1]
            else:
                loo_probs[test_idx[0]] = clf.decision_function(X_te)[0]

        try:
            auc = roc_auc_score(y, loo_probs)
        except ValueError:
            auc = 0.5

        prec_at_90 = precision_at_recall_threshold(y, loo_probs, 0.90)

        metrics = {
            "auc": round(float(auc), 4),
            "precision": round(float(precision_score(y, loo_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y, loo_preds, zero_division=0)), 4),
            "f1": round(float(f1_score(y, loo_preds, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, loo_preds)), 4),
            "precision_at_90recall": round(prec_at_90, 4),
        }
        multi_model_results[model_name] = metrics

        print(f"\n  {model_name}:")
        print(f"    AUC:              {metrics['auc']:.4f}")
        print(f"    F1:               {metrics['f1']:.4f}")
        print(f"    Precision:        {metrics['precision']:.4f}")
        print(f"    Recall:           {metrics['recall']:.4f}")
        print(f"    Precision@90%Rec: {metrics['precision_at_90recall']:.4f}")

    results["multi_model_comparison"] = multi_model_results

    # ================================================================
    # Save results
    # ================================================================
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to {RESULTS_PATH}")

    # ================================================================
    # F) Feature Importance from GBM (for paper table)
    # ================================================================
    print(f"\n{'='*70}")
    print("F) GBM Feature Importance (trained on full data)")
    print("=" * 70)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_all)
    gbm = GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1,
        min_samples_leaf=3, random_state=42,
    )
    gbm.fit(X_scaled, y)

    importances = gbm.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    for rank, idx in enumerate(sorted_idx[:15]):
        name = feature_cols[idx]
        imp = float(importances[idx])
        bar = "=" * int(imp * 100)
        print(f"  {rank+1:2d}. {name:42s} {imp:.4f} {bar}")

    results["gbm_feature_importance"] = {
        feature_cols[idx]: round(float(importances[idx]), 4)
        for idx in sorted_idx
    }

    # Re-save with importance data
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*70}")
    print("Baseline comparison complete.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
