#!/usr/bin/env python
"""
Paper 0: 31-Feature Multi-Class Taxonomy Classifier
=====================================================
Merges Paper 0's 23 behavioral features with Paper 3's 8 AI-detection
features to create a 31-feature set, then evaluates GBM and RF classifiers
on the 8-class taxonomy with 10-fold stratified CV + bootstrap 95% CIs.

Features:
  - 23 original: temporal, gas, interaction, approval, DeFi features
  -  8 new (P3): gas_price_precision, hour_entropy, behavioral_consistency,
                  action_sequence_perplexity, error_recovery_pattern,
                  response_latency_variance, gas_nonce_gap_regularity,
                  eip1559_tip_precision

Evaluation:
  - 10-fold stratified CV (GBM + RF)
  - Per-class precision / recall / F1
  - Bootstrap 95% CIs on per-class F1 (1000 iterations)
  - Confusion matrix figure

Outputs:
  - multiclass_31features_results.json
  - confusion_matrix_31features.png
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
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
    / "multiclass_31features_results.json"
)
FIG_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "confusion_matrix_31features.png"
)

# ── Feature columns ──────────────────────────────────────────────────
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
N_BOOT = 1000
SEED = 42


def load_and_merge():
    """Load Paper 0's 23-feature dataset and merge with Paper 3's 8 AI features."""
    # Load base dataset (2,744 agents)
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["label"] == 1].copy()
    print(f"Paper 0 agents: {len(df)}")

    # Load AI features
    with open(AI_FEATURES_JSON) as f:
        ai_raw = json.load(f)

    ai_rows = []
    for addr, feats in ai_raw["per_address"].items():
        row = {"address": addr}
        for col in AI_8:
            row[col] = feats.get(col, np.nan)
        ai_rows.append(row)

    df_ai = pd.DataFrame(ai_rows).set_index("address")
    print(f"Paper 3 AI features: {len(df_ai)} addresses, {len(AI_8)} features")

    # Join: left join keeps all 2,744 Paper 0 agents
    overlap_before = len(set(df.index) & set(df_ai.index))
    print(f"Address overlap: {overlap_before}")

    df = df.join(df_ai[AI_8], how="left")
    n_missing = df[AI_8].isna().any(axis=1).sum()
    print(f"Agents missing AI features (will impute): {n_missing}")

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


def bootstrap_per_class_ci(y_true, y_pred, classes, n_boot=N_BOOT, seed=SEED):
    """Bootstrap 95% CIs on per-class F1, precision, recall."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    n_cls = len(classes)

    f1_mat = np.zeros((n_boot, n_cls))
    prec_mat = np.zeros((n_boot, n_cls))
    rec_mat = np.zeros((n_boot, n_cls))

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        yp = y_pred[idx]
        f1_mat[b] = f1_score(yt, yp, labels=classes, average=None, zero_division=0)
        prec_mat[b] = precision_score(yt, yp, labels=classes, average=None, zero_division=0)
        rec_mat[b] = recall_score(yt, yp, labels=classes, average=None, zero_division=0)

    results = {}
    for i, cls in enumerate(classes):
        name = TAXONOMY_NAMES.get(int(cls), f"Class{cls}")
        results[name] = {
            "taxonomy_index": int(cls),
            "f1_point": round(float(f1_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]), 4),
            "f1_boot_mean": round(float(np.mean(f1_mat[:, i])), 4),
            "f1_ci_lo": round(float(np.percentile(f1_mat[:, i], 2.5)), 4),
            "f1_ci_hi": round(float(np.percentile(f1_mat[:, i], 97.5)), 4),
            "precision_point": round(float(precision_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]), 4),
            "precision_ci_lo": round(float(np.percentile(prec_mat[:, i], 2.5)), 4),
            "precision_ci_hi": round(float(np.percentile(prec_mat[:, i], 97.5)), 4),
            "recall_point": round(float(recall_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0]), 4),
            "recall_ci_lo": round(float(np.percentile(rec_mat[:, i], 2.5)), 4),
            "recall_ci_hi": round(float(np.percentile(rec_mat[:, i], 97.5)), 4),
        }
    return results


def plot_confusion_matrix(y_true, y_pred, classes, model_name):
    """Save confusion matrix heatmap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    # Normalize rows (recall per class)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    labels = [TAXONOMY_NAMES.get(int(c), f"C{c}") for c in classes]
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"31-Feature {model_name} Confusion Matrix (row-normalized)")

    # Annotate cells with counts
    for i in range(len(classes)):
        for j in range(len(classes)):
            text = f"{cm[i, j]}"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(str(FIG_PATH), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved confusion matrix figure: {FIG_PATH}")


def main():
    print("=" * 80)
    print("Paper 0: 31-Feature Multi-Class Taxonomy Classifier")
    print("  23 behavioral (P0) + 8 AI-detection (P3) features")
    print("=" * 80)

    # Load and merge
    df = load_and_merge()

    X = df[ALL_31].values.astype(float)
    y = df["taxonomy_index"].values.astype(int)

    # Impute NaN and clip
    X = impute_and_clip(X)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u} ({TAXONOMY_NAMES.get(int(u), '?'):<25}) n={c}")
    total = len(y)

    # Keep all classes (don't drop) -- the 23-feature baseline dropped <20,
    # but with 10-fold CV and n>=25 for the smallest class, we can keep them.
    # Actually let's check minimum
    min_count = counts.min()
    min_class = unique[counts.argmin()]
    print(f"\nSmallest class: {TAXONOMY_NAMES.get(int(min_class))} (n={min_count})")
    if min_count < N_FOLDS:
        print(f"  WARNING: class {min_class} has fewer samples than folds ({N_FOLDS}).")
        print(f"  Reducing folds to {min_count} for this run.")
        actual_folds = min(N_FOLDS, min_count)
    else:
        actual_folds = N_FOLDS

    # Models
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1,
        ),
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "description": "31-feature classifier: 23 P0 behavioral + 8 P3 AI-detection features",
        "n_samples": int(total),
        "n_features": len(ALL_31),
        "feature_names_original_23": ORIGINAL_23,
        "feature_names_ai_8": AI_8,
        "feature_names_all_31": ALL_31,
        "n_folds": actual_folds,
        "n_bootstrap": N_BOOT,
        "classes_present": sorted([int(c) for c in unique]),
        "class_counts": {
            TAXONOMY_NAMES.get(int(u), f"C{u}"): int(c)
            for u, c in zip(unique, counts)
        },
        "models": {},
    }

    classes_sorted = sorted(set(y.tolist()))
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)

    best_model_name = None
    best_f1_macro = -1
    best_y_true = None
    best_y_pred = None

    for model_name, model_template in models.items():
        print(f"\n{'─' * 60}")
        print(f"Training {model_name} ({actual_folds}-fold stratified CV)...")
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

            acc = accuracy_score(y[te_idx], y_pred)
            f1m = f1_score(y[te_idx], y_pred, average="macro", zero_division=0)
            f1w = f1_score(y[te_idx], y_pred, average="weighted", zero_division=0)

            fold_accs.append(acc)
            fold_f1_macro.append(f1m)
            fold_f1_weighted.append(f1w)
            all_y_true.extend(y[te_idx].tolist())
            all_y_pred.extend(y_pred.tolist())

            if fold < 3 or fold == actual_folds - 1:
                print(f"  Fold {fold+1:2d}: Acc={acc:.4f}  F1-macro={f1m:.4f}")

        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)

        # Aggregate metrics
        acc_mean = float(np.mean(fold_accs))
        acc_std = float(np.std(fold_accs))
        f1m_mean = float(np.mean(fold_f1_macro))
        f1m_std = float(np.std(fold_f1_macro))
        f1w_mean = float(np.mean(fold_f1_weighted))

        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=classes_sorted).tolist()
        report = classification_report(
            y_true_arr, y_pred_arr,
            labels=classes_sorted,
            target_names=[TAXONOMY_NAMES.get(int(c), f"C{c}") for c in classes_sorted],
            output_dict=True,
            zero_division=0,
        )

        print(f"\n  {model_name} Summary:")
        print(f"    Accuracy:    {acc_mean:.4f} +/- {acc_std:.4f}")
        print(f"    F1-macro:    {f1m_mean:.4f} +/- {f1m_std:.4f}")
        print(f"    F1-weighted: {f1w_mean:.4f}")
        print(f"    Per-class F1:")
        for cls_name, stats in report.items():
            if isinstance(stats, dict) and "f1-score" in stats:
                print(f"      {cls_name:<25} F1={stats['f1-score']:.4f} "
                      f"P={stats['precision']:.4f} R={stats['recall']:.4f} "
                      f"(n={stats['support']})")

        # Bootstrap CIs
        print(f"\n  Computing bootstrap 95% CIs ({N_BOOT} iterations)...")
        boot_ci = bootstrap_per_class_ci(y_true_arr, y_pred_arr, classes_sorted)
        for cls_name, ci in boot_ci.items():
            print(f"    {cls_name:<25} F1={ci['f1_point']:.4f} "
                  f"[{ci['f1_ci_lo']:.4f}, {ci['f1_ci_hi']:.4f}]")

        results["models"][model_name] = {
            "accuracy_mean": round(acc_mean, 4),
            "accuracy_std": round(acc_std, 4),
            "f1_macro_mean": round(f1m_mean, 4),
            "f1_macro_std": round(f1m_std, 4),
            "f1_weighted_mean": round(f1w_mean, 4),
            "confusion_matrix": cm,
            "confusion_labels": [int(c) for c in classes_sorted],
            "per_class_report": report,
            "bootstrap_95ci": boot_ci,
        }

        if f1m_mean > best_f1_macro:
            best_f1_macro = f1m_mean
            best_model_name = model_name
            best_y_true = y_true_arr
            best_y_pred = y_pred_arr

    # GBM feature importance (full data)
    print(f"\n{'─' * 60}")
    print("Computing GBM feature importance on full data...")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gbm_full = clone(models["GradientBoosting"])
    gbm_full.fit(Xs, y)
    sorted_idx = np.argsort(gbm_full.feature_importances_)[::-1]
    fi = {
        ALL_31[i]: round(float(gbm_full.feature_importances_[i]), 4)
        for i in sorted_idx
    }
    results["gbm_feature_importance"] = fi
    print("  Top 10 features:")
    for rank, (feat, imp) in enumerate(fi.items()):
        if rank >= 10:
            break
        marker = " [P3]" if feat in AI_8 else ""
        print(f"    {rank+1:2d}. {feat:<35} {imp:.4f}{marker}")

    # Confusion matrix figure (best model)
    print(f"\nGenerating confusion matrix figure (best model: {best_model_name})...")
    plot_confusion_matrix(best_y_true, best_y_pred, classes_sorted, best_model_name)

    # ── Comparison summary ─────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("COMPARISON: 23-feature baseline vs 31-feature (this run)")
    print("=" * 80)
    baseline = {
        "accuracy": 0.922,
        "f1_macro": 0.58,
        "LLMPoweredAgent_f1": 0.47,
        "AutonomousDAOAgent_f1": 0.14,
        "CrossChainBridgeAgent_f1": 0.31,
        "RLTradingAgent_f1": 0.07,
    }
    gbm_results = results["models"].get("GradientBoosting", {})
    gbm_report = gbm_results.get("per_class_report", {})

    new_metrics = {
        "accuracy": gbm_results.get("accuracy_mean", 0),
        "f1_macro": gbm_results.get("f1_macro_mean", 0),
        "LLMPoweredAgent_f1": gbm_report.get("LLMPoweredAgent", {}).get("f1-score", 0),
        "AutonomousDAOAgent_f1": gbm_report.get("AutonomousDAOAgent", {}).get("f1-score", 0),
        "CrossChainBridgeAgent_f1": gbm_report.get("CrossChainBridgeAgent", {}).get("f1-score", 0),
        "RLTradingAgent_f1": gbm_report.get("RLTradingAgent", {}).get("f1-score", 0),
    }

    results["comparison_vs_23features"] = {
        "baseline_23feat": baseline,
        "new_31feat": {k: round(v, 4) for k, v in new_metrics.items()},
        "delta": {k: round(new_metrics[k] - baseline[k], 4) for k in baseline},
    }

    for metric in baseline:
        b = baseline[metric]
        n = new_metrics[metric]
        delta = n - b
        arrow = "+" if delta > 0 else ""
        print(f"  {metric:<30} {b:.4f} -> {n:.4f} ({arrow}{delta:.4f})")

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Figure saved:  {FIG_PATH}")


if __name__ == "__main__":
    main()
