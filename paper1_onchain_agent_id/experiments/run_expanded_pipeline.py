"""
Paper 1: Expanded-Dataset Experiment Pipeline
================================================
Runs the full classifier + ablation + baseline comparison on the
expanded dataset (features_expanded.parquet: 3316 addresses).

Key differences from run_full_pipeline.py:
  - Uses Stratified 5-Fold repeated CV (NOT LOO-CV which is O(N) fits)
  - Loads labels directly from the parquet 'label' column
  - Reports per-class metrics (dataset is imbalanced: 2590:726)
  - Saves to experiments/expanded/pipeline_results.json
"""

import json
import logging
import sys
import time
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
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "expanded"
FEATURES_PARQUET = DATA_DIR / "features_expanded.parquet"

sys.path.insert(0, str(PROJECT_ROOT.parent))


# ============================================================
# FEATURE GROUPS (identical to run_full_pipeline.py)
# ============================================================

FEATURE_GROUPS = {
    "temporal": [
        "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
        "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
        "burst_frequency",
    ],
    "gas": [
        "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
        "gas_limit_precision", "gas_price_cv",
        "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
    ],
    "interaction": [
        "unique_contracts_ratio", "top_contract_concentration",
        "method_id_diversity", "contract_to_eoa_ratio",
        "sequential_pattern_score",
    ],
    "approval_security": [
        "unlimited_approve_ratio", "approve_revoke_ratio",
        "unverified_contract_approve_ratio",
        "multi_protocol_interaction_count", "flash_loan_usage",
    ],
}

ALL_FEATURES = []
for group_features in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group_features)


# ============================================================
# MODELS
# ============================================================

def get_models() -> dict:
    return {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
    }


# ============================================================
# DATA LOADING
# ============================================================

def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    logger.info("Loading features from %s ...", FEATURES_PARQUET)
    df = pd.read_parquet(FEATURES_PARQUET)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    missing = [c for c in ALL_FEATURES if c not in df.columns]
    if missing:
        logger.warning("Missing features: %s", missing)

    X = df[feature_cols].copy()
    y = df["label"].values.astype(int)

    # Impute NaN with column medians
    nan_mask = np.isnan(X.values)
    if nan_mask.any():
        col_medians = np.nanmedian(X.values, axis=0)
        # Fill any column-median NaNs (all-NaN columns) with 0
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X.iloc[nan_mask[:, j], j] = col_medians[j]
        logger.info("Imputed %d NaN values", int(nan_mask.sum()))

    # Clip extreme values (the expanded dataset has outliers from mined addrs)
    for col in X.columns:
        lo, hi = np.nanpercentile(X[col], [0.5, 99.5])
        X[col] = X[col].clip(lo, hi)

    logger.info("Feature matrix: %d samples x %d features", *X.shape)
    logger.info("  Agents: %d, Humans: %d (ratio %.2f:1)",
                int(y.sum()), int((y == 0).sum()),
                float(y.sum() / max((y == 0).sum(), 1)))

    return X, y, feature_cols


# ============================================================
# PER-FEATURE AUC
# ============================================================

def per_feature_auc(X_np, y, feature_names) -> dict:
    aucs = {}
    for i, name in enumerate(feature_names):
        col = X_np[:, i]
        if np.std(col) == 0:
            aucs[name] = 0.5
            continue
        auc = roc_auc_score(y, col)
        aucs[name] = max(auc, 1.0 - auc)
    return dict(sorted(aucs.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# REPEATED STRATIFIED K-FOLD CV
# ============================================================

def run_repeated_cv(
    X_np, y, model_template, n_splits: int = 5, n_repeats: int = 5,
) -> dict:
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42,
    )
    fold_metrics = {
        "auc": [], "precision": [], "recall": [], "f1": [], "accuracy": [],
        "precision_human": [], "recall_human": [], "f1_human": [],
    }
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for train_idx, test_idx in rskf.split(X_np, y):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_te)[:, 1]
        else:
            y_prob = clf.decision_function(X_te)

        fold_metrics["auc"].append(roc_auc_score(y_te, y_prob))
        fold_metrics["precision"].append(
            precision_score(y_te, y_pred, pos_label=1, zero_division=0))
        fold_metrics["recall"].append(
            recall_score(y_te, y_pred, pos_label=1, zero_division=0))
        fold_metrics["f1"].append(
            f1_score(y_te, y_pred, pos_label=1, zero_division=0))
        fold_metrics["accuracy"].append(accuracy_score(y_te, y_pred))
        fold_metrics["precision_human"].append(
            precision_score(y_te, y_pred, pos_label=0, zero_division=0))
        fold_metrics["recall_human"].append(
            recall_score(y_te, y_pred, pos_label=0, zero_division=0))
        fold_metrics["f1_human"].append(
            f1_score(y_te, y_pred, pos_label=0, zero_division=0))

        all_y_true.extend(y_te.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

    mean_m = {k: round(float(np.mean(v)), 4) for k, v in fold_metrics.items()}
    std_m = {k: round(float(np.std(v)), 4) for k, v in fold_metrics.items()}

    return {
        "mean": mean_m,
        "std": std_m,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "y_true": all_y_true,
        "y_pred": all_y_pred,
        "y_prob": all_y_prob,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> dict:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 65)
    logger.info("Paper 1: Expanded Dataset Pipeline (N=3316)")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 65)

    X, y, feature_names = load_data()
    X_np = X.values.astype(float)

    results: dict = {
        "run_timestamp": timestamp,
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "source": "features_expanded.parquet",
        },
        "models": {},
    }

    # --------------------------------------------------
    # Per-feature AUC
    # --------------------------------------------------
    logger.info("Computing per-feature AUC ...")
    feat_aucs = per_feature_auc(X_np, y, feature_names)
    results["per_feature_auc"] = {k: round(v, 4) for k, v in feat_aucs.items()}
    top10 = list(feat_aucs.keys())[:10]
    results["top10_features"] = top10
    logger.info("Top 10 features by univariate AUC:")
    for name in top10:
        logger.info("  %-40s AUC=%.4f", name, feat_aucs[name])

    # --------------------------------------------------
    # Train/evaluate all models
    # --------------------------------------------------
    models = get_models()
    for model_name, model_template in models.items():
        logger.info("Training %s ...", model_name)
        rcv = run_repeated_cv(X_np, y, model_template)

        results["models"][model_name] = {
            "repeated_cv": {
                "mean": rcv["mean"],
                "std": rcv["std"],
                "n_splits": rcv["n_splits"],
                "n_repeats": rcv["n_repeats"],
            },
        }
        logger.info(
            "  AUC=%.4f±%.4f  F1(agent)=%.4f  Prec=%.4f  Recall=%.4f  Acc=%.4f",
            rcv["mean"]["auc"], rcv["std"]["auc"],
            rcv["mean"]["f1"], rcv["mean"]["precision"],
            rcv["mean"]["recall"], rcv["mean"]["accuracy"],
        )

    # --------------------------------------------------
    # GBM feature importance + confusion matrix on full data
    # --------------------------------------------------
    logger.info("Computing GBM feature importance + confusion matrix ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    gbm = clone(models["GradientBoosting"])
    gbm.fit(X_scaled, y)
    importances = gbm.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    gbm_importance = {
        feature_names[i]: round(float(importances[i]), 4)
        for i in sorted_idx
    }
    results["gbm_feature_importance"] = gbm_importance

    # Held-out confusion matrix: use the last CV's predictions pooled
    gbm_rcv = run_repeated_cv(X_np, y, models["GradientBoosting"],
                               n_splits=5, n_repeats=1)
    y_true = np.array(gbm_rcv["y_true"])
    y_pred = np.array(gbm_rcv["y_pred"])
    cm = confusion_matrix(y_true, y_pred).tolist()
    results["confusion_matrix_gbm_cv"] = cm
    results["classification_report_gbm_cv"] = classification_report(
        y_true, y_pred, target_names=["Human", "Agent"], output_dict=True,
    )

    # --------------------------------------------------
    # Ablation: each feature group alone + leave-one-group-out
    # --------------------------------------------------
    logger.info("Running ablation study ...")
    ablation = {}
    gbm_template = models["GradientBoosting"]

    full_rcv = run_repeated_cv(X_np, y, gbm_template)
    ablation["all_features"] = {
        "n_features": len(feature_names),
        "features": feature_names,
        "mean_auc": full_rcv["mean"]["auc"],
        "std_auc": full_rcv["std"]["auc"],
        "mean_f1": full_rcv["mean"]["f1"],
        "mean_accuracy": full_rcv["mean"]["accuracy"],
    }
    logger.info("  all_features (%d): AUC=%.4f", len(feature_names),
                full_rcv["mean"]["auc"])

    for group_name, group_features in FEATURE_GROUPS.items():
        available = [f for f in group_features if f in feature_names]
        if not available:
            continue
        idx = [feature_names.index(f) for f in available]
        X_group = X_np[:, idx]
        rcv = run_repeated_cv(X_group, y, gbm_template)
        ablation[group_name] = {
            "n_features": len(available),
            "features": available,
            "mean_auc": rcv["mean"]["auc"],
            "std_auc": rcv["std"]["auc"],
            "mean_f1": rcv["mean"]["f1"],
            "mean_accuracy": rcv["mean"]["accuracy"],
        }
        logger.info("  %-18s (%d): AUC=%.4f", group_name, len(available),
                    rcv["mean"]["auc"])

    for group_name, group_features in FEATURE_GROUPS.items():
        remaining = [f for f in feature_names if f not in group_features]
        if not remaining:
            continue
        idx = [feature_names.index(f) for f in remaining]
        X_remaining = X_np[:, idx]
        rcv = run_repeated_cv(X_remaining, y, gbm_template)
        key = f"without_{group_name}"
        ablation[key] = {
            "n_features": len(remaining),
            "mean_auc": rcv["mean"]["auc"],
            "std_auc": rcv["std"]["auc"],
            "mean_f1": rcv["mean"]["f1"],
            "mean_accuracy": rcv["mean"]["accuracy"],
        }
        logger.info("  without_%-10s (%d): AUC=%.4f", group_name,
                    len(remaining), rcv["mean"]["auc"])

    results["ablation_study"] = ablation

    # --------------------------------------------------
    # Baselines
    # --------------------------------------------------
    logger.info("Running baseline comparison ...")
    baselines = {}

    # Heuristic: unique_contracts_ratio > 0.5 -> agent (high variety)
    # (Only if that feature dominates; otherwise burst_frequency)
    best_feature = list(feat_aucs.keys())[0]
    logger.info("  Best univariate feature: %s (AUC=%.4f)",
                best_feature, feat_aucs[best_feature])

    best_idx = feature_names.index(best_feature)
    lr_template = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    single_rcv = run_repeated_cv(
        X_np[:, [best_idx]], y, lr_template,
    )
    baselines["single_best_feature"] = {
        "feature": best_feature,
        "univariate_auc": round(feat_aucs[best_feature], 4),
        "mean_auc": single_rcv["mean"]["auc"],
        "mean_f1": single_rcv["mean"]["f1"],
        "mean_accuracy": single_rcv["mean"]["accuracy"],
    }

    # Full GBM
    baselines["full_model_gbm"] = {
        "n_features": len(feature_names),
        "mean_auc": full_rcv["mean"]["auc"],
        "mean_f1": full_rcv["mean"]["f1"],
        "mean_accuracy": full_rcv["mean"]["accuracy"],
    }

    # Majority-class baseline
    majority = int(y.sum() > (len(y) - y.sum()))
    majority_pred = np.full_like(y, majority)
    baselines["majority_class"] = {
        "class": "Agent" if majority == 1 else "Human",
        "accuracy": round(float(accuracy_score(y, majority_pred)), 4),
        "f1_agent": round(
            float(f1_score(y, majority_pred, pos_label=1, zero_division=0)),
            4,
        ),
    }
    results["baseline_comparison"] = baselines

    # --------------------------------------------------
    # ROC data for figures
    # --------------------------------------------------
    roc_data = {}
    for model_name, model_template in models.items():
        rcv = run_repeated_cv(X_np, y, model_template,
                              n_splits=5, n_repeats=1)
        fpr, tpr, _ = roc_curve(
            np.array(rcv["y_true"]), np.array(rcv["y_prob"]),
        )
        roc_data[model_name] = {
            "fpr": [round(float(x), 6) for x in fpr.tolist()],
            "tpr": [round(float(x), 6) for x in tpr.tolist()],
            "auc": round(float(roc_auc_score(
                np.array(rcv["y_true"]), np.array(rcv["y_prob"]),
            )), 4),
        }
    results["roc_curve_data"] = roc_data

    # --------------------------------------------------
    # Save results
    # --------------------------------------------------
    results["elapsed_seconds"] = round(time.time() - t0, 2)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_serializable = json.loads(json.dumps(results, default=_convert))

    out_path = OUTPUT_DIR / "pipeline_results.json"
    with open(out_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    logger.info("Saved results to %s", out_path)

    logger.info("=" * 65)
    logger.info("Pipeline complete in %.1f seconds", results["elapsed_seconds"])
    logger.info("  Best model: GBM AUC=%.4f", results["models"][
        "GradientBoosting"]["repeated_cv"]["mean"]["auc"])
    logger.info("=" * 65)

    return results


if __name__ == "__main__":
    main()
