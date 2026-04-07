"""
Paper 1: Train agent/human classifier on 44 Ethereum addresses.

Uses Gradient Boosting (primary), Random Forest, and Logistic Regression
with repeated stratified K-fold and leave-one-out CV for stable estimates
on this small dataset (N=44, 22 agents, 22 humans).
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = Path(__file__).parent
FEATURES_PATH = DATA_DIR / "features.parquet"
RESULTS_PATH = OUTPUT_DIR / "classifier_results.json"


def per_feature_auc(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Compute univariate AUC for each feature."""
    aucs = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if np.std(col) == 0:
            aucs[name] = 0.5
            continue
        auc = roc_auc_score(y, col)
        # Flip if AUC < 0.5 (feature negatively associated)
        aucs[name] = max(auc, 1 - auc)
    return dict(sorted(aucs.items(), key=lambda x: x[1], reverse=True))


def train_and_evaluate():
    # ── Load data ──────────────────────────────────────────────────
    df = pd.read_parquet(FEATURES_PATH)
    print(f"Loaded {len(df)} samples ({df['label'].sum()} agents, "
          f"{(df['label'] == 0).sum()} humans)")

    feature_cols = [c for c in df.columns if c not in ("label", "name")]
    X = df[feature_cols].values.astype(float)
    y = df["label"].values
    feature_names = feature_cols

    # Handle NaN (though current data has none)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]
        print(f"Filled {nan_mask.sum()} NaN values with column medians")

    print(f"Features: {len(feature_names)}")
    print(f"  {feature_names}")
    print()

    # ── Per-feature univariate AUC ─────────────────────────────────
    feat_aucs = per_feature_auc(X, y, feature_names)
    print("=" * 65)
    print("Per-Feature Univariate AUC (descending)")
    print("=" * 65)
    for name, auc in feat_aucs.items():
        bar = "█" * int(auc * 40)
        print(f"  {name:42s}  AUC={auc:.4f}  {bar}")
    print()

    top10_features = list(feat_aucs.keys())[:10]
    top10_idx = [feature_names.index(f) for f in top10_features]

    # ── Define models ──────────────────────────────────────────────
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=3,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        ),
    }

    results = {
        "dataset": {
            "n_samples": int(len(df)),
            "n_agents": int(df["label"].sum()),
            "n_humans": int((df["label"] == 0).sum()),
            "n_features": len(feature_names),
            "feature_names": feature_names,
        },
        "per_feature_auc": {k: round(v, 4) for k, v in feat_aucs.items()},
        "top10_features": top10_features,
        "models": {},
    }

    # ── Evaluate each model ────────────────────────────────────────
    feature_sets = {
        "all_features": (X, feature_names),
        "top10_features": (X[:, top10_idx], top10_features),
    }

    for model_name, model_template in models.items():
        print("=" * 65)
        print(f"Model: {model_name}")
        print("=" * 65)
        results["models"][model_name] = {}

        for fset_name, (X_set, fset_names) in feature_sets.items():
            print(f"\n  Feature set: {fset_name} ({X_set.shape[1]} features)")

            # ── 1) Repeated Stratified 5-Fold CV (10 repeats) ──────
            rskf = RepeatedStratifiedKFold(
                n_splits=5, n_repeats=10, random_state=42
            )
            fold_metrics = {
                "auc": [], "precision": [], "recall": [], "f1": [], "accuracy": []
            }

            for fold_i, (train_idx, test_idx) in enumerate(rskf.split(X_set, y)):
                X_tr, X_te = X_set[train_idx], X_set[test_idx]
                y_tr, y_te = y[train_idx], y[test_idx]

                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_tr)
                X_te = scaler.transform(X_te)

                from sklearn.base import clone
                clf = clone(model_template)
                clf.fit(X_tr, y_tr)

                y_pred = clf.predict(X_te)
                y_prob = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(X_te)

                fold_metrics["auc"].append(roc_auc_score(y_te, y_prob))
                fold_metrics["precision"].append(precision_score(y_te, y_pred, zero_division=0))
                fold_metrics["recall"].append(recall_score(y_te, y_pred, zero_division=0))
                fold_metrics["f1"].append(f1_score(y_te, y_pred, zero_division=0))
                fold_metrics["accuracy"].append(accuracy_score(y_te, y_pred))

            avg_metrics = {k: round(float(np.mean(v)), 4) for k, v in fold_metrics.items()}
            std_metrics = {k: round(float(np.std(v)), 4) for k, v in fold_metrics.items()}

            print(f"    Repeated 5-Fold CV (10 repeats, 50 folds total):")
            for metric in ["auc", "precision", "recall", "f1", "accuracy"]:
                print(f"      {metric:12s}: {avg_metrics[metric]:.4f} +/- {std_metrics[metric]:.4f}")

            # ── 2) Leave-One-Out CV ────────────────────────────────
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
                if hasattr(clf, "predict_proba"):
                    loo_probs[test_idx[0]] = clf.predict_proba(X_te)[0, 1]
                else:
                    loo_probs[test_idx[0]] = clf.decision_function(X_te)[0]

            loo_metrics = {
                "auc": round(float(roc_auc_score(y, loo_probs)), 4),
                "precision": round(float(precision_score(y, loo_preds, zero_division=0)), 4),
                "recall": round(float(recall_score(y, loo_preds, zero_division=0)), 4),
                "f1": round(float(f1_score(y, loo_preds, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y, loo_preds)), 4),
            }

            print(f"\n    Leave-One-Out CV:")
            for metric in ["auc", "precision", "recall", "f1", "accuracy"]:
                print(f"      {metric:12s}: {loo_metrics[metric]:.4f}")

            # Store results
            results["models"][model_name][fset_name] = {
                "n_features": int(X_set.shape[1]),
                "repeated_cv": {
                    "mean": avg_metrics,
                    "std": std_metrics,
                    "n_repeats": 10,
                    "n_splits": 5,
                },
                "loo_cv": loo_metrics,
            }

        print()

    # ── Feature importance from GBM (full feature set) ─────────────
    print("=" * 65)
    print("GBM Feature Importance (trained on all data)")
    print("=" * 65)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gbm = clone(models["GradientBoosting"])
    gbm.fit(X_scaled, y)

    importances = gbm.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]

    gbm_importance = {}
    for rank, idx in enumerate(sorted_idx):
        name = feature_names[idx]
        imp = float(importances[idx])
        gbm_importance[name] = round(imp, 4)
        bar = "█" * int(imp * 100)
        print(f"  {rank+1:2d}. {name:42s}  {imp:.4f}  {bar}")

    results["gbm_feature_importance"] = gbm_importance

    # ── Classification report (LOO predictions from GBM) ───────────
    print("\n" + "=" * 65)
    print("Classification Report (GBM LOO predictions, all features)")
    print("=" * 65)

    # Re-run LOO for GBM to get predictions
    loo = LeaveOneOut()
    loo_preds_gbm = np.zeros(len(y))
    loo_probs_gbm = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        clf = clone(models["GradientBoosting"])
        clf.fit(X_tr, y_tr)
        loo_preds_gbm[test_idx[0]] = clf.predict(X_te)[0]
        loo_probs_gbm[test_idx[0]] = clf.predict_proba(X_te)[0, 1]

    report = classification_report(
        y, loo_preds_gbm, target_names=["Human", "Agent"], output_dict=True
    )
    print(classification_report(y, loo_preds_gbm, target_names=["Human", "Agent"]))
    results["classification_report_loo_gbm"] = report

    # Misclassified samples
    misclassified = []
    for i in range(len(y)):
        if loo_preds_gbm[i] != y[i]:
            misclassified.append({
                "address": df.index[i],
                "name": df["name"].iloc[i],
                "true_label": "Agent" if y[i] == 1 else "Human",
                "predicted_label": "Agent" if loo_preds_gbm[i] == 1 else "Human",
                "predicted_prob": round(float(loo_probs_gbm[i]), 4),
            })
    results["misclassified_loo"] = misclassified

    if misclassified:
        print(f"\nMisclassified samples ({len(misclassified)}):")
        for m in misclassified:
            print(f"  {m['name']:35s}  true={m['true_label']:6s}  "
                  f"pred={m['predicted_label']:6s}  prob={m['predicted_prob']:.4f}")

    # ── Save results ───────────────────────────────────────────────
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    train_and_evaluate()
