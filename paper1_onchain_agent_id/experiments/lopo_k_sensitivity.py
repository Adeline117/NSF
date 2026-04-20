#!/usr/bin/env python
"""
Paper 1: K-Sensitivity Analysis for Mixed-Class LOPO
=====================================================
Reviewer concern: "The mixed-class LOPO uses K-Means k=5. Is this cherry-picked?"

This script:
  1. Loads the same features/labels used in mixed_class_lopo.py
  2. Runs mixed-class LOPO with k = 3, 4, 5, 6, 7, 8, 10
  3. For each k, reports RF pooled AUC and whether all clusters are mixed-class
  4. Saves to lopo_k_sensitivity_results.json
  5. Key question: is the AUC ~0.74 stable across k values, or does it collapse?

Outputs:
  experiments/lopo_k_sensitivity_results.json
"""

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PARQUET = DATA_DIR / "features_provenance_v4.parquet"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "lopo_k_sensitivity_results.json"

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
for gf in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(gf)

SEED = 42
K_VALUES = [3, 4, 5, 6, 7, 8, 10]


def get_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=3,
            random_state=SEED, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=3, random_state=SEED,
        ),
    }


def load_data():
    df = pd.read_parquet(FEATURES_PARQUET)
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df["label"].values.astype(int)
    categories = df["category"].copy()

    # Impute NaN
    X_vals = X.values.astype(float)
    nan_mask = np.isnan(X_vals)
    if nan_mask.any():
        col_medians = np.nanmedian(X_vals, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X_vals.shape[1]):
            X_vals[nan_mask[:, j], j] = col_medians[j]
        X = pd.DataFrame(X_vals, columns=feature_cols, index=df.index)

    # Clip
    for col in X.columns:
        lo, hi = np.nanpercentile(X[col], [1, 99])
        X[col] = X[col].clip(lo, hi)

    return X, y, categories, feature_cols


def run_cluster_lopo(X_np, y, cluster_labels, model_template, model_name, k):
    """Leave-one-CLUSTER-out CV for a given k."""
    unique_clusters = sorted(np.unique(cluster_labels))

    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    fold_aucs = []
    per_cluster = {}

    for c in unique_clusters:
        test_mask = cluster_labels == c
        train_mask = ~test_mask

        X_train, X_test = X_np[train_mask], X_np[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        n_test = int(test_mask.sum())
        n_test_agents = int(y_test.sum())
        n_test_humans = int((y_test == 0).sum())

        # Need both classes in train
        if int(y_train.sum()) == 0 or int((y_train == 0).sum()) == 0:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        clf = clone(model_template)
        clf.fit(X_tr_s, y_train)

        y_pred = clf.predict(X_te_s)
        y_prob = clf.predict_proba(X_te_s)[:, 1]

        both_classes = n_test_agents > 0 and n_test_humans > 0
        if both_classes:
            auc = float(roc_auc_score(y_test, y_prob))
            fold_aucs.append(auc)
        else:
            auc = None

        per_cluster[int(c)] = {
            "n_test": n_test,
            "n_test_agents": n_test_agents,
            "n_test_humans": n_test_humans,
            "auc": round(auc, 4) if auc is not None else None,
            "both_classes_in_test": both_classes,
        }

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_y_pred.extend(y_pred.tolist())

    # Pooled metrics
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    pooled_auc = None
    if len(np.unique(all_y_true)) == 2:
        pooled_auc = round(float(roc_auc_score(all_y_true, all_y_prob)), 4)

    pooled_acc = round(float(accuracy_score(all_y_true, (all_y_prob >= 0.5).astype(int))), 4)
    pooled_f1 = round(float(f1_score(all_y_true, (all_y_prob >= 0.5).astype(int), zero_division=0)), 4)

    return {
        "model": model_name,
        "k": k,
        "n_clusters_evaluated": len(per_cluster),
        "n_clusters_with_both_classes": sum(1 for v in per_cluster.values() if v["both_classes_in_test"]),
        "mean_per_cluster_auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
        "std_per_cluster_auc": round(float(np.std(fold_aucs)), 4) if fold_aucs else None,
        "min_per_cluster_auc": round(float(np.min(fold_aucs)), 4) if fold_aucs else None,
        "max_per_cluster_auc": round(float(np.max(fold_aucs)), 4) if fold_aucs else None,
        "pooled_auc": pooled_auc,
        "pooled_accuracy": pooled_acc,
        "pooled_f1": pooled_f1,
        "per_cluster": per_cluster,
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("Paper 1: K-Sensitivity for Mixed-Class Cluster LOPO")
    print("=" * 80)

    X, y, categories, feature_cols = load_data()
    X_np = X.values.astype(float)
    print(f"Dataset: n={len(y)}, agents={int(y.sum())}, humans={int((y==0).sum())}")
    print(f"Features: {len(feature_cols)}")

    # Scale once for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    models = get_models()
    results_by_k = {}

    print(f"\nRunning k = {K_VALUES}")
    print("-" * 80)

    for k in K_VALUES:
        print(f"\n--- k = {k} ---")

        km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        cluster_labels = km.fit_predict(X_scaled)

        # Check cluster composition
        n_mixed = 0
        all_mixed = True
        cluster_composition = {}
        for c in range(k):
            mask = cluster_labels == c
            n_agents = int(y[mask].sum())
            n_humans = int((y[mask] == 0).sum())
            is_mixed = n_agents > 0 and n_humans > 0
            if is_mixed:
                n_mixed += 1
            else:
                all_mixed = False
            cluster_composition[int(c)] = {
                "n_total": int(mask.sum()),
                "n_agents": n_agents,
                "n_humans": n_humans,
                "is_mixed": is_mixed,
            }

        print(f"  Mixed clusters: {n_mixed}/{k} (all_mixed={all_mixed})")

        k_results = {
            "k": k,
            "all_clusters_mixed": all_mixed,
            "n_mixed_clusters": n_mixed,
            "cluster_composition": cluster_composition,
            "models": {},
        }

        for model_name, model_template in models.items():
            res = run_cluster_lopo(X_np, y, cluster_labels, model_template, model_name, k)
            k_results["models"][model_name] = res
            print(f"  {model_name}: pooled_AUC={res['pooled_auc']}  "
                  f"mean_fold_AUC={res['mean_per_cluster_auc']}  "
                  f"std={res['std_per_cluster_auc']}")

        results_by_k[str(k)] = k_results

    # Summary table
    print(f"\n{'='*80}")
    print("SUMMARY: RF Pooled AUC across k values")
    print("=" * 80)
    print(f"  {'k':<5} {'All Mixed':<12} {'RF Pooled AUC':<16} {'GBM Pooled AUC':<16} "
          f"{'RF Mean Fold':<14} {'RF Std':<8}")
    print(f"  {'-'*5} {'-'*12} {'-'*16} {'-'*16} {'-'*14} {'-'*8}")

    rf_aucs = []
    gbm_aucs = []
    for k in K_VALUES:
        r = results_by_k[str(k)]
        rf_auc = r["models"]["RandomForest"]["pooled_auc"]
        gbm_auc = r["models"]["GradientBoosting"]["pooled_auc"]
        rf_mean = r["models"]["RandomForest"]["mean_per_cluster_auc"]
        rf_std = r["models"]["RandomForest"]["std_per_cluster_auc"]
        all_mixed = r["all_clusters_mixed"]
        rf_aucs.append(rf_auc)
        gbm_aucs.append(gbm_auc)
        print(f"  {k:<5} {'YES' if all_mixed else 'NO':<12} {rf_auc:<16} {gbm_auc:<16} "
              f"{rf_mean:<14} {rf_std:<8}")

    # Stability assessment
    rf_aucs_valid = [a for a in rf_aucs if a is not None]
    gbm_aucs_valid = [a for a in gbm_aucs if a is not None]

    rf_range = max(rf_aucs_valid) - min(rf_aucs_valid) if rf_aucs_valid else None
    gbm_range = max(gbm_aucs_valid) - min(gbm_aucs_valid) if gbm_aucs_valid else None
    rf_mean_all = float(np.mean(rf_aucs_valid)) if rf_aucs_valid else None
    gbm_mean_all = float(np.mean(gbm_aucs_valid)) if gbm_aucs_valid else None
    rf_std_all = float(np.std(rf_aucs_valid)) if rf_aucs_valid else None
    gbm_std_all = float(np.std(gbm_aucs_valid)) if gbm_aucs_valid else None

    stability_assessment = (
        f"RF pooled AUC across k={K_VALUES}: "
        f"mean={rf_mean_all:.4f}, std={rf_std_all:.4f}, range={rf_range:.4f}. "
        f"GBM pooled AUC: mean={gbm_mean_all:.4f}, std={gbm_std_all:.4f}, range={gbm_range:.4f}. "
    )

    if rf_range is not None and rf_range < 0.05:
        stability_assessment += (
            "The pooled AUC is STABLE across k (range < 0.05 AUC). "
            "The k=5 result is NOT cherry-picked; any k in [3,10] yields similar performance."
        )
    elif rf_range is not None and rf_range < 0.10:
        stability_assessment += (
            "The pooled AUC shows MODERATE sensitivity to k (range 0.05-0.10). "
            "The k=5 result is representative but not the unique optimum."
        )
    else:
        stability_assessment += (
            "The pooled AUC shows HIGH sensitivity to k (range > 0.10). "
            "The k=5 result may be cherry-picked; results vary substantially with k."
        )

    print(f"\nSTABILITY ASSESSMENT:")
    print(f"  {stability_assessment}")

    # Assemble final results
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": (
            "K-sensitivity analysis for mixed-class cluster LOPO. "
            "Tests whether the pooled AUC at k=5 is stable or cherry-picked "
            "by evaluating k = 3, 4, 5, 6, 7, 8, 10."
        ),
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(feature_cols),
        },
        "k_values_tested": K_VALUES,
        "results_by_k": results_by_k,
        "summary": {
            "rf_pooled_aucs": {str(k): results_by_k[str(k)]["models"]["RandomForest"]["pooled_auc"] for k in K_VALUES},
            "gbm_pooled_aucs": {str(k): results_by_k[str(k)]["models"]["GradientBoosting"]["pooled_auc"] for k in K_VALUES},
            "rf_mean_across_k": round(rf_mean_all, 4) if rf_mean_all else None,
            "rf_std_across_k": round(rf_std_all, 4) if rf_std_all else None,
            "rf_range_across_k": round(rf_range, 4) if rf_range else None,
            "gbm_mean_across_k": round(gbm_mean_all, 4) if gbm_mean_all else None,
            "gbm_std_across_k": round(gbm_std_all, 4) if gbm_std_all else None,
            "gbm_range_across_k": round(gbm_range, 4) if gbm_range else None,
            "all_k_have_all_mixed_clusters": all(
                results_by_k[str(k)]["all_clusters_mixed"] for k in K_VALUES
            ),
        },
        "stability_assessment": stability_assessment,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # JSON-safe conversion
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_safe = json.loads(json.dumps(results, default=_convert))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_safe, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")
    print(f"Elapsed: {results['elapsed_seconds']:.1f}s")
    print("Done.")


if __name__ == "__main__":
    main()
