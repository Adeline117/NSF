#!/usr/bin/env python
"""
Paper 1: Mixed-Class Leave-One-Out (Behavioral Cluster LOPO)
=============================================================
The original LOPO has a fatal flaw: every held-out platform is SINGLE-CLASS
(all agents or all humans), so AUC is undefined within each fold and the
pooled AUC of 0.42 is dominated by domain shift rather than classification
ability.

Fix: Instead of holding out by CATEGORY, hold out by BEHAVIORAL CLUSTER:
  1. Cluster all 1,147 addresses by K-Means(k=5) on the 23 features
  2. Each cluster naturally contains both agents and humans (mixed-class)
  3. Hold out one cluster at a time -> 5-fold cluster-based LOPO
  4. Each fold has both classes in the test set -> AUC is well-defined

This separates two confounds:
  - If cluster-LOPO AUC >> 0.42 (original LOPO): the original LOPO collapse
    was caused by single-class domain shift, not by genuine inability to
    distinguish agents from humans.
  - If cluster-LOPO AUC ~ random CV (0.80): the classifier generalises
    well across behavioral clusters; the problem is purely domain shift.

Outputs:
  experiments/mixed_class_lopo_results.json
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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PARQUET = DATA_DIR / "features_provenance_v4.parquet"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "mixed_class_lopo_results.json"

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
N_CLUSTERS = 5


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


def cluster_lopo(X_np, y, cluster_labels, model_template, model_name):
    """Leave-one-CLUSTER-out CV."""
    unique_clusters = sorted(np.unique(cluster_labels))

    per_cluster = {}
    all_y_true = []
    all_y_prob = []
    all_y_pred = []
    fold_aucs = []

    for c in unique_clusters:
        test_mask = cluster_labels == c
        train_mask = ~test_mask

        X_train, X_test = X_np[train_mask], X_np[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        n_test = int(test_mask.sum())
        n_test_agents = int(y_test.sum())
        n_test_humans = int((y_test == 0).sum())
        n_train_agents = int(y_train.sum())
        n_train_humans = int((y_train == 0).sum())

        # Need both classes in train
        if n_train_agents == 0 or n_train_humans == 0:
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        clf = clone(model_template)
        clf.fit(X_tr_s, y_train)

        y_pred = clf.predict(X_te_s)
        y_prob = clf.predict_proba(X_te_s)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))

        # AUC
        both_classes = n_test_agents > 0 and n_test_humans > 0
        if both_classes:
            auc = float(roc_auc_score(y_test, y_prob))
        else:
            auc = None

        per_cluster[int(c)] = {
            "n_test": n_test,
            "n_test_agents": n_test_agents,
            "n_test_humans": n_test_humans,
            "n_train": int(train_mask.sum()),
            "n_train_agents": n_train_agents,
            "n_train_humans": n_train_humans,
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "auc": round(auc, 4) if auc is not None else None,
            "both_classes_in_test": both_classes,
        }

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())
        all_y_pred.extend(y_pred.tolist())
        if auc is not None:
            fold_aucs.append(auc)

        print(f"    Cluster {c}: n={n_test} (agents={n_test_agents}, "
              f"humans={n_test_humans})  "
              f"AUC={'%.4f' % auc if auc else 'N/A':>6}  "
              f"Acc={acc:.4f}  F1={f1:.4f}")

    # Pooled metrics
    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)
    all_y_pred = np.array(all_y_pred)

    pooled_auc = None
    if len(np.unique(all_y_true)) == 2:
        pooled_auc = round(float(roc_auc_score(all_y_true, all_y_prob)), 4)

    return {
        "model": model_name,
        "n_clusters_evaluated": len(per_cluster),
        "n_clusters_with_auc": len(fold_aucs),
        "mean_per_cluster_auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
        "std_per_cluster_auc": round(float(np.std(fold_aucs)), 4) if fold_aucs else None,
        "min_per_cluster_auc": round(float(np.min(fold_aucs)), 4) if fold_aucs else None,
        "max_per_cluster_auc": round(float(np.max(fold_aucs)), 4) if fold_aucs else None,
        "pooled_auc": pooled_auc,
        "pooled_accuracy": round(float(accuracy_score(all_y_true, all_y_pred)), 4),
        "pooled_f1": round(float(f1_score(all_y_true, all_y_pred, zero_division=0)), 4),
        "pooled_precision": round(float(precision_score(all_y_true, all_y_pred, zero_division=0)), 4),
        "pooled_recall": round(float(recall_score(all_y_true, all_y_pred, zero_division=0)), 4),
        "per_cluster": per_cluster,
    }


def random_cv(X_np, y, model_template, n_splits=5, n_repeats=10):
    """Standard stratified random CV for reference."""
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=SEED)
    aucs, accs, f1s = [], [], []

    for train_idx, test_idx in rskf.split(X_np, y):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        y_prob = clf.predict_proba(X_te)[:, 1]

        aucs.append(roc_auc_score(y_te, y_prob))
        accs.append(accuracy_score(y_te, y_pred))
        f1s.append(f1_score(y_te, y_pred, zero_division=0))

    return {
        "auc_mean": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "accuracy_mean": round(float(np.mean(accs)), 4),
        "f1_mean": round(float(np.mean(f1s)), 4),
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("Paper 1: Mixed-Class Behavioral Cluster LOPO")
    print("=" * 80)

    X, y, categories, feature_cols = load_data()
    X_np = X.values.astype(float)
    print(f"Dataset: n={len(y)}, agents={int(y.sum())}, humans={int((y==0).sum())}")
    print(f"Features: {len(feature_cols)}")
    print(f"Original categories: {categories.nunique()}")

    # ── Step 1: Cluster with K-Means ─────────────────────────────────
    print(f"\nClustering with K-Means (k={N_CLUSTERS})...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED)
    cluster_labels = km.fit_predict(X_scaled)

    print(f"\nCluster composition:")
    print(f"  {'Cluster':<10} {'Total':>6} {'Agents':>7} {'Humans':>7} "
          f"{'Agent%':>8} {'Mixed':>6}")
    print(f"  {'-'*10} {'-'*6} {'-'*7} {'-'*7} {'-'*8} {'-'*6}")

    cluster_info = {}
    for c in range(N_CLUSTERS):
        mask = cluster_labels == c
        n_total = int(mask.sum())
        n_agents = int(y[mask].sum())
        n_humans = int((y[mask] == 0).sum())
        pct_agent = n_agents / n_total * 100 if n_total > 0 else 0
        is_mixed = n_agents > 0 and n_humans > 0
        print(f"  {c:<10} {n_total:>6} {n_agents:>7} {n_humans:>7} "
              f"{pct_agent:>7.1f}% {'YES' if is_mixed else 'NO':>6}")

        # Which original categories are in this cluster
        cat_dist = {}
        for cat in categories[mask].unique():
            cat_dist[cat] = int((categories[mask] == cat).sum())

        cluster_info[int(c)] = {
            "n_total": n_total,
            "n_agents": n_agents,
            "n_humans": n_humans,
            "agent_ratio": round(n_agents / n_total, 4) if n_total > 0 else 0.0,
            "is_mixed_class": is_mixed,
            "category_distribution": dict(sorted(cat_dist.items(), key=lambda x: -x[1])),
        }

    n_mixed = sum(1 for v in cluster_info.values() if v["is_mixed_class"])
    print(f"\n  Mixed-class clusters: {n_mixed} / {N_CLUSTERS}")

    # ── Step 2: Cluster-based LOPO ────────────────────────────────────
    print(f"\n--- Cluster-based LOPO (k={N_CLUSTERS}) ---")
    models = get_models()
    cluster_lopo_results = {}

    for model_name, model_template in models.items():
        print(f"\n  {model_name}:")
        cluster_lopo_results[model_name] = cluster_lopo(
            X_np, y, cluster_labels, model_template, model_name
        )

    # ── Step 3: Random CV baseline ────────────────────────────────────
    print(f"\n--- Random 5x10 Stratified CV (baseline) ---")
    random_cv_results = {}
    for model_name, model_template in models.items():
        cv_res = random_cv(X_np, y, model_template)
        random_cv_results[model_name] = cv_res
        print(f"  {model_name}: AUC={cv_res['auc_mean']:.4f} +/- {cv_res['auc_std']:.4f}  "
              f"F1={cv_res['f1_mean']:.4f}")

    # ── Step 4: Try different k values for sensitivity ────────────────
    print(f"\n--- Sensitivity to k (number of clusters) ---")
    k_sensitivity = {}
    for k in [3, 5, 7, 10]:
        km_k = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        cl_k = km_k.fit_predict(X_scaled)

        # Quick GBM-only evaluation
        gbm = get_models()["GradientBoosting"]
        all_y_true, all_y_prob = [], []
        fold_aucs = []

        for c in range(k):
            test_mask = cl_k == c
            train_mask = ~test_mask
            X_tr, X_te = X_np[train_mask], X_np[test_mask]
            y_tr, y_te = y[train_mask], y[test_mask]

            if len(np.unique(y_tr)) < 2:
                continue

            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr)
            X_te_s = sc.transform(X_te)
            clf = clone(gbm)
            clf.fit(X_tr_s, y_tr)
            y_prob = clf.predict_proba(X_te_s)[:, 1]

            all_y_true.extend(y_te.tolist())
            all_y_prob.extend(y_prob.tolist())

            if len(np.unique(y_te)) == 2:
                fold_aucs.append(roc_auc_score(y_te, y_prob))

        all_y_true = np.array(all_y_true)
        all_y_prob = np.array(all_y_prob)
        pooled_auc = float(roc_auc_score(all_y_true, all_y_prob)) if len(np.unique(all_y_true)) == 2 else None

        # Count mixed clusters
        mixed = sum(1 for c in range(k) if (y[cl_k == c].sum() > 0 and ((cl_k == c) & (y == 0)).sum() > 0))

        k_sensitivity[k] = {
            "n_mixed_clusters": mixed,
            "n_clusters": k,
            "mean_fold_auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
            "pooled_auc": round(pooled_auc, 4) if pooled_auc else None,
        }
        mean_auc_str = f"{np.mean(fold_aucs):.4f}" if fold_aucs else "N/A"
        pooled_auc_str = f"{pooled_auc:.4f}" if pooled_auc else "N/A"
        print(f"  k={k:<3}  mixed={mixed}/{k}  "
              f"mean_fold_AUC={mean_auc_str:>6}  "
              f"pooled_AUC={pooled_auc_str:>6}")

    # ── Comparison with original LOPO ─────────────────────────────────
    original_lopo_gbm_auc = 0.4221  # from leave_one_platform_out_results.json
    random_cv_gbm_auc = random_cv_results["GradientBoosting"]["auc_mean"]
    cluster_lopo_gbm_auc = cluster_lopo_results["GradientBoosting"]["pooled_auc"]

    if cluster_lopo_gbm_auc and cluster_lopo_gbm_auc > 0.65:
        diagnosis = (
            f"DOMAIN SHIFT CONFIRMED: Cluster-LOPO pooled AUC = "
            f"{cluster_lopo_gbm_auc:.4f} >> original LOPO = "
            f"{original_lopo_gbm_auc:.4f}. "
            f"The original LOPO's collapse to 0.42 was caused by single-class "
            f"domain shift (each category being all-agent or all-human), NOT by "
            f"the classifier's inability to separate agents from humans. "
            f"With mixed-class behavioral clusters, the classifier maintains "
            f"{'strong' if cluster_lopo_gbm_auc > 0.75 else 'moderate'} "
            f"discrimination ability."
        )
    elif cluster_lopo_gbm_auc and cluster_lopo_gbm_auc > 0.55:
        diagnosis = (
            f"PARTIAL DOMAIN SHIFT: Cluster-LOPO pooled AUC = "
            f"{cluster_lopo_gbm_auc:.4f} > original LOPO = "
            f"{original_lopo_gbm_auc:.4f}, but still well below "
            f"random CV = {random_cv_gbm_auc:.4f}. "
            f"Some of the LOPO collapse is due to single-class folds, "
            f"but there is also genuine behavioral heterogeneity that "
            f"hurts cross-cluster generalisation."
        )
    else:
        diagnosis = (
            f"FUNDAMENTAL GENERALIZATION FAILURE: Cluster-LOPO pooled AUC = "
            f"{cluster_lopo_gbm_auc} is still near chance, similar to "
            f"original LOPO = {original_lopo_gbm_auc:.4f}. "
            f"The features do not generalise across behavioral clusters "
            f"even when both classes are present."
        )

    print(f"\n{'='*80}")
    print("COMPARISON")
    print("=" * 80)
    print(f"  Original LOPO (GBM, pooled AUC):          {original_lopo_gbm_auc:.4f}")
    print(f"  Cluster LOPO (GBM, pooled AUC):           {cluster_lopo_gbm_auc}")
    print(f"  Random CV (GBM, mean AUC):                {random_cv_gbm_auc:.4f}")
    print(f"\nDIAGNOSIS:")
    print(diagnosis)

    # ── Assemble results ──────────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Mixed-class behavioral cluster LOPO: clusters addresses by "
            "K-Means on features, holds out one cluster at a time. "
            "Each cluster has both agents and humans (mixed-class), "
            "fixing the single-class fold problem in the original LOPO."
        ),
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(feature_cols),
            "feature_names": feature_cols,
        },
        "clustering": {
            "method": f"K-Means (k={N_CLUSTERS}), fitted on StandardScaler(X)",
            "n_clusters": N_CLUSTERS,
            "n_mixed_class_clusters": n_mixed,
            "cluster_info": cluster_info,
        },
        "cluster_lopo": cluster_lopo_results,
        "random_cv": random_cv_results,
        "k_sensitivity": k_sensitivity,
        "comparison": {
            "original_lopo_gbm_pooled_auc": original_lopo_gbm_auc,
            "cluster_lopo_gbm_pooled_auc": cluster_lopo_gbm_auc,
            "random_cv_gbm_mean_auc": random_cv_gbm_auc,
            "delta_cluster_vs_original": round(
                (cluster_lopo_gbm_auc or 0) - original_lopo_gbm_auc, 4
            ),
            "delta_cluster_vs_random_cv": round(
                (cluster_lopo_gbm_auc or 0) - random_cv_gbm_auc, 4
            ),
        },
        "diagnosis": diagnosis,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # JSON-safe
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
