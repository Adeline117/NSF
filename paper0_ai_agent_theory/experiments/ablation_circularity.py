#!/usr/bin/env python
"""
Paper 0: Circularity Ablation — Reviewer-requested experiment
==============================================================
The taxonomy projection rules (Tier 2) share 4 behavioral features with the
classifier feature set.  A reviewer concern is that this creates circularity:
the same features used to *assign* labels are then used to *predict* them.

This ablation removes those 4 shared features and re-evaluates.

Shared features (Tier 2 projection rules):
  1. burst_frequency
  2. gas_price_round_number_ratio
  3. unique_contracts_ratio
  4. multi_protocol_interaction_count

Conditions:
  A. 23-feature baseline (all original features)
  B. 19-feature ablated (remove 4 shared)
  C. 31-feature baseline (23 original + 8 AI-detection)
  D. 27-feature ablated (31 minus 4 shared)

Additionally, runs HDBSCAN clustering comparison vs K-Means.

Outputs:
  - ablation_circularity_results.json
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    classification_report,
    f1_score,
    normalized_mutual_info_score,
    silhouette_score,
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
    / "ablation_circularity_results.json"
)

# ── Feature sets ──────────────────────────────────────────────────────
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

# The 4 shared features used in Tier 2 projection rules
SHARED_4 = [
    "burst_frequency",
    "gas_price_round_number_ratio",
    "unique_contracts_ratio",
    "multi_protocol_interaction_count",
]

# Ablated sets
ABLATED_19 = [f for f in ORIGINAL_23 if f not in SHARED_4]
ABLATED_27 = [f for f in ALL_31 if f not in SHARED_4]

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

SEED = 42
N_FOLDS = 5


# ── Helpers ───────────────────────────────────────────────────────────

def impute_and_clip(X):
    """Median-impute NaNs and clip to [1st, 99th] percentile."""
    X = X.copy()
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


def run_gbm_cv(X, y, feature_names, n_folds=N_FOLDS, seed=SEED):
    """Run GBM with stratified CV and return metrics dict."""
    model_template = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=seed,
    )

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    classes_sorted = sorted(set(y.tolist()))

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
        fold_f1_macro.append(f1_score(y[te_idx], y_pred, average="macro", zero_division=0))
        fold_f1_weighted.append(f1_score(y[te_idx], y_pred, average="weighted", zero_division=0))
        all_y_true.extend(y[te_idx].tolist())
        all_y_pred.extend(y_pred.tolist())

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)

    report = classification_report(
        y_true_arr, y_pred_arr,
        labels=classes_sorted,
        target_names=[TAXONOMY_NAMES.get(int(c), f"C{c}") for c in classes_sorted],
        output_dict=True,
        zero_division=0,
    )

    # Feature importance on full data
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    gbm_full = clone(model_template)
    gbm_full.fit(Xs, y)
    fi = {
        feature_names[i]: round(float(gbm_full.feature_importances_[i]), 4)
        for i in np.argsort(gbm_full.feature_importances_)[::-1]
    }

    per_class_f1 = {}
    for cls_name, stats in report.items():
        if isinstance(stats, dict) and "f1-score" in stats:
            per_class_f1[cls_name] = round(stats["f1-score"], 4)

    return {
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "accuracy_mean": round(float(np.mean(fold_accs)), 4),
        "accuracy_std": round(float(np.std(fold_accs)), 4),
        "f1_macro_mean": round(float(np.mean(fold_f1_macro)), 4),
        "f1_macro_std": round(float(np.std(fold_f1_macro)), 4),
        "f1_weighted_mean": round(float(np.mean(fold_f1_weighted)), 4),
        "per_class_f1": per_class_f1,
        "per_class_report": report,
        "feature_importance": fi,
    }


def load_and_merge():
    """Load Paper 0's 23-feature dataset and merge with Paper 3's 8 AI features."""
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["label"] == 1].copy()

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


# ── Main ──────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("Paper 0: Circularity Ablation Experiment")
    print("  Removing 4 shared features: " + ", ".join(SHARED_4))
    print("=" * 80)

    # Load data
    df = load_and_merge()
    y = df["taxonomy_index"].values.astype(int)
    print(f"\nTotal agents: {len(df)}")
    print(f"Classes: {sorted(set(y.tolist()))}")

    unique, counts = np.unique(y, return_counts=True)
    print("\nClass distribution:")
    for u, c in zip(unique, counts):
        print(f"  {u} ({TAXONOMY_NAMES.get(int(u), '?'):<25}) n={c}")

    # ── Condition A: 23-feature baseline ──────────────────────────────
    print("\n" + "=" * 80)
    print("CONDITION A: 23-feature baseline (all original)")
    print("=" * 80)
    X_23 = df[ORIGINAL_23].values.astype(float)
    X_23 = impute_and_clip(X_23)
    results_a = run_gbm_cv(X_23, y, ORIGINAL_23)
    print(f"  Accuracy: {results_a['accuracy_mean']:.4f} +/- {results_a['accuracy_std']:.4f}")
    print(f"  F1-macro: {results_a['f1_macro_mean']:.4f} +/- {results_a['f1_macro_std']:.4f}")
    print(f"  Per-class F1:")
    for cls_name, f1 in results_a["per_class_f1"].items():
        print(f"    {cls_name:<25} {f1:.4f}")

    # ── Condition B: 19-feature ablated ───────────────────────────────
    print("\n" + "=" * 80)
    print("CONDITION B: 19-feature ablated (remove 4 shared)")
    print(f"  Removed: {SHARED_4}")
    print("=" * 80)
    X_19 = df[ABLATED_19].values.astype(float)
    X_19 = impute_and_clip(X_19)
    results_b = run_gbm_cv(X_19, y, ABLATED_19)
    print(f"  Accuracy: {results_b['accuracy_mean']:.4f} +/- {results_b['accuracy_std']:.4f}")
    print(f"  F1-macro: {results_b['f1_macro_mean']:.4f} +/- {results_b['f1_macro_std']:.4f}")
    print(f"  Per-class F1:")
    for cls_name, f1 in results_b["per_class_f1"].items():
        print(f"    {cls_name:<25} {f1:.4f}")

    # ── Condition C: 31-feature baseline ──────────────────────────────
    print("\n" + "=" * 80)
    print("CONDITION C: 31-feature baseline (23 + 8 AI-detection)")
    print("=" * 80)
    X_31 = df[ALL_31].values.astype(float)
    X_31 = impute_and_clip(X_31)
    results_c = run_gbm_cv(X_31, y, ALL_31)
    print(f"  Accuracy: {results_c['accuracy_mean']:.4f} +/- {results_c['accuracy_std']:.4f}")
    print(f"  F1-macro: {results_c['f1_macro_mean']:.4f} +/- {results_c['f1_macro_std']:.4f}")
    print(f"  Per-class F1:")
    for cls_name, f1 in results_c["per_class_f1"].items():
        print(f"    {cls_name:<25} {f1:.4f}")

    # ── Condition D: 27-feature ablated ───────────────────────────────
    print("\n" + "=" * 80)
    print("CONDITION D: 27-feature ablated (31 minus 4 shared)")
    print(f"  Removed: {SHARED_4}")
    print("=" * 80)
    X_27 = df[ABLATED_27].values.astype(float)
    X_27 = impute_and_clip(X_27)
    results_d = run_gbm_cv(X_27, y, ABLATED_27)
    print(f"  Accuracy: {results_d['accuracy_mean']:.4f} +/- {results_d['accuracy_std']:.4f}")
    print(f"  F1-macro: {results_d['f1_macro_mean']:.4f} +/- {results_d['f1_macro_std']:.4f}")
    print(f"  Per-class F1:")
    for cls_name, f1 in results_d["per_class_f1"].items():
        print(f"    {cls_name:<25} {f1:.4f}")

    # ── Deltas ────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("ABLATION DELTAS (ablated minus baseline)")
    print("=" * 80)

    delta_23_to_19 = {
        "accuracy_delta": round(results_b["accuracy_mean"] - results_a["accuracy_mean"], 4),
        "f1_macro_delta": round(results_b["f1_macro_mean"] - results_a["f1_macro_mean"], 4),
        "per_class_f1_delta": {},
    }
    for cls_name in results_a["per_class_f1"]:
        base = results_a["per_class_f1"].get(cls_name, 0)
        ablated = results_b["per_class_f1"].get(cls_name, 0)
        delta_23_to_19["per_class_f1_delta"][cls_name] = round(ablated - base, 4)

    delta_31_to_27 = {
        "accuracy_delta": round(results_d["accuracy_mean"] - results_c["accuracy_mean"], 4),
        "f1_macro_delta": round(results_d["f1_macro_mean"] - results_c["f1_macro_mean"], 4),
        "per_class_f1_delta": {},
    }
    for cls_name in results_c["per_class_f1"]:
        base = results_c["per_class_f1"].get(cls_name, 0)
        ablated = results_d["per_class_f1"].get(cls_name, 0)
        delta_31_to_27["per_class_f1_delta"][cls_name] = round(ablated - base, 4)

    print(f"\n23 -> 19 features:")
    print(f"  Accuracy delta: {delta_23_to_19['accuracy_delta']:+.4f}")
    print(f"  F1-macro delta: {delta_23_to_19['f1_macro_delta']:+.4f}")
    for cls_name, d in delta_23_to_19["per_class_f1_delta"].items():
        print(f"    {cls_name:<25} {d:+.4f}")

    print(f"\n31 -> 27 features:")
    print(f"  Accuracy delta: {delta_31_to_27['accuracy_delta']:+.4f}")
    print(f"  F1-macro delta: {delta_31_to_27['f1_macro_delta']:+.4f}")
    for cls_name, d in delta_31_to_27["per_class_f1_delta"].items():
        print(f"    {cls_name:<25} {d:+.4f}")

    # ── HDBSCAN Clustering Comparison ─────────────────────────────────
    print("\n" + "=" * 80)
    print("HDBSCAN CLUSTERING COMPARISON")
    print("=" * 80)

    # Use the 23-feature set for clustering comparison (same as cluster_validation.py)
    X_clust = impute_and_clip(df[ORIGINAL_23].values.astype(float))
    scaler = StandardScaler()
    Xs_clust = scaler.fit_transform(X_clust)
    tax_labels = y

    clustering_results = {}

    # K-Means baselines (k=3 and k=8)
    for k in [3, 8]:
        km = KMeans(n_clusters=k, n_init=20, random_state=SEED)
        labels = km.fit_predict(Xs_clust)
        sil = silhouette_score(Xs_clust, labels, sample_size=2000, random_state=SEED)
        ari = adjusted_rand_score(tax_labels, labels)
        nmi = normalized_mutual_info_score(tax_labels, labels)

        key = f"KMeans_k{k}"
        clustering_results[key] = {
            "method": "KMeans",
            "k": k,
            "n_clusters_found": k,
            "n_noise": 0,
            "silhouette": round(float(sil), 4),
            "ari_vs_taxonomy": round(float(ari), 4),
            "nmi_vs_taxonomy": round(float(nmi), 4),
        }
        print(f"\n  KMeans k={k}:")
        print(f"    Clusters found: {k}")
        print(f"    Silhouette:     {sil:.4f}")
        print(f"    ARI:            {ari:.4f}")
        print(f"    NMI:            {nmi:.4f}")

    # HDBSCAN with different min_cluster_size
    for mcs in [20, 50, 100]:
        hdb = HDBSCAN(min_cluster_size=mcs, n_jobs=-1)
        labels = hdb.fit_predict(Xs_clust)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = int((labels == -1).sum())
        noise_pct = round(100.0 * n_noise / len(labels), 1)

        # Silhouette: only on non-noise points
        non_noise_mask = labels != -1
        if n_clusters >= 2 and non_noise_mask.sum() > n_clusters:
            sil = silhouette_score(
                Xs_clust[non_noise_mask], labels[non_noise_mask],
                sample_size=min(2000, int(non_noise_mask.sum())),
                random_state=SEED,
            )
        else:
            sil = float("nan")

        # ARI and NMI: only on non-noise points
        if n_clusters >= 2 and non_noise_mask.sum() > 0:
            ari = adjusted_rand_score(tax_labels[non_noise_mask], labels[non_noise_mask])
            nmi = normalized_mutual_info_score(tax_labels[non_noise_mask], labels[non_noise_mask])
        else:
            ari = float("nan")
            nmi = float("nan")

        # Cluster sizes
        cluster_sizes = {}
        for cid in sorted(set(labels)):
            if cid == -1:
                continue
            cluster_sizes[str(cid)] = int((labels == cid).sum())

        key = f"HDBSCAN_mcs{mcs}"
        clustering_results[key] = {
            "method": "HDBSCAN",
            "min_cluster_size": mcs,
            "n_clusters_found": n_clusters,
            "n_noise": n_noise,
            "noise_pct": noise_pct,
            "silhouette": round(float(sil), 4) if not np.isnan(sil) else None,
            "ari_vs_taxonomy": round(float(ari), 4) if not np.isnan(ari) else None,
            "nmi_vs_taxonomy": round(float(nmi), 4) if not np.isnan(nmi) else None,
            "cluster_sizes": cluster_sizes,
        }
        print(f"\n  HDBSCAN min_cluster_size={mcs}:")
        print(f"    Clusters found: {n_clusters}")
        print(f"    Noise points:   {n_noise} ({noise_pct}%)")
        print(f"    Silhouette:     {sil:.4f}" if not np.isnan(sil) else "    Silhouette:     N/A")
        print(f"    ARI:            {ari:.4f}" if not np.isnan(ari) else "    ARI:            N/A")
        print(f"    NMI:            {nmi:.4f}" if not np.isnan(nmi) else "    NMI:            N/A")
        if cluster_sizes:
            print(f"    Cluster sizes:  {cluster_sizes}")

    # ── Save results ──────────────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": "Circularity ablation: removing 4 shared Tier-2 projection features",
        "shared_features_removed": SHARED_4,
        "n_samples": int(len(y)),
        "class_counts": {
            TAXONOMY_NAMES.get(int(u), f"C{u}"): int(c)
            for u, c in zip(*np.unique(y, return_counts=True))
        },
        "conditions": {
            "A_23feat_baseline": results_a,
            "B_19feat_ablated": results_b,
            "C_31feat_baseline": results_c,
            "D_27feat_ablated": results_d,
        },
        "deltas": {
            "23_to_19": delta_23_to_19,
            "31_to_27": delta_31_to_27,
        },
        "clustering_comparison": clustering_results,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
