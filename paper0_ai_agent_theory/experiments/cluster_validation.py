"""
Paper 0: Cluster Validation of Taxonomy Completeness
=====================================================
Runs unsupervised clustering on the 23-feature matrix of Paper 1's
expanded agent set and compares the discovered clusters to the
rule-based taxonomy projection from taxonomy_projection.py.

Methods:
  1. K-Means sweep over k in [3, 15] with silhouette scoring.
  2. Best-k K-Means → compare to taxonomy labels via ARI, NMI, purity.
  3. Identify orphan clusters (purity < 0.5) as candidate new categories.
  4. Report cluster → category mapping.

Outputs:
  - cluster_validation_results.json
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PARQUET = (
    PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
)
RESULTS_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "cluster_validation_results.json"
)

# The 23 behavioral features (match pipeline)
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


def main():
    print("=" * 80)
    print("Paper 0: Cluster Validation of Taxonomy")
    print("=" * 80)

    df = pd.read_parquet(FEATURES_PARQUET)
    # Restrict to agents only for taxonomy validation
    df = df[df["label"] == 1].copy()
    print(f"Agents: {len(df)}")

    # Feature matrix
    X = df[FEATURE_COLS].values.astype(float)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]

    # Clip extremes
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Taxonomy labels (from rule-based projection)
    tax_labels = df["taxonomy_index"].values.astype(int)
    unique_tax = sorted(set(tax_labels))
    print(f"Taxonomy categories present: {len(unique_tax)}")
    for t in unique_tax:
        print(f"  {t}: {int((tax_labels == t).sum())}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_agents": int(len(df)),
        "n_features": len(FEATURE_COLS),
        "n_taxonomy_categories_present": int(len(unique_tax)),
    }

    # K-Means sweep
    print("\nK-Means sweep:")
    sweep_results = {}
    for k in range(3, 16):
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        labels = km.fit_predict(Xs)
        try:
            sil = silhouette_score(Xs, labels, sample_size=2000, random_state=42)
        except Exception:
            sil = float("nan")
        ari = adjusted_rand_score(tax_labels, labels)
        nmi = normalized_mutual_info_score(tax_labels, labels)
        sweep_results[str(k)] = {
            "silhouette": round(float(sil), 4),
            "ari_vs_taxonomy": round(float(ari), 4),
            "nmi_vs_taxonomy": round(float(nmi), 4),
            "inertia": round(float(km.inertia_), 2),
        }
        print(f"  k={k:>2}  silhouette={sil:.4f}  ARI={ari:.4f}  NMI={nmi:.4f}")
    results["kmeans_sweep"] = sweep_results

    # Pick the best k by silhouette
    best_k = max(sweep_results, key=lambda kk: sweep_results[kk]["silhouette"])
    best_k_int = int(best_k)
    print(f"\nBest k by silhouette: {best_k_int}")

    # Compare taxonomy (8 cats) baseline + best-k mapping
    for k_target in [8, best_k_int]:
        print(f"\n--- Analysis at k={k_target} ---")
        km = KMeans(n_clusters=k_target, n_init=20, random_state=42)
        labels = km.fit_predict(Xs)

        # Confusion matrix
        confusion = {}
        for cluster_id in range(k_target):
            cluster_mask = labels == cluster_id
            cluster_size = int(cluster_mask.sum())
            if cluster_size == 0:
                continue
            tax_in_cluster = tax_labels[cluster_mask]
            tax_counts = pd.Series(tax_in_cluster).value_counts()
            top_cat = int(tax_counts.idxmax())
            purity = float(tax_counts.max() / cluster_size)
            confusion[f"cluster_{cluster_id}"] = {
                "size": cluster_size,
                "dominant_tax_category": top_cat,
                "purity": round(purity, 4),
                "tax_distribution": {
                    int(k): int(v) for k, v in tax_counts.items()
                },
                "is_orphan": bool(purity < 0.5),
            }

        ari = adjusted_rand_score(tax_labels, labels)
        nmi = normalized_mutual_info_score(tax_labels, labels)
        mean_purity = float(np.mean([c["purity"] for c in confusion.values()]))
        n_orphan = sum(1 for c in confusion.values() if c["is_orphan"])

        key = f"k_{k_target}_detail"
        results[key] = {
            "ari": round(ari, 4),
            "nmi": round(nmi, 4),
            "mean_purity": round(mean_purity, 4),
            "n_orphan_clusters": n_orphan,
            "confusion": confusion,
        }
        print(f"  ARI={ari:.4f}  NMI={nmi:.4f}  mean_purity={mean_purity:.4f}"
              f"  orphan_clusters={n_orphan}")

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    Xp = pca.fit_transform(Xs)
    results["pca_2d"] = {
        "explained_variance_ratio": [
            round(float(v), 4) for v in pca.explained_variance_ratio_
        ],
        "total_explained": round(float(pca.explained_variance_ratio_.sum()), 4),
        "x_range": [round(float(Xp[:, 0].min()), 4), round(float(Xp[:, 0].max()), 4)],
        "y_range": [round(float(Xp[:, 1].min()), 4), round(float(Xp[:, 1].max()), 4)],
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
