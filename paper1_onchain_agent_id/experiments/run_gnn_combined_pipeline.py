"""
Paper 1: GNN on Combined Features (P1 + P3)
==============================================
Runs the GraphSAGE and GAT classifiers on the 31-feature combined
matrix (Paper 1's 23 + Paper 3's 8 AI features).

Compares against the previous gnn_results.json (P1-only features).
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

from paper1_onchain_agent_id.models.gnn_classifier import (
    GraphSAGEClassifier, GATClassifier, build_transaction_graph,
    train_one_fold,
)

COMBINED_PARQUET = PROJECT_ROOT / "data" / "features_combined.parquet"
OUT_PATH = PROJECT_ROOT / "experiments" / "expanded" / "gnn_combined_results.json"

P1_FEATURES = [
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
P3_FEATURES = [
    "gas_price_precision", "hour_entropy", "behavioral_consistency",
    "action_sequence_perplexity", "error_recovery_pattern",
    "response_latency_variance", "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]


def main():
    t0 = time.time()
    print("=" * 70)
    print("Paper 1: GNN on Combined 31-Feature Set")
    print("=" * 70)

    df = pd.read_parquet(COMBINED_PARQUET)
    df.index = df.index.astype(str).str.lower()
    print(f"Loaded {len(df)} addresses")

    # Build graph
    edge_index, addresses = build_transaction_graph(df)

    # Reorder df to graph
    df_ordered = df.loc[addresses]

    feat_set = P1_FEATURES + P3_FEATURES
    X = df_ordered[feat_set].values.astype(float)
    y = df_ordered["label"].values.astype(int)

    # Impute / clip
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    x_t = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x_t, edge_index=edge_index_t, y=y_t)

    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_node_features} features")

    results = {
        "timestamp": datetime.now().isoformat(),
        "feature_set": "combined_31",
        "n_features": int(data.num_node_features),
        "splits": {},
    }

    # Trusted subset
    trusted_sources = {"strategy_c_human", "strategy2_paper0", "strategy_b_mev"}
    trusted_mask_arr = (
        df_ordered["source"].isna() | df_ordered["source"].isin(trusted_sources)
    ).values
    trusted_indices = np.where(trusted_mask_arr)[0]

    # Full set 5-fold
    print("\n--- Full set (n=3316, combined-31) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    full_results = {"GraphSAGE": [], "GAT": []}
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
        train_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)
        train_mask[tr_idx] = True
        test_mask[te_idx] = True

        for name, ModelCls in [
            ("GraphSAGE", GraphSAGEClassifier),
            ("GAT", GATClassifier),
        ]:
            model = ModelCls(in_dim=data.num_node_features)
            metrics = train_one_fold(model, data, train_mask, test_mask)
            full_results[name].append(metrics)
            print(f"  Fold {fold_idx+1} {name}: AUC={metrics.get('auc', 0):.4f}")

    results["splits"]["full_3316"] = {
        name: {
            "mean_auc": round(float(np.mean([r.get("auc", 0) for r in folds])), 4),
            "std_auc": round(float(np.std([r.get("auc", 0) for r in folds])), 4),
            "mean_f1": round(float(np.mean([r.get("f1", 0) for r in folds])), 4),
        }
        for name, folds in full_results.items()
    }

    # Trusted 64
    if len(trusted_indices) >= 25:
        print(f"\n--- Trusted (n={len(trusted_indices)}) ---")
        y_trusted = y[trusted_indices]
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        trusted_results = {"GraphSAGE": [], "GAT": []}
        for fold_idx, (tr_idx, te_idx) in enumerate(
            skf2.split(np.zeros(len(y_trusted)), y_trusted)
        ):
            train_mask = torch.zeros(len(y), dtype=torch.bool)
            test_mask = torch.zeros(len(y), dtype=torch.bool)
            train_mask[trusted_indices[tr_idx]] = True
            test_mask[trusted_indices[te_idx]] = True

            for name, ModelCls in [
                ("GraphSAGE", GraphSAGEClassifier),
                ("GAT", GATClassifier),
            ]:
                model = ModelCls(in_dim=data.num_node_features)
                metrics = train_one_fold(model, data, train_mask, test_mask, n_epochs=300)
                trusted_results[name].append(metrics)
                print(f"  Fold {fold_idx+1} {name}: AUC={metrics.get('auc', 0):.4f}")

        results["splits"]["trusted_64"] = {
            name: {
                "mean_auc": round(float(np.mean([r.get("auc", 0) for r in folds])), 4),
                "std_auc": round(float(np.std([r.get("auc", 0) for r in folds])), 4),
                "mean_f1": round(float(np.mean([r.get("f1", 0) for r in folds])), 4),
            }
            for name, folds in trusted_results.items()
        }

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {OUT_PATH}")
    print(f"Elapsed: {results['elapsed_seconds']:.1f}s")

    # Compare to P1-only GNN
    p1_only_path = PROJECT_ROOT / "experiments" / "expanded" / "gnn_results.json"
    if p1_only_path.exists():
        with open(p1_only_path) as f:
            p1_only = json.load(f)
        print("\n=== Comparison: P1-only (23) vs Combined (31) ===")
        for split in ["full_3316", "trusted_64"]:
            print(f"\n{split}:")
            for model in ["GraphSAGE", "GAT"]:
                p1 = p1_only.get("splits", {}).get(split, {}).get(model, {}).get("mean_auc")
                comb = results["splits"].get(split, {}).get(model, {}).get("mean_auc")
                if p1 is not None and comb is not None:
                    print(f"  {model:<10} P1={p1:.4f}  Combined={comb:.4f}  delta={comb-p1:+.4f}")


if __name__ == "__main__":
    main()
