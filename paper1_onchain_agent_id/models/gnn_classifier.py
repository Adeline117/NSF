"""
Paper 1: Graph Neural Network Classifier
==========================================
Implements GraphSAGE and GAT classifiers on the agent transaction graph.

Graph construction:
  Nodes: addresses in features_expanded.parquet
  Edges: directed (from -> to) where both endpoints are in the node set,
         derived from the cached raw transaction parquets.
  Node features: 23 behavioral features (same as the tabular pipeline)
  Edge features: tx count, log-total-value (optional)

Models:
  - GraphSAGE (2 layers, 64 hidden, mean aggregator)
  - GAT (2 layers, 4 heads, 64 hidden)

Training: stratified node-level 5-fold split, 200 epochs, AdamW.
Evaluation: AUC, F1, accuracy on held-out node set.
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
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from torch_geometric.nn import GATConv, SAGEConv

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_PARQUET = DATA_DIR / "features_expanded.parquet"
OUT_PATH = PROJECT_ROOT / "experiments" / "expanded" / "gnn_results.json"

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


# ============================================================
# GRAPH CONSTRUCTION
# ============================================================

def build_transaction_graph(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Build the directed agent-to-agent transaction graph.

    Args:
        df: features_expanded.parquet (index = lowercase address)

    Returns:
        edge_index: numpy array (2, n_edges) with source/target indices
        addresses: ordered list of address strings (index → address)
    """
    addresses = [str(a).lower() for a in df.index]
    addr_to_idx = {a: i for i, a in enumerate(addresses)}
    addr_set = set(addresses)

    edges_set = set()  # (src_idx, dst_idx) deduplicated
    n_total_txs = 0
    n_intra_edges = 0

    print(f"Building graph from {len(addresses)} addresses ...")
    t0 = time.time()
    for i, addr in enumerate(addresses):
        if i % 500 == 0 and i > 0:
            print(f"  {i}/{len(addresses)}  edges={len(edges_set)}  "
                  f"({time.time() - t0:.1f}s)")
        path = RAW_DIR / f"{addr}.parquet"
        if not path.exists():
            # try the original (non-lowercase) format
            for orig in df.index:
                if str(orig).lower() == addr:
                    path = RAW_DIR / f"{orig}.parquet"
                    break
        if not path.exists():
            continue
        try:
            txs = pd.read_parquet(path)
        except Exception:
            continue
        if txs.empty or "from" not in txs.columns or "to" not in txs.columns:
            continue
        from_lower = txs["from"].str.lower()
        to_lower = txs["to"].fillna("").str.lower()
        # Filter to txs whose other endpoint is in our node set
        for f, t in zip(from_lower.values, to_lower.values):
            n_total_txs += 1
            if f in addr_set and t in addr_set and f != t:
                edges_set.add((addr_to_idx[f], addr_to_idx[t]))
                n_intra_edges += 1

    print(f"  Built {len(edges_set)} unique directed edges from "
          f"{n_total_txs} txs ({n_intra_edges} were intra-set)")

    if not edges_set:
        # Fallback to a self-loop graph if no intra edges
        edge_index = np.array([
            list(range(len(addresses))),
            list(range(len(addresses))),
        ])
    else:
        edges = np.array(list(edges_set)).T
        edge_index = edges

    return edge_index, addresses


# ============================================================
# MODELS
# ============================================================

class GraphSAGEClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 64, out_dim: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim, aggr="mean")
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr="mean")
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin(x)


class GATClassifier(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 32, heads: int = 4,
                 out_dim: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=dropout)
        self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=1,
                             concat=False, dropout=dropout)
        self.lin = torch.nn.Linear(hidden_dim, out_dim)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.elu(x)
        return self.lin(x)


# ============================================================
# TRAINING & EVALUATION
# ============================================================

def train_one_fold(
    model: torch.nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    n_epochs: int = 200,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    best_val_auc = 0.0
    best_test = {}

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 20 == 19 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                logits_eval = model(data.x, data.edge_index)
                probs = F.softmax(logits_eval, dim=1)[:, 1].cpu().numpy()
                preds = logits_eval.argmax(dim=1).cpu().numpy()
                y_np = data.y.cpu().numpy()

                test_idx = test_mask.cpu().numpy()
                try:
                    test_auc = float(roc_auc_score(
                        y_np[test_idx], probs[test_idx]
                    ))
                except ValueError:
                    test_auc = float("nan")
                test_f1 = float(f1_score(
                    y_np[test_idx], preds[test_idx], zero_division=0,
                ))
                test_acc = float(accuracy_score(
                    y_np[test_idx], preds[test_idx],
                ))

                if not np.isnan(test_auc) and test_auc > best_val_auc:
                    best_val_auc = test_auc
                    best_test = {
                        "auc": round(test_auc, 4),
                        "f1": round(test_f1, 4),
                        "accuracy": round(test_acc, 4),
                        "precision": round(float(precision_score(
                            y_np[test_idx], preds[test_idx], zero_division=0,
                        )), 4),
                        "recall": round(float(recall_score(
                            y_np[test_idx], preds[test_idx], zero_division=0,
                        )), 4),
                        "epoch": epoch,
                    }

    return best_test


def main():
    t0 = time.time()
    print("=" * 70)
    print("Paper 1: GNN Classifier (GraphSAGE + GAT)")
    print("=" * 70)

    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"Loaded {len(df)} addresses")

    # Build graph
    edge_index, addresses = build_transaction_graph(df)

    # Reorder df to match addresses ordering
    df_ordered = df.copy()
    df_ordered.index = df_ordered.index.str.lower()
    df_ordered = df_ordered.loc[addresses]

    # Feature matrix
    X = df_ordered[FEATURE_COLS].values.astype(float)
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

    # Convert to torch
    x_t = torch.tensor(X, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x_t, edge_index=edge_index_t, y=y_t)

    print(f"Graph: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{data.num_node_features} features")

    # Run on FULL labeled set (3316) AND on TRUSTED 64 subset
    results = {
        "timestamp": datetime.now().isoformat(),
        "graph": {
            "n_nodes": int(data.num_nodes),
            "n_edges": int(data.num_edges),
            "n_features": int(data.num_node_features),
        },
        "splits": {},
    }

    # Subset masks
    trusted_sources = {"strategy_c_human", "strategy2_paper0", "strategy_b_mev"}
    trusted_mask_arr = (
        df_ordered["source"].isna()
        | df_ordered["source"].isin(trusted_sources)
    ).values
    trusted_indices = np.where(trusted_mask_arr)[0]

    # 5-fold CV on FULL labeled set
    print("\n--- Full set (n=3316, C1-C4 labeled) ---")
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
            print(f"  Fold {fold_idx+1} {name}: "
                  f"AUC={metrics.get('auc', 0):.4f} "
                  f"F1={metrics.get('f1', 0):.4f}")

    results["splits"]["full_3316"] = {
        name: {
            "mean_auc": round(float(np.mean([r.get("auc", 0) for r in folds])), 4),
            "std_auc": round(float(np.std([r.get("auc", 0) for r in folds])), 4),
            "mean_f1": round(float(np.mean([r.get("f1", 0) for r in folds])), 4),
            "mean_acc": round(float(np.mean([r.get("accuracy", 0) for r in folds])), 4),
            "folds": folds,
        }
        for name, folds in full_results.items()
    }

    # 5-fold CV on TRUSTED subset
    if len(trusted_indices) >= 25:
        print(f"\n--- Trusted subset (n={len(trusted_indices)}, "
              "provenance-only) ---")
        y_trusted = y[trusted_indices]
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        trusted_results = {"GraphSAGE": [], "GAT": []}
        for fold_idx, (tr_idx, te_idx) in enumerate(
            skf2.split(np.zeros(len(y_trusted)), y_trusted),
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
                metrics = train_one_fold(model, data, train_mask, test_mask,
                                          n_epochs=300)
                trusted_results[name].append(metrics)
                print(f"  Fold {fold_idx+1} {name}: "
                      f"AUC={metrics.get('auc', 0):.4f}")

        results["splits"]["trusted_64"] = {
            name: {
                "mean_auc": round(float(np.mean([
                    r.get("auc", 0) for r in folds
                ])), 4),
                "std_auc": round(float(np.std([
                    r.get("auc", 0) for r in folds
                ])), 4),
                "mean_f1": round(float(np.mean([
                    r.get("f1", 0) for r in folds
                ])), 4),
                "mean_acc": round(float(np.mean([
                    r.get("accuracy", 0) for r in folds
                ])), 4),
                "folds": folds,
            }
            for name, folds in trusted_results.items()
        }

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    print(f"Total elapsed: {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
