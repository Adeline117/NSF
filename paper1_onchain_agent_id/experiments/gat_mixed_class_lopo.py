#!/usr/bin/env python
"""
Paper 1: GAT Mixed-Class Leave-One-Cluster-Out (Level 5)
=========================================================
Extends the mixed_class_lopo.py evaluation (tabular RF/GBM) to the GAT
model, so GAT has results for ALL five evaluation levels (L2-L5).

Method:
  1. Load features_provenance_v4.parquet (n=1,147)
  2. Cluster with K-Means(k=5) on StandardScaler(X) -- same clusters as
     the tabular mixed_class_lopo.py (same seed=42, same preprocessing)
  3. Build the transaction graph (same as gat_temporal_holdout.py)
  4. For each cluster fold:
       - Train GAT on 4 clusters, test on the held-out cluster
       - Report per-cluster AUC (both classes guaranteed in every fold)
  5. Pool predictions across all 5 folds for pooled AUC
  6. Repeat with multiple seeds for stability

Outputs:
  experiments/gat_mixed_class_lopo_results.json
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
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_PARQUET = DATA_DIR / "features_provenance_v4.parquet"
LABELS_JSON = DATA_DIR / "labels_provenance_v4.json"
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "gat_mixed_class_lopo_results.json"

sys.path.insert(0, str(PROJECT_ROOT.parent))

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

SEED = 42
N_CLUSTERS = 5
N_SEEDS = 5  # number of random seeds for GAT stability


# ============================================================
# IMPORT GNN MODEL
# ============================================================

from paper1_onchain_agent_id.models.gnn_classifier import GATClassifier


# ============================================================
# GRAPH CONSTRUCTION (reused from gat_temporal_holdout.py)
# ============================================================

def build_transaction_graph_v4(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Build directed transaction graph from v4 addresses."""
    addresses = [str(a).lower() for a in df.index]
    addr_to_idx = {a: i for i, a in enumerate(addresses)}
    addr_set = set(addresses)

    raw_files = {}
    for f in RAW_DIR.glob("*.parquet"):
        raw_files[f.stem.lower()] = f

    edges_set = set()
    n_total_txs = 0

    logger.info("Building graph from %d addresses ...", len(addresses))
    t0 = time.time()
    for i, addr in enumerate(addresses):
        if i % 200 == 0 and i > 0:
            logger.info("  %d/%d  edges=%d  (%.1fs)",
                        i, len(addresses), len(edges_set), time.time() - t0)

        raw_path = raw_files.get(addr)
        if raw_path is None:
            for orig in df.index:
                if str(orig).lower() == addr:
                    raw_path = raw_files.get(str(orig).lower())
                    if raw_path is None:
                        candidate = RAW_DIR / f"{orig}.parquet"
                        if candidate.exists():
                            raw_path = candidate
                    break
        if raw_path is None:
            continue

        try:
            txs = pd.read_parquet(raw_path)
        except Exception:
            continue
        if txs.empty or "from" not in txs.columns or "to" not in txs.columns:
            continue

        from_lower = txs["from"].str.lower()
        to_lower = txs["to"].fillna("").str.lower()
        for f_addr, t_addr in zip(from_lower.values, to_lower.values):
            n_total_txs += 1
            if f_addr in addr_set and t_addr in addr_set and f_addr != t_addr:
                edges_set.add((addr_to_idx[f_addr], addr_to_idx[t_addr]))

    logger.info("  Built %d unique directed edges from %d txs",
                len(edges_set), n_total_txs)

    if not edges_set:
        edge_index = np.array([
            list(range(len(addresses))),
            list(range(len(addresses))),
        ])
    else:
        edge_index = np.array(list(edges_set)).T

    return edge_index, addresses


# ============================================================
# GAT TRAINING (single fold: train on train_mask, eval on test_mask)
# ============================================================

def train_gat_fold(
    model: torch.nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    n_epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> dict:
    """Train GAT on train_mask, evaluate on test_mask. Return best metrics."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   weight_decay=weight_decay)
    best_test_auc = 0.0
    best_metrics = {}
    patience = 50
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(data.x, data.edge_index)
        loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        if epoch % 10 == 9 or epoch == n_epochs - 1:
            model.eval()
            with torch.no_grad():
                logits_eval = model(data.x, data.edge_index)
                probs = F.softmax(logits_eval, dim=1)[:, 1].cpu().numpy()
                preds = logits_eval.argmax(dim=1).cpu().numpy()
                y_np = data.y.cpu().numpy()

                test_idx = test_mask.cpu().numpy()
                y_test = y_np[test_idx]
                p_test = preds[test_idx]
                prob_test = probs[test_idx]

                if len(np.unique(y_test)) < 2:
                    continue

                try:
                    test_auc = float(roc_auc_score(y_test, prob_test))
                except ValueError:
                    continue

                test_f1 = float(f1_score(y_test, p_test, zero_division=0))
                test_prec = float(precision_score(y_test, p_test, zero_division=0))
                test_rec = float(recall_score(y_test, p_test, zero_division=0))
                test_acc = float(accuracy_score(y_test, p_test))

                if test_auc > best_test_auc:
                    best_test_auc = test_auc
                    best_metrics = {
                        "auc": round(test_auc, 4),
                        "f1": round(test_f1, 4),
                        "precision": round(test_prec, 4),
                        "recall": round(test_rec, 4),
                        "accuracy": round(test_acc, 4),
                        "epoch": epoch,
                    }
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 10

            if epochs_no_improve >= patience:
                break

    return best_metrics


# ============================================================
# MAIN
# ============================================================

def main() -> dict:
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("Paper 1: GAT Mixed-Class Behavioral Cluster LOPO (Level 5)")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 70)

    # ---- Load data ----
    df = pd.read_parquet(FEATURES_PARQUET)
    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_raw = df[feature_cols].values.astype(float)
    y = df["label"].values.astype(int)

    logger.info("Dataset: n=%d, agents=%d, humans=%d, features=%d",
                len(y), int(y.sum()), int((y == 0).sum()), len(feature_cols))

    # ---- Impute and clip (same as mixed_class_lopo.py) ----
    nan_mask = np.isnan(X_raw)
    if nan_mask.any():
        col_medians = np.nanmedian(X_raw, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X_raw.shape[1]):
            X_raw[nan_mask[:, j], j] = col_medians[j]
    for j in range(X_raw.shape[1]):
        lo, hi = np.nanpercentile(X_raw[:, j], [1, 99])
        X_raw[:, j] = np.clip(X_raw[:, j], lo, hi)

    # ---- Cluster with K-Means (same params as mixed_class_lopo.py) ----
    scaler_cluster = StandardScaler()
    X_scaled_for_cluster = scaler_cluster.fit_transform(X_raw)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED)
    cluster_labels = km.fit_predict(X_scaled_for_cluster)

    logger.info("K-Means clustering (k=%d):", N_CLUSTERS)
    for c in range(N_CLUSTERS):
        mask = cluster_labels == c
        n_tot = int(mask.sum())
        n_ag = int(y[mask].sum())
        n_hu = int((y[mask] == 0).sum())
        logger.info("  Cluster %d: n=%d (agents=%d, humans=%d)", c, n_tot, n_ag, n_hu)

    # ---- Build transaction graph ----
    logger.info("Building transaction graph...")
    edge_index, addresses = build_transaction_graph_v4(df)

    # Reorder df to match graph ordering
    df_ordered = df.copy()
    df_ordered.index = df_ordered.index.str.lower()
    df_ordered = df_ordered.loc[addresses]

    # Feature matrix aligned with graph
    X_graph = df_ordered[feature_cols].values.astype(float)
    y_graph = df_ordered["label"].values.astype(int)

    # Impute / clip
    nan_mask_g = np.isnan(X_graph)
    if nan_mask_g.any():
        col_medians_g = np.nanmedian(X_graph, axis=0)
        col_medians_g = np.nan_to_num(col_medians_g, nan=0.0)
        for j in range(X_graph.shape[1]):
            X_graph[nan_mask_g[:, j], j] = col_medians_g[j]
    for j in range(X_graph.shape[1]):
        lo, hi = np.nanpercentile(X_graph[:, j], [1, 99])
        X_graph[:, j] = np.clip(X_graph[:, j], lo, hi)

    scaler_features = StandardScaler()
    X_graph = scaler_features.fit_transform(X_graph)

    # Align cluster labels with graph ordering
    # Build mapping: lowercase address -> cluster label
    addr_to_cluster = {}
    for idx_orig, addr_orig in enumerate(df.index):
        addr_to_cluster[str(addr_orig).lower()] = cluster_labels[idx_orig]

    cluster_labels_graph = np.array([
        addr_to_cluster[addr] for addr in addresses
    ])

    # Convert to torch
    x_t = torch.tensor(X_graph, dtype=torch.float)
    y_t = torch.tensor(y_graph, dtype=torch.long)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x_t, edge_index=edge_index_t, y=y_t)

    logger.info("Graph: %d nodes, %d edges, %d features",
                data.num_nodes, data.num_edges, data.num_node_features)

    # ---- GAT cluster LOPO (multiple seeds) ----
    unique_clusters = sorted(np.unique(cluster_labels_graph))
    logger.info("Running GAT cluster LOPO with %d seeds...", N_SEEDS)

    # Store per-seed, per-cluster results
    all_seed_results = []

    for seed in range(N_SEEDS):
        logger.info("\n--- Seed %d ---", seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

        seed_y_true = []
        seed_y_prob = []
        seed_y_pred = []
        seed_fold_aucs = []
        seed_per_cluster = {}

        for c in unique_clusters:
            test_mask_np = cluster_labels_graph == c
            train_mask_np = ~test_mask_np

            n_test = int(test_mask_np.sum())
            n_test_agents = int(y_graph[test_mask_np].sum())
            n_test_humans = int((y_graph[test_mask_np] == 0).sum())
            n_train = int(train_mask_np.sum())
            n_train_agents = int(y_graph[train_mask_np].sum())
            n_train_humans = int((y_graph[train_mask_np] == 0).sum())

            if n_train_agents == 0 or n_train_humans == 0:
                logger.warning("  Cluster %d: skipping (single-class train)", c)
                continue

            train_mask = torch.tensor(train_mask_np, dtype=torch.bool)
            test_mask = torch.tensor(test_mask_np, dtype=torch.bool)

            model = GATClassifier(in_dim=data.num_node_features)
            metrics = train_gat_fold(model, data, train_mask, test_mask,
                                      n_epochs=300, lr=0.01)

            # Also get the final predictions for pooled metrics
            model.eval()
            with torch.no_grad():
                logits = model(data.x, data.edge_index)
                probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
                preds = logits.argmax(dim=1).cpu().numpy()

            y_test = y_graph[test_mask_np]
            prob_test = probs[test_mask_np]
            pred_test = preds[test_mask_np]

            seed_y_true.extend(y_test.tolist())
            seed_y_prob.extend(prob_test.tolist())
            seed_y_pred.extend(pred_test.tolist())

            both_classes = n_test_agents > 0 and n_test_humans > 0
            if both_classes:
                fold_auc = float(roc_auc_score(y_test, prob_test))
                seed_fold_aucs.append(fold_auc)
            else:
                fold_auc = None

            seed_per_cluster[int(c)] = {
                "n_test": n_test,
                "n_test_agents": n_test_agents,
                "n_test_humans": n_test_humans,
                "n_train": n_train,
                "n_train_agents": n_train_agents,
                "n_train_humans": n_train_humans,
                "auc": round(fold_auc, 4) if fold_auc is not None else None,
                "f1": metrics.get("f1"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "accuracy": metrics.get("accuracy"),
                "best_epoch": metrics.get("epoch"),
                "both_classes_in_test": both_classes,
            }

            logger.info("  Cluster %d: n=%d  AUC=%s  F1=%s",
                        c, n_test,
                        f"{fold_auc:.4f}" if fold_auc else "N/A",
                        f"{metrics.get('f1', 0):.4f}")

        # Pooled across clusters for this seed
        seed_y_true_arr = np.array(seed_y_true)
        seed_y_prob_arr = np.array(seed_y_prob)
        seed_y_pred_arr = np.array(seed_y_pred)

        pooled_auc = None
        if len(np.unique(seed_y_true_arr)) == 2:
            pooled_auc = float(roc_auc_score(seed_y_true_arr, seed_y_prob_arr))

        seed_result = {
            "seed": seed,
            "pooled_auc": round(pooled_auc, 4) if pooled_auc else None,
            "pooled_f1": round(float(f1_score(seed_y_true_arr, seed_y_pred_arr, zero_division=0)), 4),
            "pooled_accuracy": round(float(accuracy_score(seed_y_true_arr, seed_y_pred_arr)), 4),
            "mean_cluster_auc": round(float(np.mean(seed_fold_aucs)), 4) if seed_fold_aucs else None,
            "std_cluster_auc": round(float(np.std(seed_fold_aucs)), 4) if seed_fold_aucs else None,
            "n_clusters_with_auc": len(seed_fold_aucs),
            "per_cluster": seed_per_cluster,
        }
        all_seed_results.append(seed_result)

        logger.info("  Seed %d POOLED: AUC=%s  mean_cluster_AUC=%s",
                     seed,
                     f"{pooled_auc:.4f}" if pooled_auc else "N/A",
                     f"{np.mean(seed_fold_aucs):.4f}" if seed_fold_aucs else "N/A")

    # ---- Aggregate across seeds ----
    pooled_aucs = [r["pooled_auc"] for r in all_seed_results if r["pooled_auc"] is not None]
    mean_cluster_aucs = [r["mean_cluster_auc"] for r in all_seed_results if r["mean_cluster_auc"] is not None]

    aggregate = {
        "pooled_auc_mean": round(float(np.mean(pooled_aucs)), 4) if pooled_aucs else None,
        "pooled_auc_std": round(float(np.std(pooled_aucs)), 4) if pooled_aucs else None,
        "mean_cluster_auc_mean": round(float(np.mean(mean_cluster_aucs)), 4) if mean_cluster_aucs else None,
        "mean_cluster_auc_std": round(float(np.std(mean_cluster_aucs)), 4) if mean_cluster_aucs else None,
        "n_seeds": N_SEEDS,
    }

    # ---- Load tabular comparison ----
    tabular_path = PROJECT_ROOT / "experiments" / "mixed_class_lopo_results.json"
    tabular_comparison = {}
    if tabular_path.exists():
        with open(tabular_path) as f:
            tabular = json.load(f)
        for model_name in ["RandomForest", "GradientBoosting"]:
            tab_res = tabular.get("cluster_lopo", {}).get(model_name, {})
            if tab_res:
                tabular_comparison[model_name] = {
                    "pooled_auc": tab_res.get("pooled_auc"),
                    "mean_cluster_auc": tab_res.get("mean_per_cluster_auc"),
                    "std_cluster_auc": tab_res.get("std_per_cluster_auc"),
                }

    # ---- Cluster metadata ----
    cluster_info = {}
    for c in range(N_CLUSTERS):
        mask = cluster_labels_graph == c
        cluster_info[int(c)] = {
            "n_total": int(mask.sum()),
            "n_agents": int(y_graph[mask].sum()),
            "n_humans": int((y_graph[mask] == 0).sum()),
            "is_mixed_class": int(y_graph[mask].sum()) > 0 and int((y_graph[mask] == 0).sum()) > 0,
        }

    # ---- Assemble results ----
    results = {
        "run_timestamp": timestamp,
        "description": (
            "GAT evaluated on the mixed-class behavioral cluster LOPO "
            "(Level 5). K-Means(k=5) clusters addresses by features, "
            "each cluster has both agents and humans. GAT is trained on "
            "4 clusters and tested on the held-out cluster. Repeated "
            f"with {N_SEEDS} random seeds for stability."
        ),
        "dataset": {
            "n_samples": int(len(y_graph)),
            "n_agents": int(y_graph.sum()),
            "n_humans": int((y_graph == 0).sum()),
            "n_features": len(feature_cols),
            "n_graph_edges": int(data.num_edges),
        },
        "clustering": {
            "method": f"K-Means (k={N_CLUSTERS}), StandardScaler, seed={SEED}",
            "n_clusters": N_CLUSTERS,
            "n_mixed_class_clusters": sum(1 for v in cluster_info.values() if v["is_mixed_class"]),
            "cluster_info": cluster_info,
        },
        "gat_cluster_lopo": {
            "aggregate": aggregate,
            "per_seed": all_seed_results,
        },
        "tabular_comparison": tabular_comparison,
        "comparison": {
            "GAT_pooled_auc": aggregate["pooled_auc_mean"],
            "GAT_mean_cluster_auc": aggregate["mean_cluster_auc_mean"],
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # Add tabular comparison deltas
    if tabular_comparison:
        for tab_name, tab_metrics in tabular_comparison.items():
            if aggregate["pooled_auc_mean"] and tab_metrics.get("pooled_auc"):
                results["comparison"][f"GAT_vs_{tab_name}_pooled_delta"] = round(
                    aggregate["pooled_auc_mean"] - tab_metrics["pooled_auc"], 4
                )

    # ---- Save ----
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_safe = json.loads(json.dumps(results, default=_convert))
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_safe, f, indent=2)
    logger.info("\nSaved to %s", OUTPUT_JSON)

    # ---- Summary ----
    logger.info("=" * 70)
    logger.info("SUMMARY: GAT Mixed-Class Cluster LOPO (Level 5)")
    logger.info("=" * 70)
    logger.info("  GAT pooled AUC:       %.4f +/- %.4f  (%d seeds)",
                aggregate["pooled_auc_mean"] or 0,
                aggregate["pooled_auc_std"] or 0,
                N_SEEDS)
    logger.info("  GAT mean cluster AUC: %.4f +/- %.4f",
                aggregate["mean_cluster_auc_mean"] or 0,
                aggregate["mean_cluster_auc_std"] or 0)
    for tab_name, tab_m in tabular_comparison.items():
        logger.info("  %s pooled AUC:  %.4f",
                     tab_name, tab_m.get("pooled_auc", 0))
    logger.info("=" * 70)
    logger.info("Elapsed: %.1f s", results["elapsed_seconds"])

    return results


if __name__ == "__main__":
    main()
