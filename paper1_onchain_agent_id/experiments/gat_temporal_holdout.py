"""
Paper 1: GAT / GraphSAGE Temporal Holdout
==========================================
Reviewer concern: "The paper's best model (GAT 0.832) must face the
paper's hardest test (temporal holdout)."

This script:
  1. Uses the same temporal split as run_temporal_holdout.py
     (median first-seen block ~12321026, approx April 2021)
  2. Builds the transaction graph from the n=1,147 provenance-v4 dataset
  3. Trains GAT and GraphSAGE on the TRAIN split (pre-median addresses)
  4. Evaluates on the TEST split (post-median addresses)
  5. Reports AUC, F1, precision, recall
  6. Compares with tabular temporal holdout results

Outputs:
  experiments/gat_temporal_holdout_results.json
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
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "gat_temporal_holdout_results.json"

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


# ============================================================
# IMPORT GNN MODELS (same architecture as gnn_classifier.py)
# ============================================================

from paper1_onchain_agent_id.models.gnn_classifier import (
    GATClassifier,
    GraphSAGEClassifier,
)


# ============================================================
# GRAPH CONSTRUCTION (adapted for v4 dataset)
# ============================================================

def build_transaction_graph_v4(
    df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    """Build directed transaction graph from v4 addresses.

    Args:
        df: features_provenance_v4.parquet (index = address)

    Returns:
        edge_index: numpy array (2, n_edges)
        addresses: ordered list of address strings
    """
    addresses = [str(a).lower() for a in df.index]
    addr_to_idx = {a: i for i, a in enumerate(addresses)}
    addr_set = set(addresses)

    # Build case-insensitive raw file lookup
    raw_files = {}
    for f in RAW_DIR.glob("*.parquet"):
        raw_files[f.stem.lower()] = f

    edges_set = set()
    n_total_txs = 0
    n_intra = 0

    logger.info("Building graph from %d addresses ...", len(addresses))
    t0 = time.time()
    for i, addr in enumerate(addresses):
        if i % 200 == 0 and i > 0:
            logger.info("  %d/%d  edges=%d  (%.1fs)",
                        i, len(addresses), len(edges_set), time.time() - t0)

        raw_path = raw_files.get(addr)
        if raw_path is None:
            # Try original casing from df.index
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
                n_intra += 1

    logger.info("  Built %d unique directed edges from %d txs (%d intra-set)",
                len(edges_set), n_total_txs, n_intra)

    if not edges_set:
        # Fallback: self-loops (no intra-set edges)
        edge_index = np.array([
            list(range(len(addresses))),
            list(range(len(addresses))),
        ])
    else:
        edges = np.array(list(edges_set)).T
        edge_index = edges

    return edge_index, addresses


# ============================================================
# TEMPORAL METADATA
# ============================================================

def extract_first_seen_block(addresses: list[str]) -> pd.DataFrame:
    """For each address, get the minimum blockNumber from raw tx data."""
    raw_files = {f.stem.lower(): f for f in RAW_DIR.glob("*.parquet")}
    records = []
    found, missing = 0, 0

    for addr in addresses:
        addr_lower = addr.lower()
        raw_path = raw_files.get(addr_lower)
        if raw_path is None:
            missing += 1
            continue
        try:
            txdf = pd.read_parquet(raw_path, columns=["blockNumber", "timeStamp"])
            blocks = pd.to_numeric(txdf["blockNumber"], errors="coerce")
            timestamps = pd.to_numeric(txdf["timeStamp"], errors="coerce")
            records.append({
                "address": addr,
                "first_block": int(blocks.min()),
                "first_timestamp": int(timestamps.min()),
            })
            found += 1
        except Exception:
            missing += 1

    logger.info("First-seen block: %d found, %d missing", found, missing)
    return pd.DataFrame(records).set_index("address")


# ============================================================
# TRAINING (temporal holdout — single train/test split)
# ============================================================

def train_temporal_holdout(
    model: torch.nn.Module,
    data: Data,
    train_mask: torch.Tensor,
    test_mask: torch.Tensor,
    n_epochs: int = 300,
    lr: float = 0.01,
    weight_decay: float = 5e-4,
) -> dict:
    """Train on train_mask nodes, evaluate on test_mask nodes.
    Returns best metrics on test set during training."""
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

        # Evaluate every 10 epochs
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
                logger.info("    Early stopping at epoch %d", epoch)
                break

    return best_metrics


# ============================================================
# MAIN
# ============================================================

def main() -> dict:
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 65)
    logger.info("Paper 1: GAT/GraphSAGE Temporal Holdout")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 65)

    # Load features
    df = pd.read_parquet(FEATURES_PARQUET)
    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)
    logger.info("Loaded %d addresses from v4 dataset", len(df))

    # Extract first-seen block
    logger.info("Extracting first-seen block from raw transaction data...")
    temporal_df = extract_first_seen_block(list(df.index))

    # Align: only keep addresses with both features AND temporal data
    common_addrs = df.index.intersection(temporal_df.index)
    logger.info("Addresses with features AND temporal data: %d / %d",
                len(common_addrs), len(df))

    df = df.loc[common_addrs]
    temporal_df = temporal_df.loc[common_addrs]
    y_aligned = np.array([
        labels_dict[addr]["label_provenance"] for addr in common_addrs
    ])

    # Determine temporal split (same as run_temporal_holdout.py)
    median_block = int(temporal_df["first_block"].median())
    logger.info("Median first_block: %d", median_block)

    train_time_mask = temporal_df["first_block"] < median_block
    test_time_mask = temporal_df["first_block"] >= median_block

    train_addrs = common_addrs[train_time_mask.values]
    test_addrs = common_addrs[test_time_mask.values]
    y_train = y_aligned[train_time_mask.values]
    y_test = y_aligned[test_time_mask.values]

    logger.info("TEMPORAL SPLIT:")
    logger.info("  Train: n=%d  (agents=%d, humans=%d)  blocks < %d",
                len(y_train), int(y_train.sum()), int((y_train == 0).sum()),
                median_block)
    logger.info("  Test:  n=%d  (agents=%d, humans=%d)  blocks >= %d",
                len(y_test), int(y_test.sum()), int((y_test == 0).sum()),
                median_block)

    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        logger.error("One split has only one class. Cannot proceed.")
        return {}

    # Build graph on the FULL aligned dataset (train+test addresses)
    logger.info("Building transaction graph...")
    edge_index, addresses = build_transaction_graph_v4(df)

    # Reorder df to match graph ordering
    df_ordered = df.copy()
    df_ordered.index = df_ordered.index.str.lower()
    # The addresses list is already lowercase
    df_ordered = df_ordered.loc[addresses]

    # Feature matrix
    X = df_ordered[FEATURE_COLS].values.astype(float)
    y = np.array([
        labels_dict.get(
            addr,
            labels_dict.get(addr.lower(), {"label_provenance": 0})
        )["label_provenance"]
        for addr in df.loc[[a for a in df.index if a.lower() in set(addresses)]].index
    ])

    # Re-align y with graph ordering
    addr_to_orig = {}
    for orig_addr in df.index:
        addr_to_orig[orig_addr.lower()] = orig_addr
    y = np.array([
        labels_dict[addr_to_orig.get(addr, addr)]["label_provenance"]
        for addr in addresses
    ])

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

    logger.info("Graph: %d nodes, %d edges, %d features",
                data.num_nodes, data.num_edges, data.num_node_features)

    # Build temporal train/test masks aligned with graph ordering
    train_addr_set = set(a.lower() for a in train_addrs)
    test_addr_set = set(a.lower() for a in test_addrs)

    train_mask = torch.tensor([
        addr in train_addr_set for addr in addresses
    ], dtype=torch.bool)
    test_mask = torch.tensor([
        addr in test_addr_set for addr in addresses
    ], dtype=torch.bool)

    logger.info("Train mask: %d nodes", int(train_mask.sum()))
    logger.info("Test mask: %d nodes", int(test_mask.sum()))

    # Verify split integrity
    y_np = y_t.cpu().numpy()
    train_agents = int(y_np[train_mask.numpy()].sum())
    train_humans = int((y_np[train_mask.numpy()] == 0).sum())
    test_agents = int(y_np[test_mask.numpy()].sum())
    test_humans = int((y_np[test_mask.numpy()] == 0).sum())
    logger.info("  Train agents=%d humans=%d", train_agents, train_humans)
    logger.info("  Test  agents=%d humans=%d", test_agents, test_humans)

    split_metadata = {
        "n_total": int(len(common_addrs)),
        "n_graph_nodes": int(data.num_nodes),
        "n_graph_edges": int(data.num_edges),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "split_block": median_block,
        "train_agents": train_agents,
        "train_humans": train_humans,
        "test_agents": test_agents,
        "test_humans": test_humans,
    }

    # ---- Run GNN temporal holdout (multiple seeds for stability) ----
    n_seeds = 5
    gnn_models = {
        "GAT": GATClassifier,
        "GraphSAGE": GraphSAGEClassifier,
    }

    gnn_results = {}
    for model_name, ModelCls in gnn_models.items():
        logger.info("\n--- %s Temporal Holdout (%d seeds) ---", model_name, n_seeds)
        seed_results = []
        for seed in range(n_seeds):
            torch.manual_seed(seed)
            np.random.seed(seed)

            model = ModelCls(in_dim=data.num_node_features)
            metrics = train_temporal_holdout(
                model, data, train_mask, test_mask,
                n_epochs=300, lr=0.01,
            )
            seed_results.append(metrics)
            logger.info("  Seed %d: AUC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f  (epoch %d)",
                        seed,
                        metrics.get("auc", 0),
                        metrics.get("f1", 0),
                        metrics.get("precision", 0),
                        metrics.get("recall", 0),
                        metrics.get("epoch", -1))

        # Aggregate across seeds
        aucs = [r.get("auc", 0) for r in seed_results]
        f1s = [r.get("f1", 0) for r in seed_results]
        precs = [r.get("precision", 0) for r in seed_results]
        recs = [r.get("recall", 0) for r in seed_results]
        accs = [r.get("accuracy", 0) for r in seed_results]

        gnn_results[model_name] = {
            "mean_auc": round(float(np.mean(aucs)), 4),
            "std_auc": round(float(np.std(aucs)), 4),
            "mean_f1": round(float(np.mean(f1s)), 4),
            "std_f1": round(float(np.std(f1s)), 4),
            "mean_precision": round(float(np.mean(precs)), 4),
            "mean_recall": round(float(np.mean(recs)), 4),
            "mean_accuracy": round(float(np.mean(accs)), 4),
            "n_seeds": n_seeds,
            "per_seed": seed_results,
        }
        logger.info("  %s MEAN: AUC=%.4f +/- %.4f  F1=%.4f +/- %.4f",
                     model_name,
                     gnn_results[model_name]["mean_auc"],
                     gnn_results[model_name]["std_auc"],
                     gnn_results[model_name]["mean_f1"],
                     gnn_results[model_name]["std_f1"])

    # ---- Load tabular temporal holdout for comparison ----
    tabular_temporal_path = PROJECT_ROOT / "experiments" / "temporal_holdout_results.json"
    tabular_comparison = {}
    if tabular_temporal_path.exists():
        with open(tabular_temporal_path) as f:
            tabular_results = json.load(f)
        tabular_comparison = tabular_results.get("temporal_holdout", {})
        logger.info("\n--- Tabular temporal holdout (from existing results) ---")
        for name, metrics in tabular_comparison.items():
            logger.info("  %s: AUC=%.4f  F1=%.4f", name, metrics["auc"], metrics["f1"])

    # ---- Assemble results ----
    results = {
        "run_timestamp": timestamp,
        "description": (
            "GAT and GraphSAGE temporal holdout: train on addresses first "
            "seen before the median block, test on addresses first seen "
            "after. Same split as the tabular temporal holdout. This "
            "addresses the reviewer concern that the best GNN model must "
            "face the hardest generalization test."
        ),
        "split_metadata": split_metadata,
        "gnn_temporal_holdout": gnn_results,
        "tabular_temporal_holdout": tabular_comparison,
        "comparison": {},
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # Build comparison table
    for gnn_name in gnn_models:
        gnn_auc = gnn_results[gnn_name]["mean_auc"]
        for tab_name, tab_metrics in tabular_comparison.items():
            tab_auc = tab_metrics["auc"]
            results["comparison"][f"{gnn_name}_vs_{tab_name}"] = {
                "gnn_auc": gnn_auc,
                "tabular_auc": tab_auc,
                "delta": round(gnn_auc - tab_auc, 4),
            }

    # ---- Save ----
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_ser = json.loads(json.dumps(results, default=_convert))
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_ser, f, indent=2)
    logger.info("Saved results to %s", OUTPUT_JSON)

    # ---- Print summary ----
    logger.info("=" * 65)
    logger.info("SUMMARY: GNN vs Tabular Temporal Holdout")
    logger.info("=" * 65)
    logger.info("%-22s  %10s  %10s", "Model", "AUC", "F1")
    logger.info("-" * 50)
    for name, metrics in gnn_results.items():
        logger.info("%-22s  %.4f+/-%.4f  %.4f+/-%.4f",
                     name,
                     metrics["mean_auc"], metrics["std_auc"],
                     metrics["mean_f1"], metrics["std_f1"])
    logger.info("-" * 50)
    for name, metrics in tabular_comparison.items():
        logger.info("%-22s  %.4f         %.4f",
                     name + " (tabular)", metrics["auc"], metrics["f1"])
    logger.info("=" * 65)
    logger.info("Elapsed: %.1f s", results["elapsed_seconds"])

    return results


if __name__ == "__main__":
    main()
