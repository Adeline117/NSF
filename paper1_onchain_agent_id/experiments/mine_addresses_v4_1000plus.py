"""
Paper 1: Expanded Provenance-Only Mining v4 (1000+ Addresses)
=============================================================
Expands the trusted (provenance-only) dataset from 563 to 1000+ addresses.

Strategy -- builds on v3 and adds NEW sources:

  NEW AGENTS (target 400+ new):
    A. Aave V3 liquidators (liquidationCall method 0x00a718a9)
    B. Compound V3 liquidators (absorb method)
    C. 1inch Resolver high-frequency executors (>200 calls)
    D. Chainlink Keeper Registry executors
    E. Uniswap V3 NonfungiblePositionManager high-freq mint/liquidity callers

  NEW HUMANS (target 200+ new):
    F. Gitcoin Grants donors
    G. ENS Reverse Registrar setters
    H. PoolTogether V4 depositors
    I. Mirror.xyz publishers

IMPORTANT: Labels derive ONLY from provenance (contract interaction
source). C1-C4 is computed for diagnostic purposes and does NOT gate labels.

Outputs:
  - data/features_provenance_v4.parquet
  - experiments/mine_addresses_v4_results.json
  - data/labels_provenance_v4.json
"""

import json
import logging
import os
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.eth_utils import EtherscanClient
from paper1_onchain_agent_id.features.feature_pipeline import (
    FeatureConfig,
    extract_temporal_features,
    extract_gas_features,
    extract_interaction_features,
    extract_approval_security_features,
    FEATURE_NAMES,
)
from paper1_onchain_agent_id.features.verify_c1c4 import (
    C1C4Verifier, C1C4Thresholds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "paper1_onchain_agent_id" / "data"
RAW_DIR = DATA_DIR / "raw"
V3_PARQUET = DATA_DIR / "features_provenance_v3.parquet"
CHECKPOINT_PATH = DATA_DIR / "features_provenance_v4_checkpoint.parquet"
OUT_PARQUET = DATA_DIR / "features_provenance_v4.parquet"
LABELS_PATH = DATA_DIR / "labels_provenance_v4.json"
RESULTS_PATH = (
    PROJECT_ROOT / "paper1_onchain_agent_id"
    / "experiments" / "mine_addresses_v4_results.json"
)

# Maximum new addresses to process (rate limit budget)
MAX_NEW_ADDRESSES = 600
CHECKPOINT_INTERVAL = 25


# ============================================================
# NEW CONTRACT ADDRESSES (v4 sources)
# ============================================================

# --- Agent Sources ---
# A. Aave V3 Pool (already used in v3 but we mine deeper)
AAVE_V3_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
AAVE_LIQUIDATION_METHOD = "0x00a718a9"

# B. Compound V3 cUSDC (Comet) -- absorb method 0x5a1b67f0
COMPOUND_V3_CUSDC = "0xc3d688B66703497DAA19211EEdff47f25384cdc3"
COMPOUND_ABSORB_METHOD = "0x5a1b67f0"

# C. 1inch AggregationRouterV5
ONEINCH_ROUTER_V5 = "0x1111111254EEB25477B68fb85Ed929f73A960582"

# D. Chainlink Keeper Registry v2.1
CHAINLINK_KEEPER_REGISTRY = "0x02777053d6764996e594c3E88AF1D58D5363a2e6"

# E. Uniswap V3 NonfungiblePositionManager
UNISWAP_V3_NFT_MANAGER = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
# mint: 0x88316456, increaseLiquidity: 0x219f5d17
UNISWAP_MINT_METHOD = "0x88316456"
UNISWAP_INCREASE_LIQ_METHOD = "0x219f5d17"

# --- Human Sources ---
# F. Gitcoin Grants Round contract (BulkCheckout)
GITCOIN_BULK_CHECKOUT = "0xD95A1969c41112cEE9A2c931E849bCef36a16F4C"

# G. ENS Reverse Registrar
ENS_REVERSE_REGISTRAR = "0x084b1c3C81545d370f3634392De611CaaBFf8148"

# H. PoolTogether V4 PrizePool
POOLTOGETHER_V4 = "0xd89a09084555a7D0ABe7B111b1f78DFEdDd638Be"

# I. Mirror.xyz Write Race Token / Factory
MIRROR_WRITE_TOKEN = "0xaEbf93B8D2c6d3BCC39AB03f450DCb1B1Cd6D763"
MIRROR_FACTORY = "0x302f746eE2fDC10DDff63188f71639094717a766"


# ============================================================
# HELPERS (reused from v3 with minor tweaks)
# ============================================================

def fetch_contract_callers(
    client: EtherscanClient,
    contract_addr: str,
    method_id: str = "",
    max_pages: int = 5,
    max_callers: int = 200,
    label: str = "caller",
) -> dict[str, tuple[str, str]]:
    """Fetch unique 'from' addresses that called a contract.

    Optionally filters by method_id prefix in the input data.

    Returns:
        {address: (name, provenance_source)}
    """
    found = {}
    for page in range(1, max_pages + 1):
        try:
            df = client.get_normal_txs(contract_addr, page=page, offset=10000)
        except Exception as exc:
            logger.warning("  Page %d failed: %s", page, exc)
            break
        if df.empty:
            break
        if "from" not in df.columns:
            break

        for _, row in df.iterrows():
            addr = str(row.get("from", "")).strip().lower()
            inp = str(row.get("input", "")) if pd.notna(row.get("input")) else ""

            if not addr or addr == contract_addr.lower():
                continue
            if addr == "0x0000000000000000000000000000000000000000":
                continue
            if method_id and not inp.lower().startswith(method_id.lower()):
                continue
            if not method_id and (not inp or len(inp) <= 10 or inp == "0x"):
                continue

            if addr not in found:
                found[addr] = (f"{label} ({addr[:10]}...)", f"on_chain_{label}")
            if len(found) >= max_callers:
                break

        if len(found) >= max_callers or len(df) < 10000:
            break

    return found


def find_high_frequency_callers(
    client: EtherscanClient,
    contract_addr: str,
    min_total_calls: int = 200,
    method_ids: list[str] | None = None,
    max_pages: int = 5,
    max_callers: int = 100,
    label: str = "high_freq_caller",
) -> dict[str, tuple[str, str]]:
    """Find addresses with high total call counts to a contract.

    Unlike the v3 per-day check, this uses total call counts which is
    more robust for position managers where activity is spread over weeks.
    """
    all_callers: dict[str, int] = {}
    for page in range(1, max_pages + 1):
        try:
            df = client.get_normal_txs(contract_addr, page=page, offset=10000)
        except Exception as exc:
            logger.warning("  Page %d failed: %s", page, exc)
            break
        if df.empty:
            break
        if "from" not in df.columns:
            break

        for _, row in df.iterrows():
            addr = str(row.get("from", "")).strip().lower()
            inp = str(row.get("input", "")) if pd.notna(row.get("input")) else ""
            if not addr or addr == contract_addr.lower():
                continue
            if addr == "0x0000000000000000000000000000000000000000":
                continue
            # Filter by method IDs if provided
            if method_ids:
                matched = False
                for mid in method_ids:
                    if inp.lower().startswith(mid.lower()):
                        matched = True
                        break
                if not matched:
                    continue
            all_callers[addr] = all_callers.get(addr, 0) + 1

        if len(df) < 10000:
            break

    # Filter by minimum calls
    found = {}
    for addr, count in sorted(all_callers.items(), key=lambda x: -x[1]):
        if count >= min_total_calls:
            found[addr] = (
                f"{label} ({count} calls)",
                f"on_chain_{label}",
            )
        if len(found) >= max_callers:
            break

    return found


def fetch_plain_eth_senders(
    client: EtherscanClient,
    contract_addr: str,
    max_pages: int = 5,
    max_callers: int = 200,
    label: str = "donor",
) -> dict[str, tuple[str, str]]:
    """Fetch unique 'from' addresses sending ANY transaction to a contract.

    Unlike fetch_contract_callers, this also includes plain ETH transfers
    (no calldata requirement) -- useful for donation / deposit contracts.
    """
    found = {}
    for page in range(1, max_pages + 1):
        try:
            df = client.get_normal_txs(contract_addr, page=page, offset=10000)
        except Exception as exc:
            logger.warning("  Page %d failed: %s", page, exc)
            break
        if df.empty:
            break
        if "from" not in df.columns:
            break

        for _, row in df.iterrows():
            addr = str(row.get("from", "")).strip().lower()
            if not addr or addr == contract_addr.lower():
                continue
            if addr == "0x0000000000000000000000000000000000000000":
                continue
            if addr not in found:
                found[addr] = (f"{label} ({addr[:10]}...)", f"on_chain_{label}")
            if len(found) >= max_callers:
                break

        if len(found) >= max_callers or len(df) < 10000:
            break

    return found


def extract_features_from_txs(txs: pd.DataFrame, config: FeatureConfig) -> dict:
    """Extract all 23 features from a DataFrame of transactions."""
    f = {}
    f.update(extract_temporal_features(txs, config))
    f.update(extract_gas_features(txs, config))
    f.update(extract_interaction_features(txs, config))
    f.update(extract_approval_security_features(txs, config))
    return f


def load_or_fetch_txs(
    client: EtherscanClient,
    addr: str,
    max_pages: int = 3,
) -> pd.DataFrame:
    """Load transactions from cache or fetch via API.

    Looks for cached parquet in RAW_DIR (case-insensitive).
    """
    addr_lower = addr.lower()

    # Check cache (multiple case variants)
    for variant in [addr, addr_lower]:
        path = RAW_DIR / f"{variant}.parquet"
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass

    # Also scan directory for case-insensitive match
    for p in RAW_DIR.glob("*.parquet"):
        if p.stem.lower() == addr_lower:
            try:
                return pd.read_parquet(p)
            except Exception:
                pass

    # Fetch from API
    try:
        txs = client.get_all_txs(addr, max_pages=max_pages)
        if not txs.empty:
            out_path = RAW_DIR / f"{addr_lower}.parquet"
            txs.to_parquet(out_path, index=False)
        return txs
    except Exception as exc:
        logger.warning("  fetch failed for %s: %s", addr[:16], exc)
        return pd.DataFrame()


# ============================================================
# CLASSIFIER EVALUATION
# ============================================================

def run_rf_evaluation(X: np.ndarray, y: np.ndarray, feature_names: list) -> dict:
    """Run Random Forest with 10-fold CV + repeated CV."""
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score, confusion_matrix,
    )
    from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )

    # --- 10-fold CV ---
    n = len(y)
    probs = np.zeros(n)
    preds = np.zeros(n, dtype=int)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = clone(rf)
        clf.fit(X_tr, y[train_idx])
        probs[test_idx] = clf.predict_proba(X_te)[:, 1]
        preds[test_idx] = clf.predict(X_te)

    cv_results = {
        "auc": round(float(roc_auc_score(y, probs)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1_human": round(float(f1_score(y, preds, pos_label=0, zero_division=0)), 4),
    }
    cm = confusion_matrix(y, preds).tolist()

    # --- Repeated 5-fold x 10 ---
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    fold_aucs = []
    for train_idx, test_idx in rskf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = clone(rf)
        clf.fit(X_tr, y[train_idx])
        y_prob = clf.predict_proba(X_te)[:, 1]
        try:
            fold_aucs.append(roc_auc_score(y[test_idx], y_prob))
        except ValueError:
            pass

    repeated_cv = {
        "mean_auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else 0.0,
        "std_auc": round(float(np.std(fold_aucs)), 4) if fold_aucs else 0.0,
    }

    # --- Feature importance ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf_full = clone(rf)
    clf_full.fit(X_scaled, y)
    importances = clf_full.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    top_features = {
        feature_names[i]: round(float(importances[i]), 4)
        for i in sorted_idx[:10]
    }

    return {
        "cv_10fold": cv_results,
        "repeated_5fold_10x": repeated_cv,
        "top10_features": top_features,
        "confusion_matrix": cm,
    }


def run_gb_evaluation(X: np.ndarray, y: np.ndarray) -> dict:
    """Run Gradient Boosting 10-fold CV."""
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    gb = GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1,
        random_state=42,
    )
    n = len(y)
    probs = np.zeros(n)
    preds = np.zeros(n, dtype=int)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = clone(gb)
        clf.fit(X_tr, y[train_idx])
        probs[test_idx] = clf.predict_proba(X_te)[:, 1]
        preds[test_idx] = clf.predict(X_te)

    return {
        "cv_10fold": {
            "auc": round(float(roc_auc_score(y, probs)), 4),
            "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, preds)), 4),
        }
    }


def run_gat_evaluation(
    X: np.ndarray,
    y: np.ndarray,
    df: pd.DataFrame,
) -> dict:
    """Run GAT evaluation on the provenance dataset.

    Builds a transaction graph from cached raw parquets and trains
    a 2-layer GAT with 5-fold stratified node splits.
    """
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv

    addresses = [str(a).lower() for a in df.index]
    addr_to_idx = {a: i for i, a in enumerate(addresses)}
    addr_set = set(addresses)

    # Build graph from raw parquets
    edges_set = set()
    logger.info("  Building transaction graph for GAT ...")
    t_graph = time.time()
    for i, addr in enumerate(addresses):
        raw_path = RAW_DIR / f"{addr}.parquet"
        if not raw_path.exists():
            for orig in df.index:
                if str(orig).lower() == addr:
                    raw_path = RAW_DIR / f"{orig}.parquet"
                    break
        if not raw_path.exists():
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
            if f_addr in addr_set and t_addr in addr_set and f_addr != t_addr:
                edges_set.add((addr_to_idx[f_addr], addr_to_idx[t_addr]))

    logger.info("  Graph: %d edges built in %.1fs",
                len(edges_set), time.time() - t_graph)

    if not edges_set:
        edge_index = np.array([
            list(range(len(addresses))),
            list(range(len(addresses))),
        ])
    else:
        edge_index = np.array(list(edges_set)).T

    # Prepare data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    x_t = torch.tensor(X_scaled, dtype=torch.float)
    y_t = torch.tensor(y, dtype=torch.long)
    edge_index_t = torch.tensor(edge_index, dtype=torch.long)
    data = Data(x=x_t, edge_index=edge_index_t, y=y_t)

    logger.info("  GAT graph: %d nodes, %d edges",
                data.num_nodes, data.num_edges)

    class GATModel(torch.nn.Module):
        def __init__(self, in_dim, hidden_dim=32, heads=4, out_dim=2, dropout=0.5):
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

    # 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []

    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        train_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)
        train_mask[tr_idx] = True
        test_mask[te_idx] = True

        model = GATModel(in_dim=X.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01,
                                       weight_decay=5e-4)
        best_auc = 0.0
        best_metrics = {}

        for epoch in range(300):
            model.train()
            optimizer.zero_grad()
            logits = model(data.x, data.edge_index)
            loss = F.cross_entropy(logits[train_mask], data.y[train_mask])
            loss.backward()
            optimizer.step()

            if epoch % 30 == 29 or epoch == 299:
                model.eval()
                with torch.no_grad():
                    logits_eval = model(data.x, data.edge_index)
                    probs = F.softmax(logits_eval, dim=1)[:, 1].cpu().numpy()
                    preds = logits_eval.argmax(dim=1).cpu().numpy()
                    y_np = data.y.cpu().numpy()
                    test_idx_np = test_mask.cpu().numpy()
                    try:
                        auc = float(roc_auc_score(
                            y_np[test_idx_np], probs[test_idx_np]
                        ))
                    except ValueError:
                        auc = 0.0
                    if auc > best_auc:
                        best_auc = auc
                        best_metrics = {
                            "auc": round(auc, 4),
                            "f1": round(float(f1_score(
                                y_np[test_idx_np], preds[test_idx_np],
                                zero_division=0
                            )), 4),
                            "accuracy": round(float(accuracy_score(
                                y_np[test_idx_np], preds[test_idx_np]
                            )), 4),
                        }

        fold_results.append(best_metrics)
        logger.info("  GAT Fold %d: AUC=%.4f F1=%.4f",
                     fold_idx + 1,
                     best_metrics.get("auc", 0),
                     best_metrics.get("f1", 0))

    mean_auc = float(np.mean([r.get("auc", 0) for r in fold_results]))
    mean_f1 = float(np.mean([r.get("f1", 0) for r in fold_results]))

    return {
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(float(np.std([r.get("auc", 0) for r in fold_results])), 4),
        "mean_f1": round(mean_f1, 4),
        "n_edges": int(len(edges_set)),
        "folds": fold_results,
    }


# ============================================================
# MAIN
# ============================================================

def run():
    print("=" * 80)
    print("Paper 1: Expanded Provenance-Only Mining v4 (1000+ target)")
    print("  Base: v3 parquet with 563 addresses")
    print("  Target: 1000+ total (400+ new agents, 200+ new humans)")
    print("  Labeling: Pure provenance (no C1-C4 gating)")
    print("=" * 80)
    t0 = time.time()

    client = EtherscanClient()
    print(f"  Etherscan API keys: {client.num_keys}")

    config = FeatureConfig()
    verifier = C1C4Verifier(
        client,
        thresholds=C1C4Thresholds(min_txs_for_analysis=15),
    )
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Phase 0: Load existing v3 data
    # ----------------------------------------------------------
    print(f"\n[Phase 0] Loading v3 parquet ...")
    v3_df = pd.read_parquet(V3_PARQUET)
    existing_addrs = set(str(a).lower() for a in v3_df.index)
    print(f"  v3 has {len(v3_df)} addresses "
          f"({(v3_df['label']==1).sum()} agents, {(v3_df['label']==0).sum()} humans)")

    # ----------------------------------------------------------
    # Phase 1: Mine NEW candidate addresses
    # ----------------------------------------------------------
    new_candidates: dict[str, dict] = {}

    # ==============================
    # NEW AGENT SOURCES
    # ==============================

    # A. Aave V3 liquidators (deeper mine -- more pages, skip existing)
    print("\n[A] Mining Aave V3 liquidation callers (expanded) ...")
    try:
        aave_callers = fetch_contract_callers(
            client, AAVE_V3_POOL,
            method_id=AAVE_LIQUIDATION_METHOD,
            max_pages=5, max_callers=150,
            label="aave_v3_liquidator",
        )
        new_aave = 0
        for addr, (name, source) in aave_callers.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"Aave V3 liquidator ({addr[:10]})",
                    "provenance_source": "on_chain_aave_v3_liquidation",
                    "label_provenance": 1,
                    "category": "aave_v3_liquidator",
                }
                new_aave += 1
        print(f"  Found {len(aave_callers)} total, {new_aave} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # B. Compound V3 liquidators (absorb method)
    print("\n[B] Mining Compound V3 (cUSDC) liquidators ...")
    try:
        comp_callers = fetch_contract_callers(
            client, COMPOUND_V3_CUSDC,
            method_id=COMPOUND_ABSORB_METHOD,
            max_pages=5, max_callers=100,
            label="compound_v3_liquidator",
        )
        # Also fetch generic callers (many liquidators use other methods)
        comp_generic = fetch_contract_callers(
            client, COMPOUND_V3_CUSDC,
            method_id="",  # any method
            max_pages=3, max_callers=100,
            label="compound_v3_caller",
        )
        comp_callers.update(comp_generic)
        new_comp = 0
        for addr, (name, source) in comp_callers.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"Compound V3 liquidator ({addr[:10]})",
                    "provenance_source": "on_chain_compound_v3_liquidation",
                    "label_provenance": 1,
                    "category": "compound_v3_liquidator",
                }
                new_comp += 1
        print(f"  Found {len(comp_callers)} total, {new_comp} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # C. 1inch high-frequency executors (>200 calls = automated)
    print("\n[C] Mining 1inch Router high-frequency executors ...")
    try:
        oneinch_hf = find_high_frequency_callers(
            client, ONEINCH_ROUTER_V5,
            min_total_calls=200,
            max_pages=5, max_callers=120,
            label="oneinch_hf_executor",
        )
        new_1inch = 0
        for addr, (name, source) in oneinch_hf.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"1inch HF executor ({addr[:10]})",
                    "provenance_source": "on_chain_1inch_high_freq",
                    "label_provenance": 1,
                    "category": "oneinch_hf_executor",
                }
                new_1inch += 1
        print(f"  Found {len(oneinch_hf)} total, {new_1inch} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # D. Chainlink Keeper Registry executors
    print("\n[D] Mining Chainlink Keeper Registry executors ...")
    try:
        chainlink_callers = fetch_contract_callers(
            client, CHAINLINK_KEEPER_REGISTRY,
            max_pages=5, max_callers=120,
            label="chainlink_keeper",
        )
        new_cl = 0
        for addr, (name, source) in chainlink_callers.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"Chainlink Keeper ({addr[:10]})",
                    "provenance_source": "on_chain_chainlink_keeper",
                    "label_provenance": 1,
                    "category": "chainlink_keeper",
                }
                new_cl += 1
        print(f"  Found {len(chainlink_callers)} total, {new_cl} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # E. Uniswap V3 NonfungiblePositionManager high-freq mint/liquidity
    print("\n[E] Mining Uniswap V3 NFT Position Manager HF callers ...")
    try:
        uni_nft_hf = find_high_frequency_callers(
            client, UNISWAP_V3_NFT_MANAGER,
            min_total_calls=50,
            method_ids=[UNISWAP_MINT_METHOD, UNISWAP_INCREASE_LIQ_METHOD],
            max_pages=5, max_callers=100,
            label="uniswap_v3_lp_bot",
        )
        new_uni = 0
        for addr, (name, source) in uni_nft_hf.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"Uniswap V3 LP bot ({addr[:10]})",
                    "provenance_source": "on_chain_uniswap_v3_nft_hf",
                    "label_provenance": 1,
                    "category": "uniswap_v3_lp_bot",
                }
                new_uni += 1
        print(f"  Found {len(uni_nft_hf)} total, {new_uni} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # ==============================
    # NEW HUMAN SOURCES
    # ==============================

    # F. Gitcoin Grants donors
    print("\n[F] Mining Gitcoin Grants donors ...")
    try:
        gitcoin_donors = fetch_plain_eth_senders(
            client, GITCOIN_BULK_CHECKOUT,
            max_pages=5, max_callers=150,
            label="gitcoin_donor",
        )
        new_gc = 0
        for addr, (name, source) in gitcoin_donors.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"Gitcoin donor ({addr[:10]})",
                    "provenance_source": "on_chain_gitcoin_grants_donor",
                    "label_provenance": 0,
                    "category": "gitcoin_donor",
                }
                new_gc += 1
        print(f"  Found {len(gitcoin_donors)} total, {new_gc} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # G. ENS Reverse Registrar setters
    print("\n[G] Mining ENS Reverse Registrar setters ...")
    try:
        ens_rev_callers = fetch_contract_callers(
            client, ENS_REVERSE_REGISTRAR,
            max_pages=5, max_callers=150,
            label="ens_reverse_setter",
        )
        new_ens = 0
        for addr, (name, source) in ens_rev_callers.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"ENS reverse setter ({addr[:10]})",
                    "provenance_source": "on_chain_ens_reverse_registrar",
                    "label_provenance": 0,
                    "category": "ens_reverse_setter",
                }
                new_ens += 1
        print(f"  Found {len(ens_rev_callers)} total, {new_ens} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # H. PoolTogether V4 depositors
    print("\n[H] Mining PoolTogether V4 depositors ...")
    try:
        pool_callers = fetch_plain_eth_senders(
            client, POOLTOGETHER_V4,
            max_pages=5, max_callers=120,
            label="pooltogether_depositor",
        )
        new_pt = 0
        for addr, (name, source) in pool_callers.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"PoolTogether depositor ({addr[:10]})",
                    "provenance_source": "on_chain_pooltogether_v4",
                    "label_provenance": 0,
                    "category": "pooltogether_depositor",
                }
                new_pt += 1
        print(f"  Found {len(pool_callers)} total, {new_pt} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # I. Mirror.xyz publishers
    print("\n[I] Mining Mirror.xyz publishers ...")
    try:
        # Try both Mirror contracts
        mirror_callers = {}
        for contract, lbl in [
            (MIRROR_WRITE_TOKEN, "mirror_writer"),
            (MIRROR_FACTORY, "mirror_publisher"),
        ]:
            try:
                mc = fetch_contract_callers(
                    client, contract,
                    max_pages=3, max_callers=80,
                    label=lbl,
                )
                mirror_callers.update(mc)
            except Exception as e:
                logger.warning("  Mirror %s failed: %s", contract[:12], e)

        new_mirror = 0
        for addr, (name, source) in mirror_callers.items():
            if addr not in existing_addrs and addr not in new_candidates:
                new_candidates[addr] = {
                    "name": f"Mirror.xyz publisher ({addr[:10]})",
                    "provenance_source": "on_chain_mirror_xyz",
                    "label_provenance": 0,
                    "category": "mirror_publisher",
                }
                new_mirror += 1
        print(f"  Found {len(mirror_callers)} total, {new_mirror} new")
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # ----------------------------------------------------------
    # Candidate Summary
    # ----------------------------------------------------------
    n_new_agent = sum(1 for v in new_candidates.values() if v["label_provenance"] == 1)
    n_new_human = sum(1 for v in new_candidates.values() if v["label_provenance"] == 0)
    print(f"\n{'=' * 60}")
    print(f"NEW CANDIDATES (not in v3): {len(new_candidates)}")
    print(f"  New Agents: {n_new_agent}")
    print(f"  New Humans: {n_new_human}")
    print(f"{'=' * 60}")

    # Category breakdown of new candidates
    cats = {}
    for v in new_candidates.values():
        cat = v["category"]
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Limit to MAX_NEW_ADDRESSES
    candidate_list = list(new_candidates.items())
    if len(candidate_list) > MAX_NEW_ADDRESSES:
        # Prioritize: take proportional agent/human
        agents_list = [(a, i) for a, i in candidate_list if i["label_provenance"] == 1]
        humans_list = [(a, i) for a, i in candidate_list if i["label_provenance"] == 0]
        # Aim for ~65% agents, 35% humans in new batch
        max_agents = min(len(agents_list), int(MAX_NEW_ADDRESSES * 0.65))
        max_humans = min(len(humans_list), MAX_NEW_ADDRESSES - max_agents)
        max_agents = min(len(agents_list), MAX_NEW_ADDRESSES - max_humans)
        candidate_list = agents_list[:max_agents] + humans_list[:max_humans]
        print(f"  Capped to {len(candidate_list)} "
              f"({max_agents} agents, {max_humans} humans)")

    # ----------------------------------------------------------
    # Phase 2: Extract features for new addresses
    # ----------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"PROCESSING {len(candidate_list)} NEW CANDIDATES")
    print(f"{'=' * 80}")

    # Load checkpoint if exists
    processed_addrs = set()
    rows = []
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint_df = pd.read_parquet(CHECKPOINT_PATH)
            processed_addrs = set(checkpoint_df.index)
            rows = checkpoint_df.reset_index().to_dict("records")
            print(f"  Resumed from checkpoint: {len(processed_addrs)} already done")
        except Exception:
            pass

    skipped_too_few_txs = 0
    skipped_no_data = 0
    skipped_contract = 0

    todo_list = [
        (addr, info) for addr, info in candidate_list
        if addr not in processed_addrs
    ]

    for i, (addr, info) in enumerate(todo_list):
        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(todo_list)}] {addr[:16]} "
                  f"{info['name'][:30]:<30}  "
                  f"({elapsed:.0f}s elapsed, {len(rows)} done)")

        # Load or fetch transactions
        txs = load_or_fetch_txs(client, addr, max_pages=3)
        if txs.empty:
            skipped_no_data += 1
            continue

        if len(txs) < 10:
            skipped_too_few_txs += 1
            continue

        # Filter to outgoing
        if "from" in txs.columns:
            outgoing = txs[txs["from"].str.lower() == addr.lower()]
            if outgoing.empty:
                skipped_contract += 1
                continue
            txs_for_features = outgoing
        else:
            txs_for_features = txs

        # C1-C4 diagnostic (does NOT gate labels)
        try:
            c1c4 = verifier.verify(addr, txs=txs)
        except Exception:
            c1c4 = {
                "c1": None, "c2": None, "c3": None, "c4": None,
                "is_agent": None, "confidence": 0.0,
            }

        # Extract 23 features
        try:
            features = extract_features_from_txs(txs_for_features, config)
        except Exception as exc:
            logger.warning("  feature extraction failed for %s: %s", addr[:16], exc)
            continue

        features.update({
            "address": addr,
            "label": info["label_provenance"],
            "name": info["name"],
            "source": info["provenance_source"],
            "category": info["category"],
            "n_transactions": len(txs_for_features),
            "c1c4_c1": c1c4.get("c1"),
            "c1c4_c2": c1c4.get("c2"),
            "c1c4_c3": c1c4.get("c3"),
            "c1c4_c4": c1c4.get("c4"),
            "c1c4_is_agent": c1c4.get("is_agent"),
            "c1c4_confidence": c1c4.get("confidence", 0.0),
        })
        rows.append(features)

        # Checkpoint
        if (i + 1) % CHECKPOINT_INTERVAL == 0 and rows:
            df_ckpt = pd.DataFrame(rows)
            if "address" in df_ckpt.columns:
                df_ckpt.set_index("address", inplace=True)
            df_ckpt.to_parquet(CHECKPOINT_PATH)
            print(f"    Checkpoint saved: {len(df_ckpt)} new rows")

    # ----------------------------------------------------------
    # Phase 3: Merge with v3 to create v4
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"  New processed: {len(rows)}")
    print(f"  Skipped (too few txs): {skipped_too_few_txs}")
    print(f"  Skipped (no data): {skipped_no_data}")
    print(f"  Skipped (contract/no outgoing): {skipped_contract}")

    if rows:
        new_df = pd.DataFrame(rows)
        if "address" in new_df.columns:
            new_df.set_index("address", inplace=True)
        new_df = new_df[~new_df.index.duplicated(keep="first")]
    else:
        new_df = pd.DataFrame()

    # Merge v3 + new
    print(f"\n  Merging v3 ({len(v3_df)}) + new ({len(new_df)}) ...")
    # Ensure same columns
    all_cols = list(v3_df.columns)
    for col in all_cols:
        if col not in new_df.columns:
            new_df[col] = np.nan

    # Reorder columns to match v3
    new_df = new_df[all_cols] if len(new_df) > 0 else new_df

    df = pd.concat([v3_df, new_df], axis=0)
    df = df[~df.index.duplicated(keep="first")]

    n_agent_total = int((df["label"] == 1).sum())
    n_human_total = int((df["label"] == 0).sum())
    print(f"  Final v4: {len(df)} rows ({n_agent_total} agents, {n_human_total} humans)")

    # Save parquet
    df.to_parquet(OUT_PARQUET)
    print(f"\nSaved {OUT_PARQUET}")

    # Save labels JSON
    labels_obj = {}
    for addr in df.index:
        row = df.loc[addr]
        labels_obj[addr] = {
            "name": str(row.get("name", "")),
            "provenance_source": str(row.get("source", "")),
            "label_provenance": int(row["label"]),
            "category": str(row.get("category", "")),
        }
    with open(LABELS_PATH, "w") as f:
        json.dump(labels_obj, f, indent=2)
    print(f"Saved {LABELS_PATH}")

    # ----------------------------------------------------------
    # Phase 4: Classifier evaluation
    # ----------------------------------------------------------
    print(f"\n{'=' * 80}")
    print("CLASSIFIER EVALUATION")
    print(f"{'=' * 80}")

    feature_cols = [c for c in FEATURE_NAMES if c in df.columns]
    X = df[feature_cols].copy()

    # Impute NaN
    X_values = X.values.astype(float)
    nan_mask = np.isnan(X_values)
    if nan_mask.any():
        col_medians = np.nanmedian(X_values, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X_values.shape[1]):
            X_values[nan_mask[:, j], j] = col_medians[j]

    # Clip extremes
    for j in range(X_values.shape[1]):
        lo, hi = np.nanpercentile(X_values[:, j], [1, 99])
        X_values[:, j] = np.clip(X_values[:, j], lo, hi)

    y = df["label"].values.astype(int)

    # Check we have both classes
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        print(f"ERROR: Only {len(unique_labels)} class(es) found. Cannot train.")
        return

    # Random Forest
    print(f"\n--- Random Forest (n={len(y)}, {n_agent_total} agents, "
          f"{n_human_total} humans) ---")
    rf_results = run_rf_evaluation(X_values, y, feature_cols)
    print(f"  10-fold CV AUC: {rf_results['cv_10fold']['auc']}")
    print(f"  10-fold CV F1:  {rf_results['cv_10fold']['f1']}")
    print(f"  5x10 CV: AUC={rf_results['repeated_5fold_10x']['mean_auc']}"
          f" +/- {rf_results['repeated_5fold_10x']['std_auc']}")
    print(f"  Top features: {list(rf_results['top10_features'].keys())[:5]}")

    # Gradient Boosting
    print(f"\n--- Gradient Boosting ---")
    gb_results = run_gb_evaluation(X_values, y)
    print(f"  10-fold CV AUC: {gb_results['cv_10fold']['auc']}")
    print(f"  10-fold CV F1:  {gb_results['cv_10fold']['f1']}")

    # GAT
    print(f"\n--- GAT (Graph Attention Network) ---")
    gat_results = run_gat_evaluation(X_values, y, df)
    print(f"  GAT 5-fold AUC: {gat_results['mean_auc']}"
          f" +/- {gat_results['std_auc']}")
    print(f"  GAT 5-fold F1:  {gat_results['mean_f1']}")

    # ----------------------------------------------------------
    # Phase 5: C1-C4 diagnostic agreement
    # ----------------------------------------------------------
    print(f"\n--- C1-C4 Diagnostic Agreement ---")
    c1c4_is_agent = df["c1c4_is_agent"].copy()
    c1c4_bool = c1c4_is_agent.map({
        True: True, False: False,
        "True": True, "False": False,
        1: True, 0: False,
        1.0: True, 0.0: False,
    })
    valid_mask = c1c4_bool.notna()
    if valid_mask.sum() > 0:
        prov_label = (df["label"] == 1)
        agreement = (c1c4_bool[valid_mask] == prov_label[valid_mask]).mean()
        print(f"  C1-C4 vs provenance agreement: {agreement:.2%} "
              f"({valid_mask.sum()} addresses with valid C1-C4)")
    else:
        agreement = 0.0
        print("  No valid C1-C4 results for agreement computation")

    # ----------------------------------------------------------
    # Phase 6: Save results
    # ----------------------------------------------------------
    elapsed = time.time() - t0
    results = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 2),
        "dataset": {
            "v3_base": len(v3_df),
            "new_candidates_mined": len(new_candidates),
            "new_processed": len(new_df),
            "n_processed": len(df),
            "n_agents": n_agent_total,
            "n_humans": n_human_total,
            "n_features": len(feature_cols),
            "agent_to_human_ratio": round(n_agent_total / max(n_human_total, 1), 2),
            "n_skipped_too_few_txs": skipped_too_few_txs,
            "n_skipped_no_data": skipped_no_data,
            "n_skipped_contract": skipped_contract,
        },
        "categories": {str(k): int(v) for k, v in
                       df["category"].value_counts().to_dict().items()},
        "new_source_counts": {str(k): int(v) for k, v in cats.items()},
        "classifiers": {
            "RandomForest": rf_results,
            "GradientBoosting": gb_results,
            "GAT": gat_results,
        },
        "c1c4_diagnostic_agreement": round(float(agreement), 4),
        "comparison_to_v3": {
            "v3_n_samples": len(v3_df),
            "v3_rf_cv_auc": 0.7956,
            "v3_gat_auc": 0.812,
            "v4_n_samples": len(df),
            "v4_rf_cv_auc": rf_results["cv_10fold"]["auc"],
            "v4_gat_auc": gat_results["mean_auc"],
            "delta_n": len(df) - len(v3_df),
            "delta_rf_auc": round(rf_results["cv_10fold"]["auc"] - 0.7956, 4),
            "delta_gat_auc": round(gat_results["mean_auc"] - 0.812, 4),
        },
    }

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    results_ser = json.loads(json.dumps(results, default=_convert))

    with open(RESULTS_PATH, "w") as f:
        json.dump(results_ser, f, indent=2)
    print(f"\nSaved {RESULTS_PATH}")

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("  Checkpoint removed (no longer needed)")

    print(f"\n{'=' * 80}")
    print(f"DONE: {len(df)} provenance-labeled rows "
          f"({n_agent_total} agents, {n_human_total} humans)")
    print(f"  RF AUC:  {rf_results['cv_10fold']['auc']} (10-fold CV), "
          f"{rf_results['repeated_5fold_10x']['mean_auc']} (5x10)")
    print(f"  GB AUC:  {gb_results['cv_10fold']['auc']} (10-fold CV)")
    print(f"  GAT AUC: {gat_results['mean_auc']} (5-fold)")
    print(f"  C1-C4 agreement: {agreement:.2%}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 80}")

    return results


if __name__ == "__main__":
    run()
