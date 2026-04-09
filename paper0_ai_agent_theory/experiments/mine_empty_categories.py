"""
Paper 0: Mine Addresses for Empty Taxonomy Categories (4, 5, 7)
================================================================
Populates the three taxonomy categories that currently have zero entries:

  Category 4 — Autonomous DAO Agent
  Category 5 — Cross-Chain Bridge Agent
  Category 7 — RL Trading Agent

Mining strategies:
  DAO Agents:
    - Gnosis Safe execTransaction(0x6a761202) callers via Safe Proxy Factory
    - Nouns DAO executor outgoing transactions
  Bridge Relayers:
    - Across Protocol SpokePool fillRelay callers
    - Stargate Router relay callers
    - LayerZero Endpoint relayer addresses
  RL Trading Agents:
    - Behavioral proxy from features_expanded.parquet:
      addresses with high active_hour_entropy, method_id_diversity,
      and multi_protocol_interaction_count
    - Multi-protocol orchestrators (interact with Uniswap + Aave + Compound
      in the same day)

For each address:
  1. Load cached raw parquet or fetch via EtherscanClient
  2. Extract 23 behavioral features
  3. Run C1-C4 diagnostic
  4. Label with taxonomy category and provenance source

Outputs:
  - empty_categories_results.json
  - Updates features_with_taxonomy.parquet
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
    C1C4Verifier,
    C1C4Thresholds,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "paper1_onchain_agent_id" / "data"
RAW_DIR = DATA_DIR / "raw"
TAXONOMY_PARQUET = DATA_DIR / "features_with_taxonomy.parquet"
RESULTS_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "empty_categories_results.json"
)

# ============================================================
# CONTRACT ADDRESSES
# ============================================================

# --- DAO Agent sources ---
SAFE_PROXY_FACTORY = "0xa6B71E26C5e0845f74c812102Ca7114b6a896AB2"
SAFE_MASTER_COPY = "0xd9Db270c1B5E3Bd161E8c8503c55cEABeE709552"
NOUNS_DAO_EXECUTOR = "0x0BC3807Ec262cB779b38D65b38158acc3bfedE10"
EXEC_TRANSACTION_METHOD = "0x6a761202"

# --- Bridge Agent sources ---
ACROSS_SPOKEPOOL = "0x5c7BCd6E7De5423a257D81B442095A1a6ced35C5"
STARGATE_ROUTER = "0x8731d54E9D02c286767d56ac03e8037C07e01e98"
LAYERZERO_ENDPOINT = "0x66A71Dcef29A0fFBDBE3c6a460a3B5BC225Cd675"

# fillRelay / fillV3Relay selectors for Across
FILL_RELAY_METHODS = ["0xe2a7515e", "0x376f43a8", "0xb8201484"]

# --- RL Trading Agent sources (multi-protocol orchestration) ---
UNISWAP_V2_ROUTER = "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D"
UNISWAP_V3_ROUTER = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
AAVE_V3_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
COMPOUND_V3_COMET = "0xc3d688B66703497DAA19211EEdff47f25384cdc3"

# ============================================================
# HELPERS
# ============================================================


def load_or_fetch_txs(
    client: EtherscanClient,
    addr: str,
    max_pages: int = 3,
) -> pd.DataFrame:
    """Load transactions from cache or fetch via API."""
    addr_lower = addr.lower()

    for variant in [addr, addr_lower]:
        path = RAW_DIR / f"{variant}.parquet"
        if path.exists():
            try:
                return pd.read_parquet(path)
            except Exception:
                pass

    # Case-insensitive scan
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


def extract_features_from_txs(txs: pd.DataFrame, config: FeatureConfig) -> dict:
    """Extract all 23 features from a DataFrame of transactions."""
    f = {}
    f.update(extract_temporal_features(txs, config))
    f.update(extract_gas_features(txs, config))
    f.update(extract_interaction_features(txs, config))
    f.update(extract_approval_security_features(txs, config))
    return f


def fetch_contract_callers(
    client: EtherscanClient,
    contract_addr: str,
    method_ids: list[str] | None = None,
    max_pages: int = 5,
    max_callers: int = 200,
    label: str = "caller",
) -> dict[str, tuple[str, str]]:
    """Fetch unique 'from' addresses that called a contract.

    Optionally filters by any of the given method_id prefixes.

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

            # Filter by method ID(s) if specified
            if method_ids:
                matched = any(inp.lower().startswith(m.lower()) for m in method_ids)
                if not matched:
                    continue
            else:
                # Skip plain ETH transfers
                if not inp or len(inp) <= 10 or inp == "0x":
                    continue

            if addr not in found:
                found[addr] = (f"{label} ({addr[:10]}...)", f"on_chain_{label}")
            if len(found) >= max_callers:
                break

        if len(found) >= max_callers or len(df) < 10000:
            break

    return found


def fetch_outgoing_callers(
    client: EtherscanClient,
    from_addr: str,
    max_pages: int = 3,
    max_callers: int = 100,
    label: str = "dao_executor_target",
) -> dict[str, tuple[str, str]]:
    """Fetch addresses that a DAO executor sends txs TO.

    For DAO executors, the interesting addresses are the 'to' targets
    that receive automated execution calls, but more importantly,
    the addresses that TRIGGER executions (which are the 'from' addresses
    on txs TO the executor). We fetch txs TO the executor to find
    the triggering bot EOAs.
    """
    found = {}
    for page in range(1, max_pages + 1):
        try:
            df = client.get_normal_txs(from_addr, page=page, offset=10000)
        except Exception as exc:
            logger.warning("  Page %d failed: %s", page, exc)
            break
        if df.empty:
            break

        for _, row in df.iterrows():
            addr = str(row.get("from", "")).strip().lower()
            inp = str(row.get("input", "")) if pd.notna(row.get("input")) else ""

            if not addr or addr == from_addr.lower():
                continue
            if addr == "0x0000000000000000000000000000000000000000":
                continue
            # Must have calldata (not plain ETH transfers)
            if not inp or len(inp) <= 10 or inp == "0x":
                continue

            if addr not in found:
                found[addr] = (
                    f"{label} ({addr[:10]}...)",
                    f"on_chain_{label}",
                )
            if len(found) >= max_callers:
                break

        if len(found) >= max_callers or len(df) < 10000:
            break

    return found


# ============================================================
# MINING FUNCTIONS
# ============================================================


def mine_dao_agents(client: EtherscanClient) -> dict[str, dict]:
    """Mine Autonomous DAO Agent addresses (category 4).

    Strategy:
      1. Gnosis Safe execTransaction callers — EOAs that call
         execTransaction(0x6a761202) on Safe multisig contracts.
         These are often automated DAO executor bots.
      2. Nouns DAO executor — addresses that interact with the
         Nouns DAO timelock executor contract.
    """
    logger.info("=" * 60)
    logger.info("Mining DAO Agents (Category 4)")
    logger.info("=" * 60)

    candidates = {}

    # 1. Gnosis Safe execTransaction callers (via Safe Master Copy)
    logger.info("  Fetching Safe execTransaction callers (Master Copy)...")
    safe_callers = fetch_contract_callers(
        client,
        SAFE_MASTER_COPY,
        method_ids=[EXEC_TRANSACTION_METHOD],
        max_pages=5,
        max_callers=50,
        label="safe_execTransaction",
    )
    logger.info("  Found %d Safe Master Copy callers", len(safe_callers))

    for addr, (name, source) in safe_callers.items():
        candidates[addr] = {
            "name": f"DAO Agent: {name}",
            "provenance_source": "gnosis_safe_execTransaction",
            "taxonomy_index": 4,
            "taxonomy_category": "AutonomousDAOAgent",
        }

    # 2. Gnosis Safe Proxy Factory callers
    logger.info("  Fetching Safe Proxy Factory callers...")
    proxy_callers = fetch_contract_callers(
        client,
        SAFE_PROXY_FACTORY,
        method_ids=None,  # any interaction
        max_pages=3,
        max_callers=30,
        label="safe_proxy_factory",
    )
    logger.info("  Found %d Proxy Factory callers", len(proxy_callers))

    for addr, (name, source) in proxy_callers.items():
        if addr not in candidates:
            candidates[addr] = {
                "name": f"DAO Agent: {name}",
                "provenance_source": "gnosis_safe_proxy_factory",
                "taxonomy_index": 4,
                "taxonomy_category": "AutonomousDAOAgent",
            }

    # 3. Nouns DAO executor interactions
    logger.info("  Fetching Nouns DAO executor interactions...")
    nouns_callers = fetch_outgoing_callers(
        client,
        NOUNS_DAO_EXECUTOR,
        max_pages=3,
        max_callers=30,
        label="nouns_dao_executor",
    )
    logger.info("  Found %d Nouns DAO executor callers", len(nouns_callers))

    for addr, (name, source) in nouns_callers.items():
        if addr not in candidates:
            candidates[addr] = {
                "name": f"DAO Agent: {name}",
                "provenance_source": "nouns_dao_executor",
                "taxonomy_index": 4,
                "taxonomy_category": "AutonomousDAOAgent",
            }

    logger.info("  Total DAO Agent candidates: %d", len(candidates))
    return candidates


def mine_bridge_agents(client: EtherscanClient) -> dict[str, dict]:
    """Mine Cross-Chain Bridge Agent addresses (category 5).

    Strategy:
      1. Across Protocol SpokePool — fillRelay callers (bridge relayers)
      2. Stargate Router — relay transaction senders
      3. LayerZero Endpoint — relayer addresses
    """
    logger.info("=" * 60)
    logger.info("Mining Bridge Agents (Category 5)")
    logger.info("=" * 60)

    candidates = {}

    # 1. Across Protocol SpokePool relayers
    logger.info("  Fetching Across SpokePool fillRelay callers...")
    across_callers = fetch_contract_callers(
        client,
        ACROSS_SPOKEPOOL,
        method_ids=FILL_RELAY_METHODS,
        max_pages=5,
        max_callers=50,
        label="across_fillRelay",
    )
    logger.info("  Found %d Across relayers", len(across_callers))

    for addr, (name, source) in across_callers.items():
        candidates[addr] = {
            "name": f"Bridge Agent: {name}",
            "provenance_source": "across_protocol_spokepool",
            "taxonomy_index": 5,
            "taxonomy_category": "CrossChainBridgeAgent",
        }

    # 2. Stargate Router callers
    logger.info("  Fetching Stargate Router relay callers...")
    stargate_callers = fetch_contract_callers(
        client,
        STARGATE_ROUTER,
        method_ids=None,
        max_pages=5,
        max_callers=50,
        label="stargate_relay",
    )
    logger.info("  Found %d Stargate callers", len(stargate_callers))

    for addr, (name, source) in stargate_callers.items():
        if addr not in candidates:
            candidates[addr] = {
                "name": f"Bridge Agent: {name}",
                "provenance_source": "stargate_router",
                "taxonomy_index": 5,
                "taxonomy_category": "CrossChainBridgeAgent",
            }

    # 3. LayerZero Endpoint relayers
    logger.info("  Fetching LayerZero Endpoint relayers...")
    lz_callers = fetch_contract_callers(
        client,
        LAYERZERO_ENDPOINT,
        method_ids=None,
        max_pages=5,
        max_callers=50,
        label="layerzero_relayer",
    )
    logger.info("  Found %d LayerZero relayers", len(lz_callers))

    for addr, (name, source) in lz_callers.items():
        if addr not in candidates:
            candidates[addr] = {
                "name": f"Bridge Agent: {name}",
                "provenance_source": "layerzero_endpoint",
                "taxonomy_index": 5,
                "taxonomy_category": "CrossChainBridgeAgent",
            }

    logger.info("  Total Bridge Agent candidates: %d", len(candidates))
    return candidates


def mine_rl_trading_agents(
    client: EtherscanClient,
) -> dict[str, dict]:
    """Mine RL Trading Agent addresses (category 7).

    Strategy:
      1. Behavioral proxy from features_expanded.parquet:
         Addresses with high active_hour_entropy (>3.0), high
         method_id_diversity (>0.3), and high
         multi_protocol_interaction_count (>3.0).
      2. Multi-protocol orchestrators: find addresses from existing raw
         parquets that interact with Uniswap AND Aave AND Compound.
      3. High-frequency multi-protocol callers from Aave V3 Pool.
    """
    logger.info("=" * 60)
    logger.info("Mining RL Trading Agents (Category 7)")
    logger.info("=" * 60)

    candidates = {}

    # Strategy 1: Behavioral proxy from existing features
    expanded_path = DATA_DIR / "features_expanded.parquet"
    if expanded_path.exists():
        logger.info("  Scanning features_expanded.parquet for RL-like profiles...")
        df = pd.read_parquet(expanded_path)
        agents = df[df["label"] == 1]

        high_entropy = agents["active_hour_entropy"] > 3.0
        high_method = agents["method_id_diversity"] > 0.3
        high_multi = agents["multi_protocol_interaction_count"] > 3.0

        # Relaxed criteria: any 2 of 3 conditions
        mask = (
            (high_entropy & high_method)
            | (high_entropy & high_multi)
            | (high_method & high_multi)
        )
        rl_candidates = agents[mask]
        logger.info(
            "  Found %d behavioral RL-proxy candidates", len(rl_candidates)
        )

        for addr in rl_candidates.index[:30]:
            addr_lower = str(addr).lower()
            name = str(rl_candidates.loc[addr].get("name", ""))
            candidates[addr_lower] = {
                "name": f"RL Agent (behavioral): {name}" if name and name != "nan" else f"RL Agent (behavioral): {addr_lower[:10]}...",
                "provenance_source": "behavioral_proxy_features_expanded",
                "taxonomy_index": 7,
                "taxonomy_category": "RLTradingAgent",
            }

    # Strategy 2: Multi-protocol orchestrators from raw parquets
    logger.info("  Scanning raw parquets for multi-protocol orchestrators...")
    protocol_addrs = {
        UNISWAP_V2_ROUTER.lower(),
        UNISWAP_V3_ROUTER.lower(),
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Uniswap Universal Router
        AAVE_V3_POOL.lower(),
        "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",  # Aave V2 Pool
        COMPOUND_V3_COMET.lower(),
        "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",  # Compound Comptroller
    }
    uniswap_set = {
        UNISWAP_V2_ROUTER.lower(),
        UNISWAP_V3_ROUTER.lower(),
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",
    }
    aave_set = {
        AAVE_V3_POOL.lower(),
        "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",
    }
    compound_set = {
        COMPOUND_V3_COMET.lower(),
        "0x3d9819210a31b4961b30ef54be2aed79b9c9cd3b",
    }

    # Scan a sample of raw parquets to find multi-protocol users
    raw_files = sorted(RAW_DIR.glob("*.parquet"))
    scanned = 0
    multi_proto_found = 0
    for raw_path in raw_files:
        if multi_proto_found >= 25 or scanned >= 2000:
            break
        scanned += 1
        addr = raw_path.stem.lower()
        if addr in candidates:
            continue
        try:
            txs = pd.read_parquet(raw_path)
            if txs.empty or "to" not in txs.columns or len(txs) < 50:
                continue
            targets = set(txs["to"].str.lower().dropna().unique())
            has_uniswap = bool(targets & uniswap_set)
            has_aave = bool(targets & aave_set)
            has_compound = bool(targets & compound_set)

            if has_uniswap and has_aave and has_compound:
                # Check same-day multi-protocol use
                if "timeStamp" in txs.columns:
                    txs_ts = txs.copy()
                    txs_ts["day"] = pd.to_numeric(
                        txs_ts["timeStamp"], errors="coerce"
                    ) // 86400
                    txs_ts["to_lower"] = txs_ts["to"].str.lower()

                    for day_val, day_group in txs_ts.groupby("day"):
                        day_targets = set(day_group["to_lower"].dropna())
                        if (
                            bool(day_targets & uniswap_set)
                            and bool(day_targets & aave_set)
                            and bool(day_targets & compound_set)
                        ):
                            candidates[addr] = {
                                "name": f"RL Agent (multi-protocol): {addr[:10]}...",
                                "provenance_source": "multi_protocol_orchestrator",
                                "taxonomy_index": 7,
                                "taxonomy_category": "RLTradingAgent",
                            }
                            multi_proto_found += 1
                            break
        except Exception:
            continue

    logger.info(
        "  Found %d multi-protocol orchestrators (scanned %d parquets)",
        multi_proto_found,
        scanned,
    )

    # Strategy 3: Aave V3 high-frequency callers that also touch Uniswap
    logger.info("  Fetching Aave V3 high-frequency callers...")
    aave_callers = fetch_contract_callers(
        client,
        AAVE_V3_POOL,
        method_ids=None,
        max_pages=3,
        max_callers=100,
        label="aave_v3_caller",
    )
    # Filter to those that also interact with other protocols
    aave_addrs = list(aave_callers.keys())[:60]  # Limit API calls
    rl_from_aave = 0
    for addr in aave_addrs:
        if len(candidates) >= 50 or rl_from_aave >= 15:
            break
        if addr in candidates:
            continue
        try:
            txs = load_or_fetch_txs(client, addr, max_pages=2)
            if txs.empty or "to" not in txs.columns or len(txs) < 30:
                continue
            targets = set(txs["to"].str.lower().dropna().unique())
            has_uniswap = bool(targets & uniswap_set)
            if has_uniswap and len(targets) >= 5:
                candidates[addr] = {
                    "name": f"RL Agent (Aave+Uniswap): {addr[:10]}...",
                    "provenance_source": "aave_uniswap_multi_protocol",
                    "taxonomy_index": 7,
                    "taxonomy_category": "RLTradingAgent",
                }
                rl_from_aave += 1
        except Exception:
            continue

    logger.info("  Found %d from Aave+Uniswap cross-check", rl_from_aave)
    logger.info("  Total RL Trading Agent candidates: %d", len(candidates))
    return candidates


# ============================================================
# MAIN PIPELINE
# ============================================================


def main():
    print("=" * 80)
    print("Paper 0: Mining Empty Taxonomy Categories (4, 5, 7)")
    print("=" * 80)

    client = EtherscanClient()
    logger.info("EtherscanClient initialized with %d API keys", client.num_keys)

    config = FeatureConfig()
    verifier = C1C4Verifier(client, C1C4Thresholds())

    # Load existing taxonomy parquet
    if TAXONOMY_PARQUET.exists():
        existing_df = pd.read_parquet(TAXONOMY_PARQUET)
        existing_addrs = set(str(a).lower() for a in existing_df.index)
        logger.info(
            "Loaded existing taxonomy parquet: %d rows", len(existing_df)
        )
    else:
        existing_df = pd.DataFrame()
        existing_addrs = set()

    # ----------------------------------------------------------
    # Phase 1: Mine candidate addresses
    # ----------------------------------------------------------
    all_candidates = {}

    dao_candidates = mine_dao_agents(client)
    bridge_candidates = mine_bridge_agents(client)
    rl_candidates = mine_rl_trading_agents(client)

    all_candidates.update(dao_candidates)
    all_candidates.update(bridge_candidates)
    all_candidates.update(rl_candidates)

    # Remove addresses already in the dataset
    new_candidates = {
        a: info for a, info in all_candidates.items()
        if a not in existing_addrs
    }
    logger.info(
        "Candidates: %d total, %d new (excluding %d already in dataset)",
        len(all_candidates),
        len(new_candidates),
        len(all_candidates) - len(new_candidates),
    )

    # ----------------------------------------------------------
    # Phase 2: Featurize each address
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 2: Featurizing %d new addresses", len(new_candidates))
    logger.info("=" * 60)

    records = []
    c1c4_results = []
    failed = []
    processed = 0

    for addr, info in new_candidates.items():
        processed += 1
        if processed % 10 == 0:
            logger.info(
                "  Progress: %d/%d (cat=%d, %s)",
                processed,
                len(new_candidates),
                info["taxonomy_index"],
                info["taxonomy_category"],
            )

        try:
            txs = load_or_fetch_txs(client, addr, max_pages=3)
            if txs.empty or len(txs) < 5:
                logger.debug("  Skipping %s: too few txs (%d)", addr[:12], len(txs))
                failed.append({"address": addr, "reason": "too_few_txs"})
                continue

            # Extract 23 features
            features = extract_features_from_txs(txs, config)
            features["address"] = addr

            # Add metadata
            features["label"] = 1  # agent
            features["name"] = info["name"]
            features["source"] = info["provenance_source"]
            features["taxonomy_index"] = info["taxonomy_index"]
            features["taxonomy_category"] = info["taxonomy_category"]
            features["n_transactions"] = len(txs)

            # C1-C4 diagnostic
            try:
                c1c4 = verifier.verify(addr, txs=txs)
                features["c1c4_confidence"] = c1c4["confidence"]
                features["c1"] = c1c4["c1"]
                features["c2"] = c1c4["c2"]
                features["c3"] = c1c4["c3"]
                features["c4"] = c1c4["c4"]
                features["is_agent_c1c4"] = c1c4["is_agent"]
                c1c4_results.append(c1c4)
            except Exception as exc:
                features["c1c4_confidence"] = 0.0
                features["c1"] = None
                features["c2"] = None
                features["c3"] = None
                features["c4"] = None
                features["is_agent_c1c4"] = None
                logger.debug("  C1C4 failed for %s: %s", addr[:12], exc)

            # Taxonomy metadata
            features["confidence"] = 0.85  # high confidence provenance
            features["rule"] = info["provenance_source"]
            features["source_tier"] = "provenance_contract_interaction"

            records.append(features)

        except Exception as exc:
            logger.warning("  Failed %s: %s", addr[:12], exc)
            failed.append({"address": addr, "reason": str(exc)})

    logger.info(
        "Featurized: %d succeeded, %d failed out of %d",
        len(records),
        len(failed),
        len(new_candidates),
    )

    # ----------------------------------------------------------
    # Phase 3: Build results DataFrame and merge
    # ----------------------------------------------------------
    if not records:
        logger.error("No records produced. Exiting.")
        return

    new_df = pd.DataFrame(records)
    new_df = new_df.set_index("address")

    # Ensure all required columns exist
    for col in FEATURE_NAMES:
        if col not in new_df.columns:
            new_df[col] = 0.0

    # Count by category
    cat_counts = {}
    for cat_idx, cat_name in [(4, "AutonomousDAOAgent"), (5, "CrossChainBridgeAgent"), (7, "RLTradingAgent")]:
        mask = new_df["taxonomy_index"] == cat_idx
        count = mask.sum()
        cat_counts[cat_name] = int(count)
        logger.info("  Category %d (%s): %d addresses", cat_idx, cat_name, count)

    # C1-C4 pass rates
    c1c4_pass_rates = {}
    for cat_idx in [4, 5, 7]:
        cat_mask = new_df["taxonomy_index"] == cat_idx
        cat_rows = new_df[cat_mask]
        if len(cat_rows) > 0:
            for c in ["c1", "c2", "c3", "c4", "is_agent_c1c4"]:
                if c in cat_rows.columns:
                    pass_rate = cat_rows[c].dropna().astype(bool).mean()
                    c1c4_pass_rates[f"cat{cat_idx}_{c}"] = round(float(pass_rate), 3)

    # ----------------------------------------------------------
    # Phase 4: Update features_with_taxonomy.parquet
    # ----------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Phase 4: Updating taxonomy parquet")
    logger.info("=" * 60)

    # Align columns for merge
    existing_cols = list(existing_df.columns) if not existing_df.empty else []
    merge_cols = [c for c in existing_cols if c in new_df.columns]

    # Only keep columns that exist in the original parquet
    if existing_cols:
        new_for_merge = new_df[[c for c in existing_cols if c in new_df.columns]].copy()
        # Add any missing columns with defaults
        for c in existing_cols:
            if c not in new_for_merge.columns:
                new_for_merge[c] = np.nan
        new_for_merge = new_for_merge[existing_cols]

        updated_df = pd.concat([existing_df, new_for_merge])
        # Drop duplicates (keep first / existing)
        updated_df = updated_df[~updated_df.index.duplicated(keep="first")]
    else:
        updated_df = new_df

    updated_df.to_parquet(TAXONOMY_PARQUET)
    logger.info("  Updated parquet: %d rows (was %d)", len(updated_df), len(existing_df))

    # Verify new category counts in updated parquet
    agents_updated = updated_df[updated_df["label"] == 1]
    final_counts = {}
    for cat_idx, cat_name in [(4, "AutonomousDAOAgent"), (5, "CrossChainBridgeAgent"), (7, "RLTradingAgent")]:
        count = (agents_updated["taxonomy_index"] == cat_idx).sum()
        final_counts[cat_name] = int(count)
        logger.info("  Final count for %s: %d", cat_name, count)

    # Full taxonomy distribution
    print("\n" + "=" * 80)
    print("Updated Taxonomy Distribution (agents only):")
    print("=" * 80)
    tax_dist = agents_updated["taxonomy_index"].value_counts().sort_index()
    TAXONOMY_NAMES_MAP = {
        0: "SimpleTradingBot",
        1: "MEVSearcher",
        2: "DeFiManagementAgent",
        3: "LLMPoweredAgent",
        4: "AutonomousDAOAgent",
        5: "CrossChainBridgeAgent",
        6: "DeterministicScript",
        7: "RLTradingAgent",
    }
    for idx, count in tax_dist.items():
        name = TAXONOMY_NAMES_MAP.get(int(idx), f"Unknown({idx})")
        pct = count / len(agents_updated) * 100
        print(f"  {idx}: {name:<28} {count:>5}  ({pct:>5.1f}%)")

    # ----------------------------------------------------------
    # Phase 5: Save results JSON
    # ----------------------------------------------------------
    results = {
        "timestamp": datetime.now().isoformat(),
        "new_addresses_featurized": len(records),
        "failed_addresses": len(failed),
        "category_counts_new": cat_counts,
        "category_counts_final": final_counts,
        "c1c4_pass_rates": c1c4_pass_rates,
        "updated_parquet": str(TAXONOMY_PARQUET),
        "total_rows_in_parquet": len(updated_df),
        "examples_per_category": {},
        "failed_list": failed[:20],
    }

    for cat_idx, cat_name in [(4, "AutonomousDAOAgent"), (5, "CrossChainBridgeAgent"), (7, "RLTradingAgent")]:
        cat_rows = new_df[new_df["taxonomy_index"] == cat_idx]
        examples = []
        for addr in cat_rows.index[:5]:
            examples.append({
                "address": str(addr),
                "name": str(cat_rows.loc[addr, "name"]),
                "provenance": str(cat_rows.loc[addr, "source"]),
                "c1c4_confidence": float(cat_rows.loc[addr, "c1c4_confidence"]),
                "n_transactions": int(cat_rows.loc[addr, "n_transactions"]),
            })
        results["examples_per_category"][cat_name] = examples

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved results to %s", RESULTS_PATH)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"New addresses featurized: {len(records)}")
    print(f"Failed: {len(failed)}")
    for cat_name, count in cat_counts.items():
        print(f"  {cat_name}: {count} new")
    for cat_name, count in final_counts.items():
        print(f"  {cat_name} (final in parquet): {count}")
    print(f"Updated parquet: {TAXONOMY_PARQUET}")
    print(f"Results JSON: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
