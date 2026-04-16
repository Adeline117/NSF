#!/usr/bin/env python3
"""
Paper 0: Advanced Agent Features — Reaction Time & Strategy Coherence
======================================================================
Two new feature families that exploit theoretically strong signals for
distinguishing LLM-powered agents from bots and humans:

Feature Family 5: Event-to-Transaction Reaction Time
  - LLMs have inference latency (1-5s floor), bots react <1s, humans >10s
  - Uses block timestamps and transactionIndex as coarse timing proxy
  - Features: reaction_time_p10, reaction_time_median, reaction_time_cv,
              reaction_time_p10_to_median_ratio

Feature Family 6: Strategy Coherence
  - LLMs plan multi-step strategies where tx_n output feeds tx_{n+1} input
  - Measures token flow continuity, contract dependency chains, action diversity
  - Features: token_flow_continuity_rate, strategy_chain_max_length,
              unique_action_sequence_ratio, action_diversity_within_strategy

Sampling: 200 addresses proportional to class sizes, minimum 10 per class.

Outputs:
  - advanced_agent_features_results.json
"""

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.eth_utils import EtherscanClient

# ── Paths ─────────────────────────────────────────────────────────────
FEATURES_PARQUET = (
    PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
)
RAW_DIR = PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "raw"
AI_FEATURES_JSON = (
    PROJECT_ROOT / "paper3_ai_sybil" / "experiments" / "real_ai_features.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "advanced_agent_features_results.json"
)
ERC20_CACHE_DIR = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments" / "erc20_cache"
)

# ── Constants ─────────────────────────────────────────────────────────
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

API_KEYS = [
    "413E91H5PDDZYTVRY8AMK4D7FVTE65Z989",
    "EBHB3GSMY5AWS5AR3HYBNJZ49R6WA2NCAY",
    "RZ9XC5V6ZZQMN25J3C63AUXWMWA1UC78R6B",
]

# Known DeFi protocol method IDs (4-byte selectors)
DEFI_METHOD_IDS = {
    "0x095ea7b3",  # approve
    "0x38ed1739",  # swapExactTokensForTokens (Uniswap V2)
    "0x8803dbee",  # swapTokensForExactTokens (Uniswap V2)
    "0x7ff36ab5",  # swapExactETHForTokens
    "0x18cbafe5",  # swapExactTokensForETH
    "0x5c11d795",  # swapExactTokensForTokensSupportingFeeOnTransferTokens
    "0xd46b02c3",  # Uniswap V2 Router removeLiquidity variant
    "0xe8e33700",  # addLiquidity
    "0xf305d719",  # addLiquidityETH
    "0x414bf389",  # Uniswap V3 exactInputSingle
    "0xc04b8d59",  # Uniswap V3 exactInput
    "0x5ae401dc",  # Uniswap V3 Router multicall
    "0xac9650d8",  # multicall (generic)
    "0xa9059cbb",  # transfer
    "0x23b872dd",  # transferFrom
    "0x2e1a7d4d",  # withdraw (WETH)
    "0xd0e30db0",  # deposit (WETH)
    "0xe449022e",  # 1inch uniswapV3Swap
    "0x0502b1c5",  # 1inch clipperSwap
    "0x12aa3caf",  # 1inch swap
    "0xfb3bdb41",  # swapETHForExactTokens
    "0xb6f9de95",  # swapExactETHForTokensSupportingFeeOnTransferTokens
    "0x1249c58b",  # mint
    "0xa0712d68",  # mint(uint256)
    "0x2eb2c2d6",  # safeBatchTransferFrom (ERC1155)
    "0xab834bab",  # atomicMatch_ (OpenSea)
    "0x3593564c",  # Uniswap Universal Router execute
}

# Ethereum block time ~12 seconds post-merge
ETH_BLOCK_TIME = 12.0

SAMPLE_SIZE = 200
MIN_PER_CLASS = 10
SEED = 42
INTER_API_DELAY = 0.25


def select_sample(df, n_total=SAMPLE_SIZE, min_per_class=MIN_PER_CLASS, seed=SEED):
    """Select proportional stratified sample with minimum per class."""
    rng = np.random.RandomState(seed)
    agents = df[df["label"] == 1].copy()

    class_counts = agents["taxonomy_index"].value_counts().sort_index()
    total_agents = len(agents)

    # Compute proportional allocation with minimum guarantee
    allocation = {}
    remaining = n_total
    for cls, count in class_counts.items():
        alloc = max(min_per_class, int(round(n_total * count / total_agents)))
        alloc = min(alloc, count)  # can't sample more than available
        allocation[cls] = alloc
        remaining -= alloc

    # If over-allocated, trim from largest classes
    while sum(allocation.values()) > n_total:
        largest = max(allocation, key=allocation.get)
        if allocation[largest] > min_per_class:
            allocation[largest] -= 1

    # If under-allocated, add to largest classes
    while sum(allocation.values()) < n_total:
        largest = max(class_counts.to_dict(), key=class_counts.to_dict().get)
        if allocation[largest] < class_counts[largest]:
            allocation[largest] += 1

    print(f"\nSampling {sum(allocation.values())} addresses:")
    sampled = []
    for cls in sorted(allocation):
        cls_addrs = agents[agents["taxonomy_index"] == cls].index.tolist()
        n = min(allocation[cls], len(cls_addrs))
        chosen = rng.choice(cls_addrs, size=n, replace=False).tolist()
        sampled.extend([(addr, int(cls)) for addr in chosen])
        print(f"  {TAXONOMY_NAMES[cls]:<25} n={n} (of {len(cls_addrs)})")

    return sampled


def load_raw_txs(address):
    """Load cached raw transactions for an address."""
    # Try exact case match first, then lowercase
    path = RAW_DIR / f"{address}.parquet"
    if not path.exists():
        path = RAW_DIR / f"{address.lower()}.parquet"
    if not path.exists():
        # Try case-insensitive search
        for f in RAW_DIR.iterdir():
            if f.stem.lower() == address.lower() and f.suffix == ".parquet":
                path = f
                break
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    return df


def load_erc20_cache(address):
    """Load cached ERC20 transfers."""
    path = ERC20_CACHE_DIR / f"{address.lower()}.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        if data:
            return pd.DataFrame(data)
    return None


def save_erc20_cache(address, df):
    """Save ERC20 transfers to cache."""
    ERC20_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = ERC20_CACHE_DIR / f"{address.lower()}.json"
    if df is not None and not df.empty:
        with open(path, "w") as f:
            json.dump(df.to_dict(orient="records"), f)
    else:
        with open(path, "w") as f:
            json.dump([], f)


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 5: Reaction Time
# ══════════════════════════════════════════════════════════════════════

def extract_reaction_time_features(txs_df):
    """
    Extract reaction time features from transaction data.

    The key insight: for each transaction, compute how quickly after the
    previous block it was included. This is approximated by:
      reaction_time ≈ (tx_timestamp - (tx_timestamp - ETH_BLOCK_TIME))
    But more precisely, we use transactionIndex as a within-block ordering
    proxy: lower index = arrived earlier in the block.

    For DeFi transactions specifically, the reaction time matters most
    because they respond to on-chain price movements.
    """
    if txs_df.empty or len(txs_df) < 5:
        return None

    df = txs_df.copy()
    df["timeStamp"] = pd.to_numeric(df["timeStamp"], errors="coerce")
    df["blockNumber"] = pd.to_numeric(df["blockNumber"], errors="coerce")
    df["transactionIndex"] = pd.to_numeric(df["transactionIndex"], errors="coerce")
    df = df.sort_values("timeStamp").reset_index(drop=True)

    # Filter to outgoing transactions (where this address is the sender)
    # Use all txs since raw data is fetched per-address
    if "from" in df.columns:
        # Keep rows where 'from' matches the most frequent sender (the address itself)
        from_counts = df["from"].str.lower().value_counts()
        if len(from_counts) > 0:
            primary_addr = from_counts.index[0]
            df_out = df[df["from"].str.lower() == primary_addr].copy()
        else:
            df_out = df.copy()
    else:
        df_out = df.copy()

    if len(df_out) < 5:
        df_out = df.copy()

    # ── Approach 1: Inter-block timing ────────────────────────────────
    # For each consecutive pair of blocks, compute time gap
    df_out = df_out.drop_duplicates(subset=["hash"]).sort_values("timeStamp")
    timestamps = df_out["timeStamp"].values.astype(float)
    block_nums = df_out["blockNumber"].values.astype(float)
    tx_indices = df_out["transactionIndex"].values.astype(float)
    method_ids = df_out["methodId"].values if "methodId" in df_out.columns else []

    # Identify DeFi transactions
    defi_mask = np.array([
        str(m).lower() in DEFI_METHOD_IDS for m in method_ids
    ]) if len(method_ids) > 0 else np.ones(len(df_out), dtype=bool)

    # ── Reaction times: time since previous block ────────────────────
    # For each tx, compute the gap to the previous tx in a different block
    reaction_times = []
    for i in range(1, len(timestamps)):
        if block_nums[i] != block_nums[i - 1]:
            # Different block: reaction_time = time between blocks
            dt = timestamps[i] - timestamps[i - 1]
            if 0 < dt < 86400:  # sanity: < 1 day
                reaction_times.append(dt)

    # ── Position-adjusted reaction time ──────────────────────────────
    # Use transactionIndex / max_tx_in_block as a fractional position
    # Lower position = faster arrival
    position_fractions = []
    for i in range(len(tx_indices)):
        idx = tx_indices[i]
        if not np.isnan(idx) and idx >= 0:
            # Normalize: assume ~200 txs per block on average
            position_fractions.append(idx / 200.0)

    # ── DeFi-specific reaction times ─────────────────────────────────
    defi_reaction_times = []
    for i in range(1, len(timestamps)):
        if defi_mask[i] and block_nums[i] != block_nums[i - 1]:
            dt = timestamps[i] - timestamps[i - 1]
            if 0 < dt < 86400:
                defi_reaction_times.append(dt)

    # ── Compute features ─────────────────────────────────────────────
    features = {}

    if len(reaction_times) >= 3:
        rt = np.array(reaction_times)
        features["reaction_time_p10"] = float(np.percentile(rt, 10))
        features["reaction_time_median"] = float(np.median(rt))
        features["reaction_time_cv"] = float(np.std(rt) / np.mean(rt)) if np.mean(rt) > 0 else 0.0
        p10 = np.percentile(rt, 10)
        med = np.median(rt)
        features["reaction_time_p10_to_median_ratio"] = float(p10 / med) if med > 0 else 0.0
    else:
        features["reaction_time_p10"] = np.nan
        features["reaction_time_median"] = np.nan
        features["reaction_time_cv"] = np.nan
        features["reaction_time_p10_to_median_ratio"] = np.nan

    # Position-based features (finer granularity)
    if len(position_fractions) >= 5:
        pf = np.array(position_fractions)
        features["block_position_p10"] = float(np.percentile(pf, 10))
        features["block_position_median"] = float(np.median(pf))
        features["block_position_cv"] = float(np.std(pf) / np.mean(pf)) if np.mean(pf) > 0 else 0.0
    else:
        features["block_position_p10"] = np.nan
        features["block_position_median"] = np.nan
        features["block_position_cv"] = np.nan

    # DeFi-specific reaction time
    if len(defi_reaction_times) >= 3:
        drt = np.array(defi_reaction_times)
        features["defi_reaction_time_p10"] = float(np.percentile(drt, 10))
        features["defi_reaction_time_median"] = float(np.median(drt))
    else:
        features["defi_reaction_time_p10"] = np.nan
        features["defi_reaction_time_median"] = np.nan

    # Burst detection: fraction of consecutive txs in same block
    same_block_count = 0
    for i in range(1, len(block_nums)):
        if block_nums[i] == block_nums[i - 1]:
            same_block_count += 1
    features["same_block_fraction"] = float(same_block_count / max(len(block_nums) - 1, 1))

    return features


# ══════════════════════════════════════════════════════════════════════
# FEATURE FAMILY 6: Strategy Coherence
# ══════════════════════════════════════════════════════════════════════

def extract_strategy_coherence_features(txs_df, erc20_df=None):
    """
    Extract strategy coherence features.

    The key insight: LLM agents plan multi-step strategies where the output
    of tx_n feeds the input of tx_{n+1}. This creates token flow continuity
    and contract dependency patterns that differ from both bots (repetitive)
    and humans (inconsistent).
    """
    if txs_df.empty or len(txs_df) < 5:
        return None

    df = txs_df.copy()
    df["timeStamp"] = pd.to_numeric(df["timeStamp"], errors="coerce")
    df = df.sort_values("timeStamp").reset_index(drop=True)

    # Focus on outgoing transactions
    if "from" in df.columns:
        from_counts = df["from"].str.lower().value_counts()
        if len(from_counts) > 0:
            primary_addr = from_counts.index[0]
            df_out = df[df["from"].str.lower() == primary_addr].copy()
        else:
            df_out = df.copy()
    else:
        df_out = df.copy()

    if len(df_out) < 5:
        df_out = df.copy()

    df_out = df_out.drop_duplicates(subset=["hash"]).sort_values("timeStamp").reset_index(drop=True)

    features = {}

    # ── Token Flow Continuity (from ERC20 transfers) ─────────────────
    token_flow_continuity = 0
    token_flow_pairs = 0
    max_chain_length = 0

    if erc20_df is not None and not erc20_df.empty:
        erc20 = erc20_df.copy()
        erc20["timeStamp"] = pd.to_numeric(erc20["timeStamp"], errors="coerce")
        erc20 = erc20.sort_values("timeStamp").reset_index(drop=True)

        if "from" in erc20.columns and "to" in erc20.columns:
            from_counts = erc20["from"].str.lower().value_counts()
            if len(from_counts) > 0:
                primary_addr = from_counts.index[0]
            else:
                primary_addr = ""

            # Group ERC20 transfers by transaction hash
            if "hash" in erc20.columns:
                tx_hashes = erc20["hash"].unique()

                # For each tx, find tokens received and tokens sent
                tx_tokens_received = {}  # hash -> set of token contract addresses
                tx_tokens_sent = {}      # hash -> set of token contract addresses

                for _, row in erc20.iterrows():
                    h = str(row.get("hash", ""))
                    token_addr = str(row.get("contractAddress", "")).lower()
                    from_addr = str(row.get("from", "")).lower()
                    to_addr = str(row.get("to", "")).lower()

                    if to_addr == primary_addr:
                        tx_tokens_received.setdefault(h, set()).add(token_addr)
                    if from_addr == primary_addr:
                        tx_tokens_sent.setdefault(h, set()).add(token_addr)

                # Get unique tx hashes in order (by first appearance timestamp)
                hash_to_ts = {}
                for _, row in erc20.iterrows():
                    h = str(row.get("hash", ""))
                    ts = row.get("timeStamp", 0)
                    if h not in hash_to_ts:
                        hash_to_ts[h] = ts
                ordered_hashes = sorted(hash_to_ts.keys(), key=lambda h: hash_to_ts[h])

                # Check consecutive tx pairs for token flow continuity
                chain_length = 1
                for i in range(1, len(ordered_hashes)):
                    prev_h = ordered_hashes[i - 1]
                    curr_h = ordered_hashes[i]

                    received_prev = tx_tokens_received.get(prev_h, set())
                    sent_curr = tx_tokens_sent.get(curr_h, set())

                    token_flow_pairs += 1
                    if received_prev & sent_curr:  # intersection: token received then sent
                        token_flow_continuity += 1
                        chain_length += 1
                    else:
                        max_chain_length = max(max_chain_length, chain_length)
                        chain_length = 1
                max_chain_length = max(max_chain_length, chain_length)

    features["token_flow_continuity_rate"] = (
        float(token_flow_continuity / max(token_flow_pairs, 1))
    )
    features["strategy_chain_max_length"] = int(max_chain_length)

    # ── Contract Dependency: consecutive txs targeting same/related contracts ──
    if "to" in df_out.columns:
        to_addrs = df_out["to"].str.lower().values
        contract_dep_count = 0
        for i in range(1, len(to_addrs)):
            if str(to_addrs[i]) == str(to_addrs[i - 1]) and str(to_addrs[i]) != "":
                contract_dep_count += 1
        features["contract_dependency_rate"] = float(
            contract_dep_count / max(len(to_addrs) - 1, 1)
        )
    else:
        features["contract_dependency_rate"] = np.nan

    # ── Action Sequence Analysis (methodId n-grams) ──────────────────
    method_ids = df_out["methodId"].values if "methodId" in df_out.columns else []
    method_ids = [str(m) for m in method_ids if str(m) not in ("", "nan", "None")]

    if len(method_ids) >= 5:
        # 3-gram analysis
        trigrams = []
        for i in range(len(method_ids) - 2):
            trigram = (method_ids[i], method_ids[i + 1], method_ids[i + 2])
            trigrams.append(trigram)

        if len(trigrams) > 0:
            trigram_counts = Counter(trigrams)
            unique_trigrams = sum(1 for c in trigram_counts.values() if c == 1)
            features["unique_action_sequence_ratio"] = float(
                unique_trigrams / len(trigram_counts)
            )

            # Action diversity within strategy chains
            # A "strategy chain" is a run of non-repeated method IDs
            chain_diversities = []
            current_chain = {method_ids[0]}
            for i in range(1, len(method_ids)):
                if method_ids[i] != method_ids[i - 1]:
                    current_chain.add(method_ids[i])
                else:
                    if len(current_chain) > 1:
                        chain_diversities.append(len(current_chain))
                    current_chain = {method_ids[i]}
            if len(current_chain) > 1:
                chain_diversities.append(len(current_chain))

            features["action_diversity_within_strategy"] = (
                float(np.mean(chain_diversities)) if chain_diversities else 1.0
            )
        else:
            features["unique_action_sequence_ratio"] = np.nan
            features["action_diversity_within_strategy"] = np.nan

        # Method entropy (diversity of actions)
        method_counter = Counter(method_ids)
        total_methods = sum(method_counter.values())
        probs = np.array([c / total_methods for c in method_counter.values()])
        features["method_sequence_entropy"] = float(-np.sum(probs * np.log2(probs + 1e-12)))

        # Repetition ratio: how often the same method is used consecutively
        repeats = sum(1 for i in range(1, len(method_ids)) if method_ids[i] == method_ids[i - 1])
        features["method_repetition_ratio"] = float(repeats / max(len(method_ids) - 1, 1))

    else:
        features["unique_action_sequence_ratio"] = np.nan
        features["action_diversity_within_strategy"] = np.nan
        features["method_sequence_entropy"] = np.nan
        features["method_repetition_ratio"] = np.nan

    return features


# ══════════════════════════════════════════════════════════════════════
# ANALYSIS: Cohen's d, distributions, classification
# ══════════════════════════════════════════════════════════════════════

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = np.nanmean(group1), np.nanmean(group2)
    s1, s2 = np.nanstd(group1, ddof=1), np.nanstd(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return float("nan")
    return float((m1 - m2) / pooled_std)


def per_class_stats(feat_dict, feature_names, sampled_addresses):
    """Compute per-class distribution summary stats."""
    class_features = {}
    for addr, cls in sampled_addresses:
        cls_name = TAXONOMY_NAMES[cls]
        if cls_name not in class_features:
            class_features[cls_name] = {fn: [] for fn in feature_names}
        if addr in feat_dict:
            for fn in feature_names:
                val = feat_dict[addr].get(fn, np.nan)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    class_features[cls_name][fn].append(val)

    results = {}
    for cls_name in sorted(class_features):
        results[cls_name] = {}
        for fn in feature_names:
            vals = class_features[cls_name][fn]
            if len(vals) >= 2:
                results[cls_name][fn] = {
                    "n": len(vals),
                    "mean": round(float(np.mean(vals)), 6),
                    "std": round(float(np.std(vals)), 6),
                    "median": round(float(np.median(vals)), 6),
                    "p10": round(float(np.percentile(vals, 10)), 6),
                    "p25": round(float(np.percentile(vals, 25)), 6),
                    "p75": round(float(np.percentile(vals, 75)), 6),
                    "p90": round(float(np.percentile(vals, 90)), 6),
                }
            else:
                results[cls_name][fn] = {
                    "n": len(vals),
                    "mean": float(vals[0]) if vals else None,
                }
    return results


def compute_effect_sizes(feat_dict, feature_names, sampled_addresses):
    """Compute Cohen's d between LLMPoweredAgent and every other class."""
    # Group by class
    class_vals = {}
    for addr, cls in sampled_addresses:
        cls_name = TAXONOMY_NAMES[cls]
        if cls_name not in class_vals:
            class_vals[cls_name] = {fn: [] for fn in feature_names}
        if addr in feat_dict:
            for fn in feature_names:
                val = feat_dict[addr].get(fn, np.nan)
                if val is not None and not (isinstance(val, float) and np.isnan(val)):
                    class_vals[cls_name][fn].append(val)

    results = {}
    target_class = "LLMPoweredAgent"
    comparison_classes = ["DeFiManagementAgent", "MEVSearcher", "SimpleTradingBot",
                          "DeterministicScript", "AutonomousDAOAgent",
                          "CrossChainBridgeAgent", "RLTradingAgent"]

    for comp_class in comparison_classes:
        if target_class not in class_vals or comp_class not in class_vals:
            continue
        results[f"LLMPoweredAgent_vs_{comp_class}"] = {}
        for fn in feature_names:
            g1 = class_vals[target_class].get(fn, [])
            g2 = class_vals[comp_class].get(fn, [])
            d = cohens_d(g1, g2)
            results[f"LLMPoweredAgent_vs_{comp_class}"][fn] = round(d, 4)

    return results


def classification_experiment(feat_dict, feature_names, sampled_addresses, df_full):
    """
    Run classification with the new features added to the 31-feature baseline.
    Uses the full 2744-agent dataset: imputes NaN for addresses not in the sample.
    """
    from sklearn.base import clone
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, classification_report, f1_score,
    )
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    # ── Load 31-feature baseline ─────────────────────────────────────
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

    agents = df_full[df_full["label"] == 1].copy()

    # Load AI features
    with open(AI_FEATURES_JSON) as f:
        ai_raw = json.load(f)
    ai_rows = []
    for addr, feats in ai_raw["per_address"].items():
        row = {"address": addr}
        for col in AI_8:
            row[col] = feats.get(col, np.nan)
        ai_rows.append(row)
    df_ai = pd.DataFrame(ai_rows).set_index("address")
    agents = agents.join(df_ai[AI_8], how="left")

    # Add new advanced features
    new_feat_rows = []
    for addr in agents.index:
        row = {"address": addr}
        if addr in feat_dict:
            for fn in feature_names:
                row[fn] = feat_dict[addr].get(fn, np.nan)
        else:
            for fn in feature_names:
                row[fn] = np.nan
        new_feat_rows.append(row)
    df_new = pd.DataFrame(new_feat_rows).set_index("address")
    agents = agents.join(df_new[feature_names], how="left")

    ALL_FEATURES = ALL_31 + feature_names
    n_features_new = len(ALL_FEATURES)

    X = agents[ALL_FEATURES].values.astype(float)
    y = agents["taxonomy_index"].values.astype(int)

    # Impute and clip
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)

    # Replace any remaining NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # ── 10-fold CV ───────────────────────────────────────────────────
    models = {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, random_state=SEED,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=SEED, n_jobs=-1,
        ),
    }

    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    n_folds = min(10, min_count)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    classes_sorted = sorted(set(y.tolist()))

    model_results = {}
    for model_name, model_template in models.items():
        fold_accs = []
        fold_f1_macro = []
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
            all_y_true.extend(y[te_idx].tolist())
            all_y_pred.extend(y_pred.tolist())

        y_true_arr = np.array(all_y_true)
        y_pred_arr = np.array(all_y_pred)

        report = classification_report(
            y_true_arr, y_pred_arr,
            labels=classes_sorted,
            target_names=[TAXONOMY_NAMES.get(int(c), f"C{c}") for c in classes_sorted],
            output_dict=True, zero_division=0,
        )

        # Feature importance (train on full data)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf_full = clone(model_template)
        clf_full.fit(Xs, y)

        fi = {}
        if hasattr(clf_full, "feature_importances_"):
            sorted_idx = np.argsort(clf_full.feature_importances_)[::-1]
            fi = {
                ALL_FEATURES[i]: round(float(clf_full.feature_importances_[i]), 6)
                for i in sorted_idx
            }

        model_results[model_name] = {
            "accuracy_mean": round(float(np.mean(fold_accs)), 4),
            "accuracy_std": round(float(np.std(fold_accs)), 4),
            "f1_macro_mean": round(float(np.mean(fold_f1_macro)), 4),
            "f1_macro_std": round(float(np.std(fold_f1_macro)), 4),
            "n_features": n_features_new,
            "n_folds": n_folds,
            "per_class_report": report,
            "feature_importance": fi,
        }

    return model_results


def main():
    print("=" * 80)
    print("Paper 0: Advanced Agent Features — Reaction Time & Strategy Coherence")
    print("  Feature Family 5: Event-to-Transaction Reaction Time")
    print("  Feature Family 6: Strategy Coherence")
    print("=" * 80)

    # ── Load dataset and select sample ───────────────────────────────
    df = pd.read_parquet(FEATURES_PARQUET)
    sampled = select_sample(df)

    # Initialize EtherscanClient for ERC20 fetching
    client = EtherscanClient(api_keys=API_KEYS)
    print(f"\nEtherscan client initialized with {client.num_keys} API keys")

    # ── Extract features for each sampled address ────────────────────
    all_features = {}
    n_success = 0
    n_erc20_fetched = 0
    n_erc20_cached = 0

    NEW_FEATURE_NAMES = [
        # Family 5: Reaction Time
        "reaction_time_p10",
        "reaction_time_median",
        "reaction_time_cv",
        "reaction_time_p10_to_median_ratio",
        "block_position_p10",
        "block_position_median",
        "block_position_cv",
        "defi_reaction_time_p10",
        "defi_reaction_time_median",
        "same_block_fraction",
        # Family 6: Strategy Coherence
        "token_flow_continuity_rate",
        "strategy_chain_max_length",
        "contract_dependency_rate",
        "unique_action_sequence_ratio",
        "action_diversity_within_strategy",
        "method_sequence_entropy",
        "method_repetition_ratio",
    ]

    print(f"\nProcessing {len(sampled)} addresses...")
    for idx, (addr, cls) in enumerate(sampled):
        if idx % 20 == 0:
            print(f"  [{idx+1}/{len(sampled)}] Processing {TAXONOMY_NAMES[cls]}...")

        # Load cached raw transactions
        txs_df = load_raw_txs(addr)
        if txs_df.empty:
            print(f"    SKIP {addr[:10]}... — no raw txs cached")
            continue

        # Load or fetch ERC20 transfers
        erc20_df = load_erc20_cache(addr)
        if erc20_df is None:
            # Fetch from Etherscan
            try:
                time.sleep(INTER_API_DELAY)
                erc20_df = client.get_erc20_transfers(addr)
                save_erc20_cache(addr, erc20_df)
                n_erc20_fetched += 1
            except Exception as e:
                print(f"    WARN: ERC20 fetch failed for {addr[:10]}...: {e}")
                erc20_df = pd.DataFrame()
                save_erc20_cache(addr, erc20_df)
        else:
            n_erc20_cached += 1

        # Extract Feature Family 5: Reaction Time
        rt_features = extract_reaction_time_features(txs_df)

        # Extract Feature Family 6: Strategy Coherence
        sc_features = extract_strategy_coherence_features(txs_df, erc20_df)

        if rt_features is not None or sc_features is not None:
            combined = {}
            if rt_features:
                combined.update(rt_features)
            if sc_features:
                combined.update(sc_features)
            all_features[addr] = combined
            n_success += 1

    print(f"\nFeature extraction complete:")
    print(f"  Addresses processed: {n_success}/{len(sampled)}")
    print(f"  ERC20 fetched (API): {n_erc20_fetched}")
    print(f"  ERC20 from cache:    {n_erc20_cached}")

    # ── Per-class distribution stats ─────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Per-Class Distribution Summary")
    print("=" * 60)

    dist_stats = per_class_stats(all_features, NEW_FEATURE_NAMES, sampled)

    # Print summary for key features
    key_features = [
        "reaction_time_p10", "reaction_time_median", "reaction_time_cv",
        "token_flow_continuity_rate", "strategy_chain_max_length",
        "unique_action_sequence_ratio",
    ]
    for fn in key_features:
        print(f"\n  {fn}:")
        for cls_name in sorted(dist_stats):
            stats = dist_stats[cls_name].get(fn, {})
            if "median" in stats:
                print(f"    {cls_name:<25} median={stats['median']:8.3f}  "
                      f"p10={stats['p10']:8.3f}  p90={stats['p90']:8.3f}  "
                      f"(n={stats['n']})")
            elif "mean" in stats and stats.get("mean") is not None:
                print(f"    {cls_name:<25} mean={stats['mean']:8.3f} (n={stats.get('n', '?')})")

    # ── Cohen's d effect sizes ───────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("Cohen's d Effect Sizes: LLMPoweredAgent vs Others")
    print("=" * 60)

    effect_sizes = compute_effect_sizes(all_features, NEW_FEATURE_NAMES, sampled)

    for comparison, effects in sorted(effect_sizes.items()):
        print(f"\n  {comparison}:")
        for fn in sorted(effects, key=lambda x: abs(effects[x]) if not np.isnan(effects[x]) else 0, reverse=True):
            d = effects[fn]
            size_label = (
                "LARGE" if abs(d) >= 0.8 else
                "MEDIUM" if abs(d) >= 0.5 else
                "SMALL" if abs(d) >= 0.2 else
                "negligible"
            ) if not np.isnan(d) else "N/A"
            print(f"    {fn:<40} d={d:+.3f}  [{size_label}]")

    # ── Classification with new features ─────────────────────────────
    print(f"\n{'=' * 60}")
    print("Classification: 31-Feature Baseline + New Advanced Features")
    print("=" * 60)

    clf_results = classification_experiment(
        all_features, NEW_FEATURE_NAMES, sampled, df
    )

    # 31-feature baseline numbers (from previous run)
    baseline_31 = {
        "accuracy": 0.9526,
        "f1_macro": 0.6869,
        "LLMPoweredAgent_f1": 0.4375,
    }

    for model_name, res in clf_results.items():
        print(f"\n  {model_name}:")
        print(f"    Accuracy: {res['accuracy_mean']:.4f} +/- {res['accuracy_std']:.4f}")
        print(f"    F1-macro: {res['f1_macro_mean']:.4f} +/- {res['f1_macro_std']:.4f}")
        print(f"    N features: {res['n_features']}")

        report = res["per_class_report"]
        print(f"    Per-class F1:")
        for cls_name in sorted(TAXONOMY_NAMES.values()):
            if cls_name in report:
                f1 = report[cls_name]["f1-score"]
                print(f"      {cls_name:<25} F1={f1:.4f}")

        # Top new features by importance
        fi = res.get("feature_importance", {})
        new_fi = {k: v for k, v in fi.items() if k in NEW_FEATURE_NAMES}
        if new_fi:
            print(f"    Top new feature importances:")
            for rank, (feat, imp) in enumerate(sorted(new_fi.items(), key=lambda x: -x[1])):
                if rank >= 5:
                    break
                print(f"      {feat:<40} {imp:.6f}")

    # ── Comparison vs 31-feature baseline ────────────────────────────
    print(f"\n{'=' * 60}")
    print("COMPARISON: 31-Feature Baseline vs 31+17 Advanced Features")
    print("=" * 60)

    gbm_new = clf_results.get("GradientBoosting", {})
    gbm_report = gbm_new.get("per_class_report", {})
    new_metrics = {
        "accuracy": gbm_new.get("accuracy_mean", 0),
        "f1_macro": gbm_new.get("f1_macro_mean", 0),
        "LLMPoweredAgent_f1": gbm_report.get("LLMPoweredAgent", {}).get("f1-score", 0),
    }

    for metric in baseline_31:
        b = baseline_31[metric]
        n = new_metrics.get(metric, 0)
        delta = n - b
        arrow = "+" if delta > 0 else ""
        print(f"  {metric:<30} {b:.4f} -> {n:.4f} ({arrow}{delta:.4f})")

    # ── Save results ─────────────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Advanced agent features: reaction time (Family 5) + "
            "strategy coherence (Family 6) — extracted from 200-address "
            "stratified sample with ERC20 transfer data"
        ),
        "sample_size": len(sampled),
        "n_successful": n_success,
        "n_erc20_api_fetched": n_erc20_fetched,
        "n_erc20_cached": n_erc20_cached,
        "feature_names": NEW_FEATURE_NAMES,
        "n_new_features": len(NEW_FEATURE_NAMES),
        "sample_allocation": {
            TAXONOMY_NAMES[cls]: sum(1 for _, c in sampled if c == cls)
            for cls in sorted(set(c for _, c in sampled))
        },
        "per_class_distributions": dist_stats,
        "effect_sizes_cohens_d": effect_sizes,
        "classification": {
            "baseline_31feat": baseline_31,
            "new_48feat": {
                "accuracy": gbm_new.get("accuracy_mean", 0),
                "f1_macro": gbm_new.get("f1_macro_mean", 0),
                "per_class_f1": {
                    cls_name: round(gbm_report.get(cls_name, {}).get("f1-score", 0), 4)
                    for cls_name in TAXONOMY_NAMES.values()
                },
            },
            "delta": {
                k: round(new_metrics.get(k, 0) - baseline_31[k], 4)
                for k in baseline_31
            },
            "model_results": clf_results,
        },
        "per_address_features": {
            addr: feats for addr, feats in all_features.items()
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
