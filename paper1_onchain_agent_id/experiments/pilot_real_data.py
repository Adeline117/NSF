"""
Paper 1 Pilot: Real Ethereum Data Feature Extraction
=====================================================
First real data experiment using Etherscan API with 6-key rotation.

Targets:
- Known agents: MEV bots, Autonolas-registered addresses
- Known humans: Vitalik.eth, whale addresses
- Compute all 4 feature groups on real transaction data
- Statistical tests to confirm feature separation on real data
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from scipy import stats
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from shared.utils.eth_utils import EtherscanClient, load_config


# ============================================================
# TARGET ADDRESSES
# ============================================================

TARGETS = {
    "agents": {
        # Known MEV bots
        "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9": "jaredfromsubway.eth (MEV)",
        "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80": "MEV bot 2",
        "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13": "jaredfromsubway v2",
        # DeFi automation / bot addresses
        "0x00000000009726632680AF5D2882e70d69d89a5C": "MEV bot (0x0000)",
        "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C": "Wintermute (market maker)",
    },
    "humans": {
        # Well-known human-operated addresses
        "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": "vitalik.eth",
        "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B": "vitalik old address",
        "0x220866B1A2219f40e72f5c628B65D54268cA3A9D": "Hayden Adams (Uniswap)",
    },
}


# ============================================================
# FEATURE EXTRACTION (real data version)
# ============================================================

def extract_temporal(txs: pd.DataFrame) -> dict:
    """Extract temporal features from real transaction data."""
    if txs.empty or len(txs) < 5:
        return {}

    timestamps = pd.to_numeric(txs["timeStamp"], errors="coerce").dropna().sort_values()
    if len(timestamps) < 5:
        return {}

    intervals = timestamps.diff().dropna().values
    hours = pd.to_datetime(timestamps, unit="s").dt.hour

    # Hour distribution entropy
    hour_counts = np.bincount(hours, minlength=24).astype(float)
    hour_probs = hour_counts / hour_counts.sum()
    hour_probs = hour_probs[hour_probs > 0]
    hour_entropy = -np.sum(hour_probs * np.log2(hour_probs))

    # Burst detection (txs within 10 seconds)
    burst_count = (intervals < 10).sum()

    return {
        "tx_count": len(txs),
        "tx_interval_mean": float(np.mean(intervals)),
        "tx_interval_std": float(np.std(intervals)),
        "tx_interval_skewness": float(stats.skew(intervals)) if len(intervals) > 2 else 0,
        "active_hour_entropy": float(hour_entropy),
        "max_possible_entropy": float(np.log2(24)),
        "night_activity_ratio": float((hours.between(0, 6)).mean()),
        "weekend_ratio": float(pd.to_datetime(timestamps, unit="s").dt.dayofweek.isin([5, 6]).mean()),
        "burst_ratio": float(burst_count / len(intervals)) if len(intervals) > 0 else 0,
    }


def extract_gas(txs: pd.DataFrame) -> dict:
    """Extract gas behavior features from real data."""
    if txs.empty:
        return {}

    gas_prices = pd.to_numeric(txs.get("gasPrice", pd.Series()), errors="coerce").dropna()
    if gas_prices.empty:
        return {}

    # Round number analysis
    def trailing_zeros(x):
        if x == 0:
            return 0
        s = str(int(x))
        return len(s) - len(s.rstrip("0"))

    trailing = gas_prices.apply(trailing_zeros)
    round_ratio = (trailing >= 9).mean()  # >=9 trailing zeros = Gwei-level rounding

    # Gas limit precision
    gas_used = pd.to_numeric(txs.get("gasUsed", pd.Series()), errors="coerce").dropna()
    gas_limit = pd.to_numeric(txs.get("gas", pd.Series()), errors="coerce").dropna()
    precision = (gas_used / gas_limit.replace(0, np.nan)).dropna()

    mean_gp = gas_prices.mean()
    cv = float(gas_prices.std() / mean_gp) if mean_gp > 0 else 0

    return {
        "gas_price_mean_gwei": float(mean_gp / 1e9),
        "gas_price_cv": cv,
        "gas_price_round_number_ratio": float(round_ratio),
        "gas_price_trailing_zeros_mean": float(trailing.mean()),
        "gas_limit_precision_mean": float(precision.mean()) if not precision.empty else 0,
        "gas_limit_precision_std": float(precision.std()) if not precision.empty else 0,
    }


def extract_interaction(txs: pd.DataFrame) -> dict:
    """Extract interaction pattern features."""
    if txs.empty:
        return {}

    to_addrs = txs["to"].dropna()
    if to_addrs.empty:
        return {}

    unique_ratio = to_addrs.nunique() / len(to_addrs)
    shares = to_addrs.value_counts(normalize=True)
    hhi = float((shares ** 2).sum())

    # Method ID diversity
    method_ids = txs["input"].apply(
        lambda x: x[:10] if isinstance(x, str) and len(x) >= 10 else "0x"
    )
    non_transfer = method_ids[method_ids != "0x"]
    method_diversity = non_transfer.nunique() / len(non_transfer) if len(non_transfer) > 0 else 0

    # Contract call ratio
    contract_calls = txs["input"].apply(
        lambda x: isinstance(x, str) and len(x) > 2 and x != "0x"
    )

    return {
        "unique_contracts_ratio": float(unique_ratio),
        "top_contract_concentration_hhi": hhi,
        "method_id_diversity": float(method_diversity),
        "unique_method_ids": int(non_transfer.nunique()) if len(non_transfer) > 0 else 0,
        "contract_call_ratio": float(contract_calls.mean()),
    }


def extract_approval(txs: pd.DataFrame) -> dict:
    """Extract token approval features."""
    if txs.empty or "input" not in txs.columns:
        return {}

    APPROVE_SIG = "0x095ea7b3"
    MAX_UINT = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

    approvals = txs[txs["input"].str.startswith(APPROVE_SIG, na=False)]
    if approvals.empty:
        return {"total_approvals": 0, "unlimited_approve_ratio": 0}

    unlimited = approvals["input"].str.contains(MAX_UINT, na=False)
    return {
        "total_approvals": len(approvals),
        "unlimited_approve_ratio": float(unlimited.mean()),
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def run_real_data_pilot():
    print("=" * 70)
    print("Paper 1: Real Ethereum Data Feature Extraction")
    print("=" * 70)

    client = EtherscanClient()
    print(f"Loaded {client.num_keys} API keys (effective rate: ~{client.num_keys * 5} calls/sec)")

    if client.num_keys == 0:
        print("[ERROR] No API keys found. Check shared/configs/config.yaml")
        return

    results = {"agents": {}, "humans": {}}

    for category, addresses in TARGETS.items():
        print(f"\n--- Processing {category} ({len(addresses)} addresses) ---")
        for addr, label in addresses.items():
            print(f"\n  [{label}] {addr[:12]}...")
            try:
                txs = client.get_normal_txs(addr, offset=2000)
                if txs.empty:
                    print(f"    No transactions found, skipping")
                    continue

                features = {
                    "label": label,
                    "address": addr,
                    "temporal": extract_temporal(txs),
                    "gas": extract_gas(txs),
                    "interaction": extract_interaction(txs),
                    "approval": extract_approval(txs),
                }

                results[category][addr] = features

                # Print key features
                t = features["temporal"]
                g = features["gas"]
                if t:
                    print(f"    Txs: {t.get('tx_count', 0)}, "
                          f"Interval: {t.get('tx_interval_mean', 0):.0f}s ± {t.get('tx_interval_std', 0):.0f}s, "
                          f"Hour entropy: {t.get('active_hour_entropy', 0):.2f}/{t.get('max_possible_entropy', 0):.2f}")
                if g:
                    print(f"    Gas CV: {g.get('gas_price_cv', 0):.4f}, "
                          f"Round ratio: {g.get('gas_price_round_number_ratio', 0):.3f}, "
                          f"Precision: {g.get('gas_limit_precision_mean', 0):.3f}")
            except Exception as e:
                print(f"    Error: {e}")

    # ============================================================
    # STATISTICAL COMPARISON
    # ============================================================
    print(f"\n{'='*70}")
    print("Statistical Comparison: Agents vs Humans")
    print("="*70)

    # Collect comparable features
    comparison_features = [
        ("temporal", "tx_interval_mean"),
        ("temporal", "active_hour_entropy"),
        ("temporal", "night_activity_ratio"),
        ("temporal", "burst_ratio"),
        ("gas", "gas_price_cv"),
        ("gas", "gas_price_round_number_ratio"),
        ("gas", "gas_limit_precision_mean"),
        ("interaction", "unique_contracts_ratio"),
        ("interaction", "contract_call_ratio"),
    ]

    print(f"\n  {'Feature':<35} {'Agent Mean':>12} {'Human Mean':>12} {'Δ':>10}")
    print("  " + "-" * 75)

    for group, feat in comparison_features:
        agent_vals = [
            results["agents"][a][group][feat]
            for a in results["agents"]
            if feat in results["agents"][a].get(group, {})
        ]
        human_vals = [
            results["humans"][a][group][feat]
            for a in results["humans"]
            if feat in results["humans"][a].get(group, {})
        ]

        if agent_vals and human_vals:
            a_mean = np.mean(agent_vals)
            h_mean = np.mean(human_vals)
            delta = a_mean - h_mean
            print(f"  {group}.{feat:<28} {a_mean:>12.4f} {h_mean:>12.4f} {delta:>+10.4f}")

    print(f"\n  Total API calls: {client._total_calls}")

    # Save results
    output_path = Path(__file__).parent / "pilot_real_data_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {output_path}")

    return results


if __name__ == "__main__":
    run_real_data_pilot()
