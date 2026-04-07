"""
Paper 1: Large-Scale Real Data Collection (50+ addresses)
==========================================================
Collects Ethereum transaction data for a curated set of known agent and
human addresses, extracts all 23 behavioral features using the production
pipeline, and runs preliminary statistical analysis.

Pipeline:
1. Load API keys from shared/configs/config.yaml (6-key rotation)
2. For each address, fetch up to 5000 transactions via paginated API calls
3. Save raw transactions to data/raw/{address}.parquet (checkpoint)
4. Extract 23 features using FeaturePipeline
5. Save combined features to data/features.parquet
6. Run statistical analysis (Mann-Whitney U, individual AUCs)
7. Save analysis to experiments/large_scale_results.json

Checkpointing: addresses with existing raw parquet files are skipped.
"""

import os
import sys
import json
import time
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.eth_utils import EtherscanClient, load_config
from paper1_onchain_agent_id.features.feature_pipeline import (
    FeaturePipeline,
    FeatureConfig,
    FEATURE_NAMES,
    extract_temporal_features,
    extract_gas_features,
    extract_interaction_features,
    extract_approval_security_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==================================================================
# TARGET ADDRESSES
# ==================================================================

AGENT_ADDRESSES = {
    # --- MEV Bots (verified EOAs) ---
    "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9": "jaredfromsubway.eth",
    "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80": "MEV bot",
    "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13": "jaredfromsubway v2",
    "0x51C72848c68a965f66FA7a88855F9f7784502a7F": "MEV bot 3",
    "0x3B17056cc4439c61ceA41Fe1c9f517Af75A978F7": "MEV bot 4",
    "0x00000000003b3cc22aF3aE1EAc0440BcEe416B40": "MEV bot 5",
    "0x000000000000cd17345801aa8147b8D3950260FF": "MEV bot 6",
    "0x00000000009726632680AF5D2882e70d69d89a5C": "MEV bot 7",
    "0x98C3d3183C4b8A650614ad179A1a98be0a8d6B8E": "MEV bot 8",
    "0x6F1cDea15B82C4fC0b8ABBc1Fa51B17B3409dACf": "MEV bot 9",
    "0xE8c060F8052E07423f71D445277c61AC5138A2e5": "Flashbots builder",

    # --- Market Makers (verified EOAs) ---
    "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C": "Wintermute",
    "0x280027dd00eE0050d3F9d168EFD6B40090009246": "Wintermute 2",
    "0xDBF5E9c5206d0dB70a90108bf936DA60221dC080": "Wintermute 3",
    "0x9507c04B10486547584C37bCBd931B437623E531": "Jump Trading",
    "0xe8c19dB00287e3536075114B2c8C5B0089b8F6Db": "DWF Labs",

    # --- DeFi Automation Bot EOAs ---
    "0xFa4FC4ec2F81A4897743C5b4f45907c02CE06199": "1inch resolver bot",

    # REMOVED: Smart contracts (NOT EOAs, cannot be agents):
    #   - 0x1111111254EEB25477B68fb85Ed929f73A960582 (1inch v5 router)
    #   - 0xDef1C0ded9bec7F1a1670819833240f027b25EfF (0x Exchange Proxy)
    #   - 0x3328F7f4A1D1C57c35df56bBf0c9dCAFCA309C49 (Banana Gun Router)
    #   - 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D (Uniswap V2 Router)
    #   - 0x000000000035B5e5ad9019092C665357240f594e (Seaport Bot)
    #   - 0x00000000006c3852cbEf3e08E8dF289169EdE581 (Seaport 1.1)
    #   - 0x000000000000Ad05Ccc4F10045630fb830B95127 (Blur Exchange)
    #   - 0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45 (Uniswap V3 Router 02)
    # REMOVED: Exchange hot wallet (not an agent):
    #   - 0x75e89d5979E4f6Fba9F97c104c2F0AFB3F1dcB88 (MEXC hot wallet)
}

HUMAN_ADDRESSES = {
    # --- Known Public Figures (ENS + social proof, verified EOAs) ---
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": "vitalik.eth",
    "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B": "vitalik old",
    "0x220866B1A2219f40e72f5c628B65D54268cA3A9D": "Hayden Adams",
    "0x8103683202aa8DA10536036EDef04CDd865C225E": "Paradigm Fund",
    "0x0716a17FBAeE714f1E6aB0f9d59edbC5f09815C0": "a16z wallet",

    # --- ENS-verified active users ---
    "0x983110309620D911731Ac0932219af06091b6744": "brantly.eth",
    "0xCB42Ac441fCadeB7a0B36E38F1d5E8cBe1832599": "sassal.eth",
    "0x36928500Bc1dCd7af6a2B4008875CC336b927D57": "Large ETH holder",
    "0x176F3DAb24a159341c0509bB36B833E7fdd0a132": "Justin Sun",
    "0x3DdfA8eC3052539b6C9549F12cEA2C295cfF5296": "SBF Alameda",

    # REMOVED: Exchange hot/cold wallets (not humans, not agents -- exclude):
    #   - 0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8 (Binance 7)
    #   - 0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf (Kraken 13)
    #   - 0xF977814e90dA44bFA03b6295A0616a897441aceC (Binance 8)
    #   - 0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503 (Binance 14)
    #   - 0x28C6c06298d514Db089934071355E5743bf21d60 (Binance 14 hot)
    #   - 0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549 (Binance 15)
    #   - 0xDFd5293D8e347dFe59E90eFd55b2956a1343963d (Binance 16)
    #   - 0x4862733B5FdDFd35f35ea8CCf08F5045e57388B3 (Bitfinex cold)
    #   - 0x1B3cB81E51011b549d78bf720b0d924ac763A7C2 (Gemini 4)
    #   - 0xBF72Da2Bd84c5170618Fbe5914B0ECA9638d5eb5 (DWF Labs hot)
    #   - 0xC098B2a3Aa256D2140208C3de6543aAEf5cd3A94 (FTX Exchange)
    #   - 0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2 (FTX Exchange 2)
    #   - 0x267be1C1D684F78cb4F6a176C4911b741E4Ffdc0 (Kraken 4)
}

# ==================================================================
# COLLECTION FUNCTIONS
# ==================================================================

def fetch_transactions(
    client: EtherscanClient,
    address: str,
    max_txs: int = 5000,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Fetch up to max_txs transactions for an address with pagination.

    Args:
        client: Etherscan API client with key rotation.
        address: Ethereum address.
        max_txs: Maximum number of transactions to retrieve.
        page_size: Number of transactions per API call.

    Returns:
        DataFrame of transactions sorted by timestamp ascending.
    """
    all_dfs = []
    max_pages = (max_txs + page_size - 1) // page_size

    for page in range(1, max_pages + 1):
        try:
            df = client.get_normal_txs(
                address, page=page, offset=page_size
            )
        except Exception as exc:
            logger.warning(
                "  Page %d failed for %s: %s", page, address[:12], exc
            )
            break

        if df.empty:
            break

        all_dfs.append(df)

        if len(df) < page_size:
            break  # Last page

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    # Deduplicate by transaction hash
    if "hash" in combined.columns:
        combined = combined.drop_duplicates(subset=["hash"])
    return combined


def extract_features_from_txs(
    txs: pd.DataFrame,
    config: FeatureConfig,
) -> dict:
    """Extract all 23 features from a transaction DataFrame.

    Uses the individual feature extraction functions from the production
    pipeline, but operates on an already-fetched DataFrame (no API calls).

    Args:
        txs: Transaction DataFrame (from Etherscan API format).
        config: Feature extraction configuration.

    Returns:
        Dictionary of feature_name -> float.
    """
    temporal = extract_temporal_features(txs, config)
    gas = extract_gas_features(txs, config)
    interaction = extract_interaction_features(txs, config)
    approval = extract_approval_security_features(txs, config)

    features = {}
    features.update(temporal)
    features.update(gas)
    features.update(interaction)
    features.update(approval)
    return features


# ==================================================================
# ANALYSIS FUNCTIONS
# ==================================================================

def run_analysis(features_df: pd.DataFrame, labels: pd.Series) -> dict:
    """Run statistical analysis comparing agent vs human features.

    Computes for each feature:
      - Agent and human means and standard deviations
      - Mann-Whitney U test (p-value)
      - Individual AUC score (feature as single predictor)

    Args:
        features_df: DataFrame with shape (n_addresses, 23), indexed by
            address.
        labels: Series indexed by address with values 0 (human) or
            1 (agent).

    Returns:
        Analysis results dictionary.
    """
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_agents": int(labels.sum()),
        "n_humans": int((labels == 0).sum()),
        "n_total": len(labels),
        "feature_analysis": {},
        "summary": {},
    }

    # Align features and labels
    common = features_df.index.intersection(labels.index)
    feat = features_df.loc[common]
    lab = labels.loc[common]

    agent_mask = lab == 1
    human_mask = lab == 0

    significant_features = []
    auc_scores = {}

    for col in FEATURE_NAMES:
        if col not in feat.columns:
            continue

        agent_vals = feat.loc[agent_mask, col].dropna().values
        human_vals = feat.loc[human_mask, col].dropna().values

        entry = {
            "agent_mean": float(np.nanmean(agent_vals)) if len(agent_vals) > 0 else None,
            "agent_std": float(np.nanstd(agent_vals)) if len(agent_vals) > 0 else None,
            "human_mean": float(np.nanmean(human_vals)) if len(human_vals) > 0 else None,
            "human_std": float(np.nanstd(human_vals)) if len(human_vals) > 0 else None,
            "n_agent": len(agent_vals),
            "n_human": len(human_vals),
        }

        # Mann-Whitney U test
        if len(agent_vals) >= 2 and len(human_vals) >= 2:
            try:
                u_stat, p_val = stats.mannwhitneyu(
                    agent_vals, human_vals, alternative="two-sided"
                )
                entry["mann_whitney_u"] = float(u_stat)
                entry["p_value"] = float(p_val)
                entry["significant_005"] = bool(p_val < 0.05)
                entry["significant_001"] = bool(p_val < 0.01)
                if p_val < 0.05:
                    significant_features.append(col)
            except Exception:
                entry["mann_whitney_u"] = None
                entry["p_value"] = None

        # Individual AUC
        all_vals = feat[col].dropna()
        all_labs = lab.loc[all_vals.index]
        if len(all_vals) >= 4 and all_labs.nunique() == 2:
            try:
                auc = roc_auc_score(all_labs, all_vals)
                # Flip if AUC < 0.5 (feature negatively correlated)
                auc_adjusted = max(auc, 1 - auc)
                entry["auc"] = float(auc)
                entry["auc_adjusted"] = float(auc_adjusted)
                auc_scores[col] = auc_adjusted
            except Exception:
                entry["auc"] = None

        results["feature_analysis"][col] = entry

    # Summary
    results["summary"] = {
        "significant_features_005": significant_features,
        "n_significant_005": len(significant_features),
        "n_total_features": len(FEATURE_NAMES),
        "top_features_by_auc": sorted(
            auc_scores.items(), key=lambda x: x[1], reverse=True
        )[:10],
        "mean_auc_adjusted": float(np.mean(list(auc_scores.values())))
        if auc_scores else 0.0,
    }

    return results


def print_comparison_table(analysis: dict) -> None:
    """Pretty-print the feature comparison table."""
    print("\n" + "=" * 100)
    print("Feature Comparison: Agent vs Human")
    print("=" * 100)
    print(
        f"{'Feature':<40} {'Agent Mean':>12} {'Human Mean':>12} "
        f"{'p-value':>12} {'AUC':>8} {'Sig':>5}"
    )
    print("-" * 100)

    for feat_name in FEATURE_NAMES:
        entry = analysis["feature_analysis"].get(feat_name, {})
        a_mean = entry.get("agent_mean")
        h_mean = entry.get("human_mean")
        p_val = entry.get("p_value")
        auc = entry.get("auc_adjusted")

        a_str = f"{a_mean:12.4f}" if a_mean is not None else f"{'N/A':>12}"
        h_str = f"{h_mean:12.4f}" if h_mean is not None else f"{'N/A':>12}"
        p_str = f"{p_val:12.6f}" if p_val is not None else f"{'N/A':>12}"
        auc_str = f"{auc:8.4f}" if auc is not None else f"{'N/A':>8}"
        sig = "**" if entry.get("significant_001") else (
            "*" if entry.get("significant_005") else ""
        )

        print(f"{feat_name:<40} {a_str} {h_str} {p_str} {auc_str} {sig:>5}")

    print("-" * 100)
    summary = analysis["summary"]
    print(
        f"\nSignificant features (p<0.05): "
        f"{summary['n_significant_005']}/{summary['n_total_features']}"
    )
    print(f"Mean adjusted AUC: {summary['mean_auc_adjusted']:.4f}")
    print(f"\nTop features by AUC:")
    for feat_name, auc in summary["top_features_by_auc"][:5]:
        print(f"  {feat_name:<40} AUC={auc:.4f}")


# ==================================================================
# MAIN
# ==================================================================

def main():
    """Run the large-scale data collection pipeline."""
    print("=" * 70)
    print("Paper 1: Large-Scale Real Data Collection")
    print(f"Started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # --- Setup ---
    data_dir = PROJECT_ROOT / "paper1_onchain_agent_id" / "data"
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    features_path = data_dir / "features.parquet"
    results_path = (
        PROJECT_ROOT
        / "paper1_onchain_agent_id"
        / "experiments"
        / "large_scale_results.json"
    )

    # --- Initialize client ---
    client = EtherscanClient()
    print(
        f"\nAPI keys loaded: {client.num_keys} "
        f"(effective rate: ~{client.num_keys * 5} calls/sec)"
    )
    if client.num_keys == 0:
        print("[ERROR] No API keys found. Check shared/configs/config.yaml")
        return

    config = FeatureConfig(tx_fetch_limit=5000)

    # --- Build address list ---
    all_addresses = {}
    for addr, label in AGENT_ADDRESSES.items():
        all_addresses[addr] = ("agent", label)
    for addr, label in HUMAN_ADDRESSES.items():
        all_addresses[addr] = ("human", label)

    n_agents = len(AGENT_ADDRESSES)
    n_humans = len(HUMAN_ADDRESSES)
    print(f"Target addresses: {n_agents} agents + {n_humans} humans = {len(all_addresses)} total")

    # --- Collection loop ---
    feature_records = []
    label_records = []
    errors = []
    skipped = 0
    collected = 0

    total = len(all_addresses)
    for idx, (addr, (category, name)) in enumerate(all_addresses.items(), 1):
        raw_path = raw_dir / f"{addr}.parquet"

        # Checkpoint: skip if raw data already exists
        if raw_path.exists():
            logger.info(
                "[%d/%d] SKIP (cached) %s (%s)", idx, total, name, addr[:12]
            )
            try:
                txs = pd.read_parquet(raw_path)
                features = extract_features_from_txs(txs, config)
                features["address"] = addr
                feature_records.append(features)
                label_records.append(
                    {"address": addr, "label": 1 if category == "agent" else 0, "name": name}
                )
                skipped += 1
            except Exception as exc:
                logger.warning("  Failed to reload %s: %s", addr[:12], exc)
                errors.append({"address": addr, "name": name, "error": str(exc)})
            continue

        # Fetch transactions
        logger.info(
            "[%d/%d] Collecting %s (%s) ...", idx, total, name, addr[:12]
        )
        try:
            txs = fetch_transactions(
                client, addr, max_txs=5000, page_size=1000
            )
            if txs.empty:
                logger.warning("  No transactions found for %s", name)
                errors.append(
                    {"address": addr, "name": name, "error": "no_transactions"}
                )
                continue

            # Save raw transactions
            txs.to_parquet(raw_path, index=False)
            logger.info(
                "  Saved %d txs to %s", len(txs), raw_path.name
            )

            # Extract features
            features = extract_features_from_txs(txs, config)
            features["address"] = addr
            feature_records.append(features)
            label_records.append(
                {"address": addr, "label": 1 if category == "agent" else 0, "name": name}
            )
            collected += 1

            # Print progress summary
            if collected % 5 == 0:
                logger.info(
                    "  Progress: %d collected, %d skipped, %d errors",
                    collected, skipped, len(errors),
                )

        except Exception as exc:
            logger.warning("  ERROR for %s: %s", name, exc)
            errors.append({"address": addr, "name": name, "error": str(exc)})

        # Save intermediate features every 10 addresses
        if (collected + skipped) % 10 == 0 and feature_records:
            _save_features(feature_records, label_records, features_path)

    # --- Save final features ---
    if not feature_records:
        print("\n[ERROR] No features extracted. Check API keys and network.")
        return

    features_df, labels = _save_features(
        feature_records, label_records, features_path
    )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("Collection Summary")
    print("=" * 70)
    print(f"  Successfully processed: {collected + skipped}")
    print(f"    - Newly collected:    {collected}")
    print(f"    - From cache:         {skipped}")
    print(f"  Errors:                 {len(errors)}")
    print(f"  Total API calls:        {client._total_calls}")
    print(f"  Features shape:         {features_df.shape}")
    if errors:
        print(f"\n  Failed addresses:")
        for e in errors:
            print(f"    {e['name']}: {e['error']}")

    # --- Run analysis ---
    print("\n" + "=" * 70)
    print("Running Statistical Analysis")
    print("=" * 70)

    analysis = run_analysis(features_df, labels)

    # Print results
    print_comparison_table(analysis)

    # Save analysis results
    # Convert tuples to lists for JSON serialization
    analysis_json = json.loads(json.dumps(analysis, default=str))
    with open(results_path, "w") as f:
        json.dump(analysis_json, f, indent=2)
    print(f"\nAnalysis saved to {results_path}")

    print(f"\n{'='*70}")
    print(f"Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    return analysis


def _save_features(
    feature_records: list[dict],
    label_records: list[dict],
    features_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Save feature records to Parquet and return (features_df, labels)."""
    features_df = pd.DataFrame(feature_records).set_index("address")
    # Ensure all expected columns exist
    for col in FEATURE_NAMES:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[FEATURE_NAMES]

    labels_df = pd.DataFrame(label_records).set_index("address")
    labels = labels_df["label"]

    # Combine features and labels for saving
    combined = features_df.copy()
    combined["label"] = labels
    combined["name"] = labels_df["name"]
    combined.to_parquet(features_path)

    logger.info("Features saved to %s (%d addresses)", features_path, len(combined))
    return features_df, labels


if __name__ == "__main__":
    main()
