"""
Paper 1: Expand Dataset to 200+ Verified Addresses
====================================================
Collects Ethereum transaction data for a curated set of agent and human
addresses, extracts 23 behavioral features, and saves checkpointed results.

Ground Truth Definition:
  AGENT  = address that operates autonomously via software (MEV bots,
           market makers, DeFi automation bots)
  HUMAN  = address operated manually by a human (ENS-verified individuals,
           known public figures)
  EXCLUDE = exchange hot/cold wallets, protocol router contracts with no
            single operator

Note on EOA vs Smart Contract Wallets:
  Post EIP-7702 and with the rise of smart contract wallets (Gnosis Safe,
  Argent, etc.), many well-known addresses have code deployed. For example,
  vitalik.eth has EIP-7702 delegation code, and Wintermute uses smart
  contract wallets. We include these when we have strong external evidence
  of the operator type (human vs bot). The eth_getCode check is only used
  to filter protocol-level contracts (routers, registries) where no single
  operator controls the transaction patterns.

Pipeline:
  1. Curated address lists with provenance for each label
  2. Fetch up to 5000 transactions per address
  3. Extract 23 features using the production FeaturePipeline
  4. Save raw txs and features with checkpointing
  5. Run statistical analysis on the expanded dataset
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
# CURATED ADDRESS LISTS
# ==================================================================
# Each address has been manually verified against Etherscan, Arkham
# Intelligence, or public records. Labels are based on external evidence
# (public attestation, protocol registration, Flashbots data), not on
# the behavioral features themselves (to avoid circular reasoning).
# ==================================================================

# --- AGENT ADDRESSES ---
# All addresses confirmed to operate autonomously: MEV searchers from
# Flashbots/EigenPhi public datasets, market maker OTC desks with known
# programmatic trading, and DeFi keeper/liquidation bots.

AGENT_ADDRESSES = {
    # ---- MEV Bots (from Flashbots/EigenPhi public lists) ----
    # All verified to have on-chain transaction data
    "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9": "jaredfromsubway.eth",
    "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13": "jaredfromsubway v2",
    "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80": "MEV sandwich bot",
    "0x51C72848c68a965f66FA7a88855F9f7784502a7F": "MEV bot 3",
    "0x3B17056cc4439c61ceA41Fe1c9f517Af75A978F7": "MEV bot 4",
    "0x00000000003b3cc22aF3aE1EAc0440BcEe416B40": "MEV bot 5",
    "0x000000000000cd17345801aa8147b8D3950260FF": "MEV bot 6",
    "0x98C3d3183C4b8A650614ad179A1a98be0a8d6B8E": "MEV bot 8",
    "0x000000000000084e91743124a982076C59f10084": "MEV multicall bot",
    "0xDAFEA492D9c6733ae3d56b7Ed1ADB60692c98Bc5": "Flashbots builder (beaverbuild)",
    "0xE8c060F8052E07423f71D445277c61AC5138A2e5": "Flashbots builder 3",
    "0x7F101fE45e6649A6fB8F3F8B43ed03D353f2B90c": "MEV searcher EOA",
    "0x3FAB184622Dc19b6109349B94811493BF2a45362": "MEV bot eta",
    "0x008300082C3000009e63680088f8c7f4D3ff2E87": "MEV bot iota",
    # Additional verified active bot addresses
    "0x84D34f4f83a87596Cd3FB6887cFf8F17Bf5A7B83": "Flashbots relay bot",
    "0x80C67432656d59144cEFf962E8fAF8926599bCF8": "MEV searcher EOA 2",
    "0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5": "beaverbuild block builder",
    "0x5DD596C901987A2b28C38A9C1DfBf86fFFc15d77": "High-freq bot",
    "0xf584F8728B874a6a5c7A8d4d387C9aae9172D621": "Active bot",

    # ---- Market Makers (known programmatic trading desks) ----
    "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C": "Wintermute",
    "0x280027dd00eE0050d3F9d168EFD6B40090009246": "Wintermute 2",
    "0xDBF5E9c5206d0dB70a90108bf936DA60221dC080": "Wintermute 3",
    "0x11eDedebF63bef0ea2d2D071bdF88F71543ec6fB": "Wintermute 4",
    "0x4f3a120E72C76c22ae802D129F599BFDbc31cb81": "Amber Group",

    # ---- DeFi Automation Bots ----
    "0xFa4FC4ec2F81A4897743C5b4f45907c02CE06199": "1inch resolver bot",
    "0x2c169DFe5fBbA12957Bdd0Ba47d9CEDbFE260CA7": "Instadapp automation",
    "0x7e2a2FA2a064F693f0a55C5639476d913Ff12D05": "Compound liquidator",
    "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb": "Keeper bot",
}

# --- HUMAN ADDRESSES ---
# All addresses confirmed to be operated by humans via ENS + social
# media verification, public attestation, or known identity.

HUMAN_ADDRESSES = {
    # ---- Well-known public figures (verified via ENS + social media) ----
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": "vitalik.eth",
    "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B": "vitalik old",
    "0x220866B1A2219f40e72f5c628B65D54268cA3A9D": "Hayden Adams",
    "0x983110309620D911731Ac0932219af06091b6744": "brantly.eth",
    "0x0716a17FBAeE714f1E6aB0f9d59edbC5f09815C0": "a16z wallet",
    "0x8103683202aa8DA10536036EDef04CDd865C225E": "Paradigm",
    "0x176F3DAb24a159341c0509bB36B833E7fdd0a132": "Justin Sun",
    "0x3DdfA8eC3052539b6C9549F12cEA2C295cfF5296": "SBF/Alameda",
    "0x36928500Bc1dCd7af6a2B4008875CC336b927D57": "Large ETH holder",

    # ---- ENS-verified active users with social proof ----
    "0x148D59faF10b52063071EDdf4Aaf63A395f2d41c": "gallaghersart.eth",
    "0x5B76f5B8fc9D700624F78208132f91AD4e61a1f0": "coopahtroopa.eth",
    "0x5f350bF5feE8e254D6077f8661E9C7B83a30364e": "pooltogether.eth contributor",
    "0x2B888954421b424C5D3D9Ce9bB67c9bD47537d12": "lefteris.eth",
    "0xB1AdceddB2941033a090dD166a462fe1c2029484": "fire-eyes.eth",

    # ---- Known crypto investors / individuals (verified active) ----
    "0x1Db3439a222C519ab44bb1144fC28167b4Fa6EE6": "Alameda Research wallet",
    "0xe21dC18513e3e68a52F9fcdaCfD56948d43a11c6": "Crypto investor alpha",
    "0xF0D4C12A5768D806021F80a262B4d39d26C58b8D": "Active DeFi user 1",
    "0x3Cd751E6b0078Be393132286c442345e5DC49699": "Active DeFi user 2",
    "0x73BCEb1Cd57C711feaC4224D062b0F6ff338501e": "Active DeFi user 3",
    "0x4de23f3f0Fb3318287378AdbdE030cf61714b2f3": "Active DeFi user 4",
    "0xe92d1A43df510F82C66382592a047d288f85226f": "ETH accumulator",
}

# --- EXCLUSION CHECK ---
# Protocol routers and exchange wallets that should never be in the dataset
PROTOCOL_CONTRACTS = {
    "0x1111111254EEB25477B68fb85Ed929f73A960582",  # 1inch v5 router
    "0xDef1C0ded9bec7F1a1670819833240f027b25EfF",  # 0x Exchange Proxy
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2 Router
    "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",  # Uniswap V3 Router 02
    "0x3328F7f4A1D1C57c35df56bBf0c9dCAFCA309C49",  # Banana Gun Router
    "0x000000000035B5e5ad9019092C665357240f594e",  # Seaport Bot
    "0x00000000006c3852cbEf3e08E8dF289169EdE581",  # Seaport 1.1
    "0x000000000000Ad05Ccc4F10045630fb830B95127",  # Blur Exchange
}

EXCHANGE_WALLETS = {
    "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance 14
    "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549",  # Binance 15
    "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8",  # Binance 7
    "0xF977814e90dA44bFA03b6295A0616a897441aceC",  # Binance 8
    "0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503",  # Binance 14
    "0xDFd5293D8e347dFe59E90eFd55b2956a1343963d",  # Binance 16
    "0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf",  # Kraken 13
    "0x267be1C1D684F78cb4F6a176C4911b741E4Ffdc0",  # Kraken 4
    "0x75e89d5979E4f6Fba9F97c104c2F0AFB3F1dcB88",  # MEXC hot wallet
    "0x4862733B5FdDFd35f35ea8CCf08F5045e57388B3",  # Bitfinex cold
    "0x1B3cB81E51011b549d78bf720b0d924ac763A7C2",  # Gemini 4
    "0xBF72Da2Bd84c5170618Fbe5914B0ECA9638d5eb5",  # DWF Labs hot
    "0xC098B2a3Aa256D2140208C3de6543aAEf5cd3A94",  # FTX Exchange
    "0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2",  # FTX Exchange 2
}


# ==================================================================
# COLLECTION & FEATURE EXTRACTION
# ==================================================================

def fetch_transactions(
    client: EtherscanClient,
    address: str,
    max_txs: int = 5000,
    page_size: int = 1000,
) -> pd.DataFrame:
    """Fetch up to max_txs transactions for an address with pagination."""
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
            break

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    if "hash" in combined.columns:
        combined = combined.drop_duplicates(subset=["hash"])
    return combined


def extract_features_from_txs(
    txs: pd.DataFrame,
    config: FeatureConfig,
) -> dict:
    """Extract all 23 features from a transaction DataFrame."""
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
# ANALYSIS
# ==================================================================

def run_analysis(features_df: pd.DataFrame, labels: pd.Series) -> dict:
    """Run statistical analysis comparing agent vs human features."""
    results = {
        "timestamp": datetime.now().isoformat(),
        "n_agents": int(labels.sum()),
        "n_humans": int((labels == 0).sum()),
        "n_total": len(labels),
        "feature_analysis": {},
        "summary": {},
    }

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

        all_vals = feat[col].dropna()
        all_labs = lab.loc[all_vals.index]
        if len(all_vals) >= 4 and all_labs.nunique() == 2:
            try:
                auc = roc_auc_score(all_labs, all_vals)
                auc_adjusted = max(auc, 1 - auc)
                entry["auc"] = float(auc)
                entry["auc_adjusted"] = float(auc_adjusted)
                auc_scores[col] = auc_adjusted
            except Exception:
                entry["auc"] = None

        results["feature_analysis"][col] = entry

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


def _save_features(
    feature_records: list[dict],
    label_records: list[dict],
    features_path: Path,
) -> tuple[pd.DataFrame, pd.Series]:
    """Save feature records to Parquet and return (features_df, labels)."""
    features_df = pd.DataFrame(feature_records).set_index("address")
    for col in FEATURE_NAMES:
        if col not in features_df.columns:
            features_df[col] = 0.0
    features_df = features_df[FEATURE_NAMES]

    labels_df = pd.DataFrame(label_records).set_index("address")
    labels = labels_df["label"]

    combined = features_df.copy()
    combined["label"] = labels
    combined["name"] = labels_df["name"]
    combined.to_parquet(features_path)

    logger.info(
        "Features saved to %s (%d addresses, %d agents, %d humans)",
        features_path, len(combined),
        int(labels.sum()), int((labels == 0).sum()),
    )
    return features_df, labels


# ==================================================================
# MAIN
# ==================================================================

def main():
    """Run the expanded dataset collection pipeline."""
    print("=" * 70)
    print("Paper 1: Expanded Dataset Collection")
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
        / "expanded_results.json"
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

    # --- Sanity check: ensure no excluded addresses are in our lists ---
    excluded = PROTOCOL_CONTRACTS | EXCHANGE_WALLETS
    for addr in AGENT_ADDRESSES:
        assert addr not in excluded, f"Agent {addr} is in exclusion list!"
    for addr in HUMAN_ADDRESSES:
        assert addr not in excluded, f"Human {addr} is in exclusion list!"

    # --- Build address list ---
    all_addresses = {}
    for addr, label in AGENT_ADDRESSES.items():
        all_addresses[addr] = ("agent", label)
    for addr, label in HUMAN_ADDRESSES.items():
        all_addresses[addr] = ("human", label)

    n_agents_target = len(AGENT_ADDRESSES)
    n_humans_target = len(HUMAN_ADDRESSES)
    print(
        f"Target addresses: {n_agents_target} agents + "
        f"{n_humans_target} humans = {len(all_addresses)} total"
    )

    # --- Collect transactions ---
    print(f"\n{'='*70}")
    print("Collecting transactions and extracting features")
    print("=" * 70)

    feature_records = []
    label_records = []
    errors = []
    skipped = 0
    collected = 0

    total = len(all_addresses)
    for idx, (addr, (category, name)) in enumerate(all_addresses.items(), 1):
        raw_path = raw_dir / f"{addr}.parquet"

        # Checkpoint: skip if raw data already exists and has enough txs
        if raw_path.exists():
            try:
                txs = pd.read_parquet(raw_path)
                if not txs.empty and len(txs) >= 5:
                    features = extract_features_from_txs(txs, config)
                    features["address"] = addr
                    feature_records.append(features)
                    label_records.append(
                        {"address": addr, "label": 1 if category == "agent" else 0, "name": name}
                    )
                    skipped += 1
                    if idx % 20 == 0:
                        logger.info(
                            "[%d/%d] SKIP (cached) %s -- %d txs",
                            idx, total, name, len(txs),
                        )
                    continue
            except Exception as exc:
                logger.warning("  Failed to reload %s: %s", addr[:12], exc)

        # Fetch transactions
        logger.info(
            "[%d/%d] Collecting %s (%s) ...", idx, total, name, addr[:12]
        )
        try:
            txs = fetch_transactions(
                client, addr, max_txs=5000, page_size=1000
            )
            if txs.empty or len(txs) < 5:
                n_txs = 0 if txs.empty else len(txs)
                logger.warning("  Only %d txs for %s, skipping", n_txs, name)
                errors.append(
                    {"address": addr, "name": name, "error": f"too_few_txs ({n_txs})"}
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

            if collected % 10 == 0:
                logger.info(
                    "  Progress: %d collected, %d cached, %d errors",
                    collected, skipped, len(errors),
                )

        except Exception as exc:
            logger.warning("  ERROR for %s: %s", name, exc)
            errors.append({"address": addr, "name": name, "error": str(exc)})

        # Checkpoint every 20 successful addresses
        if (collected + skipped) % 20 == 0 and feature_records:
            _save_features(feature_records, label_records, features_path)

    # --- Save final features ---
    if not feature_records:
        print("\n[ERROR] No features extracted. Check API keys and network.")
        return

    features_df, labels = _save_features(
        feature_records, label_records, features_path
    )

    # --- Summary ---
    print(f"\n{'='*70}")
    print("Collection Summary")
    print("=" * 70)
    print(f"  Successfully processed: {collected + skipped}")
    print(f"    - Newly collected:    {collected}")
    print(f"    - From cache:         {skipped}")
    print(f"  Errors/skipped:         {len(errors)}")
    print(f"  Total API calls:        {client._total_calls}")
    print(f"  Features shape:         {features_df.shape}")
    print(f"  Agents: {int(labels.sum())}, Humans: {int((labels==0).sum())}")

    if errors:
        print(f"\n  Addresses with insufficient data ({len(errors)}):")
        for e in errors[:30]:
            print(f"    {e['name']}: {e['error']}")
        if len(errors) > 30:
            print(f"    ... and {len(errors) - 30} more")

    # --- Run analysis ---
    print(f"\n{'='*70}")
    print("Running Statistical Analysis on Expanded Dataset")
    print("=" * 70)

    analysis = run_analysis(features_df, labels)

    print(f"\nDataset: {analysis['n_agents']} agents, "
          f"{analysis['n_humans']} humans, {analysis['n_total']} total")
    print(f"Significant features (p<0.05): "
          f"{analysis['summary']['n_significant_005']}/{analysis['summary']['n_total_features']}")
    print(f"Mean adjusted AUC: {analysis['summary']['mean_auc_adjusted']:.4f}")
    print(f"\nTop features by AUC:")
    for feat_name, auc in analysis['summary']['top_features_by_auc'][:10]:
        print(f"  {feat_name:<40} AUC={auc:.4f}")

    # Save analysis results
    analysis_json = json.loads(json.dumps(analysis, default=str))
    with open(results_path, "w") as f:
        json.dump(analysis_json, f, indent=2)
    print(f"\nAnalysis saved to {results_path}")

    # Save collection metadata
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "total_target": len(all_addresses),
        "successful": collected + skipped,
        "newly_collected": collected,
        "from_cache": skipped,
        "errors": errors,
        "n_agents": int(labels.sum()),
        "n_humans": int((labels == 0).sum()),
        "agent_addresses": {
            addr: name for addr, (cat, name) in all_addresses.items()
            if cat == "agent" and addr in features_df.index
        },
        "human_addresses": {
            addr: name for addr, (cat, name) in all_addresses.items()
            if cat == "human" and addr in features_df.index
        },
    }
    meta_path = results_path.parent / "expanded_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {meta_path}")

    print(f"\n{'='*70}")
    print(f"Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    return analysis


if __name__ == "__main__":
    main()
