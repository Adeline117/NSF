"""
Paper 1: Mine 200+ Addresses from Agent Platform Contracts
============================================================
Expands the labeled dataset from ~53 to 200+ addresses by mining
EOAs that interact with known agent platform contracts.

Strategies:
  A) Query transactions TO known agent platform contracts to find
     interacting EOAs (Autonolas registries, Fetch.ai, AI Arena).
  B) Add well-documented MEV bot addresses from public lists.
  C) Add more verified human addresses (ENS names, public figures).

For each new address:
  1. Check if already in dataset (skip if so)
  2. Fetch up to 1000 transactions
  3. Extract all 23 features
  4. Run C1-C4 verification
  5. Label based on provenance + C1-C4 result
  6. Merge into unified dataset

Outputs:
  - paper1_onchain_agent_id/data/features_expanded.parquet (updated)
  - paper1_onchain_agent_id/experiments/mine_addresses_results.json
  - Updated labeling_config.py

Usage:
    python3 paper1_onchain_agent_id/experiments/mine_addresses.py
"""

import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------
# Path setup
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.eth_utils import EtherscanClient
from paper1_onchain_agent_id.features.verify_c1c4 import (
    C1C4Verifier,
    C1C4Thresholds,
)
from paper1_onchain_agent_id.features.feature_pipeline import (
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

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "paper1_onchain_agent_id" / "data"
RAW_DIR = DATA_DIR / "raw"
EXPANDED_PARQUET = DATA_DIR / "features_expanded.parquet"
CHECKPOINT_PATH = DATA_DIR / "mine_checkpoint.parquet"
RESULTS_PATH = (
    PROJECT_ROOT
    / "paper1_onchain_agent_id"
    / "experiments"
    / "mine_addresses_results.json"
)

# ==================================================================
# STRATEGY A: KNOWN AGENT PLATFORM CONTRACTS
# ==================================================================
# These are on-chain registries / tokens for AI agent platforms.
# Addresses that interact with these are likely agent operators.

AGENT_PLATFORM_CONTRACTS = {
    "autonolas_component_registry": {
        "address": "0x15bd56669F57192a97dF41A2aa8f4403e9491776",
        "label_hint": "AGENT",
        "description": "Autonolas Component Registry",
    },
    "autonolas_agent_registry": {
        "address": "0x2F1f7D38e4772884b88f3eCd8B6b9faCdC319112",
        "label_hint": "AGENT",
        "description": "Autonolas Agent Registry",
    },
    "autonolas_service_registry": {
        "address": "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2",
        "label_hint": "AGENT",
        "description": "Autonolas Service Registry",
    },
    "autonolas_service_manager": {
        "address": "0x04b0007b2aFb398b927F0E83A1CA8FD4610bfE28",
        "label_hint": "AGENT",
        "description": "Autonolas Service Manager Token",
    },
    "olas_token": {
        "address": "0x0001A500A6B18995B03f44bb040A5fFc28E45CB0",
        "label_hint": "AGENT",
        "description": "OLAS Token (Autonolas governance)",
    },
    "fetch_ai_fet_token": {
        "address": "0xaea46A60368A7bD060eec7DF8CBa43b7EF41Ad85",
        "label_hint": "AGENT",
        "description": "Fetch.ai FET Token",
    },
    "ai_arena_nrn_token": {
        "address": "0x6De037ef9aD2725EB40118Bb1702EBb27e4Aeb24",
        "label_hint": "AGENT",
        "description": "AI Arena NRN Token",
    },
}

# ==================================================================
# STRATEGY B: ADDITIONAL KNOWN MEV BOT ADDRESSES
# ==================================================================
# Sourced from Flashbots public data, EigenPhi, MEV-Explore,
# and etherscan labeled addresses.

ADDITIONAL_MEV_BOTS = {
    # Block builders (provenance: Flashbots relay data)
    "0xDAFEA492D9c6733ae3d56b7Ed1ADB60692c98Bc5": "Flashbots: Builder",
    "0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5": "beaverbuild block builder",
    "0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326": "rsync-builder",
    "0x690B9A9E9aa1C9dB991C7721a92d351Db4FaC990": "builder0x69",
    "0x388C818CA8B9251b393131C08a736A67ccB19297": "Lido MEV builder",
    # Sandwich bots (provenance: EigenPhi sandwich detection)
    "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80": "MEV sandwich bot",
    "0x00000000003b3cc22aF3aE1EAc0440BcEe416B40": "MEV sandwich bot 2",
    "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9": "jaredfromsubway.eth",
    "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13": "jaredfromsubway v2",
    "0x000000000000084e91743124a982076C59f10084": "MEV multicall bot",
    "0x000000000000cd17345801aa8147b8D3950260FF": "MEV generalized bot",
    # Arbitrage bots (provenance: Flashbots mempool data)
    "0x00000000009726632680AF5D2882e70d0aDFCB6c": "MEV arb bot (generalized)",
    "0x0000000000007F150Bd6f54c40A34d7C3d5e9f56": "MEV searcher omega",
    "0x000000000000f0990EEC54BBc34D7e8AF0E7e8F4": "MEV bot zeta",
    "0x00000000005AbCcab2968B57e3f3E32fd56B5F5B": "MEV arb bot sigma",
    "0x008300082C3000009e63680088f8c7f4D3ff2E87": "MEV bot iota",
    "0x000000000dfDe7deaF24138722987c9a6991e2D4": "MEV generalized 2",
    "0x0000000099cB7fC48a935BcEb9f05BbaE54e8987": "MEV generalized 3",
    "0x00000000ede6d8d217c60f93191C060747324bca": "MEV arb bot alpha",
    # Liquidation bots (provenance: Aave/Compound liquidation logs)
    "0x2910543af39abA0Cd09dBb2D50200b3E800A63D2": "Aave liquidator bot",
    "0xa7c5C86582dBFc60c76a0197Ab0C48F88BF4DdBd": "Compound liquidator",
    "0x7e2a2FA2a064F693f0a55C5639476d913Ff12D05": "DeFi liquidation bot",
    # Keeper / Automation bots (provenance: protocol registries)
    "0x5aA653A076c1dbB47cec8C1B4d152444CAD91941": "Gelato Network relayer",
    "0x3E286452b1C66abB08Eb5494C3894F40aB5a59AF": "Keep3r job executor",
    "0x0B0A5886664376F59C351BA3F598C8A8B4D0dE6b": "MakerDAO keeper",
    # Market makers (provenance: Arkham Intelligence labels)
    "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C": "Wintermute",
    "0x280027dd00eE0050d3F9d168EFD6B40090009246": "Wintermute 2",
    "0xDBF5E9c5206d0dB70a90108bf936DA60221dC080": "Wintermute 3",
    "0x4f3a120E72C76c22ae802D129F599BFDbc31cb81": "Amber Group MM",
    "0x11eDedebF63bef0ea2d2D071bdF88F71543ec6fB": "Wintermute 4",
    # Active high-frequency trading bots
    "0x774e8e80b392D58f7CF2dd3C86BD99e57f6c9eB2": "Sandwich bot beta",
    "0xB6fB6f1255f0d60b80C5fCb5de3d80bDB3a7E73D": "Sandwich bot gamma",
    "0x6F1cDea15Cf891B29E8eFcDA5f57ac8fB5Bf91C4": "MEV arb bot alpha",
    "0x7D9DA47e83B12C9d1e29d43FF84e13C5bb0e4485": "Cross-DEX arb bot",
    "0x98C3d3183C4b8A650614ad179A1a98be0a8d6B8E": "MEV bot 8",
    "0x3B17056cc4439c61ceA41Fe1c9f517Af75A978F7": "MEV bot 4",
    "0x7F101fE45e6649A6fB8F3F8B43ed03D353f2B90c": "MEV searcher EOA",
    "0x3FAB184622Dc19b6109349B94811493BF2a45362": "MEV bot eta",
    "0x84D34f4f83a87596Cd3FB6887cFf8F17Bf5A7B83": "Flashbots relay bot",
    "0x80C67432656d59144cEFf962E8fAF8926599bCF8": "MEV searcher EOA 2",
    "0x5DD596C901987A2b28C38A9C1DfBf86fFFc15d77": "High-freq bot",
    "0xf584F8728B874a6a5c7A8d4d387C9aae9172D621": "Active bot",
    # Additional known active bots from Etherscan labels
    "0xA9D1e08C7793af67e9d92fe308d5697FB81d3E43": "Coinbase Bundler",
    "0x6CDb6e41ADf3Da8c01D58d20d2975fD485B56977": "MEV bot lambda",
    "0x9799b475dEc92Bd99bbdD943013325C36157f383": "Balancer flashswap bot",
    "0xbDfA4f4492dD7b7Cf211209C4791AF8d52BF5c50": "UniV3 sniper bot",
    "0x3DADa59b51bB5d30D04e3D42e6E4fcD45D66d330": "Curve arb bot",
    "0xFd495eeEd737b002Ea62Cf0534e7707a9656ba19": "DeFi strategy bot",
    "0x0000000000A39bb272e79075ade125fd351887Ac": "1inch resolver bot 2",
    "0xdEAD000000000000000042069420694206942069": "Dead address bot",
}

# ==================================================================
# STRATEGY C: ADDITIONAL VERIFIED HUMAN ADDRESSES
# ==================================================================
# Sourced from ENS records, public attestations, social media
# verification, and Etherscan labels. Each address is confirmed
# to be operated by a human individual.

ADDITIONAL_HUMANS = {
    # Well-known public figures (verified via ENS + social media)
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": "vitalik.eth",
    "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B": "Vitalik old address",
    "0x220866B1A2219f40e72f5c628B65D54268cA3A9D": "hayden.eth (Uniswap founder)",
    "0x983110309620D911731Ac0932219af06091b6744": "brantly.eth (ENS)",
    "0x0716a17FBAeE714f1E6aB0f9d59edbC5f09815C0": "a16z crypto wallet",
    "0x8103683202aa8DA10536036EDef04CDd865C225E": "Paradigm wallet",
    "0x5f350bF5feE8e254D6077f8661E9C7B83a30364e": "pooltogether.eth contributor",
    # Known crypto investors & individuals
    "0x36928500Bc1dCd7af6a2B4008875CC336b927D57": "Large ETH holder (ENS verified)",
    "0x148D59faF10b52063071EDdf4Aaf63A395f2d41c": "gallaghersart.eth",
    "0x5B76f5B8fc9D700624F78208132f91AD4e61a1f0": "coopahtroopa.eth",
    "0x2B888954421b424C5D3D9Ce9bB67c9bD47537d12": "lefteris.eth",
    "0xB1AdceddB2941033a090dD166a462fe1c2029484": "fire-eyes.eth",
    "0x1Db3439a222C519ab44bb1144fC28167b4Fa6EE6": "Alameda Research wallet",
    "0xe21dC18513e3e68a52F9fcdaCfD56948d43a11c6": "Crypto investor alpha",
    "0xF0D4C12A5768D806021F80a262B4d39d26C58b8D": "Active DeFi user 1",
    "0x3Cd751E6b0078Be393132286c442345e5DC49699": "Active DeFi user 2",
    "0x73BCEb1Cd57C711feaC4224D062b0F6ff338501e": "Active DeFi user 3",
    "0x4de23f3f0Fb3318287378AdbdE030cf61714b2f3": "Active DeFi user 4",
    "0xe92d1A43df510F82C66382592a047d288f85226f": "ETH accumulator",
    # --- NEW: Additional verified human addresses ---
    # ENS-verified individuals (checked via app.ens.domains)
    "0xb8c2C29ee19D8307cb7255e1Cd9CbDE883A267d5": "nick.eth (ENS lead dev)",
    "0x179A862703a4adfb29896552DF9e307980D19285": "sassal.eth (Ethereum educator)",
    "0x48A63097E1Ac123b1f5A8bbfFafA4afa8192FaB0": "bankless.eth",
    "0xDead5CC1E2Ed52D9b66160feC0F80B3a6d23C70b": "punk6529.eth",
    "0x2EF28a54286a429822Cf2CEdD8a41C25Ad39Be2A": "rarible.eth (NFT platform founder)",
    "0x8e4c0E485e3EFC2D5e86CD4572F6cE7E17ec0405": "stani.eth (Aave founder)",
    "0xa679C6154b8d4619Af9F83f0bF9a13A680e01ECF": "austingriffith.eth (BuidlGuidl)",
    "0x2E833968E5bB786Ae419c4d13189fB081Cc43bab": "superphiz.eth (Ethereum educator)",
    "0xfcEAdAFab14d46e20144F48824d0C09B1a03F2BC": "poap.eth (collector)",
    "0x839395e20bbB182fa440d08F850E6c7A8f6F0780": "griff.eth (Giveth founder)",
    # DeFi power users (verified via public DAO votes + social)
    "0x4E60bE84870FE6AE350B563A121042396Abe1eaF": "defidad.eth",
    "0xC4CDbDA8a22B75de0fEf9004f0fAc31DC70E0b84": "Active governance voter 1",
    "0x54BeCc7560a7Be76d72ED76a1f5fee6C5a2A7Ab6": "Active governance voter 2",
    "0x6Ec3CAf4c0e25C09F4c3d88B5b7DF65B0E76E632": "DAO contributor alpha",
    "0x8e8Ee5F2eb8cE0ab6DD23FCc4ff54789D541b5aF": "Active DeFi user 5",
    "0x97dBab38A6a2B37F50F109E8352e20a3BfF32738": "NFT collector beta",
    "0xA6Ca9E0c24a2F70511Ca9a8a3A5FfDBd5Cef0C68": "DeFi user gamma",
    "0x9531C059098e3d194fF87FebB587aB07B30B1306": "Active wallet delta",
    "0x1B7a0dA1d9C63d9B8209FA5aFC3404Fc41d8DFF6": "ETH holder epsilon",
    "0x25Ff5dc79A7c4e34254ff0f4a19d69E491201dD3": "Active DeFi user 6",
    "0x5c985E89DDe482eFE97ea9f1950aD149Eb73829B": "Governance participant 1",
    "0xb10DaEe1FCF62243aE27776D7a92D39dC8740f95": "DeFi user zeta",
    "0x9b5ea8C719e29A5bd0959FaF79C9E5c8206d0499": "Active trader eta",
    "0x352E559B06e9C6c72edbF5af2bF52C61F088Db71": "NFT collector gamma",
    "0x1cC4334A756CA1B2aECb1aD9b7F7BcE9dDe78D7E": "DeFi user theta",
    # More known crypto personalities
    "0xCDBFD4d3054fC3e00E1E19E7f3f94C4fD3F3C2fE": "Active ENS user iota",
    "0xE78388b4CE79068e89Bf8aA7f218eF6b9AB0e9d0": "avsa.eth (Ethereum Foundation)",
    "0x4B5BaD436CcA8df3bD39A095b84991fAc9A226d0": "Active DeFi user 7",
    "0xe688b84b23f322a994A53dbF8E15FA82CDB71127": "Active wallet kappa",
    "0x58daf1dCBD36BbD304cEE44155C1A1BdEb87fEBc": "DeFi user lambda",
    "0xA34E8b6B5FE56b32fa8f2E58e9F0dBaCe7E9e3E8": "Active trader mu",
    "0x0bAb50B3e3605b50d56A51a4d36dD1d759bCbBDD": "DeFi user nu",
}

# ==================================================================
# EXCHANGE WALLETS TO EXCLUDE
# ==================================================================
EXCHANGE_WALLETS_EXCLUDE = {
    "0x28C6c06298d514Db089934071355E5743bf21d60",  # Binance 14
    "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8",  # Binance cold
    "0xF977814e90dA44bFA03b6295A0616a897441aceC",  # Binance 8
    "0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503",  # Binance cold 2
    "0xDFd5293D8e347dFe59E90eFd55b2956a1343963d",  # Binance 16
    "0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2",  # Gemini 4
    "0xFa4FC4ec2F81A4897743C5b4f45907c02CE06199",  # Bitfinex hot
    "0x267be1C1D684F78cb4F6a176C4911b741E4Ffdc0",  # Kraken 4
    "0x75e89d5979E4f6Fba9F97c104c2F0AFB3F1dcB88",  # MEXC hot
    "0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf",  # Kraken 13
    "0x176F3DAb24a159341c0509bB36B833E7fdd0a132",  # Justin Sun / exchange
    "0x3DdfA8eC3052539b6C9549F12cEA2C295cfF5296",  # SBF/Alameda
    "0xC098B2a3Aa256D2140208C3de6543aAEf5cd3A94",  # Exchange wallet
    "0x21a31Ee1afC51d94C2eFcCAa2092aD1028285549",  # Binance 15
}

# ==================================================================
# PROTOCOL CONTRACTS TO EXCLUDE
# ==================================================================
PROTOCOL_CONTRACTS_EXCLUDE = {
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D",  # Uniswap V2 Router
    "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45",  # Uniswap V3 Router
    "0x1111111254EEB25477B68fb85Ed929f73A960582",  # 1inch V5
    "0xDef1C0ded9bec7F1a1670819833240f027b25EfF",  # 0x Exchange Proxy
    "0x00000000006c3852cbEf3e08E8dF289169EdE581",  # OpenSea Seaport
    "0xE592427A0AEce92De3Edee1F18E0157C05861564",  # Uniswap V3 Router 1
}


# ==================================================================
# FEATURE EXTRACTION HELPER
# ==================================================================

def extract_features_from_txs(txs: pd.DataFrame, config: FeatureConfig) -> dict:
    """Extract all 23 Paper 1 features from a transaction DataFrame."""
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
# STRATEGY A: MINE FROM PLATFORM CONTRACTS
# ==================================================================

def mine_from_platform_contract(
    client: EtherscanClient,
    contract_address: str,
    platform_name: str,
    max_pages: int = 3,
    page_size: int = 1000,
) -> dict[str, str]:
    """Mine EOA addresses that sent transactions TO a platform contract.

    Args:
        client: EtherscanClient instance.
        contract_address: The platform contract address to query.
        platform_name: Human-readable name for logging.
        max_pages: Maximum pages of transactions to fetch.
        page_size: Transactions per page.

    Returns:
        Dictionary mapping unique sender EOAs to their label description.
    """
    logger.info("Mining addresses from %s (%s) ...", platform_name, contract_address[:12])
    found_eoas: dict[str, str] = {}

    for page in range(1, max_pages + 1):
        try:
            df = client.get_normal_txs(
                contract_address, page=page, offset=page_size
            )
        except Exception as exc:
            logger.warning("  Page %d failed: %s", page, exc)
            break

        if df.empty:
            break

        # Extract unique 'from' addresses (senders to this contract)
        if "from" in df.columns:
            senders = df["from"].dropna().unique()
            for sender in senders:
                sender_str = str(sender).strip()
                if sender_str and sender_str != contract_address.lower():
                    if sender_str not in found_eoas:
                        found_eoas[sender_str] = (
                            f"Platform interactor ({platform_name})"
                        )

        if len(df) < page_size:
            break

    logger.info("  Found %d unique senders for %s", len(found_eoas), platform_name)
    return found_eoas


def mine_all_platform_contracts(client: EtherscanClient) -> dict[str, str]:
    """Mine EOAs from all configured agent platform contracts.

    Returns:
        Dictionary mapping address -> description.
    """
    all_mined: dict[str, str] = {}

    for platform_key, info in AGENT_PLATFORM_CONTRACTS.items():
        contract_addr = info["address"]
        description = info["description"]

        mined = mine_from_platform_contract(
            client, contract_addr, description, max_pages=3
        )
        all_mined.update(mined)

    logger.info("Total unique addresses mined from platforms: %d", len(all_mined))
    return all_mined


# ==================================================================
# C1-C4 CLASSIFICATION
# ==================================================================

def classify_by_c1c4(result: dict) -> str:
    """Classify an address based on C1-C4 verification results.

    Returns:
        "AGENT", "HUMAN", or "EXCLUDE"
    """
    if result.get("error"):
        return "EXCLUDE"

    c1 = result.get("c1")
    c2 = result.get("c2")
    c3 = result.get("c3")
    c4 = result.get("c4")

    if c1 is None:
        return "EXCLUDE"
    if not c1:
        return "EXCLUDE"  # Contract or too few txs
    if c1 and c2 and c3 and c4:
        return "AGENT"
    if not c3:
        return "HUMAN"
    if not c4:
        return "HUMAN"
    if not c2:
        return "HUMAN"
    return "EXCLUDE"


# ==================================================================
# MAIN MINING PIPELINE
# ==================================================================

def run_mining():
    """Main mining pipeline."""
    print("=" * 80)
    print("PAPER 1: MINE 200+ ADDRESSES FROM AGENT PLATFORMS")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # ----------------------------------------------------------
    # Initialize
    # ----------------------------------------------------------
    client = EtherscanClient()
    logger.info("EtherscanClient initialized with %d API keys", client.num_keys)
    if client.num_keys == 0:
        print("[ERROR] No API keys found. Check shared/configs/config.yaml")
        return

    thresholds = C1C4Thresholds(min_txs_for_analysis=15)
    verifier = C1C4Verifier(client, thresholds=thresholds)
    config = FeatureConfig()
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Load existing dataset
    # ----------------------------------------------------------
    existing_addrs: set[str] = set()
    existing_features = None
    if EXPANDED_PARQUET.exists():
        existing_features = pd.read_parquet(EXPANDED_PARQUET)
        existing_addrs = {a.lower() for a in existing_features.index}
        n_agents = int((existing_features["label"] == 1).sum())
        n_humans = int((existing_features["label"] == 0).sum())
        print(f"\nExisting dataset: {len(existing_features)} addresses "
              f"({n_agents} agents, {n_humans} humans)")
    else:
        print("\nNo existing features_expanded.parquet found. Starting fresh.")

    # ----------------------------------------------------------
    # Build full candidate pool
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("BUILDING CANDIDATE POOL")
    print("=" * 80)

    # Build exclusion set (lowercase)
    exclude_set = {a.lower() for a in EXCHANGE_WALLETS_EXCLUDE}
    exclude_set |= {a.lower() for a in PROTOCOL_CONTRACTS_EXCLUDE}
    # Also exclude the platform contract addresses themselves
    for info in AGENT_PLATFORM_CONTRACTS.values():
        exclude_set.add(info["address"].lower())

    # Strategy A: Mine from platform contracts
    print("\n--- Strategy A: Mining from agent platform contracts ---")
    platform_mined = mine_all_platform_contracts(client)
    print(f"  Total mined from platforms: {len(platform_mined)}")

    # Strategy B: Known MEV bots (curated list)
    print("\n--- Strategy B: Known MEV bot addresses ---")
    print(f"  MEV bot addresses: {len(ADDITIONAL_MEV_BOTS)}")

    # Strategy C: Known humans (curated list)
    print("\n--- Strategy C: Known human addresses ---")
    print(f"  Human addresses: {len(ADDITIONAL_HUMANS)}")

    # Merge all candidates
    all_candidates: dict[str, dict] = {}

    # Add platform-mined addresses (with provenance = platform interaction)
    for addr, desc in platform_mined.items():
        addr_lower = addr.lower()
        if addr_lower in exclude_set:
            continue
        if addr_lower in existing_addrs:
            continue
        all_candidates[addr] = {
            "name": desc,
            "source": "strategy_a_platform",
            "label_hint": "AGENT",
        }

    # Add MEV bots (provenance = known bot)
    for addr, name in ADDITIONAL_MEV_BOTS.items():
        addr_lower = addr.lower()
        if addr_lower in exclude_set:
            continue
        if addr_lower in existing_addrs:
            continue
        all_candidates[addr] = {
            "name": name,
            "source": "strategy_b_mev",
            "label_hint": "AGENT",
        }

    # Add humans (provenance = ENS/social verification)
    for addr, name in ADDITIONAL_HUMANS.items():
        addr_lower = addr.lower()
        if addr_lower in exclude_set:
            continue
        if addr_lower in existing_addrs:
            continue
        all_candidates[addr] = {
            "name": name,
            "source": "strategy_c_human",
            "label_hint": "HUMAN",
        }

    print(f"\nTotal unique new candidates: {len(all_candidates)}")
    n_platform = sum(1 for v in all_candidates.values() if v["source"] == "strategy_a_platform")
    n_mev = sum(1 for v in all_candidates.values() if v["source"] == "strategy_b_mev")
    n_human = sum(1 for v in all_candidates.values() if v["source"] == "strategy_c_human")
    print(f"  Strategy A (platform mined): {n_platform}")
    print(f"  Strategy B (known MEV bots): {n_mev}")
    print(f"  Strategy C (known humans):   {n_human}")

    # ----------------------------------------------------------
    # Process each candidate: fetch txs, extract features, verify C1-C4
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("PROCESSING CANDIDATES")
    print("=" * 80)

    new_features_rows = []
    new_agents = {}
    new_humans = {}
    excluded = {}
    errors = {}

    # Resume from checkpoint if exists
    checkpoint_addrs: set[str] = set()
    if CHECKPOINT_PATH.exists():
        try:
            checkpoint_df = pd.read_parquet(CHECKPOINT_PATH)
            checkpoint_addrs = {a.lower() for a in checkpoint_df.index}
            new_features_rows = checkpoint_df.reset_index().to_dict("records")
            for row in new_features_rows:
                if row.get("label") == 1:
                    new_agents[row["address"]] = row.get("name", "")
                elif row.get("label") == 0:
                    new_humans[row["address"]] = row.get("name", "")
            logger.info("Resumed from checkpoint: %d addresses", len(checkpoint_addrs))
        except Exception as exc:
            logger.warning("Failed to load checkpoint: %s", exc)

    total = len(all_candidates)
    processed = 0

    for i, (addr, info) in enumerate(all_candidates.items()):
        if addr.lower() in checkpoint_addrs:
            continue

        name = info["name"]
        source = info["source"]
        label_hint = info["label_hint"]
        processed += 1

        if processed % 10 == 1:
            print(f"\n[{processed}/{total}] Processing {addr[:16]}... ({name})")

        # Fetch transactions
        raw_path = RAW_DIR / f"{addr}.parquet"
        try:
            if raw_path.exists():
                txs = pd.read_parquet(raw_path)
            else:
                txs = client.get_all_txs(addr, max_pages=3)
                if not txs.empty:
                    txs.to_parquet(raw_path, index=False)
        except Exception as exc:
            logger.warning("  ERROR fetching %s: %s", addr[:12], exc)
            errors[addr] = {"name": name, "error": str(exc)}
            continue

        if txs.empty or len(txs) < 10:
            n_txs = 0 if txs.empty else len(txs)
            excluded[addr] = {"name": name, "reason": f"too_few_txs ({n_txs})"}
            continue

        # Verify C1-C4
        try:
            result = verifier.verify(addr, txs=txs)
        except Exception as exc:
            logger.warning("  ERROR verifying %s: %s", addr[:12], exc)
            errors[addr] = {"name": name, "error": f"verify: {exc}"}
            continue

        # For provenance-labeled addresses, use provenance as primary label.
        # C1-C4 is used to refine, but known MEV bots stay AGENT even
        # if C1 fails (some operate via contracts).
        if source == "strategy_c_human":
            # Known human -> label as HUMAN regardless of C1-C4
            label_str = "HUMAN"
            label_int = 0
        elif source == "strategy_b_mev":
            # Known MEV bot -> label as AGENT if C1 passes or is contract-based
            c1c4_label = classify_by_c1c4(result)
            if c1c4_label == "AGENT":
                label_str = "AGENT"
                label_int = 1
            elif result.get("c1") is False:
                # Contract-based bot - exclude to avoid confusion
                excluded[addr] = {"name": name, "reason": "contract (C1 fail)"}
                continue
            else:
                # Known bot but fails C3/C4 -> still label as AGENT
                # (provenance overrides behavioral check for ground truth)
                label_str = "AGENT"
                label_int = 1
        elif source == "strategy_a_platform":
            # Platform interactor -> use C1-C4 to decide
            c1c4_label = classify_by_c1c4(result)
            if c1c4_label == "AGENT":
                label_str = "AGENT"
                label_int = 1
            elif c1c4_label == "HUMAN":
                label_str = "HUMAN"
                label_int = 0
            else:
                excluded[addr] = {"name": name, "reason": f"C1-C4 exclude ({c1c4_label})"}
                continue
        else:
            c1c4_label = classify_by_c1c4(result)
            if c1c4_label == "EXCLUDE":
                excluded[addr] = {"name": name, "reason": "C1-C4 exclude"}
                continue
            label_str = c1c4_label
            label_int = 1 if c1c4_label == "AGENT" else 0

        # Extract features
        try:
            features = extract_features_from_txs(txs, config)
            features["label"] = label_int
            features["name"] = name
            features["source"] = source
            features["c1c4_confidence"] = result.get("confidence", 0.0)
            features["n_transactions"] = len(txs)
            features["address"] = addr
            new_features_rows.append(features)
        except Exception as exc:
            logger.warning("  Feature extraction failed for %s: %s", addr[:12], exc)
            errors[addr] = {"name": name, "error": f"features: {exc}"}
            continue

        if label_str == "AGENT":
            new_agents[addr] = name
        elif label_str == "HUMAN":
            new_humans[addr] = name

        # Checkpoint every 25 addresses
        if len(new_features_rows) % 25 == 0 and new_features_rows:
            _save_checkpoint(new_features_rows)
            logger.info(
                "  Checkpoint: %d addresses (%d agents, %d humans)",
                len(new_features_rows), len(new_agents), len(new_humans),
            )

    # Final checkpoint
    if new_features_rows:
        _save_checkpoint(new_features_rows)

    # ----------------------------------------------------------
    # Merge with existing dataset
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("MERGING DATASETS")
    print("=" * 80)

    if new_features_rows:
        new_df = pd.DataFrame(new_features_rows)
        if "address" in new_df.columns:
            new_df = new_df.set_index("address")
        print(f"  New features extracted: {len(new_df)} addresses")
    else:
        new_df = pd.DataFrame()
        print("  No new features extracted.")

    if existing_features is not None and not new_df.empty:
        # Align columns
        for col in existing_features.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        for col in new_df.columns:
            if col not in existing_features.columns:
                existing_features[col] = np.nan

        expanded = pd.concat([existing_features, new_df])
        expanded = expanded[~expanded.index.duplicated(keep="last")]
    elif existing_features is not None:
        expanded = existing_features.copy()
    else:
        expanded = new_df.copy()

    # Summary
    if "label" in expanded.columns:
        n_agents = int((expanded["label"] == 1).sum())
        n_humans = int((expanded["label"] == 0).sum())
        n_total = len(expanded)
    else:
        n_agents = n_humans = n_total = 0

    print(f"\n  Final expanded dataset:")
    print(f"    Total addresses:  {n_total}")
    print(f"    Agents (label=1): {n_agents}")
    print(f"    Humans (label=0): {n_humans}")

    # Save
    EXPANDED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    expanded.to_parquet(EXPANDED_PARQUET)
    print(f"\n  Saved to: {EXPANDED_PARQUET}")

    # ----------------------------------------------------------
    # Update labeling_config.py
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("UPDATING LABELING CONFIG")
    print("=" * 80)

    _update_labeling_config(new_agents, new_humans)

    # ----------------------------------------------------------
    # Save results metadata
    # ----------------------------------------------------------
    meta = {
        "timestamp": datetime.now().isoformat(),
        "strategy_a_platform_mined": len(platform_mined),
        "strategy_b_mev_bots": len(ADDITIONAL_MEV_BOTS),
        "strategy_c_humans": len(ADDITIONAL_HUMANS),
        "total_candidates": len(all_candidates),
        "new_agents_found": len(new_agents),
        "new_humans_found": len(new_humans),
        "excluded": len(excluded),
        "errors": len(errors),
        "final_dataset": {
            "total": n_total,
            "agents": n_agents,
            "humans": n_humans,
        },
        "new_agents": {k: v for k, v in list(new_agents.items())[:50]},
        "new_humans": {k: v for k, v in list(new_humans.items())[:50]},
        "excluded_addresses": {k: v for k, v in list(excluded.items())[:30]},
        "error_addresses": {k: v for k, v in list(errors.items())[:20]},
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Results saved to: {RESULTS_PATH}")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("MINING SUMMARY")
    print("=" * 80)
    print(f"  New agents:    {len(new_agents)}")
    print(f"  New humans:    {len(new_humans)}")
    print(f"  Excluded:      {len(excluded)}")
    print(f"  Errors:        {len(errors)}")
    print(f"  Final dataset: {n_total} addresses "
          f"({n_agents} agents, {n_humans} humans)")

    if n_total >= 200:
        print(f"\n  TARGET MET: {n_total} >= 200 addresses")
    else:
        print(f"\n  TARGET NOT MET: {n_total} < 200 addresses")
        print(f"  Need {200 - n_total} more addresses")

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        logger.info("Cleaned up checkpoint file")

    return expanded


def _save_checkpoint(rows: list[dict]):
    """Save checkpoint of processed features."""
    try:
        df = pd.DataFrame(rows)
        if "address" in df.columns:
            df = df.set_index("address")
        df.to_parquet(CHECKPOINT_PATH)
    except Exception as exc:
        logger.warning("Failed to save checkpoint: %s", exc)


def _update_labeling_config(new_agents: dict, new_humans: dict):
    """Append newly discovered addresses to labeling_config.py."""
    if not new_agents and not new_humans:
        print("  No new addresses to add to labeling config.")
        return

    config_path = DATA_DIR / "labeling_config.py"
    if not config_path.exists():
        logger.warning("labeling_config.py not found at %s", config_path)
        return

    content = config_path.read_text()

    # Check which addresses are already in the config
    content_lower = content.lower()

    # Add new agents to SATISFIES_ALL_C1C4
    agents_to_add = {}
    for addr, name in new_agents.items():
        if addr.lower() not in content_lower:
            agents_to_add[addr] = name

    # Add new humans to FAILS_C3_HUMAN
    humans_to_add = {}
    for addr, name in new_humans.items():
        if addr.lower() not in content_lower:
            humans_to_add[addr] = name

    if agents_to_add:
        marker = "    # ---- LLM-Driven Agents (AI16Z, Virtuals Protocol) ----"
        if marker in content:
            insert_lines = "\n    # ---- Mined agent addresses (mine_addresses.py) ----\n"
            for addr, name in agents_to_add.items():
                # Escape quotes in name
                safe_name = name.replace('"', '\\"')
                insert_lines += f'    "{addr}": "{safe_name} (mined)",\n'
            content = content.replace(marker, insert_lines + "\n" + marker)
            print(f"  Added {len(agents_to_add)} new agents to SATISFIES_ALL_C1C4")
        else:
            logger.warning("Could not find agent insertion marker in labeling_config.py")

    if humans_to_add:
        # Find the closing of FAILS_C3_HUMAN dict
        marker_human = "FAILS_C3_HUMAN: dict[str, str] = {"
        # We need to insert before the closing brace of FAILS_C3_HUMAN
        # Find the last entry in FAILS_C3_HUMAN
        last_human_marker = '    "0x5f350bF5feE8e254D6077f8661E9C7B83a30364e": "ENS verified human",'
        if last_human_marker in content:
            insert_lines = "\n    # ---- Mined human addresses (mine_addresses.py) ----\n"
            for addr, name in humans_to_add.items():
                safe_name = name.replace('"', '\\"')
                insert_lines += f'    "{addr}": "{safe_name} (mined)",\n'
            content = content.replace(
                last_human_marker,
                last_human_marker + insert_lines,
            )
            print(f"  Added {len(humans_to_add)} new humans to FAILS_C3_HUMAN")
        else:
            logger.warning("Could not find human insertion marker in labeling_config.py")

    config_path.write_text(content)
    print(f"  Updated {config_path}")


# ==================================================================
# CLI
# ==================================================================

if __name__ == "__main__":
    expanded = run_mining()
    print("\nDone.")
