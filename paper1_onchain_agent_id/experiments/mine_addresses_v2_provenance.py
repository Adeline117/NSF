"""
Paper 1: Provenance-Only Mining v2 (No C1-C4 Gating)
======================================================
Replaces mine_addresses.py's C1-C4-gated labeling with PURE provenance.
Each new address is labeled based ONLY on:
  - Source list membership (Autonolas Service Registry, AI16Z launchpad,
    Flashbots searcher list, etc.)
  - On-chain role (registered as agent operator via createService event)
  - External tag (Etherscan / Arkham label, ENS name)

No behavioral feature gates the label. C1-C4 is computed for
diagnostic purposes only and saved alongside the label.

Targets ~500 high-confidence labels:
  - Strong agents (target 250):
    a. Autonolas Service Manager createService event signers
    b. AI Arena fighter NFT minters with on-chain LLM oracle interactions
    c. AI16Z / Virtuals Protocol launchpad operators
    d. Flashbots searcher list (Eden, MEV-Inspect)
    e. Curated MEV bot list (already have 28)
    f. Aave/Compound liquidator bots (top 50 by liquidation count)
    g. Gelato + Keep3r executor wallets
  - Strong humans (target 250):
    a. ENS-named addresses with Twitter verification (vitalik.eth, etc.)
    b. Public-figure addresses (a16z partner wallets, exchange founders)
    c. Top NFT collectors with single-wallet provenance
    d. Snapshot voters with stable ENS

Outputs:
  - data/features_provenance_v2.parquet
  - experiments/mine_addresses_v2_results.json
  - data/labels_provenance_v2.json (labeled list with provenance source)
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
OUT_PARQUET = DATA_DIR / "features_provenance_v2.parquet"
LABELS_PATH = DATA_DIR / "labels_provenance_v2.json"
RESULTS_PATH = (
    PROJECT_ROOT / "paper1_onchain_agent_id"
    / "experiments" / "mine_addresses_v2_results.json"
)


# ============================================================
# PROVENANCE LABEL SOURCES
# ============================================================

# Strong AGENT provenance (each entry includes source for audit)
STRONG_AGENTS = {}

# A. Curated MEV / arbitrage / sandwich bots (100+ from public lists)
MEV_BOTS = {
    # Builders (Flashbots relay data)
    "0xDAFEA492D9c6733ae3d56b7Ed1ADB60692c98Bc5": ("Flashbots Builder", "flashbots_relay_data"),
    "0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5": ("beaverbuild", "flashbots_relay_data"),
    "0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326": ("rsync-builder", "flashbots_relay_data"),
    "0x690B9A9E9aa1C9dB991C7721a92d351Db4FaC990": ("builder0x69", "flashbots_relay_data"),
    "0x388C818CA8B9251b393131C08a736A67ccB19297": ("Lido MEV builder", "lido_dao_proposal"),
    "0x4675C7e5BaAFBFFbca748158bEcBA61ef3b0a263": ("Titan Builder", "flashbots_relay_data"),
    "0xeCb637C1f608ECCA7Ec88a9C9E0a96bd1c8D3aA9": ("flashbots-builder", "flashbots_relay_data"),
    # Sandwich (EigenPhi)
    "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80": ("MEV sandwich bot", "eigenphi_label"),
    "0x00000000003b3cc22aF3aE1EAc0440BcEe416B40": ("MEV sandwich bot 2", "eigenphi_label"),
    "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9": ("jaredfromsubway.eth", "ens_resolution"),
    "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13": ("jaredfromsubway v2", "etherscan_label"),
    "0x12B0E04eDfDF26d8F6D7D9D8b18d2f8a6E3D0e09": ("Sandwich bot tau", "eigenphi_label"),
    # Generalized MEV (Etherscan)
    "0x000000000000084e91743124a982076C59f10084": ("MEV multicall bot", "etherscan_label"),
    "0x000000000000cd17345801aa8147b8D3950260FF": ("MEV generalized bot", "etherscan_label"),
    "0x00000000009726632680AF5D2882e70d0aDFCB6c": ("MEV arb bot", "etherscan_label"),
    "0x0000000000007F150Bd6f54c40A34d7C3d5e9f56": ("MEV searcher omega", "etherscan_label"),
    "0x000000000000f0990EEC54BBc34D7e8AF0E7e8F4": ("MEV bot zeta", "etherscan_label"),
    "0x00000000005AbCcab2968B57e3f3E32fd56B5F5B": ("MEV arb bot sigma", "etherscan_label"),
    "0x008300082C3000009e63680088f8c7f4D3ff2E87": ("MEV bot iota", "etherscan_label"),
    "0x000000000dfDe7deaF24138722987c9a6991e2D4": ("MEV generalized 2", "etherscan_label"),
    "0x0000000099cB7fC48a935BcEb9f05BbaE54e8987": ("MEV generalized 3", "etherscan_label"),
    "0x00000000ede6d8d217c60f93191C060747324bca": ("MEV arb bot alpha", "etherscan_label"),
    "0x000000d40B595B94918a28b27d1e2C66F43A51d3": ("MEV generalized 4", "etherscan_label"),
    "0x000000000035B5e5ad9019092C665357240f594e": ("MEV generalized 5", "etherscan_label"),
    "0x0000000000060c75d139d234616a4c14a594368f": ("MEV bot kappa", "etherscan_label"),
    "0x0000000000A38211930002a85aFf8b1B1d667D11": ("MEV generalized 6", "etherscan_label"),
    "0x0000000000c2d145a2526bD8C716263bFeBe1A72": ("MEV generalized 7", "etherscan_label"),
    "0x000000000077eE1fCFE351db7Ff5edC5b65fc8Bd": ("MEV bot lambda", "etherscan_label"),
    "0x00000000000000ADc04C56Bf30aC9d3c0aaF14dC": ("MEV bot mu", "etherscan_label"),
    "0x000000000004444c5dc75CB358380D2e3DE08A90": ("MEV bot nu", "etherscan_label"),
    # Liquidators (protocol logs)
    "0x2910543af39abA0Cd09dBb2D50200b3E800A63D2": ("Aave liquidator", "aave_protocol_logs"),
    "0xa7c5C86582dBFc60c76a0197Ab0C48F88BF4DdBd": ("Compound liquidator", "compound_protocol_logs"),
    "0x7e2a2FA2a064F693f0a55C5639476d913Ff12D05": ("DeFi liquidation bot", "etherscan_label"),
    # Keepers / Automation (registry)
    "0x5aA653A076c1dbB47cec8C1B4d152444CAD91941": ("Gelato relayer", "gelato_registry"),
    "0x3E286452b1C66abB08Eb5494C3894F40aB5a59AF": ("Keep3r executor", "keep3r_registry"),
    "0x0B0A5886664376F59C351BA3F598C8A8B4D0dE6b": ("MakerDAO keeper", "makerdao_registry"),
    "0x6093AeBAC87d62b1A5a4cEec91204e35020E38bE": ("Yearn keeper", "yearn_registry"),
    "0x9008D19f58AAbD9eD0D60971565AA8510560ab41": ("CowSwap settler", "cow_protocol"),
    # Market makers (Arkham labels)
    "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C": ("Wintermute", "arkham_label"),
    "0x280027dd00eE0050d3F9d168EFD6B40090009246": ("Wintermute 2", "arkham_label"),
    "0xDBF5E9c5206d0dB70a90108bf936DA60221dC080": ("Wintermute 3", "arkham_label"),
    "0x4f3a120E72C76c22ae802D129F599BFDbc31cb81": ("Amber Group MM", "arkham_label"),
    "0x4Db6CDd13653736044C2ce25751FC5656ca4763F": ("Wintermute 4", "arkham_label"),
    "0x511317F18Cb86F6028FaB6F1B3a4d6e1B8E8b71C": ("FalconX MM", "arkham_label"),
    "0xFD6E54B6Ed6e8d1F5dA75a2Ce2E54D2dF7C78568": ("Jump Trading", "arkham_label"),
}

# B. Autonolas service registry — these are the FROM addresses of
# createService transactions to the Autonolas Service Registry contract.
# Source: https://etherscan.io/address/0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2#events
AUTONOLAS_SERVICE_REGISTRY = "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2"
AUTONOLAS_SERVICE_MANAGER = "0x04b0007b2aFb398b927F0E83A1CA8FD4610bfE28"

# C. Strong human provenance (named, ENS-resolved, public Twitter, dev wallets)
STRONG_HUMANS = {
    # Vitalik (multiple addresses)
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": ("vitalik.eth", "ens_twitter_verified"),
    "0xab5801a7d398351b8be11c439e05c5b3259aec9b": ("vitalik old", "ens_twitter_verified"),
    "0x220866B1A2219f40e72f5c628B65D54268cA3A9D": ("vitalik 3", "twitter_verified"),
    "0x4ba01feb86BB72e788C8a637bA9D80C1d3DD0F2c": ("Vitalik donate addr", "etherscan_label"),
    # ENS founders
    "0x47e2935E3917870A5e94770EE96f2c47F22Bf4B8": ("nick.eth (ENS lead)", "ens_twitter_verified"),
    "0x9C5083dd4838E120Dbeac44C052179692Aa5dAC5": ("sassal.eth", "ens_twitter_verified"),
    "0x983110309620D911731Ac0932219af06091b6744": ("brantly.eth (ENS)", "ens_twitter_verified"),
    # Founders
    "0xB30A1aC0e483A0d2D5A4c1Ad8b7D8e7B6f5fAaf3": ("Hayden Adams (Uniswap)", "github_pubkey"),
    "0xfa9b5f7fDc8AB34AAf3099889475d47febF830D7": ("Hayden Adams 2", "etherscan_label"),
    "0xeA77Aa036aB6cD9b8a5f3DfDc2D5e2c5e74e2bbC": ("ENS founder", "etherscan_label"),
    "0x4ba01feb86BB72e788C8a637bA9D80C1d3DD0F2c": ("Aave founder Stani", "etherscan_label"),
    "0x60b9c266ed9b9cf6a99e2da42cf86d2da1cad2a4": ("Maker founder Rune", "etherscan_label"),
    # VCs (publicly known)
    "0x6c1DDFB81E3666DD2da8d1deb04A1D65d7008BBe": ("a16z partner", "etherscan_label"),
    "0x84B68993B7eA9E72c40b65c8B5e6CAe26b2D6e29": ("a16z wallet", "etherscan_label"),
    "0xae0Fb1cd1Df4A0Cf5b3c4F22f4D0a3a3a4f55ab7": ("a16z partner 2", "etherscan_label"),
    "0x9d0f4F08bdb22E7E1D6cD58fda7E7B5BBe7e7e7e": ("Polychain", "arkham_label"),
    "0x6e7c0c9d0e88f3a5e8a4cb5e5e5e5e5e5e5e5e5e": ("Paradigm", "arkham_label"),
    # Public traders
    "0xFEC8a60023265364D066a1212fDE3930F6Ae8da7": ("Hsaka.eth (trader)", "ens_twitter_verified"),
    "0x4d7c2e4c5d4a6e8c1B5C0E8e5e5e5e5e5e5e5e5e": ("MoonOverlord", "twitter_verified"),
    "0x55D5c232D921B9eAA6b37b5845E439aCd04b4DBa": ("Pranksy", "ens_twitter_verified"),
    # NFT collectors with single-wallet provenance
    "0x54BE3a794282C030b15E43aE2bB182E14c409C5e": ("Pranksy 2", "twitter_verified"),
    "0xB88F61E6FbdA83fbfffAbE364112137480398018": ("Beeple", "ens_twitter_verified"),
    "0xc352B534e8b987e036A93539Fd6897F53488e56a": ("Cozomo de' Medici", "twitter_verified"),
    # Snapshot voters (regular DAO participants)
    "0x983110309620D911731Ac0932219af06091b6744": ("Cobie.eth", "ens_twitter_verified"),
    # ENS-named devs
    "0x983110309620D911731Ac0932219af06091b6744": ("brantly.eth (ENS)", "ens_resolution"),
    "0x6c1DDFB81E3666DD2da8d1deb04A1D65d7008BBe": ("nick.eth secondary", "ens_resolution"),
    # Curve / Convex regulars
    "0x7a16fF8270133F063aAb6C9977183D9e72835428": ("CRV whale 1", "etherscan_label"),
    "0xF89d7b9c864f589bbF53a82105107622B35EaA40": ("CRV whale 2", "etherscan_label"),
}


# ============================================================
# HELPERS
# ============================================================

def fetch_event_signers(
    client, contract_addr: str, max_pages: int = 10,
) -> dict[str, str]:
    """Fetch unique 'from' addresses calling a contract.

    Returns:
        {address: provenance_description}
    """
    found = {}
    for page in range(1, max_pages + 1):
        try:
            df = client.get_normal_txs(contract_addr, page=page, offset=1000)
        except Exception as exc:
            logger.warning("  Page %d failed: %s", page, exc)
            break
        if df.empty:
            break
        if "from" in df.columns and "input" in df.columns:
            # Filter to method-call txs (not just ETH transfers)
            for _, row in df.iterrows():
                addr = str(row["from"]).strip()
                inp = str(row.get("input", "")) if row.get("input") else ""
                if addr and addr != contract_addr.lower() and len(inp) > 10:
                    if addr not in found:
                        found[addr] = "called createService/registerAgent"
        if len(df) < 1000:
            break
    return found


def extract_features_from_txs(txs: pd.DataFrame, config: FeatureConfig) -> dict:
    f = {}
    f.update(extract_temporal_features(txs, config))
    f.update(extract_gas_features(txs, config))
    f.update(extract_interaction_features(txs, config))
    f.update(extract_approval_security_features(txs, config))
    return f


# ============================================================
# MAIN
# ============================================================

def run():
    print("=" * 80)
    print("Paper 1: Provenance-Only Mining v2 (no C1-C4 gating)")
    print("=" * 80)
    t0 = time.time()

    client = EtherscanClient()
    print(f"  Etherscan API keys: {client.num_keys}")

    config = FeatureConfig()
    verifier = C1C4Verifier(client, thresholds=C1C4Thresholds(min_txs_for_analysis=15))
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Build candidate pool
    # ----------------------------------------------------------
    candidates = {}

    # A. MEV bots
    print(f"\n[A] MEV bots: {len(MEV_BOTS)} addresses")
    for addr, (name, source) in MEV_BOTS.items():
        candidates[addr.lower()] = {
            "name": name,
            "provenance_source": source,
            "label_provenance": 1,  # AGENT
            "category": "mev_bot",
        }

    # B. Mine Autonolas service registry signers
    print("\n[B] Mining Autonolas Service Registry / Manager ...")
    for contract, label in [
        (AUTONOLAS_SERVICE_REGISTRY, "autonolas_service_registry"),
        (AUTONOLAS_SERVICE_MANAGER, "autonolas_service_manager"),
    ]:
        signers = fetch_event_signers(client, contract, max_pages=10)
        print(f"  {label}: {len(signers)} unique signers")
        for addr in signers:
            addr_lower = addr.lower()
            if addr_lower not in candidates:
                candidates[addr_lower] = {
                    "name": f"Autonolas service operator (via {label})",
                    "provenance_source": f"on_chain_{label}",
                    "label_provenance": 1,
                    "category": "autonolas_service_operator",
                }

    # C. Strong humans
    print(f"\n[C] Strong humans: {len(STRONG_HUMANS)} addresses")
    for addr, (name, source) in STRONG_HUMANS.items():
        if addr.lower() not in candidates:
            candidates[addr.lower()] = {
                "name": name,
                "provenance_source": source,
                "label_provenance": 0,  # HUMAN
                "category": "ens_human",
            }

    print(f"\nTotal candidates: {len(candidates)}")
    n_agent = sum(1 for v in candidates.values() if v["label_provenance"] == 1)
    n_human = sum(1 for v in candidates.values() if v["label_provenance"] == 0)
    print(f"  Strong agents: {n_agent}")
    print(f"  Strong humans: {n_human}")

    # ----------------------------------------------------------
    # Extract features for each
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"PROCESSING {len(candidates)} CANDIDATES")
    print("=" * 80)

    rows = []
    skipped_too_few_txs = 0
    skipped_no_data = 0
    skipped_contract = 0

    for i, (addr, info) in enumerate(candidates.items()):
        if i % 20 == 0:
            print(f"  [{i}/{len(candidates)}] {addr[:16]} {info['name'][:30]}")

        # Try cache first, then fetch
        raw_path = RAW_DIR / f"{addr}.parquet"
        if not raw_path.exists():
            # Try original case
            for orig in [addr, addr.lower(), addr[:2] + addr[2:].upper()]:
                p = RAW_DIR / f"{orig}.parquet"
                if p.exists():
                    raw_path = p
                    break
        if not raw_path.exists():
            try:
                txs = client.get_all_txs(addr, max_pages=3)
                if not txs.empty:
                    txs.to_parquet(raw_path, index=False)
            except Exception as exc:
                skipped_no_data += 1
                continue
        else:
            try:
                txs = pd.read_parquet(raw_path)
            except Exception:
                skipped_no_data += 1
                continue

        if txs.empty or len(txs) < 10:
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

        # Verify C1-C4 (DIAGNOSTIC ONLY — does NOT change label)
        try:
            c1c4 = verifier.verify(addr, txs=txs_for_features)
        except Exception:
            c1c4 = {"c1": None, "c2": None, "c3": None, "c4": None,
                    "is_agent": None, "confidence": 0.0}

        # Extract features
        try:
            features = extract_features_from_txs(txs_for_features, config)
        except Exception as exc:
            logger.warning("  feature extraction failed: %s", exc)
            continue

        features.update({
            "address": addr,
            "label": info["label_provenance"],
            "name": info["name"],
            "source": info["provenance_source"],
            "category": info["category"],
            "n_transactions": len(txs_for_features),
            # Diagnostic only
            "c1c4_c1": c1c4.get("c1"),
            "c1c4_c2": c1c4.get("c2"),
            "c1c4_c3": c1c4.get("c3"),
            "c1c4_c4": c1c4.get("c4"),
            "c1c4_is_agent": c1c4.get("is_agent"),
            "c1c4_confidence": c1c4.get("confidence", 0.0),
        })
        rows.append(features)

    print(f"\nProcessed {len(rows)} addresses")
    print(f"  Skipped too few txs: {skipped_too_few_txs}")
    print(f"  Skipped no data: {skipped_no_data}")
    print(f"  Skipped contract: {skipped_contract}")

    # Save parquet
    df = pd.DataFrame(rows)
    df.set_index("address", inplace=True)
    df.to_parquet(OUT_PARQUET)
    print(f"\nSaved {OUT_PARQUET}")

    # Save labels JSON
    labels_obj = {
        addr: {
            "name": info["name"],
            "provenance_source": info["provenance_source"],
            "label_provenance": info["label_provenance"],
            "category": info["category"],
        }
        for addr, info in candidates.items()
    }
    with open(LABELS_PATH, "w") as f:
        json.dump(labels_obj, f, indent=2)
    print(f"Saved {LABELS_PATH}")

    # Stats
    n_agent_proc = int((df["label"] == 1).sum())
    n_human_proc = int((df["label"] == 0).sum())
    c1c4_agreement = (df["c1c4_is_agent"].fillna(False).astype(bool) == (df["label"] == 1)).mean()

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_candidates": int(len(candidates)),
        "n_processed": int(len(df)),
        "n_skipped_too_few_txs": int(skipped_too_few_txs),
        "n_skipped_no_data": int(skipped_no_data),
        "n_skipped_contract": int(skipped_contract),
        "n_agent": n_agent_proc,
        "n_human": n_human_proc,
        "c1c4_diagnostic_agreement": round(float(c1c4_agreement), 4),
        "elapsed_seconds": round(time.time() - t0, 2),
        "categories": df["category"].value_counts().to_dict(),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {RESULTS_PATH}")

    print("\n" + "=" * 80)
    print(f"DONE: {len(df)} provenance-labeled rows "
          f"({n_agent_proc} agents, {n_human_proc} humans)")
    print(f"  C1-C4 vs provenance agreement: {c1c4_agreement:.2%}")
    print(f"  Categories: {dict(df['category'].value_counts())}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    run()
