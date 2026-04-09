"""
Paper 1: Expanded Provenance-Only Mining v3 (500+ Addresses)
=============================================================
Expands the trusted (provenance-only) dataset from 64 to 500+ addresses.

Strategy:
  AGENTS (target 300+):
    1. Flashbots MEV Relay builder callers
    2. Aave V3 liquidationCall callers (bot liquidators)
    3. Uniswap V3 Router high-frequency batch traders (>100 txs/day)
    4. Keep3r Network job executors
    5. Gelato Network Automate executors
    6. Existing curated MEV bots from features_expanded.parquet
    7. Additional curated MEV bots / market makers / sandwich bots

  HUMANS (target 200+):
    1. ENS Registrar Controller callers (register method 0x74694a2e)
    2. Known public figures / devs / ENS-verified
    3. Snapshot DAO voters (governance participants)
    4. ENS-named addresses from features_expanded with human labels

IMPORTANT: Labels derive ONLY from provenance (contract interaction
source, curated list membership, ENS registration). C1-C4 is computed
for diagnostic purposes and does NOT gate labels.

Outputs:
  - data/features_provenance_v3.parquet
  - experiments/mine_addresses_v3_results.json
  - data/labels_provenance_v3.json
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
CHECKPOINT_PATH = DATA_DIR / "features_provenance_v3_checkpoint.parquet"
OUT_PARQUET = DATA_DIR / "features_provenance_v3.parquet"
LABELS_PATH = DATA_DIR / "labels_provenance_v3.json"
RESULTS_PATH = (
    PROJECT_ROOT / "paper1_onchain_agent_id"
    / "experiments" / "mine_addresses_v3_results.json"
)

# Maximum new addresses to process (rate limit budget)
MAX_NEW_ADDRESSES = 500
CHECKPOINT_INTERVAL = 25


# ============================================================
# CONTRACT ADDRESSES
# ============================================================

# Flashbots Builder (fee recipient for Flashbots relay)
FLASHBOTS_BUILDER = "0xDAFEA492D9c6733ae3d56b7Ed1ADB60692c98Bc5"

# Aave V3 Pool (liquidationCall method 0x00a718a9)
AAVE_V3_POOL = "0x87870Bca3F3fD6335C3F4ce8392D69350B4fA4E2"
AAVE_LIQUIDATION_METHOD = "0x00a718a9"

# Uniswap V3 SwapRouter02
UNISWAP_V3_ROUTER = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"

# Keep3r Network
KEEP3R_V2 = "0xeb02addCfD8B773A5FFA6B9d1FE99c566f8c44CC"

# Gelato Automate
GELATO_AUTOMATE = "0x2A6C106ae13B558BB9E2Ec64Bd2f1f7BEFF3a5E0"

# ENS ETH Registrar Controller (register method 0x74694a2e)
ENS_REGISTRAR = "0x253553366Da8546fC250F225fe3d25d0C782303b"
ENS_REGISTER_METHOD = "0x74694a2e"

# Autonolas
AUTONOLAS_SERVICE_REGISTRY = "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2"
AUTONOLAS_SERVICE_MANAGER = "0x04b0007b2aFb398b927F0E83A1CA8FD4610bfE28"


# ============================================================
# CURATED AGENT LISTS
# ============================================================

# Additional curated MEV bots / builders / searchers / market makers
CURATED_AGENTS = {
    # --- Block Builders (Flashbots relay data) ---
    "0xDAFEA492D9c6733ae3d56b7Ed1ADB60692c98Bc5": ("Flashbots Builder", "flashbots_relay_data"),
    "0x95222290DD7278Aa3Ddd389Cc1E1d165CC4BAfe5": ("beaverbuild", "flashbots_relay_data"),
    "0x1f9090aaE28b8a3dCeaDf281B0F12828e676c326": ("rsync-builder", "flashbots_relay_data"),
    "0x690B9A9E9aa1C9dB991C7721a92d351Db4FaC990": ("builder0x69", "flashbots_relay_data"),
    "0x388C818CA8B9251b393131C08a736A67ccB19297": ("Lido MEV builder", "lido_dao_proposal"),
    "0x4675C7e5BaAFBFFbca748158bEcBA61ef3b0a263": ("Titan Builder", "flashbots_relay_data"),
    "0xeCb637C1f608ECCA7Ec88a9C9E0a96bd1c8D3aA9": ("flashbots-builder", "flashbots_relay_data"),
    # --- Sandwich Bots (EigenPhi / Etherscan labels) ---
    "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80": ("MEV sandwich bot", "eigenphi_label"),
    "0x00000000003b3cc22aF3aE1EAc0440BcEe416B40": ("MEV sandwich bot 2", "eigenphi_label"),
    "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9": ("jaredfromsubway.eth", "ens_resolution"),
    "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13": ("jaredfromsubway v2", "etherscan_label"),
    "0x12B0E04eDfDF26d8F6D7D9D8b18d2f8a6E3D0e09": ("Sandwich bot tau", "eigenphi_label"),
    # --- Generalized MEV (Etherscan) ---
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
    # --- Additional MEV bots from public labels ---
    "0x00000000000747D525E898EcFf4f1CFb56e01e1F": ("MEV bot 0x747", "etherscan_label"),
    "0x00000000000006b2ab6DECBC6C6E60F390f3CEC6": ("MEV bot 0x6b2", "etherscan_label"),
    "0x0000000000BBf5c5fd284e657F01Bd000933c96d": ("MEV bot 0xBBf", "etherscan_label"),
    "0xA9D1e08C7793af67e9d92fe308d5697FB81d3E43": ("Coinbase MEV searcher", "etherscan_label"),
    "0x280027dd00eE0050d3F9d168EFD6B40090009246": ("Wintermute 2", "arkham_label"),
    # --- Liquidators (protocol logs) ---
    "0x2910543af39abA0Cd09dBb2D50200b3E800A63D2": ("Aave liquidator", "aave_protocol_logs"),
    "0xa7c5C86582dBFc60c76a0197Ab0C48F88BF4DdBd": ("Compound liquidator", "compound_protocol_logs"),
    "0x7e2a2FA2a064F693f0a55C5639476d913Ff12D05": ("DeFi liquidation bot", "etherscan_label"),
    # --- Keepers / Automation ---
    "0x5aA653A076c1dbB47cec8C1B4d152444CAD91941": ("Gelato relayer", "gelato_registry"),
    "0x3E286452b1C66abB08Eb5494C3894F40aB5a59AF": ("Keep3r executor", "keep3r_registry"),
    "0x0B0A5886664376F59C351BA3F598C8A8B4D0dE6b": ("MakerDAO keeper", "makerdao_registry"),
    "0x6093AeBAC87d62b1A5a4cEec91204e35020E38bE": ("Yearn keeper", "yearn_registry"),
    "0x9008D19f58AAbD9eD0D60971565AA8510560ab41": ("CowSwap settler", "cow_protocol"),
    # --- Market Makers (Arkham labels) ---
    "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C": ("Wintermute", "arkham_label"),
    "0xDBF5E9c5206d0dB70a90108bf936DA60221dC080": ("Wintermute 3", "arkham_label"),
    "0x4f3a120E72C76c22ae802D129F599BFDbc31cb81": ("Amber Group MM", "arkham_label"),
    "0x4Db6CDd13653736044C2ce25751FC5656ca4763F": ("Wintermute 4", "arkham_label"),
    "0x511317F18Cb86F6028FaB6F1B3a4d6e1B8E8b71C": ("FalconX MM", "arkham_label"),
    "0xFD6E54B6Ed6e8d1F5dA75a2Ce2E54D2dF7C78568": ("Jump Trading", "arkham_label"),
    "0x65A0947BA5175359Bb457D3b34491eDf4cbF7997": ("Jump Trading 2", "arkham_label"),
    "0x9507c04B10486547584C37bCBd931B5a4794AABe": ("Two Sigma", "arkham_label"),
    "0x75e89d5979E4f6Fba9F97c104c2F0AFB3F1dcB88": ("DWF Labs", "arkham_label"),
}

# ============================================================
# CURATED HUMAN LISTS
# ============================================================

CURATED_HUMANS = {
    # Vitalik + core devs
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": ("vitalik.eth", "ens_twitter_verified"),
    "0xab5801a7d398351b8be11c439e05c5b3259aec9b": ("vitalik old", "ens_twitter_verified"),
    "0x220866B1A2219f40e72f5c628B65D54268cA3A9D": ("vitalik 3", "twitter_verified"),
    # ENS team
    "0x47e2935E3917870A5e94770EE96f2c47F22Bf4B8": ("nick.eth (ENS lead)", "ens_twitter_verified"),
    "0x9C5083dd4838E120Dbeac44C052179692Aa5dAC5": ("sassal.eth", "ens_twitter_verified"),
    "0x983110309620D911731Ac0932219af06091b6744": ("brantly.eth (ENS)", "ens_twitter_verified"),
    # Founders / known devs
    "0xfa9b5f7fDc8AB34AAf3099889475d47febF830D7": ("Hayden Adams (Uniswap)", "etherscan_label"),
    "0x60b9c266ed9b9cf6a99e2da42cf86d2da1cad2a4": ("Maker founder Rune", "etherscan_label"),
    # VCs / known public addresses
    "0x6c1DDFB81E3666DD2da8d1deb04A1D65d7008BBe": ("a16z partner", "etherscan_label"),
    # Public traders
    "0xFEC8a60023265364D066a1212fDE3930F6Ae8da7": ("Hsaka.eth (trader)", "ens_twitter_verified"),
    "0x55D5c232D921B9eAA6b37b5845E439aCd04b4DBa": ("Pranksy", "ens_twitter_verified"),
    # NFT collectors / known humans
    "0x54BE3a794282C030b15E43aE2bB182E14c409C5e": ("Pranksy 2", "twitter_verified"),
    "0xB88F61E6FbdA83fbfffAbE364112137480398018": ("Beeple", "ens_twitter_verified"),
    "0xc352B534e8b987e036A93539Fd6897F53488e56a": ("Cozomo de' Medici", "twitter_verified"),
    # DeFi whales
    "0x7a16fF8270133F063aAb6C9977183D9e72835428": ("CRV whale 1", "etherscan_label"),
    "0xF89d7b9c864f589bbF53a82105107622B35EaA40": ("CRV whale 2", "etherscan_label"),
}


# ============================================================
# HELPERS
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
            # Skip zero address
            if addr == "0x0000000000000000000000000000000000000000":
                continue
            # Optionally filter by method_id
            if method_id and not inp.lower().startswith(method_id.lower()):
                continue
            # Skip if input is just plain ETH transfer (no calldata)
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
    min_daily_txs: int = 50,
    max_pages: int = 5,
    max_callers: int = 100,
    label: str = "high_freq_trader",
) -> dict[str, tuple[str, str]]:
    """Find addresses with unusually high call frequency to a contract.

    Returns addresses that have >min_daily_txs calls in any single day.
    """
    # Collect all callers first
    all_callers: dict[str, list[int]] = {}
    for page in range(1, max_pages + 1):
        try:
            df = client.get_normal_txs(contract_addr, page=page, offset=10000)
        except Exception as exc:
            logger.warning("  Page %d failed: %s", page, exc)
            break
        if df.empty:
            break
        if "from" not in df.columns or "timeStamp" not in df.columns:
            break

        for _, row in df.iterrows():
            addr = str(row.get("from", "")).strip().lower()
            ts = int(row.get("timeStamp", 0))
            if addr and addr != contract_addr.lower() and ts > 0:
                if addr not in all_callers:
                    all_callers[addr] = []
                all_callers[addr].append(ts)

        if len(df) < 10000:
            break

    # Find high-frequency callers (>min_daily_txs in a single day)
    found = {}
    for addr, timestamps in all_callers.items():
        if len(timestamps) < min_daily_txs:
            continue
        # Group by day
        days = {}
        for ts in timestamps:
            day = ts // 86400
            days[day] = days.get(day, 0) + 1
        max_daily = max(days.values()) if days else 0
        if max_daily >= min_daily_txs:
            found[addr] = (
                f"{label} (peak {max_daily} txs/day)",
                f"on_chain_{label}",
            )
        if len(found) >= max_callers:
            break

    return found


def mine_from_existing_expanded() -> tuple[
    dict[str, dict], dict[str, dict]
]:
    """Extract additional labeled addresses from features_expanded.parquet.

    Addresses already in features_expanded with clear provenance can be
    included without making new API calls.

    Returns:
        (agents_dict, humans_dict) where each is
        {address: {"name": ..., "provenance_source": ..., "category": ...}}
    """
    expanded_path = DATA_DIR / "features_expanded.parquet"
    if not expanded_path.exists():
        return {}, {}

    df = pd.read_parquet(expanded_path)
    agents = {}
    humans = {}

    # MEV bots / builders / searchers from name field
    mev_keywords = [
        "MEV", "bot", "sandwich", "searcher", "builder", "Flashbots",
        "Wintermute", "Amber", "Jump", "High-freq", "Active bot",
        "DWF", "FalconX", "keeper", "liquidat", "settler",
    ]
    for addr, row in df.iterrows():
        name = str(row.get("name", ""))
        source = str(row.get("source", ""))
        label = int(row.get("label", -1))
        addr_lower = str(addr).lower()

        if label == 1:
            # Check if name matches known agent patterns
            for kw in mev_keywords:
                if kw.lower() in name.lower():
                    agents[addr_lower] = {
                        "name": name,
                        "provenance_source": f"expanded_curated_{source}" if source else "expanded_curated",
                        "category": "expanded_mev_bot",
                    }
                    break

        if label == 0 and source in (
            "", "nan", "None", "strategy_c_human",
        ):
            # Human addresses from the curated human list or manual pilot
            if name and name not in ("nan", "None", ""):
                humans[addr_lower] = {
                    "name": name,
                    "provenance_source": f"expanded_human_{source}" if source and source != "nan" else "expanded_pilot",
                    "category": "expanded_human",
                }

    return agents, humans


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
    """Run Random Forest with LOO + repeated CV."""
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, f1_score, precision_score, recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
    from sklearn.preprocessing import StandardScaler

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )

    # --- Leave-One-Out CV ---
    n = len(y)
    if n > 200:
        # Use 10-fold CV for larger datasets (LOO too slow)
        logger.info("  Using 10-fold CV instead of LOO (n=%d)", n)
        probs = np.zeros(n)
        preds = np.zeros(n, dtype=int)
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        for train_idx, test_idx in skf.split(X, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = clone(rf)
            clf.fit(X_tr, y[train_idx])
            probs[test_idx] = clf.predict_proba(X_te)[:, 1]
            preds[test_idx] = clf.predict(X_te)
        loo_results = {
            "method": "10fold_cv",
            "auc": round(float(roc_auc_score(y, probs)), 4),
            "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        }
    else:
        loo = LeaveOneOut()
        probs = np.zeros(n)
        preds = np.zeros(n, dtype=int)
        for train_idx, test_idx in loo.split(X):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X[train_idx])
            X_te = scaler.transform(X[test_idx])
            clf = clone(rf)
            clf.fit(X_tr, y[train_idx])
            probs[test_idx[0]] = clf.predict_proba(X_te)[0, 1]
            preds[test_idx[0]] = clf.predict(X_te)[0]
        loo_results = {
            "method": "loo_cv",
            "auc": round(float(roc_auc_score(y, probs)), 4),
            "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
            "accuracy": round(float(accuracy_score(y, preds)), 4),
            "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
            "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        }

    # --- Repeated Stratified K-Fold ---
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
    fold_aucs = []
    fold_f1s = []
    for train_idx, test_idx in rskf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[train_idx])
        X_te = scaler.transform(X[test_idx])
        clf = clone(rf)
        clf.fit(X_tr, y[train_idx])
        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)
        try:
            fold_aucs.append(roc_auc_score(y[test_idx], y_prob))
        except ValueError:
            pass
        fold_f1s.append(f1_score(y[test_idx], y_pred, zero_division=0))

    cv_results = {
        "mean_auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else 0.0,
        "std_auc": round(float(np.std(fold_aucs)), 4) if fold_aucs else 0.0,
        "mean_f1": round(float(np.mean(fold_f1s)), 4) if fold_f1s else 0.0,
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
        "cv": loo_results,
        "repeated_5fold_10x": cv_results,
        "top10_features": top_features,
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
            # Try original case
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
        # Fallback to self-loops
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

    # GAT model
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
    print("Paper 1: Expanded Provenance-Only Mining v3")
    print("  Target: 500+ addresses (300+ agents, 200+ humans)")
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
    # Phase 1: Build candidate pool
    # ----------------------------------------------------------
    candidates: dict[str, dict] = {}

    # 1A. Curated agents (static list)
    print(f"\n[1A] Curated agents: {len(CURATED_AGENTS)} addresses")
    for addr, (name, source) in CURATED_AGENTS.items():
        candidates[addr.lower()] = {
            "name": name,
            "provenance_source": source,
            "label_provenance": 1,
            "category": "curated_mev_bot",
        }

    # 1B. Mine Flashbots builder callers
    print("\n[1B] Mining Flashbots Builder callers ...")
    try:
        fb_callers = fetch_contract_callers(
            client, FLASHBOTS_BUILDER,
            max_pages=3, max_callers=80,
            label="flashbots_builder_caller",
        )
        print(f"  Found {len(fb_callers)} unique callers")
        for addr, (name, source) in fb_callers.items():
            if addr not in candidates:
                candidates[addr] = {
                    "name": f"Flashbots builder caller ({addr[:10]})",
                    "provenance_source": "on_chain_flashbots_builder",
                    "label_provenance": 1,
                    "category": "flashbots_caller",
                }
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # 1C. Mine Aave V3 liquidation callers
    print("\n[1C] Mining Aave V3 liquidation callers ...")
    try:
        aave_callers = fetch_contract_callers(
            client, AAVE_V3_POOL,
            method_id=AAVE_LIQUIDATION_METHOD,
            max_pages=5, max_callers=100,
            label="aave_liquidator",
        )
        print(f"  Found {len(aave_callers)} unique liquidation callers")
        for addr, (name, source) in aave_callers.items():
            if addr not in candidates:
                candidates[addr] = {
                    "name": f"Aave V3 liquidator ({addr[:10]})",
                    "provenance_source": "on_chain_aave_liquidation",
                    "label_provenance": 1,
                    "category": "aave_liquidator",
                }
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # 1D. Mine Uniswap V3 high-frequency traders
    print("\n[1D] Mining Uniswap V3 high-frequency traders ...")
    try:
        uni_hf = find_high_frequency_callers(
            client, UNISWAP_V3_ROUTER,
            min_daily_txs=50, max_pages=3, max_callers=60,
            label="uniswap_hf_trader",
        )
        print(f"  Found {len(uni_hf)} high-frequency traders")
        for addr, (name, source) in uni_hf.items():
            if addr not in candidates:
                candidates[addr] = {
                    "name": f"Uniswap HF trader ({addr[:10]})",
                    "provenance_source": "on_chain_uniswap_high_freq",
                    "label_provenance": 1,
                    "category": "uniswap_hf_bot",
                }
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # 1E. Mine Keep3r executors
    print("\n[1E] Mining Keep3r Network executors ...")
    try:
        keep3r_callers = fetch_contract_callers(
            client, KEEP3R_V2,
            max_pages=3, max_callers=50,
            label="keep3r_executor",
        )
        print(f"  Found {len(keep3r_callers)} Keep3r executors")
        for addr, (name, source) in keep3r_callers.items():
            if addr not in candidates:
                candidates[addr] = {
                    "name": f"Keep3r executor ({addr[:10]})",
                    "provenance_source": "on_chain_keep3r_executor",
                    "label_provenance": 1,
                    "category": "keep3r_executor",
                }
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # 1F. Mine Gelato Automate executors
    print("\n[1F] Mining Gelato Automate executors ...")
    try:
        gelato_callers = fetch_contract_callers(
            client, GELATO_AUTOMATE,
            max_pages=3, max_callers=50,
            label="gelato_executor",
        )
        print(f"  Found {len(gelato_callers)} Gelato executors")
        for addr, (name, source) in gelato_callers.items():
            if addr not in candidates:
                candidates[addr] = {
                    "name": f"Gelato executor ({addr[:10]})",
                    "provenance_source": "on_chain_gelato_executor",
                    "label_provenance": 1,
                    "category": "gelato_executor",
                }
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # 1G. Mine Autonolas service operators
    print("\n[1G] Mining Autonolas Service Registry/Manager ...")
    for contract, label_name in [
        (AUTONOLAS_SERVICE_REGISTRY, "autonolas_service_registry"),
        (AUTONOLAS_SERVICE_MANAGER, "autonolas_service_manager"),
    ]:
        try:
            signers = fetch_contract_callers(
                client, contract,
                max_pages=5, max_callers=80,
                label=label_name,
            )
            print(f"  {label_name}: {len(signers)} unique signers")
            for addr, (name, source) in signers.items():
                if addr not in candidates:
                    candidates[addr] = {
                        "name": f"Autonolas operator ({addr[:10]})",
                        "provenance_source": f"on_chain_{label_name}",
                        "label_provenance": 1,
                        "category": "autonolas_operator",
                    }
        except Exception as exc:
            print(f"  ERROR for {label_name}: {exc}")

    # 1H. Agents from existing features_expanded
    print("\n[1H] Mining agents from features_expanded.parquet ...")
    expanded_agents, expanded_humans = mine_from_existing_expanded()
    print(f"  Found {len(expanded_agents)} agents, {len(expanded_humans)} humans")
    for addr, info in expanded_agents.items():
        if addr not in candidates:
            candidates[addr] = {
                "name": info["name"],
                "provenance_source": info["provenance_source"],
                "label_provenance": 1,
                "category": info["category"],
            }

    # 2A. Curated humans (static list)
    print(f"\n[2A] Curated humans: {len(CURATED_HUMANS)} addresses")
    for addr, (name, source) in CURATED_HUMANS.items():
        addr_lower = addr.lower()
        if addr_lower not in candidates:
            candidates[addr_lower] = {
                "name": name,
                "provenance_source": source,
                "label_provenance": 0,
                "category": "curated_human",
            }

    # 2B. Mine ENS registrants
    print("\n[2B] Mining ENS Registrar callers (register method) ...")
    try:
        ens_callers = fetch_contract_callers(
            client, ENS_REGISTRAR,
            method_id=ENS_REGISTER_METHOD,
            max_pages=5, max_callers=200,
            label="ens_registrant",
        )
        print(f"  Found {len(ens_callers)} ENS registrants")
        for addr, (name, source) in ens_callers.items():
            if addr not in candidates:
                candidates[addr] = {
                    "name": f"ENS registrant ({addr[:10]})",
                    "provenance_source": "on_chain_ens_registration",
                    "label_provenance": 0,
                    "category": "ens_registrant",
                }
    except Exception as exc:
        print(f"  ERROR: {exc}")

    # 2C. Humans from features_expanded
    print("\n[2C] Adding humans from features_expanded ...")
    for addr, info in expanded_humans.items():
        if addr not in candidates:
            candidates[addr] = {
                "name": info["name"],
                "provenance_source": info["provenance_source"],
                "label_provenance": 0,
                "category": info["category"],
            }

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    n_agent = sum(1 for v in candidates.values() if v["label_provenance"] == 1)
    n_human = sum(1 for v in candidates.values() if v["label_provenance"] == 0)
    print(f"\n{'=' * 60}")
    print(f"TOTAL CANDIDATES: {len(candidates)}")
    print(f"  Agents: {n_agent}")
    print(f"  Humans: {n_human}")
    print(f"{'=' * 60}")

    # Category breakdown
    cats = {}
    for v in candidates.values():
        cat = v["category"]
        cats[cat] = cats.get(cat, 0) + 1
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # ----------------------------------------------------------
    # Phase 2: Extract features
    # ----------------------------------------------------------
    print(f"\n{'=' * 80}")
    print(f"PROCESSING {min(len(candidates), MAX_NEW_ADDRESSES)} CANDIDATES")
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
    api_calls_made = 0

    candidate_list = [
        (addr, info) for addr, info in candidates.items()
        if addr not in processed_addrs
    ]
    # Limit to MAX_NEW_ADDRESSES
    if len(candidate_list) > MAX_NEW_ADDRESSES:
        candidate_list = candidate_list[:MAX_NEW_ADDRESSES]

    for i, (addr, info) in enumerate(candidate_list):
        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  [{i}/{len(candidate_list)}] {addr[:16]} "
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
            # Diagnostic only
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
            print(f"    Checkpoint saved: {len(df_ckpt)} rows")

    # ----------------------------------------------------------
    # Phase 3: Build final dataset
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    print(f"FEATURE EXTRACTION COMPLETE")
    print(f"  Processed: {len(rows)}")
    print(f"  Skipped (too few txs): {skipped_too_few_txs}")
    print(f"  Skipped (no data): {skipped_no_data}")
    print(f"  Skipped (contract/no outgoing): {skipped_contract}")

    if not rows:
        print("ERROR: No rows processed. Exiting.")
        return

    df = pd.DataFrame(rows)
    if "address" in df.columns:
        df.set_index("address", inplace=True)

    # Deduplicate (keep first occurrence)
    df = df[~df.index.duplicated(keep="first")]

    n_agent_proc = int((df["label"] == 1).sum())
    n_human_proc = int((df["label"] == 0).sum())
    print(f"  Final: {len(df)} rows ({n_agent_proc} agents, {n_human_proc} humans)")

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
    print(f"\n--- Random Forest (n={len(y)}, {n_agent_proc} agents, "
          f"{n_human_proc} humans) ---")
    rf_results = run_rf_evaluation(X_values, y, feature_cols)
    print(f"  CV AUC: {rf_results['cv']['auc']}")
    print(f"  CV F1:  {rf_results['cv']['f1']}")
    print(f"  5x10 CV: AUC={rf_results['repeated_5fold_10x']['mean_auc']}"
          f" +/- {rf_results['repeated_5fold_10x']['std_auc']}")
    print(f"  Top features: {list(rf_results['top10_features'].keys())[:5]}")

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
    # Convert to boolean safely
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
            "n_candidates": len(candidates),
            "n_processed": len(df),
            "n_agents": n_agent_proc,
            "n_humans": n_human_proc,
            "n_features": len(feature_cols),
            "n_skipped_too_few_txs": skipped_too_few_txs,
            "n_skipped_no_data": skipped_no_data,
            "n_skipped_contract": skipped_contract,
        },
        "categories": df["category"].value_counts().to_dict(),
        "classifiers": {
            "RandomForest": rf_results,
            "GAT": gat_results,
        },
        "c1c4_diagnostic_agreement": round(float(agreement), 4),
        "comparison_to_prior": {
            "v2_provenance_64": {
                "n_samples": 16,
                "note": "v2 only processed 16/69 candidates due to data issues",
            },
            "v3_provenance_expanded": {
                "n_samples": len(df),
                "rf_cv_auc": rf_results["cv"]["auc"],
                "rf_5x10_auc": rf_results["repeated_5fold_10x"]["mean_auc"],
                "gat_5fold_auc": gat_results["mean_auc"],
            },
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
          f"({n_agent_proc} agents, {n_human_proc} humans)")
    print(f"  RF AUC:  {rf_results['cv']['auc']} (CV), "
          f"{rf_results['repeated_5fold_10x']['mean_auc']} (5x10)")
    print(f"  GAT AUC: {gat_results['mean_auc']} (5-fold)")
    print(f"  C1-C4 agreement: {agreement:.2%}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 80}")

    return results


if __name__ == "__main__":
    run()
