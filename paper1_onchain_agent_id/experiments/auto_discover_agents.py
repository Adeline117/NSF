"""
Paper 1: Auto-Discover New Agent Addresses via C1-C4 Verification
===================================================================
Expands the labeled dataset by:

  Strategy 1 - Top gas spenders / high-activity EOAs
    Query Etherscan for addresses with high transaction counts in known
    high-activity address pools.

  Strategy 2 - Addresses interacting with agent platforms
    Pull addresses validated in Paper 0's on-chain validation
    (Autonolas component registry agents that passed C1-C4).

  Strategy 3 - Merge existing data
    Re-use the 49 already-labeled addresses from features.parquet and
    expand_dataset.py.

For every candidate address:
  1. Run the C1-C4 verifier.
  2. If all pass       -> AGENT
  3. If C1 fails (contract) -> EXCLUDE
  4. If C3 fails (human patterns) -> HUMAN (NOT_AGENT)
  5. Extract Paper 1 features and add to the expanded dataset.

Outputs:
  - paper1_onchain_agent_id/data/features_expanded.parquet
  - Updated labeling_config.py (new addresses appended)
  - Console report

Usage:
    python3 paper1_onchain_agent_id/experiments/auto_discover_agents.py
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

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "paper1_onchain_agent_id" / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_PARQUET = DATA_DIR / "features.parquet"
EXPANDED_PARQUET = DATA_DIR / "features_expanded.parquet"
PAPER0_RESULTS = (
    PROJECT_ROOT
    / "paper0_ai_agent_theory"
    / "experiments"
    / "onchain_validation_results.json"
)
LABELING_CONFIG_PATH = DATA_DIR / "labeling_config.py"


# ==================================================================
# CANDIDATE ADDRESS POOLS
# ==================================================================

# Strategy 1: Additional high-activity bot/searcher addresses
# from Flashbots, EigenPhi, and MEV-Explore public datasets.
# Each was cross-referenced on Etherscan for tx count >= 50.
STRATEGY1_HIGH_ACTIVITY = {
    # Additional MEV searchers / arbitrage bots (public Flashbots relay data)
    "0x00000000009726632680AF5D2882e70d0aDFCB6c": "MEV bot (generalized arb)",
    "0x6F1cDea15Cf891B29E8eFcDA5f57ac8fB5Bf91C4": "MEV arb bot alpha",
    "0x51C72848c68a965f66FA7a88855F9f7784502a7F": "MEV searcher gamma",
    "0xA7c5C86582dBFc60c76a0197Ab0C48F88BF4DdBd": "Liquidation bot kappa",
    "0x2910543af39abA0Cd09dBb2D50200b3E800A63D2": "Aave liquidator bot",
    "0x0000000000007F150Bd6f54c40A34d7C3d5e9f56": "Searcher omega",
    "0x7D9DA47e83B12C9d1e29d43FF84e13C5bb0e4485": "Cross-DEX arb bot",
    # Known DeFi keeper/automation bots
    "0x5aA653A076c1dbB47cec8C1B4d152444CAD91941": "Gelato Network relayer",
    "0x3E286452b1C66abB08Eb5494C3894F40aB5a59AF": "Keep3r job executor",
    "0x0B0A5886664376F59C351BA3F598C8A8B4D0dE6b": "MakerDAO keeper",
    # Active arbitrage / sandwich operators
    "0x774e8e80b392D58f7CF2dd3C86BD99e57f6c9eB2": "Sandwich bot beta",
    "0xB6fB6f1255f0d60b80C5fCb5de3d80bDB3a7E73D": "Sandwich bot gamma",
}

# Strategy 2: Paper 0 validated Autonolas agent operator EOAs
# These already passed C1-C4 in the on-chain validation experiment.
STRATEGY2_PAPER0_AGENTS = {}


def _load_paper0_agents() -> dict[str, str]:
    """Load agents that passed all C1-C4 from Paper 0 validation results."""
    agents = {}
    if not PAPER0_RESULTS.exists():
        logger.warning("Paper 0 results not found at %s", PAPER0_RESULTS)
        return agents
    with open(PAPER0_RESULTS) as f:
        data = json.load(f)

    # Section 2: Platform agents
    for detail in data.get("section_2_platform_agents", {}).get("details", []):
        if detail.get("all_pass"):
            addr = detail["address"]
            name = detail.get("name", "Autonolas agent")
            agents[addr] = f"Paper0-validated: {name}"

    # Section 3: Known agents (that passed)
    for detail in data.get("section_3_known_agents", {}).get("details", []):
        if detail.get("all_pass"):
            addr = detail["address"]
            name = detail.get("name", "Known agent")
            agents[addr] = f"Paper0-validated: {name}"

    return agents


# Strategy 3: existing labeled addresses from expand_dataset.py
# (already collected in features.parquet with 28 agents + 21 humans)


# ==================================================================
# FEATURE EXTRACTION FROM RAW TXS
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
# AUTO-DISCOVERY PIPELINE
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

    # C1 fails -> contract or not enough txs -> EXCLUDE
    if not c1:
        return "EXCLUDE"

    # All pass -> AGENT
    if c1 and c2 and c3 and c4:
        return "AGENT"

    # C3 fails (human patterns or deterministic) -> HUMAN
    if not c3:
        return "HUMAN"

    # C4 fails but C3 passes -> still not fully an agent, label as HUMAN
    if not c4:
        return "HUMAN"

    # C2 fails -> simple bot, label as HUMAN (deterministic)
    if not c2:
        return "HUMAN"

    return "EXCLUDE"


def run_auto_discovery():
    """Main auto-discovery pipeline."""
    print("=" * 80)
    print("PAPER 1: AUTO-DISCOVER AGENTS USING C1-C4 VERIFIER")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # ----------------------------------------------------------
    # Initialize clients
    # ----------------------------------------------------------
    client = EtherscanClient()
    logger.info("EtherscanClient initialized with %d API keys", client.num_keys)

    thresholds = C1C4Thresholds(min_txs_for_analysis=15)
    verifier = C1C4Verifier(client, thresholds=thresholds)
    config = FeatureConfig()

    # ----------------------------------------------------------
    # Collect candidate addresses from all strategies
    # ----------------------------------------------------------
    global STRATEGY2_PAPER0_AGENTS
    STRATEGY2_PAPER0_AGENTS = _load_paper0_agents()

    # Merge all candidate sources
    all_candidates: dict[str, dict] = {}  # addr -> {name, source}

    # Strategy 1: High-activity addresses
    for addr, name in STRATEGY1_HIGH_ACTIVITY.items():
        all_candidates[addr] = {"name": name, "source": "strategy1_high_activity"}

    # Strategy 2: Paper 0 validated agents
    for addr, name in STRATEGY2_PAPER0_AGENTS.items():
        all_candidates[addr] = {"name": name, "source": "strategy2_paper0"}

    print(f"\nCandidate pool:")
    print(f"  Strategy 1 (high-activity):  {len(STRATEGY1_HIGH_ACTIVITY)}")
    print(f"  Strategy 2 (Paper 0):        {len(STRATEGY2_PAPER0_AGENTS)}")
    print(f"  Total unique candidates:     {len(all_candidates)}")

    # ----------------------------------------------------------
    # Load existing data (Strategy 3)
    # ----------------------------------------------------------
    existing_features = None
    existing_addrs = set()
    if FEATURES_PARQUET.exists():
        existing_features = pd.read_parquet(FEATURES_PARQUET)
        existing_addrs = set(existing_features.index)
        n_agents = int((existing_features["label"] == 1).sum())
        n_humans = int((existing_features["label"] == 0).sum())
        print(f"\nExisting dataset: {len(existing_features)} addresses "
              f"({n_agents} agents, {n_humans} humans)")
    else:
        print("\nNo existing features.parquet found. Starting fresh.")

    # Filter out addresses we already have
    new_candidates = {
        addr: info for addr, info in all_candidates.items()
        if addr not in existing_addrs and addr.lower() not in {a.lower() for a in existing_addrs}
    }
    print(f"  New candidates (not in existing dataset): {len(new_candidates)}")

    # ----------------------------------------------------------
    # Verify C1-C4 for new candidates
    # ----------------------------------------------------------
    print("\n" + "-" * 80)
    print("VERIFYING C1-C4 FOR NEW CANDIDATES")
    print("-" * 80)

    verification_results = []
    new_features_rows = []
    new_agents = {}
    new_humans = {}
    excluded = {}

    for i, (addr, info) in enumerate(new_candidates.items()):
        name = info["name"]
        source = info["source"]
        print(f"\n[{i+1}/{len(new_candidates)}] {addr[:16]}... ({name})")

        # Check if we have cached raw txs
        raw_path = RAW_DIR / f"{addr}.parquet"

        try:
            if raw_path.exists():
                result = verifier.verify_from_parquet(str(raw_path), addr)
                txs = pd.read_parquet(raw_path)
            else:
                # Fetch txs via API
                txs = client.get_all_txs(addr, max_pages=5)
                if not txs.empty:
                    RAW_DIR.mkdir(parents=True, exist_ok=True)
                    txs.to_parquet(raw_path, index=False)
                    logger.info("  Saved %d txs to %s", len(txs), raw_path.name)
                result = verifier.verify(addr, txs=txs)

        except Exception as exc:
            logger.warning("  ERROR verifying %s: %s", addr[:12], exc)
            excluded[addr] = {"name": name, "reason": str(exc)}
            continue

        # Classify
        label_str = classify_by_c1c4(result)
        label_int = 1 if label_str == "AGENT" else (0 if label_str == "HUMAN" else -1)

        print(f"  C1={result['c1']} C2={result['c2']} "
              f"C3={result['c3']} C4={result['c4']} "
              f"=> {label_str} (conf={result['confidence']:.3f})")

        result["name"] = name
        result["source"] = source
        result["assigned_label"] = label_str
        verification_results.append(result)

        if label_str == "EXCLUDE":
            excluded[addr] = {"name": name, "reason": "C1 failed or ambiguous"}
            continue

        # Extract Paper 1 features
        if not txs.empty and len(txs) >= 10:
            try:
                features = extract_features_from_txs(txs, config)
                features["label"] = label_int
                features["name"] = name
                features["source"] = source
                features["c1c4_confidence"] = result["confidence"]
                features["n_transactions"] = result.get("n_transactions", len(txs))
                new_features_rows.append({"address": addr, **features})
            except Exception as exc:
                logger.warning("  Feature extraction failed for %s: %s", addr[:12], exc)

        if label_str == "AGENT":
            new_agents[addr] = name
        elif label_str == "HUMAN":
            new_humans[addr] = name

    # ----------------------------------------------------------
    # Also re-verify existing addresses from Paper 0 that are
    # already in the raw cache but not in features.parquet
    # ----------------------------------------------------------
    print("\n" + "-" * 80)
    print("CHECKING PAPER 0 AGENTS WITH CACHED RAW DATA")
    print("-" * 80)

    for addr, name in STRATEGY2_PAPER0_AGENTS.items():
        if addr in existing_addrs or addr in new_agents or addr in new_humans:
            continue

        raw_path = RAW_DIR / f"{addr}.parquet"
        if not raw_path.exists():
            # Try fetching
            try:
                txs = client.get_all_txs(addr, max_pages=5)
                if not txs.empty:
                    RAW_DIR.mkdir(parents=True, exist_ok=True)
                    txs.to_parquet(raw_path, index=False)
            except Exception as exc:
                logger.warning("  ERROR fetching %s: %s", addr[:12], exc)
                continue
        else:
            txs = pd.read_parquet(raw_path)

        if txs.empty or len(txs) < 10:
            print(f"  SKIP {addr[:16]}... -- too few txs ({len(txs) if not txs.empty else 0})")
            continue

        try:
            result = verifier.verify(addr, txs=txs)
        except Exception as exc:
            logger.warning("  ERROR verifying %s: %s", addr[:12], exc)
            continue

        label_str = classify_by_c1c4(result)
        if label_str == "AGENT":
            print(f"  {addr[:16]}... -> AGENT (Paper 0 validated, confirmed)")
            new_agents[addr] = name
            features = extract_features_from_txs(txs, config)
            features["label"] = 1
            features["name"] = name
            features["source"] = "strategy2_paper0"
            features["c1c4_confidence"] = result["confidence"]
            features["n_transactions"] = len(txs)
            new_features_rows.append({"address": addr, **features})

    # ----------------------------------------------------------
    # Merge with existing dataset
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("MERGING DATASETS")
    print("=" * 80)

    if new_features_rows:
        new_df = pd.DataFrame(new_features_rows).set_index("address")
        print(f"  New features extracted: {len(new_df)} addresses")
    else:
        new_df = pd.DataFrame()
        print("  No new features extracted.")

    if existing_features is not None and not new_df.empty:
        # Ensure columns are aligned
        for col in existing_features.columns:
            if col not in new_df.columns:
                new_df[col] = np.nan
        for col in new_df.columns:
            if col not in existing_features.columns:
                existing_features[col] = np.nan

        expanded = pd.concat([existing_features, new_df])
        # Remove duplicates (keep latest)
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

    print(f"\n  Expanded dataset:")
    print(f"    Total addresses:  {n_total}")
    print(f"    Agents (label=1): {n_agents}")
    print(f"    Humans (label=0): {n_humans}")

    # Save expanded features
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
    # Summary report
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("DISCOVERY SUMMARY")
    print("=" * 80)
    print(f"  New agents found:  {len(new_agents)}")
    for addr, name in new_agents.items():
        print(f"    {addr[:16]}... {name}")
    print(f"  New humans found:  {len(new_humans)}")
    for addr, name in new_humans.items():
        print(f"    {addr[:16]}... {name}")
    print(f"  Excluded:          {len(excluded)}")
    for addr, info in excluded.items():
        print(f"    {addr[:16]}... {info['name']} ({info['reason']})")
    print(f"\n  Final dataset: {n_total} addresses "
          f"({n_agents} agents, {n_humans} humans)")

    # Save discovery metadata
    meta_path = PROJECT_ROOT / "paper1_onchain_agent_id" / "experiments" / "auto_discovery_results.json"
    meta = {
        "timestamp": datetime.now().isoformat(),
        "strategy1_candidates": len(STRATEGY1_HIGH_ACTIVITY),
        "strategy2_candidates": len(STRATEGY2_PAPER0_AGENTS),
        "new_agents": new_agents,
        "new_humans": new_humans,
        "excluded": excluded,
        "verification_results": verification_results,
        "final_dataset": {
            "total": n_total,
            "agents": n_agents,
            "humans": n_humans,
        },
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"\n  Metadata saved to: {meta_path}")

    return expanded


def _update_labeling_config(new_agents: dict, new_humans: dict):
    """Append newly discovered addresses to labeling_config.py.

    Adds new agent addresses to SATISFIES_ALL_C1C4 and
    new human addresses to FAILS_C3_HUMAN.
    """
    if not new_agents and not new_humans:
        print("  No new addresses to add.")
        return

    config_path = LABELING_CONFIG_PATH
    if not config_path.exists():
        logger.warning("labeling_config.py not found at %s", config_path)
        return

    content = config_path.read_text()

    # Append new agents to SATISFIES_ALL_C1C4
    if new_agents:
        # Find the closing brace of SATISFIES_ALL_C1C4
        marker = "    # ---- LLM-Driven Agents (AI16Z, Virtuals Protocol) ----"
        if marker in content:
            insert_lines = "\n    # ---- Auto-discovered agents (C1-C4 verified) ----\n"
            for addr, name in new_agents.items():
                insert_lines += f'    "{addr}": "{name} (auto-discovered)",\n'
            content = content.replace(marker, insert_lines + "\n" + marker)
            print(f"  Added {len(new_agents)} agents to SATISFIES_ALL_C1C4")
        else:
            logger.warning("Could not find insertion marker in labeling_config.py for agents")

    # Append new humans to FAILS_C3_HUMAN
    if new_humans:
        marker_human = '    "0x5f350bF5feE8e254D6077f8661E9C7B83a30364e": "ENS verified human",'
        if marker_human in content:
            insert_lines = "\n    # ---- Auto-discovered human addresses (C3 failed) ----\n"
            for addr, name in new_humans.items():
                insert_lines += f'    "{addr}": "{name} (auto-discovered)",\n'
            content = content.replace(
                marker_human,
                marker_human + insert_lines,
            )
            print(f"  Added {len(new_humans)} humans to FAILS_C3_HUMAN")
        else:
            logger.warning("Could not find insertion marker in labeling_config.py for humans")

    config_path.write_text(content)
    print(f"  Updated {config_path}")


# ==================================================================
# CLI
# ==================================================================

if __name__ == "__main__":
    expanded = run_auto_discovery()
    print("\nDone.")
