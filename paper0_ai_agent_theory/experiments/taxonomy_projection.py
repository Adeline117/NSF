"""
Paper 0: Rule-Based Taxonomy Projection onto Paper 1 Expanded Dataset
======================================================================
Projects each of the 2,590 labeled agents in Paper 1's expanded dataset
(features_expanded.parquet) onto one of the 8 taxonomy categories from
pilot_taxonomy.py. Uses a two-tier rule system:

  Tier 1: Provenance (source + name pattern) — strong prior
  Tier 2: Feature-based refinement — distinguishes within provenance

Categories (from pilot_taxonomy.TAXONOMY):
  0. Simple Trading Bot
  1. MEV Searcher
  2. DeFi Management Agent
  3. LLM-Powered Agent
  4. Autonomous DAO Agent
  5. Cross-Chain Bridge Agent
  6. Deterministic Script
  7. RL Trading Agent

Outputs:
  - features_with_taxonomy.parquet (copies expanded + adds taxonomy_category column)
  - taxonomy_projection_results.json (counts, confidence, examples)
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper0_ai_agent_theory.experiments.pilot_taxonomy import TAXONOMY

FEATURES_PARQUET = (
    PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_expanded.parquet"
)
OUT_PARQUET = (
    PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
)
RESULTS_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "taxonomy_projection_results.json"
)

TAXONOMY_NAMES = [t.name for t in TAXONOMY]

# ----------------------------------------------------------
# Provenance rules (high confidence)
# ----------------------------------------------------------

NAME_PATTERNS = [
    # (regex substring list, category_index, confidence, reason)
    (["jaredfromsubway", "sandwich", "MEV bot", "MEV searcher", "MEV multicall",
      "MEV sandwich", "MEV bot eta", "MEV bot iota", "searcher EOA",
      "Wintermute", "beaverbuild", "Flashbots builder", "Flashbots relay",
      "Flashbots block", "DeFi strategy bot"], 1, 0.95, "MEV/searcher name"),
    (["Autonolas Agent Registry", "Paper0-validated", "autonolas_agent_registry",
      "autonolas_component_registry"], 2, 0.90, "Autonolas agent registry"),
    (["Fetch.ai FET", "Fetch.ai", "FET Token"], 2, 0.75,
     "Fetch.ai token holder (DeFi management likely)"),
    (["AI Arena NRN", "AI Arena", "NRN Token"], 3, 0.75,
     "AI Arena NRN holder (LLM-powered agent likely)"),
    (["OLAS Token", "Autonolas governance", "Component Registry"], 2, 0.70,
     "Autonolas token/component holder"),
]

HUMAN_NAME_PATTERNS = [
    ".eth", "ENS", "Vitalik", "Hayden", "nick", "sassal", "bankless",
    "punk", "dev", "founder", "lead", "trader", "collector", "a16z",
    "SBF", "Alameda", "Binance", "Coinbase", "exchange",
]


def categorize_by_provenance(source, name) -> tuple[int, float, str]:
    """Return (category_index, confidence, reason) based on name pattern.

    Returns (-1, 0.0, 'unknown') if no rule matches.
    """
    if pd.isna(name):
        return -1, 0.0, "missing name"

    name_lower = str(name).lower()

    for patterns, cat_idx, conf, reason in NAME_PATTERNS:
        for pat in patterns:
            if pat.lower() in name_lower:
                return cat_idx, conf, reason

    # Check if name suggests human
    for hp in HUMAN_NAME_PATTERNS:
        if hp.lower() in name_lower:
            return -2, 0.9, f"human name pattern ({hp})"

    return -1, 0.0, "unknown provenance"


# ----------------------------------------------------------
# Feature-based refinement
# ----------------------------------------------------------

def refine_by_features(row: pd.Series, base_cat: int) -> tuple[int, float]:
    """Refine a provenance-based category using behavioral features.

    Returns (final_cat, refined_confidence).
    """
    burst = row.get("burst_frequency", 0.0) or 0.0
    gas_cv = row.get("gas_price_cv", 0.0) or 0.0
    unique_contracts = row.get("unique_contracts_ratio", 0.0) or 0.0
    multi_proto = row.get("multi_protocol_interaction_count", 0.0) or 0.0
    method_div = row.get("method_id_diversity", 0.0) or 0.0
    gas_round = row.get("gas_price_round_number_ratio", 0.0) or 0.0
    seq_pattern = row.get("sequential_pattern_score", 0.0) or 0.0
    active_hour = row.get("active_hour_entropy", 0.0) or 0.0

    # Very high burst + low gas CV → MEV Searcher
    if burst > 0.2 and gas_cv > 0.3 and unique_contracts < 0.3:
        return 1, 0.9  # MEV Searcher

    # High gas round + sequential pattern + low method diversity → Deterministic Script
    if gas_round > 0.6 and seq_pattern > 0.3 and method_div < 1.5:
        return 6, 0.8  # Deterministic Script

    # High multi_protocol + moderate unique_contracts → DeFi Management
    if multi_proto >= 5 and unique_contracts > 0.4:
        if base_cat in (2, -1):
            return 2, 0.85  # DeFi Management Agent

    # Very high unique_contracts + high active_hour + high method_div → LLM-Powered
    if unique_contracts > 0.6 and active_hour > 3.0 and method_div > 2.0:
        if base_cat in (3, -1):
            return 3, 0.75  # LLM-Powered Agent

    # Cross-chain indicator: bridge-like contract use (proxy via
    # moderate unique_contracts and specific patterns is weak; fall through)
    # Simple Trading Bot: moderate burst + low method diversity + high gas precision
    if burst > 0.05 and method_div < 2.0 and gas_round > 0.4:
        return 0, 0.7  # Simple Trading Bot

    # Fall through: keep base category if any
    if base_cat >= 0:
        return base_cat, 0.6

    # Default for agents we can't categorize: call them DeFi Management Agent
    # (largest category in Autonolas/Fetch landscape)
    return 2, 0.5


def project_taxonomy(row: pd.Series) -> dict:
    """Project a single row to a taxonomy category."""
    cat, conf, reason = categorize_by_provenance(
        row.get("source"), row.get("name"),
    )

    # Human short-circuit
    if cat == -2:
        return {
            "taxonomy_category": "HUMAN",
            "taxonomy_index": -1,
            "confidence": conf,
            "rule": reason,
            "source_tier": "provenance_human",
        }

    # Agent path
    if row.get("label", 0) == 0:
        # Was labeled human in parquet — respect it
        return {
            "taxonomy_category": "HUMAN",
            "taxonomy_index": -1,
            "confidence": 0.95,
            "rule": "parquet label=0",
            "source_tier": "parquet_label",
        }

    # Provenance hit: refine with features
    base_cat = cat if cat >= 0 else -1
    final_cat, final_conf = refine_by_features(row, base_cat)

    return {
        "taxonomy_category": TAXONOMY_NAMES[final_cat],
        "taxonomy_index": final_cat,
        "confidence": round(float(final_conf), 3),
        "rule": reason if cat >= 0 else "feature_refinement_fallback",
        "source_tier": "provenance_then_features" if cat >= 0 else "features_only",
    }


def main():
    print("=" * 80)
    print("Paper 0: Taxonomy Projection on Paper 1 Expanded Dataset")
    print("=" * 80)

    print(f"Loading {FEATURES_PARQUET}")
    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"  {len(df)} rows, {(df['label']==1).sum()} agents, "
          f"{(df['label']==0).sum()} humans")

    # Project every row
    projections = []
    for _, row in df.iterrows():
        projections.append(project_taxonomy(row))

    proj_df = pd.DataFrame(projections, index=df.index)
    df_out = pd.concat([df, proj_df], axis=1)
    df_out.to_parquet(OUT_PARQUET)
    print(f"\nSaved {OUT_PARQUET}")

    # Summary statistics
    print("\n" + "=" * 80)
    print("Taxonomy distribution (agents only):")
    print("=" * 80)
    agents_only = df_out[df_out["label"] == 1]
    cat_counts = agents_only["taxonomy_category"].value_counts()
    for cat, count in cat_counts.items():
        pct = count / len(agents_only) * 100
        print(f"  {cat:<30} {count:>5}  ({pct:>5.1f}%)")

    # Confidence distribution
    conf_bins = pd.cut(
        agents_only["confidence"],
        bins=[0, 0.5, 0.7, 0.85, 1.01],
        labels=["<0.5", "0.5-0.7", "0.7-0.85", "0.85+"],
    )
    print("\nConfidence distribution (agents):")
    for bin_label, count in conf_bins.value_counts().sort_index().items():
        print(f"  {bin_label:<12} {count:>5}")

    # Source tier distribution
    print("\nSource tier (agents):")
    for tier, count in agents_only["source_tier"].value_counts().items():
        print(f"  {tier:<30} {count:>5}")

    # Examples per category
    examples = {}
    for cat_idx, cat_name in enumerate(TAXONOMY_NAMES):
        rows = agents_only[agents_only["taxonomy_index"] == cat_idx]
        examples[cat_name] = {
            "count": int(len(rows)),
            "mean_confidence": round(float(rows["confidence"].mean()), 3)
            if len(rows) > 0 else 0.0,
            "examples": rows["name"].head(5).tolist(),
        }

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_total": int(len(df)),
        "n_agents_projected": int(len(agents_only)),
        "n_humans": int((df_out["label"] == 0).sum()),
        "category_counts": {k: int(v) for k, v in cat_counts.items()},
        "confidence_distribution": {
            str(k): int(v) for k, v in conf_bins.value_counts().sort_index().items()
        },
        "source_tier_distribution": {
            k: int(v) for k, v in agents_only["source_tier"].value_counts().items()
        },
        "category_examples": examples,
        "output_parquet": str(OUT_PARQUET),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved summary to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
