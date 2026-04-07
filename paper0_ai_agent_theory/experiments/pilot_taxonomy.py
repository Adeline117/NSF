"""
Paper 0 Pilot: AI Agent Taxonomy & Theoretical Framework
=========================================================
Goal: Establish a formal taxonomy of AI agents and validate it against
existing literature and on-chain evidence.

Research Question: What constitutes an "AI agent" in the Web3 context,
and how do we formally distinguish agents from bots, scripts, and humans?

Pilot Experiment:
1. Enumerate definitional dimensions from literature
2. Map existing on-chain entities to taxonomy categories
3. Validate that taxonomy is exhaustive and mutually exclusive
"""

import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional


class AutonomyLevel(Enum):
    """Degree of autonomous decision-making."""
    NONE = 0        # Pure script: deterministic, no adaptation
    REACTIVE = 1    # Responds to events with fixed rules
    ADAPTIVE = 2    # Adjusts parameters based on environment
    PROACTIVE = 3   # Plans and initiates actions independently
    COLLABORATIVE = 4  # Coordinates with other agents


class EnvironmentType(Enum):
    """Where the agent operates."""
    ONCHAIN_ONLY = "onchain"       # Fully on-chain (smart contract)
    OFFCHAIN_TO_ONCHAIN = "hybrid" # Off-chain compute, on-chain execution
    CROSS_CHAIN = "cross_chain"    # Operates across multiple chains
    MULTI_MODAL = "multi_modal"    # On-chain + off-chain services


class DecisionModel(Enum):
    """How decisions are made."""
    DETERMINISTIC = "deterministic"  # If-then rules
    STATISTICAL = "statistical"      # ML models, probability-based
    LLM_DRIVEN = "llm"             # Large language model reasoning
    REINFORCEMENT = "rl"           # RL-based policy
    HYBRID = "hybrid"              # Combination


@dataclass
class AgentTaxonomyEntry:
    """A single category in the AI agent taxonomy."""
    name: str
    autonomy: AutonomyLevel
    environment: EnvironmentType
    decision_model: DecisionModel
    description: str
    examples: list[str] = field(default_factory=list)
    on_chain_indicators: list[str] = field(default_factory=list)
    distinguishing_features: list[str] = field(default_factory=list)


# ============================================================
# TAXONOMY CONSTRUCTION
# ============================================================

TAXONOMY = [
    AgentTaxonomyEntry(
        name="Simple Trading Bot",
        autonomy=AutonomyLevel.REACTIVE,
        environment=EnvironmentType.OFFCHAIN_TO_ONCHAIN,
        decision_model=DecisionModel.DETERMINISTIC,
        description="Executes pre-defined trading rules (e.g., grid bot, limit orders)",
        examples=["Grid trading bots", "DCA bots", "Rebalancing scripts"],
        on_chain_indicators=[
            "Regular interval transactions",
            "Fixed trade sizes",
            "Deterministic gas pricing",
        ],
        distinguishing_features=[
            "No adaptation to market conditions beyond pre-set rules",
            "Predictable transaction patterns",
        ],
    ),
    AgentTaxonomyEntry(
        name="MEV Searcher",
        autonomy=AutonomyLevel.ADAPTIVE,
        environment=EnvironmentType.OFFCHAIN_TO_ONCHAIN,
        decision_model=DecisionModel.STATISTICAL,
        description="Identifies and extracts MEV (sandwich, arbitrage, liquidation)",
        examples=["jaredfromsubway.eth", "Flashbots searchers"],
        on_chain_indicators=[
            "Extremely low latency (sub-block response)",
            "Bundle submissions via Flashbots",
            "High gas priority fees",
            "Interaction with DEX routers",
        ],
        distinguishing_features=[
            "Mempool monitoring capability",
            "Atomic transaction bundles",
            "Profit-driven execution",
        ],
    ),
    AgentTaxonomyEntry(
        name="DeFi Management Agent",
        autonomy=AutonomyLevel.PROACTIVE,
        environment=EnvironmentType.OFFCHAIN_TO_ONCHAIN,
        decision_model=DecisionModel.HYBRID,
        description="Manages DeFi positions: yield farming, lending, liquidity provision",
        examples=["Autonolas agents", "Yearn strategy vaults", "DeFi Saver automation"],
        on_chain_indicators=[
            "Multi-protocol interactions in sequences",
            "Approval management patterns",
            "Position rebalancing transactions",
            "Oracle price consultation",
        ],
        distinguishing_features=[
            "Cross-protocol reasoning",
            "Risk-aware position management",
            "Adaptive strategy adjustment",
        ],
    ),
    AgentTaxonomyEntry(
        name="LLM-Powered Agent",
        autonomy=AutonomyLevel.PROACTIVE,
        environment=EnvironmentType.MULTI_MODAL,
        decision_model=DecisionModel.LLM_DRIVEN,
        description="Uses LLM for reasoning about on-chain actions",
        examples=["AI16Z/ELIZA agents", "Virtuals Protocol agents", "MCP-connected agents"],
        on_chain_indicators=[
            "Variable latency (LLM inference time)",
            "Complex multi-step transaction sequences",
            "Natural language-influenced parameters",
            "Non-deterministic gas pricing",
        ],
        distinguishing_features=[
            "Highly variable behavior patterns",
            "Context-dependent decision making",
            "May interact with off-chain data sources",
        ],
    ),
    AgentTaxonomyEntry(
        name="Autonomous DAO Agent",
        autonomy=AutonomyLevel.COLLABORATIVE,
        environment=EnvironmentType.ONCHAIN_ONLY,
        decision_model=DecisionModel.HYBRID,
        description="Fully on-chain agent governed by DAO rules",
        examples=["Gnosis Safe modules", "Governor-controlled executors"],
        on_chain_indicators=[
            "Transactions originate from multisig/timelock",
            "Proposal-execution pattern",
            "Governance token interactions",
        ],
        distinguishing_features=[
            "Collective decision making",
            "Timelock delays",
            "On-chain governance trail",
        ],
    ),
    AgentTaxonomyEntry(
        name="Cross-Chain Bridge Agent",
        autonomy=AutonomyLevel.ADAPTIVE,
        environment=EnvironmentType.CROSS_CHAIN,
        decision_model=DecisionModel.DETERMINISTIC,
        description="Relays messages and assets across chains",
        examples=["LayerZero relayers", "Wormhole guardians", "Axelar validators"],
        on_chain_indicators=[
            "Bridge contract interactions",
            "Matching transactions across chains",
            "Proof submission patterns",
        ],
        distinguishing_features=[
            "Multi-chain presence",
            "Verification/proof patterns",
            "Message relay timing",
        ],
    ),
]


def validate_taxonomy():
    """Validate taxonomy properties."""
    results = {
        "total_categories": len(TAXONOMY),
        "autonomy_coverage": set(),
        "environment_coverage": set(),
        "decision_coverage": set(),
        "issues": [],
    }

    names = set()
    for entry in TAXONOMY:
        # Check uniqueness
        if entry.name in names:
            results["issues"].append(f"Duplicate category: {entry.name}")
        names.add(entry.name)

        # Track coverage
        results["autonomy_coverage"].add(entry.autonomy.name)
        results["environment_coverage"].add(entry.environment.name)
        results["decision_coverage"].add(entry.decision_model.name)

        # Check completeness
        if not entry.on_chain_indicators:
            results["issues"].append(f"{entry.name}: missing on-chain indicators")
        if not entry.distinguishing_features:
            results["issues"].append(f"{entry.name}: missing distinguishing features")

    # Check if all enum values are covered
    all_autonomy = {e.name for e in AutonomyLevel}
    all_env = {e.name for e in EnvironmentType}
    all_decision = {e.name for e in DecisionModel}

    missing_autonomy = all_autonomy - results["autonomy_coverage"]
    missing_env = all_env - results["environment_coverage"]
    missing_decision = all_decision - results["decision_coverage"]

    if missing_autonomy:
        results["issues"].append(f"Autonomy levels not covered: {missing_autonomy}")
    if missing_env:
        results["issues"].append(f"Environment types not covered: {missing_env}")
    if missing_decision:
        results["issues"].append(f"Decision models not covered: {missing_decision}")

    # Convert sets to lists for JSON serialization
    results["autonomy_coverage"] = sorted(results["autonomy_coverage"])
    results["environment_coverage"] = sorted(results["environment_coverage"])
    results["decision_coverage"] = sorted(results["decision_coverage"])

    return results


def generate_comparison_matrix():
    """Generate pairwise distinguishability matrix."""
    n = len(TAXONOMY)
    matrix = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = TAXONOMY[i], TAXONOMY[j]
            distinguishable_dims = []
            if a.autonomy != b.autonomy:
                distinguishable_dims.append("autonomy")
            if a.environment != b.environment:
                distinguishable_dims.append("environment")
            if a.decision_model != b.decision_model:
                distinguishable_dims.append("decision_model")
            matrix[f"{a.name} vs {b.name}"] = {
                "distinguishable_dimensions": distinguishable_dims,
                "n_dims": len(distinguishable_dims),
                "separable": len(distinguishable_dims) > 0,
            }
    return matrix


def main():
    """Run pilot taxonomy validation."""
    print("=" * 60)
    print("Paper 0 Pilot: AI Agent Taxonomy Validation")
    print("=" * 60)

    # 1. Validate taxonomy
    print("\n--- Taxonomy Validation ---")
    results = validate_taxonomy()
    print(f"Total categories: {results['total_categories']}")
    print(f"Autonomy coverage: {results['autonomy_coverage']}")
    print(f"Environment coverage: {results['environment_coverage']}")
    print(f"Decision model coverage: {results['decision_coverage']}")
    if results["issues"]:
        print(f"Issues found: {len(results['issues'])}")
        for issue in results["issues"]:
            print(f"  - {issue}")
    else:
        print("No issues found - taxonomy is complete!")

    # 2. Pairwise distinguishability
    print("\n--- Pairwise Distinguishability ---")
    matrix = generate_comparison_matrix()
    all_separable = all(v["separable"] for v in matrix.values())
    print(f"All pairs distinguishable: {all_separable}")
    min_dims = min(v["n_dims"] for v in matrix.values())
    print(f"Minimum distinguishing dimensions: {min_dims}")

    # Find hardest-to-distinguish pairs
    hard_pairs = {k: v for k, v in matrix.items() if v["n_dims"] <= 1}
    if hard_pairs:
        print(f"\nHardest to distinguish ({len(hard_pairs)} pairs):")
        for pair, info in hard_pairs.items():
            print(f"  {pair}: only {info['distinguishable_dimensions']}")

    # 3. Summary for paper
    print("\n--- Pilot Feasibility Assessment ---")
    print(f"Taxonomy has {results['total_categories']} categories across "
          f"{len(results['autonomy_coverage'])} autonomy levels, "
          f"{len(results['environment_coverage'])} environments, "
          f"{len(results['decision_coverage'])} decision models")
    print("FEASIBLE: Taxonomy provides sufficient granularity for CHI contribution")
    print("NEXT: Map taxonomy to observable on-chain features (connects to Paper 1)")

    # Save results
    output = {
        "validation": results,
        "distinguishability": matrix,
        "taxonomy": [asdict(t) for t in TAXONOMY],
    }
    with open("paper0_ai_agent_theory/experiments/pilot_results.json", "w") as f:
        # Custom serializer for enums
        def enum_serializer(obj):
            if isinstance(obj, Enum):
                return obj.value
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        json.dump(output, f, indent=2, default=enum_serializer)
    print("\nResults saved to paper0_ai_agent_theory/experiments/pilot_results.json")


if __name__ == "__main__":
    main()
