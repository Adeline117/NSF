"""
AI Sybil Behavior Generator
============================
Generates realistic AI-agent-driven Sybil transaction sequences that
specifically evade HasciDB's 5 indicators.

Evasion Strategies:

Strategy vs BT (Batch Trading, threshold >=5):
- Vary transaction patterns to avoid fingerprint clustering
- Randomize (destination, value_band) pairs across windows
- Insert non-airdrop transactions as noise

Strategy vs BW (Batch Wallets, threshold >=10):
- Use diverse funding sources: DEX swaps, bridges, tornado-like mixing
- Spread wallet creation across >30-day epochs
- Use different funders per wallet batch

Strategy vs HF (High Frequency, threshold >=0.80):
- Maintain diverse activity: interact with multiple protocols
- Keep window_txs / total_txs < 0.80
- Generate organic-looking non-airdrop activity

Strategy vs RF (Rapid Funds, threshold >=0.50):
- Delay token consolidation beyond 30-day window
- Split outflows across multiple intermediate addresses
- Never consolidate to single receiver from >=3 sources directly

Strategy vs MA (Multi-Address, threshold >=5):
- Avoid 2-hop and 3-hop ETH cycles
- Use bridge/DEX for return flows (breaks cycle detection)
- Keep return values below 80% of preceding leg

Three Sophistication Levels:
- Basic: Simple parameter randomization
- Moderate: Strategy-aware evasion with human-like noise
- Advanced: Full LLM-orchestrated behavior with personality modeling

References:
- Li et al. CHI'26: HasciDB five-indicator framework
- HasciDB API: hascidb.org (3.6M addresses, 1.09M sybils, 16 projects)
- Adeline117/pre-airdrop-detection: LightGBM baseline
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


# ============================================================
# CONSTANTS
# ============================================================

HASCIDB_THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}

HASCIDB_PROJECTS = [
    "uniswap", "ens", "1inch", "blur_s1", "blur_s2", "gitcoin",
    "looksrare", "eigenlayer", "x2y2", "dydx", "apecoin",
    "paraswap", "badger", "ampleforth", "etherfi", "pengu",
]

# HasciDB indicator trigger prevalence among real sybils
# Updated from real HasciDB data (16 projects, 1.09M sybils, 2026-04-07)
INDICATOR_PREVALENCE = {"BT": 0.28, "BW": 0.08, "HF": 0.22, "RF": 0.33, "MA": 0.31}

AI_FEATURES = [
    "gas_price_precision",
    "hour_entropy",
    "behavioral_consistency",
    "action_sequence_perplexity",
    "error_recovery_pattern",
    "response_latency_variance",
    "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]


# ============================================================
# EVASION LEVEL ENUM
# ============================================================

class EvasionLevel(str, Enum):
    BASIC = "basic"
    MODERATE = "moderate"
    ADVANCED = "advanced"


# ============================================================
# SYBIL STRATEGY DATACLASS PER INDICATOR
# ============================================================

@dataclass
class SybilStrategy:
    """Evasion strategy configuration for a single HasciDB indicator.

    Attributes:
        indicator: The HasciDB indicator code (BT, BW, HF, RF, MA).
        threshold: The HasciDB classification threshold.
        target_max: Maximum value the AI sybil will produce (must stay
                    below threshold for evasion).
        noise_method: How noise is injected to appear organic.
            - "uniform": Uniform random below target_max.
            - "exponential": Exponential decay centered near 0.
            - "beta": Beta distribution shaped to mimic human patterns.
        noise_params: Parameters for the noise distribution.
        description: Human-readable description of the evasion approach.
    """
    indicator: str
    threshold: float
    target_max: float
    noise_method: str = "beta"
    noise_params: dict = field(default_factory=dict)
    description: str = ""


# ============================================================
# PER-LEVEL STRATEGY CONFIGURATIONS
# ============================================================

def _build_strategies(level: EvasionLevel) -> dict[str, SybilStrategy]:
    """Build indicator evasion strategies for a given sophistication level.

    Basic:   Simple randomization; may still occasionally brush thresholds.
    Moderate: Strategy-aware; uses human-mimicking distributions.
    Advanced: Full evasion with personality modeling; distributions
              indistinguishable from legitimate users.
    """
    configs = {
        EvasionLevel.BASIC: {
            "BT": SybilStrategy(
                indicator="BT", threshold=5, target_max=8,
                noise_method="uniform", noise_params={"low": 0, "high": 8},
                description="Random BT values; may occasionally exceed threshold",
            ),
            "BW": SybilStrategy(
                indicator="BW", threshold=10, target_max=15,
                noise_method="uniform", noise_params={"low": 0, "high": 15},
                description="Some wallet reuse; not fully diversified funders",
            ),
            "HF": SybilStrategy(
                indicator="HF", threshold=0.80, target_max=0.85,
                noise_method="beta", noise_params={"a": 3, "b": 3},
                description="Some activity diversification; may brush threshold",
            ),
            "RF": SybilStrategy(
                indicator="RF", threshold=0.50, target_max=0.55,
                noise_method="beta", noise_params={"a": 2, "b": 4},
                description="Basic consolidation delay; may still trigger",
            ),
            "MA": SybilStrategy(
                indicator="MA", threshold=5, target_max=6,
                noise_method="uniform", noise_params={"low": 0, "high": 6},
                description="Reduced but not eliminated fund cycles",
            ),
        },
        EvasionLevel.MODERATE: {
            "BT": SybilStrategy(
                indicator="BT", threshold=5, target_max=4,
                noise_method="exponential", noise_params={"scale": 1.2},
                description="Vary tx patterns; randomize (dest, value_band) pairs",
            ),
            "BW": SybilStrategy(
                indicator="BW", threshold=10, target_max=8,
                noise_method="exponential", noise_params={"scale": 2.0},
                description="Use DEX swaps and bridges for diverse funding",
            ),
            "HF": SybilStrategy(
                indicator="HF", threshold=0.80, target_max=0.78,
                noise_method="beta", noise_params={"a": 2, "b": 4},
                description="Multi-protocol activity; keep window_txs/total < 0.80",
            ),
            "RF": SybilStrategy(
                indicator="RF", threshold=0.50, target_max=0.48,
                noise_method="beta", noise_params={"a": 2, "b": 5},
                description="Split outflows across intermediate addresses",
            ),
            "MA": SybilStrategy(
                indicator="MA", threshold=5, target_max=4,
                noise_method="exponential", noise_params={"scale": 0.8},
                description="Use bridge/DEX for return flows; break cycle detection",
            ),
        },
        EvasionLevel.ADVANCED: {
            "BT": SybilStrategy(
                indicator="BT", threshold=5, target_max=3,
                noise_method="beta", noise_params={"a": 1.5, "b": 5},
                description="Per-tx personality variation; unique fingerprints per address",
            ),
            "BW": SybilStrategy(
                indicator="BW", threshold=10, target_max=5,
                noise_method="beta", noise_params={"a": 1.5, "b": 6},
                description="Unique funder per wallet; mixing across 30-day epochs",
            ),
            "HF": SybilStrategy(
                indicator="HF", threshold=0.80, target_max=0.75,
                noise_method="beta", noise_params={"a": 2, "b": 5},
                description="Full organic activity simulation; multi-protocol spread",
            ),
            "RF": SybilStrategy(
                indicator="RF", threshold=0.50, target_max=0.45,
                noise_method="beta", noise_params={"a": 1.5, "b": 6},
                description="Gradual consolidation beyond 30-day window; split receivers",
            ),
            "MA": SybilStrategy(
                indicator="MA", threshold=5, target_max=3,
                noise_method="beta", noise_params={"a": 1.5, "b": 6},
                description="No direct cycles; return value <80% of preceding leg",
            ),
        },
    }
    return configs[level]


# ============================================================
# TRANSACTION SEQUENCE
# ============================================================

@dataclass
class Transaction:
    """A single mock on-chain transaction."""
    tx_hash: str
    from_addr: str
    to_addr: str
    value_eth: float
    gas_price_gwei: float
    gas_used: int
    timestamp: int
    block_number: int
    nonce: int
    tx_type: str  # "airdrop_interact", "defi_swap", "bridge", "transfer", "noise"
    protocol: str  # e.g., "uniswap", "aave", "hop_bridge"


@dataclass
class SybilSequence:
    """A generated sequence of transactions for one sybil address."""
    address: str
    evasion_level: str
    transactions: list[Transaction] = field(default_factory=list)
    indicator_scores: dict = field(default_factory=dict)
    ai_feature_values: dict = field(default_factory=dict)
    evades_hascidb: bool = False


# ============================================================
# INDICATOR VALUE GENERATION
# ============================================================

def _generate_indicator_value(
    strategy: SybilStrategy, rng: np.random.RandomState
) -> float:
    """Generate a single indicator value according to the evasion strategy."""
    method = strategy.noise_method
    params = strategy.noise_params

    if method == "uniform":
        val = rng.uniform(params.get("low", 0), params.get("high", strategy.target_max))
    elif method == "exponential":
        val = rng.exponential(params.get("scale", 1.0))
    elif method == "beta":
        val = rng.beta(params.get("a", 2), params.get("b", 5))
        val *= strategy.target_max
    else:
        val = rng.uniform(0, strategy.target_max)

    return np.clip(val, 0, strategy.target_max)


def _generate_ai_feature_values(
    level: EvasionLevel, rng: np.random.RandomState
) -> dict[str, float]:
    """Generate AI-specific feature values.

    Even advanced AI sybils leak behavioral signatures through these
    features because they reflect inherent LLM execution characteristics
    that are hard to fake:
    - Gas precision: LLMs compute exact gas, humans use round numbers
    - Hour entropy: No circadian rhythm for 24/7 agents
    - Behavioral consistency: Same LLM prompt -> correlated behavior
    - Action perplexity: LLM-generated sequences have characteristic range
    - Error recovery: Systematic retry/fallback patterns
    - Response latency: LLM inference time signature
    - Nonce regularity: Regular nonce increments
    - EIP-1559 tip: Precise priority fee calculation
    """
    # As evasion level increases, AI sybils better approximate human patterns
    # on indicators, but AI-specific features remain discriminative
    level_configs = {
        EvasionLevel.BASIC: {
            "gas_beta": (7, 2),       # High precision (clearly bot)
            "entropy_scale": 0.90,    # High entropy
            "consistency_beta": (6, 2),
            "perplexity_mu": 1.2,
            "perplexity_sigma": 0.4,
            "error_beta": (6, 2),
            "latency_mu": 0.3,
            "latency_sigma": 0.4,
            "nonce_beta": (6, 2),
            "tip_beta": (6, 2),
        },
        EvasionLevel.MODERATE: {
            "gas_beta": (5, 3),       # Moderate precision
            "entropy_scale": 0.80,
            "consistency_beta": (5, 3),
            "perplexity_mu": 1.5,
            "perplexity_sigma": 0.5,
            "error_beta": (5, 3),
            "latency_mu": 0.5,
            "latency_sigma": 0.5,
            "nonce_beta": (5, 3),
            "tip_beta": (5, 3),
        },
        EvasionLevel.ADVANCED: {
            "gas_beta": (4, 4),       # Closer to human but still detectable
            "entropy_scale": 0.70,
            "consistency_beta": (4, 3),
            "perplexity_mu": 1.8,
            "perplexity_sigma": 0.6,
            "error_beta": (4, 3),
            "latency_mu": 0.7,
            "latency_sigma": 0.6,
            "nonce_beta": (4, 3),
            "tip_beta": (4, 3),
        },
    }

    cfg = level_configs[level]
    max_entropy = 3.178  # log2(24) = max hour entropy for uniform distribution

    return {
        "gas_price_precision": float(rng.beta(*cfg["gas_beta"])),
        "hour_entropy": float(rng.beta(5, 3) * max_entropy * cfg["entropy_scale"]),
        "behavioral_consistency": float(rng.beta(*cfg["consistency_beta"])),
        "action_sequence_perplexity": float(rng.lognormal(cfg["perplexity_mu"],
                                                           cfg["perplexity_sigma"])),
        "error_recovery_pattern": float(rng.beta(*cfg["error_beta"])),
        "response_latency_variance": float(rng.lognormal(cfg["latency_mu"],
                                                          cfg["latency_sigma"])),
        "gas_nonce_gap_regularity": float(rng.beta(*cfg["nonce_beta"])),
        "eip1559_tip_precision": float(rng.beta(*cfg["tip_beta"])),
    }


# ============================================================
# MOCK TRANSACTION GENERATION
# ============================================================

PROTOCOLS = [
    "uniswap_v3", "aave_v3", "compound_v3", "lido", "eigenlayer",
    "hop_bridge", "across_bridge", "stargate", "curve", "balancer",
    "opensea", "blur", "sushiswap", "1inch", "paraswap",
]

TX_TYPES = ["airdrop_interact", "defi_swap", "bridge", "transfer", "noise"]


def _generate_mock_transactions(
    address: str,
    level: EvasionLevel,
    rng: np.random.RandomState,
    n_txs: int = 50,
) -> list[Transaction]:
    """Generate a mock transaction sequence for a sybil address.

    Transaction composition varies by evasion level:
    - Basic: Mostly airdrop interactions with little noise
    - Moderate: Mixed airdrop + DeFi activity
    - Advanced: Organic-looking multi-protocol activity with buried
                airdrop interactions
    """
    tx_type_weights = {
        EvasionLevel.BASIC: [0.60, 0.15, 0.05, 0.10, 0.10],
        EvasionLevel.MODERATE: [0.35, 0.25, 0.10, 0.15, 0.15],
        EvasionLevel.ADVANCED: [0.20, 0.30, 0.15, 0.15, 0.20],
    }
    weights = tx_type_weights[level]

    base_timestamp = 1700000000  # ~Nov 2023
    base_block = 18500000
    transactions = []

    for i in range(n_txs):
        tx_type = rng.choice(TX_TYPES, p=weights)
        protocol = rng.choice(PROTOCOLS)

        # Time spacing: advanced sybils mimic human circadian patterns
        if level == EvasionLevel.ADVANCED:
            # Simulate circadian rhythm (more activity during "waking hours")
            hour = int(rng.choice(24, p=_circadian_weights()))
            time_gap = rng.exponential(3600) + hour * 100
        elif level == EvasionLevel.MODERATE:
            time_gap = rng.exponential(1800)
        else:
            time_gap = rng.exponential(600)

        timestamp = base_timestamp + int(i * time_gap)
        block = base_block + int(i * time_gap / 12)

        # Value: human-like amounts for advanced, round numbers for basic
        if level == EvasionLevel.ADVANCED:
            value = float(rng.lognormal(0, 1.5))
        elif level == EvasionLevel.MODERATE:
            value = float(rng.choice([0.1, 0.5, 1.0, 2.0, 5.0]) *
                          rng.uniform(0.8, 1.2))
        else:
            value = float(rng.choice([0.1, 0.5, 1.0, 5.0, 10.0]))

        # Gas price: AI agents compute precise gas, humans round
        if level == EvasionLevel.ADVANCED:
            gas_price = float(rng.normal(25, 8))  # Slightly less precise
        else:
            gas_price = float(rng.normal(25, 3))  # Very precise

        gas_price = max(1.0, gas_price)

        tx = Transaction(
            tx_hash=f"0x{rng.bytes(32).hex()}",
            from_addr=address,
            to_addr=f"0x{rng.bytes(20).hex()}",
            value_eth=round(value, 6),
            gas_price_gwei=round(gas_price, 4),
            gas_used=int(rng.normal(100000, 30000)),
            timestamp=timestamp,
            block_number=block,
            nonce=i,
            tx_type=tx_type,
            protocol=protocol,
        )
        transactions.append(tx)

    return transactions


def _circadian_weights() -> np.ndarray:
    """Generate hour-of-day weights mimicking human circadian rhythm.

    Peak activity 9am-11pm, low activity 2am-6am.
    """
    weights = np.array([
        0.02, 0.01, 0.01, 0.01, 0.01, 0.01,  # 0-5 (sleeping)
        0.02, 0.03, 0.04, 0.06, 0.07, 0.07,   # 6-11 (morning)
        0.06, 0.06, 0.07, 0.07, 0.06, 0.06,   # 12-17 (afternoon)
        0.05, 0.05, 0.05, 0.04, 0.04, 0.03,   # 18-23 (evening)
    ])
    return weights / weights.sum()


# ============================================================
# CORE GENERATION FUNCTIONS
# ============================================================

def generate_evasive_sequence(
    level: EvasionLevel = EvasionLevel.MODERATE,
    seed: int = 42,
    n_txs: int = 50,
) -> SybilSequence:
    """Produce a single sybil address with an evasive transaction sequence.

    Args:
        level: Evasion sophistication level.
        seed: Random seed for reproducibility.
        n_txs: Number of transactions to generate per address.

    Returns:
        SybilSequence with transactions, indicator scores, and AI features.
    """
    rng = np.random.RandomState(seed)
    strategies = _build_strategies(level)

    address = f"0x{rng.bytes(20).hex()}"

    # Generate indicator scores using evasion strategies
    indicator_scores = {}
    for ind_name, strategy in strategies.items():
        val = _generate_indicator_value(strategy, rng)
        # For count-type indicators, round to integer
        if ind_name in ("BT", "BW", "MA"):
            val = int(round(val))
        indicator_scores[ind_name] = val

    # Generate AI-specific feature values
    ai_features = _generate_ai_feature_values(level, rng)

    # Generate mock transaction sequence
    transactions = _generate_mock_transactions(address, level, rng, n_txs)

    # Check if this address would evade HasciDB
    evades = evaluate_evasion(indicator_scores)

    return SybilSequence(
        address=address,
        evasion_level=level.value,
        transactions=transactions,
        indicator_scores=indicator_scores,
        ai_feature_values=ai_features,
        evades_hascidb=evades,
    )


def generate_batch(
    n_addresses: int,
    level: EvasionLevel = EvasionLevel.MODERATE,
    seed: int = 42,
    n_txs_per_address: int = 50,
) -> list[SybilSequence]:
    """Generate a batch of N sybil addresses with evasive sequences.

    Args:
        n_addresses: Number of addresses to generate.
        level: Evasion sophistication level.
        seed: Base random seed.
        n_txs_per_address: Transactions per address.

    Returns:
        List of SybilSequence objects.
    """
    sequences = []
    for i in range(n_addresses):
        seq = generate_evasive_sequence(
            level=level,
            seed=seed + i,
            n_txs=n_txs_per_address,
        )
        sequences.append(seq)
    return sequences


def batch_to_dataframe(sequences: list[SybilSequence]) -> pd.DataFrame:
    """Convert a batch of SybilSequences to a DataFrame for ML experiments.

    Columns: BT, BW, HF, RF, MA, [8 AI features], label, evasion_level,
             evades_hascidb, address.
    """
    rows = []
    for seq in sequences:
        row = {}
        row.update(seq.indicator_scores)
        row.update(seq.ai_feature_values)
        row["label"] = 1  # All generated sequences are sybils
        row["evasion_level"] = seq.evasion_level
        row["evades_hascidb"] = int(seq.evades_hascidb)
        row["address"] = seq.address
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# EVALUATION
# ============================================================

def evaluate_evasion(indicator_scores: dict) -> bool:
    """Check if a set of indicator scores would evade HasciDB detection.

    HasciDB classification logic:
        ops_flag  = (BT >= 5) OR (BW >= 10) OR (HF >= 0.80)
        fund_flag = (RF >= 0.50) OR (MA >= 5)
        is_sybil  = ops_flag OR fund_flag

    Returns True if the address would NOT be flagged (i.e., evades).
    """
    ops_flag = (
        indicator_scores.get("BT", 0) >= HASCIDB_THRESHOLDS["BT"]
        or indicator_scores.get("BW", 0) >= HASCIDB_THRESHOLDS["BW"]
        or indicator_scores.get("HF", 0) >= HASCIDB_THRESHOLDS["HF"]
    )
    fund_flag = (
        indicator_scores.get("RF", 0) >= HASCIDB_THRESHOLDS["RF"]
        or indicator_scores.get("MA", 0) >= HASCIDB_THRESHOLDS["MA"]
    )
    is_sybil = ops_flag or fund_flag
    return not is_sybil  # Evades if NOT flagged


def evaluate_batch_evasion(sequences: list[SybilSequence]) -> dict:
    """Evaluate evasion rates across a batch of generated sequences.

    Returns:
        Dictionary with evasion rate, per-indicator trigger rates,
        and summary statistics.
    """
    n_total = len(sequences)
    n_evades = sum(1 for s in sequences if s.evades_hascidb)

    # Per-indicator trigger rates
    trigger_counts = {ind: 0 for ind in HASCIDB_THRESHOLDS}
    for seq in sequences:
        scores = seq.indicator_scores
        if scores.get("BT", 0) >= HASCIDB_THRESHOLDS["BT"]:
            trigger_counts["BT"] += 1
        if scores.get("BW", 0) >= HASCIDB_THRESHOLDS["BW"]:
            trigger_counts["BW"] += 1
        if scores.get("HF", 0) >= HASCIDB_THRESHOLDS["HF"]:
            trigger_counts["HF"] += 1
        if scores.get("RF", 0) >= HASCIDB_THRESHOLDS["RF"]:
            trigger_counts["RF"] += 1
        if scores.get("MA", 0) >= HASCIDB_THRESHOLDS["MA"]:
            trigger_counts["MA"] += 1

    trigger_rates = {k: v / n_total for k, v in trigger_counts.items()}

    # Indicator score statistics
    indicator_stats = {}
    for ind in HASCIDB_THRESHOLDS:
        values = [s.indicator_scores.get(ind, 0) for s in sequences]
        indicator_stats[ind] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "median": float(np.median(values)),
            "max": float(np.max(values)),
            "pct_below_threshold": float(
                np.mean([v < HASCIDB_THRESHOLDS[ind] for v in values])
            ),
        }

    return {
        "n_total": n_total,
        "n_evades": n_evades,
        "evasion_rate": n_evades / n_total if n_total > 0 else 0.0,
        "trigger_rates": trigger_rates,
        "indicator_stats": indicator_stats,
    }


# ============================================================
# CONVENIENCE: GENERATE DATAFRAME (compatible with pilot v2)
# ============================================================

def generate_ai_sybil_dataframe(
    n: int,
    level: EvasionLevel = EvasionLevel.MODERATE,
    seed: int = 44,
) -> pd.DataFrame:
    """Generate AI sybil data as a DataFrame, compatible with pilot v2 API.

    This is a high-level wrapper that produces the same column schema
    as the pilot's generate_ai_sybils() function:
    BT, BW, HF, RF, MA, [8 AI features], label.

    Args:
        n: Number of sybil addresses.
        level: Evasion sophistication level.
        seed: Random seed.

    Returns:
        DataFrame with HasciDB indicators, AI features, and label=1.
    """
    sequences = generate_batch(n, level=level, seed=seed, n_txs_per_address=50)
    df = batch_to_dataframe(sequences)
    # Drop metadata columns for ML compatibility
    df = df.drop(columns=["evasion_level", "evades_hascidb", "address"], errors="ignore")
    return df


# ============================================================
# MAIN (demo)
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AI Sybil Generator Demo")
    print("=" * 70)

    for level in EvasionLevel:
        print(f"\n--- {level.value.upper()} Level ---")
        sequences = generate_batch(100, level=level, seed=42)
        evaluation = evaluate_batch_evasion(sequences)

        print(f"  Evasion rate: {evaluation['evasion_rate']:.1%}")
        print(f"  Per-indicator trigger rates:")
        for ind, rate in evaluation["trigger_rates"].items():
            threshold = HASCIDB_THRESHOLDS[ind]
            print(f"    {ind} (>={threshold}): {rate:.1%}")

        df = batch_to_dataframe(sequences)
        print(f"  DataFrame shape: {df.shape}")
        print(f"  Mean indicator scores:")
        for ind in HASCIDB_THRESHOLDS:
            print(f"    {ind}: {df[ind].mean():.3f} (threshold={HASCIDB_THRESHOLDS[ind]})")
