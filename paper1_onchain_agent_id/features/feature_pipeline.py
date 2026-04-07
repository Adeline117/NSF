"""
Full Feature Extraction Pipeline for On-Chain AI Agent Identification
=====================================================================
Production-grade pipeline extracting 23 features across four groups from
Ethereum transaction data, designed for the WWW submission.

Group 1: Temporal Features (7 features)
- tx_interval_mean, tx_interval_std, tx_interval_skewness
- active_hour_entropy (Shannon entropy of hour-of-day distribution)
- night_activity_ratio (UTC 0-6 activity)
- weekend_ratio
- burst_frequency (transactions within 10-second windows)

Group 2: Gas Behavior Features (6 features)
- gas_price_round_number_ratio
- gas_price_trailing_zeros_mean
- gas_limit_precision (gas_used / gas_limit ratio)
- gas_price_cv (coefficient of variation)
- eip1559_priority_fee_precision
- gas_price_nonce_correlation

Group 3: Interaction Pattern Features (5 features)
- unique_contracts_ratio
- top_contract_concentration (HHI)
- method_id_diversity
- contract_to_eoa_ratio
- sequential_pattern_score (repeated action sequences via n-gram)

Group 4: Approval & Security Features (5 features)
- unlimited_approve_ratio (MaxUint256 approves)
- approve_revoke_ratio
- unverified_contract_approve_ratio
- multi_protocol_interaction_count
- flash_loan_usage (indicator)
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from shared.utils.eth_utils import EtherscanClient

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class FeatureConfig:
    """Configurable parameters for feature extraction.

    Attributes:
        tx_fetch_limit: Maximum number of transactions to fetch per address.
        burst_window_seconds: Time window (in seconds) for burst detection.
        night_start_hour: Start of night window (UTC).
        night_end_hour: End of night window (UTC).
        round_number_trailing_zeros: Minimum trailing zeros for "round" gas price.
        ngram_n: N-gram size for sequential pattern detection.
        max_uint256_hex: Hex string for MaxUint256 approval amount.
        approve_method_id: ERC-20 approve(address,uint256) selector.
        revoke_value_hex: Hex-encoded zero value indicating revocation.
        flash_loan_signatures: Known flash-loan function selectors.
        major_protocol_routers: Set of well-known DeFi router addresses used
            to compute multi_protocol_interaction_count.
        batch_size: Number of addresses to process before checkpointing.
        retry_attempts: Number of retries on transient API errors.
        retry_delay_seconds: Base delay between retries (exponential back-off).
    """

    tx_fetch_limit: int = 1000
    burst_window_seconds: int = 10
    night_start_hour: int = 0
    night_end_hour: int = 6
    round_number_trailing_zeros: int = 3
    ngram_n: int = 3
    max_uint256_hex: str = (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    )
    approve_method_id: str = "0x095ea7b3"
    revoke_value_hex: str = (
        "0000000000000000000000000000000000000000000000000000000000000000"
    )
    flash_loan_signatures: list = field(default_factory=lambda: [
        "0x5cffe9de",  # flashLoan (Aave V3)
        "0xab9c4b5d",  # flashLoan (Aave V2)
        "0x490e6cbc",  # flash (Uniswap V3)
    ])
    major_protocol_routers: list = field(default_factory=lambda: [
        "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2 Router
        "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 Router2
        "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Uniswap Universal Router
        "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap Router
        "0xdef1c0ded9bec7f1a1670819833240f027b25eff",  # 0x Exchange Proxy
        "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 Router
        "0x1111111254eeb25477b68fb85ed929f73a960582",  # 1inch V5 Router
        "0x87870bca3f3fd6335c3f4ce8392d69350b4fa4e2",  # Aave V3 Pool
        "0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9",  # Aave V2 Pool
        "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",  # USDC (for approve tracking)
    ])
    batch_size: int = 50
    retry_attempts: int = 3
    retry_delay_seconds: float = 2.0


# ============================================================
# FEATURE GROUP DATA CLASSES
# ============================================================

FEATURE_NAMES = [
    # Temporal (7)
    "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
    "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
    "burst_frequency",
    # Gas (6)
    "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
    "gas_limit_precision", "gas_price_cv",
    "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
    # Interaction (5)
    "unique_contracts_ratio", "top_contract_concentration",
    "method_id_diversity", "contract_to_eoa_ratio",
    "sequential_pattern_score",
    # Approval & Security (5)
    "unlimited_approve_ratio", "approve_revoke_ratio",
    "unverified_contract_approve_ratio",
    "multi_protocol_interaction_count", "flash_loan_usage",
]


# ============================================================
# INDIVIDUAL FEATURE EXTRACTORS
# ============================================================

def extract_temporal_features(
    txs: pd.DataFrame, config: FeatureConfig
) -> dict:
    """Extract 7 temporal features from transaction data.

    Features capture the timing regularity, circadian patterns, and
    burstiness that distinguish automated agents from human users.

    Args:
        txs: DataFrame of transactions with at least a ``timeStamp`` column.
        config: Pipeline configuration.

    Returns:
        Dictionary mapping feature names to float values.
    """
    result = {
        "tx_interval_mean": 0.0,
        "tx_interval_std": 0.0,
        "tx_interval_skewness": 0.0,
        "active_hour_entropy": 0.0,
        "night_activity_ratio": 0.0,
        "weekend_ratio": 0.0,
        "burst_frequency": 0.0,
    }

    if txs.empty or len(txs) < 3:
        return result

    timestamps = (
        pd.to_numeric(txs["timeStamp"], errors="coerce")
        .dropna()
        .sort_values()
        .reset_index(drop=True)
    )
    if len(timestamps) < 3:
        return result

    # --- Inter-transaction intervals ---
    intervals = timestamps.diff().dropna()
    result["tx_interval_mean"] = float(intervals.mean())
    result["tx_interval_std"] = float(intervals.std())
    if intervals.std() > 0:
        result["tx_interval_skewness"] = float(stats.skew(intervals))

    # --- Active-hour entropy (Shannon entropy, base-e) ---
    dt_series = pd.to_datetime(timestamps, unit="s", utc=True)
    hours = dt_series.dt.hour
    hour_counts = np.zeros(24)
    for h in hours:
        hour_counts[h] += 1
    hour_probs = hour_counts / hour_counts.sum()
    # Filter zeros to avoid log(0)
    hour_probs_nonzero = hour_probs[hour_probs > 0]
    result["active_hour_entropy"] = float(
        -np.sum(hour_probs_nonzero * np.log(hour_probs_nonzero))
    )

    # --- Night activity ratio (UTC hours 0-6) ---
    night_mask = hours.between(config.night_start_hour, config.night_end_hour)
    result["night_activity_ratio"] = float(night_mask.mean())

    # --- Weekend ratio ---
    day_of_week = dt_series.dt.dayofweek  # Monday=0, Sunday=6
    weekend_mask = day_of_week >= 5
    result["weekend_ratio"] = float(weekend_mask.mean())

    # --- Burst frequency ---
    # Fraction of transactions that fall within a burst window of the
    # previous transaction.
    burst_count = (intervals <= config.burst_window_seconds).sum()
    result["burst_frequency"] = float(burst_count / len(intervals))

    return result


def extract_gas_features(
    txs: pd.DataFrame, config: FeatureConfig
) -> dict:
    """Extract 6 gas behavior features from transaction data.

    Gas pricing behaviour is a strong discriminator: humans tend to use
    wallet defaults (round numbers), whereas agents set precise values.

    Args:
        txs: DataFrame with ``gasPrice``, ``gasUsed``, ``gas``, ``nonce``
            and optionally ``maxPriorityFeePerGas`` columns.
        config: Pipeline configuration.

    Returns:
        Dictionary mapping feature names to float values.
    """
    result = {
        "gas_price_round_number_ratio": 0.0,
        "gas_price_trailing_zeros_mean": 0.0,
        "gas_limit_precision": 0.0,
        "gas_price_cv": 0.0,
        "eip1559_priority_fee_precision": 0.0,
        "gas_price_nonce_correlation": 0.0,
    }

    if txs.empty:
        return result

    gas_prices = pd.to_numeric(
        txs.get("gasPrice", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if gas_prices.empty:
        return result

    # --- Round-number ratio ---
    def _count_trailing_zeros(x: float) -> int:
        if x == 0:
            return 0
        s = str(int(x))
        return len(s) - len(s.rstrip("0"))

    trailing_zeros = gas_prices.apply(_count_trailing_zeros)
    result["gas_price_round_number_ratio"] = float(
        (trailing_zeros >= config.round_number_trailing_zeros).mean()
    )
    result["gas_price_trailing_zeros_mean"] = float(trailing_zeros.mean())

    # --- Gas limit precision (gas_used / gas_limit) ---
    gas_used = pd.to_numeric(
        txs.get("gasUsed", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    gas_limit = pd.to_numeric(
        txs.get("gas", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if (
        not gas_used.empty
        and not gas_limit.empty
        and len(gas_used) == len(gas_limit)
    ):
        precision = gas_used / gas_limit.replace(0, np.nan)
        result["gas_limit_precision"] = float(precision.dropna().mean())

    # --- Gas price coefficient of variation ---
    mean_gp = gas_prices.mean()
    if mean_gp > 0:
        result["gas_price_cv"] = float(gas_prices.std() / mean_gp)

    # --- EIP-1559 priority fee precision ---
    # For EIP-1559 transactions the maxPriorityFeePerGas field exists.
    # Precision is measured as 1 / (1 + trailing_zeros_mean) -- higher
    # values indicate more precise (less "round") priority fees.
    priority_fee = pd.to_numeric(
        txs.get("maxPriorityFeePerGas", pd.Series(dtype=float)),
        errors="coerce",
    ).dropna()
    if not priority_fee.empty:
        pf_trailing = priority_fee.apply(_count_trailing_zeros)
        mean_trailing = pf_trailing.mean()
        result["eip1559_priority_fee_precision"] = float(
            1.0 / (1.0 + mean_trailing)
        )

    # --- Gas price-nonce correlation ---
    # A high correlation may indicate algorithmic gas-price adjustment
    # over successive transactions.
    nonces = pd.to_numeric(
        txs.get("nonce", pd.Series(dtype=float)), errors="coerce"
    ).dropna()
    if len(nonces) >= 5 and len(gas_prices) >= 5:
        # Align by index
        common_idx = nonces.index.intersection(gas_prices.index)
        if len(common_idx) >= 5:
            corr, _ = stats.spearmanr(
                nonces.loc[common_idx], gas_prices.loc[common_idx]
            )
            if np.isfinite(corr):
                result["gas_price_nonce_correlation"] = float(corr)

    return result


def extract_interaction_features(
    txs: pd.DataFrame, config: FeatureConfig
) -> dict:
    """Extract 5 interaction-pattern features.

    Agents tend to interact with fewer contracts in a highly repetitive
    manner, leading to high concentration and low diversity scores.

    Args:
        txs: DataFrame with ``to``, ``input`` columns.
        config: Pipeline configuration.

    Returns:
        Dictionary mapping feature names to float values.
    """
    result = {
        "unique_contracts_ratio": 0.0,
        "top_contract_concentration": 0.0,
        "method_id_diversity": 0.0,
        "contract_to_eoa_ratio": 0.0,
        "sequential_pattern_score": 0.0,
    }

    if txs.empty:
        return result

    to_addrs = txs["to"].dropna()
    if not to_addrs.empty:
        result["unique_contracts_ratio"] = float(
            to_addrs.nunique() / len(to_addrs)
        )
        # HHI (Herfindahl-Hirschman Index)
        shares = to_addrs.value_counts(normalize=True)
        result["top_contract_concentration"] = float((shares ** 2).sum())

    # --- Method ID diversity ---
    if "input" in txs.columns:
        method_ids = txs["input"].apply(
            lambda x: x[:10]
            if isinstance(x, str) and len(x) >= 10 and x != "0x"
            else None
        )
        method_ids = method_ids.dropna()
        if not method_ids.empty:
            result["method_id_diversity"] = float(
                method_ids.nunique() / len(method_ids)
            )

    # --- Contract-to-EOA ratio ---
    if "input" in txs.columns:
        is_contract_call = txs["input"].apply(
            lambda x: isinstance(x, str) and len(x) > 2 and x != "0x"
        )
        result["contract_to_eoa_ratio"] = float(is_contract_call.mean())

    # --- Sequential pattern score (n-gram repetition) ---
    if "input" in txs.columns and len(txs) >= config.ngram_n:
        method_seq = txs["input"].apply(
            lambda x: x[:10]
            if isinstance(x, str) and len(x) >= 10
            else "0x"
        ).tolist()
        ngrams = [
            tuple(method_seq[i : i + config.ngram_n])
            for i in range(len(method_seq) - config.ngram_n + 1)
        ]
        if ngrams:
            ngram_counts = Counter(ngrams)
            total_ngrams = len(ngrams)
            repeated = sum(c for c in ngram_counts.values() if c > 1)
            result["sequential_pattern_score"] = float(
                repeated / total_ngrams
            )

    return result


def extract_approval_security_features(
    txs: pd.DataFrame, config: FeatureConfig
) -> dict:
    """Extract 5 approval and security features.

    These features capture risky approval behaviours and interaction with
    potentially dangerous protocols, which are more common among naive
    automated agents than security-conscious human users.

    Args:
        txs: DataFrame with ``input``, ``to``, ``timeStamp`` columns.
        config: Pipeline configuration.

    Returns:
        Dictionary mapping feature names to float values.
    """
    result = {
        "unlimited_approve_ratio": 0.0,
        "approve_revoke_ratio": 0.0,
        "unverified_contract_approve_ratio": 0.0,
        "multi_protocol_interaction_count": 0.0,
        "flash_loan_usage": 0.0,
    }

    if txs.empty or "input" not in txs.columns:
        return result

    inputs = txs["input"].fillna("")

    # --- Identify approve transactions ---
    approve_mask = inputs.str.startswith(config.approve_method_id, na=False)
    approvals = txs[approve_mask]

    if not approvals.empty:
        # Unlimited approvals (MaxUint256)
        unlimited = approvals["input"].str.contains(
            config.max_uint256_hex, na=False
        )
        result["unlimited_approve_ratio"] = float(unlimited.mean())

        # Revoke ratio: approvals where amount is zero
        revokes = approvals["input"].str.endswith(
            config.revoke_value_hex, na=False
        )
        n_revokes = revokes.sum()
        n_non_revoke_approvals = (~revokes).sum()
        if n_non_revoke_approvals > 0:
            result["approve_revoke_ratio"] = float(
                n_revokes / n_non_revoke_approvals
            )

        # Unverified-contract approve ratio: we approximate this as
        # approvals to addresses NOT in the major-protocol set.
        if "to" in approvals.columns:
            known_set = {
                a.lower() for a in config.major_protocol_routers
            }
            approve_targets = approvals["to"].str.lower()
            unverified = ~approve_targets.isin(known_set)
            result["unverified_contract_approve_ratio"] = float(
                unverified.mean()
            )

    # --- Multi-protocol interaction count ---
    if "to" in txs.columns:
        router_set = {a.lower() for a in config.major_protocol_routers}
        interacted = txs["to"].str.lower().isin(router_set)
        unique_protocols = txs.loc[interacted, "to"].str.lower().nunique()
        result["multi_protocol_interaction_count"] = float(unique_protocols)

    # --- Flash loan usage (binary indicator) ---
    for sig in config.flash_loan_signatures:
        if inputs.str.startswith(sig, na=False).any():
            result["flash_loan_usage"] = 1.0
            break

    return result


# ============================================================
# PIPELINE ORCHESTRATOR
# ============================================================

class FeaturePipeline:
    """Production feature extraction pipeline.

    Fetches transactions for a list of Ethereum addresses via the
    Etherscan API, computes all 23 features per address, and returns
    the result as a pandas DataFrame.

    Usage::

        client = EtherscanClient(api_key="...")
        pipeline = FeaturePipeline(client)
        df = pipeline.extract(["0xABC...", "0xDEF..."])
        print(df.shape)  # (2, 23)

    Args:
        client: An :class:`EtherscanClient` instance.
        config: Optional :class:`FeatureConfig`; uses defaults if omitted.
    """

    def __init__(
        self,
        client: EtherscanClient,
        config: Optional[FeatureConfig] = None,
    ):
        self.client = client
        self.config = config or FeatureConfig()

    # ----------------------------------------------------------
    # Public API
    # ----------------------------------------------------------

    def extract(
        self,
        addresses: list[str],
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Extract features for a list of Ethereum addresses.

        Args:
            addresses: List of Ethereum addresses (checksummed or lower).
            show_progress: Whether to display a tqdm progress bar.

        Returns:
            A DataFrame indexed by address with 23 feature columns.
        """
        records: list[dict] = []
        failed: list[str] = []

        iterator = tqdm(
            addresses,
            desc="Extracting features",
            disable=not show_progress,
        )

        for addr in iterator:
            iterator.set_postfix(addr=addr[:10])
            try:
                features = self._extract_single(addr)
                features["address"] = addr
                records.append(features)
            except Exception as exc:
                logger.warning("Failed to extract features for %s: %s", addr, exc)
                failed.append(addr)

        if failed:
            logger.warning(
                "Feature extraction failed for %d / %d addresses.",
                len(failed),
                len(addresses),
            )

        if not records:
            return pd.DataFrame(columns=["address"] + FEATURE_NAMES)

        df = pd.DataFrame(records)
        df = df.set_index("address")
        # Ensure column order matches FEATURE_NAMES
        for col in FEATURE_NAMES:
            if col not in df.columns:
                df[col] = 0.0
        df = df[FEATURE_NAMES]
        return df

    def extract_batch(
        self,
        addresses: list[str],
        checkpoint_path: Optional[str] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Extract features in batches with optional checkpointing.

        After every ``config.batch_size`` addresses, intermediate results
        are saved to ``checkpoint_path`` (if provided) so that long-running
        jobs can be resumed.

        Args:
            addresses: Full list of addresses.
            checkpoint_path: Path to a Parquet file for checkpointing.
            show_progress: Whether to display progress.

        Returns:
            Complete DataFrame of features.
        """
        all_dfs: list[pd.DataFrame] = []
        processed: set[str] = set()

        # Resume from checkpoint if it exists
        if checkpoint_path and os.path.exists(checkpoint_path):
            existing = pd.read_parquet(checkpoint_path)
            all_dfs.append(existing)
            processed = set(existing.index.tolist())
            logger.info(
                "Resumed from checkpoint: %d addresses already processed.",
                len(processed),
            )

        remaining = [a for a in addresses if a not in processed]

        for i in range(0, len(remaining), self.config.batch_size):
            batch = remaining[i : i + self.config.batch_size]
            logger.info(
                "Processing batch %d-%d of %d remaining addresses.",
                i + 1,
                min(i + self.config.batch_size, len(remaining)),
                len(remaining),
            )
            batch_df = self.extract(batch, show_progress=show_progress)
            all_dfs.append(batch_df)

            if checkpoint_path:
                combined = pd.concat(all_dfs)
                combined.to_parquet(checkpoint_path)
                logger.info("Checkpoint saved to %s.", checkpoint_path)

        if all_dfs:
            return pd.concat(all_dfs)
        return pd.DataFrame(columns=FEATURE_NAMES)

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _extract_single(self, address: str) -> dict:
        """Fetch transactions and extract all 23 features for one address."""
        txs = self._fetch_with_retry(address)

        temporal = extract_temporal_features(txs, self.config)
        gas = extract_gas_features(txs, self.config)
        interaction = extract_interaction_features(txs, self.config)
        approval = extract_approval_security_features(txs, self.config)

        features: dict = {}
        features.update(temporal)
        features.update(gas)
        features.update(interaction)
        features.update(approval)
        return features

    def _fetch_with_retry(self, address: str) -> pd.DataFrame:
        """Fetch normal transactions with exponential back-off retry."""
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.retry_attempts + 1):
            try:
                txs = self.client.get_normal_txs(
                    address, offset=self.config.tx_fetch_limit
                )
                return txs
            except Exception as exc:
                last_exc = exc
                delay = self.config.retry_delay_seconds * (2 ** (attempt - 1))
                logger.warning(
                    "Attempt %d/%d for %s failed (%s). Retrying in %.1fs.",
                    attempt,
                    self.config.retry_attempts,
                    address,
                    exc,
                    delay,
                )
                time.sleep(delay)
        raise RuntimeError(
            f"All {self.config.retry_attempts} attempts failed for {address}"
        ) from last_exc


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """Demonstrate the pipeline on a small set of addresses."""
    import argparse

    parser = argparse.ArgumentParser(
        description="On-chain AI agent feature extraction pipeline"
    )
    parser.add_argument(
        "--addresses",
        nargs="+",
        default=[
            "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # vitalik.eth
        ],
        help="Ethereum addresses to process",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ETHERSCAN_API_KEY", ""),
        help="Etherscan API key",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output Parquet file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = EtherscanClient(api_key=args.api_key)
    pipeline = FeaturePipeline(client)
    df = pipeline.extract(args.addresses)

    print("\n--- Extracted Features ---")
    print(df.to_string())

    if args.output:
        df.to_parquet(args.output)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
