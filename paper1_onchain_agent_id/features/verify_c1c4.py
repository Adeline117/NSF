"""
Automated C1-C4 Verification
==============================
For any Ethereum address, check each of the 4 conditions from Definition 1
(shared/definition.md).

C1: On-chain Actuation
    is_eoa() -- eth_getCode == "0x" AND has outgoing txs

C2: Environmental Perception
    has_environmental_perception() -- calldata entropy > 0,
    tx timing correlates with environment changes

C3: Autonomous Decision-Making
    is_autonomous_nondeterministic() -- tx_interval_cv > threshold,
    hour_entropy > threshold, burst_ratio > threshold

C4: Adaptiveness
    is_adaptive() -- behavior parameters change over time
    (gas strategy drift, target contract set evolution)

Returns:
    {
        "c1": bool,
        "c2": bool,
        "c3": bool,
        "c4": bool,
        "is_agent": bool,   # True iff c1 AND c2 AND c3 AND c4
        "confidence": float, # 0.0-1.0
        "details": {
            "c1_is_eoa": bool,
            "c1_has_outgoing_txs": bool,
            "c2_calldata_entropy": float,
            "c2_tx_timing_varies": bool,
            "c3_tx_interval_cv": float,
            "c3_hour_entropy": float,
            "c3_burst_ratio": float,
            "c4_gas_strategy_drift": float,
            "c4_target_set_evolution": float,
        }
    }

Usage::

    from paper1_onchain_agent_id.features.verify_c1c4 import C1C4Verifier

    verifier = C1C4Verifier(client)
    result = verifier.verify("0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9")
    print(result["is_agent"])  # True
    print(result["details"])
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from collections import Counter
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from shared.utils.eth_utils import EtherscanClient

logger = logging.getLogger(__name__)


# ============================================================
# THRESHOLDS (from Definition 1 operationalization)
# POST-DELPHI: Calibrate thresholds based on expert consensus
# ============================================================

@dataclass
class C1C4Thresholds:
    """Configurable thresholds for C1-C4 condition checks.

    These thresholds determine the boundary between agent and non-agent
    behavior. They should be calibrated using the labeled dataset.

    Attributes:
        min_outgoing_txs: Minimum number of outgoing txs for C1.
        calldata_entropy_min: Minimum calldata entropy for C2.
        tx_interval_cv_min: Minimum tx interval CV for C3 (non-determinism).
        hour_entropy_min: Minimum active-hour entropy for C3 (no circadian rhythm).
        burst_ratio_min: Minimum burst ratio for C3 (automated speed).
        gas_drift_threshold: Minimum gas strategy drift for C4.
        target_evolution_threshold: Minimum target set Jaccard distance for C4.
        time_window_blocks: Number of time windows for C4 comparison.
        min_txs_for_analysis: Minimum transactions needed for meaningful analysis.
    """
    min_outgoing_txs: int = 10
    calldata_entropy_min: float = 0.5
    tx_interval_cv_min: float = 0.5
    hour_entropy_min: float = 2.5       # max is ln(24) ~ 3.178
    burst_ratio_min: float = 0.05       # 5% of txs within 10s of prior tx
    gas_drift_threshold: float = 0.15
    target_evolution_threshold: float = 0.2
    time_window_blocks: int = 4         # Split tx history into N windows
    min_txs_for_analysis: int = 20


# ============================================================
# VERIFIER
# ============================================================

class C1C4Verifier:
    """Automated C1-C4 condition verifier for Ethereum addresses.

    Uses the Etherscan API to fetch transaction data and evaluate each
    of the four conditions from Definition 1.

    Args:
        client: EtherscanClient instance.
        thresholds: Configurable thresholds (default values used if omitted).
        cache_txs: If True, cache fetched transactions for repeated queries.
    """

    def __init__(
        self,
        client: EtherscanClient,
        thresholds: Optional[C1C4Thresholds] = None,
        cache_txs: bool = True,
    ):
        self.client = client
        self.thresholds = thresholds or C1C4Thresholds()
        self._cache: dict[str, pd.DataFrame] = {}
        self._cache_enabled = cache_txs

    def verify(
        self,
        address: str,
        txs: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Verify all C1-C4 conditions for a given address.

        Args:
            address: Ethereum address to verify.
            txs: Pre-fetched transactions (optional). If None, fetches via API.

        Returns:
            Dictionary with c1, c2, c3, c4, is_agent, confidence, details.
        """
        logger.info("Verifying C1-C4 for %s ...", address[:12])

        if txs is None:
            txs = self._get_transactions(address)

        # Filter to outgoing transactions only
        if not txs.empty and "from" in txs.columns:
            outgoing = txs[txs["from"].str.lower() == address.lower()]
        else:
            outgoing = txs

        # C1: On-chain Actuation
        c1_result = self._check_c1(address, outgoing)

        # C2: Environmental Perception
        c2_result = self._check_c2(outgoing)

        # C3: Autonomous Decision-Making
        c3_result = self._check_c3(outgoing)

        # C4: Adaptiveness
        c4_result = self._check_c4(outgoing)

        # Aggregate
        is_agent = (c1_result["passed"] and c2_result["passed"]
                    and c3_result["passed"] and c4_result["passed"])

        # Confidence: average of individual condition confidences
        confidences = [
            c1_result["confidence"],
            c2_result["confidence"],
            c3_result["confidence"],
            c4_result["confidence"],
        ]
        overall_confidence = float(np.mean(confidences))

        result = {
            "address": address,
            "c1": c1_result["passed"],
            "c2": c2_result["passed"],
            "c3": c3_result["passed"],
            "c4": c4_result["passed"],
            "is_agent": is_agent,
            "confidence": round(overall_confidence, 3),
            "n_transactions": len(outgoing),
            "details": {
                "c1_is_eoa": c1_result.get("is_eoa"),
                "c1_has_outgoing_txs": c1_result.get("has_outgoing_txs"),
                "c1_outgoing_tx_count": c1_result.get("outgoing_tx_count"),
                "c2_calldata_entropy": round(c2_result.get("calldata_entropy", 0), 4),
                "c2_tx_timing_varies": c2_result.get("tx_timing_varies"),
                "c3_tx_interval_cv": round(c3_result.get("tx_interval_cv", 0), 4),
                "c3_hour_entropy": round(c3_result.get("hour_entropy", 0), 4),
                "c3_burst_ratio": round(c3_result.get("burst_ratio", 0), 4),
                "c4_gas_strategy_drift": round(c4_result.get("gas_drift", 0), 4),
                "c4_target_set_evolution": round(c4_result.get("target_evolution", 0), 4),
            },
        }

        logger.info(
            "  C1=%s C2=%s C3=%s C4=%s -> is_agent=%s (conf=%.3f)",
            c1_result["passed"], c2_result["passed"],
            c3_result["passed"], c4_result["passed"],
            is_agent, overall_confidence,
        )

        return result

    def verify_batch(
        self,
        addresses: list[str],
    ) -> list[dict]:
        """Verify C1-C4 for multiple addresses.

        Args:
            addresses: List of Ethereum addresses.

        Returns:
            List of verification result dictionaries.
        """
        results = []
        for addr in addresses:
            try:
                result = self.verify(addr)
                results.append(result)
            except Exception as exc:
                logger.warning("Failed to verify %s: %s", addr[:12], exc)
                results.append({
                    "address": addr,
                    "c1": None, "c2": None, "c3": None, "c4": None,
                    "is_agent": None,
                    "confidence": 0.0,
                    "error": str(exc),
                })
        return results

    # ----------------------------------------------------------
    # C1: On-chain Actuation
    # ----------------------------------------------------------

    def _check_c1(
        self,
        address: str,
        outgoing_txs: pd.DataFrame,
    ) -> dict:
        """Check C1: Is the address an EOA with outgoing transactions?

        C1 requires:
          - eth_getCode(address) == "0x" (not a smart contract)
          - Address has initiated transactions (from == address)
        """
        is_eoa = not self.client.is_contract(address)
        has_outgoing = len(outgoing_txs) >= self.thresholds.min_outgoing_txs
        outgoing_count = len(outgoing_txs)

        passed = is_eoa and has_outgoing
        confidence = 1.0 if passed else (0.3 if is_eoa else 0.0)

        return {
            "passed": passed,
            "confidence": confidence,
            "is_eoa": is_eoa,
            "has_outgoing_txs": has_outgoing,
            "outgoing_tx_count": outgoing_count,
        }

    # ----------------------------------------------------------
    # C2: Environmental Perception
    # ----------------------------------------------------------

    def _check_c2(
        self,
        txs: pd.DataFrame,
    ) -> dict:
        """Check C2: Does the entity perceive and respond to environment?

        Indicators:
          - Calldata entropy > threshold (varied function calls)
          - Transaction timing varies (not fixed cron schedule)
        """
        if txs.empty or len(txs) < self.thresholds.min_txs_for_analysis:
            return {
                "passed": False,
                "confidence": 0.0,
                "calldata_entropy": 0.0,
                "tx_timing_varies": False,
            }

        # Calldata entropy: Shannon entropy of method IDs
        calldata_entropy = 0.0
        if "input" in txs.columns:
            method_ids = txs["input"].apply(
                lambda x: x[:10]
                if isinstance(x, str) and len(x) >= 10 and x != "0x"
                else "0x_plain"
            )
            counts = method_ids.value_counts(normalize=True)
            probs = counts.values
            probs_nonzero = probs[probs > 0]
            calldata_entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)))

        # Transaction timing: check if intervals are non-uniform
        tx_timing_varies = False
        if "timeStamp" in txs.columns:
            timestamps = pd.to_numeric(
                txs["timeStamp"], errors="coerce"
            ).dropna().sort_values()
            if len(timestamps) >= 3:
                intervals = timestamps.diff().dropna()
                if intervals.std() > 0:
                    cv = intervals.std() / intervals.mean()
                    # If CV > 0.1, timing varies (not a fixed cron)
                    tx_timing_varies = cv > 0.1

        calldata_ok = calldata_entropy > self.thresholds.calldata_entropy_min
        passed = calldata_ok or tx_timing_varies

        conf = 0.0
        if calldata_ok and tx_timing_varies:
            conf = 0.9
        elif calldata_ok:
            conf = 0.7
        elif tx_timing_varies:
            conf = 0.5

        return {
            "passed": passed,
            "confidence": conf,
            "calldata_entropy": calldata_entropy,
            "tx_timing_varies": tx_timing_varies,
        }

    # ----------------------------------------------------------
    # C3: Autonomous Decision-Making
    # ----------------------------------------------------------

    def _check_c3(
        self,
        txs: pd.DataFrame,
    ) -> dict:
        """Check C3: Is decision-making autonomous and non-deterministic?

        Indicators:
          - tx_interval_cv > threshold: interval variability (non-deterministic)
          - active_hour_entropy > threshold: no circadian rhythm (not human)
          - burst_ratio > threshold: rapid-fire transactions (automated)
        """
        if txs.empty or len(txs) < self.thresholds.min_txs_for_analysis:
            return {
                "passed": False,
                "confidence": 0.0,
                "tx_interval_cv": 0.0,
                "hour_entropy": 0.0,
                "burst_ratio": 0.0,
            }

        timestamps = pd.to_numeric(
            txs["timeStamp"], errors="coerce"
        ).dropna().sort_values()

        # Tx interval coefficient of variation
        tx_interval_cv = 0.0
        if len(timestamps) >= 3:
            intervals = timestamps.diff().dropna()
            mean_interval = intervals.mean()
            if mean_interval > 0:
                tx_interval_cv = float(intervals.std() / mean_interval)

        # Active hour entropy
        hour_entropy = 0.0
        dt_series = pd.to_datetime(timestamps, unit="s", utc=True)
        hours = dt_series.dt.hour
        hour_counts = np.zeros(24)
        for h in hours:
            hour_counts[h] += 1
        total = hour_counts.sum()
        if total > 0:
            probs = hour_counts / total
            probs_nonzero = probs[probs > 0]
            hour_entropy = float(-np.sum(probs_nonzero * np.log(probs_nonzero)))

        # Burst ratio: fraction of txs within 10s of prior tx
        burst_ratio = 0.0
        if len(timestamps) >= 3:
            intervals = timestamps.diff().dropna()
            burst_count = (intervals <= 10).sum()
            burst_ratio = float(burst_count / len(intervals))

        # Evaluate conditions
        cv_ok = tx_interval_cv > self.thresholds.tx_interval_cv_min
        entropy_ok = hour_entropy > self.thresholds.hour_entropy_min
        burst_ok = burst_ratio > self.thresholds.burst_ratio_min

        # C3 passes if non-deterministic (cv_ok) AND not human-operated
        # (entropy_ok OR burst_ok)
        passed = cv_ok and (entropy_ok or burst_ok)

        # Confidence based on how many sub-conditions are met
        conditions_met = sum([cv_ok, entropy_ok, burst_ok])
        conf = conditions_met / 3.0

        return {
            "passed": passed,
            "confidence": conf,
            "tx_interval_cv": tx_interval_cv,
            "hour_entropy": hour_entropy,
            "burst_ratio": burst_ratio,
        }

    # ----------------------------------------------------------
    # C4: Adaptiveness
    # ----------------------------------------------------------

    def _check_c4(
        self,
        txs: pd.DataFrame,
    ) -> dict:
        """Check C4: Does behavior change systematically over time?

        Indicators:
          - Gas strategy drift: gas price distribution shifts between
            time windows (measured via KS statistic)
          - Target contract set evolution: Jaccard distance between
            target sets in early vs. late time windows
        """
        if txs.empty or len(txs) < self.thresholds.min_txs_for_analysis:
            return {
                "passed": False,
                "confidence": 0.0,
                "gas_drift": 0.0,
                "target_evolution": 0.0,
            }

        n_windows = self.thresholds.time_window_blocks
        n = len(txs)
        window_size = n // n_windows

        if window_size < 5:
            # Too few transactions per window for meaningful comparison
            return {
                "passed": False,
                "confidence": 0.1,
                "gas_drift": 0.0,
                "target_evolution": 0.0,
            }

        # Gas strategy drift: KS test between first and last window
        gas_drift = 0.0
        if "gasPrice" in txs.columns:
            gas_prices = pd.to_numeric(
                txs["gasPrice"], errors="coerce"
            ).dropna()
            if len(gas_prices) >= 2 * window_size:
                early_gas = gas_prices.iloc[:window_size].values
                late_gas = gas_prices.iloc[-window_size:].values
                if np.std(early_gas) > 0 or np.std(late_gas) > 0:
                    ks_stat, _ = stats.ks_2samp(early_gas, late_gas)
                    gas_drift = float(ks_stat)

        # Target contract set evolution: Jaccard distance
        target_evolution = 0.0
        if "to" in txs.columns:
            to_addrs = txs["to"].dropna()
            if len(to_addrs) >= 2 * window_size:
                early_targets = set(to_addrs.iloc[:window_size].str.lower())
                late_targets = set(to_addrs.iloc[-window_size:].str.lower())
                union = early_targets | late_targets
                intersection = early_targets & late_targets
                if union:
                    jaccard_sim = len(intersection) / len(union)
                    target_evolution = 1.0 - jaccard_sim  # Jaccard distance

        gas_ok = gas_drift > self.thresholds.gas_drift_threshold
        target_ok = target_evolution > self.thresholds.target_evolution_threshold

        passed = gas_ok or target_ok

        conf = 0.0
        if gas_ok and target_ok:
            conf = 0.9
        elif gas_ok:
            conf = 0.6
        elif target_ok:
            conf = 0.5

        return {
            "passed": passed,
            "confidence": conf,
            "gas_drift": gas_drift,
            "target_evolution": target_evolution,
        }

    # ----------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------

    def _get_transactions(self, address: str) -> pd.DataFrame:
        """Fetch transactions for an address, with optional caching."""
        addr_lower = address.lower()
        if self._cache_enabled and addr_lower in self._cache:
            return self._cache[addr_lower]

        txs = self.client.get_all_txs(address, max_pages=5)
        if self._cache_enabled:
            self._cache[addr_lower] = txs
        return txs

    def verify_from_parquet(
        self,
        parquet_path: str,
        address: str,
    ) -> dict:
        """Verify C1-C4 using pre-collected transaction data from a parquet file.

        This avoids API calls by loading cached raw transactions.

        Args:
            parquet_path: Path to address-specific parquet file.
            address: Ethereum address.

        Returns:
            Verification result dictionary.
        """
        txs = pd.read_parquet(parquet_path)
        return self.verify(address, txs=txs)


# ============================================================
# CLI
# ============================================================

def main():
    """Verify C1-C4 conditions for given addresses."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Verify C1-C4 conditions for Ethereum addresses"
    )
    parser.add_argument(
        "addresses",
        nargs="+",
        help="Ethereum addresses to verify",
    )
    parser.add_argument(
        "--parquet-dir",
        default=None,
        help="Directory containing address-specific parquet files",
    )
    parser.add_argument(
        "--min-txs",
        type=int,
        default=20,
        help="Minimum transactions for analysis (default: 20)",
    )
    args = parser.parse_args()

    client = EtherscanClient()
    thresholds = C1C4Thresholds(min_txs_for_analysis=args.min_txs)
    verifier = C1C4Verifier(client, thresholds=thresholds)

    print("=" * 70)
    print("C1-C4 Verification Results")
    print("=" * 70)

    for addr in args.addresses:
        if args.parquet_dir:
            parquet_file = os.path.join(args.parquet_dir, f"{addr}.parquet")
            if os.path.exists(parquet_file):
                result = verifier.verify_from_parquet(parquet_file, addr)
            else:
                logger.warning("Parquet not found for %s, using API", addr[:12])
                result = verifier.verify(addr)
        else:
            result = verifier.verify(addr)

        print(f"\nAddress: {addr}")
        print(f"  C1 (On-chain Actuation):      {result['c1']}")
        print(f"  C2 (Environmental Perception): {result['c2']}")
        print(f"  C3 (Autonomous Decision):      {result['c3']}")
        print(f"  C4 (Adaptiveness):             {result['c4']}")
        print(f"  => IS AGENT: {result['is_agent']}  "
              f"(confidence: {result['confidence']:.3f})")

        details = result.get("details", {})
        if details:
            print("  Details:")
            for key, value in details.items():
                print(f"    {key}: {value}")


if __name__ == "__main__":
    main()
