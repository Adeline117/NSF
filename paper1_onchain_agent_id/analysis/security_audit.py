"""
Four-Dimensional Security Audit for Identified AI Agents
=========================================================
Once an address is classified as an AI agent, this module audits its
security posture along four dimensions:

Dimension 1: Permission Exposure
  - Count of active unlimited approvals (MaxUint256).
  - Approval duration (time since approve with no revoke).
  - Approved contracts: verified vs unverified.
  - Total value at risk from outstanding approvals.

Dimension 2: Agent Network Topology
  - Build agent-to-agent transfer and call graph.
  - Compute: degree centrality, clustering coefficient, betweenness.
  - Identify systemic-risk nodes (high centrality agents).
  - Detect agent clusters via community detection.

Dimension 3: MEV Exposure
  - Sandwich attack rate: fraction of DEX trades that are sandwiched.
  - Compare to human baseline sandwich rate.
  - Frontrunning exposure estimation.
  - Flashbots / MEV-Boost data integration (when available).

Dimension 4: Failure Analysis
  - Revert rate: fraction of transactions that revert.
  - Revert reason classification (out of gas, require fail, etc.).
  - Compare agent revert rate to human baseline.
  - Error-recovery patterns (retry behaviour).

Usage::

    from paper1_onchain_agent_id.analysis.security_audit import SecurityAuditor

    auditor = SecurityAuditor(client)
    report = auditor.full_audit(agent_addresses)
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from shared.utils.eth_utils import EtherscanClient

logger = logging.getLogger(__name__)

# Constants
MAX_UINT256_HEX = (
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
)
APPROVE_METHOD_ID = "0x095ea7b3"

# Known DEX router addresses for MEV-exposure analysis
DEX_ROUTERS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 Router2
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Universal Router
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff",  # 0x Proxy
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 Router
    "0x1111111254eeb25477b68fb85ed929f73a960582",  # 1inch V5
}

# Swap method selectors (common DEX swap functions)
SWAP_SIGNATURES = {
    "0x38ed1739",  # swapExactTokensForTokens
    "0x8803dbee",  # swapTokensForExactTokens
    "0x7ff36ab5",  # swapExactETHForTokens
    "0x18cbafe5",  # swapExactTokensForETH
    "0xfb3bdb41",  # swapETHForExactTokens
    "0x5ae401dc",  # multicall (Uniswap V3)
    "0x3593564c",  # execute (Universal Router)
}


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass
class PermissionExposure:
    """Results of permission-exposure audit for one address."""

    address: str
    total_approvals: int = 0
    unlimited_approvals: int = 0
    active_approvals: int = 0  # approve without matching revoke
    avg_approval_age_days: float = 0.0
    max_approval_age_days: float = 0.0
    unverified_contract_approvals: int = 0
    verified_contract_approvals: int = 0
    estimated_value_at_risk_usd: float = 0.0  # placeholder for future pricing


@dataclass
class NetworkMetrics:
    """Network topology metrics for an agent node."""

    address: str
    degree: int = 0
    in_degree: int = 0
    out_degree: int = 0
    degree_centrality: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    community_id: int = -1


@dataclass
class MEVExposure:
    """MEV exposure metrics for one address."""

    address: str
    total_dex_trades: int = 0
    sandwiched_trades: int = 0
    sandwich_rate: float = 0.0
    frontrun_exposure_count: int = 0
    avg_slippage_bps: float = 0.0


@dataclass
class FailureAnalysis:
    """Transaction failure analysis for one address."""

    address: str
    total_transactions: int = 0
    reverted_transactions: int = 0
    revert_rate: float = 0.0
    out_of_gas_count: int = 0
    require_fail_count: int = 0
    other_revert_count: int = 0
    retry_detected: bool = False
    avg_retry_delay_seconds: float = 0.0


@dataclass
class SecurityReport:
    """Complete security audit report for one address."""

    address: str
    permission: Optional[PermissionExposure] = None
    network: Optional[NetworkMetrics] = None
    mev: Optional[MEVExposure] = None
    failure: Optional[FailureAnalysis] = None
    risk_score: float = 0.0  # composite 0-100


# ============================================================
# SECURITY AUDITOR
# ============================================================

class SecurityAuditor:
    """Performs four-dimensional security audit on identified AI agents.

    Args:
        client: EtherscanClient for on-chain queries.
        sandwich_window_blocks: Number of blocks around a swap to search
            for potential sandwich transactions.
    """

    def __init__(
        self,
        client: EtherscanClient,
        sandwich_window_blocks: int = 2,
    ):
        self.client = client
        self.sandwich_window_blocks = sandwich_window_blocks

    # ----------------------------------------------------------
    # Full Audit
    # ----------------------------------------------------------

    def full_audit(
        self,
        agent_addresses: list[str],
    ) -> list[SecurityReport]:
        """Run all four audit dimensions on a list of agent addresses.

        Args:
            agent_addresses: Addresses classified as AI agents.

        Returns:
            List of SecurityReport, one per address.
        """
        reports: list[SecurityReport] = []

        for addr in agent_addresses:
            logger.info("Auditing %s ...", addr[:12])
            try:
                report = SecurityReport(address=addr)

                txs = self.client.get_normal_txs(addr, offset=1000)
                if txs.empty:
                    logger.warning("No transactions for %s, skipping.", addr)
                    reports.append(report)
                    continue

                report.permission = self.audit_permissions(addr, txs)
                report.mev = self.audit_mev_exposure(addr, txs)
                report.failure = self.audit_failures(addr, txs)
                report.risk_score = self._compute_risk_score(report)

                reports.append(report)
            except Exception as exc:
                logger.error("Audit failed for %s: %s", addr, exc)
                reports.append(SecurityReport(address=addr))

        # Network analysis requires all addresses together
        network_metrics = self.audit_network_topology(agent_addresses)
        for report in reports:
            report.network = network_metrics.get(report.address)

        return reports

    # ----------------------------------------------------------
    # Dimension 1: Permission Exposure
    # ----------------------------------------------------------

    def audit_permissions(
        self, address: str, txs: pd.DataFrame
    ) -> PermissionExposure:
        """Audit token approval permissions for an address.

        Analyses all ERC-20 ``approve`` transactions to identify
        outstanding unlimited approvals and unverified target contracts.

        Args:
            address: The agent address.
            txs: Pre-fetched normal transactions.

        Returns:
            PermissionExposure dataclass.
        """
        result = PermissionExposure(address=address)

        if "input" not in txs.columns:
            return result

        inputs = txs["input"].fillna("")
        approve_mask = inputs.str.startswith(APPROVE_METHOD_ID, na=False)
        approvals = txs[approve_mask].copy()

        if approvals.empty:
            return result

        result.total_approvals = len(approvals)

        # Unlimited approvals
        unlimited_mask = approvals["input"].str.contains(
            MAX_UINT256_HEX, na=False
        )
        result.unlimited_approvals = int(unlimited_mask.sum())

        # Identify revocations (approve with zero amount)
        zero_amount = (
            "0000000000000000000000000000000000000000000000000000000000000000"
        )
        revoke_mask = approvals["input"].str.endswith(zero_amount, na=False)
        revoked_targets = set(
            approvals.loc[revoke_mask, "to"].str.lower().tolist()
        )

        # Active (non-revoked) approvals
        all_approve_targets = set(
            approvals.loc[~revoke_mask, "to"].str.lower().tolist()
        )
        active_targets = all_approve_targets - revoked_targets
        result.active_approvals = len(active_targets)

        # Approval age (for non-revoked approvals)
        if "timeStamp" in approvals.columns:
            ts = pd.to_numeric(approvals["timeStamp"], errors="coerce")
            non_revoked = approvals[~revoke_mask]
            if not non_revoked.empty:
                non_revoked_ts = pd.to_numeric(
                    non_revoked["timeStamp"], errors="coerce"
                ).dropna()
                if not non_revoked_ts.empty:
                    import time as _time
                    now = _time.time()
                    ages_days = (now - non_revoked_ts) / 86400
                    result.avg_approval_age_days = float(ages_days.mean())
                    result.max_approval_age_days = float(ages_days.max())

        # Verified vs unverified target contracts
        verified_count = 0
        unverified_count = 0
        for target in list(active_targets)[:20]:  # cap API calls
            try:
                abi = self.client.get_contract_abi(target)
                if abi is not None:
                    verified_count += 1
                else:
                    unverified_count += 1
            except Exception:
                unverified_count += 1
        result.verified_contract_approvals = verified_count
        result.unverified_contract_approvals = unverified_count

        return result

    # ----------------------------------------------------------
    # Dimension 2: Network Topology
    # ----------------------------------------------------------

    def audit_network_topology(
        self, agent_addresses: list[str]
    ) -> dict[str, NetworkMetrics]:
        """Build agent-to-agent interaction graph and compute metrics.

        Constructs a directed graph where nodes are agent addresses and
        edges represent ETH transfers or contract calls between agents.
        Computes centrality measures and detects communities.

        Args:
            agent_addresses: List of agent addresses.

        Returns:
            Dict mapping address to NetworkMetrics.
        """
        try:
            import networkx as nx
        except ImportError:
            logger.warning("networkx not installed; skipping topology analysis.")
            return {}

        agent_set = {a.lower() for a in agent_addresses}
        G = nx.DiGraph()

        for addr in agent_addresses:
            G.add_node(addr.lower())

        # Build edges from transaction data
        for addr in agent_addresses:
            try:
                txs = self.client.get_normal_txs(addr, offset=500)
                if txs.empty:
                    continue
                for _, row in txs.iterrows():
                    from_addr = str(row.get("from", "")).lower()
                    to_addr = str(row.get("to", "")).lower()
                    if from_addr in agent_set and to_addr in agent_set:
                        if G.has_edge(from_addr, to_addr):
                            G[from_addr][to_addr]["weight"] += 1
                        else:
                            G.add_edge(from_addr, to_addr, weight=1)
            except Exception as exc:
                logger.warning(
                    "Failed to fetch txs for network analysis of %s: %s",
                    addr, exc,
                )

        # Compute metrics
        results: dict[str, NetworkMetrics] = {}

        degree_cent = nx.degree_centrality(G) if len(G) > 1 else {}
        betweenness = nx.betweenness_centrality(G) if len(G) > 1 else {}
        clustering = nx.clustering(G.to_undirected()) if len(G) > 1 else {}

        # Community detection
        communities: dict[str, int] = {}
        if len(G) > 1:
            try:
                undirected = G.to_undirected()
                from networkx.algorithms.community import greedy_modularity_communities
                comms = list(greedy_modularity_communities(undirected))
                for comm_id, comm in enumerate(comms):
                    for node in comm:
                        communities[node] = comm_id
            except Exception:
                pass

        for addr in agent_addresses:
            addr_lower = addr.lower()
            metrics = NetworkMetrics(address=addr)
            if addr_lower in G:
                metrics.degree = G.degree(addr_lower)
                metrics.in_degree = G.in_degree(addr_lower)
                metrics.out_degree = G.out_degree(addr_lower)
                metrics.degree_centrality = degree_cent.get(addr_lower, 0.0)
                metrics.betweenness_centrality = betweenness.get(addr_lower, 0.0)
                metrics.clustering_coefficient = clustering.get(addr_lower, 0.0)
                metrics.community_id = communities.get(addr_lower, -1)
            results[addr] = metrics

        logger.info(
            "Network analysis complete: %d nodes, %d edges.",
            G.number_of_nodes(),
            G.number_of_edges(),
        )

        return results

    # ----------------------------------------------------------
    # Dimension 3: MEV Exposure
    # ----------------------------------------------------------

    def audit_mev_exposure(
        self, address: str, txs: pd.DataFrame
    ) -> MEVExposure:
        """Estimate MEV exposure for an agent address.

        Identifies DEX swap transactions and checks for potential
        sandwich attacks by looking for same-pool transactions in
        adjacent blocks.

        Args:
            address: The agent address.
            txs: Pre-fetched normal transactions.

        Returns:
            MEVExposure dataclass.
        """
        result = MEVExposure(address=address)

        if "input" not in txs.columns or "blockNumber" not in txs.columns:
            return result

        inputs = txs["input"].fillna("")
        to_addrs = txs["to"].fillna("").str.lower()

        # Identify DEX swap transactions
        is_swap = inputs.apply(
            lambda x: isinstance(x, str) and len(x) >= 10 and x[:10] in SWAP_SIGNATURES
        )
        is_dex = to_addrs.isin(DEX_ROUTERS)
        swap_mask = is_swap | is_dex
        swap_txs = txs[swap_mask]

        result.total_dex_trades = len(swap_txs)
        if swap_txs.empty:
            return result

        # Sandwich detection heuristic:
        # For each swap, check whether there are transactions to the same
        # router in the immediately preceding and following blocks that
        # are NOT from this address (potential sandwich).
        sandwiched_count = 0
        block_numbers = pd.to_numeric(
            swap_txs["blockNumber"], errors="coerce"
        ).dropna()

        for _, swap_row in swap_txs.iterrows():
            block = int(swap_row.get("blockNumber", 0))
            target = str(swap_row.get("to", "")).lower()
            if block == 0 or not target:
                continue

            # Look for surrounding transactions in nearby blocks
            # from addresses other than the agent
            nearby = txs[
                (pd.to_numeric(txs["blockNumber"], errors="coerce") >= block - self.sandwich_window_blocks)
                & (pd.to_numeric(txs["blockNumber"], errors="coerce") <= block + self.sandwich_window_blocks)
                & (txs["from"].str.lower() != address.lower())
                & (txs["to"].str.lower() == target)
            ]
            if len(nearby) >= 2:
                # Two transactions from other addresses bracketing ours
                # is a strong sandwich signal
                sandwiched_count += 1

        result.sandwiched_trades = sandwiched_count
        if result.total_dex_trades > 0:
            result.sandwich_rate = (
                sandwiched_count / result.total_dex_trades
            )

        return result

    # ----------------------------------------------------------
    # Dimension 4: Failure Analysis
    # ----------------------------------------------------------

    def audit_failures(
        self, address: str, txs: pd.DataFrame
    ) -> FailureAnalysis:
        """Analyse transaction failures for an agent address.

        Classifies reverted transactions by reason and detects retry
        patterns (same method+target within short time after a revert).

        Args:
            address: The agent address.
            txs: Pre-fetched normal transactions.

        Returns:
            FailureAnalysis dataclass.
        """
        result = FailureAnalysis(address=address)
        result.total_transactions = len(txs)

        if txs.empty:
            return result

        # Identify reverted transactions
        if "isError" in txs.columns:
            is_error = pd.to_numeric(
                txs["isError"], errors="coerce"
            ).fillna(0)
            reverted = txs[is_error == 1]
            result.reverted_transactions = len(reverted)
        elif "txreceipt_status" in txs.columns:
            status = pd.to_numeric(
                txs["txreceipt_status"], errors="coerce"
            ).fillna(1)
            reverted = txs[status == 0]
            result.reverted_transactions = len(reverted)
        else:
            return result

        if result.total_transactions > 0:
            result.revert_rate = (
                result.reverted_transactions / result.total_transactions
            )

        # Classify revert reasons (from Etherscan's error field if present)
        if not reverted.empty:
            gas_used = pd.to_numeric(
                reverted.get("gasUsed", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0)
            gas_limit = pd.to_numeric(
                reverted.get("gas", pd.Series(dtype=float)),
                errors="coerce",
            ).fillna(0)

            # Out of gas: gas_used ~= gas_limit
            oog_mask = (gas_limit > 0) & (gas_used / gas_limit > 0.99)
            result.out_of_gas_count = int(oog_mask.sum())
            result.require_fail_count = (
                result.reverted_transactions - result.out_of_gas_count
            )
            result.other_revert_count = 0  # refined in future work

        # Retry detection: look for same (to, method_id) pairs within
        # 60 seconds after a revert
        if not reverted.empty and "input" in txs.columns:
            result.retry_detected, result.avg_retry_delay_seconds = (
                self._detect_retries(txs, reverted)
            )

        return result

    @staticmethod
    def _detect_retries(
        all_txs: pd.DataFrame, reverted: pd.DataFrame
    ) -> tuple[bool, float]:
        """Detect retry behaviour after reverted transactions.

        Returns (retry_detected, avg_retry_delay_seconds).
        """
        retry_delays: list[float] = []

        timestamps = pd.to_numeric(
            all_txs["timeStamp"], errors="coerce"
        )
        all_txs = all_txs.copy()
        all_txs["_ts"] = timestamps

        for _, rev_row in reverted.iterrows():
            rev_ts = pd.to_numeric(
                rev_row.get("timeStamp", 0), errors="coerce"
            )
            rev_to = str(rev_row.get("to", "")).lower()
            rev_method = ""
            inp = rev_row.get("input", "")
            if isinstance(inp, str) and len(inp) >= 10:
                rev_method = inp[:10]

            if not rev_method or rev_ts == 0:
                continue

            # Look for a successful retry within 120 seconds
            candidates = all_txs[
                (all_txs["_ts"] > rev_ts)
                & (all_txs["_ts"] <= rev_ts + 120)
                & (all_txs["to"].str.lower() == rev_to)
                & (all_txs["input"].str[:10] == rev_method)
            ]

            if "isError" in candidates.columns:
                success = candidates[
                    pd.to_numeric(
                        candidates["isError"], errors="coerce"
                    ).fillna(0) == 0
                ]
            else:
                success = candidates

            if not success.empty:
                delay = float(success["_ts"].iloc[0] - rev_ts)
                retry_delays.append(delay)

        if retry_delays:
            return True, float(np.mean(retry_delays))
        return False, 0.0

    # ----------------------------------------------------------
    # Risk Score
    # ----------------------------------------------------------

    def _compute_risk_score(self, report: SecurityReport) -> float:
        """Compute a composite risk score (0-100) from audit dimensions.

        Weighting:
          - Permission exposure: 35%
          - MEV exposure: 25%
          - Failure rate: 25%
          - Network systemic risk: 15%

        Each dimension contributes a sub-score normalised to [0, 1].
        """
        score = 0.0

        # Permission sub-score
        if report.permission is not None:
            p = report.permission
            perm_score = 0.0
            if p.total_approvals > 0:
                perm_score += 0.3 * min(p.unlimited_approvals / max(p.total_approvals, 1), 1.0)
                perm_score += 0.3 * min(p.active_approvals / 10.0, 1.0)
                perm_score += 0.2 * min(p.avg_approval_age_days / 365.0, 1.0)
                perm_score += 0.2 * (
                    p.unverified_contract_approvals
                    / max(p.unverified_contract_approvals + p.verified_contract_approvals, 1)
                )
            score += 35.0 * perm_score

        # MEV sub-score
        if report.mev is not None:
            m = report.mev
            mev_score = min(m.sandwich_rate * 5.0, 1.0)  # 20% sandwich = max
            score += 25.0 * mev_score

        # Failure sub-score
        if report.failure is not None:
            f = report.failure
            fail_score = min(f.revert_rate * 5.0, 1.0)  # 20% revert = max
            score += 25.0 * fail_score

        # Network sub-score (added after network analysis)
        if report.network is not None:
            n = report.network
            net_score = min(n.betweenness_centrality * 10.0, 1.0)
            score += 15.0 * net_score

        return round(score, 2)

    # ----------------------------------------------------------
    # Reporting
    # ----------------------------------------------------------

    def to_dataframe(
        self, reports: list[SecurityReport]
    ) -> pd.DataFrame:
        """Convert audit reports to a flat DataFrame for analysis."""
        rows = []
        for r in reports:
            row: dict = {"address": r.address, "risk_score": r.risk_score}
            if r.permission:
                row.update({
                    "perm_total_approvals": r.permission.total_approvals,
                    "perm_unlimited": r.permission.unlimited_approvals,
                    "perm_active": r.permission.active_approvals,
                    "perm_avg_age_days": r.permission.avg_approval_age_days,
                    "perm_unverified": r.permission.unverified_contract_approvals,
                })
            if r.mev:
                row.update({
                    "mev_dex_trades": r.mev.total_dex_trades,
                    "mev_sandwiched": r.mev.sandwiched_trades,
                    "mev_sandwich_rate": r.mev.sandwich_rate,
                })
            if r.failure:
                row.update({
                    "fail_total": r.failure.total_transactions,
                    "fail_reverted": r.failure.reverted_transactions,
                    "fail_revert_rate": r.failure.revert_rate,
                    "fail_out_of_gas": r.failure.out_of_gas_count,
                    "fail_retry": r.failure.retry_detected,
                })
            if r.network:
                row.update({
                    "net_degree": r.network.degree,
                    "net_betweenness": r.network.betweenness_centrality,
                    "net_clustering": r.network.clustering_coefficient,
                    "net_community": r.network.community_id,
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def print_summary(self, reports: list[SecurityReport]) -> None:
        """Print a human-readable summary of audit results."""
        print("\n" + "=" * 70)
        print("SECURITY AUDIT SUMMARY")
        print("=" * 70)

        for r in reports:
            print(f"\nAddress: {r.address}")
            print(f"  Risk Score: {r.risk_score}/100")

            if r.permission:
                p = r.permission
                print(f"  [Permission] Approvals: {p.total_approvals} total, "
                      f"{p.unlimited_approvals} unlimited, "
                      f"{p.active_approvals} active")
                print(f"               Unverified targets: {p.unverified_contract_approvals}")
                print(f"               Max approval age: {p.max_approval_age_days:.0f} days")

            if r.mev:
                m = r.mev
                print(f"  [MEV]        DEX trades: {m.total_dex_trades}, "
                      f"sandwiched: {m.sandwiched_trades} "
                      f"({m.sandwich_rate:.1%})")

            if r.failure:
                f = r.failure
                print(f"  [Failure]    Revert rate: {f.revert_rate:.1%} "
                      f"({f.reverted_transactions}/{f.total_transactions})")
                print(f"               Out of gas: {f.out_of_gas_count}, "
                      f"Retry detected: {f.retry_detected}")

            if r.network:
                n = r.network
                print(f"  [Network]    Degree: {n.degree}, "
                      f"Betweenness: {n.betweenness_centrality:.4f}, "
                      f"Community: {n.community_id}")

        # Aggregate stats
        if reports:
            scores = [r.risk_score for r in reports]
            print(f"\n{'=' * 70}")
            print(f"Aggregate: {len(reports)} agents audited")
            print(f"  Risk scores: mean={np.mean(scores):.1f}, "
                  f"median={np.median(scores):.1f}, "
                  f"max={np.max(scores):.1f}")

            high_risk = [r for r in reports if r.risk_score >= 50]
            print(f"  High-risk agents (score >= 50): {len(high_risk)}")
            print("=" * 70)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """Demo: run security audit on known MEV bots."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Security audit for identified AI agents"
    )
    parser.add_argument(
        "--addresses",
        nargs="+",
        default=[
            "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9",
        ],
        help="Agent addresses to audit",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ETHERSCAN_API_KEY", ""),
        help="Etherscan API key",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = EtherscanClient(api_key=args.api_key)
    auditor = SecurityAuditor(client)
    reports = auditor.full_audit(args.addresses)

    auditor.print_summary(reports)

    if args.output:
        df = auditor.to_dataframe(reports)
        df.to_csv(args.output, index=False)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
