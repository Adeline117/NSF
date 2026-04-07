"""
Ground Truth Labeling for On-Chain AI Agent Identification
==========================================================
Defines how we establish binary labels (AGENT / HUMAN) with high
confidence, suitable for training and evaluating a supervised classifier.

Labeling Strategy
-----------------
We combine multiple orthogonal evidence sources so that each label
carries strong provenance.  Addresses that cannot be labeled with
high confidence are marked UNKNOWN and excluded from training.

Confirmed Agents (positive labels):
  1. Autonolas ServiceRegistry -- registered autonomous agents.
  2. Known MEV bots -- from Flashbots, EigenPhi public lists.
  3. Known AI trading bots -- from protocol registries
     (AI16Z ELIZA, Virtuals Protocol launchpad interactions).
  4. Addresses whose *only* interactions are with AI-agent launchpads.

Confirmed Humans (negative labels):
  1. ENS-named addresses with public social verification.
  2. Known whale / institution addresses (Arkham, Nansen labels).
  3. Addresses with clear human behavioral pattern (manual DEX trades
     via UI, irregular timing, circadian rhythm).

Labeling Rules:
  AGENT:   registered in an agent protocol OR confirmed bot by MEV
           tracker.
  HUMAN:   ENS + social proof AND no agent-protocol interaction.
  UNKNOWN: everything else -- excluded from training set.
"""

import os
import sys
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from shared.utils.eth_utils import EtherscanClient

logger = logging.getLogger(__name__)


# ============================================================
# LABEL DEFINITIONS
# ============================================================

class Label(str, Enum):
    """Ground truth label for an Ethereum address."""

    AGENT = "AGENT"
    HUMAN = "HUMAN"
    UNKNOWN = "UNKNOWN"


@dataclass
class LabeledAddress:
    """An Ethereum address together with its ground-truth label and
    provenance metadata."""

    address: str
    label: Label
    source: str          # e.g. "autonolas_registry", "flashbots_mev"
    confidence: float    # 0.0-1.0 subjective confidence
    notes: str = ""


# ============================================================
# KNOWN ADDRESS REGISTRIES
# ============================================================

@dataclass
class AgentRegistry:
    """Static registries of known agent and human addresses.

    These lists are curated from public on-chain data and third-party
    labeling services.  They are intentionally kept small and
    high-confidence; bulk expansion happens in :meth:`GroundTruthLabeler.expand_from_chain`.
    """

    # ---- Autonolas / OLAS ----
    autonolas_service_registry: str = (
        "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2"
    )

    # ---- AI Agent Launchpads ----
    # Addresses associated with AI16Z ELIZA framework deployments
    # and Virtuals Protocol agent launchpad.
    ai16z_deployer: str = ""   # populated at runtime from on-chain query
    virtuals_launchpad: str = ""

    # ---- Known MEV Bots (Flashbots / EigenPhi) ----
    # All entries must be verified EOAs, NOT smart contracts
    known_mev_bots: list = field(default_factory=lambda: [
        "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9",  # jaredfromsubway.eth
        "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80",  # Known sandwich bot
        "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13",  # jaredfromsubway v2
        "0x000000000000cd17345801aa8147b8D3950260FF",  # MEV bot
        "0x00000000003b3cc22aF3aE1EAc0440BcEe416B40",  # MEV bot
    ])

    # ---- Market Maker EOAs ----
    known_market_makers: list = field(default_factory=lambda: [
        "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C",  # Wintermute
        "0x280027dd00eE0050d3F9d168EFD6B40090009246",  # Wintermute 2
        "0xDBF5E9c5206d0dB70a90108bf936DA60221dC080",  # Wintermute 3
    ])

    # ---- Known AI Trading Bots (protocol-registered) ----
    known_ai_bots: list = field(default_factory=lambda: [
        # Addresses confirmed via on-chain registration in agent protocols
        # Populated during data collection phase
    ])

    # ---- Known Human Addresses (ENS + social proof, verified EOAs) ----
    known_humans: list = field(default_factory=lambda: [
        "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",  # vitalik.eth
        "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B",  # Vitalik older addr
        "0x220866B1A2219f40e72f5c628B65D54268cA3A9D",  # hayden.eth (Uniswap)
        "0x983110309620D911731Ac0932219af06091b6744",  # brantly.eth
        "0xCB42Ac441fCadeB7a0B36E38F1d5E8cBe1832599",  # sassal.eth
    ])

    # NOTE: Exchange hot/cold wallets are EXCLUDED entirely.
    # They are neither human-operated EOAs nor autonomous agents.
    # Removed: Binance 7/8/14/15/16, Kraken 4/13, Bitfinex, Gemini,
    #          FTX, DWF Labs hot, MEXC hot wallet.
    known_institutions: list = field(default_factory=lambda: [
        # Intentionally empty -- exchange wallets excluded from study
    ])


# ============================================================
# GROUND TRUTH LABELER
# ============================================================

class GroundTruthLabeler:
    """Assigns ground-truth labels to Ethereum addresses.

    The labeler first checks static registries, then optionally queries
    on-chain data to discover additional agent addresses from protocol
    registries (Autonolas, AI16Z, Virtuals).

    Usage::

        labeler = GroundTruthLabeler(client)
        labeled = labeler.label_addresses([
            "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
            "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9",
        ])
        for la in labeled:
            print(la.address, la.label, la.source)

    Args:
        client: EtherscanClient for on-chain queries.
        registry: Optional custom AgentRegistry; uses defaults if omitted.
    """

    def __init__(
        self,
        client: EtherscanClient,
        registry: Optional[AgentRegistry] = None,
    ):
        self.client = client
        self.registry = registry or AgentRegistry()
        self._agent_set: set[str] = set()
        self._human_set: set[str] = set()
        self._build_lookup_sets()

    def _build_lookup_sets(self) -> None:
        """Build O(1) lookup sets from registry lists."""
        for addr in self.registry.known_mev_bots:
            self._agent_set.add(addr.lower())
        for addr in self.registry.known_market_makers:
            self._agent_set.add(addr.lower())
        for addr in self.registry.known_ai_bots:
            self._agent_set.add(addr.lower())

        for addr in self.registry.known_humans:
            self._human_set.add(addr.lower())
        for addr in self.registry.known_institutions:
            self._human_set.add(addr.lower())

    # ----------------------------------------------------------
    # Core labeling
    # ----------------------------------------------------------

    def label_single(self, address: str) -> LabeledAddress:
        """Label a single Ethereum address.

        Priority order:
        1. Check agent set (MEV bots, AI bots).
        2. Check human set (ENS-verified, institutions).
        3. Mark as UNKNOWN.
        """
        addr_lower = address.lower()

        if addr_lower in self._agent_set:
            return LabeledAddress(
                address=address,
                label=Label.AGENT,
                source=self._agent_source(addr_lower),
                confidence=0.95,
            )

        if addr_lower in self._human_set:
            return LabeledAddress(
                address=address,
                label=Label.HUMAN,
                source=self._human_source(addr_lower),
                confidence=0.90,
            )

        return LabeledAddress(
            address=address,
            label=Label.UNKNOWN,
            source="no_match",
            confidence=0.0,
        )

    def label_addresses(
        self, addresses: list[str]
    ) -> list[LabeledAddress]:
        """Label a batch of addresses."""
        return [self.label_single(a) for a in addresses]

    def to_dataframe(
        self, labeled: list[LabeledAddress]
    ) -> pd.DataFrame:
        """Convert labeled addresses to a DataFrame."""
        records = [
            {
                "address": la.address,
                "label": la.label.value,
                "source": la.source,
                "confidence": la.confidence,
                "notes": la.notes,
            }
            for la in labeled
        ]
        return pd.DataFrame(records)

    # ----------------------------------------------------------
    # On-chain expansion
    # ----------------------------------------------------------

    def expand_from_autonolas(self) -> list[str]:
        """Query the Autonolas ServiceRegistry for registered agent
        addresses.

        This queries event logs from the ServiceRegistry contract to
        discover addresses that have been registered as autonomous agents.
        The discovered addresses are added to the internal agent set.

        Returns:
            List of newly discovered agent addresses.
        """
        registry_addr = self.registry.autonolas_service_registry
        logger.info(
            "Querying Autonolas ServiceRegistry at %s ...", registry_addr
        )
        discovered: list[str] = []

        try:
            # Query ERC20 transfers FROM the registry as a proxy for
            # service registrations (the registry emits token events
            # when new services are created).
            txs = self.client.get_normal_txs(registry_addr, offset=1000)
            if txs.empty:
                logger.warning("No transactions found for Autonolas registry.")
                return discovered

            # Addresses that sent transactions TO the registry are
            # likely agent operators; the 'from' addresses of those
            # transactions are candidate agent addresses.
            candidate_addrs = txs["from"].unique().tolist()
            for addr in candidate_addrs:
                addr_lower = addr.lower()
                if addr_lower not in self._agent_set:
                    self._agent_set.add(addr_lower)
                    discovered.append(addr)

            logger.info(
                "Discovered %d candidate agent addresses from Autonolas.",
                len(discovered),
            )
        except Exception as exc:
            logger.error("Failed to query Autonolas registry: %s", exc)

        return discovered

    def expand_from_launchpad_interactions(
        self,
        launchpad_address: str,
        label_source: str = "launchpad_interaction",
    ) -> list[str]:
        """Discover agent addresses by finding addresses that interacted
        with a known AI-agent launchpad contract (e.g., Virtuals, AI16Z).

        Args:
            launchpad_address: Address of the launchpad contract.
            label_source: Provenance label for discovered addresses.

        Returns:
            List of newly discovered agent addresses.
        """
        discovered: list[str] = []
        try:
            txs = self.client.get_normal_txs(
                launchpad_address, offset=1000
            )
            if txs.empty:
                return discovered

            interactors = txs["from"].unique().tolist()
            for addr in interactors:
                addr_lower = addr.lower()
                if (
                    addr_lower not in self._agent_set
                    and addr_lower not in self._human_set
                ):
                    self._agent_set.add(addr_lower)
                    discovered.append(addr)

            logger.info(
                "Discovered %d candidate agents from launchpad %s.",
                len(discovered),
                launchpad_address[:10],
            )
        except Exception as exc:
            logger.error(
                "Failed to query launchpad %s: %s",
                launchpad_address,
                exc,
            )

        return discovered

    # ----------------------------------------------------------
    # Dataset construction
    # ----------------------------------------------------------

    def build_labeled_dataset(
        self,
        additional_addresses: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        """Build the full labeled dataset for model training.

        Combines all known agent and human addresses, labels them, and
        returns a DataFrame.  Addresses labeled UNKNOWN are included in
        the output but should be filtered before training.

        Args:
            additional_addresses: Extra addresses to label (will be
                labeled UNKNOWN if not in any registry).

        Returns:
            DataFrame with columns: address, label, source, confidence.
        """
        all_addresses: list[str] = []

        # Collect from registries
        all_addresses.extend(self.registry.known_mev_bots)
        all_addresses.extend(self.registry.known_market_makers)
        all_addresses.extend(self.registry.known_ai_bots)
        all_addresses.extend(self.registry.known_humans)
        all_addresses.extend(self.registry.known_institutions)

        if additional_addresses:
            all_addresses.extend(additional_addresses)

        # Deduplicate while preserving order
        seen: set[str] = set()
        unique: list[str] = []
        for addr in all_addresses:
            if addr.lower() not in seen:
                seen.add(addr.lower())
                unique.append(addr)

        labeled = self.label_addresses(unique)
        df = self.to_dataframe(labeled)

        counts = df["label"].value_counts()
        logger.info("Label distribution:\n%s", counts.to_string())

        return df

    # ----------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------

    def _agent_source(self, addr_lower: str) -> str:
        """Determine the provenance source for an agent label."""
        mev_set = {a.lower() for a in self.registry.known_mev_bots}
        if addr_lower in mev_set:
            return "flashbots_mev"
        mm_set = {a.lower() for a in self.registry.known_market_makers}
        if addr_lower in mm_set:
            return "market_maker"
        ai_set = {a.lower() for a in self.registry.known_ai_bots}
        if addr_lower in ai_set:
            return "ai_protocol_registry"
        return "autonolas_registry"

    def _human_source(self, addr_lower: str) -> str:
        """Determine the provenance source for a human label."""
        human_set = {a.lower() for a in self.registry.known_humans}
        if addr_lower in human_set:
            return "ens_social_proof"
        return "institution_label"


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """Demonstrate ground truth labeling."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Ground truth labeling for on-chain AI agent ID"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ETHERSCAN_API_KEY", ""),
        help="Etherscan API key",
    )
    parser.add_argument(
        "--expand-autonolas",
        action="store_true",
        help="Query Autonolas registry for additional agent addresses",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    client = EtherscanClient(api_key=args.api_key)
    labeler = GroundTruthLabeler(client)

    if args.expand_autonolas:
        new_agents = labeler.expand_from_autonolas()
        print(f"Discovered {len(new_agents)} new agent candidates.")

    df = labeler.build_labeled_dataset()
    print("\n--- Labeled Dataset ---")
    print(df.to_string(index=False))
    print(f"\nTotal: {len(df)} addresses")

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
