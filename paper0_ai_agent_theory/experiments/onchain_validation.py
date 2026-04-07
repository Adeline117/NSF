"""
On-Chain Registry Validation of C1-C4 AI Agent Definition
==========================================================
Validates the AI Agent definition (C1-C4) using on-chain registries as
objective ground truth. This replaces the Delphi expert study with
blockchain-verifiable evidence.

Approach:
1. Query Autonolas Service Registry for registered agent services
2. Query known agent platform interactions (AI16Z, Virtuals, Fetch.ai)
3. Verify C1-C4 for discovered agents and known agents
4. Validate against known NON-agents (contracts, humans, scripts)
5. Counterexample search (find entities that break the definition)
6. Ablation experiment (show what happens when each condition is removed)

Uses EtherscanClient from shared/utils/eth_utils.py (V2 API).
"""

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from shared.utils.eth_utils import EtherscanClient


# ============================================================
# CONSTANTS: Known Addresses
# ============================================================

# Autonolas Service Registry
AUTONOLAS_REGISTRY = "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2"

# Known agent platform contracts for interaction queries
AGENT_PLATFORM_CONTRACTS = {
    "autonolas_registry": "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2",
    "autonolas_component_registry": "0x15bd56669F57192a97dF41A2aa8f4403e9491776",
    "autonolas_agent_registry": "0x2F1f7D38e4772884b88f3eCd8B6b9faCdC319112",
    "fetch_ai_token": "0xaea46A60368A7bD060eec7DF8CBa43b7EF41Ad85",
    "ai_arena_token": "0x6De037ef9aD2725EB40118Bb1702EBb27e4Aeb24",
}

# Known AI agents that SHOULD satisfy C1-C4
KNOWN_AGENTS = {
    # --- EOA-based agents (should pass C1 directly) ---
    "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9": {
        "name": "jaredfromsubway.eth",
        "type": "mev_bot",
        "expected": {"C1": True, "C2": True, "C3": True, "C4": True},
    },
    "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13": {
        "name": "jaredfromsubway v2",
        "type": "mev_bot",
        "expected": {"C1": True, "C2": True, "C3": True, "C4": True},
    },
    # --- Contract-based agents (operate through contracts, C1 applies to controller) ---
    # These are interesting edge cases: the AGENT controls a contract, but the
    # on-chain address IS a contract. C1 says "directly or indirectly controls an EOA".
    # For contract-based agents, C1 checks if the contract has an EOA owner/operator.
    "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80": {
        "name": "MEV bot (contract-based)",
        "type": "mev_bot_contract",
        "expected": {"C1": True, "C2": True, "C3": True, "C4": True},
        "note": "Operates via smart contract; C1 validated by checking EOA controller",
    },
    "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C": {
        "name": "Wintermute",
        "type": "market_maker_contract",
        "expected": {"C1": True, "C2": True, "C3": True, "C4": True},
        "note": "Operates via smart contract; C1 validated by checking EOA controller",
    },
}

# Known NON-agents that should FAIL at least one of C1-C4
NON_AGENTS = {
    # Smart contracts: should fail C1 (not EOA)
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": {
        "name": "Uniswap V2 Router",
        "type": "contract",
        "expected_fail": "C1",
        "reason": "Smart contract, not an EOA - cannot autonomously sign txs",
    },
    "0x1111111254EEB25477B68fb85Ed929f73A960582": {
        "name": "1inch v5 Router",
        "type": "contract",
        "expected_fail": "C1",
        "reason": "Smart contract router, passively called by users",
    },
    "0xE592427A0AEce92De3Edee1F18E0157C05861564": {
        "name": "Uniswap V3 Router",
        "type": "contract",
        "expected_fail": "C1",
        "reason": "Smart contract, passively invoked",
    },
    # Human wallets: should fail C3 (human-operated, not autonomous)
    # Note: Some prominent wallets (vitalik.eth) have been upgraded to smart
    # contract wallets (ERC-4337), so they may fail C1 instead of C3.
    # We include both types to test the definition robustly.
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": {
        "name": "vitalik.eth",
        "type": "human",
        "expected_fail": "C1_or_C3",
        "reason": "Human-operated wallet; may now be smart contract wallet (ERC-4337)",
    },
    "0x220866B1A2219f40e72f5c628B65D54268cA3A9D": {
        "name": "Hayden Adams (Uniswap founder)",
        "type": "human",
        "expected_fail": "C1_or_C3",
        "reason": "Human wallet; may use smart contract wallet",
    },
    "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B": {
        "name": "Satoshi Nakamoto (known ETH holder)",
        "type": "human",
        "expected_fail": "C1_or_C3",
        "reason": "Human wallet; may use smart contract wallet",
    },
    # Additional human EOA wallets (less likely to have upgraded)
    "0x28C6c06298d514Db089934071355E5743bf21d60": {
        "name": "Binance 14 (exchange hot wallet)",
        "type": "exchange_wallet",
        "expected_fail": "C3",
        "reason": (
            "Exchange hot wallet. Interesting edge case: may pass all C1-C4 "
            "because exchange operations are heavily automated, but they require "
            "human/institutional oversight and are not truly autonomous agents."
        ),
    },
}


# ============================================================
# C1-C4 VERIFICATION FUNCTIONS
# ============================================================

def verify_c1(client: EtherscanClient, address: str) -> tuple[bool, str]:
    """C1: On-chain Actuation - Controls an EOA that can sign transactions.

    Per the definition: "directly or indirectly controls an EOA".
    - If address is an EOA with outgoing transactions -> C1 passes
    - If address is a smart contract, check if it has an EOA controller
      that sends transactions to it (agent operates through contract)
    - Pure passive contracts (routers, pools) with no single controller -> C1 fails
    """
    try:
        is_contract = client.is_contract(address)

        if not is_contract:
            # Direct EOA case
            txs = client.get_normal_txs(address, offset=10)
            if txs.empty:
                return False, "EOA but no transactions found"
            has_outgoing = (txs["from"].str.lower() == address.lower()).any()
            tx_count = len(txs[txs["from"].str.lower() == address.lower()])
            return has_outgoing, f"EOA=True, has_outgoing={has_outgoing}, outgoing_count={tx_count}"

        # Contract case: check if there's a dominant EOA controller
        # (an agent operating through a contract will have a single EOA
        #  that sends most transactions to this contract)
        txs = client.get_normal_txs(address, offset=100)
        if txs.empty:
            return False, "Contract with no normal transactions"

        # Look at incoming transactions (who calls this contract)
        incoming = txs[txs["to"].str.lower() == address.lower()]
        if incoming.empty:
            return False, "Contract with no incoming calls"

        sender_counts = incoming["from"].str.lower().value_counts()
        if sender_counts.empty:
            return False, "Contract with no identifiable callers"

        # Check if the top sender is an EOA (not another contract)
        top_sender = sender_counts.index[0]
        top_sender_ratio = sender_counts.iloc[0] / len(incoming)
        top_sender_is_eoa = not client.is_contract(top_sender)

        # Agent-operated contract: single dominant EOA controller (>50% of calls)
        has_eoa_controller = top_sender_is_eoa and top_sender_ratio > 0.5

        detail = (
            f"is_contract=True, top_sender={top_sender[:12]}..., "
            f"top_sender_ratio={top_sender_ratio:.2f}, "
            f"top_sender_is_eoa={top_sender_is_eoa}, "
            f"has_eoa_controller={has_eoa_controller}"
        )
        return has_eoa_controller, detail
    except Exception as e:
        return False, f"Error: {e}"


def verify_c2(txs: pd.DataFrame) -> tuple[bool, str]:
    """C2: Environmental Perception - Do transaction parameters vary with environment?

    An agent that perceives its environment will produce varying calldata,
    gas prices, and timing patterns that correlate with external conditions.
    """
    if txs.empty or len(txs) < 5:
        return False, f"Too few transactions ({len(txs)})"

    # Check if calldata varies (not all identical)
    unique_inputs = 0
    calldata_varies = False
    if "input" in txs.columns:
        unique_inputs = txs["input"].nunique()
        calldata_varies = unique_inputs > 1

    # Check if gas prices vary (responds to market conditions)
    gas_varies = False
    gas_std = 0.0
    if "gasPrice" in txs.columns:
        gas_prices = pd.to_numeric(txs["gasPrice"], errors="coerce").dropna()
        if len(gas_prices) > 1:
            gas_std = float(gas_prices.std())
            gas_varies = gas_std > 0

    # Check if target addresses vary
    target_varies = False
    unique_targets = 0
    if "to" in txs.columns:
        unique_targets = txs["to"].nunique()
        target_varies = unique_targets > 1

    passed = calldata_varies or gas_varies or target_varies
    detail = (
        f"calldata_unique={unique_inputs}, gas_varies={gas_varies} "
        f"(std={gas_std:.0f}), target_unique={unique_targets}"
    )
    return passed, detail


def verify_c3(txs: pd.DataFrame) -> tuple[bool, str]:
    """C3: Autonomous Decision-Making - Non-deterministic + no human approval.

    Key indicators:
    - Non-deterministic: transaction interval CV > 0.1 (not perfectly periodic)
    - Autonomous: active across all hours (high hour entropy) or burst patterns
    """
    if txs.empty or len(txs) < 10:
        return False, f"Too few transactions ({len(txs)})"

    timestamps = pd.to_numeric(txs["timeStamp"], errors="coerce").dropna().sort_values()
    if len(timestamps) < 10:
        return False, f"Too few valid timestamps ({len(timestamps)})"

    intervals = timestamps.diff().dropna()

    # Non-deterministic: interval CV > 0.1 (not perfectly periodic)
    mean_interval = intervals.mean()
    cv = float(intervals.std() / (mean_interval + 1e-10))
    non_deterministic = cv > 0.1

    # No human approval: active across all hours (high hour entropy)
    hours = pd.to_datetime(timestamps, unit="s").dt.hour
    hour_counts = np.bincount(hours.astype(int), minlength=24).astype(float)
    hour_probs = hour_counts / hour_counts.sum()
    hour_probs_nz = hour_probs[hour_probs > 0]
    hour_entropy = float(-np.sum(hour_probs_nz * np.log2(hour_probs_nz)))
    max_entropy = np.log2(24)
    autonomous = hour_entropy > max_entropy * 0.7  # >70% of max = likely no human

    # Also check burst ratio (rapid-fire transactions impossible for humans)
    burst = float((intervals < 10).mean())  # < 10 seconds apart
    has_bursts = burst > 0.05

    passed = non_deterministic and (autonomous or has_bursts)
    detail = (
        f"CV={cv:.3f}, hour_entropy={hour_entropy:.2f}/{max_entropy:.2f}, "
        f"burst_ratio={burst:.3f}, non_det={non_deterministic}, "
        f"autonomous={autonomous}, has_bursts={has_bursts}"
    )
    return passed, detail


def verify_c4(txs: pd.DataFrame) -> tuple[bool, str]:
    """C4: Adaptiveness - Behavior changes over time.

    Compare first-half vs second-half of transaction history:
    - Gas price distribution shift (KS test)
    - Target contract distribution change
    - Value distribution change
    """
    if txs.empty or len(txs) < 20:
        return False, f"Too few transactions ({len(txs)})"

    mid = len(txs) // 2
    first_half = txs.iloc[:mid]
    second_half = txs.iloc[mid:]

    # Compare gas price distributions (KS test)
    gas_adapted = False
    ks_stat, p_val = 0.0, 1.0
    gp1 = pd.to_numeric(first_half.get("gasPrice", pd.Series()), errors="coerce").dropna()
    gp2 = pd.to_numeric(second_half.get("gasPrice", pd.Series()), errors="coerce").dropna()
    if len(gp1) > 5 and len(gp2) > 5:
        ks_stat, p_val = stats.ks_2samp(gp1, gp2)
        gas_adapted = p_val < 0.05  # Significant difference

    # Compare target contract distributions
    target_changed = False
    if "to" in txs.columns:
        to1 = first_half["to"].value_counts(normalize=True)
        to2 = second_half["to"].value_counts(normalize=True)
        target_changed = set(to1.head(3).index) != set(to2.head(3).index)

    # Compare transaction value distributions
    value_adapted = False
    v1 = pd.to_numeric(first_half.get("value", pd.Series()), errors="coerce").dropna()
    v2 = pd.to_numeric(second_half.get("value", pd.Series()), errors="coerce").dropna()
    if len(v1) > 5 and len(v2) > 5:
        ks_v, p_v = stats.ks_2samp(v1, v2)
        value_adapted = p_v < 0.05

    passed = gas_adapted or target_changed or value_adapted
    detail = (
        f"gas_adapted={gas_adapted} (KS={ks_stat:.3f}, p={p_val:.4f}), "
        f"target_changed={target_changed}, value_adapted={value_adapted}"
    )
    return passed, detail


# ============================================================
# REGISTRY QUERY FUNCTIONS
# ============================================================

def query_autonolas_registry(client: EtherscanClient) -> list[dict]:
    """Query the Autonolas Service Registry for registered agent services.

    Gets transactions TO the registry contract to find agent registrations.
    Also checks internal transactions if normal txs are empty.
    Parses unique 'from' addresses as potential agent operators.
    """
    print("=== Querying Autonolas Service Registry ===")
    print(f"Registry address: {AUTONOLAS_REGISTRY}")

    # Try normal transactions first
    txs = client.get_normal_txs(AUTONOLAS_REGISTRY, offset=500)
    source_type = "normal_txs"

    if txs.empty:
        print("  No normal transactions found, trying internal transactions...")
        try:
            txs = client.get_internal_txs(AUTONOLAS_REGISTRY)
            source_type = "internal_txs"
        except Exception as e:
            print(f"  Error fetching internal txs: {e}")

    if txs.empty:
        print("  No transactions found for registry via either method.")
        # Fall back to querying the related registries
        print("  Falling back to Autonolas Agent Registry and Component Registry...")
        return []

    print(f"  Found {len(txs)} transactions ({source_type}) for registry")

    # Unique 'from' addresses that interacted with the registry
    from_col = "from" if "from" in txs.columns else None
    if from_col is None:
        print("  No 'from' column in transactions")
        return []

    unique_senders = txs[from_col].str.lower().unique().tolist()
    print(f"  Found {len(unique_senders)} unique interacting addresses")

    agents = []
    for addr in unique_senders[:15]:  # Limit to avoid rate limiting
        agents.append({
            "address": addr,
            "source": "autonolas_registry",
            "tx_count_with_registry": int(
                (txs[from_col].str.lower() == addr).sum()
            ),
        })

    return agents


def query_agent_platform_interactions(client: EtherscanClient) -> list[dict]:
    """Query for addresses that interacted with known agent platform contracts."""
    print("\n=== Querying Agent Platform Interactions ===")
    agents = []
    seen_addresses = set()

    for platform_name, contract_addr in AGENT_PLATFORM_CONTRACTS.items():
        if platform_name == "autonolas_registry":
            continue  # Already queried separately

        print(f"  Querying {platform_name}: {contract_addr}")
        try:
            txs = client.get_normal_txs(contract_addr, offset=200)
            if txs.empty:
                print(f"    No transactions found.")
                continue

            unique_senders = txs["from"].str.lower().unique().tolist()
            print(f"    Found {len(unique_senders)} unique senders")

            for addr in unique_senders[:5]:
                if addr not in seen_addresses:
                    seen_addresses.add(addr)
                    agents.append({
                        "address": addr,
                        "source": platform_name,
                    })
        except Exception as e:
            print(f"    Error querying {platform_name}: {e}")

    return agents


# ============================================================
# FULL VALIDATION PIPELINE
# ============================================================

@dataclass
class ValidationResult:
    address: str
    name: str
    entity_type: str
    source: str
    c1_pass: Optional[bool] = None
    c1_detail: str = ""
    c2_pass: Optional[bool] = None
    c2_detail: str = ""
    c3_pass: Optional[bool] = None
    c3_detail: str = ""
    c4_pass: Optional[bool] = None
    c4_detail: str = ""
    all_pass: Optional[bool] = None
    tx_count: int = 0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def validate_address(
    client: EtherscanClient,
    address: str,
    name: str = "",
    entity_type: str = "unknown",
    source: str = "manual",
) -> ValidationResult:
    """Run full C1-C4 validation on a single address."""
    result = ValidationResult(
        address=address, name=name, entity_type=entity_type, source=source
    )

    try:
        # C1: On-chain Actuation
        print(f"  Verifying C1 for {name or address[:12]}...")
        result.c1_pass, result.c1_detail = verify_c1(client, address)

        # Get transactions for C2-C4 (need more data)
        print(f"  Fetching transactions for C2-C4...")
        txs = client.get_normal_txs(address, offset=1000)
        result.tx_count = len(txs) if not txs.empty else 0

        if not txs.empty:
            # Filter to outgoing transactions for behavior analysis
            outgoing = txs[txs["from"].str.lower() == address.lower()]
            if len(outgoing) < 5:
                # Use all txs if too few outgoing
                outgoing = txs

            # C2: Environmental Perception
            print(f"  Verifying C2...")
            result.c2_pass, result.c2_detail = verify_c2(outgoing)

            # C3: Autonomous Decision-Making
            print(f"  Verifying C3...")
            result.c3_pass, result.c3_detail = verify_c3(outgoing)

            # C4: Adaptiveness
            print(f"  Verifying C4...")
            result.c4_pass, result.c4_detail = verify_c4(outgoing)
        else:
            result.c2_pass = False
            result.c2_detail = "No transactions available"
            result.c3_pass = False
            result.c3_detail = "No transactions available"
            result.c4_pass = False
            result.c4_detail = "No transactions available"

        result.all_pass = all([
            result.c1_pass, result.c2_pass, result.c3_pass, result.c4_pass
        ])

    except Exception as e:
        result.error = f"{type(e).__name__}: {e}"
        traceback.print_exc()

    return result


def run_validation():
    """Run the full on-chain validation pipeline."""
    print("=" * 70)
    print("ON-CHAIN REGISTRY VALIDATION OF C1-C4 AI AGENT DEFINITION")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    client = EtherscanClient()
    print(f"EtherscanClient initialized with {client.num_keys} API keys\n")

    all_results: list[ValidationResult] = []

    # ----------------------------------------------------------
    # 1. Query Autonolas Registry
    # ----------------------------------------------------------
    registry_agents = query_autonolas_registry(client)

    # Validate a sample of registry-discovered agents
    print(f"\n=== Validating Autonolas Registry Agents (sample of {min(8, len(registry_agents))}) ===")
    registry_results = []
    for agent_info in registry_agents[:8]:
        addr = agent_info["address"]
        print(f"\n  [{addr[:10]}...] from {agent_info['source']}")
        result = validate_address(
            client, addr,
            name=f"Autonolas agent ({addr[:10]})",
            entity_type="registry_agent",
            source="autonolas_registry",
        )
        registry_results.append(result)
        all_results.append(result)
        print(f"    C1={result.c1_pass}, C2={result.c2_pass}, "
              f"C3={result.c3_pass}, C4={result.c4_pass}, ALL={result.all_pass}")

    # ----------------------------------------------------------
    # 2. Query Agent Platform Interactions
    # ----------------------------------------------------------
    platform_agents = query_agent_platform_interactions(client)

    print(f"\n=== Validating Platform-Interacting Agents (sample of {min(5, len(platform_agents))}) ===")
    platform_results = []
    for agent_info in platform_agents[:5]:
        addr = agent_info["address"]
        print(f"\n  [{addr[:10]}...] from {agent_info['source']}")
        result = validate_address(
            client, addr,
            name=f"Platform agent ({agent_info['source']})",
            entity_type="platform_agent",
            source=agent_info["source"],
        )
        platform_results.append(result)
        all_results.append(result)
        print(f"    C1={result.c1_pass}, C2={result.c2_pass}, "
              f"C3={result.c3_pass}, C4={result.c4_pass}, ALL={result.all_pass}")

    # ----------------------------------------------------------
    # 3. Validate Known Agents (positive ground truth)
    # ----------------------------------------------------------
    print("\n=== Validating Known Agents (positive ground truth) ===")
    known_agent_results = []
    for addr, info in KNOWN_AGENTS.items():
        print(f"\n  [{info['name']}] ({info['type']})")
        result = validate_address(
            client, addr,
            name=info["name"],
            entity_type=info["type"],
            source="known_agent",
        )
        known_agent_results.append(result)
        all_results.append(result)
        expected_all = all(info["expected"].values())
        match = result.all_pass == expected_all
        print(f"    C1={result.c1_pass}, C2={result.c2_pass}, "
              f"C3={result.c3_pass}, C4={result.c4_pass}, ALL={result.all_pass} "
              f"(expected={expected_all}, match={'OK' if match else 'MISMATCH'})")

    # ----------------------------------------------------------
    # 4. Validate Known Non-Agents (negative ground truth)
    # ----------------------------------------------------------
    print("\n=== Validating Known Non-Agents (negative ground truth) ===")
    non_agent_results = []
    for addr, info in NON_AGENTS.items():
        print(f"\n  [{info['name']}] ({info['type']}) - should fail {info['expected_fail']}")
        result = validate_address(
            client, addr,
            name=info["name"],
            entity_type=info["type"],
            source="known_non_agent",
        )
        non_agent_results.append(result)
        all_results.append(result)

        # Check if the expected condition fails
        expected_fail = info["expected_fail"]
        if "or" in expected_fail.lower():
            # Handle C1_or_C3 case (e.g., human wallets that may be smart contracts)
            conditions = [c.strip() for c in expected_fail.split("_or_")]
            actual_fail = any(
                not getattr(result, f"{c.lower()}_pass", True)
                for c in conditions
            )
            failed_which = [
                c for c in conditions
                if not getattr(result, f"{c.lower()}_pass", True)
            ]
            print(f"    C1={result.c1_pass}, C2={result.c2_pass}, "
                  f"C3={result.c3_pass}, C4={result.c4_pass}, ALL={result.all_pass} "
                  f"(expected any of {conditions}=False, "
                  f"failed={failed_which}, {'OK' if actual_fail else 'UNEXPECTED'})")
        else:
            actual_fail = not getattr(result, f"{expected_fail.lower()}_pass", True)
            print(f"    C1={result.c1_pass}, C2={result.c2_pass}, "
                  f"C3={result.c3_pass}, C4={result.c4_pass}, ALL={result.all_pass} "
                  f"(expected {expected_fail}=False, actual={actual_fail}, "
                  f"{'OK' if actual_fail else 'UNEXPECTED'})")

    # ----------------------------------------------------------
    # 5. Counterexample Search
    # ----------------------------------------------------------
    print("\n=== Counterexample Search ===")
    counterexamples = search_counterexamples(all_results)

    # ----------------------------------------------------------
    # 6. Ablation Experiment
    # ----------------------------------------------------------
    print("\n=== Ablation Experiment ===")
    ablation = run_ablation(all_results)

    # ----------------------------------------------------------
    # 7. Produce Validation Report
    # ----------------------------------------------------------
    report = produce_report(
        registry_results=registry_results,
        platform_results=platform_results,
        known_agent_results=known_agent_results,
        non_agent_results=non_agent_results,
        counterexamples=counterexamples,
        ablation=ablation,
        all_results=all_results,
    )

    return report


# ============================================================
# COUNTEREXAMPLE SEARCH
# ============================================================

def search_counterexamples(all_results: list[ValidationResult]) -> dict:
    """Search for entities that break the definition.

    Type A (definition too broad): Satisfies C1-C4 but intuitively NOT an agent
    Type B (definition too narrow): Fails C1-C4 but intuitively IS an agent
    """
    counterexamples = {
        "type_a_too_broad": [],
        "type_b_too_narrow": [],
        "discussion": [],
    }

    for r in all_results:
        # Type A: Passes all C1-C4 but is a known non-agent
        if r.source == "known_non_agent" and r.all_pass:
            counterexamples["type_a_too_broad"].append({
                "address": r.address,
                "name": r.name,
                "type": r.entity_type,
                "detail": "Passes all C1-C4 but is a known non-agent",
                "c1": r.c1_detail,
                "c2": r.c2_detail,
                "c3": r.c3_detail,
                "c4": r.c4_detail,
            })

        # Type B: Fails C1-C4 but is a known agent
        if r.source == "known_agent" and not r.all_pass:
            failed_conditions = []
            if not r.c1_pass:
                failed_conditions.append("C1")
            if not r.c2_pass:
                failed_conditions.append("C2")
            if not r.c3_pass:
                failed_conditions.append("C3")
            if not r.c4_pass:
                failed_conditions.append("C4")
            counterexamples["type_b_too_narrow"].append({
                "address": r.address,
                "name": r.name,
                "type": r.entity_type,
                "failed_conditions": failed_conditions,
                "detail": f"Known agent but fails {', '.join(failed_conditions)}",
                "c1": r.c1_detail,
                "c2": r.c2_detail,
                "c3": r.c3_detail,
                "c4": r.c4_detail,
            })

    # Add discussion for edge cases
    for r in all_results:
        if r.source == "autonolas_registry" and r.all_pass is False:
            counterexamples["discussion"].append({
                "address": r.address,
                "name": r.name,
                "note": (
                    "Registry-listed but fails C1-C4. This could mean: "
                    "(1) the address is a service owner, not the agent itself, "
                    "(2) the agent operates through a different EOA, or "
                    "(3) the registry includes non-agent services."
                ),
                "failed": [
                    c for c in ["C1", "C2", "C3", "C4"]
                    if not getattr(r, f"{c.lower()}_pass", True)
                ],
            })

    # Add interpretive discussion for Type A counterexamples
    for ce in counterexamples["type_a_too_broad"]:
        if "exchange" in ce.get("type", "").lower() or "exchange" in ce.get("name", "").lower():
            ce["interpretation"] = (
                "Exchange hot wallets are a known edge case. They are heavily "
                "automated (passing C3) and adapt to market conditions (passing C4), "
                "but are institutional infrastructure, not autonomous agents. "
                "This suggests C1-C4 may need an additional condition (C5) "
                "distinguishing institutional automation from agent autonomy, "
                "or the C3 threshold needs adjustment to require higher entropy "
                "or more distinctive non-human patterns."
            )

    # Add interpretive discussion for Type B counterexamples
    for ce in counterexamples["type_b_too_narrow"]:
        if "contract" in ce.get("type", "").lower():
            ce["interpretation"] = (
                "This agent operates through a smart contract rather than an EOA. "
                "C1's definition says 'directly or indirectly controls an EOA'. "
                "For multi-operator contracts, identifying the controlling EOA is "
                "harder, but the agent clearly exists. This suggests C1 verification "
                "needs refinement for contract-based agent architectures."
            )

    type_a_count = len(counterexamples["type_a_too_broad"])
    type_b_count = len(counterexamples["type_b_too_narrow"])
    print(f"  Type A (too broad - non-agent passes): {type_a_count}")
    print(f"  Type B (too narrow - agent fails):     {type_b_count}")
    print(f"  Discussion items:                       {len(counterexamples['discussion'])}")

    return counterexamples


# ============================================================
# ABLATION EXPERIMENT
# ============================================================

def run_ablation(all_results: list[ValidationResult]) -> dict:
    """Show what happens when each condition is removed.

    For each ablation, count how many known non-agents would be
    incorrectly classified as agents (false positives).
    """
    non_agents = [r for r in all_results if r.source == "known_non_agent"]
    known_agents = [r for r in all_results if r.source == "known_agent"]

    ablation_results = {}

    conditions = {
        "C1": "On-chain Actuation (includes off-chain-only systems)",
        "C2": "Environmental Perception (includes blind cron jobs)",
        "C3": "Autonomous Decision-Making (includes human wallets & deterministic bots)",
        "C4": "Adaptiveness (includes fixed-rule bots that never learn)",
    }

    for removed_c, description in conditions.items():
        # Without this condition, how many non-agents would pass?
        remaining = [c for c in ["c1", "c2", "c3", "c4"] if c != removed_c.lower()]

        false_positives = []
        for r in non_agents:
            passes_remaining = all(
                getattr(r, f"{c}_pass", False) for c in remaining
            )
            if passes_remaining:
                false_positives.append({
                    "address": r.address,
                    "name": r.name,
                    "type": r.entity_type,
                })

        # Also check: how many known agents would still pass?
        true_positives_lost = 0
        for r in known_agents:
            passes_remaining = all(
                getattr(r, f"{c}_pass", False) for c in remaining
            )
            if not passes_remaining:
                true_positives_lost += 1

        ablation_results[removed_c] = {
            "removed_condition": removed_c,
            "description": description,
            "false_positives_count": len(false_positives),
            "false_positives": false_positives,
            "true_positives_retained": len(known_agents) - true_positives_lost,
            "total_known_agents": len(known_agents),
        }

        print(f"  Without {removed_c} ({description}):")
        print(f"    False positives: {len(false_positives)} non-agents incorrectly pass")
        if false_positives:
            for fp in false_positives:
                print(f"      - {fp['name']} ({fp['type']})")
        print(f"    True positives retained: "
              f"{len(known_agents) - true_positives_lost}/{len(known_agents)}")

    return ablation_results


# ============================================================
# REPORT GENERATION
# ============================================================

def produce_report(
    registry_results: list[ValidationResult],
    platform_results: list[ValidationResult],
    known_agent_results: list[ValidationResult],
    non_agent_results: list[ValidationResult],
    counterexamples: dict,
    ablation: dict,
    all_results: list[ValidationResult],
) -> dict:
    """Produce structured validation report."""

    def pass_rate(results: list[ValidationResult], field: str) -> float:
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
        return sum(vals) / len(vals) * 100 if vals else 0.0

    def pass_count(results: list[ValidationResult], field: str) -> tuple[int, int]:
        vals = [getattr(r, field) for r in results if getattr(r, field) is not None]
        return sum(vals), len(vals)

    # --- Section 1: Autonolas Registry Agents ---
    reg_count = len(registry_results)
    reg_c1 = pass_rate(registry_results, "c1_pass")
    reg_c2 = pass_rate(registry_results, "c2_pass")
    reg_c3 = pass_rate(registry_results, "c3_pass")
    reg_c4 = pass_rate(registry_results, "c4_pass")
    reg_all = pass_rate(registry_results, "all_pass")

    # --- Section 2: Known Agents ---
    ka_c1 = pass_rate(known_agent_results, "c1_pass")
    ka_c2 = pass_rate(known_agent_results, "c2_pass")
    ka_c3 = pass_rate(known_agent_results, "c3_pass")
    ka_c4 = pass_rate(known_agent_results, "c4_pass")
    ka_all = pass_rate(known_agent_results, "all_pass")

    # --- Section 3: Non-Agents ---
    contracts = [r for r in non_agent_results if r.entity_type == "contract"]
    humans = [r for r in non_agent_results if r.entity_type in ("human", "exchange_wallet")]

    contracts_fail_c1 = all(not r.c1_pass for r in contracts) if contracts else True
    # Humans should fail C1 or C3 (some use smart contract wallets now)
    humans_correctly_rejected = all(
        not r.all_pass for r in humans
    ) if humans else True
    humans_fail_c3_or_c1 = all(
        (not r.c3_pass or not r.c1_pass) for r in humans
    ) if humans else True

    # --- Determine conclusion ---
    type_a_count = len(counterexamples["type_a_too_broad"])
    type_b_count = len(counterexamples["type_b_too_narrow"])

    if type_a_count == 0 and type_b_count == 0:
        conclusion = "VALIDATED: C1-C4 definition correctly classifies all tested entities"
    elif type_a_count > 0 and type_b_count > 0:
        conclusion = (
            f"NEEDS ADJUSTMENT: {type_a_count} false positive(s) and "
            f"{type_b_count} false negative(s) found"
        )
    elif type_a_count > 0:
        conclusion = f"PARTIALLY VALIDATED: definition may be too broad ({type_a_count} false positive(s))"
    else:
        conclusion = f"PARTIALLY VALIDATED: definition may be too narrow ({type_b_count} false negative(s))"

    report = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "method": "on-chain registry validation",
            "etherscan_api": "V2",
            "total_addresses_tested": len(all_results),
        },
        "section_1_autonolas_registry": {
            "agents_found": reg_count,
            "c1_pass_rate": round(reg_c1, 1),
            "c2_pass_rate": round(reg_c2, 1),
            "c3_pass_rate": round(reg_c3, 1),
            "c4_pass_rate": round(reg_c4, 1),
            "all_c1_c4_pass_rate": round(reg_all, 1),
            "details": [r.to_dict() for r in registry_results],
        },
        "section_2_platform_agents": {
            "agents_found": len(platform_results),
            "c1_pass_rate": round(pass_rate(platform_results, "c1_pass"), 1),
            "c2_pass_rate": round(pass_rate(platform_results, "c2_pass"), 1),
            "c3_pass_rate": round(pass_rate(platform_results, "c3_pass"), 1),
            "c4_pass_rate": round(pass_rate(platform_results, "c4_pass"), 1),
            "all_c1_c4_pass_rate": round(pass_rate(platform_results, "all_pass"), 1),
            "details": [r.to_dict() for r in platform_results],
        },
        "section_3_known_agents": {
            "count": len(known_agent_results),
            "c1_pass_rate": round(ka_c1, 1),
            "c2_pass_rate": round(ka_c2, 1),
            "c3_pass_rate": round(ka_c3, 1),
            "c4_pass_rate": round(ka_c4, 1),
            "all_c1_c4_pass_rate": round(ka_all, 1),
            "details": [r.to_dict() for r in known_agent_results],
        },
        "section_4_known_non_agents": {
            "contracts_all_fail_c1": contracts_fail_c1,
            "humans_correctly_rejected": humans_correctly_rejected,
            "humans_fail_c3_or_c1": humans_fail_c3_or_c1,
            "note": (
                "Some human wallets (vitalik.eth) have migrated to smart contract "
                "wallets (ERC-4337), causing them to fail C1 instead of C3. "
                "This is actually correct: the definition still rejects them."
            ),
            "contract_details": [r.to_dict() for r in contracts],
            "human_details": [r.to_dict() for r in humans],
        },
        "section_5_counterexamples": counterexamples,
        "section_6_ablation": ablation,
        "conclusion": conclusion,
    }

    # Print text report
    print("\n" + "=" * 70)
    print("=== C1-C4 On-Chain Validation Report ===")
    print("=" * 70)

    print(f"\n1. Autonolas Registry Agents:")
    print(f"   - {reg_count} addresses tested from registry")
    print(f"   - C1 pass rate: {reg_c1:.1f}%")
    print(f"   - C2 pass rate: {reg_c2:.1f}%")
    print(f"   - C3 pass rate: {reg_c3:.1f}%")
    print(f"   - C4 pass rate: {reg_c4:.1f}%")
    print(f"   - All C1-C4 pass rate: {reg_all:.1f}%")

    print(f"\n2. Known Agents (positive ground truth):")
    print(f"   - {len(known_agent_results)} agents tested")
    print(f"   - C1 pass rate: {ka_c1:.1f}%")
    print(f"   - C2 pass rate: {ka_c2:.1f}%")
    print(f"   - C3 pass rate: {ka_c3:.1f}%")
    print(f"   - C4 pass rate: {ka_c4:.1f}%")
    print(f"   - All C1-C4 pass rate: {ka_all:.1f}%")

    print(f"\n3. Known Non-Agents (negative ground truth):")
    print(f"   - Contracts: all fail C1 {'PASS' if contracts_fail_c1 else 'FAIL'}")
    print(f"   - Humans: all correctly rejected (fail C1 or C3) "
          f"{'PASS' if humans_correctly_rejected else 'FAIL'}")

    print(f"\n4. Counterexamples Found:")
    print(f"   - Type A (too broad): {type_a_count}")
    print(f"   - Type B (too narrow): {type_b_count}")
    if counterexamples["type_a_too_broad"]:
        for ce in counterexamples["type_a_too_broad"]:
            print(f"     - {ce['name']}: passes all C1-C4 but is non-agent")
    if counterexamples["type_b_too_narrow"]:
        for ce in counterexamples["type_b_too_narrow"]:
            print(f"     - {ce['name']}: known agent but fails {ce['failed_conditions']}")
    if counterexamples["discussion"]:
        print(f"   - Discussion items: {len(counterexamples['discussion'])}")
        for d in counterexamples["discussion"][:3]:
            print(f"     - {d['address'][:12]}...: fails {d['failed']}")

    print(f"\n5. Ablation Results:")
    for c, data in ablation.items():
        fp = data["false_positives_count"]
        tp = data["true_positives_retained"]
        total = data["total_known_agents"]
        print(f"   - Without {c}: {fp} false positives, {tp}/{total} true positives retained")

    print(f"\n6. Conclusion: {conclusion}")
    print("=" * 70)

    return report


# ============================================================
# MAIN
# ============================================================

def main():
    report = run_validation()

    # Save results
    output_path = os.path.join(
        os.path.dirname(__file__),
        "onchain_validation_results.json",
    )
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return report


if __name__ == "__main__":
    main()
