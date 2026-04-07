"""
Configurable Ground Truth Labeling
====================================
Labels addresses as AGENT / NOT_AGENT / EXCLUDE based on
Definition 1 (C1-C4) from Paper 0.

After Delphi expert validation, update the criteria here
and re-run all experiments via run_full_pipeline.py.

C1: On-chain Actuation (must be EOA with outgoing txs)
C2: Environmental Perception (tx params vary with environment)
C3: Autonomous Decision-Making (non-deterministic + no human approval)
C4: Adaptiveness (behavior changes over time)

Boundary table (from shared/definition.md):
  - Human user:        C1 yes, C2 yes, C3 NO  (human approval), C4 yes  -> NOT_AGENT
  - Deterministic bot:  C1 yes, C2 no,  C3 NO  (deterministic),  C4 no   -> NOT_AGENT
  - Simple trading bot: C1 yes, C2 yes, C3 NO  (deterministic),  C4 no   -> NOT_AGENT
  - Smart contract:     C1 NO  (passive),                                 -> EXCLUDE
  - Exchange wallet:    C1 yes, C2 yes, C3 ambiguous,            C4 no   -> EXCLUDE
  - MEV Searcher:       C1 yes, C2 yes, C3 yes, C4 yes                   -> AGENT
  - RL Trading Agent:   C1 yes, C2 yes, C3 yes, C4 yes                   -> AGENT
  - DeFi Mgmt Agent:    C1 yes, C2 yes, C3 yes, C4 yes                   -> AGENT
  - LLM-driven Agent:   C1 yes, C2 yes, C3 yes, C4 yes                   -> AGENT
  - Cross-chain relayer: C1 yes, C2 yes, C3 partial, C4 partial          -> BOUNDARY
"""

from enum import Enum
from typing import Optional


# ============================================================
# LABEL ENUM
# ============================================================

class C1C4Label(str, Enum):
    """Ground-truth label derived from C1-C4 assessment."""
    AGENT = "AGENT"           # Satisfies all C1+C2+C3+C4
    NOT_AGENT = "NOT_AGENT"   # Fails at least one of C3/C4
    EXCLUDE = "EXCLUDE"       # Ambiguous or not applicable (contracts, exchanges)
    BOUNDARY = "BOUNDARY"     # Expert disagreement expected


# ============================================================
# ADDRESS LISTS ORGANIZED BY C1-C4 SATISFACTION
# POST-DELPHI: Update these based on expert consensus
# ============================================================

SATISFIES_ALL_C1C4: dict[str, str] = {
    # ---- MEV Searchers (C1+C2+C3+C4 all satisfied) ----
    # C3: non-deterministic strategy selection under competition
    # C4: strategy adaptation to MEV landscape changes
    "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9": "jaredfromsubway.eth -- MEV sandwich bot",
    "0xae2Fc483527B8EF99EB5D9B44875F005ba1FaE13": "jaredfromsubway v2",
    "0x000000000000084e91743124a982076C59f10084": "MEV multicall bot",
    "0x000000000000Ad05Ccc4F10045630fb830B95127": "MEV bot (Flashbots)",
    "0x000000000000cd17345801aa8147b8D3950260FF": "MEV bot (generalized)",
    "0x00000000003b3cc22aF3aE1EAc0440BcEe416B40": "MEV bot (sandwich)",
    "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80": "MEV bot (known sandwich)",

    # ---- Algorithmic Market Makers (C1+C2+C3+C4) ----
    # C3: autonomous pricing + inventory management
    # C4: spread/inventory parameters adapt to volatility regime
    "0xA69babEF1cA67A37Ffaf7a485DfFF3382056e78C": "Wintermute (market maker)",
    "0x280027dd00eE0050d3F9d168EFD6B40090009246": "Wintermute 2",
    "0xDBF5E9c5206d0dB70a90108bf936DA60221dC080": "Wintermute 3",

    # ---- DeFi Management Agents (Autonolas / OLAS) ----
    # POST-DELPHI: Add confirmed Autonolas agent operator EOAs here

    # ---- LLM-Driven Agents (AI16Z, Virtuals Protocol) ----
    # POST-DELPHI: Add confirmed LLM agent EOAs here
}

FAILS_C3_DETERMINISTIC: dict[str, str] = {
    # Satisfies C1+C2 but NOT C3 (purely deterministic rules, no adaptation)
    # = Simple bots, NOT agents
    # POST-DELPHI: Move addresses here if experts determine they are
    # deterministic rule-followers without non-deterministic components
}

FAILS_C3_HUMAN: dict[str, str] = {
    # Satisfies C1+C2 but NOT C3 (human approves each tx)
    # = Human-operated wallets
    "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045": "vitalik.eth",
    "0xAb5801a7D398351b8bE11C439e05C5B3259aeC9B": "Vitalik older address",
    "0x220866B1A2219f40e72f5c628B65D54268cA3A9D": "hayden.eth (Uniswap founder)",
    "0x983110309620D911731Ac0932219af06091b6744": "brantly.eth (ENS)",
    "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb": "ENS verified human",
    "0xDA9dfA130Df4dE4673b89022EE50ff26f6EA73Cf": "ENS verified human",
    "0x3328F7f4A1D1C57c35df56bBf0c9dCAFCA309C49": "ENS verified human",
    "0x3DdfA8eC3052539b6C9549F12cEA2C295cfF5296": "ENS verified human",
    "0x51C72848c68a965f66FA7a88855F9f7784502a7F": "ENS verified human",
    "0x8103683202aa8DA10536036EDef04CDd865C225E": "ENS verified human",
    "0xE8c060F8052E07423f71D445277c61AC5138A2e5": "ENS verified human",
    "0x5f350bF5feE8e254D6077f8661E9C7b83a30364e": "ENS verified human",
}

SMART_CONTRACTS: dict[str, str] = {
    # Fails C1 (not an EOA -- passive, invoked by others)
    # = Protocols, routers, AMMs -- EXCLUDE from study
    "0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D": "Uniswap V2 Router",
    "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45": "Uniswap V3 Router 02",
    "0x1111111254EEB25477B68fb85Ed929f73A960582": "1inch V5 Router",
    "0xDef1C0ded9bec7F1a1670819833240f027b25EfF": "0x Exchange Proxy",
    "0x00000000006c3852cbEf3e08E8dF289169EdE581": "OpenSea Seaport",
}

EXCHANGE_WALLETS: dict[str, str] = {
    # Ambiguous C3 (mix of human + automated operations)
    # = EXCLUDE from training
    "0x28C6c06298d514Db089934071355E5743bf21d60": "Binance 14 hot wallet",
    "0xBE0eB53F46cd790Cd13851d5EFf43D12404d33E8": "Binance cold wallet",
    "0xF977814e90dA44bFA03b6295A0616a897441aceC": "Binance 8",
    "0x47ac0Fb4F2D84898e4D9E7b4DaB3C24507a6D503": "Binance cold 2",
    "0xDFd5293D8e347dFe59E90eFd55b2956a1343963d": "Binance 16",
    "0x2FAF487A4414Fe77e2327F0bf4AE2a264a776AD2": "Gemini 4",
    "0xFa4FC4ec2F81A4897743C5b4f45907c02CE06199": "Bitfinex hot",
    "0x267be1C1D684F78cb4F6a176C4911b741E4Ffdc0": "Kraken 4",
    "0x75e89d5979E4f6Fba9F97c104c2F0AFB3F1dcB88": "MEXC hot wallet",
    "0x2B888954421b424C5D3D9Ce9bB67c9bD47537d12": "FTX (pre-collapse)",
    "0xB1AdceddB2941033a090dD166a462fe1c2029484": "DWF Labs hot",
    "0x3FAB184622Dc19b6109349B94811493BF2a45362": "Exchange wallet",
    "0xC098B2a3Aa256D2140208C3de6543aAEf5cd3A94": "Exchange wallet",
    "0x176F3DAb24a159341c0509bB36B833E7fdd0a132": "Exchange wallet",
}

BOUNDARY_CASES: dict[str, str] = {
    # Expert disagreement expected on C3/C4 satisfaction
    # = EXCLUDE from training, include in qualitative analysis
    # POST-DELPHI: Reclassify based on expert consensus
    # e.g., cross-chain relayers, oracle bots, liquidation bots
}


# ============================================================
# LABEL RESOLUTION
# ============================================================

def get_label(address: str) -> C1C4Label:
    """Return the C1-C4 based label for a given address.

    Priority order:
      1. EXCLUDE if in SMART_CONTRACTS or EXCHANGE_WALLETS
      2. BOUNDARY if in BOUNDARY_CASES
      3. AGENT if in SATISFIES_ALL_C1C4
      4. NOT_AGENT if in FAILS_C3_DETERMINISTIC or FAILS_C3_HUMAN
      5. EXCLUDE (unknown address)

    Args:
        address: Ethereum address (any case).

    Returns:
        C1C4Label enum value.
    """
    addr = address.strip()

    # Normalize for lookup: check both original and checksummed
    for registry_addr in SMART_CONTRACTS:
        if addr.lower() == registry_addr.lower():
            return C1C4Label.EXCLUDE
    for registry_addr in EXCHANGE_WALLETS:
        if addr.lower() == registry_addr.lower():
            return C1C4Label.EXCLUDE
    for registry_addr in BOUNDARY_CASES:
        if addr.lower() == registry_addr.lower():
            return C1C4Label.BOUNDARY
    for registry_addr in SATISFIES_ALL_C1C4:
        if addr.lower() == registry_addr.lower():
            return C1C4Label.AGENT
    for registry_addr in FAILS_C3_DETERMINISTIC:
        if addr.lower() == registry_addr.lower():
            return C1C4Label.NOT_AGENT
    for registry_addr in FAILS_C3_HUMAN:
        if addr.lower() == registry_addr.lower():
            return C1C4Label.NOT_AGENT

    return C1C4Label.EXCLUDE  # Unknown address -> exclude


def get_all_labeled_addresses() -> dict[str, C1C4Label]:
    """Return a dict mapping every configured address to its label.

    Returns:
        Dictionary {address: C1C4Label}.
    """
    result: dict[str, C1C4Label] = {}
    for addr in SATISFIES_ALL_C1C4:
        result[addr] = C1C4Label.AGENT
    for addr in FAILS_C3_DETERMINISTIC:
        result[addr] = C1C4Label.NOT_AGENT
    for addr in FAILS_C3_HUMAN:
        result[addr] = C1C4Label.NOT_AGENT
    for addr in SMART_CONTRACTS:
        result[addr] = C1C4Label.EXCLUDE
    for addr in EXCHANGE_WALLETS:
        result[addr] = C1C4Label.EXCLUDE
    for addr in BOUNDARY_CASES:
        result[addr] = C1C4Label.BOUNDARY
    return result


def get_training_addresses() -> tuple[list[str], list[int]]:
    """Return addresses and binary labels for model training.

    Only includes AGENT (label=1) and NOT_AGENT (label=0).
    EXCLUDE and BOUNDARY addresses are filtered out.

    Returns:
        Tuple of (addresses_list, labels_list) where labels are 0/1.
    """
    all_labeled = get_all_labeled_addresses()
    addresses: list[str] = []
    labels: list[int] = []
    for addr, lbl in all_labeled.items():
        if lbl == C1C4Label.AGENT:
            addresses.append(addr)
            labels.append(1)
        elif lbl == C1C4Label.NOT_AGENT:
            addresses.append(addr)
            labels.append(0)
        # EXCLUDE and BOUNDARY are skipped
    return addresses, labels


def get_address_name(address: str) -> Optional[str]:
    """Look up a human-readable name for an address.

    Args:
        address: Ethereum address.

    Returns:
        Name string or None if not found.
    """
    addr_lower = address.lower()
    for registry in [SATISFIES_ALL_C1C4, FAILS_C3_DETERMINISTIC,
                     FAILS_C3_HUMAN, SMART_CONTRACTS,
                     EXCHANGE_WALLETS, BOUNDARY_CASES]:
        for reg_addr, name in registry.items():
            if reg_addr.lower() == addr_lower:
                return name
    return None


def summary() -> dict[str, int]:
    """Print a summary of the labeling configuration.

    Returns:
        Dictionary with counts per label category.
    """
    counts = {
        "AGENT (C1+C2+C3+C4)": len(SATISFIES_ALL_C1C4),
        "NOT_AGENT (fails C3, deterministic)": len(FAILS_C3_DETERMINISTIC),
        "NOT_AGENT (fails C3, human)": len(FAILS_C3_HUMAN),
        "EXCLUDE (smart contracts)": len(SMART_CONTRACTS),
        "EXCLUDE (exchange wallets)": len(EXCHANGE_WALLETS),
        "BOUNDARY (expert disagreement)": len(BOUNDARY_CASES),
    }
    total_trainable = (len(SATISFIES_ALL_C1C4)
                       + len(FAILS_C3_DETERMINISTIC)
                       + len(FAILS_C3_HUMAN))
    counts["TOTAL trainable (AGENT + NOT_AGENT)"] = total_trainable
    counts["TOTAL all addresses"] = sum(
        len(d) for d in [SATISFIES_ALL_C1C4, FAILS_C3_DETERMINISTIC,
                         FAILS_C3_HUMAN, SMART_CONTRACTS,
                         EXCHANGE_WALLETS, BOUNDARY_CASES]
    )
    return counts


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Labeling Configuration Summary (C1-C4)")
    print("=" * 60)
    for category, count in summary().items():
        print(f"  {category:45s}: {count}")

    print("\n--- Training Set ---")
    addrs, labels = get_training_addresses()
    print(f"  Agents:     {sum(labels)}")
    print(f"  Not-agents: {len(labels) - sum(labels)}")
    print(f"  Total:      {len(labels)}")

    print("\n--- All Labeled Addresses ---")
    for addr, lbl in get_all_labeled_addresses().items():
        name = get_address_name(addr) or "?"
        print(f"  {addr}  {lbl.value:12s}  {name}")
