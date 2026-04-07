"""
Paper 1: Security audit on 22 identified agent addresses.

Analyzes raw transaction data (already saved as Parquet) to extract:
- ERC20 approval counts and unlimited approval detection
- Transaction revert rates
- Interaction network statistics (unique counterparties, concentration)
- Gas usage patterns relevant to security
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_PATH = DATA_DIR / "features.parquet"
RESULTS_PATH = Path(__file__).parent / "security_audit_results.json"

# ERC20 approve method signature: approve(address,uint256) = 0x095ea7b3
APPROVE_METHOD_ID = "0x095ea7b3"
# MaxUint256 in hex (unlimited approval)
MAX_UINT256 = "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
# increaseAllowance(address,uint256) = 0x39509351
INCREASE_ALLOWANCE_METHOD_ID = "0x39509351"


def parse_approve_amount(input_data: str) -> tuple[bool, str]:
    """Parse an approve() call to check if it's unlimited.

    Returns (is_unlimited, spender_address).
    """
    if not isinstance(input_data, str) or len(input_data) < 138:
        return False, ""

    # input: 0x + 8 chars method + 64 chars address + 64 chars amount
    try:
        spender = "0x" + input_data[34:74]
        amount_hex = input_data[74:138].lower()
        is_unlimited = amount_hex == MAX_UINT256 or int(amount_hex, 16) > 2**200
        return is_unlimited, spender
    except (ValueError, IndexError):
        return False, ""


def analyze_address(address: str, raw_path: Path) -> dict:
    """Run security audit on a single address."""
    df = pd.read_parquet(raw_path)

    if df.empty:
        return {
            "address": address,
            "total_txs": 0,
            "error": "No transaction data",
        }

    n_total = len(df)

    # Ensure proper types
    if "isError" in df.columns:
        df["isError"] = pd.to_numeric(df["isError"], errors="coerce").fillna(0).astype(int)
    else:
        df["isError"] = 0

    if "input" not in df.columns:
        df["input"] = ""
    if "methodId" not in df.columns:
        df["methodId"] = ""
    if "from" not in df.columns:
        df["from"] = ""
    if "to" not in df.columns:
        df["to"] = ""

    # ── 1) Revert rate ────────────────────────────────────────────
    n_errors = int(df["isError"].sum())
    revert_rate = n_errors / n_total if n_total > 0 else 0.0

    # ── 2) ERC20 Approvals ────────────────────────────────────────
    addr_lower = address.lower()
    # Outgoing txs from this address
    outgoing = df[df["from"].str.lower() == addr_lower]
    n_outgoing = len(outgoing)

    # Find approve calls
    approve_mask = outgoing["methodId"].str.lower() == APPROVE_METHOD_ID
    # Also check input data starts with approve signature
    approve_mask2 = outgoing["input"].str[:10].str.lower() == APPROVE_METHOD_ID
    approve_txs = outgoing[approve_mask | approve_mask2]
    n_approvals = len(approve_txs)

    # Check for unlimited approvals
    n_unlimited = 0
    spenders_unlimited = set()
    for _, row in approve_txs.iterrows():
        is_unlimited, spender = parse_approve_amount(str(row.get("input", "")))
        if is_unlimited:
            n_unlimited += 1
            if spender:
                spenders_unlimited.add(spender)

    # Also check increaseAllowance
    increase_mask = outgoing["methodId"].str.lower() == INCREASE_ALLOWANCE_METHOD_ID
    increase_mask2 = outgoing["input"].str[:10].str.lower() == INCREASE_ALLOWANCE_METHOD_ID
    n_increase_allowance = int((increase_mask | increase_mask2).sum())

    # ── 3) Interaction network stats ──────────────────────────────
    # Unique counterparties (addresses interacted with)
    all_froms = df["from"].str.lower().unique()
    all_tos = df["to"].str.lower().dropna().unique()
    all_counterparties = set(all_froms) | set(all_tos)
    all_counterparties.discard(addr_lower)
    all_counterparties.discard("")
    n_unique_counterparties = len(all_counterparties)

    # Outgoing destination concentration
    if n_outgoing > 0:
        to_counts = outgoing["to"].str.lower().value_counts()
        top_dest_pct = float(to_counts.iloc[0] / n_outgoing) if len(to_counts) > 0 else 0.0
        n_unique_destinations = len(to_counts)
    else:
        top_dest_pct = 0.0
        n_unique_destinations = 0

    # Incoming vs outgoing ratio
    n_incoming = len(df[df["to"].str.lower() == addr_lower])
    incoming_outgoing_ratio = n_incoming / n_outgoing if n_outgoing > 0 else float("inf")

    # ── 4) Contract interaction patterns ──────────────────────────
    # Count distinct method IDs used (functional diversity)
    method_ids = outgoing["methodId"].dropna()
    method_ids = method_ids[method_ids != ""]
    method_ids = method_ids[method_ids != "0x"]
    n_distinct_methods = len(method_ids.unique())

    # Simple ETH transfers (no input data) vs contract calls
    simple_transfers = outgoing[
        (outgoing["input"].str.strip() == "0x") |
        (outgoing["input"].str.strip() == "") |
        (outgoing["input"].isna())
    ]
    n_simple_transfers = len(simple_transfers)
    n_contract_calls = n_outgoing - n_simple_transfers

    # ── 5) Gas analysis (security-relevant) ───────────────────────
    if "gasUsed" in df.columns:
        gas_used = pd.to_numeric(df["gasUsed"], errors="coerce").dropna()
        avg_gas = float(gas_used.mean()) if len(gas_used) > 0 else 0
        max_gas = float(gas_used.max()) if len(gas_used) > 0 else 0
    else:
        avg_gas = 0
        max_gas = 0

    # ── 6) Timing patterns (burst detection) ─────────────────────
    if "timeStamp" in df.columns and len(outgoing) > 1:
        timestamps = pd.to_numeric(outgoing["timeStamp"], errors="coerce").dropna().sort_values()
        if len(timestamps) > 1:
            intervals = timestamps.diff().dropna()
            # Bursts: intervals < 15 seconds
            n_bursts = int((intervals < 15).sum())
            burst_pct = n_bursts / len(intervals)
        else:
            n_bursts = 0
            burst_pct = 0.0
    else:
        n_bursts = 0
        burst_pct = 0.0

    # ── Risk assessment ───────────────────────────────────────────
    risk_flags = []
    if n_unlimited > 0:
        risk_flags.append(f"unlimited_approvals:{n_unlimited}")
    if revert_rate > 0.3:
        risk_flags.append(f"high_revert_rate:{revert_rate:.2f}")
    if burst_pct > 0.5:
        risk_flags.append(f"high_burst_activity:{burst_pct:.2f}")
    if top_dest_pct > 0.95:
        risk_flags.append(f"single_target_dominance:{top_dest_pct:.2f}")
    if n_distinct_methods <= 1 and n_contract_calls > 10:
        risk_flags.append("single_method_bot_pattern")

    risk_score = 0
    if n_unlimited > 0:
        risk_score += 3
    if revert_rate > 0.3:
        risk_score += 2
    elif revert_rate > 0.1:
        risk_score += 1
    if burst_pct > 0.5:
        risk_score += 1
    if n_distinct_methods <= 2 and n_outgoing > 100:
        risk_score += 1
    risk_level = "LOW" if risk_score <= 1 else ("MEDIUM" if risk_score <= 3 else "HIGH")

    return {
        "address": address,
        "total_txs": n_total,
        "outgoing_txs": n_outgoing,
        "incoming_txs": n_incoming,
        "revert_count": n_errors,
        "revert_rate": round(revert_rate, 4),
        "approvals": {
            "total_approve_calls": n_approvals,
            "unlimited_approvals": n_unlimited,
            "increase_allowance_calls": n_increase_allowance,
            "unique_unlimited_spenders": len(spenders_unlimited),
        },
        "network": {
            "unique_counterparties": n_unique_counterparties,
            "unique_destinations": n_unique_destinations,
            "top_destination_pct": round(top_dest_pct, 4),
            "incoming_outgoing_ratio": round(incoming_outgoing_ratio, 4)
            if incoming_outgoing_ratio != float("inf") else "inf",
        },
        "behavior": {
            "distinct_method_ids": n_distinct_methods,
            "simple_eth_transfers": n_simple_transfers,
            "contract_calls": n_contract_calls,
            "burst_tx_count": n_bursts,
            "burst_pct": round(burst_pct, 4),
        },
        "gas": {
            "avg_gas_used": round(avg_gas, 2),
            "max_gas_used": round(max_gas, 2),
        },
        "risk_assessment": {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "risk_flags": risk_flags,
        },
    }


def run_audit():
    # Load features to identify agent addresses
    df_feat = pd.read_parquet(FEATURES_PATH)
    agents = df_feat[df_feat["label"] == 1]
    agent_addresses = agents.index.tolist()
    agent_names = agents["name"].tolist()

    print(f"Running security audit on {len(agent_addresses)} agent addresses")
    print("=" * 70)

    all_results = []
    for addr, name in zip(agent_addresses, agent_names):
        raw_path = RAW_DIR / f"{addr}.parquet"
        if not raw_path.exists():
            print(f"  SKIP {name:35s} - no raw data file")
            continue

        result = analyze_address(addr, raw_path)
        result["name"] = name
        all_results.append(result)

        risk = result["risk_assessment"]
        print(
            f"  {name:35s}  txs={result['total_txs']:6d}  "
            f"revert={result['revert_rate']:.2%}  "
            f"approvals={result['approvals']['total_approve_calls']:3d}  "
            f"unlimited={result['approvals']['unlimited_approvals']:3d}  "
            f"risk={risk['risk_level']:6s}"
        )
        if risk["risk_flags"]:
            print(f"    {'':35s}  flags: {', '.join(risk['risk_flags'])}")

    # ── Summary statistics ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_agents = len(all_results)
    n_with_approvals = sum(1 for r in all_results if r["approvals"]["total_approve_calls"] > 0)
    n_with_unlimited = sum(1 for r in all_results if r["approvals"]["unlimited_approvals"] > 0)
    total_unlimited = sum(r["approvals"]["unlimited_approvals"] for r in all_results)
    avg_revert = np.mean([r["revert_rate"] for r in all_results])
    high_revert = sum(1 for r in all_results if r["revert_rate"] > 0.1)
    risk_counts = {
        "HIGH": sum(1 for r in all_results if r["risk_assessment"]["risk_level"] == "HIGH"),
        "MEDIUM": sum(1 for r in all_results if r["risk_assessment"]["risk_level"] == "MEDIUM"),
        "LOW": sum(1 for r in all_results if r["risk_assessment"]["risk_level"] == "LOW"),
    }

    print(f"  Agents analyzed:         {n_agents}")
    print(f"  With ERC20 approvals:    {n_with_approvals} ({n_with_approvals/n_agents:.0%})")
    print(f"  With unlimited approvals:{n_with_unlimited} ({n_with_unlimited/n_agents:.0%})")
    print(f"  Total unlimited approvals: {total_unlimited}")
    print(f"  Avg revert rate:         {avg_revert:.2%}")
    print(f"  High revert (>10%):      {high_revert} ({high_revert/n_agents:.0%})")
    print(f"  Risk levels:             HIGH={risk_counts['HIGH']}, "
          f"MEDIUM={risk_counts['MEDIUM']}, LOW={risk_counts['LOW']}")

    summary = {
        "n_agents_audited": n_agents,
        "agents_with_approvals": n_with_approvals,
        "agents_with_unlimited_approvals": n_with_unlimited,
        "total_unlimited_approvals": total_unlimited,
        "avg_revert_rate": round(avg_revert, 4),
        "agents_high_revert": high_revert,
        "risk_distribution": risk_counts,
    }

    # ── Save results ───────────────────────────────────────────────
    output = {
        "summary": summary,
        "agent_audits": all_results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_audit()
