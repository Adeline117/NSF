"""
Paper 1: Security Audit at Scale (2590 Agents)
================================================
Runs the four-dimensional security audit on all 2590 labeled agents
in features_expanded.parquet. Uses the pre-cached raw parquet files
in paper1_onchain_agent_id/data/raw/ so no Etherscan API calls needed.

Dimensions:
  1. Permission exposure (unlimited approvals, approval age, verified ratio)
  2. MEV exposure (DEX interaction rate, swap concentration)
  3. Failure analysis (revert rate, error classification)
  4. Aggregate statistics and Mann-Whitney agent vs human comparison

Skipped: Network topology (requires graph, too expensive for 2590 nodes).

Outputs:
  - experiments/expanded/security_audit_expanded.json
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

FEATURES_PARQUET = PROJECT_ROOT / "data" / "features_expanded.parquet"
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUT_PATH = PROJECT_ROOT / "experiments" / "expanded" / "security_audit_expanded.json"

MAX_UINT256_HEX = "f" * 64
APPROVE_METHOD_ID = "0x095ea7b3"

# Known DEX routers (subset — most common on mainnet)
DEX_ROUTERS = {
    "0x7a250d5630b4cf539739df2c5dacb4c659f2488d",  # Uniswap V2
    "0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45",  # Uniswap V3 Router2
    "0x3fc91a3afd70395cd496c647d5a6cc9d4b2b7fad",  # Universal Router
    "0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f",  # SushiSwap
    "0xdef1c0ded9bec7f1a1670819833240f027b25eff",  # 0x Proxy
    "0xe592427a0aece92de3edee1f18e0157c05861564",  # Uniswap V3 Router
    "0x1111111254eeb25477b68fb85ed929f73a960582",  # 1inch V5
    "0x1111111254fb6c44bac0bed2854e76f90643097d",  # 1inch V4
}

SWAP_METHOD_IDS = {
    "0x38ed1739",  # swapExactTokensForTokens
    "0x8803dbee",  # swapTokensForExactTokens
    "0x7ff36ab5",  # swapExactETHForTokens
    "0x18cbafe5",  # swapExactTokensForETH
    "0xfb3bdb41",  # swapETHForExactTokens
    "0x5ae401dc",  # multicall (Uniswap V3)
    "0x3593564c",  # execute (Universal Router)
    "0x12aa3caf",  # 1inch swap
}


def audit_one_address(addr: str, label: int) -> dict:
    """Audit one address from its cached parquet file."""
    path = RAW_DATA_DIR / f"{addr}.parquet"
    if not path.exists():
        return None

    try:
        txs = pd.read_parquet(path)
    except Exception as e:
        return None

    if txs.empty or len(txs) < 5:
        return None

    # Filter to outgoing (from == addr)
    outgoing = txs[txs["from"].str.lower() == addr.lower()]
    if outgoing.empty:
        return None

    # ----- PERMISSION EXPOSURE -----
    inputs = outgoing["input"].fillna("").astype(str)
    approve_mask = inputs.str.startswith(APPROVE_METHOD_ID, na=False)
    approvals = outgoing[approve_mask].copy()
    approval_inputs = approvals["input"].fillna("").astype(str)

    n_approvals = len(approvals)
    n_unlimited = int(approval_inputs.str.contains(MAX_UINT256_HEX, na=False).sum())

    # Revocation = approve with zero amount
    zero_amount = "0" * 64
    revoke_mask = approval_inputs.str.endswith(zero_amount, na=False)
    n_revokes = int(revoke_mask.sum())
    n_active_approvals = n_approvals - n_revokes

    # Approval age (non-revoked)
    avg_approval_age_days = 0.0
    max_approval_age_days = 0.0
    non_revoked = approvals[~revoke_mask]
    if not non_revoked.empty and "timeStamp" in non_revoked.columns:
        ts = pd.to_numeric(non_revoked["timeStamp"], errors="coerce").dropna()
        if len(ts) > 0:
            now = time.time()
            ages_days = (now - ts) / 86400
            avg_approval_age_days = float(ages_days.mean())
            max_approval_age_days = float(ages_days.max())

    # ----- MEV / DEX EXPOSURE -----
    to_lower = outgoing["to"].fillna("").astype(str).str.lower()
    dex_interactions = int(to_lower.isin(DEX_ROUTERS).sum())

    method_ids = inputs.str[:10]
    swap_interactions = int(method_ids.isin(SWAP_METHOD_IDS).sum())

    n_total = len(outgoing)
    dex_rate = dex_interactions / max(n_total, 1)
    swap_rate = swap_interactions / max(n_total, 1)

    # ----- FAILURE ANALYSIS -----
    reverted = 0
    if "isError" in outgoing.columns:
        reverted = int(
            pd.to_numeric(outgoing["isError"], errors="coerce").fillna(0).sum()
        )
    revert_rate = reverted / max(n_total, 1)

    # Gas usage efficiency
    gas_used_rate = 0.0
    if "gasUsed" in outgoing.columns and "gas" in outgoing.columns:
        gas = pd.to_numeric(outgoing["gas"], errors="coerce")
        gas_used = pd.to_numeric(outgoing["gasUsed"], errors="coerce")
        mask = (gas > 0) & gas_used.notna()
        if mask.any():
            gas_used_rate = float((gas_used[mask] / gas[mask]).mean())

    return {
        "address": addr,
        "label": label,
        "n_transactions": n_total,
        # Permission
        "n_approvals": n_approvals,
        "n_unlimited_approvals": n_unlimited,
        "n_active_approvals": n_active_approvals,
        "avg_approval_age_days": round(avg_approval_age_days, 2),
        "max_approval_age_days": round(max_approval_age_days, 2),
        "unlimited_approval_rate": round(
            n_unlimited / max(n_approvals, 1), 4,
        ),
        # MEV / DEX
        "dex_interactions": dex_interactions,
        "swap_interactions": swap_interactions,
        "dex_interaction_rate": round(dex_rate, 4),
        "swap_rate": round(swap_rate, 4),
        # Failure
        "reverted_txs": reverted,
        "revert_rate": round(revert_rate, 4),
        "gas_used_efficiency": round(gas_used_rate, 4),
    }


def summarize_audit(audit_rows: list[dict]) -> dict:
    """Aggregate statistics + Mann-Whitney agent vs human."""
    df = pd.DataFrame(audit_rows)
    agents = df[df["label"] == 1]
    humans = df[df["label"] == 0]

    print(f"\nN agents audited: {len(agents)}")
    print(f"N humans audited: {len(humans)}")

    metrics = [
        "n_approvals", "n_unlimited_approvals", "n_active_approvals",
        "avg_approval_age_days", "max_approval_age_days",
        "unlimited_approval_rate", "dex_interaction_rate", "swap_rate",
        "revert_rate", "gas_used_efficiency",
    ]

    summary = {}
    for m in metrics:
        agent_vals = agents[m].values
        human_vals = humans[m].values

        # Compute stats
        a_mean = float(np.mean(agent_vals))
        a_std = float(np.std(agent_vals))
        a_median = float(np.median(agent_vals))
        h_mean = float(np.mean(human_vals))
        h_std = float(np.std(human_vals))
        h_median = float(np.median(human_vals))

        # Cohen's d
        pooled_std = np.sqrt((a_std ** 2 + h_std ** 2) / 2) if (a_std + h_std) > 0 else 1e-9
        d = (a_mean - h_mean) / pooled_std if pooled_std > 0 else 0.0

        # Mann-Whitney U
        try:
            u, p = scipy_stats.mannwhitneyu(agent_vals, human_vals, alternative="two-sided")
        except ValueError:
            u, p = float("nan"), 1.0

        # Ratio (avoid div by zero)
        ratio = a_mean / h_mean if h_mean > 0 else float("inf")

        summary[m] = {
            "agent": {
                "mean": round(a_mean, 4),
                "std": round(a_std, 4),
                "median": round(a_median, 4),
                "n": int(len(agent_vals)),
            },
            "human": {
                "mean": round(h_mean, 4),
                "std": round(h_std, 4),
                "median": round(h_median, 4),
                "n": int(len(human_vals)),
            },
            "agent_human_ratio": round(float(ratio), 3) if not np.isinf(ratio) else "inf",
            "cohens_d": round(float(d), 4),
            "mann_whitney_U": float(u) if not np.isnan(u) else None,
            "mann_whitney_p": round(float(p), 6),
            "significant_0.05": bool(p < 0.05),
            "significant_0.01": bool(p < 0.01),
        }

    return summary


def main():
    print("=" * 70)
    print("Paper 1: Security Audit at Scale (2590+ agents)")
    print("=" * 70)

    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"Loaded {len(df)} labeled addresses")

    audit_rows = []
    skipped = 0
    t0 = time.time()

    for i, (addr, row) in enumerate(df.iterrows()):
        if i % 200 == 0:
            print(f"  progress: {i}/{len(df)}  ({time.time() - t0:.1f}s elapsed)")
        result = audit_one_address(str(addr), int(row["label"]))
        if result is None:
            skipped += 1
            continue
        audit_rows.append(result)

    print(f"\nAudited {len(audit_rows)} addresses ({skipped} skipped)")

    # Summarize
    summary = summarize_audit(audit_rows)

    # Print key findings
    print("\n" + "=" * 90)
    print(f"{'Metric':<30} {'Agent mean':>12} {'Human mean':>12} {'Ratio':>8} "
          f"{'d':>7} {'p-val':>10}")
    print("=" * 90)
    for m, stats in summary.items():
        print(f"{m:<30} {stats['agent']['mean']:>12.4f} "
              f"{stats['human']['mean']:>12.4f} "
              f"{str(stats['agent_human_ratio']):>8} "
              f"{stats['cohens_d']:>+7.3f} "
              f"{stats['mann_whitney_p']:>10.6f}"
              f"{' ***' if stats['significant_0.01'] else (' *' if stats['significant_0.05'] else '')}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_total_labeled": int(len(df)),
        "n_audited": int(len(audit_rows)),
        "n_skipped": int(skipped),
        "n_agents_audited": int(sum(1 for r in audit_rows if r["label"] == 1)),
        "n_humans_audited": int(sum(1 for r in audit_rows if r["label"] == 0)),
        "summary": summary,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
