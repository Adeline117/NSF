"""
Paper 1 + Paper 0: External Validation via Etherscan Labels
============================================================
Uses Etherscan's independently-maintained data to cross-validate
the labels in P1 (agent/human provenance) and P0 (8-class taxonomy).

For a sample of labeled addresses, we query Etherscan for:
  - Contract vs EOA status (proxy for agent identity)
  - Transaction count (activity proxy)
  - First transaction timestamp (longevity)
  - Contract ABI availability (verified contract = known bot infra)

We then check agreement between our labels and Etherscan's signals:
  - Agents: expect higher contract-interaction, more txs, contract status
  - Humans: expect EOA, fewer txs, no bot-infrastructure signatures
  - P0 MEVSearcher: expect known MEV patterns, high tx counts
  - P0 DeterministicScript: expect contract addresses
  - P0 LLMPoweredAgent: expect EOA (externally-owned, not contract)

Rate limit: 0.25s between calls (conservative).

Output: experiments/etherscan_label_validation_results.json
"""

import json
import logging
import random
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from shared.utils.eth_utils import EtherscanClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────
P1_LABELS = PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "labels_provenance_v4.json"
P0_FEATURES = PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
RESULTS_PATH = Path(__file__).resolve().parent / "etherscan_label_validation_results.json"

# ── Taxonomy name mapping (P0) ────────────────────────────────────
TAXONOMY_NAMES = {
    0: "SimpleTradingBot",
    1: "MEVSearcher",
    2: "DeFiManagementAgent",
    3: "LLMPoweredAgent",
    4: "AutonomousDAOAgent",
    5: "CrossChainBridgeAgent",
    6: "DeterministicScript",
    7: "RLTradingAgent",
}

# ── P1 category → agent/human mapping ─────────────────────────────
AGENT_CATEGORIES = {
    "curated_mev_bot", "expanded_mev_bot", "pilot_agent",
    "keep3r_executor", "gelato_executor", "chainlink_keeper",
    "compound_v3_liquidator", "defi_hf_trader", "flash_loan_user",
}
HUMAN_CATEGORIES = {
    "curated_human", "expanded_human", "pilot_human",
    "human_ens_interaction", "human_exchange_depositor",
    "gitcoin_donor", "ens_reverse_setter", "pooltogether_depositor",
    "human_ens_interaction_governance_voter",
}

# ── Seed for reproducibility ──────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def stratified_sample_p1(labels: dict, n_agent: int = 50,
                         n_human: int = 50, n_random: int = 100) -> list[dict]:
    """Sample P1 addresses stratified by agent/human with random remainder."""
    agents, humans = [], []
    for addr, info in labels.items():
        cat = info["category"]
        if cat in AGENT_CATEGORIES:
            agents.append((addr, info))
        elif cat in HUMAN_CATEGORIES:
            humans.append((addr, info))

    random.shuffle(agents)
    random.shuffle(humans)

    sampled_addrs = set()
    result = []

    # Agent stratified sample: proportional to sub-category sizes
    agent_by_cat = defaultdict(list)
    for addr, info in agents:
        agent_by_cat[info["category"]].append((addr, info))
    total_agents = len(agents)
    for cat, items in agent_by_cat.items():
        k = max(1, round(n_agent * len(items) / total_agents))
        for addr, info in items[:k]:
            if len([x for x in result if x["sample_group"] == "agent"]) >= n_agent:
                break
            result.append({
                "address": addr,
                "our_label": "agent",
                "our_category": info["category"],
                "our_name": info.get("name", ""),
                "sample_group": "agent",
            })
            sampled_addrs.add(addr)

    # Pad if we haven't hit n_agent
    for addr, info in agents:
        if addr not in sampled_addrs and len([x for x in result if x["sample_group"] == "agent"]) < n_agent:
            result.append({
                "address": addr,
                "our_label": "agent",
                "our_category": info["category"],
                "our_name": info.get("name", ""),
                "sample_group": "agent",
            })
            sampled_addrs.add(addr)

    # Human stratified sample
    human_by_cat = defaultdict(list)
    for addr, info in humans:
        human_by_cat[info["category"]].append((addr, info))
    total_humans = len(humans)
    for cat, items in human_by_cat.items():
        k = max(1, round(n_human * len(items) / total_humans))
        for addr, info in items[:k]:
            if len([x for x in result if x["sample_group"] == "human"]) >= n_human:
                break
            result.append({
                "address": addr,
                "our_label": "human",
                "our_category": info["category"],
                "our_name": info.get("name", ""),
                "sample_group": "human",
            })
            sampled_addrs.add(addr)

    # Pad if we haven't hit n_human
    for addr, info in humans:
        if addr not in sampled_addrs and len([x for x in result if x["sample_group"] == "human"]) < n_human:
            result.append({
                "address": addr,
                "our_label": "human",
                "our_category": info["category"],
                "our_name": info.get("name", ""),
                "sample_group": "human",
            })
            sampled_addrs.add(addr)

    # Random from the rest
    remaining = [(addr, info) for addr, info in labels.items() if addr not in sampled_addrs]
    random.shuffle(remaining)
    for addr, info in remaining[:n_random]:
        cat = info["category"]
        lbl = "agent" if cat in AGENT_CATEGORIES else "human"
        result.append({
            "address": addr,
            "our_label": lbl,
            "our_category": cat,
            "our_name": info.get("name", ""),
            "sample_group": "random",
        })
        sampled_addrs.add(addr)

    return result


def sample_p0_taxonomy(df: pd.DataFrame, n: int = 50) -> list[dict]:
    """Sample P0 taxonomy addresses: focus on MEVSearcher, DeterministicScript, LLMPoweredAgent."""
    agents = df[df["label"] == 1].copy()
    result = []
    sampled = set()

    # Target categories with specific counts
    targets = {
        1: 15,   # MEVSearcher
        6: 15,   # DeterministicScript
        3: 10,   # LLMPoweredAgent
    }
    # Fill rest from other categories
    remaining_budget = n - sum(targets.values())

    for tidx, count in targets.items():
        subset = agents[agents["taxonomy_index"] == tidx]
        if len(subset) == 0:
            continue
        sample_addrs = subset.index.tolist()
        random.shuffle(sample_addrs)
        for addr in sample_addrs[:count]:
            cat_name = TAXONOMY_NAMES.get(tidx, f"idx_{tidx}")
            result.append({
                "address": addr,
                "taxonomy_category": cat_name,
                "taxonomy_index": int(tidx),
                "our_name": str(subset.loc[addr, "name"]) if "name" in subset.columns else "",
                "sample_group": "p0_taxonomy",
            })
            sampled.add(addr)

    # Fill remaining from other taxonomy categories
    other = agents[~agents["taxonomy_index"].isin(targets.keys())]
    other = other[other["taxonomy_index"] >= 0]  # skip unclassified (-1)
    other_addrs = other.index.tolist()
    random.shuffle(other_addrs)
    for addr in other_addrs[:remaining_budget]:
        tidx = int(other.loc[addr, "taxonomy_index"])
        cat_name = TAXONOMY_NAMES.get(tidx, f"idx_{tidx}")
        if addr not in sampled:
            result.append({
                "address": addr,
                "taxonomy_category": cat_name,
                "taxonomy_index": tidx,
                "our_name": str(other.loc[addr, "name"]) if "name" in other.columns else "",
                "sample_group": "p0_taxonomy",
            })
            sampled.add(addr)

    return result


def query_etherscan_signals(client: EtherscanClient, address: str) -> dict:
    """Query Etherscan for validation signals about one address.

    Returns dict with:
      - is_contract: bool
      - has_verified_abi: bool
      - tx_count: int (from first page of normal txs)
      - first_tx_timestamp: int or None
      - first_tx_date: str or None
      - tx_sample_size: int (how many txs we actually got back)
    """
    signals = {
        "is_contract": None,
        "has_verified_abi": False,
        "tx_count": 0,
        "first_tx_timestamp": None,
        "first_tx_date": None,
        "tx_sample_size": 0,
        "error": None,
    }

    # 1. Contract check
    try:
        signals["is_contract"] = client.is_contract(address)
    except Exception as e:
        signals["error"] = f"is_contract: {e}"
        logger.warning(f"  is_contract failed for {address[:10]}...: {e}")

    # 2. Contract ABI (verified = known project, often bot infrastructure)
    try:
        abi = client.get_contract_abi(address)
        signals["has_verified_abi"] = abi is not None
    except Exception:
        pass  # Many addresses won't have verified ABIs

    # 3. Transaction list (first page, small offset for speed)
    try:
        df_txs = client.get_normal_txs(address, page=1, offset=50)
        if not df_txs.empty:
            signals["tx_sample_size"] = len(df_txs)
            # First tx info
            if "timeStamp" in df_txs.columns:
                first_ts = int(df_txs.iloc[0]["timeStamp"])
                signals["first_tx_timestamp"] = first_ts
                signals["first_tx_date"] = datetime.utcfromtimestamp(first_ts).strftime("%Y-%m-%d")

            # Estimate total tx count: if we got 50, there are likely more
            # Use the nonce of the latest tx as a proxy for total tx count
            if "nonce" in df_txs.columns:
                max_nonce = int(df_txs["nonce"].astype(int).max())
                signals["tx_count"] = max_nonce + 1  # nonce is 0-indexed
            else:
                signals["tx_count"] = len(df_txs)
        else:
            signals["tx_count"] = 0
    except Exception as e:
        signals["error"] = f"txlist: {e}"
        logger.warning(f"  txlist failed for {address[:10]}...: {e}")

    return signals


def compute_p1_agreement(results: list[dict]) -> dict:
    """Compute agreement between our P1 labels and Etherscan signals.

    Etherscan's public API doesn't expose human-readable tags directly.
    We use two tiers of proxy validation:

    Tier 1 — Hard signals (contract vs EOA):
      - Human addresses SHOULD be EOAs (not contracts).
      - Agent addresses CAN be either (MEV bots use EOAs; keeper contracts exist).

    Tier 2 — Consistency checks:
      - Addresses with a verified ABI are known contract infrastructure
        (supports agent label if present).
      - Addresses active before 2020 that are labeled "agent" should still
        show high nonce (they've been running a long time).

    We report per-signal rates rather than a single "agreement" number,
    since no single Etherscan signal perfectly maps to agent/human.
    """
    stats = {
        "total": 0,
        "agents_total": 0,
        "agents_is_contract": 0,
        "agents_has_verified_abi": 0,
        "humans_total": 0,
        "humans_is_eoa": 0,
        "humans_is_contract": 0,
        "humans_has_verified_abi": 0,
        "by_category": defaultdict(lambda: {
            "total": 0, "is_contract": 0,
            "tx_counts": [], "has_abi": 0,
            "first_dates": [],
        }),
        "errors": 0,
    }

    for r in results:
        if r.get("sample_group") == "p0_taxonomy":
            continue
        stats["total"] += 1
        signals = r.get("etherscan_signals", {})
        cat = r["our_category"]

        if signals.get("error"):
            stats["errors"] += 1

        cat_stats = stats["by_category"][cat]
        cat_stats["total"] += 1
        if signals.get("is_contract"):
            cat_stats["is_contract"] += 1
        cat_stats["tx_counts"].append(signals.get("tx_count", 0))
        if signals.get("has_verified_abi"):
            cat_stats["has_abi"] += 1
        if signals.get("first_tx_date"):
            cat_stats["first_dates"].append(signals["first_tx_date"])

        if r["our_label"] == "agent":
            stats["agents_total"] += 1
            if signals.get("is_contract"):
                stats["agents_is_contract"] += 1
            if signals.get("has_verified_abi"):
                stats["agents_has_verified_abi"] += 1

        elif r["our_label"] == "human":
            stats["humans_total"] += 1
            if not signals.get("is_contract"):
                stats["humans_is_eoa"] += 1
            else:
                stats["humans_is_contract"] += 1
            if signals.get("has_verified_abi"):
                stats["humans_has_verified_abi"] += 1

    # ── Tier 1: Hard signal — human ↔ EOA consistency ────────────
    # Human addresses should overwhelmingly be EOAs.
    # If a "human" address is a contract, that's a potential mislabel.
    humans_eoa_rate = round(
        stats["humans_is_eoa"] / max(1, stats["humans_total"]), 4
    )
    # Agent addresses being contracts is a supporting (not required) signal.
    agents_contract_rate = round(
        stats["agents_is_contract"] / max(1, stats["agents_total"]), 4
    )

    # ── Tier 2: Consistency — no contradictions ──────────────────
    # A human address with a verified contract ABI is suspicious.
    humans_suspicious = stats["humans_has_verified_abi"]
    # An agent address that is a contract with verified ABI strongly confirms.
    agents_confirmed_by_abi = stats["agents_has_verified_abi"]

    # ── Overall consistency rate ──────────────────────────────────
    # "Consistent" = human is EOA, OR agent is contract/has-ABI, OR
    #   agent is EOA (acceptable — most bots operate via EOAs)
    # "Inconsistent" = human is contract (potential mislabel)
    n_inconsistent = stats["humans_is_contract"]
    consistency_rate = round(
        1 - (n_inconsistent / max(1, stats["total"])), 4
    )

    summary = {
        "total_queried": stats["total"],
        "errors": stats["errors"],
        "consistency_rate": consistency_rate,
        "n_inconsistent_human_is_contract": n_inconsistent,
        "agent_sample": {
            "n": stats["agents_total"],
            "is_contract_count": stats["agents_is_contract"],
            "is_contract_rate": agents_contract_rate,
            "has_verified_abi_count": agents_confirmed_by_abi,
            "note": "Most Ethereum bots (MEV, keepers) operate as EOAs, so low contract rate is expected.",
        },
        "human_sample": {
            "n": stats["humans_total"],
            "is_eoa_count": stats["humans_is_eoa"],
            "is_eoa_rate": humans_eoa_rate,
            "is_contract_count": stats["humans_is_contract"],
            "has_verified_abi_count": stats["humans_has_verified_abi"],
            "note": "Human addresses that are contracts may be mislabeled or multisig wallets.",
        },
    }

    # Per-category breakdown
    cat_summary = {}
    for cat, cs in stats["by_category"].items():
        txs = cs["tx_counts"]
        cat_summary[cat] = {
            "n": cs["total"],
            "is_contract_rate": round(cs["is_contract"] / max(1, cs["total"]), 4),
            "mean_tx_count": round(np.mean(txs), 1) if txs else 0,
            "median_tx_count": round(float(np.median(txs)), 1) if txs else 0,
            "has_verified_abi_rate": round(cs["has_abi"] / max(1, cs["total"]), 4),
            "first_date_range": (
                f"{min(cs['first_dates'])} to {max(cs['first_dates'])}"
                if cs["first_dates"] else "N/A"
            ),
        }
    summary["by_category"] = cat_summary

    return summary


def compute_p0_agreement(results: list[dict]) -> dict:
    """Compute agreement for P0 taxonomy addresses.

    - MEVSearcher: expect is_contract=True OR extremely high tx counts
    - DeterministicScript: expect is_contract=True (on-chain scripts)
    - LLMPoweredAgent: expect EOA (LLMs operate via EOA wallets)
    """
    by_tax = defaultdict(lambda: {
        "n": 0, "is_contract": 0, "has_abi": 0,
        "tx_counts": [], "first_dates": [],
    })

    for r in results:
        if r.get("sample_group") != "p0_taxonomy":
            continue
        cat = r["taxonomy_category"]
        signals = r.get("etherscan_signals", {})

        ts = by_tax[cat]
        ts["n"] += 1
        if signals.get("is_contract"):
            ts["is_contract"] += 1
        if signals.get("has_verified_abi"):
            ts["has_abi"] += 1
        ts["tx_counts"].append(signals.get("tx_count", 0))
        if signals.get("first_tx_date"):
            ts["first_dates"].append(signals["first_tx_date"])

    summary = {}
    for cat, ts in by_tax.items():
        txc = ts["tx_counts"]
        summary[cat] = {
            "n": ts["n"],
            "is_contract_count": ts["is_contract"],
            "is_contract_rate": round(ts["is_contract"] / max(1, ts["n"]), 4),
            "has_verified_abi_count": ts["has_abi"],
            "has_verified_abi_rate": round(ts["has_abi"] / max(1, ts["n"]), 4),
            "mean_tx_count": round(np.mean(txc), 1) if txc else 0,
            "median_tx_count": round(float(np.median(txc)), 1) if txc else 0,
            "max_tx_count": int(max(txc)) if txc else 0,
            "first_dates_range": (
                f"{min(ts['first_dates'])} to {max(ts['first_dates'])}"
                if ts["first_dates"] else "N/A"
            ),
        }

    # Specific agreement checks
    checks = {}
    if "MEVSearcher" in summary:
        m = summary["MEVSearcher"]
        # MEV bots are typically contracts or have extremely high tx counts
        checks["MEVSearcher_contract_or_high_activity"] = (
            m["is_contract_rate"] > 0.3 or m["median_tx_count"] > 5000
        )
        checks["MEVSearcher_median_txs"] = m["median_tx_count"]

    if "DeterministicScript" in summary:
        d = summary["DeterministicScript"]
        # Deterministic scripts often run as contracts
        checks["DeterministicScript_contract_rate"] = d["is_contract_rate"]
        checks["DeterministicScript_high_activity_rate"] = (
            round(len([t for t in by_tax["DeterministicScript"]["tx_counts"] if t > 100])
                  / max(1, d["n"]), 4)
        )

    if "LLMPoweredAgent" in summary:
        lp = summary["LLMPoweredAgent"]
        # LLM agents typically use EOA wallets (not contracts)
        checks["LLMPoweredAgent_eoa_rate"] = round(1 - lp["is_contract_rate"], 4)
        checks["LLMPoweredAgent_median_txs"] = lp["median_tx_count"]

    return {
        "by_taxonomy": summary,
        "agreement_checks": checks,
    }


def main():
    print("=" * 80)
    print("External Validation: Etherscan Labels vs Our Provenance Labels")
    print("=" * 80)

    t0 = time.time()
    client = EtherscanClient()
    logger.info(f"EtherscanClient initialized with {client.num_keys} API keys")

    # ── Load P1 labels ─────────────────────────────────────────────
    with open(P1_LABELS) as f:
        p1_labels = json.load(f)
    logger.info(f"P1 labels loaded: {len(p1_labels)} addresses")

    # ── Load P0 taxonomy ───────────────────────────────────────────
    p0_df = pd.read_parquet(P0_FEATURES)
    n_p0_agents = (p0_df["label"] == 1).sum()
    logger.info(f"P0 taxonomy loaded: {len(p0_df)} total, {n_p0_agents} agents")

    # ── Sample P1: 50 agents + 50 humans + 100 random ─────────────
    p1_sample = stratified_sample_p1(p1_labels, n_agent=50, n_human=50, n_random=100)
    logger.info(f"P1 sample: {len(p1_sample)} addresses "
                f"({sum(1 for s in p1_sample if s['sample_group']=='agent')} agent, "
                f"{sum(1 for s in p1_sample if s['sample_group']=='human')} human, "
                f"{sum(1 for s in p1_sample if s['sample_group']=='random')} random)")

    # ── Sample P0: 50 taxonomy addresses ──────────────────────────
    p0_sample = sample_p0_taxonomy(p0_df, n=50)
    logger.info(f"P0 sample: {len(p0_sample)} addresses")
    for cat in set(s["taxonomy_category"] for s in p0_sample):
        n = sum(1 for s in p0_sample if s["taxonomy_category"] == cat)
        logger.info(f"  {cat}: {n}")

    # ── Combined sample ───────────────────────────────────────────
    all_samples = p1_sample + p0_sample
    total = len(all_samples)
    logger.info(f"Total addresses to query: {total}")

    # ── Query Etherscan for each address ──────────────────────────
    for i, sample in enumerate(all_samples):
        addr = sample["address"]
        pct = (i + 1) / total * 100
        label_info = sample.get("our_category", sample.get("taxonomy_category", "?"))
        logger.info(f"[{i+1}/{total}] ({pct:.0f}%) Querying {addr[:12]}... ({label_info})")

        signals = query_etherscan_signals(client, addr)
        sample["etherscan_signals"] = signals

        contract_str = "CONTRACT" if signals.get("is_contract") else "EOA"
        tx_str = f"txs={signals.get('tx_count', '?')}"
        abi_str = "ABI=YES" if signals.get("has_verified_abi") else ""
        logger.info(f"  -> {contract_str}, {tx_str} {abi_str}")

    elapsed = time.time() - t0
    logger.info(f"All queries complete in {elapsed:.1f}s "
                f"({client._total_calls} API calls)")

    # ── Compute P1 agreement ──────────────────────────────────────
    p1_agreement = compute_p1_agreement(all_samples)
    logger.info(f"\n{'='*60}")
    logger.info(f"P1 EXTERNAL VALIDATION SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Consistency rate: {p1_agreement['consistency_rate']:.1%} "
                f"({p1_agreement['n_inconsistent_human_is_contract']} humans flagged as contracts)")
    logger.info(f"Agent sample (n={p1_agreement['agent_sample']['n']}):")
    logger.info(f"  - is_contract: {p1_agreement['agent_sample']['is_contract_rate']:.1%} "
                f"(low is expected — most bots use EOAs)")
    logger.info(f"  - has_verified_abi: {p1_agreement['agent_sample']['has_verified_abi_count']}")
    logger.info(f"Human sample (n={p1_agreement['human_sample']['n']}):")
    logger.info(f"  - is_eoa: {p1_agreement['human_sample']['is_eoa_rate']:.1%} "
                f"({p1_agreement['human_sample']['is_eoa_count']}/{p1_agreement['human_sample']['n']})")
    logger.info(f"  - is_contract (potential mislabels): {p1_agreement['human_sample']['is_contract_count']}")

    # Per-category details
    logger.info(f"\nPer-category breakdown:")
    for cat, cs in sorted(p1_agreement["by_category"].items()):
        logger.info(f"  {cat:40s} n={cs['n']:3d}  contract={cs['is_contract_rate']:.0%}  "
                     f"median_tx={cs['median_tx_count']:.0f}  "
                     f"abi={cs['has_verified_abi_rate']:.0%}  "
                     f"dates={cs['first_date_range']}")

    # ── Compute P0 agreement ──────────────────────────────────────
    p0_agreement = compute_p0_agreement(all_samples)
    logger.info(f"\n{'='*60}")
    logger.info(f"P0 TAXONOMY AGREEMENT SUMMARY")
    logger.info(f"{'='*60}")
    for cat, ts in sorted(p0_agreement["by_taxonomy"].items()):
        logger.info(f"  {cat:25s} n={ts['n']:3d}  contract={ts['is_contract_rate']:.0%}  "
                     f"abi={ts['has_verified_abi_rate']:.0%}  "
                     f"median_tx={ts['median_tx_count']:.0f}  "
                     f"dates={ts['first_dates_range']}")
    logger.info(f"\nAgreement checks:")
    for k, v in p0_agreement["agreement_checks"].items():
        logger.info(f"  {k}: {v}")

    # ── Assemble results ──────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "api_calls": client._total_calls,
        "seed": SEED,
        "sample_sizes": {
            "p1_agents": sum(1 for s in p1_sample if s["sample_group"] == "agent"),
            "p1_humans": sum(1 for s in p1_sample if s["sample_group"] == "human"),
            "p1_random": sum(1 for s in p1_sample if s["sample_group"] == "random"),
            "p0_taxonomy": len(p0_sample),
            "total": total,
        },
        "p1_agreement": p1_agreement,
        "p0_agreement": p0_agreement,
        "address_details": [],
    }

    # Include per-address details (for auditing)
    for s in all_samples:
        detail = {
            "address": s["address"],
            "sample_group": s["sample_group"],
        }
        if "our_label" in s:
            detail["our_label"] = s["our_label"]
            detail["our_category"] = s["our_category"]
        if "taxonomy_category" in s:
            detail["taxonomy_category"] = s["taxonomy_category"]
        if "our_name" in s:
            detail["our_name"] = s["our_name"]
        sig = s.get("etherscan_signals", {})
        detail["is_contract"] = sig.get("is_contract")
        detail["has_verified_abi"] = sig.get("has_verified_abi")
        detail["tx_count"] = sig.get("tx_count", 0)
        detail["first_tx_date"] = sig.get("first_tx_date")
        detail["error"] = sig.get("error")
        results["address_details"].append(detail)

    # ── Save ──────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {RESULTS_PATH}")

    print(f"\n{'='*80}")
    print(f"DONE — {total} addresses validated in {elapsed:.0f}s")
    print(f"P1 consistency rate: {p1_agreement['consistency_rate']:.1%}")
    print(f"  Humans confirmed EOA: {p1_agreement['human_sample']['is_eoa_rate']:.1%}")
    print(f"  Agents is_contract: {p1_agreement['agent_sample']['is_contract_rate']:.1%}")
    print(f"Results: {RESULTS_PATH}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
