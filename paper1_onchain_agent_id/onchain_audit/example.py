"""Worked example: audit the provenance-v4 classifier on the 1,147 dataset.

Run:

    python -m onchain_audit.example

Outputs a Markdown audit report to stdout and writes it to
``onchain_audit/example_report.md`` for inspection.

The example reproduces the headline finding from Wen et al. (WWW '26):
Step 1 flags direct overlap between the C1-C4 mining rules and the top
classifier features; Step 2 shows AUC collapses from ~0.98 on the heuristic-
mined set to ~0.77 on the strict curated subset; Step 3 shows LOPO AUC sits
far below random-split AUC.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .audit import (
    check_label_feature_overlap,
    compare_purity_tiers,
    cross_scheme_transfer,
)
from .report import AuditDataset, generate_audit_report

PKG_ROOT = Path(__file__).resolve().parent.parent
FEATURES = PKG_ROOT / "data" / "features_provenance_v4.parquet"
LABELS = PKG_ROOT / "data" / "labels_provenance_v4.json"
SPLITS_DIR = PKG_ROOT / "benchmark" / "splits"


FEATURE_COLS = [
    "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
    "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
    "burst_frequency",
    "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
    "gas_limit_precision", "gas_price_cv",
    "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
    "unique_contracts_ratio", "top_contract_concentration",
    "method_id_diversity", "contract_to_eoa_ratio",
    "sequential_pattern_score",
    "unlimited_approve_ratio", "approve_revoke_ratio",
    "unverified_contract_approve_ratio",
    "multi_protocol_interaction_count", "flash_loan_usage",
]


def _clean_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[[c for c in FEATURE_COLS if c in df.columns]].copy()
    # median impute + 1/99 clip
    for col in X.columns:
        vals = X[col].astype(float)
        med = float(np.nanmedian(vals)) if np.isfinite(np.nanmedian(vals)) else 0.0
        vals = vals.fillna(med)
        lo, hi = np.nanpercentile(vals, [1, 99])
        X[col] = vals.clip(lo, hi)
    return X


def load_example_dataset() -> AuditDataset:
    df = pd.read_parquet(FEATURES)
    with open(LABELS) as f:
        labels_meta = json.load(f)

    X = _clean_features(df)
    y = df["label"].astype(int)
    y.name = "label"

    # Mining rules: the C1-C4 gate columns and a few provenance flags.
    # These are the rules that actually produced the labels.
    rules = pd.DataFrame(index=df.index)
    if "c1c4_c1" in df.columns:
        rules["c1c4_c1_tx_interval_cv"] = df["c1c4_c1"].astype(float)
    if "c1c4_c2" in df.columns:
        rules["c1c4_c2_gas_precision"] = df["c1c4_c2"].astype(float)
    if "c1c4_c3" in df.columns:
        rules["c1c4_c3_hour_entropy"] = df["c1c4_c3"].astype(float)
    if "c1c4_c4" in df.columns:
        rules["c1c4_c4_contract_diversity"] = df["c1c4_c4"].astype(float)
    # Source-based rules
    rules["rule_chainlink_keeper"] = df["source"].eq("on_chain_chainlink_keeper").astype(int)
    rules["rule_defi_hf_trader"] = df["source"].eq("on_chain_defi_high_frequency").astype(int)
    rules["rule_ens_interaction"] = df["source"].eq("on_chain_ens_interaction").astype(int)

    # Purity tiers
    with open(SPLITS_DIR / "level4_strict_core.json") as f:
        strict = json.load(f)
    strict_addrs = set(strict["addresses"])
    curated_mask = df.index.isin(strict_addrs)

    tiers = {
        "all_mined_n1147": (X, y),
        "strict_curated": (X[curated_mask], y[curated_mask]),
    }

    # Cross-scheme: MEV/liquidation-style (high-purity on-chain) vs ENS humans
    agent_scheme = df["source"].isin(
        ["on_chain_chainlink_keeper", "on_chain_compound_v3_liquidation",
         "on_chain_gelato_executor", "on_chain_keep3r_executor"]
    )
    # Add the ENS human tier: make it a mixed "defi_hf_trader (agent)" vs "ens_interaction (human)"
    scheme_a_mask = df["source"].isin(
        ["on_chain_chainlink_keeper", "on_chain_compound_v3_liquidation"]
    ) | df["source"].isin(["on_chain_ens_interaction"])
    scheme_b_mask = df["source"].isin(
        ["on_chain_gelato_executor", "on_chain_keep3r_executor"]
    ) | df["source"].isin(["on_chain_gitcoin_grants_donor", "on_chain_ens_reverse_registrar"])

    schemes = {
        "scheme_A_keepers_vs_ENS": (X[scheme_a_mask], y[scheme_a_mask]),
        "scheme_B_executors_vs_gitcoin_ensrev": (X[scheme_b_mask], y[scheme_b_mask]),
    }

    return AuditDataset(
        features=X,
        labels=y,
        mining_rules=rules,
        purity_tiers=tiers,
        schemes=schemes,
        name="OnChainAgentID v4 (N=1,147)",
    )


def main(output: str | None = None) -> str:
    dataset = load_example_dataset()
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    report = generate_audit_report(clf, dataset)
    out_path = Path(output) if output else (Path(__file__).parent / "example_report.md")
    out_path.write_text(report)
    print(report)
    print(f"\n(report also saved to {out_path})")
    return report


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
