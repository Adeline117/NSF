"""Generate canonical train/test/LOPO/temporal splits for the OnChainAgentID benchmark.

Run once; produces JSON files with address lists that downstream users load.

Outputs (all in this directory):
    - level2_random_10fold.json        Random stratified 10-fold CV (Level 2)
    - level3_temporal_holdout.json     Split by first-seen block (Level 3)
    - level4_strict_core.json          Hand-curated provenance subset (Level 4)
    - level5_lopo.json                 Leave-One-Platform-Out (Level 5)
    - all_addresses.json               Full 1,147-address index

All addresses are returned in the exact case-preserving form used by
``data/features_provenance_v4.parquet``. Users should load addresses and index
into that parquet to recover feature vectors.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent
FEATURES = PKG_ROOT / "data" / "features_provenance_v4.parquet"
LABELS = PKG_ROOT / "data" / "labels_provenance_v4.json"
RAW_DIR = PKG_ROOT / "data" / "raw"

# Hand-curated sources = Level 4 "strict core" (no on-chain mining rules)
CURATED_SOURCES = {
    "flashbots_relay_data",
    "ens_resolution",
    "etherscan_label",
    "arkham_label",
    "ens_twitter_verified",
    "twitter_verified",
    "pilot_trusted",
    "strategy2_paper0",
    "strategy_c_human",
    "expanded_curated_nan",
    "expanded_curated_strategy_b_mev",
    "expanded_pilot",
    "expanded_human_strategy_c_human",
}


def load_dataset() -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(FEATURES)
    with open(LABELS) as f:
        labels = json.load(f)
    return df, labels


def make_level2_random_cv(df: pd.DataFrame, n_splits: int = 10, seed: int = 42) -> dict:
    """Stratified 10-fold CV on all 1,147 addresses."""
    addresses = list(df.index)
    y = df["label"].values.astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(addresses, y)):
        folds.append(
            {
                "fold": fold_idx,
                "train_addresses": [addresses[i] for i in train_idx],
                "test_addresses": [addresses[i] for i in test_idx],
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )
    return {
        "level": 2,
        "name": "Random 10-fold stratified CV",
        "description": "Upper-bound sanity check. All 1,147 addresses, stratified by agent/human label. Random shuffle with seed=42.",
        "n_addresses": len(addresses),
        "n_splits": n_splits,
        "seed": seed,
        "folds": folds,
    }


def make_level3_temporal(df: pd.DataFrame) -> dict:
    """Temporal holdout: split by median first-seen block from raw transaction data."""
    raw_files = {p.stem.lower(): p for p in RAW_DIR.glob("*.parquet")}
    first_blocks = {}
    for addr in df.index:
        p = raw_files.get(addr.lower())
        if p is None:
            continue
        try:
            raw = pd.read_parquet(p, columns=["blockNumber"])
            blocks = pd.to_numeric(raw["blockNumber"], errors="coerce").dropna()
            if len(blocks):
                first_blocks[addr] = int(blocks.min())
        except Exception:
            continue

    if not first_blocks:
        raise RuntimeError("No first-block data found; run temporal holdout extraction first.")

    blocks_series = pd.Series(first_blocks)
    median_block = int(blocks_series.median())

    train_addresses = [a for a, b in first_blocks.items() if b <= median_block]
    test_addresses = [a for a, b in first_blocks.items() if b > median_block]

    return {
        "level": 3,
        "name": "Temporal holdout",
        "description": (
            "Split by first-seen block height. Addresses whose first tx block is <= median "
            "go to TRAIN; those > median go to TEST. Simulates realistic deployment where "
            "the classifier must generalise to future addresses."
        ),
        "median_block": median_block,
        "n_with_blocks": len(first_blocks),
        "n_missing_raw": int(len(df) - len(first_blocks)),
        "train_addresses": train_addresses,
        "test_addresses": test_addresses,
        "n_train": len(train_addresses),
        "n_test": len(test_addresses),
    }


def make_level4_strict(labels: dict, df: pd.DataFrame) -> dict:
    """Hand-curated subset (no on-chain mining rules).

    This is the highest-purity subset where agent labels come from relay data,
    Etherscan labels, Arkham labels, ENS verification, or manual pilot
    curation (N~=70). Used to estimate an honest upper bound on classifier
    quality in the absence of any feature/label leakage.
    """
    strict_addresses = [
        a for a in df.index
        if a in labels and labels[a]["provenance_source"] in CURATED_SOURCES
    ]
    strict_labels = [int(labels[a]["label_provenance"]) for a in strict_addresses]
    cats = Counter(labels[a]["category"] for a in strict_addresses)
    return {
        "level": 4,
        "name": "Strict curated core",
        "description": (
            "Hand-curated addresses only: Flashbots relay data, ENS-verified, "
            "Etherscan labels, Arkham labels, manual pilot curation. No on-chain "
            "heuristic mining. This is the honest upper-bound subset: ~N=70."
        ),
        "curated_sources": sorted(CURATED_SOURCES),
        "addresses": strict_addresses,
        "labels": strict_labels,
        "category_counts": dict(cats),
        "n_addresses": len(strict_addresses),
        "n_agents": sum(strict_labels),
        "n_humans": len(strict_labels) - sum(strict_labels),
    }


def make_level5_lopo(labels: dict, df: pd.DataFrame, min_platform: int = 10) -> dict:
    """Leave-One-Platform-Out: each fold holds out a single category."""
    categories = df["category"] if "category" in df.columns else pd.Series(
        {a: labels[a]["category"] for a in df.index if a in labels}
    )
    cat_counts = Counter(categories)
    folds = []
    for cat, n in sorted(cat_counts.items(), key=lambda kv: -kv[1]):
        if n < min_platform:
            continue
        mask = (categories == cat).values
        train_addresses = [a for a, m in zip(df.index, mask) if not m]
        test_addresses = [a for a, m in zip(df.index, mask) if m]
        folds.append(
            {
                "held_out_platform": cat,
                "train_addresses": train_addresses,
                "test_addresses": test_addresses,
                "n_train": len(train_addresses),
                "n_test": len(test_addresses),
            }
        )
    return {
        "level": 5,
        "name": "Leave-One-Platform-Out",
        "description": (
            "For each category with >=10 addresses, hold out the entire category as "
            "the test set and train on all other categories. Tests cross-platform "
            "generalisation, the strongest generalisation claim in the benchmark."
        ),
        "min_platform_size": min_platform,
        "n_folds": len(folds),
        "folds": folds,
    }


def main() -> None:
    df, labels = load_dataset()
    print(f"Loaded {len(df)} addresses from {FEATURES.name}")

    out = {
        "all_addresses.json": {
            "n_addresses": len(df),
            "addresses": list(df.index),
            "labels": [int(x) for x in df["label"].values],
            "categories": list(df["category"].values),
            "sources": list(df["source"].values),
        },
        "level2_random_10fold.json": make_level2_random_cv(df),
        "level3_temporal_holdout.json": make_level3_temporal(df),
        "level4_strict_core.json": make_level4_strict(labels, df),
        "level5_lopo.json": make_level5_lopo(labels, df),
    }

    for name, payload in out.items():
        path = HERE / name
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"  wrote {path.name}")

    print("\nSummary")
    print(f"  Level 2 (random CV):   {out['level2_random_10fold.json']['n_addresses']} addresses, 10 folds")
    print(f"  Level 3 (temporal):    {out['level3_temporal_holdout.json']['n_train']} train / {out['level3_temporal_holdout.json']['n_test']} test")
    print(f"  Level 4 (strict):      {out['level4_strict_core.json']['n_addresses']} addresses ({out['level4_strict_core.json']['n_agents']}A / {out['level4_strict_core.json']['n_humans']}H)")
    print(f"  Level 5 (LOPO):        {out['level5_lopo.json']['n_folds']} platform folds")


if __name__ == "__main__":
    main()
