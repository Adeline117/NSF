"""
Paper 1: External Baseline — Niedermayer et al. (2024)
======================================================
Implements the method from:
  Niedermayer, Kitzler, Tovanich & Khorrami (2024).
  "Detecting Financial Bots on the Ethereum Blockchain."
  WWW Companion '24.

Their approach: 7 hand-crafted features + Random Forest
(n_estimators=100, max_depth=10).

Feature mapping (Niedermayer -> our dataset):
  1. tx_frequency          ≈ 1 / tx_interval_mean
  2. gas_price_std         ≈ gas_price_cv
  3. unique_counterparties ≈ unique_contracts_ratio
  4. active_hours          ≈ active_hour_entropy
  5. burst_ratio           ≈ burst_frequency
  6. value_std             ≈ gas_price_trailing_zeros_mean  (proxy)
  7. contract_interaction_ratio ≈ contract_to_eoa_ratio

Evaluation levels:
  - Level 2: 10-fold stratified CV
  - Level 3: Temporal holdout (train pre-median-block, test post)
  - Level 5: Leave-one-platform-out (LOPO-CV)

Each level also trains our full RF (200 trees, max_depth=None,
23 features) for direct comparison.

Output: experiments/external_baseline_niedermayer_results.json
"""

import json
import logging
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_PARQUET = DATA_DIR / "features_provenance_v4.parquet"
LABELS_JSON = DATA_DIR / "labels_provenance_v4.json"
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "external_baseline_niedermayer_results.json"

# ── Full 23 features (same groups as main pipeline) ────────────────
TEMPORAL_FEATURES = [
    "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
    "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
    "burst_frequency",
]
GAS_FEATURES = [
    "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
    "gas_limit_precision", "gas_price_cv",
    "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
]
INTERACTION_FEATURES = [
    "unique_contracts_ratio", "top_contract_concentration",
    "method_id_diversity", "contract_to_eoa_ratio",
    "sequential_pattern_score",
]
APPROVAL_FEATURES = [
    "unlimited_approve_ratio", "approve_revoke_ratio",
    "unverified_contract_approve_ratio",
    "multi_protocol_interaction_count", "flash_loan_usage",
]
ALL_FEATURES = (
    TEMPORAL_FEATURES + GAS_FEATURES + INTERACTION_FEATURES + APPROVAL_FEATURES
)

# ── Niedermayer et al. (2024): 7 features mapped to ours ──────────
# Mapping rationale in the docstring above.
NIEDERMAYER_FEATURE_MAP = {
    "tx_frequency":                "tx_interval_mean",      # we invert below
    "gas_price_std":               "gas_price_cv",
    "unique_counterparties":       "unique_contracts_ratio",
    "active_hours":                "active_hour_entropy",
    "burst_ratio":                 "burst_frequency",
    "value_std":                   "gas_price_trailing_zeros_mean",
    "contract_interaction_ratio":  "contract_to_eoa_ratio",
}
NIEDERMAYER_OUR_COLS = list(NIEDERMAYER_FEATURE_MAP.values())

# ── Model configs ──────────────────────────────────────────────────
# Niedermayer: RF(100, max_depth=10)
NIEDERMAYER_RF = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
)
# Ours: RF(200, max_depth=None) on all 23 features
OUR_RF = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    random_state=42,
    n_jobs=-1,
)


# ====================================================================
# DATA LOADING
# ====================================================================

def load_dataset() -> tuple[pd.DataFrame, np.ndarray, list[str], pd.Series]:
    """Load n=1,147 provenance-labeled dataset.

    Returns (X_full, y, feature_names, categories).
    X_full has all 23 features; caller selects subsets.
    """
    logger.info("Loading %s ...", FEATURES_PARQUET)
    df = pd.read_parquet(FEATURES_PARQUET)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df["label"].values.astype(int)
    categories = df["category"].copy()

    # Impute NaN with column medians
    X_vals = X.values.astype(float)
    nan_mask = np.isnan(X_vals)
    if nan_mask.any():
        col_medians = np.nanmedian(X_vals, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X_vals.shape[1]):
            X_vals[nan_mask[:, j], j] = col_medians[j]
        X = pd.DataFrame(X_vals, columns=feature_cols, index=df.index)

    # Clip extremes at 1st/99th percentile
    for col in X.columns:
        lo, hi = np.nanpercentile(X[col], [1, 99])
        X[col] = X[col].clip(lo, hi)

    logger.info("Features: %d  |  Agents: %d  |  Humans: %d",
                len(feature_cols), int(y.sum()), int((y == 0).sum()))
    return X, y, feature_cols, categories


def prepare_niedermayer_features(X_full: pd.DataFrame) -> pd.DataFrame:
    """Extract and transform the 7 Niedermayer features from our full set.

    tx_frequency = 1 / tx_interval_mean  (inverted; clipped to avoid inf)
    All others are direct column references.
    """
    X_nied = pd.DataFrame(index=X_full.index)

    # tx_frequency = 1 / tx_interval_mean
    interval = X_full["tx_interval_mean"].values.astype(float)
    # Avoid division by zero: replace 0 with a small epsilon
    interval_safe = np.where(interval < 1e-8, 1e-8, interval)
    X_nied["tx_frequency"] = 1.0 / interval_safe

    # Remaining 6 features: direct mappings
    X_nied["gas_price_std"] = X_full["gas_price_cv"].values
    X_nied["unique_counterparties"] = X_full["unique_contracts_ratio"].values
    X_nied["active_hours"] = X_full["active_hour_entropy"].values
    X_nied["burst_ratio"] = X_full["burst_frequency"].values
    X_nied["value_std"] = X_full["gas_price_trailing_zeros_mean"].values
    X_nied["contract_interaction_ratio"] = X_full["contract_to_eoa_ratio"].values

    return X_nied


def extract_first_seen_block(addresses: list[str]) -> pd.DataFrame:
    """For each address, read its raw .parquet and return the minimum
    blockNumber for temporal split."""
    records = []
    addr_lower_to_orig = {a.lower(): a for a in addresses}
    raw_files = {f.stem.lower(): f for f in RAW_DIR.glob("*.parquet")}

    found, missing = 0, 0
    for addr_lower, addr_orig in addr_lower_to_orig.items():
        raw_path = raw_files.get(addr_lower)
        if raw_path is None:
            candidate = RAW_DIR / f"{addr_orig}.parquet"
            if candidate.exists():
                raw_path = candidate
        if raw_path is None:
            missing += 1
            continue
        try:
            df_raw = pd.read_parquet(raw_path, columns=["blockNumber"])
            blocks = pd.to_numeric(df_raw["blockNumber"], errors="coerce")
            records.append({
                "address": addr_orig,
                "first_block": int(blocks.min()),
            })
            found += 1
        except Exception:
            missing += 1

    logger.info("Extracted first-seen block for %d/%d addresses",
                found, found + missing)
    return pd.DataFrame(records).set_index("address")


# ====================================================================
# EVALUATION HELPERS
# ====================================================================

def train_eval(X_train: np.ndarray, y_train: np.ndarray,
               X_test: np.ndarray, y_test: np.ndarray,
               model_template) -> dict:
    """Train a model, predict on test set, return metrics."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = clone(model_template)
    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)[:, 1]

    if len(np.unique(y_test)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_test, y_prob)

    return {
        "auc": round(float(auc), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    }


# ====================================================================
# LEVEL 2: 10-Fold Stratified CV
# ====================================================================

def eval_10fold_cv(X_full: pd.DataFrame, X_nied: pd.DataFrame,
                   y: np.ndarray) -> dict:
    """10-fold stratified CV for both Niedermayer and our full RF."""
    logger.info("=" * 70)
    logger.info("Level 2: 10-Fold Stratified CV")
    logger.info("=" * 70)

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

    methods = {
        "Niedermayer et al. (2024) RF(100, depth=10, 7 feat)": {
            "X": X_nied.values.astype(float),
            "model": NIEDERMAYER_RF,
            "aucs": [], "f1s": [], "precs": [], "recs": [], "accs": [],
        },
        "Ours: RF(200, depth=None, 23 feat)": {
            "X": X_full.values.astype(float),
            "model": OUR_RF,
            "aucs": [], "f1s": [], "precs": [], "recs": [], "accs": [],
        },
    }

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_full.values, y)):
        y_train, y_test = y[train_idx], y[test_idx]
        if len(np.unique(y_test)) < 2:
            continue

        for mname, mdata in methods.items():
            X_m = mdata["X"]
            m = train_eval(X_m[train_idx], y_train,
                           X_m[test_idx], y_test,
                           mdata["model"])
            mdata["aucs"].append(m["auc"])
            mdata["f1s"].append(m["f1"])
            mdata["precs"].append(m["precision"])
            mdata["recs"].append(m["recall"])
            mdata["accs"].append(m["accuracy"])

    summary = {}
    for mname, mdata in methods.items():
        summary[mname] = {
            "auc_mean": round(float(np.mean(mdata["aucs"])), 4),
            "auc_std": round(float(np.std(mdata["aucs"])), 4),
            "f1_mean": round(float(np.mean(mdata["f1s"])), 4),
            "f1_std": round(float(np.std(mdata["f1s"])), 4),
            "precision_mean": round(float(np.mean(mdata["precs"])), 4),
            "recall_mean": round(float(np.mean(mdata["recs"])), 4),
            "accuracy_mean": round(float(np.mean(mdata["accs"])), 4),
            "n_folds": len(mdata["aucs"]),
        }
        logger.info("  %-55s AUC=%.4f +/- %.4f  F1=%.4f +/- %.4f",
                     mname,
                     summary[mname]["auc_mean"], summary[mname]["auc_std"],
                     summary[mname]["f1_mean"], summary[mname]["f1_std"])

    return summary


# ====================================================================
# LEVEL 3: Temporal Holdout
# ====================================================================

def eval_temporal_holdout(X_full: pd.DataFrame, X_nied: pd.DataFrame,
                          y: np.ndarray) -> tuple[dict, dict]:
    """Temporal holdout: train on pre-median-block, test on post."""
    logger.info("=" * 70)
    logger.info("Level 3: Temporal Holdout")
    logger.info("=" * 70)

    addresses = list(X_full.index)
    temporal_df = extract_first_seen_block(addresses)
    common = X_full.index.intersection(temporal_df.index)

    X_full_c = X_full.loc[common]
    X_nied_c = X_nied.loc[common]

    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)
    y_common = np.array([
        labels_dict[addr]["label_provenance"] for addr in common
    ])
    temporal_df = temporal_df.loc[common]

    median_block = int(temporal_df["first_block"].median())
    train_mask = temporal_df["first_block"] < median_block
    test_mask = temporal_df["first_block"] >= median_block

    y_train = y_common[train_mask.values]
    y_test = y_common[test_mask.values]

    logger.info("  Temporal split: train=%d (agents=%d) | test=%d (agents=%d) | median_block=%d",
                len(y_train), int(y_train.sum()),
                len(y_test), int(y_test.sum()), median_block)

    metadata = {
        "n_total": int(len(y_common)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "split_block": median_block,
        "train_agents": int(y_train.sum()),
        "train_humans": int((y_train == 0).sum()),
        "test_agents": int(y_test.sum()),
        "test_humans": int((y_test == 0).sum()),
    }

    results = {}

    # Niedermayer
    X_nied_train = X_nied_c.loc[train_mask].values.astype(float)
    X_nied_test = X_nied_c.loc[test_mask].values.astype(float)
    nied_name = "Niedermayer et al. (2024) RF(100, depth=10, 7 feat)"
    results[nied_name] = train_eval(
        X_nied_train, y_train, X_nied_test, y_test, NIEDERMAYER_RF)

    # Ours
    X_full_train = X_full_c.loc[train_mask].values.astype(float)
    X_full_test = X_full_c.loc[test_mask].values.astype(float)
    ours_name = "Ours: RF(200, depth=None, 23 feat)"
    results[ours_name] = train_eval(
        X_full_train, y_train, X_full_test, y_test, OUR_RF)

    for mname, m in results.items():
        logger.info("  %-55s AUC=%.4f  F1=%.4f", mname, m["auc"], m["f1"])

    return results, metadata


# ====================================================================
# LEVEL 5: Leave-One-Platform-Out (LOPO-CV)
# ====================================================================

def eval_lopo(X_full: pd.DataFrame, X_nied: pd.DataFrame,
              y: np.ndarray, categories: pd.Series) -> dict:
    """LOPO-CV: for each platform with >= 10 addresses, hold it out
    entirely, train on all others, predict on held-out.
    Pool all held-out predictions for a single AUC/F1."""
    logger.info("=" * 70)
    logger.info("Level 5: Leave-One-Platform-Out")
    logger.info("=" * 70)

    cat_counts = Counter(categories)
    min_size = 10

    method_configs = {
        "Niedermayer et al. (2024) RF(100, depth=10, 7 feat)": {
            "X": X_nied.values.astype(float),
            "model": NIEDERMAYER_RF,
        },
        "Ours: RF(200, depth=None, 23 feat)": {
            "X": X_full.values.astype(float),
            "model": OUR_RF,
        },
    }

    pooled = {
        mname: {"y_true": [], "y_score": [], "y_pred": []}
        for mname in method_configs
    }
    per_platform = {mname: {} for mname in method_configs}

    for cat in sorted(cat_counts.keys()):
        if cat_counts[cat] < min_size:
            continue

        test_mask = (categories == cat).values
        train_mask = ~test_mask
        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        for mname, mcfg in method_configs.items():
            X_m = mcfg["X"]
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_m[train_mask])
            X_te = scaler.transform(X_m[test_mask])

            clf = clone(mcfg["model"])
            clf.fit(X_tr, y_train)
            y_prob = clf.predict_proba(X_te)[:, 1]
            y_pred = clf.predict(X_te)

            pooled[mname]["y_true"].extend(y_test.tolist())
            pooled[mname]["y_score"].extend(y_prob.tolist())
            pooled[mname]["y_pred"].extend(y_pred.tolist())

            # Per-platform metrics (when both classes exist)
            both = len(np.unique(y_test)) == 2
            plat_auc = round(float(roc_auc_score(y_test, y_prob)), 4) if both else None
            per_platform[mname][cat] = {
                "n": int(cat_counts[cat]),
                "agents": int(y_test.sum()),
                "humans": int((y_test == 0).sum()),
                "auc": plat_auc,
                "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
                "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            }

        logger.info("  Platform %-40s n=%3d  agents=%3d  humans=%3d",
                     cat, cat_counts[cat],
                     int(y_test.sum()), int((y_test == 0).sum()))

    # Compute pooled metrics
    results = {}
    for mname in method_configs:
        yt = np.array(pooled[mname]["y_true"])
        ys = np.array(pooled[mname]["y_score"])
        yp = np.array(pooled[mname]["y_pred"])

        if len(np.unique(yt)) < 2:
            auc_val = float("nan")
        else:
            auc_val = roc_auc_score(yt, ys)

        results[mname] = {
            "pooled_auc": round(float(auc_val), 4),
            "pooled_f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "pooled_precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "pooled_recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "pooled_accuracy": round(float(accuracy_score(yt, yp)), 4),
            "n_predictions": int(len(yt)),
            "per_platform": per_platform[mname],
        }
        logger.info("  %-55s Pooled AUC=%.4f  F1=%.4f",
                     mname,
                     results[mname]["pooled_auc"],
                     results[mname]["pooled_f1"])

    return results


# ====================================================================
# MAIN
# ====================================================================

def main():
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("External Baseline: Niedermayer et al. (2024)")
    logger.info("  'Detecting Financial Bots on the Ethereum Blockchain'")
    logger.info("  WWW Companion '24")
    logger.info("=" * 70)

    # Load data
    X_full, y, feature_names, categories = load_dataset()
    X_nied = prepare_niedermayer_features(X_full)

    logger.info("Niedermayer features (%d): %s",
                len(X_nied.columns), list(X_nied.columns))
    logger.info("Our features (%d): %s",
                len(feature_names), feature_names)

    results = {
        "run_timestamp": timestamp,
        "experiment": "external_baseline_niedermayer_2024",
        "reference": (
            "Niedermayer, Kitzler, Tovanich & Khorrami (2024). "
            "'Detecting Financial Bots on the Ethereum Blockchain.' "
            "WWW Companion '24."
        ),
        "feature_mapping": {
            nied_feat: our_feat
            for nied_feat, our_feat in NIEDERMAYER_FEATURE_MAP.items()
        },
        "feature_mapping_notes": {
            "tx_frequency": "Computed as 1/tx_interval_mean (inverted)",
            "gas_price_std": "Mapped to gas_price_cv (coefficient of variation)",
            "unique_counterparties": "Mapped to unique_contracts_ratio",
            "active_hours": "Mapped to active_hour_entropy",
            "burst_ratio": "Mapped to burst_frequency",
            "value_std": "Mapped to gas_price_trailing_zeros_mean (proxy: "
                         "no direct value_std available; trailing zeros "
                         "captures value-precision patterns)",
            "contract_interaction_ratio": "Mapped to contract_to_eoa_ratio",
        },
        "models": {
            "niedermayer": "RF(n_estimators=100, max_depth=10) on 7 features",
            "ours": "RF(n_estimators=200, max_depth=None) on 23 features",
        },
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features_ours": len(feature_names),
            "n_features_niedermayer": len(X_nied.columns),
        },
    }

    # ── Level 2: 10-Fold CV ──
    cv_results = eval_10fold_cv(X_full, X_nied, y)
    results["level2_10fold_cv"] = cv_results

    # ── Level 3: Temporal Holdout ──
    temporal_results, temporal_metadata = eval_temporal_holdout(X_full, X_nied, y)
    results["level3_temporal_holdout"] = temporal_results
    results["level3_temporal_metadata"] = temporal_metadata

    # ── Level 5: LOPO ──
    lopo_results = eval_lopo(X_full, X_nied, y, categories)
    results["level5_lopo"] = lopo_results

    # ── Summary Table ──
    nied_name = "Niedermayer et al. (2024) RF(100, depth=10, 7 feat)"
    ours_name = "Ours: RF(200, depth=None, 23 feat)"

    logger.info("\n" + "=" * 90)
    logger.info("COMPARISON TABLE: Niedermayer (2024) vs. Ours")
    logger.info("=" * 90)

    header = f"{'Method':<58} {'10-fold CV':>12} {'Temporal':>12} {'LOPO':>12}"
    logger.info(header)
    logger.info("-" * 95)

    comparison_table = []
    for mname in [nied_name, ours_name]:
        cv_auc = cv_results[mname]["auc_mean"]
        cv_std = cv_results[mname]["auc_std"]
        temp_auc = temporal_results[mname]["auc"]
        lopo_auc = lopo_results[mname]["pooled_auc"]

        row = {
            "method": mname,
            "cv_10fold_auc": cv_auc,
            "cv_10fold_auc_std": cv_std,
            "temporal_holdout_auc": temp_auc,
            "lopo_pooled_auc": lopo_auc,
        }
        comparison_table.append(row)

        logger.info(
            "%-58s %5.3f+/-%.3f  %10.4f  %10.4f",
            mname, cv_auc, cv_std, temp_auc, lopo_auc,
        )

    results["comparison_table_auc"] = comparison_table

    # F1 table
    logger.info("\n" + "=" * 90)
    logger.info("COMPARISON TABLE: F1 Scores")
    logger.info("=" * 90)

    f1_table = []
    for mname in [nied_name, ours_name]:
        cv_f1 = cv_results[mname]["f1_mean"]
        cv_f1_std = cv_results[mname]["f1_std"]
        temp_f1 = temporal_results[mname]["f1"]
        lopo_f1 = lopo_results[mname]["pooled_f1"]

        row = {
            "method": mname,
            "cv_10fold_f1": cv_f1,
            "cv_10fold_f1_std": cv_f1_std,
            "temporal_holdout_f1": temp_f1,
            "lopo_pooled_f1": lopo_f1,
        }
        f1_table.append(row)

        logger.info(
            "%-58s %5.3f+/-%.3f  %10.4f  %10.4f",
            mname, cv_f1, cv_f1_std, temp_f1, lopo_f1,
        )

    results["comparison_table_f1"] = f1_table

    # ── Delta analysis ──
    delta = {
        "cv_auc_delta": round(
            cv_results[ours_name]["auc_mean"] - cv_results[nied_name]["auc_mean"], 4),
        "temporal_auc_delta": round(
            temporal_results[ours_name]["auc"] - temporal_results[nied_name]["auc"], 4),
        "lopo_auc_delta": round(
            lopo_results[ours_name]["pooled_auc"] - lopo_results[nied_name]["pooled_auc"], 4),
        "cv_f1_delta": round(
            cv_results[ours_name]["f1_mean"] - cv_results[nied_name]["f1_mean"], 4),
        "temporal_f1_delta": round(
            temporal_results[ours_name]["f1"] - temporal_results[nied_name]["f1"], 4),
        "lopo_f1_delta": round(
            lopo_results[ours_name]["pooled_f1"] - lopo_results[nied_name]["pooled_f1"], 4),
    }
    results["deltas_ours_minus_niedermayer"] = delta

    logger.info("\n  DELTA (Ours - Niedermayer):")
    logger.info("    10-fold CV  AUC: %+.4f   F1: %+.4f",
                delta["cv_auc_delta"], delta["cv_f1_delta"])
    logger.info("    Temporal    AUC: %+.4f   F1: %+.4f",
                delta["temporal_auc_delta"], delta["temporal_f1_delta"])
    logger.info("    LOPO        AUC: %+.4f   F1: %+.4f",
                delta["lopo_auc_delta"], delta["lopo_f1_delta"])

    # ── Key finding ──
    results["key_finding"] = (
        f"Niedermayer et al. (2024) achieves "
        f"{cv_results[nied_name]['auc_mean']:.3f} AUC (10-fold CV), "
        f"{temporal_results[nied_name]['auc']:.3f} AUC (temporal), "
        f"{lopo_results[nied_name]['pooled_auc']:.3f} AUC (LOPO). "
        f"Our 23-feature RF improves by "
        f"{delta['cv_auc_delta']:+.3f} / "
        f"{delta['temporal_auc_delta']:+.3f} / "
        f"{delta['lopo_auc_delta']:+.3f} AUC respectively. "
        f"The additional 16 features (gas precision, approval patterns, "
        f"interaction diversity) capture complementary behavioral signals "
        f"that the Niedermayer 7-feature set does not."
    )

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    # ── Save ──
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_safe = json.loads(json.dumps(results, default=_convert))
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_safe, f, indent=2)
    logger.info("\nResults saved to %s", OUTPUT_JSON)

    logger.info("=" * 70)
    logger.info("Niedermayer baseline comparison complete in %.1fs",
                results["elapsed_seconds"])
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
