#!/usr/bin/env python
"""
Paper 1: Bootstrap 95% Confidence Intervals for Key AUC Numbers
================================================================
Computes bootstrap 95% CIs (BCa or percentile) on:
  - Level 2 (random 10-fold CV): RF AUC — bootstrap over out-of-fold predictions
  - Level 3 (temporal holdout): RF AUC — bootstrap over test-set predictions
  - Level 5 (mixed-class LOPO): RF pooled AUC — bootstrap over cluster predictions

Uses 1,000 bootstrap resamples, matching the same data loading and model
configurations as the existing experiment scripts.

Output: experiments/bootstrap_auc_ci_results.json
"""

import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
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
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "bootstrap_auc_ci_results.json"

FEATURE_GROUPS = {
    "temporal": [
        "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
        "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
        "burst_frequency",
    ],
    "gas": [
        "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
        "gas_limit_precision", "gas_price_cv",
        "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
    ],
    "interaction": [
        "unique_contracts_ratio", "top_contract_concentration",
        "method_id_diversity", "contract_to_eoa_ratio",
        "sequential_pattern_score",
    ],
    "approval_security": [
        "unlimited_approve_ratio", "approve_revoke_ratio",
        "unverified_contract_approve_ratio",
        "multi_protocol_interaction_count", "flash_loan_usage",
    ],
}

ALL_FEATURES = []
for gf in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(gf)

SEED = 42
N_BOOTSTRAP = 1000
N_CLUSTERS = 5


def get_rf():
    """Return the same RF config used across all experiments."""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )


def load_v4_data():
    """Load features_provenance_v4.parquet and prepare X, y, categories."""
    df = pd.read_parquet(FEATURES_PARQUET)
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

    return X, y, categories, feature_cols


def bootstrap_auc(y_true, y_score, n_bootstrap=N_BOOTSTRAP, seed=SEED):
    """Compute bootstrap 95% CI for AUC using percentile method.

    Returns (point_auc, ci_lo, ci_hi).
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    point_auc = roc_auc_score(y_true, y_score)

    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_t = y_true[idx]
        y_s = y_score[idx]

        # Need both classes in the resample
        if len(np.unique(y_t)) < 2:
            continue

        boot_aucs.append(roc_auc_score(y_t, y_s))

    boot_aucs = np.array(boot_aucs)
    ci_lo = float(np.percentile(boot_aucs, 2.5))
    ci_hi = float(np.percentile(boot_aucs, 97.5))

    return point_auc, ci_lo, ci_hi, len(boot_aucs)


# ------------------------------------------------------------------
# Level 2: Random 10-Fold CV — bootstrap over out-of-fold predictions
# ------------------------------------------------------------------

def level2_random_cv_bootstrap(X_np, y):
    """
    Run a single 10-fold stratified CV with RF, collect out-of-fold
    predictions for all samples, then bootstrap over those predictions.
    """
    logger.info("Level 2: Random 10-fold CV with RF...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
    model_template = get_rf()

    oof_probs = np.zeros(len(y))

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_np, y)):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr = y[train_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)
        oof_probs[test_idx] = clf.predict_proba(X_te)[:, 1]

    point_auc, ci_lo, ci_hi, n_valid = bootstrap_auc(y, oof_probs)

    logger.info("  RF AUC = %.4f  [%.4f, %.4f]  (n_valid_resamples=%d)",
                point_auc, ci_lo, ci_hi, n_valid)

    return {
        "level2_rf_auc": round(point_auc, 4),
        "level2_rf_auc_ci_lo": round(ci_lo, 4),
        "level2_rf_auc_ci_hi": round(ci_hi, 4),
        "level2_n_samples": int(len(y)),
        "level2_n_folds": 10,
        "level2_n_valid_resamples": n_valid,
    }


# ------------------------------------------------------------------
# Level 3: Temporal Holdout — bootstrap over test-set predictions
# ------------------------------------------------------------------

def extract_first_seen_block(addresses):
    """For each address, read raw .parquet and return min blockNumber."""
    records = []
    addr_lower_to_orig = {a.lower(): a for a in addresses}
    raw_files = {f.stem.lower(): f for f in RAW_DIR.glob("*.parquet")}

    for addr_lower, addr_orig in addr_lower_to_orig.items():
        raw_path = raw_files.get(addr_lower)
        if raw_path is None:
            candidate = RAW_DIR / f"{addr_orig}.parquet"
            if candidate.exists():
                raw_path = candidate
        if raw_path is None:
            continue
        try:
            df = pd.read_parquet(raw_path, columns=["blockNumber", "timeStamp"])
            blocks = pd.to_numeric(df["blockNumber"], errors="coerce")
            records.append({
                "address": addr_orig,
                "first_block": int(blocks.min()),
            })
        except Exception:
            continue

    return pd.DataFrame(records).set_index("address")


def level3_temporal_holdout_bootstrap(X, y):
    """
    Replicate the temporal holdout split from run_temporal_holdout.py,
    train RF on train split, predict on test split, then bootstrap
    over the test-set predictions.
    """
    logger.info("Level 3: Temporal holdout with RF...")

    # Load labels_provenance_v4.json for label_provenance field
    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)

    # Extract first-seen block
    temporal_df = extract_first_seen_block(list(X.index))
    common_addrs = X.index.intersection(temporal_df.index)
    logger.info("  Addresses with temporal data: %d / %d", len(common_addrs), len(X))

    X_common = X.loc[common_addrs]
    temporal_df = temporal_df.loc[common_addrs]

    # Re-derive y using label_provenance (same as run_temporal_holdout.py)
    y_aligned = np.array([
        labels_dict[addr]["label_provenance"] for addr in common_addrs
    ])

    # Median first_block split
    median_block = int(temporal_df["first_block"].median())
    logger.info("  Median first_block: %d", median_block)

    train_mask = temporal_df["first_block"] < median_block
    test_mask = temporal_df["first_block"] >= median_block

    X_train = X_common.loc[train_mask].values.astype(float)
    X_test = X_common.loc[test_mask].values.astype(float)
    y_train = y_aligned[train_mask.values]
    y_test = y_aligned[test_mask.values]

    logger.info("  Train: n=%d (agents=%d, humans=%d)",
                len(y_train), int(y_train.sum()), int((y_train == 0).sum()))
    logger.info("  Test:  n=%d (agents=%d, humans=%d)",
                len(y_test), int(y_test.sum()), int((y_test == 0).sum()))

    # Train RF and predict on test
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    clf = clone(get_rf())
    clf.fit(X_tr_s, y_train)
    y_prob_test = clf.predict_proba(X_te_s)[:, 1]

    # Bootstrap over test predictions
    point_auc, ci_lo, ci_hi, n_valid = bootstrap_auc(y_test, y_prob_test)

    logger.info("  RF AUC = %.4f  [%.4f, %.4f]  (n_valid_resamples=%d)",
                point_auc, ci_lo, ci_hi, n_valid)

    return {
        "level3_rf_auc": round(point_auc, 4),
        "level3_rf_auc_ci_lo": round(ci_lo, 4),
        "level3_rf_auc_ci_hi": round(ci_hi, 4),
        "level3_n_train": int(len(y_train)),
        "level3_n_test": int(len(y_test)),
        "level3_split_block": median_block,
        "level3_n_valid_resamples": n_valid,
    }


# ------------------------------------------------------------------
# Level 5: Mixed-class LOPO — bootstrap over pooled cluster predictions
# ------------------------------------------------------------------

def level5_mixed_class_lopo_bootstrap(X_np, y):
    """
    Replicate the mixed-class cluster LOPO from mixed_class_lopo.py,
    collect pooled out-of-cluster predictions, then bootstrap over them.
    """
    logger.info("Level 5: Mixed-class cluster LOPO with RF...")

    # Cluster with K-Means (same as mixed_class_lopo.py)
    scaler_cluster = StandardScaler()
    X_scaled = scaler_cluster.fit_transform(X_np)

    km = KMeans(n_clusters=N_CLUSTERS, n_init=20, random_state=SEED)
    cluster_labels = km.fit_predict(X_scaled)

    # Log cluster composition
    for c in range(N_CLUSTERS):
        mask = cluster_labels == c
        n_agents = int(y[mask].sum())
        n_humans = int((y[mask] == 0).sum())
        logger.info("  Cluster %d: n=%d (agents=%d, humans=%d)",
                     c, int(mask.sum()), n_agents, n_humans)

    # Leave-one-cluster-out, collect pooled predictions
    model_template = get_rf()
    all_y_true = []
    all_y_prob = []

    for c in sorted(np.unique(cluster_labels)):
        test_mask = cluster_labels == c
        train_mask = ~test_mask

        X_train, X_test = X_np[train_mask], X_np[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        # Need both classes in train
        if len(np.unique(y_train)) < 2:
            logger.warning("  Cluster %d: skipping (single-class train)", c)
            continue

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)

        clf = clone(model_template)
        clf.fit(X_tr_s, y_train)
        y_prob = clf.predict_proba(X_te_s)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_prob.extend(y_prob.tolist())

    all_y_true = np.array(all_y_true)
    all_y_prob = np.array(all_y_prob)

    logger.info("  Pooled predictions: n=%d (agents=%d, humans=%d)",
                len(all_y_true), int(all_y_true.sum()),
                int((all_y_true == 0).sum()))

    # Bootstrap over pooled predictions
    point_auc, ci_lo, ci_hi, n_valid = bootstrap_auc(all_y_true, all_y_prob)

    logger.info("  RF pooled AUC = %.4f  [%.4f, %.4f]  (n_valid_resamples=%d)",
                point_auc, ci_lo, ci_hi, n_valid)

    return {
        "level5_mixed_rf_auc": round(point_auc, 4),
        "level5_mixed_rf_auc_ci_lo": round(ci_lo, 4),
        "level5_mixed_rf_auc_ci_hi": round(ci_hi, 4),
        "level5_n_clusters": N_CLUSTERS,
        "level5_n_pooled_predictions": int(len(all_y_true)),
        "level5_n_valid_resamples": n_valid,
    }


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("Paper 1: Bootstrap 95%% CIs for Key AUC Numbers")
    logger.info("Timestamp: %s", timestamp)
    logger.info("N_BOOTSTRAP = %d", N_BOOTSTRAP)
    logger.info("=" * 70)

    # Load data
    X, y, categories, feature_cols = load_v4_data()
    X_np = X.values.astype(float)
    logger.info("Dataset: n=%d, agents=%d, humans=%d, features=%d",
                len(y), int(y.sum()), int((y == 0).sum()), len(feature_cols))

    results = {
        "run_timestamp": timestamp,
        "description": (
            "Bootstrap 95% confidence intervals (percentile method, "
            "1000 resamples) for key AUC numbers reported in Paper 1. "
            "Level 2 = random 10-fold CV; Level 3 = temporal holdout; "
            "Level 5 = mixed-class behavioral cluster LOPO."
        ),
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(feature_cols),
        },
    }

    # Level 2: Random 10-fold CV
    logger.info("")
    level2 = level2_random_cv_bootstrap(X_np, y)
    results.update(level2)

    # Level 3: Temporal holdout
    logger.info("")
    level3 = level3_temporal_holdout_bootstrap(X, y)
    results.update(level3)

    # Level 5: Mixed-class LOPO
    logger.info("")
    level5 = level5_mixed_class_lopo_bootstrap(X_np, y)
    results.update(level5)

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    # Save
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("")
    logger.info("Results saved to %s", OUTPUT_JSON)

    # Summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info("Level 2 (random 10-fold CV):     RF AUC = %.4f  [%.4f, %.4f]",
                results["level2_rf_auc"],
                results["level2_rf_auc_ci_lo"],
                results["level2_rf_auc_ci_hi"])
    logger.info("Level 3 (temporal holdout):       RF AUC = %.4f  [%.4f, %.4f]",
                results["level3_rf_auc"],
                results["level3_rf_auc_ci_lo"],
                results["level3_rf_auc_ci_hi"])
    logger.info("Level 5 (mixed-class LOPO):       RF AUC = %.4f  [%.4f, %.4f]",
                results["level5_mixed_rf_auc"],
                results["level5_mixed_rf_auc_ci_lo"],
                results["level5_mixed_rf_auc_ci_hi"])
    logger.info("=" * 70)
    logger.info("Elapsed: %.1f s", results["elapsed_seconds"])

    return results


if __name__ == "__main__":
    main()
