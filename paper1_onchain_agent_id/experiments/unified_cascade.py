"""
Paper 1: Unified Cascade — ALL five levels with the SAME model (RF)
on the SAME dataset (v4, n=1,147).

FATAL-1 fix: The original cascade mixed GBM/RF, old 3316 dataset /
new 1147 v4 dataset, and 64-address subset vs 70-address strict core.
This script computes ALL levels consistently.

Levels:
  1. C1-C4 leaky labels on v4 — RF trained to predict the LEAKY
     heuristic labels. Shows how easy it is to fit heuristic noise.
  2. Provenance labels, 10-fold CV — the HONEST random baseline.
  3. Temporal holdout — train < median block, test >= median block.
  4. Strict curated core (N=70) — LOO-CV on hand-verified addresses.
  5. Leave-One-Platform-Out (LOPO) — pooled AUC across held-out platforms.

All five levels use:
  - RandomForestClassifier(n_estimators=200, max_depth=4,
      min_samples_leaf=3, random_state=42)
  - 23 behavioral features
  - StandardScaler
  - Same preprocessing (median impute, 1/99 percentile clip)

Output: experiments/unified_cascade_results.json
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
from sklearn.model_selection import (
    LeaveOneOut,
    RepeatedStratifiedKFold,
    StratifiedKFold,
)
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
STRICT_CORE_JSON = PROJECT_ROOT / "benchmark" / "splits" / "level4_strict_core.json"
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "unified_cascade_results.json"

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

RF_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)


# ============================================================
# DATA LOADING
# ============================================================

def load_v4_data():
    """Load v4 features, provenance labels, and C1-C4 leaky labels."""
    logger.info("Loading %s ...", FEATURES_PARQUET)
    df = pd.read_parquet(FEATURES_PARQUET)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    feature_cols = [c for c in FEATURE_COLS if c in df.columns]
    X = df[feature_cols].copy()
    y_provenance = df["label"].values.astype(int)
    categories = df["category"].copy()

    # C1-C4 leaky labels
    c1c4_is_agent = df["c1c4_is_agent"].copy()
    y_leaky = c1c4_is_agent.map({True: 1, False: 0}).values
    leaky_valid_mask = ~np.isnan(y_leaky.astype(float))

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

    return df, X, y_provenance, y_leaky, leaky_valid_mask, categories


# ============================================================
# LEVEL 1: LEAKY C1-C4 LABELS
# ============================================================

def run_level1(X, y_leaky, leaky_valid_mask):
    """RF 10-fold CV predicting the C1-C4 heuristic labels."""
    logger.info("=" * 65)
    logger.info("LEVEL 1: C1-C4 leaky labels, RF 10-fold CV")
    logger.info("=" * 65)

    X_valid = X.values[leaky_valid_mask].astype(float)
    y_valid = y_leaky[leaky_valid_mask].astype(int)

    n_agents = int(y_valid.sum())
    n_humans = int((y_valid == 0).sum())
    logger.info("  n=%d (agents=%d, humans=%d)", len(y_valid), n_agents, n_humans)

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    aucs, f1s = [], []

    for train_idx, test_idx in rskf.split(X_valid, y_valid):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_valid[train_idx])
        X_te = scaler.transform(X_valid[test_idx])

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_valid[train_idx])

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)

        aucs.append(roc_auc_score(y_valid[test_idx], y_prob))
        f1s.append(f1_score(y_valid[test_idx], y_pred, zero_division=0))

    result = {
        "level": 1,
        "name": "C1-C4 leaky labels (RF 10x10-fold CV)",
        "dataset": "v4 n=1,147 subset with C1-C4 labels",
        "n_samples": int(len(y_valid)),
        "n_agents": n_agents,
        "n_humans": n_humans,
        "model": "RandomForest",
        "auc": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "f1": round(float(np.mean(f1s)), 4),
        "note": "Leaky labels: heuristic C1-C4 rules, NOT ground truth.",
    }

    logger.info("  AUC = %.4f +/- %.4f", result["auc"], result["auc_std"])
    return result


# ============================================================
# LEVEL 2: HONEST 10-FOLD CV (PROVENANCE LABELS)
# ============================================================

def run_level2(X, y):
    """RF repeated 10-fold stratified CV on provenance labels."""
    logger.info("=" * 65)
    logger.info("LEVEL 2: Provenance labels, RF 10x10-fold CV")
    logger.info("=" * 65)

    X_np = X.values.astype(float)
    n_agents = int(y.sum())
    n_humans = int((y == 0).sum())
    logger.info("  n=%d (agents=%d, humans=%d)", len(y), n_agents, n_humans)

    rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=10, random_state=42)
    aucs, f1s, precs, recs, accs = [], [], [], [], []

    for train_idx, test_idx in rskf.split(X_np, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_np[train_idx])
        X_te = scaler.transform(X_np[test_idx])

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y[train_idx])

        y_prob = clf.predict_proba(X_te)[:, 1]
        y_pred = clf.predict(X_te)

        aucs.append(roc_auc_score(y[test_idx], y_prob))
        f1s.append(f1_score(y[test_idx], y_pred, zero_division=0))
        precs.append(precision_score(y[test_idx], y_pred, zero_division=0))
        recs.append(recall_score(y[test_idx], y_pred, zero_division=0))
        accs.append(accuracy_score(y[test_idx], y_pred))

    result = {
        "level": 2,
        "name": "Provenance labels, RF 10x10-fold CV",
        "dataset": "v4 n=1,147",
        "n_samples": int(len(y)),
        "n_agents": n_agents,
        "n_humans": n_humans,
        "model": "RandomForest",
        "auc": round(float(np.mean(aucs)), 4),
        "auc_std": round(float(np.std(aucs)), 4),
        "f1": round(float(np.mean(f1s)), 4),
        "precision": round(float(np.mean(precs)), 4),
        "recall": round(float(np.mean(recs)), 4),
        "accuracy": round(float(np.mean(accs)), 4),
    }

    logger.info("  AUC = %.4f +/- %.4f  F1 = %.4f", result["auc"],
                result["auc_std"], result["f1"])
    return result


# ============================================================
# LEVEL 3: TEMPORAL HOLDOUT
# ============================================================

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
            txs = pd.read_parquet(raw_path, columns=["blockNumber"])
            blocks = pd.to_numeric(txs["blockNumber"], errors="coerce")
            records.append({
                "address": addr_orig,
                "first_block": int(blocks.min()),
            })
        except Exception:
            continue

    return pd.DataFrame(records).set_index("address")


def run_level3(X, y):
    """RF temporal holdout: train on early addresses, test on later ones."""
    logger.info("=" * 65)
    logger.info("LEVEL 3: Temporal holdout (RF)")
    logger.info("=" * 65)

    temporal_df = extract_first_seen_block(list(X.index))
    common = X.index.intersection(temporal_df.index)
    logger.info("  Addresses with temporal data: %d / %d", len(common), len(X))

    X_sub = X.loc[common]
    # Reload labels aligned
    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)
    y_sub = np.array([labels_dict[a]["label_provenance"] for a in common])
    temporal_df = temporal_df.loc[common]

    median_block = int(temporal_df["first_block"].median())
    train_mask = temporal_df["first_block"] < median_block
    test_mask = temporal_df["first_block"] >= median_block

    X_train = X_sub.loc[train_mask].values.astype(float)
    X_test = X_sub.loc[test_mask].values.astype(float)
    y_train = y_sub[train_mask.values]
    y_test = y_sub[test_mask.values]

    logger.info("  Train: n=%d (agents=%d, humans=%d)",
                len(y_train), int(y_train.sum()), int((y_train == 0).sum()))
    logger.info("  Test:  n=%d (agents=%d, humans=%d)",
                len(y_test), int(y_test.sum()), int((y_test == 0).sum()))

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = RandomForestClassifier(**RF_PARAMS)
    clf.fit(X_tr, y_train)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = clf.predict(X_te)

    result = {
        "level": 3,
        "name": "Temporal holdout (RF)",
        "dataset": "v4 n=1,147",
        "n_total": int(len(y_sub)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "split_block": median_block,
        "train_agents": int(y_train.sum()),
        "train_humans": int((y_train == 0).sum()),
        "test_agents": int(y_test.sum()),
        "test_humans": int((y_test == 0).sum()),
        "model": "RandomForest",
        "auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        "f1": round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    }

    logger.info("  AUC = %.4f  F1 = %.4f", result["auc"], result["f1"])
    return result


# ============================================================
# LEVEL 4: STRICT CURATED CORE (N=70) — LOO-CV
# ============================================================

def run_level4(X, df):
    """RF LOO-CV on the 70-address strict curated core."""
    logger.info("=" * 65)
    logger.info("LEVEL 4: Strict curated core N=70 (RF LOO-CV)")
    logger.info("=" * 65)

    with open(STRICT_CORE_JSON) as f:
        strict = json.load(f)

    strict_addrs = strict["addresses"]
    strict_labels = strict["labels"]
    n_strict = len(strict_addrs)
    logger.info("  Strict core addresses: %d", n_strict)

    # Filter to addresses present in v4 features
    valid_mask = [a in X.index for a in strict_addrs]
    addrs_valid = [a for a, v in zip(strict_addrs, valid_mask) if v]
    labels_valid = [l for l, v in zip(strict_labels, valid_mask) if v]
    logger.info("  Present in v4 features: %d / %d", len(addrs_valid), n_strict)

    X_strict = X.loc[addrs_valid].values.astype(float)
    y_strict = np.array(labels_valid)

    n_agents = int(y_strict.sum())
    n_humans = int((y_strict == 0).sum())
    logger.info("  n=%d (agents=%d, humans=%d)", len(y_strict), n_agents, n_humans)

    # LOO-CV
    loo = LeaveOneOut()
    y_probs = np.zeros(len(y_strict))
    y_preds = np.zeros(len(y_strict), dtype=int)

    for train_idx, test_idx in loo.split(X_strict):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_strict[train_idx])
        X_te = scaler.transform(X_strict[test_idx])

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_strict[train_idx])

        y_probs[test_idx] = clf.predict_proba(X_te)[:, 1]
        y_preds[test_idx] = clf.predict(X_te)

    result = {
        "level": 4,
        "name": "Strict curated core LOO-CV (RF)",
        "dataset": "v4 strict core n=70",
        "n_samples": int(len(y_strict)),
        "n_agents": n_agents,
        "n_humans": n_humans,
        "model": "RandomForest",
        "auc": round(float(roc_auc_score(y_strict, y_probs)), 4),
        "f1": round(float(f1_score(y_strict, y_preds, zero_division=0)), 4),
        "precision": round(float(precision_score(y_strict, y_preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_strict, y_preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_strict, y_preds)), 4),
    }

    logger.info("  AUC = %.4f  F1 = %.4f  Acc = %.4f",
                result["auc"], result["f1"], result["accuracy"])
    return result


# ============================================================
# LEVEL 5: LEAVE-ONE-PLATFORM-OUT (LOPO)
# ============================================================

def run_level5(X, y, categories):
    """RF LOPO-CV: hold out each platform, train on the rest."""
    logger.info("=" * 65)
    logger.info("LEVEL 5: Leave-One-Platform-Out (RF)")
    logger.info("=" * 65)

    X_np = X.values.astype(float)
    cat_counts = Counter(categories)
    min_platform_size = 10

    per_platform = {}
    all_y_true, all_y_pred, all_y_prob = [], [], []

    for cat in sorted(cat_counts.keys()):
        n_cat = cat_counts[cat]
        if n_cat < min_platform_size:
            continue

        test_mask = (categories == cat).values
        train_mask = ~test_mask

        X_train, X_test = X_np[train_mask], X_np[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_train)
        X_te = scaler.transform(X_test)

        clf = RandomForestClassifier(**RF_PARAMS)
        clf.fit(X_tr, y_train)

        y_pred_plat = clf.predict(X_te)
        y_prob_plat = clf.predict_proba(X_te)[:, 1]

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred_plat.tolist())
        all_y_prob.extend(y_prob_plat.tolist())

        n_test_agents = int(y_test.sum())
        n_test_humans = int((y_test == 0).sum())
        both_classes = n_test_agents > 0 and n_test_humans > 0

        plat_result = {
            "n_test": n_cat,
            "n_test_agents": n_test_agents,
            "n_test_humans": n_test_humans,
            "accuracy": round(float(accuracy_score(y_test, y_pred_plat)), 4),
            "f1": round(float(f1_score(y_test, y_pred_plat, zero_division=0)), 4),
        }
        if both_classes:
            plat_result["auc"] = round(float(roc_auc_score(y_test, y_prob_plat)), 4)
        else:
            plat_result["auc"] = None
            plat_result["auc_note"] = "single-class platform"

        per_platform[cat] = plat_result
        logger.info("  %-40s n=%3d  Acc=%.3f  F1=%.3f",
                     cat, n_cat, plat_result["accuracy"], plat_result["f1"])

    # Pooled metrics
    all_y_true_arr = np.array(all_y_true)
    all_y_prob_arr = np.array(all_y_prob)
    all_y_pred_arr = np.array(all_y_pred)

    pooled_auc = round(float(roc_auc_score(all_y_true_arr, all_y_prob_arr)), 4)
    pooled_acc = round(float(accuracy_score(all_y_true_arr, all_y_pred_arr)), 4)
    pooled_f1 = round(float(f1_score(all_y_true_arr, all_y_pred_arr, zero_division=0)), 4)

    result = {
        "level": 5,
        "name": "Leave-One-Platform-Out (RF)",
        "dataset": "v4 n=1,147",
        "n_samples": int(len(y)),
        "n_platforms_evaluated": len(per_platform),
        "model": "RandomForest",
        "pooled_auc": pooled_auc,
        "pooled_f1": pooled_f1,
        "pooled_accuracy": pooled_acc,
        "pooled_precision": round(float(
            precision_score(all_y_true_arr, all_y_pred_arr, zero_division=0)), 4),
        "pooled_recall": round(float(
            recall_score(all_y_true_arr, all_y_pred_arr, zero_division=0)), 4),
        "total_predictions": len(all_y_true),
        "per_platform": per_platform,
    }

    logger.info("  Pooled AUC = %.4f  F1 = %.4f  Acc = %.4f",
                pooled_auc, pooled_f1, pooled_acc)
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("UNIFIED CASCADE: ALL 5 LEVELS, RF ONLY, v4 (n=1,147)")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 70)

    df, X, y_provenance, y_leaky, leaky_valid_mask, categories = load_v4_data()

    # Run all levels
    level1 = run_level1(X, y_leaky, leaky_valid_mask)
    level2 = run_level2(X, y_provenance)
    level3 = run_level3(X, y_provenance)
    level4 = run_level4(X, df)
    level5 = run_level5(X, y_provenance, categories)

    elapsed = round(time.time() - t0, 2)

    results = {
        "run_timestamp": timestamp,
        "description": (
            "Unified cascade: ALL five levels computed with the SAME model "
            "(RandomForest) on the SAME dataset (v4, n=1,147). "
            "Fixes FATAL-1: the original cascade mixed GBM/RF and different "
            "datasets. Level 4 uses the 70-address strict core (fixes FATAL-3: "
            "previously reported as n=64 but is actually n=70)."
        ),
        "model_config": {
            "model": "RandomForest",
            "params": RF_PARAMS,
        },
        "dataset": {
            "name": "v4",
            "n_samples": int(len(y_provenance)),
            "n_agents": int(y_provenance.sum()),
            "n_humans": int((y_provenance == 0).sum()),
            "n_features": len(FEATURE_COLS),
        },
        "levels": {
            "level1": level1,
            "level2": level2,
            "level3": level3,
            "level4": level4,
            "level5": level5,
        },
        "cascade_summary": {
            "Level 1 (leaky C1-C4)": level1["auc"],
            "Level 2 (10-fold CV)": level2["auc"],
            "Level 3 (temporal holdout)": level3["auc"],
            "Level 4 (strict core N=70)": level4["auc"],
            "Level 5 (LOPO pooled)": level5["pooled_auc"],
        },
        "elapsed_seconds": elapsed,
    }

    # Save
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_safe = json.loads(json.dumps(results, default=_convert))
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_safe, f, indent=2)
    logger.info("\nSaved to %s", OUTPUT_JSON)

    # Print summary table
    logger.info("\n" + "=" * 70)
    logger.info("CASCADE SUMMARY (all RF, all v4)")
    logger.info("=" * 70)
    logger.info("%-35s  %6s", "Level", "AUC")
    logger.info("-" * 45)
    for label, auc in results["cascade_summary"].items():
        logger.info("%-35s  %.4f", label, auc)
    logger.info("=" * 70)
    logger.info("Total elapsed: %.1f s", elapsed)

    return results


if __name__ == "__main__":
    main()
