"""
Paper 1: Published Baseline Comparison on n=1,147 Provenance Dataset
=====================================================================
Reviewer concern: "No comparison with any existing blockchain bot
detection method."

This script implements 3 published/standard baselines plus a random
reference, and evaluates each at three evaluation levels:
  - Level 1: 10-fold Stratified CV (standard)
  - Level 3: Temporal holdout (train on pre-median-block, test post)
  - Level 5: Leave-one-platform-out (LOPO-CV)

Baselines:
  1. Heuristic Bot Score (Victor & Weintraud 2021 style)
       Flag if tx_interval_mean < 60s, active_hour_entropy > 3.5,
       or gas_price_round_number_ratio > 0.8.  Score = rules_triggered / 3.
  2. Single-feature threshold (tx_interval_mean)
       Best univariate predictor from the n=64 pilot.
       Score = 1 - normalized(tx_interval_mean).
  3. Logistic Regression on 6 gas features only
       Tests whether a minimal, easy-to-compute feature set competes.
  4. Random baseline (AUC = 0.50 reference line).

Full model: Random Forest (23 features, same hyperparameters as
the main pipeline) is the paper's proposed method.

Output: experiments/baseline_comparison_results.json
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
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
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
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "baseline_comparison_results.json"

# ── Feature groups (same as the main pipeline) ──────────────────────
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
ALL_FEATURES = TEMPORAL_FEATURES + GAS_FEATURES + INTERACTION_FEATURES + APPROVAL_FEATURES


# ====================================================================
# DATA LOADING
# ====================================================================

def load_dataset() -> tuple[pd.DataFrame, np.ndarray, list[str], pd.Series]:
    """Load the n=1,147 provenance-labeled dataset.

    Returns (X, y, feature_names, categories).
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
# BASELINE 1: Heuristic Bot Score (Victor & Weintraud 2021 style)
# ====================================================================

def heuristic_bot_score(X: pd.DataFrame) -> np.ndarray:
    """Score = (triggered_rules) / 3.

    Rules:
      R1: tx_interval_mean < 60 seconds   (high frequency)
      R2: active_hour_entropy > 3.5        (24/7 activity)
      R3: gas_price_round_number_ratio > 0.8 (programmatic gas)

    Returns continuous score in [0, 1].
    """
    r1 = (X["tx_interval_mean"].values < 60).astype(float)
    r2 = (X["active_hour_entropy"].values > 3.5).astype(float)
    r3 = (X["gas_price_round_number_ratio"].values > 0.8).astype(float)
    return (r1 + r2 + r3) / 3.0


# ====================================================================
# BASELINE 2: Single-feature (tx_interval_mean)
# ====================================================================

def single_feature_score(X: pd.DataFrame) -> np.ndarray:
    """Use 1 - normalized(tx_interval_mean) as agent probability.

    Lower interval = more likely agent. We min-max normalize so the
    score is in [0, 1].
    """
    vals = X["tx_interval_mean"].values.astype(float)
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-10:
        return np.full(len(vals), 0.5)
    normalized = (vals - vmin) / (vmax - vmin)
    return 1.0 - normalized  # lower interval -> higher score


# ====================================================================
# EVALUATION HELPERS
# ====================================================================

def metrics_from_score(y_true: np.ndarray, y_score: np.ndarray,
                       threshold: float = 0.5) -> dict:
    """Compute AUC, F1, precision, recall, accuracy from a continuous score."""
    y_pred = (y_score >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_score)
    return {
        "auc": round(float(auc), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }


def metrics_from_model(X_train, y_train, X_test, y_test,
                       model_template) -> dict:
    """Fit a model, predict on test set, return metrics."""
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
# LEVEL 1: 10-Fold Stratified CV
# ====================================================================

def eval_10fold_cv(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Evaluate all methods under 10-fold stratified CV."""
    logger.info("Running Level 1: 10-Fold Stratified CV ...")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    X_np = X.values.astype(float)
    feature_names = X.columns.tolist()

    # Model templates
    rf_full = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    lr_gas = LogisticRegression(C=1.0, max_iter=2000, random_state=42)

    gas_idx = [feature_names.index(f) for f in GAS_FEATURES if f in feature_names]

    # Accumulators: method -> list of fold dicts
    methods = {
        "Heuristic (Victor & Weintraud style)": {"aucs": [], "f1s": []},
        "Single-feature (tx_interval_mean)": {"aucs": [], "f1s": []},
        "LogReg (6 gas features)": {"aucs": [], "f1s": []},
        "Random baseline": {"aucs": [], "f1s": []},
        "Full RF (23 features) [proposed]": {"aucs": [], "f1s": []},
    }

    fold_details = {m: [] for m in methods}

    for fold_i, (train_idx, test_idx) in enumerate(skf.split(X_np, y)):
        X_train_df = X.iloc[train_idx]
        X_test_df = X.iloc[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        X_train_np = X_np[train_idx]
        X_test_np = X_np[test_idx]

        if len(np.unique(y_test)) < 2:
            continue

        # 1) Heuristic bot score
        h_scores = heuristic_bot_score(X_test_df)
        # For heuristic, find best threshold on training set
        h_scores_train = heuristic_bot_score(X_train_df)
        # Find threshold that maximizes F1 on train
        best_thresh_h = 0.5
        best_f1_h = 0
        for t in [0.0, 1/3, 2/3, 1.0]:
            preds_t = (h_scores_train >= t).astype(int)
            f1_t = f1_score(y_train, preds_t, zero_division=0)
            if f1_t > best_f1_h:
                best_f1_h = f1_t
                best_thresh_h = t
        m_h = metrics_from_score(y_test, h_scores, threshold=best_thresh_h)
        methods["Heuristic (Victor & Weintraud style)"]["aucs"].append(m_h["auc"])
        methods["Heuristic (Victor & Weintraud style)"]["f1s"].append(m_h["f1"])
        fold_details["Heuristic (Victor & Weintraud style)"].append(m_h)

        # 2) Single feature (tx_interval_mean)
        sf_scores = single_feature_score(X_test_df)
        # Compute threshold on train
        sf_train = single_feature_score(X_train_df)
        best_thresh_sf = 0.5
        best_f1_sf = 0
        for t in np.linspace(0.0, 1.0, 21):
            preds_t = (sf_train >= t).astype(int)
            f1_t = f1_score(y_train, preds_t, zero_division=0)
            if f1_t > best_f1_sf:
                best_f1_sf = f1_t
                best_thresh_sf = t
        m_sf = metrics_from_score(y_test, sf_scores, threshold=best_thresh_sf)
        methods["Single-feature (tx_interval_mean)"]["aucs"].append(m_sf["auc"])
        methods["Single-feature (tx_interval_mean)"]["f1s"].append(m_sf["f1"])
        fold_details["Single-feature (tx_interval_mean)"].append(m_sf)

        # 3) Logistic regression on gas features
        X_gas_train = X_train_np[:, gas_idx]
        X_gas_test = X_test_np[:, gas_idx]
        m_lr = metrics_from_model(X_gas_train, y_train, X_gas_test, y_test, lr_gas)
        methods["LogReg (6 gas features)"]["aucs"].append(m_lr["auc"])
        methods["LogReg (6 gas features)"]["f1s"].append(m_lr["f1"])
        fold_details["LogReg (6 gas features)"].append(m_lr)

        # 4) Random baseline
        rng = np.random.RandomState(42 + fold_i)
        random_scores = rng.uniform(0, 1, len(y_test))
        m_rand = metrics_from_score(y_test, random_scores)
        methods["Random baseline"]["aucs"].append(m_rand["auc"])
        methods["Random baseline"]["f1s"].append(m_rand["f1"])
        fold_details["Random baseline"].append(m_rand)

        # 5) Full RF (23 features)
        m_rf = metrics_from_model(X_train_np, y_train, X_test_np, y_test, rf_full)
        methods["Full RF (23 features) [proposed]"]["aucs"].append(m_rf["auc"])
        methods["Full RF (23 features) [proposed]"]["f1s"].append(m_rf["f1"])
        fold_details["Full RF (23 features) [proposed]"].append(m_rf)

    # Aggregate
    summary = {}
    for method_name, accum in methods.items():
        aucs = accum["aucs"]
        f1s = accum["f1s"]
        summary[method_name] = {
            "auc_mean": round(float(np.mean(aucs)), 4),
            "auc_std": round(float(np.std(aucs)), 4),
            "f1_mean": round(float(np.mean(f1s)), 4),
            "f1_std": round(float(np.std(f1s)), 4),
            "n_folds": len(aucs),
        }
        logger.info("  %-45s AUC=%.4f +/- %.4f   F1=%.4f +/- %.4f",
                     method_name,
                     np.mean(aucs), np.std(aucs),
                     np.mean(f1s), np.std(f1s))

    return summary


# ====================================================================
# LEVEL 3: Temporal Holdout
# ====================================================================

def eval_temporal_holdout(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Evaluate all methods under temporal holdout split."""
    logger.info("Running Level 3: Temporal Holdout ...")

    addresses = list(X.index)
    temporal_df = extract_first_seen_block(addresses)
    common = X.index.intersection(temporal_df.index)
    X_common = X.loc[common]
    # Re-derive y for common addresses
    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)
    y_common = np.array([
        labels_dict[addr]["label_provenance"] for addr in common
    ])
    temporal_df = temporal_df.loc[common]

    median_block = int(temporal_df["first_block"].median())
    train_mask = temporal_df["first_block"] < median_block
    test_mask = temporal_df["first_block"] >= median_block

    X_train_df = X_common.loc[train_mask]
    X_test_df = X_common.loc[test_mask]
    y_train = y_common[train_mask.values]
    y_test = y_common[test_mask.values]

    X_train_np = X_train_df.values.astype(float)
    X_test_np = X_test_df.values.astype(float)
    feature_names = X_common.columns.tolist()

    logger.info("  Temporal split: train=%d (agents=%d) | test=%d (agents=%d) | median_block=%d",
                len(y_train), int(y_train.sum()),
                len(y_test), int(y_test.sum()), median_block)

    rf_full = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    lr_gas = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    gas_idx = [feature_names.index(f) for f in GAS_FEATURES if f in feature_names]

    results = {}

    # 1) Heuristic
    h_scores = heuristic_bot_score(X_test_df)
    h_train = heuristic_bot_score(X_train_df)
    best_t, best_f1 = 0.5, 0
    for t in [0.0, 1/3, 2/3, 1.0]:
        f1_t = f1_score(y_train, (h_train >= t).astype(int), zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_t = t
    results["Heuristic (Victor & Weintraud style)"] = metrics_from_score(
        y_test, h_scores, threshold=best_t)

    # 2) Single feature
    sf_scores = single_feature_score(X_test_df)
    sf_train = single_feature_score(X_train_df)
    best_t, best_f1 = 0.5, 0
    for t in np.linspace(0.0, 1.0, 21):
        f1_t = f1_score(y_train, (sf_train >= t).astype(int), zero_division=0)
        if f1_t > best_f1:
            best_f1 = f1_t
            best_t = t
    results["Single-feature (tx_interval_mean)"] = metrics_from_score(
        y_test, sf_scores, threshold=best_t)

    # 3) LogReg gas
    results["LogReg (6 gas features)"] = metrics_from_model(
        X_train_np[:, gas_idx], y_train,
        X_test_np[:, gas_idx], y_test, lr_gas)

    # 4) Random
    rng = np.random.RandomState(42)
    results["Random baseline"] = metrics_from_score(
        y_test, rng.uniform(0, 1, len(y_test)))

    # 5) Full RF
    results["Full RF (23 features) [proposed]"] = metrics_from_model(
        X_train_np, y_train, X_test_np, y_test, rf_full)

    for name, m in results.items():
        logger.info("  %-45s AUC=%.4f  F1=%.4f", name, m["auc"], m["f1"])

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
    return results, metadata


# ====================================================================
# LEVEL 5: Leave-One-Platform-Out
# ====================================================================

def eval_lopo(X: pd.DataFrame, y: np.ndarray,
              categories: pd.Series) -> dict:
    """Evaluate all methods under LOPO-CV.

    For each platform with >=10 addresses, hold it out entirely, train
    on all others, predict on the held-out set, then pool all held-out
    predictions for a single AUC.
    """
    logger.info("Running Level 5: Leave-One-Platform-Out ...")
    X_np = X.values.astype(float)
    feature_names = X.columns.tolist()
    gas_idx = [feature_names.index(f) for f in GAS_FEATURES if f in feature_names]

    rf_full = RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_leaf=3,
        random_state=42, n_jobs=-1,
    )
    lr_gas = LogisticRegression(C=1.0, max_iter=2000, random_state=42)

    cat_counts = Counter(categories)
    min_size = 10

    # Accumulators per method: list of (y_true, y_score) for pooling
    method_names = [
        "Heuristic (Victor & Weintraud style)",
        "Single-feature (tx_interval_mean)",
        "LogReg (6 gas features)",
        "Random baseline",
        "Full RF (23 features) [proposed]",
    ]
    pooled = {m: {"y_true": [], "y_score": [], "y_pred": []} for m in method_names}
    per_platform = {m: {} for m in method_names}

    for cat in sorted(cat_counts.keys()):
        if cat_counts[cat] < min_size:
            continue

        test_mask = (categories == cat).values
        train_mask = ~test_mask
        y_train = y[train_mask]
        y_test = y[test_mask]

        if len(np.unique(y_train)) < 2:
            continue

        X_train_df = X.iloc[np.where(train_mask)[0]]
        X_test_df = X.iloc[np.where(test_mask)[0]]
        X_train_np = X_np[train_mask]
        X_test_np = X_np[test_mask]

        # 1) Heuristic
        h_scores = heuristic_bot_score(X_test_df)
        h_train = heuristic_bot_score(X_train_df)
        best_t, best_f1 = 0.5, 0
        for t in [0.0, 1/3, 2/3, 1.0]:
            f1_t = f1_score(y_train, (h_train >= t).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_t = t
        h_pred = (h_scores >= best_t).astype(int)
        pooled["Heuristic (Victor & Weintraud style)"]["y_true"].extend(y_test.tolist())
        pooled["Heuristic (Victor & Weintraud style)"]["y_score"].extend(h_scores.tolist())
        pooled["Heuristic (Victor & Weintraud style)"]["y_pred"].extend(h_pred.tolist())

        # 2) Single feature
        sf_scores = single_feature_score(X_test_df)
        sf_train = single_feature_score(X_train_df)
        best_t, best_f1 = 0.5, 0
        for t in np.linspace(0.0, 1.0, 21):
            f1_t = f1_score(y_train, (sf_train >= t).astype(int), zero_division=0)
            if f1_t > best_f1:
                best_f1 = f1_t
                best_t = t
        sf_pred = (sf_scores >= best_t).astype(int)
        pooled["Single-feature (tx_interval_mean)"]["y_true"].extend(y_test.tolist())
        pooled["Single-feature (tx_interval_mean)"]["y_score"].extend(sf_scores.tolist())
        pooled["Single-feature (tx_interval_mean)"]["y_pred"].extend(sf_pred.tolist())

        # 3) LogReg gas
        scaler = StandardScaler()
        X_gas_tr = scaler.fit_transform(X_train_np[:, gas_idx])
        X_gas_te = scaler.transform(X_test_np[:, gas_idx])
        clf_lr = clone(lr_gas)
        clf_lr.fit(X_gas_tr, y_train)
        lr_prob = clf_lr.predict_proba(X_gas_te)[:, 1]
        lr_pred = clf_lr.predict(X_gas_te)
        pooled["LogReg (6 gas features)"]["y_true"].extend(y_test.tolist())
        pooled["LogReg (6 gas features)"]["y_score"].extend(lr_prob.tolist())
        pooled["LogReg (6 gas features)"]["y_pred"].extend(lr_pred.tolist())

        # 4) Random
        rng = np.random.RandomState(hash(cat) % (2**31))
        rand_scores = rng.uniform(0, 1, len(y_test))
        rand_pred = (rand_scores >= 0.5).astype(int)
        pooled["Random baseline"]["y_true"].extend(y_test.tolist())
        pooled["Random baseline"]["y_score"].extend(rand_scores.tolist())
        pooled["Random baseline"]["y_pred"].extend(rand_pred.tolist())

        # 5) Full RF
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train_np)
        X_te_s = scaler.transform(X_test_np)
        clf_rf = clone(rf_full)
        clf_rf.fit(X_tr_s, y_train)
        rf_prob = clf_rf.predict_proba(X_te_s)[:, 1]
        rf_pred = clf_rf.predict(X_te_s)
        pooled["Full RF (23 features) [proposed]"]["y_true"].extend(y_test.tolist())
        pooled["Full RF (23 features) [proposed]"]["y_score"].extend(rf_prob.tolist())
        pooled["Full RF (23 features) [proposed]"]["y_pred"].extend(rf_pred.tolist())

        logger.info("  Platform %-40s n=%3d  agents=%3d  humans=%3d",
                     cat, cat_counts[cat],
                     int(y_test.sum()), int((y_test == 0).sum()))

    # Compute pooled metrics
    results = {}
    for method_name in method_names:
        yt = np.array(pooled[method_name]["y_true"])
        ys = np.array(pooled[method_name]["y_score"])
        yp = np.array(pooled[method_name]["y_pred"])

        if len(np.unique(yt)) < 2:
            auc_val = float("nan")
        else:
            auc_val = roc_auc_score(yt, ys)

        results[method_name] = {
            "pooled_auc": round(float(auc_val), 4),
            "pooled_f1": round(float(f1_score(yt, yp, zero_division=0)), 4),
            "pooled_accuracy": round(float(accuracy_score(yt, yp)), 4),
            "pooled_precision": round(float(precision_score(yt, yp, zero_division=0)), 4),
            "pooled_recall": round(float(recall_score(yt, yp, zero_division=0)), 4),
            "n_predictions": int(len(yt)),
        }
        logger.info("  %-45s Pooled AUC=%.4f  F1=%.4f",
                     method_name, results[method_name]["pooled_auc"],
                     results[method_name]["pooled_f1"])

    return results


# ====================================================================
# MAIN
# ====================================================================

def main():
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("Paper 1: Published Baseline Comparison (n=1,147)")
    logger.info("=" * 70)

    X, y, feature_names, categories = load_dataset()

    results = {
        "run_timestamp": timestamp,
        "experiment": "baseline_comparison_3_levels",
        "description": (
            "Compares the proposed 23-feature RF classifier against "
            "3 baselines (heuristic bot score, single-feature, logistic "
            "regression on gas features) and a random reference. "
            "Evaluated at three rigor levels: 10-fold stratified CV, "
            "temporal holdout, and leave-one-platform-out."
        ),
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(feature_names),
        },
        "baselines": {
            "heuristic": (
                "Victor & Weintraud (2021) style: 3 rules "
                "(tx_interval<60s, hour_entropy>3.5, gas_round_ratio>0.8). "
                "Score = triggered_rules / 3."
            ),
            "single_feature": (
                "Best univariate predictor: tx_interval_mean. "
                "Score = 1 - normalized(tx_interval_mean)."
            ),
            "logreg_gas": (
                "Logistic regression on 6 gas-behavior features only: "
                "gas_price_round_number_ratio, gas_price_trailing_zeros_mean, "
                "gas_limit_precision, gas_price_cv, "
                "eip1559_priority_fee_precision, gas_price_nonce_correlation."
            ),
            "random": "Uniform random scores in [0, 1]. AUC ~ 0.50.",
        },
    }

    # ── Level 1: 10-Fold CV ──
    cv_results = eval_10fold_cv(X, y)
    results["level1_10fold_cv"] = cv_results

    # ── Level 3: Temporal Holdout ──
    temporal_results, temporal_metadata = eval_temporal_holdout(X, y)
    results["level3_temporal_holdout"] = temporal_results
    results["level3_temporal_metadata"] = temporal_metadata

    # ── Level 5: LOPO ──
    lopo_results = eval_lopo(X, y, categories)
    results["level5_lopo"] = lopo_results

    # ── Summary Table ──
    logger.info("\n" + "=" * 90)
    logger.info("COMPARISON TABLE: Method x Evaluation Level -> AUC")
    logger.info("=" * 90)
    header = f"{'Method':<45} {'10-fold CV':>12} {'Temporal':>12} {'LOPO':>12}"
    logger.info(header)
    logger.info("-" * 90)

    method_names = [
        "Random baseline",
        "Heuristic (Victor & Weintraud style)",
        "Single-feature (tx_interval_mean)",
        "LogReg (6 gas features)",
        "Full RF (23 features) [proposed]",
    ]

    comparison_table = []
    for m in method_names:
        cv_auc = cv_results[m]["auc_mean"]
        cv_std = cv_results[m]["auc_std"]
        temp_auc = temporal_results[m]["auc"]
        lopo_auc = lopo_results[m]["pooled_auc"]

        row = {
            "method": m,
            "cv_10fold_auc": cv_auc,
            "cv_10fold_auc_std": cv_std,
            "temporal_holdout_auc": temp_auc,
            "lopo_pooled_auc": lopo_auc,
        }
        comparison_table.append(row)

        logger.info(
            "%-45s %5.3f+/-%.3f  %10.4f  %10.4f",
            m, cv_auc, cv_std, temp_auc, lopo_auc,
        )

    results["comparison_table"] = comparison_table

    # ── F1 summary table ──
    logger.info("\n" + "=" * 90)
    logger.info("COMPARISON TABLE: Method x Evaluation Level -> F1")
    logger.info("=" * 90)
    header = f"{'Method':<45} {'10-fold CV':>12} {'Temporal':>12} {'LOPO':>12}"
    logger.info(header)
    logger.info("-" * 90)

    f1_table = []
    for m in method_names:
        cv_f1 = cv_results[m]["f1_mean"]
        temp_f1 = temporal_results[m]["f1"]
        lopo_f1 = lopo_results[m]["pooled_f1"]

        row = {
            "method": m,
            "cv_10fold_f1": cv_f1,
            "temporal_holdout_f1": temp_f1,
            "lopo_pooled_f1": lopo_f1,
        }
        f1_table.append(row)

        logger.info("%-45s %12.4f  %10.4f  %10.4f", m, cv_f1, temp_f1, lopo_f1)

    results["f1_comparison_table"] = f1_table

    # ── Key finding ──
    proposed_cv = cv_results["Full RF (23 features) [proposed]"]["auc_mean"]
    best_baseline_cv = max(
        cv_results[m]["auc_mean"]
        for m in method_names if m != "Full RF (23 features) [proposed]"
    )
    improvement = proposed_cv - best_baseline_cv

    results["key_finding"] = {
        "proposed_model_cv_auc": proposed_cv,
        "best_baseline_cv_auc": best_baseline_cv,
        "improvement_auc": round(improvement, 4),
        "interpretation": (
            f"The proposed 23-feature RF achieves {proposed_cv:.3f} AUC "
            f"under 10-fold CV, outperforming the best baseline by "
            f"{improvement:+.3f} AUC. The heuristic and single-feature "
            f"baselines show that behavioral features carry signal, but "
            f"the multi-feature model captures complementary patterns "
            f"that no single rule or feature can match."
        ),
    }

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
    logger.info("Baseline comparison complete in %.1fs", results["elapsed_seconds"])
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
