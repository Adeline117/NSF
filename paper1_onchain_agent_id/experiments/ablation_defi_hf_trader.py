"""
Paper 1: Ablation Study — Exclude defi_hf_trader Category
===========================================================
Reviewer concern: "37% of agents (199/533) are labeled by a behavioral
criterion (>50 daily Uniswap calls) that correlates with classifier
features. This undermines the provenance-only claim."

This script:
  1. Loads the n=1,147 provenance-v4 dataset
  2. Removes all 199 defi_hf_trader addresses → n=948 (334 agents, 614 humans)
  3. Runs RF, GBM, LR with 10-fold stratified CV on n=948
  4. Compares AUC, F1, precision, recall vs full n=1,147 results
  5. Saves results to experiments/ablation_defi_hf_trader_results.json

Outputs:
  experiments/ablation_defi_hf_trader_results.json
"""

import json
import logging
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
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
FEATURES_PARQUET = DATA_DIR / "features_provenance_v4.parquet"
LABELS_JSON = DATA_DIR / "labels_provenance_v4.json"
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "ablation_defi_hf_trader_results.json"

ALL_FEATURES = [
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


def get_models() -> dict:
    """Same model configs as the main pipeline."""
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=3,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
    }


def load_data() -> tuple[pd.DataFrame, np.ndarray, dict]:
    """Load features and labels from the v4 dataset.

    Returns:
        (df, labels_dict)  where df has features + metadata columns
    """
    logger.info("Loading features from %s", FEATURES_PARQUET)
    df = pd.read_parquet(FEATURES_PARQUET)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)

    return df, labels_dict


def prepare_features(df: pd.DataFrame, labels_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract feature matrix and label vector."""
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = np.array([labels_dict[addr]["label_provenance"] for addr in df.index])

    # Impute NaN with column medians
    nan_mask = np.isnan(X.values)
    if nan_mask.any():
        col_medians = np.nanmedian(X.values, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X.iloc[nan_mask[:, j], j] = col_medians[j]

    # Clip extremes at 1st/99th percentile
    for col in X.columns:
        lo, hi = np.nanpercentile(X[col], [1, 99])
        X[col] = X[col].clip(lo, hi)

    return X.values.astype(float), y.astype(int)


def run_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_template,
    n_splits: int = 10,
    n_repeats: int = 10,
) -> dict:
    """Repeated stratified k-fold CV. Returns dict of metrics."""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    fold_aucs, fold_f1s, fold_precs, fold_recs, fold_accs = [], [], [], [], []

    for train_idx, test_idx in rskf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)

        y_pred = clf.predict(X_te)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_te)[:, 1]
        else:
            y_prob = clf.decision_function(X_te)

        fold_aucs.append(roc_auc_score(y_te, y_prob))
        fold_f1s.append(f1_score(y_te, y_pred, zero_division=0))
        fold_precs.append(precision_score(y_te, y_pred, zero_division=0))
        fold_recs.append(recall_score(y_te, y_pred, zero_division=0))
        fold_accs.append(accuracy_score(y_te, y_pred))

    return {
        "auc": round(float(np.mean(fold_aucs)), 4),
        "auc_std": round(float(np.std(fold_aucs)), 4),
        "f1": round(float(np.mean(fold_f1s)), 4),
        "f1_std": round(float(np.std(fold_f1s)), 4),
        "precision": round(float(np.mean(fold_precs)), 4),
        "precision_std": round(float(np.std(fold_precs)), 4),
        "recall": round(float(np.mean(fold_recs)), 4),
        "recall_std": round(float(np.std(fold_recs)), 4),
        "accuracy": round(float(np.mean(fold_accs)), 4),
        "accuracy_std": round(float(np.std(fold_accs)), 4),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
    }


def main() -> dict:
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 65)
    logger.info("Paper 1: Ablation — Exclude defi_hf_trader")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 65)

    # Load data
    df, labels_dict = load_data()

    # Identify defi_hf_trader addresses
    defi_hf_addrs = [
        addr for addr, meta in labels_dict.items()
        if meta.get("category") == "defi_hf_trader"
    ]
    logger.info("defi_hf_trader addresses found: %d", len(defi_hf_addrs))

    # Verify they are all agents
    defi_hf_labels = [labels_dict[a]["label_provenance"] for a in defi_hf_addrs]
    n_agent = sum(defi_hf_labels)
    n_human = len(defi_hf_labels) - n_agent
    logger.info("  agents=%d, humans=%d (should be 199/0)", n_agent, n_human)

    # ---- FULL dataset (n=1147) ----
    logger.info("\n--- Full dataset (n=%d) ---", len(df))
    X_full, y_full = prepare_features(df, labels_dict)
    logger.info("  agents=%d, humans=%d", int(y_full.sum()), int((y_full == 0).sum()))

    models = get_models()
    full_results = {}
    for name, model in models.items():
        logger.info("  Full CV: %s ...", name)
        metrics = run_cv(X_full, y_full, model, n_splits=10, n_repeats=10)
        full_results[name] = metrics
        logger.info("    AUC=%.4f +/- %.4f  F1=%.4f  Prec=%.4f  Rec=%.4f",
                     metrics["auc"], metrics["auc_std"],
                     metrics["f1"], metrics["precision"], metrics["recall"])

    # ---- ABLATED dataset (n=948, excluding defi_hf_trader) ----
    defi_hf_set = set(a.lower() for a in defi_hf_addrs)
    ablated_mask = ~df.index.str.lower().isin(defi_hf_set)
    df_ablated = df[ablated_mask]
    logger.info("\n--- Ablated dataset (n=%d, excluding %d defi_hf_trader) ---",
                len(df_ablated), len(defi_hf_addrs))

    # Build a filtered labels dict for the ablated set
    labels_ablated = {
        addr: labels_dict[addr] for addr in df_ablated.index
    }
    X_ablated, y_ablated = prepare_features(df_ablated, labels_ablated)
    n_agents_abl = int(y_ablated.sum())
    n_humans_abl = int((y_ablated == 0).sum())
    logger.info("  agents=%d, humans=%d", n_agents_abl, n_humans_abl)

    ablated_results = {}
    for name, model in models.items():
        logger.info("  Ablated CV: %s ...", name)
        metrics = run_cv(X_ablated, y_ablated, model, n_splits=10, n_repeats=10)
        ablated_results[name] = metrics
        logger.info("    AUC=%.4f +/- %.4f  F1=%.4f  Prec=%.4f  Rec=%.4f",
                     metrics["auc"], metrics["auc_std"],
                     metrics["f1"], metrics["precision"], metrics["recall"])

    # ---- Compute deltas ----
    deltas = {}
    for name in models:
        deltas[name] = {
            "auc_delta": round(
                ablated_results[name]["auc"] - full_results[name]["auc"], 4
            ),
            "f1_delta": round(
                ablated_results[name]["f1"] - full_results[name]["f1"], 4
            ),
            "precision_delta": round(
                ablated_results[name]["precision"] - full_results[name]["precision"], 4
            ),
            "recall_delta": round(
                ablated_results[name]["recall"] - full_results[name]["recall"], 4
            ),
        }

    # ---- Assemble results ----
    results = {
        "run_timestamp": timestamp,
        "description": (
            "Ablation study: remove all 199 defi_hf_trader addresses "
            "(labeled by >50 daily Uniswap calls, a behavioral criterion "
            "that may correlate with classifier features) and re-evaluate "
            "classifier performance. This addresses the reviewer concern "
            "that 37% of agent labels are behaviorally-derived."
        ),
        "full_dataset": {
            "n_samples": int(len(y_full)),
            "n_agents": int(y_full.sum()),
            "n_humans": int((y_full == 0).sum()),
            "models": full_results,
        },
        "ablated_dataset": {
            "n_samples": int(len(y_ablated)),
            "n_agents": n_agents_abl,
            "n_humans": n_humans_abl,
            "n_removed": len(defi_hf_addrs),
            "removed_category": "defi_hf_trader",
            "removal_reason": (
                "These 199 addresses were labeled as agents based on "
                ">50 daily Uniswap swap calls (behavioral criterion). "
                "This correlates with temporal features (burst_frequency, "
                "tx_interval_std) used by the classifier, creating a "
                "potential circularity concern."
            ),
            "models": ablated_results,
        },
        "deltas": deltas,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # ---- Save ----
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_ser = json.loads(json.dumps(results, default=_convert))
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results_ser, f, indent=2)
    logger.info("Saved results to %s", OUTPUT_JSON)

    # ---- Print summary ----
    logger.info("=" * 65)
    logger.info("SUMMARY: Full (n=1147) vs Ablated (n=948)")
    logger.info("=" * 65)
    logger.info("%-22s  %14s  %14s  %8s", "", "Full AUC", "Ablated AUC", "Delta")
    logger.info("-" * 65)
    for name in models:
        logger.info(
            "%-22s  %5.4f +/- %.4f  %5.4f +/- %.4f  %+.4f",
            name,
            full_results[name]["auc"], full_results[name]["auc_std"],
            ablated_results[name]["auc"], ablated_results[name]["auc_std"],
            deltas[name]["auc_delta"],
        )
    logger.info("")
    logger.info("%-22s  %14s  %14s  %8s", "", "Full F1", "Ablated F1", "Delta")
    logger.info("-" * 65)
    for name in models:
        logger.info(
            "%-22s  %5.4f +/- %.4f  %5.4f +/- %.4f  %+.4f",
            name,
            full_results[name]["f1"], full_results[name]["f1_std"],
            ablated_results[name]["f1"], ablated_results[name]["f1_std"],
            deltas[name]["f1_delta"],
        )
    logger.info("=" * 65)
    logger.info("Elapsed: %.1f s", results["elapsed_seconds"])

    return results


if __name__ == "__main__":
    main()
