"""
Paper 1: Temporal Holdout Validation
=====================================
Instead of random cross-validation, this script splits train/test by
first-seen block height.  Addresses whose earliest transaction falls
before the median block go to TRAIN; addresses first seen after the
median go to TEST.  This simulates a realistic deployment scenario
where the classifier must generalize to *future* addresses it has
never observed during training.

Outputs:
  experiments/temporal_holdout_results.json
  figures/fig_temporal_holdout.pdf
"""

import json
import logging
import os
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
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
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
OUTPUT_JSON = PROJECT_ROOT / "experiments" / "temporal_holdout_results.json"
FIGURES_DIR = PROJECT_ROOT / "figures"

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
for group_features in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group_features)


def get_models() -> dict:
    """Return the same model configs as run_provenance_pipeline.py."""
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


# ------------------------------------------------------------------
# Step 1: Extract first-seen block from raw transaction files
# ------------------------------------------------------------------

def extract_first_seen_block(addresses: list[str]) -> pd.DataFrame:
    """For each address, read its raw .parquet and return the minimum
    blockNumber and timeStamp.

    Returns a DataFrame indexed by address with columns:
        first_block, first_timestamp, last_block, n_raw_tx
    """
    records = []
    addr_lower_to_orig = {a.lower(): a for a in addresses}
    raw_files = {f.stem.lower(): f for f in RAW_DIR.glob("*.parquet")}

    found, missing = 0, 0
    for addr_lower, addr_orig in addr_lower_to_orig.items():
        # Try case-insensitive match
        raw_path = raw_files.get(addr_lower)
        if raw_path is None:
            # Try original casing
            candidate = RAW_DIR / f"{addr_orig}.parquet"
            if candidate.exists():
                raw_path = candidate

        if raw_path is None:
            missing += 1
            continue

        try:
            df = pd.read_parquet(raw_path, columns=["blockNumber", "timeStamp"])
            blocks = pd.to_numeric(df["blockNumber"], errors="coerce")
            timestamps = pd.to_numeric(df["timeStamp"], errors="coerce")
            records.append({
                "address": addr_orig,
                "first_block": int(blocks.min()),
                "first_timestamp": int(timestamps.min()),
                "last_block": int(blocks.max()),
                "n_raw_tx": len(df),
            })
            found += 1
        except Exception as exc:
            logger.warning("Error reading %s: %s", raw_path.name, exc)
            missing += 1

    logger.info("Extracted first-seen block for %d addresses (%d missing)",
                found, missing)
    result = pd.DataFrame(records).set_index("address")
    return result


# ------------------------------------------------------------------
# Step 2: Prepare features, labels, and temporal metadata
# ------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load features, labels, and temporal metadata.

    Returns:
        (X, y, temporal_df) where temporal_df has first_block per address
    """
    logger.info("Loading features from %s", FEATURES_PARQUET)
    df = pd.read_parquet(FEATURES_PARQUET)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    # Load labels
    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)

    # Build label array aligned with features index
    y = np.array([
        labels_dict[addr]["label_provenance"]
        for addr in df.index
    ])

    # Features
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()

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

    # Extract temporal metadata
    logger.info("Extracting first-seen block from raw transaction data...")
    temporal_df = extract_first_seen_block(list(df.index))

    return X, y, temporal_df


# ------------------------------------------------------------------
# Step 3: Evaluate a model on a fixed train/test split
# ------------------------------------------------------------------

def evaluate_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_template,
) -> dict:
    """Train on X_train, evaluate on X_test. Returns metrics dict."""
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    clf = clone(model_template)
    clf.fit(X_tr, y_train)

    y_pred = clf.predict(X_te)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_te)[:, 1]
    else:
        y_prob = clf.decision_function(X_te)

    # AUC requires both classes in y_test
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
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }


# ------------------------------------------------------------------
# Step 4: Random CV baseline (same as provenance pipeline)
# ------------------------------------------------------------------

def run_random_cv(
    X_np: np.ndarray,
    y: np.ndarray,
    model_template,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> dict:
    """Repeated stratified k-fold CV."""
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    fold_aucs, fold_f1s, fold_precs, fold_recs, fold_accs = [], [], [], [], []

    for train_idx, test_idx in rskf.split(X_np, y):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
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
        "precision": round(float(np.mean(fold_precs)), 4),
        "recall": round(float(np.mean(fold_recs)), 4),
        "accuracy": round(float(np.mean(fold_accs)), 4),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
    }


# ------------------------------------------------------------------
# Step 5: Generate comparison figure
# ------------------------------------------------------------------

def generate_figure(results: dict, output_path: Path) -> None:
    """Bar chart comparing random CV vs temporal holdout AUC per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results["random_cv"].keys())
    random_aucs = [results["random_cv"][m]["auc"] for m in models]
    random_stds = [results["random_cv"][m].get("auc_std", 0) for m in models]
    temporal_aucs = [results["temporal_holdout"][m]["auc"] for m in models]

    x = np.arange(len(models))
    width = 0.32

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(
        x - width / 2, random_aucs, width,
        yerr=random_stds, capsize=4,
        label="Random 5x10 CV", color="#4878CF", edgecolor="black", linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2, temporal_aucs, width,
        label="Temporal Holdout", color="#D65F5F", edgecolor="black", linewidth=0.5,
    )

    # Add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("AUC-ROC", fontsize=12)
    ax.set_title("Random CV vs. Temporal Holdout Validation", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4, label="chance")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add annotation with dataset info
    split_info = results["split_metadata"]
    note = (
        f"n={split_info['n_total']}  |  "
        f"Train: {split_info['n_train']} (blocks < {split_info['split_block']:,})  |  "
        f"Test: {split_info['n_test']} (blocks >= {split_info['split_block']:,})"
    )
    ax.text(0.5, -0.12, note, transform=ax.transAxes,
            ha="center", va="top", fontsize=9, color="gray")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", output_path)


# ------------------------------------------------------------------
# Step 6: Generate F1 / Precision / Recall comparison figure
# ------------------------------------------------------------------

def generate_detailed_figure(results: dict, output_path: Path) -> None:
    """Grouped bar chart: AUC, F1, Precision, Recall for each model,
    comparing random CV vs temporal holdout."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(results["random_cv"].keys())
    metrics = ["auc", "f1", "precision", "recall"]
    metric_labels = ["AUC", "F1", "Precision", "Recall"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 5), sharey=True)

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        random_vals = [results["random_cv"][m][metric] for m in models]
        temporal_vals = [results["temporal_holdout"][m][metric] for m in models]

        x = np.arange(len(models))
        width = 0.32
        ax.bar(x - width / 2, random_vals, width,
               label="Random CV", color="#4878CF", edgecolor="black", linewidth=0.5)
        ax.bar(x + width / 2, temporal_vals, width,
               label="Temporal", color="#D65F5F", edgecolor="black", linewidth=0.5)

        for j, (rv, tv) in enumerate(zip(random_vals, temporal_vals)):
            ax.text(j - width / 2, rv + 0.01, f"{rv:.2f}",
                    ha="center", va="bottom", fontsize=8)
            ax.text(j + width / 2, tv + 0.01, f"{tv:.2f}",
                    ha="center", va="bottom", fontsize=8)

        ax.set_title(label, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([m[:4] for m in models], fontsize=9)
        ax.set_ylim(0.0, 1.15)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if i == 0:
            ax.set_ylabel("Score", fontsize=11)
        if i == len(metrics) - 1:
            ax.legend(fontsize=9)

    fig.suptitle("Random CV vs. Temporal Holdout: All Metrics", fontsize=14, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved detailed figure to %s", output_path)


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main() -> dict:
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 65)
    logger.info("Paper 1: Temporal Holdout Validation")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 65)

    # Load data
    X, y, temporal_df = load_data()
    feature_names = X.columns.tolist()

    # Align: only keep addresses that have both features AND temporal data
    common_addrs = X.index.intersection(temporal_df.index)
    logger.info("Addresses with features AND raw temporal data: %d / %d",
                len(common_addrs), len(X))

    X = X.loc[common_addrs]
    y_series = pd.Series(y, index=X.index[:len(y)])
    # Re-align after intersection
    df_features = pd.read_parquet(FEATURES_PARQUET)
    with open(LABELS_JSON) as f:
        labels_dict = json.load(f)
    y_aligned = np.array([
        labels_dict[addr]["label_provenance"] for addr in common_addrs
    ])
    X = X.loc[common_addrs]
    temporal_df = temporal_df.loc[common_addrs]

    # Determine temporal split point (median first_block)
    median_block = int(temporal_df["first_block"].median())
    logger.info("Median first_block: %d", median_block)

    # Split
    train_mask = temporal_df["first_block"] < median_block
    test_mask = temporal_df["first_block"] >= median_block

    X_train = X.loc[train_mask].values.astype(float)
    X_test = X.loc[test_mask].values.astype(float)
    y_train = y_aligned[train_mask.values]
    y_test = y_aligned[test_mask.values]

    logger.info("TEMPORAL SPLIT:")
    logger.info("  Train: n=%d  (agents=%d, humans=%d)  blocks < %d",
                len(y_train), int(y_train.sum()), int((y_train == 0).sum()),
                median_block)
    logger.info("  Test:  n=%d  (agents=%d, humans=%d)  blocks >= %d",
                len(y_test), int(y_test.sum()), int((y_test == 0).sum()),
                median_block)

    # Check class balance viability
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        logger.error("One split has only one class. Cannot compute AUC.")
        return {}

    # Block range summary
    train_blocks = temporal_df.loc[train_mask, "first_block"]
    test_blocks = temporal_df.loc[test_mask, "first_block"]
    train_timestamps = temporal_df.loc[train_mask, "first_timestamp"]
    test_timestamps = temporal_df.loc[test_mask, "first_timestamp"]

    split_metadata = {
        "n_total": int(len(y_aligned)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "split_block": median_block,
        "train_block_range": [int(train_blocks.min()), int(train_blocks.max())],
        "test_block_range": [int(test_blocks.min()), int(test_blocks.max())],
        "train_date_range": [
            datetime.utcfromtimestamp(int(train_timestamps.min())).strftime("%Y-%m-%d"),
            datetime.utcfromtimestamp(int(train_timestamps.max())).strftime("%Y-%m-%d"),
        ],
        "test_date_range": [
            datetime.utcfromtimestamp(int(test_timestamps.min())).strftime("%Y-%m-%d"),
            datetime.utcfromtimestamp(int(test_timestamps.max())).strftime("%Y-%m-%d"),
        ],
        "train_agents": int(y_train.sum()),
        "train_humans": int((y_train == 0).sum()),
        "test_agents": int(y_test.sum()),
        "test_humans": int((y_test == 0).sum()),
    }

    # ---- Temporal holdout evaluation ----
    models = get_models()
    temporal_results = {}
    for name, model_template in models.items():
        logger.info("Temporal holdout: training %s ...", name)
        metrics = evaluate_split(X_train, y_train, X_test, y_test, model_template)
        temporal_results[name] = metrics
        logger.info("  %s: AUC=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f",
                     name, metrics["auc"], metrics["f1"],
                     metrics["precision"], metrics["recall"])

    # ---- Random CV baseline ----
    X_all = X.values.astype(float)
    random_cv_results = {}
    for name, model_template in models.items():
        logger.info("Random 5x10 CV: training %s ...", name)
        metrics = run_random_cv(X_all, y_aligned, model_template)
        random_cv_results[name] = metrics
        logger.info("  %s: AUC=%.4f +/- %.4f  F1=%.4f",
                     name, metrics["auc"], metrics["auc_std"], metrics["f1"])

    # ---- Compute deltas ----
    deltas = {}
    for name in models:
        deltas[name] = {
            "auc_delta": round(
                temporal_results[name]["auc"] - random_cv_results[name]["auc"], 4
            ),
            "f1_delta": round(
                temporal_results[name]["f1"] - random_cv_results[name]["f1"], 4
            ),
        }

    # ---- Assemble results ----
    results = {
        "run_timestamp": timestamp,
        "description": (
            "Temporal holdout validation: train on addresses first seen "
            "before the median block, test on addresses first seen after. "
            "Compares with standard random 5x10-fold stratified CV."
        ),
        "split_metadata": split_metadata,
        "temporal_holdout": temporal_results,
        "random_cv": random_cv_results,
        "deltas": deltas,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    # ---- Save JSON ----
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

    # ---- Generate figures ----
    generate_figure(results, FIGURES_DIR / "fig_temporal_holdout.pdf")
    generate_detailed_figure(results, FIGURES_DIR / "fig_temporal_holdout_detail.pdf")

    # ---- Print summary ----
    logger.info("=" * 65)
    logger.info("SUMMARY")
    logger.info("=" * 65)
    logger.info("%-22s  %8s  %8s  %8s", "", "Random CV", "Temporal", "Delta")
    logger.info("-" * 55)
    for name in models:
        logger.info(
            "%-22s  AUC %5.3f   AUC %5.3f   %+.3f",
            name,
            random_cv_results[name]["auc"],
            temporal_results[name]["auc"],
            deltas[name]["auc_delta"],
        )
    logger.info("=" * 65)
    logger.info("Elapsed: %.1f s", results["elapsed_seconds"])

    return results


if __name__ == "__main__":
    main()
