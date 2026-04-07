"""
Paper 1: Re-runnable Full Experiment Pipeline
================================================
One-click script that reproduces ALL Paper 1 results.

After Delphi expert validation:
  1. Update labeling_config.py with new C1-C4 labels
  2. Run:  python -m paper1_onchain_agent_id.experiments.run_full_pipeline
  3. New results + figures are produced automatically

Pipeline steps:
  a) Load labels from labeling_config.py
  b) Filter to AGENT vs NOT_AGENT (exclude EXCLUDE and BOUNDARY)
  c) Load features from existing parquet or collect new ones
  d) Train classifiers (GBM, RF, LR) with LOO-CV
  e) Run ablation study (each feature group alone)
  f) Run baseline comparison (heuristic, single-feature, full model)
  g) Produce all result JSONs
  h) Generate figures (English) for the paper
"""

import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
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
    roc_curve,
)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---- Paths ----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
FIGURES_DIR = PROJECT_ROOT / "figures"
FEATURES_PARQUET = DATA_DIR / "features.parquet"

# Add project root to path for imports
sys.path.insert(0, str(PROJECT_ROOT.parent))


# ============================================================
# FEATURE GROUP DEFINITIONS
# ============================================================

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


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models() -> dict:
    """Return the three classifiers with tuned hyper-parameters."""
    return {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=3,
            random_state=42,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=3,
            random_state=42,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        ),
    }


# ============================================================
# STEP 1: LOAD LABELS
# ============================================================

def load_labels() -> tuple[list[str], list[int]]:
    """Load AGENT / NOT_AGENT labels from labeling_config.py.

    Returns:
        Tuple (addresses, labels) where labels are binary 0/1.
    """
    from paper1_onchain_agent_id.data.labeling_config import (
        get_training_addresses,
        summary,
    )

    logger.info("Loading labels from labeling_config.py ...")
    for category, count in summary().items():
        logger.info("  %s: %d", category, count)

    addresses, labels = get_training_addresses()
    n_agent = sum(labels)
    n_not = len(labels) - n_agent
    logger.info("Training set: %d AGENT, %d NOT_AGENT, %d total",
                n_agent, n_not, len(labels))
    return addresses, labels


# ============================================================
# STEP 2: LOAD FEATURES
# ============================================================

def load_features(
    addresses: list[str],
    labels: list[int],
    features_path: Optional[Path] = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load feature matrix for the given addresses.

    First tries to load from the existing parquet file. If an address
    is missing, attempts to collect features via the API pipeline.

    Args:
        addresses: List of Ethereum addresses.
        labels: Corresponding binary labels.
        features_path: Path to features parquet file.

    Returns:
        Tuple (X_dataframe, y_array) aligned and filtered.
    """
    if features_path is None:
        features_path = FEATURES_PARQUET

    logger.info("Loading features from %s ...", features_path)

    if features_path.exists():
        df = pd.read_parquet(features_path)
        logger.info("Loaded parquet with %d rows, %d columns",
                     len(df), len(df.columns))
    else:
        logger.warning("Features file not found: %s", features_path)
        logger.info("Attempting to collect features via API ...")
        df = _collect_features(addresses)

    # Filter to requested addresses only
    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    if not feature_cols:
        raise ValueError("No feature columns found in parquet file")

    # Build aligned X, y
    addr_lower_to_label = {
        a.lower(): l for a, l in zip(addresses, labels)
    }

    available_addrs = []
    available_labels = []

    for idx_addr in df.index:
        addr_lower = idx_addr.lower() if isinstance(idx_addr, str) else str(idx_addr).lower()
        if addr_lower in addr_lower_to_label:
            available_addrs.append(idx_addr)
            available_labels.append(addr_lower_to_label[addr_lower])

    if not available_addrs:
        # Fall back: try matching addresses that exist in parquet
        # even if not in labeling_config (use parquet's own labels)
        if "label" in df.columns:
            logger.warning("No address match via labeling_config; "
                           "falling back to parquet 'label' column.")
            X = df[feature_cols].copy()
            y = df["label"].values.astype(int)

            # Handle NaN
            nan_mask = np.isnan(X.values)
            if nan_mask.any():
                col_medians = np.nanmedian(X.values, axis=0)
                for j in range(X.shape[1]):
                    X.iloc[nan_mask[:, j], j] = col_medians[j]
            return X, y
        raise ValueError("No addresses matched between labels and features")

    X = df.loc[available_addrs, feature_cols].copy()
    y = np.array(available_labels)

    # Handle NaN
    nan_mask = np.isnan(X.values)
    if nan_mask.any():
        col_medians = np.nanmedian(X.values, axis=0)
        for j in range(X.shape[1]):
            X.iloc[nan_mask[:, j], j] = col_medians[j]
        logger.info("Filled %d NaN values with column medians", nan_mask.sum())

    logger.info("Feature matrix: %d samples x %d features", *X.shape)
    logger.info("  Agents: %d, Not-agents: %d", y.sum(), (y == 0).sum())

    return X, y


def _collect_features(addresses: list[str]) -> pd.DataFrame:
    """Collect features via the Etherscan API pipeline.

    Falls back to an empty DataFrame if the pipeline is not available
    or API keys are missing.
    """
    try:
        from shared.utils.eth_utils import EtherscanClient
        from paper1_onchain_agent_id.features.feature_pipeline import (
            FeaturePipeline,
        )

        client = EtherscanClient()
        pipeline = FeaturePipeline(client)
        df = pipeline.extract(addresses, show_progress=True)
        # Save for future runs
        df.to_parquet(FEATURES_PARQUET)
        logger.info("Saved features to %s", FEATURES_PARQUET)
        return df
    except Exception as exc:
        logger.error("Feature collection failed: %s", exc)
        raise


# ============================================================
# STEP 3: PER-FEATURE AUC
# ============================================================

def per_feature_auc(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
) -> dict[str, float]:
    """Compute univariate AUC for each feature.

    For features negatively associated with the positive class,
    the AUC is flipped to max(auc, 1 - auc).
    """
    aucs = {}
    for i, name in enumerate(feature_names):
        col = X[:, i]
        if np.std(col) == 0:
            aucs[name] = 0.5
            continue
        auc = roc_auc_score(y, col)
        aucs[name] = max(auc, 1.0 - auc)
    return dict(sorted(aucs.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# STEP 4: TRAIN & EVALUATE WITH LOO-CV
# ============================================================

def run_loo_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_template,
) -> dict:
    """Run Leave-One-Out cross-validation.

    Returns:
        Dictionary with metrics and per-sample predictions.
    """
    loo = LeaveOneOut()
    n = len(y)
    preds = np.zeros(n)
    probs = np.zeros(n)

    for train_idx, test_idx in loo.split(X):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr = y[train_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)

        preds[test_idx[0]] = clf.predict(X_te)[0]
        if hasattr(clf, "predict_proba"):
            probs[test_idx[0]] = clf.predict_proba(X_te)[0, 1]
        else:
            probs[test_idx[0]] = clf.decision_function(X_te)[0]

    metrics = {
        "auc": round(float(roc_auc_score(y, probs)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
    }

    return {
        "metrics": metrics,
        "predictions": preds,
        "probabilities": probs,
    }


def run_repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_template,
    n_splits: int = 5,
    n_repeats: int = 10,
) -> dict:
    """Run repeated stratified K-fold cross-validation.

    Returns:
        Dictionary with mean and std metrics.
    """
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    fold_metrics = {
        "auc": [], "precision": [], "recall": [], "f1": [], "accuracy": [],
    }

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

        fold_metrics["auc"].append(roc_auc_score(y_te, y_prob))
        fold_metrics["precision"].append(
            precision_score(y_te, y_pred, zero_division=0))
        fold_metrics["recall"].append(
            recall_score(y_te, y_pred, zero_division=0))
        fold_metrics["f1"].append(
            f1_score(y_te, y_pred, zero_division=0))
        fold_metrics["accuracy"].append(accuracy_score(y_te, y_pred))

    mean_m = {k: round(float(np.mean(v)), 4) for k, v in fold_metrics.items()}
    std_m = {k: round(float(np.std(v)), 4) for k, v in fold_metrics.items()}

    return {
        "mean": mean_m,
        "std": std_m,
        "n_repeats": n_repeats,
        "n_splits": n_splits,
    }


def train_all_models(
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict:
    """Train all three classifiers and evaluate with LOO-CV and repeated CV.

    Returns:
        Results dictionary with metrics per model.
    """
    models = get_models()
    feature_names = X.columns.tolist()
    X_np = X.values.astype(float)

    results: dict = {
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(feature_names),
            "feature_names": feature_names,
        },
        "models": {},
    }

    # Per-feature AUC
    feat_aucs = per_feature_auc(X_np, y, feature_names)
    results["per_feature_auc"] = {k: round(v, 4) for k, v in feat_aucs.items()}

    for model_name, model_template in models.items():
        logger.info("Training %s ...", model_name)

        # LOO-CV
        loo_result = run_loo_cv(X_np, y, model_template)
        logger.info("  LOO-CV AUC=%.4f  F1=%.4f  Acc=%.4f",
                     loo_result["metrics"]["auc"],
                     loo_result["metrics"]["f1"],
                     loo_result["metrics"]["accuracy"])

        # Repeated 5-fold CV
        rcv_result = run_repeated_cv(X_np, y, model_template)
        logger.info("  Repeated CV AUC=%.4f +/- %.4f",
                     rcv_result["mean"]["auc"], rcv_result["std"]["auc"])

        results["models"][model_name] = {
            "loo_cv": loo_result["metrics"],
            "repeated_cv": rcv_result,
            "loo_predictions": loo_result["predictions"].tolist(),
            "loo_probabilities": loo_result["probabilities"].tolist(),
        }

    # GBM feature importance (trained on full data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    gbm = clone(models["GradientBoosting"])
    gbm.fit(X_scaled, y)
    importances = gbm.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    gbm_importance = {}
    for idx in sorted_idx:
        gbm_importance[feature_names[idx]] = round(float(importances[idx]), 4)
    results["gbm_feature_importance"] = gbm_importance

    # Confusion matrix from GBM LOO
    gbm_preds = np.array(results["models"]["GradientBoosting"]["loo_predictions"])
    cm = confusion_matrix(y, gbm_preds).tolist()
    results["confusion_matrix_gbm_loo"] = cm

    # Classification report
    report = classification_report(
        y, gbm_preds, target_names=["Human", "Agent"], output_dict=True,
    )
    results["classification_report_loo_gbm"] = report

    # Misclassified samples
    gbm_probs = np.array(
        results["models"]["GradientBoosting"]["loo_probabilities"]
    )
    misclassified = []
    for i in range(len(y)):
        if gbm_preds[i] != y[i]:
            addr = X.index[i] if hasattr(X, "index") else f"sample_{i}"
            misclassified.append({
                "address": str(addr),
                "true_label": "Agent" if y[i] == 1 else "Human",
                "predicted_label": "Agent" if gbm_preds[i] == 1 else "Human",
                "predicted_prob": round(float(gbm_probs[i]), 4),
            })
    results["misclassified_loo"] = misclassified
    if misclassified:
        logger.info("Misclassified %d samples in GBM LOO-CV:", len(misclassified))
        for m in misclassified:
            logger.info("  %s true=%s pred=%s prob=%.4f",
                        m["address"][:20], m["true_label"],
                        m["predicted_label"], m["predicted_prob"])

    return results


# ============================================================
# STEP 5: ABLATION STUDY
# ============================================================

def run_ablation(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Run ablation study: evaluate each feature group alone.

    Trains GBM with LOO-CV on each feature group individually,
    then compares to the full model.

    Returns:
        Dictionary mapping feature group name -> LOO-CV AUC.
    """
    logger.info("Running ablation study ...")
    gbm_template = get_models()["GradientBoosting"]
    feature_names = X.columns.tolist()

    ablation_results = {}

    # Full model
    full_loo = run_loo_cv(X.values.astype(float), y, gbm_template)
    ablation_results["all_features"] = {
        "n_features": len(feature_names),
        "features": feature_names,
        "loo_auc": full_loo["metrics"]["auc"],
        "loo_f1": full_loo["metrics"]["f1"],
        "loo_accuracy": full_loo["metrics"]["accuracy"],
    }
    logger.info("  all_features (%d): AUC=%.4f",
                len(feature_names), full_loo["metrics"]["auc"])

    # Each group alone
    for group_name, group_features in FEATURE_GROUPS.items():
        available = [f for f in group_features if f in X.columns]
        if not available:
            logger.warning("  %s: no features available, skipping", group_name)
            continue

        X_group = X[available].values.astype(float)
        loo_result = run_loo_cv(X_group, y, gbm_template)
        ablation_results[group_name] = {
            "n_features": len(available),
            "features": available,
            "loo_auc": loo_result["metrics"]["auc"],
            "loo_f1": loo_result["metrics"]["f1"],
            "loo_accuracy": loo_result["metrics"]["accuracy"],
        }
        logger.info("  %s (%d): AUC=%.4f",
                    group_name, len(available), loo_result["metrics"]["auc"])

    # Leave-one-group-out
    for group_name, group_features in FEATURE_GROUPS.items():
        remaining = [f for f in feature_names if f not in group_features]
        if not remaining:
            continue
        X_remaining = X[remaining].values.astype(float)
        loo_result = run_loo_cv(X_remaining, y, gbm_template)
        key = f"without_{group_name}"
        ablation_results[key] = {
            "n_features": len(remaining),
            "features": remaining,
            "loo_auc": loo_result["metrics"]["auc"],
            "loo_f1": loo_result["metrics"]["f1"],
            "loo_accuracy": loo_result["metrics"]["accuracy"],
        }
        logger.info("  without_%s (%d): AUC=%.4f",
                    group_name, len(remaining), loo_result["metrics"]["auc"])

    return ablation_results


# ============================================================
# STEP 6: BASELINE COMPARISON
# ============================================================

def run_baselines(X: pd.DataFrame, y: np.ndarray) -> dict:
    """Compare full model against heuristic and single-feature baselines.

    Baselines:
      1. Heuristic: burst_frequency > 0.1 -> agent
      2. Single best feature (highest univariate AUC)
      3. Full GBM model (LOO-CV)

    Returns:
        Dictionary of baseline results.
    """
    logger.info("Running baseline comparison ...")
    feature_names = X.columns.tolist()
    X_np = X.values.astype(float)
    baselines = {}

    # Baseline 1: Heuristic (burst_frequency threshold)
    if "burst_frequency" in feature_names:
        bf_idx = feature_names.index("burst_frequency")
        bf_values = X_np[:, bf_idx]
        heuristic_preds = (bf_values > 0.1).astype(int)
        baselines["heuristic_burst_frequency"] = {
            "rule": "burst_frequency > 0.1",
            "accuracy": round(float(accuracy_score(y, heuristic_preds)), 4),
            "f1": round(float(f1_score(y, heuristic_preds, zero_division=0)), 4),
            "precision": round(float(precision_score(
                y, heuristic_preds, zero_division=0)), 4),
            "recall": round(float(recall_score(
                y, heuristic_preds, zero_division=0)), 4),
        }
        logger.info("  Heuristic (burst>0.1): Acc=%.4f F1=%.4f",
                    baselines["heuristic_burst_frequency"]["accuracy"],
                    baselines["heuristic_burst_frequency"]["f1"])

    # Baseline 2: Single best feature (LR on 1 feature, LOO-CV)
    feat_aucs = per_feature_auc(X_np, y, feature_names)
    best_feature = list(feat_aucs.keys())[0]
    best_feature_idx = feature_names.index(best_feature)
    X_single = X_np[:, best_feature_idx].reshape(-1, 1)

    lr_template = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    single_loo = run_loo_cv(X_single, y, lr_template)
    baselines["single_best_feature"] = {
        "feature": best_feature,
        "univariate_auc": round(feat_aucs[best_feature], 4),
        "loo_auc": single_loo["metrics"]["auc"],
        "loo_f1": single_loo["metrics"]["f1"],
        "loo_accuracy": single_loo["metrics"]["accuracy"],
    }
    logger.info("  Single best (%s): LOO-AUC=%.4f F1=%.4f",
                best_feature,
                single_loo["metrics"]["auc"],
                single_loo["metrics"]["f1"])

    # Baseline 3: Full GBM (already computed, repeated here for comparison)
    gbm_template = get_models()["GradientBoosting"]
    full_loo = run_loo_cv(X_np, y, gbm_template)
    baselines["full_model_gbm"] = {
        "n_features": len(feature_names),
        "loo_auc": full_loo["metrics"]["auc"],
        "loo_f1": full_loo["metrics"]["f1"],
        "loo_accuracy": full_loo["metrics"]["accuracy"],
    }
    logger.info("  Full GBM (%d features): LOO-AUC=%.4f F1=%.4f",
                len(feature_names),
                full_loo["metrics"]["auc"],
                full_loo["metrics"]["f1"])

    return baselines


# ============================================================
# STEP 7: ROC CURVE DATA (for figure generation)
# ============================================================

def compute_roc_data(
    X: pd.DataFrame,
    y: np.ndarray,
) -> dict:
    """Compute ROC curve data for all three models using LOO-CV probabilities.

    Returns:
        Dictionary mapping model_name -> {fpr, tpr, auc}.
    """
    models = get_models()
    roc_data = {}

    for model_name, model_template in models.items():
        loo_result = run_loo_cv(X.values.astype(float), y, model_template)
        probs = loo_result["probabilities"]
        fpr, tpr, thresholds = roc_curve(y, probs)
        auc_val = roc_auc_score(y, probs)

        roc_data[model_name] = {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "auc": round(float(auc_val), 4),
        }

    return roc_data


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_pipeline(
    features_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    generate_figs: bool = True,
) -> dict:
    """Execute the full experiment pipeline.

    Args:
        features_path: Override path to features parquet file.
        output_dir: Override path for experiment results output.
        generate_figs: Whether to generate publication figures.

    Returns:
        Complete results dictionary.
    """
    if output_dir:
        experiments_out = Path(output_dir)
    else:
        experiments_out = EXPERIMENTS_DIR

    experiments_out.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 65)
    logger.info("Paper 1: Full Experiment Pipeline")
    logger.info("Run timestamp: %s", timestamp)
    logger.info("=" * 65)

    # Step 1: Load labels
    addresses, labels = load_labels()

    # Step 2: Load features
    feat_path = Path(features_path) if features_path else None
    X, y = load_features(addresses, labels, feat_path)

    # Step 3: Train and evaluate all models
    results = train_all_models(X, y)
    results["run_timestamp"] = timestamp

    # Step 4: Ablation study
    ablation = run_ablation(X, y)
    results["ablation_study"] = ablation

    # Step 5: Baseline comparison
    baselines = run_baselines(X, y)
    results["baseline_comparison"] = baselines

    # Step 6: ROC data for figures
    roc_data = compute_roc_data(X, y)
    results["roc_curve_data"] = roc_data

    # Step 7: Save results
    results_path = experiments_out / "pipeline_results.json"

    # Convert numpy types for JSON serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_serializable = json.loads(
        json.dumps(results, default=_convert)
    )

    with open(results_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # Step 8: Generate figures
    if generate_figs:
        _generate_figures(X, y, results_serializable)

    logger.info("=" * 65)
    logger.info("Pipeline complete!")
    logger.info("=" * 65)

    return results


def _generate_figures(
    X: pd.DataFrame,
    y: np.ndarray,
    results: dict,
) -> None:
    """Generate all publication figures."""
    try:
        from paper1_onchain_agent_id.figures.generate_figures import (
            generate_all_figures,
        )
        generate_all_figures(X, y, results, output_dir=str(FIGURES_DIR))
        logger.info("Figures saved to %s/", FIGURES_DIR)
    except ImportError:
        logger.warning(
            "Could not import generate_figures module. "
            "Run figures/generate_figures.py separately."
        )
    except Exception as exc:
        logger.warning("Figure generation failed: %s", exc)


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Paper 1: Full experiment pipeline (post-Delphi ready)"
    )
    parser.add_argument(
        "--features-path",
        default=None,
        help="Path to features parquet file (default: data/features.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for results (default: experiments/)",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Skip figure generation",
    )
    args = parser.parse_args()

    run_pipeline(
        features_path=args.features_path,
        output_dir=args.output_dir,
        generate_figs=not args.no_figures,
    )
