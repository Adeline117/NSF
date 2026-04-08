"""
Paper 1: Provenance-Only Labeling Pipeline (Leakage-Free)
==========================================================
The C1-C4 verifier's C3 gate uses tx_interval_cv, hour_entropy, and
burst_ratio — which are algebraically the same as three of the top
classifier features (tx_interval_std, active_hour_entropy,
burst_frequency). This causes target leakage: the classifier is
re-learning the labeling rule.

This script builds a leakage-free label set using ONLY external
provenance (contract-interaction source + manual name annotations),
then trains the classifier on that honest subset and reports honest
AUC alongside the leaky AUC for comparison.

Trusted label sources:
  - Original pilot (NaN source in features_expanded.parquet): 49 rows
    manually labeled as MEV bots (jaredfromsubway, Flashbots, etc.)
    and named humans (vitalik.eth, nick.eth, etc.).
  - strategy2_paper0: 4 rows externally validated by Paper 0 Delphi.
  - strategy_b_mev: 1 row from DeFi strategy bot list.
  - strategy_c_human: 10 rows from curated ENS human list.
  TOTAL: 64 rows (33 agents, 31 humans).

The 3,252 strategy_a_platform rows (OLAS/FET/NRN token holders) are
DROPPED from training because their agent/human labels came from
C1-C4 verification. They can optionally be scored transductively
for a class-distribution report.

Outputs:
  - experiments/expanded/pipeline_results_provenance.json
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "expanded"
FEATURES_PARQUET = DATA_DIR / "features_expanded.parquet"

sys.path.insert(0, str(PROJECT_ROOT.parent))


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
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=2000,
            random_state=42,
        ),
    }


def load_provenance_only() -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Load the trusted (provenance-only) subset and the remaining
    token-holder inference set.

    Returns:
        (X_trusted, y_trusted, df_inference)
    """
    logger.info("Loading %s ...", FEATURES_PARQUET)
    df = pd.read_parquet(FEATURES_PARQUET)
    logger.info("Loaded %d rows x %d columns", len(df), len(df.columns))

    # Trusted mask: NOT gated by C1-C4
    trusted_sources = {"strategy_c_human", "strategy2_paper0", "strategy_b_mev"}
    trusted_mask = df["source"].isna() | df["source"].isin(trusted_sources)
    df_trusted = df[trusted_mask].copy()
    df_inference = df[~trusted_mask].copy()

    logger.info("TRUSTED (provenance-only):")
    logger.info("  N=%d  Agents=%d  Humans=%d",
                len(df_trusted),
                int((df_trusted["label"] == 1).sum()),
                int((df_trusted["label"] == 0).sum()))
    logger.info("INFERENCE (token holders, dropped from training):")
    logger.info("  N=%d  (was leaky-labeled %d/%d agent/human via C1-C4)",
                len(df_inference),
                int((df_inference["label"] == 1).sum()),
                int((df_inference["label"] == 0).sum()))

    feature_cols = [c for c in ALL_FEATURES if c in df_trusted.columns]
    X = df_trusted[feature_cols].copy()
    y = df_trusted["label"].values.astype(int)

    # Impute NaN with column medians
    nan_mask = np.isnan(X.values)
    if nan_mask.any():
        col_medians = np.nanmedian(X.values, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X.iloc[nan_mask[:, j], j] = col_medians[j]

    # Clip extremes
    for col in X.columns:
        lo, hi = np.nanpercentile(X[col], [1, 99])
        X[col] = X[col].clip(lo, hi)

    return X, y, df_inference


def per_feature_auc(X_np, y, feature_names) -> dict:
    aucs = {}
    for i, name in enumerate(feature_names):
        col = X_np[:, i]
        if np.std(col) == 0:
            aucs[name] = 0.5
            continue
        auc = roc_auc_score(y, col)
        aucs[name] = max(auc, 1.0 - auc)
    return dict(sorted(aucs.items(), key=lambda x: x[1], reverse=True))


def run_loo(X_np, y, model_template) -> dict:
    """Leave-one-out CV — appropriate for N=64."""
    loo = LeaveOneOut()
    n = len(y)
    probs = np.zeros(n)
    preds = np.zeros(n, dtype=int)

    for train_idx, test_idx in loo.split(X_np):
        X_tr, X_te = X_np[train_idx], X_np[test_idx]
        y_tr = y[train_idx]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)

        clf = clone(model_template)
        clf.fit(X_tr, y_tr)

        if hasattr(clf, "predict_proba"):
            probs[test_idx[0]] = clf.predict_proba(X_te)[0, 1]
        else:
            probs[test_idx[0]] = clf.decision_function(X_te)[0]
        preds[test_idx[0]] = clf.predict(X_te)[0]

    return {
        "auc": round(float(roc_auc_score(y, probs)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision_human": round(float(precision_score(
            y, preds, pos_label=0, zero_division=0)), 4),
        "recall_human": round(float(recall_score(
            y, preds, pos_label=0, zero_division=0)), 4),
        "f1_human": round(float(f1_score(
            y, preds, pos_label=0, zero_division=0)), 4),
        "probs": probs.tolist(),
        "preds": preds.tolist(),
    }


def run_repeated_cv(X_np, y, model_template, n_splits=5, n_repeats=10) -> dict:
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42
    )
    fold_aucs, fold_f1s, fold_accs = [], [], []
    fold_precs, fold_recs = [], []

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
        fold_accs.append(accuracy_score(y_te, y_pred))
        fold_precs.append(precision_score(y_te, y_pred, zero_division=0))
        fold_recs.append(recall_score(y_te, y_pred, zero_division=0))

    return {
        "mean_auc": round(float(np.mean(fold_aucs)), 4),
        "std_auc": round(float(np.std(fold_aucs)), 4),
        "mean_f1": round(float(np.mean(fold_f1s)), 4),
        "mean_accuracy": round(float(np.mean(fold_accs)), 4),
        "mean_precision": round(float(np.mean(fold_precs)), 4),
        "mean_recall": round(float(np.mean(fold_recs)), 4),
        "n_splits": n_splits,
        "n_repeats": n_repeats,
    }


def main() -> dict:
    t0 = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info("=" * 65)
    logger.info("Paper 1: Provenance-Only Pipeline (leakage-free)")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 65)

    X, y, df_inference = load_provenance_only()
    feature_names = X.columns.tolist()
    X_np = X.values.astype(float)

    results: dict = {
        "run_timestamp": timestamp,
        "labeling_scheme": "provenance_only_no_c1c4",
        "leakage_note": (
            "C1-C4 is DEMOTED from labeling oracle to a downstream "
            "evaluation tool. Labels come from external provenance only "
            "(original pilot manual labels + Paper 0 Delphi validation "
            "+ curated MEV-bot/ENS-human lists). The 3252 platform-token "
            "holders are excluded from training because their prior "
            "labels were C1-C4 gated."
        ),
        "dataset": {
            "n_samples_trusted": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "n_inference_dropped": int(len(df_inference)),
            "inference_reason": (
                "strategy_a_platform rows were labeled via C1-C4 "
                "which leaks tx_interval_cv, hour_entropy, burst_ratio "
                "into the classifier features."
            ),
        },
        "models": {},
    }

    # Per-feature AUC
    feat_aucs = per_feature_auc(X_np, y, feature_names)
    results["per_feature_auc"] = {k: round(v, 4) for k, v in feat_aucs.items()}
    results["top10_features"] = list(feat_aucs.keys())[:10]
    logger.info("Top 10 features (univariate AUC):")
    for name in results["top10_features"]:
        logger.info("  %-40s AUC=%.4f", name, feat_aucs[name])

    # Train models
    models = get_models()
    for model_name, model_template in models.items():
        logger.info("Training %s ...", model_name)
        loo = run_loo(X_np, y, model_template)
        rcv = run_repeated_cv(X_np, y, model_template)

        results["models"][model_name] = {
            "loo_cv": {k: v for k, v in loo.items() if k not in ("probs", "preds")},
            "repeated_5fold_10x": rcv,
        }
        logger.info(
            "  LOO: AUC=%.4f F1=%.4f Acc=%.4f  |  5x10CV: AUC=%.4f±%.4f",
            loo["auc"], loo["f1"], loo["accuracy"],
            rcv["mean_auc"], rcv["std_auc"],
        )

    # GBM feature importance on full (trusted) data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_np)
    gbm = clone(models["GradientBoosting"])
    gbm.fit(X_scaled, y)
    importances = gbm.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    results["gbm_feature_importance"] = {
        feature_names[i]: round(float(importances[i]), 4)
        for i in sorted_idx
    }

    # Confusion matrix from GBM LOO
    gbm_loo = run_loo(X_np, y, models["GradientBoosting"])
    gbm_preds = np.array(gbm_loo["preds"])
    cm = confusion_matrix(y, gbm_preds).tolist()
    results["confusion_matrix_gbm_loo"] = cm
    results["classification_report_gbm_loo"] = classification_report(
        y, gbm_preds, target_names=["Human", "Agent"], output_dict=True,
    )

    # Transductive inference on the 3252 token holders
    logger.info("Transductive inference on %d token holders ...",
                len(df_inference))
    feat_cols = [c for c in ALL_FEATURES if c in df_inference.columns]
    X_inf = df_inference[feat_cols].copy()
    nan_mask = np.isnan(X_inf.values)
    if nan_mask.any():
        col_medians = np.nanmedian(X_inf.values, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X_inf.shape[1]):
            X_inf.iloc[nan_mask[:, j], j] = col_medians[j]
    X_inf_scaled = scaler.transform(X_inf.values)
    inf_probs = gbm.predict_proba(X_inf_scaled)[:, 1]
    inf_preds = (inf_probs >= 0.5).astype(int)

    # Compare to the leaky C1-C4 labels that exist on these rows
    leaky_labels = df_inference["label"].values.astype(int)
    agreement = float((inf_preds == leaky_labels).mean())

    results["transductive_inference"] = {
        "n_inference_rows": int(len(df_inference)),
        "predicted_agent_rate": round(float(inf_preds.mean()), 4),
        "predicted_human_rate": round(float((inf_preds == 0).mean()), 4),
        "mean_prob": round(float(inf_probs.mean()), 4),
        "agreement_with_leaky_c1c4_label": round(agreement, 4),
        "note": (
            "This is the GBM trained on 64 provenance-only labels, "
            "then applied transductively to the 3252 token-holder rows "
            "whose labels we DO NOT trust (they came from C1-C4 which "
            "leaks into classifier features)."
        ),
    }
    logger.info(
        "  Predicted agent rate=%.3f  |  agreement with leaky label=%.3f",
        inf_preds.mean(), agreement,
    )

    # Compute leaky-vs-provenance delta (headline result)
    # From earlier run: leaky GBM AUC on 3316 = 0.9803
    # Our provenance GBM AUC on 64: should be much more modest
    results["honest_vs_leaky_comparison"] = {
        "leaky_c1c4_gated": {
            "n_samples": 3316,
            "gbm_auc_5fold_5rep": 0.9803,
            "source": "experiments/expanded/pipeline_results.json",
            "issue": "C1-C4 labeling oracle uses the same 3 quantities (hour_entropy, burst_ratio, tx_interval_cv) that appear as top classifier features. Direct target leakage.",
        },
        "provenance_only": {
            "n_samples": int(len(y)),
            "gbm_auc_loo": results["models"]["GradientBoosting"]["loo_cv"]["auc"],
            "gbm_auc_5fold_10rep": results["models"][
                "GradientBoosting"]["repeated_5fold_10x"]["mean_auc"],
            "source": "experiments/expanded/pipeline_results_provenance.json",
        },
    }

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_serializable = json.loads(json.dumps(results, default=_convert))

    out_path = OUTPUT_DIR / "pipeline_results_provenance.json"
    with open(out_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    logger.info("Saved to %s", out_path)

    logger.info("=" * 65)
    logger.info("Provenance-only pipeline complete in %.1fs",
                results["elapsed_seconds"])
    logger.info("=" * 65)

    return results


if __name__ == "__main__":
    main()
