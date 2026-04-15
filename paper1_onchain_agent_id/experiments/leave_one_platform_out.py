"""
Paper 1: Leave-One-Platform-Out Cross-Validation
=================================================
Tests whether the agent-vs-human classifier generalises across
operational contexts (platforms/categories), not just across random
splits within the same platform mix.

For each platform with >=10 agent addresses AND a testable class
distribution, we:
  1. Hold out ALL addresses from that platform.
  2. Train RF and GBM on every other platform's addresses.
  3. Predict on the held-out platform.
  4. Record AUC (when both classes present in train+test) or accuracy.

Addresses come from data/features_provenance_v4.parquet (n=1,147)
with category labels from data/labels_provenance_v4.json.

Output: experiments/leave_one_platform_out_results.json
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
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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
OUTPUT_DIR = PROJECT_ROOT / "experiments"
FEATURES_PARQUET = DATA_DIR / "features_provenance_v4.parquet"

# 23 behavioral features (same groups as the main pipeline)
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
    """Return model templates matching the main pipeline."""
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
    }


def load_data() -> tuple[pd.DataFrame, np.ndarray, list[str], pd.Series]:
    """Load v4 provenance features and return X, y, feature_names, categories."""
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
    logger.info("Categories: %d unique", categories.nunique())

    return X, y, feature_cols, categories


def leave_one_platform_out(
    X: pd.DataFrame,
    y: np.ndarray,
    categories: pd.Series,
    model_template,
    model_name: str,
    min_platform_size: int = 10,
) -> dict:
    """Run LOPO-CV for a single model.

    For each held-out platform:
      - Train on all other platforms.
      - Predict on the held-out platform.
      - Record metrics; AUC only when both classes exist in the test set.

    Returns per-platform breakdown and aggregate statistics.
    """
    feature_cols = X.columns.tolist()
    X_np = X.values.astype(float)

    unique_cats = sorted(categories.unique())
    cat_counts = Counter(categories)

    per_platform = {}
    all_aucs = []
    all_accs = []
    all_y_true = []
    all_y_pred = []
    all_y_prob = []

    for cat in unique_cats:
        n_cat = cat_counts[cat]
        if n_cat < min_platform_size:
            logger.info("  SKIP %-45s n=%d (< %d)", cat, n_cat, min_platform_size)
            continue

        test_mask = (categories == cat).values
        train_mask = ~test_mask

        X_train, X_test = X_np[train_mask], X_np[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        n_train_agents = int(y_train.sum())
        n_train_humans = int((y_train == 0).sum())
        n_test_agents = int(y_test.sum())
        n_test_humans = int((y_test == 0).sum())

        # Need both classes in training set
        if n_train_agents == 0 or n_train_humans == 0:
            logger.info("  SKIP %-45s train has only one class", cat)
            continue

        # Scale features
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        clf = clone(model_template)
        clf.fit(X_train_s, y_train)

        y_pred = clf.predict(X_test_s)
        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_test_s)[:, 1]
        else:
            y_prob = clf.decision_function(X_test_s)

        acc = float(accuracy_score(y_test, y_pred))
        all_accs.append(acc)
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_y_prob.extend(y_prob.tolist())

        result = {
            "n_test": n_cat,
            "n_test_agents": n_test_agents,
            "n_test_humans": n_test_humans,
            "n_train": int(train_mask.sum()),
            "n_train_agents": n_train_agents,
            "n_train_humans": n_train_humans,
            "accuracy": round(acc, 4),
        }

        # AUC requires both classes in test set
        both_classes_in_test = n_test_agents > 0 and n_test_humans > 0
        if both_classes_in_test:
            auc = float(roc_auc_score(y_test, y_prob))
            result["auc"] = round(auc, 4)
            all_aucs.append(auc)
        else:
            result["auc"] = None
            result["auc_note"] = "Test set is single-class (all agent or all human)"

        # Additional metrics
        result["precision"] = round(float(
            precision_score(y_test, y_pred, zero_division=0)), 4)
        result["recall"] = round(float(
            recall_score(y_test, y_pred, zero_division=0)), 4)
        result["f1"] = round(float(
            f1_score(y_test, y_pred, zero_division=0)), 4)

        # Dominant class in this platform
        test_label = "agent" if n_test_agents > n_test_humans else "human"
        result["platform_dominant_class"] = test_label

        per_platform[cat] = result
        auc_str = f"AUC={result['auc']:.4f}" if result['auc'] is not None else "AUC=N/A (single-class)"
        logger.info("  %-45s n=%3d  %s  Acc=%.4f  F1=%.4f",
                     cat, n_cat, auc_str, acc, result["f1"])

    # Aggregate metrics
    # Pooled AUC across all held-out predictions (where both classes exist overall)
    all_y_true_arr = np.array(all_y_true)
    all_y_prob_arr = np.array(all_y_prob)
    all_y_pred_arr = np.array(all_y_pred)

    pooled_auc = None
    if len(np.unique(all_y_true_arr)) == 2:
        pooled_auc = round(float(roc_auc_score(all_y_true_arr, all_y_prob_arr)), 4)

    pooled_acc = round(float(accuracy_score(all_y_true_arr, all_y_pred_arr)), 4)
    pooled_f1 = round(float(f1_score(all_y_true_arr, all_y_pred_arr, zero_division=0)), 4)
    pooled_precision = round(float(
        precision_score(all_y_true_arr, all_y_pred_arr, zero_division=0)), 4)
    pooled_recall = round(float(
        recall_score(all_y_true_arr, all_y_pred_arr, zero_division=0)), 4)

    summary = {
        "model": model_name,
        "n_platforms_evaluated": len(per_platform),
        "n_platforms_with_auc": len(all_aucs),
        "mean_per_platform_auc": round(float(np.mean(all_aucs)), 4) if all_aucs else None,
        "std_per_platform_auc": round(float(np.std(all_aucs)), 4) if all_aucs else None,
        "min_per_platform_auc": round(float(np.min(all_aucs)), 4) if all_aucs else None,
        "max_per_platform_auc": round(float(np.max(all_aucs)), 4) if all_aucs else None,
        "mean_per_platform_accuracy": round(float(np.mean(all_accs)), 4),
        "pooled_auc": pooled_auc,
        "pooled_accuracy": pooled_acc,
        "pooled_f1": pooled_f1,
        "pooled_precision": pooled_precision,
        "pooled_recall": pooled_recall,
        "total_predictions": len(all_y_true),
        "per_platform": per_platform,
    }

    return summary


def main() -> dict:
    t0 = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 70)
    logger.info("Paper 1: Leave-One-Platform-Out Cross-Validation")
    logger.info("Timestamp: %s", timestamp)
    logger.info("=" * 70)

    X, y, feature_names, categories = load_data()

    # Log platform summary
    cat_counts = Counter(categories)
    logger.info("\nPlatform/category breakdown (n >= 10 will be evaluated):")
    for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
        cat_labels = y[categories == cat]
        agents = int(cat_labels.sum())
        humans = int((cat_labels == 0).sum())
        flag = " <-- SKIP (n<10)" if n < 10 else ""
        logger.info("  %-45s n=%3d  (agent=%d, human=%d)%s",
                     cat, n, agents, humans, flag)

    models = get_models()
    results = {
        "run_timestamp": timestamp,
        "experiment": "leave_one_platform_out_cross_validation",
        "description": (
            "For each platform/category with >= 10 addresses, hold out "
            "all addresses from that platform as the test set, train on "
            "all other platforms, and evaluate. Tests cross-platform "
            "generalisation — whether the classifier transfers to "
            "operational contexts it has never seen."
        ),
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(feature_names),
            "feature_names": feature_names,
            "n_categories": int(categories.nunique()),
            "categories": {
                cat: {
                    "n": int(n),
                    "n_agents": int(y[categories == cat].sum()),
                    "n_humans": int((y[categories == cat] == 0).sum()),
                }
                for cat, n in sorted(cat_counts.items(), key=lambda x: -x[1])
            },
        },
        "min_platform_size": 10,
        "models": {},
    }

    for model_name, model_template in models.items():
        logger.info("\n%s LOPO-CV:", model_name)
        logger.info("-" * 70)
        summary = leave_one_platform_out(
            X, y, categories, model_template, model_name,
            min_platform_size=10,
        )
        results["models"][model_name] = summary

        logger.info("\n  %s SUMMARY:", model_name)
        logger.info("    Platforms evaluated:      %d", summary["n_platforms_evaluated"])
        logger.info("    Platforms with AUC:       %d", summary["n_platforms_with_auc"])
        if summary["mean_per_platform_auc"] is not None:
            logger.info("    Mean per-platform AUC:   %.4f +/- %.4f",
                        summary["mean_per_platform_auc"],
                        summary["std_per_platform_auc"])
        logger.info("    Pooled AUC:              %s",
                     f"{summary['pooled_auc']:.4f}" if summary['pooled_auc'] else "N/A")
        logger.info("    Pooled accuracy:         %.4f", summary["pooled_accuracy"])
        logger.info("    Pooled F1:               %.4f", summary["pooled_f1"])

    # ── Cross-class analysis: agent-platform recall & human-platform specificity ──
    logger.info("\n" + "=" * 70)
    logger.info("CROSS-CLASS ANALYSIS (GradientBoosting)")
    logger.info("=" * 70)
    gbm_pp = results["models"]["GradientBoosting"]["per_platform"]

    agent_platforms = {k: v for k, v in gbm_pp.items()
                       if v["platform_dominant_class"] == "agent"}
    human_platforms = {k: v for k, v in gbm_pp.items()
                       if v["platform_dominant_class"] == "human"}

    # Agent-platform recall: fraction of agent addresses correctly predicted=1
    agent_correct = sum(int(v["accuracy"] * v["n_test"]) for v in agent_platforms.values())
    agent_total = sum(v["n_test"] for v in agent_platforms.values())
    agent_recall = agent_correct / agent_total if agent_total else 0.0

    # Human-platform specificity: fraction of human addresses correctly predicted=0
    human_correct = sum(int(v["accuracy"] * v["n_test"]) for v in human_platforms.values())
    human_total = sum(v["n_test"] for v in human_platforms.values())
    human_specificity = human_correct / human_total if human_total else 0.0

    logger.info("  Agent platforms held out (%d platforms, %d addresses):",
                len(agent_platforms), agent_total)
    for cat, v in sorted(agent_platforms.items(), key=lambda x: -x[1]["n_test"]):
        logger.info("    %-40s n=%3d  recall=%.2f%%",
                     cat, v["n_test"], v["accuracy"] * 100)
    logger.info("    WEIGHTED AGENT-PLATFORM RECALL: %.2f%%", agent_recall * 100)

    logger.info("  Human platforms held out (%d platforms, %d addresses):",
                len(human_platforms), human_total)
    for cat, v in sorted(human_platforms.items(), key=lambda x: -x[1]["n_test"]):
        logger.info("    %-40s n=%3d  specificity=%.2f%%",
                     cat, v["n_test"], v["accuracy"] * 100)
    logger.info("    WEIGHTED HUMAN-PLATFORM SPECIFICITY: %.2f%%",
                human_specificity * 100)

    results["cross_class_analysis_gbm"] = {
        "agent_platform_recall": round(agent_recall, 4),
        "agent_platforms_n": agent_total,
        "agent_platforms_count": len(agent_platforms),
        "human_platform_specificity": round(human_specificity, 4),
        "human_platforms_n": human_total,
        "human_platforms_count": len(human_platforms),
        "interpretation": (
            "Agent-platform recall = fraction of held-out agent-platform "
            "addresses correctly classified as agents. Human-platform "
            "specificity = fraction of held-out human-platform addresses "
            "correctly classified as humans. Each platform is entirely "
            "unseen during training, so this measures true cross-platform "
            "transfer. Because each category is single-class, accuracy "
            "equals recall for agent platforms and specificity for human "
            "platforms."
        ),
        "per_agent_platform": {
            cat: {"n": v["n_test"], "recall": v["accuracy"]}
            for cat, v in agent_platforms.items()
        },
        "per_human_platform": {
            cat: {"n": v["n_test"], "specificity": v["accuracy"]}
            for cat, v in human_platforms.items()
        },
    }

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    # JSON-safe serialization
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    results_safe = json.loads(json.dumps(results, default=_convert))

    out_path = OUTPUT_DIR / "leave_one_platform_out_results.json"
    with open(out_path, "w") as f:
        json.dump(results_safe, f, indent=2)
    logger.info("\nResults saved to %s", out_path)

    logger.info("=" * 70)
    logger.info("LOPO-CV complete in %.1fs", results["elapsed_seconds"])
    logger.info("=" * 70)

    return results


if __name__ == "__main__":
    main()
