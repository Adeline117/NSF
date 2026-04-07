"""
Cross-Project Transfer Analysis
=================================
Test whether AI Sybil detection transfers across airdrop projects.

Methodology:
- Leave-One-Project-Out (LOPO): train on 15 projects, test on 1
- Temporal transfer: train on pre-2024 projects, test on 2024+
- Protocol transfer: train on DeFi airdrops, test on NFT airdrops

HasciDB Projects (chronological):
  2020: uniswap, 1inch, badger
  2021: ens, dydx, ampleforth, paraswap
  2022: apecoin, x2y2, looksrare, gitcoin, blur_s1
  2023: blur_s2
  2024: eigenlayer, etherfi, pengu

Analysis:
- Per-project AUC with confidence intervals
- Feature importance stability across projects
- Indicator threshold sensitivity analysis

References:
- Li et al. CHI'26: HasciDB five-indicator framework, 16 projects
- Wen et al.: pre-airdrop-detection temporal generalization
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats


# ============================================================
# CONSTANTS
# ============================================================

HASCIDB_FEATURES = ["BT", "BW", "HF", "RF", "MA"]

AI_FEATURES = [
    "gas_price_precision",
    "hour_entropy",
    "behavioral_consistency",
    "action_sequence_perplexity",
    "error_recovery_pattern",
    "response_latency_variance",
    "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]

ALL_FEATURES = HASCIDB_FEATURES + AI_FEATURES

HASCIDB_THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}

# Chronological ordering
PROJECTS_CHRONOLOGICAL = {
    2020: ["uniswap", "1inch", "badger"],
    2021: ["ens", "dydx", "ampleforth", "paraswap"],
    2022: ["apecoin", "x2y2", "looksrare", "gitcoin", "blur_s1"],
    2023: ["blur_s2"],
    2024: ["eigenlayer", "etherfi", "pengu"],
}

# Protocol categories
DEFI_PROJECTS = [
    "uniswap", "1inch", "dydx", "paraswap", "badger",
    "ampleforth", "eigenlayer", "etherfi",
]
NFT_PROJECTS = [
    "ens", "apecoin", "x2y2", "looksrare", "blur_s1", "blur_s2",
]
GOVERNANCE_PROJECTS = ["gitcoin", "pengu"]


# ============================================================
# RESULT DATACLASS
# ============================================================

@dataclass
class TransferResult:
    """Results from a single transfer evaluation."""
    train_projects: list[str]
    test_project: str
    auc: float
    average_precision: float
    n_train: int
    n_test: int
    feature_importances: dict = field(default_factory=dict)


# ============================================================
# LEAVE-ONE-PROJECT-OUT (LOPO)
# ============================================================

def lopo_analysis(
    project_datasets: dict[str, pd.DataFrame],
    features: Optional[list[str]] = None,
    model_params: Optional[dict] = None,
) -> dict:
    """Leave-One-Project-Out cross-validation across HasciDB projects.

    For each of the 16 projects, trains on the remaining 15 and tests
    on the held-out project. Reports per-project AUC and aggregate
    statistics with 95% confidence intervals.

    Args:
        project_datasets: Dict mapping project name -> DataFrame.
                          Each DataFrame must have columns matching
                          `features` plus a "label" column.
        features: Feature columns. Defaults to ALL_FEATURES (13).
        model_params: GradientBoosting hyperparameters.

    Returns:
        Dictionary with per-project results, mean AUC, CI, and
        feature importance stability analysis.
    """
    features = features or ALL_FEATURES
    model_params = model_params or {
        "n_estimators": 200, "max_depth": 5,
        "learning_rate": 0.1, "subsample": 0.8, "random_state": 42,
    }

    project_names = sorted(project_datasets.keys())
    results = {}
    all_importances = {}

    for test_proj in project_names:
        train_dfs = [
            project_datasets[p] for p in project_names if p != test_proj
        ]
        train_data = pd.concat(train_dfs, ignore_index=True)
        test_data = project_datasets[test_proj]

        # Skip projects with too few samples or single-class
        if len(test_data) < 10 or test_data["label"].nunique() < 2:
            continue

        clf = GradientBoostingClassifier(**model_params)
        clf.fit(train_data[features].values, train_data["label"].values)
        y_pred = clf.predict_proba(test_data[features].values)[:, 1]
        y_test = test_data["label"].values

        auc = roc_auc_score(y_test, y_pred)
        ap = average_precision_score(y_test, y_pred)

        # Feature importances for this fold
        fold_importances = dict(zip(features, clf.feature_importances_))
        all_importances[test_proj] = fold_importances

        results[test_proj] = TransferResult(
            train_projects=[p for p in project_names if p != test_proj],
            test_project=test_proj,
            auc=float(auc),
            average_precision=float(ap),
            n_train=len(train_data),
            n_test=len(test_data),
            feature_importances=fold_importances,
        )

    # Aggregate statistics
    aucs = [r.auc for r in results.values()]
    mean_auc = float(np.mean(aucs)) if aucs else 0.0
    std_auc = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0
    ci_95 = 1.96 * std_auc / np.sqrt(len(aucs)) if aucs else 0.0

    # Feature importance stability
    importance_stability = _compute_importance_stability(all_importances, features)

    return {
        "per_project": {k: {
            "auc": v.auc,
            "ap": v.average_precision,
            "n_test": v.n_test,
        } for k, v in results.items()},
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "ci_95_lower": mean_auc - ci_95,
        "ci_95_upper": mean_auc + ci_95,
        "n_projects": len(aucs),
        "best_project": max(results.items(), key=lambda x: x[1].auc)[0] if results else None,
        "worst_project": min(results.items(), key=lambda x: x[1].auc)[0] if results else None,
        "importance_stability": importance_stability,
    }


# ============================================================
# TEMPORAL TRANSFER
# ============================================================

def temporal_transfer(
    project_datasets: dict[str, pd.DataFrame],
    train_cutoff_year: int = 2023,
    features: Optional[list[str]] = None,
    model_params: Optional[dict] = None,
) -> dict:
    """Train on pre-cutoff projects, test on post-cutoff projects.

    Tests whether detection generalizes to temporally later airdrops.

    Args:
        project_datasets: Dict mapping project name -> DataFrame.
        train_cutoff_year: Train on years <= cutoff, test on > cutoff.
        features: Feature columns.
        model_params: GradientBoosting hyperparameters.

    Returns:
        Dictionary with overall and per-project results.
    """
    features = features or ALL_FEATURES
    model_params = model_params or {
        "n_estimators": 200, "max_depth": 5, "random_state": 42,
    }

    train_projects = []
    test_projects = []
    for year, projs in PROJECTS_CHRONOLOGICAL.items():
        for p in projs:
            if p in project_datasets:
                if year <= train_cutoff_year:
                    train_projects.append(p)
                else:
                    test_projects.append(p)

    if not train_projects or not test_projects:
        return {"error": "Insufficient projects for temporal split",
                "train_projects": train_projects,
                "test_projects": test_projects}

    train_data = pd.concat(
        [project_datasets[p] for p in train_projects], ignore_index=True
    )
    test_data = pd.concat(
        [project_datasets[p] for p in test_projects], ignore_index=True
    )

    clf = GradientBoostingClassifier(**model_params)
    clf.fit(train_data[features].values, train_data["label"].values)

    # Overall
    y_pred_all = clf.predict_proba(test_data[features].values)[:, 1]
    y_test_all = test_data["label"].values
    overall_auc = roc_auc_score(y_test_all, y_pred_all)
    overall_ap = average_precision_score(y_test_all, y_pred_all)

    # Per-project
    per_project = {}
    for p in test_projects:
        p_data = project_datasets[p]
        if p_data["label"].nunique() < 2:
            continue
        y_pred_p = clf.predict_proba(p_data[features].values)[:, 1]
        per_project[p] = {
            "auc": float(roc_auc_score(p_data["label"].values, y_pred_p)),
            "ap": float(average_precision_score(p_data["label"].values, y_pred_p)),
            "n": len(p_data),
        }

    return {
        "train_projects": train_projects,
        "test_projects": test_projects,
        "train_cutoff_year": train_cutoff_year,
        "overall_auc": float(overall_auc),
        "overall_ap": float(overall_ap),
        "per_project": per_project,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "feature_importances": dict(zip(features, clf.feature_importances_)),
    }


# ============================================================
# PROTOCOL TRANSFER
# ============================================================

def protocol_transfer(
    project_datasets: dict[str, pd.DataFrame],
    train_category: str = "defi",
    test_category: str = "nft",
    features: Optional[list[str]] = None,
    model_params: Optional[dict] = None,
) -> dict:
    """Train on one protocol category, test on another.

    Categories:
    - defi: uniswap, 1inch, dydx, paraswap, badger, ampleforth,
            eigenlayer, etherfi
    - nft: ens, apecoin, x2y2, looksrare, blur_s1, blur_s2
    - governance: gitcoin, pengu

    Args:
        project_datasets: Dict mapping project name -> DataFrame.
        train_category: Category to train on.
        test_category: Category to test on.
        features: Feature columns.
        model_params: GradientBoosting hyperparameters.

    Returns:
        Dictionary with overall and per-project results.
    """
    features = features or ALL_FEATURES
    model_params = model_params or {
        "n_estimators": 200, "max_depth": 5, "random_state": 42,
    }

    category_map = {
        "defi": DEFI_PROJECTS,
        "nft": NFT_PROJECTS,
        "governance": GOVERNANCE_PROJECTS,
    }

    train_projects = [
        p for p in category_map.get(train_category, [])
        if p in project_datasets
    ]
    test_projects = [
        p for p in category_map.get(test_category, [])
        if p in project_datasets
    ]

    if not train_projects or not test_projects:
        return {"error": f"No projects found for categories: "
                         f"train={train_category}, test={test_category}"}

    train_data = pd.concat(
        [project_datasets[p] for p in train_projects], ignore_index=True
    )
    test_data = pd.concat(
        [project_datasets[p] for p in test_projects], ignore_index=True
    )

    clf = GradientBoostingClassifier(**model_params)
    clf.fit(train_data[features].values, train_data["label"].values)

    y_pred = clf.predict_proba(test_data[features].values)[:, 1]
    y_test = test_data["label"].values

    # Per-project
    per_project = {}
    for p in test_projects:
        p_data = project_datasets[p]
        if p_data["label"].nunique() < 2:
            continue
        y_pred_p = clf.predict_proba(p_data[features].values)[:, 1]
        per_project[p] = {
            "auc": float(roc_auc_score(p_data["label"].values, y_pred_p)),
            "ap": float(average_precision_score(p_data["label"].values, y_pred_p)),
            "n": len(p_data),
        }

    return {
        "train_category": train_category,
        "test_category": test_category,
        "train_projects": train_projects,
        "test_projects": test_projects,
        "overall_auc": float(roc_auc_score(y_test, y_pred)),
        "overall_ap": float(average_precision_score(y_test, y_pred)),
        "per_project": per_project,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "feature_importances": dict(zip(features, clf.feature_importances_)),
    }


# ============================================================
# FEATURE IMPORTANCE STABILITY
# ============================================================

def _compute_importance_stability(
    all_importances: dict[str, dict[str, float]],
    features: list[str],
) -> dict:
    """Analyze feature importance stability across LOPO folds.

    Computes:
    - Mean and std of importance per feature across folds.
    - Rank stability (Kendall's tau between folds).
    - Consistent top-K features across all folds.

    Args:
        all_importances: Dict mapping project -> {feature: importance}.
        features: List of feature names.

    Returns:
        Dictionary with stability metrics.
    """
    if not all_importances:
        return {}

    # Collect importance vectors
    imp_matrix = []
    for proj, imp_dict in all_importances.items():
        imp_matrix.append([imp_dict.get(f, 0.0) for f in features])
    imp_array = np.array(imp_matrix)  # shape: (n_folds, n_features)

    # Per-feature statistics
    per_feature = {}
    for i, feat in enumerate(features):
        vals = imp_array[:, i]
        per_feature[feat] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "cv": float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0.0,
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "type": "AI-specific" if feat in AI_FEATURES else "HasciDB",
        }

    # Rank stability: pairwise Kendall's tau between folds
    n_folds = len(imp_matrix)
    taus = []
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            tau, _ = stats.kendalltau(imp_array[i], imp_array[j])
            taus.append(tau)

    mean_tau = float(np.mean(taus)) if taus else 0.0

    # Top-5 consistency: how often does each feature appear in top-5
    top5_counts = {f: 0 for f in features}
    for row in imp_matrix:
        ranked = sorted(zip(features, row), key=lambda x: -x[1])
        for feat, _ in ranked[:5]:
            top5_counts[feat] += 1

    top5_consistency = {
        f: count / n_folds for f, count in top5_counts.items()
    }

    return {
        "per_feature": per_feature,
        "mean_kendall_tau": mean_tau,
        "top5_consistency": top5_consistency,
        "n_folds": n_folds,
    }


# ============================================================
# THRESHOLD SENSITIVITY ANALYSIS
# ============================================================

def threshold_sensitivity(
    project_datasets: dict[str, pd.DataFrame],
    indicator: str = "BT",
    threshold_range: Optional[list[float]] = None,
) -> dict:
    """Analyze how HasciDB indicator threshold changes affect detection.

    Varies a single indicator's threshold and measures the impact on
    sybil detection rate across projects.

    Args:
        project_datasets: Dict mapping project name -> DataFrame.
        indicator: Which indicator to vary (BT, BW, HF, RF, MA).
        threshold_range: List of threshold values to test.

    Returns:
        Dictionary with per-threshold detection rates.
    """
    default_ranges = {
        "BT": [2, 3, 4, 5, 6, 7, 8, 10],
        "BW": [3, 5, 7, 10, 12, 15, 20],
        "HF": [0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95],
        "RF": [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80],
        "MA": [2, 3, 4, 5, 7, 10, 15],
    }
    threshold_range = threshold_range or default_ranges.get(indicator, [])

    all_data = pd.concat(list(project_datasets.values()), ignore_index=True)
    y_true = all_data["label"].values

    results = []
    for threshold in threshold_range:
        # Modify only this indicator's threshold
        modified_thresholds = dict(HASCIDB_THRESHOLDS)
        modified_thresholds[indicator] = threshold

        # Compute sybil flag with modified threshold
        ops_flag = (
            (all_data["BT"] >= modified_thresholds["BT"])
            | (all_data["BW"] >= modified_thresholds["BW"])
            | (all_data["HF"] >= modified_thresholds["HF"])
        )
        fund_flag = (
            (all_data["RF"] >= modified_thresholds["RF"])
            | (all_data["MA"] >= modified_thresholds["MA"])
        )
        y_pred = (ops_flag | fund_flag).astype(int).values

        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        results.append({
            "threshold": float(threshold),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
        })

    return {
        "indicator": indicator,
        "original_threshold": HASCIDB_THRESHOLDS[indicator],
        "results": results,
        "n_total": len(all_data),
    }


# ============================================================
# FULL TRANSFER ANALYSIS PIPELINE
# ============================================================

def run_full_transfer_analysis(
    project_datasets: dict[str, pd.DataFrame],
    features: Optional[list[str]] = None,
) -> dict:
    """Run the complete cross-project transfer analysis pipeline.

    Includes:
    1. LOPO cross-validation (HasciDB-only, AI-only, Enhanced)
    2. Temporal transfer (train pre-2024, test 2024+)
    3. Protocol transfer (DeFi -> NFT, NFT -> DeFi)
    4. Threshold sensitivity for each indicator

    Args:
        project_datasets: Dict mapping project name -> DataFrame.
        features: Feature columns. Defaults to ALL_FEATURES.

    Returns:
        Comprehensive dictionary with all analysis results.
    """
    features = features or ALL_FEATURES
    results = {}

    # 1. LOPO with different feature sets
    print("Running LOPO analysis...")
    results["lopo_hascidb"] = lopo_analysis(
        project_datasets, features=HASCIDB_FEATURES
    )
    results["lopo_ai_only"] = lopo_analysis(
        project_datasets, features=AI_FEATURES
    )
    results["lopo_enhanced"] = lopo_analysis(
        project_datasets, features=ALL_FEATURES
    )

    # 2. Temporal transfer
    print("Running temporal transfer analysis...")
    for cutoff in [2021, 2022, 2023]:
        key = f"temporal_{cutoff}"
        results[key] = temporal_transfer(
            project_datasets, train_cutoff_year=cutoff, features=features
        )

    # 3. Protocol transfer
    print("Running protocol transfer analysis...")
    results["protocol_defi_to_nft"] = protocol_transfer(
        project_datasets, train_category="defi", test_category="nft",
        features=features,
    )
    results["protocol_nft_to_defi"] = protocol_transfer(
        project_datasets, train_category="nft", test_category="defi",
        features=features,
    )

    # 4. Threshold sensitivity
    print("Running threshold sensitivity analysis...")
    for indicator in HASCIDB_THRESHOLDS:
        results[f"threshold_{indicator}"] = threshold_sensitivity(
            project_datasets, indicator=indicator
        )

    # Summary
    results["summary"] = {
        "lopo_enhanced_mean_auc": results["lopo_enhanced"]["mean_auc"],
        "lopo_hascidb_mean_auc": results["lopo_hascidb"]["mean_auc"],
        "lopo_ai_mean_auc": results["lopo_ai_only"]["mean_auc"],
        "temporal_2023_auc": results["temporal_2023"].get("overall_auc", None),
        "defi_to_nft_auc": results["protocol_defi_to_nft"].get("overall_auc", None),
        "nft_to_defi_auc": results["protocol_nft_to_defi"].get("overall_auc", None),
    }

    return results


# ============================================================
# MAIN (demo)
# ============================================================

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from adversarial.ai_sybil_generator import (
        generate_ai_sybil_dataframe,
        EvasionLevel,
    )
    from experiments.pilot_sybil_evasion import (
        generate_real_calibrated_legitimate,
        generate_real_calibrated_traditional_sybil,
    )

    print("=" * 70)
    print("Cross-Project Transfer Analysis Demo")
    print("=" * 70)

    # Simulate per-project datasets
    project_datasets = {}
    for i, proj in enumerate(PROJECTS_CHRONOLOGICAL.get(2020, []) +
                              PROJECTS_CHRONOLOGICAL.get(2021, []) +
                              PROJECTS_CHRONOLOGICAL.get(2022, []) +
                              PROJECTS_CHRONOLOGICAL.get(2023, []) +
                              PROJECTS_CHRONOLOGICAL.get(2024, [])):
        seed = 100 + i * 7
        legit = generate_real_calibrated_legitimate(200, seed=seed)
        trad = generate_real_calibrated_traditional_sybil(60, seed=seed + 1)
        ai = generate_ai_sybil_dataframe(40, level=EvasionLevel.MODERATE, seed=seed + 2)
        project_datasets[proj] = pd.concat([legit, trad, ai], ignore_index=True)

    # LOPO analysis
    print("\n--- LOPO (Enhanced, 13 features) ---")
    lopo_result = lopo_analysis(project_datasets)
    print(f"  Mean AUC: {lopo_result['mean_auc']:.3f} "
          f"[{lopo_result['ci_95_lower']:.3f}, {lopo_result['ci_95_upper']:.3f}]")
    print(f"  Best:  {lopo_result['best_project']}")
    print(f"  Worst: {lopo_result['worst_project']}")

    # Temporal transfer
    print("\n--- Temporal Transfer (train <=2023, test 2024) ---")
    temporal_result = temporal_transfer(project_datasets, train_cutoff_year=2023)
    if "error" not in temporal_result:
        print(f"  Overall AUC: {temporal_result['overall_auc']:.3f}")
        for p, info in temporal_result["per_project"].items():
            print(f"    {p}: AUC={info['auc']:.3f}")

    # Protocol transfer
    print("\n--- Protocol Transfer (DeFi -> NFT) ---")
    protocol_result = protocol_transfer(
        project_datasets, train_category="defi", test_category="nft"
    )
    if "error" not in protocol_result:
        print(f"  Overall AUC: {protocol_result['overall_auc']:.3f}")
