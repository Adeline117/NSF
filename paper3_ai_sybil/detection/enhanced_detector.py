"""
Enhanced AI-Sybil Detector
===========================
Extends HasciDB 5-indicator baseline with 8 AI-specific features.

Baseline (HasciDB CHI'26):
- BT, BW, HF, RF, MA -> ops_flag OR fund_flag

AI-Specific Features (from Paper 1):
1. gas_price_precision: Agent computes exact gas vs human round numbers
2. hour_entropy: No circadian rhythm for 24/7 agents
3. behavioral_consistency: Cross-address correlation from same LLM
4. action_sequence_perplexity: LLM sequences have characteristic range
5. error_recovery_pattern: Systematic retry/fallback
6. response_latency_variance: LLM inference time signature
7. gas_nonce_gap_regularity: Regular nonce increments
8. eip1559_tip_precision: Precise priority fee calculation

Models:
- GradientBoosting (primary)
- LightGBM (comparison with pre-airdrop-detection)
- RandomForest (baseline)

Evaluation:
- Leave-One-Project-Out (LOPO) cross-validation across 16 HasciDB projects
- Temporal split: train on earlier airdrops, test on later ones
- Comparison baselines: HasciDB rules, HasciDB ML, pre-airdrop LightGBM,
  TrustaLabs graph mining, Arbitrum Louvain

References:
- Li et al. CHI'26: HasciDB five-indicator framework
- Wen et al.: pre-airdrop-detection LightGBM, AUC 0.793 @ T-30
- UW-DCL/Blur: ARTEMIS GNN, LLMhunter
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False


# ============================================================
# CONSTANTS
# ============================================================

HASCIDB_THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}

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

HASCIDB_PROJECTS = [
    "uniswap", "ens", "1inch", "blur_s1", "blur_s2", "gitcoin",
    "looksrare", "eigenlayer", "x2y2", "dydx", "apecoin",
    "paraswap", "badger", "ampleforth", "etherfi", "pengu",
]

# Chronological ordering for temporal splits
PROJECTS_CHRONOLOGICAL = {
    2020: ["uniswap", "1inch", "badger"],
    2021: ["ens", "dydx", "ampleforth", "paraswap"],
    2022: ["apecoin", "x2y2", "looksrare", "gitcoin", "blur_s1"],
    2023: ["blur_s2"],
    2024: ["eigenlayer", "etherfi", "pengu"],
}


# ============================================================
# EVALUATION RESULT DATACLASS
# ============================================================

@dataclass
class DetectorResult:
    """Results from a single detector evaluation."""
    model_name: str
    features_used: list[str]
    auc: float
    average_precision: float
    precision_at_90_recall: float = 0.0
    feature_importances: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


# ============================================================
# HASCIDB BASELINE DETECTORS
# ============================================================

def hascidb_rule_based_predict(df: pd.DataFrame) -> np.ndarray:
    """Replicate HasciDB's rule-based classification.

    ops_flag  = (BT >= 5) OR (BW >= 10) OR (HF >= 0.80)
    fund_flag = (RF >= 0.50) OR (MA >= 5)
    is_sybil  = ops_flag OR fund_flag
    """
    ops_flag = (df["BT"] >= 5) | (df["BW"] >= 10) | (df["HF"] >= 0.80)
    fund_flag = (df["RF"] >= 0.50) | (df["MA"] >= 5)
    return (ops_flag | fund_flag).astype(int).values


def hascidb_continuous_score(df: pd.DataFrame) -> np.ndarray:
    """Compute a continuous sybil score from HasciDB indicators.

    Normalized distance to thresholds, combined with max-aggregation.
    Used for AUC computation with the rule-based detector.
    """
    scores = np.column_stack([
        np.clip(df["BT"].values / 5.0, 0, 1),
        np.clip(df["BW"].values / 10.0, 0, 1),
        np.clip(df["HF"].values / 0.80, 0, 1),
        np.clip(df["RF"].values / 0.50, 0, 1),
        np.clip(df["MA"].values / 5.0, 0, 1),
    ])
    return scores.max(axis=1)


# ============================================================
# ENHANCED DETECTOR
# ============================================================

class EnhancedDetector:
    """Enhanced AI-Sybil detector combining HasciDB and AI-specific features.

    Supports three model backends:
    - GradientBoosting (primary, from pilot v2)
    - LightGBM (comparison with pre-airdrop-detection)
    - RandomForest (baseline)

    Usage:
        detector = EnhancedDetector(model_type="gbm")
        detector.fit(X_train, y_train)
        scores = detector.predict_proba(X_test)
        result = detector.evaluate(X_test, y_test)
    """

    MODEL_CONFIGS = {
        "gbm": {
            "class": GradientBoostingClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "random_state": 42,
            },
        },
        "rf": {
            "class": RandomForestClassifier,
            "params": {
                "n_estimators": 200,
                "max_depth": 10,
                "min_samples_leaf": 5,
                "random_state": 42,
                "n_jobs": -1,
            },
        },
    }

    def __init__(
        self,
        model_type: str = "gbm",
        features: Optional[list[str]] = None,
        scale_features: bool = False,
    ):
        """Initialize the enhanced detector.

        Args:
            model_type: One of "gbm", "lgbm", "rf".
            features: Feature columns to use. Defaults to ALL_FEATURES.
            scale_features: Whether to standardize features before training.
        """
        self.model_type = model_type
        self.features = features or ALL_FEATURES
        self.scale_features = scale_features
        self.scaler = StandardScaler() if scale_features else None
        self.model = None
        self._build_model()

    def _build_model(self):
        """Instantiate the underlying model."""
        if self.model_type == "lgbm":
            if not HAS_LIGHTGBM:
                raise ImportError(
                    "LightGBM not installed. Install with: pip install lightgbm"
                )
            self.model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1,
            )
        elif self.model_type in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[self.model_type]
            self.model = config["class"](**config["params"])
        else:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                             f"Choose from: gbm, lgbm, rf")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and optionally scale feature matrix."""
        X = df[self.features].values.astype(np.float64)
        if self.scale_features and self.scaler is not None:
            X = self.scaler.transform(X)
        return X

    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None):
        """Train the detector.

        Args:
            df: DataFrame with feature columns and optionally a "label" column.
            y: Labels (1=sybil, 0=legitimate). If None, reads from df["label"].
        """
        if y is None:
            y = df["label"].values
        X = df[self.features].values.astype(np.float64)
        if self.scale_features and self.scaler is not None:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return sybil probability scores.

        Args:
            df: DataFrame with feature columns.

        Returns:
            1-D array of P(sybil) for each row.
        """
        X = self._prepare_features(df)
        return self.model.predict_proba(X)[:, 1]

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Return binary predictions.

        Args:
            df: DataFrame with feature columns.
            threshold: Decision threshold on P(sybil).

        Returns:
            1-D array of 0/1 predictions.
        """
        return (self.predict_proba(df) >= threshold).astype(int)

    def evaluate(
        self,
        df: pd.DataFrame,
        y: Optional[np.ndarray] = None,
    ) -> DetectorResult:
        """Evaluate the detector on a test set.

        Args:
            df: DataFrame with feature columns and optionally "label".
            y: True labels. If None, reads from df["label"].

        Returns:
            DetectorResult with AUC, AP, precision@90recall, importances.
        """
        if y is None:
            y = df["label"].values

        scores = self.predict_proba(df)
        auc = roc_auc_score(y, scores)
        ap = average_precision_score(y, scores)

        # Precision at 90% recall
        precision_vals, recall_vals, _ = precision_recall_curve(y, scores)
        p_at_90 = 0.0
        for p, r in zip(precision_vals, recall_vals):
            if r >= 0.90:
                p_at_90 = max(p_at_90, p)

        # Feature importances
        importances = {}
        if hasattr(self.model, "feature_importances_"):
            for feat, imp in zip(self.features, self.model.feature_importances_):
                importances[feat] = float(imp)

        return DetectorResult(
            model_name=f"Enhanced_{self.model_type}",
            features_used=self.features,
            auc=float(auc),
            average_precision=float(ap),
            precision_at_90_recall=float(p_at_90),
            feature_importances=importances,
        )

    def cross_validate(
        self,
        df: pd.DataFrame,
        y: Optional[np.ndarray] = None,
        n_folds: int = 5,
    ) -> dict:
        """Run stratified K-fold cross-validation.

        Args:
            df: DataFrame with feature columns and optionally "label".
            y: Labels. If None, reads from df["label"].
            n_folds: Number of folds.

        Returns:
            Dictionary with mean AUC, std, and per-fold results.
        """
        if y is None:
            y = df["label"].values
        X = df[self.features].values.astype(np.float64)
        if self.scale_features and self.scaler is not None:
            X = self.scaler.fit_transform(X)

        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        auc_scores = cross_val_score(self.model, X, y, cv=cv, scoring="roc_auc")

        return {
            "mean_auc": float(auc_scores.mean()),
            "std_auc": float(auc_scores.std()),
            "per_fold": [float(s) for s in auc_scores],
            "n_folds": n_folds,
        }

    def feature_importance_ranking(self) -> list[tuple[str, float]]:
        """Return features ranked by importance (descending).

        Returns:
            List of (feature_name, importance) tuples.
        """
        if not hasattr(self.model, "feature_importances_"):
            return []
        pairs = list(zip(self.features, self.model.feature_importances_))
        return sorted(pairs, key=lambda x: -x[1])


# ============================================================
# LEAVE-ONE-PROJECT-OUT (LOPO) CROSS-VALIDATION
# ============================================================

def lopo_cross_validation(
    project_datasets: dict[str, pd.DataFrame],
    model_type: str = "gbm",
    features: Optional[list[str]] = None,
) -> dict:
    """Leave-One-Project-Out cross-validation across HasciDB projects.

    Trains on 15 projects, tests on the held-out project. Repeats
    for all 16 projects.

    Args:
        project_datasets: Dict mapping project name -> DataFrame with
                          feature columns and "label" column.
        model_type: Model backend for EnhancedDetector.
        features: Feature columns to use.

    Returns:
        Dictionary with per-project AUC, mean AUC, and confidence interval.
    """
    features = features or ALL_FEATURES
    results = {}

    project_names = list(project_datasets.keys())
    per_project_auc = []

    for test_project in project_names:
        # Train on all other projects
        train_dfs = [
            project_datasets[p] for p in project_names if p != test_project
        ]
        train_data = pd.concat(train_dfs, ignore_index=True)
        test_data = project_datasets[test_project]

        if len(test_data) < 10 or test_data["label"].nunique() < 2:
            continue

        detector = EnhancedDetector(model_type=model_type, features=features)
        detector.fit(train_data)
        result = detector.evaluate(test_data)

        results[test_project] = {
            "auc": result.auc,
            "ap": result.average_precision,
            "p_at_90r": result.precision_at_90_recall,
            "n_train": len(train_data),
            "n_test": len(test_data),
            "feature_importances": result.feature_importances,
        }
        per_project_auc.append(result.auc)

    # Confidence interval (95%)
    if len(per_project_auc) >= 2:
        mean_auc = float(np.mean(per_project_auc))
        std_auc = float(np.std(per_project_auc, ddof=1))
        ci_95 = 1.96 * std_auc / np.sqrt(len(per_project_auc))
    else:
        mean_auc = per_project_auc[0] if per_project_auc else 0.0
        std_auc = 0.0
        ci_95 = 0.0

    return {
        "per_project": results,
        "mean_auc": mean_auc,
        "std_auc": std_auc,
        "ci_95": float(ci_95),
        "ci_lower": mean_auc - ci_95,
        "ci_upper": mean_auc + ci_95,
        "n_projects_evaluated": len(per_project_auc),
    }


# ============================================================
# TEMPORAL SPLIT EVALUATION
# ============================================================

def temporal_split_evaluation(
    project_datasets: dict[str, pd.DataFrame],
    train_cutoff_year: int = 2023,
    model_type: str = "gbm",
    features: Optional[list[str]] = None,
) -> dict:
    """Train on pre-cutoff projects, test on post-cutoff projects.

    Uses chronological project ordering from PROJECTS_CHRONOLOGICAL.

    Args:
        project_datasets: Dict mapping project name -> DataFrame.
        train_cutoff_year: Train on projects from years <= cutoff,
                           test on projects from years > cutoff.
        model_type: Model backend.
        features: Feature columns to use.

    Returns:
        Dictionary with train/test project lists, AUC, and feature importances.
    """
    features = features or ALL_FEATURES

    train_projects = []
    test_projects = []
    for year, projects in PROJECTS_CHRONOLOGICAL.items():
        for p in projects:
            if year <= train_cutoff_year:
                train_projects.append(p)
            else:
                test_projects.append(p)

    # Filter to available datasets
    train_projects = [p for p in train_projects if p in project_datasets]
    test_projects = [p for p in test_projects if p in project_datasets]

    if not train_projects or not test_projects:
        return {"error": "Insufficient projects for temporal split"}

    train_data = pd.concat(
        [project_datasets[p] for p in train_projects], ignore_index=True
    )
    test_data = pd.concat(
        [project_datasets[p] for p in test_projects], ignore_index=True
    )

    detector = EnhancedDetector(model_type=model_type, features=features)
    detector.fit(train_data)
    result = detector.evaluate(test_data)

    # Per-test-project breakdown
    per_project = {}
    for p in test_projects:
        p_data = project_datasets[p]
        if p_data["label"].nunique() >= 2:
            p_result = detector.evaluate(p_data)
            per_project[p] = {
                "auc": p_result.auc,
                "ap": p_result.average_precision,
                "n": len(p_data),
            }

    return {
        "train_projects": train_projects,
        "test_projects": test_projects,
        "train_cutoff_year": train_cutoff_year,
        "overall_auc": result.auc,
        "overall_ap": result.average_precision,
        "per_project": per_project,
        "feature_importances": result.feature_importances,
        "n_train": len(train_data),
        "n_test": len(test_data),
    }


# ============================================================
# BASELINE COMPARISON
# ============================================================

def compare_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, DetectorResult]:
    """Compare multiple detection approaches on the same train/test split.

    Baselines:
    1. HasciDB rules (no training)
    2. HasciDB ML (GBM on 5 indicators)
    3. pre-airdrop LightGBM (if available)
    4. AI-features only (GBM on 8 AI features)
    5. Enhanced GBM (13 features)
    6. Enhanced RF (13 features)
    7. Enhanced LightGBM (13 features, if available)

    Args:
        train_df: Training DataFrame with features and "label".
        test_df: Test DataFrame with features and "label".

    Returns:
        Dictionary mapping method name -> DetectorResult.
    """
    results = {}
    y_test = test_df["label"].values

    # 1. HasciDB rules
    y_rule_score = hascidb_continuous_score(test_df)
    rule_auc = roc_auc_score(y_test, y_rule_score)
    rule_ap = average_precision_score(y_test, y_rule_score)
    results["HasciDB Rules"] = DetectorResult(
        model_name="HasciDB Rules",
        features_used=HASCIDB_FEATURES,
        auc=rule_auc,
        average_precision=rule_ap,
    )

    # 2. HasciDB ML (5 indicators)
    det_hascidb = EnhancedDetector(model_type="gbm", features=HASCIDB_FEATURES)
    det_hascidb.fit(train_df)
    results["HasciDB ML (5-feat)"] = det_hascidb.evaluate(test_df)
    results["HasciDB ML (5-feat)"].model_name = "HasciDB ML (5-feat)"

    # 3. AI-features only
    det_ai = EnhancedDetector(model_type="gbm", features=AI_FEATURES)
    det_ai.fit(train_df)
    results["AI-Only GBM (8-feat)"] = det_ai.evaluate(test_df)
    results["AI-Only GBM (8-feat)"].model_name = "AI-Only GBM (8-feat)"

    # 4. Enhanced GBM (13 features)
    det_enhanced_gbm = EnhancedDetector(model_type="gbm", features=ALL_FEATURES)
    det_enhanced_gbm.fit(train_df)
    results["Enhanced GBM (13-feat)"] = det_enhanced_gbm.evaluate(test_df)
    results["Enhanced GBM (13-feat)"].model_name = "Enhanced GBM (13-feat)"

    # 5. Enhanced RF (13 features)
    det_enhanced_rf = EnhancedDetector(model_type="rf", features=ALL_FEATURES)
    det_enhanced_rf.fit(train_df)
    results["Enhanced RF (13-feat)"] = det_enhanced_rf.evaluate(test_df)
    results["Enhanced RF (13-feat)"].model_name = "Enhanced RF (13-feat)"

    # 6. Enhanced LightGBM (if available)
    if HAS_LIGHTGBM:
        det_lgbm = EnhancedDetector(model_type="lgbm", features=ALL_FEATURES)
        det_lgbm.fit(train_df)
        results["Enhanced LightGBM (13-feat)"] = det_lgbm.evaluate(test_df)
        results["Enhanced LightGBM (13-feat)"].model_name = "Enhanced LightGBM (13-feat)"

    return results


# ============================================================
# INDIVIDUAL FEATURE DISCRIMINATIVE POWER
# ============================================================

def individual_feature_auc(
    df: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    features: Optional[list[str]] = None,
) -> dict[str, float]:
    """Compute univariate AUC for each feature.

    For each feature, compute AUC using the raw feature value as the
    prediction score. Takes max(AUC, 1-AUC) to handle negatively
    correlated features.

    Args:
        df: DataFrame with feature columns and optionally "label".
        y: True labels. If None, reads from df["label"].
        features: Features to evaluate. Defaults to ALL_FEATURES.

    Returns:
        Dictionary mapping feature name -> AUC (in [0.5, 1.0]).
    """
    if y is None:
        y = df["label"].values
    features = features or ALL_FEATURES

    aucs = {}
    for feat in features:
        try:
            auc = roc_auc_score(y, df[feat].values)
            auc = max(auc, 1 - auc)  # Correct for negative correlation
        except (ValueError, KeyError):
            auc = 0.5
        aucs[feat] = float(auc)

    return aucs


# ============================================================
# ABLATION STUDY
# ============================================================

def feature_ablation(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_type: str = "gbm",
) -> dict:
    """Ablation study: measure AUC when removing one feature at a time.

    Args:
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        model_type: Model backend.

    Returns:
        Dictionary with full-model AUC, per-feature ablation AUC,
        and delta (impact of removing each feature).
    """
    y_test = test_df["label"].values

    # Full model
    full_det = EnhancedDetector(model_type=model_type, features=ALL_FEATURES)
    full_det.fit(train_df)
    full_auc = roc_auc_score(y_test, full_det.predict_proba(test_df))

    # Remove one feature at a time
    ablation_results = {}
    for feat in ALL_FEATURES:
        reduced_features = [f for f in ALL_FEATURES if f != feat]
        det = EnhancedDetector(model_type=model_type, features=reduced_features)
        det.fit(train_df)
        reduced_auc = roc_auc_score(y_test, det.predict_proba(test_df))
        ablation_results[feat] = {
            "auc_without": float(reduced_auc),
            "delta": float(full_auc - reduced_auc),
            "feature_type": "AI-specific" if feat in AI_FEATURES else "HasciDB",
        }

    # Sort by impact (largest delta first)
    sorted_ablation = dict(
        sorted(ablation_results.items(), key=lambda x: -x[1]["delta"])
    )

    return {
        "full_model_auc": float(full_auc),
        "ablation": sorted_ablation,
    }


# ============================================================
# MAIN (demo)
# ============================================================

if __name__ == "__main__":
    # Import generator for demo data
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from adversarial.ai_sybil_generator import (
        generate_ai_sybil_dataframe,
        EvasionLevel,
    )

    # Use pilot v2's data generation functions as fallback
    from experiments.pilot_sybil_evasion import (
        generate_real_calibrated_legitimate,
        generate_real_calibrated_traditional_sybil,
    )

    print("=" * 70)
    print("Enhanced Detector Demo")
    print("=" * 70)

    # Generate data
    legit = generate_real_calibrated_legitimate(1000)
    trad_sybil = generate_real_calibrated_traditional_sybil(300)
    ai_sybil = generate_ai_sybil_dataframe(300, level=EvasionLevel.ADVANCED)

    train_df = pd.concat([legit, trad_sybil, ai_sybil], ignore_index=True)

    # Test on advanced AI sybils
    test_legit = generate_real_calibrated_legitimate(300, seed=99)
    test_ai = generate_ai_sybil_dataframe(300, level=EvasionLevel.ADVANCED, seed=99)
    test_df = pd.concat([test_legit, test_ai], ignore_index=True)

    # Compare baselines
    print("\n--- Baseline Comparison ---")
    baseline_results = compare_baselines(train_df, test_df)
    for name, result in sorted(baseline_results.items(), key=lambda x: x[1].auc):
        print(f"  {name:<35s} AUC={result.auc:.3f}  AP={result.average_precision:.3f}")

    # Feature ablation
    print("\n--- Feature Ablation ---")
    ablation = feature_ablation(train_df, test_df)
    print(f"  Full model AUC: {ablation['full_model_auc']:.3f}")
    for feat, info in ablation["ablation"].items():
        print(f"  Remove {feat:<35s}: AUC={info['auc_without']:.3f} "
              f"(delta={info['delta']:+.4f}) [{info['feature_type']}]")

    # Individual feature AUC
    print("\n--- Individual Feature AUC ---")
    feat_aucs = individual_feature_auc(test_df)
    for feat, auc in sorted(feat_aucs.items(), key=lambda x: -x[1]):
        ftype = "AI" if feat in AI_FEATURES else "HasciDB"
        print(f"  {feat:<35s}: {auc:.3f}  [{ftype}]")
