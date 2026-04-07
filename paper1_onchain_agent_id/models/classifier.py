"""
AI Agent Classifier
===================
Multi-model ensemble for identifying AI agents from on-chain
behavioral features.

Models:
  1. LightGBM (primary) -- handles mixed feature types, fast training,
     built-in feature importance.
  2. Random Forest -- robust ensemble baseline.
  3. Logistic Regression -- interpretable linear baseline.

Evaluation protocol:
  - 5-fold stratified cross-validation (preserves class balance).
  - Metrics: AUC-ROC, Precision@K, Recall@K, F1-score.
  - SHAP feature importance analysis (for LightGBM).
  - Calibration curve to assess predicted probability reliability.

Usage::

    from paper1_onchain_agent_id.models.classifier import AgentClassifier

    clf = AgentClassifier()
    results = clf.train_and_evaluate(X, y)
    clf.plot_results(results, output_dir="figures/")
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class ClassifierConfig:
    """Hyper-parameters and evaluation settings.

    Attributes:
        n_folds: Number of cross-validation folds.
        random_state: Seed for reproducibility.
        lgbm_params: LightGBM parameters dictionary.
        rf_params: Random Forest parameters dictionary.
        lr_params: Logistic Regression parameters dictionary.
        precision_at_k: List of K values for Precision@K evaluation.
    """

    n_folds: int = 5
    random_state: int = 42
    lgbm_params: dict = field(default_factory=lambda: {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbose": -1,
        "n_jobs": -1,
    })
    rf_params: dict = field(default_factory=lambda: {
        "n_estimators": 300,
        "max_depth": 10,
        "min_samples_leaf": 5,
        "random_state": 42,
        "n_jobs": -1,
    })
    lr_params: dict = field(default_factory=lambda: {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    })
    precision_at_k: list = field(default_factory=lambda: [50, 100, 200])


# ============================================================
# EVALUATION RESULTS
# ============================================================

@dataclass
class FoldResult:
    """Metrics from a single cross-validation fold."""

    fold: int
    model_name: str
    auc_roc: float
    precision: float
    recall: float
    f1: float
    precision_at_k: dict  # {k: precision_value}
    recall_at_k: dict     # {k: recall_value}
    y_true: np.ndarray = field(repr=False)
    y_prob: np.ndarray = field(repr=False)


@dataclass
class EvalResults:
    """Aggregated evaluation results across all folds and models."""

    fold_results: list[FoldResult] = field(default_factory=list)
    feature_importances: Optional[pd.DataFrame] = None
    shap_values: Optional[np.ndarray] = None

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame summarising mean +/- std metrics per model."""
        rows = []
        for model_name in {fr.model_name for fr in self.fold_results}:
            model_folds = [
                fr for fr in self.fold_results if fr.model_name == model_name
            ]
            rows.append({
                "model": model_name,
                "auc_roc_mean": np.mean([f.auc_roc for f in model_folds]),
                "auc_roc_std": np.std([f.auc_roc for f in model_folds]),
                "f1_mean": np.mean([f.f1 for f in model_folds]),
                "f1_std": np.std([f.f1 for f in model_folds]),
                "precision_mean": np.mean([f.precision for f in model_folds]),
                "recall_mean": np.mean([f.recall for f in model_folds]),
            })
        return pd.DataFrame(rows).sort_values("auc_roc_mean", ascending=False)


# ============================================================
# CLASSIFIER
# ============================================================

class AgentClassifier:
    """Multi-model classifier for AI agent identification.

    Args:
        config: Classifier configuration; uses defaults if omitted.
    """

    def __init__(self, config: Optional[ClassifierConfig] = None):
        self.config = config or ClassifierConfig()
        self._models: dict = {}
        self._scaler = StandardScaler()

    # ----------------------------------------------------------
    # Training & Evaluation
    # ----------------------------------------------------------

    def train_and_evaluate(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> EvalResults:
        """Run stratified k-fold CV for all three models.

        Args:
            X: Feature matrix (n_samples x 23 features).
            y: Binary labels (1 = AGENT, 0 = HUMAN).

        Returns:
            EvalResults containing per-fold and aggregated metrics.
        """
        results = EvalResults()
        skf = StratifiedKFold(
            n_splits=self.config.n_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )

        # Accumulate feature importances from LightGBM folds
        lgbm_importances: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            logger.info("Fold %d/%d ...", fold_idx + 1, self.config.n_folds)

            # ---- LightGBM ----
            lgbm_result, lgbm_imp = self._train_lgbm(
                X_train, y_train, X_test, y_test, fold_idx
            )
            results.fold_results.append(lgbm_result)
            if lgbm_imp is not None:
                lgbm_importances.append(lgbm_imp)

            # ---- Random Forest ----
            rf_result = self._train_rf(
                X_train, y_train, X_test, y_test, fold_idx
            )
            results.fold_results.append(rf_result)

            # ---- Logistic Regression ----
            lr_result = self._train_lr(
                X_train, y_train, X_test, y_test, fold_idx
            )
            results.fold_results.append(lr_result)

        # Aggregate feature importances
        if lgbm_importances:
            mean_imp = np.mean(lgbm_importances, axis=0)
            results.feature_importances = pd.DataFrame({
                "feature": X.columns.tolist(),
                "importance": mean_imp,
            }).sort_values("importance", ascending=False)

        # SHAP analysis (on full training set, final fold model)
        results.shap_values = self._compute_shap(X, y)

        logger.info("\n--- Evaluation Summary ---")
        logger.info("\n%s", results.summary().to_string(index=False))

        return results

    def predict(self, X: pd.DataFrame, model_name: str = "lgbm") -> np.ndarray:
        """Predict agent probabilities using a trained model.

        Args:
            X: Feature matrix.
            model_name: One of 'lgbm', 'rf', 'lr'.

        Returns:
            Array of predicted probabilities for class AGENT.
        """
        model = self._models.get(model_name)
        if model is None:
            raise ValueError(
                f"Model '{model_name}' not trained. "
                f"Available: {list(self._models.keys())}"
            )
        if model_name == "lr":
            X_scaled = self._scaler.transform(X)
            return model.predict_proba(X_scaled)[:, 1]
        return model.predict_proba(X)[:, 1]

    # ----------------------------------------------------------
    # Model-specific training
    # ----------------------------------------------------------

    def _train_lgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        fold_idx: int,
    ) -> tuple:
        """Train LightGBM and return (FoldResult, feature_importances)."""
        try:
            import lightgbm as lgb
        except ImportError:
            logger.warning(
                "LightGBM not installed; skipping. "
                "Install with: pip install lightgbm"
            )
            return self._dummy_fold(fold_idx, "lgbm", y_test), None

        model = lgb.LGBMClassifier(**self.config.lgbm_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )
        self._models["lgbm"] = model

        y_prob = model.predict_proba(X_test)[:, 1]
        fold_result = self._compute_fold_metrics(
            fold_idx, "lgbm", y_test, y_prob
        )
        return fold_result, model.feature_importances_

    def _train_rf(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        fold_idx: int,
    ) -> FoldResult:
        """Train Random Forest and return FoldResult."""
        model = RandomForestClassifier(**self.config.rf_params)
        model.fit(X_train, y_train)
        self._models["rf"] = model

        y_prob = model.predict_proba(X_test)[:, 1]
        return self._compute_fold_metrics(fold_idx, "rf", y_test, y_prob)

    def _train_lr(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        fold_idx: int,
    ) -> FoldResult:
        """Train Logistic Regression (with scaling) and return FoldResult."""
        self._scaler.fit(X_train)
        X_train_scaled = self._scaler.transform(X_train)
        X_test_scaled = self._scaler.transform(X_test)

        model = LogisticRegression(**self.config.lr_params)
        model.fit(X_train_scaled, y_train)
        self._models["lr"] = model

        y_prob = model.predict_proba(X_test_scaled)[:, 1]
        return self._compute_fold_metrics(fold_idx, "lr", y_test, y_prob)

    # ----------------------------------------------------------
    # Metrics
    # ----------------------------------------------------------

    def _compute_fold_metrics(
        self,
        fold_idx: int,
        model_name: str,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> FoldResult:
        """Compute all metrics for one fold."""
        y_pred = (y_prob >= 0.5).astype(int)

        auc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        # Precision@K and Recall@K
        prec_at_k = {}
        rec_at_k = {}
        sorted_indices = np.argsort(-y_prob)
        n_positive = int(y_true.sum())
        for k in self.config.precision_at_k:
            if k > len(y_true):
                k = len(y_true)
            top_k_true = y_true[sorted_indices[:k]]
            prec_at_k[k] = float(top_k_true.mean())
            rec_at_k[k] = float(top_k_true.sum() / max(n_positive, 1))

        return FoldResult(
            fold=fold_idx,
            model_name=model_name,
            auc_roc=auc,
            precision=prec,
            recall=rec,
            f1=f1,
            precision_at_k=prec_at_k,
            recall_at_k=rec_at_k,
            y_true=y_true,
            y_prob=y_prob,
        )

    @staticmethod
    def _dummy_fold(fold_idx: int, name: str, y_test: np.ndarray) -> FoldResult:
        """Return a placeholder FoldResult when a model is unavailable."""
        return FoldResult(
            fold=fold_idx,
            model_name=name,
            auc_roc=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            precision_at_k={},
            recall_at_k={},
            y_true=y_test,
            y_prob=np.zeros_like(y_test, dtype=float),
        )

    # ----------------------------------------------------------
    # SHAP Analysis
    # ----------------------------------------------------------

    def _compute_shap(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Compute SHAP values for the LightGBM model.

        Returns None if SHAP or LightGBM is not available.
        """
        try:
            import shap
        except ImportError:
            logger.warning("SHAP not installed; skipping importance analysis.")
            return None

        model = self._models.get("lgbm")
        if model is None:
            return None

        # Retrain on full data for SHAP
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classification, shap_values may be a list of two arrays
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # class 1 = AGENT

        logger.info("SHAP values computed: shape %s", shap_values.shape)
        return shap_values

    # ----------------------------------------------------------
    # Plotting
    # ----------------------------------------------------------

    def plot_results(
        self,
        results: EvalResults,
        output_dir: str = "figures",
    ) -> None:
        """Generate evaluation plots and save to disk.

        Creates:
          1. ROC curves for all models (one plot).
          2. Feature importance bar chart (from LightGBM).
          3. Calibration curve.
          4. SHAP summary plot (if SHAP values available).

        Args:
            results: EvalResults from :meth:`train_and_evaluate`.
            output_dir: Directory to save figures.
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed; skipping plots.")
            return

        # 1. ROC Curves
        self._plot_roc_curves(results, output_dir, plt)

        # 2. Feature Importance
        if results.feature_importances is not None:
            self._plot_feature_importance(
                results.feature_importances, output_dir, plt
            )

        # 3. Calibration Curve
        self._plot_calibration(results, output_dir, plt)

        # 4. SHAP
        if results.shap_values is not None:
            self._plot_shap(results, output_dir)

        logger.info("Plots saved to %s/", output_dir)

    def _plot_roc_curves(self, results, output_dir, plt):
        """Plot ROC curves averaged across folds for each model."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_name in ["lgbm", "rf", "lr"]:
            model_folds = [
                fr for fr in results.fold_results
                if fr.model_name == model_name
            ]
            if not model_folds:
                continue

            mean_auc = np.mean([f.auc_roc for f in model_folds])
            # Plot the last fold as representative
            last = model_folds[-1]
            fpr, tpr, _ = roc_curve(last.y_true, last.y_prob)
            label = f"{model_name} (AUC={mean_auc:.3f})"
            ax.plot(fpr, tpr, label=label, linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curves: AI Agent Identification")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "roc_curves.pdf"), dpi=300)
        plt.close(fig)

    @staticmethod
    def _plot_feature_importance(fi_df, output_dir, plt):
        """Plot LightGBM feature importance as horizontal bar chart."""
        fig, ax = plt.subplots(figsize=(8, 8))
        fi_sorted = fi_df.sort_values("importance", ascending=True)
        ax.barh(fi_sorted["feature"], fi_sorted["importance"])
        ax.set_xlabel("Feature Importance (Gain)")
        ax.set_title("LightGBM Feature Importance")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "feature_importance.pdf"), dpi=300
        )
        plt.close(fig)

    def _plot_calibration(self, results, output_dir, plt):
        """Plot calibration curves for all models."""
        fig, ax = plt.subplots(figsize=(8, 6))
        for model_name in ["lgbm", "rf", "lr"]:
            model_folds = [
                fr for fr in results.fold_results
                if fr.model_name == model_name
            ]
            if not model_folds:
                continue
            last = model_folds[-1]
            if len(np.unique(last.y_true)) < 2:
                continue
            prob_true, prob_pred = calibration_curve(
                last.y_true, last.y_prob, n_bins=10, strategy="uniform"
            )
            ax.plot(prob_pred, prob_true, marker="o", label=model_name)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfectly calibrated")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curves")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "calibration_curves.pdf"), dpi=300
        )
        plt.close(fig)

    @staticmethod
    def _plot_shap(results, output_dir):
        """Generate SHAP summary plot."""
        try:
            import shap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(10, 8))
            shap.summary_plot(
                results.shap_values,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "shap_summary.pdf"), dpi=300
            )
            plt.close()
        except Exception as exc:
            logger.warning("SHAP plot failed: %s", exc)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """Demo: train and evaluate on synthetic data."""
    logging.basicConfig(level=logging.INFO)

    np.random.seed(42)
    n_agents, n_humans = 200, 200
    n_features = 23

    # Synthetic features: agents have lower interval variance, higher
    # entropy, lower gas round-number ratio.
    X_agents = np.random.randn(n_agents, n_features)
    X_agents[:, 0] += 2.0   # tx_interval_mean shift
    X_agents[:, 3] += 1.5   # active_hour_entropy shift
    X_agents[:, 7] -= 1.0   # gas_price_round lower

    X_humans = np.random.randn(n_humans, n_features)

    from paper1_onchain_agent_id.features.feature_pipeline import FEATURE_NAMES

    X = pd.DataFrame(
        np.vstack([X_agents, X_humans]),
        columns=FEATURE_NAMES,
    )
    y = np.array([1] * n_agents + [0] * n_humans)

    clf = AgentClassifier()
    results = clf.train_and_evaluate(X, y)

    print("\n=== Summary ===")
    print(results.summary().to_string(index=False))

    if results.feature_importances is not None:
        print("\n=== Top 10 Features (LightGBM) ===")
        print(results.feature_importances.head(10).to_string(index=False))

    clf.plot_results(results, output_dir="paper1_onchain_agent_id/figures")
    print("\nPlots saved to paper1_onchain_agent_id/figures/")


if __name__ == "__main__":
    main()
