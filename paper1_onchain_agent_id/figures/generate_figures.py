"""
Paper 1: Publication-Quality Figure Generation
================================================
Generates all figures for the Paper 1 submission in ENGLISH.

Figures:
  a) Feature importance bar chart (horizontal, colored by feature group)
  b) ROC curve comparing 3 classifiers
  c) Feature distribution violin plots (agent vs human, top-6 features)
  d) Confusion matrix heatmap
  e) Ablation study bar chart (AUC by feature group)

All figures are saved as PDF for LaTeX inclusion.
Style: seaborn-v0_8-whitegrid, 300 DPI, Times-like serif font.
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ensure project is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


# ============================================================
# STYLE CONFIGURATION
# ============================================================

# Feature group -> color mapping (colorblind-friendly palette)
GROUP_COLORS = {
    "temporal": "#4C72B0",         # steel blue
    "gas": "#DD8452",              # burnt orange
    "interaction": "#55A868",      # sage green
    "approval_security": "#C44E52",  # brick red
}

FEATURE_TO_GROUP = {}
_GROUPS = {
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
for group, features in _GROUPS.items():
    for feat in features:
        FEATURE_TO_GROUP[feat] = group

# Human-readable feature labels for figures
FEATURE_DISPLAY_NAMES = {
    "tx_interval_mean": "Tx Interval Mean",
    "tx_interval_std": "Tx Interval Std",
    "tx_interval_skewness": "Tx Interval Skewness",
    "active_hour_entropy": "Active Hour Entropy",
    "night_activity_ratio": "Night Activity Ratio",
    "weekend_ratio": "Weekend Ratio",
    "burst_frequency": "Burst Frequency",
    "gas_price_round_number_ratio": "Gas Round Number Ratio",
    "gas_price_trailing_zeros_mean": "Gas Trailing Zeros Mean",
    "gas_limit_precision": "Gas Limit Precision",
    "gas_price_cv": "Gas Price CV",
    "eip1559_priority_fee_precision": "EIP-1559 Priority Fee Precision",
    "gas_price_nonce_correlation": "Gas-Nonce Correlation",
    "unique_contracts_ratio": "Unique Contracts Ratio",
    "top_contract_concentration": "Contract Concentration (HHI)",
    "method_id_diversity": "Method ID Diversity",
    "contract_to_eoa_ratio": "Contract-to-EOA Ratio",
    "sequential_pattern_score": "Sequential Pattern Score",
    "unlimited_approve_ratio": "Unlimited Approve Ratio",
    "approve_revoke_ratio": "Approve/Revoke Ratio",
    "unverified_contract_approve_ratio": "Unverified Approve Ratio",
    "multi_protocol_interaction_count": "Multi-Protocol Count",
    "flash_loan_usage": "Flash Loan Usage",
}

# Display names for feature groups
GROUP_DISPLAY_NAMES = {
    "temporal": "Temporal",
    "gas": "Gas Behavior",
    "interaction": "Interaction Pattern",
    "approval_security": "Approval & Security",
}


def _setup_matplotlib():
    """Configure matplotlib for publication-quality output."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use a clean style
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        try:
            plt.style.use("seaborn-whitegrid")
        except OSError:
            pass

    # Font settings for academic papers
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    return plt, sns


# ============================================================
# FIGURE a: FEATURE IMPORTANCE BAR CHART
# ============================================================

def plot_feature_importance(
    results: dict,
    output_dir: str,
) -> None:
    """Horizontal bar chart of GBM feature importance, colored by group.

    Args:
        results: Pipeline results dict with 'gbm_feature_importance' key.
        output_dir: Directory to save the figure.
    """
    plt, sns = _setup_matplotlib()

    importance = results.get("gbm_feature_importance", {})
    if not importance:
        logger.warning("No feature importance data; skipping plot.")
        return

    # Sort ascending for horizontal bars (top feature at the top)
    sorted_items = sorted(importance.items(), key=lambda x: x[1])
    features = [item[0] for item in sorted_items]
    values = [item[1] for item in sorted_items]
    colors = [GROUP_COLORS.get(FEATURE_TO_GROUP.get(f, ""), "#888888")
              for f in features]
    display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in features]

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(range(len(features)), values, color=colors, edgecolor="white",
                   linewidth=0.5)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(display_names, fontsize=9)
    ax.set_xlabel("Feature Importance (Gain)")
    ax.set_title("GBM Feature Importance by Group")

    # Legend for feature groups
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color, label=GROUP_DISPLAY_NAMES.get(group, group))
        for group, color in GROUP_COLORS.items()
    ]
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.9)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "feature_importance.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ============================================================
# FIGURE b: ROC CURVES
# ============================================================

def plot_roc_curves(
    results: dict,
    output_dir: str,
) -> None:
    """ROC curves for all three classifiers.

    Args:
        results: Pipeline results dict with 'roc_curve_data' key.
        output_dir: Directory to save the figure.
    """
    plt, sns = _setup_matplotlib()

    roc_data = results.get("roc_curve_data", {})
    if not roc_data:
        logger.warning("No ROC curve data; skipping plot.")
        return

    model_styles = {
        "GradientBoosting": {"color": "#4C72B0", "ls": "-", "lw": 2.5},
        "RandomForest": {"color": "#55A868", "ls": "--", "lw": 2.0},
        "LogisticRegression": {"color": "#DD8452", "ls": "-.", "lw": 2.0},
    }
    model_display = {
        "GradientBoosting": "Gradient Boosting",
        "RandomForest": "Random Forest",
        "LogisticRegression": "Logistic Regression",
    }

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for model_name, data in roc_data.items():
        fpr = np.array(data["fpr"])
        tpr = np.array(data["tpr"])
        auc_val = data["auc"]
        style = model_styles.get(model_name, {"color": "gray", "ls": "-", "lw": 1.5})
        display = model_display.get(model_name, model_name)
        ax.plot(fpr, tpr,
                label=f"{display} (AUC = {auc_val:.3f})",
                **style)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1, label="Random (AUC = 0.500)")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves: On-Chain AI Agent Identification (LOO-CV)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_aspect("equal")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "roc_curves.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ============================================================
# FIGURE c: VIOLIN PLOTS (top-6 features)
# ============================================================

def plot_feature_violins(
    X: pd.DataFrame,
    y: np.ndarray,
    results: dict,
    output_dir: str,
    n_top: int = 6,
) -> None:
    """Violin plots comparing agent vs human distributions for top features.

    Args:
        X: Feature matrix DataFrame.
        y: Binary labels (1=agent, 0=human).
        results: Pipeline results dict with 'per_feature_auc'.
        output_dir: Directory to save the figure.
        n_top: Number of top features to plot.
    """
    plt, sns = _setup_matplotlib()

    feat_aucs = results.get("per_feature_auc", {})
    if not feat_aucs:
        # Fall back to column order
        top_features = X.columns[:n_top].tolist()
    else:
        top_features = list(feat_aucs.keys())[:n_top]

    # Filter to features that exist in X
    top_features = [f for f in top_features if f in X.columns][:n_top]
    if not top_features:
        logger.warning("No features available for violin plot.")
        return

    n_rows = 2
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 7))
    axes = axes.flatten()

    for i, feat in enumerate(top_features):
        ax = axes[i]
        data = pd.DataFrame({
            "value": X[feat].values,
            "label": ["Agent" if yi == 1 else "Human" for yi in y],
        })

        sns.violinplot(
            data=data, x="label", y="value", hue="label", ax=ax,
            palette={"Agent": "#4C72B0", "Human": "#DD8452"},
            inner="box", linewidth=1.0,
            cut=0, legend=False,
        )

        display_name = FEATURE_DISPLAY_NAMES.get(feat, feat)
        auc_val = feat_aucs.get(feat, 0.0)
        ax.set_title(f"{display_name}\n(AUC = {auc_val:.3f})", fontsize=10)
        ax.set_xlabel("")
        ax.set_ylabel("")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused subplots
    for j in range(len(top_features), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Feature Distributions: Agent vs. Human (Top-6 by AUC)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    path = os.path.join(output_dir, "feature_violins.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ============================================================
# FIGURE d: CONFUSION MATRIX HEATMAP
# ============================================================

def plot_confusion_matrix(
    results: dict,
    output_dir: str,
) -> None:
    """Confusion matrix heatmap from GBM LOO-CV.

    Args:
        results: Pipeline results dict with 'confusion_matrix_gbm_loo'.
        output_dir: Directory to save the figure.
    """
    plt, sns = _setup_matplotlib()

    cm = results.get("confusion_matrix_gbm_loo")
    if cm is None:
        # Try to reconstruct from predictions
        models = results.get("models", {})
        gbm_data = models.get("GradientBoosting", {})
        preds = gbm_data.get("loo_predictions")
        dataset = results.get("dataset", {})
        n_agents = dataset.get("n_agents", 0)
        n_humans = dataset.get("n_humans", 0)

        if preds is not None:
            y = np.array([0] * n_humans + [1] * n_agents)
            from sklearn.metrics import confusion_matrix as cm_func
            cm = cm_func(y, np.array(preds)).tolist()
        else:
            logger.warning("No confusion matrix data; skipping plot.")
            return

    cm_array = np.array(cm)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    sns.heatmap(
        cm_array,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Human", "Agent"],
        yticklabels=["Human", "Agent"],
        ax=ax,
        linewidths=1,
        linecolor="white",
        cbar_kws={"shrink": 0.8},
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix (GBM, LOO-CV)")

    fig.tight_layout()
    path = os.path.join(output_dir, "confusion_matrix.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ============================================================
# FIGURE e: ABLATION STUDY BAR CHART
# ============================================================

def plot_ablation(
    results: dict,
    output_dir: str,
) -> None:
    """Bar chart showing AUC by feature group (ablation study).

    Shows:
      - Each group alone
      - All features combined

    Args:
        results: Pipeline results dict with 'ablation_study' key.
        output_dir: Directory to save the figure.
    """
    plt, sns = _setup_matplotlib()

    ablation = results.get("ablation_study", {})
    if not ablation:
        logger.warning("No ablation data; skipping plot.")
        return

    # Collect group-only results
    groups_to_plot = []
    for group_name in ["temporal", "gas", "interaction", "approval_security"]:
        if group_name in ablation:
            groups_to_plot.append((group_name, ablation[group_name]))

    if "all_features" in ablation:
        groups_to_plot.append(("all_features", ablation["all_features"]))

    if not groups_to_plot:
        logger.warning("No ablation groups to plot.")
        return

    names = []
    aucs = []
    colors = []
    for group_name, data in groups_to_plot:
        display = GROUP_DISPLAY_NAMES.get(group_name, group_name)
        if group_name == "all_features":
            display = "All Features"
        n_feat = data.get("n_features", "?")
        names.append(f"{display}\n(n={n_feat})")
        aucs.append(data["loo_auc"])
        if group_name == "all_features":
            colors.append("#333333")
        else:
            colors.append(GROUP_COLORS.get(group_name, "#888888"))

    fig, ax = plt.subplots(figsize=(8, 5))
    x_pos = range(len(names))
    bars = ax.bar(x_pos, aucs, color=colors, edgecolor="white", linewidth=1.0,
                  width=0.65)

    # Add value labels on bars
    for bar, auc_val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{auc_val:.3f}", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("LOO-CV AUC")
    ax.set_title("Ablation Study: AUC by Feature Group")
    ax.set_ylim([0, 1.08])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3, label="Random")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    path = os.path.join(output_dir, "ablation_study.pdf")
    fig.savefig(path)
    plt.close(fig)
    logger.info("Saved: %s", path)


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def generate_all_figures(
    X: pd.DataFrame,
    y: np.ndarray,
    results: dict,
    output_dir: str = None,
) -> None:
    """Generate all publication figures.

    Args:
        X: Feature matrix DataFrame.
        y: Binary labels.
        results: Full pipeline results dictionary.
        output_dir: Directory to save figures (created if needed).
    """
    if output_dir is None:
        output_dir = str(Path(__file__).parent)

    os.makedirs(output_dir, exist_ok=True)
    logger.info("Generating figures in %s ...", output_dir)

    plot_feature_importance(results, output_dir)
    plot_roc_curves(results, output_dir)
    plot_feature_violins(X, y, results, output_dir)
    plot_confusion_matrix(results, output_dir)
    plot_ablation(results, output_dir)

    logger.info("All figures generated successfully.")


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    """Standalone usage: generate figures from existing pipeline results."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Generate Paper 1 publication figures"
    )
    parser.add_argument(
        "--results-json",
        default=None,
        help="Path to pipeline_results.json (default: experiments/pipeline_results.json)",
    )
    parser.add_argument(
        "--features-parquet",
        default=None,
        help="Path to features.parquet (default: data/features.parquet)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures (default: figures/)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    # Load results
    results_path = args.results_json or str(
        project_root / "experiments" / "pipeline_results.json"
    )
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
    else:
        # Fall back to classifier_results.json
        fallback = project_root / "experiments" / "classifier_results.json"
        if fallback.exists():
            with open(fallback) as f:
                results = json.load(f)
            logger.info("Loaded fallback results from %s", fallback)
        else:
            logger.error("No results JSON found. Run the pipeline first.")
            sys.exit(1)

    # Load features
    feat_path = args.features_parquet or str(
        project_root / "data" / "features.parquet"
    )
    if os.path.exists(feat_path):
        df = pd.read_parquet(feat_path)
        feature_cols = [c for c in df.columns if c not in ("label", "name")]
        X = df[feature_cols]
        y = df["label"].values.astype(int) if "label" in df.columns else np.zeros(len(df))
    else:
        logger.error("Features parquet not found: %s", feat_path)
        sys.exit(1)

    out_dir = args.output_dir or str(project_root / "figures")
    generate_all_figures(X, y, results, out_dir)
