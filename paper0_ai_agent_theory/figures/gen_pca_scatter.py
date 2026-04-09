#!/usr/bin/env python
"""Generate PCA 2D scatter plot colored by taxonomy category."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Load data ──────────────────────────────────────────────────────────
DATA_PATH = "/Users/adelinewen/NSF/paper1_onchain_agent_id/data/features_with_taxonomy.parquet"
OUT_PATH = "/Users/adelinewen/NSF/paper0_ai_agent_theory/figures/fig_pca_scatter.pdf"

df = pd.read_parquet(DATA_PATH)

# Keep only the 5 agent categories (drop HUMAN)
df = df[df["taxonomy_category"] != "HUMAN"].copy()
print(f"Agents after dropping HUMAN: {len(df)}")

# 23 behavioral features (from multiclass_classifier_results.json)
FEATURES = [
    "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
    "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
    "burst_frequency", "gas_price_round_number_ratio",
    "gas_price_trailing_zeros_mean", "gas_limit_precision", "gas_price_cv",
    "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
    "unique_contracts_ratio", "top_contract_concentration",
    "method_id_diversity", "contract_to_eoa_ratio",
    "sequential_pattern_score", "unlimited_approve_ratio",
    "approve_revoke_ratio", "unverified_contract_approve_ratio",
    "multi_protocol_interaction_count", "flash_loan_usage",
]

X = df[FEATURES].values
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# ── PCA ────────────────────────────────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_std)
evr = pca.explained_variance_ratio_
print(f"Explained variance ratio: PC1={evr[0]:.4f}, PC2={evr[1]:.4f}, total={sum(evr):.4f}")

# ── Plot ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "dejavuserif",
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
})

CATEGORY_ORDER = [
    "DeFi Management Agent",
    "Deterministic Script",
    "Simple Trading Bot",
    "MEV Searcher",
    "LLM-Powered Agent",
]
COLORS = {
    "DeFi Management Agent": "#1f77b4",
    "Deterministic Script":  "#ff7f0e",
    "Simple Trading Bot":    "#2ca02c",
    "MEV Searcher":          "#d62728",
    "LLM-Powered Agent":     "#9467bd",
}

fig, ax = plt.subplots(figsize=(7, 5))

for cat in CATEGORY_ORDER:
    mask = df["taxonomy_category"].values == cat
    ax.scatter(
        X_pca[mask, 0], X_pca[mask, 1],
        c=COLORS[cat], label=cat,
        alpha=0.5, s=15, edgecolors="none",
    )

ax.set_xlabel(f"PC 1 ({evr[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC 2 ({evr[1]*100:.1f}% variance)")
ax.set_title("PCA Projection of 23 Behavioral Features (5 Agent Categories)")

# Legend outside plot
ax.legend(
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),
    borderaxespad=0,
    frameon=True,
    framealpha=0.9,
)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
plt.close(fig)
