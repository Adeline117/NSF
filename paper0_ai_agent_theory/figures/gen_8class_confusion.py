#!/usr/bin/env python
"""Generate 8-class confusion matrix figure for Paper 0."""

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# ── paths ──
DATA = Path(__file__).resolve().parent.parent.parent / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
OUT  = Path(__file__).resolve().parent / "fig_8class_confusion.pdf"

# ── load ──
df = pd.read_parquet(DATA)
agents = df[df["label"] == 1].copy()

# Map taxonomy_category -> short class names
CLASS_MAP = {
    "Simple Trading Bot":    "SimpleTradBot",
    "MEV Searcher":          "MEVSearch",
    "DeFi Management Agent": "DeFiMgmt",
    "LLM-Powered Agent":     "LLMPower",
    "AutonomousDAOAgent":    "DAOAgent",
    "CrossChainBridgeAgent": "Bridge",
    "Deterministic Script":  "DetScript",
    "RLTradingAgent":        "RLTrade",
}

CLASS_ORDER = ["SimpleTradBot", "MEVSearch", "DeFiMgmt", "LLMPower",
               "DAOAgent", "Bridge", "DetScript", "RLTrade"]

agents["class_name"] = agents["taxonomy_category"].map(CLASS_MAP)
agents = agents.dropna(subset=["class_name"])

# ── features ──
feat_cols = [c for c in agents.columns if c not in
             ["label", "name", "source", "c1c4_confidence", "n_transactions",
              "taxonomy_category", "taxonomy_index", "confidence", "rule",
              "source_tier", "class_name"]]

X = agents[feat_cols].values.astype(np.float64)
# Fill NaNs with column medians
col_medians = np.nanmedian(X, axis=0)
for j in range(X.shape[1]):
    mask = np.isnan(X[:, j])
    X[mask, j] = col_medians[j]

le = LabelEncoder()
le.classes_ = np.array(CLASS_ORDER)
y = le.transform(agents["class_name"].values)

# ── 5-fold CV, pooled predictions ──
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []

for train_idx, test_idx in skf.split(X, y):
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(X[train_idx], y[train_idx])
    preds = clf.predict(X[test_idx])
    y_true_all.extend(y[test_idx])
    y_pred_all.extend(preds)

y_true_all = np.array(y_true_all)
y_pred_all = np.array(y_pred_all)

# ── confusion matrix ──
cm = confusion_matrix(y_true_all, y_pred_all, labels=np.arange(len(CLASS_ORDER)))

# ── plot ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

fig, ax = plt.subplots(figsize=(7.5, 6.5))

# Normalize for color intensity, but show raw counts
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)

im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues",
               vmin=0, vmax=1, aspect="equal")

# Annotate cells with raw counts
thresh = 0.5
for i in range(len(CLASS_ORDER)):
    for j in range(len(CLASS_ORDER)):
        color = "white" if cm_norm[i, j] > thresh else "black"
        ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                color=color, fontsize=9, fontweight="bold")

ax.set_xticks(np.arange(len(CLASS_ORDER)))
ax.set_yticks(np.arange(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_ORDER, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(CLASS_ORDER, fontsize=9)
ax.set_xlabel("Predicted Class")
ax.set_ylabel("True Class")
ax.set_title("8-Class Agent Taxonomy: GBM 5-Fold CV Confusion Matrix")

cbar = fig.colorbar(im, ax=ax, shrink=0.8, label="Recall (row-normalized)")

fig.tight_layout()
fig.savefig(str(OUT), dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
print(f"Overall accuracy: {(y_true_all == y_pred_all).mean():.4f}")
print(f"Confusion matrix:\n{cm}")
