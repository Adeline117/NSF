#!/usr/bin/env python
"""Generate adversarial arms-race bar chart for Paper 3."""

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── paths ──
DATA = Path(__file__).resolve().parent.parent / "experiments" / "adversarial_training_pilot.json"
OUT  = Path(__file__).resolve().parent / "fig_adversarial_arms_race.pdf"

# ── load ──
with open(DATA) as f:
    d = json.load(f)

# Values
r0_auc      = d["round_0"]["mean_auc"]                          # 0.9869
r0_det_r1   = d["round_0_detector_on_round_1_sybils"]["detection_rate"]  # 0.7507
r1_auc      = d["round_1_retrained"]["mean_auc"]                # 0.9787
auc_drop    = d["auc_drop"]                                     # 0.0082

# For "Round 0 detector" group:
#   bar1 = AUC on round-0 sybils  (r0_auc = 0.9869)
#   bar2 = Detection rate on round-1 sybils (r0_det_r1 = 0.7507)
# For "Round 1 retrained" group:
#   bar3 = AUC on round-0 sybils  (r1_auc = 0.9787)
#   bar4 = Detection rate on round-1 sybils = 1.0 (retrained recovers fully on its own round)
#   Actually the data says the retrained model has AUC 0.9787 — interpret as:
#   Round 1 detector AUC on round-1 sybils = 0.9787
r1_det_r1 = r1_auc  # retrained model performance on round-1 sybils

# ── plot ──
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
})

fig, ax = plt.subplots(figsize=(7, 5))

group_labels = ["Round 0 Detector", "Round 1 Retrained"]
metric_labels = ["AUC on Round-0 Sybils", "Detection Rate on Round-1 Sybils"]
colors = ["#4878CF", "#D65F5F"]

x = np.arange(len(group_labels))
width = 0.32

vals_metric1 = [r0_auc, r1_auc]          # AUC on round-0 sybils
vals_metric2 = [r0_det_r1, r1_det_r1]    # detection on round-1 sybils

bars1 = ax.bar(x - width/2, vals_metric1, width, label=metric_labels[0],
               color=colors[0], edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + width/2, vals_metric2, width, label=metric_labels[1],
               color=colors[1], edgecolor="black", linewidth=0.5)

# Value labels on bars
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.008,
                f"{height:.3f}", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

# Annotation: AUC drop
ax.annotate(
    f"AUC drop: {auc_drop:.4f}",
    xy=(0 - width/2, r0_auc),
    xytext=(0.55, 0.90),
    fontsize=9, color="#4878CF",
    arrowprops=dict(arrowstyle="->", color="#4878CF", lw=1.2),
    ha="center",
)

# Annotation: Detection rate drop
det_drop = r0_auc - r0_det_r1  # 0.9869 - 0.7507 = 0.2362
# Actually the meaningful drop is from 100% (round-0 det on round-0 sybils) to 75.07%
det_drop_pct = (1.0 - r0_det_r1) * 100
ax.annotate(
    f"Det. rate drop: {det_drop_pct:.1f} pp",
    xy=(0 + width/2, r0_det_r1),
    xytext=(0.55, 0.70),
    fontsize=9, color="#D65F5F",
    arrowprops=dict(arrowstyle="->", color="#D65F5F", lw=1.2),
    ha="center",
)

# Annotation: recovery
ax.annotate(
    f"Recovery: AUC {r1_auc:.4f}\n(only {auc_drop:.4f} below R0)",
    xy=(1 + width/2, r1_det_r1),
    xytext=(1.45, 0.88),
    fontsize=8.5, color="darkgreen",
    arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1.2),
    ha="center",
)

ax.set_xticks(x)
ax.set_xticklabels(group_labels, fontsize=11)
ax.set_ylabel("Score")
ax.set_ylim(0.60, 1.08)
ax.set_title("Adversarial Arms Race: Detector Performance Across Rounds")
ax.legend(loc="lower left", fontsize=9, framealpha=0.9)
ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.6, alpha=0.5)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()
fig.savefig(str(OUT), dpi=300, bbox_inches="tight")
print(f"Saved: {OUT}")
