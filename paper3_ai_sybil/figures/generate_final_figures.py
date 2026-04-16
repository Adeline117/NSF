#!/usr/bin/env python3
"""
Generate final publication figures for Paper 3 (AI Sybil).

Produces:
  a) fig_graph_evasion.pdf       — bar chart: evasion rates across 4 detector types x 3 tiers
  b) fig_equilibrium_convergence.pdf — AUC trajectory over 10 rounds (3-seed mean +/- std)
  c) fig_closed_loop_comparison.pdf  — Round-1 evasion: LLM 0.980 vs parametric 0.691

Data sources:
  - experiments/adversarial_best_response_results.json
  - experiments/experiment_graph_detector_results.json
  - experiments/closed_loop_full_results.json
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

BASE = Path(__file__).resolve().parent.parent
EXP = BASE / "experiments"
FIG = BASE / "figures"

# ── Load data ────────────────────────────────────────────────────────────
with open(EXP / "adversarial_best_response_results.json") as f:
    br_data = json.load(f)

with open(EXP / "experiment_graph_detector_results.json") as f:
    graph_data = json.load(f)

with open(EXP / "closed_loop_full_results.json") as f:
    cl_data = json.load(f)


# ════════════════════════════════════════════════════════════════════════
# (a) fig_graph_evasion.pdf — Cross-modality evasion bar chart
# ════════════════════════════════════════════════════════════════════════
def make_graph_evasion():
    comp = graph_data["comparison"]
    detectors = ["Rules\n(HasciDB)", "ML\n(LightGBM)", "GBM\n(Cross-axis)", "Graph\n(Louvain)"]
    tiers = ["Basic", "Moderate", "Advanced"]
    keys = ["rules", "lightgbm", "cross_axis_gbm", "graph_louvain"]

    vals = np.array([[comp[k][t.lower()] for t in tiers] for k in keys])

    x = np.arange(len(detectors))
    width = 0.22
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, (tier, color) in enumerate(zip(tiers, colors)):
        bars = ax.bar(x + (i - 1) * width, vals[:, i], width,
                      label=tier, color=color, edgecolor="white", linewidth=0.4)
        for bar, v in zip(bars, vals[:, i]):
            if v < 0.999:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                        f"{v:.1%}", ha="center", va="bottom", fontsize=6)

    ax.set_ylabel("Evasion Rate")
    ax.set_ylim(0.88, 1.02)
    ax.set_xticks(x)
    ax.set_xticklabels(detectors)
    ax.legend(title="Tier", frameon=True, framealpha=0.9)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_title("Cross-Modality Evasion: LLM Sybils vs. Four Detector Types")

    fig.savefig(FIG / "fig_graph_evasion.pdf")
    plt.close(fig)
    print("  Saved fig_graph_evasion.pdf")


# ════════════════════════════════════════════════════════════════════════
# (b) fig_equilibrium_convergence.pdf — AUC trajectory over 10 rounds
# ════════════════════════════════════════════════════════════════════════
def make_equilibrium_convergence():
    seeds = ["42", "137", "314"]
    rounds = list(range(10))

    auc_matrix = []
    for s in seeds:
        auc_matrix.append([r["detector_auc"] for r in br_data["best_response_runs"][s]])
    auc_matrix = np.array(auc_matrix)  # (3, 10)
    auc_mean = auc_matrix.mean(axis=0)
    auc_std = auc_matrix.std(axis=0)

    # Evasion rates (skip round 0 which is None)
    eva_matrix = []
    for s in seeds:
        row = []
        for r in br_data["best_response_runs"][s]:
            row.append(r["evasion_rate"] if r["evasion_rate"] is not None else np.nan)
        eva_matrix.append(row)
    eva_matrix = np.array(eva_matrix)
    eva_mean = np.nanmean(eva_matrix, axis=0)
    eva_std = np.nanstd(eva_matrix, axis=0)

    fig, ax1 = plt.subplots(figsize=(3.4, 2.4))

    # AUC on left axis
    color_auc = "#4C72B0"
    ax1.plot(rounds, auc_mean, "o-", color=color_auc, markersize=4, linewidth=1.5,
             label="Detector AUC", zorder=3)
    ax1.fill_between(rounds, auc_mean - auc_std, auc_mean + auc_std,
                     color=color_auc, alpha=0.15)
    ax1.set_xlabel("Adversarial Round")
    ax1.set_ylabel("Detector AUC", color=color_auc)
    ax1.tick_params(axis="y", labelcolor=color_auc)
    ax1.set_ylim(0.50, 1.02)
    ax1.axhline(0.578, color=color_auc, linestyle="--", linewidth=0.8, alpha=0.5)
    ax1.text(9.2, 0.578, "0.578", fontsize=7, color=color_auc, va="center")

    # Evasion on right axis
    ax2 = ax1.twinx()
    color_eva = "#C44E52"
    valid = ~np.isnan(eva_mean)
    ax2.plot(np.array(rounds)[valid], eva_mean[valid], "s--", color=color_eva,
             markersize=4, linewidth=1.5, label="Evasion Rate", zorder=3)
    ax2.fill_between(np.array(rounds)[valid],
                     (eva_mean - eva_std)[valid], (eva_mean + eva_std)[valid],
                     color=color_eva, alpha=0.12)
    ax2.set_ylabel("Evasion Rate", color=color_eva)
    ax2.tick_params(axis="y", labelcolor=color_eva)
    ax2.set_ylim(-0.02, 1.02)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=7)

    ax1.set_xticks(rounds)
    ax1.spines["top"].set_visible(False)
    ax1.set_title("Best-Response Adversarial Convergence (3-seed mean)")

    fig.savefig(FIG / "fig_equilibrium_convergence.pdf")
    plt.close(fig)
    print("  Saved fig_equilibrium_convergence.pdf")


# ════════════════════════════════════════════════════════════════════════
# (c) fig_closed_loop_comparison.pdf — Round-1 evasion paired bars
# ════════════════════════════════════════════════════════════════════════
def make_closed_loop_comparison():
    seeds = ["42", "137", "314"]

    llm_r1 = []
    param_r1 = []
    for s in seeds:
        r1 = cl_data["closed_loop_runs"][s][1]  # round 1
        llm_r1.append(r1["llm_evasion_rate"])
        param_r1.append(r1["parametric_probe_evasion"])

    llm_mean = np.mean(llm_r1)
    llm_std = np.std(llm_r1)
    param_mean = np.mean(param_r1)
    param_std = np.std(param_r1)

    fig, ax = plt.subplots(figsize=(2.8, 2.4))

    x = np.array([0, 1])
    bars = ax.bar(x, [llm_mean, param_mean],
                  yerr=[llm_std, param_std],
                  color=["#C44E52", "#4C72B0"],
                  edgecolor="white", linewidth=0.5,
                  width=0.55, capsize=4, error_kw={"linewidth": 1.2})

    ax.set_xticks(x)
    ax.set_xticklabels(["LLM\n(Claude Opus)", "Parametric\n(Scalar Blend)"], fontsize=8)
    ax.set_ylabel("Round-1 Evasion Rate")
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Annotate values
    for i, (mean, std) in enumerate([(llm_mean, llm_std), (param_mean, param_std)]):
        ax.text(i, mean + std + 0.03, f"{mean:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # Delta annotation
    mid_y = (llm_mean + param_mean) / 2
    ax.annotate("", xy=(0.15, llm_mean - 0.02), xytext=(0.15, param_mean + 0.02),
                arrowprops=dict(arrowstyle="<->", color="black", linewidth=1.0))
    delta = llm_mean - param_mean
    ax.text(-0.25, mid_y, f"$\\Delta$ = +{delta:.1%}",
            fontsize=8, ha="center", va="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="wheat", alpha=0.8))

    ax.set_title("Round-1 Evasion: LLM vs. Parametric", fontsize=9)

    fig.savefig(FIG / "fig_closed_loop_comparison.pdf")
    plt.close(fig)
    print("  Saved fig_closed_loop_comparison.pdf")


# ════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating final figures for Paper 3...")
    make_graph_evasion()
    make_equilibrium_convergence()
    make_closed_loop_comparison()
    print("Done. All figures saved to", FIG)
