#!/usr/bin/env python3
"""
Generate the four main figures for the Nature Machine Intelligence submission.

    Fig 1: MI vs accuracy (Fano ceiling curve + measured points for X_23/X_31/X_47)
    Fig 2: Clustering saturation (K-Means silhouette vs k, HDBSCAN point)
    Fig 3: Execution vs decision mechanism (paired bar chart)
    Fig 4: Cross-chain (ETH vs Polygon accuracy comparison)

Reads data from ../experiments/*.json and writes PDFs to figures/.
"""

import json
import pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

HERE = pathlib.Path(__file__).resolve().parent
EXP = HERE.parent / "experiments"
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: tight Fano inversion
# ---------------------------------------------------------------------------
def fano_ceiling(mi_bits, h_y_bits, K):
    """Return accuracy ceiling by numerically inverting Fano's inequality."""
    h_cond = h_y_bits - mi_bits
    if h_cond <= 0:
        return 1.0
    pe_grid = np.linspace(0, 1 - 1 / K, 10000)
    for pe in pe_grid:
        if pe == 0:
            h_pe = 0.0
        elif pe == 1:
            h_pe = 0.0
        else:
            h_pe = -pe * np.log2(pe) - (1 - pe) * np.log2(1 - pe)
        lhs = h_pe + pe * np.log2(K - 1)
        if lhs >= h_cond:
            return 1 - pe
    return 1 / K


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(EXP / "information_theoretic_bound_results.json") as f:
    itb = json.load(f)

with open(EXP / "cluster_validation_results.json") as f:
    clust = json.load(f)

with open(EXP / "reaction_time_full_results.json") as f:
    rt = json.load(f)

with open(EXP / "polygon_replication_real_results.json") as f:
    poly = json.load(f)


# ===================================================================
# Fig 1: MI vs accuracy (Fano ceiling curve + measured points)
# ===================================================================
def fig1():
    H_Y = itb["label_sets"]["Y_8"]["H_Y_bits"]
    K = 8

    # Build the Fano ceiling curve
    mi_range = np.linspace(0, H_Y, 500)
    ceiling_curve = np.array([fano_ceiling(mi, H_Y, K) for mi in mi_range])

    # Measured points
    feature_sets = ["X_23", "X_31", "X_47"]
    mi_keys = [f"MI({fs};Y_8)" for fs in feature_sets]
    mi_vals = [itb["mi_table"][k]["MI_joint_knn_bits"] for k in mi_keys]
    fano_vals = [itb["mi_table"][k]["fano_joint"]["accuracy_ceiling_tight"] for k in mi_keys]
    acc_vals = [itb["measured_accuracies_Y8"][fs] for fs in feature_sets]

    fig, ax = plt.subplots(figsize=(5.5, 4))

    # Fano ceiling curve
    ax.plot(mi_range, ceiling_curve * 100, "k-", linewidth=1.8, label="Fano ceiling")

    # Measured ceilings (on the curve)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    markers = ["o", "s", "D"]
    for i, fs in enumerate(feature_sets):
        # Ceiling point (on the curve)
        ax.plot(mi_vals[i], fano_vals[i] * 100, markers[i], color=colors[i],
                markersize=8, zorder=5)
        # Measured accuracy (below the curve)
        ax.plot(mi_vals[i], acc_vals[i] * 100, markers[i], color=colors[i],
                markersize=8, markerfacecolor="white", markeredgewidth=1.5,
                zorder=5, label=f"${fs.replace('_', '{')+'}'[:-1] + '}'}$")
        # Vertical headroom bar
        ax.vlines(mi_vals[i], acc_vals[i] * 100, fano_vals[i] * 100,
                  colors=colors[i], linewidth=1.2, linestyle="--", alpha=0.6)

    # Shade the headroom band
    ax.fill_between(mi_range, ceiling_curve * 100, 100, alpha=0.06, color="gray")

    ax.set_xlabel("Mutual information $I(X; Y)$ (bits)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(85, 101)
    ax.set_xlim(0, H_Y * 1.05)
    ax.yaxis.set_major_formatter(PercentFormatter(decimals=0))

    # Custom legend: filled = ceiling, open = measured
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="k", linewidth=1.8, label="Fano ceiling"),
        Line2D([0], [0], marker="o", color="w", markeredgecolor="#1f77b4",
               markersize=8, markeredgewidth=1.5, label="$X_{23}$ (23 feat.)"),
        Line2D([0], [0], marker="s", color="w", markeredgecolor="#ff7f0e",
               markersize=8, markeredgewidth=1.5, label="$X_{31}$ (31 feat.)"),
        Line2D([0], [0], marker="D", color="w", markeredgecolor="#2ca02c",
               markersize=8, markeredgewidth=1.5, label="$X_{47}$ (47 feat.)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", frameon=True,
              framealpha=0.9, edgecolor="0.8")

    ax.set_title("Information-theoretic ceiling for AI agent detection")
    fig.savefig(FIG / "fig1_mi_vs_accuracy.pdf")
    plt.close(fig)
    print(f"  Saved {FIG / 'fig1_mi_vs_accuracy.pdf'}")


# ===================================================================
# Fig 2: Clustering saturation (silhouette vs k, with HDBSCAN)
# ===================================================================
def fig2():
    sweep = clust["kmeans_sweep"]
    ks = sorted(int(k) for k in sweep)
    sils = [sweep[str(k)]["silhouette"] for k in ks]
    aris = [sweep[str(k)]["ari_vs_taxonomy"] for k in ks]

    fig, ax1 = plt.subplots(figsize=(5.5, 3.8))

    color_sil = "#1f77b4"
    color_ari = "#d62728"

    ax1.plot(ks, sils, "o-", color=color_sil, linewidth=1.5, markersize=5,
             label="Silhouette score")
    ax1.set_xlabel("Number of clusters $k$")
    ax1.set_ylabel("Silhouette score", color=color_sil)
    ax1.tick_params(axis="y", labelcolor=color_sil)
    ax1.set_ylim(0.08, 0.18)

    # Mark k=3 peak
    ax1.annotate("$k=3$\n(peak)", xy=(3, sils[0]),
                 xytext=(5, sils[0] + 0.015),
                 arrowprops=dict(arrowstyle="->", color=color_sil),
                 fontsize=9, color=color_sil, ha="center")

    # HDBSCAN marker (2 clusters, plotted at k=2 on same y-scale)
    ax1.axvline(x=2, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax1.annotate("HDBSCAN\n(2 clusters)", xy=(2, 0.155),
                 fontsize=8, color="gray", ha="center", va="bottom")

    # Secondary axis: ARI
    ax2 = ax1.twinx()
    ax2.plot(ks, aris, "s--", color=color_ari, linewidth=1.2, markersize=4,
             alpha=0.7, label="ARI vs. taxonomy")
    ax2.set_ylabel("Adjusted Rand Index", color=color_ari)
    ax2.tick_params(axis="y", labelcolor=color_ari)
    ax2.set_ylim(0.05, 0.40)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right",
               frameon=True, framealpha=0.9, edgecolor="0.8")

    ax1.set_title("Unsupervised clustering saturates at 2--3 groups")
    ax1.set_xticks(range(2, 16))
    fig.savefig(FIG / "fig2_cluster_saturation.pdf")
    plt.close(fig)
    print(f"  Saved {FIG / 'fig2_cluster_saturation.pdf'}")


# ===================================================================
# Fig 3: Execution vs Decision mechanism (paired bar chart)
# ===================================================================
def fig3():
    # DeFi execution time (median): LLM vs DeFi Management
    llm_exec = rt["per_class_distributions"]["LLMPoweredAgent"]["defi_reaction_time_median"]["median"]
    defi_exec = rt["per_class_distributions"]["DeFiManagementAgent"]["defi_reaction_time_median"]["median"]

    # General decision interval (median): LLM vs DeFi Management
    llm_decision = rt["per_class_distributions"]["LLMPoweredAgent"]["reaction_time_median"]["median"]
    defi_decision = rt["per_class_distributions"]["DeFiManagementAgent"]["reaction_time_median"]["median"]

    # Cohen's d values
    d_exec = rt["headline_cohens_d_llm_vs_defi"]["defi_reaction_time_median"]["d"]
    d_decision = rt["headline_cohens_d_llm_vs_defi"]["reaction_time_median"]["d"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=False)

    bar_width = 0.35
    x = np.array([0])

    # Left panel: Execution time
    ax1.bar(x - bar_width / 2, [llm_exec], bar_width, color="#4c72b0",
            label="LLM-powered", edgecolor="white", linewidth=0.5)
    ax1.bar(x + bar_width / 2, [defi_exec], bar_width, color="#dd8452",
            label="Rule-based DeFi", edgecolor="white", linewidth=0.5)
    ax1.set_ylabel("Median latency (seconds)")
    ax1.set_title(f"DeFi execution time\n($d = {d_exec:.2f}$, n.s.)")
    ax1.set_xticks([])
    ax1.legend(fontsize=8, loc="upper left")

    # Annotate values
    ax1.text(x[0] - bar_width / 2, llm_exec + 5, f"{llm_exec:.0f}s",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.text(x[0] + bar_width / 2, defi_exec + 5, f"{defi_exec:.0f}s",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Right panel: Decision interval
    ax2.bar(x - bar_width / 2, [llm_decision], bar_width, color="#4c72b0",
            label="LLM-powered", edgecolor="white", linewidth=0.5)
    ax2.bar(x + bar_width / 2, [defi_decision], bar_width, color="#dd8452",
            label="Rule-based DeFi", edgecolor="white", linewidth=0.5)
    ax2.set_ylabel("Median latency (seconds)")
    ax2.set_title(f"General decision interval\n($d = {d_decision:.2f}$, $p < 0.004$)")
    ax2.set_xticks([])
    ax2.legend(fontsize=8, loc="upper left")

    # Annotate values
    ax2.text(x[0] - bar_width / 2, llm_decision + 8, f"{llm_decision:.0f}s",
             ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.text(x[0] + bar_width / 2, defi_decision + 8, f"{defi_decision:.0f}s",
             ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle("Execution time is a tool signature; decision interval is an agent signature",
                 fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_execution_vs_decision.pdf")
    plt.close(fig)
    print(f"  Saved {FIG / 'fig3_execution_vs_decision.pdf'}")


# ===================================================================
# Fig 4: Cross-chain (ETH vs Polygon accuracy comparison)
# ===================================================================
def fig4():
    # Ethereum baseline accuracy (X_31 GradientBoosting)
    eth_acc = itb["measured_accuracies_Y8"]["X_31"]  # 0.9526

    # Polygon: direct transfer
    poly_transfer = poly["experiment_a_cross_chain_transfer"]["accuracy"]  # 0.2222

    # Polygon: retrained (best = LogisticRegression)
    poly_retrain = poly["experiment_b_polygon_only_cv"]["LogisticRegression"]["accuracy_mean"]

    # Random baseline for 5-class
    random_5class = 0.20

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    categories = [
        "Ethereum\n(8-class, $n$=2,744)",
        "Polygon transfer\n(5-class, $n$=27)",
        "Polygon retrained\n(5-class, $n$=27)",
    ]
    accs = [eth_acc * 100, poly_transfer * 100, poly_retrain * 100]
    colors = ["#4c72b0", "#c44e52", "#dd8452"]

    bars = ax.bar(categories, accs, color=colors, edgecolor="white",
                  linewidth=0.5, width=0.55)

    # Random baseline line
    ax.axhline(y=random_5class * 100, color="gray", linestyle="--",
               linewidth=1, alpha=0.7)
    ax.text(2.35, random_5class * 100 + 1.5, "Random baseline (20%)",
            fontsize=8, color="gray", ha="right")

    # Value labels
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.1f}%", ha="center", va="bottom", fontsize=10,
                fontweight="bold")

    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Cross-chain validation: Ethereum vs. Polygon")

    fig.tight_layout()
    fig.savefig(FIG / "fig4_cross_chain.pdf")
    plt.close(fig)
    print(f"  Saved {FIG / 'fig4_cross_chain.pdf'}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("Generating NatMI figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    print("Done. All figures saved to", FIG)
