#!/usr/bin/env python3
"""
Generate publication-quality figures for Paper 2: Agent Tool Security.

Reads:
  experiments/full_catalog_scan_results.json  (138-server scan)
  experiments/recalibrated_risk_scores.json   (risk scores per repo)
  dynamic_testing/llm_dynamic_results_opus.json (16 dynamic scenarios)
  data/taxonomy.json                          (category -> attack surface)

Outputs (all in figures/):
  fig_category_protocol_heatmap.pdf
  fig_severity_distribution.pdf
  fig_risk_score_distribution.pdf
  fig_dynamic_harness_matrix.pdf
  fig_system_architecture.pdf
"""

import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
EXP_DIR = os.path.join(PAPER_DIR, "experiments")
DYN_DIR = os.path.join(PAPER_DIR, "dynamic_testing")
DATA_DIR = os.path.join(PAPER_DIR, "data")
FIG_DIR = os.path.join(PAPER_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Global style -- colorblind-friendly, publication-ready
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Colorblind-safe palette (IBM Design / Wong)
CB_BLUE    = "#0072B2"
CB_ORANGE  = "#E69F00"
CB_GREEN   = "#009E73"
CB_RED     = "#D55E00"
CB_PURPLE  = "#CC79A7"
CB_CYAN    = "#56B4E9"
CB_YELLOW  = "#F0E442"
CB_BLACK   = "#000000"

SEVERITY_COLORS = {
    "critical": CB_RED,
    "high": CB_ORANGE,
    "medium": CB_CYAN,
}

PROTOCOL_ORDER = ["MCP", "OpenAI", "LangChain", "Web3"]
PROTOCOL_MAP = {
    "mcp": "MCP",
    "openai": "OpenAI",
    "langchain": "LangChain",
    "web3_native": "Web3",
}

SURFACE_COLORS = {
    "S1": CB_BLUE,
    "S2": CB_ORANGE,
    "S3": CB_RED,
    "S4": CB_GREEN,
    "S5": CB_PURPLE,
}
SURFACE_LABELS = {
    "S1": "S1: Tool Definition",
    "S2": "S2: Input Construction",
    "S3": "S3: Execution",
    "S4": "S4: Output Handling",
    "S5": "S5: Cross-Tool",
}


def load_json(path):
    if not os.path.exists(path):
        print(f"  [SKIP] {path} not found")
        return None
    with open(path) as f:
        return json.load(f)


# ===================================================================
# Figure 1: Category x Protocol Heatmap
# ===================================================================
def fig_category_protocol_heatmap(scan_data):
    """Heatmap of finding counts: rows=top-15 vuln categories, cols=4 protocols."""
    if scan_data is None:
        print("  [SKIP] fig_category_protocol_heatmap: no scan data")
        return

    # Aggregate by (protocol, category)
    proto_cats = defaultdict(lambda: Counter())
    for repo in scan_data["repo_results"]:
        proto_raw = repo.get("detected_protocol") or repo.get("catalog_protocol", "unknown")
        proto = PROTOCOL_MAP.get(proto_raw)
        if proto is None:
            continue  # skip "unknown"
        for cat, cnt in repo.get("by_category", {}).items():
            proto_cats[proto][cat] += cnt

    # Determine top-15 categories by total count
    total_by_cat = Counter()
    for proto in PROTOCOL_ORDER:
        for cat, cnt in proto_cats[proto].items():
            total_by_cat[cat] += cnt
    top15 = [cat for cat, _ in total_by_cat.most_common(15)]

    # Build matrix (rows = categories, cols = protocols)
    matrix = np.zeros((len(top15), len(PROTOCOL_ORDER)), dtype=int)
    for j, proto in enumerate(PROTOCOL_ORDER):
        for i, cat in enumerate(top15):
            matrix[i, j] = proto_cats[proto].get(cat, 0)

    # Row and column marginal totals
    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)

    # Pretty category labels
    cat_labels = [c.replace("_", " ").replace("tx ", "TX ").title() for c in top15]

    fig, ax = plt.subplots(figsize=(7.5, 7.0))

    # Use log-normalized color scale for better visibility
    from matplotlib.colors import LogNorm, Normalize
    vmax = matrix.max()
    # Use a sequential colorblind-friendly colormap
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    # Plot heatmap
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   norm=LogNorm(vmin=max(1, matrix[matrix > 0].min()), vmax=vmax))

    # Annotate cells
    for i in range(len(top15)):
        for j in range(len(PROTOCOL_ORDER)):
            val = matrix[i, j]
            color = "white" if val > vmax * 0.4 else "black"
            if val == 0:
                txt = "-"
                color = "#999999"
            else:
                txt = str(val)
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    fontweight="bold" if val > vmax * 0.3 else "normal",
                    color=color)

    # Axis labels
    ax.set_xticks(range(len(PROTOCOL_ORDER)))
    ax.set_xticklabels(PROTOCOL_ORDER, fontsize=10, fontweight="bold")
    ax.set_yticks(range(len(top15)))
    ax.set_yticklabels(cat_labels, fontsize=9)

    # Marginal totals -- right side column header aligned with data
    margin_x = len(PROTOCOL_ORDER) - 0.5 + 0.35
    ax.text(margin_x, -0.65, "Total",
            ha="center", va="center", fontsize=9.5, fontweight="bold", color="#333333")
    for i, total in enumerate(row_totals):
        ax.text(margin_x, i, f"{total:,}",
                ha="center", va="center", fontsize=8.5, color="#333333",
                fontweight="bold")

    # Marginal totals -- bottom
    for j, total in enumerate(col_totals):
        ax.text(j, len(top15) + 0.15, f"{total:,}",
                ha="center", va="top", fontsize=9, color="#333333",
                fontweight="bold")

    ax.set_xlim(-0.5, len(PROTOCOL_ORDER) + 0.5)
    ax.set_ylim(len(top15) - 0.5, -0.85)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.12)
    cbar.set_label("Finding Count (log scale)", fontsize=10)

    ax.set_title("Vulnerability Findings by Category and Protocol", fontsize=13, pad=10)
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_category_protocol_heatmap.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ===================================================================
# Figure 2: Severity Distribution (stacked bar + percentages)
# ===================================================================
def fig_severity_distribution(scan_data):
    """Stacked bar chart: x=4 protocols, y=finding count, stacked by severity."""
    if scan_data is None:
        print("  [SKIP] fig_severity_distribution: no scan data")
        return

    proto_sevs = defaultdict(lambda: Counter())
    for repo in scan_data["repo_results"]:
        proto_raw = repo.get("detected_protocol") or repo.get("catalog_protocol", "unknown")
        proto = PROTOCOL_MAP.get(proto_raw)
        if proto is None:
            continue
        for sev, cnt in repo.get("by_severity", {}).items():
            proto_sevs[proto][sev] += cnt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5),
                                    gridspec_kw={"width_ratios": [1.1, 1]})

    x = np.arange(len(PROTOCOL_ORDER))
    width = 0.55

    sev_order = ["critical", "high", "medium"]

    # --- Left panel: absolute counts ---
    bottoms = np.zeros(len(PROTOCOL_ORDER))
    bars_per_sev = {}
    for sev in sev_order:
        vals = np.array([proto_sevs[p].get(sev, 0) for p in PROTOCOL_ORDER])
        bars = ax1.bar(x, vals, width, bottom=bottoms,
                       label=sev.capitalize(), color=SEVERITY_COLORS[sev],
                       edgecolor="white", linewidth=0.5)
        bars_per_sev[sev] = vals
        bottoms += vals

    # Total labels on top
    totals = bottoms.astype(int)
    for i, total in enumerate(totals):
        ax1.text(i, total + max(totals) * 0.02, f"n={total:,}",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(PROTOCOL_ORDER, fontsize=10)
    ax1.set_ylabel("Number of Findings")
    ax1.set_title("(a) Absolute Counts", fontsize=11)
    ax1.legend(loc="upper right", framealpha=0.9)
    ax1.grid(axis="y", alpha=0.2, linestyle="--")

    # --- Right panel: percentage ---
    bottoms_pct = np.zeros(len(PROTOCOL_ORDER))
    for sev in sev_order:
        vals = np.array([proto_sevs[p].get(sev, 0) for p in PROTOCOL_ORDER], dtype=float)
        pcts = vals / totals * 100
        ax2.bar(x, pcts, width, bottom=bottoms_pct,
                label=sev.capitalize(), color=SEVERITY_COLORS[sev],
                edgecolor="white", linewidth=0.5)
        # Annotate percentage inside bar if large enough
        for i, pct in enumerate(pcts):
            if pct > 8:
                ax2.text(i, bottoms_pct[i] + pct / 2,
                         f"{pct:.0f}%", ha="center", va="center",
                         fontsize=8, color="white", fontweight="bold")
        bottoms_pct += pcts

    ax2.set_xticks(x)
    ax2.set_xticklabels(PROTOCOL_ORDER, fontsize=10)
    ax2.set_ylabel("Percentage of Findings")
    ax2.set_title("(b) Severity Proportions", fontsize=11)
    ax2.set_ylim(0, 105)
    ax2.grid(axis="y", alpha=0.2, linestyle="--")

    fig.suptitle("Severity Distribution Across Protocols (N=5,429 findings, 4 protocols)",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_severity_distribution.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ===================================================================
# Figure 3: Risk Score Distribution (violin + box overlay)
# ===================================================================
def fig_risk_score_distribution(risk_data):
    """Violin/box plot of risk scores by protocol with saturation threshold."""
    if risk_data is None:
        print("  [SKIP] fig_risk_score_distribution: no risk data")
        return

    per_repo = risk_data.get("per_repo_scores", [])
    if not per_repo:
        print("  [SKIP] fig_risk_score_distribution: no per_repo_scores")
        return

    # Use the recommended calibration: linear_p90
    score_key = "score_linear_p90"
    saturation_threshold = risk_data["calibrations"]["linear_p90"]["expected_max"]

    # Group scores by protocol
    proto_scores = defaultdict(list)
    for repo in per_repo:
        proto_raw = repo.get("protocol", "unknown")
        proto = PROTOCOL_MAP.get(proto_raw)
        if proto is None:
            continue
        proto_scores[proto].append(repo[score_key])

    # Build arrays in protocol order (only protocols with data)
    plot_protocols = [p for p in PROTOCOL_ORDER if proto_scores[p]]
    plot_data = [proto_scores[p] for p in plot_protocols]

    fig, ax = plt.subplots(figsize=(7, 5))

    # Only draw violins for protocols with enough data points (n >= 4)
    violin_positions = [i for i, d in enumerate(plot_data) if len(d) >= 4]
    violin_data = [plot_data[i] for i in violin_positions]
    if violin_data:
        parts = ax.violinplot(violin_data, positions=violin_positions,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(CB_CYAN)
            pc.set_alpha(0.3)
            pc.set_edgecolor("none")

    # For low-n protocols, scatter individual points instead
    for i, d in enumerate(plot_data):
        if len(d) < 4:
            jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(d))
            ax.scatter(np.full(len(d), i) + jitter, d,
                       color=CB_BLUE, alpha=0.7, s=40, zorder=5,
                       edgecolors="white", linewidth=0.5)

    # Overlay box plots (only for n >= 2)
    box_positions = [i for i, d in enumerate(plot_data) if len(d) >= 2]
    box_data = [plot_data[i] for i in box_positions]
    if box_data:
        bp = ax.boxplot(box_data, positions=box_positions,
                        widths=0.25, patch_artist=True,
                        showfliers=True, flierprops=dict(marker="o", markersize=4,
                                                          markerfacecolor=CB_RED,
                                                          markeredgecolor=CB_RED,
                                                          alpha=0.6))
        for patch in bp["boxes"]:
            patch.set_facecolor(CB_BLUE)
            patch.set_alpha(0.6)
        for median in bp["medians"]:
            median.set_color("white")
            median.set_linewidth(2)
        for whisker in bp["whiskers"]:
            whisker.set_color(CB_BLACK)
            whisker.set_linewidth(0.8)
        for cap in bp["caps"]:
            cap.set_color(CB_BLACK)
            cap.set_linewidth(0.8)

    # Saturation threshold line (scores are 0-100, threshold is at 100)
    ax.axhline(y=100, color=CB_RED, linestyle="--", linewidth=1.2, alpha=0.7,
               label="Saturation (score=100)")

    # Annotate stats
    for i, (proto, scores) in enumerate(zip(plot_protocols, plot_data)):
        scores_arr = np.array(scores)
        n = len(scores_arr)
        med = np.median(scores_arr)
        sat = int(np.sum(scores_arr >= 100))
        y_top = max(scores_arr) if len(scores_arr) > 0 else 0
        note = f"  (n<4)" if n < 4 else ""
        ax.text(i, y_top + 4,
                f"n={n}{note}\nmed={med:.1f}\nsat={sat}",
                ha="center", va="bottom", fontsize=7, color="#333333")

    ax.set_xticks(range(len(plot_protocols)))
    ax.set_xticklabels(plot_protocols, fontsize=11, fontweight="bold")
    ax.set_ylabel("Risk Score (linear p90 calibration)")
    ax.set_title("Risk Score Distribution by Protocol", fontsize=13, pad=10)
    ax.set_ylim(-5, 130)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.2, linestyle="--")

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_risk_score_distribution.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ===================================================================
# Figure 4: Dynamic Harness Matrix
# ===================================================================
def fig_dynamic_harness_matrix(dyn_data, taxonomy):
    """Matrix of dynamic test results: rows=16 scenarios, cols=outcome, color=attack surface."""
    if dyn_data is None:
        print("  [SKIP] fig_dynamic_harness_matrix: no dynamic data")
        return

    # Build category -> primary attack surface from taxonomy
    cat_to_surface = {}
    if taxonomy:
        for p in taxonomy.get("patterns", []):
            cat = p["category"]
            if cat not in cat_to_surface:
                cat_to_surface[cat] = p["attack_surface"]

    scenarios = dyn_data["scenarios"]
    n = len(scenarios)

    # Determine outcome for each scenario
    # Three possible outcomes: attack_success, refused (warned), safe_proceed
    outcome_cols = ["Attack\nSuccess", "Refused /\nWarned", "Safe\nProceed"]

    # Build matrix: rows=scenarios, cols=outcomes, values = 0 or 1
    matrix = np.zeros((n, 3), dtype=int)
    surface_ids = []

    for i, s in enumerate(scenarios):
        cat = s["category"]
        surface_raw = cat_to_surface.get(cat, "S3_execution")
        surface_id = surface_raw.split("_")[0]  # "S1", "S2", etc.
        surface_ids.append(surface_id)

        if s["attack_success"]:
            matrix[i, 0] = 1
        elif s["refused_or_warned"]:
            matrix[i, 1] = 1
        else:
            matrix[i, 2] = 1

    # Sort scenarios by surface, then by scenario_id
    sort_idx = sorted(range(n), key=lambda i: (surface_ids[i], scenarios[i]["scenario_id"]))
    matrix = matrix[sort_idx]
    sorted_scenarios = [scenarios[i] for i in sort_idx]
    sorted_surfaces = [surface_ids[i] for i in sort_idx]

    # Build row labels
    row_labels = []
    for s, surf in zip(sorted_scenarios, sorted_surfaces):
        short_name = s["scenario_name"]
        if len(short_name) > 35:
            short_name = short_name[:33] + ".."
        row_labels.append(f"[{surf}] {short_name}")

    fig, ax = plt.subplots(figsize=(7, 7.5))

    # Custom colormap: white=0, colored=1 per outcome
    outcome_colors = [CB_RED, CB_GREEN, CB_CYAN]

    # Draw cells
    for i in range(n):
        for j in range(3):
            if matrix[i, j] == 1:
                color = outcome_colors[j]
                alpha = 0.85
            else:
                color = "#F5F5F5"
                alpha = 1.0
            rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 facecolor=color, alpha=alpha,
                                 edgecolor="white", linewidth=1.5)
            ax.add_patch(rect)
            if matrix[i, j] == 1:
                # Use clear ASCII markers: FAIL / PASS / neutral
                markers = ["FAIL", "OK", "safe"]
                fsizes = [9, 9, 8]
                ax.text(j, i, markers[j], ha="center", va="center",
                        fontsize=fsizes[j], fontweight="bold",
                        color="white",
                        fontfamily="sans-serif")

    # Color-code row labels by attack surface
    ax.set_yticks(range(n))
    ax.set_yticklabels(row_labels, fontsize=8)
    for i, (label, surf) in enumerate(zip(ax.get_yticklabels(), sorted_surfaces)):
        label.set_color(SURFACE_COLORS.get(surf, CB_BLACK))
        label.set_fontweight("bold")

    ax.set_xticks(range(3))
    ax.set_xticklabels(outcome_cols, fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False)

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(n - 0.5, -0.5)

    # Summary bar at bottom
    totals = matrix.sum(axis=0)
    summary_text = (f"Attack Success: {totals[0]}/{n} ({totals[0]/n*100:.0f}%)   |   "
                    f"Refused: {totals[1]}/{n} ({totals[1]/n*100:.0f}%)   |   "
                    f"Safe: {totals[2]}/{n} ({totals[2]/n*100:.0f}%)")
    ax.text(1, n + 0.6, summary_text, ha="center", va="top",
            fontsize=8.5, fontweight="bold", color="#333333",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#F0F0F0",
                      edgecolor="#CCCCCC"))

    # Legend for attack surfaces
    legend_handles = [mpatches.Patch(facecolor=SURFACE_COLORS[s], label=SURFACE_LABELS[s])
                      for s in ["S1", "S2", "S3", "S4", "S5"]]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=7,
              title="Attack Surface", title_fontsize=8,
              framealpha=0.9, ncol=1)

    ax.set_title("Dynamic Harness: LLM Response to 16 Attack Scenarios\n"
                 "(Claude Opus 4.6, single-turn evaluation)",
                 fontsize=12, pad=12)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_dynamic_harness_matrix.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ===================================================================
# Figure 5: System Architecture Diagram
# ===================================================================
def fig_system_architecture():
    """System architecture diagram showing 5 components, trust boundaries, attack surfaces."""

    fig, ax = plt.subplots(figsize=(11, 7.5))
    ax.set_xlim(-0.5, 11)
    ax.set_ylim(-0.5, 8)
    ax.set_aspect("equal")
    ax.axis("off")

    def draw_box(x, y, w, h, label, color, sublabel=None, fontsize=11):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.15",
                             facecolor=color, edgecolor="#333333",
                             linewidth=1.8, alpha=0.85, zorder=3)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2 + (0.15 if sublabel else 0),
                label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", zorder=4)
        if sublabel:
            ax.text(x + w / 2, y + h / 2 - 0.25,
                    sublabel, ha="center", va="center",
                    fontsize=8, color="#444444", zorder=4)

    def draw_arrow(x1, y1, x2, y2, label=None, label_offset=(0, 0.18),
                   color="#333333", style="->", ha="center", va="bottom"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color,
                                    linewidth=1.8, connectionstyle="arc3,rad=0"),
                    zorder=2)
        if label:
            mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
            ax.text(mx, my, label, ha=ha, va=va,
                    fontsize=7.5, color="#555555", fontstyle="italic",
                    bbox=dict(boxstyle="round,pad=0.08", facecolor="white",
                              edgecolor="none", alpha=0.9),
                    zorder=5)

    def draw_attack_surface(x, y, label, color):
        ax.plot(x, y, marker="*", markersize=18, color=color,
                markeredgecolor="white", markeredgewidth=0.8, zorder=6)
        ax.text(x, y - 0.28, label, ha="center", va="top",
                fontsize=8, fontweight="bold", color=color, zorder=6,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white",
                          edgecolor=color, alpha=0.95, linewidth=1.0))

    def draw_trust_boundary(x, y, w, h, label):
        rect = plt.Rectangle((x, y), w, h, fill=False,
                              edgecolor="#AAAAAA", linewidth=1.5,
                              linestyle=(0, (5, 3)), zorder=1)
        ax.add_patch(rect)
        ax.text(x + 0.15, y + h - 0.12, label, fontsize=8,
                color="#888888", va="top", fontstyle="italic", zorder=1)

    # Trust boundaries -- wider spacing
    draw_trust_boundary(0.0, 3.2, 3.8, 4.3, "User Trust Domain")
    draw_trust_boundary(4.5, 0.2, 6.2, 7.3, "Tool Server Domain")

    # Components -- more separated
    # 1. Agent (LLM + orchestration)
    draw_box(0.3, 4.5, 3.2, 2.0, "AI Agent", "#B3D9FF",
             sublabel="LLM + Orchestration")

    # 2. Tool Registry
    draw_box(5.0, 5.5, 2.4, 1.5, "Tool Registry", "#FFD9B3",
             sublabel="Descriptions + Schemas")

    # 3. Tool Server (execution)
    draw_box(5.0, 2.8, 2.4, 1.8, "Tool Server", "#FFB3B3",
             sublabel="MCP / OpenAI / LC / Web3")

    # 4. Execution Environment
    draw_box(8.0, 2.8, 2.3, 1.8, "Execution Env", "#D9FFB3",
             sublabel="Sandbox / Runtime")

    # 5. Blockchain
    draw_box(8.0, 0.5, 2.3, 1.5, "Blockchain", "#E0B3FF",
             sublabel="ETH / Polygon / etc.")

    # Arrows between components (carefully spaced to avoid overlap)
    # Agent <-> Tool Registry (top path)
    draw_arrow(3.5, 5.9, 5.0, 6.3, "discover tools",
               label_offset=(0, 0.2))
    draw_arrow(5.0, 5.8, 3.5, 5.5, "tool metadata",
               label_offset=(0, -0.25), va="top")

    # Agent <-> Tool Server (middle path -- separate invoke/result vertically)
    draw_arrow(3.5, 4.9, 5.0, 4.2, "invoke(params)",
               label_offset=(-0.35, 0.22))
    draw_arrow(5.0, 3.2, 3.5, 3.9, "result / output",
               label_offset=(-0.6, -0.28), va="top")

    # Tool Server -> Execution Env
    draw_arrow(7.4, 3.7, 8.0, 3.7, "exec",
               label_offset=(0, 0.22))

    # Execution Env -> Blockchain
    draw_arrow(9.15, 2.8, 9.15, 2.0, "tx / call",
               label_offset=(0.4, 0), ha="left")

    # Attack surface markers -- positioned at trust boundary crossings, clear of labels
    draw_attack_surface(4.3, 6.6, "S1", SURFACE_COLORS["S1"])   # between Agent & Registry
    draw_attack_surface(4.5, 4.8, "S2", SURFACE_COLORS["S2"])   # Agent -> Tool Server input
    draw_attack_surface(7.5, 2.4, "S3", SURFACE_COLORS["S3"])   # Tool Server / Exec Env boundary
    draw_attack_surface(4.5, 3.2, "S4", SURFACE_COLORS["S4"])   # Tool Server -> Agent output
    draw_attack_surface(1.9, 3.5, "S5", SURFACE_COLORS["S5"])   # cross-tool within Agent

    # Attack surface legend -- bottom left, well separated from components
    legend_entries = [
        ("S1", "S1: Tool Definition -- poisoned metadata"),
        ("S2", "S2: Input Construction -- parameter injection"),
        ("S3", "S3: Execution -- runtime privileges, credentials"),
        ("S4", "S4: Output Handling -- data leakage, XSS"),
        ("S5", "S5: Cross-Tool -- confused deputy, state confusion"),
    ]
    for i, (sid, txt) in enumerate(legend_entries):
        y_pos = 2.8 - i * 0.4
        ax.plot(-0.1, y_pos, marker="*", markersize=10,
                color=SURFACE_COLORS[sid], markeredgecolor="white",
                markeredgewidth=0.3, zorder=5)
        ax.text(0.2, y_pos, txt, fontsize=7.5,
                color=SURFACE_COLORS[sid], fontweight="bold",
                va="center", zorder=5)

    ax.set_title("Agent-Tool Interface: System Architecture and Attack Surfaces (S1--S5)",
                 fontsize=13, fontweight="bold", pad=12)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, "fig_system_architecture.pdf")
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


# ===================================================================
# Main
# ===================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Paper 2: Generating publication-quality figures...")
    print("=" * 60)

    scan_data = load_json(os.path.join(EXP_DIR, "full_catalog_scan_results.json"))
    risk_data = load_json(os.path.join(EXP_DIR, "recalibrated_risk_scores.json"))
    dyn_data = load_json(os.path.join(DYN_DIR, "llm_dynamic_results_opus.json"))
    taxonomy = load_json(os.path.join(DATA_DIR, "taxonomy.json"))

    print("\n[1/5] Category x Protocol Heatmap...")
    fig_category_protocol_heatmap(scan_data)

    print("[2/5] Severity Distribution...")
    fig_severity_distribution(scan_data)

    print("[3/5] Risk Score Distribution...")
    fig_risk_score_distribution(risk_data)

    print("[4/5] Dynamic Harness Matrix...")
    fig_dynamic_harness_matrix(dyn_data, taxonomy)

    print("[5/5] System Architecture...")
    fig_system_architecture()

    print("\n" + "=" * 60)
    print("All figures saved to:", FIG_DIR)
    print("=" * 60)
