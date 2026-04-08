#!/usr/bin/env python3
"""Generate publication-quality figures for Paper 0: AI Agent Theory."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Style configuration
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAPER_DIR = os.path.dirname(SCRIPT_DIR)
EXP_DIR = os.path.join(PAPER_DIR, 'experiments')
FIG_DIR = SCRIPT_DIR

palette = sns.color_palette("Set2", 10)


def load_json(filename):
    path = os.path.join(EXP_DIR, filename)
    if not os.path.exists(path):
        print(f"  [SKIP] {filename} not found")
        return None
    with open(path) as f:
        return json.load(f)


def fig1_taxonomy_overview(data):
    """Heatmap showing 3 dimensions x 8 categories of the taxonomy."""
    if data is None:
        print("  [SKIP] fig1: taxonomy_comparison_results.json missing")
        return

    # Our 8 categories (from the coverage_matrix structure)
    our_categories = [
        "Deterministic\nScript",
        "Stateful\nBot",
        "MEV\nSearcher",
        "Bridge\nAgent",
        "RL Trading\nAgent",
        "DAO\nAgent",
        "LLM-Powered\nAgent",
        "Multi-Modal\nAgent",
    ]

    # 3 dimensions: Autonomy, Environment, Decision
    # Map each category to dimension values
    # Autonomy: NONE(0), SUPERVISED(1), COLLABORATIVE(2), FULL(3)
    autonomy = [0, 1, 1, 1, 2, 3, 2, 3]
    # Environment: ON_CHAIN(0), HYBRID(1), CROSS_CHAIN(2), MULTI_MODAL(3)
    environment = [0, 0, 0, 2, 1, 1, 1, 3]
    # Decision: DETERMINISTIC(0), STATISTICAL(1), RL(2), LLM(3)
    decision = [0, 0, 1, 0, 2, 1, 3, 3]

    dim_labels = ['Autonomy Level', 'Environment Type', 'Decision Model']
    dim_tick_labels = {
        'Autonomy Level': ['NONE', 'SUPERVISED', 'COLLABORATIVE', 'FULL'],
        'Environment Type': ['ON-CHAIN', 'HYBRID', 'CROSS-CHAIN', 'MULTI-MODAL'],
        'Decision Model': ['DETERMINISTIC', 'STATISTICAL', 'RL', 'LLM'],
    }

    matrix = np.array([autonomy, environment, decision])

    fig, ax = plt.subplots(figsize=(10, 4))
    cmap = sns.color_palette("YlOrRd", 4)
    cmap_obj = matplotlib.colors.ListedColormap(cmap)

    im = ax.imshow(matrix, cmap=cmap_obj, aspect='auto', vmin=0, vmax=3)

    ax.set_xticks(range(len(our_categories)))
    ax.set_xticklabels(our_categories, fontsize=9, ha='center')
    ax.set_yticks(range(len(dim_labels)))
    ax.set_yticklabels(dim_labels, fontsize=11)

    # Add text annotations
    for i in range(len(dim_labels)):
        ticks = dim_tick_labels[dim_labels[i]]
        for j in range(len(our_categories)):
            val = matrix[i, j]
            text_color = 'white' if val >= 2 else 'black'
            ax.text(j, i, ticks[val], ha='center', va='center',
                    fontsize=7.5, fontweight='bold', color=text_color)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.ax.set_yticklabels(['Level 0', 'Level 1', 'Level 2', 'Level 3'], fontsize=9)

    ax.set_title('Web3 AI Agent Taxonomy: 3 Dimensions x 8 Categories', fontsize=13, pad=10)
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_taxonomy_overview.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig2_validation_results(data):
    """Bar chart showing C1-C4 pass rates for agents vs non-agents."""
    if data is None:
        print("  [SKIP] fig2: onchain_validation_results.json missing")
        return

    conditions = ['C1', 'C2', 'C3', 'C4']

    # Combine platform_agents + known_agents as "agents"
    agent_sections = []
    if 'section_2_platform_agents' in data:
        agent_sections.append(data['section_2_platform_agents'])
    if 'section_3_known_agents' in data:
        agent_sections.append(data['section_3_known_agents'])

    # Compute agent pass rates
    agent_details = []
    for sec in agent_sections:
        agent_details.extend(sec.get('details', []))

    if not agent_details:
        print("  [SKIP] fig2: no agent details found")
        return

    n_agents = len(agent_details)
    agent_rates = []
    for c in ['c1_pass', 'c2_pass', 'c3_pass', 'c4_pass']:
        passes = sum(1 for d in agent_details
                     if str(d.get(c, '')).lower() == 'true' or d.get(c) is True)
        agent_rates.append(passes / n_agents * 100)

    # Non-agents: contracts + humans
    non_agent_details = []
    sec4 = data.get('section_4_known_non_agents', {})
    non_agent_details.extend(sec4.get('contract_details', []))
    non_agent_details.extend(sec4.get('human_details', []))

    n_non = len(non_agent_details)
    non_rates = []
    for c in ['c1_pass', 'c2_pass', 'c3_pass', 'c4_pass']:
        passes = sum(1 for d in non_agent_details
                     if str(d.get(c, '')).lower() == 'true' or d.get(c) is True)
        non_rates.append(passes / n_non * 100 if n_non > 0 else 0)

    x = np.arange(len(conditions))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, agent_rates, width, label=f'Known Agents (n={n_agents})',
                   color=palette[0], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, non_rates, width, label=f'Non-Agents (n={n_non})',
                   color=palette[1], edgecolor='black', linewidth=0.5)

    ax.set_ylabel('Pass Rate (%)')
    ax.set_xlabel('Condition')
    ax.set_title('Validation: C1-C4 Pass Rates for Agents vs Non-Agents')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.0f}%',
                ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 1.5, f'{h:.0f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_validation_results.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig3_ablation(data):
    """Bar chart showing false positives when each condition is removed."""
    if data is None:
        print("  [SKIP] fig3: onchain_validation_results.json missing")
        return

    ablation = data.get('section_6_ablation', {})
    if not ablation:
        print("  [SKIP] fig3: no ablation data")
        return

    conditions = sorted(ablation.keys())
    fp_counts = [ablation[c].get('false_positives_count', 0) for c in conditions]
    descs = [ablation[c].get('description', '').split('(')[0].strip() for c in conditions]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = [palette[i] for i in range(len(conditions))]
    bars = ax.bar(range(len(conditions)), fp_counts, color=colors,
                  edgecolor='black', linewidth=0.5)

    ax.set_xticks(range(len(conditions)))
    labels = [f'{c}\n({d})' for c, d in zip(conditions, descs)]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('False Positives (non-agents incorrectly classified)')
    ax.set_title('Ablation Study: False Positives When Removing Each Condition')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(fp_counts) * 1.3 if max(fp_counts) > 0 else 5)

    for bar, val in zip(bars, fp_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_ablation.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig4_framework_comparison(data):
    """Heatmap comparing 6 frameworks x coverage of our 8 categories."""
    if data is None:
        print("  [SKIP] fig4: taxonomy_comparison_results.json missing")
        return

    frameworks_data = data.get('frameworks', {})
    if not frameworks_data:
        print("  [SKIP] fig4: no frameworks data")
        return

    our_categories = [
        "Det.\nScript",
        "Stateful\nBot",
        "MEV\nSearcher",
        "Bridge\nAgent",
        "RL Trading\nAgent",
        "DAO\nAgent",
        "LLM\nAgent",
        "Multi-Modal\nAgent",
    ]

    # Add our framework as row 0 (perfect coverage)
    framework_names = ['Ours (Web3 AI Agent)']
    matrix_rows = [np.array([2, 2, 2, 2, 2, 2, 2, 2])]

    for name in ["Russell & Norvig (2020)", "Wooldridge & Jennings (1995)",
                  "Franklin & Graesser (1996)", "Parasuraman, Sheridan & Wickens (2000)",
                  "He et al. (2025)"]:
        fw = frameworks_data.get(name, {})
        cov_mat = fw.get('coverage_matrix', [])
        if not cov_mat:
            continue
        # Max coverage across their categories for each of our 8
        arr = np.array(cov_mat)
        max_coverage = arr.max(axis=0) if arr.ndim == 2 else np.zeros(8)
        framework_names.append(name.replace(' (', '\n('))
        matrix_rows.append(max_coverage)

    matrix = np.array(matrix_rows, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    cmap = sns.color_palette("RdYlGn", as_cmap=False, n_colors=3)
    cmap_obj = matplotlib.colors.ListedColormap(cmap)

    im = ax.imshow(matrix, cmap=cmap_obj, aspect='auto', vmin=0, vmax=2)

    ax.set_xticks(range(len(our_categories)))
    ax.set_xticklabels(our_categories, fontsize=8, ha='center')
    ax.set_yticks(range(len(framework_names)))
    ax.set_yticklabels(framework_names, fontsize=9)

    labels_map = {0: 'None', 1: 'Partial', 2: 'Strong'}
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = int(matrix[i, j])
            color = 'white' if val == 0 else 'black'
            ax.text(j, i, labels_map[val], ha='center', va='center',
                    fontsize=7, fontweight='bold', color=color)

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], shrink=0.6)
    cbar.ax.set_yticklabels(['None', 'Partial', 'Strong'], fontsize=9)

    ax.set_title('Framework Comparison: Coverage of Our 8 Agent Categories', fontsize=12, pad=10)
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_framework_comparison.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


if __name__ == '__main__':
    print("Paper 0: Generating figures...")

    validation = load_json('onchain_validation_results.json')
    taxonomy = load_json('taxonomy_comparison_results.json')
    case_study = load_json('case_study_results.json')

    fig1_taxonomy_overview(taxonomy)
    fig2_validation_results(validation)
    fig3_ablation(validation)
    fig4_framework_comparison(taxonomy)

    print("Paper 0: Done.")
