#!/usr/bin/env python3
"""Generate publication-quality figures for Paper 3: AI Sybil Detection."""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

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


def fig1_evasion_comparison(data):
    """Grouped bar chart: baseline AUC vs AI sybil AUC at 3 evasion levels."""
    if data is None:
        print("  [SKIP] fig1: experiment_real_results.json missing")
        return

    exp2 = data.get('exp2_ai_evasion', {})
    exp3 = data.get('exp3_enhanced_detector', {})
    if not exp2:
        print("  [SKIP] fig1: no exp2_ai_evasion data")
        return

    baseline_auc = exp2.get('baseline_auc_in_dist', 1.0)
    evasion = exp2.get('evasion_by_level', {})
    recovery = exp3.get('recovery_by_level', {})

    levels = ['basic', 'moderate', 'advanced']
    level_labels = ['Basic', 'Moderate', 'Advanced']

    baseline_aucs = [baseline_auc] * 3
    evaded_aucs = [evasion.get(l, {}).get('ml_auc', 0) for l in levels]
    recovered_aucs = [recovery.get(l, {}).get('enhanced_auc', 0) for l in levels]

    x = np.arange(len(levels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, baseline_aucs, width, label='Baseline (HasciDB only)',
           color=palette[0], edgecolor='black', linewidth=0.5)
    ax.bar(x, evaded_aucs, width, label='After AI Evasion',
           color=palette[1], edgecolor='black', linewidth=0.5)
    ax.bar(x + width, recovered_aucs, width, label='Enhanced (+ AI features)',
           color=palette[2], edgecolor='black', linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(level_labels)
    ax.set_ylabel('AUC')
    ax.set_title('Detection AUC at 3 AI Evasion Levels')
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.15)

    # Add value labels
    for bars in [ax.containers[0], ax.containers[1], ax.containers[2]]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=8, rotation=0)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_evasion_comparison.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig2_feature_importance(data):
    """Bar chart of 13-feature importance (color AI-specific vs HasciDB)."""
    if data is None:
        print("  [SKIP] fig2: experiment_real_results.json missing")
        return

    exp3 = data.get('exp3_enhanced_detector', {})
    importance = exp3.get('feature_importances', {})
    if not importance:
        print("  [SKIP] fig2: no feature_importances")
        return

    hascidb_features = {'BT', 'BW', 'HF', 'RF', 'MA'}
    ai_features = {
        'gas_price_precision', 'hour_entropy', 'behavioral_consistency',
        'action_sequence_perplexity', 'error_recovery_pattern',
        'response_latency_variance', 'gas_nonce_gap_regularity', 'eip1559_tip_precision'
    }

    sorted_feats = sorted(importance.items(), key=lambda x: x[1], reverse=False)
    features = [f for f, v in sorted_feats]
    values = [v for f, v in sorted_feats]
    colors = ['#DD8452' if f in ai_features else '#4C72B0' for f in features]

    pretty_names = {
        'BT': 'Batch Timing (BT)',
        'BW': 'Batch Wallet (BW)',
        'HF': 'High Frequency (HF)',
        'RF': 'Recycled Funds (RF)',
        'MA': 'Multi-Address (MA)',
        'gas_price_precision': 'Gas Price Precision',
        'hour_entropy': 'Hour Entropy',
        'behavioral_consistency': 'Behavioral Consistency',
        'action_sequence_perplexity': 'Action Seq. Perplexity',
        'error_recovery_pattern': 'Error Recovery Pattern',
        'response_latency_variance': 'Response Latency Var.',
        'gas_nonce_gap_regularity': 'Gas-Nonce Gap Regularity',
        'eip1559_tip_precision': 'EIP-1559 Tip Precision',
    }
    display = [pretty_names.get(f, f) for f in features]

    fig, ax = plt.subplots(figsize=(8, 6))
    y_pos = np.arange(len(features))
    ax.barh(y_pos, values, color=colors, edgecolor='black', linewidth=0.3, height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(display, fontsize=9)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Enhanced Detector: Feature Importance (13 Features)')
    ax.grid(axis='x', alpha=0.3)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#DD8452', edgecolor='black', label='AI-Specific Features'),
        Patch(facecolor='#4C72B0', edgecolor='black', label='HasciDB Indicators'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_feature_importance.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig3_indicator_cooccurrence(data):
    """Heatmap of indicator co-occurrence matrix (aggregate from first project as proxy)."""
    if data is None:
        print("  [SKIP] fig3: indicator_cooccurrence_results.json missing")
        return

    # Use the first project's co-occurrence matrix as representative
    per_project = data.get('per_project', {})
    if not per_project:
        print("  [SKIP] fig3: no per_project data")
        return

    # Average across all projects
    indicators = ['BT', 'BW', 'HF', 'RF', 'MA']
    n = len(indicators)
    avg_matrix = np.zeros((n, n))
    count = 0

    for proj, pdata in per_project.items():
        cmat = pdata.get('cooccurrence_matrix', {})
        if not cmat:
            continue
        for i, ind_i in enumerate(indicators):
            for j, ind_j in enumerate(indicators):
                avg_matrix[i, j] += cmat.get(ind_i, {}).get(ind_j, 0)
        count += 1

    if count > 0:
        avg_matrix /= count

    fig, ax = plt.subplots(figsize=(7, 6))
    mask = np.zeros_like(avg_matrix, dtype=bool)

    sns.heatmap(avg_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
                xticklabels=indicators, yticklabels=indicators,
                ax=ax, linewidths=0.5, linecolor='white',
                cbar_kws={'label': 'Co-occurrence Rate'})

    ax.set_title(f'Indicator Co-occurrence Matrix\n(Average across {count} projects)')
    ax.set_xlabel('Indicator')
    ax.set_ylabel('Indicator')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_indicator_cooccurrence.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig4_serial_sybil(data):
    """Distribution of how many projects each serial sybil appears in."""
    if data is None:
        print("  [SKIP] fig4: serial_sybil_results.json missing")
        return

    dist = data.get('n_projects_distribution', {})
    if not dist:
        print("  [SKIP] fig4: no n_projects_distribution")
        return

    n_projects = sorted(int(k) for k in dist.keys())
    counts = [dist[str(k)] for k in n_projects]

    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['#4C72B0' if n == 1 else '#DD8452' if n <= 3 else '#C44E52'
              for n in n_projects]
    bars = ax.bar(range(len(n_projects)), counts, color=colors,
                  edgecolor='black', linewidth=0.3)

    ax.set_xticks(range(len(n_projects)))
    ax.set_xticklabels([str(n) for n in n_projects])
    ax.set_xlabel('Number of Projects Appearing In')
    ax.set_ylabel('Number of Addresses (log scale)')
    ax.set_title('Serial Sybil Distribution: Cross-Project Recurrence')
    ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)

    # Annotate key statistics
    summary = data.get('summary', {})
    serial_2 = summary.get('serial_2plus_pct', 0)
    serial_5 = summary.get('serial_5plus_pct', 0)
    ax.annotate(f'{serial_2:.1f}% appear in 2+ projects\n{serial_5:.1f}% appear in 5+ projects',
                xy=(0.95, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', edgecolor='gray'))

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_serial_sybil.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig5_boundary_analysis(boundary_data):
    """Chart showing non-sybil percentile at 80% threshold for each indicator."""
    if boundary_data is None:
        print("  [SKIP] fig5: boundary_analysis_results.json missing")
        return

    agg = boundary_data.get('aggregate', {})
    if not agg:
        print("  [SKIP] fig5: no aggregate data")
        return

    safe_zone = agg.get('safe_zone_avg_percentile', {})
    within_20 = agg.get('per_indicator_within_20pct', {})

    indicators = ['BT', 'BW', 'HF', 'RF', 'MA']
    percentiles = [safe_zone.get(ind, 0) for ind in indicators]
    n_close = [within_20.get(ind, 0) for ind in indicators]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Safe zone percentile
    colors1 = [palette[i] for i in range(len(indicators))]
    bars1 = ax1.bar(range(len(indicators)), percentiles, color=colors1,
                    edgecolor='black', linewidth=0.5, width=0.6)
    ax1.set_xticks(range(len(indicators)))
    ax1.set_xticklabels(indicators)
    ax1.set_ylabel('Average Percentile (%)')
    ax1.set_title('Non-Sybil Safe Zone\n(Higher = Better Separation)')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(90, 101)
    for bar, val in zip(bars1, percentiles):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10)

    # Right: Number of non-sybils within 20% of threshold
    bars2 = ax2.bar(range(len(indicators)), n_close, color=colors1,
                    edgecolor='black', linewidth=0.5, width=0.6)
    ax2.set_xticks(range(len(indicators)))
    ax2.set_xticklabels(indicators)
    ax2.set_ylabel('Non-Sybils Near Threshold')
    ax2.set_title('Non-Sybils Within 20% of Threshold\n(Lower = Fewer False Positives)')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')
    for bar, val in zip(bars2, n_close):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.1,
                f'{val:,}', ha='center', va='bottom', fontsize=9)

    plt.suptitle('Boundary Analysis: Threshold Safety Margins', fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig5_boundary_analysis.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig6_project_diversity(dist_data):
    """Heatmap of project pairwise sybil rate diversity."""
    if dist_data is None:
        print("  [SKIP] fig6: real_distribution_results.json missing")
        return

    per_project = dist_data.get('per_project', [])
    if not per_project:
        print("  [SKIP] fig6: no per_project data")
        return

    # Extract project names and trigger rates
    projects = []
    trigger_rates = []  # shape: (n_projects, 5) for BT, BW, HF, RF, MA

    indicators = ['BT', 'BW', 'HF', 'RF', 'MA']

    for proj in per_project:
        pname = proj.get('project', 'unknown')
        projects.append(pname)

        rates = []
        trigger = proj.get('indicator_trigger_rates_among_sybils', {})
        for ind in indicators:
            rates.append(trigger.get(ind, 0))
        trigger_rates.append(rates)

    trigger_rates = np.array(trigger_rates)
    n = len(projects)

    # Compute pairwise Jensen-Shannon divergence of trigger rate vectors
    from scipy.spatial.distance import jensenshannon
    jsd_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # Add small epsilon to avoid zeros
            p = trigger_rates[i] + 1e-10
            q = trigger_rates[j] + 1e-10
            p = p / p.sum()
            q = q / q.sum()
            jsd_matrix[i, j] = jensenshannon(p, q)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(jsd_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=projects, yticklabels=projects,
                ax=ax, linewidths=0.3, linecolor='white',
                cbar_kws={'label': 'Jensen-Shannon Divergence'},
                annot_kws={'fontsize': 6})

    ax.set_title('Project Pairwise Diversity\n(JSD of Sybil Indicator Trigger Rate Distributions)')
    ax.set_xticklabels(projects, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(projects, fontsize=8)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig6_project_diversity.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


if __name__ == '__main__':
    print("Paper 3: Generating figures...")

    experiment = load_json('experiment_real_results.json')
    ai_features = load_json('real_ai_feature_distributions.json')
    distribution = load_json('real_distribution_results.json')
    serial = load_json('serial_sybil_results.json')
    cooccurrence = load_json('indicator_cooccurrence_results.json')
    boundary = load_json('boundary_analysis_results.json')

    fig1_evasion_comparison(experiment)
    fig2_feature_importance(experiment)
    fig3_indicator_cooccurrence(cooccurrence)
    fig4_serial_sybil(serial)
    fig5_boundary_analysis(boundary)
    fig6_project_diversity(distribution)

    print("Paper 3: Done.")
