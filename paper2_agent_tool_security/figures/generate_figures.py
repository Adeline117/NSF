#!/usr/bin/env python3
"""Generate publication-quality figures for Paper 2: Agent Tool Security."""

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
DYN_DIR = os.path.join(PAPER_DIR, 'dynamic_testing')
FIG_DIR = SCRIPT_DIR

palette = sns.color_palette("Set2", 10)


def load_json(filepath):
    if not os.path.exists(filepath):
        print(f"  [SKIP] {filepath} not found")
        return None
    with open(filepath) as f:
        return json.load(f)


def fig1_vulnerability_distribution(data):
    """Horizontal bar chart of finding counts by category."""
    if data is None:
        print("  [SKIP] fig1: unified_web3_results.json missing")
        return

    by_cat = data.get('static_scan_summary', {}).get('by_category', {})
    if not by_cat:
        print("  [SKIP] fig1: no by_category data")
        return

    sorted_cats = sorted(by_cat.items(), key=lambda x: x[1], reverse=False)
    categories = [c.replace('_', ' ').title() for c, v in sorted_cats]
    counts = [v for c, v in sorted_cats]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("tab10", len(categories))
    y_pos = np.arange(len(categories))
    ax.barh(y_pos, counts, color=colors, edgecolor='black', linewidth=0.3, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(categories, fontsize=9)
    ax.set_xlabel('Number of Findings')
    ax.set_title(f'Static Scan: Vulnerability Distribution by Category (N={sum(counts)})')
    ax.grid(axis='x', alpha=0.3)

    for i, v in enumerate(counts):
        ax.text(v + max(counts)*0.01, i, str(v), va='center', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_vulnerability_distribution.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig2_severity_breakdown(data):
    """Stacked bar chart: per-protocol findings by severity."""
    if data is None:
        print("  [SKIP] fig2: unified_web3_results.json missing")
        return

    by_protocol = data.get('static_scan_summary', {}).get('by_protocol', {})
    if not by_protocol:
        print("  [SKIP] fig2: no by_protocol data")
        return

    protocols = []
    critical_vals = []
    high_vals = []
    medium_vals = []

    for proto, info in sorted(by_protocol.items()):
        sev = info.get('by_severity', {})
        protocols.append(proto.replace('_', ' ').title())
        critical_vals.append(sev.get('critical', 0))
        high_vals.append(sev.get('high', 0))
        medium_vals.append(sev.get('medium', 0))

    x = np.arange(len(protocols))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x, critical_vals, width, label='Critical', color='#d62728', edgecolor='black', linewidth=0.3)
    ax.bar(x, high_vals, width, bottom=critical_vals, label='High',
           color='#ff7f0e', edgecolor='black', linewidth=0.3)
    bottoms = [c + h for c, h in zip(critical_vals, high_vals)]
    ax.bar(x, medium_vals, width, bottom=bottoms, label='Medium',
           color='#ffbb78', edgecolor='black', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(protocols, fontsize=10)
    ax.set_ylabel('Number of Findings')
    ax.set_title('Severity Breakdown by Protocol Type')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)

    # Add total labels
    totals = [c + h + m for c, h, m in zip(critical_vals, high_vals, medium_vals)]
    for i, total in enumerate(totals):
        ax.text(i, total + max(totals)*0.02, str(total), ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_severity_breakdown.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig3_dynamic_test_results(data):
    """Bar chart of 5 attack vector success rates."""
    if data is None:
        print("  [SKIP] fig3: unified_web3_results.json missing")
        return

    attack_vectors = data.get('dynamic_test_summary', {}).get('attack_vectors', {})
    if not attack_vectors:
        print("  [SKIP] fig3: no attack_vectors data")
        return

    vectors = []
    rates = []
    n_vulns = []

    display_names = {
        'tool_poisoning': 'Tool\nPoisoning',
        'prompt_injection_output': 'Prompt Injection\n(Output)',
        'parameter_injection': 'Parameter\nInjection',
        'tx_validation': 'Transaction\nValidation',
        'private_key_handling': 'Private Key\nHandling',
    }

    for vec_name in ['tool_poisoning', 'prompt_injection_output', 'parameter_injection',
                      'tx_validation', 'private_key_handling']:
        info = attack_vectors.get(vec_name, {})
        vectors.append(display_names.get(vec_name, vec_name))
        rates.append(info.get('rate_pct', 0))
        n_vulns.append(info.get('vulnerable', 0))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#d62728' if r >= 50 else '#ff7f0e' if r >= 10 else '#2ca02c' for r in rates]
    bars = ax.bar(range(len(vectors)), rates, color=colors,
                  edgecolor='black', linewidth=0.5, width=0.6)

    ax.set_xticks(range(len(vectors)))
    ax.set_xticklabels(vectors, fontsize=9)
    ax.set_ylabel('Vulnerability Rate (%)')
    ax.set_title('Dynamic Testing: Attack Vector Success Rates')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 115)

    for bar, rate, vuln in zip(bars, rates, n_vulns):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{rate:.1f}%\n({vuln})', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_dynamic_test_results.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


def fig4_risk_scores(data):
    """Bar chart of per-repo risk scores (top 20)."""
    if data is None:
        print("  [SKIP] fig4: unified_web3_results.json missing")
        return

    risk_ranking = data.get('static_scan_summary', {}).get('risk_ranking', [])
    if not risk_ranking:
        print("  [SKIP] fig4: no risk_ranking data")
        return

    # Take top 20 repos
    top_repos = risk_ranking[:20]
    repo_names = [r['repo'].split('/')[-1][:20] for r in top_repos]
    scores = [r['risk_score'] for r in top_repos]
    ratings = [r.get('risk_rating', 'unknown') for r in top_repos]

    color_map = {
        'critical': '#d62728',
        'high': '#ff7f0e',
        'medium': '#ffbb78',
        'low': '#2ca02c',
        'safe': '#98df8a',
    }
    colors = [color_map.get(r, '#999999') for r in ratings]

    fig, ax = plt.subplots(figsize=(10, 7))
    y_pos = np.arange(len(repo_names))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor='black', linewidth=0.3, height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(repo_names, fontsize=8)
    ax.set_xlabel('Risk Score (0-100)')
    ax.set_title('Top 20 Repositories by Risk Score')
    ax.grid(axis='x', alpha=0.3)
    ax.invert_yaxis()

    for i, (score, rating) in enumerate(zip(scores, ratings)):
        ax.text(score + 1, i, f'{score:.0f} ({rating})', va='center', fontsize=8)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map['critical'], edgecolor='black', label='Critical'),
        Patch(facecolor=color_map['high'], edgecolor='black', label='High'),
        Patch(facecolor=color_map['medium'], edgecolor='black', label='Medium'),
        Patch(facecolor=color_map['low'], edgecolor='black', label='Low'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_risk_scores.pdf')
    plt.savefig(path)
    plt.close()
    print(f"  Saved {path}")


if __name__ == '__main__':
    print("Paper 2: Generating figures...")

    unified = load_json(os.path.join(EXP_DIR, 'unified_web3_results.json'))
    dynamic = load_json(os.path.join(DYN_DIR, 'dynamic_test_results.json'))

    fig1_vulnerability_distribution(unified)
    fig2_severity_breakdown(unified)

    # Use unified data (which contains dynamic_test_summary) for fig3
    fig3_dynamic_test_results(unified)
    fig4_risk_scores(unified)

    print("Paper 2: Done.")
