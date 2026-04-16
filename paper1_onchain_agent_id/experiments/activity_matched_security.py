#!/usr/bin/env python
"""
Paper 1: Activity-Matched Security Comparison
===============================================
The security reversal (humans 66.1% vs bots 49.1% unlimited approvals)
might be an activity-level artifact: humans with fewer transactions
have fewer chances to approve, so the *rate* is inflated by small
denominators.

Fix: match each agent address to the human address with the closest
total transaction count (nearest-neighbor matching on activity), then
compare unlimited_approve_ratio within matched pairs.

Also includes propensity score matching as a robustness check.

Outputs:
  experiments/activity_matched_security_results.json
"""

import json
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PARQUET = DATA_DIR / "features_provenance_v4.parquet"
RESULTS_PATH = PROJECT_ROOT / "experiments" / "activity_matched_security_results.json"

# 23 features used in the main pipeline
FEATURE_GROUPS = {
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

ALL_FEATURES = []
for group_features in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(group_features)


def load_data():
    """Load n=1,147 with features + labels."""
    df = pd.read_parquet(FEATURES_PARQUET)

    feature_cols = [c for c in ALL_FEATURES if c in df.columns]
    X = df[feature_cols].copy()
    y = df["label"].values.astype(int)
    categories = df["category"].copy()
    n_transactions = df["n_transactions"].values.astype(float)

    # Impute NaN
    for col in X.columns:
        if X[col].isna().any():
            X[col] = X[col].fillna(X[col].median())

    return df, X, y, categories, n_transactions, feature_cols


def nearest_neighbor_matching(agents_df, humans_df, match_col="n_transactions"):
    """For each agent, find the human with the closest value on match_col.

    Uses 1:1 matching without replacement.
    Returns matched pairs as a list of (agent_idx, human_idx) tuples.
    """
    agent_vals = agents_df[match_col].values
    human_vals = humans_df[match_col].values.copy()
    human_indices = list(humans_df.index)

    matched_pairs = []
    used_humans = set()

    # Sort agents by their match value for more stable matching
    agent_order = np.argsort(agent_vals)

    for ai in agent_order:
        agent_idx = agents_df.index[ai]
        agent_val = agent_vals[ai]

        best_dist = np.inf
        best_hi = None

        for hi_pos, hi in enumerate(human_indices):
            if hi in used_humans:
                continue
            dist = abs(agent_val - human_vals[hi_pos])
            if dist < best_dist:
                best_dist = dist
                best_hi = hi

        if best_hi is not None:
            matched_pairs.append((agent_idx, best_hi))
            used_humans.add(best_hi)

    return matched_pairs


def propensity_score_matching(df, y, feature_cols, caliper=0.2):
    """Propensity score matching using logistic regression on all features.

    Estimates P(agent | X), then matches each agent to the nearest-propensity
    human within a caliper of 0.2 * std(propensity score).

    Returns matched pairs.
    """
    X_vals = df[feature_cols].values.astype(float)

    # Handle NaN
    nan_mask = np.isnan(X_vals)
    if nan_mask.any():
        col_medians = np.nanmedian(X_vals, axis=0)
        for j in range(X_vals.shape[1]):
            X_vals[nan_mask[:, j], j] = col_medians[j]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_vals)

    lr = LogisticRegression(C=1.0, max_iter=5000, random_state=42)
    lr.fit(X_scaled, y)
    propensity = lr.predict_proba(X_scaled)[:, 1]

    ps_std = propensity.std()
    caliper_val = caliper * ps_std

    agent_mask = y == 1
    human_mask = y == 0

    agent_indices = np.where(agent_mask)[0]
    human_indices = np.where(human_mask)[0]

    agent_ps = propensity[agent_indices]
    human_ps = propensity[human_indices]

    matched_pairs = []
    used_humans = set()

    for ai, aps in zip(agent_indices, agent_ps):
        best_dist = np.inf
        best_hi = None
        for hi_pos, (hi, hps) in enumerate(zip(human_indices, human_ps)):
            if hi in used_humans:
                continue
            dist = abs(aps - hps)
            if dist < best_dist and dist <= caliper_val:
                best_dist = dist
                best_hi = hi

        if best_hi is not None:
            matched_pairs.append((int(ai), int(best_hi)))
            used_humans.add(best_hi)

    return matched_pairs, propensity


def compare_matched_pairs(df, matched_pairs, metric_cols):
    """Compare metric values between matched agent-human pairs."""
    agent_vals = {col: [] for col in metric_cols}
    human_vals = {col: [] for col in metric_cols}

    addresses = list(df.index)

    for agent_idx, human_idx in matched_pairs:
        for col in metric_cols:
            # Handle both positional (int) and label-based indices
            if isinstance(agent_idx, (int, np.integer)) and not isinstance(agent_idx, str):
                av = df.iloc[agent_idx][col] if agent_idx < len(df) else df.loc[agent_idx][col]
                hv = df.iloc[human_idx][col] if human_idx < len(df) else df.loc[human_idx][col]
            else:
                av = df.loc[agent_idx][col]
                hv = df.loc[human_idx][col]
            agent_vals[col].append(float(av) if not pd.isna(av) else 0.0)
            human_vals[col].append(float(hv) if not pd.isna(hv) else 0.0)

    results = {}
    for col in metric_cols:
        a = np.array(agent_vals[col])
        h = np.array(human_vals[col])

        a_mean = float(np.mean(a))
        h_mean = float(np.mean(h))
        a_std = float(np.std(a))
        h_std = float(np.std(h))

        # Paired test (Wilcoxon signed-rank)
        try:
            stat, p_wilcoxon = scipy_stats.wilcoxon(a, h, alternative="two-sided")
        except ValueError:
            stat, p_wilcoxon = float("nan"), 1.0

        # Unpaired test (Mann-Whitney)
        try:
            u, p_mw = scipy_stats.mannwhitneyu(a, h, alternative="two-sided")
        except ValueError:
            u, p_mw = float("nan"), 1.0

        # Cohen's d (paired)
        diff = a - h
        d_paired = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0

        # Effect direction
        if a_mean > h_mean:
            direction = "agent > human"
        elif a_mean < h_mean:
            direction = "human > agent"
        else:
            direction = "equal"

        results[col] = {
            "agent_mean": round(a_mean, 4),
            "agent_std": round(a_std, 4),
            "human_mean": round(h_mean, 4),
            "human_std": round(h_std, 4),
            "mean_difference": round(float(a_mean - h_mean), 4),
            "direction": direction,
            "cohens_d_paired": round(d_paired, 4),
            "wilcoxon_stat": round(float(stat), 4) if not np.isnan(stat) else None,
            "wilcoxon_p": round(float(p_wilcoxon), 6),
            "mann_whitney_U": round(float(u), 4) if not np.isnan(u) else None,
            "mann_whitney_p": round(float(p_mw), 6),
            "significant_wilcoxon_0.05": bool(p_wilcoxon < 0.05),
            "significant_wilcoxon_0.01": bool(p_wilcoxon < 0.01),
        }

    return results


def main():
    print("=" * 80)
    print("Paper 1: Activity-Matched Security Comparison")
    print("=" * 80)

    df, X, y, categories, n_transactions, feature_cols = load_data()

    n_agents = int((y == 1).sum())
    n_humans = int((y == 0).sum())
    print(f"Dataset: n={len(df)}, agents={n_agents}, humans={n_humans}")
    print(f"Features: {len(feature_cols)}")

    # Security-relevant metrics to compare
    security_metrics = [
        "unlimited_approve_ratio",
        "approve_revoke_ratio",
        "unverified_contract_approve_ratio",
        "multi_protocol_interaction_count",
        "flash_loan_usage",
    ]

    # ── UNMATCHED comparison (baseline) ───────────────────────────────
    print("\n--- UNMATCHED comparison (baseline) ---")
    agents_df = df[y == 1]
    humans_df = df[y == 0]

    unmatched_results = {}
    for col in security_metrics:
        a = agents_df[col].fillna(0).values
        h = humans_df[col].fillna(0).values
        a_mean = float(np.mean(a))
        h_mean = float(np.mean(h))
        try:
            u, p = scipy_stats.mannwhitneyu(a, h, alternative="two-sided")
        except ValueError:
            u, p = float("nan"), 1.0

        unmatched_results[col] = {
            "agent_mean": round(a_mean, 4),
            "human_mean": round(h_mean, 4),
            "direction": "agent > human" if a_mean > h_mean else "human > agent",
            "mann_whitney_p": round(float(p), 6),
        }
        print(f"  {col:<40} agent={a_mean:.4f}  human={h_mean:.4f}  "
              f"{'agent>human' if a_mean > h_mean else 'human>agent'}  p={p:.4f}")

    # Activity level comparison
    agent_tx = n_transactions[y == 1]
    human_tx = n_transactions[y == 0]
    print(f"\n  Activity levels:")
    print(f"    Agent tx count: mean={np.mean(agent_tx):.1f}, "
          f"median={np.median(agent_tx):.1f}")
    print(f"    Human tx count: mean={np.mean(human_tx):.1f}, "
          f"median={np.median(human_tx):.1f}")

    # ── NEAREST-NEIGHBOR MATCHING on transaction count ────────────────
    print("\n--- Nearest-Neighbor Matching (by n_transactions) ---")

    # Prepare DataFrames for matching
    agents_for_match = df[y == 1].copy()
    humans_for_match = df[y == 0].copy()

    matched_pairs_nn = nearest_neighbor_matching(
        agents_for_match, humans_for_match, match_col="n_transactions"
    )
    print(f"  Matched pairs: {len(matched_pairs_nn)}")

    # Verify matching quality
    matched_agent_tx = []
    matched_human_tx = []
    for agent_idx, human_idx in matched_pairs_nn:
        matched_agent_tx.append(df.loc[agent_idx, "n_transactions"])
        matched_human_tx.append(df.loc[human_idx, "n_transactions"])
    matched_agent_tx = np.array(matched_agent_tx)
    matched_human_tx = np.array(matched_human_tx)
    tx_diffs = np.abs(matched_agent_tx - matched_human_tx)

    print(f"  Matching quality (|tx_agent - tx_human|):")
    print(f"    Mean: {np.mean(tx_diffs):.1f}")
    print(f"    Median: {np.median(tx_diffs):.1f}")
    print(f"    Max: {np.max(tx_diffs):.1f}")
    print(f"  Matched agent tx: mean={np.mean(matched_agent_tx):.1f}")
    print(f"  Matched human tx: mean={np.mean(matched_human_tx):.1f}")

    # Compare security metrics on matched pairs
    # Build a matched DataFrame for comparison
    matched_agent_rows = []
    matched_human_rows = []
    for agent_idx, human_idx in matched_pairs_nn:
        matched_agent_rows.append(df.loc[agent_idx])
        matched_human_rows.append(df.loc[human_idx])

    matched_agents_df = pd.DataFrame(matched_agent_rows)
    matched_humans_df = pd.DataFrame(matched_human_rows)

    print("\n  Security metrics after activity matching:")
    nn_results = {}
    for col in security_metrics:
        a = matched_agents_df[col].fillna(0).values
        h = matched_humans_df[col].fillna(0).values
        a_mean = float(np.mean(a))
        h_mean = float(np.mean(h))

        # Paired Wilcoxon signed-rank test
        try:
            stat, p_w = scipy_stats.wilcoxon(a, h, alternative="two-sided")
        except ValueError:
            stat, p_w = float("nan"), 1.0

        # Unpaired Mann-Whitney
        try:
            u, p_mw = scipy_stats.mannwhitneyu(a, h, alternative="two-sided")
        except ValueError:
            u, p_mw = float("nan"), 1.0

        # Cohen's d (paired)
        diff = a - h
        d_paired = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0

        direction = "agent > human" if a_mean > h_mean else (
            "human > agent" if a_mean < h_mean else "equal"
        )

        nn_results[col] = {
            "agent_mean": round(a_mean, 4),
            "agent_std": round(float(np.std(a)), 4),
            "human_mean": round(h_mean, 4),
            "human_std": round(float(np.std(h)), 4),
            "mean_difference": round(a_mean - h_mean, 4),
            "direction": direction,
            "cohens_d_paired": round(d_paired, 4),
            "wilcoxon_p": round(float(p_w), 6),
            "mann_whitney_p": round(float(p_mw), 6),
            "significant_0.05": bool(p_w < 0.05),
        }

        sig = " ***" if p_w < 0.01 else (" *" if p_w < 0.05 else "")
        print(f"    {col:<40} agent={a_mean:.4f}  human={h_mean:.4f}  "
              f"d={d_paired:+.3f}  p_wilcoxon={p_w:.4f}{sig}")

    # ── PROPENSITY SCORE MATCHING ─────────────────────────────────────
    print("\n--- Propensity Score Matching (robustness check) ---")

    matched_pairs_ps, propensity = propensity_score_matching(
        df, y, feature_cols, caliper=0.25
    )
    print(f"  Matched pairs (PSM): {len(matched_pairs_ps)}")

    if len(matched_pairs_ps) > 10:
        # Compare security metrics
        print("\n  Security metrics after propensity matching:")
        ps_results = {}
        for col in security_metrics:
            a_vals = [float(df.iloc[ai][col]) if not pd.isna(df.iloc[ai][col]) else 0.0
                      for ai, _ in matched_pairs_ps]
            h_vals = [float(df.iloc[hi][col]) if not pd.isna(df.iloc[hi][col]) else 0.0
                      for _, hi in matched_pairs_ps]
            a = np.array(a_vals)
            h = np.array(h_vals)

            a_mean = float(np.mean(a))
            h_mean = float(np.mean(h))

            try:
                stat, p_w = scipy_stats.wilcoxon(a, h, alternative="two-sided")
            except ValueError:
                stat, p_w = float("nan"), 1.0

            diff = a - h
            d_paired = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0

            direction = "agent > human" if a_mean > h_mean else (
                "human > agent" if a_mean < h_mean else "equal"
            )

            ps_results[col] = {
                "agent_mean": round(a_mean, 4),
                "human_mean": round(h_mean, 4),
                "mean_difference": round(a_mean - h_mean, 4),
                "direction": direction,
                "cohens_d_paired": round(d_paired, 4),
                "wilcoxon_p": round(float(p_w), 6),
                "significant_0.05": bool(p_w < 0.05),
            }

            sig = " ***" if p_w < 0.01 else (" *" if p_w < 0.05 else "")
            print(f"    {col:<40} agent={a_mean:.4f}  human={h_mean:.4f}  "
                  f"d={d_paired:+.3f}  p={p_w:.4f}{sig}")
    else:
        ps_results = {"note": f"Too few matched pairs ({len(matched_pairs_ps)})"}
        print("  Too few matched pairs for PSM analysis")

    # ── DIAGNOSIS ─────────────────────────────────────────────────────
    # Check if the unlimited_approve_ratio reversal persists
    nn_approve = nn_results.get("unlimited_approve_ratio", {})
    unmatched_approve = unmatched_results.get("unlimited_approve_ratio", {})

    reversal_persists_nn = (
        nn_approve.get("direction", "") == unmatched_approve.get("direction", "")
    )

    if reversal_persists_nn and nn_approve.get("significant_0.05", False):
        diagnosis = (
            "REVERSAL IS REAL: After activity matching, the approval-rate "
            "direction persists AND is statistically significant. The security "
            "behavior difference is not an artifact of transaction-count "
            "imbalance."
        )
    elif reversal_persists_nn:
        diagnosis = (
            "REVERSAL PERSISTS BUT NOT SIGNIFICANT: Direction unchanged after "
            "matching, but the paired test is not significant (p > 0.05). "
            "The effect may be real but weak, or the matched sample is too "
            "small for power."
        )
    else:
        diagnosis = (
            "REVERSAL IS AN ARTIFACT: After activity matching, the direction "
            "flips or disappears. The original human>agent unlimited approval "
            "finding was driven by activity-level confounding."
        )

    print(f"\n{'='*80}")
    print("DIAGNOSIS")
    print("=" * 80)
    print(diagnosis)

    # ── Assemble results ──────────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Activity-matched security comparison: tests whether the "
            "human>agent unlimited_approve_ratio reversal survives after "
            "matching by transaction count (nearest-neighbor) and propensity "
            "score matching."
        ),
        "dataset": {
            "n_total": int(len(df)),
            "n_agents": n_agents,
            "n_humans": n_humans,
            "n_features": len(feature_cols),
        },
        "activity_levels": {
            "agent_tx_mean": round(float(np.mean(agent_tx)), 1),
            "agent_tx_median": round(float(np.median(agent_tx)), 1),
            "human_tx_mean": round(float(np.mean(human_tx)), 1),
            "human_tx_median": round(float(np.median(human_tx)), 1),
        },
        "unmatched_comparison": unmatched_results,
        "nn_matching": {
            "method": "Nearest-neighbor on n_transactions (1:1 without replacement)",
            "n_matched_pairs": len(matched_pairs_nn),
            "matching_quality": {
                "tx_diff_mean": round(float(np.mean(tx_diffs)), 1),
                "tx_diff_median": round(float(np.median(tx_diffs)), 1),
                "tx_diff_max": round(float(np.max(tx_diffs)), 1),
                "matched_agent_tx_mean": round(float(np.mean(matched_agent_tx)), 1),
                "matched_human_tx_mean": round(float(np.mean(matched_human_tx)), 1),
            },
            "security_comparison": nn_results,
        },
        "propensity_score_matching": {
            "method": (
                "Logistic regression on all 23 features, caliper=0.25*std(PS), "
                "1:1 without replacement"
            ),
            "n_matched_pairs": len(matched_pairs_ps),
            "security_comparison": ps_results,
        },
        "diagnosis": diagnosis,
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
