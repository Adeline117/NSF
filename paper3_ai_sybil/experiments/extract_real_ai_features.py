"""
Paper 3 -- Extract Real AI-Agent Features from Paper 1 Raw Transaction Data
===========================================================================
Extracts the 8 AI-specific features from REAL on-chain transaction data
for all labeled addresses in Paper 1.  Replaces the synthetic (rng.beta)
calibration values with empirical distributions.

For each address we load raw Etherscan transaction data and compute:
  1. gas_price_precision      -- fraction of gas prices NOT round numbers
  2. hour_entropy              -- Shannon entropy of hour-of-day distribution
  3. behavioral_consistency    -- 1/(1+CV) of transaction intervals
  4. action_sequence_perplexity-- Shannon entropy of method-ID distribution
  5. error_recovery_pattern    -- fraction of failed transactions (revert rate)
  6. response_latency_variance -- CV of transaction intervals
  7. gas_nonce_gap_regularity  -- fraction of nonce increments == 1
  8. eip1559_tip_precision     -- gas-price trailing-digit precision proxy

Statistical tests:
  - Mann-Whitney U (two-sided) for each feature: agent vs human

Outputs:
  - paper3_ai_sybil/experiments/real_ai_feature_distributions.json
  - paper3_ai_sybil/experiments/real_ai_features.json  (legacy compat)
  - Console summary table

Usage:
    python3 paper3_ai_sybil/experiments/extract_real_ai_features.py
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "raw"
FEATURES_FILE = PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features.parquet"
EXPANDED_FILE = PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_expanded.parquet"
OUTPUT_FILE = SCRIPT_DIR / "real_ai_features.json"
DISTRIBUTIONS_FILE = SCRIPT_DIR / "real_ai_feature_distributions.json"

# The 8 AI-specific features
AI_FEATURE_NAMES = [
    "gas_price_precision",
    "hour_entropy",
    "behavioral_consistency",
    "action_sequence_perplexity",
    "error_recovery_pattern",
    "response_latency_variance",
    "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]


# ============================================================
# FEATURE EXTRACTION
# ============================================================


def _trailing_zeros(x: int) -> int:
    """Count trailing zeros in an integer."""
    if x == 0:
        return 0
    s = str(x)
    return len(s) - len(s.rstrip("0"))


def _shannon_entropy_bits(counts: np.ndarray) -> float:
    """Shannon entropy in bits from a count array."""
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def extract_ai_features(df: pd.DataFrame) -> dict:
    """Extract the 8 AI-specific features from real transaction data.

    Args:
        df: DataFrame of raw Etherscan transactions for one address.

    Returns:
        Dictionary of feature values, or None if data is insufficient.
    """
    if len(df) < 5:
        return None

    features = {}
    features["n_transactions"] = len(df)

    # ----------------------------------------------------------
    # 1. gas_price_precision
    #    Fraction of gas prices that are NOT round numbers.
    #    "Round" = has >= 9 trailing zeros in wei representation.
    # ----------------------------------------------------------
    gas_prices = pd.to_numeric(df.get("gasPrice", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(gas_prices) > 0:
        gas_ints = gas_prices.astype(np.int64)
        trailing = gas_ints.apply(lambda x: _trailing_zeros(int(x)))
        frac_round = (trailing >= 9).mean()
        features["gas_price_precision"] = float(1.0 - frac_round)
    else:
        features["gas_price_precision"] = 0.5

    # ----------------------------------------------------------
    # 2. hour_entropy
    #    Shannon entropy of hour-of-day distribution (bits).
    #    Max = log2(24) = 4.585.  Agents -> high, humans -> low.
    # ----------------------------------------------------------
    timestamps = pd.to_numeric(df.get("timeStamp", pd.Series(dtype=float)), errors="coerce").dropna()
    if len(timestamps) > 0:
        hours = pd.to_datetime(timestamps.astype(int), unit="s").dt.hour
        hour_counts = np.bincount(hours.values, minlength=24)
        features["hour_entropy"] = _shannon_entropy_bits(hour_counts)
    else:
        features["hour_entropy"] = 0.0

    # ----------------------------------------------------------
    # 3. behavioral_consistency
    #    1 / (1 + CV) of transaction intervals.
    #    High = regular intervals (agent-like).
    # ----------------------------------------------------------
    intervals = np.array([], dtype=float)
    if len(timestamps) > 1:
        ts_sorted = timestamps.astype(int).sort_values().values
        intervals = np.diff(ts_sorted).astype(float)
        intervals = intervals[intervals > 0]

    if len(intervals) > 1 and intervals.mean() > 0:
        cv = intervals.std() / intervals.mean()
        features["behavioral_consistency"] = float(1.0 / (1.0 + cv))
    else:
        features["behavioral_consistency"] = 0.5

    # ----------------------------------------------------------
    # 4. action_sequence_perplexity
    #    Shannon entropy of method-ID distribution (bits).
    #    Higher = more diverse function calls = more adaptive.
    # ----------------------------------------------------------
    if "input" in df.columns:
        method_ids = df["input"].apply(
            lambda x: x[:10] if isinstance(x, str) and len(x) >= 10 else "0x"
        )
        mid_counts = method_ids.value_counts()
        mid_probs = mid_counts.values / mid_counts.values.sum()
        mid_probs = mid_probs[mid_probs > 0]
        features["action_sequence_perplexity"] = float(-np.sum(mid_probs * np.log2(mid_probs)))
    else:
        features["action_sequence_perplexity"] = 0.0

    # ----------------------------------------------------------
    # 5. error_recovery_pattern
    #    Fraction of failed (reverted) transactions.
    #    Agents may have higher revert rates from competitive strategies
    #    (MEV, frontrunning) or may have lower rates from programmatic
    #    pre-checks.  Key: it's a discriminating signal.
    # ----------------------------------------------------------
    if "isError" in df.columns:
        errors = pd.to_numeric(df["isError"], errors="coerce").fillna(0).astype(int)
        features["error_recovery_pattern"] = float(errors.mean())
    else:
        features["error_recovery_pattern"] = 0.0

    # ----------------------------------------------------------
    # 6. response_latency_variance
    #    std(intervals) / mean(intervals) = coefficient of variation.
    #    Agents: lower CV; humans: higher CV.
    # ----------------------------------------------------------
    if len(intervals) > 1 and intervals.mean() > 0:
        features["response_latency_variance"] = float(
            intervals.std() / (intervals.mean() + 1e-10)
        )
    else:
        features["response_latency_variance"] = 1.0

    # ----------------------------------------------------------
    # 7. gas_nonce_gap_regularity
    #    Fraction of nonce increments == 1.
    #    Agents: near 1.0 (programmatic nonce management).
    # ----------------------------------------------------------
    if "nonce" in df.columns:
        nonces = pd.to_numeric(df["nonce"], errors="coerce").dropna()
        if len(nonces) > 1:
            nonce_sorted = nonces.astype(int).sort_values().values
            nonce_diffs = np.diff(nonce_sorted)
            features["gas_nonce_gap_regularity"] = float(
                (nonce_diffs == 1).mean()
            ) if len(nonce_diffs) > 0 else 0.5
        else:
            features["gas_nonce_gap_regularity"] = 0.5
    else:
        features["gas_nonce_gap_regularity"] = 0.5

    # ----------------------------------------------------------
    # 8. eip1559_tip_precision
    #    Fraction of gasPrice with < 6 trailing zeros
    #    (sub-Gwei precision proxy for maxPriorityFeePerGas).
    # ----------------------------------------------------------
    if len(gas_prices) > 0:
        gas_ints = gas_prices.astype(np.int64)
        trailing = gas_ints.apply(lambda x: _trailing_zeros(int(x)))
        features["eip1559_tip_precision"] = float((trailing < 6).mean())
    else:
        features["eip1559_tip_precision"] = 0.5

    # Bonus: gas_price_cv for cross-reference
    if len(gas_prices) > 1 and gas_prices.mean() > 0:
        features["gas_price_cv"] = float(gas_prices.std() / gas_prices.mean())
    else:
        features["gas_price_cv"] = 0.0

    # Bonus: hour_entropy_normalized
    if features["hour_entropy"] > 0:
        features["hour_entropy_normalized"] = features["hour_entropy"] / np.log2(24)
    else:
        features["hour_entropy_normalized"] = 0.0

    return features


# ============================================================
# DISTRIBUTION STATISTICS
# ============================================================


def compute_distribution_stats(values: list[float]) -> dict:
    """Compute summary statistics for a list of values."""
    arr = np.array(values)
    if len(arr) == 0:
        return {"count": 0, "mean": 0, "std": 0, "min": 0, "max": 0,
                "p25": 0, "median": 0, "p75": 0, "p90": 0}
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def fit_beta_params(values: list[float]) -> dict:
    """Fit Beta distribution parameters to [0,1]-bounded values.

    Returns alpha, beta for np.random.beta(alpha, beta, n) calibration.
    """
    arr = np.array(values, dtype=float)
    arr = np.clip(arr, 0.001, 0.999)
    if len(arr) < 3:
        return {"alpha": 2.0, "beta": 2.0, "method": "fallback"}

    mean = arr.mean()
    var = arr.var()
    if var <= 0 or mean <= 0 or mean >= 1:
        return {"alpha": 2.0, "beta": 2.0, "method": "fallback"}

    common = mean * (1 - mean) / var - 1
    if common <= 0:
        return {"alpha": 2.0, "beta": 2.0, "method": "fallback"}
    alpha = float(mean * common)
    beta = float((1 - mean) * common)
    alpha = max(0.1, min(100.0, alpha))
    beta = max(0.1, min(100.0, beta))
    return {"alpha": alpha, "beta": beta, "method": "moments"}


def mann_whitney_test(agent_vals: list, human_vals: list) -> dict:
    """Run Mann-Whitney U test between agent and human distributions."""
    if len(agent_vals) < 3 or len(human_vals) < 3:
        return {"U_statistic": None, "p_value": None, "significant": None,
                "note": "Too few samples"}
    try:
        u_stat, p_val = scipy_stats.mannwhitneyu(
            agent_vals, human_vals, alternative="two-sided"
        )
        return {
            "U_statistic": float(u_stat),
            "p_value": float(p_val),
            "significant_0.05": bool(p_val < 0.05),
            "significant_0.01": bool(p_val < 0.01),
            "effect_size_r": float(
                u_stat / (len(agent_vals) * len(human_vals))
            ),
        }
    except Exception as e:
        return {"U_statistic": None, "p_value": None, "error": str(e)}


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 80)
    print("EXTRACT REAL AI FEATURES FROM PAPER 1 RAW TRANSACTION DATA")
    print("  8 AI-Specific Features + Mann-Whitney Statistical Tests")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")

    # Try expanded first, fall back to original
    if EXPANDED_FILE.exists():
        labels_df = pd.read_parquet(EXPANDED_FILE)
        source_file = str(EXPANDED_FILE)
        print(f"Using expanded dataset: {EXPANDED_FILE.name}")
    elif FEATURES_FILE.exists():
        labels_df = pd.read_parquet(FEATURES_FILE)
        source_file = str(FEATURES_FILE)
        print(f"Using original dataset: {FEATURES_FILE.name}")
    else:
        print(f"ERROR: No features file found at {FEATURES_FILE} or {EXPANDED_FILE}")
        sys.exit(1)

    n_total = len(labels_df)
    n_agents = int((labels_df["label"] == 1).sum())
    n_humans = int((labels_df["label"] == 0).sum())
    print(f"Loaded {n_total} labeled addresses ({n_agents} agents, {n_humans} humans)")

    # ----------------------------------------------------------
    # Extract features for each address
    # ----------------------------------------------------------
    all_features = {}
    agent_features = {}
    human_features = {}

    for addr in labels_df.index:
        parquet_path = RAW_DATA_DIR / f"{addr}.parquet"
        if not parquet_path.exists():
            continue

        label = int(labels_df.loc[addr, "label"])
        name = str(labels_df.loc[addr, "name"]) if "name" in labels_df.columns else ""
        label_str = "AGENT" if label == 1 else "HUMAN"

        features = extract_ai_features(pd.read_parquet(parquet_path))
        if features is None:
            print(f"  SKIP: {addr[:12]}... (insufficient data)")
            continue

        features["label"] = label
        features["label_str"] = label_str
        features["name"] = name
        all_features[addr] = features

        if label == 1:
            agent_features[addr] = features
        else:
            human_features[addr] = features

    print(f"\nSuccessfully extracted features for {len(all_features)} addresses")
    print(f"  Agents: {len(agent_features)}, Humans: {len(human_features)}")

    # ----------------------------------------------------------
    # Console summary with statistical tests
    # ----------------------------------------------------------
    print("\n" + "=" * 120)
    print("FEATURE SUMMARY: AGENTS vs HUMANS (8 AI-Specific Features)")
    print("=" * 120)

    header = (f"{'Feature':<30} | {'Agent Mean':>10} {'Std':>8} "
              f"| {'Human Mean':>10} {'Std':>8} "
              f"| {'Cohen d':>8} | {'MW p-val':>10} {'Sig':>4}")
    print(header)
    print("-" * 120)

    statistical_tests = {}

    for feat in AI_FEATURE_NAMES:
        agent_vals = [f[feat] for f in agent_features.values() if feat in f]
        human_vals = [f[feat] for f in human_features.values() if feat in f]

        if agent_vals and human_vals:
            a_mean = np.mean(agent_vals)
            a_std = np.std(agent_vals)
            h_mean = np.mean(human_vals)
            h_std = np.std(human_vals)

            # Cohen's d
            pooled_std = np.sqrt((a_std ** 2 + h_std ** 2) / 2) if (a_std + h_std) > 0 else 1
            cohens_d = abs(a_mean - h_mean) / pooled_std if pooled_std > 0 else 0

            # Mann-Whitney test
            mw = mann_whitney_test(agent_vals, human_vals)
            p_val = mw.get("p_value")
            sig = "**" if (p_val and p_val < 0.01) else ("*" if (p_val and p_val < 0.05) else "ns")
            p_str = f"{p_val:.6f}" if p_val is not None else "N/A"

            statistical_tests[feat] = {
                "cohens_d": float(cohens_d),
                "mann_whitney": mw,
            }

            print(f"{feat:<30} | {a_mean:>10.4f} {a_std:>8.4f} "
                  f"| {h_mean:>10.4f} {h_std:>8.4f} "
                  f"| {cohens_d:>8.3f} | {p_str:>10} {sig:>4}")

    # ----------------------------------------------------------
    # Compute distributions and beta fits
    # ----------------------------------------------------------
    distributions = {"agent": {}, "human": {}}
    beta_params = {"agent": {}, "human": {}}

    for feat in AI_FEATURE_NAMES:
        agent_vals = [f[feat] for f in agent_features.values() if feat in f]
        human_vals = [f[feat] for f in human_features.values() if feat in f]

        distributions["agent"][feat] = compute_distribution_stats(agent_vals)
        distributions["human"][feat] = compute_distribution_stats(human_vals)

        # Beta fit for [0,1]-bounded features
        if feat not in ("hour_entropy", "action_sequence_perplexity",
                         "response_latency_variance"):
            beta_params["agent"][feat] = fit_beta_params(agent_vals)
            beta_params["human"][feat] = fit_beta_params(human_vals)

    # ----------------------------------------------------------
    # Calibration output for Paper 3 AI sybil generator
    # ----------------------------------------------------------
    print("\n" + "=" * 100)
    print("CALIBRATION PARAMETERS FOR AI SYBIL GENERATOR")
    print("=" * 100)
    print("\nReplace rng.beta(...) calls in generate_ai_sybils() with these:")
    print()

    for feat in AI_FEATURE_NAMES:
        if feat in beta_params.get("agent", {}):
            ap = beta_params["agent"][feat]
            hp = beta_params["human"][feat]
            print(f"# {feat}")
            print(f"#   Agent: Beta(alpha={ap['alpha']:.3f}, beta={ap['beta']:.3f})")
            print(f"#   Human: Beta(alpha={hp['alpha']:.3f}, beta={hp['beta']:.3f})")
            print()
        else:
            ad = distributions["agent"].get(feat, {})
            hd = distributions["human"].get(feat, {})
            if ad and hd:
                print(f"# {feat} (unbounded, use normal/lognormal)")
                print(f"#   Agent: mean={ad.get('mean',0):.4f}, std={ad.get('std',0):.4f}")
                print(f"#   Human: mean={hd.get('mean',0):.4f}, std={hd.get('std',0):.4f}")
                print()

    # ----------------------------------------------------------
    # Save: real_ai_feature_distributions.json
    # ----------------------------------------------------------
    distributions_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source": source_file,
            "n_agents": len(agent_features),
            "n_humans": len(human_features),
            "features": AI_FEATURE_NAMES,
            "description": (
                "Real AI-specific feature distributions computed from "
                "Paper 1 labeled on-chain agent/human addresses. "
                "Includes Mann-Whitney U tests for each feature."
            ),
        },
        "agent_distributions": distributions["agent"],
        "human_distributions": distributions["human"],
        "statistical_tests": statistical_tests,
        "beta_params": beta_params,
    }

    DISTRIBUTIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(DISTRIBUTIONS_FILE, "w") as f:
        json.dump(distributions_output, f, indent=2)
    print(f"\nDistributions saved to {DISTRIBUTIONS_FILE}")

    # ----------------------------------------------------------
    # Save: real_ai_features.json (legacy + per-address data)
    # ----------------------------------------------------------
    legacy_output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "source": source_file,
            "n_agents": len(agent_features),
            "n_humans": len(human_features),
            "features_extracted": AI_FEATURE_NAMES,
        },
        "per_address": {
            addr: {k: v for k, v in feats.items()}
            for addr, feats in all_features.items()
        },
        "distributions": distributions,
        "beta_params": beta_params,
        "statistical_tests": statistical_tests,
        "calibration_note": (
            "Use beta_params to calibrate AI sybil generation. "
            "Agent distributions represent REAL on-chain AI agent behavior. "
            "Human distributions represent REAL human behavior. "
            "For AI sybil evasion experiments, use agent distributions "
            "scaled by an evasion sophistication factor."
        ),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(legacy_output, f, indent=2)
    print(f"Per-address features saved to {OUTPUT_FILE}")

    # ----------------------------------------------------------
    # Print significance summary
    # ----------------------------------------------------------
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE SUMMARY")
    print("=" * 80)

    sig_count = 0
    for feat in AI_FEATURE_NAMES:
        test = statistical_tests.get(feat, {})
        mw = test.get("mann_whitney", {})
        p = mw.get("p_value")
        d = test.get("cohens_d", 0)
        if p is not None and p < 0.05:
            sig_count += 1
            star = "***" if p < 0.001 else ("**" if p < 0.01 else "*")
            effect = "large" if abs(d) > 0.8 else ("medium" if abs(d) > 0.5 else "small")
            print(f"  {star} {feat:<35} p={p:.6f}  d={d:.3f} ({effect} effect)")
        else:
            print(f"  ns  {feat:<35} p={p if p else 'N/A'}")

    print(f"\n  {sig_count}/{len(AI_FEATURE_NAMES)} features significantly "
          f"different between agents and humans (p<0.05)")
    print("\nDone.")


if __name__ == "__main__":
    main()
