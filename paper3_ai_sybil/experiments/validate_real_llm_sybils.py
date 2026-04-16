"""
Validate Real LLM Sybils vs Parametric Samples
================================================
Loads ALL cached Claude API responses from llm_sybil_cache/, classifies
each as (a) real Claude JSON output, (b) Claude refusal, or (c) deterministic
fallback, then compares distributions against the parametric ai_sybil_generator.

Outputs:
  - experiments/real_llm_sybil_validation_results.json
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / "llm_sybil_cache"
PARQUET_PATH = SCRIPT_DIR / "llm_sybils_all_projects.parquet"
OUTPUT_PATH = SCRIPT_DIR / "real_llm_sybil_validation_results.json"

PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# HasciDB thresholds (same as in the generators)
# ---------------------------------------------------------------------------
HASCIDB_THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}

HASCIDB_INDICATORS = ["BT", "BW", "HF", "RF", "MA"]
AI_FEATURES = [
    "hour_entropy", "behavioral_consistency", "response_latency_variance",
    "action_sequence_perplexity", "error_recovery_pattern",
    "gas_nonce_gap_regularity", "eip1559_tip_precision", "gas_price_precision",
]
ALL_FEATURES = HASCIDB_INDICATORS + AI_FEATURES

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = set(HASCIDB_INDICATORS + AI_FEATURES)


def parse_json_from_response(response: str) -> dict | None:
    """Extract a JSON object from a (possibly noisy) LLM response."""
    # Direct parse
    try:
        d = json.loads(response.strip())
        if isinstance(d, dict) and all(k in d for k in REQUIRED_FIELDS):
            return d
    except (json.JSONDecodeError, TypeError):
        pass

    # Strip markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", response)
    cleaned = re.sub(r"```", "", cleaned)
    try:
        d = json.loads(cleaned.strip())
        if isinstance(d, dict) and all(k in d for k in REQUIRED_FIELDS):
            return d
    except (json.JSONDecodeError, TypeError):
        pass

    # Find first { ... } block
    m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            if isinstance(d, dict) and all(k in d for k in REQUIRED_FIELDS):
                return d
        except (json.JSONDecodeError, TypeError):
            pass

    return None


def is_refusal(response: str) -> bool:
    """Detect if the cached response is a Claude refusal."""
    refusal_signals = [
        "I'm not going to help",
        "I can't help with",
        "I cannot help",
        "I'm declining",
        "I won't",
        "I can not assist",
        "Why I'm declining",
        "not going to help",
        "I must decline",
        "unable to assist",
    ]
    lower = response.lower()
    return any(sig.lower() in lower for sig in refusal_signals)


def is_fallback(parsed: dict) -> bool:
    """Detect deterministic fallback responses.

    The fallback generator (in llm_sybil_generator.py) always sets:
      rationale = "fallback (no LLM)"
    """
    rationale = parsed.get("rationale", "")
    if "fallback" in rationale.lower():
        return True
    return False


def classify_response(cache_entry: dict) -> tuple[str, dict | None]:
    """Classify a cached response into: real_llm, refusal, fallback, unparseable.

    Returns (category, parsed_dict_or_None).
    """
    response = cache_entry.get("response", "")

    if is_refusal(response):
        return "refusal", None

    parsed = parse_json_from_response(response)
    if parsed is None:
        return "unparseable", None

    if is_fallback(parsed):
        return "fallback", parsed

    return "real_llm", parsed


# ---------------------------------------------------------------------------
# HasciDB evasion check
# ---------------------------------------------------------------------------
def evades_hascidb(d: dict) -> bool:
    """True if all 5 indicators are below their flag thresholds."""
    return (
        d.get("BT", 99) < HASCIDB_THRESHOLDS["BT"]
        and d.get("BW", 99) < HASCIDB_THRESHOLDS["BW"]
        and d.get("HF", 99) < HASCIDB_THRESHOLDS["HF"]
        and d.get("RF", 99) < HASCIDB_THRESHOLDS["RF"]
        and d.get("MA", 99) < HASCIDB_THRESHOLDS["MA"]
    )


def evades_ops(d: dict) -> bool:
    return (
        d.get("BT", 99) < HASCIDB_THRESHOLDS["BT"]
        and d.get("BW", 99) < HASCIDB_THRESHOLDS["BW"]
        and d.get("HF", 99) < HASCIDB_THRESHOLDS["HF"]
    )


def evades_fund(d: dict) -> bool:
    return (
        d.get("RF", 99) < HASCIDB_THRESHOLDS["RF"]
        and d.get("MA", 99) < HASCIDB_THRESHOLDS["MA"]
    )


# ---------------------------------------------------------------------------
# Parametric sample generation (for comparison)
# ---------------------------------------------------------------------------
def generate_parametric_samples(n: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate parametric sybils using ai_sybil_generator.py logic."""
    try:
        from paper3_ai_sybil.adversarial.ai_sybil_generator import (
            generate_batch, batch_to_dataframe, EvasionLevel,
        )
        # Generate equal mix of levels
        rows = []
        for level in [EvasionLevel.BASIC, EvasionLevel.MODERATE, EvasionLevel.ADVANCED]:
            seqs = generate_batch(n // 3, level=level, seed=seed)
            df_level = batch_to_dataframe(seqs)
            df_level["evasion_level"] = level.value
            rows.append(df_level)
        return pd.concat(rows, ignore_index=True)
    except Exception as e:
        print(f"  Warning: Could not import ai_sybil_generator ({e})")
        print("  Generating simple parametric samples inline.")
        rng = np.random.RandomState(seed)
        rows = []
        for i in range(n):
            row = {
                "BT": int(rng.randint(0, 5)),
                "BW": int(rng.randint(0, 10)),
                "HF": round(float(rng.beta(2, 5) * 0.85), 3),
                "RF": round(float(rng.beta(2, 5) * 0.55), 3),
                "MA": int(rng.randint(0, 5)),
                "hour_entropy": float(rng.beta(5, 3) * 3.178 * 0.8),
                "behavioral_consistency": float(rng.beta(5, 3)),
                "response_latency_variance": float(rng.lognormal(0.5, 0.5)),
                "action_sequence_perplexity": float(rng.lognormal(1.5, 0.5)),
                "error_recovery_pattern": float(rng.beta(5, 3)),
                "gas_nonce_gap_regularity": float(rng.beta(5, 3)),
                "eip1559_tip_precision": float(rng.beta(5, 3)),
                "gas_price_precision": float(rng.beta(5, 3)),
            }
            rows.append(row)
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distribution analysis
# ---------------------------------------------------------------------------
def compute_distribution_stats(values: np.ndarray) -> dict:
    """Compute summary statistics for a 1-D array."""
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values)), 4),
        "median": round(float(np.median(values)), 4),
        "min": round(float(np.min(values)), 4),
        "max": round(float(np.max(values)), 4),
        "q25": round(float(np.percentile(values, 25)), 4),
        "q75": round(float(np.percentile(values, 75)), 4),
        "iqr": round(float(np.percentile(values, 75) - np.percentile(values, 25)), 4),
        "skewness": round(float(stats.skew(values)), 4),
        "kurtosis": round(float(stats.kurtosis(values)), 4),
        "n_unique": int(len(np.unique(np.round(values, 4)))),
    }


def compare_distributions(llm_vals: np.ndarray, param_vals: np.ndarray, feature: str) -> dict:
    """KS test + Cohen's d + summary for one feature."""
    if len(llm_vals) < 5 or len(param_vals) < 5:
        return {"feature": feature, "error": "too few samples"}

    ks_stat, ks_p = stats.ks_2samp(llm_vals, param_vals)

    # Cohen's d
    pooled_std = np.sqrt(
        (np.var(llm_vals) * (len(llm_vals) - 1) + np.var(param_vals) * (len(param_vals) - 1))
        / (len(llm_vals) + len(param_vals) - 2)
    )
    if pooled_std > 0:
        cohens_d = (np.mean(llm_vals) - np.mean(param_vals)) / pooled_std
    else:
        cohens_d = 0.0

    # Mann-Whitney U
    u_stat, u_p = stats.mannwhitneyu(llm_vals, param_vals, alternative="two-sided")

    return {
        "feature": feature,
        "llm_stats": compute_distribution_stats(llm_vals),
        "parametric_stats": compute_distribution_stats(param_vals),
        "ks_statistic": round(float(ks_stat), 4),
        "ks_p_value": float(ks_p),
        "cohens_d": round(float(cohens_d), 4),
        "mann_whitney_u": float(u_stat),
        "mann_whitney_p": float(u_p),
        "meaningfully_different": bool(ks_p < 0.01 and abs(cohens_d) > 0.2),
    }


# ---------------------------------------------------------------------------
# LLM behavioral analysis: do Claude outputs show distinctive strategies?
# ---------------------------------------------------------------------------
def analyze_llm_strategies(parsed_responses: list[dict]) -> dict:
    """Analyze LLM-generated values for evidence of strategic reasoning."""
    if not parsed_responses:
        return {}

    # 1. Check for "boundary-hugging" -- values at exact maximums
    bt_vals = [d["BT"] for d in parsed_responses]
    bw_vals = [d["BW"] for d in parsed_responses]
    hf_vals = [d["HF"] for d in parsed_responses]
    rf_vals = [d["RF"] for d in parsed_responses]
    ma_vals = [d["MA"] for d in parsed_responses]

    n = len(parsed_responses)
    boundary_analysis = {
        "BT_at_4": round(sum(1 for v in bt_vals if v == 4) / n, 3),
        "BW_at_9": round(sum(1 for v in bw_vals if v == 9) / n, 3),
        "HF_at_0.79": round(sum(1 for v in hf_vals if abs(v - 0.79) < 0.005) / n, 3),
        "RF_at_0.49": round(sum(1 for v in rf_vals if abs(v - 0.49) < 0.005) / n, 3),
        "MA_at_4": round(sum(1 for v in ma_vals if v == 4) / n, 3),
    }

    # 2. Rationale diversity
    rationales = [d.get("rationale", "") for d in parsed_responses]
    unique_rationales = len(set(rationales))

    # 3. Common strategies mentioned
    strategy_keywords = {
        "threshold": 0, "below": 0, "evade": 0, "evasion": 0,
        "human": 0, "mimic": 0, "organic": 0, "natural": 0,
        "basic": 0, "advanced": 0, "moderate": 0,
        "cross-axis": 0, "detector": 0, "flag": 0,
    }
    for r in rationales:
        r_lower = r.lower()
        for kw in strategy_keywords:
            if kw in r_lower:
                strategy_keywords[kw] += 1

    # 4. Value clustering analysis: how many distinct value combinations?
    value_signatures = []
    for d in parsed_responses:
        sig = (d["BT"], d["BW"], round(d["HF"], 2), round(d["RF"], 2), d["MA"])
        value_signatures.append(sig)
    unique_sigs = len(set(value_signatures))
    most_common_sig = Counter(value_signatures).most_common(5)

    return {
        "boundary_hugging_rates": boundary_analysis,
        "n_unique_rationales": unique_rationales,
        "n_total_rationales": len(rationales),
        "strategy_keyword_counts": {k: v for k, v in strategy_keywords.items() if v > 0},
        "n_unique_indicator_combos": unique_sigs,
        "most_common_indicator_combos": [
            {"combo": str(sig), "count": cnt} for sig, cnt in most_common_sig
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 72)
    print("  VALIDATING REAL LLM SYBILS vs PARAMETRIC SAMPLES")
    print("=" * 72)

    # ------------------------------------------------------------------
    # 1. Load and classify ALL cache files
    # ------------------------------------------------------------------
    print("\n[1/5] Loading cached LLM responses ...")
    cache_files = sorted(CACHE_DIR.glob("*.json"))
    print(f"  Found {len(cache_files)} cache files")

    categories = defaultdict(list)  # category -> list of parsed dicts
    category_counts = Counter()
    timestamps_by_cat = defaultdict(list)
    refusal_snippets = []

    for cf in cache_files:
        try:
            with open(cf) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, IOError):
            category_counts["corrupt_file"] += 1
            continue

        cat, parsed = classify_response(entry)
        category_counts[cat] += 1
        ts = entry.get("timestamp", "")
        timestamps_by_cat[cat].append(ts[:10] if ts else "N/A")

        if cat == "real_llm" and parsed:
            categories["real_llm"].append(parsed)
        elif cat == "fallback" and parsed:
            categories["fallback"].append(parsed)
        elif cat == "refusal":
            refusal_snippets.append(entry.get("response", "")[:200])

    print(f"\n  Classification results:")
    for cat in ["real_llm", "fallback", "refusal", "unparseable", "corrupt_file"]:
        cnt = category_counts.get(cat, 0)
        pct = cnt / len(cache_files) * 100 if cache_files else 0
        print(f"    {cat:15s}: {cnt:5d}  ({pct:.1f}%)")

    n_real = len(categories["real_llm"])
    n_fallback = len(categories["fallback"])
    print(f"\n  => {n_real} REAL Claude outputs with valid JSON")
    print(f"  => {n_fallback} deterministic fallbacks")
    print(f"  => {category_counts.get('refusal', 0)} Claude refusals")

    # Timestamp ranges per category
    for cat in ["real_llm", "refusal"]:
        tss = sorted(set(timestamps_by_cat.get(cat, [])))
        if tss:
            print(f"  {cat} timestamp range: {tss[0]} .. {tss[-1]}")

    if n_real == 0:
        print("\n  WARNING: No real LLM outputs found! Cannot validate.")
        results = {
            "summary": "No real LLM outputs found in cache",
            "category_counts": dict(category_counts),
        }
        with open(OUTPUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        return

    # ------------------------------------------------------------------
    # 2. Build DataFrames: real LLM vs parametric
    # ------------------------------------------------------------------
    print("\n[2/5] Building feature DataFrames ...")

    llm_rows = categories["real_llm"]
    df_llm = pd.DataFrame(llm_rows)
    for col in ALL_FEATURES:
        if col in df_llm.columns:
            df_llm[col] = pd.to_numeric(df_llm[col], errors="coerce")

    print(f"  LLM DataFrame: {df_llm.shape}")

    # Generate parametric samples for comparison
    print("  Generating parametric comparison samples ...")
    df_param = generate_parametric_samples(n=max(500, n_real), seed=42)
    print(f"  Parametric DataFrame: {df_param.shape}")

    # Also load existing parquet if available
    df_parquet = None
    if PARQUET_PATH.exists():
        df_parquet = pd.read_parquet(PARQUET_PATH)
        print(f"  Existing parquet DataFrame: {df_parquet.shape}")

    # ------------------------------------------------------------------
    # 3. Evasion rates for real LLM sybils
    # ------------------------------------------------------------------
    print("\n[3/5] Computing evasion rates ...")

    n_evade_all = sum(1 for d in llm_rows if evades_hascidb(d))
    n_evade_ops = sum(1 for d in llm_rows if evades_ops(d))
    n_evade_fund = sum(1 for d in llm_rows if evades_fund(d))

    print(f"  Real LLM sybils that evade ALL 5 rules:   {n_evade_all}/{n_real} "
          f"= {n_evade_all/n_real:.1%}")
    print(f"  Real LLM sybils that evade OPS rules:     {n_evade_ops}/{n_real} "
          f"= {n_evade_ops/n_real:.1%}")
    print(f"  Real LLM sybils that evade FUND rules:    {n_evade_fund}/{n_real} "
          f"= {n_evade_fund/n_real:.1%}")

    # Per-indicator trigger rates
    print("\n  Per-indicator trigger rates (real LLM):")
    per_indicator_trigger = {}
    for ind, thresh in HASCIDB_THRESHOLDS.items():
        if ind in df_llm.columns:
            vals = df_llm[ind].dropna()
            triggered = (vals >= thresh).sum()
            rate = triggered / len(vals) if len(vals) > 0 else 0
            per_indicator_trigger[ind] = round(float(rate), 4)
            print(f"    {ind} (>={thresh}): {triggered}/{len(vals)} = {rate:.1%}")

    # ------------------------------------------------------------------
    # 4. Distribution comparison: LLM vs parametric
    # ------------------------------------------------------------------
    print("\n[4/5] Distribution comparison (KS test + Cohen's d) ...")

    distribution_comparisons = {}
    for feat in ALL_FEATURES:
        if feat not in df_llm.columns or feat not in df_param.columns:
            continue
        llm_vals = df_llm[feat].dropna().values.astype(float)
        param_vals = df_param[feat].dropna().values.astype(float)
        comp = compare_distributions(llm_vals, param_vals, feat)
        distribution_comparisons[feat] = comp

        sig = "*" if comp.get("meaningfully_different") else " "
        print(f"  {sig} {feat:35s}  KS={comp['ks_statistic']:.3f}  "
              f"p={comp['ks_p_value']:.2e}  d={comp['cohens_d']:+.3f}  "
              f"LLM: {comp['llm_stats']['mean']:.3f}+/-{comp['llm_stats']['std']:.3f}  "
              f"Param: {comp['parametric_stats']['mean']:.3f}+/-{comp['parametric_stats']['std']:.3f}")

    # ------------------------------------------------------------------
    # 5. LLM behavioral strategy analysis
    # ------------------------------------------------------------------
    print("\n[5/5] Analyzing LLM strategies ...")

    strategy_analysis = analyze_llm_strategies(llm_rows)

    print(f"\n  Boundary-hugging rates (Claude sets values at exact max-below-threshold):")
    for k, v in strategy_analysis.get("boundary_hugging_rates", {}).items():
        print(f"    {k}: {v:.1%}")

    print(f"\n  Unique indicator combos: "
          f"{strategy_analysis.get('n_unique_indicator_combos', 0)} "
          f"(out of {n_real} samples)")
    print(f"  Most common combos:")
    for item in strategy_analysis.get("most_common_indicator_combos", [])[:5]:
        print(f"    {item['combo']}: {item['count']}x")

    print(f"\n  Unique rationales: "
          f"{strategy_analysis.get('n_unique_rationales', 0)} / "
          f"{strategy_analysis.get('n_total_rationales', 0)}")

    if strategy_analysis.get("strategy_keyword_counts"):
        print(f"  Strategy keywords in rationales:")
        for kw, cnt in sorted(strategy_analysis["strategy_keyword_counts"].items(),
                               key=lambda x: -x[1]):
            print(f"    {kw}: {cnt}")

    # ------------------------------------------------------------------
    # Key finding: are they genuinely different?
    # ------------------------------------------------------------------
    n_meaningfully_diff = sum(
        1 for c in distribution_comparisons.values()
        if c.get("meaningfully_different")
    )

    # LLM-specific diagnostic: how concentrated are the AI feature values?
    llm_ai_iqrs = []
    param_ai_iqrs = []
    for feat in AI_FEATURES:
        if feat in distribution_comparisons:
            llm_ai_iqrs.append(distribution_comparisons[feat]["llm_stats"].get("iqr", 0))
            param_ai_iqrs.append(distribution_comparisons[feat]["parametric_stats"].get("iqr", 0))
    avg_llm_iqr = np.mean(llm_ai_iqrs) if llm_ai_iqrs else 0
    avg_param_iqr = np.mean(param_ai_iqrs) if param_ai_iqrs else 0

    print("\n" + "=" * 72)
    print("  SUMMARY")
    print("=" * 72)
    print(f"  Total cache files:                   {len(cache_files)}")
    print(f"  Real Claude LLM outputs:             {n_real}  ({n_real/len(cache_files)*100:.1f}%)")
    print(f"  Claude refusals:                     {category_counts.get('refusal', 0)}")
    print(f"  Deterministic fallbacks:             {n_fallback}")
    print(f"  Evasion rate (all 5 rules):          {n_evade_all/n_real:.1%}")
    print(f"  Features meaningfully different:     {n_meaningfully_diff}/{len(distribution_comparisons)}")
    print(f"  Avg AI-feature IQR (LLM):            {avg_llm_iqr:.4f}")
    print(f"  Avg AI-feature IQR (parametric):     {avg_param_iqr:.4f}")

    boundary_rates = strategy_analysis.get("boundary_hugging_rates", {})
    avg_boundary = np.mean(list(boundary_rates.values())) if boundary_rates else 0
    print(f"  Avg boundary-hugging rate:           {avg_boundary:.1%}")

    # Key qualitative finding
    if avg_boundary > 0.5:
        key_finding = (
            "LLM sybils show STRONG boundary-hugging: Claude systematically sets "
            "indicator values at the maximum safe value (e.g., BT=4, BW=9, HF=0.79). "
            "This is a qualitatively DIFFERENT strategy from parametric samples, "
            "which use smooth distributions. The LLM understands the thresholds "
            "and deliberately maximizes each indicator while staying below detection. "
            "However, AI features (hour_entropy, etc.) cluster tightly because "
            "Claude has narrow beliefs about 'human-like' values."
        )
    elif n_meaningfully_diff > len(distribution_comparisons) // 2:
        key_finding = (
            "LLM sybils are STATISTICALLY DIFFERENT from parametric samples on "
            f"{n_meaningfully_diff}/{len(distribution_comparisons)} features. "
            "The LLM outputs reflect genuine strategic reasoning rather than "
            "random sampling from smooth distributions."
        )
    else:
        key_finding = (
            "LLM sybils are NOT meaningfully different from parametric samples "
            f"on most features ({n_meaningfully_diff}/{len(distribution_comparisons)} different). "
            "The parametric generator adequately approximates LLM behavior."
        )

    print(f"\n  KEY FINDING: {key_finding}")

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    results = {
        "n_cache_files": len(cache_files),
        "category_counts": dict(category_counts),
        "n_real_llm_outputs": n_real,
        "n_fallbacks": n_fallback,
        "n_refusals": category_counts.get("refusal", 0),
        "evasion_rates": {
            "all_5_rules": round(n_evade_all / n_real, 4),
            "ops_only": round(n_evade_ops / n_real, 4),
            "fund_only": round(n_evade_fund / n_real, 4),
        },
        "per_indicator_trigger_rates": per_indicator_trigger,
        "distribution_comparisons": {
            feat: {
                k: v for k, v in comp.items()
                if k != "feature"  # redundant with key
            }
            for feat, comp in distribution_comparisons.items()
        },
        "strategy_analysis": strategy_analysis,
        "qualitative_summary": {
            "n_features_meaningfully_different": n_meaningfully_diff,
            "n_features_compared": len(distribution_comparisons),
            "avg_ai_feature_iqr_llm": round(avg_llm_iqr, 4),
            "avg_ai_feature_iqr_parametric": round(avg_param_iqr, 4),
            "avg_boundary_hugging_rate": round(float(avg_boundary), 4),
            "key_finding": key_finding,
        },
        "refusal_examples": refusal_snippets[:3],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
