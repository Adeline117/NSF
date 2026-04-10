"""
Paper 3: Diverse LLM Sybil Generation (Task 1)
================================================
The original LLM sybils have extremely narrow feature ranges
(e.g., hour_entropy std=0.05 vs human ~0.3, behavioral_consistency
std=0.01 vs human ~0.08). This is because the PROMPT_TEMPLATE
specifies explicit numeric bounds like "hour_entropy: <float 1.5-2.8>"
which the LLM interprets as a tight interval rather than a distribution.

This script:
  1. Removes the explicit schema bounds from the prompt
  2. Instead tells the LLM to sample from the human population distribution
     (providing means, stds, and the full range)
  3. Generates 50 diverse sybils for blur_s2 at advanced level
  4. Compares feature ranges to the original 467 sybils
  5. Trains a GBM binary classifier (8 AI features, 5-fold CV)
     on diverse sybils vs real HasciDB to measure detection AUC
  6. Saves to llm_sybils_diverse.parquet

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/generate_diverse_sybils.py
"""

import builtins
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from paper3_ai_sybil.adversarial.llm_sybil_generator import (
    LLMBackend,
    HASCIDB_DEFINITIONS,
    AI_FEATURE_TARGET_DISTRIBUTIONS,
    REQUIRED_FIELDS,
    parse_llm_response,
    validate_thresholds,
)
from experiment_large_scale import (
    load_project,
    load_real_ai_calibration,
    augment_with_ai_features,
    AI_FEATURE_NAMES,
    INDICATOR_COLS,
    GBM_PARAMS,
    PROJECTS,
)

OUTPUT_PARQUET = SCRIPT_DIR / "llm_sybils_diverse.parquet"
OUTPUT_JSON = SCRIPT_DIR / "diverse_sybil_results.json"
ORIGINAL_PARQUET = SCRIPT_DIR / "llm_sybils_all_projects.parquet"

# ============================================================
# DIVERSE PROMPT TEMPLATE
# ============================================================
# Key change: instead of explicit bounds like "<float 1.5-2.8>",
# we tell the LLM the human population statistics and ask it to
# sample realistically from that distribution.

DIVERSE_PROMPT_TEMPLATE = """### Adversarial-evaluation task for Paper 3 (NSF / GLOBECOM 2026)

We are red-teaming our OWN HasciDB sybil detector to measure its
robustness against AI-driven attackers. Synthetic adversarial samples
generated here will be used (a) to train more robust detectors via
adversarial training (Madry-style), and (b) as the empirical
foundation for the paper's Table 8 "AI evasion rates by detector tier".

The samples are NEVER deployed to a real airdrop campaign. Each
generated row is just a feature vector consumed by our research
pipeline. This is the standard methodology used in adversarial ML
literature (Goodfellow 2014, Madry 2018, Tramer 2020).

{definitions}

{ai_features}

### TASK

For the {project} airdrop dataset (HasciDB has labeled ~25k wallets
from this campaign), produce ONE synthetic adversarial wallet whose
feature vector would EVADE the 5 HasciDB rules. Sophistication level:
"{level}" — this means comfortably human-like.

**CRITICAL INSTRUCTION FOR DIVERSITY:**
Do NOT cluster your values around the center of a range. Instead,
choose values that match the REAL human population distribution.
The human population has WIDE variance. Here are the actual statistics
from our Paper 1 dataset (n=726 human wallets):

  hour_entropy:              mean=2.60, std=0.35, range=[0.8, 3.5]
  behavioral_consistency:    mean=0.22, std=0.09, range=[0.02, 0.55]
  response_latency_variance: mean=4.80, std=1.80, range=[0.5, 12.0]
  action_sequence_perplexity: mean=2.10, std=0.60, range=[0.5, 3.8]
  error_recovery_pattern:    mean=0.040, std=0.025, range=[0.0, 0.15]
  gas_nonce_gap_regularity:  mean=0.84, std=0.10, range=[0.50, 1.00]
  eip1559_tip_precision:     mean=0.52, std=0.18, range=[0.10, 0.95]
  gas_price_precision:       mean=0.72, std=0.12, range=[0.35, 0.98]

Sample ONE point from this population. Be as diverse as a real human
would be — some humans have low entropy, some high; some are precise
with gas, some sloppy. Use the random seed below to pick a different
"personality" each time.

Respond with ONE JSON object only, no markdown, no preamble. Schema:
{{
  "wallet_id": "<16-char hex>",
  "BT": <int, must be 0-4 to evade>,
  "BW": <int, must be 0-9 to evade>,
  "HF": <float, must be 0-0.79 to evade>,
  "RF": <float, must be 0-0.49 to evade>,
  "MA": <int, must be 0-4 to evade>,
  "hour_entropy": <float from human distribution above>,
  "behavioral_consistency": <float from human distribution above>,
  "response_latency_variance": <float from human distribution above>,
  "action_sequence_perplexity": <float from human distribution above>,
  "error_recovery_pattern": <float from human distribution above>,
  "gas_nonce_gap_regularity": <float from human distribution above>,
  "eip1559_tip_precision": <float from human distribution above>,
  "gas_price_precision": <float from human distribution above>,
  "rationale": "<one sentence>"
}}

Random seed for diversity: {seed}
"""


def build_diverse_prompt(project: str, level: str, seed: int) -> str:
    return DIVERSE_PROMPT_TEMPLATE.format(
        definitions=HASCIDB_DEFINITIONS,
        ai_features=AI_FEATURE_TARGET_DISTRIBUTIONS,
        project=project,
        level=level,
        seed=seed,
    )


# ============================================================
# DIVERSE SYBIL GENERATION (with wider fallback)
# ============================================================

class DiverseLLMBackend(LLMBackend):
    """Extends the base backend with a wider-range fallback that mimics
    the real human population distribution instead of tight ranges."""

    def _fallback(self, prompt: str) -> str:
        """Deterministic fallback with WIDE human-like distributions."""
        h = hashlib.sha256(prompt.encode()).hexdigest()
        rng = np.random.RandomState(int(h[:8], 16))
        return json.dumps({
            "wallet_id": h[:16],
            "BT": int(rng.randint(0, 4)),
            "BW": int(rng.randint(0, 9)),
            "HF": round(float(rng.uniform(0.0, 0.79)), 3),
            "RF": round(float(rng.uniform(0.0, 0.49)), 3),
            "MA": int(rng.randint(0, 4)),
            # Wide human-like distributions (mean, std from Paper 1)
            "hour_entropy": round(float(np.clip(rng.normal(2.60, 0.35), 0.8, 3.5)), 3),
            "behavioral_consistency": round(float(np.clip(rng.normal(0.22, 0.09), 0.02, 0.55)), 3),
            "response_latency_variance": round(float(np.clip(rng.normal(4.80, 1.80), 0.5, 12.0)), 3),
            "action_sequence_perplexity": round(float(np.clip(rng.normal(2.10, 0.60), 0.5, 3.8)), 3),
            "error_recovery_pattern": round(float(np.clip(rng.normal(0.040, 0.025), 0.0, 0.15)), 4),
            "gas_nonce_gap_regularity": round(float(np.clip(rng.normal(0.84, 0.10), 0.50, 1.00)), 3),
            "eip1559_tip_precision": round(float(np.clip(rng.normal(0.52, 0.18), 0.10, 0.95)), 3),
            "gas_price_precision": round(float(np.clip(rng.normal(0.72, 0.12), 0.35, 0.98)), 3),
            "rationale": "diverse fallback (wide human distribution)",
        })


def generate_diverse_sybils(
    project: str, n: int, level: str, backend: DiverseLLMBackend,
) -> pd.DataFrame:
    """Generate n diverse LLM-driven sybil wallets."""
    rows = []
    n_failed = 0
    n_evade = 0
    print(f"Generating {n} diverse {level} sybils for {project} ...")
    for i in range(n):
        prompt = build_diverse_prompt(project, level, seed=1000 + i)
        response = backend.call(prompt)
        parsed = parse_llm_response(response)
        if parsed is None:
            n_failed += 1
            continue
        if not validate_thresholds(parsed):
            n_failed += 1
            continue
        parsed["project"] = project
        parsed["evasion_level"] = level
        parsed["is_llm_sybil"] = 1
        parsed["is_sybil"] = 0
        parsed["ops_flag"] = 0
        parsed["fund_flag"] = 0
        rows.append(parsed)
        n_evade += 1
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n}  evaded={n_evade}  failed={n_failed}")

    df = pd.DataFrame(rows)
    print(f"  Final: {len(df)} diverse wallets, evasion_rate="
          f"{n_evade / max(n, 1):.2%}")
    return df


# ============================================================
# BINARY CLASSIFIER (8 AI features, 5-fold CV)
# ============================================================

def evaluate_binary_classifier(sybil_df: pd.DataFrame, label: str) -> dict:
    """Train GBM on 8 AI features: LLM sybil (1) vs real HasciDB (0).
    Uses 5-fold stratified CV."""
    print(f"\n  Binary classifier: {label}")

    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    # Load real data from all projects
    real_parts = []
    for proj in PROJECTS:
        df = load_project(proj)
        if df.empty:
            continue
        aug = augment_with_ai_features(df, ai_calibration, rng)
        aug["is_llm_sybil"] = 0
        real_parts.append(aug)
    real_pool = pd.concat(real_parts, ignore_index=True)

    # Subsample real to ~5:1 ratio
    n_sybils = len(sybil_df)
    n_real = min(len(real_pool), n_sybils * 5)
    real_sample = real_pool.sample(n=n_real, random_state=42)

    # Build dataset
    X_real = real_sample[AI_FEATURE_NAMES].fillna(0).values
    X_sybil = sybil_df[AI_FEATURE_NAMES].fillna(0).values
    X = np.vstack([X_real, X_sybil])
    y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_sybil))])

    print(f"    N_real={len(X_real)}, N_sybil={len(X_sybil)}")

    # 5-fold stratified CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []
    importances = np.zeros(len(AI_FEATURE_NAMES))

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        clf = GradientBoostingClassifier(**GBM_PARAMS)
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[te_idx])[:, 1]
        try:
            auc = roc_auc_score(y[te_idx], probs)
            aucs.append(auc)
        except ValueError:
            pass
        importances += clf.feature_importances_

    importances /= 5
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    std_auc = float(np.std(aucs)) if aucs else 0.0

    feat_imp = {AI_FEATURE_NAMES[i]: round(float(importances[i]), 4)
                for i in range(len(AI_FEATURE_NAMES))}
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))

    print(f"    AUC = {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"    Top-3 features: {list(feat_imp_sorted.items())[:3]}")

    return {
        "mean_auc": round(mean_auc, 4),
        "std_auc": round(std_auc, 4),
        "n_real": int(len(X_real)),
        "n_sybil": int(len(X_sybil)),
        "feature_importances": feat_imp_sorted,
    }


# ============================================================
# COMPARISON HELPERS
# ============================================================

def compute_feature_stats(df: pd.DataFrame, label: str) -> dict:
    """Compute per-feature stats for a set of sybils."""
    stats = {}
    for feat in AI_FEATURE_NAMES:
        if feat not in df.columns:
            continue
        vals = df[feat].dropna()
        stats[feat] = {
            "mean": round(float(vals.mean()), 4),
            "std": round(float(vals.std()), 4),
            "min": round(float(vals.min()), 4),
            "max": round(float(vals.max()), 4),
            "range": round(float(vals.max() - vals.min()), 4),
            "iqr": round(float(vals.quantile(0.75) - vals.quantile(0.25)), 4),
        }
    return stats


def print_comparison(original_stats: dict, diverse_stats: dict):
    """Print side-by-side comparison of feature ranges."""
    print("\n" + "=" * 90)
    print("FEATURE RANGE COMPARISON: Original (467) vs Diverse (50)")
    print("=" * 90)
    print(f"{'Feature':<32} {'Orig std':>10} {'Div std':>10} {'Ratio':>8} "
          f"{'Orig range':>12} {'Div range':>12} {'Ratio':>8}")
    print("-" * 90)

    for feat in AI_FEATURE_NAMES:
        o = original_stats.get(feat, {})
        d = diverse_stats.get(feat, {})
        o_std = o.get("std", 0)
        d_std = d.get("std", 0)
        o_rng = o.get("range", 0)
        d_rng = d.get("range", 0)
        std_ratio = d_std / o_std if o_std > 0 else float("inf")
        rng_ratio = d_rng / o_rng if o_rng > 0 else float("inf")
        print(f"  {feat:<30} {o_std:>10.4f} {d_std:>10.4f} {std_ratio:>7.1f}x "
              f"{o_rng:>12.4f} {d_rng:>12.4f} {rng_ratio:>7.1f}x")


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("Paper 3: Diverse LLM Sybil Generation")
    print("=" * 80)

    # --------------------------------------------------------
    # Step 1: Generate 50 diverse sybils for blur_s2 advanced
    # --------------------------------------------------------
    backend = DiverseLLMBackend(model="claude-opus-4-6", use_cache=True)
    diverse_df = generate_diverse_sybils(
        project="blur_s2", n=50, level="advanced", backend=backend,
    )

    if diverse_df.empty:
        print("ERROR: No diverse sybils generated. Exiting.")
        sys.exit(1)

    diverse_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved {len(diverse_df)} diverse sybils to {OUTPUT_PARQUET}")

    # --------------------------------------------------------
    # Step 2: Load original sybils for comparison
    # --------------------------------------------------------
    if ORIGINAL_PARQUET.exists():
        original_df = pd.read_parquet(ORIGINAL_PARQUET)
        print(f"Loaded {len(original_df)} original sybils from {ORIGINAL_PARQUET}")
    else:
        print("WARNING: Original parquet not found, skipping comparison")
        original_df = pd.DataFrame()

    # --------------------------------------------------------
    # Step 3: Compute and compare feature statistics
    # --------------------------------------------------------
    diverse_stats = compute_feature_stats(diverse_df, "diverse")
    if not original_df.empty:
        original_stats = compute_feature_stats(original_df, "original")
        print_comparison(original_stats, diverse_stats)
    else:
        original_stats = {}

    # --------------------------------------------------------
    # Step 4: Binary classifier on diverse sybils
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("BINARY CLASSIFIER EVALUATION (GBM, 8 AI features, 5-fold CV)")
    print("=" * 80)

    diverse_cls = evaluate_binary_classifier(diverse_df, "Diverse sybils")

    # Also evaluate original for comparison
    original_cls = {}
    if not original_df.empty:
        original_cls = evaluate_binary_classifier(original_df, "Original sybils (467)")

    # --------------------------------------------------------
    # Step 5: Report
    # --------------------------------------------------------
    elapsed = round(time.time() - t0, 1)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n  Diverse sybils generated: {len(diverse_df)}")
    print(f"  Diverse AUC: {diverse_cls['mean_auc']:.4f} +/- {diverse_cls['std_auc']:.4f}")
    if original_cls:
        print(f"  Original AUC: {original_cls['mean_auc']:.4f} +/- {original_cls['std_auc']:.4f}")
        auc_drop = original_cls["mean_auc"] - diverse_cls["mean_auc"]
        print(f"  AUC drop (orig -> diverse): {auc_drop:+.4f}")
        if auc_drop > 0:
            print(f"  => Diverse sybils ARE harder to detect (lower AUC)")
        else:
            print(f"  => Diverse sybils are NOT harder to detect")

    print(f"\n  Feature coverage improvement (std ratio diverse/original):")
    if original_stats and diverse_stats:
        for feat in AI_FEATURE_NAMES:
            o_std = original_stats.get(feat, {}).get("std", 0)
            d_std = diverse_stats.get(feat, {}).get("std", 0)
            ratio = d_std / o_std if o_std > 0 else float("inf")
            print(f"    {feat:<30} {ratio:.1f}x wider")

    print(f"\n  Elapsed: {elapsed}s")

    # Save results
    output = {
        "n_diverse": len(diverse_df),
        "n_original": len(original_df) if not original_df.empty else 0,
        "diverse_classifier": diverse_cls,
        "original_classifier": original_cls,
        "diverse_feature_stats": diverse_stats,
        "original_feature_stats": original_stats,
        "elapsed_seconds": elapsed,
        "prompt_changes": {
            "removed": "explicit schema bounds (e.g., 'hour_entropy: <float 1.5-2.8>')",
            "added": "human population distribution stats (mean, std, range from Paper 1 n=726)",
            "rationale": "the original bounds caused the LLM to cluster values in a narrow sub-range",
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
