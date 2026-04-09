"""
Paper 3: Batch LLM Sybil Generation Across All 16 HasciDB Projects
====================================================================
Generates LLM-driven sybil wallets for all 16 projects at 3 evasion
levels (basic, moderate, advanced), then evaluates them against:
  (a) HasciDB 5-indicator rules (evasion rate)
  (b) Pre-airdrop LightGBM baseline (evasion rate)
  (c) P3-B ablation binary classifier: LLM sybil vs real HasciDB

Uses the existing llm_sybil_generator.py infrastructure with caching.
Each LLM call is cached to disk, so re-runs are deterministic and free.

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/batch_llm_sybil_generation.py

Output:
    paper3_ai_sybil/experiments/llm_sybils_all_projects.parquet
    paper3_ai_sybil/experiments/llm_sybil_batch_results.json
"""

import builtins
import json
import sys
import time
import traceback
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

# Force flush on all prints
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

HASCIDB_DIR = (
    PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
)
OUTPUT_PARQUET = SCRIPT_DIR / "llm_sybils_all_projects.parquet"
OUTPUT_JSON = SCRIPT_DIR / "llm_sybil_batch_results.json"

# ============================================================
# IMPORTS FROM EXISTING MODULES
# ============================================================

from paper3_ai_sybil.adversarial.llm_sybil_generator import (
    LLMBackend,
    generate_sybils,
    PROJECT_LAUNCH_DATES,
    REQUIRED_FIELDS,
    validate_thresholds,
)
from experiment_large_scale import (
    load_project,
    load_real_ai_calibration,
    augment_with_ai_features,
    AI_FEATURE_NAMES,
    OPS_COLS,
    FUND_COLS,
    GBM_PARAMS,
    INDICATOR_COLS,
    THRESHOLDS,
    MAX_ROWS,
)

# ============================================================
# CONFIGURATION
# ============================================================

PROJECTS = [
    "1inch", "uniswap", "looksrare", "apecoin", "ens", "gitcoin",
    "etherfi", "x2y2", "dydx", "badger", "blur_s1", "blur_s2",
    "paraswap", "ampleforth", "eigenlayer", "pengu",
]

LEVELS = ["basic", "moderate", "advanced"]
N_PER_PROJECT_PER_LEVEL = 10  # 10 x 16 x 3 = 480 total attempts

# LightGBM params (from pre_airdrop_lightgbm.py)
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 63,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

TEMPORAL_FEATURES = ["BT", "BW", "HF", "RF", "MA", "n_indicators"]


# ============================================================
# EVALUATION HELPERS
# ============================================================

def evaluate_metrics(y_true, y_prob) -> dict:
    """Compute standard classification metrics."""
    y_pred = (y_prob >= 0.5).astype(int)
    metrics = {}
    try:
        metrics["auc_roc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
    except ValueError:
        metrics["auc_roc"] = None
    try:
        metrics["avg_precision"] = round(float(average_precision_score(y_true, y_prob)), 4)
    except ValueError:
        metrics["avg_precision"] = None
    metrics["precision"] = round(float(precision_score(y_true, y_pred, zero_division=0)), 4)
    metrics["recall"] = round(float(recall_score(y_true, y_pred, zero_division=0)), 4)
    metrics["f1"] = round(float(f1_score(y_true, y_pred, zero_division=0)), 4)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"], metrics["fp"] = int(cm[0, 0]), int(cm[0, 1])
    metrics["fn"], metrics["tp"] = int(cm[1, 0]), int(cm[1, 1])
    return metrics


# ============================================================
# STEP 1: GENERATE ALL LLM SYBILS
# ============================================================

def generate_all_sybils(backend: LLMBackend) -> pd.DataFrame:
    """Generate LLM sybils for all 16 projects x 3 levels."""
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING LLM SYBILS")
    print(f"  {len(PROJECTS)} projects x {len(LEVELS)} levels x "
          f"{N_PER_PROJECT_PER_LEVEL} per = {len(PROJECTS) * len(LEVELS) * N_PER_PROJECT_PER_LEVEL} total attempts")
    print("=" * 80)

    all_dfs = []
    stats = {
        "per_project": {},
        "per_level": {level: {"attempted": 0, "generated": 0} for level in LEVELS},
        "total_attempted": 0,
        "total_generated": 0,
    }

    for proj in PROJECTS:
        launch_date = PROJECT_LAUNCH_DATES.get(proj, "2023-01")
        proj_stats = {}

        for level in LEVELS:
            t0 = time.time()
            df = generate_sybils(
                project=proj,
                launch_date=launch_date,
                n=N_PER_PROJECT_PER_LEVEL,
                level=level,
                backend=backend,
            )
            elapsed = time.time() - t0

            n_attempted = N_PER_PROJECT_PER_LEVEL
            n_generated = len(df)

            stats["per_level"][level]["attempted"] += n_attempted
            stats["per_level"][level]["generated"] += n_generated
            stats["total_attempted"] += n_attempted
            stats["total_generated"] += n_generated

            proj_stats[level] = {
                "attempted": n_attempted,
                "generated": n_generated,
                "success_rate": round(n_generated / max(n_attempted, 1), 4),
                "elapsed_seconds": round(elapsed, 1),
            }

            if not df.empty:
                all_dfs.append(df)

            print(f"  {proj}/{level}: {n_generated}/{n_attempted} "
                  f"({n_generated/max(n_attempted,1):.0%}) in {elapsed:.1f}s")

        stats["per_project"][proj] = proj_stats

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
    else:
        combined = pd.DataFrame()

    stats["total_generated"] = len(combined)
    stats["overall_success_rate"] = round(
        len(combined) / max(stats["total_attempted"], 1), 4
    )

    print(f"\n  TOTAL: {len(combined)} generated / "
          f"{stats['total_attempted']} attempted = "
          f"{stats['overall_success_rate']:.1%}")

    return combined, stats


# ============================================================
# STEP 2: HASCIDB RULE EVASION ANALYSIS
# ============================================================

def analyze_rule_evasion(llm_df: pd.DataFrame) -> dict:
    """Check how many LLM sybils evade HasciDB rules."""
    print("\n" + "=" * 80)
    print("STEP 2: HASCIDB RULE EVASION ANALYSIS")
    print("=" * 80)

    results = {}
    for level in LEVELS:
        subset = llm_df[llm_df["evasion_level"] == level]
        if subset.empty:
            results[level] = {"n": 0, "evade_all": 0, "evasion_rate": 0.0}
            continue

        n = len(subset)
        # Check each indicator
        bt_flag = (subset["BT"] >= 5).sum()
        bw_flag = (subset["BW"] >= 10).sum()
        hf_flag = (subset["HF"] >= 0.8).sum()
        rf_flag = (subset["RF"] >= 0.5).sum()
        ma_flag = (subset["MA"] >= 5).sum()
        any_flag = (
            (subset["BT"] >= 5) |
            (subset["BW"] >= 10) |
            (subset["HF"] >= 0.8) |
            (subset["RF"] >= 0.5) |
            (subset["MA"] >= 5)
        ).sum()
        evade_all = n - any_flag

        results[level] = {
            "n": int(n),
            "evade_all": int(evade_all),
            "evasion_rate": round(evade_all / n, 4),
            "flagged_BT": int(bt_flag),
            "flagged_BW": int(bw_flag),
            "flagged_HF": int(hf_flag),
            "flagged_RF": int(rf_flag),
            "flagged_MA": int(ma_flag),
            "any_flagged": int(any_flag),
        }
        print(f"  {level:<10}: {evade_all}/{n} evade all rules "
              f"({evade_all/n:.1%})")
        print(f"    Flagged: BT={bt_flag} BW={bw_flag} HF={hf_flag} "
              f"RF={rf_flag} MA={ma_flag}")

    # Overall
    n_total = len(llm_df)
    any_flag_total = (
        (llm_df["BT"] >= 5) |
        (llm_df["BW"] >= 10) |
        (llm_df["HF"] >= 0.8) |
        (llm_df["RF"] >= 0.5) |
        (llm_df["MA"] >= 5)
    ).sum() if n_total > 0 else 0
    evade_total = n_total - any_flag_total
    results["overall"] = {
        "n": int(n_total),
        "evade_all": int(evade_total),
        "evasion_rate": round(evade_total / max(n_total, 1), 4),
    }
    print(f"\n  OVERALL: {evade_total}/{n_total} evade all rules "
          f"({evade_total/max(n_total,1):.1%})")

    return results


# ============================================================
# STEP 3: PRE-AIRDROP LIGHTGBM EVASION
# ============================================================

def analyze_lightgbm_evasion(llm_df: pd.DataFrame) -> dict:
    """Train pre-airdrop LightGBM on each project, test LLM sybils."""
    print("\n" + "=" * 80)
    print("STEP 3: PRE-AIRDROP LIGHTGBM EVASION")
    print("=" * 80)

    # Add n_indicators column
    llm_with_ni = llm_df.copy()
    llm_with_ni["n_indicators"] = (
        (llm_with_ni["BT"] >= 5).astype(int)
        + (llm_with_ni["BW"] >= 10).astype(int)
        + (llm_with_ni["HF"] >= 0.8).astype(int)
        + (llm_with_ni["RF"] >= 0.5).astype(int)
        + (llm_with_ni["MA"] >= 5).astype(int)
    )

    # Train a pooled LightGBM on ALL projects combined
    print("\n  Training pooled LightGBM on all 16 projects ...")
    train_dfs = []
    for proj in PROJECTS:
        df = load_project(proj)
        if df.empty:
            continue
        # Add n_indicators
        df["n_indicators"] = (
            (df["BT"] >= 5).astype(int)
            + (df["BW"] >= 10).astype(int)
            + (df["HF"] >= 0.8).astype(int)
            + (df["RF"] >= 0.5).astype(int)
            + (df["MA"] >= 5).astype(int)
        )
        train_dfs.append(df)

    pooled = pd.concat(train_dfs, ignore_index=True)
    X_train = pooled[TEMPORAL_FEATURES].fillna(0).values
    y_train = pooled["is_sybil"].values.astype(int)

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_train, y_train)
    print(f"  Trained on {len(pooled)} rows ({int(y_train.sum())} sybils)")

    results = {}
    for level in LEVELS:
        subset = llm_with_ni[llm_with_ni["evasion_level"] == level]
        if subset.empty:
            results[level] = {"n": 0, "detected": 0, "evasion_rate": 0.0}
            continue

        X_llm = subset[TEMPORAL_FEATURES].fillna(0).values
        probs = clf.predict_proba(X_llm)[:, 1]
        detected = int((probs >= 0.5).sum())
        n = len(subset)
        evasion_rate = round(1 - detected / max(n, 1), 4)

        results[level] = {
            "n": int(n),
            "detected": detected,
            "evasion_rate": evasion_rate,
            "mean_sybil_score": round(float(probs.mean()), 4),
            "median_sybil_score": round(float(np.median(probs)), 4),
            "max_sybil_score": round(float(probs.max()), 4),
        }
        print(f"  {level:<10}: {n - detected}/{n} evade LightGBM "
              f"({evasion_rate:.1%}), mean_score={probs.mean():.4f}")

    # Overall
    X_all = llm_with_ni[TEMPORAL_FEATURES].fillna(0).values
    if len(X_all) > 0:
        probs_all = clf.predict_proba(X_all)[:, 1]
        detected_all = int((probs_all >= 0.5).sum())
        results["overall"] = {
            "n": int(len(llm_with_ni)),
            "detected": detected_all,
            "evasion_rate": round(1 - detected_all / max(len(llm_with_ni), 1), 4),
            "mean_sybil_score": round(float(probs_all.mean()), 4),
        }
        print(f"\n  OVERALL: {len(llm_with_ni) - detected_all}/{len(llm_with_ni)} "
              f"evade LightGBM ({results['overall']['evasion_rate']:.1%})")

    return results


# ============================================================
# STEP 4: P3-B ABLATION BINARY CLASSIFIER
#         (LLM sybil vs real HasciDB addresses)
# ============================================================

def analyze_binary_classifier(llm_df: pd.DataFrame) -> dict:
    """Train a binary classifier: LLM sybil (1) vs real HasciDB (0).

    Uses LOPO: for each project, train on the other 15 projects'
    real data + LLM sybils from those 15, test on the held-out
    project's real data + LLM sybils.

    Features: 5 indicators + 8 AI features = 13 total.
    """
    print("\n" + "=" * 80)
    print("STEP 4: P3-B ABLATION -- BINARY CLASSIFIER (LLM sybil vs real HasciDB)")
    print("=" * 80)

    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    # Load real data and augment with AI features (human distribution)
    real_data = {}
    for proj in PROJECTS:
        df = load_project(proj)
        if df.empty:
            continue
        df_aug = augment_with_ai_features(df, ai_calibration, rng)
        df_aug["is_llm_sybil"] = 0
        real_data[proj] = df_aug

    # Prepare LLM sybils with AI features already present
    all_features = INDICATOR_COLS + AI_FEATURE_NAMES

    results_per_level = {}

    for level in LEVELS:
        print(f"\n  --- Level: {level} ---")
        llm_subset = llm_df[llm_df["evasion_level"] == level].copy()
        if llm_subset.empty:
            results_per_level[level] = {"note": "no LLM sybils at this level"}
            continue

        # Ensure all features exist
        for feat in all_features:
            if feat not in llm_subset.columns:
                llm_subset[feat] = 0.0

        # Per-project LLM sybils
        llm_by_proj = {}
        for proj in PROJECTS:
            proj_llm = llm_subset[llm_subset["project"] == proj]
            if not proj_llm.empty:
                llm_by_proj[proj] = proj_llm

        # LOPO evaluation
        lopo_aucs = {}
        for test_proj in PROJECTS:
            if test_proj not in real_data:
                continue
            test_real = real_data[test_proj]
            test_llm = llm_by_proj.get(test_proj, pd.DataFrame())

            if test_llm.empty:
                continue

            # Train on other projects
            train_real_parts = []
            train_llm_parts = []
            for p in PROJECTS:
                if p == test_proj:
                    continue
                if p in real_data:
                    train_real_parts.append(real_data[p])
                if p in llm_by_proj:
                    train_llm_parts.append(llm_by_proj[p])

            if not train_real_parts or not train_llm_parts:
                continue

            train_real = pd.concat(train_real_parts, ignore_index=True)
            train_llm = pd.concat(train_llm_parts, ignore_index=True)

            # Subsample real data to balance (~3:1 ratio real:llm)
            n_llm_train = len(train_llm)
            n_real_sample = min(len(train_real), n_llm_train * 3)
            train_real_sampled = train_real.sample(
                n=n_real_sample, random_state=42
            )

            # Build train set
            X_train_real = train_real_sampled[all_features].fillna(0).values
            X_train_llm = train_llm[all_features].fillna(0).values
            X_train = np.vstack([X_train_real, X_train_llm])
            y_train = np.concatenate([
                np.zeros(len(X_train_real)),
                np.ones(len(X_train_llm)),
            ])

            # Build test set
            # Subsample test real for speed
            n_test_llm = len(test_llm)
            n_test_real = min(len(test_real), n_test_llm * 5)
            test_real_sampled = test_real.sample(
                n=n_test_real, random_state=42
            )
            X_test_real = test_real_sampled[all_features].fillna(0).values
            X_test_llm = test_llm[all_features].fillna(0).values
            X_test = np.vstack([X_test_real, X_test_llm])
            y_test = np.concatenate([
                np.zeros(len(X_test_real)),
                np.ones(len(X_test_llm)),
            ])

            if y_test.sum() < 2 or (len(y_test) - y_test.sum()) < 2:
                continue

            clf = GradientBoostingClassifier(**GBM_PARAMS)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]

            try:
                auc = round(float(roc_auc_score(y_test, probs)), 4)
            except ValueError:
                auc = None

            lopo_aucs[test_proj] = auc
            print(f"    {test_proj:<14} AUC={auc}")

        # Summary for this level
        valid_aucs = [v for v in lopo_aucs.values() if v is not None]
        mean_auc = round(float(np.mean(valid_aucs)), 4) if valid_aucs else None
        std_auc = round(float(np.std(valid_aucs)), 4) if valid_aucs else None

        results_per_level[level] = {
            "per_project_auc": lopo_aucs,
            "mean_auc": mean_auc,
            "std_auc": std_auc,
            "n_projects_tested": len(valid_aucs),
            "n_llm_sybils": int(len(llm_subset)),
        }
        print(f"    Mean AUC = {mean_auc} +/- {std_auc} "
              f"({len(valid_aucs)} projects)")

    return results_per_level


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("Paper 3: Batch LLM Sybil Generation & Evaluation")
    print(f"  Projects: {len(PROJECTS)}")
    print(f"  Levels: {LEVELS}")
    print(f"  N per project per level: {N_PER_PROJECT_PER_LEVEL}")
    print(f"  Total attempts: {len(PROJECTS) * len(LEVELS) * N_PER_PROJECT_PER_LEVEL}")
    print("=" * 80)

    # Initialize LLM backend with caching
    backend = LLMBackend(model="claude-opus-4-6", use_cache=True)

    # Step 1: Generate all sybils
    llm_df, gen_stats = generate_all_sybils(backend)

    if llm_df.empty:
        print("\nERROR: No sybils generated. Exiting.")
        sys.exit(1)

    # Save combined parquet
    llm_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved {len(llm_df)} sybils to {OUTPUT_PARQUET}")

    # Step 2: HasciDB rule evasion
    rule_results = analyze_rule_evasion(llm_df)

    # Step 3: LightGBM evasion
    lgbm_results = analyze_lightgbm_evasion(llm_df)

    # Step 4: Binary classifier
    binary_results = analyze_binary_classifier(llm_df)

    # ============================================================
    # FINAL REPORT
    # ============================================================
    elapsed = round(time.time() - t0, 1)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)

    print(f"\n  Generation:")
    print(f"    Total attempted:  {gen_stats['total_attempted']}")
    print(f"    Total generated:  {gen_stats['total_generated']}")
    print(f"    Success rate:     {gen_stats['overall_success_rate']:.1%}")

    print(f"\n  HasciDB Rule Evasion (all levels evade by construction):")
    for level in LEVELS:
        r = rule_results.get(level, {})
        print(f"    {level:<10}: {r.get('evasion_rate', 0):.1%} "
              f"({r.get('evade_all', 0)}/{r.get('n', 0)})")

    print(f"\n  LightGBM Evasion:")
    for level in LEVELS:
        r = lgbm_results.get(level, {})
        print(f"    {level:<10}: {r.get('evasion_rate', 0):.1%} "
              f"(detected {r.get('detected', 0)}/{r.get('n', 0)}, "
              f"mean_score={r.get('mean_sybil_score', 0):.4f})")

    print(f"\n  Binary Classifier AUC (LLM sybil vs real HasciDB):")
    for level in LEVELS:
        r = binary_results.get(level, {})
        print(f"    {level:<10}: AUC={r.get('mean_auc', 'N/A')} "
              f"+/- {r.get('std_auc', 'N/A')} "
              f"({r.get('n_projects_tested', 0)} projects)")

    print(f"\n  Elapsed: {elapsed}s")

    # Save results
    output = {
        "generation_stats": gen_stats,
        "hascidb_rule_evasion": rule_results,
        "lightgbm_evasion": lgbm_results,
        "binary_classifier": binary_results,
        "elapsed_seconds": elapsed,
        "config": {
            "n_projects": len(PROJECTS),
            "n_levels": len(LEVELS),
            "n_per_project_per_level": N_PER_PROJECT_PER_LEVEL,
            "total_attempts": len(PROJECTS) * len(LEVELS) * N_PER_PROJECT_PER_LEVEL,
            "projects": PROJECTS,
            "levels": LEVELS,
        },
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
