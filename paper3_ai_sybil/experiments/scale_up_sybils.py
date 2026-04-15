"""
Paper 3: Scale-Up LLM Sybil Generation with Confidence Intervals
================================================================
CCS/S&P Reviewer Fix: "155 sybils per tier is underpowered."

This script generates 500 sybils per tier (basic/moderate/advanced)
for 4 key projects (blur_s2, 1inch, uniswap, eigenlayer), tests
all 4 detectors on each batch, and computes bootstrap 95% CIs on
evasion rates.

4 detectors:
  1. HasciDB rules (5-indicator threshold)
  2. Pre-airdrop LightGBM (6 temporal features)
  3. Cross-axis GBM (OPS -> fund_flag)
  4. Enhanced 13-feature GBM (5 indicators + 8 AI features)

Outputs:
  - scale_up_sybils_results.json

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/scale_up_sybils.py
"""

import builtins
import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

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
    generate_sybils,
    PROJECT_LAUNCH_DATES,
)
from experiment_large_scale import (
    load_project,
    load_real_ai_calibration,
    augment_with_ai_features,
    AI_FEATURE_NAMES,
    OPS_COLS,
    FUND_COLS,
    INDICATOR_COLS,
    GBM_PARAMS,
    THRESHOLDS,
    PROJECTS as ALL_PROJECTS,
)

OUTPUT_JSON = SCRIPT_DIR / "scale_up_sybils_results.json"

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_PROJECTS = ["blur_s2", "1inch", "uniswap", "eigenlayer"]
LEVELS = ["basic", "moderate", "advanced"]
N_PER_PROJECT_PER_LEVEL = 500  # 500 per tier per project
N_BOOTSTRAP = 2000
BOOTSTRAP_CI = 0.95

# LightGBM params (from pre_airdrop_lightgbm.py / batch_llm_sybil_generation.py)
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
# BOOTSTRAP CI COMPUTATION
# ============================================================

def bootstrap_ci(data: np.ndarray, stat_fn, n_boot: int = N_BOOTSTRAP,
                 ci: float = BOOTSTRAP_CI, seed: int = 42) -> dict:
    """Compute bootstrap confidence interval for a statistic.

    Args:
        data: 1D array of values (e.g., binary evasion indicators)
        stat_fn: function to compute statistic (e.g., np.mean)
        n_boot: number of bootstrap resamples
        ci: confidence level (e.g., 0.95)
        seed: random seed

    Returns:
        dict with point estimate, ci_lower, ci_upper, ci_width
    """
    rng = np.random.RandomState(seed)
    n = len(data)
    point_estimate = float(stat_fn(data))

    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        resample = rng.choice(data, size=n, replace=True)
        boot_stats[b] = stat_fn(resample)

    alpha = (1 - ci) / 2
    ci_lower = float(np.percentile(boot_stats, 100 * alpha))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha)))

    return {
        "point": round(point_estimate, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "ci_width": round(ci_upper - ci_lower, 4),
        "n": int(n),
        "n_bootstrap": n_boot,
        "ci_level": ci,
    }


# ============================================================
# STEP 1: GENERATE SYBILS
# ============================================================

def generate_all_sybils(backend: LLMBackend) -> pd.DataFrame:
    """Generate 500 sybils per tier per project."""
    print("\n" + "=" * 80)
    print("STEP 1: GENERATING LLM SYBILS (SCALE-UP)")
    total = len(TARGET_PROJECTS) * len(LEVELS) * N_PER_PROJECT_PER_LEVEL
    print(f"  {len(TARGET_PROJECTS)} projects x {len(LEVELS)} levels x "
          f"{N_PER_PROJECT_PER_LEVEL} per = {total} total attempts")
    print("=" * 80)

    all_dfs = []
    gen_stats = {}

    for proj in TARGET_PROJECTS:
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

            n_gen = len(df)
            proj_stats[level] = {
                "attempted": N_PER_PROJECT_PER_LEVEL,
                "generated": n_gen,
                "success_rate": round(n_gen / max(N_PER_PROJECT_PER_LEVEL, 1), 4),
                "elapsed_seconds": round(elapsed, 1),
            }

            if not df.empty:
                all_dfs.append(df)

            print(f"  {proj}/{level}: {n_gen}/{N_PER_PROJECT_PER_LEVEL} "
                  f"({n_gen/max(N_PER_PROJECT_PER_LEVEL,1):.0%}) in {elapsed:.1f}s")

        gen_stats[proj] = proj_stats

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
    n_total = len(combined)
    total_attempted = len(TARGET_PROJECTS) * len(LEVELS) * N_PER_PROJECT_PER_LEVEL

    print(f"\n  TOTAL: {n_total} generated / {total_attempted} attempted "
          f"= {n_total/max(total_attempted,1):.1%}")

    return combined, gen_stats


# ============================================================
# STEP 2: TRAIN DETECTORS
# ============================================================

def train_detectors() -> dict:
    """Train all 4 detector types on the full HasciDB dataset."""
    print("\n" + "=" * 80)
    print("STEP 2: TRAINING DETECTORS")
    print("=" * 80)

    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    # Load all project data
    all_dfs = []
    all_dfs_augmented = []
    for proj in ALL_PROJECTS:
        df = load_project(proj)
        if df.empty:
            continue
        df["n_indicators"] = (
            (df["BT"] >= 5).astype(int)
            + (df["BW"] >= 10).astype(int)
            + (df["HF"] >= 0.8).astype(int)
            + (df["RF"] >= 0.5).astype(int)
            + (df["MA"] >= 5).astype(int)
        )
        all_dfs.append(df)

        aug = augment_with_ai_features(df, ai_calibration, rng)
        aug["n_indicators"] = df["n_indicators"]
        aug["is_llm_sybil"] = 0
        all_dfs_augmented.append(aug)

    pooled = pd.concat(all_dfs, ignore_index=True)
    pooled_aug = pd.concat(all_dfs_augmented, ignore_index=True)

    detectors = {}

    # Detector 1: HasciDB rules (no training needed, just threshold checks)
    detectors["hascidb_rules"] = {"type": "rules"}
    print(f"  [1] HasciDB rules: threshold-based (no training)")

    # Detector 2: Pre-airdrop LightGBM
    X_lgbm = pooled[TEMPORAL_FEATURES].fillna(0).values
    y_lgbm = pooled["is_sybil"].values.astype(int)
    clf_lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf_lgbm.fit(X_lgbm, y_lgbm)
    detectors["lightgbm"] = {"type": "lightgbm", "model": clf_lgbm}
    print(f"  [2] LightGBM: trained on {len(pooled)} rows "
          f"({int(y_lgbm.sum())} sybils)")

    # Detector 3: Cross-axis GBM (OPS -> fund_flag)
    X_cross = pooled[OPS_COLS].fillna(0).values
    y_cross = pooled["fund_flag"].values.astype(int)
    clf_cross = GradientBoostingClassifier(**GBM_PARAMS)
    clf_cross.fit(X_cross, y_cross)
    detectors["cross_axis"] = {"type": "cross_axis", "model": clf_cross}
    print(f"  [3] Cross-axis GBM (OPS->fund): trained on {len(pooled)} rows")

    # Detector 4: Enhanced 13-feature GBM (indicators + AI features)
    # Train: real (human AI features) as class 0 vs simulated LLM sybils as class 1
    from experiment_large_scale import generate_ai_sybils_calibrated
    nonsybil_df = pooled_aug[pooled_aug["is_sybil"] == 0]
    n_sybil_train = min(10000, len(nonsybil_df) // 3)
    rng_gen = np.random.RandomState(42)
    sybil_train = generate_ai_sybils_calibrated(
        n_sybil_train, ai_calibration, nonsybil_df, rng_gen, "advanced"
    )
    sybil_train["is_llm_sybil"] = 1

    all_features = INDICATOR_COLS + AI_FEATURE_NAMES
    # Subsample real for balance
    n_real_train = min(len(pooled_aug), n_sybil_train * 3)
    real_sample = pooled_aug.sample(n=n_real_train, random_state=42)

    X_enh_real = real_sample[all_features].fillna(0).values
    X_enh_sybil = sybil_train[all_features].fillna(0).values
    X_enh = np.vstack([X_enh_real, X_enh_sybil])
    y_enh = np.concatenate([np.zeros(len(X_enh_real)), np.ones(len(X_enh_sybil))])

    clf_enh = GradientBoostingClassifier(**GBM_PARAMS)
    clf_enh.fit(X_enh, y_enh)
    detectors["enhanced_13feat"] = {"type": "enhanced", "model": clf_enh}
    print(f"  [4] Enhanced 13-feature GBM: trained on {len(X_enh)} rows "
          f"({int(y_enh.sum())} sybils)")

    return detectors, pooled_aug


# ============================================================
# STEP 3: EVALUATE EVASION WITH BOOTSTRAP CIs
# ============================================================

def evaluate_detector(detector: dict, sybil_df: pd.DataFrame) -> np.ndarray:
    """Return binary array: 1 = evaded, 0 = detected."""
    dtype = detector["type"]

    if dtype == "rules":
        # HasciDB rules: evaded iff no indicator is flagged
        flagged = (
            (sybil_df["BT"] >= 5) |
            (sybil_df["BW"] >= 10) |
            (sybil_df["HF"] >= 0.8) |
            (sybil_df["RF"] >= 0.5) |
            (sybil_df["MA"] >= 5)
        )
        return (~flagged).astype(int).values

    elif dtype == "lightgbm":
        sybil_with_ni = sybil_df.copy()
        sybil_with_ni["n_indicators"] = (
            (sybil_with_ni["BT"] >= 5).astype(int)
            + (sybil_with_ni["BW"] >= 10).astype(int)
            + (sybil_with_ni["HF"] >= 0.8).astype(int)
            + (sybil_with_ni["RF"] >= 0.5).astype(int)
            + (sybil_with_ni["MA"] >= 5).astype(int)
        )
        X = sybil_with_ni[TEMPORAL_FEATURES].fillna(0).values
        probs = detector["model"].predict_proba(X)[:, 1]
        return (probs < 0.5).astype(int)  # evaded = not detected

    elif dtype == "cross_axis":
        X = sybil_df[OPS_COLS].fillna(0).values
        probs = detector["model"].predict_proba(X)[:, 1]
        return (probs < 0.5).astype(int)

    elif dtype == "enhanced":
        all_features = INDICATOR_COLS + AI_FEATURE_NAMES
        X = sybil_df[all_features].fillna(0).values
        probs = detector["model"].predict_proba(X)[:, 1]
        return (probs < 0.5).astype(int)

    else:
        raise ValueError(f"Unknown detector type: {dtype}")


def evaluate_all(sybil_df: pd.DataFrame, detectors: dict) -> dict:
    """Evaluate all detectors on sybil_df with bootstrap CIs.

    Returns nested dict: results[project][level][detector] = {point, ci_lower, ...}
    """
    print("\n" + "=" * 80)
    print("STEP 3: EVALUATING EVASION RATES WITH BOOTSTRAP 95% CIs")
    print(f"  Bootstrap resamples: {N_BOOTSTRAP}")
    print("=" * 80)

    results = {}
    detector_names = list(detectors.keys())

    for proj in TARGET_PROJECTS:
        results[proj] = {}
        for level in LEVELS:
            subset = sybil_df[(sybil_df["project"] == proj) &
                               (sybil_df["evasion_level"] == level)]
            if subset.empty:
                results[proj][level] = {
                    "n": 0,
                    "detectors": {d: {"point": None} for d in detector_names},
                }
                continue

            print(f"\n  {proj}/{level} (n={len(subset)}):")
            det_results = {}
            for det_name in detector_names:
                evaded = evaluate_detector(detectors[det_name], subset)
                ci = bootstrap_ci(evaded, np.mean)
                det_results[det_name] = ci
                print(f"    {det_name:<20}: evasion={ci['point']:.4f} "
                      f"95% CI [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] "
                      f"width={ci['ci_width']:.4f}")

            results[proj][level] = {
                "n": len(subset),
                "detectors": det_results,
            }

    return results


# ============================================================
# STEP 4: AGGREGATE STATISTICS
# ============================================================

def compute_aggregates(results: dict) -> dict:
    """Compute aggregate evasion rates across projects."""
    print("\n" + "=" * 80)
    print("STEP 4: AGGREGATE EVASION RATES")
    print("=" * 80)

    detector_names = ["hascidb_rules", "lightgbm", "cross_axis", "enhanced_13feat"]
    aggregates = {}

    for level in LEVELS:
        print(f"\n  --- {level} ---")
        level_agg = {}
        for det_name in detector_names:
            all_points = []
            total_evaded = 0
            total_n = 0
            for proj in TARGET_PROJECTS:
                pdata = results.get(proj, {}).get(level, {})
                det_data = pdata.get("detectors", {}).get(det_name, {})
                n = pdata.get("n", 0)
                point = det_data.get("point")
                if point is not None and n > 0:
                    all_points.append(point)
                    total_evaded += int(round(point * n))
                    total_n += n

            if total_n > 0:
                pooled_rate = total_evaded / total_n
                # Bootstrap on the pooled count
                pooled_binary = np.concatenate([
                    np.ones(total_evaded, dtype=int),
                    np.zeros(total_n - total_evaded, dtype=int),
                ])
                pooled_ci = bootstrap_ci(pooled_binary, np.mean)
            else:
                pooled_rate = None
                pooled_ci = {"point": None, "ci_lower": None, "ci_upper": None, "ci_width": None}

            level_agg[det_name] = {
                "pooled_evasion_rate": round(pooled_rate, 4) if pooled_rate is not None else None,
                "pooled_ci": pooled_ci,
                "per_project_rates": {
                    proj: results.get(proj, {}).get(level, {}).get("detectors", {}).get(det_name, {}).get("point")
                    for proj in TARGET_PROJECTS
                },
                "total_n": total_n,
            }
            if pooled_rate is not None:
                print(f"    {det_name:<20}: {pooled_rate:.4f} "
                      f"[{pooled_ci['ci_lower']:.4f}, {pooled_ci['ci_upper']:.4f}] "
                      f"(n={total_n})")

        aggregates[level] = level_agg

    return aggregates


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("Paper 3: Scale-Up LLM Sybil Generation + Confidence Intervals")
    print(f"  Projects: {TARGET_PROJECTS}")
    print(f"  Levels: {LEVELS}")
    print(f"  N per project per level: {N_PER_PROJECT_PER_LEVEL}")
    print(f"  Total: {len(TARGET_PROJECTS) * len(LEVELS) * N_PER_PROJECT_PER_LEVEL}")
    print(f"  Bootstrap resamples: {N_BOOTSTRAP}")
    print("=" * 80)

    # Initialize LLM backend with caching (deterministic fallback)
    backend = LLMBackend(model="claude-opus-4-6", use_cache=True)

    # Step 1: Generate sybils
    sybil_df, gen_stats = generate_all_sybils(backend)

    if sybil_df.empty:
        print("\nERROR: No sybils generated. Exiting.")
        sys.exit(1)

    print(f"\nGenerated {len(sybil_df)} total sybils")

    # Step 2: Train detectors
    detectors, real_data = train_detectors()

    # Step 3: Evaluate with bootstrap CIs
    results = evaluate_all(sybil_df, detectors)

    # Step 4: Aggregate
    aggregates = compute_aggregates(results)

    # ============================================================
    # FINAL REPORT
    # ============================================================
    elapsed = round(time.time() - t0, 1)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY TABLE")
    print("=" * 80)

    detector_names = ["hascidb_rules", "lightgbm", "cross_axis", "enhanced_13feat"]
    short_names = {"hascidb_rules": "HasciDB", "lightgbm": "LightGBM",
                   "cross_axis": "CrossAxis", "enhanced_13feat": "Enhanced13"}

    # Print table: rows = (project, level), cols = detectors
    header = f"{'Project':<12} {'Level':<10} {'N':>5}"
    for d in detector_names:
        header += f" {short_names[d]+' (95% CI)':>28}"
    print(header)
    print("-" * len(header))

    for proj in TARGET_PROJECTS:
        for level in LEVELS:
            pdata = results.get(proj, {}).get(level, {})
            n = pdata.get("n", 0)
            row = f"{proj:<12} {level:<10} {n:>5}"
            for d in detector_names:
                ci = pdata.get("detectors", {}).get(d, {})
                pt = ci.get("point")
                lo = ci.get("ci_lower")
                hi = ci.get("ci_upper")
                if pt is not None:
                    row += f" {pt:.3f} [{lo:.3f},{hi:.3f}]"
                else:
                    row += f" {'N/A':>28}"
            print(row)

    # Aggregated
    print("\nAGGREGATED (pooled across 4 projects):")
    print("-" * 80)
    for level in LEVELS:
        row = f"{'ALL':<12} {level:<10}"
        for d in detector_names:
            agg = aggregates.get(level, {}).get(d, {})
            pt = agg.get("pooled_evasion_rate")
            ci = agg.get("pooled_ci", {})
            lo = ci.get("ci_lower")
            hi = ci.get("ci_upper")
            n = agg.get("total_n", 0)
            if pt is not None:
                row += f"  {short_names[d]}: {pt:.3f} [{lo:.3f},{hi:.3f}] (n={n})"
            else:
                row += f"  {short_names[d]}: N/A"
        print(row)

    print(f"\n  Elapsed: {elapsed}s")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        "config": {
            "target_projects": TARGET_PROJECTS,
            "levels": LEVELS,
            "n_per_project_per_level": N_PER_PROJECT_PER_LEVEL,
            "n_bootstrap": N_BOOTSTRAP,
            "bootstrap_ci": BOOTSTRAP_CI,
            "detector_types": detector_names,
        },
        "generation_stats": gen_stats,
        "per_project_results": results,
        "aggregated_results": aggregates,
        "elapsed_seconds": elapsed,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
