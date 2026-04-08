"""
Paper 3 Baseline: Pre-Airdrop-Detection LightGBM
==================================================
Reimplements the pre-airdrop-detection LightGBM baseline (from
Adeline117/pre-airdrop-detection) using the HasciDB feature columns
as proxies for the original temporal feature set.

The original paper proposes a T-30 (30-day-pre-snapshot) temporal
LightGBM trained on:
  - tx count
  - active days
  - hour-of-day histogram entropy
  - protocol interaction vector
  - value distribution moments
  - gas-price percentiles

We map these to the HasciDB columns + Paper-1-derived AI features
because the raw transaction histories are not all readily available
for the 386k HasciDB addresses.

Models:
  - LightGBM with 200 trees, max_depth=6, learning_rate=0.05
  - Identical hyperparameters to the original paper

Evaluation:
  1. Train on Blur S2 sybil labels, 5-fold CV
  2. Train on Blur S2, test on (a) Blur S1, (b) other projects
  3. Test against rule-based AI sybils (advanced level)
  4. Test against LLM-generated sybils (from llm_sybil_generator.py)

Output: pre_airdrop_lightgbm_results.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

HASCIDB_DIR = (
    PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
)
OUT_PATH = (
    PROJECT_ROOT / "paper3_ai_sybil" / "experiments"
    / "pre_airdrop_lightgbm_results.json"
)

# Same hyperparameters as the original paper
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

# Pre-airdrop temporal feature proxies (using HasciDB cols)
TEMPORAL_FEATURES = ["BT", "BW", "HF", "RF", "MA", "n_indicators"]


def load_project(name: str, max_rows: int = 25000) -> pd.DataFrame:
    path = HASCIDB_DIR / f"{name}_chi26_v3.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, nrows=max_rows)
    return df


def evaluate(model, X, y) -> dict:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, probs)
    except ValueError:
        auc = float("nan")
    return {
        "auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "accuracy": round(float(accuracy_score(y, preds)), 4),
        "precision": round(float(precision_score(y, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
    }


def cv_blur_s2():
    """Train on Blur S2 with 5-fold stratified CV."""
    print("\n--- Blur S2 5-fold CV ---")
    df = load_project("blur_s2")
    print(f"  Loaded {len(df)} addresses")
    if df.empty:
        return None

    X = df[TEMPORAL_FEATURES].fillna(0).values
    y = df["is_sybil"].values.astype(int)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_results = []
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        clf = lgb.LGBMClassifier(**LGBM_PARAMS)
        clf.fit(X[tr_idx], y[tr_idx])
        m = evaluate(clf, X[te_idx], y[te_idx])
        fold_results.append(m)
        print(f"  Fold {fold_idx+1}: AUC={m['auc']:.4f} F1={m['f1']:.4f}")
    aucs = [f["auc"] for f in fold_results]
    return {
        "mean_auc": round(float(np.mean(aucs)), 4),
        "std_auc": round(float(np.std(aucs)), 4),
        "folds": fold_results,
    }


def cross_project_transfer(train: str, test_projects: list[str]) -> dict:
    """Train on one project, test on others."""
    print(f"\n--- Train: {train} ---")
    train_df = load_project(train)
    if train_df.empty:
        return None
    X_tr = train_df[TEMPORAL_FEATURES].fillna(0).values
    y_tr = train_df["is_sybil"].values.astype(int)

    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    clf.fit(X_tr, y_tr)
    print(f"  Trained on {len(train_df)} rows ({y_tr.sum()} sybils)")

    results = {}
    for test in test_projects:
        if test == train:
            continue
        test_df = load_project(test)
        if test_df.empty:
            continue
        X_te = test_df[TEMPORAL_FEATURES].fillna(0).values
        y_te = test_df["is_sybil"].values.astype(int)
        m = evaluate(clf, X_te, y_te)
        results[test] = m
        print(f"  -> {test:<14} AUC={m['auc'] or '-':<8} "
              f"F1={m['f1']:<8}")
    return results


def evaluate_against_ai_sybils(train_project: str = "blur_s2") -> dict:
    """Train pre-airdrop LightGBM, then test on synthetic AI sybils."""
    print(f"\n--- Test against synthetic AI sybils ---")
    train_df = load_project(train_project)
    if train_df.empty:
        return {}
    clf = lgb.LGBMClassifier(**LGBM_PARAMS)
    X_tr = train_df[TEMPORAL_FEATURES].fillna(0).values
    y_tr = train_df["is_sybil"].values.astype(int)
    clf.fit(X_tr, y_tr)
    print(f"  Trained on {train_project} ({len(train_df)} rows)")

    # Try to load LLM-generated sybils first
    llm_paths = list((PROJECT_ROOT / "paper3_ai_sybil" / "experiments").glob(
        "llm_sybils_*.parquet"
    ))
    results = {}

    if llm_paths:
        print(f"\n  Loading {len(llm_paths)} LLM sybil files ...")
        all_llm = []
        for p in llm_paths:
            try:
                d = pd.read_parquet(p)
                all_llm.append(d)
            except Exception:
                pass
        if all_llm:
            llm = pd.concat(all_llm, ignore_index=True)
            # Add n_indicators column
            llm["n_indicators"] = (
                (llm["BT"] >= 5).astype(int)
                + (llm["BW"] >= 10).astype(int)
                + (llm["HF"] > 0.8).astype(int)
                + (llm["RF"] > 0.5).astype(int)
                + (llm["MA"] >= 5).astype(int)
            )
            X_llm = llm[TEMPORAL_FEATURES].fillna(0).values
            probs = clf.predict_proba(X_llm)[:, 1]
            detected = int((probs >= 0.5).sum())
            results["llm_sybils"] = {
                "n": int(len(llm)),
                "detected": detected,
                "evasion_rate": round(1 - detected / max(len(llm), 1), 4),
                "mean_score": round(float(probs.mean()), 4),
            }
            print(f"  LLM sybils:    n={len(llm)}  "
                  f"evasion={1 - detected/len(llm):.2%}")

    # Synthetic rule-based AI sybils (basic / moderate / advanced)
    sys.path.insert(0, str(PROJECT_ROOT / "paper3_ai_sybil" / "experiments"))
    from experiment_large_scale import (
        generate_ai_sybils_calibrated,
        load_real_ai_calibration,
    )

    nonsybil_train = train_df[train_df["is_sybil"] == 0]
    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    for level in ["basic", "moderate", "advanced"]:
        sybils = generate_ai_sybils_calibrated(
            3000, ai_calibration, nonsybil_train, rng, level,
        )
        sybils["n_indicators"] = (
            (sybils["BT"] >= 5).astype(int)
            + (sybils["BW"] >= 10).astype(int)
            + (sybils["HF"] > 0.8).astype(int)
            + (sybils["RF"] > 0.5).astype(int)
            + (sybils["MA"] >= 5).astype(int)
        )
        X_syn = sybils[TEMPORAL_FEATURES].fillna(0).values
        probs = clf.predict_proba(X_syn)[:, 1]
        detected = int((probs >= 0.5).sum())
        results[f"rule_based_{level}"] = {
            "n": int(len(sybils)),
            "detected": detected,
            "evasion_rate": round(1 - detected / max(len(sybils), 1), 4),
            "mean_score": round(float(probs.mean()), 4),
        }
        print(f"  Rule {level:<8}: n={len(sybils)}  "
              f"evasion={1 - detected/len(sybils):.2%}")

    return results


def main():
    t0 = time.time()
    print("=" * 70)
    print("Paper 3 Baseline: Pre-Airdrop-Detection LightGBM")
    print("=" * 70)

    results = {
        "model": "LightGBM",
        "params": LGBM_PARAMS,
        "features": TEMPORAL_FEATURES,
    }

    # 1. 5-fold CV on Blur S2 (the canonical test from the paper)
    results["blur_s2_5fold"] = cv_blur_s2()

    # 2. Cross-project transfer
    results["cross_project_blur_s2"] = cross_project_transfer(
        "blur_s2", ["blur_s1", "uniswap", "1inch", "ens", "looksrare"],
    )

    # 3. AI sybil evasion test
    results["ai_sybil_evasion"] = evaluate_against_ai_sybils("blur_s2")

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")
    print(f"Elapsed: {results['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
