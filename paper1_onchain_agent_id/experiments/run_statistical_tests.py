"""
Paper 1: Statistical Significance Tests
==========================================
Runs paired-sample significance tests comparing classifiers on the
provenance-only labeled dataset (N=64, 33 agents + 31 humans).

Tests:
  1. McNemar's test on LOO predictions — GBM vs LR, GBM vs RF,
     GBM vs single-feature baseline, GBM vs majority heuristic.
  2. 10k bootstrap CI on AUC differences between pairs.
  3. DeLong's test on ROC AUC differences.
  4. Precision@k and recall@k sweeps (k = 5, 10, 20).
  5. Calibration curve + Brier score for each model.

All metrics are computed using LOO-CV on the provenance-only set
since that is the honest evaluation (after B1 leakage fix).
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

FEATURES_PARQUET = (
    PROJECT_ROOT / "data" / "features_expanded.parquet"
)
OUT_PATH = (
    PROJECT_ROOT / "experiments" / "expanded" / "statistical_tests.json"
)


def load_provenance_trusted() -> tuple[np.ndarray, np.ndarray, list[str]]:
    df = pd.read_parquet(FEATURES_PARQUET)
    trusted_sources = {"strategy_c_human", "strategy2_paper0", "strategy_b_mev"}
    trusted_mask = df["source"].isna() | df["source"].isin(trusted_sources)
    df_trusted = df[trusted_mask].copy()

    exclude = {"label", "name", "source", "c1c4_confidence", "n_transactions"}
    feature_cols = [c for c in df_trusted.columns if c not in exclude]
    X = df_trusted[feature_cols].values.astype(float)
    y = df_trusted["label"].values.astype(int)

    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]

    # Clip extremes
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)

    return X, y, feature_cols


def loo_predict(X: np.ndarray, y: np.ndarray, model) -> tuple[np.ndarray, np.ndarray]:
    """Return LOO (predictions, probabilities)."""
    n = len(y)
    preds = np.zeros(n, dtype=int)
    probs = np.zeros(n)
    loo = LeaveOneOut()
    for tr_idx, te_idx in loo.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        clf = clone(model)
        clf.fit(X_tr, y[tr_idx])
        preds[te_idx[0]] = clf.predict(X_te)[0]
        if hasattr(clf, "predict_proba"):
            probs[te_idx[0]] = clf.predict_proba(X_te)[0, 1]
        else:
            probs[te_idx[0]] = clf.decision_function(X_te)[0]
    return preds, probs


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict:
    """McNemar's test: A correct & B wrong vs A wrong & B correct."""
    correct_a = (pred_a == y)
    correct_b = (pred_b == y)

    b = int(np.sum(correct_a & ~correct_b))  # A correct, B wrong
    c = int(np.sum(~correct_a & correct_b))  # A wrong, B correct

    # Exact binomial when b+c small
    n = b + c
    if n == 0:
        return {"b": 0, "c": 0, "p_value": 1.0, "significant": False}
    if n < 25:
        # Exact binomial test: probability of getting <=min(b,c) successes out of n
        k = min(b, c)
        p_value = 2 * scipy_stats.binom.cdf(k, n, 0.5)
        p_value = min(p_value, 1.0)
    else:
        # Use chi-square with continuity correction
        chi2 = (abs(b - c) - 1) ** 2 / n
        p_value = 1 - scipy_stats.chi2.cdf(chi2, df=1)

    return {
        "b_A_correct_B_wrong": b,
        "c_A_wrong_B_correct": c,
        "p_value": round(float(p_value), 6),
        "significant_0.05": bool(p_value < 0.05),
        "significant_0.01": bool(p_value < 0.01),
    }


def bootstrap_auc_diff(
    y: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
    n_boot: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """Bootstrap CI on AUC(A) - AUC(B)."""
    rng = np.random.RandomState(42)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            auc_a = roc_auc_score(y_b, probs_a[idx])
            auc_b = roc_auc_score(y_b, probs_b[idx])
            diffs.append(auc_a - auc_b)
        except ValueError:
            pass

    diffs = np.array(diffs)
    mean_diff = float(np.mean(diffs))
    lo = float(np.percentile(diffs, 100 * alpha / 2))
    hi = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    return {
        "mean_auc_diff": round(mean_diff, 4),
        "ci_95": [round(lo, 4), round(hi, 4)],
        "n_bootstrap": len(diffs),
        "significant_ci_excludes_0": bool(lo > 0 or hi < 0),
    }


def delong_auc_variance(y: np.ndarray, probs: np.ndarray) -> float:
    """DeLong's AUC variance estimator (single model)."""
    pos = probs[y == 1]
    neg = probs[y == 0]
    m, n = len(pos), len(neg)
    if m == 0 or n == 0:
        return float("nan")

    # Placements
    # V10[i] = (1/n) * sum_j I(pos[i] > neg[j]) + 0.5*I(pos[i] == neg[j])
    V10 = np.zeros(m)
    for i in range(m):
        V10[i] = (np.sum(pos[i] > neg) + 0.5 * np.sum(pos[i] == neg)) / n
    V01 = np.zeros(n)
    for j in range(n):
        V01[j] = (np.sum(pos > neg[j]) + 0.5 * np.sum(pos == neg[j])) / m

    auc = roc_auc_score(y, probs)
    s10 = float(np.var(V10, ddof=1))
    s01 = float(np.var(V01, ddof=1))
    var = s10 / m + s01 / n
    return var


def delong_test(
    y: np.ndarray,
    probs_a: np.ndarray,
    probs_b: np.ndarray,
) -> dict:
    """Paired DeLong test via bootstrap on paired differences."""
    auc_a = roc_auc_score(y, probs_a)
    auc_b = roc_auc_score(y, probs_b)
    diff = auc_a - auc_b

    # Paired bootstrap variance
    rng = np.random.RandomState(42)
    n = len(y)
    diffs = []
    for _ in range(5000):
        idx = rng.choice(n, n, replace=True)
        y_b = y[idx]
        if len(np.unique(y_b)) < 2:
            continue
        try:
            diffs.append(
                roc_auc_score(y_b, probs_a[idx])
                - roc_auc_score(y_b, probs_b[idx])
            )
        except ValueError:
            pass

    diffs = np.array(diffs)
    std_diff = float(np.std(diffs))
    if std_diff == 0:
        return {
            "auc_a": round(auc_a, 4),
            "auc_b": round(auc_b, 4),
            "diff": round(diff, 4),
            "std_diff_bootstrap": 0.0,
            "z_statistic": float("nan"),
            "p_value": 1.0,
            "significant_0.05": False,
        }
    z = diff / std_diff
    p_val = 2 * (1 - scipy_stats.norm.cdf(abs(z)))

    return {
        "auc_a": round(auc_a, 4),
        "auc_b": round(auc_b, 4),
        "diff": round(diff, 4),
        "std_diff_bootstrap": round(std_diff, 4),
        "z_statistic": round(float(z), 4),
        "p_value": round(float(p_val), 6),
        "significant_0.05": bool(p_val < 0.05),
        "significant_0.01": bool(p_val < 0.01),
    }


def precision_recall_at_k(
    y: np.ndarray, probs: np.ndarray, ks: list[int],
) -> dict:
    """Precision@k and Recall@k."""
    order = np.argsort(-probs)
    results = {}
    n_pos = int(y.sum())
    for k in ks:
        if k > len(y):
            continue
        top_k = order[:k]
        tp = int(y[top_k].sum())
        precision = tp / k
        recall = tp / max(n_pos, 1)
        results[f"k={k}"] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "tp": tp,
        }
    return results


def main():
    print("=" * 70)
    print("Paper 1: Statistical Significance Tests")
    print("=" * 70)

    X, y, feat_names = load_provenance_trusted()
    print(f"Loaded N={len(y)} trusted samples ({int(y.sum())} agents, "
          f"{int((y == 0).sum())} humans), {len(feat_names)} features")

    models = {
        "GBM": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=3, random_state=42,
        ),
        "RF": RandomForestClassifier(
            n_estimators=200, max_depth=4, min_samples_leaf=3,
            random_state=42, n_jobs=-1,
        ),
        "LR": LogisticRegression(C=1.0, max_iter=2000, random_state=42),
    }

    # LOO predictions for each model
    print("\nComputing LOO predictions for each model ...")
    loo_preds = {}
    loo_probs = {}
    for name, m in models.items():
        preds, probs = loo_predict(X, y, m)
        loo_preds[name] = preds
        loo_probs[name] = probs
        auc = roc_auc_score(y, probs)
        acc = (preds == y).mean()
        print(f"  {name:<5} AUC={auc:.4f}  Acc={acc:.4f}")

    # Single-best-feature baseline
    # Find best univariate feature
    best_feat_idx = 0
    best_feat_auc = 0.5
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.std(col) == 0:
            continue
        auc = roc_auc_score(y, col)
        auc = max(auc, 1 - auc)
        if auc > best_feat_auc:
            best_feat_auc = auc
            best_feat_idx = j

    print(f"\nBest univariate feature: {feat_names[best_feat_idx]} "
          f"(AUC={best_feat_auc:.4f})")
    X_single = X[:, [best_feat_idx]]
    lr_single = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    single_preds, single_probs = loo_predict(X_single, y, lr_single)
    # Flip if AUC < 0.5 (feature anti-correlated with label)
    if roc_auc_score(y, single_probs) < 0.5:
        single_probs = 1.0 - single_probs
        single_preds = 1 - single_preds
    loo_preds["SingleBestFeature"] = single_preds
    loo_probs["SingleBestFeature"] = single_probs
    print(f"  Single: AUC={roc_auc_score(y, single_probs):.4f}")

    # Majority-class baseline
    majority = int(y.sum() > (len(y) - y.sum()))
    loo_preds["Majority"] = np.full_like(y, majority)
    loo_probs["Majority"] = np.full_like(y, float(majority), dtype=float)

    results = {
        "timestamp": datetime.now().isoformat(),
        "n_samples": int(len(y)),
        "n_agents": int(y.sum()),
        "n_humans": int((y == 0).sum()),
        "best_univariate_feature": {
            "name": feat_names[best_feat_idx],
            "auc": round(float(best_feat_auc), 4),
        },
        "model_summary": {},
        "mcnemar_tests": {},
        "bootstrap_auc_diffs": {},
        "delong_tests": {},
        "precision_recall_at_k": {},
        "calibration": {},
    }

    # Per-model summary
    for name in loo_preds:
        preds = loo_preds[name]
        probs = loo_probs[name]
        try:
            auc = roc_auc_score(y, probs)
        except ValueError:
            auc = float("nan")
        results["model_summary"][name] = {
            "auc": round(float(auc), 4) if not np.isnan(auc) else None,
            "accuracy": round(float((preds == y).mean()), 4),
            "precision_agent": round(float(precision_score(
                y, preds, pos_label=1, zero_division=0)), 4),
            "recall_agent": round(float(recall_score(
                y, preds, pos_label=1, zero_division=0)), 4),
            "precision_human": round(float(precision_score(
                y, preds, pos_label=0, zero_division=0)), 4),
            "recall_human": round(float(recall_score(
                y, preds, pos_label=0, zero_division=0)), 4),
        }

    # McNemar tests
    print("\nMcNemar tests (vs GBM):")
    for opponent in ["RF", "LR", "SingleBestFeature", "Majority"]:
        m = mcnemar(y, loo_preds["GBM"], loo_preds[opponent])
        results["mcnemar_tests"][f"GBM_vs_{opponent}"] = m
        print(f"  GBM vs {opponent:<20} b={m['b_A_correct_B_wrong']:>3} "
              f"c={m['c_A_wrong_B_correct']:>3} p={m['p_value']:.4f} "
              f"sig={'YES' if m['significant_0.05'] else 'no'}")

    # Bootstrap CI on AUC diffs
    print("\nBootstrap CI on AUC differences (10000 samples):")
    for opponent in ["RF", "LR", "SingleBestFeature"]:
        bs = bootstrap_auc_diff(y, loo_probs["GBM"], loo_probs[opponent])
        results["bootstrap_auc_diffs"][f"GBM_vs_{opponent}"] = bs
        print(f"  GBM vs {opponent:<20} mean_diff={bs['mean_auc_diff']:+.4f} "
              f"CI95={bs['ci_95']}")

    # DeLong tests
    print("\nDeLong tests (paired bootstrap on AUC):")
    for opponent in ["RF", "LR", "SingleBestFeature"]:
        dl = delong_test(y, loo_probs["GBM"], loo_probs[opponent])
        results["delong_tests"][f"GBM_vs_{opponent}"] = dl
        print(f"  GBM vs {opponent:<20} z={dl['z_statistic']:+.3f} "
              f"p={dl['p_value']:.4f}")

    # Precision/Recall @ k
    print("\nPrecision/Recall @ k (GBM):")
    pr = precision_recall_at_k(y, loo_probs["GBM"], ks=[5, 10, 20, 30])
    results["precision_recall_at_k"]["GBM"] = pr
    for k_lbl, vals in pr.items():
        print(f"  {k_lbl:<8} P={vals['precision']:.4f} R={vals['recall']:.4f}")

    # Calibration + Brier
    print("\nCalibration (Brier score, lower is better):")
    for name, probs in loo_probs.items():
        if name == "Majority":
            continue
        try:
            brier = brier_score_loss(y, probs)
        except Exception:
            brier = float("nan")
        try:
            frac_pos, mean_pred = calibration_curve(
                y, probs, n_bins=5, strategy="uniform",
            )
        except Exception:
            frac_pos, mean_pred = np.array([]), np.array([])
        results["calibration"][name] = {
            "brier_score": round(float(brier), 4)
            if not np.isnan(brier) else None,
            "curve": {
                "mean_pred": mean_pred.tolist(),
                "frac_positive": frac_pos.tolist(),
            },
        }
        print(f"  {name:<20} Brier={brier:.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
