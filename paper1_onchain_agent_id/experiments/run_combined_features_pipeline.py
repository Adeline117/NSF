"""
Paper 1 + Paper 3 Combined Feature Pipeline
=============================================
Joins Paper 3's 8 AI-specific features (extracted by
extract_real_ai_features.py) onto Paper 1's 23 behavioral features
to produce a 31-feature matrix. Then trains all four classifiers
(GBM, RF, LR, GNN) on the combined feature set.

Hypothesis: Paper 3's features (hour_entropy, behavioral_consistency,
response_latency_variance, etc.) carry signal that Paper 1's
features miss. Combined model should outperform Paper 1 alone on
both the leaky 3316 set and the trusted 64 set.

Outputs:
  - experiments/expanded/combined_features.parquet
  - experiments/expanded/combined_pipeline_results.json
"""

import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import LeaveOneOut, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT.parent))

FEATURES_PARQUET = PROJECT_ROOT / "data" / "features_expanded.parquet"
AI_FEATURES_JSON = (
    PROJECT_ROOT.parent / "paper3_ai_sybil" / "experiments" / "real_ai_features.json"
)
COMBINED_PARQUET = PROJECT_ROOT / "data" / "features_combined.parquet"
OUT_PATH = PROJECT_ROOT / "experiments" / "expanded" / "combined_pipeline_results.json"

P1_FEATURES = [
    "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
    "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
    "burst_frequency",
    "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
    "gas_limit_precision", "gas_price_cv",
    "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
    "unique_contracts_ratio", "top_contract_concentration",
    "method_id_diversity", "contract_to_eoa_ratio",
    "sequential_pattern_score",
    "unlimited_approve_ratio", "approve_revoke_ratio",
    "unverified_contract_approve_ratio",
    "multi_protocol_interaction_count", "flash_loan_usage",
]

P3_FEATURES = [
    "gas_price_precision",
    "hour_entropy",
    "behavioral_consistency",
    "action_sequence_perplexity",
    "error_recovery_pattern",
    "response_latency_variance",
    "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]


def load_combined() -> pd.DataFrame:
    """Load Paper 1 features + Paper 3 AI features, merged on address."""
    df = pd.read_parquet(FEATURES_PARQUET)
    df.index = df.index.str.lower()

    with open(AI_FEATURES_JSON) as f:
        ai_data = json.load(f)
    ai_features = ai_data.get("per_address", {})

    # Build a dict of {addr -> {feat -> val}}
    ai_rows = {}
    for addr, feats in ai_features.items():
        ai_rows[addr.lower()] = {
            f: feats.get(f, np.nan) for f in P3_FEATURES
        }
    ai_df = pd.DataFrame.from_dict(ai_rows, orient="index")
    ai_df.index.name = "address"

    # Join
    combined = df.join(ai_df, how="left")
    print(f"Combined: {len(combined)} rows, "
          f"{combined[P3_FEATURES].notna().sum().min()} have AI features")
    return combined


def clean(X: np.ndarray) -> np.ndarray:
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)
    return X


def cv_score(X, y, model_template, n_splits=5, n_repeats=3):
    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits, n_repeats=n_repeats, random_state=42,
    )
    aucs, f1s, accs = [], [], []
    for tr_idx, te_idx in rskf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        clf = clone(model_template)
        clf.fit(X_tr, y[tr_idx])
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(X_te)[:, 1]
        else:
            probs = clf.decision_function(X_te)
        preds = clf.predict(X_te)
        try:
            aucs.append(roc_auc_score(y[te_idx], probs))
        except ValueError:
            pass
        f1s.append(f1_score(y[te_idx], preds, zero_division=0))
        accs.append(accuracy_score(y[te_idx], preds))
    return {
        "mean_auc": round(float(np.mean(aucs)), 4) if aucs else None,
        "std_auc": round(float(np.std(aucs)), 4) if aucs else None,
        "mean_f1": round(float(np.mean(f1s)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
    }


def loo_score(X, y, model_template):
    if len(y) > 100:
        return None
    n = len(y)
    probs = np.zeros(n)
    preds = np.zeros(n, dtype=int)
    loo = LeaveOneOut()
    for tr_idx, te_idx in loo.split(X):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])
        clf = clone(model_template)
        clf.fit(X_tr, y[tr_idx])
        if hasattr(clf, "predict_proba"):
            probs[te_idx[0]] = clf.predict_proba(X_te)[0, 1]
        else:
            probs[te_idx[0]] = clf.decision_function(X_te)[0]
        preds[te_idx[0]] = clf.predict(X_te)[0]
    return {
        "auc": round(float(roc_auc_score(y, probs)), 4),
        "f1": round(float(f1_score(y, preds, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, preds)), 4),
    }


def evaluate_split(name: str, X: np.ndarray, y: np.ndarray, label: str):
    print(f"\n--- {name}: {label} (N={len(y)}, "
          f"{y.sum()} agents, {(y==0).sum()} humans) ---")
    X = clean(X.copy())

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
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.1,
            min_child_samples=3, random_state=42, verbose=-1, n_jobs=-1,
        ),
    }

    out = {}
    for m_name, m in models.items():
        cv = cv_score(X, y, m)
        loo = loo_score(X, y, m) if len(y) <= 100 else None
        out[m_name] = {"5fold_3rep": cv, "loo": loo}
        print(f"  {m_name:<10}  CV AUC={cv['mean_auc']:.4f}  "
              f"LOO AUC={(loo['auc'] if loo else 'N/A')}")
    return out


def main():
    t0 = time.time()
    print("=" * 70)
    print("Paper 1 + Paper 3 Combined Feature Pipeline")
    print("=" * 70)

    df = load_combined()
    df.to_parquet(COMBINED_PARQUET)
    print(f"Saved combined parquet to {COMBINED_PARQUET}")

    # Three feature sets
    feat_sets = {
        "p1_only_23": P1_FEATURES,
        "p3_only_8": P3_FEATURES,
        "combined_31": P1_FEATURES + P3_FEATURES,
    }

    results = {
        "timestamp": datetime.now().isoformat(),
        "feature_sets": {k: len(v) for k, v in feat_sets.items()},
        "splits": {},
    }

    # Trusted subset
    trusted_sources = {"strategy_c_human", "strategy2_paper0", "strategy_b_mev"}
    trusted_mask = df["source"].isna() | df["source"].isin(trusted_sources)
    df_trusted = df[trusted_mask]
    df_full = df

    for fset_name, features in feat_sets.items():
        print(f"\n=== Feature set: {fset_name} ({len(features)} features) ===")

        # Full set
        X_full = df_full[features].values.astype(float)
        y_full = df_full["label"].values.astype(int)
        full_res = evaluate_split(fset_name, X_full, y_full, "full_3316")

        # Trusted subset
        X_tr = df_trusted[features].values.astype(float)
        y_tr = df_trusted["label"].values.astype(int)
        trust_res = evaluate_split(fset_name, X_tr, y_tr, "trusted_64")

        results["splits"][fset_name] = {
            "full_3316": full_res,
            "trusted_64": trust_res,
        }

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    # Compute headline deltas
    delta_table = {}
    for split in ["full_3316", "trusted_64"]:
        delta_table[split] = {}
        for model in ["GBM", "RF", "LR", "LightGBM"]:
            p1 = results["splits"]["p1_only_23"][split][model]["5fold_3rep"]["mean_auc"]
            comb = results["splits"]["combined_31"][split][model]["5fold_3rep"]["mean_auc"]
            delta_table[split][model] = {
                "p1_only": p1,
                "combined": comb,
                "delta": round(comb - p1, 4),
            }
    results["headline_deltas"] = delta_table

    print("\n" + "=" * 70)
    print("HEADLINE: P1-only vs Combined-31 (delta)")
    print("=" * 70)
    print(f"{'Model':<10}  {'Split':<14}  {'P1 only':<10}  {'Combined':<10}  {'Delta':<10}")
    for split, model_dict in delta_table.items():
        for m, vals in model_dict.items():
            print(f"  {m:<10} {split:<14} {vals['p1_only']:<10.4f} "
                  f"{vals['combined']:<10.4f} {vals['delta']:+.4f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {OUT_PATH}")
    print(f"Elapsed: {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
