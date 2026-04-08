"""
Paper 1: Cross-Platform Generalization Evaluation
====================================================
Tests whether the classifier generalizes across agent platforms.

The expanded mining dataset is 97% concentrated in Autonolas/OLAS
(1771 OLAS interactors), 28% Fetch.ai (923), and 17% AI Arena (549).
The provenance-trusted set (n=64) is more diverse but small.

Splits:
  Split 1: Train on Autonolas/OLAS agents, test on (MEV + Paper0 + named humans)
  Split 2: Train on (MEV bots + Paper0 + named humans), test on Autonolas
  Split 3: Train on Fetch.ai, test on AI Arena (cross-LLM-project)

Since Autonolas agents were labeled via C1-C4 (leaky), these splits
inherit some C1-C4 bias. The TRUSTED split 2 is the most honest.

Outputs:
  - experiments/expanded/cross_platform_eval.json
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_PARQUET = PROJECT_ROOT / "data" / "features_expanded.parquet"
OUT_PATH = PROJECT_ROOT / "experiments" / "expanded" / "cross_platform_eval.json"

FEATURE_COLS = [
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


def clean(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLS].values.astype(float)
    y = df["label"].values.astype(int)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        col_medians = np.nanmedian(X, axis=0)
        col_medians = np.nan_to_num(col_medians, nan=0.0)
        for j in range(X.shape[1]):
            X[nan_mask[:, j], j] = col_medians[j]
    for j in range(X.shape[1]):
        lo, hi = np.nanpercentile(X[:, j], [1, 99])
        X[:, j] = np.clip(X[:, j], lo, hi)
    return X, y


def evaluate(model, X_tr, y_tr, X_te, y_te) -> dict:
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    clf = clone(model)
    clf.fit(X_tr_s, y_tr)
    y_pred = clf.predict(X_te_s)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_te_s)[:, 1]
    else:
        y_prob = clf.decision_function(X_te_s)

    try:
        auc = roc_auc_score(y_te, y_prob)
    except ValueError:
        auc = float("nan")

    return {
        "auc": round(float(auc), 4) if not np.isnan(auc) else None,
        "accuracy": round(float(accuracy_score(y_te, y_pred)), 4),
        "precision": round(float(precision_score(y_te, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_te, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_te, y_pred, zero_division=0)), 4),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "n_train_agent": int(y_tr.sum()),
        "n_test_agent": int(y_te.sum()),
    }


def main():
    print("=" * 70)
    print("Paper 1: Cross-Platform Generalization Evaluation")
    print("=" * 70)

    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"Loaded {len(df)} rows")

    # Extract contract source from name
    def parse_src(name):
        if pd.isna(name):
            return "pilot_manual"
        if "Autonolas" in str(name) or "OLAS" in str(name):
            return "Autonolas"
        if "Fetch.ai" in str(name) or "FET Token" in str(name):
            return "Fetch.ai"
        if "AI Arena" in str(name) or "NRN Token" in str(name):
            return "AI_Arena"
        if any(t in str(name).lower() for t in ["mev", "flashbots", "jared",
                                                  "sandwich", "builder",
                                                  "beaverbuild", "wintermute",
                                                  "searcher"]):
            return "MEV_curated"
        if "Paper0" in str(name) or "paper0" in str(name):
            return "Paper0_validated"
        if any(t in str(name).lower() for t in [".eth", "ens", "vitalik",
                                                 "dev", "founder", "a16z",
                                                 "sbf", "coinbase"]):
            return "ENS_human"
        return "other"

    df["platform"] = df["name"].apply(parse_src)
    print("\nPlatform distribution:")
    for plat, cnt in df["platform"].value_counts().items():
        by_label = df[df["platform"] == plat]["label"].value_counts().to_dict()
        print(f"  {plat:<20} {cnt:>5}  agent={by_label.get(1, 0)} human={by_label.get(0, 0)}")

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

    splits = {}

    # Split 1: Autonolas agents (C1-C4 gated) vs trusted humans
    auto_train = df[df["platform"] == "Autonolas"]
    trust_test = df[df["platform"].isin(["MEV_curated", "Paper0_validated",
                                          "ENS_human", "pilot_manual"])]
    if len(auto_train) > 0 and len(trust_test) > 0:
        X_tr, y_tr = clean(auto_train)
        X_te, y_te = clean(trust_test)
        splits["auto_train_to_trusted_test"] = {
            "description": "Train on Autonolas (C1-C4 labeled), "
                           "test on trusted provenance set",
            "models": {name: evaluate(m, X_tr, y_tr, X_te, y_te)
                       for name, m in models.items()},
        }

    # Split 2: Trusted -> Autonolas (the honest direction)
    if len(trust_test) > 0 and len(auto_train) > 0:
        X_tr, y_tr = clean(trust_test)
        X_te, y_te = clean(auto_train)
        splits["trusted_train_to_auto_test"] = {
            "description": "Train on trusted provenance set, "
                           "test on Autonolas (C1-C4 labeled)",
            "models": {name: evaluate(m, X_tr, y_tr, X_te, y_te)
                       for name, m in models.items()},
        }

    # Split 3: Fetch.ai -> AI Arena
    fet = df[df["platform"] == "Fetch.ai"]
    ai_ar = df[df["platform"] == "AI_Arena"]
    if len(fet) > 0 and len(ai_ar) > 0:
        X_tr, y_tr = clean(fet)
        X_te, y_te = clean(ai_ar)
        splits["fetch_to_ai_arena"] = {
            "description": "Train on Fetch.ai holders, test on AI Arena "
                           "holders (both C1-C4 labeled)",
            "models": {name: evaluate(m, X_tr, y_tr, X_te, y_te)
                       for name, m in models.items()},
        }

    # Split 4: AI Arena -> Fetch.ai
    if len(ai_ar) > 0 and len(fet) > 0:
        X_tr, y_tr = clean(ai_ar)
        X_te, y_te = clean(fet)
        splits["ai_arena_to_fetch"] = {
            "description": "Train on AI Arena, test on Fetch.ai "
                           "(both C1-C4 labeled)",
            "models": {name: evaluate(m, X_tr, y_tr, X_te, y_te)
                       for name, m in models.items()},
        }

    # Print summary
    print("\n" + "=" * 80)
    print(f"{'Split':<35} {'Model':<6} {'n_tr':<6} {'n_te':<6} "
          f"{'AUC':<7} {'F1':<7}")
    print("=" * 80)
    for split_name, split_data in splits.items():
        for m_name, m_stats in split_data["models"].items():
            print(f"{split_name:<35} {m_name:<6} "
                  f"{m_stats['n_train']:<6} {m_stats['n_test']:<6} "
                  f"{m_stats['auc'] or '-':<7} {m_stats['f1']:<7}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "platform_distribution": {
            str(k): int(v) for k, v in df["platform"].value_counts().items()
        },
        "splits": splits,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_PATH}")


if __name__ == "__main__":
    main()
