"""
Feature-group ablation on OnChainAgentID v4 (N=1,147, 23 features).

Groups:
  - temporal (7): tx_interval_mean, tx_interval_std, tx_interval_skewness,
                  active_hour_entropy, night_activity_ratio, weekend_ratio, burst_frequency
  - gas (6): gas_price_round_number_ratio, gas_price_trailing_zeros_mean,
             gas_limit_precision, gas_price_cv, eip1559_priority_fee_precision,
             gas_price_nonce_correlation
  - interaction (5): unique_contracts_ratio, top_contract_concentration,
                     method_id_diversity, contract_to_eoa_ratio, sequential_pattern_score
  - approval_security (5): unlimited_approve_ratio, approve_revoke_ratio,
                           unverified_contract_approve_ratio,
                           multi_protocol_interaction_count, flash_loan_usage

For each group:
  1. Train RF using ONLY that group -> 10-fold CV AUC
  2. Train RF with that group REMOVED -> 10-fold CV AUC
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).parent.parent / "data" / "features_provenance_v4.parquet"
OUTPUT_PATH = Path(__file__).parent / "feature_group_ablation_v4_results.json"

FEATURE_GROUPS = {
    "temporal": [
        "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
        "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
        "burst_frequency"
    ],
    "gas": [
        "gas_price_round_number_ratio", "gas_price_trailing_zeros_mean",
        "gas_limit_precision", "gas_price_cv", "eip1559_priority_fee_precision",
        "gas_price_nonce_correlation"
    ],
    "interaction": [
        "unique_contracts_ratio", "top_contract_concentration",
        "method_id_diversity", "contract_to_eoa_ratio", "sequential_pattern_score"
    ],
    "approval_security": [
        "unlimited_approve_ratio", "approve_revoke_ratio",
        "unverified_contract_approve_ratio", "multi_protocol_interaction_count",
        "flash_loan_usage"
    ]
}

ALL_FEATURES = []
for feats in FEATURE_GROUPS.values():
    ALL_FEATURES.extend(feats)


def make_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
    ])


def evaluate(X, y, n_splits=10, n_repeats=10, seed=42):
    """Return mean AUC and std from repeated stratified K-fold."""
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    pipe = make_pipeline()
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(np.mean(scores)), float(np.std(scores))


def main():
    t0 = time.time()
    df = pd.read_parquet(DATA_PATH)
    X_all = df[ALL_FEATURES].values
    y = df["label"].values

    # Full model baseline
    full_auc, full_std = evaluate(X_all, y)
    print(f"Full model (all 23 features): AUC = {full_auc:.4f} +/- {full_std:.4f}")

    results = {
        "run_timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "description": "Feature-group ablation on OnChainAgentID v4 (N=1147, 23 features). "
                       "For each of 4 groups: (1) train RF on ONLY that group, "
                       "(2) train RF with that group REMOVED. 10x10 repeated stratified CV.",
        "dataset": {
            "n_samples": int(len(y)),
            "n_agents": int(y.sum()),
            "n_humans": int((y == 0).sum()),
            "n_features": len(ALL_FEATURES)
        },
        "full_model": {
            "auc": full_auc,
            "auc_std": full_std,
            "features_used": len(ALL_FEATURES)
        },
        "group_only": {},
        "group_removed": {}
    }

    for group_name, group_feats in FEATURE_GROUPS.items():
        print(f"\n--- Group: {group_name} ({len(group_feats)} features) ---")

        # Only this group
        X_only = df[group_feats].values
        auc_only, std_only = evaluate(X_only, y)
        print(f"  ONLY {group_name}: AUC = {auc_only:.4f} +/- {std_only:.4f}")
        results["group_only"][group_name] = {
            "auc": auc_only,
            "auc_std": std_only,
            "n_features": len(group_feats),
            "features": group_feats
        }

        # Remove this group
        remaining = [f for f in ALL_FEATURES if f not in group_feats]
        X_removed = df[remaining].values
        auc_removed, std_removed = evaluate(X_removed, y)
        print(f"  WITHOUT {group_name}: AUC = {auc_removed:.4f} +/- {std_removed:.4f}")
        results["group_removed"][group_name] = {
            "auc": auc_removed,
            "auc_std": std_removed,
            "n_features": len(remaining),
            "delta_vs_full": round(auc_removed - full_auc, 4)
        }

    results["elapsed_seconds"] = round(time.time() - t0, 2)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")
    print(f"Elapsed: {results['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
