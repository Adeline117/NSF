#!/usr/bin/env python
"""Bootstrap 95% CIs on per-class F1 for the GBM 5-fold CV classifier."""

import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from datetime import datetime

# ── Config ─────────────────────────────────────────────────────────────
DATA_PATH = "/Users/adelinewen/NSF/paper1_onchain_agent_id/data/features_with_taxonomy.parquet"
OUT_PATH = "/Users/adelinewen/NSF/paper0_ai_agent_theory/experiments/bootstrap_per_class_ci.json"
N_BOOT = 1000
N_FOLDS = 5
SEED = 42

FEATURES = [
    "tx_interval_mean", "tx_interval_std", "tx_interval_skewness",
    "active_hour_entropy", "night_activity_ratio", "weekend_ratio",
    "burst_frequency", "gas_price_round_number_ratio",
    "gas_price_trailing_zeros_mean", "gas_limit_precision", "gas_price_cv",
    "eip1559_priority_fee_precision", "gas_price_nonce_correlation",
    "unique_contracts_ratio", "top_contract_concentration",
    "method_id_diversity", "contract_to_eoa_ratio",
    "sequential_pattern_score", "unlimited_approve_ratio",
    "approve_revoke_ratio", "unverified_contract_approve_ratio",
    "multi_protocol_interaction_count", "flash_loan_usage",
]

CLASS_NAMES = {
    0: "SimpleTradingBot",
    1: "MEVSearcher",
    2: "DeFiManagementAgent",
    3: "LLMPoweredAgent",
    6: "DeterministicScript",
}

# ── Load data ──────────────────────────────────────────────────────────
df = pd.read_parquet(DATA_PATH)
df = df[df["taxonomy_category"] != "HUMAN"].copy()
print(f"Agents: {len(df)}")

X = df[FEATURES].values
y = df["taxonomy_index"].values

# Encode labels to 0..4 for sklearn
le = LabelEncoder()
le.fit(sorted(set(y)))
y_enc = le.transform(y)
classes_orig = le.classes_  # the original taxonomy_index values

print(f"Classes (taxonomy_index): {classes_orig}")
print(f"Class counts: {dict(zip(classes_orig, np.bincount(y_enc)))}")

# ── Helper: GBM 5-fold CV producing concatenated predictions ───────────
def run_cv(X_data, y_data, seed):
    """Run 5-fold stratified CV and return (y_true, y_pred) arrays."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    y_true_all = []
    y_pred_all = []
    for train_idx, test_idx in skf.split(X_data, y_data):
        X_tr, X_te = X_data[train_idx], X_data[test_idx]
        y_tr, y_te = y_data[train_idx], y_data[test_idx]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te = scaler.transform(X_te)
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=seed,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_true_all.append(y_te)
        y_pred_all.append(y_pred)
    return np.concatenate(y_true_all), np.concatenate(y_pred_all)


# ── Bootstrap ──────────────────────────────────────────────────────────
rng = np.random.RandomState(SEED)
n = len(X)
n_classes = len(classes_orig)

# Store per-class F1 for each bootstrap iteration
f1_matrix = np.zeros((N_BOOT, n_classes))  # (1000, 5)

print(f"Running {N_BOOT} bootstrap iterations (each with {N_FOLDS}-fold CV)...")
for b in range(N_BOOT):
    if (b + 1) % 100 == 0:
        print(f"  iteration {b+1}/{N_BOOT}")
    # Bootstrap resample
    idx = rng.choice(n, size=n, replace=True)
    X_b = X[idx]
    y_b = y_enc[idx]
    # Check all classes present
    if len(set(y_b)) < n_classes:
        # Extremely unlikely but handle gracefully: re-draw
        idx = rng.choice(n, size=n, replace=True)
        X_b = X[idx]
        y_b = y_enc[idx]
    y_true, y_pred = run_cv(X_b, y_b, seed=b)
    # Per-class F1
    per_class_f1 = f1_score(y_true, y_pred, labels=list(range(n_classes)), average=None)
    f1_matrix[b] = per_class_f1

# ── Compute 95% CIs ───────────────────────────────────────────────────
results = {}
for i, cls_idx in enumerate(classes_orig):
    col = f1_matrix[:, i]
    lo = float(np.percentile(col, 2.5))
    hi = float(np.percentile(col, 97.5))
    mean = float(np.mean(col))
    std = float(np.std(col))
    cls_name = CLASS_NAMES[cls_idx]
    results[cls_name] = {
        "taxonomy_index": int(cls_idx),
        "f1_mean": round(mean, 4),
        "f1_std": round(std, 4),
        "ci_95_lower": round(lo, 4),
        "ci_95_upper": round(hi, 4),
    }
    print(f"  {cls_name:25s}  F1={mean:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

output = {
    "timestamp": datetime.now().isoformat(),
    "method": "GBM_5fold_CV_bootstrap",
    "n_bootstrap": N_BOOT,
    "n_folds": N_FOLDS,
    "n_agents": n,
    "per_class_ci": results,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
