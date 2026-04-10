#!/usr/bin/env python
"""
Bootstrap 95% CIs on per-class F1 for the GBM 5-fold CV classifier.

Strategy: Run a single stratified 5-fold CV pass to collect out-of-fold
predictions for every sample, then bootstrap-resample the (y_true, y_pred)
pairs 1000 times to compute 95% CIs on per-class F1.  This is the standard
"bootstrap the test metric" approach and avoids re-fitting the GBM 5000 times.
"""

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

le = LabelEncoder()
le.fit(sorted(set(y)))
y_enc = le.transform(y)
classes_orig = le.classes_  # original taxonomy_index values
n_classes = len(classes_orig)

print(f"Classes (taxonomy_index): {classes_orig}")

# ── Step 1: Single 5-fold CV to collect out-of-fold predictions ────────
print(f"Running {N_FOLDS}-fold stratified CV with GBM...")
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
y_true_oof = np.zeros(len(X), dtype=int)
y_pred_oof = np.zeros(len(X), dtype=int)

for fold_i, (train_idx, test_idx) in enumerate(skf.split(X, y_enc)):
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])
    clf = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=SEED,
    )
    clf.fit(X_tr, y_enc[train_idx])
    preds = clf.predict(X_te)
    y_true_oof[test_idx] = y_enc[test_idx]
    y_pred_oof[test_idx] = preds
    fold_f1 = f1_score(y_enc[test_idx], preds, labels=list(range(n_classes)), average="macro")
    print(f"  Fold {fold_i+1}: macro-F1 = {fold_f1:.4f}")

# Point estimates
point_f1 = f1_score(y_true_oof, y_pred_oof, labels=list(range(n_classes)), average=None)
print(f"\nPoint-estimate per-class F1: {point_f1}")

# ── Step 2: Bootstrap resample (y_true, y_pred) 1000 times ────────────
print(f"\nBootstrapping {N_BOOT} iterations on out-of-fold predictions...")
rng = np.random.RandomState(SEED)
n = len(y_true_oof)
f1_matrix = np.zeros((N_BOOT, n_classes))

for b in range(N_BOOT):
    idx = rng.choice(n, size=n, replace=True)
    yt = y_true_oof[idx]
    yp = y_pred_oof[idx]
    f1_matrix[b] = f1_score(yt, yp, labels=list(range(n_classes)), average=None, zero_division=0)

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
        "f1_point": round(float(point_f1[i]), 4),
        "f1_boot_mean": round(mean, 4),
        "f1_boot_std": round(std, 4),
        "ci_95_lower": round(lo, 4),
        "ci_95_upper": round(hi, 4),
    }
    print(f"  {cls_name:25s}  F1={point_f1[i]:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

output = {
    "timestamp": datetime.now().isoformat(),
    "method": "GBM_5fold_CV_then_bootstrap_predictions",
    "n_bootstrap": N_BOOT,
    "n_folds": N_FOLDS,
    "n_agents": n,
    "per_class_ci": results,
}

with open(OUT_PATH, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: {OUT_PATH}")
