#!/usr/bin/env python
"""
Paper 0: Decision-Process Feature Experiment
=============================================
Extracts 4 new feature families (16 features) from raw transaction data
that capture *decision-process* signals — how an address reacts to errors,
structures sessions, manages nonces, and tunes gas prices.

Feature Families:
  1. Error Recovery (5 features)   — from isError field
  2. Session Patterns (5 features) — from timestamps
  3. Nonce Patterns (3 features)   — from nonce field
  4. Gas Microstructure (3 features) — from gasPrice

Experiment:
  Condition A: Original 23 features (baseline)
  Condition B: 23 + 16 decision-process = 39 features
  Condition C: 31 (23 + 8 AI) + 16 = 47 features (full)

All conditions use 10-fold stratified CV with GBM, per-class F1, and
feature importance ranking.

Outputs:
  - decision_process_results.json
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Paths ─────────────────────────────────────────────────────────────
RAW_DIR = PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "raw"
FEATURES_PARQUET = (
    PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
)
AI_FEATURES_JSON = (
    PROJECT_ROOT / "paper3_ai_sybil" / "experiments" / "real_ai_features.json"
)
RESULTS_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "decision_process_results.json"
)

# ── Feature column definitions ────────────────────────────────────────
ORIGINAL_23 = [
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

AI_8 = [
    "gas_price_precision", "hour_entropy", "behavioral_consistency",
    "action_sequence_perplexity", "error_recovery_pattern",
    "response_latency_variance", "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]

DECISION_PROCESS_16 = [
    # Family 1: Error Recovery
    "error_rate",
    "post_error_delay_median",
    "post_error_method_change_rate",
    "post_error_target_change_rate",
    "error_abandon_rate",
    # Family 2: Session Patterns
    "session_count",
    "session_length_mean",
    "session_length_cv",
    "intra_session_tx_rate",
    "inter_session_gap_cv",
    # Family 3: Nonce Patterns
    "nonce_sequential_rate",
    "nonce_gap_max",
    "nonce_completeness",
    # Family 4: Gas Microstructure
    "gas_price_unique_ratio",
    "gas_price_autocorrelation",
    "gas_price_cluster_count",
]

TAXONOMY_NAMES = {
    0: "SimpleTradingBot",
    1: "MEVSearcher",
    2: "DeFiManagementAgent",
    3: "LLMPoweredAgent",
    4: "AutonomousDAOAgent",
    5: "CrossChainBridgeAgent",
    6: "DeterministicScript",
    7: "RLTradingAgent",
}

N_FOLDS = 10
N_BOOT = 1000
SEED = 42


# ═══════════════════════════════════════════════════════════════════════
# Feature Extraction
# ═══════════════════════════════════════════════════════════════════════

def _safe_cv(arr):
    """Coefficient of variation, safe for empty/zero arrays."""
    if len(arr) < 2:
        return 0.0
    m = np.mean(arr)
    if m == 0:
        return 0.0
    return float(np.std(arr) / m)


def extract_error_recovery(df_tx):
    """Family 1: Error Recovery features."""
    n = len(df_tx)
    if n == 0:
        return {
            "error_rate": 0.0,
            "post_error_delay_median": np.nan,
            "post_error_method_change_rate": np.nan,
            "post_error_target_change_rate": np.nan,
            "error_abandon_rate": np.nan,
        }

    # Ensure isError is numeric
    is_error = pd.to_numeric(df_tx["isError"], errors="coerce").fillna(0).astype(int)
    timestamps = pd.to_numeric(df_tx["timeStamp"], errors="coerce").values
    methods = df_tx["methodId"].astype(str).values
    targets = df_tx["to"].astype(str).values

    error_mask = is_error.values == 1
    n_errors = int(error_mask.sum())
    error_rate = n_errors / n

    if n_errors == 0:
        return {
            "error_rate": 0.0,
            "post_error_delay_median": np.nan,
            "post_error_method_change_rate": np.nan,
            "post_error_target_change_rate": np.nan,
            "error_abandon_rate": np.nan,
        }

    error_indices = np.where(error_mask)[0]
    post_delays = []
    method_changes = 0
    target_changes = 0
    abandoned = 0
    valid_retries = 0

    for idx in error_indices:
        if idx + 1 < n:
            delay = timestamps[idx + 1] - timestamps[idx]
            if delay > 3600:
                abandoned += 1
            else:
                post_delays.append(delay)
                valid_retries += 1
                if methods[idx + 1] != methods[idx]:
                    method_changes += 1
                if targets[idx + 1] != targets[idx]:
                    target_changes += 1
        else:
            # Last tx was error, no subsequent tx
            abandoned += 1

    return {
        "error_rate": error_rate,
        "post_error_delay_median": float(np.median(post_delays)) if post_delays else np.nan,
        "post_error_method_change_rate": method_changes / valid_retries if valid_retries > 0 else np.nan,
        "post_error_target_change_rate": target_changes / valid_retries if valid_retries > 0 else np.nan,
        "error_abandon_rate": abandoned / n_errors if n_errors > 0 else np.nan,
    }


def extract_session_patterns(df_tx):
    """Family 2: Session Patterns features."""
    n = len(df_tx)
    if n < 2:
        return {
            "session_count": 1 if n == 1 else 0,
            "session_length_mean": 0.0,
            "session_length_cv": 0.0,
            "intra_session_tx_rate": 0.0,
            "inter_session_gap_cv": 0.0,
        }

    timestamps = pd.to_numeric(df_tx["timeStamp"], errors="coerce").values
    gaps = np.diff(timestamps)

    # Session boundary = gap > 1800 seconds (30 minutes)
    session_boundary = gaps > 1800
    session_ids = np.cumsum(np.concatenate([[0], session_boundary.astype(int)]))
    n_sessions = int(session_ids[-1] + 1)

    # Session lengths and tx counts
    session_lengths = []
    session_tx_counts = []
    for sid in range(n_sessions):
        mask = session_ids == sid
        ts_in_session = timestamps[mask]
        session_len = float(ts_in_session[-1] - ts_in_session[0])
        session_lengths.append(session_len)
        session_tx_counts.append(int(mask.sum()))

    session_lengths = np.array(session_lengths)
    session_tx_counts = np.array(session_tx_counts)

    # Intra-session tx rate (txs per minute, excluding single-tx sessions)
    tx_rates = []
    for sl, stc in zip(session_lengths, session_tx_counts):
        if sl > 0 and stc > 1:
            tx_rates.append(stc / (sl / 60.0))
    intra_rate = float(np.mean(tx_rates)) if tx_rates else 0.0

    # Inter-session gaps
    if n_sessions > 1:
        session_end_times = []
        session_start_times = []
        for sid in range(n_sessions):
            mask = session_ids == sid
            ts_in_session = timestamps[mask]
            session_start_times.append(ts_in_session[0])
            session_end_times.append(ts_in_session[-1])
        inter_gaps = [
            session_start_times[i + 1] - session_end_times[i]
            for i in range(n_sessions - 1)
        ]
        inter_gap_cv = _safe_cv(inter_gaps)
    else:
        inter_gap_cv = 0.0

    return {
        "session_count": n_sessions,
        "session_length_mean": float(np.mean(session_lengths)),
        "session_length_cv": _safe_cv(session_lengths),
        "intra_session_tx_rate": intra_rate,
        "inter_session_gap_cv": inter_gap_cv,
    }


def extract_nonce_patterns(df_tx):
    """Family 3: Nonce Patterns features."""
    n = len(df_tx)
    if n < 2:
        return {
            "nonce_sequential_rate": 1.0 if n == 1 else np.nan,
            "nonce_gap_max": 0,
            "nonce_completeness": 1.0 if n == 1 else np.nan,
        }

    nonces = pd.to_numeric(df_tx["nonce"], errors="coerce").dropna().astype(int).values
    if len(nonces) < 2:
        return {
            "nonce_sequential_rate": np.nan,
            "nonce_gap_max": 0,
            "nonce_completeness": np.nan,
        }

    diffs = np.diff(nonces)
    sequential_rate = float(np.mean(diffs == 1))
    gap_max = int(np.max(np.abs(diffs))) if len(diffs) > 0 else 0

    max_nonce = int(np.max(nonces))
    n_txs = len(nonces)
    # nonce_completeness: if max_nonce is 0, we only have nonce=0 txs
    if max_nonce > 0:
        completeness = n_txs / (max_nonce + 1)
    else:
        completeness = 1.0

    return {
        "nonce_sequential_rate": sequential_rate,
        "nonce_gap_max": gap_max,
        "nonce_completeness": min(completeness, 2.0),  # cap at 2.0 (possible with reorgs)
    }


def extract_gas_microstructure(df_tx):
    """Family 4: EIP-1559 Gas Microstructure features."""
    n = len(df_tx)
    if n < 2:
        return {
            "gas_price_unique_ratio": 1.0 if n == 1 else np.nan,
            "gas_price_autocorrelation": np.nan,
            "gas_price_cluster_count": 1 if n == 1 else 0,
        }

    gas_prices = pd.to_numeric(df_tx["gasPrice"], errors="coerce").dropna().values.astype(float)
    if len(gas_prices) < 2:
        return {
            "gas_price_unique_ratio": np.nan,
            "gas_price_autocorrelation": np.nan,
            "gas_price_cluster_count": 0,
        }

    n_gp = len(gas_prices)

    # Unique ratio
    unique_ratio = len(np.unique(gas_prices)) / n_gp

    # Lag-1 autocorrelation
    gp_mean = np.mean(gas_prices)
    gp_std = np.std(gas_prices)
    if gp_std > 0 and n_gp >= 3:
        gp_centered = gas_prices - gp_mean
        autocorr = float(np.sum(gp_centered[:-1] * gp_centered[1:]) / ((n_gp - 1) * gp_std ** 2))
        autocorr = np.clip(autocorr, -1.0, 1.0)
    else:
        autocorr = 0.0

    # Gas price cluster count via k-means with silhouette
    n_unique = len(np.unique(gas_prices))
    if n_unique <= 1:
        cluster_count = 1
    elif n_unique <= 3 or n_gp < 10:
        cluster_count = min(n_unique, 2)
    else:
        gp_reshaped = gas_prices.reshape(-1, 1)
        # Subsample for speed if too many
        if n_gp > 2000:
            rng = np.random.RandomState(SEED)
            idx = rng.choice(n_gp, 2000, replace=False)
            gp_reshaped = gp_reshaped[idx]

        # Normalize for k-means
        gp_norm = (gp_reshaped - np.mean(gp_reshaped)) / (np.std(gp_reshaped) + 1e-12)

        max_k = min(8, n_unique, len(gp_norm) - 1)
        if max_k < 2:
            cluster_count = 1
        else:
            best_k = 1
            best_score = -1
            from sklearn.metrics import silhouette_score
            for k in range(2, max_k + 1):
                km = KMeans(n_clusters=k, n_init=5, max_iter=100, random_state=SEED)
                labels = km.fit_predict(gp_norm)
                if len(set(labels)) < 2:
                    continue
                try:
                    score = silhouette_score(gp_norm, labels, sample_size=min(1000, len(gp_norm)))
                except ValueError:
                    continue
                if score > best_score:
                    best_score = score
                    best_k = k
            cluster_count = best_k

    return {
        "gas_price_unique_ratio": unique_ratio,
        "gas_price_autocorrelation": autocorr,
        "gas_price_cluster_count": cluster_count,
    }


def extract_all_features(address):
    """Extract all 16 decision-process features for one address."""
    parquet_path = RAW_DIR / f"{address}.parquet"
    if not parquet_path.exists():
        return None

    try:
        df_tx = pd.read_parquet(parquet_path)
    except Exception:
        return None

    if len(df_tx) == 0:
        return None

    # Sort by timestamp
    df_tx["timeStamp"] = pd.to_numeric(df_tx["timeStamp"], errors="coerce")
    df_tx = df_tx.sort_values("timeStamp").reset_index(drop=True)

    feats = {}
    feats.update(extract_error_recovery(df_tx))
    feats.update(extract_session_patterns(df_tx))
    feats.update(extract_nonce_patterns(df_tx))
    feats.update(extract_gas_microstructure(df_tx))
    return feats


# ═══════════════════════════════════════════════════════════════════════
# Classification Experiment
# ═══════════════════════════════════════════════════════════════════════

def impute_and_clip(X):
    """Median-impute NaNs and clip to [1st, 99th] percentile."""
    X = X.copy()
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


def bootstrap_per_class_ci(y_true, y_pred, classes, n_boot=N_BOOT, seed=SEED):
    """Bootstrap 95% CIs on per-class F1."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    n_cls = len(classes)
    f1_mat = np.zeros((n_boot, n_cls))

    for b in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        f1_mat[b] = f1_score(y_true[idx], y_pred[idx], labels=classes,
                             average=None, zero_division=0)

    results = {}
    for i, cls in enumerate(classes):
        name = TAXONOMY_NAMES.get(int(cls), f"Class{cls}")
        point = float(f1_score(y_true, y_pred, labels=[cls], average=None, zero_division=0)[0])
        results[name] = {
            "f1_point": round(point, 4),
            "f1_ci_lo": round(float(np.percentile(f1_mat[:, i], 2.5)), 4),
            "f1_ci_hi": round(float(np.percentile(f1_mat[:, i], 97.5)), 4),
        }
    return results


def run_condition(X, y, feature_names, condition_name, classes_sorted):
    """Run 10-fold stratified CV with GBM for one experimental condition."""
    print(f"\n{'─' * 70}")
    print(f"Condition: {condition_name}")
    print(f"  Features: {len(feature_names)}, Samples: {len(y)}")
    print(f"{'─' * 70}")

    X = impute_and_clip(X)

    model = GradientBoostingClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, random_state=SEED,
    )

    min_count = min(np.bincount(y)[np.bincount(y) > 0])
    actual_folds = min(N_FOLDS, min_count)

    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=SEED)

    fold_accs = []
    fold_f1_macro = []
    fold_f1_weighted = []
    all_y_true = []
    all_y_pred = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_te = scaler.transform(X[te_idx])

        clf = clone(model)
        clf.fit(X_tr, y[tr_idx])
        y_pred = clf.predict(X_te)

        acc = accuracy_score(y[te_idx], y_pred)
        f1m = f1_score(y[te_idx], y_pred, average="macro", zero_division=0)
        f1w = f1_score(y[te_idx], y_pred, average="weighted", zero_division=0)

        fold_accs.append(acc)
        fold_f1_macro.append(f1m)
        fold_f1_weighted.append(f1w)
        all_y_true.extend(y[te_idx].tolist())
        all_y_pred.extend(y_pred.tolist())

    y_true_arr = np.array(all_y_true)
    y_pred_arr = np.array(all_y_pred)

    report = classification_report(
        y_true_arr, y_pred_arr,
        labels=classes_sorted,
        target_names=[TAXONOMY_NAMES.get(int(c), f"C{c}") for c in classes_sorted],
        output_dict=True,
        zero_division=0,
    )

    acc_mean = float(np.mean(fold_accs))
    f1m_mean = float(np.mean(fold_f1_macro))
    f1w_mean = float(np.mean(fold_f1_weighted))

    print(f"  Accuracy:    {acc_mean:.4f} +/- {np.std(fold_accs):.4f}")
    print(f"  F1-macro:    {f1m_mean:.4f} +/- {np.std(fold_f1_macro):.4f}")
    print(f"  F1-weighted: {f1w_mean:.4f}")
    print(f"  Per-class F1:")
    for cls_name in [TAXONOMY_NAMES[c] for c in classes_sorted]:
        if cls_name in report:
            s = report[cls_name]
            print(f"    {cls_name:<25} F1={s['f1-score']:.4f} "
                  f"P={s['precision']:.4f} R={s['recall']:.4f} "
                  f"(n={s['support']})")

    # Bootstrap CIs
    print(f"  Bootstrap 95% CIs ({N_BOOT} iterations)...")
    boot_ci = bootstrap_per_class_ci(y_true_arr, y_pred_arr, classes_sorted)
    for cls_name, ci in boot_ci.items():
        print(f"    {cls_name:<25} F1={ci['f1_point']:.4f} "
              f"[{ci['f1_ci_lo']:.4f}, {ci['f1_ci_hi']:.4f}]")

    # Feature importance (full-data refit)
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X)
    gbm_full = clone(model)
    gbm_full.fit(X_full, y)
    sorted_idx = np.argsort(gbm_full.feature_importances_)[::-1]
    fi = {
        feature_names[i]: round(float(gbm_full.feature_importances_[i]), 4)
        for i in sorted_idx
    }

    # Print new-feature importance specifically
    print(f"\n  Feature importance (new decision-process features):")
    dp_fi = {k: v for k, v in fi.items() if k in DECISION_PROCESS_16}
    for rank, (feat, imp) in enumerate(fi.items()):
        if feat in DECISION_PROCESS_16:
            # Find overall rank
            overall_rank = list(fi.keys()).index(feat) + 1
            print(f"    rank {overall_rank:2d}/{len(feature_names)}: "
                  f"{feat:<35} {imp:.4f}")

    return {
        "condition": condition_name,
        "n_features": len(feature_names),
        "n_samples": int(len(y)),
        "n_folds": actual_folds,
        "accuracy_mean": round(acc_mean, 4),
        "accuracy_std": round(float(np.std(fold_accs)), 4),
        "f1_macro_mean": round(f1m_mean, 4),
        "f1_macro_std": round(float(np.std(fold_f1_macro)), 4),
        "f1_weighted_mean": round(f1w_mean, 4),
        "per_class_report": report,
        "bootstrap_95ci": boot_ci,
        "feature_importance": fi,
        "decision_process_feature_importance": dp_fi,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Paper 0: Decision-Process Feature Experiment")
    print("  16 new features from raw tx data (error, session, nonce, gas)")
    print("=" * 80)

    # ── Step 1: Load base dataset ─────────────────────────────────────
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["label"] == 1].copy()
    addresses = list(df.index)
    print(f"\nAgent addresses: {len(addresses)}")

    # ── Step 2: Extract decision-process features ─────────────────────
    print(f"\nExtracting 16 decision-process features from {len(addresses)} raw parquet files...")
    dp_rows = {}
    missing_count = 0
    error_count = 0

    for i, addr in enumerate(addresses):
        if (i + 1) % 500 == 0 or i == 0:
            print(f"  Processing {i+1}/{len(addresses)}...")

        feats = extract_all_features(addr)
        if feats is None:
            missing_count += 1
            dp_rows[addr] = {col: np.nan for col in DECISION_PROCESS_16}
        else:
            dp_rows[addr] = feats

    df_dp = pd.DataFrame.from_dict(dp_rows, orient="index")
    df_dp.index.name = "address"

    print(f"\nExtraction complete:")
    print(f"  Addresses processed: {len(addresses)}")
    print(f"  Missing raw data: {missing_count}")
    print(f"  NaN rate per feature:")
    for col in DECISION_PROCESS_16:
        nan_pct = df_dp[col].isna().mean() * 100
        print(f"    {col:<35} {nan_pct:5.1f}% NaN")

    # ── Step 3: Load AI features ──────────────────────────────────────
    with open(AI_FEATURES_JSON) as f:
        ai_raw = json.load(f)

    ai_rows = []
    for addr, feats in ai_raw["per_address"].items():
        row = {"address": addr}
        for col in AI_8:
            row[col] = feats.get(col, np.nan)
        ai_rows.append(row)

    df_ai = pd.DataFrame(ai_rows).set_index("address")
    print(f"\nAI features: {len(df_ai)} addresses")

    # ── Step 4: Merge everything ──────────────────────────────────────
    df = df.join(df_dp[DECISION_PROCESS_16], how="left")
    df = df.join(df_ai[AI_8], how="left")

    y = df["taxonomy_index"].values.astype(int)
    classes_sorted = sorted(set(y.tolist()))

    print(f"\nClass distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  {u} ({TAXONOMY_NAMES.get(int(u), '?'):<25}) n={c}")

    # ── Step 5: Run 3 conditions ──────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": "Decision-process feature experiment: 4 families, 16 features",
        "n_agents": int(len(df)),
        "n_missing_raw": missing_count,
        "feature_families": {
            "error_recovery": ["error_rate", "post_error_delay_median",
                               "post_error_method_change_rate",
                               "post_error_target_change_rate", "error_abandon_rate"],
            "session_patterns": ["session_count", "session_length_mean",
                                 "session_length_cv", "intra_session_tx_rate",
                                 "inter_session_gap_cv"],
            "nonce_patterns": ["nonce_sequential_rate", "nonce_gap_max",
                               "nonce_completeness"],
            "gas_microstructure": ["gas_price_unique_ratio",
                                   "gas_price_autocorrelation",
                                   "gas_price_cluster_count"],
        },
        "conditions": {},
    }

    # Condition A: Original 23 features (baseline)
    X_a = df[ORIGINAL_23].values.astype(float)
    feat_names_a = list(ORIGINAL_23)
    res_a = run_condition(X_a, y, feat_names_a,
                          "A: 23 original features (baseline)", classes_sorted)
    results["conditions"]["A_23_baseline"] = res_a

    # Condition B: 23 + 16 decision-process = 39 features
    feat_names_b = ORIGINAL_23 + DECISION_PROCESS_16
    X_b = df[feat_names_b].values.astype(float)
    res_b = run_condition(X_b, y, feat_names_b,
                          "B: 23 + 16 decision-process = 39 features", classes_sorted)
    results["conditions"]["B_39_with_dp"] = res_b

    # Condition C: 23 + 8 AI + 16 decision-process = 47 features
    feat_names_c = ORIGINAL_23 + AI_8 + DECISION_PROCESS_16
    X_c = df[feat_names_c].values.astype(float)
    res_c = run_condition(X_c, y, feat_names_c,
                          "C: 23 + 8 AI + 16 decision-process = 47 features (full)",
                          classes_sorted)
    results["conditions"]["C_47_full"] = res_c

    # ── Step 6: Delta comparison ──────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("COMPARISON: Decision-Process Feature Impact")
    print("=" * 80)

    header = f"{'Metric':<30} {'A (23)':<10} {'B (39)':<10} {'C (47)':<10} {'A→B Δ':<10} {'A→C Δ':<10}"
    print(header)
    print("-" * len(header))

    comparison = {}
    for metric_key, metric_name in [
        ("accuracy_mean", "Accuracy"),
        ("f1_macro_mean", "F1-macro"),
        ("f1_weighted_mean", "F1-weighted"),
    ]:
        va = res_a[metric_key]
        vb = res_b[metric_key]
        vc = res_c[metric_key]
        delta_ab = vb - va
        delta_ac = vc - va
        comparison[metric_name] = {
            "A": va, "B": vb, "C": vc,
            "delta_A_to_B": round(delta_ab, 4),
            "delta_A_to_C": round(delta_ac, 4),
        }
        print(f"  {metric_name:<28} {va:<10.4f} {vb:<10.4f} {vc:<10.4f} "
              f"{'+' if delta_ab >= 0 else ''}{delta_ab:<10.4f} "
              f"{'+' if delta_ac >= 0 else ''}{delta_ac:<10.4f}")

    print()
    print(f"{'Per-class F1':<30} {'A (23)':<10} {'B (39)':<10} {'C (47)':<10} {'A→B Δ':<10} {'A→C Δ':<10}")
    print("-" * 80)

    for cls_idx in classes_sorted:
        cls_name = TAXONOMY_NAMES.get(int(cls_idx), f"C{cls_idx}")
        f1_a = res_a["per_class_report"].get(cls_name, {}).get("f1-score", 0)
        f1_b = res_b["per_class_report"].get(cls_name, {}).get("f1-score", 0)
        f1_c = res_c["per_class_report"].get(cls_name, {}).get("f1-score", 0)
        delta_ab = f1_b - f1_a
        delta_ac = f1_c - f1_a
        comparison[f"{cls_name}_f1"] = {
            "A": round(f1_a, 4), "B": round(f1_b, 4), "C": round(f1_c, 4),
            "delta_A_to_B": round(delta_ab, 4),
            "delta_A_to_C": round(delta_ac, 4),
        }
        marker = " ***" if cls_name == "LLMPoweredAgent" else ""
        print(f"  {cls_name:<28} {f1_a:<10.4f} {f1_b:<10.4f} {f1_c:<10.4f} "
              f"{'+' if delta_ab >= 0 else ''}{delta_ab:<10.4f} "
              f"{'+' if delta_ac >= 0 else ''}{delta_ac:<10.4f}{marker}")

    results["comparison"] = comparison

    # ── Step 7: Bootstrap CIs for LLMPoweredAgent specifically ────────
    print(f"\n{'─' * 70}")
    print("LLMPoweredAgent F1 — Bootstrap 95% CIs across conditions:")
    print("─" * 70)
    for cond_label, cond_key in [("A (23)", "A_23_baseline"),
                                  ("B (39)", "B_39_with_dp"),
                                  ("C (47)", "C_47_full")]:
        ci = results["conditions"][cond_key]["bootstrap_95ci"].get("LLMPoweredAgent", {})
        print(f"  {cond_label}: F1={ci.get('f1_point', 0):.4f} "
              f"[{ci.get('f1_ci_lo', 0):.4f}, {ci.get('f1_ci_hi', 0):.4f}]")

    # ── Save ──────────────────────────────────────────────────────────
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")
    print("Done.")


if __name__ == "__main__":
    main()
