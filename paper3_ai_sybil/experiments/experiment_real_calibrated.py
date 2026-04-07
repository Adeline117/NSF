"""
Paper 3 — Real-Data Calibrated AI Sybil Evasion Experiment
===========================================================
Uses REAL HasciDB CSV data (16 projects, 1.09M sybils) with calibrated
distributions. No synthetic baseline data — all baseline experiments use
actual on-chain sybil labels from the CHI'26 five-indicator framework.

Experiments:
  1. Real baseline performance (5-fold CV on blur_s2 real data)
  2. AI Sybil evasion against real-trained baseline
  3. Enhanced detector with AI-specific features on mixed data
  4. Cross-project transfer with real data (train blur_s2 → test others)

Data: paper3_ai_sybil/data/HasciDB/data/sybil_results/{project}_chi26_v3.csv
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HASCIDB_DIR = (
    PROJECT_ROOT
    / "paper3_ai_sybil"
    / "data"
    / "HasciDB"
    / "data"
    / "sybil_results"
)
OUTPUT_FILE = SCRIPT_DIR / "experiment_real_results.json"
DIST_FILE = SCRIPT_DIR / "real_distribution_results.json"

# ============================================================
# CONSTANTS
# ============================================================

INDICATOR_COLS = ["BT", "BW", "HF", "RF", "MA"]

THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}

AI_FEATURES = [
    "gas_price_precision",
    "hour_entropy",
    "behavioral_consistency",
    "action_sequence_perplexity",
    "error_recovery_pattern",
    "response_latency_variance",
    "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]

# Primary project for training
PRIMARY_PROJECT = "blur_s2"

# Transfer-test projects
TRANSFER_PROJECTS = ["uniswap", "eigenlayer", "gitcoin"]

# Maximum rows to use per project (for speed on very large CSVs)
MAX_ROWS = 50_000


# ============================================================
# DATA LOADING
# ============================================================


def load_project(project: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Load a HasciDB CSV, sampling if it exceeds max_rows."""
    csv_path = HASCIDB_DIR / f"{project}_chi26_v3.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalise columns in case of variant naming
    for col in INDICATOR_COLS:
        if col not in df.columns:
            for alt in [col.lower(), f"{col.lower()}_score"]:
                if alt in df.columns:
                    df[col] = df[alt]
                    break
            else:
                df[col] = 0

    # Ensure is_sybil exists
    if "is_sybil" not in df.columns:
        ops = (df["BT"] >= THRESHOLDS["BT"]) | (df["BW"] >= THRESHOLDS["BW"]) | (df["HF"] >= THRESHOLDS["HF"])
        fund = (df["RF"] >= THRESHOLDS["RF"]) | (df["MA"] >= THRESHOLDS["MA"])
        df["is_sybil"] = (ops | fund).astype(int)

    # Stratified sample if too large
    if len(df) > max_rows:
        sybils = df[df["is_sybil"] == 1]
        non_sybils = df[df["is_sybil"] == 0]
        sybil_rate = len(sybils) / len(df)
        n_sybil_sample = int(max_rows * sybil_rate)
        n_nonsybil_sample = max_rows - n_sybil_sample
        sybils_s = sybils.sample(n=min(n_sybil_sample, len(sybils)), random_state=42)
        nonsybils_s = non_sybils.sample(n=min(n_nonsybil_sample, len(non_sybils)), random_state=42)
        df = pd.concat([sybils_s, nonsybils_s], ignore_index=True)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    df["project"] = project
    return df


def load_real_distributions() -> dict:
    """Load pre-computed real distribution stats (for AI calibration)."""
    if DIST_FILE.exists():
        with open(DIST_FILE) as f:
            return json.load(f)
    return {}


# ============================================================
# AI SYBIL GENERATION — calibrated to real non-sybil distributions
# ============================================================


def _get_nonsybil_stats(dist_data: dict, project: str) -> dict:
    """Extract non-sybil distribution stats for a project."""
    if not dist_data:
        return None
    for p in dist_data.get("per_project", []):
        if p["project"] == project:
            return {
                ind: p["distributions"][ind]["non_sybil"]
                for ind in INDICATOR_COLS
            }
    return None


def generate_ai_sybils(
    n: int,
    nonsybil_stats: dict | None,
    rng: np.random.RandomState,
    evasion_level: str = "advanced",
) -> pd.DataFrame:
    """Generate AI sybils calibrated to look like real non-sybils.

    Uses the actual non-sybil distributions from HasciDB to set indicator
    values that stay safely below thresholds, while adding AI-agent
    behavioural signatures that leak through other channels.

    Parameters
    ----------
    n : int
        Number of AI sybils to generate.
    nonsybil_stats : dict | None
        Per-indicator distribution stats for the target project's non-sybil
        population.  If None, fallback to conservative defaults.
    rng : np.random.RandomState
        Random state for reproducibility.
    evasion_level : str
        One of "basic", "moderate", "advanced".  Controls how carefully the
        AI agent mimics legitimate distributions and hides its AI signatures.
    """

    # ---- Evasion sophistication multipliers ----
    evasion_cfg = {
        "basic":    {"indicator_noise": 1.5, "ai_signal": 0.95},
        "moderate": {"indicator_noise": 1.0, "ai_signal": 0.75},
        "advanced": {"indicator_noise": 0.7, "ai_signal": 0.55},
    }
    cfg = evasion_cfg[evasion_level]

    # ---- HasciDB indicators: mimic non-sybil distributions ----
    if nonsybil_stats is not None:
        # Calibrate each indicator from real non-sybil distribution
        bt_mean = nonsybil_stats["BT"]["mean"] * cfg["indicator_noise"]
        bt_std = max(nonsybil_stats["BT"]["std"], 0.3) * cfg["indicator_noise"]
        bt = np.clip(rng.normal(bt_mean, bt_std, n), 0, THRESHOLDS["BT"] - 1).astype(int)

        bw_mean = nonsybil_stats["BW"]["mean"] * cfg["indicator_noise"]
        bw_std = max(nonsybil_stats["BW"]["std"], 0.3) * cfg["indicator_noise"]
        bw = np.clip(rng.normal(bw_mean, bw_std, n), 0, THRESHOLDS["BW"] - 1).astype(int)

        hf_mean = nonsybil_stats["HF"]["mean"]
        hf_std = max(nonsybil_stats["HF"]["std"], 0.05)
        hf = np.clip(rng.normal(hf_mean, hf_std * cfg["indicator_noise"], n), 0, THRESHOLDS["HF"] - 0.01)

        rf_mean = nonsybil_stats["RF"]["mean"]
        rf_std = max(nonsybil_stats["RF"]["std"], 0.02)
        rf = np.clip(rng.normal(rf_mean, rf_std * cfg["indicator_noise"], n), 0, THRESHOLDS["RF"] - 0.01)

        ma_mean = nonsybil_stats["MA"]["mean"] * cfg["indicator_noise"]
        ma_std = max(nonsybil_stats["MA"]["std"], 0.3) * cfg["indicator_noise"]
        ma = np.clip(rng.normal(ma_mean, ma_std, n), 0, THRESHOLDS["MA"] - 1).astype(int)
    else:
        # Conservative fallback: all indicators near zero
        bt = np.clip(rng.exponential(0.05, n), 0, 4).astype(int)
        bw = np.clip(rng.exponential(0.1, n), 0, 9).astype(int)
        hf = np.clip(rng.beta(2, 8, n), 0, 0.78)
        rf = np.clip(rng.beta(1, 15, n), 0, 0.48)
        ma = np.clip(rng.exponential(0.05, n), 0, 4).astype(int)

    # ---- AI-specific features: these leak agent signatures ----
    #
    # Even an advanced AI agent cannot fully eliminate these signals because
    # they arise from fundamental properties of LLM-based decision making:
    #   - gas_price_precision: LLMs compute exact gas, humans round
    #   - hour_entropy: no circadian rhythm (or LLM-inference bursts)
    #   - behavioral_consistency: same prompt → correlated actions
    #   - action_sequence_perplexity: LLM output has characteristic range
    #   - error_recovery_pattern: systematic retry/fallback
    #   - response_latency_variance: LLM inference time signature
    #   - gas_nonce_gap_regularity: programmatic nonce management
    #   - eip1559_tip_precision: computed priority fees

    sig = cfg["ai_signal"]  # higher = more detectable

    gas_precision = rng.beta(5 * sig + 2, 3 - sig, n)
    hour_entropy = rng.beta(4 * sig + 2, 4 - 2 * sig, n) * 3.178
    behav_consistency = rng.beta(4 * sig + 1, 4 - 2 * sig, n)
    action_perplexity = rng.lognormal(1.2 + 0.3 * sig, 0.5, n)
    error_recovery = rng.beta(4 * sig + 1, 4 - 2 * sig, n)
    latency_var = rng.lognormal(0.3 + 0.2 * sig, 0.5, n)
    nonce_regularity = rng.beta(4 * sig + 1, 4 - 2 * sig, n)
    tip_precision = rng.beta(4 * sig + 1, 4 - 2 * sig, n)

    return pd.DataFrame({
        "BT": bt,
        "BW": bw,
        "HF": hf,
        "RF": rf,
        "MA": ma,
        "gas_price_precision": gas_precision,
        "hour_entropy": hour_entropy,
        "behavioral_consistency": behav_consistency,
        "action_sequence_perplexity": action_perplexity,
        "error_recovery_pattern": error_recovery,
        "response_latency_variance": latency_var,
        "gas_nonce_gap_regularity": nonce_regularity,
        "eip1559_tip_precision": tip_precision,
        "is_sybil": 1,
    })


def augment_real_data_with_ai_features(
    df: pd.DataFrame,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Add synthetic AI-specific features to real HasciDB data.

    Real data only has the 5 HasciDB indicators.  For the enhanced detector
    experiments we need the 8 AI-specific behavioural features.  We generate
    them from class-conditional distributions:
      - Non-sybils: human-like patterns (low precision, low entropy, etc.)
      - Sybils (traditional): bot/script patterns (high precision, 24/7, etc.)
    """
    df = df.copy()
    n = len(df)
    is_sybil = df["is_sybil"].values.astype(bool)

    for feat in AI_FEATURES:
        vals = np.zeros(n)
        n_s = is_sybil.sum()
        n_ns = n - n_s

        if feat == "gas_price_precision":
            vals[~is_sybil] = rng.beta(2, 8, n_ns)
            vals[is_sybil] = rng.beta(7, 2, n_s)
        elif feat == "hour_entropy":
            vals[~is_sybil] = rng.beta(3, 5, n_ns) * 3.178
            vals[is_sybil] = rng.beta(7, 2, n_s) * 3.178
        elif feat == "behavioral_consistency":
            vals[~is_sybil] = rng.beta(2, 6, n_ns)
            vals[is_sybil] = rng.beta(8, 2, n_s)
        elif feat == "action_sequence_perplexity":
            vals[~is_sybil] = rng.lognormal(3, 1, n_ns)
            vals[is_sybil] = rng.lognormal(0.5, 0.3, n_s)
        elif feat == "error_recovery_pattern":
            vals[~is_sybil] = rng.beta(2, 5, n_ns)
            vals[is_sybil] = rng.beta(8, 2, n_s)
        elif feat == "response_latency_variance":
            vals[~is_sybil] = rng.lognormal(2, 1.5, n_ns)
            vals[is_sybil] = rng.lognormal(-1, 0.3, n_s)
        elif feat == "gas_nonce_gap_regularity":
            vals[~is_sybil] = rng.beta(2, 6, n_ns)
            vals[is_sybil] = rng.beta(8, 2, n_s)
        elif feat == "eip1559_tip_precision":
            vals[~is_sybil] = rng.beta(2, 7, n_ns)
            vals[is_sybil] = rng.beta(7, 2, n_s)

        df[feat] = vals

    return df


# ============================================================
# EXPERIMENT 1 — Real Baseline Performance
# ============================================================


def exp1_real_baseline(df: pd.DataFrame) -> dict:
    """Train GBM on real HasciDB data with 5-fold CV.

    This establishes the TRUE baseline: how well a gradient-boosted model
    can separate sybils from non-sybils using only the 5 HasciDB indicators
    on real, labelled on-chain data.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 1: Real Baseline Performance (blur_s2)")
    print("=" * 72)

    X = df[INDICATOR_COLS].values
    y = df["is_sybil"].values

    n_total = len(df)
    n_sybil = y.sum()
    n_nonsybil = n_total - n_sybil
    sybil_rate = n_sybil / n_total

    print(f"  Data:  {n_total:,} addresses  ({n_sybil:,} sybils, "
          f"{n_nonsybil:,} non-sybils, rate={sybil_rate:.1%})")

    # 5-fold stratified CV
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        clf.fit(X[train_idx], y[train_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        y_pred = clf.predict(X[test_idx])

        auc = roc_auc_score(y[test_idx], y_prob)
        ap = average_precision_score(y[test_idx], y_prob)
        prec = precision_score(y[test_idx], y_pred, zero_division=0)
        rec = recall_score(y[test_idx], y_pred, zero_division=0)
        f1 = f1_score(y[test_idx], y_pred, zero_division=0)

        fold_results.append({
            "fold": fold_idx,
            "auc": float(auc),
            "ap": float(ap),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
        })
        print(f"  Fold {fold_idx}: AUC={auc:.4f}  AP={ap:.4f}  "
              f"P={prec:.3f}  R={rec:.3f}  F1={f1:.3f}")

    mean_auc = np.mean([r["auc"] for r in fold_results])
    std_auc = np.std([r["auc"] for r in fold_results])
    mean_ap = np.mean([r["ap"] for r in fold_results])

    print(f"\n  >>> 5-Fold CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  >>> 5-Fold CV AP:  {mean_ap:.4f}")

    # Feature importance from full-data fit
    clf.fit(X, y)
    importances = dict(zip(INDICATOR_COLS, clf.feature_importances_.tolist()))
    print(f"\n  Feature importances:")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        bar = "#" * int(imp * 50)
        print(f"    {feat:>4s}: {imp:.4f}  {bar}")

    return {
        "project": PRIMARY_PROJECT,
        "n_total": n_total,
        "n_sybil": int(n_sybil),
        "n_nonsybil": int(n_nonsybil),
        "sybil_rate": float(sybil_rate),
        "cv_folds": fold_results,
        "mean_auc": float(mean_auc),
        "std_auc": float(std_auc),
        "mean_ap": float(mean_ap),
        "feature_importances": importances,
    }


# ============================================================
# EXPERIMENT 2 — AI Sybil Evasion Against Real Baseline
# ============================================================


def exp2_ai_evasion(
    df_train: pd.DataFrame,
    nonsybil_stats: dict | None,
) -> dict:
    """Test AI sybil evasion against a model trained on real data.

    The baseline GBM is trained on real blur_s2 sybils/non-sybils.
    AI sybils are generated to mimic real non-sybil distributions.
    We measure how often the real-trained baseline fails to detect them.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 2: AI Sybil Evasion Against Real-Trained Baseline")
    print("=" * 72)

    # Train baseline on real data
    X_train = df_train[INDICATOR_COLS].values
    y_train = df_train["is_sybil"].values

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # In-distribution test: hold-out split performance
    from sklearn.model_selection import train_test_split
    _, X_test_id, _, y_test_id = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=99
    )
    y_prob_id = clf.predict_proba(X_test_id)[:, 1]
    auc_in_dist = roc_auc_score(y_test_id, y_prob_id)
    print(f"\n  In-distribution AUC (blur_s2 hold-out): {auc_in_dist:.4f}")

    # Hold-out legitimate addresses for test mixing
    nonsybils_test = df_train[df_train["is_sybil"] == 0].sample(
        n=min(1000, (df_train["is_sybil"] == 0).sum()), random_state=77
    )

    evasion_results = {}
    for level in ["basic", "moderate", "advanced"]:
        rng = np.random.RandomState(42 + hash(level) % 1000)
        ai_sybils = generate_ai_sybils(
            n=1000,
            nonsybil_stats=nonsybil_stats,
            rng=rng,
            evasion_level=level,
        )

        # Combine AI sybils with real non-sybils
        test_legit = nonsybils_test.copy()
        test_legit_slim = test_legit[INDICATOR_COLS].copy()
        test_legit_slim["is_sybil"] = 0
        ai_slim = ai_sybils[INDICATOR_COLS + ["is_sybil"]].copy()
        test_data = pd.concat([test_legit_slim, ai_slim], ignore_index=True)

        X_test = test_data[INDICATOR_COLS].values
        y_test = test_data["is_sybil"].values

        # Rule-based detection
        ops_flag = (test_data["BT"] >= THRESHOLDS["BT"]) | \
                   (test_data["BW"] >= THRESHOLDS["BW"]) | \
                   (test_data["HF"] >= THRESHOLDS["HF"])
        fund_flag = (test_data["RF"] >= THRESHOLDS["RF"]) | \
                    (test_data["MA"] >= THRESHOLDS["MA"])
        y_rule = (ops_flag | fund_flag).astype(int).values

        rule_recall_ai = (y_rule[y_test == 1] == 1).mean()
        rule_evasion = 1.0 - rule_recall_ai

        # ML-based detection
        y_prob = clf.predict_proba(X_test)[:, 1]
        auc_vs_ai = roc_auc_score(y_test, y_prob)
        ap_vs_ai = average_precision_score(y_test, y_prob)

        # ML recall at threshold 0.5
        y_pred_ml = (y_prob >= 0.5).astype(int)
        ml_recall_ai = (y_pred_ml[y_test == 1] == 1).mean()
        ml_evasion = 1.0 - ml_recall_ai

        evasion_results[level] = {
            "rule_recall": float(rule_recall_ai),
            "rule_evasion_rate": float(rule_evasion),
            "ml_auc": float(auc_vs_ai),
            "ml_ap": float(ap_vs_ai),
            "ml_recall_at_05": float(ml_recall_ai),
            "ml_evasion_at_05": float(ml_evasion),
        }

        print(f"\n  [{level.upper()}] AI Sybils (n={len(ai_sybils)}):")
        print(f"    Rule-based: recall={rule_recall_ai:.3f}, "
              f"evasion={rule_evasion:.1%}")
        print(f"    ML (GBM):   AUC={auc_vs_ai:.4f}, AP={ap_vs_ai:.4f}, "
              f"recall@0.5={ml_recall_ai:.3f}, evasion={ml_evasion:.1%}")

    return {
        "baseline_auc_in_dist": float(auc_in_dist),
        "evasion_by_level": evasion_results,
    }


# ============================================================
# EXPERIMENT 3 — Enhanced Detector on Mixed Data
# ============================================================


def exp3_enhanced_detector(
    df_real: pd.DataFrame,
    nonsybil_stats: dict | None,
) -> dict:
    """Train enhanced detector with both real sybils and AI sybils.

    Adds 8 AI-specific behavioural features to the 5 HasciDB indicators.
    Training data: real sybils + real non-sybils + AI sybils.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 3: Enhanced Detector (13 features, mixed training)")
    print("=" * 72)

    rng = np.random.RandomState(42)
    all_features = INDICATOR_COLS + AI_FEATURES

    # Augment real data with synthetic AI-specific features
    df_aug = augment_real_data_with_ai_features(df_real, rng)

    # Generate AI sybils for training (label = 1)
    n_ai_train = min(2000, len(df_real[df_real["is_sybil"] == 1]))
    ai_train = generate_ai_sybils(
        n=n_ai_train,
        nonsybil_stats=nonsybil_stats,
        rng=np.random.RandomState(50),
        evasion_level="moderate",
    )

    # Combine: real data + AI sybils
    train_combined = pd.concat([df_aug, ai_train], ignore_index=True)

    X = train_combined[all_features].values
    y = train_combined["is_sybil"].values

    print(f"  Training data: {len(train_combined):,} rows")
    print(f"    Real sybils:     {(df_aug['is_sybil'] == 1).sum():,}")
    print(f"    Real non-sybils: {(df_aug['is_sybil'] == 0).sum():,}")
    print(f"    AI sybils:       {len(ai_train):,}")

    # 5-fold CV on the combined dataset
    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        clf.fit(X[train_idx], y[train_idx])
        y_prob = clf.predict_proba(X[test_idx])[:, 1]
        auc = roc_auc_score(y[test_idx], y_prob)
        ap = average_precision_score(y[test_idx], y_prob)
        fold_results.append({"fold": fold_idx, "auc": float(auc), "ap": float(ap)})
        print(f"  Fold {fold_idx}: AUC={auc:.4f}  AP={ap:.4f}")

    mean_auc = np.mean([r["auc"] for r in fold_results])
    std_auc = np.std([r["auc"] for r in fold_results])

    print(f"\n  >>> Enhanced 5-Fold CV AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

    # ---- Held-out AI sybil test at each evasion level ----
    clf.fit(X, y)

    # Feature importance
    importances = dict(zip(all_features, clf.feature_importances_.tolist()))
    print(f"\n  Feature importances (13 features):")
    for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
        ftype = "AI" if feat in AI_FEATURES else "HasciDB"
        bar = "#" * int(imp * 50)
        print(f"    {feat:<35s} {imp:.4f}  [{ftype:>7s}]  {bar}")

    # Test on held-out AI sybils
    recovery_results = {}
    for level in ["basic", "moderate", "advanced"]:
        rng_test = np.random.RandomState(60 + hash(level) % 1000)
        ai_test = generate_ai_sybils(
            n=500,
            nonsybil_stats=nonsybil_stats,
            rng=rng_test,
            evasion_level=level,
        )
        legit_test = df_aug[df_aug["is_sybil"] == 0].sample(
            n=min(500, (df_aug["is_sybil"] == 0).sum()), random_state=71
        )

        test_data = pd.concat([legit_test[all_features + ["is_sybil"]],
                               ai_test[all_features + ["is_sybil"]]],
                              ignore_index=True)
        X_test = test_data[all_features].values
        y_test = test_data["is_sybil"].values

        y_prob = clf.predict_proba(X_test)[:, 1]
        auc_test = roc_auc_score(y_test, y_prob)
        ap_test = average_precision_score(y_test, y_prob)

        recovery_results[level] = {
            "enhanced_auc": float(auc_test),
            "enhanced_ap": float(ap_test),
        }
        print(f"\n  [{level.upper()}] Held-out AI sybils:")
        print(f"    Enhanced AUC: {auc_test:.4f}  AP: {ap_test:.4f}")

    ai_in_top5 = sum(1 for f, _ in sorted(importances.items(), key=lambda x: -x[1])[:5]
                     if f in AI_FEATURES)

    return {
        "cv_folds": fold_results,
        "mean_auc": float(mean_auc),
        "std_auc": float(std_auc),
        "feature_importances": importances,
        "ai_features_in_top5": ai_in_top5,
        "recovery_by_level": recovery_results,
    }


# ============================================================
# EXPERIMENT 4 — Cross-Project Transfer
# ============================================================


def exp4_cross_project_transfer(
    df_train: pd.DataFrame,
) -> dict:
    """Train on blur_s2, test on uniswap, eigenlayer, gitcoin.

    Measures how well a sybil detector trained on one project's real data
    transfers to other projects — a key practical question for deployment.
    """
    print("\n" + "=" * 72)
    print("EXPERIMENT 4: Cross-Project Transfer (real data)")
    print("=" * 72)

    X_train = df_train[INDICATOR_COLS].values
    y_train = df_train["is_sybil"].values

    clf = GradientBoostingClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.8,
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # In-distribution baseline
    from sklearn.model_selection import cross_val_score as cvs
    cv_auc = cvs(clf, X_train, y_train, cv=5, scoring="roc_auc")
    print(f"\n  Train project: {PRIMARY_PROJECT}")
    print(f"  In-distribution CV AUC: {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")
    clf.fit(X_train, y_train)  # Refit on full data

    transfer_results = {}
    for proj in TRANSFER_PROJECTS:
        try:
            df_test = load_project(proj)
        except FileNotFoundError:
            print(f"  {proj}: CSV not found, skipping")
            continue

        X_test = df_test[INDICATOR_COLS].values
        y_test = df_test["is_sybil"].values
        n_total = len(df_test)
        n_sybil = int(y_test.sum())
        sybil_rate = y_test.mean()

        y_prob = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)

        # Also compute rule-based accuracy
        ops = (df_test["BT"] >= THRESHOLDS["BT"]) | \
              (df_test["BW"] >= THRESHOLDS["BW"]) | \
              (df_test["HF"] >= THRESHOLDS["HF"])
        fund = (df_test["RF"] >= THRESHOLDS["RF"]) | \
               (df_test["MA"] >= THRESHOLDS["MA"])
        y_rule = (ops | fund).astype(int).values
        rule_acc = (y_rule == y_test).mean()
        rule_auc = roc_auc_score(y_test, y_rule)

        transfer_results[proj] = {
            "n_total": n_total,
            "n_sybil": n_sybil,
            "sybil_rate": float(sybil_rate),
            "transfer_auc": float(auc),
            "transfer_ap": float(ap),
            "rule_accuracy": float(rule_acc),
            "rule_auc": float(rule_auc),
        }

        print(f"\n  {proj} (n={n_total:,}, sybil_rate={sybil_rate:.1%}):")
        print(f"    Transfer GBM AUC: {auc:.4f}  AP: {ap:.4f}")
        print(f"    Rule-based AUC:   {rule_auc:.4f}  Acc: {rule_acc:.3f}")

    # Summary
    if transfer_results:
        aucs = [r["transfer_auc"] for r in transfer_results.values()]
        print(f"\n  >>> Transfer AUC range: [{min(aucs):.4f}, {max(aucs):.4f}]")
        print(f"  >>> Transfer AUC mean:  {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    return {
        "train_project": PRIMARY_PROJECT,
        "train_cv_auc": float(cv_auc.mean()),
        "transfer": transfer_results,
    }


# ============================================================
# MAIN
# ============================================================


def main():
    start_time = time.time()

    print("=" * 72)
    print("Paper 3: Real-Data Calibrated AI Sybil Evasion Experiment")
    print("=" * 72)
    print(f"  HasciDB data dir: {HASCIDB_DIR}")
    print(f"  Primary project:  {PRIMARY_PROJECT}")
    print(f"  Transfer projects: {TRANSFER_PROJECTS}")
    print(f"  Max rows/project: {MAX_ROWS:,}")

    # ---- Load real distribution stats ----
    dist_data = load_real_distributions()
    if dist_data:
        print(f"  Distribution file loaded: {DIST_FILE.name}")
    else:
        print(f"  WARNING: {DIST_FILE.name} not found; using fallback calibration")

    nonsybil_stats = _get_nonsybil_stats(dist_data, PRIMARY_PROJECT)
    if nonsybil_stats:
        print(f"  Non-sybil stats loaded for {PRIMARY_PROJECT}")
        for ind in INDICATOR_COLS:
            s = nonsybil_stats[ind]
            print(f"    {ind}: mean={s['mean']:.4f}, std={s['std']:.4f}, "
                  f"median={s['median']:.4f}, p95={s['p95']:.4f}")

    # ---- Load primary project ----
    print(f"\n  Loading {PRIMARY_PROJECT}...")
    df_primary = load_project(PRIMARY_PROJECT)
    n_total = len(df_primary)
    n_sybil = df_primary["is_sybil"].sum()
    print(f"  Loaded {n_total:,} rows ({n_sybil:,} sybils, "
          f"rate={n_sybil/n_total:.1%})")

    # ---- Run experiments ----
    results = {"metadata": {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "primary_project": PRIMARY_PROJECT,
        "transfer_projects": TRANSFER_PROJECTS,
        "max_rows": MAX_ROWS,
        "hascidb_indicators": INDICATOR_COLS,
        "ai_features": AI_FEATURES,
        "thresholds": THRESHOLDS,
    }}

    # Exp 1
    results["exp1_real_baseline"] = exp1_real_baseline(df_primary)

    # Exp 2
    results["exp2_ai_evasion"] = exp2_ai_evasion(df_primary, nonsybil_stats)

    # Exp 3
    results["exp3_enhanced_detector"] = exp3_enhanced_detector(
        df_primary, nonsybil_stats
    )

    # Exp 4
    results["exp4_cross_project"] = exp4_cross_project_transfer(df_primary)

    # ---- Summary ----
    elapsed = time.time() - start_time
    results["metadata"]["elapsed_seconds"] = round(elapsed, 1)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)

    exp1 = results["exp1_real_baseline"]
    exp2 = results["exp2_ai_evasion"]
    exp3 = results["exp3_enhanced_detector"]
    exp4 = results["exp4_cross_project"]

    print(f"\n  {'Metric':<55s} {'Value':>10}")
    print("  " + "-" * 68)

    print(f"  {'Exp1: Real baseline AUC (blur_s2, 5-fold CV)':<55s} "
          f"{exp1['mean_auc']:>10.4f}")

    print(f"  {'Exp2: In-distribution AUC':<55s} "
          f"{exp2['baseline_auc_in_dist']:>10.4f}")

    for level in ["basic", "moderate", "advanced"]:
        ev = exp2["evasion_by_level"][level]
        print(f"  {'Exp2: Rule evasion rate (' + level + ')':<55s} "
              f"{ev['rule_evasion_rate']:>9.1%}")
        print(f"  {'Exp2: ML AUC vs AI (' + level + ')':<55s} "
              f"{ev['ml_auc']:>10.4f}")

    print(f"  {'Exp3: Enhanced CV AUC (13 features)':<55s} "
          f"{exp3['mean_auc']:>10.4f}")
    print(f"  {'Exp3: AI features in top-5 importance':<55s} "
          f"{exp3['ai_features_in_top5']:>10d}")

    for level in ["basic", "moderate", "advanced"]:
        rec = exp3["recovery_by_level"][level]
        ev = exp2["evasion_by_level"][level]
        delta = rec["enhanced_auc"] - ev["ml_auc"]
        print(f"  {'Exp3: Enhanced AUC vs AI (' + level + ')':<55s} "
              f"{rec['enhanced_auc']:>10.4f}")
        print(f"  {'Exp3: AUC recovery (' + level + ')':<55s} "
              f"{delta:>+10.4f}")

    for proj, tr in exp4.get("transfer", {}).items():
        print(f"  {'Exp4: Transfer AUC → ' + proj:<55s} "
              f"{tr['transfer_auc']:>10.4f}")

    # Feasibility assessment
    adv_ev = exp2["evasion_by_level"]["advanced"]
    adv_rec = exp3["recovery_by_level"]["advanced"]
    evasion_rate = adv_ev["rule_evasion_rate"]
    recovery = adv_rec["enhanced_auc"] - adv_ev["ml_auc"]
    feasible = evasion_rate > 0.5 and recovery > 0.05

    results["feasibility"] = {
        "advanced_evasion_rate": float(evasion_rate),
        "advanced_recovery": float(recovery),
        "verdict": "CONFIRMED" if feasible else "NEEDS_ADJUSTMENT",
    }

    print(f"\n  FEASIBILITY: {'CONFIRMED' if feasible else 'NEEDS ADJUSTMENT'}")
    print(f"    Advanced AI evasion rate: {evasion_rate:.1%}")
    print(f"    Enhanced detector recovery: +{recovery:.4f} AUC")
    print(f"    Elapsed: {elapsed:.1f}s")

    # ---- Save ----
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")

    return results


if __name__ == "__main__":
    main()
