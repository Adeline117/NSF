"""
Paper 3 -- Large-Scale Cross-Axis & Temporal Experiment Across All 16 Projects
================================================================================
CRITICAL DESIGN NOTE:
  is_sybil = f(BT, BW, HF, RF, MA)  <-- deterministic across ALL projects.
  Training GBM on {BT,BW,HF,RF,MA} -> predict is_sybil gives AUC=1.0
  trivially, even with LOPO, because the GBM learns the threshold rules.

  The ONLY non-circular ML experiments are:
    (a) Cross-axis: train on OPS indicators -> predict fund_flag (and vice versa)
    (b) Cross-method: train on HasciDB -> predict Gitcoin FDD labels
    (c) Temporal cross-axis: train pre-2023 OPS -> predict post-2023 fund_flag

  This script uses cross-axis prediction as the primary evaluation metric.

Experiments:
  1. Baseline statistics across all 16 projects
  2. Cross-Axis LOPO: train OPS->fund_flag / FUND->ops_flag (leave one project out)
  3. AI Sybil evasion against cross-axis detector
  4. Enhanced detector with real AI features (cross-axis evaluation)
  5. Multi-baseline comparison (rule, single-indicator, ML variants)
  6. Temporal analysis (pre-2023 train -> post-2023 test, cross-axis)

Data:
  paper3_ai_sybil/data/HasciDB/data/sybil_results/{project}_chi26_v3.csv
  paper3_ai_sybil/experiments/real_ai_features.json

Usage:
    python3 paper3_ai_sybil/experiments/experiment_large_scale.py
"""

import json
import sys
import time
import warnings
import builtins
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# Force flush on all prints
_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HASCIDB_DIR = (
    PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
)
AI_FEATURES_FILE = SCRIPT_DIR / "real_ai_features.json"
OUTPUT_FILE = SCRIPT_DIR / "experiment_large_scale_results.json"

# ============================================================
# CONSTANTS
# ============================================================

INDICATOR_COLS = ["BT", "BW", "HF", "RF", "MA"]
OPS_COLS = ["BT", "BW", "HF"]
FUND_COLS = ["RF", "MA"]

THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}

AI_FEATURE_NAMES = [
    "gas_price_precision",
    "hour_entropy",
    "behavioral_consistency",
    "action_sequence_perplexity",
    "error_recovery_pattern",
    "response_latency_variance",
    "gas_nonce_gap_regularity",
    "eip1559_tip_precision",
]

PROJECTS = [
    "1inch", "uniswap", "looksrare", "apecoin", "ens", "gitcoin",
    "etherfi", "x2y2", "dydx", "badger", "blur_s1", "blur_s2",
    "paraswap", "ampleforth", "eigenlayer", "pengu",
]

PROJECT_DATES = {
    "uniswap": "2020-09", "1inch": "2020-12", "badger": "2021-02",
    "ampleforth": "2021-04", "gitcoin": "2021-05", "dydx": "2021-09",
    "ens": "2021-11", "looksrare": "2022-01", "x2y2": "2022-02",
    "apecoin": "2022-03", "blur_s1": "2023-02", "blur_s2": "2023-05",
    "paraswap": "2023-08", "eigenlayer": "2024-05", "etherfi": "2024-06",
    "pengu": "2024-12",
}

PRE_2023 = [p for p, d in PROJECT_DATES.items() if d < "2023-01"]
POST_2023 = [p for p, d in PROJECT_DATES.items() if d >= "2023-01"]

# 25K per project keeps total around 400K - trains GBM in minutes
MAX_ROWS = 25_000

GBM_PARAMS = dict(
    n_estimators=150, max_depth=4, learning_rate=0.1,
    subsample=0.8, random_state=42,
)


# ============================================================
# DATA LOADING
# ============================================================


def load_project(project: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Load a HasciDB CSV with stratified sampling if needed."""
    csv_path = HASCIDB_DIR / f"{project}_chi26_v3.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    for col in INDICATOR_COLS:
        if col not in df.columns:
            for alt in [col.lower(), f"{col.lower()}_score"]:
                if alt in df.columns:
                    df[col] = df[alt]
                    break
            else:
                df[col] = 0

    if "ops_flag" not in df.columns:
        df["ops_flag"] = (
            (df["BT"] >= THRESHOLDS["BT"]) |
            (df["BW"] >= THRESHOLDS["BW"]) |
            (df["HF"] >= THRESHOLDS["HF"])
        ).astype(int)
    if "fund_flag" not in df.columns:
        df["fund_flag"] = (
            (df["RF"] >= THRESHOLDS["RF"]) |
            (df["MA"] >= THRESHOLDS["MA"])
        ).astype(int)
    if "is_sybil" not in df.columns:
        df["is_sybil"] = ((df["ops_flag"] == 1) | (df["fund_flag"] == 1)).astype(int)

    if len(df) > max_rows:
        sybils = df[df["is_sybil"] == 1]
        non_sybils = df[df["is_sybil"] == 0]
        rate = len(sybils) / len(df)
        n_s = min(int(max_rows * rate), len(sybils))
        n_ns = min(max_rows - n_s, len(non_sybils))
        df = pd.concat([
            sybils.sample(n=n_s, random_state=42),
            non_sybils.sample(n=n_ns, random_state=42),
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    df["project"] = project
    return df


def load_real_ai_calibration() -> dict:
    """Load real AI feature calibration from Paper 1 extraction."""
    if AI_FEATURES_FILE.exists():
        with open(AI_FEATURES_FILE) as f:
            return json.load(f)
    return {}


# ============================================================
# EVALUATION
# ============================================================


def evaluate(y_true, y_prob, y_pred=None) -> dict:
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)
    metrics = {}
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = float("nan")
    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["avg_precision"] = float("nan")
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"], metrics["fp"] = int(cm[0, 0]), int(cm[0, 1])
    metrics["fn"], metrics["tp"] = int(cm[1, 0]), int(cm[1, 1])
    return metrics


# ============================================================
# AI SYBIL GENERATION (calibrated to real distributions)
# ============================================================


def generate_ai_sybils_calibrated(
    n: int,
    ai_calibration: dict,
    nonsybil_df: pd.DataFrame,
    rng: np.random.RandomState,
    evasion_level: str = "advanced",
) -> pd.DataFrame:
    """Generate AI sybils using REAL distributions from Paper 1."""
    evasion_cfg = {
        "basic":    {"indicator_noise": 1.5, "ai_signal": 0.95},
        "moderate": {"indicator_noise": 1.0, "ai_signal": 0.75},
        "advanced": {"indicator_noise": 0.7, "ai_signal": 0.55},
    }
    cfg = evasion_cfg[evasion_level]

    ind_data = {}
    for col in INDICATOR_COLS:
        if col in nonsybil_df.columns and len(nonsybil_df) > 0:
            vals = nonsybil_df[col].values
            mean = vals.mean() * cfg["indicator_noise"]
            std = max(vals.std(), 0.1) * cfg["indicator_noise"]
            if col in ("BT", "BW", "MA"):
                ind_data[col] = np.clip(rng.normal(mean, std, n), 0, THRESHOLDS[col] - 1).astype(int)
            else:
                ind_data[col] = np.clip(rng.normal(mean, std, n), 0, THRESHOLDS[col] - 0.01)
        else:
            ind_data[col] = np.zeros(n)

    # Compute derived flags for AI sybils
    ind_data["ops_flag"] = np.zeros(n, dtype=int)
    ind_data["fund_flag"] = np.zeros(n, dtype=int)
    ind_data["is_sybil"] = np.ones(n, dtype=int)  # They ARE sybils (evading detection)

    beta_params = ai_calibration.get("beta_params", {}).get("agent", {})
    distributions = ai_calibration.get("distributions", {}).get("agent", {})
    sig = cfg["ai_signal"]

    ai_data = {}
    for feat in AI_FEATURE_NAMES:
        if feat in beta_params:
            bp = beta_params[feat]
            alpha = bp["alpha"] * sig + (1 - sig) * 2.0
            beta_p = bp["beta"] * (1 - sig * 0.3)
            ai_data[feat] = rng.beta(max(0.1, alpha), max(0.1, beta_p), n)
        elif feat in distributions:
            dist = distributions[feat]
            mean = dist.get("mean", 0.5)
            std = dist.get("std", 0.2)
            ai_data[feat] = np.clip(rng.normal(mean * sig + 0.5 * (1 - sig), std, n), 0, None)
        else:
            ai_data[feat] = rng.beta(3 * sig + 1, 3 - sig, n)

    return pd.DataFrame({**ind_data, **ai_data})


def augment_with_ai_features(
    df: pd.DataFrame,
    ai_calibration: dict,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    """Add AI features to real HasciDB data.

    CRITICAL FIX (2026-04-08): The previous version drew AI feature
    values from class-conditional distributions indexed by is_sybil
    (agent-side for sybils, human-side for non-sybils). Since
    is_sybil is correlated with fund_flag (via RF, MA), the synthetic
    AI features acted as a hidden proxy for the label, causing direct
    target leakage.

    New semantics: ALL real HasciDB rows get AI features drawn from
    the Paper 1 HUMAN distribution, regardless of the HasciDB is_sybil
    label. The assumption is that most HasciDB sybils are script-based
    sybils, not LLM-based sybils, so their AI features (gas precision,
    hour entropy, etc.) look more human than agent. Only explicitly
    LLM-generated sybils (in generate_ai_sybils_calibrated) get
    AI features from the Paper 1 AGENT distribution.

    This removes the circular shortcut. The enhanced detector now
    answers the honest question: "Given N real HasciDB addresses
    (with human-like AI features) plus M LLM sybils (with agent-like
    AI features) mixed in, can a 5+AI-feature detector separate the
    LLM sybils from the real addresses better than the 5-indicator
    detector alone?"
    """
    df = df.copy()
    n = len(df)

    beta_human = ai_calibration.get("beta_params", {}).get("human", {})
    dist_human = ai_calibration.get("distributions", {}).get("human", {})

    for feat in AI_FEATURE_NAMES:
        if feat in beta_human:
            bp = beta_human[feat]
            vals = rng.beta(max(0.1, bp["alpha"]), max(0.1, bp["beta"]), n)
        elif feat in dist_human:
            d = dist_human[feat]
            vals = np.clip(
                rng.normal(d.get("mean", 0.5), d.get("std", 0.2), n), 0, None,
            )
        else:
            vals = rng.beta(2, 5, n)
        df[feat] = vals
    return df


# ============================================================
# EXPERIMENT 1: BASELINE STATISTICS
# ============================================================


def experiment_baseline_stats(projects_data: dict) -> dict:
    print("\n" + "=" * 80)
    print("EXPERIMENT 1: FULL BASELINE STATISTICS")
    print("=" * 80)

    results = {}
    total_eligible = 0
    total_sybils = 0

    for name, df in sorted(projects_data.items()):
        n = len(df)
        n_s = int(df["is_sybil"].sum())
        n_ops = int(df["ops_flag"].sum())
        n_fund = int(df["fund_flag"].sum())
        total_eligible += n
        total_sybils += n_s

        trigger_rates = {ind: float((df[ind] >= THRESHOLDS[ind]).mean()) for ind in INDICATOR_COLS}

        results[name] = {
            "n_eligible": n, "n_sybils": n_s,
            "sybil_rate": round(n_s / n * 100, 2),
            "ops_flag_rate": round(n_ops / n * 100, 2),
            "fund_flag_rate": round(n_fund / n * 100, 2),
            "trigger_rates": trigger_rates,
            "launch_date": PROJECT_DATES.get(name, "?"),
        }
        print(f"  {name:<14} {n:>7,} eligible, {n_s:>6,} sybils ({n_s/n*100:>5.1f}%)"
              f"  ops={n_ops/n*100:.1f}% fund={n_fund/n*100:.1f}%")

    results["_aggregate"] = {
        "total_eligible": total_eligible,
        "total_sybils": total_sybils,
        "overall_sybil_rate": round(total_sybils / total_eligible * 100, 2),
    }
    print(f"\n  Total: {total_eligible:,} eligible, {total_sybils:,} sybils "
          f"({total_sybils/total_eligible*100:.1f}%)")
    return results


# ============================================================
# EXPERIMENT 2: CROSS-AXIS LOPO (NOT CIRCULAR)
# ============================================================


def experiment_cross_axis_lopo(projects_data: dict) -> dict:
    """Leave-One-Project-Out with cross-axis prediction.

    NOT circular because:
    - OPS features (BT, BW, HF) predict fund_flag which is derived ONLY from {RF, MA}
    - FUND features (RF, MA) predict ops_flag which is derived ONLY from {BT, BW, HF}
    - The input features are completely disjoint from the label derivation
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 2: CROSS-AXIS LOPO (NOT circular)")
    print("  OPS features -> fund_flag   (features DISJOINT from label)")
    print("  FUND features -> ops_flag   (features DISJOINT from label)")
    print("=" * 80)

    project_names = sorted(projects_data.keys())
    results = {"ops_to_fund": {}, "fund_to_ops": {}}

    ops_fund_aucs = []
    fund_ops_aucs = []

    for test_proj in project_names:
        test_df = projects_data[test_proj]

        # Skip if insufficient class balance
        n_fund_pos = test_df["fund_flag"].sum()
        n_ops_pos = test_df["ops_flag"].sum()

        # Collect training data from other projects
        train_dfs = [projects_data[p] for p in project_names if p != test_proj]
        train_df = pd.concat(train_dfs, ignore_index=True)

        # OPS -> fund_flag
        if n_fund_pos >= 10 and (len(test_df) - n_fund_pos) >= 10:
            X_train = train_df[OPS_COLS].values
            y_train = train_df["fund_flag"].values.astype(int)
            X_test = test_df[OPS_COLS].values
            y_test = test_df["fund_flag"].values.astype(int)

            clf = GradientBoostingClassifier(**GBM_PARAMS)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, probs)
            results["ops_to_fund"][test_proj] = metrics
            ops_fund_aucs.append(metrics["auc_roc"])
            print(f"  OPS->fund {test_proj:<14} AUC={metrics['auc_roc']:.4f} F1={metrics['f1']:.4f}")
        else:
            print(f"  OPS->fund {test_proj:<14} SKIPPED (fund_flag positives: {n_fund_pos})")

        # FUND -> ops_flag
        if n_ops_pos >= 10 and (len(test_df) - n_ops_pos) >= 10:
            X_train = train_df[FUND_COLS].values
            y_train = train_df["ops_flag"].values.astype(int)
            X_test = test_df[FUND_COLS].values
            y_test = test_df["ops_flag"].values.astype(int)

            clf = GradientBoostingClassifier(**GBM_PARAMS)
            clf.fit(X_train, y_train)
            probs = clf.predict_proba(X_test)[:, 1]
            metrics = evaluate(y_test, probs)
            results["fund_to_ops"][test_proj] = metrics
            fund_ops_aucs.append(metrics["auc_roc"])
            print(f"  FUND->ops {test_proj:<14} AUC={metrics['auc_roc']:.4f} F1={metrics['f1']:.4f}")
        else:
            print(f"  FUND->ops {test_proj:<14} SKIPPED (ops_flag positives: {n_ops_pos})")

    results["_summary"] = {
        "ops_to_fund_mean_auc": float(np.mean(ops_fund_aucs)) if ops_fund_aucs else float("nan"),
        "ops_to_fund_std_auc": float(np.std(ops_fund_aucs)) if ops_fund_aucs else float("nan"),
        "fund_to_ops_mean_auc": float(np.mean(fund_ops_aucs)) if fund_ops_aucs else float("nan"),
        "fund_to_ops_std_auc": float(np.std(fund_ops_aucs)) if fund_ops_aucs else float("nan"),
        "note": "NOT circular: features are DISJOINT from label derivation",
    }

    print(f"\n  OPS->fund LOPO AUC: {results['_summary']['ops_to_fund_mean_auc']:.4f} "
          f"+/- {results['_summary']['ops_to_fund_std_auc']:.4f}")
    print(f"  FUND->ops LOPO AUC: {results['_summary']['fund_to_ops_mean_auc']:.4f} "
          f"+/- {results['_summary']['fund_to_ops_std_auc']:.4f}")

    return results


# ============================================================
# EXPERIMENT 3: AI SYBIL EVASION (cross-axis evaluation)
# ============================================================


def experiment_ai_evasion(projects_data: dict, ai_calibration: dict) -> dict:
    """Test AI sybil evasion against cross-axis detectors.

    Since is_sybil detection is trivially AUC=1.0, we test:
    1. Rule-based evasion: do AI sybils stay below indicator thresholds?
    2. Cross-axis: can OPS features detect AI sybils that evade fund indicators?
    3. AI feature-based: can AI features detect AI sybils?
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 3: AI SYBIL EVASION (REAL distributions)")
    print("=" * 80)

    rng = np.random.RandomState(42)
    combined = pd.concat(list(projects_data.values()), ignore_index=True)
    nonsybil_df = combined[combined["is_sybil"] == 0]

    results = {}
    for level in ["basic", "moderate", "advanced"]:
        n_ai = 5000
        ai_sybils = generate_ai_sybils_calibrated(
            n_ai, ai_calibration, nonsybil_df, rng, level
        )

        # Rule-based detection
        rule_detected = 0
        for _, row in ai_sybils.head(n_ai).iterrows():
            ops = (row["BT"] >= THRESHOLDS["BT"] or
                   row["BW"] >= THRESHOLDS["BW"] or
                   row["HF"] >= THRESHOLDS["HF"])
            fund = (row["RF"] >= THRESHOLDS["RF"] or
                    row["MA"] >= THRESHOLDS["MA"])
            if ops or fund:
                rule_detected += 1

        # Cross-axis: train OPS->fund on real data, test on AI sybils
        X_train_ops = combined[OPS_COLS].values
        y_train_fund = combined["fund_flag"].values.astype(int)
        clf_ops = GradientBoostingClassifier(**GBM_PARAMS)
        clf_ops.fit(X_train_ops, y_train_fund)
        probs_ops = clf_ops.predict_proba(ai_sybils[OPS_COLS].values)[:, 1]
        ops_detected = int((probs_ops >= 0.5).sum())

        # All-5-indicator ML
        X_train_all = combined[INDICATOR_COLS].values
        y_train_sybil = combined["is_sybil"].values.astype(int)
        clf_all = GradientBoostingClassifier(**GBM_PARAMS)
        clf_all.fit(X_train_all, y_train_sybil)
        probs_all = clf_all.predict_proba(ai_sybils[INDICATOR_COLS].values)[:, 1]
        ml_detected = int((probs_all >= 0.5).sum())

        results[level] = {
            "n_ai_sybils": n_ai,
            "rule_detected": rule_detected,
            "rule_evasion_rate": float(1.0 - rule_detected / n_ai),
            "ml_5ind_detected": ml_detected,
            "ml_5ind_evasion_rate": float(1.0 - ml_detected / n_ai),
            "cross_axis_ops_detected": ops_detected,
            "cross_axis_ops_evasion_rate": float(1.0 - ops_detected / n_ai),
        }

        print(f"  {level:>8}: Rule evasion={1-rule_detected/n_ai:.1%} "
              f"ML(5-ind)={1-ml_detected/n_ai:.1%} "
              f"CrossAxis(OPS)={1-ops_detected/n_ai:.1%}")

    return results


# ============================================================
# EXPERIMENT 4: ENHANCED DETECTOR WITH AI FEATURES (cross-axis)
# ============================================================


def experiment_enhanced_detector(projects_data: dict, ai_calibration: dict) -> dict:
    """Enhanced detector using OPS + AI features to predict fund_flag.

    This adds real AI features alongside OPS indicators.
    Evaluation is cross-axis (not circular).
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: ENHANCED DETECTOR (OPS + AI features -> fund_flag)")
    print("=" * 80)

    rng = np.random.RandomState(42)
    project_names = sorted(projects_data.keys())

    # Augment all projects with AI features
    augmented = {}
    for name, df in projects_data.items():
        augmented[name] = augment_with_ai_features(df, ai_calibration, rng)

    base_features = OPS_COLS
    enhanced_features = OPS_COLS + AI_FEATURE_NAMES
    ai_only_features = AI_FEATURE_NAMES

    results = {}
    base_aucs, enh_aucs, ai_only_aucs = [], [], []

    for test_proj in project_names:
        test_df = augmented[test_proj]
        y_test = test_df["fund_flag"].values.astype(int)
        if y_test.sum() < 10 or (len(y_test) - y_test.sum()) < 10:
            continue

        train_dfs = [augmented[p] for p in project_names if p != test_proj]
        train_df = pd.concat(train_dfs, ignore_index=True)
        y_train = train_df["fund_flag"].values.astype(int)

        # Baseline: OPS only
        clf_b = GradientBoostingClassifier(**GBM_PARAMS)
        clf_b.fit(train_df[base_features].values, y_train)
        probs_b = clf_b.predict_proba(test_df[base_features].values)[:, 1]
        auc_b = roc_auc_score(y_test, probs_b) if y_test.sum() > 0 else float("nan")
        base_aucs.append(auc_b)

        # Enhanced: OPS + AI
        clf_e = GradientBoostingClassifier(**GBM_PARAMS)
        clf_e.fit(train_df[enhanced_features].values, y_train)
        probs_e = clf_e.predict_proba(test_df[enhanced_features].values)[:, 1]
        auc_e = roc_auc_score(y_test, probs_e) if y_test.sum() > 0 else float("nan")
        enh_aucs.append(auc_e)

        # AI-only
        clf_a = GradientBoostingClassifier(**GBM_PARAMS)
        clf_a.fit(train_df[ai_only_features].values, y_train)
        probs_a = clf_a.predict_proba(test_df[ai_only_features].values)[:, 1]
        auc_a = roc_auc_score(y_test, probs_a) if y_test.sum() > 0 else float("nan")
        ai_only_aucs.append(auc_a)

        # Feature importances from enhanced model
        fi = dict(zip(enhanced_features, clf_e.feature_importances_))

        results[test_proj] = {
            "baseline_auc": float(auc_b),
            "enhanced_auc": float(auc_e),
            "ai_only_auc": float(auc_a),
            "improvement": float(auc_e - auc_b),
            "top_features": {k: round(float(v), 4) for k, v in
                             sorted(fi.items(), key=lambda x: -x[1])[:5]},
        }
        print(f"  {test_proj:<14} base={auc_b:.4f} enh={auc_e:.4f} "
              f"ai_only={auc_a:.4f} delta={auc_e-auc_b:+.4f}")

    # AI Sybil recovery test
    rng2 = np.random.RandomState(42)
    combined = pd.concat(list(augmented.values()), ignore_index=True)
    nonsybil_df = combined[combined["is_sybil"] == 0]

    clf_base = GradientBoostingClassifier(**GBM_PARAMS)
    clf_base.fit(combined[base_features].values, combined["fund_flag"].values.astype(int))

    clf_enh = GradientBoostingClassifier(**GBM_PARAMS)
    clf_enh.fit(combined[enhanced_features].values, combined["fund_flag"].values.astype(int))

    recovery = {}
    for level in ["basic", "moderate", "advanced"]:
        ai_sybils = generate_ai_sybils_calibrated(
            3000, ai_calibration, nonsybil_df, rng2, level
        )
        # Need to add AI features to generated sybils for enhanced detector
        ai_sybils_aug = ai_sybils.copy()

        probs_b = clf_base.predict_proba(ai_sybils_aug[base_features].values)[:, 1]
        probs_e = clf_enh.predict_proba(ai_sybils_aug[enhanced_features].values)[:, 1]

        det_b = int((probs_b >= 0.5).sum())
        det_e = int((probs_e >= 0.5).sum())

        recovery[level] = {
            "n_ai_sybils": 3000,
            "baseline_detected": det_b,
            "baseline_rate": float(det_b / 3000),
            "enhanced_detected": det_e,
            "enhanced_rate": float(det_e / 3000),
            "recovery": float((det_e - det_b) / 3000),
        }
        print(f"  Recovery ({level}): base={det_b/3000:.1%} enh={det_e/3000:.1%} "
              f"delta={recovery[level]['recovery']:+.1%}")

    results["_summary"] = {
        "mean_baseline_auc": float(np.nanmean(base_aucs)),
        "mean_enhanced_auc": float(np.nanmean(enh_aucs)),
        "mean_ai_only_auc": float(np.nanmean(ai_only_aucs)),
        "mean_improvement": float(np.nanmean(enh_aucs) - np.nanmean(base_aucs)),
        "recovery": recovery,
    }
    return results


# ============================================================
# EXPERIMENT 5: MULTI-BASELINE COMPARISON
# ============================================================


def experiment_multi_baseline(projects_data: dict) -> dict:
    """Compare multiple detection approaches using cross-axis evaluation.

    Methods compared (all predicting fund_flag from OPS-side features):
    1. Random baseline
    2. Single indicator: BT alone, BW alone, HF alone
    3. OPS rules (any OPS indicator above threshold)
    4. GBM on OPS (3 features)
    5. RF on OPS (3 features)
    6. LR on OPS (3 features)
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 5: MULTI-BASELINE COMPARISON (cross-axis)")
    print("  All methods: OPS-side -> predict fund_flag")
    print("=" * 80)

    project_names = sorted(projects_data.keys())

    method_aucs = {
        "random": [], "rule_BT": [], "rule_BW": [], "rule_HF": [],
        "rule_any_ops": [], "gbm_ops": [], "rf_ops": [], "lr_ops": [],
    }

    for test_proj in project_names:
        test_df = projects_data[test_proj]
        y_test = test_df["fund_flag"].values.astype(int)
        if y_test.sum() < 10 or (len(y_test) - y_test.sum()) < 10:
            continue

        rng = np.random.RandomState(42)
        method_aucs["random"].append(roc_auc_score(y_test, rng.rand(len(y_test))))

        for ind in OPS_COLS:
            try:
                method_aucs[f"rule_{ind}"].append(
                    roc_auc_score(y_test, test_df[ind].values.astype(float))
                )
            except ValueError:
                pass

        ops_score = np.zeros(len(test_df))
        for ind in OPS_COLS:
            ops_score += (test_df[ind] >= THRESHOLDS[ind]).astype(float)
        try:
            method_aucs["rule_any_ops"].append(roc_auc_score(y_test, ops_score))
        except ValueError:
            pass

        train_dfs = [projects_data[p] for p in project_names if p != test_proj]
        train_df = pd.concat(train_dfs, ignore_index=True)
        X_train = train_df[OPS_COLS].values
        y_train = train_df["fund_flag"].values.astype(int)
        X_test = test_df[OPS_COLS].values

        clf_gbm = GradientBoostingClassifier(**GBM_PARAMS)
        clf_gbm.fit(X_train, y_train)
        try:
            method_aucs["gbm_ops"].append(roc_auc_score(y_test, clf_gbm.predict_proba(X_test)[:, 1]))
        except ValueError:
            pass

        clf_rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42, n_jobs=-1)
        clf_rf.fit(X_train, y_train)
        try:
            method_aucs["rf_ops"].append(roc_auc_score(y_test, clf_rf.predict_proba(X_test)[:, 1]))
        except ValueError:
            pass

        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        clf_lr = LogisticRegression(max_iter=1000, random_state=42)
        clf_lr.fit(X_train_sc, y_train)
        try:
            method_aucs["lr_ops"].append(roc_auc_score(y_test, clf_lr.predict_proba(X_test_sc)[:, 1]))
        except ValueError:
            pass

    summary = {}
    print(f"\n  {'Method':<20} {'Mean AUC':>10} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-" * 60)
    for method, aucs in method_aucs.items():
        if not aucs:
            continue
        s = {
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "min_auc": float(np.min(aucs)),
            "max_auc": float(np.max(aucs)),
            "n_projects": len(aucs),
        }
        summary[method] = s
        print(f"  {method:<20} {s['mean_auc']:>10.4f} {s['std_auc']:>8.4f} "
              f"{s['min_auc']:>8.4f} {s['max_auc']:>8.4f}")

    return {"per_project_aucs": {k: [float(v) for v in vs] for k, vs in method_aucs.items()},
            "summary": summary}


# ============================================================
# EXPERIMENT 6: TEMPORAL ANALYSIS (cross-axis)
# ============================================================


def experiment_temporal(projects_data: dict) -> dict:
    """Train on pre-2023 OPS->fund, test on post-2023 OPS->fund.

    Tests whether cross-axis prediction degrades over time.
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT 6: TEMPORAL ANALYSIS (cross-axis)")
    print(f"  Pre-2023 train: {[p for p in PRE_2023 if p in projects_data]}")
    print(f"  Post-2023 test:  {[p for p in POST_2023 if p in projects_data]}")
    print("=" * 80)

    train_dfs = [projects_data[p] for p in PRE_2023 if p in projects_data]
    if not train_dfs:
        return {"skipped": True}

    train_df = pd.concat(train_dfs, ignore_index=True)
    X_train = train_df[OPS_COLS].values
    y_train = train_df["fund_flag"].values.astype(int)

    print(f"\n  Train: {len(train_df):,} rows ({y_train.sum():,} fund_flag positives)")

    clf = GradientBoostingClassifier(**GBM_PARAMS)
    clf.fit(X_train, y_train)

    results = {"train_n": len(train_df)}
    per_project = {}

    for p in POST_2023:
        if p not in projects_data:
            continue
        test_df = projects_data[p]
        y_test = test_df["fund_flag"].values.astype(int)
        if y_test.sum() < 5 or (len(y_test) - y_test.sum()) < 5:
            print(f"  {p}: SKIPPED (insufficient fund_flag)")
            continue

        probs = clf.predict_proba(test_df[OPS_COLS].values)[:, 1]
        metrics = evaluate(y_test, probs)
        per_project[p] = {
            "date": PROJECT_DATES.get(p, "?"),
            "n_test": len(test_df),
            "fund_flag_rate": float(y_test.mean()),
            "metrics": metrics,
        }
        print(f"  {p:<14} ({PROJECT_DATES.get(p, '?')}) AUC={metrics['auc_roc']:.4f} "
              f"F1={metrics['f1']:.4f}")

    results["per_project"] = per_project
    test_aucs = [v["metrics"]["auc_roc"] for v in per_project.values()
                 if not np.isnan(v["metrics"]["auc_roc"])]
    if test_aucs:
        results["mean_test_auc"] = float(np.mean(test_aucs))
        results["std_test_auc"] = float(np.std(test_aucs))
        print(f"\n  Temporal test AUC: {results['mean_test_auc']:.4f} "
              f"+/- {results['std_test_auc']:.4f}")

    # Also test ALL indicators -> is_sybil (circular reference, for comparison)
    clf_circ = GradientBoostingClassifier(**GBM_PARAMS)
    clf_circ.fit(train_df[INDICATOR_COLS].values, train_df["is_sybil"].values.astype(int))
    circ_results = {}
    for p in POST_2023:
        if p not in projects_data:
            continue
        test_df = projects_data[p]
        y_test_circ = test_df["is_sybil"].values.astype(int)
        probs_circ = clf_circ.predict_proba(test_df[INDICATOR_COLS].values)[:, 1]
        try:
            circ_auc = roc_auc_score(y_test_circ, probs_circ)
        except ValueError:
            circ_auc = float("nan")
        circ_results[p] = float(circ_auc)
        print(f"  (circular ref) {p}: AUC={circ_auc:.4f}")

    results["circular_reference_for_comparison"] = circ_results
    results["note"] = (
        "Cross-axis AUC measures TRUE predictive power. "
        "Circular AUC (shown for reference) is always ~1.0 because "
        "is_sybil = f(BT,BW,HF,RF,MA)."
    )

    return results


# ============================================================
# MAIN
# ============================================================


def main():
    start = time.time()

    print("=" * 80)
    print("PAPER 3: LARGE-SCALE CROSS-AXIS EXPERIMENT")
    print("  Uses cross-axis prediction (NOT circular)")
    print("  OPS features -> fund_flag / FUND features -> ops_flag")
    print("=" * 80)

    print("\nLoading HasciDB data...")
    projects_data = {}
    for proj in PROJECTS:
        print(f"  {proj}...", end=" ")
        df = load_project(proj)
        if df.empty:
            print("SKIPPED")
            continue
        projects_data[proj] = df
        n_s = df["is_sybil"].sum()
        print(f"OK ({len(df):,} rows, {n_s:,} sybils)")

    print(f"\nLoaded {len(projects_data)} projects")

    ai_calibration = load_real_ai_calibration()
    if ai_calibration:
        n_a = ai_calibration.get("metadata", {}).get("n_agents", "?")
        n_h = ai_calibration.get("metadata", {}).get("n_humans", "?")
        print(f"Loaded real AI calibration ({n_a} agents, {n_h} humans)")
    else:
        print("WARNING: No AI calibration file. Using fallback distributions.")

    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "projects": list(projects_data.keys()),
            "max_rows_per_project": MAX_ROWS,
            "gbm_params": GBM_PARAMS,
            "ai_calibration_available": bool(ai_calibration),
            "design_note": (
                "ALL ML evaluations use cross-axis prediction to avoid "
                "circular reasoning. is_sybil = f(BT,BW,HF,RF,MA), so "
                "predicting is_sybil from the same 5 indicators is trivially "
                "AUC=1.0. Cross-axis: OPS features predict fund_flag "
                "(derived only from RF, MA) and vice versa."
            ),
        }
    }

    all_results["exp1_baseline_stats"] = experiment_baseline_stats(projects_data)
    all_results["exp2_cross_axis_lopo"] = experiment_cross_axis_lopo(projects_data)
    all_results["exp3_ai_evasion"] = experiment_ai_evasion(projects_data, ai_calibration)
    all_results["exp4_enhanced_detector"] = experiment_enhanced_detector(
        projects_data, ai_calibration
    )
    all_results["exp5_multi_baseline"] = experiment_multi_baseline(projects_data)
    all_results["exp6_temporal"] = experiment_temporal(projects_data)

    elapsed = time.time() - start
    all_results["metadata"]["elapsed_seconds"] = round(elapsed, 1)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 80}")
    print("KEY RESULTS SUMMARY")
    print("=" * 80)

    lopo = all_results.get("exp2_cross_axis_lopo", {}).get("_summary", {})
    print(f"  Cross-Axis OPS->fund AUC: {lopo.get('ops_to_fund_mean_auc', '?'):.4f} "
          f"+/- {lopo.get('ops_to_fund_std_auc', '?'):.4f}")
    print(f"  Cross-Axis FUND->ops AUC: {lopo.get('fund_to_ops_mean_auc', '?'):.4f} "
          f"+/- {lopo.get('fund_to_ops_std_auc', '?'):.4f}")
    print("    (These are NOT circular: features are disjoint from labels)")

    evasion = all_results.get("exp3_ai_evasion", {})
    for level in ["basic", "moderate", "advanced"]:
        if level in evasion:
            e = evasion[level]
            print(f"  AI Evasion ({level}): Rule={e.get('rule_evasion_rate',0):.1%} "
                  f"ML={e.get('ml_5ind_evasion_rate',0):.1%}")

    enhanced = all_results.get("exp4_enhanced_detector", {}).get("_summary", {})
    print(f"  Enhanced AUC: {enhanced.get('mean_enhanced_auc', '?'):.4f} "
          f"(improvement: {enhanced.get('mean_improvement', '?'):+.4f})")


if __name__ == "__main__":
    main()
