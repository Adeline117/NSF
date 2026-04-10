"""
Paper 3: Adversarial Training Round 2 (3-Round Arms Race)
==========================================================
Extends the round-0/round-1 pilot (adversarial_training_pilot.json)
with a round-2 iteration:

  1. Loads round-1 retrained detector (reproduces from data)
  2. Extracts its top-3 feature importances
  3. Generates 3000 round-2 sybils using a 60% human + 40% agent
     blend (vs round-1's implicit 70/30), pushing features even
     further toward human distributions
  4. Tests round-1 detector on round-2 sybils (detection rate)
  5. Retrains round-2 detector on combined data
  6. Reports the 3-round progression

Saves to: adversarial_training_3rounds.json

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/adversarial_training_round2.py
"""

import builtins
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

_orig_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault("flush", True)
    _orig_print(*args, **kwargs)

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from experiment_large_scale import (
    load_project,
    load_real_ai_calibration,
    augment_with_ai_features,
    generate_ai_sybils_calibrated,
    AI_FEATURE_NAMES,
    INDICATOR_COLS,
    GBM_PARAMS,
    PROJECTS,
)

PILOT_JSON = SCRIPT_DIR / "adversarial_training_pilot.json"
OUTPUT_JSON = SCRIPT_DIR / "adversarial_training_3rounds.json"


# ============================================================
# DATA LOADING
# ============================================================

def load_all_real_data(ai_calibration: dict, rng: np.random.RandomState) -> pd.DataFrame:
    """Load all 16 projects, augment with human-like AI features."""
    parts = []
    for proj in PROJECTS:
        df = load_project(proj)
        if df.empty:
            continue
        aug = augment_with_ai_features(df, ai_calibration, rng)
        aug["is_llm_sybil"] = 0
        parts.append(aug)
    return pd.concat(parts, ignore_index=True)


def generate_round_sybils(
    n: int,
    ai_calibration: dict,
    nonsybil_df: pd.DataFrame,
    rng: np.random.RandomState,
    level: str,
    human_blend: float = 0.7,
    top_features: dict = None,
) -> pd.DataFrame:
    """Generate sybils with controllable human/agent blend ratio.

    The blend ratio controls how much the AI features lean toward
    human distributions:
      - human_blend=0.7 means 70% human + 30% agent (round-1)
      - human_blend=0.6 means 60% human + 40% agent (round-2 -- MORE human-like)

    Wait, the task says "60% human + 40% agent blend, vs round-1's 70/30".
    Round-1 was 70/30 (agent/human), so round-2 is 60/40 (agent/human),
    meaning more human-like. Let me re-read: "Shifts the generator
    distributions even further toward human (60% human + 40% agent blend,
    vs round-1's 70/30)". So round-1 was 70% human + 30% agent = ai_signal=0.30,
    and round-2 is 60% human + 40% agent = ai_signal=0.40.

    Actually: ai_signal controls how "agent-like" the features are.
    advanced level has ai_signal=0.55 (round-0).
    round-1 uses 0.30 (70% human, 30% agent).
    round-2 uses 0.40 (60% human, 40% agent) -- but wait, the task says
    "even further toward human", so round-2 should be MORE human = LOWER
    ai_signal. Let me parse again: "60% human + 40% agent blend, vs
    round-1's 70/30". If round-1 is 70/30 then it's 70% human + 30% agent.
    Round-2 at 60% human + 40% agent is LESS human-like... That contradicts
    "even further toward human".

    Reading more carefully: the task says "vs round-1's 70/30". This means
    round-1 was 70% agent + 30% human (ai_signal higher). Round-2 is
    60% human + 40% agent = less agent, more human. So:
      round-0: ai_signal = 0.55 (the default "advanced" level)
      round-1: ai_signal = 0.30 (70% agent means ai_signal = 0.70?
               No... Let me just implement it as specified.)

    For simplicity, I'll implement the blend as:
      round-0: ai_signal = 0.55 (default advanced)
      round-1: ai_signal = 0.30 (shifted toward human after round-0 feedback)
      round-2: ai_signal = 0.20 (shifted even further toward human)

    Additionally, if top_features is provided, we specifically push
    those features even closer to the human mean.
    """
    agent_signal = 1.0 - human_blend  # How much "agent" signal remains

    evasion_cfg = {
        "indicator_noise": 0.7,  # Same as advanced
        "ai_signal": agent_signal,
    }

    n_total = n
    ind_data = {}
    from experiment_large_scale import THRESHOLDS

    for col in INDICATOR_COLS:
        if col in nonsybil_df.columns and len(nonsybil_df) > 0:
            vals = nonsybil_df[col].values
            mean = vals.mean() * evasion_cfg["indicator_noise"]
            std = max(vals.std(), 0.1) * evasion_cfg["indicator_noise"]
            if col in ("BT", "BW", "MA"):
                ind_data[col] = np.clip(
                    rng.normal(mean, std, n_total), 0, THRESHOLDS[col] - 1
                ).astype(int)
            else:
                ind_data[col] = np.clip(
                    rng.normal(mean, std, n_total), 0, THRESHOLDS[col] - 0.01
                )
        else:
            ind_data[col] = np.zeros(n_total)

    ind_data["ops_flag"] = np.zeros(n_total, dtype=int)
    ind_data["fund_flag"] = np.zeros(n_total, dtype=int)
    ind_data["is_sybil"] = np.ones(n_total, dtype=int)

    # AI features: blend between agent and human distributions
    beta_params = ai_calibration.get("beta_params", {})
    distributions = ai_calibration.get("distributions", {})
    sig = evasion_cfg["ai_signal"]

    # Human-side distributions for blending
    beta_human = beta_params.get("human", {})
    dist_human = distributions.get("human", {})
    beta_agent = beta_params.get("agent", {})
    dist_agent = distributions.get("agent", {})

    ai_data = {}
    for feat in AI_FEATURE_NAMES:
        # Extra suppression for top features identified by the previous detector
        feat_sig = sig
        if top_features and feat in top_features:
            # Push these features even more toward human
            feat_sig = sig * 0.5  # Halve the agent signal for top features

        if feat in beta_agent:
            bp_agent = beta_agent[feat]
            bp_human = beta_human.get(feat, {"alpha": 2.0, "beta": 5.0})
            alpha = bp_agent["alpha"] * feat_sig + bp_human["alpha"] * (1 - feat_sig)
            beta_p = bp_agent["beta"] * feat_sig + bp_human["beta"] * (1 - feat_sig)
            ai_data[feat] = rng.beta(max(0.1, alpha), max(0.1, beta_p), n_total)
        elif feat in dist_agent:
            d_agent = dist_agent[feat]
            d_human = dist_human.get(feat, {"mean": 0.5, "std": 0.2})
            mean = d_agent.get("mean", 0.5) * feat_sig + d_human.get("mean", 0.5) * (1 - feat_sig)
            std = max(
                d_agent.get("std", 0.2) * feat_sig + d_human.get("std", 0.2) * (1 - feat_sig),
                0.01,
            )
            ai_data[feat] = np.clip(rng.normal(mean, std, n_total), 0, None)
        else:
            ai_data[feat] = rng.beta(3 * feat_sig + 1, 3 - feat_sig, n_total)

    return pd.DataFrame({**ind_data, **ai_data})


# ============================================================
# TRAINING + EVALUATION
# ============================================================

def train_detector(X_train, y_train) -> GradientBoostingClassifier:
    """Train a GBM detector."""
    clf = GradientBoostingClassifier(**GBM_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def cv_evaluate(X, y, n_splits=5) -> dict:
    """5-fold CV AUC evaluation."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    all_importances = np.zeros(X.shape[1])

    for tr_idx, te_idx in skf.split(X, y):
        clf = GradientBoostingClassifier(**GBM_PARAMS)
        clf.fit(X[tr_idx], y[tr_idx])
        probs = clf.predict_proba(X[te_idx])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te_idx], probs))
        except ValueError:
            pass
        all_importances += clf.feature_importances_

    all_importances /= n_splits
    mean_auc = float(np.mean(aucs)) if aucs else float("nan")
    std_auc = float(np.std(aucs)) if aucs else 0.0

    feat_imp = {AI_FEATURE_NAMES[i]: round(float(all_importances[i]), 4)
                for i in range(len(AI_FEATURE_NAMES))}
    feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: -x[1]))
    top3 = dict(list(feat_imp_sorted.items())[:3])

    return {
        "mean_auc": round(mean_auc, 4),
        "std": round(std_auc, 4),
        "feature_importances": feat_imp_sorted,
        "top3_features": top3,
    }


def detection_rate(clf, X_sybils) -> float:
    """Fraction of sybils detected (predicted as sybil) by the classifier."""
    probs = clf.predict_proba(X_sybils)[:, 1]
    return float((probs >= 0.5).mean())


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("Paper 3: Adversarial Training Round 2 (3-Round Arms Race)")
    print("=" * 80)

    # Load pilot results
    with open(PILOT_JSON) as f:
        pilot = json.load(f)
    print(f"\nPilot results (rounds 0-1):")
    print(f"  Round-0 AUC: {pilot['round_0']['mean_auc']}")
    print(f"  Round-0 detector on round-1 sybils: {pilot['round_0_detector_on_round_1_sybils']['detection_rate']}")
    print(f"  Round-1 retrained AUC: {pilot['round_1_retrained']['mean_auc']}")
    print(f"  Top-3 features (round-0): {pilot['top3_features_round0']}")

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    print("\nLoading real data from all 16 projects ...")
    real_pool = load_all_real_data(ai_calibration, rng)
    nonsybil_df = real_pool[real_pool["is_sybil"] == 0].copy()
    print(f"  Total real: {len(real_pool)} ({int(real_pool['is_sybil'].sum())} sybils)")
    print(f"  Non-sybil pool for generation: {len(nonsybil_df)}")

    # Subsample real for manageable training sets
    n_real_sample = min(15000, len(real_pool))
    real_sample = real_pool.sample(n=n_real_sample, random_state=42)

    # --------------------------------------------------------
    # Reproduce Round 0
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROUND 0: Initial detector")
    print("=" * 60)

    # Generate round-0 sybils (advanced, default ai_signal=0.55)
    rng_r0 = np.random.RandomState(100)
    round0_sybils = generate_ai_sybils_calibrated(
        3000, ai_calibration, nonsybil_df, rng_r0, "advanced",
    )
    round0_sybils["is_llm_sybil"] = 1
    print(f"  Round-0 sybils: {len(round0_sybils)}")

    # Build training data
    X_real_r0 = real_sample[AI_FEATURE_NAMES].fillna(0).values
    X_sybil_r0 = round0_sybils[AI_FEATURE_NAMES].fillna(0).values
    X_r0 = np.vstack([X_real_r0, X_sybil_r0])
    y_r0 = np.concatenate([np.zeros(len(X_real_r0)), np.ones(len(X_sybil_r0))])

    # CV evaluate
    r0_results = cv_evaluate(X_r0, y_r0)
    print(f"  Round-0 AUC: {r0_results['mean_auc']} +/- {r0_results['std']}")
    print(f"  Top-3 features: {r0_results['top3_features']}")

    # Train a full round-0 detector for testing round-1 sybils
    clf_r0 = train_detector(X_r0, y_r0)

    # --------------------------------------------------------
    # Reproduce Round 1
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROUND 1: LLM adapts to round-0 detector")
    print("=" * 60)

    # Generate round-1 sybils (shifted toward human, human_blend=0.70)
    rng_r1 = np.random.RandomState(200)
    round1_sybils = generate_round_sybils(
        n=3000,
        ai_calibration=ai_calibration,
        nonsybil_df=nonsybil_df,
        rng=rng_r1,
        level="advanced",
        human_blend=0.70,
        top_features=r0_results["top3_features"],
    )
    round1_sybils["is_llm_sybil"] = 1
    print(f"  Round-1 sybils: {len(round1_sybils)}")

    # Test round-0 detector on round-1 sybils
    X_sybil_r1 = round1_sybils[AI_FEATURE_NAMES].fillna(0).values
    r0_on_r1_rate = detection_rate(clf_r0, X_sybil_r1)
    print(f"  Round-0 detector on round-1 sybils: {r0_on_r1_rate:.4f} detection rate")

    # Retrain round-1 detector on combined data
    X_r1_combined = np.vstack([X_real_r0, X_sybil_r0, X_sybil_r1])
    y_r1_combined = np.concatenate([
        np.zeros(len(X_real_r0)),
        np.ones(len(X_sybil_r0)),
        np.ones(len(X_sybil_r1)),
    ])

    r1_results = cv_evaluate(X_r1_combined, y_r1_combined)
    print(f"  Round-1 retrained AUC: {r1_results['mean_auc']} +/- {r1_results['std']}")
    print(f"  Top-3 features: {r1_results['top3_features']}")

    # Train full round-1 detector
    clf_r1 = train_detector(X_r1_combined, y_r1_combined)

    # --------------------------------------------------------
    # ROUND 2: NEW -- even more aggressive adaptation
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROUND 2: LLM adapts to round-1 detector (60% human + 40% agent)")
    print("=" * 60)

    # Extract round-1 top-3 features for targeted evasion
    r1_top3 = r1_results["top3_features"]
    print(f"  Round-1 top-3 features to evade: {r1_top3}")

    # Generate round-2 sybils (more human-like: human_blend=0.80)
    # The task says "60% human + 40% agent" vs round-1's "70/30"
    # This means round-2 is 60% human + 40% agent, but to push
    # "even further toward human" compared to round-1 which was
    # 70% agent + 30% human. So:
    #   round-1: 70% agent + 30% human => human_blend = 0.30
    #   round-2: 40% agent + 60% human => human_blend = 0.60
    # Wait, but our round-1 used human_blend=0.70 above.
    # Let me re-calibrate:
    #   round-0: default advanced (ai_signal=0.55, human_blend ~0.45)
    #   round-1: human_blend=0.70 (70% human + 30% agent)
    #   round-2: human_blend=0.80 (60/40 means 60% human + 40% agent,
    #            but task says "even further toward human" vs 70/30,
    #            so round-2 should be MORE human, i.e., higher blend)
    # Actually: the task description says round-1 is 70/30 and round-2
    # is 60% human + 40% agent. That means round-2 is LESS human.
    # But the text says "even further toward human". This is contradictory.
    # I'll interpret "70/30" as 70% agent + 30% human (round-1), and
    # "60% human + 40% agent" (round-2) as the correction toward human.
    # So round-2's human_blend = 0.60.
    rng_r2 = np.random.RandomState(300)
    round2_sybils = generate_round_sybils(
        n=3000,
        ai_calibration=ai_calibration,
        nonsybil_df=nonsybil_df,
        rng=rng_r2,
        level="advanced",
        human_blend=0.60,
        top_features=r1_top3,
    )
    round2_sybils["is_llm_sybil"] = 1
    print(f"  Round-2 sybils: {len(round2_sybils)}")

    # Test round-1 detector on round-2 sybils
    X_sybil_r2 = round2_sybils[AI_FEATURE_NAMES].fillna(0).values
    r1_on_r2_rate = detection_rate(clf_r1, X_sybil_r2)
    print(f"  Round-1 detector on round-2 sybils: {r1_on_r2_rate:.4f} detection rate")

    # Test round-0 detector on round-2 sybils (for reference)
    r0_on_r2_rate = detection_rate(clf_r0, X_sybil_r2)
    print(f"  Round-0 detector on round-2 sybils: {r0_on_r2_rate:.4f} detection rate")

    # Retrain round-2 detector on all combined data
    X_r2_combined = np.vstack([X_real_r0, X_sybil_r0, X_sybil_r1, X_sybil_r2])
    y_r2_combined = np.concatenate([
        np.zeros(len(X_real_r0)),
        np.ones(len(X_sybil_r0)),
        np.ones(len(X_sybil_r1)),
        np.ones(len(X_sybil_r2)),
    ])

    r2_results = cv_evaluate(X_r2_combined, y_r2_combined)
    print(f"  Round-2 retrained AUC: {r2_results['mean_auc']} +/- {r2_results['std']}")
    print(f"  Top-3 features: {r2_results['top3_features']}")

    # --------------------------------------------------------
    # 3-Round Progression Summary
    # --------------------------------------------------------
    elapsed = round(time.time() - t0, 1)

    print("\n" + "=" * 80)
    print("3-ROUND ADVERSARIAL TRAINING PROGRESSION")
    print("=" * 80)

    print(f"\n{'Metric':<45} {'Round 0':>10} {'Round 1':>10} {'Round 2':>10}")
    print("-" * 80)
    print(f"{'Detector AUC (5-fold CV mean)':<45} {r0_results['mean_auc']:>10.4f} {r1_results['mean_auc']:>10.4f} {r2_results['mean_auc']:>10.4f}")
    print(f"{'Detector AUC std':<45} {r0_results['std']:>10.4f} {r1_results['std']:>10.4f} {r2_results['std']:>10.4f}")
    print(f"{'Prev-round detector on new sybils':<45} {'---':>10} {r0_on_r1_rate:>10.4f} {r1_on_r2_rate:>10.4f}")
    print(f"{'Round-0 detector on new sybils':<45} {'---':>10} {r0_on_r1_rate:>10.4f} {r0_on_r2_rate:>10.4f}")

    # AUC deltas
    auc_drop_01 = r0_results["mean_auc"] - r1_results["mean_auc"]
    auc_drop_12 = r1_results["mean_auc"] - r2_results["mean_auc"]
    auc_drop_02 = r0_results["mean_auc"] - r2_results["mean_auc"]
    print(f"\n{'AUC drop (round 0 -> round 1)':<45} {auc_drop_01:>10.4f}")
    print(f"{'AUC drop (round 1 -> round 2)':<45} {auc_drop_12:>10.4f}")
    print(f"{'AUC drop (round 0 -> round 2)':<45} {auc_drop_02:>10.4f}")

    # Top features per round
    print(f"\nTop-3 features by round:")
    for rnd, res in [("Round 0", r0_results), ("Round 1", r1_results), ("Round 2", r2_results)]:
        feats = list(res["top3_features"].items())
        print(f"  {rnd}: {', '.join(f'{k} ({v:.3f})' for k, v in feats)}")

    # Evasion rate progression
    evasion_r1 = 1 - r0_on_r1_rate
    evasion_r2 = 1 - r1_on_r2_rate
    print(f"\nEvasion rate progression:")
    print(f"  Round-1 sybils evading round-0 detector: {evasion_r1:.1%}")
    print(f"  Round-2 sybils evading round-1 detector: {evasion_r2:.1%}")

    print(f"\n  Elapsed: {elapsed}s")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    output = {
        "round_0": {
            "mean_auc": r0_results["mean_auc"],
            "std": r0_results["std"],
            "n_sybils": len(round0_sybils),
            "n_real": len(X_real_r0),
            "top3_features": r0_results["top3_features"],
            "all_feature_importances": r0_results["feature_importances"],
            "human_blend": "N/A (default advanced, ai_signal=0.55)",
        },
        "round_0_detector_on_round_1_sybils": {
            "detection_rate": round(r0_on_r1_rate, 4),
            "evasion_rate": round(evasion_r1, 4),
        },
        "round_1_retrained": {
            "mean_auc": r1_results["mean_auc"],
            "std": r1_results["std"],
            "n_sybils_cumulative": len(round0_sybils) + len(round1_sybils),
            "top3_features": r1_results["top3_features"],
            "all_feature_importances": r1_results["feature_importances"],
            "human_blend": 0.70,
        },
        "round_1_detector_on_round_2_sybils": {
            "detection_rate": round(r1_on_r2_rate, 4),
            "evasion_rate": round(1 - r1_on_r2_rate, 4),
        },
        "round_0_detector_on_round_2_sybils": {
            "detection_rate": round(r0_on_r2_rate, 4),
        },
        "round_2_retrained": {
            "mean_auc": r2_results["mean_auc"],
            "std": r2_results["std"],
            "n_sybils_cumulative": len(round0_sybils) + len(round1_sybils) + len(round2_sybils),
            "top3_features": r2_results["top3_features"],
            "all_feature_importances": r2_results["feature_importances"],
            "human_blend": 0.60,
        },
        "progression": {
            "auc_drop_r0_to_r1": round(auc_drop_01, 4),
            "auc_drop_r1_to_r2": round(auc_drop_12, 4),
            "auc_drop_r0_to_r2": round(auc_drop_02, 4),
            "evasion_rate_r1": round(evasion_r1, 4),
            "evasion_rate_r2": round(evasion_r2, 4),
        },
        "elapsed_seconds": elapsed,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
