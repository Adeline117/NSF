"""
Paper 3: 10-Round Adversarial Equilibrium Analysis
====================================================
Extends the 3-round arms race to 10 rounds to analyze convergence
and potential Nash equilibrium in the attacker-detector game.

Each round:
  1. Attacker generates 3000 sybils targeting the previous detector's
     top-3 features, with progressively more human-like distributions
  2. Previous detector is tested on new sybils (evasion rate)
  3. Detector is retrained on cumulative data (AUC)

Outputs:
  - adversarial_equilibrium_10rounds.json: Full per-round metrics
  - ../figures/fig_equilibrium_convergence.pdf: Convergence curves

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/adversarial_equilibrium_10rounds.py
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
FIGURES_DIR = SCRIPT_DIR.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

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
    THRESHOLDS,
)

OUTPUT_JSON = SCRIPT_DIR / "adversarial_equilibrium_10rounds.json"
OUTPUT_FIG = FIGURES_DIR / "fig_equilibrium_convergence.pdf"

N_ROUNDS = 10
N_SYBILS_PER_ROUND = 3000
N_REAL_SAMPLE = 15000
N_CV_FOLDS = 5

# Human blend schedule: progressively more human-like
# Round 0: default advanced (ai_signal=0.55 → human_blend≈0.45)
# Rounds 1-9: increasing human blend with diminishing increments
HUMAN_BLEND_SCHEDULE = [
    None,  # Round 0: use default advanced
    0.70,  # Round 1
    0.78,  # Round 2
    0.84,  # Round 3
    0.88,  # Round 4
    0.91,  # Round 5
    0.93,  # Round 6
    0.95,  # Round 7
    0.96,  # Round 8
    0.97,  # Round 9
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_all_real_data(ai_calibration: dict, rng: np.random.RandomState) -> pd.DataFrame:
    """Load all 16 projects, augment with AI features."""
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
    human_blend: float,
    top_features: dict = None,
) -> pd.DataFrame:
    """Generate sybils with specified human/agent blend and targeted evasion."""
    agent_signal = 1.0 - human_blend

    evasion_cfg = {"indicator_noise": 0.7, "ai_signal": agent_signal}

    ind_data = {}
    for col in INDICATOR_COLS:
        if col in nonsybil_df.columns and len(nonsybil_df) > 0:
            vals = nonsybil_df[col].values
            mean = vals.mean() * evasion_cfg["indicator_noise"]
            std = max(vals.std(), 0.1) * evasion_cfg["indicator_noise"]
            if col in ("BT", "BW", "MA"):
                ind_data[col] = np.clip(
                    rng.normal(mean, std, n), 0, THRESHOLDS[col] - 1
                ).astype(int)
            else:
                ind_data[col] = np.clip(
                    rng.normal(mean, std, n), 0, THRESHOLDS[col] - 0.01
                )
        else:
            ind_data[col] = np.zeros(n)

    ind_data["ops_flag"] = np.zeros(n, dtype=int)
    ind_data["fund_flag"] = np.zeros(n, dtype=int)
    ind_data["is_sybil"] = np.ones(n, dtype=int)

    # AI features: blend between agent and human distributions
    beta_params = ai_calibration.get("beta_params", {})
    distributions = ai_calibration.get("distributions", {})
    sig = evasion_cfg["ai_signal"]

    beta_human = beta_params.get("human", {})
    beta_agent = beta_params.get("agent", {})
    dist_human = distributions.get("human", {})
    dist_agent = distributions.get("agent", {})

    ai_data = {}
    for feat in AI_FEATURE_NAMES:
        feat_sig = sig
        if top_features and feat in top_features:
            feat_sig = sig * 0.3  # Aggressively suppress top features

        if feat in beta_agent:
            bp_agent = beta_agent[feat]
            bp_human = beta_human.get(feat, {"alpha": 2.0, "beta": 5.0})
            alpha = bp_agent["alpha"] * feat_sig + bp_human["alpha"] * (1 - feat_sig)
            beta_p = bp_agent["beta"] * feat_sig + bp_human["beta"] * (1 - feat_sig)
            ai_data[feat] = rng.beta(max(0.1, alpha), max(0.1, beta_p), n)
        elif feat in dist_agent:
            d_agent = dist_agent[feat]
            d_human = dist_human.get(feat, {"mean": 0.5, "std": 0.2})
            mean = d_agent.get("mean", 0.5) * feat_sig + d_human.get("mean", 0.5) * (1 - feat_sig)
            std = max(
                d_agent.get("std", 0.2) * feat_sig + d_human.get("std", 0.2) * (1 - feat_sig),
                0.01,
            )
            ai_data[feat] = np.clip(rng.normal(mean, std, n), 0, None)
        else:
            ai_data[feat] = rng.beta(3 * feat_sig + 1, 3 - feat_sig, n)

    return pd.DataFrame({**ind_data, **ai_data})


def cv_evaluate(X, y, n_splits=N_CV_FOLDS) -> dict:
    """K-fold CV AUC evaluation with feature importances."""
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


def train_detector(X_train, y_train) -> GradientBoostingClassifier:
    clf = GradientBoostingClassifier(**GBM_PARAMS)
    clf.fit(X_train, y_train)
    return clf


def detection_rate(clf, X_sybils) -> float:
    probs = clf.predict_proba(X_sybils)[:, 1]
    return float((probs >= 0.5).mean())


# ============================================================
# FIGURE GENERATION
# ============================================================

def generate_convergence_figure(rounds_data: list):
    """Generate convergence curves: AUC & evasion rate over rounds."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        rounds = list(range(len(rounds_data)))
        aucs = [r["detector_auc"] for r in rounds_data]
        evasion_rates = [r.get("evasion_rate", 0) for r in rounds_data]
        # Round 0 has no evasion rate (no previous detector)
        evasion_rates[0] = None

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

        # Panel 1: Detector AUC over rounds
        ax1.plot(rounds, aucs, "b-o", linewidth=2, markersize=8, label="Detector AUC")
        ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
        ax1.set_xlabel("Round", fontsize=12)
        ax1.set_ylabel("AUC (5-fold CV)", fontsize=12)
        ax1.set_title("Detector Performance", fontsize=13)
        ax1.set_ylim(0.45, 1.02)
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

        # Panel 2: Evasion rate over rounds
        valid_rounds = [r for r in rounds if evasion_rates[r] is not None]
        valid_evasion = [evasion_rates[r] for r in valid_rounds]
        ax2.plot(valid_rounds, valid_evasion, "r-s", linewidth=2, markersize=8, label="Evasion rate")
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel("Evasion Rate", fontsize=12)
        ax2.set_title("Attacker Evasion", fontsize=13)
        ax2.set_ylim(-0.05, 1.1)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Feature importance evolution (top-3 features)
        # Track the top-3 features from round 0 across all rounds
        all_feats = set()
        for r in rounds_data:
            all_feats.update(r["top3_features"].keys())
        top_feats = list(rounds_data[0]["top3_features"].keys())[:3]

        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00"]
        for i, feat in enumerate(top_feats):
            vals = [r["feature_importances"].get(feat, 0) for r in rounds_data]
            short_name = feat.replace("_", "\n", 1) if len(feat) > 15 else feat
            ax3.plot(rounds, vals, "-D", color=colors[i % len(colors)],
                     linewidth=2, markersize=6, label=feat)
        ax3.set_xlabel("Round", fontsize=12)
        ax3.set_ylabel("Feature Importance", fontsize=12)
        ax3.set_title("Feature Importance Shift", fontsize=13)
        ax3.legend(fontsize=8, loc="best")
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        fig.savefig(str(OUTPUT_FIG), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  Saved figure to {OUTPUT_FIG}")
    except ImportError as e:
        print(f"\n  WARNING: Could not generate figure: {e}")


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print(f"Paper 3: {N_ROUNDS}-Round Adversarial Equilibrium Analysis")
    print("=" * 80)

    # Load data
    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    print("\nLoading real data from all 16 projects ...")
    real_pool = load_all_real_data(ai_calibration, rng)
    nonsybil_df = real_pool[real_pool["is_sybil"] == 0].copy()
    print(f"  Total real: {len(real_pool)} ({int(real_pool['is_sybil'].sum())} sybils)")
    print(f"  Non-sybil pool: {len(nonsybil_df)}")

    # Subsample real for training
    n_real_sample = min(N_REAL_SAMPLE, len(real_pool))
    real_sample = real_pool.sample(n=n_real_sample, random_state=42)
    X_real = real_sample[AI_FEATURE_NAMES].fillna(0).values

    # Storage for all rounds
    rounds_data = []
    all_sybil_X = []  # Cumulative sybil feature matrices
    prev_clf = None
    prev_top3 = None

    for rnd in range(N_ROUNDS):
        print(f"\n{'=' * 60}")
        print(f"ROUND {rnd}")
        print(f"{'=' * 60}")

        rng_round = np.random.RandomState(100 + rnd * 100)

        # Generate sybils
        if rnd == 0:
            # Round 0: default advanced sybils
            sybils = generate_ai_sybils_calibrated(
                N_SYBILS_PER_ROUND, ai_calibration, nonsybil_df, rng_round, "advanced",
            )
            sybils["is_llm_sybil"] = 1
            human_blend_used = 0.45  # approximate
        else:
            human_blend = HUMAN_BLEND_SCHEDULE[rnd]
            sybils = generate_round_sybils(
                n=N_SYBILS_PER_ROUND,
                ai_calibration=ai_calibration,
                nonsybil_df=nonsybil_df,
                rng=rng_round,
                human_blend=human_blend,
                top_features=prev_top3,
            )
            sybils["is_llm_sybil"] = 1
            human_blend_used = human_blend

        X_sybil = sybils[AI_FEATURE_NAMES].fillna(0).values
        all_sybil_X.append(X_sybil)
        print(f"  Generated {len(sybils)} sybils (human_blend={human_blend_used})")

        # Test previous detector on new sybils
        evasion_rate = None
        prev_det_rate = None
        if prev_clf is not None:
            prev_det_rate = detection_rate(prev_clf, X_sybil)
            evasion_rate = 1.0 - prev_det_rate
            print(f"  Previous detector detection rate: {prev_det_rate:.4f}")
            print(f"  Evasion rate: {evasion_rate:.4f}")

        # Build cumulative training data
        X_all_sybils = np.vstack(all_sybil_X)
        X_combined = np.vstack([X_real, X_all_sybils])
        y_combined = np.concatenate([
            np.zeros(len(X_real)),
            np.ones(len(X_all_sybils)),
        ])

        # CV evaluate
        results = cv_evaluate(X_combined, y_combined)
        print(f"  Retrained detector AUC: {results['mean_auc']} +/- {results['std']}")
        print(f"  Top-3 features: {results['top3_features']}")

        # Train full detector for next round
        prev_clf = train_detector(X_combined, y_combined)
        prev_top3 = results["top3_features"]

        # Store round data
        round_info = {
            "round": rnd,
            "human_blend": human_blend_used,
            "n_sybils_this_round": N_SYBILS_PER_ROUND,
            "n_sybils_cumulative": len(X_all_sybils),
            "n_real": len(X_real),
            "detector_auc": results["mean_auc"],
            "detector_auc_std": results["std"],
            "evasion_rate": round(evasion_rate, 4) if evasion_rate is not None else None,
            "detection_rate_prev": round(prev_det_rate, 4) if prev_det_rate is not None else None,
            "top3_features": results["top3_features"],
            "feature_importances": results["feature_importances"],
        }
        rounds_data.append(round_info)

    # --------------------------------------------------------
    # Convergence Analysis
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"{N_ROUNDS}-ROUND ADVERSARIAL EQUILIBRIUM ANALYSIS")
    print("=" * 80)

    aucs = [r["detector_auc"] for r in rounds_data]
    evasions = [r["evasion_rate"] for r in rounds_data]

    print(f"\n{'Round':<8} {'AUC':>8} {'ΔAUC':>8} {'Evasion':>10} {'Human Blend':>12} {'Top Feature':>30}")
    print("-" * 80)
    for i, r in enumerate(rounds_data):
        delta = f"{aucs[i] - aucs[i-1]:+.4f}" if i > 0 else "---"
        evasion = f"{r['evasion_rate']:.4f}" if r['evasion_rate'] is not None else "---"
        top_feat = list(r['top3_features'].keys())[0] if r['top3_features'] else "---"
        print(f"{i:<8} {r['detector_auc']:>8.4f} {delta:>8} {evasion:>10} {r['human_blend']:>12.2f} {top_feat:>30}")

    # Convergence check: AUC deltas in last 3 rounds
    if len(aucs) >= 4:
        last_deltas = [abs(aucs[i] - aucs[i-1]) for i in range(len(aucs)-3, len(aucs))]
        mean_delta = np.mean(last_deltas)
        print(f"\nConvergence check (last 3 rounds):")
        print(f"  Mean |ΔAUC|: {mean_delta:.4f}")
        if mean_delta < 0.01:
            print(f"  → CONVERGED: AUC changes < 0.01 threshold")
            equilibrium_auc = np.mean(aucs[-3:])
            print(f"  → Equilibrium AUC: {equilibrium_auc:.4f}")
        else:
            print(f"  → NOT YET CONVERGED (threshold: 0.01)")

    # Feature importance stability
    print(f"\nFeature rank stability (top feature per round):")
    top1_features = [list(r['top3_features'].keys())[0] for r in rounds_data]
    from collections import Counter
    stability = Counter(top1_features)
    for feat, count in stability.most_common():
        print(f"  {feat}: top-1 in {count}/{N_ROUNDS} rounds ({count/N_ROUNDS:.0%})")

    elapsed = round(time.time() - t0, 1)
    print(f"\nTotal elapsed: {elapsed}s")

    # --------------------------------------------------------
    # Generate figure
    # --------------------------------------------------------
    generate_convergence_figure(rounds_data)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    # Compute summary statistics
    auc_deltas = [aucs[i] - aucs[i-1] for i in range(1, len(aucs))]

    output = {
        "config": {
            "n_rounds": N_ROUNDS,
            "n_sybils_per_round": N_SYBILS_PER_ROUND,
            "n_real_sample": N_REAL_SAMPLE,
            "n_cv_folds": N_CV_FOLDS,
            "human_blend_schedule": [
                r["human_blend"] for r in rounds_data
            ],
            "gbm_params": GBM_PARAMS,
        },
        "rounds": rounds_data,
        "summary": {
            "initial_auc": aucs[0],
            "final_auc": aucs[-1],
            "total_auc_drop": round(aucs[0] - aucs[-1], 4),
            "auc_deltas": [round(d, 4) for d in auc_deltas],
            "mean_delta_last3": round(float(np.mean([abs(d) for d in auc_deltas[-3:]])), 4),
            "converged": float(np.mean([abs(d) for d in auc_deltas[-3:]])) < 0.01,
            "equilibrium_auc": round(float(np.mean(aucs[-3:])), 4) if float(np.mean([abs(d) for d in auc_deltas[-3:]])) < 0.01 else None,
            "max_evasion_rate": max(e for e in evasions[1:] if e is not None),
            "final_evasion_rate": evasions[-1],
            "top1_feature_stability": dict(stability),
        },
        "elapsed_seconds": elapsed,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
