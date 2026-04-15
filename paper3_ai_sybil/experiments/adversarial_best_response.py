"""
Paper 3: Best-Response Adversarial Equilibrium Analysis
========================================================
CCS/S&P Reviewer Fix: "The attacker follows a predetermined blend
schedule, not a best-response strategy.  This is not a Stackelberg
equilibrium."

This script replaces the fixed blend schedule with proper best-response
dynamics:

  Each round:
    1. Attacker OPTIMIZES the blend ratio via grid search over
       [0.30, 0.35, ..., 0.99] by picking the ratio that MAXIMIZES
       evasion rate against the current detector.
    2. Defender retrains on cumulative data (Stackelberg leader).
    3. This is a proper Stackelberg leader-follower game:
       Defender (leader) commits to a detection model, then the
       Attacker (follower) best-responds by choosing the blend
       that maximizes evasion.

Outputs:
  - adversarial_best_response_results.json
  - ../figures/fig_best_response_equilibrium.pdf

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/adversarial_best_response.py
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

OUTPUT_JSON = SCRIPT_DIR / "adversarial_best_response_results.json"
OUTPUT_FIG = FIGURES_DIR / "fig_best_response_equilibrium.pdf"

N_ROUNDS = 10
N_SYBILS_PER_ROUND = 3000
N_REAL_SAMPLE = 15000
N_CV_FOLDS = 5
N_PROBE_SYBILS = 500  # smaller batch for blend ratio search

# Blend grid: attacker searches over these values each round
BLEND_GRID = np.arange(0.30, 1.00, 0.05).round(2).tolist()
# [0.30, 0.35, 0.40, ..., 0.95]

# Stability test: 3 different initial conditions
INITIAL_SEEDS = [42, 137, 314]


# ============================================================
# HELPER FUNCTIONS (from adversarial_equilibrium_10rounds.py)
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
# BEST-RESPONSE: ATTACKER OPTIMIZES BLEND RATIO
# ============================================================

def attacker_best_response(
    detector: GradientBoostingClassifier,
    ai_calibration: dict,
    nonsybil_df: pd.DataFrame,
    rng_seed: int,
    top_features: dict = None,
) -> tuple:
    """Attacker grid-searches over blend ratios to maximize evasion.

    Returns (best_blend, best_evasion_rate, all_evasion_rates).
    """
    best_blend = None
    best_evasion = -1.0
    evasion_by_blend = {}

    for blend in BLEND_GRID:
        rng = np.random.RandomState(rng_seed)
        probe_sybils = generate_round_sybils(
            n=N_PROBE_SYBILS,
            ai_calibration=ai_calibration,
            nonsybil_df=nonsybil_df,
            rng=rng,
            human_blend=blend,
            top_features=top_features,
        )
        X_probe = probe_sybils[AI_FEATURE_NAMES].fillna(0).values
        det_rate = detection_rate(detector, X_probe)
        evasion = 1.0 - det_rate
        evasion_by_blend[blend] = round(evasion, 4)

        if evasion > best_evasion:
            best_evasion = evasion
            best_blend = blend

    return best_blend, best_evasion, evasion_by_blend


# ============================================================
# SINGLE RUN: BEST-RESPONSE DYNAMICS
# ============================================================

def run_best_response(
    seed: int,
    ai_calibration: dict,
    real_pool: pd.DataFrame,
    nonsybil_df: pd.DataFrame,
    label: str = "",
) -> list:
    """Run N_ROUNDS of best-response adversarial training.

    Returns list of per-round dicts.
    """
    rng = np.random.RandomState(seed)

    # Subsample real for training
    n_real_sample = min(N_REAL_SAMPLE, len(real_pool))
    real_sample = real_pool.sample(n=n_real_sample, random_state=seed)
    X_real = real_sample[AI_FEATURE_NAMES].fillna(0).values

    rounds_data = []
    all_sybil_X = []
    prev_clf = None
    prev_top3 = None

    for rnd in range(N_ROUNDS):
        print(f"\n  [{label}] ROUND {rnd}")

        rng_round = np.random.RandomState(seed * 1000 + rnd * 100)

        # --- Attacker's best response ---
        if rnd == 0:
            # Round 0: no detector yet -> use default advanced sybils
            sybils = generate_ai_sybils_calibrated(
                N_SYBILS_PER_ROUND, ai_calibration, nonsybil_df, rng_round, "advanced",
            )
            sybils["is_llm_sybil"] = 1
            human_blend_used = 0.45
            evasion_rate = None
            prev_det_rate = None
            evasion_by_blend = None
        else:
            # Best-response: attacker optimizes blend against current detector
            best_blend, best_evasion, evasion_by_blend = attacker_best_response(
                detector=prev_clf,
                ai_calibration=ai_calibration,
                nonsybil_df=nonsybil_df,
                rng_seed=seed * 1000 + rnd * 100,
                top_features=prev_top3,
            )
            human_blend_used = best_blend
            print(f"    Attacker best-response: blend={best_blend:.2f}, "
                  f"evasion={best_evasion:.4f}")

            # Generate full batch at optimal blend
            sybils = generate_round_sybils(
                n=N_SYBILS_PER_ROUND,
                ai_calibration=ai_calibration,
                nonsybil_df=nonsybil_df,
                rng=rng_round,
                human_blend=best_blend,
                top_features=prev_top3,
            )
            sybils["is_llm_sybil"] = 1

            # Measure evasion of full batch against current detector
            X_sybil_test = sybils[AI_FEATURE_NAMES].fillna(0).values
            prev_det_rate = detection_rate(prev_clf, X_sybil_test)
            evasion_rate = 1.0 - prev_det_rate
            print(f"    Full batch evasion: {evasion_rate:.4f}")

        X_sybil = sybils[AI_FEATURE_NAMES].fillna(0).values
        all_sybil_X.append(X_sybil)
        print(f"    Generated {len(sybils)} sybils (blend={human_blend_used})")

        # --- Defender retrains on cumulative data ---
        X_all_sybils = np.vstack(all_sybil_X)
        X_combined = np.vstack([X_real, X_all_sybils])
        y_combined = np.concatenate([
            np.zeros(len(X_real)),
            np.ones(len(X_all_sybils)),
        ])

        results = cv_evaluate(X_combined, y_combined)
        print(f"    Retrained AUC: {results['mean_auc']} +/- {results['std']}")
        print(f"    Top-3: {results['top3_features']}")

        # Train full detector for next round
        prev_clf = train_detector(X_combined, y_combined)
        prev_top3 = results["top3_features"]

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
            "evasion_by_blend": evasion_by_blend,
        }
        rounds_data.append(round_info)

    return rounds_data


# ============================================================
# LOAD FIXED-SCHEDULE RESULTS FOR COMPARISON
# ============================================================

def load_fixed_schedule_results() -> list:
    """Load the existing fixed-schedule results for comparison."""
    fixed_json = SCRIPT_DIR / "adversarial_equilibrium_10rounds.json"
    if fixed_json.exists():
        with open(fixed_json) as f:
            data = json.load(f)
        return data.get("rounds", [])
    return []


# ============================================================
# FIGURE GENERATION
# ============================================================

def generate_figure(br_runs: dict, fixed_rounds: list):
    """Generate comparison figure: best-response vs fixed schedule.

    4 panels:
      (a) AUC trajectories: fixed vs best-response (3 seeds + mean)
      (b) Evasion rate trajectories: fixed vs best-response
      (c) Attacker's chosen blend ratio per round (best-response only)
      (d) Convergence: |ΔAUC| over rounds for all runs
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError as e:
        print(f"\n  WARNING: Could not generate figure: {e}")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_auc, ax_evasion, ax_blend, ax_convergence = axes.flat

    colors = ["#e41a1c", "#377eb8", "#4daf4a"]
    alpha_runs = 0.35

    # --- Compute mean best-response trajectory ---
    n_rounds = N_ROUNDS
    all_br_aucs = []
    all_br_evasions = []
    all_br_blends = []

    for seed, rounds_data in br_runs.items():
        aucs = [r["detector_auc"] for r in rounds_data]
        evasions = [r["evasion_rate"] for r in rounds_data]
        blends = [r["human_blend"] for r in rounds_data]
        all_br_aucs.append(aucs)
        all_br_evasions.append(evasions)
        all_br_blends.append(blends)

    mean_br_aucs = np.mean(all_br_aucs, axis=0)
    round_indices = list(range(n_rounds))

    # --- Fixed schedule data ---
    fixed_aucs = [r["detector_auc"] for r in fixed_rounds] if fixed_rounds else []
    fixed_evasions = [r.get("evasion_rate") for r in fixed_rounds] if fixed_rounds else []

    # ---- Panel (a): AUC trajectories ----
    # Fixed schedule
    if fixed_aucs:
        ax_auc.plot(range(len(fixed_aucs)), fixed_aucs, "k--o",
                     linewidth=2.5, markersize=7, label="Fixed schedule", zorder=10)

    # Individual BR runs
    for i, (seed, rounds_data) in enumerate(br_runs.items()):
        aucs = [r["detector_auc"] for r in rounds_data]
        ax_auc.plot(round_indices, aucs, "-", color=colors[i % len(colors)],
                     linewidth=1.2, alpha=alpha_runs, label=f"BR seed={seed}")

    # Mean BR
    ax_auc.plot(round_indices, mean_br_aucs, "r-s", linewidth=2.5,
                 markersize=8, label="BR mean", zorder=9)

    ax_auc.axhline(y=0.5, color="gray", linestyle=":", alpha=0.4)
    ax_auc.set_xlabel("Round", fontsize=12)
    ax_auc.set_ylabel("Detector AUC (5-fold CV)", fontsize=12)
    ax_auc.set_title("(a) Detector AUC: Fixed vs Best-Response", fontsize=13)
    ax_auc.set_ylim(0.45, 1.02)
    ax_auc.legend(fontsize=9, loc="lower left")
    ax_auc.grid(True, alpha=0.3)

    # ---- Panel (b): Evasion rate trajectories ----
    if fixed_evasions:
        valid_fixed = [(i, e) for i, e in enumerate(fixed_evasions) if e is not None]
        if valid_fixed:
            fx, fy = zip(*valid_fixed)
            ax_evasion.plot(fx, fy, "k--o", linewidth=2.5, markersize=7,
                             label="Fixed schedule", zorder=10)

    for i, (seed, rounds_data) in enumerate(br_runs.items()):
        evasions = [r["evasion_rate"] for r in rounds_data]
        valid_br = [(j, e) for j, e in enumerate(evasions) if e is not None]
        if valid_br:
            bx, by = zip(*valid_br)
            ax_evasion.plot(bx, by, "-", color=colors[i % len(colors)],
                             linewidth=1.2, alpha=alpha_runs, label=f"BR seed={seed}")

    # Mean BR evasion (skip round 0)
    mean_evasions = []
    for rnd in range(n_rounds):
        vals = [all_br_evasions[i][rnd] for i in range(len(all_br_evasions))
                if all_br_evasions[i][rnd] is not None]
        mean_evasions.append(np.mean(vals) if vals else None)
    valid_mean = [(i, e) for i, e in enumerate(mean_evasions) if e is not None]
    if valid_mean:
        mx, my = zip(*valid_mean)
        ax_evasion.plot(mx, my, "r-s", linewidth=2.5, markersize=8,
                         label="BR mean", zorder=9)

    ax_evasion.set_xlabel("Round", fontsize=12)
    ax_evasion.set_ylabel("Evasion Rate", fontsize=12)
    ax_evasion.set_title("(b) Attacker Evasion Rate", fontsize=13)
    ax_evasion.set_ylim(-0.05, 1.1)
    ax_evasion.legend(fontsize=9, loc="upper right")
    ax_evasion.grid(True, alpha=0.3)

    # ---- Panel (c): Attacker's chosen blend per round ----
    for i, (seed, rounds_data) in enumerate(br_runs.items()):
        blends = [r["human_blend"] for r in rounds_data]
        ax_blend.plot(round_indices, blends, "-D", color=colors[i % len(colors)],
                       linewidth=1.5, markersize=6, label=f"BR seed={seed}")

    # Fixed schedule blends for comparison
    if fixed_rounds:
        fixed_blends = [r["human_blend"] for r in fixed_rounds]
        ax_blend.plot(range(len(fixed_blends)), fixed_blends, "k--^",
                       linewidth=2, markersize=6, label="Fixed schedule")

    ax_blend.set_xlabel("Round", fontsize=12)
    ax_blend.set_ylabel("Human Blend Ratio", fontsize=12)
    ax_blend.set_title("(c) Attacker's Chosen Blend Ratio", fontsize=13)
    ax_blend.set_ylim(0.25, 1.02)
    ax_blend.legend(fontsize=9, loc="best")
    ax_blend.grid(True, alpha=0.3)

    # ---- Panel (d): Convergence (|ΔAUC|) ----
    for i, (seed, rounds_data) in enumerate(br_runs.items()):
        aucs = [r["detector_auc"] for r in rounds_data]
        deltas = [abs(aucs[j] - aucs[j-1]) for j in range(1, len(aucs))]
        ax_convergence.plot(range(1, len(aucs)), deltas, "-o",
                             color=colors[i % len(colors)], linewidth=1.5,
                             markersize=5, label=f"BR seed={seed}")

    if fixed_aucs:
        fixed_deltas = [abs(fixed_aucs[j] - fixed_aucs[j-1]) for j in range(1, len(fixed_aucs))]
        ax_convergence.plot(range(1, len(fixed_aucs)), fixed_deltas, "k--o",
                             linewidth=2, markersize=5, label="Fixed schedule")

    ax_convergence.axhline(y=0.01, color="green", linestyle="--", alpha=0.6,
                            label="Convergence threshold (0.01)")
    ax_convergence.set_xlabel("Round", fontsize=12)
    ax_convergence.set_ylabel("|ΔAUC|", fontsize=12)
    ax_convergence.set_title("(d) Convergence: |ΔAUC| per Round", fontsize=13)
    ax_convergence.set_ylim(-0.005, 0.20)
    ax_convergence.legend(fontsize=8, loc="upper right")
    ax_convergence.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(str(OUTPUT_FIG), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved figure to {OUTPUT_FIG}")


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print(f"Paper 3: Best-Response Adversarial Equilibrium ({N_ROUNDS} rounds)")
    print(f"  Blend grid: {BLEND_GRID}")
    print(f"  Stability seeds: {INITIAL_SEEDS}")
    print(f"  Probe batch size: {N_PROBE_SYBILS}")
    print("=" * 80)

    # Load data
    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    print("\nLoading real data from all 16 projects ...")
    real_pool = load_all_real_data(ai_calibration, rng)
    nonsybil_df = real_pool[real_pool["is_sybil"] == 0].copy()
    print(f"  Total real: {len(real_pool)} ({int(real_pool['is_sybil'].sum())} sybils)")
    print(f"  Non-sybil pool: {len(nonsybil_df)}")

    # --------------------------------------------------------
    # Run best-response for 3 initial conditions
    # --------------------------------------------------------
    br_runs = {}
    for seed in INITIAL_SEEDS:
        print(f"\n{'=' * 60}")
        print(f"BEST-RESPONSE RUN: seed={seed}")
        print(f"{'=' * 60}")
        rounds_data = run_best_response(
            seed=seed,
            ai_calibration=ai_calibration,
            real_pool=real_pool,
            nonsybil_df=nonsybil_df,
            label=f"seed={seed}",
        )
        br_runs[seed] = rounds_data

    # --------------------------------------------------------
    # Load fixed-schedule results for comparison
    # --------------------------------------------------------
    fixed_rounds = load_fixed_schedule_results()

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("BEST-RESPONSE EQUILIBRIUM ANALYSIS")
    print("=" * 80)

    # Per-run summary table
    for seed, rounds_data in br_runs.items():
        aucs = [r["detector_auc"] for r in rounds_data]
        blends = [r["human_blend"] for r in rounds_data]
        evasions = [r["evasion_rate"] for r in rounds_data]

        print(f"\n--- Seed {seed} ---")
        print(f"{'Round':<8} {'AUC':>8} {'ΔAUC':>8} {'Evasion':>10} "
              f"{'Blend':>8} {'Top Feature':>30}")
        print("-" * 76)
        for i, r in enumerate(rounds_data):
            delta = f"{aucs[i] - aucs[i-1]:+.4f}" if i > 0 else "---"
            evasion = f"{r['evasion_rate']:.4f}" if r['evasion_rate'] is not None else "---"
            top_feat = list(r['top3_features'].keys())[0] if r['top3_features'] else "---"
            print(f"{i:<8} {r['detector_auc']:>8.4f} {delta:>8} {evasion:>10} "
                  f"{r['human_blend']:>8.2f} {top_feat:>30}")

    # Convergence check across runs
    print("\n\nCONVERGENCE AND STABILITY ANALYSIS")
    print("=" * 60)

    equilibrium_aucs = []
    equilibrium_blends = []
    converged_seeds = []

    for seed, rounds_data in br_runs.items():
        aucs = [r["detector_auc"] for r in rounds_data]
        blends = [r["human_blend"] for r in rounds_data]

        if len(aucs) >= 4:
            last_deltas = [abs(aucs[i] - aucs[i-1]) for i in range(len(aucs)-3, len(aucs))]
            mean_delta = np.mean(last_deltas)
            eq_auc = np.mean(aucs[-3:])
            eq_blend = np.mean(blends[-3:])

            print(f"\n  Seed {seed}:")
            print(f"    Mean |ΔAUC| (last 3): {mean_delta:.4f}")
            print(f"    Equilibrium AUC: {eq_auc:.4f}")
            print(f"    Equilibrium blend: {eq_blend:.2f}")

            if mean_delta < 0.015:
                print(f"    -> CONVERGED")
                converged_seeds.append(seed)
            else:
                print(f"    -> NOT YET CONVERGED (threshold 0.015)")

            equilibrium_aucs.append(eq_auc)
            equilibrium_blends.append(eq_blend)

    # Cross-seed stability
    if len(equilibrium_aucs) >= 2:
        auc_spread = max(equilibrium_aucs) - min(equilibrium_aucs)
        blend_spread = max(equilibrium_blends) - min(equilibrium_blends)
        print(f"\n  Cross-seed stability ({len(INITIAL_SEEDS)} seeds):")
        print(f"    AUC spread: {auc_spread:.4f}")
        print(f"    Blend spread: {blend_spread:.2f}")
        print(f"    AUC mean +/- std: {np.mean(equilibrium_aucs):.4f} "
              f"+/- {np.std(equilibrium_aucs):.4f}")
        print(f"    Blend mean +/- std: {np.mean(equilibrium_blends):.2f} "
              f"+/- {np.std(equilibrium_blends):.2f}")

        if auc_spread < 0.03:
            print(f"    -> STABLE: All seeds converge to same equilibrium (AUC spread < 0.03)")
        else:
            print(f"    -> UNSTABLE: Seeds diverge (AUC spread >= 0.03)")

    # Comparison with fixed schedule
    if fixed_rounds:
        fixed_aucs = [r["detector_auc"] for r in fixed_rounds]
        br_mean_aucs = np.mean([[r["detector_auc"] for r in rd] for rd in br_runs.values()], axis=0)

        print(f"\n\nFIXED SCHEDULE vs BEST-RESPONSE COMPARISON")
        print("=" * 60)
        print(f"{'Round':<8} {'Fixed AUC':>12} {'BR Mean AUC':>14} {'Δ(BR-Fixed)':>14}")
        print("-" * 50)
        for i in range(min(len(fixed_aucs), len(br_mean_aucs))):
            delta = br_mean_aucs[i] - fixed_aucs[i]
            print(f"{i:<8} {fixed_aucs[i]:>12.4f} {br_mean_aucs[i]:>14.4f} {delta:>+14.4f}")

        print(f"\n  Fixed final AUC: {fixed_aucs[-1]:.4f}")
        print(f"  BR mean final AUC: {br_mean_aucs[-1]:.4f}")
        print(f"  Difference: {br_mean_aucs[-1] - fixed_aucs[-1]:+.4f}")

        # Key insight: does best-response produce MORE or LESS damage?
        if br_mean_aucs[-1] < fixed_aucs[-1]:
            print(f"\n  -> Best-response attacker is MORE effective "
                  f"(AUC {br_mean_aucs[-1] - fixed_aucs[-1]:+.4f} lower)")
        else:
            print(f"\n  -> Best-response attacker is LESS effective "
                  f"(fixed schedule was already near-optimal)")

    elapsed = round(time.time() - t0, 1)
    print(f"\nTotal elapsed: {elapsed}s")

    # --------------------------------------------------------
    # Generate figure
    # --------------------------------------------------------
    generate_figure(br_runs, fixed_rounds)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    # Prepare serializable output
    br_runs_serializable = {}
    for seed, rounds_data in br_runs.items():
        clean_rounds = []
        for r in rounds_data:
            cr = dict(r)
            # Convert blend dict keys to strings for JSON
            if cr.get("evasion_by_blend"):
                cr["evasion_by_blend"] = {str(k): v for k, v in cr["evasion_by_blend"].items()}
            clean_rounds.append(cr)
        br_runs_serializable[str(seed)] = clean_rounds

    # Compute summary for each seed
    seed_summaries = {}
    for seed, rounds_data in br_runs.items():
        aucs = [r["detector_auc"] for r in rounds_data]
        blends = [r["human_blend"] for r in rounds_data]
        evasions = [r["evasion_rate"] for r in rounds_data if r["evasion_rate"] is not None]
        auc_deltas = [aucs[i] - aucs[i-1] for i in range(1, len(aucs))]

        seed_summaries[str(seed)] = {
            "initial_auc": aucs[0],
            "final_auc": aucs[-1],
            "total_auc_drop": round(aucs[0] - aucs[-1], 4),
            "mean_delta_last3": round(float(np.mean([abs(d) for d in auc_deltas[-3:]])), 4),
            "converged": float(np.mean([abs(d) for d in auc_deltas[-3:]])) < 0.015,
            "equilibrium_auc": round(float(np.mean(aucs[-3:])), 4),
            "equilibrium_blend": round(float(np.mean(blends[-3:])), 2),
            "max_evasion_rate": round(max(evasions), 4) if evasions else None,
            "final_evasion_rate": rounds_data[-1]["evasion_rate"],
            "blend_trajectory": blends,
        }

    output = {
        "config": {
            "n_rounds": N_ROUNDS,
            "n_sybils_per_round": N_SYBILS_PER_ROUND,
            "n_real_sample": N_REAL_SAMPLE,
            "n_cv_folds": N_CV_FOLDS,
            "n_probe_sybils": N_PROBE_SYBILS,
            "blend_grid": BLEND_GRID,
            "initial_seeds": INITIAL_SEEDS,
            "gbm_params": GBM_PARAMS,
        },
        "best_response_runs": br_runs_serializable,
        "seed_summaries": seed_summaries,
        "stability_analysis": {
            "n_seeds": len(INITIAL_SEEDS),
            "equilibrium_aucs": [round(a, 4) for a in equilibrium_aucs],
            "equilibrium_blends": [round(b, 2) for b in equilibrium_blends],
            "auc_mean": round(float(np.mean(equilibrium_aucs)), 4) if equilibrium_aucs else None,
            "auc_std": round(float(np.std(equilibrium_aucs)), 4) if equilibrium_aucs else None,
            "blend_mean": round(float(np.mean(equilibrium_blends)), 2) if equilibrium_blends else None,
            "blend_std": round(float(np.std(equilibrium_blends)), 2) if equilibrium_blends else None,
            "auc_spread": round(float(max(equilibrium_aucs) - min(equilibrium_aucs)), 4) if len(equilibrium_aucs) >= 2 else None,
            "stable": (max(equilibrium_aucs) - min(equilibrium_aucs)) < 0.03 if len(equilibrium_aucs) >= 2 else None,
            "converged_seeds": converged_seeds,
        },
        "comparison_with_fixed": {
            "fixed_final_auc": fixed_rounds[-1]["detector_auc"] if fixed_rounds else None,
            "br_mean_final_auc": round(float(np.mean([s["final_auc"] for s in seed_summaries.values()])), 4),
            "delta": round(
                float(np.mean([s["final_auc"] for s in seed_summaries.values()]))
                - (fixed_rounds[-1]["detector_auc"] if fixed_rounds else 0), 4
            ),
        },
        "elapsed_seconds": elapsed,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
