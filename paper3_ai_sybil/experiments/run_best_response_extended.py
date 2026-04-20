"""
Paper 3: Extended Best-Response Convergence Experiment
======================================================
Extends adversarial_best_response.py to 7 seeds x 15 rounds.

Original: 3 seeds [42, 137, 314] x 10 rounds
Extended: 7 seeds [42, 137, 314, 0, 7, 99, 256] x 15 rounds

Purpose: Reviewer request -- confirm the equilibrium plateau holds
beyond round 10, and improve statistical power with more seeds.

Output:
  adversarial_best_response_extended_results.json

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/run_best_response_extended.py
"""

import builtins
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
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
    THRESHOLDS,
)

OUTPUT_JSON = SCRIPT_DIR / "adversarial_best_response_extended_results.json"

# ============================================================
# EXTENDED CONFIG
# ============================================================

N_ROUNDS = 15  # Extended from 10 to 15
N_SYBILS_PER_ROUND = 3000
N_REAL_SAMPLE = 15000
N_CV_FOLDS = 5
N_PROBE_SYBILS = 500

BLEND_GRID = np.arange(0.30, 1.00, 0.05).round(2).tolist()

# 7 seeds total: original 3 + 4 new
ALL_SEEDS = [42, 137, 314, 0, 7, 99, 256]

# ============================================================
# HELPER FUNCTIONS (from adversarial_best_response.py)
# ============================================================

def load_all_real_data(ai_calibration: dict, rng: np.random.RandomState) -> pd.DataFrame:
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
            feat_sig = sig * 0.3

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


def attacker_best_response(
    detector: GradientBoostingClassifier,
    ai_calibration: dict,
    nonsybil_df: pd.DataFrame,
    rng_seed: int,
    top_features: dict = None,
) -> tuple:
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
# SINGLE RUN: BEST-RESPONSE DYNAMICS (15 ROUNDS)
# ============================================================

def run_best_response(
    seed: int,
    ai_calibration: dict,
    real_pool: pd.DataFrame,
    nonsybil_df: pd.DataFrame,
    n_rounds: int = N_ROUNDS,
    label: str = "",
) -> list:
    rng = np.random.RandomState(seed)

    n_real_sample = min(N_REAL_SAMPLE, len(real_pool))
    real_sample = real_pool.sample(n=n_real_sample, random_state=seed)
    X_real = real_sample[AI_FEATURE_NAMES].fillna(0).values

    rounds_data = []
    all_sybil_X = []
    prev_clf = None
    prev_top3 = None

    for rnd in range(n_rounds):
        print(f"  [{label}] Round {rnd}/{n_rounds-1}")

        rng_round = np.random.RandomState(seed * 1000 + rnd * 100)

        if rnd == 0:
            sybils = generate_ai_sybils_calibrated(
                N_SYBILS_PER_ROUND, ai_calibration, nonsybil_df, rng_round, "advanced",
            )
            sybils["is_llm_sybil"] = 1
            human_blend_used = 0.45
            evasion_rate = None
            prev_det_rate = None
            evasion_by_blend = None
        else:
            best_blend, best_evasion, evasion_by_blend = attacker_best_response(
                detector=prev_clf,
                ai_calibration=ai_calibration,
                nonsybil_df=nonsybil_df,
                rng_seed=seed * 1000 + rnd * 100,
                top_features=prev_top3,
            )
            human_blend_used = best_blend

            sybils = generate_round_sybils(
                n=N_SYBILS_PER_ROUND,
                ai_calibration=ai_calibration,
                nonsybil_df=nonsybil_df,
                rng=rng_round,
                human_blend=best_blend,
                top_features=prev_top3,
            )
            sybils["is_llm_sybil"] = 1

            X_sybil_test = sybils[AI_FEATURE_NAMES].fillna(0).values
            prev_det_rate = detection_rate(prev_clf, X_sybil_test)
            evasion_rate = 1.0 - prev_det_rate

        X_sybil = sybils[AI_FEATURE_NAMES].fillna(0).values
        all_sybil_X.append(X_sybil)

        # Defender retrains on cumulative data
        X_all_sybils = np.vstack(all_sybil_X)
        X_combined = np.vstack([X_real, X_all_sybils])
        y_combined = np.concatenate([
            np.zeros(len(X_real)),
            np.ones(len(X_all_sybils)),
        ])

        results = cv_evaluate(X_combined, y_combined)

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
            "evasion_by_blend": evasion_by_blend,
        }
        rounds_data.append(round_info)

        print(f"    AUC={results['mean_auc']:.4f} blend={human_blend_used:.2f}"
              f" evasion={evasion_rate if evasion_rate is not None else 'N/A'}")

    return rounds_data


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print(f"Paper 3: Extended Best-Response Convergence ({N_ROUNDS} rounds, {len(ALL_SEEDS)} seeds)")
    print(f"  Seeds: {ALL_SEEDS}")
    print(f"  Blend grid: {BLEND_GRID}")
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
    # Run best-response for all 7 seeds
    # --------------------------------------------------------
    br_runs = {}
    for seed in ALL_SEEDS:
        print(f"\n{'=' * 60}")
        print(f"BEST-RESPONSE RUN: seed={seed} ({N_ROUNDS} rounds)")
        print(f"{'=' * 60}")
        rounds_data = run_best_response(
            seed=seed,
            ai_calibration=ai_calibration,
            real_pool=real_pool,
            nonsybil_df=nonsybil_df,
            n_rounds=N_ROUNDS,
            label=f"seed={seed}",
        )
        br_runs[seed] = rounds_data

    # --------------------------------------------------------
    # Analysis
    # --------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXTENDED CONVERGENCE ANALYSIS")
    print("=" * 80)

    # Per-seed AUC at round 10 and round 15
    seed_metrics = {}
    for seed, rounds_data in br_runs.items():
        aucs = [r["detector_auc"] for r in rounds_data]
        blends = [r["human_blend"] for r in rounds_data]
        evasions = [r["evasion_rate"] for r in rounds_data if r["evasion_rate"] is not None]

        auc_r10 = aucs[9] if len(aucs) >= 10 else aucs[-1]  # round index 9 = round 10
        auc_r15 = aucs[14] if len(aucs) >= 15 else aucs[-1]  # round index 14 = round 15
        delta_10_to_15 = auc_r15 - auc_r10

        seed_metrics[seed] = {
            "auc_round_10": auc_r10,
            "auc_round_15": auc_r15,
            "delta_round_10_to_15": round(delta_10_to_15, 4),
            "abs_delta_10_to_15": round(abs(delta_10_to_15), 4),
            "equilibrium_auc_last3": round(float(np.mean(aucs[-3:])), 4),
            "equilibrium_blend_last3": round(float(np.mean(blends[-3:])), 2),
            "final_evasion": rounds_data[-1]["evasion_rate"],
            "converged": abs(delta_10_to_15) < 0.005,
        }

        print(f"\n  Seed {seed}:")
        print(f"    AUC @ round 10: {auc_r10:.4f}")
        print(f"    AUC @ round 15: {auc_r15:.4f}")
        print(f"    Delta (10->15): {delta_10_to_15:+.4f}")
        print(f"    Converged (|delta| < 0.005): {abs(delta_10_to_15) < 0.005}")

    # Overall mean and std across 7 seeds
    all_auc_r10 = [m["auc_round_10"] for m in seed_metrics.values()]
    all_auc_r15 = [m["auc_round_15"] for m in seed_metrics.values()]
    all_deltas = [m["delta_round_10_to_15"] for m in seed_metrics.values()]

    mean_auc_r10 = float(np.mean(all_auc_r10))
    std_auc_r10 = float(np.std(all_auc_r10))
    mean_auc_r15 = float(np.mean(all_auc_r15))
    std_auc_r15 = float(np.std(all_auc_r15))
    mean_delta = float(np.mean(all_deltas))
    mean_abs_delta = float(np.mean([abs(d) for d in all_deltas]))

    print(f"\n\n  AGGREGATE ({len(ALL_SEEDS)} seeds):")
    print(f"    Mean AUC @ round 10: {mean_auc_r10:.4f} +/- {std_auc_r10:.4f}")
    print(f"    Mean AUC @ round 15: {mean_auc_r15:.4f} +/- {std_auc_r15:.4f}")
    print(f"    Mean delta (10->15): {mean_delta:+.4f}")
    print(f"    Mean |delta| (10->15): {mean_abs_delta:.4f}")
    print(f"    Convergence confirmed (mean |delta| < 0.005): {mean_abs_delta < 0.005}")

    # 95% CI for floor estimate using t-distribution
    n_seeds = len(ALL_SEEDS)
    se_r15 = std_auc_r15 / np.sqrt(n_seeds)
    t_crit = stats.t.ppf(0.975, df=n_seeds - 1)
    ci_lower = mean_auc_r15 - t_crit * se_r15
    ci_upper = mean_auc_r15 + t_crit * se_r15

    print(f"\n  FLOOR ESTIMATE (95% CI, t-distribution, df={n_seeds-1}):")
    print(f"    Point estimate: {mean_auc_r15:.4f}")
    print(f"    95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    SE: {se_r15:.4f}")

    # Check per-seed convergence
    n_converged = sum(1 for m in seed_metrics.values() if m["converged"])
    print(f"\n  Seeds converged (|delta_10_to_15| < 0.005): {n_converged}/{n_seeds}")

    elapsed = round(time.time() - t0, 1)
    print(f"\nTotal elapsed: {elapsed}s")

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------
    br_runs_serializable = {}
    for seed, rounds_data in br_runs.items():
        clean_rounds = []
        for r in rounds_data:
            cr = dict(r)
            if cr.get("evasion_by_blend"):
                cr["evasion_by_blend"] = {str(k): v for k, v in cr["evasion_by_blend"].items()}
            clean_rounds.append(cr)
        br_runs_serializable[str(seed)] = clean_rounds

    output = {
        "config": {
            "n_rounds": N_ROUNDS,
            "n_sybils_per_round": N_SYBILS_PER_ROUND,
            "n_real_sample": N_REAL_SAMPLE,
            "n_cv_folds": N_CV_FOLDS,
            "n_probe_sybils": N_PROBE_SYBILS,
            "blend_grid": BLEND_GRID,
            "all_seeds": ALL_SEEDS,
            "gbm_params": GBM_PARAMS,
        },
        "per_seed_metrics": {str(k): v for k, v in seed_metrics.items()},
        "aggregate": {
            "n_seeds": n_seeds,
            "mean_auc_round_10": round(mean_auc_r10, 4),
            "std_auc_round_10": round(std_auc_r10, 4),
            "mean_auc_round_15": round(mean_auc_r15, 4),
            "std_auc_round_15": round(std_auc_r15, 4),
            "mean_delta_10_to_15": round(mean_delta, 4),
            "mean_abs_delta_10_to_15": round(mean_abs_delta, 4),
            "convergence_confirmed": mean_abs_delta < 0.005,
            "floor_estimate": {
                "point": round(mean_auc_r15, 4),
                "ci_95_lower": round(ci_lower, 4),
                "ci_95_upper": round(ci_upper, 4),
                "se": round(se_r15, 4),
                "df": n_seeds - 1,
            },
            "seeds_converged": n_converged,
            "seeds_total": n_seeds,
        },
        "best_response_runs": br_runs_serializable,
        "elapsed_seconds": elapsed,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
