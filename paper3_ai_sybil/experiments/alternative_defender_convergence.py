"""
Paper 3: Alternative Defender Convergence — MLP & Random Forest
================================================================
Reviewer robustness check: The best-response equilibrium (AUC ~0.58)
was established with a GBM defender. Does a different defender
architecture push the detection floor higher?

Defenders tested:
  1. MLP: 2 hidden layers (64, 32), ReLU, Adam, max_iter=500
  2. Random Forest: 500 trees, unlimited depth

Same protocol as adversarial_best_response.py:
  - 10 rounds of Stackelberg best-response dynamics
  - 3 seeds (42, 137, 314)
  - Same blend grid [0.30, 0.35, ..., 0.95]
  - Same sybil generation (parametric blend)

Outputs:
  - alternative_defender_results.json

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/alternative_defender_convergence.py
"""

import builtins
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

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

OUTPUT_JSON = SCRIPT_DIR / "alternative_defender_results.json"

N_ROUNDS = 10
N_SYBILS_PER_ROUND = 3000
N_REAL_SAMPLE = 15000
N_CV_FOLDS = 5
N_PROBE_SYBILS = 500

BLEND_GRID = np.arange(0.30, 1.00, 0.05).round(2).tolist()
INITIAL_SEEDS = [42, 137, 314]

# ============================================================
# DEFENDER CONFIGURATIONS
# ============================================================

DEFENDERS = {
    "mlp": {
        "label": "MLP (64,32) ReLU Adam",
        "needs_scaling": True,
    },
    "random_forest": {
        "label": "Random Forest (500 trees, unlimited depth)",
        "needs_scaling": False,
    },
    "gbm": {
        "label": "GBM (baseline, for comparison)",
        "needs_scaling": False,
    },
}


def make_defender(name: str, random_state: int = 42):
    """Instantiate a fresh defender model."""
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
            random_state=random_state,
        )
    elif name == "random_forest":
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,  # unlimited depth
            random_state=random_state,
            n_jobs=-1,
        )
    elif name == "gbm":
        return GradientBoostingClassifier(**GBM_PARAMS)
    else:
        raise ValueError(f"Unknown defender: {name}")


# ============================================================
# HELPER FUNCTIONS (from adversarial_best_response.py)
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


# ============================================================
# GENERIC DEFENDER: TRAIN, PREDICT, EVALUATE
# ============================================================

class DefenderWrapper:
    """Wraps a sklearn classifier with optional feature scaling.

    MLP needs StandardScaler; tree-based methods do not.
    """

    def __init__(self, defender_name: str, random_state: int = 42):
        self.name = defender_name
        self.needs_scaling = DEFENDERS[defender_name]["needs_scaling"]
        self.model = make_defender(defender_name, random_state)
        self.scaler = StandardScaler() if self.needs_scaling else None

    def fit(self, X, y):
        if self.needs_scaling:
            X = self.scaler.fit_transform(X)
        self.model.fit(X, y)
        return self

    def predict_proba(self, X):
        if self.needs_scaling:
            X = self.scaler.transform(X)
        return self.model.predict_proba(X)

    def feature_importances(self, feature_names):
        """Return feature importances where available, else zeros."""
        if hasattr(self.model, "feature_importances_"):
            imp = self.model.feature_importances_
        elif hasattr(self.model, "coefs_"):
            # For MLP, use absolute mean weight from first layer as proxy
            w = np.abs(self.model.coefs_[0]).mean(axis=1)
            imp = w / (w.sum() + 1e-12)
        else:
            imp = np.zeros(len(feature_names))

        result = {feature_names[i]: round(float(imp[i]), 4)
                  for i in range(len(feature_names))}
        return dict(sorted(result.items(), key=lambda x: -x[1]))


def detection_rate(defender: DefenderWrapper, X_sybils) -> float:
    probs = defender.predict_proba(X_sybils)[:, 1]
    return float((probs >= 0.5).mean())


# ============================================================
# CV EVALUATION (generic defender)
# ============================================================

def cv_evaluate(X, y, defender_name: str, n_splits=N_CV_FOLDS) -> dict:
    """K-fold CV AUC evaluation with a specified defender architecture."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    aucs = []
    all_importances = np.zeros(X.shape[1])
    n_folds_done = 0

    for tr_idx, te_idx in skf.split(X, y):
        dw = DefenderWrapper(defender_name)
        dw.fit(X[tr_idx], y[tr_idx])
        probs = dw.predict_proba(X[te_idx])[:, 1]
        try:
            aucs.append(roc_auc_score(y[te_idx], probs))
        except ValueError:
            pass

        imp = dw.feature_importances(AI_FEATURE_NAMES)
        for i, feat in enumerate(AI_FEATURE_NAMES):
            all_importances[i] += imp.get(feat, 0.0)
        n_folds_done += 1

    if n_folds_done > 0:
        all_importances /= n_folds_done
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


# ============================================================
# ATTACKER BEST-RESPONSE (identical logic, generic defender)
# ============================================================

def attacker_best_response(
    detector: DefenderWrapper,
    ai_calibration: dict,
    nonsybil_df: pd.DataFrame,
    rng_seed: int,
    top_features: dict = None,
) -> tuple:
    """Attacker grid-searches over blend ratios to maximize evasion."""
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
# SINGLE RUN: BEST-RESPONSE DYNAMICS (generic defender)
# ============================================================

def run_best_response(
    seed: int,
    ai_calibration: dict,
    real_pool: pd.DataFrame,
    nonsybil_df: pd.DataFrame,
    defender_name: str,
    label: str = "",
) -> list:
    """Run N_ROUNDS of best-response adversarial training with specified defender."""
    rng = np.random.RandomState(seed)

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
            print(f"    Attacker best-response: blend={best_blend:.2f}, "
                  f"evasion={best_evasion:.4f}")

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

        results = cv_evaluate(X_combined, y_combined, defender_name)
        print(f"    Retrained AUC: {results['mean_auc']} +/- {results['std']}")
        print(f"    Top-3: {results['top3_features']}")

        # Train full detector for next round
        prev_clf = DefenderWrapper(defender_name, random_state=seed)
        prev_clf.fit(X_combined, y_combined)
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
# ANALYSIS
# ============================================================

def analyze_runs(defender_name: str, runs: dict) -> dict:
    """Compute convergence and stability analysis for a set of runs."""
    equilibrium_aucs = []
    equilibrium_blends = []
    converged_seeds = []
    seed_summaries = {}

    for seed_str, rounds_data in runs.items():
        seed = int(seed_str)
        aucs = [r["detector_auc"] for r in rounds_data]
        blends = [r["human_blend"] for r in rounds_data]
        evasions = [r["evasion_rate"] for r in rounds_data if r["evasion_rate"] is not None]
        auc_deltas = [aucs[i] - aucs[i - 1] for i in range(1, len(aucs))]

        eq_auc = round(float(np.mean(aucs[-3:])), 4)
        eq_blend = round(float(np.mean(blends[-3:])), 2)
        mean_delta_last3 = round(float(np.mean([abs(d) for d in auc_deltas[-3:]])), 4)
        converged = mean_delta_last3 < 0.015

        if converged:
            converged_seeds.append(seed)
        equilibrium_aucs.append(eq_auc)
        equilibrium_blends.append(eq_blend)

        seed_summaries[seed_str] = {
            "initial_auc": aucs[0],
            "final_auc": aucs[-1],
            "total_auc_drop": round(aucs[0] - aucs[-1], 4),
            "mean_delta_last3": mean_delta_last3,
            "converged": converged,
            "equilibrium_auc": eq_auc,
            "equilibrium_blend": eq_blend,
            "max_evasion_rate": round(max(evasions), 4) if evasions else None,
            "final_evasion_rate": rounds_data[-1]["evasion_rate"],
            "blend_trajectory": blends,
        }

    stability = {
        "n_seeds": len(runs),
        "equilibrium_aucs": equilibrium_aucs,
        "equilibrium_blends": equilibrium_blends,
        "auc_mean": round(float(np.mean(equilibrium_aucs)), 4) if equilibrium_aucs else None,
        "auc_std": round(float(np.std(equilibrium_aucs)), 4) if equilibrium_aucs else None,
        "blend_mean": round(float(np.mean(equilibrium_blends)), 2) if equilibrium_blends else None,
        "blend_std": round(float(np.std(equilibrium_blends)), 2) if equilibrium_blends else None,
        "converged_seeds": converged_seeds,
    }

    if len(equilibrium_aucs) >= 2:
        spread = round(float(max(equilibrium_aucs) - min(equilibrium_aucs)), 4)
        stability["auc_spread"] = spread
        stability["stable"] = spread < 0.03

    return {
        "defender": defender_name,
        "defender_label": DEFENDERS[defender_name]["label"],
        "seed_summaries": seed_summaries,
        "stability_analysis": stability,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    t0 = time.time()
    print("=" * 80)
    print("Paper 3: Alternative Defender Convergence")
    print(f"  Defenders: {[DEFENDERS[d]['label'] for d in ['mlp', 'random_forest', 'gbm']]}")
    print(f"  {N_ROUNDS} rounds, {len(INITIAL_SEEDS)} seeds, blend grid {len(BLEND_GRID)} points")
    print("=" * 80)

    # Load data (shared across all defenders)
    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    print("\nLoading real data from all 16 projects ...")
    real_pool = load_all_real_data(ai_calibration, rng)
    nonsybil_df = real_pool[real_pool["is_sybil"] == 0].copy()
    print(f"  Total real: {len(real_pool)} ({int(real_pool['is_sybil'].sum())} sybils)")
    print(f"  Non-sybil pool: {len(nonsybil_df)}")

    all_defender_results = {}

    for defender_name in ["mlp", "random_forest", "gbm"]:
        defender_label = DEFENDERS[defender_name]["label"]
        print(f"\n{'#' * 80}")
        print(f"# DEFENDER: {defender_label}")
        print(f"{'#' * 80}")

        runs = {}
        for seed in INITIAL_SEEDS:
            print(f"\n{'=' * 60}")
            print(f"  {defender_label} | seed={seed}")
            print(f"{'=' * 60}")
            rounds_data = run_best_response(
                seed=seed,
                ai_calibration=ai_calibration,
                real_pool=real_pool,
                nonsybil_df=nonsybil_df,
                defender_name=defender_name,
                label=f"{defender_name}/seed={seed}",
            )
            runs[str(seed)] = rounds_data

        # Clean for JSON serialization
        clean_runs = {}
        for seed_str, rounds_data in runs.items():
            clean_rounds = []
            for r in rounds_data:
                cr = dict(r)
                if cr.get("evasion_by_blend"):
                    cr["evasion_by_blend"] = {str(k): v for k, v in cr["evasion_by_blend"].items()}
                clean_rounds.append(cr)
            clean_runs[seed_str] = clean_rounds

        analysis = analyze_runs(defender_name, clean_runs)
        analysis["runs"] = clean_runs

        all_defender_results[defender_name] = analysis

        # Print summary for this defender
        stab = analysis["stability_analysis"]
        print(f"\n  {defender_label} SUMMARY:")
        print(f"    Equilibrium AUC (mean +/- std): "
              f"{stab['auc_mean']} +/- {stab['auc_std']}")
        print(f"    Equilibrium AUCs per seed: {stab['equilibrium_aucs']}")
        print(f"    Converged seeds: {stab['converged_seeds']}")
        if "auc_spread" in stab:
            print(f"    AUC spread: {stab['auc_spread']}")
            print(f"    Stable: {stab['stable']}")

    # ============================================================
    # CROSS-DEFENDER COMPARISON
    # ============================================================
    print(f"\n{'=' * 80}")
    print("CROSS-DEFENDER COMPARISON")
    print("=" * 80)

    comparison = {}
    print(f"\n  {'Defender':<40} {'Eq. AUC':>10} {'Eq. Blend':>12} {'Spread':>10} {'Converged':>10}")
    print("  " + "-" * 84)

    for dname in ["gbm", "mlp", "random_forest"]:
        if dname not in all_defender_results:
            continue
        stab = all_defender_results[dname]["stability_analysis"]
        label = DEFENDERS[dname]["label"]
        spread = stab.get("auc_spread", "N/A")
        converged = f"{len(stab['converged_seeds'])}/{stab['n_seeds']}"
        print(f"  {label:<40} {stab['auc_mean']:>10.4f} {stab['blend_mean']:>12.2f} "
              f"{spread:>10} {converged:>10}")

        comparison[dname] = {
            "equilibrium_auc_mean": stab["auc_mean"],
            "equilibrium_auc_std": stab["auc_std"],
            "equilibrium_blend_mean": stab["blend_mean"],
            "auc_spread": stab.get("auc_spread"),
            "n_converged": len(stab["converged_seeds"]),
        }

    # Key question: does any defender push the floor higher?
    gbm_auc = comparison.get("gbm", {}).get("equilibrium_auc_mean")
    for dname in ["mlp", "random_forest"]:
        if dname not in comparison:
            continue
        alt_auc = comparison[dname]["equilibrium_auc_mean"]
        delta = alt_auc - gbm_auc if gbm_auc else None
        comparison[dname]["delta_vs_gbm"] = round(delta, 4) if delta is not None else None
        if delta is not None:
            label = DEFENDERS[dname]["label"]
            if abs(delta) < 0.02:
                print(f"\n  {label}: ~same floor as GBM (delta={delta:+.4f})")
            elif delta > 0:
                print(f"\n  {label}: HIGHER floor than GBM by {delta:+.4f} "
                      f"(defender advantage)")
            else:
                print(f"\n  {label}: LOWER floor than GBM by {delta:+.4f} "
                      f"(weaker defender)")

    elapsed = round(time.time() - t0, 1)
    print(f"\nTotal elapsed: {elapsed}s")

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    output = {
        "experiment": "alternative_defender_convergence",
        "question": (
            "Does the detection floor (~0.58 AUC) hold across different "
            "defender architectures, or does a non-GBM defender push it higher?"
        ),
        "config": {
            "n_rounds": N_ROUNDS,
            "n_sybils_per_round": N_SYBILS_PER_ROUND,
            "n_real_sample": N_REAL_SAMPLE,
            "n_cv_folds": N_CV_FOLDS,
            "n_probe_sybils": N_PROBE_SYBILS,
            "blend_grid": BLEND_GRID,
            "initial_seeds": INITIAL_SEEDS,
            "defenders": {k: v["label"] for k, v in DEFENDERS.items()},
        },
        "results": all_defender_results,
        "cross_defender_comparison": comparison,
        "elapsed_seconds": elapsed,
    }

    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
