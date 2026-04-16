"""
Paper 3: CLOSED-LOOP LLM Adversarial Training Harness
=======================================================
This is the "greatness path" extension of adversarial_best_response.py.

Instead of the parametric attacker (grid-searching a scalar human_blend
ratio), the attacker is Claude itself. Each round we:

  1. Train detector D_t on cumulative (real ∪ LLM sybils).
  2. Extract D_t's top-3 AI feature importances.
  3. Build a FEEDBACK-DRIVEN prompt that tells Claude exactly which
     features D_t is relying on and the evasion rate parametric
     sybils are currently achieving.
  4. Ask Claude for a BATCH of 100 sybil feature vectors in one call
     (a JSON array), staying below all 5 HasciDB thresholds and
     especially attacking the top-3 features.
  5. Parse + validate, save to the shared llm_sybil_cache.
  6. Measure D_t detection rate on those 100 LLM sybils -> evasion.
  7. Retrain D_{t+1} on cumulative data, repeat.

Pilot:  1 seed × 3 rounds × 100 sybils/round = 3 LLM calls (batched).
Full :  10 rounds × 3 seeds × 100 sybils/round = 30 LLM calls.

Uses the same LLMBackend (with caching) from
  paper3_ai_sybil/adversarial/llm_sybil_generator.py

Compares against parametric best-response baseline:
  adversarial_best_response_results.json -> final AUC ~0.578

Usage:
    /Users/adelinewen/NSF/.venv/bin/python \
        paper3_ai_sybil/experiments/adversarial_closed_loop.py \
        --model claude-haiku-4-5 --seeds 42 --rounds 3 --sybils 100
"""

import argparse
import builtins
import json
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Optional

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
    THRESHOLDS,
)
from paper3_ai_sybil.adversarial.llm_sybil_generator import LLMBackend

OUTPUT_JSON = SCRIPT_DIR / "closed_loop_pilot_results.json"

N_REAL_SAMPLE = 15_000
N_CV_FOLDS = 5

# AI feature target human ranges (used to validate LLM output + prompt)
AI_FEATURE_RANGES = {
    "gas_price_precision":        (0.50, 0.90),
    "hour_entropy":               (1.80, 3.20),
    "behavioral_consistency":     (0.05, 0.35),
    "action_sequence_perplexity": (1.00, 2.50),
    "error_recovery_pattern":     (0.00, 0.10),
    "response_latency_variance":  (2.50, 7.50),
    "gas_nonce_gap_regularity":   (0.70, 0.98),
    "eip1559_tip_precision":      (0.30, 0.75),
}

PARAMETRIC_BASELINE_AUC = 0.578  # from adversarial_best_response_results.json


# ============================================================
# CLOSED-LOOP PROMPT
# ============================================================

def build_closed_loop_prompt(
    round_idx: int,
    top3_features: dict,
    all_importances: dict,
    current_evasion_rate: Optional[float],
    prev_auc: Optional[float],
    n_sybils: int,
    seed: int,
) -> str:
    """Build a feedback-driven prompt that tells Claude which features
    the current detector is using, and asks for a batch of sybils.

    We frame as: published adversarial-ML research on HasciDB (a sybil
    detector we built ourselves), same setup as Madry/Goodfellow/Tramer.
    """
    top3_str = "\n".join(
        f"     {i+1}. {name:<30s} importance={imp:.4f}"
        for i, (name, imp) in enumerate(top3_features.items())
    )

    all_imp_str = "\n".join(
        f"     {name:<30s} {imp:.4f}"
        for name, imp in sorted(all_importances.items(), key=lambda x: -x[1])
    )

    feedback = ""
    if current_evasion_rate is not None:
        feedback = (
            f"\n### FEEDBACK FROM PREVIOUS ROUND\n"
            f"The parametric attacker baseline achieves "
            f"{current_evasion_rate:.1%} evasion against D_{round_idx}. "
            f"Detector AUC is {prev_auc:.4f} (random = 0.50). "
            f"Your goal is to MATCH or EXCEED this evasion rate while "
            f"staying below all 5 HasciDB thresholds.\n"
        )

    return f"""### Adversarial-robustness evaluation (GLOBECOM 2026 / NSF)

We are red-teaming our OWN sybil detector HasciDB, which we built and
maintain. You are the attacker in a closed-loop Stackelberg game,
analogous to Madry et al. (2018) PGD adversarial training. The
synthetic feature vectors you produce are consumed ONLY by our
offline research pipeline; they never touch a live airdrop. The
dataset is released to improve sybil defenses (Goodfellow 2014;
Tramer 2020 methodology).

### HasciDB thresholds (flag = wallet classified as sybil)
  BT (Burst Transactions): flagged if BT >= 5
  BW (Burst Wallets):      flagged if BW >= 10
  HF (Hop Frequency):      flagged if HF > 0.80
  RF (Repeat Funding):     flagged if RF > 0.50
  MA (Multi-Account):      flagged if MA >= 5

A wallet is labeled sybil iff ANY of the 5 triggers fire, so to
evade HasciDB you must keep all five strictly below threshold.

### 8 behavioral features that a learned detector D_t uses
{chr(10).join(f"  {name:<30s} range [{lo:.2f}, {hi:.2f}]  (human-like)" for name, (lo, hi) in AI_FEATURE_RANGES.items())}

### CURRENT DETECTOR D_{round_idx} — GradientBoosting trained on cumulative data

  Full feature importances (sums to ~1.0):
{all_imp_str}

  TOP-3 FEATURES to attack:
{top3_str}
{feedback}
### TASK

Generate EXACTLY {n_sybils} sybil wallets that evade detector D_{round_idx}.
Strategy: your top priority is to blend in on the TOP-3 features above
(make their values statistically indistinguishable from humans). The
remaining 5 features matter less — any human-plausible value works.

Sophistication: "advanced". Use diversity — do NOT output {n_sybils}
near-copies. Vary patterns across the batch.

Random seed for diversity: {seed}

### OUTPUT FORMAT

Respond with a single JSON array of length {n_sybils}, no markdown,
no preamble, no explanation, no ``` fences. Each element must be an
object with exactly these keys:

[
  {{
    "BT": <int 0-4>, "BW": <int 0-9>, "HF": <float 0-0.79>,
    "RF": <float 0-0.49>, "MA": <int 0-4>,
    "gas_price_precision": <float>, "hour_entropy": <float>,
    "behavioral_consistency": <float>, "action_sequence_perplexity": <float>,
    "error_recovery_pattern": <float>, "response_latency_variance": <float>,
    "gas_nonce_gap_regularity": <float>, "eip1559_tip_precision": <float>
  }},
  ... ({n_sybils} total)
]
"""


# ============================================================
# PARSE BATCH RESPONSE
# ============================================================

REQUIRED_FIELDS = set(AI_FEATURE_NAMES) | set(INDICATOR_COLS)


def extract_json_array(response: str) -> Optional[list]:
    """Robustly pull a JSON array out of noisy LLM output."""
    s = response.strip()
    # Strip markdown fences
    s = re.sub(r"^```(?:json)?", "", s, flags=re.IGNORECASE).strip()
    s = re.sub(r"```$", "", s).strip()

    # Direct parse
    try:
        data = json.loads(s)
        if isinstance(data, list):
            return data
        if isinstance(data, dict) and "sybils" in data:
            return data["sybils"]
    except json.JSONDecodeError:
        pass

    # Find the outermost [ ... ]
    first = s.find("[")
    last = s.rfind("]")
    if first != -1 and last != -1 and last > first:
        try:
            data = json.loads(s[first:last + 1])
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

    # Fallback: collect individual {...} objects
    objs = re.findall(r"\{[^{}]*\}", s, re.DOTALL)
    out = []
    for m in objs:
        try:
            d = json.loads(m)
            if all(k in d for k in REQUIRED_FIELDS):
                out.append(d)
        except json.JSONDecodeError:
            continue
    return out if out else None


def validate_sybil(d: dict) -> bool:
    try:
        if not (d["BT"] < 5 and d["BW"] < 10 and d["HF"] < 0.80
                and d["RF"] < 0.50 and d["MA"] < 5):
            return False
        for k in AI_FEATURE_NAMES:
            v = float(d[k])
            if not np.isfinite(v):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def parse_batch(response: str, n_expected: int) -> list:
    raw = extract_json_array(response)
    if not raw:
        return []
    out = []
    for d in raw:
        if not isinstance(d, dict):
            continue
        if all(k in d for k in REQUIRED_FIELDS) and validate_sybil(d):
            out.append(d)
    return out


# ============================================================
# LLM BATCH GENERATION WITH RETRY/TOPUP
# ============================================================

def llm_sybil_batch(
    backend: LLMBackend,
    round_idx: int,
    top3_features: dict,
    all_importances: dict,
    current_evasion_rate: Optional[float],
    prev_auc: Optional[float],
    n_sybils: int,
    seed: int,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Generate a batch of LLM sybils, with retry top-ups."""
    collected = []
    attempt = 0
    while len(collected) < n_sybils and attempt < max_retries:
        need = n_sybils - len(collected)
        prompt = build_closed_loop_prompt(
            round_idx=round_idx,
            top3_features=top3_features,
            all_importances=all_importances,
            current_evasion_rate=current_evasion_rate,
            prev_auc=prev_auc,
            n_sybils=need,
            seed=seed + attempt * 10_000,
        )
        print(f"    LLM call attempt {attempt+1}: requesting {need} sybils "
              f"(cached prompt hash will dedupe re-runs)")
        t_call = time.time()
        response = backend.call(prompt, timeout=180.0)
        dt = time.time() - t_call
        batch = parse_batch(response, need)
        print(f"      -> got {len(batch)}/{need} valid sybils in {dt:.1f}s "
              f"(resp_len={len(response)} chars)")
        collected.extend(batch)
        attempt += 1

    df = pd.DataFrame(collected[:n_sybils])
    return df


# ============================================================
# DETECTOR / EVALUATION
# ============================================================

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


# ============================================================
# PARAMETRIC EVASION PROBE (to feed into the prompt as feedback)
# ============================================================

def parametric_evasion_probe(
    clf: GradientBoostingClassifier,
    ai_calibration: dict,
    nonsybil_df: pd.DataFrame,
    seed: int,
    n_probe: int = 500,
) -> float:
    """Quick parametric probe: what evasion can the PARAMETRIC attacker get
    against clf? This number is reported to the LLM as the bar to beat."""
    rng = np.random.RandomState(seed)
    sybils = generate_ai_sybils_calibrated(
        n_probe, ai_calibration, nonsybil_df, rng, "advanced",
    )
    X = sybils[AI_FEATURE_NAMES].fillna(0).values
    return 1.0 - detection_rate(clf, X)


# ============================================================
# REAL-DATA LOADER
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


# ============================================================
# MAIN CLOSED LOOP
# ============================================================

def run_closed_loop(
    seed: int,
    n_rounds: int,
    n_sybils_per_round: int,
    ai_calibration: dict,
    real_pool: pd.DataFrame,
    nonsybil_df: pd.DataFrame,
    backend: LLMBackend,
    label: str = "",
) -> list:
    rng = np.random.RandomState(seed)

    n_real = min(N_REAL_SAMPLE, len(real_pool))
    real_sample = real_pool.sample(n=n_real, random_state=seed)
    X_real = real_sample[AI_FEATURE_NAMES].fillna(0).values

    rounds_data = []
    all_sybil_X = []
    prev_clf = None
    prev_top3 = None
    prev_all_imp = None
    prev_auc = None

    # Round 0 bootstrap — use the same parametric "advanced" sybils as the
    # best_response baseline so round-0 detector D_1 is identical, which
    # isolates the LLM vs parametric comparison from round 1 onward.
    for rnd in range(n_rounds):
        print(f"\n  [{label}] ROUND {rnd}")

        if rnd == 0:
            rng_round = np.random.RandomState(seed * 1000 + rnd * 100)
            sybils_df = generate_ai_sybils_calibrated(
                n_sybils_per_round, ai_calibration, nonsybil_df, rng_round,
                "advanced",
            )
            X_sybil = sybils_df[AI_FEATURE_NAMES].fillna(0).values
            evasion_rate = None
            parametric_probe_rate = None
            n_llm_requested = 0
            n_llm_returned = 0
            source = "parametric_bootstrap"
        else:
            # Probe parametric evasion rate against the current detector
            # (this is the bar the LLM is told to beat).
            parametric_probe_rate = parametric_evasion_probe(
                prev_clf, ai_calibration, nonsybil_df,
                seed=seed * 7919 + rnd, n_probe=500,
            )
            print(f"    Parametric probe vs D_{rnd}: "
                  f"evasion={parametric_probe_rate:.4f}")

            # Ask Claude for a batch of sybils targeted at D_rnd
            sybils_df = llm_sybil_batch(
                backend=backend,
                round_idx=rnd,
                top3_features=prev_top3,
                all_importances=prev_all_imp,
                current_evasion_rate=parametric_probe_rate,
                prev_auc=prev_auc,
                n_sybils=n_sybils_per_round,
                seed=seed * 7919 + rnd,
            )
            n_llm_requested = n_sybils_per_round
            n_llm_returned = len(sybils_df)
            source = "llm_closed_loop"

            if n_llm_returned == 0:
                print(f"    LLM returned 0 valid sybils; "
                      f"falling back to parametric for this round")
                rng_round = np.random.RandomState(seed * 1000 + rnd * 100)
                sybils_df = generate_ai_sybils_calibrated(
                    n_sybils_per_round, ai_calibration, nonsybil_df, rng_round,
                    "advanced",
                )
                source = "parametric_fallback"

            X_sybil = sybils_df[AI_FEATURE_NAMES].fillna(0).values
            # Measure evasion of these new sybils vs current detector
            evasion_rate = 1.0 - detection_rate(prev_clf, X_sybil)
            print(f"    LLM batch evasion vs D_{rnd}: {evasion_rate:.4f} "
                  f"({n_llm_returned}/{n_llm_requested} valid)")

        all_sybil_X.append(X_sybil)

        # Retrain defender on cumulative data
        X_all_sybils = np.vstack(all_sybil_X)
        X_combined = np.vstack([X_real, X_all_sybils])
        y_combined = np.concatenate([
            np.zeros(len(X_real)),
            np.ones(len(X_all_sybils)),
        ])

        results = cv_evaluate(X_combined, y_combined)
        print(f"    Retrained AUC: {results['mean_auc']} "
              f"+/- {results['std']}")
        print(f"    Top-3: {results['top3_features']}")

        prev_clf = train_detector(X_combined, y_combined)
        prev_top3 = results["top3_features"]
        prev_all_imp = results["feature_importances"]
        prev_auc = results["mean_auc"]

        rounds_data.append({
            "round": rnd,
            "source": source,
            "n_sybils_this_round": len(X_sybil),
            "n_sybils_cumulative": len(X_all_sybils),
            "n_real": len(X_real),
            "n_llm_requested": n_llm_requested,
            "n_llm_returned": n_llm_returned,
            "detector_auc": results["mean_auc"],
            "detector_auc_std": results["std"],
            "llm_evasion_rate": round(evasion_rate, 4) if evasion_rate is not None else None,
            "parametric_probe_evasion": round(parametric_probe_rate, 4) if parametric_probe_rate is not None else None,
            "top3_features": results["top3_features"],
            "feature_importances": results["feature_importances"],
        })

    return rounds_data


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude-haiku-4-5",
                    help="SDK model id (claude-haiku-4-5 for pilot, "
                         "claude-opus-4-6 for final)")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--sybils", type=int, default=100)
    ap.add_argument("--output", default=None,
                    help="Override output JSON path")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--control-only", action="store_true",
                    help="Run parametric-only control at matched "
                         "n_sybils / n_rounds for fair comparison")
    args = ap.parse_args()

    out_path = Path(args.output) if args.output else OUTPUT_JSON

    t0 = time.time()
    print("=" * 80)
    print(f"Paper 3: CLOSED-LOOP LLM adversarial training (pilot)")
    print(f"  model:          {args.model}")
    print(f"  seeds:          {args.seeds}")
    print(f"  rounds/seed:    {args.rounds}")
    print(f"  sybils/round:   {args.sybils}")
    total_calls = len(args.seeds) * max(0, args.rounds - 1)
    print(f"  total LLM call batches: {total_calls} "
          f"(one batch per round after round 0, each asks for "
          f"{args.sybils} sybils)")
    print("=" * 80)

    backend = LLMBackend(model=args.model, use_cache=not args.no_cache)
    if not backend._available:
        print("\nERROR: claude_agent_sdk not available. Aborting.")
        sys.exit(2)

    ai_calibration = load_real_ai_calibration()
    rng = np.random.RandomState(42)

    print("\nLoading real data from all 16 projects ...")
    real_pool = load_all_real_data(ai_calibration, rng)
    nonsybil_df = real_pool[real_pool["is_sybil"] == 0].copy()
    print(f"  Total real: {len(real_pool)}  "
          f"(sybils={int(real_pool['is_sybil'].sum())})")
    print(f"  Non-sybil pool: {len(nonsybil_df)}")

    runs = {}
    for seed in args.seeds:
        print(f"\n{'=' * 60}")
        print(f"CLOSED-LOOP RUN: seed={seed}")
        print(f"{'=' * 60}")
        runs[seed] = run_closed_loop(
            seed=seed,
            n_rounds=args.rounds,
            n_sybils_per_round=args.sybils,
            ai_calibration=ai_calibration,
            real_pool=real_pool,
            nonsybil_df=nonsybil_df,
            backend=backend,
            label=f"seed={seed}",
        )

    # Scale-matched parametric CONTROL: same n_sybils, same n_rounds,
    # but every round uses parametric sybils (not the LLM). This
    # isolates "LLM strategy" from "matched attacker budget".
    print(f"\n{'=' * 60}")
    print(f"SCALE-MATCHED PARAMETRIC CONTROL")
    print(f"  (same {args.rounds} rounds × {args.sybils} sybils, "
          f"no LLM — apples-to-apples with closed-loop)")
    print(f"{'=' * 60}")
    control_runs = {}
    for seed in args.seeds:
        rng = np.random.RandomState(seed)
        n_real = min(N_REAL_SAMPLE, len(real_pool))
        real_sample = real_pool.sample(n=n_real, random_state=seed)
        X_real = real_sample[AI_FEATURE_NAMES].fillna(0).values
        all_sybil_X = []
        prev_clf = None
        rounds_ctrl = []
        for rnd in range(args.rounds):
            rng_round = np.random.RandomState(seed * 1000 + rnd * 100)
            sybils_df = generate_ai_sybils_calibrated(
                args.sybils, ai_calibration, nonsybil_df, rng_round,
                "advanced",
            )
            X_sybil = sybils_df[AI_FEATURE_NAMES].fillna(0).values
            evasion = None
            if prev_clf is not None:
                evasion = 1.0 - detection_rate(prev_clf, X_sybil)
            all_sybil_X.append(X_sybil)
            X_all = np.vstack(all_sybil_X)
            X_comb = np.vstack([X_real, X_all])
            y_comb = np.concatenate([np.zeros(len(X_real)),
                                      np.ones(len(X_all))])
            res = cv_evaluate(X_comb, y_comb)
            prev_clf = train_detector(X_comb, y_comb)
            rounds_ctrl.append({
                "round": rnd,
                "detector_auc": res["mean_auc"],
                "parametric_evasion": round(evasion, 4) if evasion is not None else None,
            })
            print(f"  [ctrl seed={seed}] R{rnd}: "
                  f"AUC={res['mean_auc']}, evasion={evasion}")
        control_runs[seed] = rounds_ctrl

    # --------- Summary ---------
    print("\n" + "=" * 80)
    print("CLOSED-LOOP PILOT SUMMARY")
    print("=" * 80)

    seed_summaries = {}
    for seed, rounds_data in runs.items():
        aucs = [r["detector_auc"] for r in rounds_data]
        llm_evasions = [r["llm_evasion_rate"] for r in rounds_data
                        if r["llm_evasion_rate"] is not None]
        param_evasions = [r["parametric_probe_evasion"] for r in rounds_data
                          if r["parametric_probe_evasion"] is not None]

        print(f"\n--- Seed {seed} ---")
        print(f"{'Round':<6} {'Source':<22} {'AUC':>7} {'ΔAUC':>8} "
              f"{'LLM-evade':>10} {'Param-evade':>12} {'Top feat':>30}")
        print("-" * 100)
        for i, r in enumerate(rounds_data):
            delta = f"{aucs[i] - aucs[i-1]:+.4f}" if i > 0 else "---"
            llm_ev = (f"{r['llm_evasion_rate']:.4f}"
                      if r["llm_evasion_rate"] is not None else "---")
            p_ev = (f"{r['parametric_probe_evasion']:.4f}"
                    if r["parametric_probe_evasion"] is not None else "---")
            top_feat = list(r["top3_features"].keys())[0] if r["top3_features"] else "---"
            print(f"{i:<6} {r['source']:<22} {r['detector_auc']:>7.4f} "
                  f"{delta:>8} {llm_ev:>10} {p_ev:>12} {top_feat:>30}")

        seed_summaries[str(seed)] = {
            "final_auc": aucs[-1],
            "initial_auc": aucs[0],
            "total_auc_drop": round(aucs[0] - aucs[-1], 4),
            "mean_llm_evasion": round(float(np.mean(llm_evasions)), 4) if llm_evasions else None,
            "mean_parametric_evasion": round(float(np.mean(param_evasions)), 4) if param_evasions else None,
            "final_llm_evasion": rounds_data[-1]["llm_evasion_rate"],
        }

    # Compare to parametric baseline AND scale-matched control
    final_aucs = [s["final_auc"] for s in seed_summaries.values()]
    mean_final_auc = float(np.mean(final_aucs))

    ctrl_final_aucs = [c[-1]["detector_auc"] for c in control_runs.values()]
    ctrl_mean_final_auc = float(np.mean(ctrl_final_aucs))

    # Round-1 evasion comparison (head-to-head vs same D_1)
    r1_llm_evasions = [r[1]["llm_evasion_rate"] for r in runs.values()
                       if len(r) > 1 and r[1]["llm_evasion_rate"] is not None]
    r1_param_evasions = [r[1]["parametric_probe_evasion"]
                         for r in runs.values()
                         if len(r) > 1 and r[1]["parametric_probe_evasion"] is not None]

    print("\n" + "=" * 80)
    print("COMPARISON: CLOSED-LOOP vs PARAMETRIC")
    print("=" * 80)
    print(f"  Full-scale parametric baseline (10rd × 3000 sybils) final AUC: "
          f"{PARAMETRIC_BASELINE_AUC:.4f}")
    print(f"  Scale-matched parametric control ({args.rounds}rd × "
          f"{args.sybils} sybils) final AUC: {ctrl_mean_final_auc:.4f}")
    print(f"  Closed-loop LLM ({args.rounds}rd × {args.sybils} sybils) "
          f"final AUC:                 {mean_final_auc:.4f}")
    delta = mean_final_auc - ctrl_mean_final_auc
    print(f"  Δ vs scale-matched control:  {delta:+.4f}")
    if r1_llm_evasions and r1_param_evasions:
        print(f"\n  ROUND-1 EVASION (both attackers face same D_1):")
        print(f"    LLM closed-loop mean evasion:   "
              f"{float(np.mean(r1_llm_evasions)):.4f}")
        print(f"    Parametric probe mean evasion:  "
              f"{float(np.mean(r1_param_evasions)):.4f}")
        r1_delta = float(np.mean(r1_llm_evasions)) - float(np.mean(r1_param_evasions))
        print(f"    Δ (LLM - parametric) evasion:   {r1_delta:+.4f}")

    # Primary signal is round-1 evasion head-to-head (both attackers
    # face the SAME D_1 trained on the same round-0 bootstrap), which
    # is independent of scale artefacts.
    r1_delta = None
    if r1_llm_evasions and r1_param_evasions:
        r1_delta = float(np.mean(r1_llm_evasions)) - float(np.mean(r1_param_evasions))

    if r1_delta is not None and r1_delta > 0.05:
        verdict = (
            "CLOSED-LOOP IS STRICTLY HARDER. Round-1 head-to-head "
            f"evasion: LLM={float(np.mean(r1_llm_evasions)):.3f} vs "
            f"parametric={float(np.mean(r1_param_evasions)):.3f} "
            f"(Δ=+{r1_delta:.3f}). Claude found evasion strategies "
            "OUTSIDE the scalar-blend family; the parametric attacker "
            "is NOT a tight upper bound on attacker capability."
        )
    elif r1_delta is not None and r1_delta < -0.05:
        verdict = (
            "CLOSED-LOOP IS EASIER than parametric. Round-1 head-to-head "
            f"evasion: LLM={float(np.mean(r1_llm_evasions)):.3f} vs "
            f"parametric={float(np.mean(r1_param_evasions)):.3f} "
            f"(Δ={r1_delta:+.3f}). The scalar-blend attacker dominates "
            "an LLM red team at this scale — parametric AUC is a valid "
            "(or even pessimistic) attacker-capability bound."
        )
    else:
        verdict = (
            "CLOSED-LOOP AND PARAMETRIC CONVERGE in head-to-head "
            f"round-1 evasion (|Δ|={abs(r1_delta) if r1_delta else 0:.3f} < 0.05). "
            "This is a robustness result: the scalar blend-ratio "
            "approximation tracks an LLM attacker and the published "
            "parametric AUC is a valid attacker-capability bound."
        )
    print(f"\n  VERDICT: {verdict}")

    elapsed = round(time.time() - t0, 1)
    print(f"\n  Total elapsed: {elapsed}s")

    # Budget estimate for full run
    pilot_calls = total_calls
    full_calls = 10 * 3  # 10 rounds (after R0) x 3 seeds = 30 batch calls
    print(f"\n  PILOT batch calls used: {pilot_calls}")
    print(f"  FULL RUN batch calls:  {full_calls}  "
          f"(10 rounds × 3 seeds × 100 sybils/round = 3000 LLM sybils)")

    # --------- Serialise ---------
    runs_serial = {str(s): rd for s, rd in runs.items()}
    output = {
        "config": {
            "model": args.model,
            "seeds": args.seeds,
            "rounds": args.rounds,
            "sybils_per_round": args.sybils,
            "n_real_sample": N_REAL_SAMPLE,
            "n_cv_folds": N_CV_FOLDS,
            "gbm_params": GBM_PARAMS,
            "hascidb_thresholds": THRESHOLDS,
            "ai_feature_names": AI_FEATURE_NAMES,
        },
        "parametric_baseline": {
            "source": "adversarial_best_response_results.json",
            "br_mean_final_auc": PARAMETRIC_BASELINE_AUC,
        },
        "closed_loop_runs": runs_serial,
        "control_runs": {str(s): r for s, r in control_runs.items()},
        "seed_summaries": seed_summaries,
        "comparison": {
            "closed_loop_mean_final_auc": round(mean_final_auc, 4),
            "scale_matched_control_final_auc": round(ctrl_mean_final_auc, 4),
            "parametric_baseline_final_auc": PARAMETRIC_BASELINE_AUC,
            "delta_vs_scale_matched": round(delta, 4),
            "round1_llm_evasion_mean": round(float(np.mean(r1_llm_evasions)), 4) if r1_llm_evasions else None,
            "round1_parametric_evasion_mean": round(float(np.mean(r1_param_evasions)), 4) if r1_param_evasions else None,
            "round1_evasion_delta": round(
                float(np.mean(r1_llm_evasions)) - float(np.mean(r1_param_evasions)), 4
            ) if r1_llm_evasions and r1_param_evasions else None,
            "verdict": verdict,
        },
        "elapsed_seconds": elapsed,
    }
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
