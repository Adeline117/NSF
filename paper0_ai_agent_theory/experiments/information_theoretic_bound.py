#!/usr/bin/env python
"""
Paper 0: Information-Theoretic Bound Analysis (Fano's Inequality)
==================================================================
Computes the empirical mutual information I(X; Y) between each feature set
and the taxonomy labels, then applies Fano's inequality to derive a lower
bound on the Bayes classification error.

Feature sets:
  X_23  — 23 original behavioral features (Paper 0 baseline)
  X_31  — X_23 + 8 AI-detection features (Paper 3)
  X_47  — X_31 + 16 decision-process features (error recovery / session /
           nonce / gas microstructure)

Label spaces:
  Y_8   — 8-class fine-grained taxonomy (taxonomy_index)
  Y_3   — 3-cluster K-Means labels derived from X_23 (matches
           cluster_validation.py k=3 configuration)

Fano's inequality (discrete Y):
  P_error >= ( H(Y) - I(X; Y) - 1 ) / log2(|Y|)
  ⇒  accuracy_ceiling = 1 - P_error_lower_bound

Usage:
  python information_theoretic_bound.py

Outputs:
  information_theoretic_bound_results.json
  fig_mi_vs_accuracy.png
"""

from __future__ import annotations

import json
import math
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Reuse decision-process feature extractors from the companion script.
from paper0_ai_agent_theory.experiments import decision_process_features as dpf  # noqa: E402

# ── Paths ─────────────────────────────────────────────────────────────
FEATURES_PARQUET = (
    PROJECT_ROOT / "paper1_onchain_agent_id" / "data" / "features_with_taxonomy.parquet"
)
AI_FEATURES_JSON = (
    PROJECT_ROOT / "paper3_ai_sybil" / "experiments" / "real_ai_features.json"
)
DP_CACHE_PARQUET = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "decision_process_features_cache.parquet"
)
RESULTS_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "information_theoretic_bound_results.json"
)
FIG_PATH = (
    PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
    / "fig_mi_vs_accuracy.png"
)

# ── Feature columns ───────────────────────────────────────────────────
ORIGINAL_23 = dpf.ORIGINAL_23
AI_8 = dpf.AI_8
DECISION_PROCESS_16 = dpf.DECISION_PROCESS_16
ALL_47 = ORIGINAL_23 + AI_8 + DECISION_PROCESS_16

TAXONOMY_NAMES = dpf.TAXONOMY_NAMES

SEED = 42
MI_NEIGHBORS = 3  # sklearn mutual_info_classif uses k-NN Kraskov-like estimator


# ══════════════════════════════════════════════════════════════════════
# Data loading
# ══════════════════════════════════════════════════════════════════════

def load_decision_process_cache(addresses):
    """Load 16 DP features from cache, extracting on demand for missing rows."""
    if DP_CACHE_PARQUET.exists():
        cached = pd.read_parquet(DP_CACHE_PARQUET)
        print(f"  Loaded DP cache: {len(cached)} rows")
    else:
        cached = pd.DataFrame(columns=DECISION_PROCESS_16)

    missing = [a for a in addresses if a not in cached.index]
    if missing:
        print(f"  Extracting DP features for {len(missing)} addresses "
              f"(cache miss)...")
        rows = {}
        for i, addr in enumerate(missing):
            if (i + 1) % 250 == 0 or i == 0:
                print(f"    {i + 1}/{len(missing)}")
            feats = dpf.extract_all_features(addr)
            if feats is None:
                rows[addr] = {c: np.nan for c in DECISION_PROCESS_16}
            else:
                rows[addr] = feats
        new = pd.DataFrame.from_dict(rows, orient="index")
        new.index.name = "address"
        cached = pd.concat([cached, new])
        cached.to_parquet(DP_CACHE_PARQUET)
        print(f"  Wrote DP cache: {DP_CACHE_PARQUET} ({len(cached)} rows)")

    return cached.loc[[a for a in addresses if a in cached.index]]


def load_all_features():
    """Return DataFrame with all 47 features + taxonomy_index, indexed by address."""
    df = pd.read_parquet(FEATURES_PARQUET)
    df = df[df["label"] == 1].copy()
    addresses = list(df.index)
    print(f"Agents (label==1): {len(addresses)}")

    # AI 8 features
    with open(AI_FEATURES_JSON) as f:
        ai_raw = json.load(f)
    ai_rows = []
    for addr, feats in ai_raw["per_address"].items():
        row = {"address": addr}
        for col in AI_8:
            row[col] = feats.get(col, np.nan)
        ai_rows.append(row)
    df_ai = pd.DataFrame(ai_rows).set_index("address")
    df = df.join(df_ai[AI_8], how="left")
    print(f"  After AI join: AI NaN-any rate = "
          f"{df[AI_8].isna().any(axis=1).mean():.3f}")

    # Decision-process 16 features
    df_dp = load_decision_process_cache(addresses)
    df = df.join(df_dp[DECISION_PROCESS_16], how="left")
    print(f"  After DP join: DP NaN-any rate = "
          f"{df[DECISION_PROCESS_16].isna().any(axis=1).mean():.3f}")

    return df


def impute_and_clip(X):
    """Median-impute NaNs and clip to [1st, 99th] percentile (same as classifier)."""
    X = X.copy().astype(float)
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


# ══════════════════════════════════════════════════════════════════════
# Information theory
# ══════════════════════════════════════════════════════════════════════

def entropy_discrete(y, base=2):
    """Empirical Shannon entropy (bits if base=2, nats if base=e)."""
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    if base == 2:
        return float(-np.sum(p * np.log2(p + 1e-30)))
    return float(-np.sum(p * np.log(p + 1e-30)))


def kraskov_mi(X, y, n_neighbors=MI_NEIGHBORS, seed=SEED):
    """
    Sum of per-feature mutual information between each X column and y,
    using sklearn's Kraskov (k-NN) estimator. Returns total in *nats*
    (sklearn convention). Use the joint version below for the tighter
    multivariate estimate.
    """
    mi_per_feat = mutual_info_classif(
        X, y, n_neighbors=n_neighbors, random_state=seed
    )
    return float(np.sum(mi_per_feat)), mi_per_feat.tolist()


def joint_mi_knn(X, y, n_neighbors=MI_NEIGHBORS, seed=SEED):
    """
    Estimate I(X; Y) treating X as a single multivariate feature.
    Uses the identity I(X;Y) = H(Y) - H(Y|X) with a k-NN regressor
    style estimate: for each sample, look at its k nearest neighbors in
    X-space (in standardized scale) and estimate local label entropy.

    This gives a tighter upper bound on MI than the sum of per-feature
    MIs (which ignores feature dependence) while remaining non-parametric.

    Returns MI in bits.
    """
    from sklearn.neighbors import NearestNeighbors

    # Standardize
    Xs = StandardScaler().fit_transform(X)
    n = len(y)
    # +1 to include the point itself
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="auto")
    nn.fit(Xs)
    _, idx = nn.kneighbors(Xs)
    # Drop self (column 0)
    idx = idx[:, 1:]

    # Estimate local conditional entropy H(Y | X ≈ x) from neighbor labels
    conditional_entropies = np.zeros(n)
    for i in range(n):
        labels_nb = y[idx[i]]
        _, counts = np.unique(labels_nb, return_counts=True)
        p = counts / counts.sum()
        conditional_entropies[i] = -np.sum(p * np.log2(p + 1e-30))
    h_cond = float(np.mean(conditional_entropies))
    h_y = entropy_discrete(y, base=2)
    return max(0.0, h_y - h_cond), h_y, h_cond


def fano_lower_bound_error(h_y_bits, mi_bits, num_classes):
    """
    Fano's inequality (binary entropy variant), solved for P_e as a lower bound:
        H(Y|X) <= H(P_e) + P_e * log2(|Y| - 1)
    Since H(P_e) <= 1 for binary entropy, the standard *loose* form is:
        P_e >= (H(Y|X) - 1) / log2(|Y|)
               = (H(Y) - I(X;Y) - 1) / log2(|Y|)
    We compute BOTH:
      (a) loose_bound:  uses H(P_e) <= 1 relaxation -> simple closed form
      (b) tight_bound:  numerically invert Fano:
              find smallest P_e in [0, 1 - 1/|Y|] s.t.
              H_b(P_e) + P_e*log2(|Y|-1) >= H(Y|X)
    Both give lower bounds on the Bayes error.  (b) is the one people
    usually quote as "the Fano bound".
    """
    h_cond = max(0.0, h_y_bits - mi_bits)
    log_k = math.log2(num_classes) if num_classes > 1 else 1.0

    # Loose bound
    loose = max(0.0, (h_cond - 1.0) / log_k)

    # Tight bound (numerically invert Fano)
    def fano_rhs(pe):
        # H_b(pe) + pe * log2(|Y|-1)
        if pe <= 0 or pe >= 1:
            hb = 0.0
        else:
            hb = -(pe * math.log2(pe) + (1 - pe) * math.log2(1 - pe))
        return hb + (pe * math.log2(num_classes - 1) if num_classes > 1 else 0.0)

    # Binary-search smallest pe in [0, (K-1)/K] with fano_rhs(pe) >= h_cond
    max_pe = (num_classes - 1) / num_classes
    lo, hi = 0.0, max_pe
    # If even at max_pe the rhs < h_cond, the bound is uninformative (trivially max_pe).
    if fano_rhs(max_pe) < h_cond:
        tight = max_pe
    else:
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if fano_rhs(mid) >= h_cond:
                hi = mid
            else:
                lo = mid
        tight = hi
    return {
        "h_y_bits": float(h_y_bits),
        "mi_bits": float(mi_bits),
        "h_cond_bits": float(h_cond),
        "num_classes": int(num_classes),
        "p_error_lower_loose": round(loose, 4),
        "p_error_lower_tight": round(tight, 4),
        "accuracy_ceiling_loose": round(1.0 - loose, 4),
        "accuracy_ceiling_tight": round(1.0 - tight, 4),
    }


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("Paper 0: Information-Theoretic Bound (Fano's Inequality)")
    print("=" * 80)

    df = load_all_features()
    y8 = df["taxonomy_index"].values.astype(int)

    # Build feature matrices
    X23 = impute_and_clip(df[ORIGINAL_23].values)
    X31 = impute_and_clip(df[ORIGINAL_23 + AI_8].values)
    X47 = impute_and_clip(df[ALL_47].values)

    print(f"\nMatrices: X23={X23.shape}, X31={X31.shape}, X47={X47.shape}")
    print(f"Labels Y_8: {len(np.unique(y8))} classes")

    # Build Y_3 via K-Means on standardized X_23 (reproduces cluster_validation)
    scaler23 = StandardScaler()
    X23s = scaler23.fit_transform(X23)
    km3 = KMeans(n_clusters=3, n_init=20, random_state=SEED)
    y3 = km3.fit_predict(X23s).astype(int)
    print(f"K-Means(k=3) on X_23:")
    u, c = np.unique(y3, return_counts=True)
    for uu, cc in zip(u, c):
        print(f"   cluster {uu}: n={cc}")

    # Entropies
    H_Y8 = entropy_discrete(y8, base=2)
    H_Y3 = entropy_discrete(y3, base=2)
    print(f"\nH(Y_8) = {H_Y8:.4f} bits  (max log2(8) = {math.log2(8):.4f})")
    print(f"H(Y_3) = {H_Y3:.4f} bits  (max log2(3) = {math.log2(3):.4f})")

    # ── Measured classifier performance (from existing experiments) ───
    # These are GBM 10-fold stratified CV accuracies reported in:
    #   multiclass_31features_results.json (X_31 vs Y_8)
    #   decision_process_results.json      (X_23, X_47 vs Y_8)
    with open(PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
              / "decision_process_results.json") as f:
        dp_res = json.load(f)
    with open(PROJECT_ROOT / "paper0_ai_agent_theory" / "experiments"
              / "multiclass_31features_results.json") as f:
        mc31 = json.load(f)

    acc_X23_Y8 = dp_res["conditions"]["A_23_baseline"]["accuracy_mean"]
    acc_X31_Y8 = mc31["models"]["GradientBoosting"]["accuracy_mean"]
    acc_X47_Y8 = dp_res["conditions"]["C_47_full"]["accuracy_mean"]
    print(f"\nMeasured GBM accuracies (10-fold CV):")
    print(f"  X23 → Y8: {acc_X23_Y8:.4f}  (error {1 - acc_X23_Y8:.4f})")
    print(f"  X31 → Y8: {acc_X31_Y8:.4f}  (error {1 - acc_X31_Y8:.4f})")
    print(f"  X47 → Y8: {acc_X47_Y8:.4f}  (error {1 - acc_X47_Y8:.4f})")

    # ── Compute MI ─────────────────────────────────────────────────────
    feature_sets = [
        ("X_23", X23, ORIGINAL_23),
        ("X_31", X31, ORIGINAL_23 + AI_8),
        ("X_47", X47, ALL_47),
    ]

    label_sets = [
        ("Y_8", y8, 8, H_Y8, {"X_23": acc_X23_Y8, "X_31": acc_X31_Y8, "X_47": acc_X47_Y8}),
        ("Y_3", y3, 3, H_Y3, {}),  # no existing classifier on Y_3
    ]

    mi_table = {}

    for Xname, X, fnames in feature_sets:
        for Yname, yv, K, H_y, _ in label_sets:
            key = f"MI({Xname};{Yname})"
            print(f"\n─── {key}  (n={len(yv)}, |X|={X.shape[1]}, "
                  f"|Y|={K}) ───")

            # (a) Sum of per-feature MI (nats -> bits)
            mi_sum_nats, mi_per_feat = kraskov_mi(X, yv)
            mi_sum_bits = mi_sum_nats / math.log(2.0)
            # Cap at H(Y) because sum-of-per-feature is an upper bound in
            # the independent-features case; the true joint MI is ≤ H(Y).
            mi_sum_capped_bits = min(mi_sum_bits, H_y)

            # (b) Joint (multivariate) k-NN-based MI estimate
            mi_joint_bits, h_y_check, h_cond_joint = joint_mi_knn(X, yv)
            mi_joint_bits = min(mi_joint_bits, H_y)

            print(f"  H(Y)                              = {H_y:.4f} bits")
            print(f"  Σ I(X_j; Y) (sklearn Kraskov)     = {mi_sum_bits:.4f} bits"
                  f"  -> capped {mi_sum_capped_bits:.4f}")
            print(f"  I(X; Y) joint k-NN (k={MI_NEIGHBORS})         = "
                  f"{mi_joint_bits:.4f} bits")

            # Use joint estimate for the Fano bound (it is the true
            # multivariate I(X;Y); the sum-of-per-feature overcounts when
            # features are correlated).
            bound_joint = fano_lower_bound_error(H_y, mi_joint_bits, K)
            bound_sum = fano_lower_bound_error(H_y, mi_sum_capped_bits, K)

            print(f"  Fano bound (joint MI):  P_e >= "
                  f"{bound_joint['p_error_lower_tight']:.4f}  "
                  f"⇒ accuracy ≤ {bound_joint['accuracy_ceiling_tight']:.4f}")
            print(f"  Fano bound (sum MI):    P_e >= "
                  f"{bound_sum['p_error_lower_tight']:.4f}  "
                  f"⇒ accuracy ≤ {bound_sum['accuracy_ceiling_tight']:.4f}")

            mi_table[key] = {
                "feature_set": Xname,
                "label_set": Yname,
                "n_samples": int(len(yv)),
                "n_features": int(X.shape[1]),
                "num_classes": int(K),
                "H_Y_bits": round(H_y, 4),
                "MI_sum_bits": round(mi_sum_bits, 4),
                "MI_sum_capped_bits": round(mi_sum_capped_bits, 4),
                "MI_joint_knn_bits": round(mi_joint_bits, 4),
                "H_Y_given_X_joint_bits": round(h_cond_joint, 4),
                "per_feature_MI_nats": {
                    fn: round(v, 4) for fn, v in zip(fnames, mi_per_feat)
                },
                "fano_joint": bound_joint,
                "fano_sum": bound_sum,
            }

    # ── Classifier-vs-Fano comparison ─────────────────────────────────
    comparison = {}
    for Xname, X, _ in feature_sets:
        acc = {"X_23": acc_X23_Y8, "X_31": acc_X31_Y8, "X_47": acc_X47_Y8}[Xname]
        entry = mi_table[f"MI({Xname};Y_8)"]
        ceil = entry["fano_joint"]["accuracy_ceiling_tight"]
        gap = ceil - acc
        comparison[f"{Xname}→Y_8"] = {
            "measured_accuracy": round(acc, 4),
            "measured_error": round(1 - acc, 4),
            "fano_accuracy_ceiling": round(ceil, 4),
            "fano_error_lower_bound": entry["fano_joint"]["p_error_lower_tight"],
            "headroom_accuracy": round(gap, 4),
            "interpretation": (
                "TIGHT (≤1pp headroom) — ceiling is information-theoretic."
                if gap <= 0.01 else
                "NEAR-TIGHT (1-5pp headroom) — small room for better classifiers."
                if gap <= 0.05 else
                "LOOSE (>5pp headroom) — classifier is sub-optimal "
                "or MI estimate is loose."
            ),
        }

    # Marginal MI increase from adding features
    delta_mi = {
        "I(X_31;Y_8) - I(X_23;Y_8)": round(
            mi_table["MI(X_31;Y_8)"]["MI_joint_knn_bits"]
            - mi_table["MI(X_23;Y_8)"]["MI_joint_knn_bits"], 4),
        "I(X_47;Y_8) - I(X_31;Y_8)": round(
            mi_table["MI(X_47;Y_8)"]["MI_joint_knn_bits"]
            - mi_table["MI(X_31;Y_8)"]["MI_joint_knn_bits"], 4),
        "I(X_47;Y_8) - I(X_23;Y_8)": round(
            mi_table["MI(X_47;Y_8)"]["MI_joint_knn_bits"]
            - mi_table["MI(X_23;Y_8)"]["MI_joint_knn_bits"], 4),
    }
    delta_acc = {
        "acc(X_31) - acc(X_23)": round(acc_X31_Y8 - acc_X23_Y8, 4),
        "acc(X_47) - acc(X_31)": round(acc_X47_Y8 - acc_X31_Y8, 4),
        "acc(X_47) - acc(X_23)": round(acc_X47_Y8 - acc_X23_Y8, 4),
    }

    # ── Plot MI vs accuracy ───────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [
            mi_table["MI(X_23;Y_8)"]["MI_joint_knn_bits"],
            mi_table["MI(X_31;Y_8)"]["MI_joint_knn_bits"],
            mi_table["MI(X_47;Y_8)"]["MI_joint_knn_bits"],
        ]
        ys = [acc_X23_Y8, acc_X31_Y8, acc_X47_Y8]
        ceils = [
            mi_table["MI(X_23;Y_8)"]["fano_joint"]["accuracy_ceiling_tight"],
            mi_table["MI(X_31;Y_8)"]["fano_joint"]["accuracy_ceiling_tight"],
            mi_table["MI(X_47;Y_8)"]["fano_joint"]["accuracy_ceiling_tight"],
        ]
        labels = ["X_23", "X_31", "X_47"]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(xs, ys, "o-", label="Measured GBM accuracy", linewidth=2)
        ax.plot(xs, ceils, "s--", color="red",
                label="Fano ceiling (1 - P_e lower bound)", linewidth=2)
        for x, y_, c, lab in zip(xs, ys, ceils, labels):
            ax.annotate(lab, (x, y_), textcoords="offset points",
                        xytext=(6, -12), fontsize=10)
        ax.set_xlabel("Mutual information I(X; Y_8)  (bits)")
        ax.set_ylabel("Accuracy")
        ax.set_title("Fano ceiling vs measured accuracy (P0 agents, 8-class)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(str(FIG_PATH), dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\nSaved figure: {FIG_PATH}")
    except Exception as e:
        print(f"Figure generation failed: {e}")

    # ── Assemble final results ────────────────────────────────────────
    results = {
        "timestamp": datetime.now().isoformat(),
        "description": (
            "Information-theoretic bounds on 8-class and 3-class taxonomy "
            "classification via Fano's inequality."
        ),
        "n_samples": int(len(y8)),
        "feature_sets": {
            "X_23": ORIGINAL_23,
            "X_31": ORIGINAL_23 + AI_8,
            "X_47": ALL_47,
        },
        "label_sets": {
            "Y_8": {
                "source": "taxonomy_index (rule-based projection)",
                "H_Y_bits": round(H_Y8, 4),
                "class_counts": {
                    int(k): int((y8 == k).sum())
                    for k in np.unique(y8)
                },
            },
            "Y_3": {
                "source": "KMeans(k=3) on standardized X_23",
                "H_Y_bits": round(H_Y3, 4),
                "class_counts": {
                    int(k): int((y3 == k).sum())
                    for k in np.unique(y3)
                },
            },
        },
        "mi_table": mi_table,
        "measured_accuracies_Y8": {
            "X_23": round(acc_X23_Y8, 4),
            "X_31": round(acc_X31_Y8, 4),
            "X_47": round(acc_X47_Y8, 4),
        },
        "classifier_vs_fano": comparison,
        "delta_mi_bits": delta_mi,
        "delta_accuracy": delta_acc,
        "key_interpretation": (
            f"With X_31 we achieve {acc_X31_Y8:.1%} accuracy vs a Fano "
            f"ceiling of "
            f"{mi_table['MI(X_31;Y_8)']['fano_joint']['accuracy_ceiling_tight']:.1%}. "
            "See classifier_vs_fano.headroom_accuracy for the gap."
        ),
        "methodology_notes": {
            "mi_estimator_per_feature": (
                "sklearn.feature_selection.mutual_info_classif, "
                f"n_neighbors={MI_NEIGHBORS} (Kraskov-style k-NN, in nats)."
            ),
            "mi_estimator_joint": (
                f"k-NN plug-in: H(Y|X) estimated by averaging label entropy "
                f"across {MI_NEIGHBORS} nearest neighbors in standardized "
                "feature space, then I = H(Y) - H(Y|X). Bits."
            ),
            "fano_loose": "P_e >= (H(Y|X) - 1) / log2(|Y|)",
            "fano_tight": (
                "Numerically invert Fano: smallest P_e such that "
                "H_b(P_e) + P_e*log2(|Y|-1) >= H(Y|X)."
            ),
        },
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {RESULTS_PATH}")

    # ── Summary printout ──────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("SUMMARY — Fano vs measured accuracy (8-class)")
    print("=" * 80)
    hdr = f"{'Set':<6} {'I(X;Y) bits':<13} {'H(Y|X)':<10} {'Fano P_e':<10} " \
          f"{'acc ceiling':<13} {'measured acc':<14} {'headroom':<10}"
    print(hdr)
    print("-" * len(hdr))
    for Xname in ["X_23", "X_31", "X_47"]:
        key = f"MI({Xname};Y_8)"
        e = mi_table[key]
        cmp_ = comparison[f"{Xname}→Y_8"]
        print(
            f"{Xname:<6} {e['MI_joint_knn_bits']:<13.4f} "
            f"{e['H_Y_given_X_joint_bits']:<10.4f} "
            f"{e['fano_joint']['p_error_lower_tight']:<10.4f} "
            f"{e['fano_joint']['accuracy_ceiling_tight']:<13.4f} "
            f"{cmp_['measured_accuracy']:<14.4f} "
            f"{cmp_['headroom_accuracy']:<10.4f}"
        )

    print(f"\nΔMI  X23→X31 (added 8 AI):  +{delta_mi['I(X_31;Y_8) - I(X_23;Y_8)']:.4f} bits   "
          f"Δacc: {delta_acc['acc(X_31) - acc(X_23)']:+.4f}")
    print(f"ΔMI  X31→X47 (added 16 DP): +{delta_mi['I(X_47;Y_8) - I(X_31;Y_8)']:.4f} bits   "
          f"Δacc: {delta_acc['acc(X_47) - acc(X_31)']:+.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
