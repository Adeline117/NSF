"""Three-step leakage audit for on-chain agent-vs-human classifiers.

Step 1 — ``check_label_feature_overlap``
    Quantify how much the rules that produced your labels overlap with the
    features your classifier consumes. Uses Jaccard overlap on supports and
    Pearson/Spearman correlation between each mining rule and each feature.

Step 2 — ``compare_purity_tiers``
    Re-evaluate the classifier across label-purity tiers (e.g. hand-curated vs
    heuristic-mined). A large AUC drop on the purer tier indicates the lift
    came from the mining rule itself, not from behavioural signal.

Step 3 — ``cross_scheme_transfer``
    Train on labels from scheme A, evaluate on labels from scheme B where the
    two schemes share no mining rule. Transfer AUC tells you how much of the
    classifier's decision function survives a change in the label-generating
    process.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Step 1: label-feature overlap
# ---------------------------------------------------------------------------


@dataclass
class OverlapResult:
    """Result of :func:`check_label_feature_overlap`."""

    jaccard_by_rule_feature: pd.DataFrame   # rows = rules, cols = features
    pearson_by_rule_feature: pd.DataFrame
    spearman_by_rule_feature: pd.DataFrame
    max_jaccard_per_rule: pd.Series
    max_corr_per_rule: pd.Series
    risk_flags: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = ["### Step 1 — Label/feature overlap"]
        if self.risk_flags:
            lines.append("")
            lines.append("**Risk flags:**")
            for flag in self.risk_flags:
                lines.append(f"- {flag}")
        lines.append("")
        lines.append("**Max |Pearson| between each mining rule and any feature:**")
        lines.append("")
        lines.append("| Rule | max |Pearson| | max Jaccard |")
        lines.append("|------|---------------:|------------:|")
        for rule in self.max_corr_per_rule.index:
            lines.append(
                f"| `{rule}` | {self.max_corr_per_rule[rule]:.3f} | "
                f"{self.max_jaccard_per_rule[rule]:.3f} |"
            )
        return "\n".join(lines)


def _rule_to_boolean(rule_values: np.ndarray) -> np.ndarray:
    """Coerce a rule column to boolean support."""
    arr = np.asarray(rule_values)
    if arr.dtype == bool:
        return arr
    if np.issubdtype(arr.dtype, np.number):
        # >0 counts as 'rule fires'
        return arr > 0
    # strings: non-empty / non-False
    return np.array([bool(x) and str(x).lower() not in {"0", "false", "none", ""}
                     for x in arr])


def _feature_to_boolean(feat_values: np.ndarray) -> np.ndarray:
    """Binarise a feature at its median (support comparison only)."""
    arr = np.asarray(feat_values, dtype=float)
    if np.all(np.isnan(arr)):
        return np.zeros_like(arr, dtype=bool)
    med = float(np.nanmedian(arr))
    return (arr > med) & ~np.isnan(arr)


def check_label_feature_overlap(
    labels: pd.Series,
    features: pd.DataFrame,
    mining_rules: pd.DataFrame,
    *,
    corr_threshold: float = 0.6,
    jaccard_threshold: float = 0.5,
) -> OverlapResult:
    """Step 1. Quantify overlap between label-producing rules and model features.

    Parameters
    ----------
    labels : pd.Series
        Binary labels indexed by address (1 = agent, 0 = human). Only used to
        align the inputs.
    features : pd.DataFrame
        Classifier features, indexed by address. Numeric columns only.
    mining_rules : pd.DataFrame
        One column per rule used to *generate* labels, indexed by address.
        Columns can be boolean, numeric, or string; they will be coerced to
        booleans (True = rule fires on this address).
    corr_threshold, jaccard_threshold : float
        Risk flags are emitted when any (rule, feature) pair exceeds either
        threshold.

    Returns
    -------
    OverlapResult
    """
    idx = labels.index.intersection(features.index).intersection(mining_rules.index)
    features = features.loc[idx].select_dtypes(include=[np.number])
    mining_rules = mining_rules.loc[idx]

    rule_names = list(mining_rules.columns)
    feat_names = list(features.columns)

    jac = pd.DataFrame(index=rule_names, columns=feat_names, dtype=float)
    pears = pd.DataFrame(index=rule_names, columns=feat_names, dtype=float)
    spear = pd.DataFrame(index=rule_names, columns=feat_names, dtype=float)

    for rule in rule_names:
        rbool = _rule_to_boolean(mining_rules[rule].values)
        rfloat = rbool.astype(float)
        for feat in feat_names:
            fvals = features[feat].values.astype(float)
            fbool = _feature_to_boolean(fvals)

            # Jaccard on the two boolean supports
            union = int(np.sum(rbool | fbool))
            inter = int(np.sum(rbool & fbool))
            jac.at[rule, feat] = inter / union if union > 0 else 0.0

            # Pearson / Spearman on numeric values
            mask = ~np.isnan(fvals)
            if mask.sum() < 3 or rfloat[mask].std() == 0 or fvals[mask].std() == 0:
                pears.at[rule, feat] = 0.0
                spear.at[rule, feat] = 0.0
                continue
            pears.at[rule, feat] = float(abs(np.corrcoef(rfloat[mask], fvals[mask])[0, 1]))
            try:
                rho, _ = stats.spearmanr(rfloat[mask], fvals[mask])
                spear.at[rule, feat] = float(abs(rho)) if np.isfinite(rho) else 0.0
            except Exception:
                spear.at[rule, feat] = 0.0

    max_jac = jac.max(axis=1)
    max_corr = pears.max(axis=1)

    flags: list[str] = []
    for rule in rule_names:
        if max_corr[rule] >= corr_threshold:
            best = pears.loc[rule].idxmax()
            flags.append(
                f"Rule `{rule}` has |Pearson|={max_corr[rule]:.3f} with feature "
                f"`{best}` (>= {corr_threshold}) — suspected direct leakage."
            )
        if max_jac[rule] >= jaccard_threshold:
            best = jac.loc[rule].idxmax()
            flags.append(
                f"Rule `{rule}` has Jaccard={max_jac[rule]:.3f} with feature "
                f"`{best}` (>= {jaccard_threshold}) — supports nearly coincide."
            )

    return OverlapResult(
        jaccard_by_rule_feature=jac,
        pearson_by_rule_feature=pears,
        spearman_by_rule_feature=spear,
        max_jaccard_per_rule=max_jac,
        max_corr_per_rule=max_corr,
        risk_flags=flags,
    )


# ---------------------------------------------------------------------------
# Step 2: purity-tier comparison
# ---------------------------------------------------------------------------


@dataclass
class PurityTierResult:
    per_tier: dict                     # tier_name -> dict(auc_mean, auc_std, n, n_agents, n_humans)
    drop_vs_best: float                # best_auc - worst_tier_auc (flagged if large)
    warning: str | None = None

    def to_markdown(self) -> str:
        lines = ["### Step 2 — Label-purity tier comparison"]
        lines.append("")
        lines.append("| Tier | N | Agents | Humans | AUC (mean) | AUC (std) |")
        lines.append("|------|---:|------:|------:|----------:|----------:|")
        for tier, r in self.per_tier.items():
            auc_mean = f"{r['auc_mean']:.3f}" if r["auc_mean"] is not None else "n/a"
            auc_std = f"{r['auc_std']:.3f}" if r["auc_std"] is not None else "n/a"
            lines.append(
                f"| {tier} | {r['n']} | {r['n_agents']} | {r['n_humans']} | "
                f"{auc_mean} | {auc_std} |"
            )
        lines.append("")
        lines.append(f"**Drop vs best tier:** {self.drop_vs_best:.3f}")
        if self.warning:
            lines.append("")
            lines.append(f"**Warning:** {self.warning}")
        return "\n".join(lines)


def compare_purity_tiers(
    classifier,
    tiers: Mapping[str, tuple[pd.DataFrame, pd.Series]],
    *,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
    drop_threshold: float = 0.15,
) -> PurityTierResult:
    """Step 2. Evaluate ``classifier`` on each label-purity tier independently.

    Parameters
    ----------
    classifier : sklearn estimator
        Unfitted estimator (will be ``clone``-d per fold).
    tiers : mapping[str, (X, y)]
        Each entry is a named tier with its own (features, labels). Tiers should
        go from lowest to highest purity (e.g. "all_mined", "partial_curated",
        "strict_curated").
    n_splits, n_repeats : int
        Repeated stratified K-fold parameters.
    drop_threshold : float
        If (best_tier_auc - worst_tier_auc) >= this value, emit a leakage
        warning.
    """
    per_tier: dict = {}
    rng = np.random.default_rng(seed)

    for tier_name, (X, y) in tiers.items():
        X_arr = X.values.astype(float)
        y_arr = np.asarray(y).astype(int)
        n = len(y_arr)
        n_agents = int(y_arr.sum())
        n_humans = n - n_agents

        if n < n_splits * 2 or n_agents < n_splits or n_humans < n_splits:
            per_tier[tier_name] = {
                "auc_mean": None, "auc_std": None,
                "n": n, "n_agents": n_agents, "n_humans": n_humans,
                "note": "Too few samples for CV",
            }
            continue

        aucs = []
        for rep in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                  random_state=int(rng.integers(0, 2**31)))
            for tr, te in skf.split(X_arr, y_arr):
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(X_arr[tr])
                Xte = scaler.transform(X_arr[te])
                clf = clone(classifier)
                clf.fit(Xtr, y_arr[tr])
                if hasattr(clf, "predict_proba"):
                    prob = clf.predict_proba(Xte)[:, 1]
                else:
                    prob = clf.decision_function(Xte)
                if len(np.unique(y_arr[te])) > 1:
                    aucs.append(roc_auc_score(y_arr[te], prob))

        per_tier[tier_name] = {
            "auc_mean": float(np.mean(aucs)) if aucs else None,
            "auc_std": float(np.std(aucs)) if aucs else None,
            "n": n, "n_agents": n_agents, "n_humans": n_humans,
            "n_folds_used": len(aucs),
        }

    valid = [r["auc_mean"] for r in per_tier.values() if r.get("auc_mean") is not None]
    if len(valid) >= 2:
        drop = max(valid) - min(valid)
    else:
        drop = 0.0

    warning = None
    if drop >= drop_threshold:
        warning = (
            f"AUC drops by {drop:.3f} between highest and lowest tier. "
            "A large drop suggests the classifier's lift depends on the "
            "label-generating rule rather than a stable behavioural signal."
        )

    return PurityTierResult(per_tier=per_tier, drop_vs_best=drop, warning=warning)


# ---------------------------------------------------------------------------
# Step 3: cross-scheme transfer
# ---------------------------------------------------------------------------


@dataclass
class CrossSchemeResult:
    a_to_b_auc: float | None
    b_to_a_auc: float | None
    a_internal_auc: float | None
    b_internal_auc: float | None
    transfer_gap: float | None
    warning: str | None = None

    def to_markdown(self) -> str:
        def fmt(x):
            return f"{x:.3f}" if x is not None else "n/a"

        lines = ["### Step 3 — Cross-scheme transfer"]
        lines.append("")
        lines.append("| Direction | AUC |")
        lines.append("|-----------|----:|")
        lines.append(f"| Scheme A internal (CV) | {fmt(self.a_internal_auc)} |")
        lines.append(f"| Scheme B internal (CV) | {fmt(self.b_internal_auc)} |")
        lines.append(f"| Train A → Test B | {fmt(self.a_to_b_auc)} |")
        lines.append(f"| Train B → Test A | {fmt(self.b_to_a_auc)} |")
        lines.append("")
        lines.append(f"**Transfer gap (internal - transferred):** {fmt(self.transfer_gap)}")
        if self.warning:
            lines.append("")
            lines.append(f"**Warning:** {self.warning}")
        return "\n".join(lines)


def _cv_auc(
    classifier,
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 3,
    seed: int = 42,
) -> float | None:
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < n_splits:
        return None
    rng = np.random.default_rng(seed)
    aucs = []
    for rep in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=int(rng.integers(0, 2**31)))
        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            clf = clone(classifier)
            clf.fit(Xtr, y[tr])
            if hasattr(clf, "predict_proba"):
                p = clf.predict_proba(Xte)[:, 1]
            else:
                p = clf.decision_function(Xte)
            if len(np.unique(y[te])) > 1:
                aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else None


def _train_predict_auc(
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> float | None:
    if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
        return None
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)
    clf = clone(classifier)
    clf.fit(Xtr, y_train)
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(Xte)[:, 1]
    else:
        p = clf.decision_function(Xte)
    return float(roc_auc_score(y_test, p))


def cross_scheme_transfer(
    model,
    scheme_a: tuple[pd.DataFrame, pd.Series],
    scheme_b: tuple[pd.DataFrame, pd.Series],
    *,
    gap_threshold: float = 0.15,
) -> CrossSchemeResult:
    """Step 3. Test whether the classifier transfers across label schemes.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted estimator (cloned per call).
    scheme_a, scheme_b : (X, y)
        Two datasets whose labels come from *independent* schemes. For example,
        scheme A = "curated MEV bots", scheme B = "Chainlink keepers". The
        features should be identical column sets.
    gap_threshold : float
        If mean internal AUC minus mean transfer AUC exceeds this, a warning
        fires.
    """
    X_a, y_a = scheme_a
    X_b, y_b = scheme_b
    feat_cols = [c for c in X_a.columns if c in X_b.columns]
    X_a_arr = X_a[feat_cols].values.astype(float)
    X_b_arr = X_b[feat_cols].values.astype(float)
    y_a_arr = np.asarray(y_a).astype(int)
    y_b_arr = np.asarray(y_b).astype(int)

    a_int = _cv_auc(model, X_a_arr, y_a_arr)
    b_int = _cv_auc(model, X_b_arr, y_b_arr)
    a_to_b = _train_predict_auc(model, X_a_arr, y_a_arr, X_b_arr, y_b_arr)
    b_to_a = _train_predict_auc(model, X_b_arr, y_b_arr, X_a_arr, y_a_arr)

    internal_vals = [v for v in (a_int, b_int) if v is not None]
    transfer_vals = [v for v in (a_to_b, b_to_a) if v is not None]
    if internal_vals and transfer_vals:
        gap = float(np.mean(internal_vals) - np.mean(transfer_vals))
    else:
        gap = None

    warning = None
    if gap is not None and gap >= gap_threshold:
        warning = (
            f"Cross-scheme transfer drops by {gap:.3f} below internal CV. "
            "The classifier learned scheme-specific cues, not a behavioural "
            "definition of 'agent' that generalises across label sources."
        )

    return CrossSchemeResult(
        a_to_b_auc=a_to_b,
        b_to_a_auc=b_to_a,
        a_internal_auc=a_int,
        b_internal_auc=b_int,
        transfer_gap=gap,
        warning=warning,
    )
