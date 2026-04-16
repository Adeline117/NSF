"""Glue: run all three audit steps and emit a single Markdown report."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

import pandas as pd
from sklearn.base import clone

from .audit import (
    CrossSchemeResult,
    OverlapResult,
    PurityTierResult,
    check_label_feature_overlap,
    compare_purity_tiers,
    cross_scheme_transfer,
)


@dataclass
class AuditDataset:
    """A dataset in the form expected by ``generate_audit_report``.

    Fields
    ------
    features : pd.DataFrame indexed by address, numeric columns only.
    labels : pd.Series indexed by address, binary (1 = agent).
    mining_rules : pd.DataFrame indexed by address. One column per rule used
        to generate labels. For datasets without explicit rules, pass an
        empty DataFrame.
    purity_tiers : mapping[str, (X, y)]
        Named tiers from lowest to highest label purity.
    schemes : mapping[str, (X, y)]
        Pairs of label schemes for cross-scheme transfer. The dict should
        have exactly two entries; the first will be scheme A.
    """

    features: pd.DataFrame
    labels: pd.Series
    mining_rules: pd.DataFrame
    purity_tiers: Mapping[str, tuple[pd.DataFrame, pd.Series]]
    schemes: Mapping[str, tuple[pd.DataFrame, pd.Series]]
    name: str = "unnamed dataset"


def _summary_verdict(
    overlap: OverlapResult,
    purity: PurityTierResult,
    transfer: CrossSchemeResult,
) -> str:
    flags = []
    if overlap.risk_flags:
        flags.append("Step 1 flagged direct rule↔feature overlap.")
    if purity.warning:
        flags.append("Step 2 flagged a large AUC drop across purity tiers.")
    if transfer.warning:
        flags.append("Step 3 flagged a cross-scheme transfer gap.")

    if not flags:
        return (
            "PASS — no leakage indicators triggered. The classifier's AUC is "
            "plausibly driven by behavioural signal."
        )

    bullets = "\n".join(f"- {f}" for f in flags)
    return (
        "FAIL — at least one leakage indicator triggered:\n"
        f"{bullets}\n"
        "Investigate before claiming the reported AUC."
    )


def generate_audit_report(classifier, dataset: AuditDataset) -> str:
    """Run all three audit steps and return a Markdown report.

    Parameters
    ----------
    classifier : sklearn estimator (unfitted)
    dataset : AuditDataset

    Returns
    -------
    str : Markdown-formatted report suitable for writing to disk or
          pasting into a notebook/paper appendix.
    """
    # Step 1
    if dataset.mining_rules.empty:
        overlap = OverlapResult(
            jaccard_by_rule_feature=pd.DataFrame(),
            pearson_by_rule_feature=pd.DataFrame(),
            spearman_by_rule_feature=pd.DataFrame(),
            max_jaccard_per_rule=pd.Series(dtype=float),
            max_corr_per_rule=pd.Series(dtype=float),
            risk_flags=["No mining rules supplied — Step 1 skipped."],
        )
    else:
        overlap = check_label_feature_overlap(
            dataset.labels, dataset.features, dataset.mining_rules
        )

    # Step 2
    purity = compare_purity_tiers(classifier, dataset.purity_tiers)

    # Step 3
    if len(dataset.schemes) >= 2:
        keys = list(dataset.schemes.keys())
        scheme_a = dataset.schemes[keys[0]]
        scheme_b = dataset.schemes[keys[1]]
        transfer = cross_scheme_transfer(classifier, scheme_a, scheme_b)
    else:
        transfer = CrossSchemeResult(
            a_to_b_auc=None, b_to_a_auc=None,
            a_internal_auc=None, b_internal_auc=None,
            transfer_gap=None,
            warning="Fewer than two label schemes supplied — Step 3 skipped.",
        )

    verdict = _summary_verdict(overlap, purity, transfer)

    lines = [
        f"# OnChainAudit report — `{dataset.name}`",
        "",
        f"_Generated {datetime.utcnow().isoformat(timespec='seconds')}Z_",
        "",
        f"**Classifier:** `{type(classifier).__name__}`",
        f"**N addresses:** {len(dataset.labels)}",
        f"**N agents:** {int(dataset.labels.sum())}",
        f"**N humans:** {int((dataset.labels == 0).sum())}",
        "",
        "## Summary verdict",
        "",
        verdict,
        "",
        "---",
        "",
        overlap.to_markdown(),
        "",
        "---",
        "",
        purity.to_markdown(),
        "",
        "---",
        "",
        transfer.to_markdown(),
        "",
    ]
    return "\n".join(lines)
