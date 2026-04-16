"""onchain_audit — audit on-chain agent-vs-human classifiers for label/feature leakage.

Three-step protocol proposed in:
    Wen et al. (2026). "Auditing on-chain agent-vs-human classifiers for
    label leakage", WWW '26 (submitted).

Quick usage
-----------
>>> from onchain_audit import generate_audit_report
>>> report = generate_audit_report(my_classifier, my_dataset)
>>> print(report)  # Markdown string, save to disk or paste into notebook.

Individual steps
----------------
>>> from onchain_audit import (
...     check_label_feature_overlap,
...     compare_purity_tiers,
...     cross_scheme_transfer,
... )
"""
from __future__ import annotations

from .audit import (
    check_label_feature_overlap,
    compare_purity_tiers,
    cross_scheme_transfer,
)
from .report import generate_audit_report

__all__ = [
    "check_label_feature_overlap",
    "compare_purity_tiers",
    "cross_scheme_transfer",
    "generate_audit_report",
]

__version__ = "0.1.0"
