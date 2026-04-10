#!/usr/bin/env python3
"""
False-positive heuristic analysis for the 138-server scan.

Stratified random sample of 100 findings (20 critical, 30 high, 50 medium),
classified as TP/FP by automated heuristics.  Produces precision estimates
with Wilson confidence intervals, broken down by severity and category.

Output: false_positive_analysis.json  (same directory)
"""

import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
SCAN_PATH = HERE / "full_catalog_scan_results.json"
OUTPUT_PATH = HERE / "false_positive_analysis.json"

SEED = 20260409  # reproducible sampling


# ---------------------------------------------------------------------------
# Wilson score confidence interval (95%)
# ---------------------------------------------------------------------------
def wilson_ci(successes: int, total: int, z: float = 1.96) -> dict:
    """Return Wilson score interval for a proportion."""
    if total == 0:
        return {"lower": 0.0, "upper": 0.0, "point": 0.0}
    p_hat = successes / total
    denom = 1 + z**2 / total
    centre = (p_hat + z**2 / (2 * total)) / denom
    spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * total)) / total) / denom
    return {
        "point": round(p_hat, 4),
        "lower": round(max(0.0, centre - spread), 4),
        "upper": round(min(1.0, centre + spread), 4),
    }


# ---------------------------------------------------------------------------
# Heuristic TP / FP classifier
# ---------------------------------------------------------------------------

# File-path patterns indicating test / example / vendored code
_TEST_RE = re.compile(
    r"(^|/)("
    r"tests?/|__tests__/|spec/|test_|\.test\.|\.spec\."
    r"|e2e/|cypress/|playwright/"
    r")",
    re.IGNORECASE,
)

_EXAMPLE_RE = re.compile(
    r"(^|/)("
    r"examples?/|samples?/|demos?/|mock/|mocks/|fixtures?/"
    r"|snippets?/|tutorials?/"
    r")",
    re.IGNORECASE,
)

_VENDOR_RE = re.compile(
    r"(^|/)("
    r"node_modules/|vendor/|dist/|build/|\.min\."
    r"|third.?party/|external/"
    r")",
    re.IGNORECASE,
)

# Patterns whose single-keyword match inside a comment is likely a false positive
_GENERIC_PIDS = {"IV-002", "IV-001", "SC-001"}

# Pattern IDs with high intrinsic precision (structural matches)
_HIGH_PRECISION_PIDS = {"HC-001", "PKE-001", "PKE-002", "PKE-003", "UA-001", "AI-001"}


def _context_line_is_comment(context: str) -> bool:
    """Check whether the >>> matched line is purely a comment."""
    m = re.search(r">>>\s*\d+:\s*(.*)$", context, re.MULTILINE)
    if not m:
        return False
    line = m.group(1).strip()
    # JS/TS/Solidity single-line comment
    if line.startswith("//"):
        return True
    # Python / shell comment
    if line.startswith("#"):
        return True
    # Block-comment interior (* leading)
    if re.match(r"^\*", line):
        return True
    return False


def classify_finding(finding: dict) -> str:
    """
    Return 'TP' or 'FP' based on heuristic rules.

    Rules (applied in order; first match wins):
      1. File in node_modules / vendor / dist / build  -> FP
      2. File is a test/spec file AND severity != critical -> FP
      3. File is an example/demo/mock -> FP
      4. Matched line is purely a comment AND pattern is generic -> FP
      5. High-precision pattern with non-trivial matched_text -> TP
      6. Context shows actual function call/definition matching the
         pattern description -> TP
      7. Generic pattern (IV-002, IV-001, SC-001) with matched_text
         shorter than 10 chars -> FP  (overly broad regex hit)
      8. Default -> TP  (conservative: treat as true positive)
    """
    filepath = finding.get("file", "")
    matched = finding.get("matched_text", "")
    context = finding.get("context", "")
    pid = finding.get("pattern_id", "")
    severity = finding.get("severity", "")

    # Rule 1: vendored / build artefact
    if _VENDOR_RE.search(filepath):
        return "FP"

    # Rule 2: test file (except critical — test files can still reveal
    # real hardcoded secrets)
    if _TEST_RE.search(filepath) and severity != "critical":
        return "FP"

    # Rule 3: example / demo / mock
    if _EXAMPLE_RE.search(filepath):
        return "FP"

    # Rule 4: comment-only match for a generic pattern
    if pid in _GENERIC_PIDS and _context_line_is_comment(context):
        return "FP"

    # Rule 5: high-precision pattern with substantive match
    if pid in _HIGH_PRECISION_PIDS and len(matched) >= 8:
        return "TP"

    # Rule 6: context shows real function call / definition
    # (callTool, execute, transfer, approve, sign — not in a string/comment)
    if re.search(
        r"(callTool|execute|transfer|approve|signTransaction|os\.system|subprocess|eval)\s*\(",
        context,
    ):
        return "TP"

    # Rule 7: overly short match on a generic pattern
    if pid in _GENERIC_PIDS and len(matched.strip()) < 10:
        return "FP"

    # Rule 8: test file with critical severity (e.g., test secrets) —
    # still likely FP in practice since test secrets are not production
    if _TEST_RE.search(filepath) and severity == "critical":
        return "FP"

    # Default: conservative TP
    return "TP"


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def main() -> None:
    with open(SCAN_PATH) as f:
        data = json.load(f)

    # Flatten findings with repo metadata
    all_findings: list[dict] = []
    for repo in data["repo_results"]:
        for finding in repo.get("findings", []):
            entry = dict(finding)
            entry["repo"] = repo["full_name"]
            entry["catalog_protocol"] = repo["catalog_protocol"]
            all_findings.append(entry)

    print(f"Total findings loaded: {len(all_findings)}")

    # --- Stratified sampling ---
    by_severity: dict[str, list[dict]] = defaultdict(list)
    for f in all_findings:
        by_severity[f["severity"]].append(f)

    target = {"critical": 20, "high": 30, "medium": 50}
    rng = random.Random(SEED)

    sample: list[dict] = []
    for sev, n in target.items():
        pool = by_severity[sev]
        n_actual = min(n, len(pool))
        sample.extend(rng.sample(pool, n_actual))
        print(f"  Sampled {n_actual}/{len(pool)} {sev} findings")

    print(f"Sample size: {len(sample)}")

    # --- Classify each finding ---
    classifications: list[dict] = []
    for finding in sample:
        verdict = classify_finding(finding)
        classifications.append({
            "repo": finding["repo"],
            "file": finding["file"],
            "line": finding["line"],
            "pattern_id": finding["pattern_id"],
            "category": finding["category"],
            "severity": finding["severity"],
            "matched_text": finding["matched_text"][:120],
            "verdict": verdict,
            "catalog_protocol": finding["catalog_protocol"],
        })

    # --- Compute precision ---
    tp_count = sum(1 for c in classifications if c["verdict"] == "TP")
    fp_count = sum(1 for c in classifications if c["verdict"] == "FP")
    total = len(classifications)

    overall_precision = wilson_ci(tp_count, total)

    # By severity
    precision_by_severity: dict[str, dict] = {}
    for sev in ["critical", "high", "medium"]:
        sev_items = [c for c in classifications if c["severity"] == sev]
        tp = sum(1 for c in sev_items if c["verdict"] == "TP")
        precision_by_severity[sev] = {
            "n": len(sev_items),
            "tp": tp,
            "fp": len(sev_items) - tp,
            **wilson_ci(tp, len(sev_items)),
        }

    # By category
    precision_by_category: dict[str, dict] = {}
    cat_groups: dict[str, list[dict]] = defaultdict(list)
    for c in classifications:
        cat_groups[c["category"]].append(c)
    for cat, items in sorted(cat_groups.items(), key=lambda x: -len(x[1])):
        tp = sum(1 for c in items if c["verdict"] == "TP")
        precision_by_category[cat] = {
            "n": len(items),
            "tp": tp,
            "fp": len(items) - tp,
            **wilson_ci(tp, len(items)),
        }

    # By protocol
    precision_by_protocol: dict[str, dict] = {}
    proto_groups: dict[str, list[dict]] = defaultdict(list)
    for c in classifications:
        proto_groups[c["catalog_protocol"]].append(c)
    for proto, items in sorted(proto_groups.items()):
        tp = sum(1 for c in items if c["verdict"] == "TP")
        precision_by_protocol[proto] = {
            "n": len(items),
            "tp": tp,
            "fp": len(items) - tp,
            **wilson_ci(tp, len(items)),
        }

    # --- FP reasons summary ---
    fp_reasons: Counter = Counter()
    for finding, cls in zip(sample, classifications):
        if cls["verdict"] == "FP":
            filepath = finding["file"]
            pid = finding["pattern_id"]
            if _VENDOR_RE.search(filepath):
                fp_reasons["vendored/build artefact"] += 1
            elif _TEST_RE.search(filepath):
                fp_reasons["test/spec file"] += 1
            elif _EXAMPLE_RE.search(filepath):
                fp_reasons["example/demo/mock"] += 1
            elif pid in _GENERIC_PIDS and _context_line_is_comment(finding.get("context", "")):
                fp_reasons["comment-only match (generic pattern)"] += 1
            elif pid in _GENERIC_PIDS and len(finding.get("matched_text", "").strip()) < 10:
                fp_reasons["overly short generic match"] += 1
            else:
                fp_reasons["other"] += 1

    # --- Assemble output ---
    result = {
        "metadata": {
            "description": "Heuristic false-positive analysis of 138-server scan",
            "sample_size": total,
            "stratification": target,
            "seed": SEED,
            "total_findings_in_corpus": len(all_findings),
            "heuristic_rules": [
                "R1: file in node_modules/vendor/dist/build -> FP",
                "R2: file is test/spec (non-critical) -> FP",
                "R3: file is example/demo/mock -> FP",
                "R4: generic pattern + comment-only matched line -> FP",
                "R5: high-precision pattern with substantive match -> TP",
                "R6: context shows real function call/definition -> TP",
                "R7: generic pattern + matched_text < 10 chars -> FP",
                "R8: test file + critical -> FP (test secrets)",
                "R9: default -> TP (conservative)",
            ],
        },
        "overall": {
            "true_positives": tp_count,
            "false_positives": fp_count,
            "total": total,
            "precision": overall_precision,
        },
        "precision_by_severity": precision_by_severity,
        "precision_by_category": precision_by_category,
        "precision_by_protocol": precision_by_protocol,
        "fp_reason_breakdown": dict(fp_reasons.most_common()),
        "sample_classifications": classifications,
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print(f"FALSE POSITIVE ANALYSIS — 138-server scan")
    print(f"{'='*60}")
    print(f"Sample size:       {total}")
    print(f"True positives:    {tp_count}")
    print(f"False positives:   {fp_count}")
    print(f"Overall precision: {overall_precision['point']:.1%}"
          f"  (95% Wilson CI: [{overall_precision['lower']:.1%},"
          f" {overall_precision['upper']:.1%}])")
    print()
    print("Precision by severity:")
    for sev in ["critical", "high", "medium"]:
        p = precision_by_severity[sev]
        print(f"  {sev:10s}  {p['tp']}/{p['n']}"
              f"  = {p['point']:.1%}"
              f"  [{p['lower']:.1%}, {p['upper']:.1%}]")
    print()
    print("Precision by category:")
    for cat, p in precision_by_category.items():
        print(f"  {cat:35s}  {p['tp']}/{p['n']}"
              f"  = {p['point']:.1%}"
              f"  [{p['lower']:.1%}, {p['upper']:.1%}]")
    print()
    print("FP reason breakdown:")
    for reason, cnt in fp_reasons.most_common():
        print(f"  {reason:45s}  {cnt}")
    print()
    print(f"Results saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
