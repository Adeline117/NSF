"""
Paper 2: Manual Validation Simulation
======================================
Take the top 100 highest-severity findings from cleaned results and apply
heuristic validation rules to estimate precision per severity level and
per category.

Validation heuristics:
  1. Is it in a test file?              -> false positive
  2. Is it in a comment?                -> false positive
  3. Is the match in an actual code path? -> likely true positive
  4. Does surrounding context suggest intentional behavior? -> true positive
  5. Is it in a dead code block (commented out)? -> false positive
  6. Does the file path suggest example/demo code? -> false positive
  7. Is the regex match too broad for the context? -> uncertain/false positive

Usage:
    python paper2_agent_tool_security/experiments/validate_findings.py

Requires cleaned_results.json from clean_and_validate.py.
"""

import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CLEANED_PATH = os.path.join(SCRIPT_DIR, "cleaned_results.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "validation_results.json")

# ============================================================
# SEVERITY ORDERING
# ============================================================

SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}

# ============================================================
# VALIDATION RULES
# ============================================================

def is_test_file(file_path: str) -> bool:
    """Check if file is a test/spec/example file."""
    indicators = [
        "test/", "tests/", "__tests__/", "spec/", "specs/",
        ".test.", ".spec.", "_test.", "_spec.",
        "test_", "spec_", "mock/", "mocks/",
        "fixture/", "fixtures/", "__fixtures__/",
        "example/", "examples/", "demo/", "demos/",
        "sample/", "samples/", "playground/",
        "e2e/", "integration/", "cypress/",
    ]
    fp_lower = file_path.lower()
    return any(ind in fp_lower for ind in indicators)


def is_comment_match(context: str) -> bool:
    """Check if the matched line (marked with >>>) is a comment."""
    for line in context.split("\n"):
        if ">>>" in line:
            # Extract the code portion (after the line number)
            code_match = re.search(r">>>\s*\d+:\s*(.*)", line)
            if code_match:
                code = code_match.group(1).strip()
                # Single-line comment patterns
                if code.startswith("//") or code.startswith("#"):
                    return True
                if code.startswith("*") or code.startswith("/*"):
                    return True
                if code.startswith("'''") or code.startswith('"""'):
                    return True
                # Check for inline comment where the match is after the comment marker
                if re.match(r".*\s+(?://|#)\s+.*", code):
                    # The code has an inline comment, but the match might be in the code part
                    pass
            break
    return False


def is_in_block_comment(context: str) -> bool:
    """Check if the matched line is inside a block comment."""
    lines = context.split("\n")
    in_block = False
    for line in lines:
        if "/*" in line:
            in_block = True
        if ">>>" in line and in_block:
            return True
        if "*/" in line:
            in_block = False
    return False


def is_in_docstring(context: str) -> bool:
    """Check if the matched line is inside a Python docstring."""
    lines = context.split("\n")
    triple_count = 0
    for line in lines:
        code_match = re.search(r"\d+:\s*(.*)", line)
        if code_match:
            code = code_match.group(1)
            triple_count += code.count('"""') + code.count("'''")
        if ">>>" in line:
            # If we've seen an odd number of triple quotes before this line,
            # we're inside a docstring
            if triple_count % 2 == 1:
                return True
            break
    return False


def suggests_intentional_behavior(context: str, category: str) -> bool:
    """Check if context suggests the behavior is intentional (not a bug)."""
    context_lower = context.lower()

    # Common intentional patterns
    intentional_markers = [
        "// intentional", "# intentional", "// deliberate", "# deliberate",
        "// by design", "# by design", "// expected", "# expected",
        "// todo", "# todo", "// fixme", "# fixme",
        "// note:", "# note:", "// warning:", "# warning:",
    ]
    if any(marker in context_lower for marker in intentional_markers):
        return True

    # For hardcoded credentials: test/placeholder values
    if category == "hardcoded_credentials":
        placeholder_markers = [
            "placeholder", "example", "dummy", "test", "sample",
            "your_", "replace_", "xxx", "changeme", "todo",
        ]
        if any(m in context_lower for m in placeholder_markers):
            return True  # Intentional placeholder, not a real credential

    return False


def is_broad_regex_match(finding: dict) -> bool:
    """Check if the pattern matched too broadly for the context."""
    pattern_id = finding.get("pattern_id", "")
    matched_text = finding.get("matched_text", "")
    context = finding.get("context", "")

    # IV-002 matching generic 'value: string' without crypto context
    if pattern_id == "IV-002":
        if "value" in matched_text.lower():
            crypto_words = [
                "wei", "gwei", "ether", "token", "amount", "balance",
                "transfer", "approve", "allowance", "deposit", "withdraw",
                "eth", "btc", "sol", "usdt", "usdc",
            ]
            if not any(w in context.lower() for w in crypto_words):
                return True

    # IV-001 matching non-Ethereum addresses
    if pattern_id == "IV-001":
        non_crypto = [
            "proxy", "server", "remote", "ip", "email", "url",
            "host", "endpoint", "callback", "redirect",
        ]
        if any(nc in matched_text.lower() for nc in non_crypto):
            return True

    # SC-001 matching simple Python 'global' statements
    if pattern_id == "SC-001":
        if re.search(r"global\s+\w+", matched_text) and "session" not in context.lower():
            return True

    return False


def validate_finding(finding: dict) -> dict:
    """
    Apply heuristic validation to a single finding.
    Returns the finding with added validation fields:
      - validation_label: 'true_positive', 'false_positive', 'uncertain'
      - validation_reason: explanation
      - validation_confidence: 0.0 to 1.0
    """
    file_path = finding.get("file", "")
    context = finding.get("context", "")
    category = finding.get("category", "")
    severity = finding.get("severity", "")

    label = "true_positive"
    reason = "Default: finding in production code path"
    confidence = 0.7

    # Rule 1: Test file
    if is_test_file(file_path):
        label = "false_positive"
        reason = "Finding in test/example/demo file"
        confidence = 0.9

    # Rule 2: Comment
    elif is_comment_match(context):
        label = "false_positive"
        reason = "Match is in a code comment"
        confidence = 0.95

    # Rule 3: Block comment
    elif is_in_block_comment(context):
        label = "false_positive"
        reason = "Match is inside a block comment"
        confidence = 0.95

    # Rule 4: Docstring
    elif is_in_docstring(context):
        label = "false_positive"
        reason = "Match is inside a docstring"
        confidence = 0.85

    # Rule 5: Broad regex match
    elif is_broad_regex_match(finding):
        label = "false_positive"
        reason = "Regex matched too broadly for the context"
        confidence = 0.8

    # Rule 6: Intentional behavior
    elif suggests_intentional_behavior(context, category):
        if category == "hardcoded_credentials":
            label = "false_positive"
            reason = "Credential is a placeholder/example value"
            confidence = 0.85
        else:
            label = "uncertain"
            reason = "Context suggests intentional behavior"
            confidence = 0.5

    # Rule 7: Critical findings in real code paths with crypto context
    if label == "true_positive" and severity == "critical":
        crypto_evidence = [
            "private", "key", "mnemonic", "seed", "sign",
            "transfer", "approve", "delegatecall", "selfdestruct",
        ]
        if any(w in context.lower() for w in crypto_evidence):
            confidence = 0.9
            reason = "Critical finding with strong crypto-security evidence"

    # Rule 8: High findings with tool-call patterns
    if label == "true_positive" and category in ("tool_poisoning", "prompt_injection"):
        confidence = 0.85
        reason = "Agent security finding (tool poisoning / prompt injection)"

    finding["validation_label"] = label
    finding["validation_reason"] = reason
    finding["validation_confidence"] = confidence
    return finding


# ============================================================
# PRECISION CALCULATION
# ============================================================

def compute_precision(findings: list, group_key: str) -> dict:
    """Compute estimated precision grouped by a key (severity or category)."""
    groups = defaultdict(lambda: {"tp": 0, "fp": 0, "uncertain": 0, "total": 0})

    for f in findings:
        group = f.get(group_key, "unknown")
        label = f.get("validation_label", "uncertain")
        groups[group]["total"] += 1
        if label == "true_positive":
            groups[group]["tp"] += 1
        elif label == "false_positive":
            groups[group]["fp"] += 1
        else:
            groups[group]["uncertain"] += 1

    result = {}
    for group, counts in sorted(groups.items()):
        tp = counts["tp"]
        total = counts["total"]
        precision = (tp / total * 100) if total > 0 else 0.0
        result[group] = {
            "total": total,
            "true_positive": tp,
            "false_positive": counts["fp"],
            "uncertain": counts["uncertain"],
            "precision_pct": round(precision, 1),
        }
    return result


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Paper 2: Manual Validation Simulation")
    print("=" * 60)
    print()

    # Load cleaned results
    if not os.path.exists(CLEANED_PATH):
        print(f"ERROR: {CLEANED_PATH} not found.")
        print("Run clean_and_validate.py first.")
        sys.exit(1)

    with open(CLEANED_PATH, "r") as f:
        data = json.load(f)

    # Collect all findings from all kept repos
    all_findings = []
    for rr in data.get("repo_results", []):
        repo_name = rr["full_name"]
        for f in rr.get("findings", []):
            f["_repo"] = repo_name
            all_findings.append(f)

    print(f"Total findings from cleaned data: {len(all_findings)}")

    # ---- Sort by severity (critical > high > medium > low) ----
    all_findings.sort(
        key=lambda f: (SEVERITY_ORDER.get(f.get("severity", "low"), 99),
                       -len(f.get("context", "")))
    )

    # ---- Take top 100 ----
    top_n = min(100, len(all_findings))
    top_findings = all_findings[:top_n]
    print(f"Selected top {top_n} findings by severity for validation")
    print()

    # ---- Validate each finding ----
    print("--- Applying validation heuristics ---")
    for f in top_findings:
        validate_finding(f)

    # ---- Count results ----
    label_counts = Counter(f["validation_label"] for f in top_findings)
    print(f"  True positives:  {label_counts.get('true_positive', 0)}")
    print(f"  False positives: {label_counts.get('false_positive', 0)}")
    print(f"  Uncertain:       {label_counts.get('uncertain', 0)}")
    print()

    # ---- Precision by severity ----
    print("--- Precision by Severity ---")
    precision_by_severity = compute_precision(top_findings, "severity")
    print(f"  {'Severity':<12} {'Total':>6} {'TP':>6} {'FP':>6} {'?':>6} {'Precision':>10}")
    print("  " + "-" * 48)
    for sev in ["critical", "high", "medium", "low"]:
        if sev in precision_by_severity:
            p = precision_by_severity[sev]
            print(f"  {sev:<12} {p['total']:>6} {p['true_positive']:>6} "
                  f"{p['false_positive']:>6} {p['uncertain']:>6} "
                  f"{p['precision_pct']:>9.1f}%")
    print()

    # ---- Precision by category ----
    print("--- Precision by Category ---")
    precision_by_category = compute_precision(top_findings, "category")
    print(f"  {'Category':<30} {'Total':>6} {'TP':>6} {'FP':>6} {'?':>6} {'Precision':>10}")
    print("  " + "-" * 66)
    for cat in sorted(precision_by_category.keys()):
        p = precision_by_category[cat]
        print(f"  {cat:<30} {p['total']:>6} {p['true_positive']:>6} "
              f"{p['false_positive']:>6} {p['uncertain']:>6} "
              f"{p['precision_pct']:>9.1f}%")
    print()

    # ---- Precision by repo ----
    print("--- Precision by Repo ---")
    precision_by_repo = compute_precision(top_findings, "_repo")
    print(f"  {'Repo':<45} {'Total':>6} {'TP':>6} {'Prec':>8}")
    print("  " + "-" * 63)
    for repo in sorted(precision_by_repo.keys(),
                       key=lambda r: precision_by_repo[r]["total"], reverse=True):
        p = precision_by_repo[repo]
        print(f"  {repo:<45} {p['total']:>6} {p['true_positive']:>6} "
              f"{p['precision_pct']:>7.1f}%")
    print()

    # ---- Overall precision ----
    overall_tp = label_counts.get("true_positive", 0)
    overall_total = top_n
    overall_precision = (overall_tp / overall_total * 100) if overall_total > 0 else 0.0
    print(f"Overall estimated precision (top {top_n}): {overall_precision:.1f}%")
    print()

    # ---- Example findings ----
    print("--- Sample True Positive Findings ---")
    tp_examples = [f for f in top_findings if f["validation_label"] == "true_positive"][:5]
    for i, f in enumerate(tp_examples, 1):
        print(f"  {i}. [{f['severity'].upper()}] {f['category']} in {f['_repo']}")
        print(f"     File: {f['file']}:{f['line']}")
        print(f"     Pattern: {f['pattern_id']} - {f['description']}")
        print(f"     Matched: {f['matched_text'][:80]}")
        print(f"     Reason: {f['validation_reason']}")
        print()

    print("--- Sample False Positive Findings ---")
    fp_examples = [f for f in top_findings if f["validation_label"] == "false_positive"][:5]
    for i, f in enumerate(fp_examples, 1):
        print(f"  {i}. [{f['severity'].upper()}] {f['category']} in {f['_repo']}")
        print(f"     File: {f['file']}:{f['line']}")
        print(f"     Pattern: {f['pattern_id']} - {f['description']}")
        print(f"     Matched: {f['matched_text'][:80]}")
        print(f"     Reason: {f['validation_reason']}")
        print()

    # ---- Save results ----
    output = {
        "metadata": {
            "source": CLEANED_PATH,
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "total_findings_in_source": len(all_findings),
            "top_n_validated": top_n,
        },
        "overall": {
            "true_positive": label_counts.get("true_positive", 0),
            "false_positive": label_counts.get("false_positive", 0),
            "uncertain": label_counts.get("uncertain", 0),
            "precision_pct": round(overall_precision, 1),
        },
        "precision_by_severity": precision_by_severity,
        "precision_by_category": precision_by_category,
        "precision_by_repo": {
            k: v for k, v in precision_by_repo.items()
        },
        "validated_findings": [
            {
                "repo": f["_repo"],
                "file": f["file"],
                "line": f["line"],
                "pattern_id": f["pattern_id"],
                "category": f["category"],
                "severity": f["severity"],
                "description": f["description"],
                "matched_text": f["matched_text"][:200],
                "validation_label": f["validation_label"],
                "validation_reason": f["validation_reason"],
                "validation_confidence": f["validation_confidence"],
            }
            for f in top_findings
        ],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved validation results to: {OUTPUT_PATH}")
    print()


if __name__ == "__main__":
    main()
