#!/usr/bin/env python3
"""
Generate per-repo responsible disclosure reports for Web3 agent tool
security vulnerabilities discovered during the NSF research project.

Reads unified_web3_results.json and dynamic_test_results.json, then
produces:
  - Per-repo markdown disclosure reports in disclosure/reports/
  - A summary CSV at disclosure/disclosure_summary.csv
  - An updated tracking sheet at disclosure/tracking.md

Usage:
    python generate_reports.py
"""

import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent

UNIFIED_RESULTS = PROJECT_DIR / "experiments" / "unified_web3_results.json"
DYNAMIC_RESULTS = PROJECT_DIR / "dynamic_testing" / "dynamic_test_results.json"
REPORTS_DIR = SCRIPT_DIR / "reports"
SUMMARY_CSV = SCRIPT_DIR / "disclosure_summary.csv"
TRACKING_MD = SCRIPT_DIR / "tracking.md"

# ---------------------------------------------------------------------------
# Remediation recommendations keyed by vulnerability category
# ---------------------------------------------------------------------------
REMEDIATION = {
    "private_key_exposure": (
        "Never accept private keys as tool parameters. Use session keys or "
        "hardware wallet signing. If the server must sign, use a secure "
        "enclave or environment-variable-only key loading with strict "
        "file permissions."
    ),
    "unlimited_approval": (
        "Cap approval amounts to the exact needed value. Never use "
        "MaxUint256 (type(uint256).max) as an approval amount. Implement "
        "increaseAllowance/decreaseAllowance patterns instead."
    ),
    "missing_input_validation": (
        "Validate Ethereum addresses with isAddress(). Validate amounts "
        "are positive and within expected bounds. Reject unexpected or "
        "extra parameters by setting additionalProperties: false in JSON "
        "Schema definitions."
    ),
    "tool_poisoning": (
        "Keep tool descriptions factual. Never include instructions for "
        "the LLM in tool descriptions or metadata. Avoid directive "
        "language (e.g., 'always', 'you must', 'ignore previous')."
    ),
    "prompt_injection": (
        "Never interpolate user input into tool descriptions or system "
        "prompts. Sanitize all dynamic content before inclusion in LLM "
        "context. Use parameterized tool interfaces."
    ),
    "cross_tool_escalation": (
        "Implement per-tool permission checks. Do not allow one tool to "
        "invoke another without explicit authorization. Use capability-"
        "based access control for sensitive operations."
    ),
    "hardcoded_credentials": (
        "Remove all hardcoded API keys and secrets. Use environment "
        "variables or a secrets manager. Add credential files to "
        ".gitignore."
    ),
    "env_key_exposure": (
        "Do not log or display environment variable values. Mask "
        "sensitive environment variables in error messages and debug "
        "output."
    ),
    "state_confusion": (
        "Isolate per-session state. Do not share mutable state across "
        "tool invocations from different users or sessions. Use request-"
        "scoped context objects."
    ),
    "excessive_permissions": (
        "Apply the principle of least privilege. Only request the "
        "permissions each tool actually needs. Separate read-only and "
        "write tools into distinct permission scopes."
    ),
    "tx_validation_missing": (
        "Validate all transaction parameters (to, value, data, gasLimit) "
        "before signing. Require user confirmation for transactions above "
        "a configurable threshold."
    ),
    "no_gas_limit": (
        "Always specify a gas limit for transactions. Use gas estimation "
        "with a safety margin rather than omitting the limit entirely."
    ),
    "no_slippage_protection": (
        "Implement configurable slippage tolerance (e.g., 0.5-1%). "
        "Calculate minimum output amounts and set deadlines for DEX "
        "swaps."
    ),
    "mev_exposure": (
        "Use private transaction pools (e.g., Flashbots Protect) for "
        "sensitive transactions. Implement commit-reveal schemes where "
        "applicable."
    ),
    "missing_harness": (
        "Add integration tests that exercise tool interfaces with "
        "adversarial inputs. Include negative test cases for boundary "
        "conditions."
    ),
}

# Severity ordering for sorting
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "none": 4}


def load_json(path: Path) -> dict:
    """Load a JSON file, returning empty dict on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: could not load {path}: {e}")
        return {}


def safe_filename(repo_name: str) -> str:
    """Convert repo full_name to a safe filename."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", repo_name)


def get_github_issues_url(repo_name: str) -> str:
    """Build the GitHub issues URL for a repo."""
    return f"https://github.com/{repo_name}/issues"


def collect_repo_findings(web3_data: dict) -> dict:
    """
    Build a dict keyed by repo full_name with:
      - risk_score, risk_rating, protocol, stars
      - findings list (from static scan repo_results)
      - by_severity / by_category summaries
    """
    repos = {}

    # -- From risk_ranking in the static_scan_summary --
    for entry in web3_data.get("static_scan_summary", {}).get("risk_ranking", []):
        name = entry["repo"]
        repos[name] = {
            "risk_score": entry.get("risk_score", 0),
            "risk_rating": entry.get("risk_rating", "unknown"),
            "protocol": entry.get("protocol", "unknown"),
            "stars": entry.get("stars", 0),
            "total_findings": entry.get("findings", 0),
            "tools": entry.get("tools", 0),
            "sensitive_tools": entry.get("sensitive_tools", 0),
            "findings": [],
            "by_severity": {},
            "by_category": {},
        }

    # -- Merge detailed findings from web3_scan_results repo_results --
    # The unified file embeds summaries; the raw web3_scan_results has
    # individual findings. We try the raw file first.
    raw_web3 = load_json(PROJECT_DIR / "experiments" / "web3_scan_results.json")
    for repo_entry in raw_web3.get("repo_results", []):
        name = repo_entry.get("full_name", "")
        if name not in repos:
            repos[name] = {
                "risk_score": repo_entry.get("risk_score", 0),
                "risk_rating": repo_entry.get("risk_rating", "unknown"),
                "protocol": repo_entry.get("detected_protocol", "unknown"),
                "stars": repo_entry.get("stars", 0),
                "total_findings": repo_entry.get("total_findings", 0),
                "tools": 0,
                "sensitive_tools": 0,
                "findings": [],
                "by_severity": {},
                "by_category": {},
            }
        repos[name]["findings"] = repo_entry.get("findings", [])
        repos[name]["by_severity"] = repo_entry.get("by_severity", {})
        repos[name]["by_category"] = repo_entry.get("by_category", {})

    return repos


def collect_dynamic_findings(dynamic_data: dict) -> dict:
    """
    Build a dict keyed by repo name with dynamic attack results.
    """
    repos = {}
    for repo_entry in dynamic_data.get("repo_results", []):
        name = repo_entry.get("repo", "")
        vulns = [r for r in repo_entry.get("attack_results", []) if r.get("vulnerable")]
        repos[name] = {
            "url": repo_entry.get("url", ""),
            "stars": repo_entry.get("stars", 0),
            "tools_tested": repo_entry.get("tools_tested", 0),
            "total_vulnerabilities": repo_entry.get("vulnerabilities", 0),
            "overall_score": repo_entry.get("overall_score", 0),
            "dynamic_vulns": vulns,
        }
    return repos


def severity_sort_key(finding: dict) -> int:
    sev = finding.get("severity", "none").lower()
    return SEVERITY_ORDER.get(sev, 99)


def generate_repo_report(
    repo_name: str,
    static_info: dict,
    dynamic_info: dict | None,
    report_date: str,
    disclosure_deadline: str,
) -> str:
    """Generate a Markdown disclosure report for a single repo."""

    # Merge counts
    critical_count = static_info.get("by_severity", {}).get("critical", 0)
    high_count = static_info.get("by_severity", {}).get("high", 0)
    medium_count = static_info.get("by_severity", {}).get("medium", 0)
    total = static_info.get("total_findings", 0)

    # Dynamic additions
    dyn_total = 0
    if dynamic_info:
        dyn_total = dynamic_info.get("total_vulnerabilities", 0)

    combined = total + dyn_total

    lines = [
        "# Security Vulnerability Disclosure Report",
        "",
        f"**From:** NSF Research Project -- AI Agent Tool Interface Security",
        f"**Date:** {report_date}",
        f"**Subject:** Security vulnerabilities found in {repo_name}",
        "",
        "## Summary",
        "",
        f"During our academic research on AI agent tool interface security, we "
        f"identified **{combined}** potential security vulnerabilities in your "
        f"**{repo_name}** project ({total} from static analysis"
        + (f", {dyn_total} from dynamic testing" if dyn_total else "")
        + ").",
        "",
        f"- **Risk score:** {static_info.get('risk_score', 'N/A')}/100",
        f"- **Risk rating:** {static_info.get('risk_rating', 'N/A')}",
        f"- **Protocol:** {static_info.get('protocol', 'N/A')}",
        f"- **GitHub stars:** {static_info.get('stars', 'N/A')}",
        "",
    ]

    # --- Severity breakdown ---
    lines.append("## Severity Breakdown")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    for sev in ["critical", "high", "medium", "low"]:
        count = static_info.get("by_severity", {}).get(sev, 0)
        if count:
            lines.append(f"| {sev.capitalize()} | {count} |")
    lines.append("")

    # --- Category breakdown ---
    cats = static_info.get("by_category", {})
    if cats:
        lines.append("## Vulnerability Categories")
        lines.append("")
        lines.append("| Category | Count |")
        lines.append("|----------|-------|")
        for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
            lines.append(f"| {cat.replace('_', ' ').title()} | {count} |")
        lines.append("")

    # --- Top 5 most severe findings (static) ---
    findings = sorted(static_info.get("findings", []), key=severity_sort_key)
    top_findings = findings[:5]

    if top_findings:
        lines.append("## Top Findings (Most Severe)")
        lines.append("")

        for i, f in enumerate(top_findings, 1):
            vuln_id = f.get("pattern_id", f"VULN-{i:03d}")
            category = f.get("category", "unknown")
            severity = f.get("severity", "unknown").upper()
            lines.append(f"### {vuln_id}: {category.replace('_', ' ').title()} ({severity})")
            lines.append(f"- **File:** {f.get('file', 'N/A')}")
            lines.append(f"- **Line:** {f.get('line', 'N/A')}")
            lines.append(f"- **Description:** {f.get('description', 'N/A')}")
            lines.append(f"- **CWE:** {f.get('cwe', 'N/A')}")
            if f.get("context"):
                lines.append(f"- **Code context:**")
                lines.append("  ```")
                for ctx_line in f["context"].split("\n"):
                    lines.append(f"  {ctx_line}")
                lines.append("  ```")
            impact = _impact_for_category(category)
            lines.append(f"- **Impact:** {impact}")
            rec = REMEDIATION.get(category, "Review and apply security best practices.")
            lines.append(f"- **Recommendation:** {rec}")
            lines.append("")

    # --- Dynamic testing findings ---
    if dynamic_info and dynamic_info.get("dynamic_vulns"):
        lines.append("## Dynamic Testing Findings")
        lines.append("")
        dyn_vulns = sorted(dynamic_info["dynamic_vulns"], key=severity_sort_key)
        # Deduplicate by (attack_vector, severity) and show top entries
        seen = set()
        shown = 0
        for v in dyn_vulns:
            key = (v.get("attack_vector"), v.get("tool_name"), v.get("severity"))
            if key in seen:
                continue
            seen.add(key)
            if shown >= 5:
                break
            shown += 1
            lines.append(
                f"### DYN-{shown:03d}: {v.get('attack_vector', 'N/A').replace('_', ' ').title()} "
                f"({v.get('severity', 'N/A').upper()})"
            )
            lines.append(f"- **Tool:** {v.get('tool_name', 'N/A')}")
            lines.append(f"- **Details:** {v.get('details', 'N/A')}")
            lines.append(f"- **CWE:** {v.get('cwe', 'N/A')}")
            rec = REMEDIATION.get(
                _vector_to_category(v.get("attack_vector", "")),
                "Review and apply security best practices.",
            )
            lines.append(f"- **Recommendation:** {rec}")
            lines.append("")

    # --- Remediation summary ---
    all_categories = set(cats.keys())
    if dynamic_info:
        for v in dynamic_info.get("dynamic_vulns", []):
            mapped = _vector_to_category(v.get("attack_vector", ""))
            if mapped:
                all_categories.add(mapped)

    if all_categories:
        lines.append("## Remediation Recommendations")
        lines.append("")
        for cat in sorted(all_categories):
            rec = REMEDIATION.get(cat)
            if rec:
                lines.append(f"### {cat.replace('_', ' ').title()}")
                lines.append(f"{rec}")
                lines.append("")

    # --- Disclosure timeline ---
    lines.append("## Disclosure Timeline")
    lines.append("")
    lines.append(f"- **2026-04-07:** Vulnerabilities discovered during automated scanning")
    lines.append(f"- **{report_date}:** This report sent to maintainers")
    lines.append(f"- **{disclosure_deadline}:** Planned public disclosure (per responsible disclosure standard, 90 days)")
    lines.append("")

    # --- About ---
    lines.append("## About This Research")
    lines.append("")
    lines.append(
        "This work is part of an academic research project studying the security of "
        "AI agent tool interfaces across protocol families (MCP, OpenAI Function Calling, "
        "LangChain, Web3-native modules). The study analyzed 43 Web3-related repositories "
        "using both static pattern analysis and dynamic tool-interface testing."
    )
    lines.append("")
    lines.append("Paper submitted to [VENUE].")
    lines.append("")

    # --- Contact ---
    lines.append("## Contact")
    lines.append("")
    lines.append("For questions about this report, please contact:")
    lines.append("- [Your Name] <[email]>")
    lines.append("- [Advisor Name] <[email]>")
    lines.append("")
    lines.append(
        "We are happy to work with you on remediation and will update the planned "
        "public disclosure timeline if needed to allow adequate time for fixes."
    )
    lines.append("")

    return "\n".join(lines)


def _impact_for_category(category: str) -> str:
    """Return a human-readable impact statement for a category."""
    impacts = {
        "private_key_exposure": (
            "An attacker could steal private keys, gaining full control of "
            "associated wallets and funds."
        ),
        "unlimited_approval": (
            "Unlimited token approvals allow a compromised or malicious "
            "contract to drain all approved tokens from the user's wallet."
        ),
        "missing_input_validation": (
            "Unvalidated inputs could lead to injection attacks, unexpected "
            "behavior, or interaction with unintended contracts."
        ),
        "tool_poisoning": (
            "Malicious instructions in tool descriptions could manipulate "
            "the LLM into performing unintended actions on behalf of the user."
        ),
        "prompt_injection": (
            "User-controlled input interpolated into prompts could hijack "
            "the LLM's behavior, bypassing safety checks."
        ),
        "cross_tool_escalation": (
            "One tool invoking another without permission checks could "
            "allow privilege escalation or unauthorized actions."
        ),
        "hardcoded_credentials": (
            "Hardcoded secrets in source code could be extracted by anyone "
            "with repository access, compromising associated services."
        ),
        "env_key_exposure": (
            "Leaking environment variable values could expose API keys "
            "and secrets in logs or error messages."
        ),
        "state_confusion": (
            "Shared mutable state across sessions could allow one user "
            "to influence or read another user's operations."
        ),
        "excessive_permissions": (
            "Tools with excessive permissions increase the blast radius "
            "if any single tool is compromised."
        ),
        "tx_validation_missing": (
            "Missing transaction validation could allow crafted "
            "transactions to drain funds or interact with malicious contracts."
        ),
        "no_gas_limit": (
            "Missing gas limits could lead to unexpectedly expensive "
            "transactions or denial-of-service via gas exhaustion."
        ),
        "no_slippage_protection": (
            "Without slippage protection, DEX trades could suffer "
            "significant value loss through sandwich attacks."
        ),
        "mev_exposure": (
            "Transactions visible in the public mempool are vulnerable "
            "to front-running and MEV extraction."
        ),
        "missing_harness": (
            "Without proper test coverage, vulnerabilities may go "
            "undetected and regressions may be introduced."
        ),
    }
    return impacts.get(category, "Could lead to security compromise if exploited.")


def _vector_to_category(vector: str) -> str:
    """Map a dynamic attack vector name to the closest static category."""
    mapping = {
        "tool_poisoning": "tool_poisoning",
        "prompt_injection_output": "prompt_injection",
        "parameter_injection": "missing_input_validation",
        "tx_validation": "tx_validation_missing",
        "private_key_handling": "private_key_exposure",
    }
    return mapping.get(vector, "")


def main():
    print("Loading scan results...")
    unified = load_json(UNIFIED_RESULTS)
    dynamic = load_json(DYNAMIC_RESULTS)

    if not unified:
        print("ERROR: Could not load unified results. Aborting.")
        return

    # Collect data
    static_repos = collect_repo_findings(unified)
    dynamic_repos = collect_dynamic_findings(dynamic)

    # Dates
    report_date = "2026-04-07"
    deadline_date = (datetime(2026, 4, 7) + timedelta(days=90)).strftime("%Y-%m-%d")

    # Ensure output directory exists
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # Filter repos: only CRITICAL or HIGH risk_rating
    target_repos = {
        name: info
        for name, info in static_repos.items()
        if info.get("risk_rating", "").lower() in ("critical", "high")
    }

    print(f"Found {len(target_repos)} repos with CRITICAL or HIGH risk rating.")

    # Generate per-repo reports
    csv_rows = []
    tracking_rows = []

    for repo_name in sorted(target_repos.keys()):
        static_info = target_repos[repo_name]
        dynamic_info = dynamic_repos.get(repo_name)

        report = generate_repo_report(
            repo_name, static_info, dynamic_info, report_date, deadline_date
        )

        # Write report
        filename = safe_filename(repo_name) + ".md"
        report_path = REPORTS_DIR / filename
        with open(report_path, "w") as f:
            f.write(report)
        print(f"  Generated: {report_path.name}")

        # CSV row
        critical_count = static_info.get("by_severity", {}).get("critical", 0)
        high_count = static_info.get("by_severity", {}).get("high", 0)
        csv_rows.append({
            "repo_name": repo_name,
            "risk_score": static_info.get("risk_score", 0),
            "risk_rating": static_info.get("risk_rating", ""),
            "total_findings": static_info.get("total_findings", 0),
            "critical_count": critical_count,
            "high_count": high_count,
            "dynamic_vulns": dynamic_info.get("total_vulnerabilities", 0) if dynamic_info else 0,
            "maintainer_contact": get_github_issues_url(repo_name),
        })

        # Tracking row
        tracking_rows.append({
            "repo": repo_name,
            "critical": critical_count,
            "high": high_count,
            "risk_score": static_info.get("risk_score", 0),
        })

    # Write summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "repo_name", "risk_score", "risk_rating", "total_findings",
            "critical_count", "high_count", "dynamic_vulns", "maintainer_contact",
        ])
        writer.writeheader()
        for row in sorted(csv_rows, key=lambda r: -r["risk_score"]):
            writer.writerow(row)
    print(f"\nSummary CSV: {SUMMARY_CSV}")

    # Write tracking sheet
    tracking_lines = [
        "# Vulnerability Disclosure Tracking",
        "",
        f"Generated: {report_date}",
        f"Public disclosure deadline: {deadline_date}",
        "",
        "| Repo | Risk Score | Critical | High | Disclosed | Response | Fixed |",
        "|------|-----------|----------|------|-----------|----------|-------|",
    ]
    for row in sorted(tracking_rows, key=lambda r: -r["risk_score"]):
        tracking_lines.append(
            f"| {row['repo']} | {row['risk_score']} | {row['critical']} | "
            f"{row['high']} | Pending | - | - |"
        )
    tracking_lines.append("")
    tracking_lines.append("## Status Legend")
    tracking_lines.append("")
    tracking_lines.append("- **Disclosed:** Pending / Sent / N/A")
    tracking_lines.append("- **Response:** - / Acknowledged / Working on fix / Disputed / No response")
    tracking_lines.append("- **Fixed:** - / Partial / Full / Won't fix")
    tracking_lines.append("")

    with open(TRACKING_MD, "w") as f:
        f.write("\n".join(tracking_lines))
    print(f"Tracking sheet: {TRACKING_MD}")

    print(f"\nDone. {len(target_repos)} disclosure reports generated.")


if __name__ == "__main__":
    main()
