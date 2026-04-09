#!/usr/bin/env python3
"""Generate 25 responsible disclosure reports for the highest-risk servers
from the 138-server scan. Updates tracking.md and creates disclosure_batch_results.json."""

import json
import os
import re
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

# ── Load data ────────────────────────────────────────────────────────────
with open(os.path.join(PROJECT_DIR, "experiments/full_catalog_scan_results.json")) as f:
    scan_data = json.load(f)

with open(os.path.join(PROJECT_DIR, "data/taxonomy.json")) as f:
    taxonomy = json.load(f)

# ── Build pattern_id -> remediation lookup (suffix-based) ────────────────
remediation_lookup = {}
cwe_lookup = {}
for p in taxonomy["patterns"]:
    remediation_lookup[p["id"]] = p["remediation"]
    cwe_lookup[p["id"]] = p.get("cwe", "")
    # Also index by suffix (e.g., "IV-001" from "S2-IV-001")
    parts = p["id"].split("-", 1)
    if len(parts) == 2:
        suffix = parts[1]
        remediation_lookup[suffix] = p["remediation"]
        cwe_lookup[suffix] = p.get("cwe", "")
    # Three-part suffixes (e.g., "PKE-001" from "S3-PKE-001")
    id_parts = p["id"].split("-")
    if len(id_parts) >= 3:
        suffix2 = "-".join(id_parts[1:])
        remediation_lookup[suffix2] = p["remediation"]
        cwe_lookup[suffix2] = p.get("cwe", "")

def get_remediation(pattern_id):
    return remediation_lookup.get(pattern_id, "Review and apply appropriate security controls for this finding category.")

# ── Sort repos by risk_score, take top 25 ────────────────────────────────
repos = sorted(scan_data["repo_results"], key=lambda r: r.get("risk_score", 0), reverse=True)
top25 = repos[:25]

REPORT_DATE = "2026-04-08"
DEADLINE = "2026-07-07"  # 90 days

reports_dir = os.path.join(BASE_DIR, "reports")
os.makedirs(reports_dir, exist_ok=True)

# ── Helper: slugify repo name ────────────────────────────────────────────
def slugify(full_name):
    return full_name.replace("/", "_").replace(" ", "-")

# ── Generate each report ─────────────────────────────────────────────────
batch_results = []

for rank, repo in enumerate(top25, 1):
    slug = slugify(repo["full_name"])
    total_findings = repo["total_findings"]
    risk_score = repo["risk_score"]
    risk_rating = repo["risk_rating"]
    protocol = repo["detected_protocol"]
    full_name = repo["full_name"]
    stars = repo.get("stars", 0)
    language = repo.get("language", "Unknown")
    url = repo.get("url", f"https://github.com/{full_name}")

    findings = repo.get("findings", [])

    # Separate critical/high findings
    critical_high = [f for f in findings if f["severity"] in ("critical", "high")]
    # Sort: critical first, then high, then by line
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    critical_high.sort(key=lambda f: (severity_order.get(f["severity"], 9), f.get("line", 0)))

    n_critical = repo["by_severity"].get("critical", 0)
    n_high = repo["by_severity"].get("high", 0)
    n_medium = repo["by_severity"].get("medium", 0)

    # ── Build the report ──────────────────────────────────────────────
    lines = []
    lines.append(f"# Security Disclosure: {full_name}\n")
    lines.append(f"**Date:** {REPORT_DATE}")
    lines.append(f"**Reporter:** NSF AI Agent Security Research Team")
    lines.append(f"**Severity:** {risk_rating} (score: {risk_score})")
    lines.append(f"**Protocol:** {protocol}")
    lines.append(f"**Repository:** [{full_name}]({url})")
    lines.append(f"**Language:** {language}")
    lines.append(f"**Stars:** {stars}")
    lines.append("")

    lines.append("## Summary")
    lines.append(f"We identified {total_findings} potential security issues in {full_name}")
    lines.append(f"through automated static analysis of AI agent tool interfaces.")
    lines.append(f"This analysis was part of a 138-repository systematic study across four")
    lines.append(f"protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).")
    lines.append("")

    lines.append("### Severity Breakdown")
    lines.append("")
    lines.append("| Severity | Count |")
    lines.append("|----------|-------|")
    if n_critical > 0:
        lines.append(f"| Critical | {n_critical} |")
    if n_high > 0:
        lines.append(f"| High | {n_high} |")
    if n_medium > 0:
        lines.append(f"| Medium | {n_medium} |")
    lines.append("")

    # ── Critical / High Severity Findings ─────────────────────────────
    if critical_high:
        lines.append("## Critical / High Severity Findings")
        lines.append("")

        # Cap at 20 findings to keep reports manageable
        shown = critical_high[:20]
        for idx, finding in enumerate(shown, 1):
            cat = finding["category"]
            sev = finding["severity"].upper()
            pid = finding["pattern_id"]
            cwe = finding.get("cwe", cwe_lookup.get(pid, ""))
            fpath = finding.get("file", "unknown")
            fline = finding.get("line", 0)
            desc = finding.get("description", "")
            matched = finding.get("matched_text", "")
            # Truncate matched text
            if len(matched) > 100:
                matched = matched[:100] + "..."
            remediation = get_remediation(pid)

            lines.append(f"### Finding {idx}: {cat} ({sev})")
            lines.append(f"- **Pattern:** {pid}")
            lines.append(f"- **CWE:** {cwe}")
            lines.append(f"- **File:** {fpath}:{fline}")
            lines.append(f"- **Description:** {desc}")
            lines.append(f"- **Matched:** `{matched}`")
            lines.append(f"- **Remediation:** {remediation}")
            lines.append("")

        if len(critical_high) > 20:
            lines.append(f"*... and {len(critical_high) - 20} additional critical/high findings not shown.*")
            lines.append("")
    else:
        lines.append("## Critical / High Severity Findings")
        lines.append("")
        lines.append("No critical or high severity findings were identified; however,")
        lines.append(f"the repository has {n_medium} medium-severity findings contributing to the overall risk score.")
        lines.append("")

    # ── Category Breakdown ────────────────────────────────────────────
    lines.append("## Findings by Category")
    lines.append("")
    lines.append("| Category | Count |")
    lines.append("|----------|-------|")
    for cat, count in sorted(repo.get("by_category", {}).items(), key=lambda x: -x[1]):
        lines.append(f"| {cat.replace('_', ' ').title()} | {count} |")
    lines.append("")

    # ── Recommendations ───────────────────────────────────────────────
    lines.append("## Recommendations")
    lines.append("")
    # Generate unique recommendations based on the categories found
    seen_cats = set()
    rec_num = 1
    for finding in critical_high[:20]:
        cat = finding["category"]
        if cat not in seen_cats:
            seen_cats.add(cat)
            rem = get_remediation(finding["pattern_id"])
            lines.append(f"{rec_num}. **{cat.replace('_', ' ').title()}:** {rem}")
            rec_num += 1
    # If no critical/high, pull from all findings
    if not seen_cats:
        for finding in findings[:10]:
            cat = finding["category"]
            if cat not in seen_cats:
                seen_cats.add(cat)
                rem = get_remediation(finding["pattern_id"])
                lines.append(f"{rec_num}. **{cat.replace('_', ' ').title()}:** {rem}")
                rec_num += 1
    lines.append("")

    # ── Disclosure Timeline ───────────────────────────────────────────
    lines.append("## Disclosure Timeline")
    lines.append(f"- {REPORT_DATE}: Report prepared")
    lines.append(f"- {REPORT_DATE}: Report sent to maintainer (pending)")
    lines.append(f"- {DEADLINE}: 90-day disclosure deadline")
    lines.append("")

    # ── About / Contact ───────────────────────────────────────────────
    lines.append("## About This Research")
    lines.append("")
    lines.append("This work is part of an NSF-funded academic research project studying the security of")
    lines.append("AI agent tool interfaces across protocol families (MCP, OpenAI Function Calling,")
    lines.append("LangChain, Web3-native modules). The study analyzed 138 repositories using")
    lines.append("static pattern analysis based on a 27-pattern vulnerability taxonomy.")
    lines.append("")
    lines.append("## Contact")
    lines.append("")
    lines.append("For questions about this report, please contact the NSF AI Agent Security Research Team.")
    lines.append("We are happy to work with you on remediation and will adjust the disclosure")
    lines.append("timeline if needed to allow adequate time for fixes.")

    report_text = "\n".join(lines) + "\n"

    # Write report
    report_path = os.path.join(reports_dir, f"{slug}.md")
    with open(report_path, "w") as f:
        f.write(report_text)

    # Record for batch results
    batch_results.append({
        "rank": rank,
        "repo": full_name,
        "slug": slug,
        "url": url,
        "protocol": protocol,
        "language": language,
        "stars": stars,
        "risk_score": risk_score,
        "risk_rating": risk_rating,
        "total_findings": total_findings,
        "critical": n_critical,
        "high": n_high,
        "medium": n_medium,
        "report_file": f"reports/{slug}.md",
        "report_date": REPORT_DATE,
        "deadline": DEADLINE,
        "status": "prepared"
    })

    print(f"[{rank:2d}] {full_name:55s} score={risk_score:5.1f} findings={total_findings:4d} crit={n_critical} high={n_high}")

# ── Update tracking.md ────────────────────────────────────────────────────
tracking_lines = []
tracking_lines.append("# Vulnerability Disclosure Tracking")
tracking_lines.append("")
tracking_lines.append(f"Generated: {REPORT_DATE}")
tracking_lines.append(f"Public disclosure deadline: {DEADLINE}")
tracking_lines.append(f"Total repositories scanned: 138")
tracking_lines.append(f"Reports generated: {len(batch_results)}")
tracking_lines.append("")
tracking_lines.append("| # | Repo | Protocol | Risk Score | Findings | Critical | High | Status | Report Date | 90-Day Deadline |")
tracking_lines.append("|---|------|----------|-----------|----------|----------|------|--------|-------------|-----------------|")

for r in batch_results:
    tracking_lines.append(
        f"| {r['rank']} | [{r['repo']}]({r['url']}) | {r['protocol']} | {r['risk_score']} | {r['total_findings']} | {r['critical']} | {r['high']} | Pending | {r['report_date']} | {r['deadline']} |"
    )

tracking_lines.append("")
tracking_lines.append("## Status Legend")
tracking_lines.append("")
tracking_lines.append("- **Status:** Pending / Sent / Acknowledged / Working on fix / Disputed / No response / Fixed")
tracking_lines.append("- **Report links:** See `disclosure/reports/{repo_slug}.md`")
tracking_lines.append("")

tracking_path = os.path.join(BASE_DIR, "tracking.md")
with open(tracking_path, "w") as f:
    f.write("\n".join(tracking_lines) + "\n")

print(f"\nTracking updated: {tracking_path}")

# ── Save batch results JSON ───────────────────────────────────────────────
batch_json = {
    "generated": REPORT_DATE,
    "deadline": DEADLINE,
    "total_scanned": 138,
    "reports_generated": len(batch_results),
    "reports": batch_results
}

batch_path = os.path.join(BASE_DIR, "disclosure_batch_results.json")
with open(batch_path, "w") as f:
    json.dump(batch_json, f, indent=2)

print(f"Batch results saved: {batch_path}")
print(f"\nDone. Generated {len(batch_results)} reports.")
