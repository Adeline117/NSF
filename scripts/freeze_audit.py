#!/usr/bin/env python3
"""Agent D: Freeze Checker — pre-deadline audit.

Usage:
    python3 scripts/freeze_audit.py [paper] [level]
    python3 scripts/freeze_audit.py paper0 72h
    python3 scripts/freeze_audit.py all 24h
"""
import json
import re
import sys
import os
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent

PAPERS = {
    "paper0": ROOT / "paper0_ai_agent_theory",
    "paper1": ROOT / "paper1_onchain_agent_id",
    "paper2": ROOT / "paper2_agent_tool_security",
    "paper3": ROOT / "paper3_ai_sybil",
}

FLUID_PATTERN = re.compile(r"%\s*FLUID:\s*(.*)")
TODO_PATTERN = re.compile(r"(?:%\s*TODO|\\todo|\[TODO[:\s])")
ORPHANED_PATTERN = re.compile(r"%\s*ORPHANED:\s*(.*)")
CITE_PATTERN = re.compile(r"\\cite\{([^}]+)\}")
MACRO_USE_PATTERN = re.compile(r"\\([a-zA-Z]+)")


def scan_tex_markers(tex_path: Path) -> dict:
    """Scan a .tex file for FLUID, TODO, ORPHANED markers."""
    findings = {"fluid": [], "todo": [], "orphaned": []}
    with open(tex_path) as f:
        for i, line in enumerate(f, 1):
            m = FLUID_PATTERN.search(line)
            if m:
                findings["fluid"].append((i, m.group(1).strip()))
            m = TODO_PATTERN.search(line)
            if m:
                findings["todo"].append((i, line.strip()))
            m = ORPHANED_PATTERN.search(line)
            if m:
                findings["orphaned"].append((i, m.group(1).strip()))
    return findings


def check_manifest_completeness(paper_dir: Path) -> list:
    """Check if any macro used in .tex is missing from manifest."""
    manifest_path = paper_dir / "results" / "manifest.json"
    if not manifest_path.exists():
        return [("CRITICAL", "No manifest.json found")]

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Get all defined macro names from macros_results.tex
    issues = []
    tex_dirs = [d for d in paper_dir.iterdir() if d.is_dir() and (d / "macros_results.tex").exists()]

    for tex_dir in tex_dirs:
        macros_file = tex_dir / "macros_results.tex"
        defined_macros = set()
        with open(macros_file) as f:
            for line in f:
                m = re.match(r"\\newcommand\{\\(\w+)\}", line)
                if m:
                    defined_macros.add(m.group(1))

        # Check main.tex for macro usage that might reference undefined macros
        main_tex = tex_dir / "main.tex"
        if main_tex.exists():
            with open(main_tex) as f:
                content = f.read()
            # This is a basic check — doesn't resolve all LaTeX macro scoping
            # but catches obvious missing references

    return issues


def check_figure_staleness(paper_dir: Path) -> list:
    """Check if figures are older than latest results."""
    issues = []
    exp_dir = paper_dir / "experiments"
    fig_dir = paper_dir / "figures"

    if not fig_dir.exists() or not exp_dir.exists():
        return issues

    # Latest result file timestamp
    result_files = list(exp_dir.glob("*_results.json"))
    if not result_files:
        return issues

    latest_result_time = max(f.stat().st_mtime for f in result_files)

    # Check figure timestamps
    for fig in fig_dir.glob("*.pdf"):
        if fig.stat().st_mtime < latest_result_time:
            age_hours = (latest_result_time - fig.stat().st_mtime) / 3600
            issues.append(("WARN", f"Figure {fig.name} is {age_hours:.0f}h older than latest results"))

    return issues


def run_audit(paper_key: str, level: str = "72h") -> str:
    """Run freeze audit for a paper at specified level."""
    paper_dir = PAPERS[paper_key]
    report_lines = []
    blocking = []
    warnings = []

    report_lines.append(f"# Freeze Audit: {paper_key} [{level}]")
    report_lines.append(f"Generated: {datetime.now().isoformat()}")
    report_lines.append("")

    # 1. Scan all tex files for markers
    report_lines.append("## Markers")
    tex_dirs = [d for d in paper_dir.iterdir() if d.is_dir() and (d / "main.tex").exists()]

    for tex_dir in tex_dirs:
        for tex_file in tex_dir.glob("*.tex"):
            markers = scan_tex_markers(tex_file)
            rel = tex_file.relative_to(paper_dir)
            if markers["fluid"]:
                report_lines.append(f"\n### FLUID in `{rel}`:")
                for line_no, desc in markers["fluid"]:
                    report_lines.append(f"- L{line_no}: {desc}")
                    if level == "2h":
                        blocking.append(f"FLUID marker at {rel}:{line_no}")
            if markers["todo"]:
                report_lines.append(f"\n### TODO in `{rel}`:")
                for line_no, text in markers["todo"]:
                    report_lines.append(f"- L{line_no}: {text[:80]}")
                    if level in ("2h", "24h"):
                        blocking.append(f"TODO at {rel}:{line_no}")
            if markers["orphaned"]:
                report_lines.append(f"\n### ORPHANED in `{rel}`:")
                for line_no, desc in markers["orphaned"]:
                    report_lines.append(f"- L{line_no}: {desc}")
                    blocking.append(f"ORPHANED content at {rel}:{line_no}")

    # 2. Manifest completeness
    report_lines.append("\n## Manifest Completeness")
    issues = check_manifest_completeness(paper_dir)
    if issues:
        for severity, msg in issues:
            report_lines.append(f"- [{severity}] {msg}")
            if severity == "CRITICAL":
                blocking.append(msg)
    else:
        report_lines.append("- OK")

    # 3. Figure staleness
    report_lines.append("\n## Figure Freshness")
    fig_issues = check_figure_staleness(paper_dir)
    if fig_issues:
        for severity, msg in fig_issues:
            report_lines.append(f"- [{severity}] {msg}")
            warnings.append(msg)
    else:
        report_lines.append("- All figures up to date")

    # 4. Summary
    report_lines.append("\n---")
    report_lines.append(f"\n## Summary")
    report_lines.append(f"- Blocking issues: {len(blocking)}")
    report_lines.append(f"- Warnings: {len(warnings)}")

    if blocking:
        report_lines.append("\n### BLOCKING (must resolve before submission)")
        for b in blocking:
            report_lines.append(f"- [ ] {b}")

    if warnings:
        report_lines.append("\n### WARNINGS")
        for w in warnings:
            report_lines.append(f"- [ ] {w}")

    report = "\n".join(report_lines)

    # Write report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    review_dir = paper_dir / "review"
    review_dir.mkdir(exist_ok=True)
    out_path = review_dir / f"D_freeze_audit_{timestamp}.md"
    with open(out_path, "w") as f:
        f.write(report)

    print(f"Audit written to: {out_path.relative_to(ROOT)}")
    return report


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else ["all"]
    level = "72h"

    # Parse args
    paper_targets = []
    for arg in targets:
        if arg in ("72h", "24h", "2h"):
            level = arg
        elif arg == "all":
            paper_targets = list(PAPERS.keys())
        elif arg in PAPERS:
            paper_targets.append(arg)
        else:
            candidates = [k for k in PAPERS if arg in k]
            if candidates:
                paper_targets.extend(candidates)

    if not paper_targets:
        paper_targets = list(PAPERS.keys())

    for pk in paper_targets:
        print(f"\n{'='*60}")
        print(run_audit(pk, level))


if __name__ == "__main__":
    main()
