"""
Paper 2: Export Unified Vulnerability Taxonomy
================================================
Reads VULN_PATTERNS from static_analysis/analyzer.py and emits:
  - data/taxonomy.json  (machine-readable)
  - paper/taxonomy.md   (human-readable Appendix A)

Schema per pattern entry:
  {
    "id": "S1-TP-001",
    "category": "tool_poisoning",
    "attack_surface": "S1_tool_definition",
    "severity": "high",
    "protocols": ["mcp", "openai", "langchain"],
    "cwe": "CWE-913",
    "owasp_llm": "LLM01",
    "description": "...",
    "remediation": "...",
    "file_extensions": [".ts", ".js", ...]
  }
"""

import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper2_agent_tool_security.static_analysis.analyzer import VULN_PATTERNS

OUTPUT_JSON = PROJECT_ROOT / "paper2_agent_tool_security" / "data" / "taxonomy.json"
OUTPUT_MD = PROJECT_ROOT / "paper2_agent_tool_security" / "paper" / "taxonomy.md"


def pattern_to_dict(p) -> dict:
    d = dataclasses.asdict(p)
    # Enums → string
    d["severity"] = p.severity.value
    d["attack_surface"] = p.attack_surface.value
    d["protocols"] = [proto.value for proto in p.protocols]
    # Drop regex for the public taxonomy (keep in code)
    d.pop("pattern", None)
    d.pop("false_positive_hints", None)
    return d


def main():
    print(f"Loading {len(VULN_PATTERNS)} VulnPatterns from analyzer.py")

    taxonomy = [pattern_to_dict(p) for p in VULN_PATTERNS]

    # Group by attack surface and category for the MD table
    by_surface: dict[str, list] = defaultdict(list)
    for p in taxonomy:
        by_surface[p["attack_surface"]].append(p)

    # Count distinct CWEs and categories
    cwes = sorted({p["cwe"] for p in taxonomy if p["cwe"]})
    categories = sorted({p["category"] for p in taxonomy})
    surfaces = sorted(by_surface.keys())
    owasp_llms = sorted({p["owasp_llm"] for p in taxonomy if p["owasp_llm"]})

    summary = {
        "n_patterns": len(taxonomy),
        "n_categories": len(categories),
        "n_cwes": len(cwes),
        "n_attack_surfaces": len(surfaces),
        "n_owasp_llm": len(owasp_llms),
        "severity_distribution": {
            sev: sum(1 for p in taxonomy if p["severity"] == sev)
            for sev in ["critical", "high", "medium", "low"]
        },
        "categories": categories,
        "cwes": cwes,
        "attack_surfaces": surfaces,
        "owasp_llm_categories": owasp_llms,
        "patterns": taxonomy,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved JSON to {OUTPUT_JSON}")

    # Markdown Appendix
    lines = []
    lines.append("# Appendix A: Unified Vulnerability Taxonomy")
    lines.append("")
    lines.append(
        f"**{len(taxonomy)}** patterns across **{len(categories)}** categories, "
        f"**{len(cwes)}** unique CWEs, **{len(surfaces)}** attack surfaces, "
        f"**{len(owasp_llms)}** OWASP LLM Top-10 tie-ins."
    )
    lines.append("")
    lines.append(
        f"Severity distribution: "
        f"**{summary['severity_distribution']['critical']}** critical, "
        f"**{summary['severity_distribution']['high']}** high, "
        f"**{summary['severity_distribution']['medium']}** medium, "
        f"**{summary['severity_distribution']['low']}** low."
    )
    lines.append("")
    lines.append("## Attack Surfaces")
    lines.append("")
    lines.append("| ID | Surface | # Patterns |")
    lines.append("|----|---------|-----------|")
    for surf in surfaces:
        lines.append(f"| {surf.split('_')[0]} | {surf} | {len(by_surface[surf])} |")
    lines.append("")

    lines.append("## CWE Mapping Summary")
    lines.append("")
    lines.append("| CWE | Name | # Patterns |")
    lines.append("|-----|------|-----------|")
    cwe_names = {
        "CWE-20": "Improper Input Validation",
        "CWE-74": "Improper Neutralization of Special Elements (Injection)",
        "CWE-78": "OS Command Injection",
        "CWE-116": "Improper Encoding/Escaping",
        "CWE-200": "Exposure of Sensitive Information",
        "CWE-250": "Execution with Unnecessary Privileges",
        "CWE-252": "Unchecked Return Value",
        "CWE-269": "Improper Privilege Management",
        "CWE-284": "Improper Access Control",
        "CWE-319": "Cleartext Transmission of Sensitive Information",
        "CWE-345": "Insufficient Verification of Data Authenticity",
        "CWE-362": "Concurrent Execution with Improper Synchronization (Race)",
        "CWE-798": "Use of Hard-coded Credentials",
        "CWE-829": "Inclusion of Functionality from Untrusted Control Sphere",
        "CWE-862": "Missing Authorization",
        "CWE-913": "Improper Control of Dynamically-Managed Code Resources",
    }
    for cwe in cwes:
        count = sum(1 for p in taxonomy if p["cwe"] == cwe)
        lines.append(f"| {cwe} | {cwe_names.get(cwe, '?')} | {count} |")
    lines.append("")

    lines.append("## Full Pattern Catalog")
    lines.append("")
    for surf in surfaces:
        lines.append(f"### {surf}")
        lines.append("")
        lines.append(
            "| ID | Severity | Category | CWE | OWASP LLM | "
            "Protocols | Description |"
        )
        lines.append(
            "|----|----------|----------|-----|-----------|"
            "-----------|-------------|"
        )
        for p in by_surface[surf]:
            sev = p["severity"].upper()
            protos = ",".join(p["protocols"])
            desc = p["description"].replace("|", "\\|")[:80]
            if len(p["description"]) > 80:
                desc += "…"
            lines.append(
                f"| {p['id']} | {sev} | {p['category']} | {p['cwe']} | "
                f"{p['owasp_llm'] or '-'} | {protos} | {desc} |"
            )
        lines.append("")

    lines.append("## Remediation Guide")
    lines.append("")
    for surf in surfaces:
        lines.append(f"### {surf}")
        lines.append("")
        for p in by_surface[surf]:
            lines.append(f"- **{p['id']}** ({p['severity']}): {p['remediation']}")
        lines.append("")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved MD to {OUTPUT_MD}")

    # Console summary
    print()
    print("=" * 60)
    print(f"Patterns:       {len(taxonomy)}")
    print(f"Categories:     {len(categories)}")
    print(f"CWEs:           {len(cwes)}")
    print(f"Attack surfaces: {len(surfaces)}")
    print(f"OWASP LLM:      {len(owasp_llms)}")
    print()
    print("Severity distribution:")
    for sev, cnt in summary["severity_distribution"].items():
        print(f"  {sev:<10} {cnt}")
    print()
    print("Categories:")
    for cat in categories:
        cnt = sum(1 for p in taxonomy if p["category"] == cat)
        print(f"  {cat:<30} {cnt}")


if __name__ == "__main__":
    main()
