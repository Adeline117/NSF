#!/usr/bin/env python3
"""Generate macros_results.tex from results/manifest.json for each paper.

Usage:
    python scripts/generate_macros.py              # all papers
    python scripts/generate_macros.py paper0       # single paper
    python scripts/generate_macros.py paper1 paper3
"""
import json
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PAPERS = {
    "paper0": ROOT / "paper0_ai_agent_theory",
    "paper1": ROOT / "paper1_onchain_agent_id",
    "paper2": ROOT / "paper2_agent_tool_security",
    "paper3": ROOT / "paper3_ai_sybil",
}


def key_to_macro_name(key: str) -> str:
    """Convert manifest key to LaTeX macro name.

    e.g., "gbm_accuracy_31feat" -> "\\gbmaccuracyxxxifeat"
    Rules: remove underscores, lowercase, replace digits contextually.
    We keep it simple: camelCase-ish with no underscores.
    """
    # Remove leading underscore keys (meta)
    parts = key.split("_")
    # CamelCase
    name = parts[0] + "".join(p.capitalize() for p in parts[1:])
    # LaTeX macro names can't have digits at certain positions; prefix with 'n' if starts with digit
    if name[0].isdigit():
        name = "n" + name
    # Replace any remaining non-alpha chars
    name = "".join(c for c in name if c.isalpha())
    return name


def format_value(val) -> str:
    """Format a value for LaTeX display."""
    if isinstance(val, bool):
        return "Yes" if val else "No"
    if isinstance(val, float):
        # Percentages (values between 0 and 1 that look like rates)
        if abs(val) <= 1.0:
            # Keep as decimal, 3-4 significant figures
            if abs(val) < 0.01:
                return f"{val:.4f}"
            return f"{val:.3f}" if abs(val) < 1 else f"{val:.2f}"
        # Larger numbers
        if val == int(val):
            return f"{int(val)}"
        return f"{val:.2f}"
    if isinstance(val, int):
        # Add comma separators for large numbers
        return f"{val:,}"
    return str(val)


def generate_macros(paper_key: str):
    paper_dir = PAPERS[paper_key]
    manifest_path = paper_dir / "results" / "manifest.json"

    if not manifest_path.exists():
        print(f"  SKIP {paper_key}: no manifest.json")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Find the target tex directory (prefer venue-specific, fall back to latex/)
    tex_dirs = []
    for d in paper_dir.iterdir():
        if d.is_dir() and (d / "main.tex").exists():
            tex_dirs.append(d)

    lines = [
        "% AUTO-GENERATED from results/manifest.json",
        f"% Last regenerated: {manifest.get('_meta', {}).get('last_updated', 'unknown')}",
        "% DO NOT EDIT MANUALLY — run: python scripts/generate_macros.py",
        "",
    ]

    seen_names = {}
    for key, val in manifest.items():
        if key.startswith("_"):
            continue

        macro_name = key_to_macro_name(key)

        # Handle collision by appending suffix
        if macro_name in seen_names:
            macro_name = macro_name + "B"
        seen_names[macro_name] = key

        formatted = format_value(val)
        lines.append(f"\\newcommand{{\\{macro_name}}}{{{formatted}}}  % {key}")

    lines.append("")
    content = "\n".join(lines)

    # Write to each tex directory that has a main.tex
    for tex_dir in tex_dirs:
        out_path = tex_dir / "macros_results.tex"
        with open(out_path, "w") as f:
            f.write(content)
        print(f"  WROTE {out_path.relative_to(ROOT)}")


def main():
    targets = sys.argv[1:] if len(sys.argv) > 1 else list(PAPERS.keys())

    for paper_key in targets:
        if paper_key not in PAPERS:
            # Try with "paper" prefix
            candidates = [k for k in PAPERS if paper_key in k]
            if candidates:
                paper_key = candidates[0]
            else:
                print(f"  ERROR: unknown paper '{paper_key}'")
                continue
        print(f"[{paper_key}]")
        generate_macros(paper_key)


if __name__ == "__main__":
    main()
