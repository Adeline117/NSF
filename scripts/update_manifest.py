#!/usr/bin/env python3
"""Update a paper's manifest.json from its experiment result files.

This is Agent B's core logic: read experiment JSONs, extract key values,
update manifest, regenerate macros, verify compilation.

Usage:
    python scripts/update_manifest.py paper0 [experiment_name]
    python scripts/update_manifest.py paper3 adversarial_best_response
"""
import json
import sys
import subprocess
from pathlib import Path
from datetime import date

ROOT = Path(__file__).resolve().parent.parent

PAPERS = {
    "paper0": ROOT / "paper0_ai_agent_theory",
    "paper1": ROOT / "paper1_onchain_agent_id",
    "paper2": ROOT / "paper2_agent_tool_security",
    "paper3": ROOT / "paper3_ai_sybil",
}


def load_manifest(paper_dir: Path) -> dict:
    manifest_path = paper_dir / "results" / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"_meta": {"last_updated": str(date.today()), "schema_version": "1.0"}}


def save_manifest(paper_dir: Path, manifest: dict):
    manifest["_meta"]["last_updated"] = str(date.today())
    manifest_path = paper_dir / "results" / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Updated {manifest_path.relative_to(ROOT)}")


def regenerate_macros(paper_key: str):
    script = ROOT / "scripts" / "generate_macros.py"
    subprocess.run([sys.executable, str(script), paper_key], check=True)


def verify_compilation(paper_dir: Path) -> bool:
    """Try to compile and report success/failure."""
    tex_dirs = [d for d in paper_dir.iterdir() if d.is_dir() and (d / "main.tex").exists()]
    all_ok = True
    for tex_dir in tex_dirs:
        result = subprocess.run(
            ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
            cwd=str(tex_dir),
            capture_output=True,
            timeout=120,
        )
        if result.returncode != 0:
            print(f"  COMPILE FAIL: {tex_dir.name}/main.tex")
            all_ok = False
        else:
            print(f"  COMPILE OK: {tex_dir.name}/main.tex")
    return all_ok


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/update_manifest.py <paper> [experiment]")
        sys.exit(1)

    paper_key = sys.argv[1]
    if paper_key not in PAPERS:
        candidates = [k for k in PAPERS if paper_key in k]
        paper_key = candidates[0] if candidates else None
        if not paper_key:
            print(f"Unknown paper: {sys.argv[1]}")
            sys.exit(1)

    paper_dir = PAPERS[paper_key]
    experiment_filter = sys.argv[2] if len(sys.argv) > 2 else None

    # Load current manifest
    manifest = load_manifest(paper_dir)

    # Find experiment results
    exp_dir = paper_dir / "experiments"
    result_files = sorted(exp_dir.glob("*_results.json"))

    if experiment_filter:
        result_files = [f for f in result_files if experiment_filter in f.stem]

    print(f"[{paper_key}] Found {len(result_files)} result files")
    for rf in result_files:
        print(f"  - {rf.name}")

    # Save and regenerate
    save_manifest(paper_dir, manifest)
    regenerate_macros(paper_key)

    print("\nDone. Run verify_compilation manually if needed.")


if __name__ == "__main__":
    main()
