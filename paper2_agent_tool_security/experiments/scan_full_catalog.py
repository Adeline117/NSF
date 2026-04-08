"""
Paper 2: Scale Static Analysis to 200+ Servers
================================================
Stratified by protocol, scans the top 50 per protocol family from
server_catalog.json (or all available, whichever is smaller).

Targets:
  MCP: 50 (out of 205 catalogued)
  Web3-native: 50 (out of 79)
  LangChain: 40 (out of 40, all)
  OpenAI: 12 (out of 12, all)
Total: ~152 stratified servers, vs the 62 in scan_results.json + web3_scan_results.json.

Reuses select_repos / scan_single_repo / aggregate_results from
scan_real_servers.py via dynamic import.

Outputs:
  - paper2_agent_tool_security/experiments/full_catalog_scan_results.json
  - paper2_agent_tool_security/experiments/scan_progress.log
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the existing single-repo scan logic
sys.path.insert(0, str(PROJECT_ROOT / "paper2_agent_tool_security" / "experiments"))
from scan_real_servers import (
    scan_single_repo,
    aggregate_results,
)

CATALOG_PATH = (
    PROJECT_ROOT / "paper2_agent_tool_security" / "data" / "server_catalog.json"
)
OUT_PATH = (
    PROJECT_ROOT / "paper2_agent_tool_security" / "experiments"
    / "full_catalog_scan_results.json"
)
LOG_PATH = (
    PROJECT_ROOT / "paper2_agent_tool_security" / "experiments"
    / "scan_progress.log"
)

# How many per protocol
TARGETS = {
    "mcp": 50,
    "web3_native": 50,
    "langchain": 40,
    "openai": 12,
}

EXCLUDE_REPOS = {
    # Already exhausted in earlier runs
}


def select_stratified(catalog: dict) -> list:
    """Select top N per protocol, sorted by stars."""
    repos = catalog.get("repos", [])
    by_protocol: dict[str, list] = {}
    for r in repos:
        if r.get("full_name") in EXCLUDE_REPOS:
            continue
        lang = r.get("language", "Unknown")
        if lang in ("Unknown", "HTML", "Shell", "Dockerfile"):
            continue
        size_kb = r.get("size_kb", 0)
        if size_kb < 10:
            continue
        proto = r.get("protocol", "unknown")
        by_protocol.setdefault(proto, []).append(r)

    selected = []
    for proto, target in TARGETS.items():
        candidates = by_protocol.get(proto, [])
        candidates.sort(key=lambda x: x.get("stars", 0), reverse=True)
        selected.extend(candidates[:target])
        print(f"  {proto:<14} target={target:<3} available={len(candidates):<4} "
              f"selected={min(target, len(candidates))}")

    return selected


def main():
    print("=" * 70)
    print("Paper 2: Scale Static Analysis to 200+ Servers")
    print("=" * 70)

    with open(CATALOG_PATH) as f:
        catalog = json.load(f)
    print(f"Total catalog: {catalog.get('total_repos', '?')} repos")

    selected = select_stratified(catalog)
    print(f"\nSelected: {len(selected)} repos for scanning")

    # Resume support
    completed_names = set()
    existing = []
    if OUT_PATH.exists():
        try:
            with open(OUT_PATH) as f:
                prev = json.load(f)
            existing = prev.get("repo_results", [])
            completed_names = {r["full_name"] for r in existing}
            print(f"  Resuming: {len(completed_names)} already done")
        except Exception as e:
            print(f"  Could not load previous: {e}")

    log_lines = []
    new_results = list(existing)
    t0 = time.time()
    for i, repo in enumerate(selected):
        if repo["full_name"] in completed_names:
            continue
        try:
            r = scan_single_repo(repo)
            from dataclasses import asdict
            new_results.append(asdict(r))
            elapsed = time.time() - t0
            log_lines.append(
                f"[{i+1}/{len(selected)}] {repo['full_name']} "
                f"findings={r.total_findings} risk={r.risk_score:.1f}  "
                f"({elapsed:.0f}s elapsed)"
            )
            print(log_lines[-1])
        except Exception as e:
            log_lines.append(f"[{i+1}/{len(selected)}] {repo['full_name']} ERROR: {e}")
            print(log_lines[-1])

        # Periodic save
        if (i + 1) % 10 == 0:
            _save(new_results)
            with open(LOG_PATH, "w") as f:
                f.write("\n".join(log_lines))

    # Final save
    _save(new_results)
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(log_lines))

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"Total scanned: {len(new_results)}")


def _save(results: list):
    with open(OUT_PATH, "w") as f:
        json.dump({
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "n_repos": len(results),
                "stratification": TARGETS,
            },
            "repo_results": results,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
