"""
Agent-Tool-Scanner: Unified CLI Entry Point
=============================================
Single `agent-tool-scanner` command with subcommands:

  enumerate   — discover servers from GitHub (delegates to enumerate_servers.py)
  scan        — scan a single cloned repo
  batch       — scan a directory of cloned repos
  taxonomy    — export taxonomy JSON / MD

Usage:
    agent-tool-scanner enumerate --protocol mcp
    agent-tool-scanner scan /path/to/repo
    agent-tool-scanner batch ~/tmp/cloned_repos --output results.json
    agent-tool-scanner taxonomy --format json
"""

import argparse
import json
import sys
from pathlib import Path

# Path setup for in-tree use and installed use
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def cmd_enumerate(args):
    from paper2_agent_tool_security.scanner.enumerate_servers import main as enum_main
    old_argv = sys.argv
    sys.argv = ["enumerate"] + args
    try:
        return enum_main()
    finally:
        sys.argv = old_argv


def cmd_scan(args):
    from paper2_agent_tool_security.static_analysis.analyzer import StaticAnalyzer
    parser = argparse.ArgumentParser(prog="agent-tool-scanner scan")
    parser.add_argument("repo_path")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--quiet", "-q", action="store_true")
    ns = parser.parse_args(args)

    analyzer = StaticAnalyzer()
    result = analyzer.scan_repo(ns.repo_path)

    # Convert to dict
    from dataclasses import asdict
    result_dict = asdict(result)

    if ns.output:
        with open(ns.output, "w") as f:
            json.dump(result_dict, f, indent=2)
        print(f"Saved to {ns.output}", file=sys.stderr)
    else:
        if not ns.quiet:
            print(json.dumps(result_dict, indent=2))
        else:
            print(f"{result.repo_name}: {result.total_findings} findings, "
                  f"risk {result.risk_score:.1f} ({result.risk_rating})")
    return 0


def cmd_batch(args):
    from paper2_agent_tool_security.static_analysis.analyzer import StaticAnalyzer
    parser = argparse.ArgumentParser(prog="agent-tool-scanner batch")
    parser.add_argument("repos_dir")
    parser.add_argument("--catalog", default=None,
                        help="server_catalog.json for protocol hints")
    parser.add_argument("--output", "-o", default="batch_scan_results.json")
    ns = parser.parse_args(args)

    analyzer = StaticAnalyzer()
    result = analyzer.scan_batch(ns.repos_dir, catalog_path=ns.catalog)

    from dataclasses import asdict
    with open(ns.output, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"Saved {result.n_repos} repo scans to {ns.output}", file=sys.stderr)
    print(f"  Total findings: {result.total_findings}", file=sys.stderr)
    print(f"  Mean risk score: {result.mean_risk_score:.1f}", file=sys.stderr)
    return 0


def cmd_taxonomy(args):
    parser = argparse.ArgumentParser(prog="agent-tool-scanner taxonomy")
    parser.add_argument("--format", choices=["json", "md", "both"], default="both")
    parser.add_argument("--output-dir", default=".")
    ns = parser.parse_args(args)

    from paper2_agent_tool_security.scripts.export_taxonomy import main as tax_main
    return tax_main() or 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="agent-tool-scanner",
        description=(
            "Static analysis scanner for AI agent tool interfaces "
            "(MCP, OpenAI function calling, LangChain, Web3-native)"
        ),
    )
    parser.add_argument("--version", action="version", version="0.1.0")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("enumerate", help="Enumerate servers from GitHub").set_defaults(
        func=lambda ns, rest: cmd_enumerate(rest),
    )
    sub.add_parser("scan", help="Scan a single repo").set_defaults(
        func=lambda ns, rest: cmd_scan(rest),
    )
    sub.add_parser("batch", help="Scan a directory of cloned repos").set_defaults(
        func=lambda ns, rest: cmd_batch(rest),
    )
    sub.add_parser("taxonomy", help="Export vulnerability taxonomy").set_defaults(
        func=lambda ns, rest: cmd_taxonomy(rest),
    )

    ns, rest = parser.parse_known_args(argv)
    return ns.func(ns, rest)


if __name__ == "__main__":
    sys.exit(main() or 0)
