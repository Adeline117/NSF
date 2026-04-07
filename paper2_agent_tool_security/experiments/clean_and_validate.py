"""
Paper 2: Clean and Validate Scan Data
======================================
Remove non-Web3 servers from scan_results.json, keeping only repos that
actually interact with blockchain/crypto/DeFi/wallet.  For remaining
findings, classify each as likely true positive vs likely false positive
based on contextual heuristics.

Usage:
    python paper2_agent_tool_security/experiments/clean_and_validate.py
"""

import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCAN_RESULTS_PATH = os.path.join(SCRIPT_DIR, "scan_results.json")
CATALOG_PATH = os.path.join(SCRIPT_DIR, "..", "data", "server_catalog.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "cleaned_results.json")

# ============================================================
# WEB3 CLASSIFICATION
# ============================================================

# Keywords that indicate a repo interacts with blockchain/crypto
WEB3_NAME_KEYWORDS = [
    "web3", "crypto", "defi", "solana", "ethereum", "evm", "blockchain",
    "wallet", "swap", "dex", "nft", "token", "chain", "onchain",
    "binance", "uniswap", "aave", "polymarket", "flashloan", "gnosis",
    "staking", "lending", "bridge", "vault", "yield", "mev",
]

WEB3_DESC_KEYWORDS = [
    "blockchain", "cryptocurrency", "crypto", "defi", "decentralized finance",
    "wallet", "ethereum", "solana", "web3", "smart contract", "erc-20",
    "erc20", "erc-721", "erc721", "token", "swap", "dex", "nft",
    "on-chain", "onchain", "evm", "transaction", "trading",
    "binance", "uniswap", "aave", "compound", "polymarket",
    "flashloan", "gnosis", "safe", "staking", "lending", "bridge",
    "vault", "yield", "mev", "liquidity", "amm",
    "solidity", "smart-contract", "private key", "mnemonic", "seed phrase",
    "rpc", "ethers", "viem", "web3.py", "web3js", "wagmi",
]

WEB3_TOPIC_KEYWORDS = [
    "blockchain", "crypto", "defi", "wallet", "ethereum", "solana",
    "web3", "nft", "dex", "erc20", "smart-contract", "solidity",
    "trading", "binance", "uniswap", "onchain", "evm",
]

# Repos that are definitively NOT Web3, regardless of keyword matches
NON_WEB3_REPOS = {
    "getsentry/XcodeBuildMCP",          # Xcode build tools
    "sooperset/mcp-atlassian",          # Atlassian (Jira/Confluence)
    "ridafkih/keeper.sh",               # Calendar sync
    "CodeGraphContext/CodeGraphContext", # Code graph indexer
    "apify/apify-mcp-server",           # Web scraping
    "taylorwilsdon/google_workspace_mcp",  # Google Workspace
    "av/harbor",                        # LLM stack manager
    "modelcontextprotocol/inspector",   # MCP inspector/testing
    "aipotheosis-labs/aci",             # Generic tool platform
    "iannuttall/mcp-boilerplate",       # Boilerplate template
    "Pimzino/spec-workflow-mcp",        # Spec-driven dev workflow
    "Dataojitori/nocturne_memory",      # Long-term memory server
    "timescale/pg-aiguide",             # PostgreSQL guide
    "executeautomation/mcp-playwright", # Browser automation
    "MicrosoftDocs/mcp",               # Microsoft docs
    "open-webui/openapi-servers",       # OpenAPI tool servers
    "leonardsellem/n8n-mcp-server",     # n8n workflow automation
    "0x4m4/hexstrike-ai",              # Cybersecurity pentesting
    "jlumbroso/passage-of-time-mcp",   # Time/calendar tool
    "jamubc/gemini-mcp-tool",          # Gemini LLM bridge
    # Propaganda/spam repos
    "cirosantilli/china-dictatorship",
    "cirosantilli/china-dictatroship-7",
    "zszszszsz/.config",
    "mRFWq7LwNPZjaVv5v6eo/cihna-dictattorshrip-8",
    "Daravai1234/china-dictatorship",
    # Generic dev tools
    "sisig-ai/doctor",                 # Web crawler/indexer
    "elusznik/mcp-server-code-execution-mode",  # Code execution sandbox
    "grll/mcpadapt",                   # MCP adapter framework
    "GalaxyLLMCI/lyraios",             # Multi-agent OS (not Web3 specific)
}


def is_web3_repo(repo: dict) -> bool:
    """Determine if a repo is actually Web3-related based on name, description, topics."""
    name = repo.get("full_name", "")

    # Explicit exclusion list
    if name in NON_WEB3_REPOS:
        return False

    name_lower = name.lower()
    desc = (repo.get("description") or "").lower()
    topics = [t.lower() for t in repo.get("topics", [])]
    topics_str = " ".join(topics)

    # Check name
    if any(kw in name_lower for kw in WEB3_NAME_KEYWORDS):
        return True

    # Check description
    if any(kw in desc for kw in WEB3_DESC_KEYWORDS):
        return True

    # Check topics
    if any(kw in topics_str for kw in WEB3_TOPIC_KEYWORDS):
        return True

    # Check protocol tag from catalog
    protocol = repo.get("protocol", repo.get("catalog_protocol", ""))
    if protocol == "web3_native":
        return True

    return False


# ============================================================
# FALSE POSITIVE CLASSIFICATION
# ============================================================

# Heuristic rules for classifying findings as likely true/false positive
def classify_finding(finding: dict, repo_name: str) -> dict:
    """
    Add 'classification' field to a finding:
      - 'true_positive':  likely real vulnerability
      - 'false_positive': likely not a real issue
      - 'uncertain':      needs manual review

    Also add 'fp_reason' explaining the classification.
    """
    file_path = finding.get("file", "")
    matched_text = finding.get("matched_text", "")
    context = finding.get("context", "")
    category = finding.get("category", "")
    severity = finding.get("severity", "")
    pattern_id = finding.get("pattern_id", "")

    classification = "true_positive"
    fp_reason = ""

    # Rule 1: Test files are likely false positives
    test_indicators = [
        "test/", "tests/", "__tests__/", "spec/", ".test.", ".spec.",
        "_test.", "_spec.", "test_", "spec_", "mock", "fixture",
        "example/", "examples/", "demo/", "sample/",
    ]
    if any(ind in file_path.lower() for ind in test_indicators):
        classification = "false_positive"
        fp_reason = "Finding in test/example file"

    # Rule 2: Comment-only matches
    # Check if the context line starts with a comment marker
    context_lines = context.split("\n")
    matched_line = ""
    for cl in context_lines:
        if ">>>" in cl:
            matched_line = cl.strip()
            break
    comment_patterns = [
        r"^\s*(?:>>>)?\s*\d+:\s*(?://|#|/\*|\*|'''|\"\"\")",
    ]
    if matched_line and any(re.search(p, matched_line) for p in comment_patterns):
        classification = "false_positive"
        fp_reason = "Match appears in a comment"

    # Rule 3: IV-002 (amount/value without bounds) matching generic 'value: string'
    # is almost always a false positive -- it's too broad a regex
    if pattern_id == "IV-002":
        # The pattern r"(?:amount|value|quantity|wei|gwei)\s*[:=]..." is extremely broad.
        # 'value' matches any variable named 'value', which is ubiquitous.
        if "value" in matched_text.lower() and not any(
            kw in matched_text.lower() for kw in ["amount", "wei", "gwei", "quantity"]
        ):
            # Generic 'value' without crypto context
            crypto_context_words = [
                "wei", "gwei", "ether", "token", "amount", "balance",
                "transfer", "approve", "allowance", "deposit", "withdraw",
            ]
            if not any(w in context.lower() for w in crypto_context_words):
                classification = "false_positive"
                fp_reason = "Generic 'value' variable, not crypto-specific"

    # Rule 4: IV-001 (address without validation) matching generic 'address' in
    # non-crypto contexts (e.g., URL address, memory address, IP address)
    if pattern_id == "IV-001":
        non_crypto_address = [
            "proxyFullAddress", "ipAddress", "ip_address", "emailAddress",
            "email_address", "url", "hostname", "host", "endpoint",
            "serverAddress", "server_address", "remoteAddress",
        ]
        if any(nca in matched_text for nca in non_crypto_address):
            classification = "false_positive"
            fp_reason = "Non-Ethereum address (IP/URL/email)"

        # Also check context for non-crypto address usage
        if any(nca in context for nca in non_crypto_address):
            classification = "false_positive"
            fp_reason = "Non-Ethereum address context"

    # Rule 5: HC-001 hardcoded credentials that are actually env variable references
    if pattern_id == "HC-001":
        env_patterns = [
            r"process\.env", r"os\.environ", r"os\.getenv",
            r"env\(", r"dotenv", r"\.env",
        ]
        if any(re.search(p, context, re.IGNORECASE) for p in env_patterns):
            classification = "false_positive"
            fp_reason = "Credential loaded from environment variable"

    # Rule 6: CE-001 (cross-tool escalation) in framework/SDK code
    if pattern_id == "CE-001":
        sdk_paths = [
            "node_modules/", "sdk/", "lib/", "packages/core/",
            "packages/sdk/", "framework/",
        ]
        if any(sp in file_path for sp in sdk_paths):
            classification = "false_positive"
            fp_reason = "Cross-tool call in framework/SDK code, not user-facing"

    # Rule 7: SC-001 (state confusion) matching 'global' keyword in Python imports
    if pattern_id == "SC-001":
        if re.search(r"global\s+\w+", matched_text):
            # Check if it's a simple global declaration in a function
            if "def " in context:
                classification = "uncertain"
                fp_reason = "Python global declaration -- may or may not cause state confusion"

    # Rule 8: MH-001 (missing harness) in test helpers
    if pattern_id == "MH-001":
        if any(ind in file_path.lower() for ind in ["test", "spec", "example", "demo"]):
            classification = "false_positive"
            fp_reason = "Server instantiation in test/example code"

    # Rule 9: Critical findings that are in actual code paths
    # are likely true positives -- keep them as-is
    if classification == "true_positive" and severity == "critical":
        # Double-check it's not in a dead code path (commented out block)
        if "/*" in context and "*/" in context:
            classification = "false_positive"
            fp_reason = "Code appears to be in a block comment"

    finding["classification"] = classification
    finding["fp_reason"] = fp_reason
    return finding


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("Paper 2: Clean and Validate Scan Data")
    print("=" * 60)
    print()

    # Load scan results
    if not os.path.exists(SCAN_RESULTS_PATH):
        print(f"ERROR: {SCAN_RESULTS_PATH} not found.")
        sys.exit(1)

    with open(SCAN_RESULTS_PATH, "r") as f:
        data = json.load(f)

    # Load catalog for richer repo metadata
    catalog_map = {}
    if os.path.exists(CATALOG_PATH):
        with open(CATALOG_PATH, "r") as f:
            catalog = json.load(f)
        catalog_map = {r["full_name"]: r for r in catalog.get("repos", [])}

    repo_results = data.get("repo_results", [])
    print(f"Original repos scanned: {len(repo_results)}")
    print(f"Original total findings: {data['summary']['total_findings']}")
    print()

    # ---- Step 1: Filter non-Web3 repos ----
    print("--- Step 1: Remove non-Web3 servers ---")
    removed_repos = []
    kept_repos = []

    for rr in repo_results:
        name = rr["full_name"]
        # Merge catalog metadata for richer classification
        catalog_entry = catalog_map.get(name, {})
        merged = {**rr, **{k: v for k, v in catalog_entry.items() if k not in rr}}

        if is_web3_repo(merged):
            kept_repos.append(rr)
            print(f"  [KEEP]   {name} (findings={rr['total_findings']})")
        else:
            removed_repos.append(rr)
            print(f"  [REMOVE] {name} (findings={rr['total_findings']})")

    print()
    print(f"Kept: {len(kept_repos)} repos")
    print(f"Removed: {len(removed_repos)} repos")
    print()

    # ---- Step 2: Classify findings ----
    print("--- Step 2: Classify findings as TP/FP ---")
    total_tp = 0
    total_fp = 0
    total_uncertain = 0
    all_cleaned_findings = []

    for rr in kept_repos:
        repo_name = rr["full_name"]
        repo_tp = 0
        repo_fp = 0
        repo_uncertain = 0

        for finding in rr.get("findings", []):
            classify_finding(finding, repo_name)
            if finding["classification"] == "true_positive":
                repo_tp += 1
            elif finding["classification"] == "false_positive":
                repo_fp += 1
            else:
                repo_uncertain += 1

        total_tp += repo_tp
        total_fp += repo_fp
        total_uncertain += repo_uncertain
        all_cleaned_findings.extend(rr.get("findings", []))

        print(f"  {repo_name}: TP={repo_tp}, FP={repo_fp}, uncertain={repo_uncertain}")

    print()
    kept_total_findings = sum(rr["total_findings"] for rr in kept_repos)
    removed_total_findings = sum(rr["total_findings"] for rr in removed_repos)
    print(f"Kept findings: {kept_total_findings}")
    print(f"Removed findings: {removed_total_findings}")
    print(f"Classification: TP={total_tp}, FP={total_fp}, uncertain={total_uncertain}")

    if (total_tp + total_fp + total_uncertain) > 0:
        precision = total_tp / (total_tp + total_fp + total_uncertain) * 100
        print(f"Estimated precision (TP / all): {precision:.1f}%")
    print()

    # ---- Step 3: Recompute summary ----
    print("--- Step 3: Recompute summary ---")

    severity_counts = Counter()
    category_counts = Counter()
    tp_severity_counts = Counter()
    tp_category_counts = Counter()

    for rr in kept_repos:
        for f in rr.get("findings", []):
            severity_counts[f["severity"]] += 1
            category_counts[f["category"]] += 1
            if f["classification"] == "true_positive":
                tp_severity_counts[f["severity"]] += 1
                tp_category_counts[f["category"]] += 1

    print(f"  All findings by severity: {dict(severity_counts)}")
    print(f"  TP findings by severity:  {dict(tp_severity_counts)}")
    print(f"  All findings by category: {dict(category_counts)}")
    print(f"  TP findings by category:  {dict(tp_category_counts)}")
    print()

    # ---- Step 4: Build cleaned output ----
    cleaned_output = {
        "metadata": {
            "source": SCAN_RESULTS_PATH,
            "cleaned_at": datetime.now(timezone.utc).isoformat(),
            "original_repos": len(repo_results),
            "kept_repos": len(kept_repos),
            "removed_repos": len(removed_repos),
            "original_findings": data["summary"]["total_findings"],
            "kept_findings": kept_total_findings,
            "removed_findings": removed_total_findings,
        },
        "classification_summary": {
            "true_positive": total_tp,
            "false_positive": total_fp,
            "uncertain": total_uncertain,
            "estimated_precision_pct": round(
                total_tp / max(1, total_tp + total_fp + total_uncertain) * 100, 1
            ),
        },
        "summary": {
            "total_findings": kept_total_findings,
            "true_positive_findings": total_tp,
            "by_severity": dict(severity_counts),
            "by_category": dict(category_counts),
            "tp_by_severity": dict(tp_severity_counts),
            "tp_by_category": dict(tp_category_counts),
        },
        "removed_repos": [
            {"repo": rr["full_name"], "reason": "Not a Web3/blockchain tool server",
             "findings_removed": rr["total_findings"]}
            for rr in removed_repos
        ],
        "risk_ranking": [
            {
                "repo": rr["full_name"],
                "risk_score": rr["risk_score"],
                "risk_rating": rr["risk_rating"],
                "total_findings": rr["total_findings"],
                "true_positives": sum(
                    1 for f in rr.get("findings", [])
                    if f.get("classification") == "true_positive"
                ),
                "false_positives": sum(
                    1 for f in rr.get("findings", [])
                    if f.get("classification") == "false_positive"
                ),
                "protocol": rr.get("detected_protocol", rr.get("catalog_protocol", "")),
                "stars": rr["stars"],
            }
            for rr in sorted(kept_repos, key=lambda x: x["risk_score"], reverse=True)
        ],
        "repo_results": kept_repos,
    }

    # ---- Step 5: Save ----
    with open(OUTPUT_PATH, "w") as f:
        json.dump(cleaned_output, f, indent=2, default=str)
    print(f"Saved cleaned results to: {OUTPUT_PATH}")
    print(f"  Repos: {len(kept_repos)}")
    print(f"  Total findings: {kept_total_findings}")
    print(f"  True positives: {total_tp}")
    print(f"  False positives: {total_fp}")
    print()

    # ---- Summary Table ----
    print("=" * 60)
    print("CLEANING SUMMARY")
    print("=" * 60)
    print(f"{'Metric':<35} {'Before':>10} {'After':>10}")
    print("-" * 55)
    print(f"{'Repos':<35} {len(repo_results):>10} {len(kept_repos):>10}")
    print(f"{'Total Findings':<35} {data['summary']['total_findings']:>10} {kept_total_findings:>10}")
    print(f"{'True Positives':<35} {'N/A':>10} {total_tp:>10}")
    print(f"{'False Positives':<35} {'N/A':>10} {total_fp:>10}")
    print(f"{'Uncertain':<35} {'N/A':>10} {total_uncertain:>10}")
    print(f"{'Est. Precision':<35} {'N/A':>10} {cleaned_output['classification_summary']['estimated_precision_pct']:>9.1f}%")
    print()


if __name__ == "__main__":
    main()
