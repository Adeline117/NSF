"""
Paper 2: Scan Real Web3 Agent Tool Servers
===========================================
Clone top-starred repos from server_catalog.json, run static vulnerability
analysis, and produce structured findings.

Steps:
  1. Read server_catalog.json
  2. Select top 20 repos by stars (filter for actual tool servers with code)
  3. For each repo:
     - Shallow clone to temp directory
     - Detect protocol family
     - Run regex-based vulnerability patterns
     - Record findings with severity, category, file, line
     - Clean up
  4. Aggregate and save results to scan_results.json

Usage:
    python paper2_agent_tool_security/experiments/scan_real_servers.py

No external dependencies -- uses Python standard library only.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

CATALOG_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "server_catalog.json"
)
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "scan_results.json")

MAX_REPOS = 20
CLONE_DELAY_SECONDS = 1.5
CLONE_TIMEOUT_SECONDS = 60
MAX_FILES_PER_REPO = 500
MAX_FILE_SIZE_BYTES = 500_000  # 500KB per file

# Skip these directories during scanning
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv",
    "dist", "build", "out", ".next", "coverage", "artifacts",
    "cache", "typechain-types", ".tox", ".mypy_cache", ".pytest_cache",
    "vendor", "bower_components", ".yarn", ".pnp", "target",
    "lib", "deps", ".deps",
}

# Source file extensions to scan
SOURCE_EXTENSIONS = {
    ".ts", ".js", ".py", ".sol", ".tsx", ".jsx",
    ".mts", ".mjs", ".rs", ".go",
}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Finding:
    pattern_id: str
    category: str
    severity: str       # critical, high, medium, low
    description: str
    file: str           # relative path within repo
    line: int
    matched_text: str
    context: str        # surrounding lines
    cwe: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class RepoResult:
    full_name: str
    url: str
    stars: int
    catalog_protocol: str   # from catalog
    detected_protocol: str  # from static analysis
    protocol_confidence: float
    language: str
    files_scanned: int = 0
    scan_time_seconds: float = 0.0
    total_findings: int = 0
    risk_score: float = 0.0
    risk_rating: str = "unknown"
    by_severity: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    clone_success: bool = True

    def to_dict(self):
        d = asdict(self)
        return d


# ============================================================
# REPOS THAT ARE NOT ACTUAL TOOL SERVERS (filter list)
# ============================================================

# Repos that are docs-only, political propaganda, IDE frameworks, etc.
EXCLUDE_REPOS = {
    # Not tool servers -- political/spam repos that got into web3 search
    "cirosantilli/china-dictatorship",
    "cirosantilli/china-dictatroship-7",
    "zszszszsz/.config",
    "mRFWq7LwNPZjaVv5v6eo/cihna-dictattorshrip-8",
    "Daravai1234/china-dictatorship",
    # Documentation/guide repos, not tool servers
    "wesammustafa/Claude-Code-Everything-You-Need-to-Know",
    "demcp/awesome-web3-mcp-servers",
    "rhinestonewtf/awesome-modular-accounts",
    # IDE frameworks / clients, not tool servers
    "nanbingxyz/5ire",         # MCP client, not server
    "opensumi/core",           # IDE framework
    "jonigl/mcp-client-for-ollama",  # MCP client
    "patruff/ollama-mcp-bridge",     # bridge, not tool server
    # LLM inference servers, not agent tool servers
    "waybarrios/vllm-mlx",
    # Pure awesome-lists
    "SolanaRemix/node",
    # Not relevant
    "2PacOfWoW/AimSharpWoW",
    "NickBusey/jQuery-Date-Select-Boxes-Plugin",
    "dfraser74/magento2-marketplace-social-login",
    "wtcherr/lunar-lander-dqn",
    "buildbycj/smart-loginizer-free",
    "LyzrCore/lyzr-framework",
}


# ============================================================
# PROTOCOL DETECTION
# ============================================================

PROTOCOL_INDICATORS = {
    "mcp": [
        {"pattern": r"@modelcontextprotocol|from mcp import|mcp\.Server|McpServer",
         "weight": 10.0, "where": "content"},
        {"pattern": r"server\.tool\(|@tool|mcp_server",
         "weight": 5.0, "where": "content"},
        {"pattern": r"\"@modelcontextprotocol/sdk\"",
         "weight": 8.0, "where": "content"},
    ],
    "openai": [
        {"pattern": r"import openai|from openai|openai\.chat",
         "weight": 8.0, "where": "content"},
        {"pattern": r"function_call|tool_choice|\"functions\"\s*:|tools\s*=\s*\[",
         "weight": 7.0, "where": "content"},
    ],
    "langchain": [
        {"pattern": r"from langchain|import langchain|from crewai|from autogen",
         "weight": 10.0, "where": "content"},
        {"pattern": r"@tool|BaseTool|StructuredTool|initialize_agent|AgentExecutor",
         "weight": 7.0, "where": "content"},
    ],
    "web3_native": [
        {"pattern": r"pragma solidity|// SPDX-License-Identifier",
         "weight": 10.0, "where": "content"},
        {"pattern": r"ISafe|IModule|IGuard|ModuleManager|GnosisSafe",
         "weight": 8.0, "where": "content"},
        {"pattern": r"ERC4337|IEntryPoint|UserOperation|IPlugin|IValidator",
         "weight": 9.0, "where": "content"},
        {"pattern": r"delegatecall|execTransactionFromModule",
         "weight": 6.0, "where": "content"},
    ],
}


def detect_protocol(repo_path: str, max_files: int = 200) -> tuple:
    """Detect the protocol family of a repository.
    Returns (protocol_name, confidence).
    """
    scores = {p: 0.0 for p in PROTOCOL_INDICATORS}
    files_checked = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if files_checked >= max_files:
                break
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SOURCE_EXTENSIONS:
                continue
            filepath = os.path.join(root, fname)
            files_checked += 1
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(10_000)
            except (OSError, PermissionError):
                continue
            for proto, indicators in PROTOCOL_INDICATORS.items():
                for ind in indicators:
                    if ind["where"] == "content":
                        if re.search(ind["pattern"], content, re.IGNORECASE):
                            scores[proto] += ind["weight"]
        if files_checked >= max_files:
            break

    # Check package manifests
    for manifest, mappings in [
        ("package.json", [("@modelcontextprotocol", "mcp", 15),
                          ("openai", "openai", 10),
                          ("langchain", "langchain", 10)]),
        ("pyproject.toml", [("mcp", "mcp", 10),
                            ("openai", "openai", 8),
                            ("langchain", "langchain", 10)]),
        ("requirements.txt", [("mcp", "mcp", 10),
                              ("openai", "openai", 8),
                              ("langchain", "langchain", 10)]),
    ]:
        fpath = os.path.join(repo_path, manifest)
        if os.path.exists(fpath):
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    txt = f.read()
                for keyword, proto, weight in mappings:
                    if keyword.lower() in txt.lower():
                        scores[proto] += weight
            except OSError:
                pass

    total = sum(scores.values())
    if total == 0:
        return "unknown", 0.0
    best = max(scores, key=lambda p: scores[p])
    confidence = round(scores[best] / total, 3)
    return best, confidence


# ============================================================
# VULNERABILITY PATTERNS
# ============================================================

# Each pattern: (id, category, severity, description, regex, cwe, file_exts)
# file_exts = None means all source extensions

VULN_PATTERNS = [
    # --- Private Key Exposure ---
    ("PKE-001", "private_key_exposure", "critical",
     "Private key, mnemonic, or seed phrase in code or as parameter",
     r"(?:private[_\s]?key|secret[_\s]?key|mnemonic|seed[_\s]?phrase|signing[_\s]?key)\s*[=:]\s*[\"']",
     "CWE-200", None),

    ("PKE-002", "private_key_exposure", "critical",
     "Raw private key hex string (64 hex chars)",
     r"[\"']0x[a-fA-F0-9]{64}[\"']",
     "CWE-200", None),

    ("PKE-003", "private_key_exposure", "critical",
     "Wallet signing without user confirmation",
     r"(?:signTransaction|signMessage|eth_sign|personal_sign|sign_transaction|signTypedData)"
     r"(?!.*(?:confirm|approval|prompt|user|consent|review))",
     "CWE-862", None),

    # --- Unlimited Approval ---
    ("UA-001", "unlimited_approval", "high",
     "Unlimited token approval (MaxUint256 or equivalent)",
     r"(?:MaxUint256|type\(uint256\)\.max|2\s*\*\*\s*256|0xf{64}|maxApproval|UNLIMITED|uint256\(-1\)|MAX_UINT)",
     "CWE-250", None),

    # --- Transaction Validation Missing ---
    ("TV-001", "tx_validation_missing", "high",
     "Transaction sent without validation/whitelist check",
     r"(?:sendTransaction|send_transaction|sendRawTransaction)\s*\("
     r"(?!.*(?:whitelist|allowlist|valid|check|verify|approved))",
     "CWE-345", None),

    # --- Tool Poisoning ---
    ("TP-001", "tool_poisoning", "high",
     "Tool/function description contains directive instructions (always/must/never/ignore)",
     r"(?:description|desc)\s*[:=]\s*[\"'][^\"']*"
     r"(?:always|must|never|ignore\s+previous|forget|override|system\s*prompt|"
     r"do\s+not\s+ask|without\s+confirm)[^\"']*[\"']",
     "CWE-913", None),

    ("TP-002", "tool_poisoning", "medium",
     "Tool description references other tools, directing LLM to chain calls",
     r"(?:description|desc)\s*[:=]\s*[\"'][^\"']*"
     r"(?:first\s+call|then\s+use|before\s+running|after\s+calling|"
     r"requires?\s+calling)[^\"']*[\"']",
     "CWE-913", None),

    # --- Prompt Injection ---
    ("PI-001", "prompt_injection", "critical",
     "User input interpolated into description/prompt/system message",
     r"(?:description|prompt|system|template)\s*[:=]\s*"
     r"(?:f\"|f'|.*\.format\(|.*%s|.*\+\s*(?:user|input|data|query|request|msg))",
     "CWE-74", None),

    # --- Hardcoded Credentials ---
    ("HC-001", "hardcoded_credentials", "critical",
     "Hardcoded API key, password, secret, or token",
     r"(?:api[_\s]?key|password|secret|token|auth[_\s]?token)\s*[=:]\s*['\"][a-zA-Z0-9_\-]{20,}['\"]",
     "CWE-798", None),

    ("HC-002", "hardcoded_credentials", "critical",
     "Hardcoded Infura/Alchemy/QuickNode API key in RPC URL",
     r"(?:infura\.io|alchemyapi\.io|alchemy\.com|quicknode\.com)[^\s\"']*[a-zA-Z0-9]{20,}",
     "CWE-798", None),

    # --- Insecure RPC ---
    ("IR-001", "insecure_rpc", "medium",
     "HTTP (not HTTPS) RPC endpoint for blockchain communication",
     r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)[^\s\"']*(?:rpc|eth|node|alchemy|infura|quicknode|ankr|chain)",
     "CWE-319", None),

    # --- Missing Input Validation ---
    ("IV-001", "missing_input_validation", "medium",
     "Address parameter accepted without Ethereum address validation",
     r"(?:address|addr|to|from|recipient|spender)\s*[:=].*(?:string|str|any|\"type\"\s*:\s*\"string\")"
     r"(?!.*(?:isAddress|checksum|validate|ethers\.utils))",
     "CWE-20", None),

    ("IV-002", "missing_input_validation", "medium",
     "Amount/value parameter without bounds checking",
     r"(?:amount|value|quantity|wei|gwei)\s*[:=].*(?:string|str|any|number)"
     r"(?!.*(?:max|min|limit|cap|bound|validate|parse|BigNumber))",
     "CWE-20", None),

    # --- No Gas Limit ---
    ("GL-001", "no_gas_limit", "medium",
     "Transaction without gas limit specification",
     r"(?:sendTransaction|send_transaction)\s*\(\s*\{(?!.*(?:gasLimit|gas_limit|gas\s*:))",
     "CWE-400", None),

    # --- Cross-Tool Escalation ---
    ("CE-001", "cross_tool_escalation", "high",
     "Tool invokes other tools without permission check",
     r"(?:callTool|call_tool|execute_tool|invoke_tool|agent\.run|chain\.invoke)\s*\("
     r"(?!.*(?:permission|auth|check|allowed|restrict))",
     "CWE-284", None),

    # --- Missing Harness ---
    ("MH-001", "missing_harness", "medium",
     "MCP server instantiated without security wrapper/sandbox",
     r"new\s+(?:Server|McpServer)\s*\("
     r"(?!.*(?:sandbox|harness|security|restrict|policy|guard))",
     "CWE-284", None),

    ("MH-002", "missing_harness", "medium",
     "Agent initialized without safety constraints",
     r"(?:initialize_agent|AgentExecutor|create_\w+_agent)\s*\("
     r"(?!.*(?:max_iterations|handle_parsing_errors|callbacks|allowed_tools|max_execution_time))",
     "CWE-284", None),

    # --- Web3-Native: Delegatecall Abuse ---
    ("DC-001", "delegatecall_abuse", "critical",
     "delegatecall to unvalidated address",
     r"\.delegatecall\s*\((?!.*(?:trusted|whitelist|allowed|immutable|onlyOwner))",
     "CWE-829", {".sol"}),

    # --- Web3-Native: Unchecked External Call ---
    ("UC-001", "unchecked_external_call", "high",
     "External call without checking return value",
     r"\.call\s*\{.*?\}\s*\(|\.send\s*\(|\.transfer\s*\("
     r"(?!.*(?:require|assert|if\s*\(|revert))",
     "CWE-252", {".sol"}),

    # --- Web3-Native: Privilege Escalation ---
    ("PE-001", "privilege_escalation", "critical",
     "Module changes owner/modules/handler without timelock",
     r"(?:transferOwnership|addModule|enableModule|setFallbackHandler|setGuard)\s*\("
     r"(?!.*(?:timelock|multisig|onlyOwner|require.*delay|onlyEntryPoint))",
     "CWE-269", {".sol"}),

    # --- Web3-Native: Approval Injection ---
    ("AI-001", "approval_injection", "high",
     "Module sets ERC-20 approvals without whitelisting",
     r"(?:approve|increaseAllowance)\s*\("
     r"(?!.*(?:cap|limit|whitelist|onlyWhitelisted|maxApproval))",
     "CWE-862", {".sol"}),

    # --- Excessive Permissions ---
    ("EP-001", "excessive_permissions", "high",
     "Destructive operation function without scope restriction",
     r"(?:os\.system|subprocess\.(?:run|call|Popen)|exec\(|eval\(|shell\s*=\s*True)"
     r"(?!.*(?:sandbox|restrict|whitelist|allowed_commands|shlex\.quote))",
     "CWE-78", {".py"}),

    # --- State Confusion ---
    ("SC-001", "state_confusion", "medium",
     "Mutable global state shared across tool calls",
     r"(?:global|globals\(\))\s+\w+|(?:shared_state|global_context|session)\s*\[",
     "CWE-362", None),

    # --- Output Handling ---
    ("OH-001", "no_output_validation", "medium",
     "Tool result passed to LLM without sanitization",
     r"(?:function_response|tool_output|result|response)\s*=\s*(?:raw|unsanitized|direct)",
     "CWE-20", None),

    # --- Reentrancy (Solidity) ---
    ("RE-001", "reentrancy", "critical",
     "State change after external call (potential reentrancy)",
     r"\.call\s*\{.*?\}\s*\([^)]*\).*\n.*(?:balances|_balances|mapping)\s*\[",
     "CWE-841", {".sol"}),

    # --- Unprotected selfdestruct ---
    ("SD-001", "selfdestruct_unprotected", "critical",
     "selfdestruct without access control",
     r"selfdestruct\s*\((?!.*(?:onlyOwner|require|auth|modifier))",
     "CWE-284", {".sol"}),
]


# ============================================================
# SCANNING LOGIC
# ============================================================

def get_context_lines(lines: list, line_idx: int, window: int = 2) -> str:
    """Get surrounding lines for context."""
    start = max(0, line_idx - window)
    end = min(len(lines), line_idx + window + 1)
    ctx_lines = []
    for i in range(start, end):
        prefix = ">>>" if i == line_idx else "   "
        ctx_lines.append(f"{prefix} {i+1}: {lines[i].rstrip()}")
    return "\n".join(ctx_lines)


def scan_file(filepath: str, repo_root: str) -> list:
    """Scan a single file for vulnerability patterns.
    Returns list of Finding objects.
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in SOURCE_EXTENSIONS:
        return []

    try:
        size = os.path.getsize(filepath)
        if size > MAX_FILE_SIZE_BYTES:
            return []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except (OSError, PermissionError):
        return []

    lines = content.split("\n")
    rel_path = os.path.relpath(filepath, repo_root)
    findings = []

    for pat_id, category, severity, desc, regex, cwe, file_exts in VULN_PATTERNS:
        # Check if this pattern applies to this file type
        if file_exts is not None and ext not in file_exts:
            continue

        try:
            for match in re.finditer(regex, content, re.IGNORECASE | re.MULTILINE):
                # Find line number
                pos = match.start()
                line_num = content[:pos].count("\n") + 1
                line_idx = line_num - 1
                matched_text = match.group()[:200]  # Truncate long matches
                context = get_context_lines(lines, line_idx)

                findings.append(Finding(
                    pattern_id=pat_id,
                    category=category,
                    severity=severity,
                    description=desc,
                    file=rel_path,
                    line=line_num,
                    matched_text=matched_text,
                    context=context,
                    cwe=cwe,
                ))
        except re.error:
            continue

    return findings


def scan_repo(repo_path: str) -> tuple:
    """Scan all source files in a repo.
    Returns (list[Finding], files_scanned).
    """
    all_findings = []
    files_scanned = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if files_scanned >= MAX_FILES_PER_REPO:
                break
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SOURCE_EXTENSIONS:
                continue
            filepath = os.path.join(root, fname)
            findings = scan_file(filepath, repo_path)
            all_findings.extend(findings)
            files_scanned += 1
        if files_scanned >= MAX_FILES_PER_REPO:
            break

    return all_findings, files_scanned


def compute_risk_score(findings: list) -> tuple:
    """Compute risk score and rating from findings.
    Returns (score, rating).
    """
    severity_weights = {
        "critical": 10.0,
        "high": 7.5,
        "medium": 5.0,
        "low": 2.5,
    }
    if not findings:
        return 0.0, "safe"

    total = sum(severity_weights.get(f.severity, 1.0) for f in findings)
    # Normalize: log-scale to prevent huge repos from dominating
    import math
    score = min(100.0, round(10 * math.log2(1 + total), 1))

    if score >= 80:
        rating = "critical"
    elif score >= 60:
        rating = "high"
    elif score >= 40:
        rating = "medium"
    elif score >= 20:
        rating = "low"
    else:
        rating = "safe"

    return score, rating


# ============================================================
# REPO SELECTION
# ============================================================

def select_repos(catalog: dict, max_repos: int = MAX_REPOS) -> list:
    """Select top repos by stars, filtering out non-tool-servers."""
    repos = catalog.get("repos", [])

    # Filter
    filtered = []
    for r in repos:
        name = r.get("full_name", "")
        if name in EXCLUDE_REPOS:
            continue
        # Must have a known language (has code)
        lang = r.get("language", "Unknown")
        if lang in ("Unknown", "HTML", "Shell", "Dockerfile"):
            continue
        # Must have at least some code (size > 50KB or it's tiny)
        size_kb = r.get("size_kb", 0)
        if size_kb < 10:
            continue
        filtered.append(r)

    # Sort by stars descending
    filtered.sort(key=lambda x: x.get("stars", 0), reverse=True)

    # Take top N
    return filtered[:max_repos]


# ============================================================
# CLONE & SCAN
# ============================================================

def clone_repo(url: str, dest: str, timeout: int = CLONE_TIMEOUT_SECONDS) -> bool:
    """Shallow clone a repo. Returns True on success."""
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--single-branch", url, dest],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        print(f"  Clone failed: {e}")
        return False


def scan_single_repo(repo_info: dict) -> RepoResult:
    """Clone, scan, and clean up a single repo."""
    full_name = repo_info["full_name"]
    url = repo_info["url"]
    stars = repo_info.get("stars", 0)
    catalog_protocol = repo_info.get("protocol", "unknown")
    language = repo_info.get("language", "Unknown")

    result = RepoResult(
        full_name=full_name,
        url=url,
        stars=stars,
        catalog_protocol=catalog_protocol,
        detected_protocol="unknown",
        protocol_confidence=0.0,
        language=language,
    )

    # Create temp directory for clone
    tmp_dir = tempfile.mkdtemp(prefix="scan_")
    clone_path = os.path.join(tmp_dir, full_name.split("/")[-1])

    try:
        print(f"\n{'='*60}")
        print(f"Scanning: {full_name} ({stars} stars, {catalog_protocol})")
        print(f"{'='*60}")

        # Clone
        print(f"  Cloning {url}...")
        t0 = time.time()
        if not clone_repo(url, clone_path):
            print(f"  FAILED to clone {full_name}")
            result.clone_success = False
            result.errors.append("Clone failed")
            return result

        clone_time = time.time() - t0
        print(f"  Cloned in {clone_time:.1f}s")

        # Detect protocol
        detected, confidence = detect_protocol(clone_path)
        result.detected_protocol = detected
        result.protocol_confidence = confidence
        print(f"  Protocol: {detected} (confidence: {confidence:.2f})")

        # Scan for vulnerabilities
        print(f"  Scanning files...")
        t1 = time.time()
        findings, files_scanned = scan_repo(clone_path)
        scan_time = time.time() - t1

        result.files_scanned = files_scanned
        result.scan_time_seconds = round(scan_time, 2)
        result.total_findings = len(findings)
        result.findings = [f.to_dict() for f in findings]

        # Aggregate by severity
        sev_counts = {}
        for f in findings:
            sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
        result.by_severity = sev_counts

        # Aggregate by category
        cat_counts = {}
        for f in findings:
            cat_counts[f.category] = cat_counts.get(f.category, 0) + 1
        result.by_category = cat_counts

        # Risk score
        score, rating = compute_risk_score(findings)
        result.risk_score = score
        result.risk_rating = rating

        print(f"  Files scanned: {files_scanned}")
        print(f"  Findings: {len(findings)}")
        print(f"  By severity: {sev_counts}")
        print(f"  Risk: {score} ({rating})")

    except Exception as e:
        result.errors.append(str(e))
        print(f"  ERROR: {e}")

    finally:
        # Clean up clone
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    return result


# ============================================================
# AGGREGATION
# ============================================================

def aggregate_results(results: list) -> dict:
    """Produce aggregate statistics across all repos."""
    total_findings = sum(r.total_findings for r in results)
    successful = [r for r in results if r.clone_success]

    # Overall severity distribution
    overall_severity = {}
    for r in successful:
        for sev, count in r.by_severity.items():
            overall_severity[sev] = overall_severity.get(sev, 0) + count

    # Overall category distribution
    overall_category = {}
    for r in successful:
        for cat, count in r.by_category.items():
            overall_category[cat] = overall_category.get(cat, 0) + count

    # Per-protocol stats
    protocol_stats = {}
    for r in successful:
        proto = r.detected_protocol
        if proto not in protocol_stats:
            protocol_stats[proto] = {
                "repos": 0,
                "total_findings": 0,
                "avg_findings": 0,
                "avg_risk_score": 0,
                "by_severity": {},
                "by_category": {},
            }
        ps = protocol_stats[proto]
        ps["repos"] += 1
        ps["total_findings"] += r.total_findings
        ps["avg_risk_score"] += r.risk_score
        for sev, count in r.by_severity.items():
            ps["by_severity"][sev] = ps["by_severity"].get(sev, 0) + count
        for cat, count in r.by_category.items():
            ps["by_category"][cat] = ps["by_category"].get(cat, 0) + count

    # Compute averages
    for proto, ps in protocol_stats.items():
        if ps["repos"] > 0:
            ps["avg_findings"] = round(ps["total_findings"] / ps["repos"], 1)
            ps["avg_risk_score"] = round(ps["avg_risk_score"] / ps["repos"], 1)

    # Top risky repos
    risk_ranking = sorted(
        [(r.full_name, r.risk_score, r.risk_rating, r.total_findings,
          r.detected_protocol, r.stars)
         for r in successful],
        key=lambda x: x[1],
        reverse=True,
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "repos_attempted": len(results),
        "repos_successful": len(successful),
        "repos_failed": len(results) - len(successful),
        "total_findings": total_findings,
        "overall_by_severity": overall_severity,
        "overall_by_category": overall_category,
        "cross_protocol_stats": protocol_stats,
        "risk_ranking": [
            {"repo": name, "risk_score": score, "risk_rating": rating,
             "findings": findings, "protocol": proto, "stars": stars}
            for name, score, rating, findings, proto, stars in risk_ranking
        ],
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Paper 2: Real Web3 Agent Tool Server Scanner")
    print("=" * 70)

    # 1. Read catalog
    catalog_path = os.path.abspath(CATALOG_PATH)
    print(f"\nReading catalog: {catalog_path}")
    with open(catalog_path, "r") as f:
        catalog = json.load(f)
    print(f"Total repos in catalog: {catalog['total_repos']}")

    # 2. Select top repos
    selected = select_repos(catalog, MAX_REPOS)
    print(f"\nSelected {len(selected)} repos for scanning:")
    for i, r in enumerate(selected):
        print(f"  {i+1:2d}. {r['stars']:>5} stars | {r['protocol']:>12} | "
              f"{r['language']:>12} | {r['full_name']}")

    # 3. Scan each repo
    results = []
    start_time = time.time()

    for i, repo_info in enumerate(selected):
        print(f"\n[{i+1}/{len(selected)}]", end="")
        result = scan_single_repo(repo_info)
        results.append(result)

        # Rate limit between clones
        if i < len(selected) - 1:
            time.sleep(CLONE_DELAY_SECONDS)

    total_time = time.time() - start_time

    # 4. Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    aggregate = aggregate_results(results)
    aggregate["total_scan_time_seconds"] = round(total_time, 1)

    # Build output
    output = {
        "metadata": {
            "scan_type": "real_server_scan",
            "timestamp": aggregate["timestamp"],
            "catalog_source": catalog_path,
            "repos_attempted": aggregate["repos_attempted"],
            "repos_successful": aggregate["repos_successful"],
            "repos_failed": aggregate["repos_failed"],
            "total_scan_time_seconds": aggregate["total_scan_time_seconds"],
        },
        "summary": {
            "total_findings": aggregate["total_findings"],
            "overall_by_severity": aggregate["overall_by_severity"],
            "overall_by_category": aggregate["overall_by_category"],
        },
        "cross_protocol_comparison": aggregate["cross_protocol_stats"],
        "risk_ranking": aggregate["risk_ranking"],
        "repo_results": [r.to_dict() for r in results],
    }

    # 5. Save results
    output_path = os.path.abspath(OUTPUT_PATH)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    # 6. Print summary
    print("\n" + "=" * 70)
    print("SCAN SUMMARY")
    print("=" * 70)
    print(f"Repos scanned:     {aggregate['repos_successful']}/{aggregate['repos_attempted']}")
    print(f"Total findings:    {aggregate['total_findings']}")
    print(f"Total time:        {total_time:.1f}s")
    print(f"\nBy severity:")
    for sev in ["critical", "high", "medium", "low"]:
        count = aggregate["overall_by_severity"].get(sev, 0)
        print(f"  {sev:>10}: {count}")
    print(f"\nBy category:")
    for cat, count in sorted(aggregate["overall_by_category"].items(),
                              key=lambda x: x[1], reverse=True):
        print(f"  {cat:>30}: {count}")
    print(f"\nCross-protocol comparison:")
    for proto, stats in aggregate["cross_protocol_stats"].items():
        print(f"  {proto:>12}: {stats['repos']} repos, "
              f"{stats['total_findings']} findings, "
              f"avg risk={stats['avg_risk_score']}")
    print(f"\nRisk ranking (top 5):")
    for entry in aggregate["risk_ranking"][:5]:
        print(f"  {entry['risk_score']:>5.1f} ({entry['risk_rating']:>8}) | "
              f"{entry['findings']:>3} findings | {entry['repo']}")


if __name__ == "__main__":
    main()
