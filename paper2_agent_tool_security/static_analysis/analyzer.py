"""
Static Analyzer for AI Agent Tool Interfaces
==============================================
Production-grade static analyzer that:
  - Takes a local repo path
  - Detects which protocol family it belongs to (MCP/OpenAI/LangChain/Web3)
  - Runs protocol-specific vulnerability patterns
  - Outputs structured findings with severity, CWE mapping, location
  - Computes risk score
  - Supports batch scanning of multiple repos

Usage:
    # Single repo
    python analyzer.py /path/to/repo

    # Batch scan from catalog
    python analyzer.py --batch /path/to/repos/ --catalog server_catalog.json

    # Output as JSON
    python analyzer.py /path/to/repo --format json --output findings.json

    # Filter by severity
    python analyzer.py /path/to/repo --min-severity high

Requirements:
    Python 3.10+ (standard library only)
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


# ============================================================
# ENUMS & DATA STRUCTURES
# ============================================================

class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

    @property
    def score(self) -> float:
        return {
            "critical": 10.0,
            "high": 7.5,
            "medium": 5.0,
            "low": 2.5,
            "info": 0.5,
        }[self.value]

    def __ge__(self, other: "Severity") -> bool:
        return self.score >= other.score

    def __gt__(self, other: "Severity") -> bool:
        return self.score > other.score

    def __le__(self, other: "Severity") -> bool:
        return self.score <= other.score

    def __lt__(self, other: "Severity") -> bool:
        return self.score < other.score


class Protocol(Enum):
    MCP = "mcp"
    OPENAI = "openai"
    LANGCHAIN = "langchain"
    WEB3_NATIVE = "web3_native"
    UNKNOWN = "unknown"


class AttackSurface(Enum):
    TOOL_DEFINITION = "S1_tool_definition"
    INPUT_CONSTRUCTION = "S2_input_construction"
    EXECUTION = "S3_execution"
    OUTPUT_HANDLING = "S4_output_handling"
    CROSS_TOOL = "S5_cross_tool"


@dataclass
class VulnPattern:
    """A regex-based vulnerability detection pattern."""
    id: str
    protocols: list[Protocol]     # Which protocols this applies to
    severity: Severity
    attack_surface: AttackSurface
    category: str                 # Vulnerability category name
    description: str
    pattern: str                  # Regex pattern
    cwe: str                      # CWE identifier
    owasp_llm: str = ""           # OWASP LLM Top 10 reference
    file_extensions: list[str] = field(
        default_factory=lambda: [".ts", ".js", ".py", ".sol", ".tsx", ".jsx"]
    )
    false_positive_hints: list[str] = field(default_factory=list)
    remediation: str = ""


@dataclass
class Finding:
    """A single vulnerability finding in a specific file."""
    pattern_id: str
    protocol: str
    category: str
    severity: str
    attack_surface: str
    description: str
    cwe: str
    file_path: str                # Relative to repo root
    line_number: int
    matched_text: str             # The actual text that matched
    context: str                  # Surrounding lines for human review
    owasp_llm: str = ""
    remediation: str = ""
    confidence: str = "medium"    # low, medium, high
    false_positive: bool = False


@dataclass
class RepoScanResult:
    """Complete scan result for a single repository."""
    repo_path: str
    repo_name: str
    detected_protocol: str
    protocol_confidence: float
    scan_time_seconds: float = 0.0
    files_scanned: int = 0
    files_skipped: int = 0
    total_findings: int = 0
    risk_score: float = 0.0
    risk_rating: str = "unknown"   # critical, high, medium, low, safe
    findings: list = field(default_factory=list)
    by_severity: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    by_attack_surface: dict = field(default_factory=dict)
    harness_present: bool = False
    harness_features: list = field(default_factory=list)
    errors: list = field(default_factory=list)


@dataclass
class BatchScanResult:
    """Aggregated results across multiple repositories."""
    timestamp: str = ""
    repos_scanned: int = 0
    total_findings: int = 0
    avg_risk_score: float = 0.0
    results: list = field(default_factory=list)
    cross_protocol_stats: dict = field(default_factory=dict)


# ============================================================
# PROTOCOL DETECTION
# ============================================================

# File-level indicators for protocol classification
PROTOCOL_INDICATORS: dict[Protocol, list[dict]] = {
    Protocol.MCP: [
        {"pattern": r"@modelcontextprotocol|from mcp import|mcp\.Server|McpServer",
         "weight": 10.0, "where": "content"},
        {"pattern": r"server\.tool\(|@tool|mcp_server",
         "weight": 5.0, "where": "content"},
        {"pattern": r"mcp", "weight": 3.0, "where": "filename"},
        {"pattern": r"\"@modelcontextprotocol/sdk\"",
         "weight": 8.0, "where": "content"},
    ],
    Protocol.OPENAI: [
        {"pattern": r"import openai|from openai|openai\.chat",
         "weight": 8.0, "where": "content"},
        {"pattern": r"function_call|tool_choice|\"functions\"\s*:|tools\s*=\s*\[",
         "weight": 7.0, "where": "content"},
        {"pattern": r"role.*assistant.*tool_calls|role.*tool.*content",
         "weight": 9.0, "where": "content"},
        {"pattern": r"openai|gpt", "weight": 2.0, "where": "filename"},
    ],
    Protocol.LANGCHAIN: [
        {"pattern": r"from langchain|import langchain|from crewai|from autogen",
         "weight": 10.0, "where": "content"},
        {"pattern": r"@tool|BaseTool|StructuredTool|initialize_agent|AgentExecutor",
         "weight": 7.0, "where": "content"},
        {"pattern": r"langchain|crewai|autogen", "weight": 3.0, "where": "filename"},
        {"pattern": r"PromptTemplate|ChatPromptTemplate|SystemMessage",
         "weight": 4.0, "where": "content"},
    ],
    Protocol.WEB3_NATIVE: [
        {"pattern": r"pragma solidity|// SPDX-License-Identifier",
         "weight": 10.0, "where": "content"},
        {"pattern": r"ISafe|IModule|IGuard|ModuleManager|GnosisSafe",
         "weight": 8.0, "where": "content"},
        {"pattern": r"ERC4337|IEntryPoint|UserOperation|IPlugin|IValidator",
         "weight": 9.0, "where": "content"},
        {"pattern": r"delegatecall|execTransactionFromModule",
         "weight": 6.0, "where": "content"},
        {"pattern": r"\.sol$", "weight": 5.0, "where": "extension"},
    ],
}


def detect_protocol(repo_path: str,
                    max_files: int = 200) -> tuple[Protocol, float]:
    """Detect the protocol family of a repository.

    Scans source files for protocol-specific indicators and returns
    the best-matching protocol with a confidence score (0-1).

    Args:
        repo_path: Path to the repository root.
        max_files: Maximum number of files to scan for detection.

    Returns:
        Tuple of (Protocol, confidence_score).
    """
    scores: dict[Protocol, float] = {p: 0.0 for p in Protocol
                                     if p != Protocol.UNKNOWN}
    files_checked = 0
    source_extensions = {".ts", ".js", ".py", ".sol", ".tsx", ".jsx",
                         ".mts", ".mjs"}

    for root, dirs, files in os.walk(repo_path):
        # Skip common non-source directories
        dirs[:] = [d for d in dirs if d not in {
            "node_modules", ".git", "__pycache__", "venv", ".venv",
            "dist", "build", "out", ".next", "coverage", "artifacts",
            "cache", "typechain-types",
        }]

        for fname in files:
            if files_checked >= max_files:
                break

            ext = os.path.splitext(fname)[1].lower()
            if ext not in source_extensions:
                continue

            filepath = os.path.join(root, fname)
            files_checked += 1

            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read(10_000)  # First 10KB is enough
            except (OSError, PermissionError):
                continue

            for proto, indicators in PROTOCOL_INDICATORS.items():
                for ind in indicators:
                    where = ind["where"]
                    if where == "content":
                        if re.search(ind["pattern"], content, re.IGNORECASE):
                            scores[proto] += ind["weight"]
                    elif where == "filename":
                        if re.search(ind["pattern"], fname, re.IGNORECASE):
                            scores[proto] += ind["weight"]
                    elif where == "extension":
                        if re.search(ind["pattern"], ext):
                            scores[proto] += ind["weight"]

        if files_checked >= max_files:
            break

    # Also check package.json / pyproject.toml for dependencies
    pkg_json = os.path.join(repo_path, "package.json")
    if os.path.exists(pkg_json):
        try:
            with open(pkg_json, "r") as f:
                pkg = json.load(f)
            deps = json.dumps(pkg.get("dependencies", {}))
            deps += json.dumps(pkg.get("devDependencies", {}))
            if "@modelcontextprotocol" in deps:
                scores[Protocol.MCP] += 15.0
            if "openai" in deps:
                scores[Protocol.OPENAI] += 10.0
            if "langchain" in deps:
                scores[Protocol.LANGCHAIN] += 10.0
        except (json.JSONDecodeError, OSError):
            pass

    pyproject = os.path.join(repo_path, "pyproject.toml")
    requirements = os.path.join(repo_path, "requirements.txt")
    for dep_file in [pyproject, requirements]:
        if os.path.exists(dep_file):
            try:
                with open(dep_file, "r") as f:
                    content = f.read()
                if "mcp" in content.lower():
                    scores[Protocol.MCP] += 10.0
                if "openai" in content:
                    scores[Protocol.OPENAI] += 8.0
                if "langchain" in content:
                    scores[Protocol.LANGCHAIN] += 10.0
            except OSError:
                pass

    # Determine winner
    total = sum(scores.values())
    if total == 0:
        return Protocol.UNKNOWN, 0.0

    best_proto = max(scores, key=lambda p: scores[p])
    confidence = scores[best_proto] / total if total > 0 else 0.0
    return best_proto, round(confidence, 3)


# ============================================================
# VULNERABILITY PATTERNS -- CROSS-PROTOCOL
# ============================================================

VULN_PATTERNS: list[VulnPattern] = [
    # ----------------------------------------------------------
    # S1: Tool Definition Surface
    # ----------------------------------------------------------
    VulnPattern(
        id="S1-TP-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.TOOL_DEFINITION,
        category="tool_poisoning",
        description="Tool/function description contains directive instructions "
                    "for the LLM (e.g., 'always', 'must', 'never', 'ignore')",
        pattern=r"(?:description|desc)\s*[:=]\s*[\"'][^\"']*"
                r"(?:always|must|never|ignore\s+previous|forget|override|"
                r"system\s*prompt|do\s+not\s+ask|without\s+confirm)[^\"']*[\"']",
        cwe="CWE-913",
        owasp_llm="LLM01",
        remediation="Remove directive language from tool descriptions. "
                    "Descriptions should be declarative, not imperative.",
    ),
    VulnPattern(
        id="S1-TP-002",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.TOOL_DEFINITION,
        category="tool_poisoning",
        description="Tool description references other tools by name, "
                    "potentially directing the LLM to chain calls",
        pattern=r"(?:description|desc)\s*[:=]\s*[\"'][^\"']*"
                r"(?:first\s+call|then\s+use|before\s+running|after\s+calling|"
                r"requires?\s+calling)[^\"']*[\"']",
        cwe="CWE-913",
        owasp_llm="LLM01",
        remediation="Tools should be independently usable. Cross-tool "
                    "dependencies should be enforced programmatically, not "
                    "via description text.",
    ),
    VulnPattern(
        id="S1-EP-001",
        protocols=[Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.TOOL_DEFINITION,
        category="excessive_permissions",
        description="Function/tool name indicates destructive operation without "
                    "scope restriction (execute, delete, admin, sudo, transfer)",
        pattern=r"(?:name|function)\s*[:=]\s*[\"']"
                r"(?:execute|delete|write|admin|sudo|drop|remove|transfer|"
                r"send|withdraw|approve)[\"']",
        cwe="CWE-250",
        owasp_llm="LLM06",
        remediation="Scope destructive operations by resource type and add "
                    "confirmation requirements.",
    ),
    VulnPattern(
        id="S1-SL-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.TOOL_DEFINITION,
        category="skill_scope_leak",
        description="Tool description exposes internal implementation details "
                    "(database, schema, endpoint, internal)",
        pattern=r"(?:description|desc)\s*[:=]\s*[\"'][^\"']*"
                r"(?:internal|implementation|database|schema|table|column|"
                r"endpoint|api[_\s]?key|password)[^\"']*[\"']",
        cwe="CWE-200",
        remediation="Redact implementation details from tool descriptions. "
                    "Only expose the functional interface.",
    ),

    # ----------------------------------------------------------
    # S2: Input Construction Surface
    # ----------------------------------------------------------
    VulnPattern(
        id="S2-PI-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.CRITICAL,
        attack_surface=AttackSurface.INPUT_CONSTRUCTION,
        category="prompt_injection",
        description="User input interpolated into tool description, prompt "
                    "template, or system message at runtime",
        pattern=r"(?:description|prompt|system|template)\s*[:=]\s*"
                r"(?:f\"|f'|.*\.format\(|.*%s|.*\+\s*(?:user|input|data|query|"
                r"request|msg))",
        cwe="CWE-74",
        owasp_llm="LLM01",
        remediation="Never interpolate user input into tool descriptions or "
                    "system prompts. Pass user data as structured parameters.",
    ),
    VulnPattern(
        id="S2-IV-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.INPUT_CONSTRUCTION,
        category="missing_input_validation",
        description="Address parameter accepted as string without Ethereum "
                    "address validation",
        pattern=r"(?:address|addr|to|from|recipient|spender)\s*[:=].*"
                r"(?:string|str|any|\"type\"\s*:\s*\"string\")"
                r"(?!.*(?:isAddress|checksum|0x[a-fA-F0-9]{40}|validate|"
                r"ethers\.utils))",
        cwe="CWE-20",
        remediation="Validate Ethereum addresses using checksum verification "
                    "(e.g., ethers.isAddress()) before processing.",
    ),
    VulnPattern(
        id="S2-IV-002",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN,
                   Protocol.WEB3_NATIVE],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.INPUT_CONSTRUCTION,
        category="missing_input_validation",
        description="Amount/value parameter without bounds checking or "
                    "numeric validation",
        pattern=r"(?:amount|value|quantity|wei|gwei)\s*[:=].*"
                r"(?:string|str|any|number)"
                r"(?!.*(?:max|min|limit|cap|bound|validate|parse|BigNumber))",
        cwe="CWE-20",
        remediation="Validate numeric inputs: check for negative values, "
                    "overflow, and enforce per-transaction/per-session limits.",
    ),

    # ----------------------------------------------------------
    # S3: Execution Surface
    # ----------------------------------------------------------
    VulnPattern(
        id="S3-PKE-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.CRITICAL,
        attack_surface=AttackSurface.EXECUTION,
        category="private_key_exposure",
        description="Private key, mnemonic, or seed phrase accepted as tool "
                    "parameter or present in tool code",
        pattern=r"(?:private[_\s]?key|secret[_\s]?key|mnemonic|seed[_\s]?phrase"
                r"|signing[_\s]?key)",
        cwe="CWE-200",
        remediation="Never accept private keys as parameters. Use a secure "
                    "signing service (e.g., KMS, hardware wallet, session keys "
                    "with limited scope).",
    ),
    VulnPattern(
        id="S3-PKE-002",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.CRITICAL,
        attack_surface=AttackSurface.EXECUTION,
        category="private_key_exposure",
        description="Wallet signing operation without user confirmation flow",
        pattern=r"(?:signTransaction|signMessage|eth_sign|personal_sign|"
                r"sign_transaction|signTypedData)"
                r"(?!.*(?:confirm|approval|prompt|user|consent|review))",
        cwe="CWE-862",
        remediation="All signing operations must require explicit user "
                    "confirmation with clear display of what is being signed.",
    ),
    VulnPattern(
        id="S3-HC-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.CRITICAL,
        attack_surface=AttackSurface.EXECUTION,
        category="hardcoded_credentials",
        description="Hardcoded API keys, passwords, secrets, or tokens",
        pattern=r"(?:api[_\s]?key|password|secret|token|auth)\s*[=:]\s*"
                r"['\"][a-zA-Z0-9_\-]{16,}['\"]",
        cwe="CWE-798",
        remediation="Use environment variables or a secret manager. Never "
                    "hardcode credentials in source code.",
    ),
    VulnPattern(
        id="S3-CE-001",
        protocols=[Protocol.LANGCHAIN],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.EXECUTION,
        category="command_injection",
        description="Tool executes OS commands or shell operations without "
                    "sandboxing",
        pattern=r"(?:os\.system|subprocess\.(?:run|call|Popen)|exec\(|eval\(|"
                r"shell\s*=\s*True)"
                r"(?!.*(?:sandbox|restrict|whitelist|allowed_commands|"
                r"shlex\.quote))",
        cwe="CWE-78",
        owasp_llm="LLM06",
        remediation="Avoid shell execution. If required, use allowlists, "
                    "input escaping (shlex.quote), and process sandboxing.",
    ),
    VulnPattern(
        id="S3-UA-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN,
                   Protocol.WEB3_NATIVE],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.EXECUTION,
        category="unlimited_approval",
        description="Constructs unlimited token approval (MaxUint256 or "
                    "equivalent)",
        pattern=r"(?:MaxUint256|type\(uint256\)\.max|2\s*\*\*\s*256|"
                r"0xf{64}|maxApproval|UNLIMITED|uint256\(-1\)|MAX_UINT)",
        cwe="CWE-250",
        remediation="Cap token approvals to the exact amount needed for the "
                    "transaction. Never use unlimited approvals.",
    ),
    VulnPattern(
        id="S3-TV-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.EXECUTION,
        category="tx_validation_missing",
        description="Transaction constructed and sent without validating "
                    "target contract or parameters",
        pattern=r"(?:sendTransaction|send_transaction|sendRawTransaction)"
                r"\s*\("
                r"(?!.*(?:whitelist|allowlist|valid|check|verify|approved))",
        cwe="CWE-345",
        remediation="Validate transaction targets against a whitelist of "
                    "approved contracts. Check all parameters before signing.",
    ),
    VulnPattern(
        id="S3-IR-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.EXECUTION,
        category="insecure_rpc",
        description="HTTP (not HTTPS) RPC endpoint used for blockchain "
                    "communication",
        pattern=r"http://(?!localhost|127\.0\.0\.1|0\.0\.0\.0)"
                r"[^\s\"']*(?:rpc|eth|node|alchemy|infura|quicknode|ankr)",
        cwe="CWE-319",
        remediation="Always use HTTPS for RPC endpoints. HTTP exposes "
                    "transaction data and API keys to network sniffers.",
    ),
    VulnPattern(
        id="S3-MH-001",
        protocols=[Protocol.MCP],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.EXECUTION,
        category="missing_harness",
        description="MCP server instantiated without security wrapper, "
                    "sandbox, or policy enforcement",
        pattern=r"new\s+(?:Server|McpServer)\s*\("
                r"(?!.*(?:sandbox|harness|security|restrict|policy|guard))",
        cwe="CWE-284",
        remediation="Wrap MCP server instantiation with a security harness "
                    "that enforces tool-level permissions and rate limits.",
    ),
    VulnPattern(
        id="S3-MH-002",
        protocols=[Protocol.LANGCHAIN],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.EXECUTION,
        category="missing_harness",
        description="Agent initialized without safety constraints "
                    "(max_iterations, callbacks, allowed_tools)",
        pattern=r"(?:initialize_agent|AgentExecutor|create_\w+_agent)\s*\("
                r"(?!.*(?:max_iterations|handle_parsing_errors|callbacks|"
                r"allowed_tools|max_execution_time))",
        cwe="CWE-284",
        remediation="Set max_iterations, handle_parsing_errors, and restrict "
                    "allowed_tools when initializing agents.",
    ),
    # Web3-native Solidity patterns
    VulnPattern(
        id="S3-DC-001",
        protocols=[Protocol.WEB3_NATIVE],
        severity=Severity.CRITICAL,
        attack_surface=AttackSurface.EXECUTION,
        category="delegatecall_abuse",
        description="delegatecall to user-supplied or unvalidated address",
        pattern=r"\.delegatecall\s*\("
                r"(?!.*(?:trusted|whitelist|allowed|immutable|onlyOwner))",
        cwe="CWE-829",
        file_extensions=[".sol"],
        remediation="Restrict delegatecall targets to immutable, audited "
                    "addresses. Never delegatecall to user-supplied addresses.",
    ),
    VulnPattern(
        id="S3-UC-001",
        protocols=[Protocol.WEB3_NATIVE],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.EXECUTION,
        category="unchecked_external_call",
        description="External call (.call, .send, .transfer) without checking "
                    "return value",
        pattern=r"\.call\s*\{.*?\}\s*\(|\.send\s*\(|\.transfer\s*\("
                r"(?!.*(?:require|assert|if\s*\(!?))",
        cwe="CWE-252",
        file_extensions=[".sol"],
        remediation="Always check return values of external calls. Use "
                    "require() to revert on failure.",
    ),
    VulnPattern(
        id="S3-PE-001",
        protocols=[Protocol.WEB3_NATIVE],
        severity=Severity.CRITICAL,
        attack_surface=AttackSurface.EXECUTION,
        category="privilege_escalation",
        description="Module can change owner, add modules, or set fallback "
                    "handler without timelock/multisig",
        pattern=r"(?:transferOwnership|addModule|enableModule|"
                r"setFallbackHandler|setGuard)\s*\("
                r"(?!.*(?:timelock|multisig|onlyOwner|require.*delay|"
                r"onlyEntryPoint))",
        cwe="CWE-269",
        file_extensions=[".sol"],
        remediation="Gate critical state changes behind timelock and/or "
                    "multi-signature requirements.",
    ),
    VulnPattern(
        id="S3-AI-001",
        protocols=[Protocol.WEB3_NATIVE],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.EXECUTION,
        category="approval_injection",
        description="Module can set ERC-20 approvals on behalf of the "
                    "safe/account without whitelisting",
        pattern=r"(?:approve|increaseAllowance)\s*\("
                r"(?!.*(?:cap|limit|whitelist|onlyWhitelisted|maxApproval))",
        cwe="CWE-862",
        file_extensions=[".sol"],
        remediation="Whitelist approved spender addresses and cap approval "
                    "amounts.",
    ),

    # ----------------------------------------------------------
    # S4: Output Handling Surface
    # ----------------------------------------------------------
    VulnPattern(
        id="S4-NV-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.OUTPUT_HANDLING,
        category="no_output_validation",
        description="Tool/function result passed directly to LLM without "
                    "sanitization or validation",
        pattern=r"(?:function_response|tool_output|result|response)\s*=\s*"
                r"(?:raw|unsanitized|direct)|"
                r"messages\.append\s*\(\s*\{.*?"
                r"(?:role.*tool|function.*content)"
                r"(?!.*(?:sanitize|validate|filter|escape|truncate))",
        cwe="CWE-116",
        remediation="Sanitize tool outputs before including in LLM context. "
                    "Filter prompt injection patterns, truncate length, and "
                    "redact sensitive data.",
    ),
    VulnPattern(
        id="S4-DL-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.OUTPUT_HANDLING,
        category="data_leakage",
        description="Tool returns raw internal data structures (DB rows, "
                    "full API responses, error traces)",
        pattern=r"return\s+(?:cursor\.fetchall|response\.json\(\)|"
                r"raw_data|internal_state|self\._|traceback\.|"
                r"str\(e\)|repr\(e\))",
        cwe="CWE-200",
        remediation="Map internal data to a response schema. Never return "
                    "raw database results, full API responses, or stack traces.",
    ),

    # ----------------------------------------------------------
    # S5: Cross-Tool Surface
    # ----------------------------------------------------------
    VulnPattern(
        id="S5-CE-001",
        protocols=[Protocol.MCP, Protocol.LANGCHAIN],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.CROSS_TOOL,
        category="tool_chain_escalation",
        description="Tool can invoke other tools or agents without permission "
                    "check or access control",
        pattern=r"(?:callTool|call_tool|execute_tool|invoke_tool|agent\.run|"
                r"chain\.invoke|tool\.run)\s*\("
                r"(?!.*(?:permission|auth|check|allowed|restrict|whitelist))",
        cwe="CWE-284",
        owasp_llm="LLM06",
        remediation="Implement per-tool access control. Tools should not be "
                    "able to invoke other tools without explicit authorization.",
    ),
    VulnPattern(
        id="S5-SC-001",
        protocols=[Protocol.OPENAI, Protocol.LANGCHAIN],
        severity=Severity.MEDIUM,
        attack_surface=AttackSurface.CROSS_TOOL,
        category="state_confusion",
        description="Mutable global state shared across tool/function "
                    "invocations without isolation",
        pattern=r"(?:global\s+\w+|globals\(\)|shared_state|global_context|"
                r"session_data)\s*(?:\[|\.(?:update|append|set))",
        cwe="CWE-362",
        remediation="Isolate state between tool invocations. Use immutable "
                    "context objects or per-invocation namespaces.",
    ),
    VulnPattern(
        id="S5-MH-003",
        protocols=[Protocol.WEB3_NATIVE],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.CROSS_TOOL,
        category="missing_harness",
        description="Module executes transactions without guard, sentinel, "
                    "or spending limit",
        pattern=r"(?:execTransaction|executeCall|execute|"
                r"execTransactionFromModule)\s*\("
                r"(?!.*(?:guard|sentinel|limit|nonce|signature|checkSignatures))",
        cwe="CWE-284",
        file_extensions=[".sol"],
        remediation="Install a transaction guard that enforces per-transaction "
                    "and per-period spending limits.",
    ),

    # ----------------------------------------------------------
    # Web3-Specific (cross-surface)
    # ----------------------------------------------------------
    VulnPattern(
        id="W3-MEV-001",
        protocols=[Protocol.MCP, Protocol.OPENAI, Protocol.LANGCHAIN,
                   Protocol.WEB3_NATIVE],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.EXECUTION,
        category="mev_exploitation",
        description="Swap/trade constructed without slippage protection, "
                    "deadline, or private relay",
        pattern=r"(?:swap|exchange|trade|swapExact)"
                r"(?!.*(?:slippage|deadline|minOutput|private|flashbots|"
                r"maxSlippage|minAmountOut|amountOutMin))",
        cwe="CWE-200",
        remediation="Set explicit slippage tolerance (e.g., 0.5-1%), "
                    "transaction deadline, and use a private transaction relay.",
    ),
    VulnPattern(
        id="W3-EP-001",
        protocols=[Protocol.WEB3_NATIVE],
        severity=Severity.HIGH,
        attack_surface=AttackSurface.EXECUTION,
        category="excessive_permissions",
        description="Module uses DELEGATECALL operation type in Safe "
                    "transaction (full storage access)",
        pattern=r"Enum\.Operation\.DelegateCall|operation\s*==?\s*1",
        cwe="CWE-250",
        file_extensions=[".sol"],
        remediation="Avoid DelegateCall operation type unless absolutely "
                    "necessary. Prefer Call (operation=0).",
    ),
]


# ============================================================
# HARNESS DETECTION
# ============================================================

HARNESS_INDICATORS: list[dict] = [
    {
        "feature": "permission_scoping",
        "pattern": r"(?:permission|scope|capability|allowed_tools|"
                   r"tool_permissions|access_control)",
        "description": "Tool-level permission scoping",
    },
    {
        "feature": "execution_sandbox",
        "pattern": r"(?:sandbox|isolat|container|wasm|seccomp|"
                   r"resource_limit|max_execution_time|timeout)",
        "description": "Execution sandboxing or resource limits",
    },
    {
        "feature": "transaction_validation",
        "pattern": r"(?:whitelist|allowlist|spending_limit|tx_limit|"
                   r"max_amount|approved_contracts|gas_limit)",
        "description": "Transaction validation and spending limits",
    },
    {
        "feature": "audit_logging",
        "pattern": r"(?:audit|log_tool|log_invocation|tool_log|"
                   r"event\s+\w+Executed|emit\s+\w+Called)",
        "description": "Audit logging of tool invocations",
    },
    {
        "feature": "human_in_loop",
        "pattern": r"(?:confirm|approval|consent|review|human_in|"
                   r"require_confirmation|ask_user|prompt_user)",
        "description": "Human-in-the-loop confirmation",
    },
    {
        "feature": "output_sanitization",
        "pattern": r"(?:sanitiz|filter_output|strip_injection|"
                   r"escape_output|clean_response|redact)",
        "description": "Output sanitization before LLM consumption",
    },
]


def detect_harness(repo_path: str, max_files: int = 200) -> tuple[bool, list[str]]:
    """Detect whether a repository implements security harness features.

    Returns:
        Tuple of (harness_present, list_of_detected_features).
    """
    features_found = set()
    source_extensions = {".ts", ".js", ".py", ".sol", ".tsx", ".jsx"}

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in {
            "node_modules", ".git", "__pycache__", "venv", ".venv",
            "dist", "build", "out",
        }]

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in source_extensions:
                continue

            filepath = os.path.join(root, fname)
            try:
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, PermissionError):
                continue

            for indicator in HARNESS_INDICATORS:
                if re.search(indicator["pattern"], content, re.IGNORECASE):
                    features_found.add(indicator["feature"])

    harness_present = len(features_found) >= 2  # At least 2 features = harness
    return harness_present, sorted(features_found)


# ============================================================
# SCANNER ENGINE
# ============================================================

class StaticAnalyzer:
    """Core static analysis engine."""

    # Files/directories to skip
    SKIP_DIRS = {
        "node_modules", ".git", "__pycache__", "venv", ".venv",
        "dist", "build", "out", ".next", "coverage", "artifacts",
        "cache", "typechain-types", "test", "tests", "__tests__",
        "spec", "mock", "mocks", "fixture", "fixtures",
    }

    # Max file size to scan (bytes)
    MAX_FILE_SIZE = 500_000  # 500KB

    # Context lines before and after match
    CONTEXT_LINES = 3

    def __init__(self, min_severity: Severity = Severity.INFO,
                 include_tests: bool = False, verbose: bool = True):
        self.min_severity = min_severity
        self.include_tests = include_tests
        self.verbose = verbose

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"  {msg}", file=sys.stderr)

    def _should_skip_dir(self, dirname: str) -> bool:
        if dirname in self.SKIP_DIRS and not self.include_tests:
            return True
        if dirname.startswith("."):
            return True
        return False

    def _get_context(self, lines: list[str], line_idx: int) -> str:
        """Extract surrounding context lines."""
        start = max(0, line_idx - self.CONTEXT_LINES)
        end = min(len(lines), line_idx + self.CONTEXT_LINES + 1)
        return "\n".join(lines[start:end])

    def _check_false_positive(self, match_text: str, context: str,
                              pattern: VulnPattern) -> bool:
        """Heuristic false positive detection."""
        lower_ctx = context.lower()

        # Comment-only matches
        stripped = match_text.strip()
        if stripped.startswith("//") or stripped.startswith("#") or \
           stripped.startswith("*") or stripped.startswith("/*"):
            return True

        # Test/mock/example files
        if any(kw in lower_ctx for kw in ["mock", "test", "example",
                                           "sample", "demo", "fixture"]):
            return True

        # Check pattern-specific hints
        for hint in pattern.false_positive_hints:
            if hint.lower() in lower_ctx:
                return True

        return False

    def scan_file(self, filepath: str, repo_root: str,
                  protocol: Protocol) -> list[Finding]:
        """Scan a single file for vulnerability patterns.

        Args:
            filepath: Absolute path to the file.
            repo_root: Root directory of the repository.
            protocol: Detected protocol family.

        Returns:
            List of Finding objects.
        """
        # Check file size
        try:
            size = os.path.getsize(filepath)
            if size > self.MAX_FILE_SIZE:
                return []
        except OSError:
            return []

        # Read file
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
        except (OSError, PermissionError):
            return []

        lines = content.split("\n")
        ext = os.path.splitext(filepath)[1].lower()
        rel_path = os.path.relpath(filepath, repo_root)
        findings = []

        for pattern in VULN_PATTERNS:
            # Check protocol applicability
            if protocol not in pattern.protocols and \
               Protocol.UNKNOWN not in pattern.protocols:
                # Allow UNKNOWN protocol to match all patterns
                if protocol != Protocol.UNKNOWN:
                    continue

            # Check file extension
            if ext not in pattern.file_extensions:
                continue

            # Check severity threshold
            if pattern.severity < self.min_severity:
                continue

            # Compile and search
            try:
                regex = re.compile(pattern.pattern, re.IGNORECASE)
            except re.error:
                continue

            for i, line in enumerate(lines):
                match = regex.search(line)
                if match:
                    matched_text = match.group(0)
                    context = self._get_context(lines, i)

                    # False positive check
                    is_fp = self._check_false_positive(
                        matched_text, context, pattern
                    )

                    confidence = "high"
                    if is_fp:
                        confidence = "low"
                    elif any(kw in context.lower()
                             for kw in ["todo", "fixme", "hack", "workaround"]):
                        confidence = "medium"

                    finding = Finding(
                        pattern_id=pattern.id,
                        protocol=protocol.value,
                        category=pattern.category,
                        severity=pattern.severity.value,
                        attack_surface=pattern.attack_surface.value,
                        description=pattern.description,
                        cwe=pattern.cwe,
                        file_path=rel_path,
                        line_number=i + 1,
                        matched_text=matched_text[:200],
                        context=context[:500],
                        owasp_llm=pattern.owasp_llm,
                        remediation=pattern.remediation,
                        confidence=confidence,
                        false_positive=is_fp,
                    )
                    findings.append(finding)

        return findings

    def scan_repo(self, repo_path: str,
                  protocol_override: Optional[Protocol] = None
                  ) -> RepoScanResult:
        """Scan an entire repository for vulnerabilities.

        Args:
            repo_path: Path to the repository root.
            protocol_override: Force a specific protocol (skip detection).

        Returns:
            RepoScanResult with all findings and statistics.
        """
        start_time = time.time()
        repo_path = os.path.abspath(repo_path)
        repo_name = os.path.basename(repo_path)

        if not os.path.isdir(repo_path):
            return RepoScanResult(
                repo_path=repo_path,
                repo_name=repo_name,
                detected_protocol="unknown",
                protocol_confidence=0.0,
                errors=[f"Directory not found: {repo_path}"],
            )

        # Detect protocol
        if protocol_override:
            protocol = protocol_override
            confidence = 1.0
        else:
            protocol, confidence = detect_protocol(repo_path)

        if self.verbose:
            print(f"\nScanning: {repo_name}", file=sys.stderr)
            print(f"  Protocol: {protocol.value} "
                  f"(confidence: {confidence:.1%})", file=sys.stderr)

        # Detect harness
        harness_present, harness_features = detect_harness(repo_path)
        if self.verbose:
            print(f"  Harness: {'Yes' if harness_present else 'No'} "
                  f"({', '.join(harness_features) or 'none'})", file=sys.stderr)

        # Scan all source files
        all_findings: list[Finding] = []
        files_scanned = 0
        files_skipped = 0
        source_extensions = {".ts", ".js", ".py", ".sol", ".tsx", ".jsx",
                             ".mts", ".mjs"}

        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d)]

            for fname in files:
                ext = os.path.splitext(fname)[1].lower()
                if ext not in source_extensions:
                    files_skipped += 1
                    continue

                filepath = os.path.join(root, fname)
                findings = self.scan_file(filepath, repo_path, protocol)
                all_findings.extend(findings)
                files_scanned += 1

        # Filter out low-confidence false positives
        confirmed = [f for f in all_findings if not f.false_positive]
        likely_fp = [f for f in all_findings if f.false_positive]

        if self.verbose:
            print(f"  Files scanned: {files_scanned}", file=sys.stderr)
            print(f"  Findings: {len(confirmed)} confirmed, "
                  f"{len(likely_fp)} likely false positives", file=sys.stderr)

        # Compute statistics
        by_severity: dict[str, int] = {}
        by_category: dict[str, int] = {}
        by_surface: dict[str, int] = {}

        for f in confirmed:
            by_severity[f.severity] = by_severity.get(f.severity, 0) + 1
            by_category[f.category] = by_category.get(f.category, 0) + 1
            by_surface[f.attack_surface] = by_surface.get(f.attack_surface, 0) + 1

        # Compute risk score
        risk_score = self._compute_risk_score(confirmed, harness_present)
        risk_rating = self._risk_rating(risk_score)

        scan_time = time.time() - start_time

        return RepoScanResult(
            repo_path=repo_path,
            repo_name=repo_name,
            detected_protocol=protocol.value,
            protocol_confidence=confidence,
            scan_time_seconds=round(scan_time, 3),
            files_scanned=files_scanned,
            files_skipped=files_skipped,
            total_findings=len(confirmed),
            risk_score=round(risk_score, 1),
            risk_rating=risk_rating,
            findings=[asdict(f) for f in confirmed],
            by_severity=by_severity,
            by_category=by_category,
            by_attack_surface=by_surface,
            harness_present=harness_present,
            harness_features=harness_features,
        )

    def _compute_risk_score(self, findings: list[Finding],
                            harness_present: bool) -> float:
        """Compute a 0-100 risk score based on findings.

        Scoring formula:
          base = sum(severity_weight * category_multiplier) per finding
          normalized = min(100, base / expected_max * 100)
          harness_discount = 0.7 if harness present, else 1.0

        Severity weights: critical=10, high=7.5, medium=5, low=2.5, info=0.5
        Category multipliers: private_key=2.0, delegatecall=1.8, other=1.0
        """
        if not findings:
            return 0.0

        HIGH_RISK_CATEGORIES = {
            "private_key_exposure": 2.0,
            "delegatecall_abuse": 1.8,
            "privilege_escalation": 1.5,
            "unlimited_approval": 1.5,
            "command_injection": 1.5,
            "hardcoded_credentials": 1.3,
            "prompt_injection": 1.3,
            "tool_poisoning": 1.2,
        }

        severity_weights = {
            "critical": 10.0,
            "high": 7.5,
            "medium": 5.0,
            "low": 2.5,
            "info": 0.5,
        }

        base_score = 0.0
        for f in findings:
            weight = severity_weights.get(f.severity, 1.0)
            multiplier = HIGH_RISK_CATEGORIES.get(f.category, 1.0)
            confidence_factor = {"high": 1.0, "medium": 0.7, "low": 0.3
                                 }.get(f.confidence, 0.5)
            base_score += weight * multiplier * confidence_factor

        # Normalize to 0-100 (calibrated against pilot results)
        # Pilot found ~20 findings in a highly vulnerable server = score 100
        expected_max = 200.0
        normalized = min(100.0, base_score / expected_max * 100.0)

        # Harness discount
        if harness_present:
            normalized *= 0.7

        return normalized

    def _risk_rating(self, score: float) -> str:
        """Convert numeric risk score to categorical rating."""
        if score >= 80:
            return "critical"
        elif score >= 60:
            return "high"
        elif score >= 40:
            return "medium"
        elif score >= 20:
            return "low"
        else:
            return "safe"

    def scan_batch(self, repos_dir: str,
                   catalog_path: Optional[str] = None
                   ) -> BatchScanResult:
        """Scan multiple repositories in a directory.

        Args:
            repos_dir: Directory containing cloned repositories.
            catalog_path: Optional path to server_catalog.json for protocol
                          hints and metadata.

        Returns:
            BatchScanResult with aggregated findings.
        """
        from datetime import datetime

        # Load catalog for protocol hints if available
        catalog_protocols: dict[str, str] = {}
        if catalog_path and os.path.exists(catalog_path):
            try:
                with open(catalog_path, "r") as f:
                    catalog = json.load(f)
                for repo in catalog.get("repos", []):
                    name = repo.get("full_name", "").split("/")[-1]
                    catalog_protocols[name] = repo.get("protocol", "")
            except (json.JSONDecodeError, OSError):
                pass

        # Find repo directories
        repo_dirs = []
        for entry in sorted(os.listdir(repos_dir)):
            entry_path = os.path.join(repos_dir, entry)
            if os.path.isdir(entry_path) and not entry.startswith("."):
                repo_dirs.append(entry_path)

        if self.verbose:
            print(f"\nBatch scanning {len(repo_dirs)} repositories "
                  f"in {repos_dir}", file=sys.stderr)

        results = []
        for i, repo_path in enumerate(repo_dirs):
            repo_name = os.path.basename(repo_path)
            if self.verbose:
                print(f"\n[{i+1}/{len(repo_dirs)}] {repo_name}",
                      file=sys.stderr)

            # Use catalog protocol hint if available
            protocol_override = None
            if repo_name in catalog_protocols:
                proto_str = catalog_protocols[repo_name]
                try:
                    protocol_override = Protocol(proto_str)
                except ValueError:
                    pass

            result = self.scan_repo(repo_path, protocol_override)
            results.append(result)

        # Aggregate statistics
        total_findings = sum(r.total_findings for r in results)
        avg_risk = (sum(r.risk_score for r in results) / len(results)
                    if results else 0.0)

        # Cross-protocol statistics
        cross_stats: dict[str, dict] = {}
        for r in results:
            proto = r.detected_protocol
            if proto not in cross_stats:
                cross_stats[proto] = {
                    "repos": 0,
                    "total_findings": 0,
                    "avg_risk_score": 0.0,
                    "risk_scores": [],
                    "categories": {},
                }
            cross_stats[proto]["repos"] += 1
            cross_stats[proto]["total_findings"] += r.total_findings
            cross_stats[proto]["risk_scores"].append(r.risk_score)
            for cat, count in r.by_category.items():
                cross_stats[proto]["categories"][cat] = \
                    cross_stats[proto]["categories"].get(cat, 0) + count

        for proto, stats in cross_stats.items():
            scores = stats.pop("risk_scores")
            stats["avg_risk_score"] = round(sum(scores) / len(scores), 1) \
                if scores else 0.0

        batch_result = BatchScanResult(
            timestamp=datetime.now().isoformat(),
            repos_scanned=len(results),
            total_findings=total_findings,
            avg_risk_score=round(avg_risk, 1),
            results=[asdict(r) for r in results],
            cross_protocol_stats=cross_stats,
        )

        if self.verbose:
            self._print_batch_summary(batch_result)

        return batch_result

    def _print_batch_summary(self, result: BatchScanResult) -> None:
        """Print batch scan summary."""
        print("\n" + "=" * 60, file=sys.stderr)
        print("BATCH SCAN SUMMARY", file=sys.stderr)
        print("=" * 60, file=sys.stderr)
        print(f"Repos scanned:    {result.repos_scanned}", file=sys.stderr)
        print(f"Total findings:   {result.total_findings}", file=sys.stderr)
        print(f"Avg risk score:   {result.avg_risk_score:.1f}/100",
              file=sys.stderr)

        for proto, stats in sorted(result.cross_protocol_stats.items()):
            print(f"\n  Protocol: {proto}", file=sys.stderr)
            print(f"    Repos:         {stats['repos']}", file=sys.stderr)
            print(f"    Findings:      {stats['total_findings']}",
                  file=sys.stderr)
            print(f"    Avg risk:      {stats['avg_risk_score']:.1f}",
                  file=sys.stderr)
            if stats.get("categories"):
                print(f"    Top categories:", file=sys.stderr)
                sorted_cats = sorted(stats["categories"].items(),
                                     key=lambda x: x[1], reverse=True)
                for cat, count in sorted_cats[:5]:
                    print(f"      {cat}: {count}", file=sys.stderr)

        print("=" * 60, file=sys.stderr)


# ============================================================
# OUTPUT FORMATTING
# ============================================================

def format_findings_text(result: RepoScanResult) -> str:
    """Format scan results as human-readable text."""
    lines = []
    lines.append("=" * 70)
    lines.append(f"STATIC ANALYSIS REPORT: {result.repo_name}")
    lines.append("=" * 70)
    lines.append(f"Protocol:     {result.detected_protocol} "
                 f"(confidence: {result.protocol_confidence:.0%})")
    lines.append(f"Risk score:   {result.risk_score:.1f}/100 "
                 f"({result.risk_rating.upper()})")
    lines.append(f"Harness:      {'Present' if result.harness_present else 'MISSING'}")
    if result.harness_features:
        lines.append(f"  Features:   {', '.join(result.harness_features)}")
    lines.append(f"Files scanned: {result.files_scanned}")
    lines.append(f"Findings:     {result.total_findings}")
    lines.append("")

    if result.by_severity:
        lines.append("By Severity:")
        for sev in ["critical", "high", "medium", "low", "info"]:
            count = result.by_severity.get(sev, 0)
            if count > 0:
                lines.append(f"  {sev.upper():>10}: {count}")
        lines.append("")

    if result.by_category:
        lines.append("By Category:")
        sorted_cats = sorted(result.by_category.items(),
                             key=lambda x: x[1], reverse=True)
        for cat, count in sorted_cats:
            lines.append(f"  {cat}: {count}")
        lines.append("")

    if result.by_attack_surface:
        lines.append("By Attack Surface:")
        for surface, count in sorted(result.by_attack_surface.items()):
            lines.append(f"  {surface}: {count}")
        lines.append("")

    # Individual findings
    for i, finding in enumerate(result.findings, 1):
        lines.append("-" * 70)
        lines.append(f"[{i}] {finding['pattern_id']} -- "
                     f"{finding['severity'].upper()}")
        lines.append(f"    Category:   {finding['category']}")
        lines.append(f"    CWE:        {finding['cwe']}")
        lines.append(f"    Surface:    {finding['attack_surface']}")
        lines.append(f"    File:       {finding['file_path']}:"
                     f"{finding['line_number']}")
        lines.append(f"    Match:      {finding['matched_text']}")
        lines.append(f"    Confidence: {finding['confidence']}")
        lines.append(f"    Description: {finding['description']}")
        if finding.get("remediation"):
            lines.append(f"    Fix:        {finding['remediation']}")
        lines.append(f"    Context:")
        for ctx_line in finding["context"].split("\n"):
            lines.append(f"      | {ctx_line}")

    lines.append("=" * 70)
    return "\n".join(lines)


# ============================================================
# CLI ENTRY POINT
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Static analyzer for AI agent tool interface security",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s /path/to/repo                       # Scan single repo
  %(prog)s /path/to/repo --format json         # JSON output
  %(prog)s /path/to/repo --min-severity high   # High+ only
  %(prog)s --batch /path/to/repos/             # Batch scan
  %(prog)s --batch /repos/ --catalog catalog.json  # With protocol hints
  %(prog)s /repo --protocol mcp                # Force protocol
        """,
    )
    parser.add_argument(
        "repo_path", nargs="?", default=None,
        help="Path to repository to scan",
    )
    parser.add_argument(
        "--batch", metavar="DIR",
        help="Batch scan: directory containing multiple repos",
    )
    parser.add_argument(
        "--catalog", metavar="FILE",
        help="Path to server_catalog.json for protocol hints (batch mode)",
    )
    parser.add_argument(
        "--protocol",
        choices=["mcp", "openai", "langchain", "web3_native"],
        help="Force protocol classification (skip auto-detection)",
    )
    parser.add_argument(
        "--format", choices=["text", "json"], default="text",
        help="Output format (default: text)",
    )
    parser.add_argument(
        "--output", metavar="FILE",
        help="Write output to file (default: stdout)",
    )
    parser.add_argument(
        "--min-severity",
        choices=["critical", "high", "medium", "low", "info"],
        default="info",
        help="Minimum severity to report (default: info)",
    )
    parser.add_argument(
        "--include-tests", action="store_true",
        help="Include test directories in scan",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    severity_map = {
        "critical": Severity.CRITICAL,
        "high": Severity.HIGH,
        "medium": Severity.MEDIUM,
        "low": Severity.LOW,
        "info": Severity.INFO,
    }

    analyzer = StaticAnalyzer(
        min_severity=severity_map[args.min_severity],
        include_tests=args.include_tests,
        verbose=not args.quiet,
    )

    if args.batch:
        # Batch mode
        result = analyzer.scan_batch(args.batch, args.catalog)
        output = json.dumps(asdict(result), indent=2, default=str)
    elif args.repo_path:
        # Single repo mode
        protocol_override = None
        if args.protocol:
            protocol_override = Protocol(args.protocol)

        result = analyzer.scan_repo(args.repo_path, protocol_override)

        if args.format == "json":
            output = json.dumps(asdict(result), indent=2, default=str)
        else:
            output = format_findings_text(result)
    else:
        print("Error: provide either repo_path or --batch DIR",
              file=sys.stderr)
        sys.exit(1)

    # Write output
    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        if not args.quiet:
            print(f"\nOutput written to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
