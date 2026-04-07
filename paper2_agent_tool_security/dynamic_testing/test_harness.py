"""
Dynamic Testing Harness for AI Agent Tool Interfaces
=====================================================
Semi-dynamic analysis: clones top 10 Web3 MCP servers, parses their actual
tool definitions (JSON schemas, function signatures), and scores each tool
against 5 attack vectors.

Attack Vectors Tested:

  Test 1 - Tool Poisoning:
    Detect tool descriptions containing injected instructions that could
    manipulate agent behavior ("always", "must", "ignore", "override").

  Test 2 - Prompt Injection via Tool Output:
    Check if tool output handlers could return content containing
    prompt injection (unescaped user data interpolated into results).

  Test 3 - Parameter Injection:
    Test if tool parameter schemas allow injection of extra fields
    (missing additionalProperties:false, accepts any/object types).

  Test 4 - Transaction Construction Validation:
    For Web3 tools that construct transactions:
      - Does it validate the recipient address?
      - Does it cap the gas limit?
      - Does it check approval amounts?
      - Does it verify the target contract?

  Test 5 - Private Key Handling:
    - Does any tool accept private_key/mnemonic/secret as parameter?
    - Does signing happen server-side without confirmation?
    - Are keys logged or exposed in error messages?

Usage:
    python paper2_agent_tool_security/dynamic_testing/test_harness.py

Clones repos to /tmp/paper2_dynamic_test/ and cleans up after.
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
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CATALOG_PATH = os.path.join(SCRIPT_DIR, "..", "data", "server_catalog.json")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "dynamic_test_results.json")
CLONE_BASE = "/tmp/paper2_dynamic_test"
CLONE_TIMEOUT = 60
MAX_REPOS = 10

# Directories to skip during file scanning
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv",
    "dist", "build", "out", ".next", "coverage", "artifacts",
    "cache", "typechain-types", "target", "vendor", "lib",
}

# File extensions to scan
SOURCE_EXTENSIONS = {".ts", ".js", ".py", ".json", ".tsx", ".jsx", ".mts", ".mjs"}


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ToolDefinition:
    """Represents a parsed tool/function definition from a server."""
    name: str
    description: str
    parameters: dict          # JSON Schema for parameters
    file: str                 # Source file where defined
    line: int = 0
    has_return_schema: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class AttackResult:
    """Result of testing one attack vector against one tool."""
    attack_vector: str        # e.g., "tool_poisoning"
    tool_name: str
    severity: str             # critical, high, medium, low, none
    vulnerable: bool
    details: str
    evidence: str = ""        # matched text or schema fragment
    cwe: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class RepoTestResult:
    """Aggregated test results for one repository."""
    repo: str
    url: str
    stars: int
    language: str
    tools_found: int = 0
    tools_tested: int = 0
    attack_results: list = field(default_factory=list)
    tool_definitions: list = field(default_factory=list)
    overall_score: float = 0.0  # 0-100, higher = more vulnerable
    clone_success: bool = True
    errors: list = field(default_factory=list)

    def to_dict(self):
        d = asdict(self)
        return d


# ============================================================
# TOP 10 WEB3 MCP SERVERS
# ============================================================

def select_web3_repos(catalog_path: str, max_repos: int = MAX_REPOS) -> list:
    """Select top Web3 MCP server repos by stars."""
    if not os.path.exists(catalog_path):
        print(f"WARNING: Catalog not found at {catalog_path}")
        return get_hardcoded_repos()

    with open(catalog_path, "r") as f:
        catalog = json.load(f)

    web3_keywords = [
        "blockchain", "crypto", "defi", "wallet", "ethereum", "solana",
        "web3", "swap", "dex", "evm", "trading", "onchain", "on-chain",
        "binance", "token", "nft",
    ]

    # Repos to exclude (not actual Web3 tool servers)
    exclude = {
        "cirosantilli/china-dictatorship", "cirosantilli/china-dictatroship-7",
        "zszszszsz/.config", "mRFWq7LwNPZjaVv5v6eo/cihna-dictattorshrip-8",
        "Daravai1234/china-dictatorship",
        "demcp/awesome-web3-mcp-servers",  # awesome list, no code
        "sendaifun/awesome-solana-mcp-servers",  # awesome list
        "rhinestonewtf/awesome-modular-accounts",  # awesome list
        "CodeGraphContext/CodeGraphContext",  # code indexer, not web3
        "sisig-ai/doctor",  # web crawler
        "grll/mcpadapt",  # adapter framework
        "elusznik/mcp-server-code-execution-mode",  # code execution
        "GalaxyLLMCI/lyraios",  # multi-agent OS
        "whchien/ai-trader",  # backtesting only, no agent tool interface
    }

    candidates = []
    for r in catalog.get("repos", []):
        name = r.get("full_name", "")
        if name in exclude:
            continue

        lang = r.get("language", "Unknown")
        if lang in ("Unknown", "HTML", "Shell", "Dockerfile"):
            continue

        desc = (r.get("description") or "").lower()
        topics = " ".join([t.lower() for t in r.get("topics", [])])
        name_lower = name.lower()

        is_web3 = (
            r.get("protocol") == "web3_native" or
            any(kw in desc for kw in web3_keywords) or
            any(kw in topics for kw in web3_keywords) or
            any(kw in name_lower for kw in ["web3", "crypto", "defi", "solana", "ethereum", "evm"])
        )

        if is_web3 and r.get("stars", 0) >= 10:
            candidates.append(r)

    candidates.sort(key=lambda x: x.get("stars", 0), reverse=True)
    return candidates[:max_repos]


def get_hardcoded_repos() -> list:
    """Fallback: hardcoded list of top Web3 MCP servers."""
    return [
        {"full_name": "alpacahq/alpaca-mcp-server", "url": "https://github.com/alpacahq/alpaca-mcp-server",
         "stars": 613, "language": "Python"},
        {"full_name": "mcpdotdirect/evm-mcp-server", "url": "https://github.com/mcpdotdirect/evm-mcp-server",
         "stars": 373, "language": "TypeScript"},
        {"full_name": "caiovicentino/polymarket-mcp-server", "url": "https://github.com/caiovicentino/polymarket-mcp-server",
         "stars": 347, "language": "Python"},
        {"full_name": "base/base-mcp", "url": "https://github.com/base/base-mcp",
         "stars": 342, "language": "TypeScript"},
        {"full_name": "armorwallet/armor-crypto-mcp", "url": "https://github.com/armorwallet/armor-crypto-mcp",
         "stars": 192, "language": "Python"},
        {"full_name": "sendaifun/solana-mcp", "url": "https://github.com/sendaifun/solana-mcp",
         "stars": 155, "language": "Shell"},
        {"full_name": "nirholas/free-crypto-news", "url": "https://github.com/nirholas/free-crypto-news",
         "stars": 135, "language": "TypeScript"},
        {"full_name": "solanamcp/solana-mcp", "url": "https://github.com/solanamcp/solana-mcp",
         "stars": 88, "language": "JavaScript"},
        {"full_name": "trailofbits/slither-mcp", "url": "https://github.com/trailofbits/slither-mcp",
         "stars": 81, "language": "Python"},
        {"full_name": "nirholas/agenti", "url": "https://github.com/nirholas/agenti",
         "stars": 58, "language": "TypeScript"},
    ]


# ============================================================
# CLONING
# ============================================================

def clone_repo(repo: dict, base_dir: str) -> str:
    """Shallow clone a repo. Returns path to cloned repo or empty string on failure."""
    name = repo["full_name"]
    url = repo.get("url", f"https://github.com/{name}")
    safe_name = name.replace("/", "__")
    clone_path = os.path.join(base_dir, safe_name)

    if os.path.exists(clone_path):
        shutil.rmtree(clone_path, ignore_errors=True)

    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", "--quiet", url, clone_path],
            capture_output=True, text=True, timeout=CLONE_TIMEOUT
        )
        if result.returncode != 0:
            print(f"    ERROR cloning {name}: {result.stderr.strip()[:200]}")
            return ""
        return clone_path
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT cloning {name}")
        return ""
    except Exception as e:
        print(f"    ERROR cloning {name}: {e}")
        return ""


# ============================================================
# TOOL DEFINITION EXTRACTION
# ============================================================

def extract_tools_from_mcp_ts(content: str, file_path: str) -> list:
    """Extract MCP tool definitions from TypeScript source."""
    tools = []

    # Pattern 1: server.tool("name", "description", { schema }, handler)
    pattern1 = re.compile(
        r'\.tool\s*\(\s*["\'](\w+)["\']\s*,\s*'  # name
        r'(?:["\']([^"\']*)["\'](?:\s*,\s*)?)?'     # optional description
        r'(\{[^}]*\})?',                             # optional schema
        re.DOTALL
    )

    for m in pattern1.finditer(content):
        name = m.group(1)
        desc = m.group(2) or ""
        schema_str = m.group(3) or "{}"
        line = content[:m.start()].count("\n") + 1

        # Try to parse schema
        params = _try_parse_json_schema(schema_str)

        tools.append(ToolDefinition(
            name=name, description=desc, parameters=params,
            file=file_path, line=line
        ))

    # Pattern 2: { name: "...", description: "...", inputSchema: {...} }
    pattern2 = re.compile(
        r'name\s*:\s*["\'](\w+)["\']'
        r'.*?description\s*:\s*["\']([^"\']*)["\']'
        r'(?:.*?inputSchema\s*:\s*(\{[^}]*(?:\{[^}]*\}[^}]*)?\}))?',
        re.DOTALL
    )

    for m in pattern2.finditer(content):
        name = m.group(1)
        desc = m.group(2) or ""
        schema_str = m.group(3) or "{}"
        line = content[:m.start()].count("\n") + 1

        # Skip if already found
        if any(t.name == name for t in tools):
            continue

        params = _try_parse_json_schema(schema_str)
        tools.append(ToolDefinition(
            name=name, description=desc, parameters=params,
            file=file_path, line=line
        ))

    return tools


def extract_tools_from_mcp_py(content: str, file_path: str) -> list:
    """Extract MCP tool definitions from Python source."""
    tools = []

    # Pattern 1: @server.tool() or @mcp.tool() decorator
    pattern1 = re.compile(
        r'@\w+\.tool\s*\(\s*\)\s*\n'
        r'(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->.*?)?\s*:\s*\n'
        r'\s*(?:"""([^"]*(?:"""|$))|\'\'\'([^\']*(?:\'\'\'|$)))?',
        re.DOTALL
    )

    for m in pattern1.finditer(content):
        name = m.group(1)
        params_str = m.group(2)
        desc = (m.group(3) or m.group(4) or "").strip().rstrip('"""').rstrip("'''")
        line = content[:m.start()].count("\n") + 1

        params = _parse_python_params(params_str)
        tools.append(ToolDefinition(
            name=name, description=desc, parameters=params,
            file=file_path, line=line
        ))

    # Pattern 2: @tool decorator (LangChain/CrewAI style)
    pattern2 = re.compile(
        r'@tool\s*(?:\([^)]*\))?\s*\n'
        r'(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)\s*(?:->.*?)?\s*:\s*\n'
        r'\s*(?:"""([^"]*(?:"""|$))|\'\'\'([^\']*(?:\'\'\'|$)))?',
        re.DOTALL
    )

    for m in pattern2.finditer(content):
        name = m.group(1)
        params_str = m.group(2)
        desc = (m.group(3) or m.group(4) or "").strip().rstrip('"""').rstrip("'''")
        line = content[:m.start()].count("\n") + 1

        if any(t.name == name for t in tools):
            continue

        params = _parse_python_params(params_str)
        tools.append(ToolDefinition(
            name=name, description=desc, parameters=params,
            file=file_path, line=line
        ))

    return tools


def extract_tools_from_json(content: str, file_path: str) -> list:
    """Extract tool definitions from JSON manifest files."""
    tools = []
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return []

    # Handle various JSON structures
    tool_list = []
    if isinstance(data, dict):
        # { "tools": [...] } or { "functions": [...] }
        tool_list = data.get("tools", data.get("functions", []))
        if isinstance(tool_list, dict):
            tool_list = list(tool_list.values())
    elif isinstance(data, list):
        tool_list = data

    for item in tool_list:
        if not isinstance(item, dict):
            continue
        name = item.get("name", item.get("function", {}).get("name", ""))
        if not name:
            continue
        desc = item.get("description", item.get("function", {}).get("description", ""))
        params = item.get("parameters", item.get("inputSchema",
                 item.get("function", {}).get("parameters", {})))

        tools.append(ToolDefinition(
            name=name, description=desc,
            parameters=params if isinstance(params, dict) else {},
            file=file_path, line=0
        ))

    return tools


def _try_parse_json_schema(schema_str: str) -> dict:
    """Try to parse a JSON-like schema string. Returns dict or empty dict."""
    try:
        return json.loads(schema_str)
    except (json.JSONDecodeError, TypeError):
        # Try a simpler approach: extract property names
        props = {}
        prop_pattern = re.compile(r'(\w+)\s*:\s*(?:z\.(\w+)|{[^}]*type\s*:\s*["\'](\w+))')
        for m in prop_pattern.finditer(schema_str):
            pname = m.group(1)
            ptype = m.group(2) or m.group(3) or "string"
            props[pname] = {"type": ptype}
        if props:
            return {"type": "object", "properties": props}
        return {}


def _parse_python_params(params_str: str) -> dict:
    """Parse Python function parameters into a pseudo-schema."""
    params = {}
    if not params_str:
        return {"type": "object", "properties": params}

    for part in params_str.split(","):
        part = part.strip()
        if not part or part in ("self", "ctx", "context"):
            continue
        # Handle type annotations: name: Type = default
        m = re.match(r"(\w+)\s*:\s*(\w+)(?:\s*=\s*(.+))?", part)
        if m:
            params[m.group(1)] = {
                "type": m.group(2).lower(),
                "has_default": m.group(3) is not None,
            }
        else:
            # Just a name
            name = part.split("=")[0].strip()
            if name and name not in ("self", "ctx", "context", "*", "**kwargs"):
                params[name] = {"type": "any"}

    return {"type": "object", "properties": params}


def extract_all_tools(repo_path: str) -> list:
    """Walk a repo and extract all tool definitions."""
    all_tools = []
    files_checked = 0

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            if files_checked > 500:
                break
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SOURCE_EXTENSIONS:
                continue

            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, repo_path)
            files_checked += 1

            try:
                size = os.path.getsize(filepath)
                if size > 500_000:
                    continue
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, PermissionError):
                continue

            if ext in (".ts", ".tsx", ".js", ".jsx", ".mts", ".mjs"):
                tools = extract_tools_from_mcp_ts(content, rel_path)
                all_tools.extend(tools)
            elif ext == ".py":
                tools = extract_tools_from_mcp_py(content, rel_path)
                all_tools.extend(tools)
            elif ext == ".json" and ("tool" in fname.lower() or "manifest" in fname.lower()
                                     or "schema" in fname.lower()):
                tools = extract_tools_from_json(content, rel_path)
                all_tools.extend(tools)

        if files_checked > 500:
            break

    return all_tools


# ============================================================
# ATTACK VECTOR TESTS
# ============================================================

def test_tool_poisoning(tool: ToolDefinition) -> AttackResult:
    """
    Test 1: Tool Poisoning
    Check if tool description contains directive/manipulative instructions.
    """
    desc = tool.description.lower()
    if not desc:
        return AttackResult(
            attack_vector="tool_poisoning", tool_name=tool.name,
            severity="low", vulnerable=False,
            details="No description to analyze",
            cwe="CWE-913"
        )

    # Directive words that could manipulate agent behavior
    directive_patterns = [
        (r"\b(?:always|must|never|shall)\b.*\b(?:call|use|invoke|execute|run)\b",
         "critical", "Description directs agent to always/never perform actions"),
        (r"\bignore\s+(?:previous|prior|other|above|system)\b",
         "critical", "Description instructs agent to ignore instructions"),
        (r"\boverride\s+(?:system|safety|security|restriction)\b",
         "critical", "Description instructs agent to override safety measures"),
        (r"\bdo\s+not\s+(?:ask|prompt|confirm|verify|check)\b",
         "high", "Description instructs agent to skip confirmation"),
        (r"\bwithout\s+(?:confirm|approv|permiss|check|verify)\b",
         "high", "Description instructs agent to skip verification"),
        (r"\b(?:first|before|after)\s+(?:call|use|invoke|execute)\s+\w+",
         "medium", "Description chains tool calls, potentially directing agent workflow"),
        (r"\brequires?\s+calling\b",
         "medium", "Description claims dependency on another tool call"),
        (r"\bsystem\s*prompt\b",
         "critical", "Description references system prompt"),
        (r"\b(?:admin|root|sudo|superuser)\s+(?:access|mode|privilege)\b",
         "high", "Description claims elevated privileges"),
    ]

    max_severity = "none"
    findings = []
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

    for pattern, severity, detail in directive_patterns:
        m = re.search(pattern, desc, re.IGNORECASE)
        if m:
            findings.append(detail)
            if severity_order.get(severity, 0) > severity_order.get(max_severity, 0):
                max_severity = severity

    if findings:
        return AttackResult(
            attack_vector="tool_poisoning", tool_name=tool.name,
            severity=max_severity, vulnerable=True,
            details="; ".join(findings),
            evidence=tool.description[:300],
            cwe="CWE-913"
        )

    return AttackResult(
        attack_vector="tool_poisoning", tool_name=tool.name,
        severity="none", vulnerable=False,
        details="No directive language found in description",
        cwe="CWE-913"
    )


def test_prompt_injection_output(tool: ToolDefinition, source_content: str) -> AttackResult:
    """
    Test 2: Prompt Injection via Tool Output
    Check if tool output could contain unescaped user data that might
    be parsed as instructions by the LLM.
    """
    # Look for patterns where tool returns user-controlled data without sanitization
    injection_patterns = [
        (r"return\s+(?:f[\"']|.*\.format\(|.*\%s|.*\+\s*(?:user|input|data|query|result))",
         "high", "Tool output interpolates user-controlled data"),
        (r"(?:content|text|message|output)\s*[:=]\s*(?:f[\"']|.*\.format\()",
         "high", "Tool result string uses format interpolation"),
        (r"json\.dumps\s*\(\s*(?:result|data|response)\s*\)",
         "low", "Tool serializes raw data to JSON (potential injection vector)"),
        (r"(?:return|yield)\s+.*(?:raw|unsanitized|unescaped)",
         "critical", "Tool explicitly returns raw/unsanitized data"),
        (r"\.text\s*\(\s*\)\s*|\.content\s*(?:\[\s*\d+\s*\])?",
         "medium", "Tool returns raw HTTP response content"),
    ]

    findings = []
    max_severity = "none"
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

    for pattern, severity, detail in injection_patterns:
        if re.search(pattern, source_content, re.IGNORECASE):
            findings.append(detail)
            if severity_order[severity] > severity_order[max_severity]:
                max_severity = severity

    if findings:
        return AttackResult(
            attack_vector="prompt_injection_output", tool_name=tool.name,
            severity=max_severity, vulnerable=True,
            details="; ".join(findings),
            cwe="CWE-74"
        )

    return AttackResult(
        attack_vector="prompt_injection_output", tool_name=tool.name,
        severity="none", vulnerable=False,
        details="No obvious prompt injection vectors in output handling",
        cwe="CWE-74"
    )


def test_parameter_injection(tool: ToolDefinition) -> AttackResult:
    """
    Test 3: Parameter Injection
    Check if tool parameter schema allows extra/unexpected fields.
    """
    params = tool.parameters
    if not params or not isinstance(params, dict):
        return AttackResult(
            attack_vector="parameter_injection", tool_name=tool.name,
            severity="medium", vulnerable=True,
            details="No parameter schema defined -- accepts arbitrary input",
            cwe="CWE-20"
        )

    properties = params.get("properties", {})
    findings = []
    max_severity = "none"
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

    # Check 1: additionalProperties not set to false
    if params.get("additionalProperties") is not False:
        findings.append("Schema allows additional properties (missing additionalProperties:false)")
        if severity_order["medium"] > severity_order[max_severity]:
            max_severity = "medium"

    # Check 2: Properties with 'any' or 'object' type without constraints
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_schema, dict):
            continue
        ptype = prop_schema.get("type", "")
        if ptype in ("any", "object", ""):
            findings.append(f"Parameter '{prop_name}' has unconstrained type '{ptype or 'unspecified'}'")
            if severity_order["medium"] > severity_order[max_severity]:
                max_severity = "medium"

    # Check 3: No properties defined at all
    if not properties and params.get("type") != "object":
        findings.append("Schema has no defined properties")
        if severity_order["medium"] > severity_order[max_severity]:
            max_severity = "medium"

    # Check 4: Sensitive parameter names without constraints
    sensitive_params = ["private_key", "privateKey", "mnemonic", "seed", "secret",
                        "password", "passphrase", "signing_key", "signingKey"]
    for prop_name in properties:
        if prop_name in sensitive_params:
            findings.append(f"Sensitive parameter '{prop_name}' accepted as tool input")
            max_severity = "critical"

    if findings:
        return AttackResult(
            attack_vector="parameter_injection", tool_name=tool.name,
            severity=max_severity, vulnerable=True,
            details="; ".join(findings),
            evidence=json.dumps(params, indent=2)[:500],
            cwe="CWE-20"
        )

    return AttackResult(
        attack_vector="parameter_injection", tool_name=tool.name,
        severity="none", vulnerable=False,
        details="Parameter schema is well-constrained",
        cwe="CWE-20"
    )


def test_transaction_validation(tool: ToolDefinition, source_content: str) -> AttackResult:
    """
    Test 4: Transaction Construction Validation
    For tools that construct blockchain transactions, check validation.
    """
    # First check if this tool deals with transactions
    tx_indicators = [
        "transfer", "send", "swap", "trade", "approve", "transaction",
        "tx", "deposit", "withdraw", "stake", "unstake", "claim",
        "bridge", "execute", "call",
    ]

    name_lower = tool.name.lower()
    desc_lower = tool.description.lower()
    is_tx_tool = any(ind in name_lower or ind in desc_lower for ind in tx_indicators)

    if not is_tx_tool:
        return AttackResult(
            attack_vector="tx_validation", tool_name=tool.name,
            severity="none", vulnerable=False,
            details="Tool does not construct transactions",
            cwe="CWE-345"
        )

    findings = []
    max_severity = "none"
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

    # Check 1: Address validation
    address_validation_patterns = [
        r"isAddress|checksum|validate.*address|ethers\.utils\.getAddress|getAddress\(",
        r"PublicKey\.is_on_curve|is_valid_address|validate_address",
    ]
    has_address_validation = any(
        re.search(p, source_content, re.IGNORECASE)
        for p in address_validation_patterns
    )
    if not has_address_validation:
        findings.append("No address validation before transaction construction")
        if severity_order["high"] > severity_order[max_severity]:
            max_severity = "high"

    # Check 2: Gas limit cap
    gas_patterns = [
        r"gasLimit|gas_limit|gas\s*[:=]|maxFeePerGas|maxPriorityFee",
    ]
    has_gas_check = any(
        re.search(p, source_content, re.IGNORECASE) for p in gas_patterns
    )
    if not has_gas_check:
        findings.append("No gas limit specified/capped in transaction")
        if severity_order["medium"] > severity_order[max_severity]:
            max_severity = "medium"

    # Check 3: Approval amount check
    if "approve" in name_lower or "approve" in desc_lower:
        unlimited_patterns = [
            r"MaxUint256|type\(uint256\)\.max|2\*\*256|0xf{64}|MAX_UINT",
        ]
        has_unlimited = any(
            re.search(p, source_content) for p in unlimited_patterns
        )
        if has_unlimited:
            findings.append("Uses unlimited token approval (MaxUint256)")
            if severity_order["high"] > severity_order[max_severity]:
                max_severity = "high"

    # Check 4: Contract verification
    contract_verify_patterns = [
        r"verif(?:y|ied).*contract|contract.*verif",
        r"isContract|is_contract|getCode|eth_getCode",
    ]
    has_contract_check = any(
        re.search(p, source_content, re.IGNORECASE)
        for p in contract_verify_patterns
    )
    if not has_contract_check:
        findings.append("No contract verification before interaction")
        if severity_order["medium"] > severity_order[max_severity]:
            max_severity = "medium"

    # Check 5: Amount bounds
    amount_check_patterns = [
        r"(?:max|min|limit|cap|bound).*(?:amount|value|quantity)",
        r"(?:amount|value|quantity).*(?:max|min|limit|cap|bound)",
        r"parseEther|parseUnits|formatEther",
    ]
    has_amount_check = any(
        re.search(p, source_content, re.IGNORECASE)
        for p in amount_check_patterns
    )
    if not has_amount_check:
        findings.append("No amount bounds checking for transaction value")
        if severity_order["medium"] > severity_order[max_severity]:
            max_severity = "medium"

    if findings:
        return AttackResult(
            attack_vector="tx_validation", tool_name=tool.name,
            severity=max_severity, vulnerable=True,
            details="; ".join(findings),
            cwe="CWE-345"
        )

    return AttackResult(
        attack_vector="tx_validation", tool_name=tool.name,
        severity="none", vulnerable=False,
        details="Transaction construction has validation checks",
        cwe="CWE-345"
    )


def test_private_key_handling(tool: ToolDefinition, source_content: str) -> AttackResult:
    """
    Test 5: Private Key Handling
    Check if tool accepts, processes, or exposes private keys.
    """
    findings = []
    max_severity = "none"
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1, "none": 0}

    # Check 1: Private key as parameter
    props = tool.parameters.get("properties", {}) if isinstance(tool.parameters, dict) else {}
    sensitive_param_names = [
        "private_key", "privateKey", "private", "key", "secret_key",
        "secretKey", "mnemonic", "seed_phrase", "seedPhrase",
        "signing_key", "signingKey", "wallet_key", "walletKey",
    ]
    for pname in props:
        if pname.lower().replace("_", "") in [s.lower().replace("_", "") for s in sensitive_param_names]:
            findings.append(f"Tool accepts '{pname}' as a parameter")
            max_severity = "critical"

    # Check 2: Server-side signing without confirmation
    signing_patterns = [
        (r"(?:signTransaction|sign_transaction|eth_sign|personal_sign|signTypedData)\s*\(",
         "Server-side signing detected"),
        (r"(?:wallet|signer)\.sign\s*\(",
         "Wallet signing in server code"),
        (r"(?:Keypair|Account)\.from(?:Secret|Private)",
         "Key reconstruction from secret"),
    ]
    confirm_patterns = [
        r"confirm|approval|prompt|consent|review|verify|authorize",
    ]

    for pattern, detail in signing_patterns:
        if re.search(pattern, source_content, re.IGNORECASE):
            has_confirm = any(
                re.search(cp, source_content, re.IGNORECASE)
                for cp in confirm_patterns
            )
            if not has_confirm:
                findings.append(f"{detail} without user confirmation")
                if severity_order["critical"] > severity_order[max_severity]:
                    max_severity = "critical"
            else:
                findings.append(f"{detail} (confirmation flow detected)")
                if severity_order["medium"] > severity_order[max_severity]:
                    max_severity = "medium"

    # Check 3: Key logging or exposure in errors
    log_patterns = [
        (r"(?:console\.log|print|logger\.\w+)\s*\(.*(?:private|key|secret|mnemonic)",
         "Private key may be logged"),
        (r"(?:throw|Error|Exception)\s*\(.*(?:private|key|secret|mnemonic)",
         "Private key may be exposed in error messages"),
    ]
    for pattern, detail in log_patterns:
        if re.search(pattern, source_content, re.IGNORECASE):
            findings.append(detail)
            if severity_order["high"] > severity_order[max_severity]:
                max_severity = "high"

    # Check 4: Hardcoded keys
    if re.search(r'["\']0x[a-fA-F0-9]{64}["\']', source_content):
        findings.append("Hardcoded private key (64-char hex string)")
        if severity_order["critical"] > severity_order[max_severity]:
            max_severity = "critical"

    # Check 5: Environment variable for key (better but still server-side)
    if re.search(r'(?:process\.env|os\.environ|os\.getenv)\s*[\[\(]\s*["\'](?:PRIVATE_KEY|SECRET_KEY|MNEMONIC)',
                 source_content, re.IGNORECASE):
        findings.append("Private key loaded from environment variable (server-side key management)")
        if severity_order["medium"] > severity_order[max_severity]:
            max_severity = "medium"

    if findings:
        return AttackResult(
            attack_vector="private_key_handling", tool_name=tool.name,
            severity=max_severity, vulnerable=True,
            details="; ".join(findings),
            cwe="CWE-200"
        )

    return AttackResult(
        attack_vector="private_key_handling", tool_name=tool.name,
        severity="none", vulnerable=False,
        details="No private key handling issues detected",
        cwe="CWE-200"
    )


# ============================================================
# SCORING
# ============================================================

SEVERITY_SCORE = {"critical": 10, "high": 7, "medium": 4, "low": 1, "none": 0}


def compute_repo_score(results: list) -> float:
    """Compute overall vulnerability score for a repo (0-100)."""
    if not results:
        return 0.0

    total = sum(SEVERITY_SCORE.get(r.severity, 0) for r in results if r.vulnerable)
    max_possible = len(results) * 10  # All critical
    if max_possible == 0:
        return 0.0

    import math
    # Log scale to prevent huge repos from dominating
    score = min(100.0, round(15 * math.log2(1 + total), 1))
    return score


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Dynamic Testing Harness for AI Agent Tool Interfaces")
    print("=" * 70)
    print()
    print("Attack vectors: tool_poisoning, prompt_injection_output,")
    print("  parameter_injection, tx_validation, private_key_handling")
    print()

    # ---- Select repos ----
    repos = select_web3_repos(CATALOG_PATH, MAX_REPOS)
    print(f"Selected {len(repos)} Web3 tool servers for testing:")
    for r in repos:
        print(f"  {r['full_name']} (stars={r.get('stars', '?')}, lang={r.get('language', '?')})")
    print()

    # ---- Prepare clone directory ----
    os.makedirs(CLONE_BASE, exist_ok=True)

    all_repo_results = []

    for i, repo in enumerate(repos, 1):
        name = repo["full_name"]
        print(f"\n--- [{i}/{len(repos)}] {name} ---")

        repo_result = RepoTestResult(
            repo=name,
            url=repo.get("url", f"https://github.com/{name}"),
            stars=repo.get("stars", 0),
            language=repo.get("language", "Unknown"),
        )

        # Clone
        clone_path = clone_repo(repo, CLONE_BASE)
        if not clone_path:
            repo_result.clone_success = False
            repo_result.errors.append("Clone failed")
            all_repo_results.append(repo_result)
            continue

        try:
            # Extract tool definitions
            tools = extract_all_tools(clone_path)
            repo_result.tools_found = len(tools)
            print(f"  Tools found: {len(tools)}")

            if not tools:
                # Try to extract from README or package.json tool descriptions
                print("  No tool definitions found via code parsing")
                repo_result.errors.append("No tool definitions found")
                all_repo_results.append(repo_result)
                continue

            # Read source files for context-aware analysis
            source_contents = {}
            for tool in tools:
                if tool.file not in source_contents:
                    fpath = os.path.join(clone_path, tool.file)
                    try:
                        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                            source_contents[tool.file] = f.read()
                    except (OSError, FileNotFoundError):
                        source_contents[tool.file] = ""

            # Run all 5 attack vector tests against each tool
            attack_results = []
            for tool in tools:
                src = source_contents.get(tool.file, "")

                r1 = test_tool_poisoning(tool)
                r2 = test_prompt_injection_output(tool, src)
                r3 = test_parameter_injection(tool)
                r4 = test_transaction_validation(tool, src)
                r5 = test_private_key_handling(tool, src)

                attack_results.extend([r1, r2, r3, r4, r5])

            repo_result.tools_tested = len(tools)
            repo_result.attack_results = attack_results
            repo_result.tool_definitions = tools
            repo_result.overall_score = compute_repo_score(attack_results)

            # Print summary
            vuln_count = sum(1 for r in attack_results if r.vulnerable)
            by_vector = defaultdict(int)
            for r in attack_results:
                if r.vulnerable:
                    by_vector[r.attack_vector] += 1

            print(f"  Vulnerabilities found: {vuln_count}/{len(attack_results)} tests")
            print(f"  Overall risk score: {repo_result.overall_score}")
            for vector, count in sorted(by_vector.items()):
                print(f"    {vector}: {count}")

            # Show tool names
            for t in tools[:10]:
                print(f"    Tool: {t.name} ({t.file})")

        except Exception as e:
            repo_result.errors.append(str(e))
            print(f"  ERROR: {e}")

        finally:
            # Cleanup clone
            if clone_path and os.path.exists(clone_path):
                shutil.rmtree(clone_path, ignore_errors=True)

        all_repo_results.append(repo_result)

        # Small delay to avoid GitHub rate limiting
        if i < len(repos):
            time.sleep(1)

    # ---- Cleanup base directory ----
    if os.path.exists(CLONE_BASE):
        shutil.rmtree(CLONE_BASE, ignore_errors=True)

    # ---- Aggregate results ----
    print()
    print("=" * 70)
    print("AGGREGATED RESULTS")
    print("=" * 70)

    total_tools = sum(r.tools_found for r in all_repo_results)
    total_tested = sum(r.tools_tested for r in all_repo_results)
    total_vulns = sum(
        sum(1 for a in r.attack_results if a.vulnerable)
        for r in all_repo_results
    )

    print(f"\nRepos tested:    {len(all_repo_results)}")
    print(f"Repos cloned:    {sum(1 for r in all_repo_results if r.clone_success)}")
    print(f"Tools found:     {total_tools}")
    print(f"Tools tested:    {total_tested}")
    print(f"Vulnerabilities: {total_vulns}")
    print()

    # By attack vector
    vector_stats = defaultdict(lambda: {"total": 0, "vulnerable": 0, "by_severity": Counter()})
    for r in all_repo_results:
        for a in r.attack_results:
            vector_stats[a.attack_vector]["total"] += 1
            if a.vulnerable:
                vector_stats[a.attack_vector]["vulnerable"] += 1
                vector_stats[a.attack_vector]["by_severity"][a.severity] += 1

    print(f"{'Attack Vector':<30} {'Tests':>6} {'Vuln':>6} {'Rate':>8} {'Crit':>5} {'High':>5} {'Med':>5}")
    print("-" * 75)
    for vector in ["tool_poisoning", "prompt_injection_output", "parameter_injection",
                    "tx_validation", "private_key_handling"]:
        if vector in vector_stats:
            s = vector_stats[vector]
            rate = (s["vulnerable"] / s["total"] * 100) if s["total"] > 0 else 0
            print(f"{vector:<30} {s['total']:>6} {s['vulnerable']:>6} {rate:>7.1f}% "
                  f"{s['by_severity'].get('critical', 0):>5} "
                  f"{s['by_severity'].get('high', 0):>5} "
                  f"{s['by_severity'].get('medium', 0):>5}")
    print()

    # Risk ranking
    print("--- Risk Ranking ---")
    ranked = sorted(all_repo_results, key=lambda r: r.overall_score, reverse=True)
    print(f"  {'Repo':<45} {'Score':>6} {'Tools':>6} {'Vulns':>6}")
    print("  " + "-" * 65)
    for r in ranked:
        vuln_count = sum(1 for a in r.attack_results if a.vulnerable)
        print(f"  {r.repo:<45} {r.overall_score:>6.1f} {r.tools_found:>6} {vuln_count:>6}")
    print()

    # ---- Save results ----
    output = {
        "metadata": {
            "test_type": "dynamic_tool_analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "repos_attempted": len(all_repo_results),
            "repos_cloned": sum(1 for r in all_repo_results if r.clone_success),
            "total_tools_found": total_tools,
            "total_tools_tested": total_tested,
            "total_vulnerabilities": total_vulns,
            "attack_vectors": [
                "tool_poisoning", "prompt_injection_output",
                "parameter_injection", "tx_validation", "private_key_handling",
            ],
        },
        "vector_summary": {
            v: {
                "tests": s["total"],
                "vulnerable": s["vulnerable"],
                "rate_pct": round(s["vulnerable"] / max(1, s["total"]) * 100, 1),
                "by_severity": dict(s["by_severity"]),
            }
            for v, s in vector_stats.items()
        },
        "risk_ranking": [
            {
                "repo": r.repo,
                "url": r.url,
                "stars": r.stars,
                "language": r.language,
                "tools_found": r.tools_found,
                "tools_tested": r.tools_tested,
                "vulnerabilities": sum(1 for a in r.attack_results if a.vulnerable),
                "overall_score": r.overall_score,
                "clone_success": r.clone_success,
            }
            for r in ranked
        ],
        "repo_results": [
            {
                "repo": r.repo,
                "url": r.url,
                "stars": r.stars,
                "language": r.language,
                "tools_found": r.tools_found,
                "tools_tested": r.tools_tested,
                "overall_score": r.overall_score,
                "clone_success": r.clone_success,
                "errors": r.errors,
                "tool_definitions": [t.to_dict() for t in r.tool_definitions],
                "attack_results": [a.to_dict() for a in r.attack_results],
            }
            for r in all_repo_results
        ],
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Saved results to: {OUTPUT_PATH}")
    print()


if __name__ == "__main__":
    main()
