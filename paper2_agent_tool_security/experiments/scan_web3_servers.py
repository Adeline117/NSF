"""
Paper 2: Scan Web3-Only Agent Tool Servers
==========================================
Large-scale scan of Web3/blockchain-specific MCP and agent tool servers.

This script:
  1. Uses a curated list of Web3-specific repos (from GitHub search + dynamic testing)
  2. Shallow-clones each repo
  3. Runs vulnerability scanning with the same patterns from scan_real_servers.py
  4. Additionally parses tool definitions (JSON schemas, TS interfaces)
  5. Identifies sensitive operations (signing, approving, transferring)
  6. Combines results with dynamic_test_results.json
  7. Saves unified results

Usage:
    python paper2_agent_tool_security/experiments/scan_web3_servers.py

No external dependencies -- uses Python standard library only.
"""

import json
import math
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

OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "web3_scan_results.json")
UNIFIED_PATH = os.path.join(os.path.dirname(__file__), "unified_web3_results.json")
DYNAMIC_RESULTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dynamic_testing", "dynamic_test_results.json"
)

CLONE_DELAY_SECONDS = 2.0
CLONE_TIMEOUT_SECONDS = 90
MAX_FILES_PER_REPO = 500
MAX_FILE_SIZE_BYTES = 500_000  # 500KB per file

SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", "venv", ".venv",
    "dist", "build", "out", ".next", "coverage", "artifacts",
    "cache", "typechain-types", ".tox", ".mypy_cache", ".pytest_cache",
    "vendor", "bower_components", ".yarn", ".pnp", "target",
    "lib", "deps", ".deps",
}

SOURCE_EXTENSIONS = {
    ".ts", ".js", ".py", ".sol", ".tsx", ".jsx",
    ".mts", ".mjs", ".rs", ".go",
}


# ============================================================
# CURATED WEB3 REPO LIST
# ============================================================
# All repos are Web3/blockchain-specific tool servers.
# Sources: GitHub search results + dynamic_test_results.json repos.
# Filtered: only repos whose name or description contains
# web3/crypto/blockchain/ethereum/solana/defi/wallet/token/swap/nft.

WEB3_REPOS = [
    # === Already tested in dynamic testing (6 real Web3 servers) ===
    {"full_name": "armorwallet/armor-crypto-mcp", "stars": 192,
     "url": "https://github.com/armorwallet/armor-crypto-mcp",
     "description": "MCP server for interacting with Blockchain, Swaps, Strategic Planning and more.",
     "source": "dynamic_testing"},
    {"full_name": "base/base-mcp", "stars": 342,
     "url": "https://github.com/base/base-mcp",
     "description": "Base blockchain MCP server",
     "source": "dynamic_testing"},
    {"full_name": "mcpdotdirect/evm-mcp-server", "stars": 373,
     "url": "https://github.com/mcpdotdirect/evm-mcp-server",
     "description": "MCP server for interacting with EVM networks",
     "source": "dynamic_testing"},
    {"full_name": "caiovicentino/polymarket-mcp-server", "stars": 348,
     "url": "https://github.com/caiovicentino/polymarket-mcp-server",
     "description": "AI-Powered MCP Server for Polymarket prediction markets",
     "source": "dynamic_testing"},
    {"full_name": "solanamcp/solana-mcp", "stars": 88,
     "url": "https://github.com/solanamcp/solana-mcp",
     "description": "Solana MCP server",
     "source": "dynamic_testing"},
    {"full_name": "trailofbits/slither-mcp", "stars": 81,
     "url": "https://github.com/trailofbits/slither-mcp",
     "description": "Slither smart contract security analyzer MCP server",
     "source": "dynamic_testing"},
    {"full_name": "nirholas/pump-fun-sdk", "stars": 75,
     "url": "https://github.com/nirholas/pump-fun-sdk",
     "description": "Pump.fun SDK for Solana token operations",
     "source": "dynamic_testing"},
    {"full_name": "alpacahq/alpaca-mcp-server", "stars": 613,
     "url": "https://github.com/alpacahq/alpaca-mcp-server",
     "description": "Alpaca trading MCP server (crypto/stock trading)",
     "source": "dynamic_testing"},
    {"full_name": "nirholas/free-crypto-news", "stars": 135,
     "url": "https://github.com/nirholas/free-crypto-news",
     "description": "Free crypto news aggregation tool",
     "source": "dynamic_testing"},

    # === From GitHub search: high-star Web3 MCP repos ===
    {"full_name": "hydra-mcp/hydra-mcp-solana", "stars": 238,
     "url": "https://github.com/hydra-mcp/hydra-mcp-solana",
     "description": "Hydra AI Solana MCP server",
     "source": "github_search"},
    {"full_name": "berlinbra/polymarket-mcp", "stars": 130,
     "url": "https://github.com/berlinbra/polymarket-mcp",
     "description": "MCP Server for PolyMarket API",
     "source": "github_search"},
    {"full_name": "bnb-chain/bnbchain-mcp", "stars": 58,
     "url": "https://github.com/bnb-chain/bnbchain-mcp",
     "description": "MCP server for BNB Chain, BSC, opBNB, Greenfield, and EVM-compatible networks",
     "source": "github_search"},
    {"full_name": "ozgureyilmaz/polymarket-mcp", "stars": 48,
     "url": "https://github.com/ozgureyilmaz/polymarket-mcp",
     "description": "MCP server for polymarket",
     "source": "github_search"},
    {"full_name": "truss44/mcp-crypto-price", "stars": 39,
     "url": "https://github.com/truss44/mcp-crypto-price",
     "description": "MCP server for real-time cryptocurrency analysis via CoinCap API",
     "source": "github_search"},
    {"full_name": "nirholas/UCAI", "stars": 28,
     "url": "https://github.com/nirholas/UCAI",
     "description": "Universal Contract AI Interface: ABI to MCP, smart contract MCP generator",
     "source": "github_search"},
    {"full_name": "zhangzhongnan928/mcp-blockchain-server", "stars": 10,
     "url": "https://github.com/zhangzhongnan928/mcp-blockchain-server",
     "description": "MCP Server for blockchain interactions with Web DApp for secure transaction signing",
     "source": "github_search"},
    {"full_name": "dcSpark/mcp-cryptowallet-evm", "stars": 9,
     "url": "https://github.com/dcSpark/mcp-cryptowallet-evm",
     "description": "MCP crypto wallet for EVM chains",
     "source": "github_search"},
    {"full_name": "magnetai/mcp-crypto", "stars": 8,
     "url": "https://github.com/magnetai/mcp-crypto",
     "description": "A DAO ecosystem for AI x Crypto projects contributing to MCP",
     "source": "github_search"},
    {"full_name": "kukapay/wallet-inspector-mcp", "stars": 8,
     "url": "https://github.com/kukapay/wallet-inspector-mcp",
     "description": "MCP server to inspect wallet balance and onchain activity across EVM and Solana chains",
     "source": "github_search"},
    {"full_name": "pab1it0/polymarket-mcp", "stars": 7,
     "url": "https://github.com/pab1it0/polymarket-mcp",
     "description": "MCP server for Polymarket Gamma Markets API",
     "source": "github_search"},
    {"full_name": "z1labs/mcp-nft1155-badge", "stars": 6,
     "url": "https://github.com/z1labs/mcp-nft1155-badge",
     "description": "ERC-1155 context badges using MCP, eligibility check, CID verification, badge minting",
     "source": "github_search"},
    {"full_name": "Suryansh-23/hyperlane-mcp", "stars": 6,
     "url": "https://github.com/Suryansh-23/hyperlane-mcp",
     "description": "MCP server for cross-chain messaging and asset transfers via Hyperlane Protocol",
     "source": "github_search"},
    {"full_name": "InjectiveLabs/mcp-server", "stars": 6,
     "url": "https://github.com/InjectiveLabs/mcp-server",
     "description": "MCP server for Injective: perpetual futures, spot transfers, cross-chain bridging, raw EVM transactions",
     "source": "github_search"},
    {"full_name": "aryankeluskar/polymarket-mcp", "stars": 6,
     "url": "https://github.com/aryankeluskar/polymarket-mcp",
     "description": "MCP Server for Polymarket API, 2900+ downloads",
     "source": "github_search"},
    {"full_name": "wallet-agent/wallet-agent", "stars": 5,
     "url": "https://github.com/wallet-agent/wallet-agent",
     "description": "MCP server for Web3 wallet interactions on EVM-compatible chains",
     "source": "github_search"},
    {"full_name": "z1labs/mcp-nft721-contract", "stars": 5,
     "url": "https://github.com/z1labs/mcp-nft721-contract",
     "description": "ERC-721 contract for minting AI chat sessions as NFTs with MCP model hash",
     "source": "github_search"},
    {"full_name": "crazyrabbitLTC/mcp-web3-stats", "stars": 4,
     "url": "https://github.com/crazyrabbitLTC/mcp-web3-stats",
     "description": "MCP server for blockchain wallet analysis and token data via Dune API",
     "source": "github_search"},
    {"full_name": "NFTGo/mcp-nftgo-api", "stars": 4,
     "url": "https://github.com/NFTGo/mcp-nftgo-api",
     "description": "NFTGo MCP server for NFT analytics",
     "source": "github_search"},
    {"full_name": "efekucuk/etherlink-mcp-server", "stars": 4,
     "url": "https://github.com/efekucuk/etherlink-mcp-server",
     "description": "MCP server for Etherlink blockchain (EVM-compatible L2 on Tezos)",
     "source": "github_search"},
    {"full_name": "agenticvault/agentic-vault", "stars": 4,
     "url": "https://github.com/agenticvault/agentic-vault",
     "description": "Server-side EVM signing with AWS KMS and DeFi protocol awareness, expose wallet to AI agents via MCP",
     "source": "github_search"},
    {"full_name": "lucianoayres/mcp-crypto-ticker", "stars": 3,
     "url": "https://github.com/lucianoayres/mcp-crypto-ticker",
     "description": "MCP Crypto Ticker tool for real-time cryptocurrency data in AI-powered IDEs",
     "source": "github_search"},
    {"full_name": "waifuai/mcp-solana-ico", "stars": 3,
     "url": "https://github.com/waifuai/mcp-solana-ico",
     "description": "MCP server for Solana ICO management with bonding curves",
     "source": "github_search"},
    {"full_name": "z1labs/mcp-nft1155-contract", "stars": 3,
     "url": "https://github.com/z1labs/mcp-nft1155-contract",
     "description": "ERC-1155 contract for minting context-based badges using MCP",
     "source": "github_search"},
    {"full_name": "IQAIcom/mcp-defillama", "stars": 2,
     "url": "https://github.com/IQAIcom/mcp-defillama",
     "description": "MCP Server for interacting with Defillama services",
     "source": "github_search"},
    {"full_name": "waifuai/mcp-solana-dex", "stars": 2,
     "url": "https://github.com/waifuai/mcp-solana-dex",
     "description": "FastMCP server implementing basic Solana DEX operations",
     "source": "github_search"},
    {"full_name": "kukapay/chainlist-mcp", "stars": 2,
     "url": "https://github.com/kukapay/chainlist-mcp",
     "description": "MCP server for verified EVM chain information",
     "source": "github_search"},
    {"full_name": "StronglyTypedSoul/sui-mcp-server-wallet-management", "stars": 2,
     "url": "https://github.com/StronglyTypedSoul/sui-mcp-server-wallet-management",
     "description": "Comprehensive Sui blockchain MCP server for wallet management and transactions",
     "source": "github_search"},
    {"full_name": "N-45div/SolMCP-Solana-MCP-Server", "stars": 2,
     "url": "https://github.com/N-45div/SolMCP-Solana-MCP-Server",
     "description": "Solana MCP Server",
     "source": "github_search"},
    {"full_name": "dcSpark/mcp-cryptowallet-solana", "stars": 1,
     "url": "https://github.com/dcSpark/mcp-cryptowallet-solana",
     "description": "MCP crypto wallet for Solana",
     "source": "github_search"},
    {"full_name": "waifuai/mcp-solana-internet", "stars": 1,
     "url": "https://github.com/waifuai/mcp-solana-internet",
     "description": "Solana-based micropayment server using MCP",
     "source": "github_search"},
    {"full_name": "dennisonbertram/mcp-web3-wallet-tester", "stars": 1,
     "url": "https://github.com/dennisonbertram/mcp-web3-wallet-tester",
     "description": "MCP server as Ethereum wallet for LLM-controlled Web3 dApp testing",
     "source": "github_search"},
    {"full_name": "rafael-pina/polymarket-mcp", "stars": 5,
     "url": "https://github.com/rafael-pina/polymarket-mcp",
     "description": "Polymarket MCP server",
     "source": "github_search"},
]

# Total: 42 unique Web3 repos


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Finding:
    pattern_id: str
    category: str
    severity: str
    description: str
    file: str
    line: int
    matched_text: str
    context: str
    cwe: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class ToolDefinition:
    name: str
    description: str
    file: str
    line: int
    parameters: dict = field(default_factory=dict)
    is_sensitive: bool = False
    sensitive_type: str = ""
    has_return_schema: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class RepoResult:
    full_name: str
    url: str
    stars: int
    source: str
    detected_protocol: str
    protocol_confidence: float
    language: str = "Unknown"
    files_scanned: int = 0
    scan_time_seconds: float = 0.0
    total_findings: int = 0
    risk_score: float = 0.0
    risk_rating: str = "unknown"
    by_severity: dict = field(default_factory=dict)
    by_category: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    tool_definitions: list = field(default_factory=list)
    tools_found: int = 0
    sensitive_tools: int = 0
    errors: list = field(default_factory=list)
    clone_success: bool = True

    def to_dict(self):
        return asdict(self)


# ============================================================
# PROTOCOL DETECTION
# ============================================================

PROTOCOL_INDICATORS = {
    "mcp": [
        {"pattern": r"@modelcontextprotocol|from mcp import|mcp\.Server|McpServer",
         "weight": 10.0},
        {"pattern": r"server\.tool\(|@tool|mcp_server",
         "weight": 5.0},
        {"pattern": r"\"@modelcontextprotocol/sdk\"",
         "weight": 8.0},
    ],
    "openai": [
        {"pattern": r"import openai|from openai|openai\.chat",
         "weight": 8.0},
        {"pattern": r"function_call|tool_choice|\"functions\"\s*:|tools\s*=\s*\[",
         "weight": 7.0},
    ],
    "langchain": [
        {"pattern": r"from langchain|import langchain|from crewai|from autogen",
         "weight": 10.0},
        {"pattern": r"@tool|BaseTool|StructuredTool|initialize_agent|AgentExecutor",
         "weight": 7.0},
    ],
    "web3_native": [
        {"pattern": r"pragma solidity|// SPDX-License-Identifier",
         "weight": 10.0},
        {"pattern": r"ISafe|IModule|IGuard|ModuleManager|GnosisSafe",
         "weight": 8.0},
        {"pattern": r"ERC4337|IEntryPoint|UserOperation|IPlugin|IValidator",
         "weight": 9.0},
        {"pattern": r"delegatecall|execTransactionFromModule",
         "weight": 6.0},
    ],
}


def detect_protocol(repo_path: str, max_files: int = 200) -> tuple:
    """Detect the protocol family of a repository."""
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
# VULNERABILITY PATTERNS (same as scan_real_servers.py)
# ============================================================

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
     "Tool/function description contains directive instructions",
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
     "Destructive operation without scope restriction",
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

    # --- Web3-specific additional patterns ---
    ("SLP-001", "no_slippage_protection", "high",
     "Swap/trade without slippage protection",
     r"(?:swap|exchange|trade)\s*\((?!.*(?:slippage|minOutput|minAmount|deadline|amountOutMin))",
     "CWE-345", None),

    ("MEV-001", "mev_exposure", "high",
     "Transaction sent to public mempool without private relay",
     r"(?:sendTransaction|broadcast)\s*\((?!.*(?:private|flashbots|mev|protect|bundle))",
     "CWE-200", None),

    ("ENV-001", "env_key_exposure", "high",
     "Private key read from environment variable without protection",
     r"(?:process\.env|os\.environ|os\.getenv)\s*[\.\[\(]\s*[\"']?(?:PRIVATE_KEY|SECRET_KEY|MNEMONIC|SEED)",
     "CWE-312", None),
]


# ============================================================
# TOOL DEFINITION PARSING
# ============================================================

SENSITIVE_OPERATIONS = {
    "signing": ["sign", "signTransaction", "signMessage", "eth_sign",
                "personal_sign", "signTypedData", "sign_transaction"],
    "approving": ["approve", "increaseAllowance", "setApprovalForAll",
                  "unlimited_approval", "erc20_approve"],
    "transferring": ["transfer", "transferFrom", "sendTransaction",
                    "send_transaction", "sendRawTransaction", "send",
                    "withdraw", "deposit", "swap", "trade", "bridge"],
    "key_management": ["exportKey", "importKey", "generateKey",
                      "createWallet", "deriveKey", "getPrivateKey"],
}


def parse_tool_definitions(repo_path: str) -> list:
    """Parse tool definitions from MCP servers, OpenAI function defs, etc.
    Returns list of ToolDefinition objects.
    """
    tools = []

    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in SOURCE_EXTENSIONS and ext != ".json":
                continue
            filepath = os.path.join(root, fname)
            try:
                size = os.path.getsize(filepath)
                if size > MAX_FILE_SIZE_BYTES:
                    continue
                with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                    content = f.read()
            except (OSError, PermissionError):
                continue

            rel_path = os.path.relpath(filepath, repo_path)

            if ext in (".ts", ".tsx", ".js", ".jsx", ".mts", ".mjs"):
                tools.extend(_parse_ts_tools(content, rel_path))
            elif ext == ".py":
                tools.extend(_parse_py_tools(content, rel_path))
            elif ext == ".json" and "tool" in fname.lower():
                tools.extend(_parse_json_tools(content, rel_path))

    return tools


def _parse_ts_tools(content: str, rel_path: str) -> list:
    """Parse TypeScript/JavaScript tool definitions."""
    tools = []
    lines = content.split("\n")

    # Pattern 1: server.tool("name", { ... })
    for m in re.finditer(
        r'(?:server|mcp)\.tool\s*\(\s*["\']([^"\']+)["\']',
        content
    ):
        name = m.group(1)
        pos = m.start()
        line_num = content[:pos].count("\n") + 1
        desc = _extract_nearby_description(content, pos)
        tool = ToolDefinition(
            name=name, description=desc, file=rel_path, line=line_num
        )
        _check_sensitive(tool)
        tools.append(tool)

    # Pattern 2: { name: "...", description: "..." }
    for m in re.finditer(
        r'name\s*:\s*["\']([^"\']+)["\'].*?description\s*:\s*["\']([^"\']+)["\']',
        content, re.DOTALL
    ):
        name = m.group(1)
        desc = m.group(2)
        pos = m.start()
        line_num = content[:pos].count("\n") + 1
        tool = ToolDefinition(
            name=name, description=desc, file=rel_path, line=line_num
        )
        _check_sensitive(tool)
        tools.append(tool)

    return tools


def _parse_py_tools(content: str, rel_path: str) -> list:
    """Parse Python tool definitions."""
    tools = []

    # Pattern 1: @tool decorator
    for m in re.finditer(
        r'@tool\s*(?:\([^)]*\))?\s*\ndef\s+(\w+)',
        content
    ):
        name = m.group(1)
        pos = m.start()
        line_num = content[:pos].count("\n") + 1
        desc = _extract_docstring(content, m.end())
        tool = ToolDefinition(
            name=name, description=desc, file=rel_path, line=line_num
        )
        _check_sensitive(tool)
        tools.append(tool)

    # Pattern 2: Tool(name="...", ...)
    for m in re.finditer(
        r'Tool\s*\(\s*name\s*=\s*["\']([^"\']+)["\']',
        content
    ):
        name = m.group(1)
        pos = m.start()
        line_num = content[:pos].count("\n") + 1
        tool = ToolDefinition(
            name=name, description="", file=rel_path, line=line_num
        )
        _check_sensitive(tool)
        tools.append(tool)

    return tools


def _parse_json_tools(content: str, rel_path: str) -> list:
    """Parse JSON tool schema definitions."""
    tools = []
    try:
        data = json.loads(content)
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and "name" in item:
                    tool = ToolDefinition(
                        name=item["name"],
                        description=item.get("description", ""),
                        file=rel_path,
                        line=0,
                        parameters=item.get("parameters", {}),
                        has_return_schema="returns" in item or "output" in item,
                    )
                    _check_sensitive(tool)
                    tools.append(tool)
        elif isinstance(data, dict):
            if "tools" in data:
                for item in data["tools"]:
                    if isinstance(item, dict) and "name" in item:
                        tool = ToolDefinition(
                            name=item["name"],
                            description=item.get("description", ""),
                            file=rel_path,
                            line=0,
                            parameters=item.get("parameters", {}),
                            has_return_schema="returns" in item or "output" in item,
                        )
                        _check_sensitive(tool)
                        tools.append(tool)
    except (json.JSONDecodeError, TypeError):
        pass
    return tools


def _extract_nearby_description(content: str, pos: int) -> str:
    """Extract description from nearby context."""
    window = content[max(0, pos - 200):min(len(content), pos + 500)]
    m = re.search(r'description\s*[:=]\s*["\']([^"\']{5,200})["\']', window)
    if m:
        return m.group(1)
    return ""


def _extract_docstring(content: str, pos: int) -> str:
    """Extract Python docstring after function def."""
    remaining = content[pos:pos + 500]
    m = re.search(r':\s*\n\s*"""([^"]{5,200})"""', remaining)
    if m:
        return m.group(1).strip()
    m = re.search(r":\s*\n\s*'''([^']{5,200})'''", remaining)
    if m:
        return m.group(1).strip()
    return ""


def _check_sensitive(tool: ToolDefinition):
    """Check if a tool handles sensitive operations."""
    name_lower = tool.name.lower()
    desc_lower = tool.description.lower()

    for op_type, keywords in SENSITIVE_OPERATIONS.items():
        for kw in keywords:
            if kw.lower() in name_lower or kw.lower() in desc_lower:
                tool.is_sensitive = True
                tool.sensitive_type = op_type
                return


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
    """Scan a single file for vulnerability patterns."""
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
        if file_exts is not None and ext not in file_exts:
            continue
        try:
            for match in re.finditer(regex, content, re.IGNORECASE | re.MULTILINE):
                pos = match.start()
                line_num = content[:pos].count("\n") + 1
                line_idx = line_num - 1
                matched_text = match.group()[:200]
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
    """Compute risk score and rating."""
    severity_weights = {
        "critical": 10.0,
        "high": 7.5,
        "medium": 5.0,
        "low": 2.5,
    }
    if not findings:
        return 0.0, "safe"

    total = sum(severity_weights.get(f.severity, 1.0) for f in findings)
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
# CLONE & SCAN
# ============================================================

def clone_repo(url: str, dest: str, timeout: int = CLONE_TIMEOUT_SECONDS) -> bool:
    """Shallow clone a repo."""
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


def detect_language(repo_path: str) -> str:
    """Detect primary language from file extensions."""
    ext_counts = {}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SOURCE_EXTENSIONS:
                ext_counts[ext] = ext_counts.get(ext, 0) + 1

    if not ext_counts:
        return "Unknown"

    ext_to_lang = {
        ".ts": "TypeScript", ".tsx": "TypeScript", ".mts": "TypeScript",
        ".js": "JavaScript", ".jsx": "JavaScript", ".mjs": "JavaScript",
        ".py": "Python", ".sol": "Solidity", ".rs": "Rust", ".go": "Go",
    }
    top_ext = max(ext_counts, key=ext_counts.get)
    return ext_to_lang.get(top_ext, "Unknown")


def scan_single_repo(repo_info: dict) -> RepoResult:
    """Clone, scan, parse tools, and clean up a single repo."""
    full_name = repo_info["full_name"]
    url = repo_info["url"]
    stars = repo_info.get("stars", 0)
    source = repo_info.get("source", "unknown")

    result = RepoResult(
        full_name=full_name,
        url=url,
        stars=stars,
        source=source,
        detected_protocol="unknown",
        protocol_confidence=0.0,
    )

    tmp_dir = tempfile.mkdtemp(prefix="web3scan_")
    clone_path = os.path.join(tmp_dir, full_name.split("/")[-1])

    try:
        print(f"\n{'='*60}")
        print(f"Scanning: {full_name} ({stars} stars)")
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

        # Detect language
        result.language = detect_language(clone_path)

        # Detect protocol
        detected, confidence = detect_protocol(clone_path)
        result.detected_protocol = detected
        result.protocol_confidence = confidence
        print(f"  Protocol: {detected} (confidence: {confidence:.2f})")
        print(f"  Language: {result.language}")

        # Parse tool definitions
        print(f"  Parsing tool definitions...")
        tool_defs = parse_tool_definitions(clone_path)
        result.tool_definitions = [t.to_dict() for t in tool_defs]
        result.tools_found = len(tool_defs)
        result.sensitive_tools = sum(1 for t in tool_defs if t.is_sensitive)
        print(f"  Tools found: {len(tool_defs)}, Sensitive: {result.sensitive_tools}")
        for t in tool_defs:
            sens = f" [SENSITIVE: {t.sensitive_type}]" if t.is_sensitive else ""
            print(f"    - {t.name}{sens}")

        # Scan for vulnerabilities
        print(f"  Scanning for vulnerabilities...")
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
        print(f"  By category: {cat_counts}")
        print(f"  Risk: {score} ({rating})")

    except Exception as e:
        result.errors.append(str(e))
        print(f"  ERROR: {e}")

    finally:
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
    successful = [r for r in results if r.clone_success]
    total_findings = sum(r.total_findings for r in successful)
    total_tools = sum(r.tools_found for r in successful)
    total_sensitive = sum(r.sensitive_tools for r in successful)

    # Overall severity
    overall_severity = {}
    for r in successful:
        for sev, count in r.by_severity.items():
            overall_severity[sev] = overall_severity.get(sev, 0) + count

    # Overall category
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
                "repos": 0, "total_findings": 0, "avg_findings": 0,
                "avg_risk_score": 0, "total_tools": 0, "sensitive_tools": 0,
                "by_severity": {}, "by_category": {},
            }
        ps = protocol_stats[proto]
        ps["repos"] += 1
        ps["total_findings"] += r.total_findings
        ps["avg_risk_score"] += r.risk_score
        ps["total_tools"] += r.tools_found
        ps["sensitive_tools"] += r.sensitive_tools
        for sev, count in r.by_severity.items():
            ps["by_severity"][sev] = ps["by_severity"].get(sev, 0) + count
        for cat, count in r.by_category.items():
            ps["by_category"][cat] = ps["by_category"].get(cat, 0) + count

    for proto, ps in protocol_stats.items():
        if ps["repos"] > 0:
            ps["avg_findings"] = round(ps["total_findings"] / ps["repos"], 1)
            ps["avg_risk_score"] = round(ps["avg_risk_score"] / ps["repos"], 1)

    # Risk ranking
    risk_ranking = sorted(
        [(r.full_name, r.risk_score, r.risk_rating, r.total_findings,
          r.detected_protocol, r.stars, r.tools_found, r.sensitive_tools)
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
        "total_tools_found": total_tools,
        "total_sensitive_tools": total_sensitive,
        "overall_by_severity": overall_severity,
        "overall_by_category": overall_category,
        "cross_protocol_stats": protocol_stats,
        "risk_ranking": [
            {"repo": name, "risk_score": score, "risk_rating": rating,
             "findings": findings, "protocol": proto, "stars": stars,
             "tools": tools, "sensitive_tools": sens}
            for name, score, rating, findings, proto, stars, tools, sens
            in risk_ranking
        ],
    }


def load_dynamic_results() -> dict:
    """Load dynamic test results if available."""
    path = os.path.abspath(DYNAMIC_RESULTS_PATH)
    if not os.path.exists(path):
        print(f"  Dynamic results not found at {path}")
        return {}
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  Failed to load dynamic results: {e}")
        return {}


def build_unified_results(static_aggregate: dict, static_results: list,
                          dynamic_data: dict) -> dict:
    """Combine static scan + dynamic test results into unified statistics."""

    # Extract dynamic summary
    dyn_meta = dynamic_data.get("metadata", {})
    dyn_vectors = dynamic_data.get("vector_summary", {})
    dyn_repos = dynamic_data.get("risk_ranking", [])

    # Repos in dynamic that we also scanned
    static_repos = {r.full_name for r in static_results if r.clone_success}
    dyn_repo_names = {r.get("repo", "") for r in dyn_repos}
    overlap = static_repos & dyn_repo_names
    dyn_only = dyn_repo_names - static_repos

    unified = {
        "metadata": {
            "type": "unified_web3_analysis",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "static_scan": {
                "repos_scanned": static_aggregate["repos_successful"],
                "total_findings": static_aggregate["total_findings"],
                "total_tools_found": static_aggregate["total_tools_found"],
                "total_sensitive_tools": static_aggregate["total_sensitive_tools"],
            },
            "dynamic_testing": {
                "repos_tested": dyn_meta.get("repos_cloned", 0),
                "tools_tested": dyn_meta.get("total_tools_tested", 0),
                "total_vulnerabilities": dyn_meta.get("total_vulnerabilities", 0),
            },
            "overlap_repos": len(overlap),
            "total_unique_repos": len(static_repos | dyn_repo_names),
        },
        "static_scan_summary": {
            "by_severity": static_aggregate["overall_by_severity"],
            "by_category": static_aggregate["overall_by_category"],
            "by_protocol": static_aggregate["cross_protocol_stats"],
            "risk_ranking": static_aggregate["risk_ranking"],
        },
        "dynamic_test_summary": {
            "attack_vectors": dyn_vectors,
            "repos_by_risk": dyn_repos,
        },
        "combined_statistics": {
            "total_repos_analyzed": len(static_repos | dyn_repo_names),
            "total_static_findings": static_aggregate["total_findings"],
            "total_dynamic_vulnerabilities": dyn_meta.get("total_vulnerabilities", 0),
            "combined_issues": (
                static_aggregate["total_findings"]
                + dyn_meta.get("total_vulnerabilities", 0)
            ),
        },
    }

    return unified


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("Paper 2: Web3-Only Agent Tool Server Scanner")
    print(f"Repos to scan: {len(WEB3_REPOS)}")
    print("=" * 70)

    # Sort by stars descending
    repos = sorted(WEB3_REPOS, key=lambda r: r.get("stars", 0), reverse=True)

    print(f"\nWeb3 repos ({len(repos)} total):")
    for i, r in enumerate(repos):
        print(f"  {i+1:2d}. {r['stars']:>4} stars | {r['full_name']}")

    # Scan each repo
    results = []
    start_time = time.time()

    for i, repo_info in enumerate(repos):
        print(f"\n[{i+1}/{len(repos)}]", end="")
        result = scan_single_repo(repo_info)
        results.append(result)

        if i < len(repos) - 1:
            time.sleep(CLONE_DELAY_SECONDS)

    total_time = time.time() - start_time

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    aggregate = aggregate_results(results)
    aggregate["total_scan_time_seconds"] = round(total_time, 1)

    # Build static scan output
    static_output = {
        "metadata": {
            "scan_type": "web3_only_server_scan",
            "timestamp": aggregate["timestamp"],
            "repos_attempted": aggregate["repos_attempted"],
            "repos_successful": aggregate["repos_successful"],
            "repos_failed": aggregate["repos_failed"],
            "total_scan_time_seconds": aggregate["total_scan_time_seconds"],
        },
        "summary": {
            "total_findings": aggregate["total_findings"],
            "total_tools_found": aggregate["total_tools_found"],
            "total_sensitive_tools": aggregate["total_sensitive_tools"],
            "overall_by_severity": aggregate["overall_by_severity"],
            "overall_by_category": aggregate["overall_by_category"],
        },
        "cross_protocol_comparison": aggregate["cross_protocol_stats"],
        "risk_ranking": aggregate["risk_ranking"],
        "repo_results": [r.to_dict() for r in results],
    }

    # Save static scan results
    output_path = os.path.abspath(OUTPUT_PATH)
    with open(output_path, "w") as f:
        json.dump(static_output, f, indent=2, default=str)
    print(f"\nStatic scan results saved to: {output_path}")

    # Load dynamic results and build unified output
    print("\nLoading dynamic test results...")
    dynamic_data = load_dynamic_results()
    unified = build_unified_results(aggregate, results, dynamic_data)

    unified_path = os.path.abspath(UNIFIED_PATH)
    with open(unified_path, "w") as f:
        json.dump(unified, f, indent=2, default=str)
    print(f"Unified results saved to: {unified_path}")

    # Print summary
    print("\n" + "=" * 70)
    print("WEB3 SCAN SUMMARY")
    print("=" * 70)
    successful = [r for r in results if r.clone_success]
    print(f"Repos scanned:      {len(successful)}/{len(results)}")
    print(f"Total findings:     {aggregate['total_findings']}")
    print(f"Total tools found:  {aggregate['total_tools_found']}")
    print(f"Sensitive tools:    {aggregate['total_sensitive_tools']}")
    print(f"Total scan time:    {total_time:.1f}s")

    print(f"\nBy severity:")
    for sev in ["critical", "high", "medium", "low"]:
        count = aggregate["overall_by_severity"].get(sev, 0)
        print(f"  {sev:>10}: {count}")

    print(f"\nBy category:")
    sorted_cats = sorted(
        aggregate["overall_by_category"].items(),
        key=lambda x: x[1], reverse=True
    )
    for cat, count in sorted_cats:
        print(f"  {cat:>30}: {count}")

    print(f"\nTop 10 riskiest repos:")
    for entry in aggregate["risk_ranking"][:10]:
        print(f"  {entry['risk_score']:>5.1f} ({entry['risk_rating']:>8}) | "
              f"{entry['findings']:>3} findings | {entry['tools']:>2} tools | "
              f"{entry['repo']}")

    print(f"\nProtocol distribution:")
    for proto, stats in aggregate["cross_protocol_stats"].items():
        print(f"  {proto:>12}: {stats['repos']} repos, "
              f"{stats['total_findings']} findings, "
              f"avg risk {stats['avg_risk_score']:.1f}")

    if dynamic_data:
        print(f"\nDynamic testing (from previous run):")
        dyn_meta = dynamic_data.get("metadata", {})
        print(f"  Repos tested:       {dyn_meta.get('repos_cloned', 0)}")
        print(f"  Tools tested:       {dyn_meta.get('total_tools_tested', 0)}")
        print(f"  Vulnerabilities:    {dyn_meta.get('total_vulnerabilities', 0)}")

    print(f"\nCOMBINED:")
    print(f"  Unique repos:       {unified['metadata']['total_unique_repos']}")
    print(f"  Static findings:    {unified['combined_statistics']['total_static_findings']}")
    print(f"  Dynamic vulns:      {unified['combined_statistics']['total_dynamic_vulnerabilities']}")
    print(f"  Total issues:       {unified['combined_statistics']['combined_issues']}")


if __name__ == "__main__":
    main()
