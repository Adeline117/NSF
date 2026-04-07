"""
Enumerate AI Agent Tool Servers from GitHub
============================================
Search GitHub for real servers across 4 protocol families:

Protocol 1 (MCP): search "mcp server" + "web3/defi/ethereum/solana"
Protocol 2 (OpenAI): search "function_call" + "web3/crypto"
Protocol 3 (LangChain): search "langchain tool" + "web3/blockchain"
Protocol 4 (Web3-Native): search "Safe module" + "ERC-4337 plugin"

Uses GitHub API via gh CLI.
Output: JSON catalog with repo metadata, language, stars, last updated.

Usage:
    python enumerate_servers.py                   # Run all protocols
    python enumerate_servers.py --protocol mcp    # Single protocol
    python enumerate_servers.py --output results/ # Custom output dir
    python enumerate_servers.py --max-per-query 50 --dry-run  # Preview

Requirements:
    - gh CLI installed and authenticated (gh auth login)
    - Python 3.10+
    - No external Python dependencies

Output:
    server_catalog.json -- deduplicated catalog of all discovered repos
    per-protocol JSON files with raw search results
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RepoRecord:
    """Metadata for a discovered GitHub repository."""
    full_name: str              # owner/repo
    url: str                    # https://github.com/owner/repo
    description: str
    language: str
    stars: int
    forks: int
    last_updated: str           # ISO 8601
    created_at: str
    topics: list[str] = field(default_factory=list)
    protocol: str = ""          # mcp | openai | langchain | web3_native
    search_query: str = ""      # query that found this repo
    is_archived: bool = False
    license: str = ""
    default_branch: str = "main"
    open_issues: int = 0
    size_kb: int = 0

    @property
    def owner(self) -> str:
        return self.full_name.split("/")[0] if "/" in self.full_name else ""

    @property
    def repo_name(self) -> str:
        return self.full_name.split("/")[1] if "/" in self.full_name else self.full_name


@dataclass
class SearchQuery:
    """A GitHub search query with metadata."""
    protocol: str
    query: str
    language_filter: Optional[str] = None
    description: str = ""


@dataclass
class EnumerationResult:
    """Complete result of a server enumeration run."""
    timestamp: str = ""
    total_repos: int = 0
    total_deduplicated: int = 0
    per_protocol: dict = field(default_factory=dict)
    repos: list = field(default_factory=list)
    queries_executed: int = 0
    errors: list = field(default_factory=list)


# ============================================================
# SEARCH QUERIES -- ALL 4 PROTOCOL FAMILIES
# ============================================================

SEARCH_QUERIES: list[SearchQuery] = [
    # -------------------------------------------------------
    # Protocol 1: MCP (Model Context Protocol)
    # -------------------------------------------------------
    SearchQuery(
        protocol="mcp",
        query="mcp server web3",
        description="MCP servers for Web3 interactions",
    ),
    SearchQuery(
        protocol="mcp",
        query="mcp server ethereum",
        description="MCP servers for Ethereum",
    ),
    SearchQuery(
        protocol="mcp",
        query="mcp server defi",
        description="MCP servers for DeFi protocols",
    ),
    SearchQuery(
        protocol="mcp",
        query="mcp server solana",
        description="MCP servers for Solana",
    ),
    SearchQuery(
        protocol="mcp",
        query="mcp server blockchain wallet",
        description="MCP servers for blockchain wallets",
    ),
    SearchQuery(
        protocol="mcp",
        query="modelcontextprotocol server crypto",
        description="MCP SDK-based crypto servers",
    ),
    SearchQuery(
        protocol="mcp",
        query="mcp server token swap",
        description="MCP servers for token swaps",
    ),
    SearchQuery(
        protocol="mcp",
        query="mcp tool server",
        language_filter="TypeScript",
        description="MCP tool servers in TypeScript",
    ),
    SearchQuery(
        protocol="mcp",
        query="mcp tool server",
        language_filter="Python",
        description="MCP tool servers in Python",
    ),

    # -------------------------------------------------------
    # Protocol 2: OpenAI Function Calling
    # -------------------------------------------------------
    SearchQuery(
        protocol="openai",
        query="openai function_call web3",
        description="OpenAI function calling for Web3",
    ),
    SearchQuery(
        protocol="openai",
        query="openai function calling crypto wallet",
        description="OpenAI function calling for crypto wallets",
    ),
    SearchQuery(
        protocol="openai",
        query="openai tools blockchain ethereum",
        description="OpenAI tools for blockchain",
    ),
    SearchQuery(
        protocol="openai",
        query="gpt function_call defi",
        description="GPT function calling for DeFi",
    ),
    SearchQuery(
        protocol="openai",
        query="openai assistant tools web3",
        description="OpenAI Assistants API with Web3 tools",
    ),
    SearchQuery(
        protocol="openai",
        query="chatgpt plugin crypto",
        description="ChatGPT plugins for crypto",
    ),
    SearchQuery(
        protocol="openai",
        query="openai function calling solana",
        description="OpenAI function calling for Solana",
    ),

    # -------------------------------------------------------
    # Protocol 3: LangChain / Agent Framework Tools
    # -------------------------------------------------------
    SearchQuery(
        protocol="langchain",
        query="langchain tool web3",
        description="LangChain tools for Web3",
    ),
    SearchQuery(
        protocol="langchain",
        query="langchain agent blockchain",
        description="LangChain agents for blockchain",
    ),
    SearchQuery(
        protocol="langchain",
        query="langchain tool ethereum defi",
        description="LangChain tools for Ethereum DeFi",
    ),
    SearchQuery(
        protocol="langchain",
        query="langchain agent crypto wallet",
        description="LangChain agents for crypto wallets",
    ),
    SearchQuery(
        protocol="langchain",
        query="langchain tool solana",
        description="LangChain tools for Solana",
    ),
    SearchQuery(
        protocol="langchain",
        query="crewai tool blockchain",
        description="CrewAI tools for blockchain",
    ),
    SearchQuery(
        protocol="langchain",
        query="autogen tool web3",
        description="AutoGen tools for Web3",
    ),
    SearchQuery(
        protocol="langchain",
        query="ai agent framework defi tool",
        language_filter="Python",
        description="Python AI agent frameworks for DeFi",
    ),

    # -------------------------------------------------------
    # Protocol 4: Web3-Native (Smart Contract Modules)
    # -------------------------------------------------------
    SearchQuery(
        protocol="web3_native",
        query="gnosis safe module",
        language_filter="Solidity",
        description="Gnosis Safe modules",
    ),
    SearchQuery(
        protocol="web3_native",
        query="safe module ai agent",
        description="Safe modules for AI agents",
    ),
    SearchQuery(
        protocol="web3_native",
        query="ERC-4337 plugin module",
        description="ERC-4337 account abstraction plugins",
    ),
    SearchQuery(
        protocol="web3_native",
        query="account abstraction module plugin",
        language_filter="Solidity",
        description="AA modules and plugins in Solidity",
    ),
    SearchQuery(
        protocol="web3_native",
        query="smart account plugin",
        description="Smart account plugins",
    ),
    SearchQuery(
        protocol="web3_native",
        query="safe guard module solidity",
        description="Safe guard modules",
    ),
    SearchQuery(
        protocol="web3_native",
        query="modular smart account",
        description="Modular smart accounts (ERC-6900, ERC-7579)",
    ),
    SearchQuery(
        protocol="web3_native",
        query="ERC-7579 module",
        description="ERC-7579 modular account modules",
    ),
]


# ============================================================
# GITHUB API INTERFACE
# ============================================================

class GitHubSearcher:
    """Search GitHub repositories using the gh CLI."""

    # GitHub search API rate limits: 30 requests/minute for authenticated,
    # 10 requests/minute for unauthenticated. We use authenticated.
    RATE_LIMIT_DELAY = 3.0  # seconds between API calls (conservative)
    MAX_RESULTS_PER_QUERY = 100  # GitHub search returns max 100 per page

    def __init__(self, max_per_query: int = 50, dry_run: bool = False,
                 verbose: bool = True):
        self.max_per_query = min(max_per_query, self.MAX_RESULTS_PER_QUERY)
        self.dry_run = dry_run
        self.verbose = verbose
        self._last_request_time = 0.0

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}",
                  file=sys.stderr)

    def _check_gh_auth(self) -> bool:
        """Verify gh CLI is installed and authenticated."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True, text=True, timeout=10
            )
            return result.returncode == 0
        except FileNotFoundError:
            self._log("ERROR: gh CLI not found. Install: https://cli.github.com/")
            return False
        except subprocess.TimeoutExpired:
            self._log("ERROR: gh auth status timed out")
            return False

    def _rate_limit(self) -> None:
        """Enforce rate limiting between API calls."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.RATE_LIMIT_DELAY:
            sleep_time = self.RATE_LIMIT_DELAY - elapsed
            self._log(f"  Rate limiting: sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)
        self._last_request_time = time.time()

    def search_repos(self, query: SearchQuery) -> list[RepoRecord]:
        """Execute a single search query and return matching repos."""
        search_string = query.query
        if query.language_filter:
            search_string += f" language:{query.language_filter}"

        self._log(f"Searching: {search_string}")

        if self.dry_run:
            self._log(f"  [DRY RUN] Would search: {search_string}")
            return []

        self._rate_limit()

        # Use gh api for richer results (includes topics, license, etc.)
        api_query = search_string.replace(" ", "+")
        cmd = [
            "gh", "api",
            f"search/repositories?q={api_query}&sort=stars&order=desc"
            f"&per_page={self.max_per_query}",
            "--jq", ".items"
        ]

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
        except subprocess.TimeoutExpired:
            self._log(f"  TIMEOUT: {search_string}")
            return []

        if result.returncode != 0:
            self._log(f"  ERROR ({result.returncode}): {result.stderr.strip()}")
            # Fallback to gh search repos CLI
            return self._search_repos_fallback(query)
            return []

        try:
            items = json.loads(result.stdout)
        except json.JSONDecodeError:
            self._log(f"  JSON parse error for query: {search_string}")
            return []

        if not isinstance(items, list):
            self._log(f"  Unexpected response type: {type(items)}")
            return []

        repos = []
        for item in items:
            try:
                repo = RepoRecord(
                    full_name=item.get("full_name", ""),
                    url=item.get("html_url", ""),
                    description=(item.get("description") or "")[:500],
                    language=item.get("language") or "Unknown",
                    stars=item.get("stargazers_count", 0),
                    forks=item.get("forks_count", 0),
                    last_updated=item.get("updated_at", ""),
                    created_at=item.get("created_at", ""),
                    topics=item.get("topics", []),
                    protocol=query.protocol,
                    search_query=query.query,
                    is_archived=item.get("archived", False),
                    license=(item.get("license") or {}).get("spdx_id", ""),
                    default_branch=item.get("default_branch", "main"),
                    open_issues=item.get("open_issues_count", 0),
                    size_kb=item.get("size", 0),
                )
                repos.append(repo)
            except (KeyError, TypeError) as e:
                self._log(f"  Skipping malformed item: {e}")
                continue

        self._log(f"  Found {len(repos)} repos")
        return repos

    def _search_repos_fallback(self, query: SearchQuery) -> list[RepoRecord]:
        """Fallback using gh search repos CLI when API fails."""
        self._rate_limit()

        search_string = query.query
        cmd = [
            "gh", "search", "repos", search_string,
            "--limit", str(self.max_per_query),
            "--json", "fullName,url,description,language,stargazersCount,"
                      "forksCount,updatedAt,createdAt,isArchived,"
                      "defaultBranch",
            "--sort", "stars",
            "--order", "desc",
        ]
        if query.language_filter:
            cmd.extend(["--language", query.language_filter])

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
        except subprocess.TimeoutExpired:
            self._log(f"  TIMEOUT (fallback): {search_string}")
            return []

        if result.returncode != 0:
            self._log(f"  FALLBACK ERROR: {result.stderr.strip()}")
            return []

        try:
            items = json.loads(result.stdout)
        except json.JSONDecodeError:
            return []

        repos = []
        for item in items:
            try:
                repo = RepoRecord(
                    full_name=item.get("fullName", ""),
                    url=item.get("url", ""),
                    description=(item.get("description") or "")[:500],
                    language=item.get("language") or "Unknown",
                    stars=item.get("stargazersCount", 0),
                    forks=item.get("forksCount", 0),
                    last_updated=item.get("updatedAt", ""),
                    created_at=item.get("createdAt", ""),
                    protocol=query.protocol,
                    search_query=query.query,
                    is_archived=item.get("isArchived", False),
                    default_branch=item.get("defaultBranch", "main"),
                )
                repos.append(repo)
            except (KeyError, TypeError) as e:
                self._log(f"  Skipping malformed item: {e}")
                continue

        self._log(f"  Found {len(repos)} repos (fallback)")
        return repos


# ============================================================
# DEDUPLICATION & FILTERING
# ============================================================

def deduplicate_repos(repos: list[RepoRecord]) -> list[RepoRecord]:
    """Deduplicate repos by full_name, keeping the record with highest stars.

    When a repo appears in multiple search queries, keep the entry found
    by the most specific query (protocol assignment from first hit) and
    the highest star count (in case of data staleness).
    """
    seen: dict[str, RepoRecord] = {}
    for repo in repos:
        key = repo.full_name.lower()
        if key not in seen:
            seen[key] = repo
        else:
            # Keep the one with more stars (fresher data)
            if repo.stars > seen[key].stars:
                # Preserve protocol from first hit
                old_protocol = seen[key].protocol
                seen[key] = repo
                seen[key].protocol = old_protocol
    return list(seen.values())


def filter_repos(repos: list[RepoRecord],
                 min_stars: int = 0,
                 exclude_archived: bool = True,
                 exclude_forks_with_no_stars: bool = True,
                 languages: Optional[list[str]] = None) -> list[RepoRecord]:
    """Apply quality filters to the repo list."""
    filtered = []
    for repo in repos:
        if exclude_archived and repo.is_archived:
            continue
        if repo.stars < min_stars:
            continue
        if languages and repo.language not in languages:
            continue
        filtered.append(repo)
    return filtered


def classify_protocol(repo: RepoRecord) -> str:
    """Attempt to classify a repo's protocol family from its metadata.

    Uses description, topics, and language as signals. Falls back to
    the protocol assigned by the search query.
    """
    text = f"{repo.description} {' '.join(repo.topics)}".lower()

    # Strong signals
    if "mcp" in text or "model context protocol" in text:
        return "mcp"
    if "function_call" in text or "function calling" in text or "openai" in text:
        return "openai"
    if "langchain" in text or "crewai" in text or "autogen" in text:
        return "langchain"
    if any(kw in text for kw in ["safe module", "erc-4337", "erc-7579",
                                  "account abstraction", "smart account"]):
        return "web3_native"

    # Language-based heuristic for ambiguous cases
    if repo.language == "Solidity":
        return "web3_native"

    # Fall back to search query assignment
    return repo.protocol


# ============================================================
# OUTPUT & REPORTING
# ============================================================

def generate_catalog(repos: list[RepoRecord]) -> EnumerationResult:
    """Generate the final catalog with statistics."""
    result = EnumerationResult(
        timestamp=datetime.now().isoformat(),
        total_repos=len(repos),
        total_deduplicated=len(repos),
        repos=[asdict(r) for r in repos],
    )

    # Per-protocol statistics
    protocols = {}
    for repo in repos:
        proto = repo.protocol
        if proto not in protocols:
            protocols[proto] = {
                "count": 0,
                "languages": {},
                "avg_stars": 0,
                "total_stars": 0,
                "top_repos": [],
            }
        protocols[proto]["count"] += 1
        protocols[proto]["total_stars"] += repo.stars

        lang = repo.language
        protocols[proto]["languages"][lang] = \
            protocols[proto]["languages"].get(lang, 0) + 1

    for proto, stats in protocols.items():
        if stats["count"] > 0:
            stats["avg_stars"] = stats["total_stars"] / stats["count"]
        # Top 5 repos by stars
        proto_repos = [r for r in repos if r.protocol == proto]
        proto_repos.sort(key=lambda r: r.stars, reverse=True)
        stats["top_repos"] = [r.full_name for r in proto_repos[:5]]

    result.per_protocol = protocols
    return result


def print_summary(result: EnumerationResult) -> None:
    """Print a human-readable summary to stderr."""
    print("\n" + "=" * 60, file=sys.stderr)
    print("SERVER ENUMERATION SUMMARY", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"Timestamp:    {result.timestamp}", file=sys.stderr)
    print(f"Total repos:  {result.total_deduplicated}", file=sys.stderr)
    print(f"Queries run:  {result.queries_executed}", file=sys.stderr)
    print(f"Errors:       {len(result.errors)}", file=sys.stderr)
    print("-" * 60, file=sys.stderr)

    for proto, stats in sorted(result.per_protocol.items()):
        print(f"\n  Protocol: {proto}", file=sys.stderr)
        print(f"    Repos:     {stats['count']}", file=sys.stderr)
        print(f"    Avg stars: {stats['avg_stars']:.1f}", file=sys.stderr)
        print(f"    Languages: {dict(stats['languages'])}", file=sys.stderr)
        print(f"    Top repos:", file=sys.stderr)
        for name in stats.get("top_repos", []):
            print(f"      - {name}", file=sys.stderr)

    if result.errors:
        print(f"\nErrors:", file=sys.stderr)
        for err in result.errors:
            print(f"  - {err}", file=sys.stderr)

    print("=" * 60, file=sys.stderr)


# ============================================================
# MAIN ORCHESTRATION
# ============================================================

def run_enumeration(
    protocols: Optional[list[str]] = None,
    max_per_query: int = 50,
    min_stars: int = 0,
    output_dir: str = ".",
    dry_run: bool = False,
    verbose: bool = True,
) -> EnumerationResult:
    """Run the full enumeration pipeline.

    Args:
        protocols: Filter to specific protocol families (None = all).
        max_per_query: Max repos per search query.
        min_stars: Minimum star count to include.
        output_dir: Directory for output JSON files.
        dry_run: If True, print queries but don't execute.
        verbose: Print progress to stderr.

    Returns:
        EnumerationResult with all discovered repos.
    """
    searcher = GitHubSearcher(
        max_per_query=max_per_query,
        dry_run=dry_run,
        verbose=verbose,
    )

    # Check authentication
    if not dry_run and not searcher._check_gh_auth():
        print("ERROR: GitHub CLI not authenticated. Run: gh auth login",
              file=sys.stderr)
        sys.exit(1)

    # Filter queries by protocol if specified
    queries = SEARCH_QUERIES
    if protocols:
        queries = [q for q in queries if q.protocol in protocols]

    if verbose:
        print(f"\nRunning {len(queries)} search queries across "
              f"{len(set(q.protocol for q in queries))} protocol families\n",
              file=sys.stderr)

    # Execute searches
    all_repos: list[RepoRecord] = []
    errors: list[str] = []
    queries_executed = 0

    for i, query in enumerate(queries):
        if verbose:
            print(f"[{i+1}/{len(queries)}] {query.description}",
                  file=sys.stderr)
        try:
            repos = searcher.search_repos(query)
            all_repos.extend(repos)
            queries_executed += 1
        except Exception as e:
            error_msg = f"Query failed: {query.query} -- {e}"
            errors.append(error_msg)
            if verbose:
                print(f"  ERROR: {e}", file=sys.stderr)

    # Deduplicate
    if verbose:
        print(f"\nTotal raw results: {len(all_repos)}", file=sys.stderr)

    deduped = deduplicate_repos(all_repos)
    if verbose:
        print(f"After deduplication: {len(deduped)}", file=sys.stderr)

    # Reclassify protocols based on content analysis
    for repo in deduped:
        repo.protocol = classify_protocol(repo)

    # Filter
    filtered = filter_repos(deduped, min_stars=min_stars)
    if verbose:
        print(f"After filtering (min_stars={min_stars}): {len(filtered)}",
              file=sys.stderr)

    # Sort by stars descending
    filtered.sort(key=lambda r: r.stars, reverse=True)

    # Generate catalog
    result = generate_catalog(filtered)
    result.queries_executed = queries_executed
    result.errors = errors

    # Save output
    os.makedirs(output_dir, exist_ok=True)

    catalog_path = os.path.join(output_dir, "server_catalog.json")
    with open(catalog_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    if verbose:
        print(f"\nCatalog written to: {catalog_path}", file=sys.stderr)

    # Per-protocol output files
    for proto in set(r.protocol for r in filtered):
        proto_repos = [r for r in filtered if r.protocol == proto]
        proto_path = os.path.join(output_dir, f"servers_{proto}.json")
        with open(proto_path, "w") as f:
            json.dump(
                {
                    "protocol": proto,
                    "count": len(proto_repos),
                    "repos": [asdict(r) for r in proto_repos],
                },
                f, indent=2, default=str,
            )
        if verbose:
            print(f"  {proto}: {len(proto_repos)} repos -> {proto_path}",
                  file=sys.stderr)

    # Print summary
    if verbose:
        print_summary(result)

    return result


# ============================================================
# CLI ENTRY POINT
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate AI agent tool servers from GitHub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              # Enumerate all protocols
  %(prog)s --protocol mcp              # MCP servers only
  %(prog)s --protocol mcp openai       # MCP + OpenAI
  %(prog)s --min-stars 10              # Only repos with 10+ stars
  %(prog)s --max-per-query 100         # Maximize results per query
  %(prog)s --output results/           # Custom output directory
  %(prog)s --dry-run                   # Preview queries without API calls
        """,
    )
    parser.add_argument(
        "--protocol", nargs="*", default=None,
        choices=["mcp", "openai", "langchain", "web3_native"],
        help="Protocol families to enumerate (default: all)",
    )
    parser.add_argument(
        "--max-per-query", type=int, default=50,
        help="Maximum repos per search query (default: 50, max: 100)",
    )
    parser.add_argument(
        "--min-stars", type=int, default=0,
        help="Minimum star count to include (default: 0)",
    )
    parser.add_argument(
        "--output", default=".",
        help="Output directory for JSON files (default: current dir)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print queries but don't execute API calls",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_enumeration(
        protocols=args.protocol,
        max_per_query=args.max_per_query,
        min_stars=args.min_stars,
        output_dir=args.output,
        dry_run=args.dry_run,
        verbose=not args.quiet,
    )

    # Print catalog to stdout for piping
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
