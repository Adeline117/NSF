# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in waifuai/mcp-solana-dex

## Summary

During our academic research on AI agent tool interface security, we identified **19** potential security vulnerabilities in your **waifuai/mcp-solana-dex** project (19 from static analysis).

- **Risk score:** 66.6/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 2

## Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 2 |
| Medium | 17 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 11 |
| State Confusion | 6 |
| No Slippage Protection | 2 |

## Top Findings (Most Severe)

### SLP-001: No Slippage Protection (HIGH)
- **File:** mcp_solana_dex/server.py
- **Line:** 4
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      2: Solana DEX Server Implementation
      3: 
  >>> 4: This is the main server implementation for a Solana-based decentralized exchange (DEX)
      5: that operates as an MCP (Model Context Protocol) service. The server provides tools
      6: for managing ICO token orders with the following key features:
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** mcp_solana_dex/__init__.py
- **Line:** 4
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      2: MCP Solana DEX Package
      3: 
  >>> 4: This package provides the core functionality for a Solana-based decentralized exchange
      5: (DEX) server that operates as an MCP (Model Context Protocol) service. The package
      6: includes the main server implementation with tools for managing token orders, persistence
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SC-001: State Confusion (MEDIUM)
- **File:** tests/integration/conftest.py
- **Line:** 19
- **Description:** Mutable global state shared across tool calls
- **CWE:** CWE-362
- **Code context:**
  ```
      17: - mock_context: Provides mock MCP Context for tool function testing
      18: - patched_server_module: Reloads server module with patched environment and
  >>> 19:   clears global state to ensure test isolation
      20: 
      21: The module ensures that each test runs with a clean slate and doesn't interfere
  ```
- **Impact:** Shared mutable state across sessions could allow one user to influence or read another user's operations.
- **Recommendation:** Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### SC-001: State Confusion (MEDIUM)
- **File:** tests/integration/conftest.py
- **Line:** 74
- **Description:** Mutable global state shared across tool calls
- **CWE:** CWE-362
- **Code context:**
  ```
      72:     """
      73:     Patches the ORDER_BOOK_FILE path and provides the reloaded server module.
  >>> 74:     Ensures a clean state (including global order_book dict) for each test.
      75:     """
      76:     # 1. Patch the ORDER_BOOK_FILE constant *before* importing/reloading
  ```
- **Impact:** Shared mutable state across sessions could allow one user to influence or read another user's operations.
- **Recommendation:** Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### SC-001: State Confusion (MEDIUM)
- **File:** tests/integration/conftest.py
- **Line:** 81
- **Description:** Mutable global state shared across tool calls
- **CWE:** CWE-362
- **Code context:**
  ```
      79:     print(f"Set ORDER_BOOK_FILE env var to: {temp_order_book_path}")
      80: 
  >>> 81:     # 2. Reload the server module to pick up the patched path and reset global state
      82:     # Important: Need to handle potential import errors if module structure changes
      83:     try:
  ```
- **Impact:** Shared mutable state across sessions could allow one user to influence or read another user's operations.
- **Recommendation:** Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

## Remediation Recommendations

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### No Slippage Protection
Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

## Disclosure Timeline

- **2026-04-07:** Vulnerabilities discovered during automated scanning
- **2026-04-07:** This report sent to maintainers
- **2026-07-06:** Planned public disclosure (per responsible disclosure standard, 90 days)

## About This Research

This work is part of an academic research project studying the security of AI agent tool interfaces across protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native modules). The study analyzed 43 Web3-related repositories using both static pattern analysis and dynamic tool-interface testing.

Paper submitted to [VENUE].

## Contact

For questions about this report, please contact:
- [Your Name] <[email]>
- [Advisor Name] <[email]>

We are happy to work with you on remediation and will update the planned public disclosure timeline if needed to allow adequate time for fixes.
