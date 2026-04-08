# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in IQAIcom/mcp-defillama

## Summary

During our academic research on AI agent tool interface security, we identified **13** potential security vulnerabilities in your **IQAIcom/mcp-defillama** project (13 from static analysis).

- **Risk score:** 60.4/100
- **Risk rating:** high
- **Protocol:** openai
- **GitHub stars:** 2

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Medium | 13 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| State Confusion | 9 |
| Missing Input Validation | 4 |

## Top Findings (Most Severe)

### SC-001: State Confusion (MEDIUM)
- **File:** src/tools/index.ts
- **Line:** 227
- **Description:** Mutable global state shared across tool calls
- **CWE:** CWE-362
- **Code context:**
  ```
      225: 		name: "defillama_get_dexs_data",
      226: 		description:
  >>> 227: 			"Fetches DEX (decentralized exchange) trading volume data and metrics. Returns different levels of data based on parameters: specific protocol data (most detailed), chain-specific overview, or global overview (all DEXs). **AUTO-RESOLUTION ENABLED:** Protocol and chain names are automatically matched.",
      228: 		parameters: z.object({
      229: 			excludeTotalDataChart: z
  ```
- **Impact:** Shared mutable state across sessions could allow one user to influence or read another user's operations.
- **Recommendation:** Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### SC-001: State Confusion (MEDIUM)
- **File:** src/tools/index.ts
- **Line:** 247
- **Description:** Mutable global state shared across tool calls
- **CWE:** CWE-362
- **Code context:**
  ```
      245: 				.optional()
      246: 				.describe(
  >>> 247: 					"Blockchain name - auto-resolved (e.g., 'Ethereum', 'BSC', 'Polygon', 'Arbitrum'). Returns overview of all DEXs operating on that chain. Only used when protocol is not specified. If both protocol and chain are omitted, returns global overview of all DEXs",
      248: 				),
      249: 			sortCondition: z
  ```
- **Impact:** Shared mutable state across sessions could allow one user to influence or read another user's operations.
- **Recommendation:** Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### SC-001: State Confusion (MEDIUM)
- **File:** src/tools/index.ts
- **Line:** 383
- **Description:** Mutable global state shared across tool calls
- **CWE:** CWE-362
- **Code context:**
  ```
      381: 		name: "defillama_get_stablecoin_charts",
      382: 		description:
  >>> 383: 			"Fetches historical market cap charts for stablecoins over time. Returns the last 10 data points showing circulation, USD values, and bridged amounts. Can filter by blockchain or specific stablecoin, or show global aggregated data",
      384: 		parameters: z.object({
      385: 			stablecoin: z
  ```
- **Impact:** Shared mutable state across sessions could allow one user to influence or read another user's operations.
- **Recommendation:** Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### SC-001: State Confusion (MEDIUM)
- **File:** src/tools/index.ts
- **Line:** 395
- **Description:** Mutable global state shared across tool calls
- **CWE:** CWE-362
- **Code context:**
  ```
      393: 				.optional()
      394: 				.describe(
  >>> 395: 					"Blockchain name to filter stablecoin data by (e.g., 'Ethereum', 'Polygon', 'Arbitrum'). Returns stablecoin market cap data for that specific chain. **If the user mentions a blockchain but you're unsure of the exact name format, call defillama_get_chains first to discover valid chain names**. If omitted, returns global aggregated data across all chains",
      396: 				),
      397: 			_userQuery: z.string().optional(),
  ```
- **Impact:** Shared mutable state across sessions could allow one user to influence or read another user's operations.
- **Recommendation:** Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### SC-001: State Confusion (MEDIUM)
- **File:** src/tools/index.ts
- **Line:** 717
- **Description:** Mutable global state shared across tool calls
- **CWE:** CWE-362
- **Code context:**
  ```
      715: 		name: "defillama_get_options_data",
      716: 		description:
  >>> 717: 			"Fetches options protocol data including trading volume and premium metrics. Returns different levels of data: specific protocol data (most detailed), chain-specific overview, or global overview (all options protocols). **AUTO-RESOLUTION ENABLED:** Protocol and chain names are automatically matched.",
      718: 		parameters: z.object({
      719: 			dataType: z
  ```
- **Impact:** Shared mutable state across sessions could allow one user to influence or read another user's operations.
- **Recommendation:** Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

## Remediation Recommendations

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

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
