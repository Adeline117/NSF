# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in solanamcp/solana-mcp

## Summary

During our academic research on AI agent tool interface security, we identified **10** potential security vulnerabilities in your **solanamcp/solana-mcp** project (10 from static analysis).

- **Risk score:** 61.5/100
- **Risk rating:** high
- **Protocol:** unknown
- **GitHub stars:** 88

## Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 8 |
| Medium | 2 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Tx Validation Missing | 4 |
| No Slippage Protection | 4 |
| Missing Input Validation | 2 |

## Top Findings (Most Severe)

### TV-001: Tx Validation Missing (HIGH)
- **File:** scripts/solana/jupapi/jupapi.js
- **Line:** 176
- **Description:** Transaction sent without validation/whitelist check
- **CWE:** CWE-345
- **Code context:**
  ```
      174:             } else {
      175:                 transaction.sign([owner]);
  >>> 176:                 txId = await myConnection.sendRawTransaction(transaction.serialize(), {
      177:                     skipPreflight: true,
      178:                     preflightCommitment: 'processed'
  ```
- **Impact:** Missing transaction validation could allow crafted transactions to drain funds or interact with malicious contracts.
- **Recommendation:** Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

### TV-001: Tx Validation Missing (HIGH)
- **File:** scripts/solana/jupapi/jupapi.js
- **Line:** 386
- **Description:** Transaction sent without validation/whitelist check
- **CWE:** CWE-345
- **Code context:**
  ```
      384:             // Then send the main transaction
      385:             transaction.sign([owner]);
  >>> 386:             txId = await myConnection.sendRawTransaction(transaction.serialize(), {
      387:                 skipPreflight: true,
      388:                 preflightCommitment: 'processed'
  ```
- **Impact:** Missing transaction validation could allow crafted transactions to drain funds or interact with malicious contracts.
- **Recommendation:** Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

### SLP-001: No Slippage Protection (HIGH)
- **File:** scripts/solana/jupapi/jupapi.js
- **Line:** 14
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      12: const { JitoJsonRpcClient } = require('./JitoJsonRpcClient');
      13: 
  >>> 14: async function jupSwap(address, inputMint, outputMint, amount, order, from = "api") {
      15:     console.log("jupSwap", address, inputMint, outputMint, amount, order, from);
      16:     // Handle SOL address normalization
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** scripts/solana/jupapi/jupapi.js
- **Line:** 232
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      230: 
      231: // Advanced trading functionality
  >>> 232: async function advanceJupSwap(address, inputMint, outputMint, amount, percent, from = "api") {
      233:     // Handle SOL address normalization
      234:     if (inputMint === "11111111111111111111111111111111") {
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** scripts/solana/raydium/raydiumapi.js
- **Line:** 22
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      20: const pmupfunApi = require("./pump/api");
      21: 
  >>> 22: async function raySwap(address, inputMint, outputMint,amount,order,from="api") {
      23: 
      24:     if (inputMint==="11111111111111111111111111111111")
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

## Remediation Recommendations

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### No Slippage Protection
Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### Tx Validation Missing
Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

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
