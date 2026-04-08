# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in base/base-mcp

## Summary

During our academic research on AI agent tool interface security, we identified **43** potential security vulnerabilities in your **base/base-mcp** project (43 from static analysis).

- **Risk score:** 79.6/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 342

## Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 13 |
| Medium | 30 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 23 |
| Tx Validation Missing | 6 |
| No Gas Limit | 6 |
| Mev Exposure | 6 |
| Missing Harness | 1 |
| Env Key Exposure | 1 |

## Top Findings (Most Severe)

### ENV-001: Env Key Exposure (HIGH)
- **File:** src/main.ts
- **Line:** 41
- **Description:** Private key read from environment variable without protection
- **CWE:** CWE-312
- **Code context:**
  ```
      39:   const privateKey =
      40:     process.env.COINBASE_API_SECRET || process.env.COINBASE_API_PRIVATE_KEY; // Previously, was called COINBASE_API_PRIVATE_KEY
  >>> 41:   const seedPhrase = process.env.SEED_PHRASE;
      42:   const fallbackPhrase = generateMnemonic(english, 256); // Fallback in case user wants read-only operations
      43:   const chainId = process.env.CHAIN_ID ? Number(process.env.CHAIN_ID) : base.id;
  ```
- **Impact:** Leaking environment variable values could expose API keys and secrets in logs or error messages.
- **Recommendation:** Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### TV-001: Tx Validation Missing (HIGH)
- **File:** src/tools/erc20/index.ts
- **Line:** 83
- **Description:** Transaction sent without validation/whitelist check
- **CWE:** CWE-345
- **Code context:**
  ```
      81:     const atomicUnits = parseUnits(amount, decimals);
      82: 
  >>> 83:     const tx = await walletProvider.sendTransaction({
      84:       to: contractAddress,
      85:       data: encodeFunctionData({
  ```
- **Impact:** Missing transaction validation could allow crafted transactions to drain funds or interact with malicious contracts.
- **Recommendation:** Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

### MEV-001: Mev Exposure (HIGH)
- **File:** src/tools/erc20/index.ts
- **Line:** 83
- **Description:** Transaction sent to public mempool without private relay
- **CWE:** CWE-200
- **Code context:**
  ```
      81:     const atomicUnits = parseUnits(amount, decimals);
      82: 
  >>> 83:     const tx = await walletProvider.sendTransaction({
      84:       to: contractAddress,
      85:       data: encodeFunctionData({
  ```
- **Impact:** Transactions visible in the public mempool are vulnerable to front-running and MEV extraction.
- **Recommendation:** Use private transaction pools (e.g., Flashbots Protect) for sensitive transactions. Implement commit-reveal schemes where applicable.

### TV-001: Tx Validation Missing (HIGH)
- **File:** src/tools/nft/utils.ts
- **Line:** 174
- **Description:** Transaction sent without validation/whitelist check
- **CWE:** CWE-345
- **Code context:**
  ```
      172:     if (nftStandard === 'ERC721') {
      173:       // Transfer ERC721 NFT
  >>> 174:       hash = await wallet.sendTransaction({
      175:         to: contractAddress,
      176:         data: encodeFunctionData({
  ```
- **Impact:** Missing transaction validation could allow crafted transactions to drain funds or interact with malicious contracts.
- **Recommendation:** Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

### TV-001: Tx Validation Missing (HIGH)
- **File:** src/tools/nft/utils.ts
- **Line:** 184
- **Description:** Transaction sent without validation/whitelist check
- **CWE:** CWE-345
- **Code context:**
  ```
      182:     } else {
      183:       // Transfer ERC1155 NFT
  >>> 184:       hash = await wallet.sendTransaction({
      185:         to: contractAddress,
      186:         data: encodeFunctionData({
  ```
- **Impact:** Missing transaction validation could allow crafted transactions to drain funds or interact with malicious contracts.
- **Recommendation:** Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

## Dynamic Testing Findings

### DYN-001: Private Key Handling (HIGH)
- **Tool:** farcaster_username
- **Details:** Private key may be exposed in error messages
- **CWE:** CWE-200
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### DYN-002: Private Key Handling (HIGH)
- **Tool:** buy_openrouter_credits
- **Details:** Private key may be exposed in error messages
- **CWE:** CWE-200
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### DYN-003: Parameter Injection (MEDIUM)
- **Tool:** get_morpho_vaults
- **Details:** No parameter schema defined -- accepts arbitrary input
- **CWE:** CWE-20
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### DYN-004: Parameter Injection (MEDIUM)
- **Tool:** erc20_balance
- **Details:** No parameter schema defined -- accepts arbitrary input
- **CWE:** CWE-20
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### DYN-005: Parameter Injection (MEDIUM)
- **Tool:** erc20_transfer
- **Details:** No parameter schema defined -- accepts arbitrary input
- **CWE:** CWE-20
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

## Remediation Recommendations

### Env Key Exposure
Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### Mev Exposure
Use private transaction pools (e.g., Flashbots Protect) for sensitive transactions. Implement commit-reveal schemes where applicable.

### Missing Harness
Add integration tests that exercise tool interfaces with adversarial inputs. Include negative test cases for boundary conditions.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### No Gas Limit
Always specify a gas limit for transactions. Use gas estimation with a safety margin rather than omitting the limit entirely.

### Private Key Exposure
Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

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
