# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in nirholas/free-crypto-news

## Summary

During our academic research on AI agent tool interface security, we identified **273** potential security vulnerabilities in your **nirholas/free-crypto-news** project (273 from static analysis).

- **Risk score:** 100.0/100
- **Risk rating:** critical
- **Protocol:** unknown
- **GitHub stars:** 135

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 32 |
| High | 16 |
| Medium | 225 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 179 |
| State Confusion | 43 |
| Private Key Exposure | 26 |
| Unlimited Approval | 8 |
| No Slippage Protection | 4 |
| Tool Poisoning | 4 |
| Hardcoded Credentials | 3 |
| Missing Harness | 3 |
| Prompt Injection | 3 |

## Top Findings (Most Severe)

### PKE-001: Private Key Exposure (CRITICAL)
- **File:** x402-facilitator/tests/unit/verifier.test.ts
- **Line:** 25
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **CWE:** CWE-200
- **Code context:**
  ```
      23: vi.mock('../../src/config/env.js', () => ({
      24:   env: {
  >>> 25:     FACILITATOR_PRIVATE_KEY: '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',
      26:     BASE_RPC_URL: 'https://base.example.com',
      27:     BASE_SEPOLIA_RPC_URL: 'https://base-sepolia.example.com',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** x402-facilitator/tests/unit/verifier.test.ts
- **Line:** 25
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      23: vi.mock('../../src/config/env.js', () => ({
      24:   env: {
  >>> 25:     FACILITATOR_PRIVATE_KEY: '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',
      26:     BASE_RPC_URL: 'https://base.example.com',
      27:     BASE_SEPOLIA_RPC_URL: 'https://base-sepolia.example.com',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** x402-facilitator/tests/unit/verifier.test.ts
- **Line:** 64
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      62: const PAYER: Address = '0x1111111111111111111111111111111111111111';
      63: const PAYEE: Address = '0x2222222222222222222222222222222222222222';
  >>> 64: const NONCE: Hex = '0x0000000000000000000000000000000000000000000000000000000000000001';
      65: const SIGNATURE: Hex = ('0x' + 'ab'.repeat(64) + '1c') as Hex;
      66: 
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-001: Private Key Exposure (CRITICAL)
- **File:** x402-facilitator/tests/unit/settler.test.ts
- **Line:** 17
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **CWE:** CWE-200
- **Code context:**
  ```
      15: vi.mock('../../src/config/env.js', () => ({
      16:   env: {
  >>> 17:     FACILITATOR_PRIVATE_KEY: '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',
      18:     BASE_RPC_URL: 'https://base.example.com',
      19:     BASE_SEPOLIA_RPC_URL: 'https://base-sepolia.example.com',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** x402-facilitator/tests/unit/settler.test.ts
- **Line:** 17
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      15: vi.mock('../../src/config/env.js', () => ({
      16:   env: {
  >>> 17:     FACILITATOR_PRIVATE_KEY: '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',
      18:     BASE_RPC_URL: 'https://base.example.com',
      19:     BASE_SEPOLIA_RPC_URL: 'https://base-sepolia.example.com',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Dynamic Testing Findings

### DYN-001: Private Key Handling (HIGH)
- **Tool:** Sponsorship
- **Details:** Private key may be exposed in error messages
- **CWE:** CWE-200
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### DYN-002: Prompt Injection Output (HIGH)
- **Tool:** get_crypto_news
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### DYN-003: Prompt Injection Output (HIGH)
- **Tool:** search_crypto_news
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### DYN-004: Prompt Injection Output (HIGH)
- **Tool:** get_bitcoin_news
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### DYN-005: Prompt Injection Output (HIGH)
- **Tool:** get_defi_news
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

## Remediation Recommendations

### Hardcoded Credentials
Remove all hardcoded API keys and secrets. Use environment variables or a secrets manager. Add credential files to .gitignore.

### Missing Harness
Add integration tests that exercise tool interfaces with adversarial inputs. Include negative test cases for boundary conditions.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### No Slippage Protection
Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### Private Key Exposure
Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### Prompt Injection
Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### Tool Poisoning
Keep tool descriptions factual. Never include instructions for the LLM in tool descriptions or metadata. Avoid directive language (e.g., 'always', 'you must', 'ignore previous').

### Tx Validation Missing
Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

### Unlimited Approval
Cap approval amounts to the exact needed value. Never use MaxUint256 (type(uint256).max) as an approval amount. Implement increaseAllowance/decreaseAllowance patterns instead.

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
