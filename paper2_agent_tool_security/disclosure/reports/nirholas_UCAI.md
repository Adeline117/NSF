# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in nirholas/UCAI

## Summary

During our academic research on AI agent tool interface security, we identified **153** potential security vulnerabilities in your **nirholas/UCAI** project (153 from static analysis).

- **Risk score:** 100.0/100
- **Risk rating:** critical
- **Protocol:** openai
- **GitHub stars:** 28

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 49 |
| High | 8 |
| Medium | 96 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 95 |
| Private Key Exposure | 42 |
| Excessive Permissions | 8 |
| Prompt Injection | 5 |
| Hardcoded Credentials | 2 |
| State Confusion | 1 |

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

## Remediation Recommendations

### Excessive Permissions
Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### Hardcoded Credentials
Remove all hardcoded API keys and secrets. Use environment variables or a secrets manager. Add credential files to .gitignore.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### Private Key Exposure
Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### Prompt Injection
Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

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
