# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in wallet-agent/wallet-agent

## Summary

During our academic research on AI agent tool interface security, we identified **896** potential security vulnerabilities in your **wallet-agent/wallet-agent** project (896 from static analysis).

- **Risk score:** 100.0/100
- **Risk rating:** critical
- **Protocol:** mcp
- **GitHub stars:** 5

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 202 |
| High | 535 |
| Medium | 159 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Cross Tool Escalation | 463 |
| Private Key Exposure | 197 |
| Missing Input Validation | 79 |
| State Confusion | 56 |
| Tx Validation Missing | 29 |
| Mev Exposure | 29 |
| No Gas Limit | 23 |
| Unlimited Approval | 13 |
| Hardcoded Credentials | 5 |
| Missing Harness | 1 |
| Tool Poisoning | 1 |

## Top Findings (Most Severe)

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** test/schemas.test.ts
- **Line:** 52
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      50:     it("validates private keys", () => {
      51:       const result = PrivateKeySchema.safeParse(
  >>> 52:         "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
      53:       )
      54:       expect(result.success).toBe(true)
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/schemas.test.ts
- **Line:** 11
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      9:   SendTransactionArgsSchema,
      10:   SetWalletTypeArgsSchema,
  >>> 11:   SignMessageArgsSchema,
      12:   SwitchChainArgsSchema,
      13: } from "../src/schemas"
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/schemas.test.ts
- **Line:** 82
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      80:   })
      81: 
  >>> 82:   describe("SignMessageArgsSchema", () => {
      83:     it("validates sign message args", () => {
      84:       expect(SignMessageArgsSchema.safeParse({ message: "Hello world" }).success).toBe(true)
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/schemas.test.ts
- **Line:** 84
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      82:   describe("SignMessageArgsSchema", () => {
      83:     it("validates sign message args", () => {
  >>> 84:       expect(SignMessageArgsSchema.safeParse({ message: "Hello world" }).success).toBe(true)
      85:     })
      86: 
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/schemas.test.ts
- **Line:** 88
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      86: 
      87:     it("rejects empty messages", () => {
  >>> 88:       const result = SignMessageArgsSchema.safeParse({ message: "" })
      89:       expect(result.success).toBe(false)
      90:       if (!result.success) {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Remediation Recommendations

### Cross Tool Escalation
Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### Hardcoded Credentials
Remove all hardcoded API keys and secrets. Use environment variables or a secrets manager. Add credential files to .gitignore.

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
