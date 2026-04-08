# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in dennisonbertram/mcp-web3-wallet-tester

## Summary

During our academic research on AI agent tool interface security, we identified **103** potential security vulnerabilities in your **dennisonbertram/mcp-web3-wallet-tester** project (103 from static analysis).

- **Risk score:** 97.6/100
- **Risk rating:** critical
- **Protocol:** mcp
- **GitHub stars:** 1

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 64 |
| High | 13 |
| Medium | 26 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Private Key Exposure | 64 |
| Missing Input Validation | 23 |
| Mev Exposure | 8 |
| Tx Validation Missing | 3 |
| State Confusion | 2 |
| Env Key Exposure | 2 |
| Missing Harness | 1 |

## Top Findings (Most Severe)

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/wallet.ts
- **Line:** 189
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      187: 
      188:   /**
  >>> 189:    * Sign a message (personal_sign)
      190:    */
      191:   async signMessage(message: string): Promise<`0x${string}`> {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/wallet.ts
- **Line:** 191
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      189:    * Sign a message (personal_sign)
      190:    */
  >>> 191:   async signMessage(message: string): Promise<`0x${string}`> {
      192:     const signature = await this.walletClient.signMessage({
      193:       account: this.account,
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/wallet.ts
- **Line:** 192
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      190:    */
      191:   async signMessage(message: string): Promise<`0x${string}`> {
  >>> 192:     const signature = await this.walletClient.signMessage({
      193:       account: this.account,
      194:       message,
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/wallet.ts
- **Line:** 200
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      198: 
      199:   /**
  >>> 200:    * Sign typed data (eth_signTypedData_v4)
      201:    */
      202:   async signTypedData(typedData: {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/wallet.ts
- **Line:** 202
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      200:    * Sign typed data (eth_signTypedData_v4)
      201:    */
  >>> 202:   async signTypedData(typedData: {
      203:     domain: Record<string, unknown>;
      204:     types: Record<string, unknown>;
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Remediation Recommendations

### Env Key Exposure
Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### Mev Exposure
Use private transaction pools (e.g., Flashbots Protect) for sensitive transactions. Implement commit-reveal schemes where applicable.

### Missing Harness
Add integration tests that exercise tool interfaces with adversarial inputs. Include negative test cases for boundary conditions.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### Private Key Exposure
Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

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
