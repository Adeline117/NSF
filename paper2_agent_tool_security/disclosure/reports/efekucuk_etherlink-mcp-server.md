# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in efekucuk/etherlink-mcp-server

## Summary

During our academic research on AI agent tool interface security, we identified **63** potential security vulnerabilities in your **efekucuk/etherlink-mcp-server** project (63 from static analysis).

- **Risk score:** 85.8/100
- **Risk rating:** critical
- **Protocol:** mcp
- **GitHub stars:** 4

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 9 |
| High | 9 |
| Medium | 45 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 43 |
| Private Key Exposure | 9 |
| Unlimited Approval | 7 |
| Tx Validation Missing | 1 |
| No Gas Limit | 1 |
| Mev Exposure | 1 |
| Missing Harness | 1 |

## Top Findings (Most Severe)

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/core/tools.ts
- **Line:** 1240
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      1238:       try {
      1239:         const senderAddress = getWalletAddressFromKey();
  >>> 1240:         const signature = await services.signMessage(message);
      1241:         return {
      1242:           content: [{
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/core/tools.ts
- **Line:** 1248
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      1246:               signature,
      1247:               signer: senderAddress,
  >>> 1248:               messageType: "personal_sign"
      1249:             }, null, 2)
      1250:           }]
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/core/tools.ts
- **Line:** 1299
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      1297:         }
      1298: 
  >>> 1299:         const signature = await services.signTypedData(domain, types, primaryType, message);
      1300: 
      1301:         return {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/core/services/wallet.ts
- **Line:** 96
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      94:  * @returns The signature as a hex string
      95:  */
  >>> 96: export const signMessage = async (message: string): Promise<string> => {
      97:     const account = getConfiguredAccount();
      98: 
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/core/services/wallet.ts
- **Line:** 99
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      97:     const account = getConfiguredAccount();
      98: 
  >>> 99:     // Use the account's signMessage method directly
      100:     const signature = await account.signMessage({
      101:         message: message
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Remediation Recommendations

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
