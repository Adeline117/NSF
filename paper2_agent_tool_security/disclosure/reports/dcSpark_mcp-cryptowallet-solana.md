# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in dcSpark/mcp-cryptowallet-solana

## Summary

During our academic research on AI agent tool interface security, we identified **23** potential security vulnerabilities in your **dcSpark/mcp-cryptowallet-solana** project (23 from static analysis).

- **Risk score:** 73.5/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 1

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 8 |
| High | 3 |
| Medium | 12 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 11 |
| Private Key Exposure | 8 |
| Env Key Exposure | 2 |
| Missing Harness | 1 |
| Tx Validation Missing | 1 |

## Top Findings (Most Severe)

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/tools.ts
- **Line:** 6
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      4:   getTokenBalanceHandler,
      5:   createTransactionMessageHandler,
  >>> 6:   signTransactionMessageHandler,
      7:   sendTransactionHandler,
      8:   generateKeyPairHandler,
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/tools.ts
- **Line:** 68
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      66:   },
      67:   {
  >>> 68:     name: "wallet_sign_transaction",
      69:     description: "Sign a transaction with a private key (uses default wallet if no privateKey provided)",
      70:     inputSchema: {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/tools.ts
- **Line:** 162
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      160:   wallet_get_token_balance: getTokenBalanceHandler,
      161:   wallet_create_transaction: createTransactionMessageHandler,
  >>> 162:   wallet_sign_transaction: signTransactionMessageHandler,
      163:   wallet_send_transaction: sendTransactionHandler,
      164:   wallet_generate_keypair: generateKeyPairHandler,
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/tools.ts
- **Line:** 162
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      160:   wallet_get_token_balance: getTokenBalanceHandler,
      161:   wallet_create_transaction: createTransactionMessageHandler,
  >>> 162:   wallet_sign_transaction: signTransactionMessageHandler,
      163:   wallet_send_transaction: sendTransactionHandler,
      164:   wallet_generate_keypair: generateKeyPairHandler,
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/handlers/wallet.ts
- **Line:** 13
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      11:   ImportPrivateKeyInput,
      12:   SendTransactionInput,
  >>> 13:   SignTransactionInput,
      14:   ValidateAddressInput
      15: } from "./wallet.types.js";
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Remediation Recommendations

### Env Key Exposure
Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### Missing Harness
Add integration tests that exercise tool interfaces with adversarial inputs. Include negative test cases for boundary conditions.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

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
