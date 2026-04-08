# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in bnb-chain/bnbchain-mcp

## Summary

During our academic research on AI agent tool interface security, we identified **101** potential security vulnerabilities in your **bnb-chain/bnbchain-mcp** project (101 from static analysis).

- **Risk score:** 93.5/100
- **Risk rating:** critical
- **Protocol:** mcp
- **GitHub stars:** 58

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 54 |
| Medium | 45 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 42 |
| Cross Tool Escalation | 41 |
| Env Key Exposure | 9 |
| Private Key Exposure | 2 |
| Mev Exposure | 2 |
| State Confusion | 1 |
| Missing Harness | 1 |
| Unlimited Approval | 1 |
| Tx Validation Missing | 1 |
| No Gas Limit | 1 |

## Top Findings (Most Severe)

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** e2e/evm/transactions.test.ts
- **Line:** 10
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      8:   // A known transaction hash on BSC
      9:   const TX_HASH =
  >>> 10:     "0x5a40bbbe542e105089350635c44e18585a8a1ea861f41a11831ee504e5bc3250"
      11: 
      12:   it("get transaction details", async () => {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** e2e/evm/blocks.test.ts
- **Line:** 25
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      23:       arguments: {
      24:         blockHash:
  >>> 25:           "0x5443cc9cf2820982843ac91095827561f31f595e75932c9f87c6fe610b95243c",
      26:         network: "bsc"
      27:       }
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### CE-001: Cross Tool Escalation (HIGH)
- **File:** examples/client-with-anthropic/src/client.ts
- **Line:** 85
- **Description:** Tool invokes other tools without permission check
- **CWE:** CWE-284
- **Code context:**
  ```
      83:         const toolArgs = content.input as { [x: string]: unknown } | undefined
      84: 
  >>> 85:         const result = await this.mcp.callTool({
      86:           name: toolName,
      87:           arguments: toolArgs
  ```
- **Impact:** One tool invoking another without permission checks could allow privilege escalation or unauthorized actions.
- **Recommendation:** Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### ENV-001: Env Key Exposure (HIGH)
- **File:** examples/client-with-anthropic/src/client.ts
- **Line:** 37
- **Description:** Private key read from environment variable without protection
- **CWE:** CWE-312
- **Code context:**
  ```
      35:         args: ["-y", "@bnb-chain/mcp@latest"],
      36:         env: {
  >>> 37:           PRIVATE_KEY: process.env.PRIVATE_KEY || ""
      38:         }
      39:       })
  ```
- **Impact:** Leaking environment variable values could expose API keys and secrets in logs or error messages.
- **Recommendation:** Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### ENV-001: Env Key Exposure (HIGH)
- **File:** e2e/util.ts
- **Line:** 23
- **Description:** Private key read from environment variable without protection
- **CWE:** CWE-312
- **Code context:**
  ```
      21:         args: ["dist/index.js"],
      22:         env: {
  >>> 23:           PRIVATE_KEY: process.env.PRIVATE_KEY || "",
      24:           LOGLEVEL: "debug"
      25:         }
  ```
- **Impact:** Leaking environment variable values could expose API keys and secrets in logs or error messages.
- **Recommendation:** Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

## Remediation Recommendations

### Cross Tool Escalation
Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

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

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

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
