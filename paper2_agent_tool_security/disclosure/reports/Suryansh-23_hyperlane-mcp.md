# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in Suryansh-23/hyperlane-mcp

## Summary

During our academic research on AI agent tool interface security, we identified **20** potential security vulnerabilities in your **Suryansh-23/hyperlane-mcp** project (20 from static analysis).

- **Risk score:** 69.8/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 6

## Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 10 |
| Medium | 10 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 9 |
| Env Key Exposure | 8 |
| Tx Validation Missing | 1 |
| Mev Exposure | 1 |
| Missing Harness | 1 |

## Top Findings (Most Severe)

### ENV-001: Env Key Exposure (HIGH)
- **File:** src/hyperlaneDeployer.ts
- **Line:** 185
- **Description:** Private key read from environment variable without protection
- **CWE:** CWE-312
- **Code context:**
  ```
      183:   const requiredHook = await createMerkleTreeConfig();
      184: 
  >>> 185:   if (!process.env.PRIVATE_KEY) {
      186:     throw new Error('PRIVATE_KEY environment variable is required');
      187:   }
  ```
- **Impact:** Leaking environment variable values could expose API keys and secrets in logs or error messages.
- **Recommendation:** Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### ENV-001: Env Key Exposure (HIGH)
- **File:** src/hyperlaneDeployer.ts
- **Line:** 188
- **Description:** Private key read from environment variable without protection
- **CWE:** CWE-312
- **Code context:**
  ```
      186:     throw new Error('PRIVATE_KEY environment variable is required');
      187:   }
  >>> 188:   const owner = await privateKeyToSigner(process.env.PRIVATE_KEY);
      189: 
      190:   const proxyAdmin: OwnableConfig = {
  ```
- **Impact:** Leaking environment variable values could expose API keys and secrets in logs or error messages.
- **Recommendation:** Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### ENV-001: Env Key Exposure (HIGH)
- **File:** src/hyperlaneDeployer.ts
- **Line:** 215
- **Description:** Private key read from environment variable without protection
- **CWE:** CWE-312
- **Code context:**
  ```
      213:   config: CoreDeployConfig
      214: ): Promise<Record<string, string>> {
  >>> 215:   if (!process.env.PRIVATE_KEY) {
      216:     throw new Error('PRIVATE_KEY environment variable is required');
      217:   }
  ```
- **Impact:** Leaking environment variable values could expose API keys and secrets in logs or error messages.
- **Recommendation:** Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### ENV-001: Env Key Exposure (HIGH)
- **File:** src/hyperlaneDeployer.ts
- **Line:** 218
- **Description:** Private key read from environment variable without protection
- **CWE:** CWE-312
- **Code context:**
  ```
      216:     throw new Error('PRIVATE_KEY environment variable is required');
      217:   }
  >>> 218:   const signer = privateKeyToSigner(process.env.PRIVATE_KEY);
      219:   const chain = config.config.chainName;
      220: 
  ```
- **Impact:** Leaking environment variable values could expose API keys and secrets in logs or error messages.
- **Recommendation:** Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### TV-001: Tx Validation Missing (HIGH)
- **File:** src/assetTransfer.ts
- **Line:** 130
- **Description:** Transaction sent without validation/whitelist check
- **CWE:** CWE-345
- **Code context:**
  ```
      128:   for (const tx of transferTxs) {
      129:     if (tx.type === ProviderType.EthersV5) {
  >>> 130:       const txResponse = await connectedSigner.sendTransaction(tx.transaction);
      131:       const txReceipt = await multiProvider.handleTx(origin, txResponse);
      132:       txReceipts.push(txReceipt);
  ```
- **Impact:** Missing transaction validation could allow crafted transactions to drain funds or interact with malicious contracts.
- **Recommendation:** Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

## Remediation Recommendations

### Env Key Exposure
Do not log or display environment variable values. Mask sensitive environment variables in error messages and debug output.

### Mev Exposure
Use private transaction pools (e.g., Flashbots Protect) for sensitive transactions. Implement commit-reveal schemes where applicable.

### Missing Harness
Add integration tests that exercise tool interfaces with adversarial inputs. Include negative test cases for boundary conditions.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

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
