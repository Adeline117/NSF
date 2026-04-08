# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in agenticvault/agentic-vault

## Summary

During our academic research on AI agent tool interface security, we identified **559** potential security vulnerabilities in your **agenticvault/agentic-vault** project (559 from static analysis).

- **Risk score:** 100.0/100
- **Risk rating:** critical
- **Protocol:** mcp
- **GitHub stars:** 4

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 339 |
| High | 58 |
| Medium | 162 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Private Key Exposure | 302 |
| Missing Input Validation | 119 |
| Missing Harness | 41 |
| Hardcoded Credentials | 37 |
| Cross Tool Escalation | 25 |
| Mev Exposure | 13 |
| No Slippage Protection | 11 |
| Tx Validation Missing | 9 |
| State Confusion | 2 |

## Top Findings (Most Severe)

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/unit/kms-signer.test.ts
- **Line:** 119
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      117: 
      118: // ============================================================================
  >>> 119: // signTransaction
      120: // ============================================================================
      121: 
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/unit/kms-signer.test.ts
- **Line:** 122
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      120: // ============================================================================
      121: 
  >>> 122: describe('KmsSignerAdapter.signTransaction', () => {
      123:   it('should sign and return a serialized signed transaction', async () => {
      124:     const mockClient = createMockKmsClient();
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/unit/kms-signer.test.ts
- **Line:** 138
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      136:     };
      137: 
  >>> 138:     const signedTx = await signer.signTransaction(tx);
      139:     expect(signedTx).toBeDefined();
      140:     expect(typeof signedTx).toBe('string');
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/unit/kms-signer.test.ts
- **Line:** 160
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      158:     };
      159: 
  >>> 160:     await signer.signTransaction(tx);
      161:     expect(mockClient.signDigest).toHaveBeenCalledWith(
      162:       TEST_KEY_ID,
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** test/unit/kms-signer.test.ts
- **Line:** 185
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      183:     };
      184: 
  >>> 185:     await signer.signTransaction(tx);
      186: 
      187:     expect(parseDerSignature).toHaveBeenCalledWith(MOCK_DER_SIGNATURE);
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

### No Slippage Protection
Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

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
