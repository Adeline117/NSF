# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in dcSpark/mcp-cryptowallet-evm

## Summary

During our academic research on AI agent tool interface security, we identified **67** potential security vulnerabilities in your **dcSpark/mcp-cryptowallet-evm** project (67 from static analysis).

- **Risk score:** 91.1/100
- **Risk rating:** critical
- **Protocol:** mcp
- **GitHub stars:** 9

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 41 |
| High | 4 |
| Medium | 22 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Private Key Exposure | 41 |
| Missing Input Validation | 21 |
| Env Key Exposure | 2 |
| Missing Harness | 1 |
| Tx Validation Missing | 1 |
| Mev Exposure | 1 |

## Top Findings (Most Severe)

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** tests/handlers/provider.test.ts
- **Line:** 38
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      36:     }),
      37:     getCode: jest.fn().mockResolvedValue('0x'),
  >>> 38:     getStorageAt: jest.fn().mockResolvedValue('0x0000000000000000000000000000000000000000000000000000000000000000'),
      39:     estimateGas: jest.fn().mockResolvedValue(originalModule.BigNumber.from(21000)),
      40:     getLogs: jest.fn().mockResolvedValue([{
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** tests/handlers/provider.test.ts
- **Line:** 129
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      127:     expect(result.isError).toBe(false);
      128:     expect(result.toolResult).toHaveProperty('storage');
  >>> 129:     expect(result.toolResult.storage).toBe('0x0000000000000000000000000000000000000000000000000000000000000000');
      130:   });
      131: 
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-001: Private Key Exposure (CRITICAL)
- **File:** tests/handlers/wallet.test.ts
- **Line:** 28
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **CWE:** CWE-200
- **Code context:**
  ```
      26:   const mockWallet = {
      27:     address: '0x742d35Cc6634C0532925a3b844Bc454e4438f44e',
  >>> 28:     privateKey: '0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef',
      29:     publicKey: '0x04a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6',
      30:     mnemonic: {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-001: Private Key Exposure (CRITICAL)
- **File:** tests/handlers/wallet.test.ts
- **Line:** 117
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **CWE:** CWE-200
- **Code context:**
  ```
      115:   test('fromPrivateKeyHandler should create a wallet from a private key', async () => {
      116:     const result = await fromPrivateKeyHandler({
  >>> 117:       privateKey: '0x0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef'
      118:     });
      119: 
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-001: Private Key Exposure (CRITICAL)
- **File:** tests/handlers/wallet.test.ts
- **Line:** 128
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **CWE:** CWE-200
- **Code context:**
  ```
      126:   test('fromMnemonicHandler should create a wallet from a mnemonic', async () => {
      127:     const result = await fromMnemonicHandler({
  >>> 128:       mnemonic: 'test test test test test test test test test test test junk'
      129:     });
      130: 
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
