# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in InjectiveLabs/mcp-server

## Summary

During our academic research on AI agent tool interface security, we identified **210** potential security vulnerabilities in your **InjectiveLabs/mcp-server** project (210 from static analysis).

- **Risk score:** 100.0/100
- **Risk rating:** critical
- **Protocol:** mcp
- **GitHub stars:** 6

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 19 |
| High | 16 |
| Medium | 175 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 174 |
| Private Key Exposure | 18 |
| Mev Exposure | 15 |
| Missing Harness | 1 |
| Hardcoded Credentials | 1 |
| Tx Validation Missing | 1 |

## Top Findings (Most Severe)

### HC-001: Hardcoded Credentials (CRITICAL)
- **File:** src/bridges/debridge.ts
- **Line:** 29
- **Description:** Hardcoded API key, password, secret, or token
- **CWE:** CWE-798
- **Code context:**
  ```
      27: export const DEBRIDGE_INJECTIVE_CHAIN_ID = 100000029
      28: const DEBRIDGE_TIMEOUT_MS = 15_000
  >>> 29: const EVM_NATIVE_TOKEN = '0x0000000000000000000000000000000000000000'
      30: const DEBRIDGE_ALLOWED_HOSTS = new Set(['dln.debridge.finance'])
      31: 
  ```
- **Impact:** Hardcoded secrets in source code could be extracted by anyone with repository access, compromising associated services.
- **Recommendation:** Remove all hardcoded API keys and secrets. Use environment variables or a secrets manager. Add credential files to .gitignore.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** src/evm/eip712.test.ts
- **Line:** 12
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      10: vi.mock('../wallets/index.js', () => ({
      11:   wallets: {
  >>> 12:     unlock: vi.fn().mockReturnValue('0xdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef'),
      13:   },
      14: }))
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/evm/eip712.test.ts
- **Line:** 110
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      108:     Wallet: vi.fn().mockImplementation(() => ({
      109:       address: '0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266',
  >>> 110:       signTypedData: vi.fn().mockResolvedValue(
      111:         '0x' + 'ab'.repeat(65)
      112:       ),
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/evm/eip712.test.ts
- **Line:** 143
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      141:     })
      142: 
  >>> 143:     it('signs with ethers.Wallet.signTypedData', async () => {
      144:       const { Wallet } = await import('ethers')
      145:       const { eip712 } = await import('./eip712.js')
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/evm/eip712.test.ts
- **Line:** 156
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      154: 
      155:       const walletInstance = vi.mocked(Wallet).mock.results[0]?.value
  >>> 156:       expect(walletInstance?.signTypedData).toHaveBeenCalled()
      157:       // EIP712Domain must be stripped from types passed to signTypedData
      158:       const [_domain, types] = walletInstance?.signTypedData.mock.calls[0]
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Remediation Recommendations

### Hardcoded Credentials
Remove all hardcoded API keys and secrets. Use environment variables or a secrets manager. Add credential files to .gitignore.

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
