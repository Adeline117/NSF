# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in hydra-mcp/hydra-mcp-solana

## Summary

During our academic research on AI agent tool interface security, we identified **37** potential security vulnerabilities in your **hydra-mcp/hydra-mcp-solana** project (37 from static analysis).

- **Risk score:** 79.9/100
- **Risk rating:** high
- **Protocol:** web3_native
- **GitHub stars:** 238

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 11 |
| High | 5 |
| Medium | 21 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 14 |
| Private Key Exposure | 11 |
| State Confusion | 7 |
| Tx Validation Missing | 2 |
| Mev Exposure | 2 |
| No Slippage Protection | 1 |

## Top Findings (Most Severe)

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/types/phantom.ts
- **Line:** 13
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      11:     | 'signAndSendTransactionV0'
      12:     | 'signAndSendTransactionV0WithLookupTable'
  >>> 13:     | 'signTransaction'
      14:     | 'signAllTransactions'
      15:     | 'signMessage';
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/types/phantom.ts
- **Line:** 15
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      13:     | 'signTransaction'
      14:     | 'signAllTransactions'
  >>> 15:     | 'signMessage';
      16: 
      17: interface ConnectOpts {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/types/phantom.ts
- **Line:** 31
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      29:         opts?: SendOptions
      30:     ) => Promise<{ signature: string; publicKey: PublicKey }>;
  >>> 31:     signTransaction: (transaction: Transaction | VersionedTransaction) => Promise<Transaction | VersionedTransaction>;
      32:     signAllTransactions: (
      33:         transactions: (Transaction | VersionedTransaction)[]
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/types/phantom.ts
- **Line:** 35
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      33:         transactions: (Transaction | VersionedTransaction)[]
      34:     ) => Promise<(Transaction | VersionedTransaction)[]>;
  >>> 35:     signMessage: (message: Uint8Array | string, display?: DisplayEncoding) => Promise<any>;
      36:     connect: (opts?: Partial<ConnectOpts>) => Promise<{ publicKey: PublicKey }>;
      37:     disconnect: () => Promise<void>;
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** src/components/login/PhantomLoginForm.tsx
- **Line:** 66
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      64: 
      65:     // Sign message with Phantom wallet
  >>> 66:     const signMessage = async (message: string): Promise<string> => {
      67:         if (!window.phantom?.solana) {
      68:             throw new Error('Phantom wallet not available');
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Remediation Recommendations

### Mev Exposure
Use private transaction pools (e.g., Flashbots Protect) for sensitive transactions. Implement commit-reveal schemes where applicable.

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
