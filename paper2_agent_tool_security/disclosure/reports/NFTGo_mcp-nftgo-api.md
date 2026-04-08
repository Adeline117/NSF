# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in NFTGo/mcp-nftgo-api

## Summary

During our academic research on AI agent tool interface security, we identified **12** potential security vulnerabilities in your **NFTGo/mcp-nftgo-api** project (12 from static analysis).

- **Risk score:** 67.6/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 4

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 9 |
| High | 1 |
| Medium | 2 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Private Key Exposure | 9 |
| Cross Tool Escalation | 1 |
| Missing Harness | 1 |
| State Confusion | 1 |

## Top Findings (Most Severe)

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** src/openapi-spec.ts
- **Line:** 2791
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      2789:             description: 'Last sale price of the NFT',
      2790:             example: {
  >>> 2791:               tx_hash: '0x8443c7fd50cecbfac1e5a330fed8e53ee8e611c6029dcbd6097ca729c9360328',
      2792:               price_token: 98.5,
      2793:               token_symbol: 'ETH',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** src/openapi-spec.ts
- **Line:** 4724
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      4722:             description: 'Maximum historical sale price of this NFT',
      4723:             example: {
  >>> 4724:               tx_hash: '0x8443c7fd50cecbfac1e5a330fed8e53ee8e611c6029dcbd6097ca729c9360328',
      4725:               price_token: 98.5,
      4726:               token_symbol: 'ETH',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** src/openapi-spec.ts
- **Line:** 4754
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      4752:             description: 'Minimum historical sale price of this NFT',
      4753:             example: {
  >>> 4754:               tx_hash: '0xf28cc8354a14b627632a81dd7d87e0edbf9c40f9d0c94ee4cda0f13f07ec1b7c',
      4755:               price_token: 0.15,
      4756:               token_symbol: 'ETH',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** src/openapi-spec.ts
- **Line:** 4784
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      4782:             description: 'Last sale price of this NFT',
      4783:             example: {
  >>> 4784:               tx_hash: '0x8443c7fd50cecbfac1e5a330fed8e53ee8e611c6029dcbd6097ca729c9360328',
      4785:               price_token: 98.5,
      4786:               token_symbol: 'ETH',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-002: Private Key Exposure (CRITICAL)
- **File:** src/openapi-spec.ts
- **Line:** 5084
- **Description:** Raw private key hex string (64 hex chars)
- **CWE:** CWE-200
- **Code context:**
  ```
      5082:             description: 'Last sale price of the NFT',
      5083:             example: {
  >>> 5084:               tx_hash: '0x8443c7fd50cecbfac1e5a330fed8e53ee8e611c6029dcbd6097ca729c9360328',
      5085:               price_token: 98.5,
      5086:               token_symbol: 'ETH',
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Remediation Recommendations

### Cross Tool Escalation
Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### Missing Harness
Add integration tests that exercise tool interfaces with adversarial inputs. Include negative test cases for boundary conditions.

### Private Key Exposure
Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

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
