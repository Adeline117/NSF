# Security Disclosure: tommyz7/ethwaterloo-contract-account-metamask-plugin

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 99.7)
**Protocol:** web3_native
**Repository:** [tommyz7/ethwaterloo-contract-account-metamask-plugin](https://github.com/tommyz7/ethwaterloo-contract-account-metamask-plugin)
**Language:** JavaScript
**Stars:** 1

## Summary
We identified 134 potential security issues in tommyz7/ethwaterloo-contract-account-metamask-plugin
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 65 |
| High | 4 |
| Medium | 65 |

## Critical / High Severity Findings

### Finding 1: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** metamask-plugin/examples/defi-custody/index.js:6
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `PrivateKey = "`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 2: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** metamask-plugin/examples/defi-custody/index.js:6
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0xb0057716d5917badaf911b193b12b910811c1497b5bada8d7711f758981c3773"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 3: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** metamask/ui/app/components/app/transaction-activity-log/tests/transaction-activity-log.component.test.js:11
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `'0xe46c7f9b39af2fbf1c53e66f72f80343ab54c2c6dba902d51fb98ada08fe1a63'`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 4: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** metamask-plugin/examples/bls-signer/index.js:12
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 5: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** metamask/ui/app/components/app/transaction-activity-log/tests/transaction-activity-log.util.test.js:13
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `'0xa14f13d36b3901e352ce3a7acb9b47b001e5a3370f06232a0953c6fc6fad91b3'`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 6: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** metamask/ui/app/components/app/transaction-activity-log/tests/transaction-activity-log.component.test.js:17
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `'0xe46c7f9b39af2fbf1c53e66f72f80343ab54c2c6dba902d51fb98ada08fe1a63'`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 7: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** metamask/ui/app/pages/first-time-flow/seed-phrase/tests/confirm-seed-phrase-component.test.js:22
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `seedPhrase: '`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 8: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** metamask/ui/app/components/app/transaction-activity-log/tests/transaction-activity-log.component.test.js:23
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `'0x7d09d337fc6f5d6fe2dbf3a6988d69532deb0a82b665f9180b5a20db377eea87'`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 9: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** metamask/ui/app/pages/first-time-flow/seed-phrase/confirm-seed-phrase/confirm-seed-phrase.component.js:23
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `seedPhrase: '`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 10: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** metamask/ui/app/pages/first-time-flow/seed-phrase/seed-phrase.component.js:25
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `SeedPhrase: '`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 11: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** metamask/ui/app/pages/first-time-flow/create-password/import-with-seed-phrase/import-with-seed-phrase.component.js:25
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `seedPhrase: '`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 12: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** metamask-plugin/examples/custom-account/index.js:26
- **Description:** Wallet signing without user confirmation
- **Matched:** `eth_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 13: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** metamask-plugin/examples/defi-custody-sandbox/index.js:27
- **Description:** Wallet signing without user confirmation
- **Matched:** `eth_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 14: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** metamask-plugin/examples/custom-account/index.js:27
- **Description:** Wallet signing without user confirmation
- **Matched:** `eth_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 15: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** metamask/ui/app/pages/keychains/restore-vault.js:27
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `seedPhrase: '`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 16: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** metamask-plugin/examples/defi-custody-sandbox/index.js:28
- **Description:** Wallet signing without user confirmation
- **Matched:** `eth_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 17: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** metamask-plugin/examples/custom-account/index.js:28
- **Description:** Wallet signing without user confirmation
- **Matched:** `personal_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 18: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** metamask-plugin/examples/defi-custody-sandbox/index.js:29
- **Description:** Wallet signing without user confirmation
- **Matched:** `personal_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 19: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** metamask-plugin/examples/custom-account/index.js:29
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTypedData`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 20: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** metamask/ui/app/components/app/transaction-activity-log/tests/transaction-activity-log.component.test.js:29
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `'0x7d09d337fc6f5d6fe2dbf3a6988d69532deb0a82b665f9180b5a20db377eea87'`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

*... and 49 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Private Key Exposure | 65 |
| Missing Input Validation | 53 |
| Insecure Rpc | 8 |
| State Confusion | 4 |
| Unchecked External Call | 3 |
| Tx Validation Missing | 1 |

## Recommendations

1. **Private Key Exposure:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

## Disclosure Timeline
- 2026-04-08: Report prepared
- 2026-04-08: Report sent to maintainer (pending)
- 2026-07-07: 90-day disclosure deadline

## About This Research

This work is part of an NSF-funded academic research project studying the security of
AI agent tool interfaces across protocol families (MCP, OpenAI Function Calling,
LangChain, Web3-native modules). The study analyzed 138 repositories using
static pattern analysis based on a 27-pattern vulnerability taxonomy.

## Contact

For questions about this report, please contact the NSF AI Agent Security Research Team.
We are happy to work with you on remediation and will adjust the disclosure
timeline if needed to allow adequate time for fixes.
