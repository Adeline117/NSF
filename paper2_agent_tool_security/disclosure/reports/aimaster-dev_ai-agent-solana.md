# Security Disclosure: aimaster-dev/ai-agent-solana

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 100.0)
**Protocol:** langchain
**Repository:** [aimaster-dev/ai-agent-solana](https://github.com/aimaster-dev/ai-agent-solana)
**Language:** TypeScript
**Stars:** 11

## Summary
We identified 533 potential security issues in aimaster-dev/ai-agent-solana
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 17 |
| High | 55 |
| Medium | 461 |

## Critical / High Severity Findings

### Finding 1: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/services/blockchain/types.ts:5
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 2: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** src/config/__mocks__/settings.ts:6
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `PRIVATE_KEY: '`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 3: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** src/config/__mocks__/settings.js:6
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `PRIVATE_KEY: '`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 4: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** scripts/convert-key-to-array.js:9
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `private key:'`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 5: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** scripts/validate-key.js:12
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `private key:'`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 6: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** src/services/blockchain/pump.js:13
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `Private Key:"`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 7: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** src/services/blockchain/pump.ts:19
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `Private Key:"`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 8: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/providers/wallet.js:23
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 9: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/providers/wallet.js:34
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 10: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/providers/wallet.js:34
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 11: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/providers/wallet.ts:57
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 12: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/providers/wallet.ts:69
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 13: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/providers/wallet.ts:69
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 14: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions2/swapDao.js:90
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "So11111111111111111111111111111111111111112"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 15: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions2/swapDao.js:91
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 16: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions2/swapDao.ts:145
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "So11111111111111111111111111111111111111112"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 17: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions2/swapDao.ts:146
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 18: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/utils/spl-token/index.js:10
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 19: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** frontend/src/app/api/wallet/sign.ts:16
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `SendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 20: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/utils/spl-token/index.js:17
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

*... and 52 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 451 |
| Tx Validation Missing | 52 |
| Private Key Exposure | 13 |
| State Confusion | 8 |
| Hardcoded Credentials | 4 |
| Tool Poisoning | 2 |
| No Gas Limit | 2 |
| Cross Tool Escalation | 1 |

## Recommendations

1. **Private Key Exposure:** Review and apply appropriate security controls for this finding category.
2. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
3. **Tx Validation Missing:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

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
