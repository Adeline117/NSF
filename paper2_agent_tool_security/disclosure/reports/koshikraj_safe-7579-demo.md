# Security Disclosure: koshikraj/safe-7579-demo

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 99.4)
**Protocol:** web3_native
**Repository:** [koshikraj/safe-7579-demo](https://github.com/koshikraj/safe-7579-demo)
**Language:** JavaScript
**Stars:** 2

## Summary
We identified 157 potential security issues in koshikraj/safe-7579-demo
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 30 |
| High | 18 |
| Medium | 109 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** web/src/logic/networks.ts:1
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY = "05d830413c5a4ac8873c84319679c7b2"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** web/src/logic/networks.ts:2
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY = "H8IGZCCS8XCJYSXIA3GUUKW6CDECYYMNPG"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** web/src/logic/networks.ts:3
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY = "GVZS4QAMWFBGS5PK2BR76FNFPJ7X2GR44I"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 4: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** module/hardhat.config.ts:17
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `MNEMONIC = '`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 5: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** packages/4337-local-bundler/src/deploy/entrypoint.ts:17
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `'0x90d8084deab30c2a37c45e8d47f49f2f7965183cb6990a98943ef94940681de3'`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 6: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** web/src/utils/userOp.ts:35
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0x0000000000000000000000000000000000000000000000000000000000000000"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 7: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** web/src/utils/userOp.ts:36
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0x0100000000000000000000000000000000000000000000000000000000000000"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 8: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** module/src/utils/userOp.ts:47
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0x0000000000000000000000000000000000000000000000000000000000000000"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 9: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** module/src/utils/userOp.ts:48
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0x0100000000000000000000000000000000000000000000000000000000000000"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 10: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** module/contracts/interfaces/Safe.sol:58
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `enableModule(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 11: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** web/src/logic/module.ts:62
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 12: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-002
- **CWE:** CWE-798
- **File:** web/src/logic/networks.ts:62
- **Description:** Hardcoded Infura/Alchemy/QuickNode API key in RPC URL
- **Matched:** `alchemy.com/v2/K1GZzIiF6-PthdjPtfzvTOMcej2zOWWA`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 13: delegatecall_abuse (CRITICAL)
- **Pattern:** DC-001
- **CWE:** CWE-829
- **File:** module/contracts/test/SafeMock.sol:64
- **Description:** delegatecall to unvalidated address
- **Matched:** `.delegatecall(`
- **Remediation:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

### Finding 14: delegatecall_abuse (CRITICAL)
- **Pattern:** DC-001
- **CWE:** CWE-829
- **File:** module/contracts/test/SafeMock.sol:76
- **Description:** delegatecall to unvalidated address
- **Matched:** `.delegatecall(`
- **Remediation:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

### Finding 15: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** web/src/utils/execution.ts:83
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 16: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** module/src/utils/execution.ts:83
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 17: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** module/src/utils/userOp.ts:86
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTypedData`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 18: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** web/src/utils/userOp.ts:88
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTypedData`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 19: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** module/contracts/interfaces/ISafe.sol:98
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `enableModule(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 20: delegatecall_abuse (CRITICAL)
- **Pattern:** DC-001
- **CWE:** CWE-829
- **File:** module/contracts/test/TestSafeSignerLaunchpad.sol:101
- **Description:** delegatecall to unvalidated address
- **Matched:** `.delegatecall(`
- **Remediation:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

*... and 28 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 79 |
| State Confusion | 22 |
| Private Key Exposure | 17 |
| Tx Validation Missing | 10 |
| No Gas Limit | 8 |
| Delegatecall Abuse | 7 |
| Unchecked External Call | 6 |
| Hardcoded Credentials | 4 |
| Unlimited Approval | 2 |
| Privilege Escalation | 2 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Private Key Exposure:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).
3. **Privilege Escalation:** Gate critical state changes behind timelock and/or multi-signature requirements.
4. **Delegatecall Abuse:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

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
