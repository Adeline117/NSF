# Security Disclosure: bcnmy/nexus

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 100.0)
**Protocol:** web3_native
**Repository:** [bcnmy/nexus](https://github.com/bcnmy/nexus)
**Language:** Solidity
**Stars:** 51

## Summary
We identified 169 potential security issues in bcnmy/nexus
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 63 |
| High | 61 |
| Medium | 45 |

## Critical / High Severity Findings

### Finding 1: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** scripts/foundry/Base.s.sol:8
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `MNEMONIC = "`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 2: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** contracts/mocks/Mock7739PreValidationHook.sol:10
- **Description:** Wallet signing without user confirmation
- **Matched:** `PERSONAL_SIGN`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 3: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** contracts/mocks/MockResourceLockPreValidationHook.sol:20
- **Description:** Wallet signing without user confirmation
- **Matched:** `PERSONAL_SIGN`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 4: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/foundry/utils/CheatCodes.sol:24
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 5: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/foundry/unit/concrete/modulemanager/TestModuleManager_EnableMode.t.sol:53
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 6: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** contracts/mocks/Mock7739PreValidationHook.sol:59
- **Description:** Wallet signing without user confirmation
- **Matched:** `PERSONAL_SIGN`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 7: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** contracts/modules/validators/K1Validator.sol:100
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 8: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/hardhat/smart-account/Nexus.Single.Execution.specs.ts:113
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 9: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/foundry/unit/concrete/modules/TestK1Validator.t.sol:114
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 10: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/foundry/unit/concrete/modulemanager/TestModuleManager_EnableMode.t.sol:121
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 11: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/hardhat/smart-account/Nexus.Module.K1Validator.specs.ts:130
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 12: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/foundry/unit/concrete/modules/TestK1Validator.t.sol:150
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 13: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/hardhat/smart-account/Nexus.Factory.specs.ts:154
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 14: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/foundry/unit/concrete/modules/TestK1Validator.t.sol:158
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 15: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/hardhat/smart-account/Nexus.Single.Execution.specs.ts:160
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 16: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/hardhat/smart-account/Nexus.Module.K1Validator.specs.ts:173
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 17: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/foundry/unit/concrete/modules/TestK1Validator.t.sol:174
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 18: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/foundry/unit/concrete/modulemanager/TestModuleManager_EnableMode.t.sol:178
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 19: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/hardhat/smart-account/Nexus.Batch.Execution.specs.ts:186
- **Description:** Wallet signing without user confirmation
- **Matched:** `signMessage`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 20: delegatecall_abuse (CRITICAL)
- **Pattern:** DC-001
- **CWE:** CWE-829
- **File:** contracts/Nexus.sol:192
- **Description:** delegatecall to unvalidated address
- **Matched:** `.delegatecall(`
- **Remediation:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

*... and 104 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Private Key Exposure | 58 |
| Unchecked External Call | 41 |
| Missing Input Validation | 38 |
| Approval Injection | 11 |
| Tx Validation Missing | 6 |
| No Gas Limit | 6 |
| Unlimited Approval | 3 |
| Privilege Escalation | 3 |
| Delegatecall Abuse | 2 |
| State Confusion | 1 |

## Recommendations

1. **Private Key Exposure:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).
2. **Privilege Escalation:** Gate critical state changes behind timelock and/or multi-signature requirements.
3. **Delegatecall Abuse:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

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
