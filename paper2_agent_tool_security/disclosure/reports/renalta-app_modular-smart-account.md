# Security Disclosure: renalta-app/modular-smart-account

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 93.3)
**Protocol:** web3_native
**Repository:** [renalta-app/modular-smart-account](https://github.com/renalta-app/modular-smart-account)
**Language:** Solidity
**Stars:** 0

## Summary
We identified 85 potential security issues in renalta-app/modular-smart-account
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 36 |
| High | 15 |
| Medium | 34 |

## Critical / High Severity Findings

### Finding 1: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/helpers/EIP7739Helpers.sol:15
- **Description:** Wallet signing without user confirmation
- **Matched:** `PERSONAL_SIGN`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 2: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/helpers/EIP7739Helpers.sol:31
- **Description:** Wallet signing without user confirmation
- **Matched:** `PERSONAL_SIGN`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 3: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/helpers/EIP7739Helpers.sol:32
- **Description:** Wallet signing without user confirmation
- **Matched:** `PERSONAL_SIGN`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 4: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/unit/OwnerTransfer.t.sol:39
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `TransferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 5: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/unit/OwnerTransfer.t.sol:46
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 6: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** contracts/libraries/ERC7579ModuleLib.sol:55
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `addModule(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 7: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/unit/OwnerTransfer.t.sol:56
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 8: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** contracts/accounts/ModuleStorage.sol:58
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `addModule(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 9: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/unit/OwnerTransfer.t.sol:61
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `TransferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 10: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/unit/OwnerTransfer.t.sol:68
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 11: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/unit/OwnerTransfer.t.sol:80
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 12: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/unit/OwnerTransfer.t.sol:94
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 13: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/helpers/ForkHelpers.sol:95
- **Description:** Wallet signing without user confirmation
- **Matched:** `eth_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 14: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/helpers/SmartSessionTestBase.sol:96
- **Description:** Wallet signing without user confirmation
- **Matched:** `eth_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 15: delegatecall_abuse (CRITICAL)
- **Pattern:** DC-001
- **CWE:** CWE-829
- **File:** test/unit/ExecutionLib.t.sol:101
- **Description:** delegatecall to unvalidated address
- **Matched:** `.delegateCall(`
- **Remediation:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

### Finding 16: privilege_escalation (CRITICAL)
- **Pattern:** PE-001
- **CWE:** CWE-269
- **File:** test/unit/OwnerTransfer.t.sol:107
- **Description:** Module changes owner/modules/handler without timelock
- **Matched:** `transferOwnership(`
- **Remediation:** Gate critical state changes behind timelock and/or multi-signature requirements.

### Finding 17: delegatecall_abuse (CRITICAL)
- **Pattern:** DC-001
- **CWE:** CWE-829
- **File:** test/unit/ExecutionLib.t.sol:108
- **Description:** delegatecall to unvalidated address
- **Matched:** `.delegateCall(`
- **Remediation:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

### Finding 18: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/helpers/ForkHelpers.sol:115
- **Description:** Wallet signing without user confirmation
- **Matched:** `eth_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 19: delegatecall_abuse (CRITICAL)
- **Pattern:** DC-001
- **CWE:** CWE-829
- **File:** test/unit/ExecutionLib.t.sol:116
- **Description:** delegatecall to unvalidated address
- **Matched:** `.delegateCall(`
- **Remediation:** Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.

### Finding 20: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** test/helpers/modules/ECDSASessionKeyValidator.sol:116
- **Description:** Wallet signing without user confirmation
- **Matched:** `eth_sign`
- **Remediation:** Review and apply appropriate security controls for this finding category.

*... and 31 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| State Confusion | 34 |
| Privilege Escalation | 22 |
| Unlimited Approval | 9 |
| Private Key Exposure | 8 |
| Delegatecall Abuse | 6 |
| Unchecked External Call | 6 |

## Recommendations

1. **Private Key Exposure:** Review and apply appropriate security controls for this finding category.
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
