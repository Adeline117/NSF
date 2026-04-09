# Security Disclosure: ridafkih/keeper.sh

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 96.5)
**Protocol:** unknown
**Repository:** [ridafkih/keeper.sh](https://github.com/ridafkih/keeper.sh)
**Language:** TypeScript
**Stars:** 970

## Summary
We identified 147 potential security issues in ridafkih/keeper.sh
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 6 |
| High | 16 |
| Medium | 125 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** packages/auth/tests/capabilities.test.ts:9
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: "google-client-secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** applications/web/src/server/migration-check.ts:10
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `TOKEN: "VITE_VISITORS_NOW_TOKEN"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** packages/auth/tests/capabilities.test.ts:11
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: "microsoft-client-secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 4: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** packages/calendar/tests/core/oauth/google.test.ts:27
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: "google-client-secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 5: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** packages/auth/tests/capabilities.test.ts:34
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: "google-client-secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 6: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** packages/auth/tests/capabilities.test.ts:36
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: "microsoft-client-secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 7: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** applications/web/tests/hooks/use-entitlements.test.ts:9
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 8: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** services/api/src/utils/source-destination-mappings.ts:10
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 9: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** services/api/src/utils/caldav.ts:16
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 10: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** services/api/src/utils/caldav-sources.ts:23
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 11: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** services/api/src/utils/oauth-sources.ts:24
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 12: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** applications/web/src/config/plans.ts:32
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 13: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** applications/web/src/config/plans.ts:37
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `Unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 14: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** applications/web/src/config/plans.ts:38
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `Unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 15: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** services/api/src/utils/source-lifecycle.ts:39
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 16: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** applications/web/src/config/plans.ts:40
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `Unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 17: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** applications/web/src/routes/(dashboard)/dashboard/settings/api-tokens.tsx:53
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `Unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 18: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** services/api/src/utils/oauth.ts:62
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 19: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** services/api/src/routes/api/destinations/authorize.ts:63
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 20: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** applications/web/src/routes/(dashboard)/dashboard/upgrade/index.tsx:140
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

*... and 2 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 122 |
| Unlimited Approval | 16 |
| Hardcoded Credentials | 6 |
| State Confusion | 2 |
| Missing Harness | 1 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Unlimited Approval:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

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
