# Security Disclosure: CodeGraphContext/CodeGraphContext

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 96.5)
**Protocol:** mcp
**Repository:** [CodeGraphContext/CodeGraphContext](https://github.com/CodeGraphContext/CodeGraphContext)
**Language:** Python
**Stars:** 2824

## Summary
We identified 145 potential security issues in CodeGraphContext/CodeGraphContext
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 28 |
| Medium | 115 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/fixtures/sample_projects/sample_project/edge_cases/hardcoded_secrets.py:3
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `api_key = "AKIAIOSFODNN7EXAMPLE"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/fixtures/sample_projects/sample_project/edge_cases/hardcoded_secrets.py:4
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token = "ghp_16charactertoken"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/e2e/test_user_journeys.py:22
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 4: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/unit/core/test_cgcignore_patterns.py:24
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 5: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/unit/core/test_cgcignore_patterns.py:24
- **Description:** Destructive operation function without scope restriction
- **Matched:** `shell=True`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 6: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** scripts/generate_lang_contributors.py:38
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.Call`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 7: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** src/codegraphcontext/cli/config_manager.py:41
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 8: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/tools/package_resolver.py:49
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 9: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** src/codegraphcontext/cli/config_manager.py:69
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 10: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/tools/package_resolver.py:74
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 11: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/e2e/test_user_journeys.py:76
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 12: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/e2e/test_user_journeys.py:84
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 13: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/tools/languages/python.py:109
- **Description:** Destructive operation function without scope restriction
- **Matched:** `eval(`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 14: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/tools/scip_indexer.py:126
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 15: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/tools/package_resolver.py:183
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 16: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/core/database_falkordb.py:231
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.Popen`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 17: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/tools/package_resolver.py:239
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 18: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/tools/package_resolver.py:287
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 19: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/tools/package_resolver.py:297
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 20: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** src/codegraphcontext/core/cgc_bundle.py:303
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.Call`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

*... and 10 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 81 |
| State Confusion | 34 |
| Excessive Permissions | 22 |
| Unlimited Approval | 6 |
| Hardcoded Credentials | 2 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Excessive Permissions:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).
3. **Unlimited Approval:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

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
