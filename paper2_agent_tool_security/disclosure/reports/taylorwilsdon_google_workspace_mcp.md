# Security Disclosure: taylorwilsdon/google_workspace_mcp

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 90.5)
**Protocol:** mcp
**Repository:** [taylorwilsdon/google_workspace_mcp](https://github.com/taylorwilsdon/google_workspace_mcp)
**Language:** Python
**Stars:** 2054

## Summary
We identified 82 potential security issues in taylorwilsdon/google_workspace_mcp
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 20 |
| High | 8 |
| Medium | 54 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/auth/test_google_auth_callback_refresh_token.py:94
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="session-refresh-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:314
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 3: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:330
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 4: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:381
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 5: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:452
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 6: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:576
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 7: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:588
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 8: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:635
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 9: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:644
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 10: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:688
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 11: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:696
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 12: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:718
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 13: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:724
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 14: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:728
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 15: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:738
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 16: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:758
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 17: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/managers/batch_operation_manager.py:766
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 18: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/docs_tools.py:782
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 19: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/docs_tools.py:798
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 20: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** gdocs/docs_tools.py:885
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

*... and 8 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| State Confusion | 37 |
| Prompt Injection | 19 |
| Missing Input Validation | 17 |
| Cross Tool Escalation | 8 |
| Hardcoded Credentials | 1 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Prompt Injection:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

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
