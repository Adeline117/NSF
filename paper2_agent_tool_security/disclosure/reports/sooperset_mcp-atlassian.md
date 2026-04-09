# Security Disclosure: sooperset/mcp-atlassian

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 100.0)
**Protocol:** mcp
**Repository:** [sooperset/mcp-atlassian](https://github.com/sooperset/mcp-atlassian)
**Language:** Python
**Stars:** 4834

## Summary
We identified 275 potential security issues in sooperset/mcp-atlassian
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 25 |
| High | 144 |
| Medium | 106 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/auth/test_authentication.py:33
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="expired-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/auth/test_authentication.py:75
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="expired-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/auth/test_authentication.py:76
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="invalid-refresh-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 4: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/confluence/test_client_oauth.py:90
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="test-byo-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 5: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/auth/test_authentication.py:102
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="almost-expired-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 6: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/confluence/test_client_oauth.py:142
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="test-byo-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 7: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/unit/servers/test_mcp_protocol.py:142
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 8: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/unit/servers/test_mcp_protocol.py:157
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 9: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/confluence/test_client_oauth.py:164
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="test-byo-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 10: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/servers/test_confluence_server.py:172
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `secret="server_client_secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 11: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/auth/test_authentication.py:208
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="test-personal-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 12: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/unit/servers/test_mcp_protocol.py:223
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 13: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/auth/test_authentication.py:237
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="test-personal-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 14: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/unit/servers/test_mcp_protocol.py:237
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 15: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/servers/test_dependencies.py:284
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token = "modified-tenant1-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 16: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/unit/servers/test_mcp_protocol.py:308
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 17: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/servers/test_jira_server.py:354
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `secret="server_client_secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 18: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/jira/test_client_oauth.py:360
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token = "env-byo-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 19: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/confluence/test_client_oauth.py:372
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="env-byo-access-token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 20: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/unit/servers/test_mcp_protocol.py:376
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

*... and 149 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Cross Tool Escalation | 140 |
| Missing Input Validation | 60 |
| State Confusion | 46 |
| Hardcoded Credentials | 14 |
| Prompt Injection | 11 |
| Excessive Permissions | 3 |
| Tool Poisoning | 1 |

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
