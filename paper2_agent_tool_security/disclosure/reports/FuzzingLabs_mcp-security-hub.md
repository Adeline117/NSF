# Security Disclosure: FuzzingLabs/mcp-security-hub

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 95.7)
**Protocol:** mcp
**Repository:** [FuzzingLabs/mcp-security-hub](https://github.com/FuzzingLabs/mcp-security-hub)
**Language:** Python
**Stars:** 510

## Summary
We identified 98 potential security issues in FuzzingLabs/mcp-security-hub
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 12 |
| High | 84 |
| Medium | 2 |

## Critical / High Severity Findings

### Finding 1: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** web-security/waybackurls-mcp/server.py:452
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 2: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** blockchain/daml-viewer-mcp/server.py:482
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 3: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** binary-analysis/yara-mcp/server.py:545
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 4: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** secrets/gitleaks-mcp/server.py:600
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 5: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** web-security/nuclei-mcp/server.py:619
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 6: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** cloud-security/prowler-mcp/server.py:631
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 7: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** web-security/sqlmap-mcp/server.py:638
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 8: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** binary-analysis/binwalk-mcp/server.py:644
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 9: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** reconnaissance/nmap-mcp/server.py:670
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 10: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** cloud-security/trivy-mcp/server.py:703
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 11: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** web-security/ffuf-mcp/server.py:743
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 12: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** blockchain/solazy-mcp/server.py:750
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 13: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/conftest.py:31
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 14: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/conftest.py:37
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.Call`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 15: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/conftest.py:45
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 16: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/conftest.py:51
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.Call`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 17: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** web-security/ffuf-mcp/server.py:52
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 18: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/conftest.py:59
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 19: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** fuzzing/dharma-mcp/server.py:61
- **Description:** Destructive operation function without scope restriction
- **Matched:** `exec(`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 20: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/conftest.py:65
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.Call`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

*... and 76 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Cross Tool Escalation | 46 |
| Excessive Permissions | 37 |
| Prompt Injection | 12 |
| Missing Input Validation | 2 |
| Unlimited Approval | 1 |

## Recommendations

1. **Prompt Injection:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.
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
