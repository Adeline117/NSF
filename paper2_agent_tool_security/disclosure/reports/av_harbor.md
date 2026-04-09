# Security Disclosure: av/harbor

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 88.1)
**Protocol:** mcp
**Repository:** [av/harbor](https://github.com/av/harbor)
**Language:** TypeScript
**Stars:** 2768

## Summary
We identified 78 potential security issues in av/harbor
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 7 |
| High | 9 |
| Medium | 62 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** services/compose.openterminal.ts:5
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY = 'harbor-openterminal-change-me'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** services/boost/src/modules/eli5.py:71
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=eli5_prompt.format(`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 3: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** services/boost/src/modules/eli5.py:76
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=answer_prompt.format(`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 4: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** services/boost/src/custom_modules/recpl.py:117
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = prompts.init_expansion.format(`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 5: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** services/boost/src/modules/wordedit.py:319
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 6: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** services/boost/src/modules/promx.py:369
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = promx.format(`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 7: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** services/boost/src/config.py:471
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f'`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 8: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** harbor/__init__.py:11
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 9: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** services/boost/src/custom_modules/gact.py:27
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description="Conditional logic that must be met before executing command"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 10: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** .scripts/lint.ts:110
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "container_name must be ${HARBOR_CONTAINER_PREFIX}.<service>"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 11: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** .scripts/lint.ts:147
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "env_file must include ./.env"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 12: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** .scripts/lint.ts:175
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "Override env_file must point to ./services/<dir>/override.env"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 13: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** .scripts/lint.ts:210
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "Volume mounts must use ./services/ or ${HARBOR_*} for local paths"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 14: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** .scripts/lint.ts:263
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "Build context must be ./services/<dir> or a ${HARBOR_*} variable"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 15: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** .agents/skills/skill-creator/eval-viewer/generate_review.py:291
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 16: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** .scripts/lint.ts:394
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "services/<handle>/ directory and override.env must exist"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 57 |
| Tool Poisoning | 7 |
| Prompt Injection | 6 |
| State Confusion | 5 |
| Excessive Permissions | 2 |
| Hardcoded Credentials | 1 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Prompt Injection:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.
3. **Excessive Permissions:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).
4. **Tool Poisoning:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

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
