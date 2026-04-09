# Security Disclosure: waybarrios/vllm-mlx

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 99.2)
**Protocol:** mcp
**Repository:** [waybarrios/vllm-mlx](https://github.com/waybarrios/vllm-mlx)
**Language:** Python
**Stars:** 774

## Summary
We identified 140 potential security issues in waybarrios/vllm-mlx
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 15 |
| High | 78 |
| Medium | 47 |

## Critical / High Severity Findings

### Finding 1: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_continuous_batching.py:35
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 2: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_simple_engine.py:80
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 3: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** examples/mcp_chat.py:91
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 4: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/evals/gsm8k/gsm8k_eval.py:130
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 5: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_mllm_continuous_batching.py:141
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 6: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_mllm_continuous_batching.py:163
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 7: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_simple_engine.py:205
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 8: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/evals/gsm8k/gsm8k_eval.py:227
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 9: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_batching.py:326
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 10: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** vllm_mlx/api/tool_calling.py:537
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 11: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_mllm_continuous_batching.py:546
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 12: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_mllm_cache.py:746
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 13: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_mllm_cache.py:796
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 14: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_mllm_cache.py:923
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 15: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/test_mllm_cache.py:972
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 16: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** vllm_mlx/mcp/__init__.py:19
- **Description:** Tool invokes other tools without permission check
- **Matched:** `execute_tool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 17: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** vllm_mlx/vllm_platform.py:27
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 18: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/test_kv_cache_quantization.py:28
- **Description:** Destructive operation function without scope restriction
- **Matched:** `eval(`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 19: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** tests/test_paged_cache_real_model.py:29
- **Description:** Destructive operation function without scope restriction
- **Matched:** `eval(`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### Finding 20: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** examples/mcp_chat.py:38
- **Description:** Tool invokes other tools without permission check
- **Matched:** `execute_tool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

*... and 73 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Excessive Permissions | 65 |
| State Confusion | 39 |
| Prompt Injection | 15 |
| Cross Tool Escalation | 10 |
| Missing Input Validation | 6 |
| Unlimited Approval | 3 |
| No Output Validation | 2 |

## Recommendations

1. **Prompt Injection:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.
2. **Cross Tool Escalation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.
3. **Excessive Permissions:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

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
