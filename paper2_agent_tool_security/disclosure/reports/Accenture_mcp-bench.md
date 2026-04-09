# Security Disclosure: Accenture/mcp-bench

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 100.0)
**Protocol:** mcp
**Repository:** [Accenture/mcp-bench](https://github.com/Accenture/mcp-bench)
**Language:** Python
**Stars:** 465

## Summary
We identified 210 potential security issues in Accenture/mcp-bench
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 19 |
| High | 89 |
| Medium | 102 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** mcp_servers/weather_mcp/app.py:3
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY = "366fd563131a4af1bd962603252105"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** mcp_servers/weather_mcp/server.py:6
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY = "366fd563131a4af1bd962603252105"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** mcp_servers/movie-recommender-mcp/movie-reccomender-mcp/movie_recommender.py:7
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY = "c6fae702c36224d5f01778d394772520"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 4: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** mcp_servers/weather_mcp/simple_weather_server.py:12
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY = "366fd563131a4af1bd962603252105"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 5: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** synthesis/task_synthesis.py:45
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 6: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** mcp_servers/medcalc/MedCalcBench/evaluation/generate_code_prompt.py:53
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `system = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 7: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** agent/executor.py:207
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 8: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** agent/executor.py:281
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 9: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** synthesis/task_synthesis.py:319
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 10: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** benchmark/evaluator.py:486
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 11: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** synthesis/task_synthesis.py:501
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 12: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** agent/executor.py:661
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 13: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** benchmark/evaluator.py:671
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 14: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** agent/executor.py:705
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 15: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** agent/executor.py:781
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 16: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** agent/executor.py:783
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 17: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** mcp_servers/huggingface-mcp-server/src/huggingface/server.py:821
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 18: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** mcp_servers/huggingface-mcp-server/src/huggingface/server.py:850
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 19: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** agent/executor.py:879
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 20: excessive_permissions (HIGH)
- **Pattern:** EP-001
- **CWE:** CWE-78
- **File:** mcp_servers/mcp-osint-server/mcp_osint_server/mcp_osint_server/main.py:12
- **Description:** Destructive operation function without scope restriction
- **Matched:** `subprocess.run`
- **Remediation:** Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

*... and 88 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Excessive Permissions | 55 |
| Missing Input Validation | 53 |
| State Confusion | 42 |
| Cross Tool Escalation | 20 |
| Prompt Injection | 15 |
| Tool Poisoning | 13 |
| Missing Harness | 7 |
| Hardcoded Credentials | 4 |
| Unlimited Approval | 1 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Prompt Injection:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.
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
