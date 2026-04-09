# Security Disclosure: assagman/dsgo

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 88.8)
**Protocol:** openai
**Repository:** [assagman/dsgo](https://github.com/assagman/dsgo)
**Language:** Go
**Stars:** 2

## Summary
We identified 80 potential security issues in assagman/dsgo
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 4 |
| High | 20 |
| Medium | 56 |

## Critical / High Severity Findings

### Finding 1: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** examples/yaml_program/main.go:47
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `Description: %s`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 2: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** examples/chat_loop_mcp/main.go:281
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt=%d, completion=%d) cost=$%.6f latency=%s`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 3: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** internal/module/react.go:550
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = fmt.Sprintf("The %s`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 4: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** internal/module/react.go:561
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = fmt.Sprintf("%s (one of: %s`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 5: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** internal/core/history.go:9
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 6: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** internal/mcp/client_test.go:83
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 7: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** internal/mcp/tools.go:101
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 8: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** integration/filesystem_mcp_test.go:106
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 9: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** internal/mcp/client.go:178
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 10: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** internal/core/history_test.go:269
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 11: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** internal/core/history_test.go:275
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 12: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** examples/project_review/main.go:398
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 13: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** examples/project_review/main.go:405
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 14: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** examples/security_scan/main.go:407
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 15: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** examples/security_scan/main.go:414
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 16: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** internal/mcp/client_test.go:423
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 17: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** internal/mcp/client_test.go:433
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 18: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** internal/mcp/client_test.go:443
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 19: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** internal/mcp/client_test.go:453
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 20: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** internal/mcp/client_test.go:463
- **Description:** Tool invokes other tools without permission check
- **Matched:** `CallTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

*... and 4 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| State Confusion | 38 |
| Missing Input Validation | 18 |
| Cross Tool Escalation | 9 |
| Unlimited Approval | 7 |
| Prompt Injection | 4 |
| Tool Poisoning | 4 |

## Recommendations

1. **Prompt Injection:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.
2. **Unlimited Approval:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.
3. **Cross Tool Escalation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

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
