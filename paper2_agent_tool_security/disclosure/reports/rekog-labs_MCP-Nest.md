# Security Disclosure: rekog-labs/MCP-Nest

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 96.5)
**Protocol:** mcp
**Repository:** [rekog-labs/MCP-Nest](https://github.com/rekog-labs/MCP-Nest)
**Language:** TypeScript
**Stars:** 620

## Summary
We identified 113 potential security issues in rekog-labs/MCP-Nest
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 6 |
| High | 84 |
| Medium | 23 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/mcp-multi-auth.e2e.spec.ts:93
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: 'secret-a-that-is-at-least-32-characters-long'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/mcp-multi-auth.e2e.spec.ts:94
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: 'jwt-secret-a-that-is-at-least-32-characters-long'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/mcp-multi-auth.e2e.spec.ts:109
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: 'secret-b-that-is-at-least-32-characters-long'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 4: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/mcp-multi-auth.e2e.spec.ts:110
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret: 'jwt-secret-b-that-is-at-least-32-characters-long'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 5: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/mcp-oauth-auth.e2e.spec.ts:183
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Secret = 'test-jwt-secret-that-is-at-least-32-characters-long'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 6: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/mcp-oauth-auth.e2e.spec.ts:637
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token: 'invalid-refresh-token'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 7: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** tests/mcp.useFilters.e2e.spec.ts:43
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: 'Method-level filter overrides class-level'`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 8: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** playground/clients/http-sse.ts:54
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 9: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/mcp-logging.e2e.spec.ts:61
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 10: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** playground/clients/stdio-client.ts:66
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 11: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** tests/mcp.useFilters.e2e.spec.ts:75
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: 'Method-level filter overrides class-level'`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 12: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** playground/clients/stdio-client.ts:88
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 13: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/mcp-logging.e2e.spec.ts:92
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 14: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/mcp-guard-without-user.e2e.spec.ts:92
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 15: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** playground/clients/stdio-client.ts:110
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 16: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** tests/mcp.useFilters.e2e.spec.ts:110
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: 'Method-level filter overrides class-level'`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 17: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/mcp-logging.e2e.spec.ts:128
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 18: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** playground/clients/http-sse-azure-ad-oauth-client.ts:137
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 19: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/mcp-logging.e2e.spec.ts:162
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 20: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/mcp-tool-auth.e2e.spec.ts:162
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

*... and 70 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Cross Tool Escalation | 81 |
| Missing Input Validation | 15 |
| State Confusion | 7 |
| Hardcoded Credentials | 6 |
| Tool Poisoning | 3 |
| Missing Harness | 1 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Tool Poisoning:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.
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
