# Security Disclosure: apify/apify-mcp-server

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 93.4)
**Protocol:** mcp
**Repository:** [apify/apify-mcp-server](https://github.com/apify/apify-mcp-server)
**Language:** TypeScript
**Stars:** 1026

## Summary
We identified 92 potential security issues in apify/apify-mcp-server
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 3 |
| High | 68 |
| Medium | 21 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/const.ts:129
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `apiKey: 'e97714a64e2b4b8b8fe0b01cd8592870'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/const.ts:140
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `apiKey: '267679200b833c2ca1255ab276731869'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/const.ts:152
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `apiKey: '878493fcd7001e3c179b6db6796a999b'`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 4: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:60
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 5: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/web/src/utils/mock-openai.ts:70
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 6: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** evals/workflows/mcp_client.ts:113
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 7: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** evals/workflows/mcp_client.ts:119
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 8: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** evals/workflows/conversation_executor.ts:133
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 9: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/tools/core/call_actor_common.ts:148
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 10: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/helpers.ts:185
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 11: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:525
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 12: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:558
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 13: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:566
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 14: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:579
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 15: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:606
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 16: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:631
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 17: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:650
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 18: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:673
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 19: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:707
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 20: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** tests/integration/suite.ts:734
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

*... and 51 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Cross Tool Escalation | 68 |
| Missing Input Validation | 19 |
| Hardcoded Credentials | 3 |
| State Confusion | 1 |
| Missing Harness | 1 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Cross Tool Escalation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

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
