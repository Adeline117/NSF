# Security Disclosure: opensumi/core

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 94.3)
**Protocol:** mcp
**Repository:** [opensumi/core](https://github.com/opensumi/core)
**Language:** TypeScript
**Stars:** 3620

## Summary
We identified 130 potential security issues in opensumi/core
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 16 |
| Medium | 114 |

## Critical / High Severity Findings

### Finding 1: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/common/mcp-server-manager.ts:9
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 2: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/common/mcp-server-manager.ts:17
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 3: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/node/mcp-server-manager-impl.ts:72
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 4: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/node/mcp-server.sse.ts:79
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 5: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/__test__/node/mcp-server.sse.test.ts:82
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 6: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/node/mcp-server-manager-impl.ts:82
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 7: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/__test__/common/mcp-server-manager.test.ts:86
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 8: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/__test__/node/mcp-server.stdio.test.ts:90
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 9: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/__test__/node/mcp-server.sse.test.ts:93
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 10: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/node/mcp-server.stdio.ts:94
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 11: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/node/mcp-server.sse.ts:97
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 12: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/__test__/node/mcp-server.stdio.test.ts:101
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 13: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/node/mcp-server.stdio.ts:112
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 14: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/node/mcp-server-manager-impl.ts:113
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 15: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** packages/ai-native/src/node/mcp/sumi-mcp-server.ts:260
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 16: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** packages/ai-native/src/browser/context/llm-context.service.ts:407
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description="This is a rule set by the user that the agent must follow."`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 106 |
| Cross Tool Escalation | 15 |
| State Confusion | 7 |
| Tool Poisoning | 1 |
| Missing Harness | 1 |

## Recommendations

1. **Cross Tool Escalation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.
2. **Tool Poisoning:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

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
