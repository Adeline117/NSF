# Security Disclosure: getsentry/XcodeBuildMCP

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 100.0)
**Protocol:** mcp
**Repository:** [getsentry/XcodeBuildMCP](https://github.com/getsentry/XcodeBuildMCP)
**Language:** TypeScript
**Stars:** 5066

## Summary
We identified 229 potential security issues in getsentry/XcodeBuildMCP
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 150 |
| Medium | 79 |

## Critical / High Severity Findings

### Finding 1: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-scaffolding.test.ts:20
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 2: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-doctor.test.ts:22
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 3: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-logging.test.ts:24
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 4: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-swift-package.test.ts:26
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 5: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-error-paths.test.ts:31
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 6: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-scaffolding.test.ts:33
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 7: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-logging.test.ts:33
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 8: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/integrations/xcode-tools-bridge/__tests__/registry.integration.test.ts:35
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 9: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-sessions.test.ts:36
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 10: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-device-macos.test.ts:39
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 11: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-error-paths.test.ts:40
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 12: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-swift-package.test.ts:41
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 13: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-logging.test.ts:43
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 14: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-sessions.test.ts:44
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 15: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-device-macos.test.ts:48
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 16: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-error-paths.test.ts:48
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 17: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-invocation.test.ts:48
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 18: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-ui-automation.test.ts:51
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 19: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-logging.test.ts:55
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 20: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** src/smoke-tests/__tests__/e2e-mcp-swift-package.test.ts:56
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

*... and 130 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Cross Tool Escalation | 150 |
| Missing Input Validation | 63 |
| State Confusion | 13 |
| Missing Harness | 3 |

## Recommendations

1. **Cross Tool Escalation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

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
