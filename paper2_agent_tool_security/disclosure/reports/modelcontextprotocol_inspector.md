# Security Disclosure: modelcontextprotocol/inspector

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 87.3)
**Protocol:** mcp
**Repository:** [modelcontextprotocol/inspector](https://github.com/modelcontextprotocol/inspector)
**Language:** TypeScript
**Stars:** 9364

## Summary
We identified 80 potential security issues in modelcontextprotocol/inspector
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 8 |
| Medium | 71 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** client/src/components/__tests__/AuthDebugger.test.tsx:531
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `secret: "static_client_secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** cli/src/client/tools.ts:86
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 3: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** cli/src/client/tools.ts:126
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 4: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** cli/src/index.ts:142
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 5: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** client/src/__tests__/App.toolsAppsPrefill.test.tsx:154
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 6: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** client/src/components/AppsTab.tsx:219
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 7: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** client/src/components/ToolsTab.tsx:827
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 8: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** client/src/App.tsx:1550
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 9: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** client/src/App.tsx:1621
- **Description:** Tool invokes other tools without permission check
- **Matched:** `callTool(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 69 |
| Cross Tool Escalation | 8 |
| Missing Harness | 2 |
| Hardcoded Credentials | 1 |

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
