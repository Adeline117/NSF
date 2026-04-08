# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in trailofbits/slither-mcp

## Summary

During our academic research on AI agent tool interface security, we identified **46** potential security vulnerabilities in your **trailofbits/slither-mcp** project (46 from static analysis).

- **Risk score:** 84.1/100
- **Risk rating:** critical
- **Protocol:** openai
- **GitHub stars:** 81

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 2 |
| High | 40 |
| Medium | 4 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Cross Tool Escalation | 16 |
| No Slippage Protection | 14 |
| Unlimited Approval | 6 |
| State Confusion | 4 |
| Tool Poisoning | 2 |
| Excessive Permissions | 2 |
| Prompt Injection | 2 |

## Top Findings (Most Severe)

### PI-001: Prompt Injection (CRITICAL)
- **File:** slither_mcp/tools/get_inherited_contracts.py
- **Line:** 41
- **Description:** User input interpolated into description/prompt/system message
- **CWE:** CWE-74
- **Code context:**
  ```
      39:         int | None,
      40:         Field(
  >>> 41:             description=f"Maximum inheritance depth to traverse (None for unlimited, default {DEFAULT_MAX_DEPTH})"
      42:         ),
      43:     ] = DEFAULT_MAX_DEPTH
  ```
- **Impact:** User-controlled input interpolated into prompts could hijack the LLM's behavior, bypassing safety checks.
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### PI-001: Prompt Injection (CRITICAL)
- **File:** slither_mcp/tools/get_derived_contracts.py
- **Line:** 41
- **Description:** User input interpolated into description/prompt/system message
- **CWE:** CWE-74
- **Code context:**
  ```
      39:         int | None,
      40:         Field(
  >>> 41:             description=f"Maximum derivation depth to traverse (None for unlimited, default {DEFAULT_MAX_DEPTH})"
      42:         ),
      43:     ] = DEFAULT_MAX_DEPTH
  ```
- **Impact:** User-controlled input interpolated into prompts could hijack the LLM's behavior, bypassing safety checks.
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### TP-001: Tool Poisoning (HIGH)
- **File:** tests/conftest.py
- **Line:** 547
- **Description:** Tool/function description contains directive instructions
- **CWE:** CWE-913
- **Code context:**
  ```
      545:                 impact="High",
      546:                 confidence="High",
  >>> 547:                 description="Contract.storageVar (contracts/Contract.sol#3) is never initialized",
      548:                 source_locations=[
      549:                     SourceLocation(file_path="contracts/Contract.sol", start_line=3, end_line=3)
  ```
- **Impact:** Malicious instructions in tool descriptions could manipulate the LLM into performing unintended actions on behalf of the user.
- **Recommendation:** Keep tool descriptions factual. Never include instructions for the LLM in tool descriptions or metadata. Avoid directive language (e.g., 'always', 'you must', 'ignore previous').

### CE-001: Cross Tool Escalation (HIGH)
- **File:** tests/test_mcp_client.py
- **Line:** 382
- **Description:** Tool invokes other tools without permission check
- **CWE:** CWE-284
- **Code context:**
  ```
      380:         call_count = [0]
      381: 
  >>> 382:         async def mock_call_tool(tool_name, arguments):
      383:             mock_result = MagicMock()
      384:             mock_content = MagicMock()
  ```
- **Impact:** One tool invoking another without permission checks could allow privilege escalation or unauthorized actions.
- **Recommendation:** Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### CE-001: Cross Tool Escalation (HIGH)
- **File:** tests/test_mcp_client.py
- **Line:** 452
- **Description:** Tool invokes other tools without permission check
- **CWE:** CWE-284
- **Code context:**
  ```
      450: 
      451:         # Set up mock responses
  >>> 452:         async def mock_call_tool(tool_name, arguments):
      453:             mock_result = MagicMock()
      454:             mock_content = MagicMock()
  ```
- **Impact:** One tool invoking another without permission checks could allow privilege escalation or unauthorized actions.
- **Recommendation:** Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

## Remediation Recommendations

### Cross Tool Escalation
Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### Excessive Permissions
Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### No Slippage Protection
Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### Prompt Injection
Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### Tool Poisoning
Keep tool descriptions factual. Never include instructions for the LLM in tool descriptions or metadata. Avoid directive language (e.g., 'always', 'you must', 'ignore previous').

### Unlimited Approval
Cap approval amounts to the exact needed value. Never use MaxUint256 (type(uint256).max) as an approval amount. Implement increaseAllowance/decreaseAllowance patterns instead.

## Disclosure Timeline

- **2026-04-07:** Vulnerabilities discovered during automated scanning
- **2026-04-07:** This report sent to maintainers
- **2026-07-06:** Planned public disclosure (per responsible disclosure standard, 90 days)

## About This Research

This work is part of an academic research project studying the security of AI agent tool interfaces across protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native modules). The study analyzed 43 Web3-related repositories using both static pattern analysis and dynamic tool-interface testing.

Paper submitted to [VENUE].

## Contact

For questions about this report, please contact:
- [Your Name] <[email]>
- [Advisor Name] <[email]>

We are happy to work with you on remediation and will update the planned public disclosure timeline if needed to allow adequate time for fixes.
