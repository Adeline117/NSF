# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in alpacahq/alpaca-mcp-server

## Summary

During our academic research on AI agent tool interface security, we identified **40** potential security vulnerabilities in your **alpacahq/alpaca-mcp-server** project (40 from static analysis).

- **Risk score:** 82.3/100
- **Risk rating:** critical
- **Protocol:** mcp
- **GitHub stars:** 613

## Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 40 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Cross Tool Escalation | 37 |
| No Slippage Protection | 3 |

## Top Findings (Most Severe)

### CE-001: Cross Tool Escalation (HIGH)
- **File:** tests/conftest.py
- **Line:** 27
- **Description:** Tool invokes other tools without permission check
- **CWE:** CWE-284
- **Code context:**
  ```
      25:     server = build_server()
      26:     async with Client(transport=server) as mcp:
  >>> 27:         await mcp.call_tool("cancel_all_orders", {})
      28:         await mcp.call_tool("close_all_positions", {})
      29:         # close_all_positions may queue sell orders when market is closed;
  ```
- **Impact:** One tool invoking another without permission checks could allow privilege escalation or unauthorized actions.
- **Recommendation:** Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### CE-001: Cross Tool Escalation (HIGH)
- **File:** tests/conftest.py
- **Line:** 28
- **Description:** Tool invokes other tools without permission check
- **CWE:** CWE-284
- **Code context:**
  ```
      26:     async with Client(transport=server) as mcp:
      27:         await mcp.call_tool("cancel_all_orders", {})
  >>> 28:         await mcp.call_tool("close_all_positions", {})
      29:         # close_all_positions may queue sell orders when market is closed;
      30:         # cancel them so they don't trigger wash-trade rejections in tests.
  ```
- **Impact:** One tool invoking another without permission checks could allow privilege escalation or unauthorized actions.
- **Recommendation:** Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### CE-001: Cross Tool Escalation (HIGH)
- **File:** tests/conftest.py
- **Line:** 31
- **Description:** Tool invokes other tools without permission check
- **CWE:** CWE-284
- **Code context:**
  ```
      29:         # close_all_positions may queue sell orders when market is closed;
      30:         # cancel them so they don't trigger wash-trade rejections in tests.
  >>> 31:         await mcp.call_tool("cancel_all_orders", {})
      32:     yield
      33: 
  ```
- **Impact:** One tool invoking another without permission checks could allow privilege escalation or unauthorized actions.
- **Recommendation:** Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### CE-001: Cross Tool Escalation (HIGH)
- **File:** tests/test_paper_integration.py
- **Line:** 71
- **Description:** Tool invokes other tools without permission check
- **CWE:** CWE-284
- **Code context:**
  ```
      69:     server = build_server()
      70:     async with Client(transport=server) as c:
  >>> 71:         raw = await c.call_tool(tool_name, args or {})
      72:     return _parse(raw)
      73: 
  ```
- **Impact:** One tool invoking another without permission checks could allow privilege escalation or unauthorized actions.
- **Recommendation:** Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### CE-001: Cross Tool Escalation (HIGH)
- **File:** tests/test_paper_integration.py
- **Line:** 375
- **Description:** Tool invokes other tools without permission check
- **CWE:** CWE-284
- **Code context:**
  ```
      373:     server = build_server()
      374:     async with Client(transport=server) as mcp:
  >>> 375:         order = _parse(await mcp.call_tool("place_stock_order", {
      376:             "symbol": "AAPL",
      377:             "side": "buy",
  ```
- **Impact:** One tool invoking another without permission checks could allow privilege escalation or unauthorized actions.
- **Recommendation:** Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

## Remediation Recommendations

### Cross Tool Escalation
Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### No Slippage Protection
Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

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
