# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in waifuai/mcp-solana-ico

## Summary

During our academic research on AI agent tool interface security, we identified **22** potential security vulnerabilities in your **waifuai/mcp-solana-ico** project (22 from static analysis).

- **Risk score:** 68.6/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 3

## Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 2 |
| Medium | 20 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 16 |
| State Confusion | 4 |
| Excessive Permissions | 2 |

## Top Findings (Most Severe)

### EP-001: Excessive Permissions (HIGH)
- **File:** mcp_solana_ico/pricing.py
- **Line:** 98
- **Description:** Destructive operation without scope restriction
- **CWE:** CWE-78
- **Code context:**
  ```
      96:                  raise ValueError(f"Custom formula or initial_price not set for ICO {ico_id}.")
      97:             try:
  >>> 98:                 # WARNING: Using eval() is potentially unsafe. Sanitize or use a safer evaluation method.
      99:                 # Provide necessary variables in the eval context.
      100:                 eval_context = {
  ```
- **Impact:** Tools with excessive permissions increase the blast radius if any single tool is compromised.
- **Recommendation:** Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### EP-001: Excessive Permissions (HIGH)
- **File:** mcp_solana_ico/pricing.py
- **Line:** 108
- **Description:** Destructive operation without scope restriction
- **CWE:** CWE-78
- **Code context:**
  ```
      106:                     # Add other relevant variables/functions safely
      107:                 }
  >>> 108:                 base_price_per_token = eval(ico.ico.custom_formula, {"__builtins__": {}}, eval_context)
      109:             except Exception as e:
      110:                 logger.error(f"Error evaluating custom formula for ICO {ico_id}: {e}")
  ```
- **Impact:** Tools with excessive permissions increase the blast radius if any single tool is compromised.
- **Recommendation:** Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### IV-002: Missing Input Validation (MEDIUM)
- **File:** tests/integration/test_ico_server.py
- **Line:** 349
- **Description:** Amount/value parameter without bounds checking
- **CWE:** CWE-20
- **Code context:**
  ```
      347:                 mock_transfer.return_value = "mock_tx_hash"
      348:                 mock_context = MagicMock()
  >>> 349:                 result = await server.buy_tokens(context=mock_context, ico_id=ico_id, amount=1000, payment_transaction=str(Signature.default()), client_ip="127.0.0.1", sell=False)
      350:                 assert "Successfully purchased" in result
      351: 
  ```
- **Impact:** Unvalidated inputs could lead to injection attacks, unexpected behavior, or interaction with unintended contracts.
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### IV-002: Missing Input Validation (MEDIUM)
- **File:** tests/integration/test_ico_server.py
- **Line:** 365
- **Description:** Amount/value parameter without bounds checking
- **CWE:** CWE-20
- **Code context:**
  ```
      363:                 mock_transfer.return_value = "mock_tx_hash"
      364:                 mock_context = MagicMock()
  >>> 365:                 result = await server.buy_tokens(context=mock_context, ico_id=ico_id, amount=1000, payment_transaction=str(Signature.default()), client_ip="127.0.0.1", sell=False)
      366:                 assert "Successfully purchased" in result
      367: 
  ```
- **Impact:** Unvalidated inputs could lead to injection attacks, unexpected behavior, or interaction with unintended contracts.
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### IV-002: Missing Input Validation (MEDIUM)
- **File:** mcp_solana_ico/server.py
- **Line:** 92
- **Description:** Amount/value parameter without bounds checking
- **CWE:** CWE-20
- **Code context:**
  ```
      90: 
      91: 
  >>> 92: def validate_token_operation_params(ico_id: str, amount: int, payment_transaction: str, client_ip: str) -> None:
      93:     """
      94:     Validate common parameters for token operations.
  ```
- **Impact:** Unvalidated inputs could lead to injection attacks, unexpected behavior, or interaction with unintended contracts.
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

## Remediation Recommendations

### Excessive Permissions
Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

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
