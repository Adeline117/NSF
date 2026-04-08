# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in armorwallet/armor-crypto-mcp

## Summary

During our academic research on AI agent tool interface security, we identified **27** potential security vulnerabilities in your **armorwallet/armor-crypto-mcp** project (27 from static analysis).

- **Risk score:** 73.3/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 192

## Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 10 |
| Medium | 17 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 16 |
| Excessive Permissions | 8 |
| Tool Poisoning | 2 |
| State Confusion | 1 |

## Top Findings (Most Severe)

### TP-001: Tool Poisoning (HIGH)
- **File:** armor_crypto_mcp/armor_client.py
- **Line:** 295
- **Description:** Tool/function description contains directive instructions
- **CWE:** CWE-913
- **Code context:**
  ```
      293:     direction: Literal["ABOVE", "BELOW"] = Field(description="whether or not the order is above or below current market value")
      294:     token_address_watcher: str = Field(description="public address of the token to watch. should be output token for limit orders and input token for stop loss and take profit orders")
  >>> 295:     target_value: Optional[float] = Field(description="target value to execute the order. You must always specify a target value or delta percentage.")
      296:     delta_percentage: Optional[float] = Field(description="delta percentage to execute the order. You must always specify a target value or delta percentage.")
      297: 
  ```
- **Impact:** Malicious instructions in tool descriptions could manipulate the LLM into performing unintended actions on behalf of the user.
- **Recommendation:** Keep tool descriptions factual. Never include instructions for the LLM in tool descriptions or metadata. Avoid directive language (e.g., 'always', 'you must', 'ignore previous').

### TP-001: Tool Poisoning (HIGH)
- **File:** armor_crypto_mcp/armor_client.py
- **Line:** 296
- **Description:** Tool/function description contains directive instructions
- **CWE:** CWE-913
- **Code context:**
  ```
      294:     token_address_watcher: str = Field(description="public address of the token to watch. should be output token for limit orders and input token for stop loss and take profit orders")
      295:     target_value: Optional[float] = Field(description="target value to execute the order. You must always specify a target value or delta percentage.")
  >>> 296:     delta_percentage: Optional[float] = Field(description="delta percentage to execute the order. You must always specify a target value or delta percentage.")
      297: 
      298: class OrderWatcher(BaseModel):
  ```
- **Impact:** Malicious instructions in tool descriptions could manipulate the LLM into performing unintended actions on behalf of the user.
- **Recommendation:** Keep tool descriptions factual. Never include instructions for the LLM in tool descriptions or metadata. Avoid directive language (e.g., 'always', 'you must', 'ignore previous').

### EP-001: Excessive Permissions (HIGH)
- **File:** armor_crypto_mcp/armor_client.py
- **Line:** 762
- **Description:** Destructive operation without scope restriction
- **CWE:** CWE-78
- **Code context:**
  ```
      760:         ast.USub: operator.neg
      761:     }
  >>> 762:     def _eval(node):
      763:         if isinstance(node, ast.Num):
      764:             return node.n
  ```
- **Impact:** Tools with excessive permissions increase the blast radius if any single tool is compromised.
- **Recommendation:** Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### EP-001: Excessive Permissions (HIGH)
- **File:** armor_crypto_mcp/armor_client.py
- **Line:** 768
- **Description:** Destructive operation without scope restriction
- **CWE:** CWE-78
- **Code context:**
  ```
      766:             return node.value
      767:         elif isinstance(node, ast.BinOp):
  >>> 768:             return ops[type(node.op)](_eval(node.left), _eval(node.right))
      769:         elif isinstance(node, ast.UnaryOp):
      770:             return ops[type(node.op)](_eval(node.operand))
  ```
- **Impact:** Tools with excessive permissions increase the blast radius if any single tool is compromised.
- **Recommendation:** Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### EP-001: Excessive Permissions (HIGH)
- **File:** armor_crypto_mcp/armor_client.py
- **Line:** 768
- **Description:** Destructive operation without scope restriction
- **CWE:** CWE-78
- **Code context:**
  ```
      766:             return node.value
      767:         elif isinstance(node, ast.BinOp):
  >>> 768:             return ops[type(node.op)](_eval(node.left), _eval(node.right))
      769:         elif isinstance(node, ast.UnaryOp):
      770:             return ops[type(node.op)](_eval(node.operand))
  ```
- **Impact:** Tools with excessive permissions increase the blast radius if any single tool is compromised.
- **Recommendation:** Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

## Dynamic Testing Findings

### DYN-001: Prompt Injection Output (HIGH)
- **Tool:** get_armor_mcp_version
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### DYN-002: Prompt Injection Output (HIGH)
- **Tool:** wait_a_moment
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### DYN-003: Prompt Injection Output (HIGH)
- **Tool:** get_current_time
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### DYN-004: Prompt Injection Output (HIGH)
- **Tool:** calculator
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### DYN-005: Prompt Injection Output (HIGH)
- **Tool:** get_wallet_token_balance
- **Details:** Tool output interpolates user-controlled data
- **CWE:** CWE-74
- **Recommendation:** Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

## Remediation Recommendations

### Excessive Permissions
Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### Prompt Injection
Never interpolate user input into tool descriptions or system prompts. Sanitize all dynamic content before inclusion in LLM context. Use parameterized tool interfaces.

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### Tool Poisoning
Keep tool descriptions factual. Never include instructions for the LLM in tool descriptions or metadata. Avoid directive language (e.g., 'always', 'you must', 'ignore previous').

### Tx Validation Missing
Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

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
