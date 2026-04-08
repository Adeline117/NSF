# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in caiovicentino/polymarket-mcp-server

## Summary

During our academic research on AI agent tool interface security, we identified **34** potential security vulnerabilities in your **caiovicentino/polymarket-mcp-server** project (34 from static analysis).

- **Risk score:** 76.7/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 348

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 1 |
| High | 11 |
| Medium | 22 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| State Confusion | 12 |
| Missing Input Validation | 10 |
| No Slippage Protection | 6 |
| Cross Tool Escalation | 3 |
| Excessive Permissions | 1 |
| Tool Poisoning | 1 |
| Private Key Exposure | 1 |

## Top Findings (Most Severe)

### PKE-001: Private Key Exposure (CRITICAL)
- **File:** src/polymarket_mcp/auth/signer.py
- **Line:** 49
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **CWE:** CWE-200
- **Code context:**
  ```
      47:         # Ensure private key has 0x prefix for eth_account
      48:         if not private_key.startswith("0x"):
  >>> 49:             private_key = "0x" + private_key
      50: 
      51:         self.account = Account.from_key(private_key)
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### SLP-001: No Slippage Protection (HIGH)
- **File:** run_trading_tests.py
- **Line:** 139
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      137:             print(f"   FAILED: {result.get('error')}")
      138: 
  >>> 139:         # Test 4: Smart trade (dry run)
      140:         print("\n9. Testing smart trade execution...")
      141:         result = await trading_tools.execute_smart_trade(
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** run_trading_tests.py
- **Line:** 141
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      139:         # Test 4: Smart trade (dry run)
      140:         print("\n9. Testing smart trade execution...")
  >>> 141:         result = await trading_tools.execute_smart_trade(
      142:             market_id=market_id,
      143:             intent="Buy YES at a good price, be patient",
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** tests/test_trading_tools.py
- **Line:** 350
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      348:     """Test Smart Trading tools (2 tools)"""
      349: 
  >>> 350:     async def test_execute_smart_trade(self, setup):
      351:         """Test AI-powered smart trade execution"""
      352:         print("\n--- Testing execute_smart_trade ---")
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** tests/test_trading_tools.py
- **Line:** 357
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      355: 
      356:         # Test with conservative intent
  >>> 357:         result = await trading_tools.execute_smart_trade(
      358:             market_id=TEST_MARKET_ID,
      359:             intent="Buy YES at a good price, be patient",
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

## Remediation Recommendations

### Cross Tool Escalation
Implement per-tool permission checks. Do not allow one tool to invoke another without explicit authorization. Use capability-based access control for sensitive operations.

### Excessive Permissions
Apply the principle of least privilege. Only request the permissions each tool actually needs. Separate read-only and write tools into distinct permission scopes.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### No Slippage Protection
Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### Private Key Exposure
Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### State Confusion
Isolate per-session state. Do not share mutable state across tool invocations from different users or sessions. Use request-scoped context objects.

### Tool Poisoning
Keep tool descriptions factual. Never include instructions for the LLM in tool descriptions or metadata. Avoid directive language (e.g., 'always', 'you must', 'ignore previous').

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
