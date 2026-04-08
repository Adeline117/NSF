# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in nirholas/pump-fun-sdk

## Summary

During our academic research on AI agent tool interface security, we identified **39** potential security vulnerabilities in your **nirholas/pump-fun-sdk** project (39 from static analysis).

- **Risk score:** 78.5/100
- **Risk rating:** high
- **Protocol:** unknown
- **GitHub stars:** 75

## Severity Breakdown

| Severity | Count |
|----------|-------|
| High | 14 |
| Medium | 25 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 25 |
| No Slippage Protection | 14 |

## Top Findings (Most Severe)

### SLP-001: No Slippage Protection (HIGH)
- **File:** pump-fun-repos/carbon/crates/macros/src/schemas.rs
- **Line:** 39
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      37: //!     any,
      38: //!     [
  >>> 39: //!         AllInstructionTypes::JupSwap(JupiterInstructionType::SwapEvent),
      40: //!         "jup_swap_event",
      41: //!         []
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** pump-fun-repos/carbon/crates/macros/src/schemas.rs
- **Line:** 107
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      105: ///     any
      106: ///     [
  >>> 107: ///         AllInstructionTypes::JupSwap(JupiterInstructionType::SwapEvent),
      108: ///         "jup_swap_event",
      109: ///         []
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** pump-fun-repos/carbon/crates/proc-macros/src/lib.rs
- **Line:** 651
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      649: ///
      650: /// This example will generate:
  >>> 651: /// - AllInstructions enum with variants JupSwap(JupiterInstruction) and
      652: ///   MeteoraSwap(MeteoraInstruction)
      653: /// - AllInstructionTypes enum with variants JupSwap(JupiterInstructionType) and
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** pump-fun-repos/carbon/crates/proc-macros/src/lib.rs
- **Line:** 652
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      650: /// This example will generate:
      651: /// - AllInstructions enum with variants JupSwap(JupiterInstruction) and
  >>> 652: ///   MeteoraSwap(MeteoraInstruction)
      653: /// - AllInstructionTypes enum with variants JupSwap(JupiterInstructionType) and
      654: ///   MeteoraSwap(MeteoraInstructionType)
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

### SLP-001: No Slippage Protection (HIGH)
- **File:** pump-fun-repos/carbon/crates/proc-macros/src/lib.rs
- **Line:** 653
- **Description:** Swap/trade without slippage protection
- **CWE:** CWE-345
- **Code context:**
  ```
      651: /// - AllInstructions enum with variants JupSwap(JupiterInstruction) and
      652: ///   MeteoraSwap(MeteoraInstruction)
  >>> 653: /// - AllInstructionTypes enum with variants JupSwap(JupiterInstructionType) and
      654: ///   MeteoraSwap(MeteoraInstructionType)
      655: /// - AllPrograms enum with variants JupSwap and MeteoraSwap
  ```
- **Impact:** Without slippage protection, DEX trades could suffer significant value loss through sandwich attacks.
- **Recommendation:** Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

## Dynamic Testing Findings

### DYN-001: Parameter Injection (MEDIUM)
- **Tool:** pumpAmm
- **Details:** No parameter schema defined -- accepts arbitrary input
- **CWE:** CWE-20
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### DYN-002: Parameter Injection (MEDIUM)
- **Tool:** pump
- **Details:** No parameter schema defined -- accepts arbitrary input
- **CWE:** CWE-20
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### DYN-003: Parameter Injection (MEDIUM)
- **Tool:** wallet
- **Details:** No parameter schema defined -- accepts arbitrary input
- **CWE:** CWE-20
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### DYN-004: Tx Validation (MEDIUM)
- **Tool:** wallet
- **Details:** No gas limit specified/capped in transaction; No contract verification before interaction
- **CWE:** CWE-345
- **Recommendation:** Validate all transaction parameters (to, value, data, gasLimit) before signing. Require user confirmation for transactions above a configurable threshold.

### DYN-005: Parameter Injection (MEDIUM)
- **Tool:** page
- **Details:** No parameter schema defined -- accepts arbitrary input
- **CWE:** CWE-20
- **Recommendation:** Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

## Remediation Recommendations

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### No Slippage Protection
Implement configurable slippage tolerance (e.g., 0.5-1%). Calculate minimum output amounts and set deadlines for DEX swaps.

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
