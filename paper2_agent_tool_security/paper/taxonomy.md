# Appendix A: Unified Vulnerability Taxonomy

**27** patterns across **21** categories, **16** unique CWEs, **5** attack surfaces, **2** OWASP LLM Top-10 tie-ins.

Severity distribution: **6** critical, **11** high, **10** medium, **0** low.

## Attack Surfaces

| ID | Surface | # Patterns |
|----|---------|-----------|
| S1 | S1_tool_definition | 4 |
| S2 | S2_input_construction | 3 |
| S3 | S3_execution | 15 |
| S4 | S4_output_handling | 2 |
| S5 | S5_cross_tool | 3 |

## CWE Mapping Summary

| CWE | Name | # Patterns |
|-----|------|-----------|
| CWE-116 | Improper Encoding/Escaping | 1 |
| CWE-20 | Improper Input Validation | 2 |
| CWE-200 | Exposure of Sensitive Information | 4 |
| CWE-250 | Execution with Unnecessary Privileges | 3 |
| CWE-252 | Unchecked Return Value | 1 |
| CWE-269 | Improper Privilege Management | 1 |
| CWE-284 | Improper Access Control | 4 |
| CWE-319 | Cleartext Transmission of Sensitive Information | 1 |
| CWE-345 | Insufficient Verification of Data Authenticity | 1 |
| CWE-362 | Concurrent Execution with Improper Synchronization (Race) | 1 |
| CWE-74 | Improper Neutralization of Special Elements (Injection) | 1 |
| CWE-78 | OS Command Injection | 1 |
| CWE-798 | Use of Hard-coded Credentials | 1 |
| CWE-829 | Inclusion of Functionality from Untrusted Control Sphere | 1 |
| CWE-862 | Missing Authorization | 2 |
| CWE-913 | Improper Control of Dynamically-Managed Code Resources | 2 |

## Full Pattern Catalog

### S1_tool_definition

| ID | Severity | Category | CWE | OWASP LLM | Protocols | Description |
|----|----------|----------|-----|-----------|-----------|-------------|
| S1-TP-001 | HIGH | tool_poisoning | CWE-913 | LLM01 | mcp,openai,langchain | Tool/function description contains directive instructions for the LLM (e.g., 'al… |
| S1-TP-002 | MEDIUM | tool_poisoning | CWE-913 | LLM01 | mcp,openai,langchain | Tool description references other tools by name, potentially directing the LLM t… |
| S1-EP-001 | HIGH | excessive_permissions | CWE-250 | LLM06 | openai,langchain | Function/tool name indicates destructive operation without scope restriction (ex… |
| S1-SL-001 | MEDIUM | skill_scope_leak | CWE-200 | - | mcp,openai,langchain | Tool description exposes internal implementation details (database, schema, endp… |

### S2_input_construction

| ID | Severity | Category | CWE | OWASP LLM | Protocols | Description |
|----|----------|----------|-----|-----------|-----------|-------------|
| S2-PI-001 | CRITICAL | prompt_injection | CWE-74 | LLM01 | mcp,openai,langchain | User input interpolated into tool description, prompt template, or system messag… |
| S2-IV-001 | MEDIUM | missing_input_validation | CWE-20 | - | mcp,openai,langchain | Address parameter accepted as string without Ethereum address validation |
| S2-IV-002 | MEDIUM | missing_input_validation | CWE-20 | - | mcp,openai,langchain,web3_native | Amount/value parameter without bounds checking or numeric validation |

### S3_execution

| ID | Severity | Category | CWE | OWASP LLM | Protocols | Description |
|----|----------|----------|-----|-----------|-----------|-------------|
| S3-PKE-001 | CRITICAL | private_key_exposure | CWE-200 | - | mcp,openai,langchain | Private key, mnemonic, or seed phrase accepted as tool parameter or present in t… |
| S3-PKE-002 | CRITICAL | private_key_exposure | CWE-862 | - | mcp,openai,langchain | Wallet signing operation without user confirmation flow |
| S3-HC-001 | CRITICAL | hardcoded_credentials | CWE-798 | - | mcp,openai,langchain | Hardcoded API keys, passwords, secrets, or tokens |
| S3-CE-001 | HIGH | command_injection | CWE-78 | LLM06 | langchain | Tool executes OS commands or shell operations without sandboxing |
| S3-UA-001 | HIGH | unlimited_approval | CWE-250 | - | mcp,openai,langchain,web3_native | Constructs unlimited token approval (MaxUint256 or equivalent) |
| S3-TV-001 | HIGH | tx_validation_missing | CWE-345 | - | mcp,openai,langchain | Transaction constructed and sent without validating target contract or parameter… |
| S3-IR-001 | MEDIUM | insecure_rpc | CWE-319 | - | mcp,openai,langchain | HTTP (not HTTPS) RPC endpoint used for blockchain communication |
| S3-MH-001 | MEDIUM | missing_harness | CWE-284 | - | mcp | MCP server instantiated without security wrapper, sandbox, or policy enforcement |
| S3-MH-002 | MEDIUM | missing_harness | CWE-284 | - | langchain | Agent initialized without safety constraints (max_iterations, callbacks, allowed… |
| S3-DC-001 | CRITICAL | delegatecall_abuse | CWE-829 | - | web3_native | delegatecall to user-supplied or unvalidated address |
| S3-UC-001 | HIGH | unchecked_external_call | CWE-252 | - | web3_native | External call (.call, .send, .transfer) without checking return value |
| S3-PE-001 | CRITICAL | privilege_escalation | CWE-269 | - | web3_native | Module can change owner, add modules, or set fallback handler without timelock/m… |
| S3-AI-001 | HIGH | approval_injection | CWE-862 | - | web3_native | Module can set ERC-20 approvals on behalf of the safe/account without whitelisti… |
| W3-MEV-001 | HIGH | mev_exploitation | CWE-200 | - | mcp,openai,langchain,web3_native | Swap/trade constructed without slippage protection, deadline, or private relay |
| W3-EP-001 | HIGH | excessive_permissions | CWE-250 | - | web3_native | Module uses DELEGATECALL operation type in Safe transaction (full storage access… |

### S4_output_handling

| ID | Severity | Category | CWE | OWASP LLM | Protocols | Description |
|----|----------|----------|-----|-----------|-----------|-------------|
| S4-NV-001 | MEDIUM | no_output_validation | CWE-116 | - | mcp,openai,langchain | Tool/function result passed directly to LLM without sanitization or validation |
| S4-DL-001 | MEDIUM | data_leakage | CWE-200 | - | mcp,openai,langchain | Tool returns raw internal data structures (DB rows, full API responses, error tr… |

### S5_cross_tool

| ID | Severity | Category | CWE | OWASP LLM | Protocols | Description |
|----|----------|----------|-----|-----------|-----------|-------------|
| S5-CE-001 | HIGH | tool_chain_escalation | CWE-284 | LLM06 | mcp,langchain | Tool can invoke other tools or agents without permission check or access control |
| S5-SC-001 | MEDIUM | state_confusion | CWE-362 | - | openai,langchain | Mutable global state shared across tool/function invocations without isolation |
| S5-MH-003 | HIGH | missing_harness | CWE-284 | - | web3_native | Module executes transactions without guard, sentinel, or spending limit |

## Remediation Guide

### S1_tool_definition

- **S1-TP-001** (high): Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.
- **S1-TP-002** (medium): Tools should be independently usable. Cross-tool dependencies should be enforced programmatically, not via description text.
- **S1-EP-001** (high): Scope destructive operations by resource type and add confirmation requirements.
- **S1-SL-001** (medium): Redact implementation details from tool descriptions. Only expose the functional interface.

### S2_input_construction

- **S2-PI-001** (critical): Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.
- **S2-IV-001** (medium): Validate Ethereum addresses using checksum verification (e.g., ethers.isAddress()) before processing.
- **S2-IV-002** (medium): Validate numeric inputs: check for negative values, overflow, and enforce per-transaction/per-session limits.

### S3_execution

- **S3-PKE-001** (critical): Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).
- **S3-PKE-002** (critical): All signing operations must require explicit user confirmation with clear display of what is being signed.
- **S3-HC-001** (critical): Use environment variables or a secret manager. Never hardcode credentials in source code.
- **S3-CE-001** (high): Avoid shell execution. If required, use allowlists, input escaping (shlex.quote), and process sandboxing.
- **S3-UA-001** (high): Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.
- **S3-TV-001** (high): Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.
- **S3-IR-001** (medium): Always use HTTPS for RPC endpoints. HTTP exposes transaction data and API keys to network sniffers.
- **S3-MH-001** (medium): Wrap MCP server instantiation with a security harness that enforces tool-level permissions and rate limits.
- **S3-MH-002** (medium): Set max_iterations, handle_parsing_errors, and restrict allowed_tools when initializing agents.
- **S3-DC-001** (critical): Restrict delegatecall targets to immutable, audited addresses. Never delegatecall to user-supplied addresses.
- **S3-UC-001** (high): Always check return values of external calls. Use require() to revert on failure.
- **S3-PE-001** (critical): Gate critical state changes behind timelock and/or multi-signature requirements.
- **S3-AI-001** (high): Whitelist approved spender addresses and cap approval amounts.
- **W3-MEV-001** (high): Set explicit slippage tolerance (e.g., 0.5-1%), transaction deadline, and use a private transaction relay.
- **W3-EP-001** (high): Avoid DelegateCall operation type unless absolutely necessary. Prefer Call (operation=0).

### S4_output_handling

- **S4-NV-001** (medium): Sanitize tool outputs before including in LLM context. Filter prompt injection patterns, truncate length, and redact sensitive data.
- **S4-DL-001** (medium): Map internal data to a response schema. Never return raw database results, full API responses, or stack traces.

### S5_cross_tool

- **S5-CE-001** (high): Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.
- **S5-SC-001** (medium): Isolate state between tool invocations. Use immutable context objects or per-invocation namespaces.
- **S5-MH-003** (high): Install a transaction guard that enforces per-transaction and per-period spending limits.
