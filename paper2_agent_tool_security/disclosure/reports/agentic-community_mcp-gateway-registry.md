# Security Disclosure: agentic-community/mcp-gateway-registry

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 100.0)
**Protocol:** mcp
**Repository:** [agentic-community/mcp-gateway-registry](https://github.com/agentic-community/mcp-gateway-registry)
**Language:** Python
**Stars:** 561

## Summary
We identified 324 potential security issues in agentic-community/mcp-gateway-registry
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 56 |
| High | 56 |
| Medium | 212 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/utils/test_credential_encryption.py:32
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `secret = "test-secret-key-for-derivation"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** terraform/telemetry-collector/lambda/collector/index.py:35
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `SIGNING_KEY = "`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 3: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** registry/core/telemetry.py:38
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `SIGNING_KEY = "`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 4: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/federation_routes.py:62
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 5: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/unit/api/test_peer_management_routes.py:121
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="original-token-abc123"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 6: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/federation_routes.py:131
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 7: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** metrics-service/migrate.py:139
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `template = f'`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 8: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/ans_routes.py:153
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 9: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/ans_routes.py:203
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 10: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** tests/auth_server/conftest.py:214
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `secret_key="`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 11: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/federation_routes.py:215
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 12: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/schemas/peer_federation_schema.py:222
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 13: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** tests/auth_server/conftest.py:225
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `secret_key="`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 14: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/virtual_server_routes.py:226
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 15: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/ans_routes.py:235
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 16: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** tests/conftest.py:236
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `secret_key="`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 17: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** tests/e2e_agent_skills_test.py:250
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 18: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** scripts/test.py:282
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 19: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/ans_routes.py:285
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 20: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** registry/api/virtual_server_routes.py:317
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

*... and 92 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| State Confusion | 107 |
| Missing Input Validation | 98 |
| Prompt Injection | 45 |
| Excessive Permissions | 41 |
| Tool Poisoning | 10 |
| Private Key Exposure | 9 |
| Missing Harness | 7 |
| Cross Tool Escalation | 5 |
| Hardcoded Credentials | 2 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Private Key Exposure:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).
3. **Prompt Injection:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

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
