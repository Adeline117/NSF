# Security Disclosure: aipotheosis-labs/aci

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 87.3)
**Protocol:** openai
**Repository:** [aipotheosis-labs/aci](https://github.com/aipotheosis-labs/aci)
**Language:** Python
**Stars:** 4751

## Summary
We identified 67 potential security issues in aipotheosis-labs/aci
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 17 |
| High | 1 |
| Medium | 49 |

## Critical / High Severity Findings

### Finding 1: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** .github/scripts/integration_pr_review.py:19
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `prompt = f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/routes/linked_acounts/test_accounts_link_api_key.py:25
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `api_key="test_linked_account_api_key"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** backend/aci/common/tests/crud/test_linked_accounts.py:31
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `secret_key="`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 4: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** backend/aci/common/tests/crud/test_linked_accounts.py:35
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `secret_key="`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 5: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** backend/aci/common/tests/crud/test_linked_accounts.py:37
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `secret_key="`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 6: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/routes/linked_acounts/test_accounts_link_oauth2.py:62
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `secret="custom_client_secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 7: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** backend/aci/server/tests/routes/projects/test_projects_agents.py:62
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 8: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/routes/linked_acounts/test_accounts_link_api_key.py:68
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `api_key="test_linked_account_api_key"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 9: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** backend/aci/server/tests/routes/projects/test_projects_agents.py:73
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 10: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/crud/custom_sql_types/test_encrypted_security_schemes.py:85
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `secret = "app_config_secret_value"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 11: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/conftest.py:113
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="dummy_access_token_2"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 12: prompt_injection (CRITICAL)
- **Pattern:** PI-001
- **CWE:** CWE-74
- **File:** backend/evals/synthetic_intent_generator.py:127
- **Description:** User input interpolated into description/prompt/system message
- **Matched:** `description=f"`
- **Remediation:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.

### Finding 13: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/routes/linked_acounts/test_accounts_link_oauth2.py:230
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `secret="custom_client_secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 14: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** backend/aci/server/tests/conftest.py:561
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `secret_key="`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 15: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/conftest.py:569
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `secret="dummy_linked_account_oauth2_credentials_client_secret"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 16: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/conftest.py:571
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="dummy_linked_account_oauth2_credentials_access_token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 17: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** backend/aci/server/tests/conftest.py:574
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token="dummy_linked_account_oauth2_credentials_refresh_token"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 18: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** backend/aci/server/app_connectors/frontend_qa_agent.py:414
- **Description:** Tool invokes other tools without permission check
- **Matched:** `agent.run(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 47 |
| Hardcoded Credentials | 9 |
| Prompt Injection | 4 |
| Private Key Exposure | 4 |
| State Confusion | 1 |
| No Output Validation | 1 |
| Cross Tool Escalation | 1 |

## Recommendations

1. **Prompt Injection:** Never interpolate user input into tool descriptions or system prompts. Pass user data as structured parameters.
2. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
3. **Private Key Exposure:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).
4. **Cross Tool Escalation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

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
