# Security Disclosure: affaan-m/agentshield

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 91.1)
**Protocol:** mcp
**Repository:** [affaan-m/agentshield](https://github.com/affaan-m/agentshield)
**Language:** TypeScript
**Stars:** 331

## Summary
We identified 81 potential security issues in affaan-m/agentshield
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 12 |
| High | 35 |
| Medium | 34 |

## Critical / High Severity Findings

### Finding 1: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/sandbox/executor.ts:51
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY: "CANARY_anthropic_sk-ant-fake12345"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/sandbox/executor.ts:52
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `API_KEY: "CANARY_openai_sk-fake67890"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/sandbox/executor.ts:53
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `TOKEN: "CANARY_github_ghp_fake11111"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 4: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/corpus/vulnerable-configs.ts:57
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `TOKEN: "ghp_realtoken123456789012345678901234567"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 5: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** src/sandbox/executor.ts:57
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `SECRET_KEY: "`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 6: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/sandbox/executor.ts:58
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `TOKEN: "CANARY_slack_xoxb-fake55555"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 7: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/sandbox/executor.ts:59
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `TOKEN: "CANARY_npm_npm_fake66666"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 8: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/rules/mcp.test.ts:163
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `AUTH_TOKEN: "hardcoded-secret-value-here"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 9: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/rules/mcp.test.ts:202
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `TOKEN: "YOUR_GITHUB_PAT_HERE"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 10: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** tests/rules/mcp.test.ts:220
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `TOKEN: "real-secret-token-value"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 11: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** tests/rules/secrets.test.ts:286
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `SECRET_KEY="`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 12: private_key_exposure (CRITICAL)
- **Pattern:** PKE-001
- **CWE:** CWE-200
- **File:** tests/rules/secrets.test.ts:485
- **Description:** Private key, mnemonic, or seed phrase in code or as parameter
- **Matched:** `SECRET_KEY: "`
- **Remediation:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).

### Finding 13: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** src/rules/prompt-defense.ts:26
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description:
      "Prompt should state that user content cannot override, ignore, or modify higher-...`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 14: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** src/miniclaw/router.ts:46
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "System prompt override: '`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 15: unlimited_approval (HIGH)
- **Pattern:** UA-001
- **CWE:** CWE-250
- **File:** src/injection/payloads.ts:50
- **Description:** Unlimited token approval (MaxUint256 or equivalent)
- **Matched:** `unlimited`
- **Remediation:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

### Finding 16: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** src/miniclaw/router.ts:58
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "System prompt injection: '`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 17: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** src/miniclaw/router.ts:62
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "Direct system prompt override attempt"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 18: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** src/rules/prompt-defense.ts:147
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description:
      "Checks whether system prompt files contain defensive instructions against common...`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 19: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** src/rules/mcp-tool-poisoning.ts:153
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "Instruction to leak system prompt or conversation context"`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

### Finding 20: tool_poisoning (HIGH)
- **Pattern:** TP-001
- **CWE:** CWE-913
- **File:** src/rules/mcp-tool-poisoning.ts:163
- **Description:** Tool/function description contains directive instructions (always/must/never/ignore)
- **Matched:** `description: "Attempt to override the agent'`
- **Remediation:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.

*... and 27 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Tool Poisoning | 35 |
| Missing Input Validation | 25 |
| Hardcoded Credentials | 9 |
| State Confusion | 8 |
| Private Key Exposure | 3 |
| Unlimited Approval | 1 |

## Recommendations

1. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
2. **Private Key Exposure:** Never accept private keys as parameters. Use a secure signing service (e.g., KMS, hardware wallet, session keys with limited scope).
3. **Tool Poisoning:** Remove directive language from tool descriptions. Descriptions should be declarative, not imperative.
4. **Unlimited Approval:** Cap token approvals to the exact amount needed for the transaction. Never use unlimited approvals.

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
