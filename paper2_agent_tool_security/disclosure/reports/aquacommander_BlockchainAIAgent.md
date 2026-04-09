# Security Disclosure: aquacommander/BlockchainAIAgent

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 100.0)
**Protocol:** web3_native
**Repository:** [aquacommander/BlockchainAIAgent](https://github.com/aquacommander/BlockchainAIAgent)
**Language:** TypeScript
**Stars:** 1

## Summary
We identified 290 potential security issues in aquacommander/BlockchainAIAgent
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 16 |
| High | 34 |
| Medium | 240 |

## Critical / High Severity Findings

### Finding 1: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** Sui/src/hooks/useTransactionExecution.ts:3
- **Description:** Wallet signing without user confirmation
- **Matched:** `SignTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 2: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** Sui/src/constants.ts:6
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0x83076db923c7d6528e74fd7eafc568fc21e0f4e594cc97a0610ff7c1c8b16e6a"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 3: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** Sui/src/constants.ts:7
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0x3f5399e23c155e2ba8657386572b171a86a40344b251d32e300910165af9bce8"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 4: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** Sui/src/constants.ts:9
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0xe8320f49956af16e7cd93066edbca1caf6509b9a1dba66003021811abd5097bf"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 5: private_key_exposure (CRITICAL)
- **Pattern:** PKE-002
- **CWE:** CWE-200
- **File:** Sui/src/constants.ts:10
- **Description:** Raw private key hex string (64 hex chars)
- **Matched:** `"0x7f083161132e804985e833fc64b3ca5aa4093da5b6a9b41685278e001fd5e1f9"`
- **Remediation:** All signing operations must require explicit user confirmation with clear display of what is being signed.

### Finding 6: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** Sui/src/hooks/useTransactionExecution.ts:16
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 7: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** Sui/src/hooks/useTransactionExecution.ts:16
- **Description:** Wallet signing without user confirmation
- **Matched:** `SignTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 8: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** Solana/src/utils/keypair.ts:21
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 9: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** Sui/src/hooks/useTransactionExecution.ts:22
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 10: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** Solana/src/actions/jupiter/trade.ts:31
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 11: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** Solana/src/utils/keypair.ts:37
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 12: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** Solana/src/actions/solana/balance.ts:39
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 13: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** Solana/src/actions/solana/transfer.ts:48
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 14: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** Solana/src/actions/jupiter/trade.ts:50
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 15: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** Solana/src/actions/jupiter/trade.ts:51
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "So11111111111111111111111111111111111111112"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 16: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** Solana/src/tools/drift/drift.ts:57
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 17: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** Solana/examples/agent-kit-langgraph/src/agents/manager.ts:11
- **Description:** Tool invokes other tools without permission check
- **Matched:** `chain.invoke(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 18: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** Solana/src/tools/solana/transfer.ts:37
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 19: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** Solana/src/tools/jupiter/stake_with_jup.ts:39
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 20: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** Solana/src/tools/squads/create_proposal.ts:41
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendRawTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

*... and 30 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 238 |
| Tx Validation Missing | 33 |
| Private Key Exposure | 11 |
| Hardcoded Credentials | 5 |
| No Gas Limit | 2 |
| Cross Tool Escalation | 1 |

## Recommendations

1. **Private Key Exposure:** Review and apply appropriate security controls for this finding category.
2. **Hardcoded Credentials:** Use environment variables or a secret manager. Never hardcode credentials in source code.
3. **Cross Tool Escalation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.
4. **Tx Validation Missing:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

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
