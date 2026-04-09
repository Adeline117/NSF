# Security Disclosure: juliusespinosa991/Solana-Ai-Trading

**Date:** 2026-04-08
**Reporter:** NSF AI Agent Security Research Team
**Severity:** critical (score: 100.0)
**Protocol:** openai
**Repository:** [juliusespinosa991/Solana-Ai-Trading](https://github.com/juliusespinosa991/Solana-Ai-Trading)
**Language:** TypeScript
**Stars:** 0

## Summary
We identified 265 potential security issues in juliusespinosa991/Solana-Ai-Trading
through automated static analysis of AI agent tool interfaces.
This analysis was part of a 138-repository systematic study across four
protocol families (MCP, OpenAI Function Calling, LangChain, Web3-native).

### Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 8 |
| High | 31 |
| Medium | 226 |

## Critical / High Severity Findings

### Finding 1: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/utils/keypair.ts:21
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 2: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions/jupiter/trade.ts:31
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 3: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/utils/keypair.ts:37
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 4: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions/solana/balance.ts:39
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 5: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions/solana/transfer.ts:48
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 6: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions/jupiter/trade.ts:50
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 7: hardcoded_credentials (CRITICAL)
- **Pattern:** HC-001
- **CWE:** CWE-798
- **File:** src/actions/jupiter/trade.ts:51
- **Description:** Hardcoded API key, password, secret, or token
- **Matched:** `Token: "So11111111111111111111111111111111111111112"`
- **Remediation:** Use environment variables or a secret manager. Never hardcode credentials in source code.

### Finding 8: private_key_exposure (CRITICAL)
- **Pattern:** PKE-003
- **CWE:** CWE-862
- **File:** src/tools/drift/drift.ts:57
- **Description:** Wallet signing without user confirmation
- **Matched:** `signTransaction`
- **Remediation:** Review and apply appropriate security controls for this finding category.

### Finding 9: cross_tool_escalation (HIGH)
- **Pattern:** CE-001
- **CWE:** CWE-284
- **File:** examples/agent-kit-langgraph/src/agents/manager.ts:11
- **Description:** Tool invokes other tools without permission check
- **Matched:** `chain.invoke(`
- **Remediation:** Implement per-tool access control. Tools should not be able to invoke other tools without explicit authorization.

### Finding 10: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/solana/transfer.ts:37
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 11: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/jupiter/stake_with_jup.ts:39
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 12: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/squads/create_proposal.ts:41
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendRawTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 13: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/squads/execute_proposal.ts:42
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendRawTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 14: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/lulo/lend.ts:42
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 15: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/squads/reject_proposal.ts:45
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendRawTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 16: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/squads/approve_proposal.ts:45
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendRawTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 17: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/solayer/stake_with_solayer.ts:46
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 18: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/solana/close_empty_token_accounts.ts:49
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 19: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/lulo/lulo_lend.ts:49
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

### Finding 20: tx_validation_missing (HIGH)
- **Pattern:** TV-001
- **CWE:** CWE-345
- **File:** src/tools/sns/register_domain.ts:50
- **Description:** Transaction sent without validation/whitelist check
- **Matched:** `sendTransaction(`
- **Remediation:** Validate transaction targets against a whitelist of approved contracts. Check all parameters before signing.

*... and 19 additional critical/high findings not shown.*

## Findings by Category

| Category | Count |
|----------|-------|
| Missing Input Validation | 226 |
| Tx Validation Missing | 30 |
| Hardcoded Credentials | 5 |
| Private Key Exposure | 3 |
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
