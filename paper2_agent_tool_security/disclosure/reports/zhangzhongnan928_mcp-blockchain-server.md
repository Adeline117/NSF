# Security Vulnerability Disclosure Report

**From:** NSF Research Project -- AI Agent Tool Interface Security
**Date:** 2026-04-07
**Subject:** Security vulnerabilities found in zhangzhongnan928/mcp-blockchain-server

## Summary

During our academic research on AI agent tool interface security, we identified **35** potential security vulnerabilities in your **zhangzhongnan928/mcp-blockchain-server** project (35 from static analysis).

- **Risk score:** 76.9/100
- **Risk rating:** high
- **Protocol:** mcp
- **GitHub stars:** 10

## Severity Breakdown

| Severity | Count |
|----------|-------|
| Critical | 5 |
| High | 2 |
| Medium | 28 |

## Vulnerability Categories

| Category | Count |
|----------|-------|
| Missing Input Validation | 27 |
| Private Key Exposure | 5 |
| Tx Validation Missing | 1 |
| Mev Exposure | 1 |
| Missing Harness | 1 |

## Top Findings (Most Severe)

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** web/src/components/TransactionReview.tsx
- **Line:** 58
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      56:   const { txId } = useParams<{ txId: string }>();
      57:   const navigate = useNavigate();
  >>> 58:   const { account, connect, signTransaction, chainId, switchChain } = useWallet();
      59: 
      60:   const [transaction, setTransaction] = useState<Transaction | null>(null);
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** web/src/components/TransactionReview.tsx
- **Line:** 135
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      133: 
      134:       // Sign transaction
  >>> 135:       const signedTx = await signTransaction(tx);
      136: 
      137:       // Submit signed transaction
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** web/src/hooks/useWallet.ts
- **Line:** 201
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      199: 
      200:   // Sign transaction
  >>> 201:   const signTransaction = async (txRequest: TransactionRequest) => {
      202:     if (!walletState.signer) {
      203:       throw new Error('Wallet not connected');
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** web/src/hooks/useWallet.ts
- **Line:** 208
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      206:     try {
      207:       // Create transaction
  >>> 208:       const tx = await walletState.signer.signTransaction(txRequest);
      209:       return tx;
      210:     } catch (error) {
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

### PKE-003: Private Key Exposure (CRITICAL)
- **File:** web/src/hooks/useWallet.ts
- **Line:** 248
- **Description:** Wallet signing without user confirmation
- **CWE:** CWE-862
- **Code context:**
  ```
      246:     disconnect,
      247:     switchChain,
  >>> 248:     signTransaction,
      249:     sendTransaction,
      250:   };
  ```
- **Impact:** An attacker could steal private keys, gaining full control of associated wallets and funds.
- **Recommendation:** Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

## Remediation Recommendations

### Mev Exposure
Use private transaction pools (e.g., Flashbots Protect) for sensitive transactions. Implement commit-reveal schemes where applicable.

### Missing Harness
Add integration tests that exercise tool interfaces with adversarial inputs. Include negative test cases for boundary conditions.

### Missing Input Validation
Validate Ethereum addresses with isAddress(). Validate amounts are positive and within expected bounds. Reject unexpected or extra parameters by setting additionalProperties: false in JSON Schema definitions.

### Private Key Exposure
Never accept private keys as tool parameters. Use session keys or hardware wallet signing. If the server must sign, use a secure enclave or environment-variable-only key loading with strict file permissions.

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
