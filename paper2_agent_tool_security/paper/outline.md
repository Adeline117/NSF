# AI Agent Tool Interface Security: A Cross-Protocol Empirical Analysis

**Target:** IEEE S&P or USENIX Security 2027
**Page limit:** 18 pages (S&P) / 18 pages (USENIX)

---

## Abstract

AI agents increasingly rely on external tools for real-world actions --
signing blockchain transactions, querying databases, executing code, and
managing infrastructure. Multiple protocols have emerged to mediate these
interactions: Anthropic's Model Context Protocol (MCP), OpenAI Function
Calling, LangChain/agent framework tools, and Web3-native smart-contract
modules. Despite the critical security implications of granting LLMs
access to privileged operations, no systematic cross-protocol security
analysis exists. We present the first such analysis, examining 200+ tool
servers across all four protocol families. Using a combination of
automated static analysis (36 vulnerability patterns, 12 CWE mappings)
and targeted dynamic testing (50 servers), we find that 18 out of 20
vulnerability categories manifest across protocol boundaries, with 38.9%
of vulnerabilities being protocol-agnostic. We identify five attack
surfaces common to all protocols, four Web3-specific attack vectors with
irreversible financial impact, and a pervasive absence of security
harnesses. We propose six harness requirements and evaluate their
adoption. Our tools and dataset are released as open source.

## 1. Introduction

- The AI agent paradigm: LLMs as reasoning engines that invoke external
  tools to act in the world
- The tool interface as the new attack surface: agents inherit the
  vulnerabilities of every tool they can invoke
- Multiple competing protocols: MCP, OpenAI Function Calling, LangChain,
  Web3-native (ERC-4337, Safe modules, ERC-7579)
- Gap: existing security analyses are protocol-specific (MCP-only) or
  limited to prompt injection; no cross-protocol empirical study exists
- Key finding preview: 38.9% of vulnerabilities are protocol-agnostic,
  indicating systemic design flaws, not implementation bugs
- Contributions:
  1. Unified vulnerability taxonomy (20 categories) with CWE mapping
  2. First cross-protocol empirical analysis (200+ servers, 4 protocols)
  3. Five attack surfaces and four Web3-specific vectors
  4. Harness gap analysis and six mitigation requirements
  5. Open-source static analyzer and dataset

## 2. Background & Threat Model

### 2.1 AI Agent Tool Protocols
- MCP: stdio/SSE transport, tool/resource/prompt primitives, sampling
- OpenAI Function Calling: JSON Schema function definitions in chat API
- LangChain: Python @tool decorators, AgentExecutor, BaseTool
- Web3-native: Safe modules, ERC-4337 plugins, ERC-7579 modules
- Comparison table: transport, schema, execution model, trust assumptions

### 2.2 System Model
- Agent, protocol, tool server, harness, skill -- five components
- Trust boundaries (B1-B7)

### 2.3 Attacker Model
- A1: Malicious tool provider (controls server code)
- A2: Prompt injection attacker (controls data the agent processes)
- A3: Man-in-the-middle (intercepts protocol messages)
- Prevalence estimates from GitHub/npm analysis

### 2.4 Attack Surfaces
- S1: Tool definition surface (descriptions, schemas, defaults)
- S2: Input construction surface (parameter validation, type safety)
- S3: Execution surface (sandboxing, privilege, resource limits)
- S4: Output handling surface (result sanitization, injection)
- S5: Cross-tool surface (information flow, state sharing, escalation)

### 2.5 Web3-Specific Attack Vectors
- W1: Transaction construction injection (irreversible loss)
- W2: Approval hijacking (MaxUint256 as "required step")
- W3: Private key / session key extraction
- W4: MEV exploitation (slippage, frontrunning)

## 3. Methodology

### 3.1 Server Enumeration & Selection
- GitHub search across 32 queries, 4 protocol families
- Selection criteria: public, non-archived, source available
- Deduplication and protocol classification
- Final dataset: N repos (breakdown by protocol)

### 3.2 Protocol Classification
- Automated detection via import analysis, dependency scanning,
  file-level indicators
- Manual validation on 50-repo sample
- Classification confidence scoring

### 3.3 Static Analysis Framework
- 36 regex-based vulnerability patterns (from pilot)
- Mapped to 20 vulnerability categories, 12 CWE IDs, 5 attack surfaces
- False positive filtering: comment detection, test exclusion, context
  heuristics
- Risk scoring: severity-weighted, category-multiplied, harness-adjusted
- Harness detection: 6 feature categories

### 3.4 Dynamic Testing (Top 50 Servers)
- Selection: top 50 by risk score from static analysis
- Test categories:
  - Tool poisoning: inject directives, measure agent compliance
  - Prompt injection via tool output: inject payloads in results
  - Cross-tool privilege escalation: low-priv tool influences high-priv
  - Private key extraction: test key leakage vectors
  - Transaction construction attacks: parameter manipulation
- Environment: local testnet (Hardhat/Anvil), sandboxed agent
- Metrics: attack success rate, detection rate, time to exploit

### 3.5 Risk Scoring
- Per-server risk score (0-100)
- Calibration against known-vulnerable and known-secure servers
- Validation against manual expert assessment (20-repo sample)

## 4. Static Analysis Results

### 4.1 Dataset Overview
- N total servers, breakdown by protocol and language
- Stars, recency, activity level distributions
- Geographic and organizational distribution

### 4.2 Per-Protocol Vulnerability Distribution
- Table: vulnerability category x protocol family (count + percentage)
- Key protocols compared: MCP vs OpenAI vs LangChain vs Web3-native
- Statistical tests for distribution differences (chi-squared)

### 4.3 Cross-Protocol Systemic Vulnerabilities
- Categories that appear in ALL protocols (systemic)
- Categories that are protocol-specific
- Percentage of systemic vs protocol-specific findings
- Argument: systemic vulnerabilities indicate design-level problems

### 4.4 Web3-Specific Findings
- Private key exposure rates by protocol
- Unlimited approval prevalence
- Transaction validation gaps
- MEV protection gaps
- Comparison: Web3 tool servers vs general-purpose tool servers

### 4.5 Harness Gap Analysis
- Percentage of servers with any harness features
- Feature adoption breakdown (6 categories)
- Correlation: harness presence vs risk score
- Protocol comparison: which protocol ecosystems are most/least hardened

## 5. Dynamic Testing Results

### 5.1 Tool Poisoning
- Success rate of directive injection across protocols
- Which LLMs are most/least susceptible
- Effective vs ineffective poisoning techniques

### 5.2 Prompt Injection via Tool Output
- Success rate by injection technique
- Impact: what actions can be triggered
- Existing defenses and their effectiveness

### 5.3 Cross-Tool Privilege Escalation
- Confused deputy scenarios: data tool -> signing tool
- Chain escalation: tool A invokes tool B
- Shared state exploitation

### 5.4 Private Key Extraction
- Direct extraction (key as parameter)
- Indirect extraction (injection -> key in response)
- Session key scope analysis

### 5.5 Transaction Construction Attacks
- Address substitution success rate
- Amount manipulation success rate
- Approval injection success rate
- MEV exposure quantification

## 6. Risk Scoring & Remediation

- Risk score distribution across all servers
- Correlation analysis: risk factors
  - Protocol family vs risk
  - Server age vs risk
  - Stars/popularity vs risk
  - Language (TS vs Python vs Solidity) vs risk
- Six harness requirements with implementation guidance
- Cost-benefit analysis of each mitigation
- Responsible disclosure results

## 7. Discussion

- Why are systemic vulnerabilities so prevalent?
  - Protocol designers focused on functionality, not security
  - Developers copy patterns from documentation/tutorials
  - No security-by-default in any protocol
- The harness gap: why doesn't anyone build security wrappers?
  - Developer incentives analysis
  - Usability vs security tradeoff
- Limitations:
  - Static analysis false positives/negatives
  - Dynamic testing limited to 50 servers
  - GitHub-only dataset (not npm, PyPI registries)
  - Web3 focus may not generalize to other verticals
- Future work:
  - Registry-level security scanning (npm, PyPI)
  - Protocol-level security primitives (standardization)
  - Automated harness generation
  - Longitudinal study of vulnerability trends

## 8. Conclusion

- First cross-protocol security analysis of AI agent tool interfaces
- 18/20 vulnerability categories manifest across protocol boundaries
- 38.9% of vulnerabilities are protocol-agnostic (systemic)
- Pervasive absence of security harnesses
- Call to action: protocol designers must build security into the protocol
  layer, not leave it to individual tool developers

## References

[Expected: 45-55 references]
- Protocol specifications (MCP, OpenAI, LangChain, ERC-4337, ERC-7579)
- LLM security (Greshake 2023, Zhan 2024, OWASP LLM Top 10)
- Smart contract security (Atzei 2017, Chen 2020, OWASP SC Top 10)
- API security (OWASP API Top 10)
- Agent security surveys (arxiv 2601.04583)
- CWE references

## Appendices

### A. Full Vulnerability Pattern Catalog
- All 36 patterns with regex, CWE, and examples

### B. Server Catalog
- Complete list of analyzed servers with protocol classification

### C. Risk Score Calibration
- Scoring methodology details and validation results

### D. Ethical Considerations
- IRB status, responsible disclosure timeline, no-harm policy
