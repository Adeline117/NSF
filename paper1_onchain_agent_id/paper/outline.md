# On-Chain AI Agent Identification and Security Posture: An Empirical Study

**Target venue:** The Web Conference (WWW)

## Abstract

We present the first systematic approach for identifying AI agent addresses from on-chain behavioral data and quantifying their security exposure on Ethereum. As autonomous AI agents increasingly operate on public blockchains -- managing DeFi positions, executing trades, and interacting with smart contracts -- they introduce novel risks that are invisible to existing monitoring tools because agents do not self-identify. We propose a 23-feature behavioral fingerprinting framework across four dimensions (temporal patterns, gas pricing behavior, interaction patterns, and approval security) that distinguishes AI agents from human users with high accuracy. Using ground truth labels from Autonolas, Flashbots, and AI agent launchpad protocols, we train a LightGBM ensemble classifier achieving [TBD] AUC-ROC on [TBD] labeled addresses. We then conduct a four-dimensional security audit of identified agents, revealing concerning patterns: [TBD]% maintain unlimited token approvals, [TBD]% of agent DEX trades are sandwiched (compared to [TBD]% for humans), and agent transaction revert rates are [TBD]x higher than human baselines. Our agent interaction graph reveals tightly clustered agent communities with systemic risk implications. We release our identification tool and labeled dataset to enable further research.

## 1. Introduction

- The rise of autonomous AI agents on public blockchains: from simple trading bots to complex multi-step DeFi agents (Autonolas, AI16Z ELIZA, Virtuals Protocol).
- Scale of the problem: $45M+ losses attributed to AI agent vulnerabilities in 2025-2026, including approval exploits, oracle manipulation, and cascading liquidation failures.
- Fundamental gap: agents do not self-identify on-chain. There is no protocol-level distinction between a human-operated wallet and an AI-controlled one.
- Why identification matters: (1) security auditors cannot assess agent-specific risks, (2) protocol designers cannot account for agent behavior, (3) regulators lack visibility into automated financial activity.
- **RQ1:** Can we identify AI agents from on-chain behavioral features alone?
- **RQ2:** What is the security posture of identified AI agents, and how does it compare to human users?
- Contributions:
  1. A 23-feature behavioral fingerprinting framework for agent identification.
  2. A labeled dataset of confirmed agent and human addresses with provenance.
  3. A multi-model classifier achieving high accuracy on agent identification.
  4. A four-dimensional security audit revealing systemic risks in agent operations.
  5. An open-source identification and audit toolkit.

## 2. Related Work

- 2.1 Bot Detection on Blockchains
  - Chen et al. (2020): Ponzi scheme detection on Ethereum.
  - Victor & Weintraud (2021): Detecting and quantifying wash trading on DEXs.
  - Existing bot detection focuses on specific malicious behaviors, not general agent identification.
- 2.2 MEV and Automated Trading
  - Daian et al. (2020): Flash Boys 2.0 -- frontrunning in decentralized exchanges.
  - Weintraub et al. (2022): A flash(bot) in the pan -- measuring MEV on Flashbots.
  - Qin et al. (2022): Quantifying blockchain extractable value.
  - These works study the effects of bots but do not build general-purpose identifiers.
- 2.3 AI Agent Platforms
  - Autonolas: Autonomous agent service framework with on-chain registry.
  - AI16Z ELIZA: Open-source AI agent framework for crypto.
  - Virtuals Protocol: Agent tokenization and launchpad.
  - These platforms create agents but do not provide cross-platform identification.
- 2.4 Blockchain Security Analysis
  - Zhou et al. (2023): SoK on decentralized finance attacks.
  - Wen et al. (2023): DeFi security and composability risks.
  - Security analyses focus on protocol-level vulnerabilities, not agent-specific risks.
- **Gap:** No prior work systematically identifies AI agents from on-chain data and audits their security posture.

## 3. Methodology

### 3.1 Data Collection

- **Source:** Ethereum mainnet via Etherscan API and Dune Analytics.
- **Time period:** [TBD, e.g., January 2024 -- March 2026].
- **Scope:** Normal transactions, internal transactions, ERC-20 transfers, token approvals.
- **Scale:** [TBD] addresses, [TBD] total transactions.
- Ethical considerations: all data is publicly available on-chain.

### 3.2 Ground Truth Labeling

- **Confirmed Agents (positive class):**
  - Autonolas ServiceRegistry: addresses registered as autonomous service operators.
  - Flashbots/EigenPhi MEV bot lists: known MEV searcher addresses.
  - AI agent launchpad interactions: addresses deploying or operating agents via AI16Z or Virtuals.
- **Confirmed Humans (negative class):**
  - ENS-named addresses with linked social accounts (Twitter, GitHub).
  - Known institutional wallets (Arkham Intelligence, Nansen labels).
  - Addresses exhibiting clear circadian transaction patterns.
- **Exclusions:** Addresses that cannot be labeled with high confidence are marked UNKNOWN and excluded from training.
- **Label quality:** Inter-rater agreement on a sample of [TBD] addresses.

### 3.3 Feature Engineering

Four groups, 23 features total:

**Group 1: Temporal Features (7)**
| Feature | Intuition |
|---------|-----------|
| tx_interval_mean | Agents have shorter, more regular intervals |
| tx_interval_std | Agents show lower variance |
| tx_interval_skewness | Human intervals are right-skewed (bursts of activity) |
| active_hour_entropy | Agents operate 24/7 (high entropy); humans show circadian rhythm |
| night_activity_ratio | Agents active during UTC 0-6; humans are not |
| weekend_ratio | Agents maintain steady weekend activity |
| burst_frequency | Agents produce sub-10-second transaction bursts |

**Group 2: Gas Behavior (6)**
| Feature | Intuition |
|---------|-----------|
| gas_price_round_number_ratio | Wallets/humans prefer round gas prices |
| gas_price_trailing_zeros_mean | Algorithmic gas pricing has fewer trailing zeros |
| gas_limit_precision | Agents estimate gas limits more precisely |
| gas_price_cv | Agents show lower gas price variation (algorithmic) |
| eip1559_priority_fee_precision | Agents set precise priority fees |
| gas_price_nonce_correlation | Algorithmic gas adjustment across nonces |

**Group 3: Interaction Patterns (5)**
| Feature | Intuition |
|---------|-----------|
| unique_contracts_ratio | Agents interact with fewer unique contracts |
| top_contract_concentration | Agent interactions are concentrated (high HHI) |
| method_id_diversity | Agents call fewer distinct functions |
| contract_to_eoa_ratio | Agents primarily call contracts, not EOAs |
| sequential_pattern_score | Agents repeat action sequences (high n-gram overlap) |

**Group 4: Approval & Security (5)**
| Feature | Intuition |
|---------|-----------|
| unlimited_approve_ratio | Agents often set MaxUint256 approvals for convenience |
| approve_revoke_ratio | Agents rarely revoke approvals |
| unverified_contract_approve_ratio | Agents may approve unverified contracts |
| multi_protocol_interaction_count | Agents interact with many DeFi protocols |
| flash_loan_usage | Flash loan usage is predominantly agent-driven |

### 3.4 Classification Model

- **Primary model:** LightGBM with 500 estimators, max_depth=6.
- **Baselines:** Random Forest, Logistic Regression.
- **Evaluation:** 5-fold stratified cross-validation.
- **Metrics:** AUC-ROC, Precision@K, Recall@K, F1-score.
- **Interpretability:** SHAP feature importance analysis.
- **Calibration:** Calibration curves to assess prediction reliability.
- **Ablation:** Feature group ablation study to quantify each group's contribution.

### 3.5 Security Audit Framework

Four audit dimensions applied to each identified agent:

1. **Permission Exposure:** Active unlimited approvals, approval age, unverified targets.
2. **Agent Network Topology:** Directed graph of inter-agent transfers/calls, centrality metrics, community detection.
3. **MEV Exposure:** Sandwich attack rate on agent DEX trades vs. human baseline.
4. **Failure Analysis:** Transaction revert rate and reason classification, retry patterns.

## 4. Results

### 4.1 Agent Identification Performance

- Classification accuracy across all three models.
- AUC-ROC curves.
- Precision@K analysis (useful for deployment: identify top-K most likely agents).
- Comparison table: LightGBM vs. RF vs. LR.

### 4.2 Feature Importance Analysis

- SHAP summary plot: which features matter most?
- Feature group ablation: temporal > gas > interaction > approval?
- Per-feature analysis with violin plots.

### 4.3 Permission Exposure Findings

- Distribution of unlimited approvals among agents vs. humans.
- Approval age analysis: how long do agents leave approvals active?
- Value at risk from outstanding agent approvals.

### 4.4 MEV Exposure Findings

- Agent vs. human sandwich attack rates.
- Which agent types are most vulnerable to MEV?
- Correlation between agent activity patterns and MEV exposure.

### 4.5 Failure Rate Analysis

- Agent vs. human revert rates.
- Revert reason distribution.
- Retry behavior: do agents automatically retry after failures?

### 4.6 Agent Network Topology

- Network visualization of agent clusters.
- Centrality distribution: identification of systemic risk nodes.
- Community structure: are agent clusters protocol-specific?

## 5. Discussion

- **Implications for protocol designers:** agents need first-class support (rate limiting, gas optimization, approval management).
- **Implications for security auditors:** agent-specific audit frameworks.
- **Implications for regulators:** automated financial activity is invisible without identification tools.
- **Limitations:**
  - Ground truth is inherently incomplete (unknown agents exist).
  - Etherscan rate limits constrain scale.
  - Feature drift: agent behavior evolves as platforms update.
  - Adversarial evasion: sophisticated agents could mimic human behavior.
- **Future work:**
  - Real-time agent identification system.
  - Cross-chain agent tracking (Base, Arbitrum, Solana).
  - Agent capability taxonomy integration (Paper 0).

## 6. Conclusion

- Summary of contributions: identification framework, security audit, empirical findings.
- Call to action: protocols should implement agent-aware security measures.
- Open-source release: toolkit and labeled dataset.
