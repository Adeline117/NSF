# Understanding AI Agents: A Taxonomy for the Web3 Era

## Abstract
- **Problem:** No formal definition of "AI agent" exists that maps to observable behavior, particularly in the rapidly expanding Web3 and decentralized finance ecosystem.
- **Contribution:** A multi-dimensional taxonomy characterizing agents along three orthogonal axes -- autonomy level, environment type, and decision model -- yielding eight validated categories.
- **Validation:** On-chain behavioral mapping of real-world agents plus a Delphi expert survey establish that the taxonomy is exhaustive, mutually exclusive, and empirically grounded.
- **Impact:** Provides a shared vocabulary for HCI researchers, regulators, and practitioners to reason about AI agents interacting with blockchain infrastructure.

## 1. Introduction
- AI agents now autonomously manage billions of dollars in decentralized finance (DeFi) assets, from MEV extraction to portfolio rebalancing to cross-chain bridging.
- Despite their prevalence, no consensus definition of "AI agent" exists -- the term is applied indiscriminately to simple cron scripts, statistical arbitrage bots, and LLM-powered autonomous systems.
- This definitional ambiguity prevents measurement, comparison, and regulation of agent behavior.
- **RQ1:** What constitutes an AI agent as distinct from a bot, a script, or a human operating through software?
- **RQ2:** Can we map theoretical agent categories to observable on-chain behavioral signatures?
- **Contribution:** The first formal taxonomy of AI agents validated on blockchain transaction data, bridging classical AI agent theory with empirically observable on-chain behavior.
- Paper roadmap: taxonomy design (Section 3), behavioral mapping (Section 4), expert validation (Section 5), and implications for the broader research program (Section 6).

## 2. Related Work
- **2.1** Agent Definitions in AI/CS (Russell & Norvig, Wooldridge & Jennings, Franklin & Graesser)
- **2.2** Agent Definitions in HCI (Shneiderman levels of automation, human-AI interaction guidelines)
- **2.3** Blockchain Bots and MEV (Flashbots, Daian et al., Qin et al.)
- **2.4** On-chain AI Agents (Autonolas, AI16Z, Virtuals Protocol, Fetch.ai)
- **2.5** Gaps: No taxonomy bridges AI theory and on-chain observability
- *(Full text in related_work.md)*

## 3. Taxonomy Design
### 3.1 Design Methodology
- Literature synthesis across AI, HCI, and blockchain domains.
- Delphi expert consensus process: three rounds with 12 domain experts (AI researchers, blockchain security engineers, DeFi protocol designers).
- Iterative refinement until convergence on dimensions, levels, and category boundaries.

### 3.2 Dimension 1: Autonomy Level
- **NONE (0):** Pure script; deterministic, no adaptation. Executes a fixed sequence regardless of environmental state.
- **REACTIVE (1):** Responds to environmental events with fixed rules. Senses but does not learn.
- **ADAPTIVE (2):** Adjusts parameters based on environmental feedback. Learns within a bounded policy space.
- **PROACTIVE (3):** Plans and initiates actions independently. Maintains internal goals and models of the environment.
- **COLLABORATIVE (4):** Coordinates with other agents or human operators to achieve joint objectives.

### 3.3 Dimension 2: Environment Type
- **On-chain only:** Fully on-chain smart contract logic; no off-chain compute.
- **Hybrid (off-chain to on-chain):** Off-chain computation with on-chain execution -- the dominant pattern for bots and agents.
- **Cross-chain:** Operates across multiple blockchain networks, relaying state or assets.
- **Multi-modal:** Combines on-chain actions with off-chain data sources (APIs, LLM inference, social media).

### 3.4 Dimension 3: Decision Model
- **Deterministic:** If-then rules with no stochastic component.
- **Statistical:** Machine learning models producing probabilistic decisions.
- **LLM-driven:** Large language model reasoning as the primary decision engine.
- **Reinforcement learning:** Policy learned from reward signals; exploration-exploitation dynamics.
- **Hybrid:** Combination of multiple decision models (e.g., RL for strategy selection + deterministic for execution).

### 3.5 Category Definitions (8 categories with examples)
1. **Deterministic Script** -- (NONE, Hybrid, Deterministic) -- Cron-scheduled transfers, airdrop scripts.
2. **Simple Trading Bot** -- (REACTIVE, Hybrid, Deterministic) -- Grid bots, DCA bots.
3. **MEV Searcher** -- (ADAPTIVE, Hybrid, Statistical) -- Sandwich attackers, Flashbots searchers.
4. **Cross-Chain Bridge Agent** -- (ADAPTIVE, Cross-chain, Deterministic) -- LayerZero relayers, Wormhole guardians.
5. **RL Trading Agent** -- (ADAPTIVE, Hybrid, Reinforcement) -- RL market makers, policy-gradient arbitrageurs.
6. **DeFi Management Agent** -- (PROACTIVE, Hybrid, Hybrid) -- Autonolas agents, Yearn vaults.
7. **LLM-Powered Agent** -- (PROACTIVE, Multi-modal, LLM) -- AI16Z/ELIZA agents, Virtuals Protocol agents.
8. **Autonomous DAO Agent** -- (COLLABORATIVE, On-chain, Hybrid) -- Gnosis Safe modules, Governor executors.

### 3.6 Distinguishability Analysis
- Pairwise comparison of all 28 category pairs across 3 dimensions.
- Every pair is separable on at least 1 dimension; most pairs differ on 2+.
- Formal proof that no two categories share identical (autonomy, environment, decision_model) tuples.

## 4. Mapping to Observable Behavior
### 4.1 On-chain Feature Extraction
- Transaction timing distributions (periodicity, latency, burstiness).
- Calldata entropy and diversity metrics.
- Gas pricing strategies and priority fee patterns.
- Protocol interaction graphs and sequencing.
- Connects to Paper 1 (classification pipeline) for feature engineering.

### 4.2 Case Studies
- **Autonolas agents:** Multi-protocol interactions, adaptive rebalancing -- maps to DeFi Management Agent.
- **MEV bots (jaredfromsubway.eth):** Sub-block latency, bundle submissions -- maps to MEV Searcher.
- **AI16Z/ELIZA agents:** Variable latency, context-dependent actions -- maps to LLM-Powered Agent.
- **Cron transfer scripts:** Fixed calldata, perfect periodicity -- maps to Deterministic Script.

### 4.3 Feature-to-Category Mapping Validation
- Ground-truth labeling of 200 on-chain entities by 3 independent annotators.
- Inter-rater reliability (Cohen's kappa) and confusion matrix analysis.
- Feature importance ranking via mutual information with category labels.

## 5. Expert Validation
### 5.1 Delphi Study Design
- Three-round Delphi process: divergent elicitation, convergence, and final consensus.
- 12 experts spanning AI research, blockchain engineering, DeFi protocol design, and HCI.
- Each round: rate taxonomy completeness, mutual exclusivity, and practical utility on 7-point Likert scales.

### 5.2 Participant Recruitment
- Purposive sampling from (a) top-tier AI/HCI venues (CHI, NeurIPS, ICML), (b) blockchain security firms (Trail of Bits, OpenZeppelin), (c) DeFi protocols (Aave, Uniswap, Autonolas).
- IRB approval and informed consent per institutional requirements.

### 5.3 Results: Agreement on Definitions
- Final-round consensus scores (mean, SD) for each dimension and category.
- Areas of disagreement and how they were resolved.
- Comparison of expert mental models vs. taxonomy categories.

## 6. Discussion
### 6.1 Implications for Regulation
- Taxonomy provides a vocabulary for regulators to distinguish scripts (no regulation needed) from adaptive agents (may require disclosure) to collaborative agents (may need licensing).
- Maps to emerging EU AI Act risk categories.

### 6.2 Implications for Security (connects to Papers 1-3)
- Each taxonomy category implies distinct threat profiles.
- Adaptive and proactive agents can exhibit adversarial behaviors detectable by Paper 1's classifier.
- LLM-powered agents introduce prompt injection and hallucination risks (Paper 2 attack surface).
- Collaborative agents create multi-agent coordination risks (Paper 3 game theory).

### 6.3 Limitations
- Taxonomy is designed for the current Web3 agent landscape; new categories may emerge.
- On-chain observability is limited for agents operating primarily off-chain.
- Expert panel may not capture all stakeholder perspectives (e.g., end users, victims of MEV).
- Delphi consensus does not guarantee ground truth.

## 7. Conclusion
- Presented the first multi-dimensional taxonomy of AI agents grounded in on-chain observability.
- Eight categories spanning five autonomy levels, four environment types, and five decision models.
- Validated through behavioral mapping and expert consensus.
- Establishes the definitional foundation for the broader NSF research program (Papers 1-4).
- Future work: longitudinal study of category evolution as agent technology advances.

## References
*(See related_work.md for full citation details; final paper will use ACM reference format.)*
