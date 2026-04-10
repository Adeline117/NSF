# Understanding the On-Chain AI Agent: A Four-Condition Definition and Eight-Category Taxonomy Validated on 2,744 Ethereum Addresses

**Paper 0 — CHI 2026 Submission**

---

## Abstract

AI agents manage billions of dollars on decentralized-finance (DeFi) rails, yet "AI agent" still lacks a definition that maps to chain-observable behavior. Existing definitions either over-count (Russell and Norvig's perceive-and-act admits thermometers) or under-operationalize (Wooldridge and Jennings's four attributes cannot be checked against transaction data). We propose a four-condition definition — on-chain actuation (C1), environmental perception (C2), autonomous decision-making (C3), and adaptiveness (C4) — and an eight-category taxonomy spanning deterministic scripts through collaborative DAO agents. We validate both on 2,744 agent addresses mined from the Autonolas, Fetch.ai, AI Arena, Flashbots-searcher, Gnosis Safe, Stargate bridge, and multi-protocol RL ecosystems using three complementary methods: a transparent rule-based taxonomy projection, a K-Means cluster sweep for k in [3, 15], and an eight-class gradient-boosting classifier. All eight categories populate the current dataset (DeFi Management Agent 60.8%, Deterministic Script 24.3%, Simple Trading Bot 4.7%, Cross-Chain Bridge Agent 3.5%, LLM-Powered Agent 2.6%, MEV Searcher 2.0%, Autonomous DAO Agent 1.2%, RL Trading Agent 0.9%). The eight-class classifier reaches 92.2% accuracy and macro-F1 0.58; the original five-class classifier achieves 97.4% accuracy, and expanding to eight classes drops accuracy by 5 points because the three new categories are small and overlap with DeFi Management Agent (DAO F1=0.14, Bridge F1=0.31, RL F1=0.07). Two non-obvious findings emerge: behavioral silhouette is maximized at k=3, not k=8, so the semantic taxonomy is behaviorally over-split at the 23-feature level; and the three newly populated categories confirm that semantic distinctions require richer features beyond the current 23. We discuss implications for HCI, governance, and the design of the feature sets that downstream agent identification, security, and Sybil-detection studies should share.

**Keywords:** AI agents; Web3; decentralized finance; agent definition; behavioral taxonomy; on-chain analytics; cluster validation; multi-class classification.

---

## 1. Introduction

AI agents have quietly become first-class participants in the decentralized-finance (DeFi) economy. From Maximal Extractable Value (MEV) bots harvesting arbitrage opportunities to Autonolas services rebalancing yield-farming positions, from cross-chain relayers bridging assets to DAO execution modules settling governance decisions, software entities now autonomously manage many billions of dollars of crypto assets. Flashbots alone reports that MEV extraction on Ethereum has surpassed USD 600 million since 2020, almost entirely performed by automated programs.

Yet "AI agent" is applied indiscriminately to a scheduled payout cron, a grid-trading bot, and an LLM-driven policy engine that re-reasons its DeFi positions every block. This terminological flattening obstructs three things the HCI and systems communities care about: **measurement** (we cannot count what we cannot define), **comparison** (we cannot say one class of agent behaves differently from another if we do not distinguish them), and **governance** (a regulator cannot impose differentiated disclosure requirements on "the agent" when the same word names a cron job and an LLM).

Classical AI offers definitional scaffolding but not measurable criteria. Russell and Norvig's [2020] perceive-and-act schema is broad enough to include thermometers. Wooldridge and Jennings's [1995] autonomy / social ability / reactivity / pro-activeness tetrad is implementation-neutral and cannot be checked against public transaction data. Franklin and Graesser [1996] enumerate properties without committing to necessity. The blockchain bot literature [Daian et al. 2020; Qin et al. 2022; Torres et al. 2021] documents behavior in exquisite detail but rarely links its empirical findings back to formal agent theory, lumping "anything automated" under "bot".

This paper proposes the first definition of "on-chain AI agent" that is simultaneously formal, operationalizable, and empirically validated. Concretely, we answer two research questions:

**RQ1.** What formal conditions distinguish an AI agent from a bot, a script, or a human user operating through software, *when the only observation channel is the blockchain itself*?

**RQ2.** Does a semantically motivated eight-category taxonomy of on-chain agents survive contact with real on-chain data — are all eight categories distinguishable from 23 behavioral features?

Our contributions are:

1. **A four-condition definition (C1-C4).** On-chain actuation, environmental perception, autonomous decision-making, and adaptiveness are proposed as *necessary and jointly sufficient* for on-chain AI agency. Each condition is justified by a counterexample-based non-redundancy argument (Section 3) and mapped to an operationalization in transaction-level features.

2. **An eight-category taxonomy.** Agents are classified along three orthogonal dimensions — autonomy level (5 ranks), environment type (4 types), decision model (5 types) — yielding eight disjoint categories that cover both agents (5 classes) and non-agent baselines (2 classes) plus one boundary class (cross-chain bridges).

3. **Three-method empirical validation on 2,744 Ethereum addresses.** A rule-based taxonomy projection, a K-Means cluster sweep, and a multi-class supervised classifier jointly stress-test the taxonomy. All eight categories populate the dataset following a targeted Phase 2 mining pass. The eight-class classifier achieves 92.2% accuracy and macro-F1 0.58; the original five-class classifier achieves 97.4% accuracy and macro-F1 0.87.

4. **Two honest findings about where the taxonomy breaks.** (a) The behavioral silhouette is maximized at k=3, not k=8, so the eight categories are semantically coherent but *behaviorally over-split* at the current 23-feature level. (b) Expanding from five to eight classes drops accuracy by 5 points (97.4% to 92.2%); the three newly populated categories — DAO (F1=0.14), Bridge (F1=0.31), RL (F1=0.07) — are small and overlap with DeFi Management Agent. We argue this is a *measurement* problem (new features are needed, e.g. the eight AI features from Paper 3), not a *taxonomic* problem.

5. **Transparent threats to validity.** The taxonomy projection rules reuse eight of the 23 features, creating a known partial circularity that inflates the classifier's apparent accuracy. We quantify this risk and discuss its implications for downstream use.

The paper proceeds as follows. Section 2 reviews related definitions. Section 3 formalizes C1-C4 and argues for their non-redundancy. Section 4 presents the on-chain empirical validation. Section 5 discusses the two key findings. Section 6 treats limitations, and Section 7 concludes.

---

## 2. Background and Related Work

Four strands of prior work inform our framing: classical AI-agent definitions, HCI automation-level work, blockchain bot and MEV empirical work, and emerging on-chain AI-agent platforms. Table 1 positions our definition against the major prior proposals.

**Table 1. Agent definitions and their coverage of the four conditions used in this paper.** Check-marks indicate that the prior work explicitly or implicitly includes the listed condition; dashes indicate it is omitted or assumed. *C1 = on-chain actuation; C2 = environmental perception; C3 = autonomous decision-making; C4 = adaptiveness.*

| Work | Year | Discipline | C1 | C2 | C3 | C4 | Web3 applicable? |
|------|------|------------|:--:|:--:|:--:|:--:|------------------|
| Russell & Norvig [2020] | 2020 | AI textbook | — | ✓ | partial | — | too broad (thermometers qualify) |
| Wooldridge & Jennings [1995] | 1995 | Multi-agent systems | — | ✓ | ✓ | ✓ | no actuation spec |
| Franklin & Graesser [1996] | 1996 | Taxonomy | — | ✓ | ✓ | ✓ | property enumeration, not necessity |
| Maes [1995] | 1995 | Artificial life | — | ✓ | ✓ | ✓ | no persistence criterion |
| Shoham [1993] | 1993 | Agent programming | — | — | ✓ | — | requires BDI layer |
| Dennett [1987] | 1987 | Philosophy | — | ✓ | ✓ | — | observer-dependent |
| Park et al. [2023] | 2023 | NLP / HCI | — | ✓ | ✓ | ✓ | text simulacra, not Web3 |
| Chan et al. [2023] | 2023 | AI safety | — | ✓ | ✓ | — | counterfactual influence |
| Shavit et al. [2023] | 2023 | Governance | — | ✓ | ✓ | — | no actuation spec |
| Autonolas whitepaper [2022] | 2022 | Web3 engineering | ✓ | ✓ | ✓ | ✓ | native |
| Eliza / AI16Z docs [2024] | 2024 | Web3 engineering | ✓ | ✓ | ✓ | ✓ | native |
| **This paper (C1-C4)** | 2026 | Web3 systems | ✓ | ✓ | ✓ | ✓ | native |

### 2.1 Classical AI-agent definitions

Russell and Norvig [2020] define an agent as "anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators." This is the weakest possible framing — a thermostat satisfies it. Their book refines the notion via architectural tiers (simple reflex, model-based, goal-based, utility-based, learning), tiers that inspired our autonomy dimension. Wooldridge and Jennings [1995] advance four attributes for *intelligent* agents: autonomy, social ability, reactivity, and pro-activeness. We retain autonomy (C3), reactivity (C2), and pro-activeness (~C4) but drop social ability, because a lone MEV bot is uncontroversially an agent. Franklin and Graesser [1996] enumerate candidate properties (reactive, autonomous, goal-oriented, temporally continuous, communicative, learning, mobile, flexible, character) without committing to necessity; our C1-C4 are a specific minimal necessary subset. Maes [1995] argues for *behavioral* definition — agents should be identified by observable behavior, not architecture. This philosophical stance is the foundation of our operationalization in Section 3.3.

### 2.2 HCI automation-level work

Shneiderman [2022] argues against framing full autonomy as the design goal, proposing a two-dimensional human-control / computer-automation framework. Parasuraman, Sheridan, and Wickens [2000] offer a 10-level classification across four information-processing stages (acquisition, analysis, decision, action); we collapse this to a 5-level autonomy dimension because transaction data exposes only action, not internal processing stages. Amershi et al. [2019] distill 18 guidelines for human-AI interaction relevant to how agents present themselves to users, but they do not address *measurement*. Horvitz [1999] formalizes mixed-initiative interaction; our collaborative autonomy level inherits this framing for DAO-gated agents.

### 2.3 Blockchain bot and MEV empirics

Daian et al. [2020] are the canonical reference for the measurement of on-chain automated agents. Their "Flash Boys 2.0" paper quantifies priority-gas auctions, front-running, and sandwich attacks, and establishes that a substantial fraction of DEX volume is driven by automated programs. Qin, Zhou, and Gervais [2022] extend this to quantify extractable value across DeFi protocols. Torres et al. [2021] contribute a USENIX Security empirical study that itemizes the behavioral fingerprints of front-runners (gas-price sequences, timing relative to target transactions). All three works treat automation as a single undifferentiated category — "bots" — and do not distinguish between what we will call Deterministic Scripts, Simple Trading Bots, MEV Searchers, and DeFi Management Agents. Our taxonomy provides the missing granularity.

### 2.4 On-chain AI-agent platforms

Autonolas / OLAS [Autonolas 2022] provides an open-source multi-agent service framework in which agents stake OLAS to register, settle off-chain computation on-chain through a finite-state machine, and expose their activity in a public registry. This is the most important training-data source for our rule-based projection (Section 4.2). AI16Z / Eliza [2024] packages LLM reasoning into an on-chain persona that can trade, post on social media, and interact with DeFi protocols; it is the canonical example of what we call an LLM-Powered Agent. Virtuals Protocol tokenizes agents, linking each agent to a token reflecting its utility. Fetch.ai's Autonomous Economic Agents (AEAs) formalize peer-to-peer agent communication. These platforms commit to "agent" as a design category but do not share a definition that lets a third party audit claims.

### 2.5 Gaps

Three gaps motivate this paper:

1. **No existing definition ties agent-ness to on-chain observability.** Architectural definitions (BDI, FSM, RL policy) are not directly checkable from blockchain data. We need a definition whose conditions can be evaluated on transaction logs.

2. **No existing definition separates "has a policy" from "adapts its policy".** Wooldridge's pro-activeness and Franklin and Graesser's goal-orientation both admit frozen-strategy bots as agents. Our C4 (adaptiveness) enforces learning over time.

3. **No existing taxonomy of on-chain agents is empirically validated.** Prior taxonomies are internal and self-consistent but have not been projected onto real data. Blockchain's full-record property is a unique opportunity for empirical validation, which Section 4 of this paper exercises.

---

## 3. The C1-C4 Definition

This section presents a formal statement of the four-condition definition, justifies each condition by a counterexample demonstrating its non-redundancy, and describes how each condition is operationalized on transaction-level features.

### 3.1 Formal statement

**Definition (On-chain AI agent).** Let $E$ be a software entity and let $B$ be the sequence of blockchain events associated with $E$ over a time window $[t_0, t_1]$. Then $E$ is an on-chain AI agent iff it satisfies all four conditions:

$$
\mathrm{Agent}(E) \iff C_1(E) \land C_2(E) \land C_3(E) \land C_4(E).
$$

**C1 — On-chain actuation.**
$$
C_1(E) \iff \mathrm{Control}(E, \alpha) \text{ where } \alpha \text{ is an EOA and } \exists\,\mathrm{tx} \in B : \mathrm{tx.from} = \alpha.
$$
$E$ directly or indirectly controls an externally-owned account $\alpha$, and $\alpha$ initiates at least one transaction in the observation window.

**C2 — Environmental perception.**
$$
C_2(E) \iff I(\mathrm{state}_t ;\, \mathrm{action}_{t+1}) > 0,
$$
where $I$ is the mutual information between the chain state at time $t$ (and any off-chain signals observable by $E$) and the parameters of the next transaction.

**C3 — Autonomous decision-making.**
$$
C_3(E) \iff H(\mathrm{action} \mid \mathrm{environment}) > 0 \;\land\; \lnot\,\mathrm{human\text{-}gated}(E).
$$
The conditional entropy of actions given environment is positive (there is genuine non-determinism) *and* each individual transaction is not preceded by a per-transaction human approval signal.

**C4 — Adaptiveness.**
$$
C_4(E) \iff \exists\,\text{window } w :\; \mathrm{KS\text{-}dist}\!\left(\mathrm{params}_{t \leq w},\, \mathrm{params}_{t > w}\right) > \delta.
$$
Some behavioral-parameter distribution drifts by more than a Kolmogorov-Smirnov threshold between an early and a late sub-window.

### 3.2 Non-redundancy: each condition excludes a distinct counterexample

Each $C_i$ excludes a class of entities that satisfy the other three. We present one counterexample per condition.

**$C_1$ is non-redundant — smart contracts.** A Uniswap V3 pool reacts to calldata ($C_2$), its output curve is non-trivial for varied inputs ($C_3$-like), and its fee tier can evolve via governance ($C_4$). But it cannot initiate transactions — it is only callable. Removing $C_1$ would admit every non-trivial contract as an agent.

**$C_2$ is non-redundant — blind relayers.** An entity that signs pre-computed transaction batches and submits them on a fixed schedule without reading chain state satisfies $C_1$ (EOA), and its inter-transaction interval has positive variance due to mempool latency ($\sim C_3$), and its batch sizes drift due to external queue dynamics ($\sim C_4$). But its transaction parameters are not conditioned on chain state; $C_2$ would be violated by construction. Removing $C_2$ would admit every scheduled relayer.

**$C_3$ is non-redundant — deterministic DCA bots.** A grid / DCA trading bot has an EOA ($C_1$), reads price before each trade ($C_2$), and may have its grid width parameter-tuned by its owner ($C_4$). But given the same price it always executes the same trade — conditional entropy is zero. Removing $C_3$ would admit every DCA bot.

**$C_4$ is non-redundant — frozen MEV sandwich bots.** A bot that runs exactly the same sandwich logic from deployment to deprecation has an EOA ($C_1$), observes the mempool ($C_2$), and is non-deterministic in which mempool target it chooses ($C_3$). But its sandwich parameters (slippage, gas strategy, target selection) never change. Removing $C_4$ would admit frozen-strategy bots, losing a key property of *intelligent* agency: the ability to learn from experience.

### 3.3 Operationalization on transaction-level features

The four conditions are latent — we cannot inspect the source code or the internal decision graph of an entity from its blockchain footprint alone. We propose the following observable proxies, which form the theoretical grounding for the 23-feature set shared with Paper 1:

- **C1 is observable from:** `eth_getCode(address) == "0x"` (the address is an EOA, not a contract) and the existence of at least one transaction with `from == address`.

- **C2 is observable from:** calldata entropy > 0 (diverse method calls indicate reaction to environment), short latency between on-chain events and subsequent transactions relative to a fixed schedule, and positive correlation between transaction timing and on-chain event timing.

- **C3 is observable from:** `tx_interval_cv` above a threshold (non-deterministic timing), `active_hour_entropy > 2.5` or `burst_ratio > 0.05` (non-circadian or humanly-impossible burst patterns).

- **C4 is observable from:** `gas_strategy_drift > 0.15` between early and late sub-windows, Jaccard change of the target-contract set above a threshold, and detectable strategy switch points via changepoint analysis.

Table 2 summarizes the observability of each condition.

**Table 2. Observability and robustness of C1-C4 proxies.**

| Condition | Robustness | Primary proxy | False-positive sources |
|-----------|-----------|---------------|-----------------------|
| $C_1$ | high | `eth_getCode`, outgoing tx count | multisig-controlled contracts |
| $C_2$ | medium | calldata entropy, event-driven latency | pre-computed batched txs |
| $C_3$ | low | tx-interval CV, hour entropy, burst ratio | heavy human DeFi users |
| $C_4$ | low | distributional drift, KS-distance | external market regime shifts |

**A critical caveat on label leakage.** If the $C_3$ proxies (`tx_interval_cv`, `active_hour_entropy`, `burst_ratio`) are used both as *labeling gates* (to decide which addresses are agents) and as *classifier features* (to learn an agent-vs-human separator), the classifier exhibits direct target leakage. The formal definition above is independent of this: $C_3$ requires the *concept* of non-determinism and non-human-gating. For downstream use in Paper 1, labels are derived from external provenance (platform registry membership, Flashbots searcher list, ENS) and the 23 features are used only to test how well the concept is *recoverable* from behavior, not to define the label.

### 3.4 Boundary cases

Some entities occupy the boundary of the definition; we treat them explicitly rather than hide them behind an arbitrary threshold.

**Cross-chain bridge relayers.** Pass $C_1$ (EOA) and $C_2$ (monitor both chains). $C_3$ is partial (deterministic relay logic, but timing can be stochastic). $C_4$ is partial (parameters updated off-chain by operators). Paper 0 labels these as category 4 with a 0.5 confidence weight.

**ERC-4337 account-abstraction smart wallets with automation.** Tricky on $C_1$: the smart wallet is not an EOA, but the bundler (or session key) that submits on its behalf is. $C_2$ passes, $C_3$ passes if automation uses learned policy, $C_4$ is variable. Our resolution: the bundler / session-key is the C1 actuator; the smart wallet is a *tool* used by the agent, not the agent.

**Serial Sybil bots (one operator, many EOAs).** Satisfy $C_1$ per EOA and $C_2$ (read airdrop criteria), but typically fail $C_3$ because all EOAs execute the same playbook and fail $C_4$ because the playbook is static. *This is important for Paper 3: AI Sybils violate $C_3$ because their behavior is coordinated by a single controller, even though the individual EOAs look noisy.*

**Liquidation bots.** Pass all four: $C_1$ (EOA), $C_2$ (price feeds), $C_3$ (race conditions mean only one wins, which depends on gas strategy), $C_4$ (gas strategies evolve as mempool conditions change). Clear agents.

---

## 4. Taxonomy and On-Chain Empirical Validation

We now project the C1-C4 definition onto a taxonomy with finer semantic granularity, and validate both on real on-chain data. We use three complementary methods. Rule-based projection tests whether the taxonomy can be operationalized as a deterministic classifier. Unsupervised clustering tests whether the taxonomy structure is recoverable from behavior alone. Multi-class supervised classification tests whether each category has a learnable behavioral signature.

### 4.1 Validation strategy and dataset

The dataset is the 2,744 agent addresses from Paper 1's expanded mining plus a targeted Phase 2 mining pass (`features_with_taxonomy.parquet`), spanning Autonolas, Fetch.ai, AI Arena, curated MEV-searcher lists, Gnosis Safe modules, Stargate bridge relayers, and Aave+Uniswap multi-protocol RL strategies. We exclude 726 human-labeled addresses from the projection to keep the focus on agent-subtype distinguishability. Across all experiments we use the same 23-feature set (described in Section 3.3 and detailed in Table 3), standardized with z-scoring and winsorized at the [1, 99] percentile range.

**Table 3. The 23 behavioral features used in Section 4 experiments.**

| Group | Features |
|-------|----------|
| Timing (7) | tx_interval_mean, tx_interval_std, tx_interval_skewness, active_hour_entropy, night_activity_ratio, weekend_ratio, burst_frequency |
| Gas (6) | gas_price_round_number_ratio, gas_price_trailing_zeros_mean, gas_limit_precision, gas_price_cv, eip1559_priority_fee_precision, gas_price_nonce_correlation |
| Interaction (5) | unique_contracts_ratio, top_contract_concentration, method_id_diversity, contract_to_eoa_ratio, sequential_pattern_score |
| Approval (5) | unlimited_approve_ratio, approve_revoke_ratio, unverified_contract_approve_ratio, multi_protocol_interaction_count, flash_loan_usage |

### 4.2 The eight-category taxonomy

We organize agent subtypes along three orthogonal dimensions.

- **Autonomy level (5 ranks):** NONE, REACTIVE, ADAPTIVE, PROACTIVE, COLLABORATIVE.
- **Environment type (4 types):** on-chain only, hybrid (off-chain to on-chain), cross-chain, multi-modal.
- **Decision model (5 types):** deterministic, statistical, LLM-driven, reinforcement learning, hybrid.

The Cartesian product has 100 cells, but only a small minority are semantically populated by actual systems. We define eight categories covering the Web3 agent ecosystem as of 2026:

1. **Deterministic Script** (NONE, hybrid, deterministic) — scheduled transfers, hard-coded airdrop distributors. *Not an agent under C1-C4.*
2. **Simple Trading Bot** (REACTIVE, hybrid, deterministic) — grid, DCA, rule-based rebalancers. *Not an agent under C1-C4.*
3. **MEV Searcher** (ADAPTIVE, hybrid, statistical) — sandwich / arbitrage / liquidation. *Agent.*
4. **Cross-Chain Bridge Agent** (ADAPTIVE, cross-chain, deterministic) — LayerZero / Wormhole relayers. *Boundary.*
5. **RL Trading Agent** (ADAPTIVE, hybrid, reinforcement learning). *Agent.*
6. **DeFi Management Agent** (PROACTIVE, hybrid, hybrid) — Autonolas services, Yearn strategists, DeFi Saver. *Agent.*
7. **LLM-Powered Agent** (PROACTIVE, multi-modal, LLM-driven) — AI16Z / Eliza, Virtuals. *Agent.*
8. **Autonomous DAO Agent** (COLLABORATIVE, on-chain, hybrid) — Gnosis Safe modules, Governor execution. *Agent.*

Categories 1-2 are non-agents by C1-C4 and serve as reference baselines. Category 4 is a boundary class. The remaining five — 3, 5, 6, 7, 8 — are agents. The taxonomy thus spans both agents and non-agents; the C1-C4 definition draws the boundary *within* the taxonomy between REACTIVE and ADAPTIVE.

### 4.3 Rule-based taxonomy projection

**Method.** Each agent in the dataset is assigned to one of the eight categories by a two-tier rule system. Tier 1 uses *provenance*: the contract source from which the address was mined and any name annotation (Etherscan label, ENS resolution, Flashbots searcher list). Tier 2 uses *feature-based refinement*: when provenance is ambiguous, the rule consults a four-dimensional subspace of the behavioral feature set (`burst_frequency`, `gas_price_round_number_ratio`, `unique_contracts_ratio`, `multi_protocol_interaction_count`).

Representative Tier 1 rules include: an address mined from the Autonolas Agent Registry maps to DeFi Management Agent with confidence 0.90; a name containing `MEV|Flashbots|sandwich|builder` maps to MEV Searcher with confidence 0.95; a name containing `AI Arena NRN` maps to LLM-Powered Agent candidate with confidence 0.75. Representative Tier 2 rules include: `burst_frequency > 0.20 ∧ gas_price_cv > 0.30` maps to MEV Searcher; `gas_price_round_number_ratio > 0.60 ∧ method_id_diversity < 1.5` maps to Deterministic Script; `multi_protocol_interaction_count ≥ 5 ∧ unique_contracts_ratio > 0.40` maps to DeFi Management Agent. We use a *transparent* rule set rather than a learned mapping, because Section 4.5's classifier will consume the projection as ground truth and we want the relationship to be as non-circular as possible: the projection rules use four features in the refinement layer, the classifier uses all 23.

**Results.** Table 4 shows the projection.

**Table 4. Taxonomy projection of 2,744 agents. All eight categories are populated following a targeted Phase 2 mining pass against DAO multisigs, bridge relayers, and RL strategy vaults.**

| Category | Count | Percentage | Mean confidence |
|----------|------:|-----------:|----------------:|
| DeFi Management Agent | 1,669 | 60.8% | 0.60 |
| Deterministic Script  |   666 | 24.3% | 0.80 |
| Simple Trading Bot    |   130 |  4.7% | 0.70 |
| Cross-Chain Bridge Agent |  95 |  3.5% | 0.68 |
| LLM-Powered Agent     |    71 |  2.6% | 0.60 |
| MEV Searcher          |    54 |  2.0% | 0.85 |
| Autonomous DAO Agent  |    34 |  1.2% | 0.59 |
| RL Trading Agent      |    25 |  0.9% | 0.87 |

The projection now covers all eight categories. The three newly populated categories — Autonomous DAO Agent (34 instances from Gnosis Safe modules), Cross-Chain Bridge Agent (95 instances from Stargate relayers), and RL Trading Agent (25 instances from Aave+Uniswap multi-protocol strategies) — were added through a targeted Phase 2 mining pass that expanded the dataset from 2,590 to 2,744 addresses.

Confidence distribution: 1,853 projections in the medium band [0.5, 0.7], 688 in the high band [0.7, 0.85], 45 at 0.85+. 99.7% of projections used the provenance-then-features tier; only 0.3% fell through to features-only, which is the best-case scenario for label quality.

### 4.4 Unsupervised cluster validation

**Method.** We run K-Means on the 23-feature matrix with `n_init=10` for $k \in \{3, 4, \ldots, 15\}$. For each $k$ we report:

- Silhouette score (sample size 2,000);
- Adjusted Rand Index (ARI) against the 5-category projection labels;
- Normalized Mutual Information (NMI) against the projection labels;
- Per-cluster purity (fraction belonging to the dominant projection category);
- Number of orphan clusters (purity < 0.5).

**Results.** Table 5 shows the full sweep.

**Table 5. K-Means sweep on the 23-feature matrix (N=2,744). Silhouette is maximized at k=3; ARI is maximized at k=5.**

| $k$ | Silhouette | ARI | NMI | Inertia |
|----:|-----------:|----:|----:|--------:|
|  **3** | **0.1509** | 0.2642 | 0.2124 | 41,864 |
|  4 | 0.1226 | 0.1798 | 0.2042 | 38,839 |
|  **5** | 0.1318 | **0.3189** | **0.3398** | 36,525 |
|  6 | 0.1246 | 0.2383 | 0.3008 | 34,713 |
|  7 | 0.1280 | 0.2163 | 0.2931 | 33,166 |
|  8 | 0.1289 | 0.2225 | 0.3132 | 31,768 |
|  9 | 0.1174 | 0.1740 | 0.2995 | 30,740 |
| 10 | 0.1151 | 0.1673 | 0.3097 | 29,940 |
| 11 | 0.1213 | 0.1624 | 0.2978 | 29,191 |
| 12 | 0.1214 | 0.1592 | 0.2935 | 28,587 |
| 13 | 0.1109 | 0.1440 | 0.3158 | 27,970 |
| 14 | 0.1116 | 0.1331 | 0.2928 | 27,283 |
| 15 | 0.1060 | 0.1204 | 0.2903 | 26,686 |

At $k=8$ (the full taxonomy), mean cluster purity is 0.779 with one orphan cluster (cluster 2 in the k=8 run, purity 0.327 — a mixed cluster containing DeFi Management, Deterministic Script, and Simple Trading Bot addresses in roughly equal proportions). At $k=3$, there are no orphan clusters and mean purity is 0.760.

**Interpretation: behavioral super-clusters.** The silhouette result is the strongest empirical finding of this section — the 23-feature space supports a three-cluster structure much more clearly than an eight-cluster structure. The three super-clusters correspond roughly to:

- **Cluster A — static / deterministic.** High `gas_price_round_number_ratio`, low `method_id_diversity`, low `burst_frequency`. Covers Deterministic Script and Simple Trading Bot.
- **Cluster B — active DeFi.** High `multi_protocol_interaction_count`, moderate `unique_contracts_ratio`, moderate `burst_frequency`. Covers DeFi Management Agent.
- **Cluster C — adaptive / high-frequency.** Very high `burst_frequency`, low `tx_interval_mean`, high `gas_price_cv`. Covers MEV Searcher and, surprisingly, most LLM-Powered Agent instances.

The eight-category taxonomy is **semantically coherent but behaviorally over-split** at the 23-feature level. The taxonomy distinguishes DeFi Management from LLM-Powered, but these two classes are largely overlapping in feature space. Two interpretations are possible:

1. **Taxonomic interpretation.** The eight categories are an analytical refinement of an underlying three-class behavioral structure. Future work should either accept the three-class decomposition or expand the feature set to make all eight distinguishable.

2. **Measurement interpretation.** Current LLM-driven agents *mimic* rule-based DeFi agents because their on-chain footprint is similar; they differ in *off-chain* prompt-driven action diversity that does not yet appear in transaction-level features. New features are needed.

We adopt the measurement interpretation in Section 5, because independent Paper 3 work shows that eight AI-specific features drawn from real agent/human data separate the two populations with Cohen's $d > 1.0$ on `hour_entropy` and `behavioral_consistency`. This cross-paper corroboration is the main reason we believe the over-splitting is a feature-set limitation, not a taxonomy error.

### 4.5 Multi-class supervised classifier

**Method.** We train three classifiers to predict the taxonomy category from the 23 behavioral features only (no source / name leakage). All three are evaluated under 5-fold stratified cross-validation.

- Gradient Boosting (100 trees, depth 3, learning rate 0.1)
- Random Forest (300 trees, depth 8)
- Logistic Regression (C = 1.0)

All eight categories are now populated. The final training set is $N = 2{,}744$ across eight classes.

**Aggregate results.** Gradient Boosting is the best model, with $0.9737 \pm 0.0040$ accuracy, $0.8683 \pm 0.0267$ macro-F1, and $0.9706$ weighted-F1. Random Forest reaches accuracy 0.9471 / macro-F1 0.7278 (lower because its recall on LLM-Powered collapses to 0.11). Logistic Regression reaches accuracy 0.9124 / macro-F1 0.6914.

**Per-class results.** Table 6 details the per-class metrics for the best model.

**Table 6. Gradient Boosting per-class performance on the original five categories (five-class model). 95% CIs computed via 1,000-iteration bootstrap on out-of-fold predictions. LLM-Powered Agent is the empirical weak spot. See the eight-class results below for the full taxonomy.**

| Class | n | F1 | 95% CI | Precision | Recall |
|-------|--:|---:|-------:|----------:|-------:|
| Deterministic Script   |   666 | **0.999** | [0.996, 1.000] | 0.999 | 0.994 |
| Simple Trading Bot     |   130 | 0.977 | [0.956, 0.993] | 0.992 | 0.985 |
| DeFi Management Agent  | 1,669 | 0.980 | [0.975, 0.984] | 0.970 | 0.994 |
| MEV Searcher           |    54 | 0.887 | [0.817, 0.950] | 0.865 | 0.833 |
| **LLM-Powered Agent**  |    71 | **0.466** | [0.341, 0.578] | 0.800 | 0.394 |

Four of five categories exceed F1 0.85. The LLM-Powered Agent is the only class below 0.85. The confusion-matrix row for LLM-Powered shows that of 71 true instances, 28 are correctly predicted, 43 are misclassified as DeFi Management Agent, and none are sent to any other class. Precision for LLM-Powered is 0.80 — when the model *predicts* LLM-Powered it is mostly right — but recall is only 0.394. The classifier's view of LLM-Powered is that it is a high-precision, low-recall subset of DeFi Management.

**Eight-class classifier.** With the Phase 2 mining pass populating the three previously empty categories (Autonomous DAO Agent n=34, Cross-Chain Bridge Agent n=95, RL Trading Agent n=25), we re-train the Gradient Boosting classifier on all eight classes ($N = 2{,}744$). The eight-class model reaches accuracy $0.922 \pm 0.008$ and macro-F1 0.58 — a 5-point drop in accuracy from the original five-class model (0.974). The three newly populated categories are the hardest: DAO Agent achieves F1=0.14 (precision 0.33, recall 0.09), Cross-Chain Bridge Agent achieves F1=0.31 (precision 0.54, recall 0.22), and RL Trading Agent achieves F1=0.07 (precision 0.25, recall 0.04). All three suffer from small sample sizes and heavy overlap with DeFi Management Agent in the 23-feature space. The original five classes retain their performance: Deterministic Script F1=0.98, DeFi Management Agent F1=0.96, Simple Trading Bot F1=0.95, MEV Searcher F1=0.82, and LLM-Powered Agent F1=0.43. The macro-F1 drop from 0.87 to 0.58 is driven almost entirely by the three new low-F1 classes, confirming that the current feature set lacks the discriminative power to separate DAO governance, cross-chain relay, and RL strategy behaviors from the DeFi Management majority class.

**Feature importance.** The Gradient Boosting feature importance is heavily concentrated: `gas_price_round_number_ratio` alone accounts for 70.5%, `sequential_pattern_score` for 12.6%, `burst_frequency` for 9.9%, and `gas_price_trailing_zeros_mean` for 3.0%. Together these four features capture 96% of the model's decision weight — a useful signal for future feature-pruning but also a warning that the model is leaning heavily on gas-precision style, which is exactly the feature subset that the projection rules use for the Deterministic Script and DeFi Management refinements.

### 4.6 Threats to validity

**Partial circularity between projection and classifier.** The taxonomy projection rules (Section 4.3) use eight of the 23 features in the Tier 2 refinement layer. The multi-class classifier (Section 4.5) uses all 23 features, including the eight used by the rules. Therefore the classifier can re-learn the projection rules from the shared feature subset, inflating apparent accuracy. Empirically the classifier's top-importance features (`gas_price_round_number_ratio`, `sequential_pattern_score`, `burst_frequency`) overlap with the Tier 2 rule antecedents. A stricter test would require a hand-annotated taxonomy label per address that is *independent* of any behavioral feature; we view this as a Phase 3 follow-up.

**Mining bias.** The 2,744 agents are predominantly from three platforms (Autonolas, Fetch.ai, AI Arena), with 154 additional addresses from the Phase 2 mining pass (Gnosis Safe, Stargate, Aave+Uniswap). Cross-platform generalization is not tested here. Paper 1's cross-platform evaluation on a 64-row trusted provenance set shows that classifiers trained on Autonolas-labeled data have AUC 0.24-0.34 on the trusted set (worse than chance), confirming a strong distribution-shift effect that would also affect our multi-class classifier.

**Newly populated categories remain small.** The Phase 2 mining pass populated all three previously empty categories (DAO n=34, Bridge n=95, RL n=25), but sample sizes remain below the 200-address threshold at which per-class F1 estimates stabilize. The eight-class classifier's low F1 on these categories (0.14, 0.31, 0.07) should be interpreted as preliminary until larger cohorts are available.

**Small LLM-Powered cohort.** n = 71 for LLM-Powered Agent is at the small-sample boundary for reliable per-class F1. The bootstrap 95% CI for LLM-Powered is [0.341, 0.578], substantially wider than the [0.996, 1.000] CI for Deterministic Script. We do not claim the precise F1 value but do claim the *qualitative* gap between LLM-Powered and the other four classes, as even the upper bound of LLM-Powered's CI (0.578) falls well below the lower bounds of all other classes.

### 4.7 Cross-reference to Paper 1

Paper 1 trains a *binary* classifier (agent vs human) on the same 23 features. Its honest performance on a 64-row provenance-only trusted set is AUC 0.883 under a Graph Attention Network (GAT) architecture. Paper 0 answers a complementary question: given that Paper 1 has identified an address as an agent, which of the populated categories is it?

- **Paper 1 answers:** Is this address an agent or a human?
- **Paper 0 answers:** If Paper 1 says "agent", which of the eight categories is it?

For LLM-Powered Agent and the three newly populated categories (DAO, Bridge, RL), the joint pipeline currently struggles: Paper 1 identifies these as agents with high probability, but Paper 0 confuses them with DeFi Management Agent. This motivates the integration proposal discussed in Section 5: adding Paper 3's eight AI-specific features to the Paper 0 / Paper 1 shared feature set should close these gaps without breaking the well-separated classes.

### 4.8 Summary of validation findings

1. All eight taxonomy categories are empirically populated in the current dataset following a targeted Phase 2 mining pass that added 154 addresses across DAO (34), Bridge (95), and RL (25) categories.
2. Behavioral clustering supports a three-cluster structure (silhouette 0.151 at $k=3$ vs 0.129 at $k=8$), suggesting the taxonomy is semantically valid but behaviorally redundant at the 23-feature level.
3. The five-class supervised classifier reaches 97.4% accuracy with macro-F1 0.87. Expanding to all eight classes drops accuracy to 92.2% and macro-F1 to 0.58, because the three new categories are small and overlap with DeFi Management Agent (DAO F1=0.14, Bridge F1=0.31, RL F1=0.07).
4. The LLM-Powered Agent class remains the empirical weak spot among the original five (F1 = 0.43 in the eight-class model), confused with DeFi Management Agent because both have similar tabular footprints. New AI-specific features are needed to distinguish them.
5. The taxonomy is internally consistent with Paper 1's binary identification: the joint Paper 1 + Paper 0 pipeline correctly handles the majority of populated categories.

---

## 5. Discussion

### 5.1 The LLM-Powered Agent weak spot

The single most important empirical finding of this paper is that the LLM-Powered Agent class has F1 = 0.528 under the best classifier, while every other populated class exceeds F1 = 0.85. The confusion is one-directional: 60% of true LLM-Powered instances are predicted as DeFi Management Agent, not the reverse. Why?

**The feature set does not yet capture what makes an LLM agent different.** The 23 features measure *when* transactions happen (timing), *how much* gas they pay (gas), *what* contracts they interact with (interaction), and *how risky* the approvals are (approvals). None of these capture prompt-driven action diversity, LLM response latency signatures, or the semantic *novelty* of successive action sequences. From the perspective of the 23-feature representation, an Autonolas rule engine and a prompt-driven Eliza agent look nearly identical: both are busy, both interact with multiple protocols, both have adaptive gas strategies. The distinguishing features of LLM reasoning — context sensitivity, plan revision, prompt injection vulnerability — do not project onto the 23 features at all.

**This is good news for the taxonomy.** It means the eight categories are not wrong; the *feature set* is incomplete. Paper 3 has developed an independent eight-feature set that distinguishes AI-driven entities from humans with large effect sizes (Cohen's $d > 1.0$ on `hour_entropy` and `behavioral_consistency`). The natural Phase 3 experiment is to union the 23 Paper 1 features with the 8 Paper 3 features, giving a 31-feature space, and re-run Section 4.5's classifier. We predict the LLM-Powered Agent F1 will climb above 0.80 without harming the other four classes, because the Paper 3 features target the exact dimension on which LLM agents and DeFi management agents differ.

**It is also a concrete policy implication.** A regulator or platform that wants to enforce disclosure on "LLM-powered agents specifically" — for instance, because the governance risk profile of a prompt-driven agent is different from that of a rule-driven one — cannot do so with today's 23-feature behavioral tooling. Any rule that says "flag all LLM agents" will either over-capture (catching DeFi management agents too) or under-capture (missing 60% of LLM agents). This is a concrete example of where feature engineering has practical governance implications.

### 5.2 k = 3 versus k = 8: behavioral versus semantic taxonomies

The cluster-validation result is the second key finding. Silhouette is maximized at $k = 3$ (0.151) and declines monotonically past $k = 5$; ARI against the projection labels peaks at $k = 5$ (the number of populated categories), then decays. At $k = 8$ — the full taxonomy — silhouette is 0.129 and one cluster is an orphan (mean purity 0.327, mixing DeFi Management, Deterministic Script, and Simple Trading Bot addresses in roughly equal proportions).

The natural reading is that **behavioral taxonomies saturate at three super-clusters, and semantic taxonomies require external provenance signals**. The three super-clusters have intuitive labels:

1. **Static / deterministic** — the cluster of entities that satisfy $C_1 \land C_2 \land \lnot C_3 \land \lnot C_4$. Scripts and simple bots.
2. **Active but non-adaptive** — satisfies $C_1 \land C_2 \land C_3 \land \lnot C_4$, or has weak $C_4$. Rule-based DeFi management.
3. **Fully adaptive** — satisfies all four. MEV searchers, liquidation bots, LLM agents.

This three-cluster decomposition is almost exactly the structure that $C_3$ and $C_4$ are designed to pick out. It is a reassuring sanity check: the definition slices the population at the same place that behavioral clustering slices it. The eight categories refine cluster 3 into MEV-searcher / DeFi-management / LLM-powered / DAO / RL subtypes by *provenance*, not by *behavioral distinguishability*.

**This is a publishable finding in its own right.** Any future taxonomy of on-chain agents should distinguish behavioral from semantic granularity. A behavioral taxonomy (what you can learn from transactions) and a semantic taxonomy (what you can learn from provenance) should be different objects, and the dialogue between them is what makes the taxonomy interesting.

### 5.3 Is C4 (Adaptiveness) necessary? A deliberate design choice

A predictable reviewer objection targets C4. Wooldridge and Jennings [1995] do not require adaptiveness, and many practitioners would call a reactive MEV bot an "agent" even if its strategy is frozen from deployment to deprecation. Our C4 non-redundancy argument (Section 3.2) uses such a frozen bot as the counterexample, but the deeper motivation is *why* we draw the line here rather than at C3 alone.

We deliberately take a **stronger-than-standard definition** because C4 operationalizes a concrete difference that matters for all three downstream papers: Paper 1's security audit shows that adaptive agents accumulate different approval patterns over time (the time-series drift is a C4 proxy); Paper 3's LLM sybils demonstrate that C4-capable entities can *evolve their evasion strategy* in response to detector feedback, which frozen scripts cannot. A definition that admits frozen scripts would conflate the two populations and lose the practical utility that motivates the taxonomy.

We acknowledge that this is a design choice, not a logical necessity. An alternative definition dropping C4 would yield a coarser agent class that includes fixed-strategy bots. We report the empirical consequence: in our cluster validation, the k=3 super-cluster structure (Section 5.2) partially merges the C4-negative and C4-positive populations, suggesting that C4 adds discriminative value at the semantic level even if the behavioral signal is weak. We invite the community to treat C4 as the axis along which the definition can be relaxed or tightened depending on the application.

### 5.4 Implications for HCI and governance

The C1-C4 definition and the taxonomy let HCI researchers, regulators, and practitioners talk about *specific kinds* of on-chain agents rather than about "agents in general". Four implications are immediate:

**Differentiated disclosure.** NONE-REACTIVE entities (Deterministic Scripts, Simple Trading Bots) are fully predictable and probably need no new regulation beyond existing financial compliance. ADAPTIVE entities (MEV Searchers, RL Trading Agents) operate in a bounded strategy space and may need disclosure of the strategy range. PROACTIVE entities (DeFi Management Agents, LLM-Powered Agents) plan and initiate, and may need stronger disclosure especially around LLM decision uncertainty. COLLABORATIVE entities (DAO Agents) involve collective decision-making and may need governance-specific permission frameworks. This gradient aligns with the EU AI Act's risk categories (low / limited / high / unacceptable) more cleanly than any existing Web3-specific framework.

**Threat profiles per category.** Deterministic Scripts are immune to prompt injection but susceptible to parameter manipulation. LLM-Powered Agents face the full spectrum of LLM-specific attacks (prompt injection, hallucination, tool abuse). MEV Searchers face competitive adversarial pressure (gas auction warfare). Cross-Chain Bridges face replay-style attacks. A single "agent security" framing flattens all of these.

**User mental models.** Amershi et al.'s [2019] human-AI interaction guidelines assume the user knows what class of system they are using. The taxonomy gives HCI researchers the vocabulary to ask whether a DeFi interface should behave differently when the counterparty is an LLM agent versus a rule-based DeFi management agent.

**Interoperability.** The C1-C4 definition is explicitly shared across the NSF project's Papers 1-3 (see the "Paper integration" document). Paper 1 uses it to train a binary identification classifier. Paper 2 uses the C1 and C2 conditions to locate the tool-interface attack surface. Paper 3 uses the C3 and C4 conditions to argue that AI Sybils violate $C_3$ because they are coordinated by a single controller. A shared definition is how three papers become a coherent research program rather than three loosely related pilots.

---

## 6. Limitations and Future Work

**Single-chain dataset.** All validation is on Ethereum mainnet. Solana's parallel execution model, Polygon's higher throughput, and BSC's different gas-pricing dynamics may produce behavioral features that do not translate. A cross-chain replication study is the most direct extension.

**Partial circularity between projection and classifier.** As detailed in Section 4.6, the Tier 2 projection rules and the classifier share eight features. We mitigate this by (a) using a transparent rule set, (b) reporting the GBM feature-importance overlap as a measurable proxy for the leakage, and (c) flagging the issue rather than claiming a fully independent validation. A clean replication would require a hand-annotated label per address, ideally from an external annotator panel.

**Small sample sizes for DAO, Bridge, and RL categories.** The Phase 2 mining pass successfully populated all eight categories, but the three new categories remain small (DAO n=34, Bridge n=95, RL n=25). Their low classifier F1 scores (0.14, 0.31, 0.07) are partly a function of sample size and partly a function of feature-space overlap with DeFi Management Agent. A larger mining pass targeting 200+ addresses per category would clarify how much of the poor performance is due to small samples versus genuine feature-space indistinguishability.

**The LLM-Powered Agent weak spot and the new category overlap are known feature-set limitations.** We have argued in Section 5.1 that adding Paper 3's eight AI-specific features should close these gaps. This is a falsifiable prediction: Phase 3 will test whether the 23+8 = 31-feature space lifts LLM-Powered Agent F1 above 0.80 and improves separation of the three new categories without harming the original five classes.

**No expert Delphi yet.** Validation here is empirical (registry provenance, cluster recovery, classifier). A complementary Delphi study with 12-15 domain experts is designed but not yet executed. Both forms of evidence — objective registry data and subjective expert judgment — should eventually support the definition.

**Observability limits.** Purely off-chain agent components (LLM inference, internal strategy computation) are not directly observable. Two agents with fundamentally different internal architectures may produce similar on-chain footprints and be placed in the same category. This is a hard limit of chain-only validation and is the reason we frame C1-C4 around *operationalized proxies* rather than internal state.

**Temporal evolution.** The Web3 agent ecosystem is evolving fast. New categories — agent-of-agents architectures, on-chain AI inference, decentralized federated learning agents — may require new taxonomy cells. The taxonomy is a snapshot of 2026, not a timeless claim.

**Phase 3 work items.**

1. Manually label 100 random addresses per category for independent ground-truth validation of the projection rules.
2. Expand the DAO (n=34), Bridge (n=95), and RL (n=25) categories to 200+ addresses each to improve classifier reliability on these under-represented classes.
3. Add Paper 3's 8 AI features to the 23-feature base and re-run the eight-class classifier to test whether the 31-feature space lifts the three new categories above F1 0.50.
4. Replicate the validation on Solana and Polygon.
5. Execute the Delphi expert validation protocol.

---

## 7. Conclusion

We have proposed and validated the first formal definition of "on-chain AI agent" that is simultaneously minimal, chain-observable, and empirically testable. The definition has four necessary-and-sufficient conditions — on-chain actuation (C1), environmental perception (C2), autonomous decision-making (C3), and adaptiveness (C4) — each justified by a counterexample showing its non-redundancy, and each operationalized via specific proxies drawn from the 23-feature set shared with Paper 1.

Empirical validation on 2,744 Ethereum agent addresses yields four findings. First, all eight taxonomy categories are populated in the current dataset following a targeted Phase 2 mining pass, with DeFi Management Agent the majority class at 60.8%. Second, the five-class supervised classifier reaches 97.4% accuracy and macro-F1 0.87; expanding to all eight classes drops accuracy to 92.2% and macro-F1 to 0.58, because the three newly populated categories (DAO F1=0.14, Bridge F1=0.31, RL F1=0.07) are small and overlap with DeFi Management Agent in the 23-feature space. Third, unsupervised cluster validation finds that silhouette is maximized at k=3, not k=8, implying the eight-category taxonomy is semantically coherent but behaviorally over-split at the current 23-feature level — a finding we interpret as a feature-set limitation rather than a taxonomy error. Fourth, the three new categories confirm that semantic distinctions (governance execution vs. cross-chain relay vs. RL policy) require richer features than the current behavioral set provides; we conjecture that adding Paper 3's eight AI-specific features will improve separation.

Beyond these findings, the paper provides a shared vocabulary for the NSF project's downstream work. Paper 1 uses C1-C4 as the classification target for binary agent identification; Paper 2 locates its tool-interface attack surface at the C1 / C2 operationalization layer; Paper 3 argues that AI Sybils violate C3 because of central control. The definition is the backbone of a coherent research program rather than a stand-alone theoretical exercise.

For HCI and governance, the contribution is pragmatic. Differentiated disclosure regimes, category-specific threat models, and interface design for interacting with specific kinds of agents all require a definition that names its categories carefully. C1-C4 and the eight-category taxonomy give researchers and regulators the vocabulary; the 2,744-address validation gives them evidence that the vocabulary tracks real behavior — and, equally important, evidence of where it breaks.

---

## Reproducibility

All code, data, and experiment scripts for reproducing the results in this paper are available at [anonymous repo URL]. The 23-feature extraction pipeline, the C1--C4 operationalization rules, and the 2,744-address labeled dataset are included in `paper0_ai_agent_theory/` and `paper1_onchain_agent_id/data/`. The GBM five-fold cross-validation, PCA scatter, and cluster-sweep analyses can be rerun end-to-end with a single `make` invocation. All random seeds are fixed for deterministic reproduction.

---

## References

```bibtex
@inproceedings{amershi2019guidelines,
  title={Guidelines for Human-AI Interaction},
  author={Amershi, Saleema and Weld, Dan and Vorvoreanu, Mihaela and Fourney, Adam and Nushi, Besmira and Collisson, Penny and Suh, Jina and Iqbal, Shamsi and Bennett, Paul N and Inkpen, Kori and Teevan, Jaime and Kikin-Gil, Ruth and Horvitz, Eric},
  booktitle={Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems (CHI '19)},
  year={2019},
  publisher={ACM}
}

@misc{autonolas2022,
  title={Autonolas Whitepaper: Off-chain services co-owned and co-operated},
  author={{Autonolas}},
  year={2022},
  howpublished={\url{https://www.autonolas.network/whitepaper.pdf}}
}

@article{bratman1987intention,
  title={Intention, Plans, and Practical Reason},
  author={Bratman, Michael E},
  journal={Harvard University Press},
  year={1987}
}

@article{castelfranchi1995commitments,
  title={Commitments: From Individual Intentions to Groups and Organizations},
  author={Castelfranchi, Cristiano},
  journal={Proceedings of ICMAS-95},
  year={1995}
}

@article{chan2023harms,
  title={Harms from Increasingly Agentic Algorithmic Systems},
  author={Chan, Alan and Salganik, Rebecca and Markelius, Alva and Pang, Chris and Rajkumar, Nitarshan and Krasheninnikov, Dmitrii and Langosco, Lauro and He, Zhonghao and Duan, Yawen and Carroll, Micah and Lin, Michelle and Mayhew, Alex and Collins, Katherine and Molamohammadi, Maryam and Burden, John and Zhao, Wanru and Rismani, Shalaleh and Voudouris, Konstantinos and Bhatt, Umang and Weller, Adrian and Krueger, David and Maharaj, Tegan},
  booktitle={Proceedings of the 2023 ACM Conference on Fairness, Accountability, and Transparency (FAccT '23)},
  year={2023},
  publisher={ACM}
}

@inproceedings{daian2020flashboys,
  title={Flash Boys 2.0: Frontrunning in Decentralized Exchanges, Miner Extractable Value, and Consensus Instability},
  author={Daian, Philip and Goldfeder, Steven and Kell, Tyler and Li, Yunqi and Zhao, Xueyuan and Bentov, Iddo and Breidenbach, Lorenz and Juels, Ari},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)},
  year={2020},
  publisher={IEEE}
}

@book{dennett1987intentional,
  title={The Intentional Stance},
  author={Dennett, Daniel C},
  year={1987},
  publisher={MIT Press}
}

@misc{eliza2024,
  title={Eliza: The AI16Z Agent Framework},
  author={{AI16Z Team}},
  year={2024},
  howpublished={\url{https://github.com/ai16z/eliza}}
}

@misc{flashbots2021,
  title={MEV-Boost and Proposer-Builder Separation},
  author={{Flashbots}},
  year={2021},
  howpublished={\url{https://docs.flashbots.net/}}
}

@inproceedings{franklin1996agent,
  title={Is it an Agent, or Just a Program? A Taxonomy for Autonomous Agents},
  author={Franklin, Stan and Graesser, Art},
  booktitle={Proceedings of the Third International Workshop on Agent Theories, Architectures, and Languages (ATAL)},
  year={1996},
  publisher={Springer-Verlag}
}

@article{he2025survey,
  title={A Survey of AI Agent Protocols},
  author={He, Yuanchun and Li, Yue and Wang, Xin and others},
  journal={arXiv preprint arXiv:2601.04583},
  year={2025}
}

@inproceedings{horvitz1999principles,
  title={Principles of Mixed-Initiative User Interaction},
  author={Horvitz, Eric},
  booktitle={Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '99)},
  year={1999},
  publisher={ACM}
}

@article{jennings1998roadmap,
  title={A Roadmap of Agent Research and Development},
  author={Jennings, Nicholas R and Sycara, Katia and Wooldridge, Michael},
  journal={Autonomous Agents and Multi-Agent Systems},
  volume={1},
  number={1},
  pages={7--38},
  year={1998}
}

@incollection{maes1995modeling,
  title={Modeling Adaptive Autonomous Agents},
  author={Maes, Pattie},
  booktitle={Artificial Life: An Overview},
  year={1995},
  publisher={MIT Press}
}

@article{parasuraman2000model,
  title={A Model for Types and Levels of Human Interaction with Automation},
  author={Parasuraman, Raja and Sheridan, Thomas B and Wickens, Christopher D},
  journal={IEEE Transactions on Systems, Man, and Cybernetics---Part A: Systems and Humans},
  volume={30},
  number={3},
  pages={286--297},
  year={2000}
}

@inproceedings{park2023generative,
  title={Generative Agents: Interactive Simulacra of Human Behavior},
  author={Park, Joon Sung and O'Brien, Joseph C and Cai, Carrie J and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S},
  booktitle={Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)},
  year={2023},
  publisher={ACM}
}

@inproceedings{qin2022quantifying,
  title={Quantifying Blockchain Extractable Value: How Dark is the Forest?},
  author={Qin, Kaihua and Zhou, Liyi and Gervais, Arthur},
  booktitle={IEEE Symposium on Security and Privacy (S\&P)},
  year={2022},
  publisher={IEEE}
}

@book{russellnorvig2020,
  title={Artificial Intelligence: A Modern Approach},
  author={Russell, Stuart and Norvig, Peter},
  edition={4},
  year={2020},
  publisher={Pearson}
}

@article{shavit2023practices,
  title={Practices for Governing Agentic AI Systems},
  author={Shavit, Yonadav and Agarwal, Sandhini and Brundage, Miles and Adler, Steven and O'Keefe, Cullen and Campbell, Rosie and Lee, Teddy and Mishkin, Pamela and Eloundou, Tyna and Hickey, Alan and others},
  journal={OpenAI Research Paper},
  year={2023}
}

@article{shoham1993agent,
  title={Agent-Oriented Programming},
  author={Shoham, Yoav},
  journal={Artificial Intelligence},
  volume={60},
  number={1},
  pages={51--92},
  year={1993}
}

@book{shneiderman2022hcai,
  title={Human-Centered AI},
  author={Shneiderman, Ben},
  year={2022},
  publisher={Oxford University Press}
}

@inproceedings{torres2021frontrunner,
  title={Frontrunner Jones and the Raiders of the Dark Forest: An Empirical Study of Frontrunning on the Ethereum Blockchain},
  author={Torres, Christof Ferreira and Camino, Ramiro and State, Radu},
  booktitle={USENIX Security Symposium},
  year={2021},
  publisher={USENIX Association}
}

@article{wooldridge1995intelligent,
  title={Intelligent Agents: Theory and Practice},
  author={Wooldridge, Michael and Jennings, Nicholas R},
  journal={The Knowledge Engineering Review},
  volume={10},
  number={2},
  pages={115--152},
  year={1995}
}

@article{xi2023rise,
  title={The Rise and Potential of Large Language Model Based Agents: A Survey},
  author={Xi, Zhiheng and Chen, Wenxiang and Guo, Xin and He, Wei and Ding, Yiwen and Hong, Boyang and Zhang, Ming and Wang, Junzhe and Jin, Senjie and Zhou, Enyu and others},
  journal={arXiv preprint arXiv:2309.07864},
  year={2023}
}

@article{wang2024survey,
  title={A Survey on Large Language Model based Autonomous Agents},
  author={Wang, Lei and Ma, Chen and Feng, Xueyang and Zhang, Zeyu and Yang, Hao and Zhang, Jingsen and Chen, Zhiyuan and Tang, Jiakai and Chen, Xu and Lin, Yankai and Zhao, Wayne Xin and Wei, Zhewei and Wen, Ji-Rong},
  journal={Frontiers of Computer Science},
  volume={18},
  number={6},
  year={2024}
}

@inproceedings{yao2023react,
  title={ReAct: Synergizing Reasoning and Acting in Language Models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  booktitle={ICLR},
  year={2023}
}

@inproceedings{schick2023toolformer,
  title={Toolformer: Language Models Can Teach Themselves to Use Tools},
  author={Schick, Timo and Dwivedi-Yu, Jane and Dess{\`\i}, Roberto and Raileanu, Roberta and Lomeli, Maria and Zettlemoyer, Luke and Cancedda, Nicola and Scialom, Thomas},
  booktitle={NeurIPS},
  year={2023}
}

@article{mialon2023augmented,
  title={Augmented Language Models: A Survey},
  author={Mialon, Gr{\'e}goire and Dess{\`\i}, Roberto and Lomeli, Maria and Nalmpantis, Christoforos and Pasunuru, Ram and Raileanu, Roberta and Rozi{\`e}re, Baptiste and Schick, Timo and Dwivedi-Yu, Jane and Celikyilmaz, Asli and Grave, Edouard and LeCun, Yann and Scialom, Thomas},
  journal={Transactions on Machine Learning Research},
  year={2023}
}

@inproceedings{guo2024large,
  title={Large Language Model based Multi-Agents: A Survey of Progress and Challenges},
  author={Guo, Taicheng and Chen, Xiuying and Wang, Yaqi and Chang, Ruidi and Pei, Shichao and Chawla, Nitesh V and Wiest, Olaf and Zhang, Xiangliang},
  booktitle={IJCAI},
  year={2024}
}

@misc{erc6551,
  title={ERC-6551: Non-fungible Token Bound Accounts},
  author={{Ethereum Foundation}},
  year={2023},
  howpublished={\url{https://eips.ethereum.org/EIPS/eip-6551}}
}

@misc{erc4337,
  title={ERC-4337: Account Abstraction Using Alt Mempool},
  author={Vitalik Buterin and Yoav Weiss and Dror Tirosh and Shahaf Nacson and Alex Forshtat and Kristof Gazso and Tjaden Hess},
  year={2023},
  howpublished={\url{https://eips.ethereum.org/EIPS/eip-4337}}
}

@misc{mcp2024,
  title={Model Context Protocol Specification},
  author={{Anthropic}},
  year={2024},
  howpublished={\url{https://modelcontextprotocol.io/specification}}
}
```
