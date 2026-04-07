# 2. Related Work

This section surveys four bodies of literature that inform our taxonomy: classical AI agent definitions, HCI perspectives on automation and human-AI interaction, blockchain bot and MEV research, and the emerging class of on-chain AI agents. We conclude by identifying the gap our taxonomy addresses.

---

## 2.1 Agent Definitions in AI/CS

The concept of an "agent" has a long and contested history in artificial intelligence. We trace the key definitional milestones that inform our taxonomy dimensions.

**Russell & Norvig (2020).** *Artificial Intelligence: A Modern Approach* (4th edition) defines an agent as "anything that can be viewed as perceiving its environment through sensors and acting upon that environment through actuators." This perception-action loop is foundational to our taxonomy: we operationalize "environment" as our second dimension (on-chain, hybrid, cross-chain, multi-modal) and "action" as the observable on-chain transaction. Russell and Norvig further distinguish agent architectures -- simple reflex, model-based reflex, goal-based, utility-based, and learning agents -- which directly inform our autonomy level dimension. Their hierarchy maps roughly to our REACTIVE through PROACTIVE levels, though we extend it with NONE (pure scripts) and COLLABORATIVE (multi-agent coordination), which their framework treats only peripherally.

**Wooldridge & Jennings (1995).** In their seminal paper "Intelligent Agents: Theory and Practice," Wooldridge and Jennings propose four properties that distinguish intelligent agents from mere software: (1) *autonomy* -- operating without direct human intervention and having control over their own actions and internal state; (2) *social ability* -- interacting with other agents (and possibly humans) via some communication language; (3) *reactivity* -- perceiving the environment and responding in a timely fashion; and (4) *pro-activeness* -- exhibiting goal-directed behavior by taking initiative. Our autonomy dimension directly encodes their autonomy and pro-activeness properties as a spectrum. Their social ability property maps to our COLLABORATIVE level. Reactivity is captured as a baseline requirement for any category above NONE.

**Franklin & Graesser (1996).** "Is it an Agent, or Just a Program?" provides an early attempt at a formal taxonomy. They define an autonomous agent as "a system situated within and part of an environment that senses that environment and acts on it, over time, in pursuit of its own agenda and so as to effect what it senses in the future." They enumerate properties (reactive, autonomous, goal-oriented, temporally continuous, communicative, learning, mobile, flexible, character) and use subsets of these properties to classify agent types. Our taxonomy extends their approach by grounding it in observable blockchain behavior rather than abstract properties, and by adding the decision model dimension to capture the mechanism of decision-making (deterministic, statistical, LLM, RL, hybrid), which Franklin and Graesser do not address.

**Maes (1995).** Pattie Maes, in "Artificial Life Meets Entertainment: Lifelike Autonomous Agents," takes a bottom-up, behavior-based approach to agent design. She emphasizes that agents should be defined by their observable behavior rather than their internal architecture -- a principle we adopt directly. Our taxonomy is designed so that each category is distinguishable through on-chain behavioral indicators, not by inspecting source code or internal decision processes. This aligns with Maes's position that an agent's "intelligence" is in the eye of the observer.

**Park et al. (2023).** "Generative Agents: Interactive Simulacra of Human Behavior" demonstrates LLM-powered agents that maintain memory, reflect on experiences, and plan actions in a simulated environment. This work is pivotal for our LLM-Powered Agent category: it establishes that LLM-based reasoning can produce agent behaviors (planning, memory, social interaction) that meet the Wooldridge-Jennings criteria. The key question our taxonomy addresses is whether such behaviors are distinguishable on-chain from simpler bot behavior.

**Significant Gravitas -- AutoGPT (2023).** The open-source AutoGPT project demonstrated that LLMs can be wrapped in an autonomous loop (perceive-plan-act-reflect) to accomplish complex tasks with minimal human intervention. AutoGPT and its successors (BabyAGI, CrewAI, LangChain Agents) represent a new class of agent that blurs the line between tool-use and genuine autonomy. Our taxonomy must account for these systems, which typically operate at our PROACTIVE autonomy level with an LLM decision model, but may exhibit ADAPTIVE or even COLLABORATIVE behavior depending on configuration.

---

## 2.2 Agent Definitions in HCI

The HCI community has approached the agent question primarily through the lens of automation levels and human-AI interaction, providing complementary perspectives to the AI/CS definitions.

**Shneiderman (2022).** In *Human-Centered AI*, Ben Shneiderman argues against full autonomy as a design goal, instead advocating for systems that amplify human abilities while maintaining human control. He proposes a two-dimensional framework (human control vs. computer automation) rather than a single spectrum, noting that high automation and high human control are not mutually exclusive. This perspective informs our COLLABORATIVE autonomy level, where agents coordinate with humans or other agents rather than acting unilaterally. Shneiderman's critique also motivates our inclusion of the NONE level -- not as a degenerate case, but as a legitimate and often desirable design choice.

**Parasuraman, Sheridan & Wickens (2000).** "A Model for Types and Levels of Human Interaction with Automation" proposes a 10-level automation taxonomy spanning from fully manual to fully automatic, applied across four information-processing stages: information acquisition, information analysis, decision selection, and action implementation. Our 5-level autonomy dimension (NONE through COLLABORATIVE) is a deliberate simplification optimized for on-chain observability. We collapse their 10 levels because blockchain transactions provide limited visibility into internal information processing -- we can only observe actions and infer decision quality from outcomes. However, their insight that automation can be partial and stage-specific informs our decision model dimension: an agent may use LLM reasoning for decision selection but deterministic logic for action implementation.

**Amershi et al. (2019).** "Guidelines for Human-AI Interaction" (CHI Best Paper) provides 18 design guidelines organized by interaction phase (initially, during interaction, when wrong, over time). Several guidelines are relevant to our taxonomy: G1 (make clear what the system can do) relates to the transparency implications of our categories; G11 (make clear why the system did what it did) connects to the explainability differences between deterministic and LLM-driven agents; G16 (provide global controls) maps to the governance mechanisms in our COLLABORATIVE category. We argue that the appropriate interaction guidelines differ systematically by taxonomy category, which is a contribution to the HCI community.

**Horvitz (1999).** "Principles of Mixed-Initiative User Interaction" establishes foundational principles for systems where humans and AI agents share control. Horvitz emphasizes that agents should (1) develop significant value-added automation, (2) allow efficient direct invocation and termination, (3) maintain working memory of user preferences, and (4) provide transparency about uncertainty. These principles are most relevant to our PROACTIVE and COLLABORATIVE categories, where the agent has sufficient autonomy to benefit from mixed-initiative design. For our NONE and REACTIVE categories, the interaction is sufficiently simple that mixed-initiative considerations do not arise.

---

## 2.3 Blockchain Bots and MEV

A substantial body of empirical research documents the behavior of automated agents on blockchain networks, though this literature rarely engages with formal agent definitions.

**Daian et al. (2020).** "Flash Boys 2.0: Frontrunning in Decentralized Exchanges, Miner Extractable Value, and Consensus Instability" (IEEE S&P) is the foundational work on MEV. Daian et al. document priority gas auctions (PGAs) among competing bots on decentralized exchanges and quantify the value extracted through frontrunning, backrunning, and sandwich attacks. Their characterization of bot behavior -- monitoring the mempool, computing profitable trades, submitting competitive gas bids -- maps directly to our MEV Searcher category (ADAPTIVE autonomy, Hybrid environment, Statistical decision model). Critically, they do not distinguish between "bots" and "agents," treating all automated extractors as equivalent. Our taxonomy provides the missing granularity.

**Qin, Zhou & Gervais (2022).** "Quantifying Blockchain Extractable Value: How Dark is the Forest?" extends the MEV analysis with a systematic framework for quantifying extractable value across DeFi protocols. They classify MEV strategies (arbitrage, liquidation, sandwich) and measure their prevalence on Ethereum. Their strategy classification is orthogonal to our taxonomy: a single MEV Searcher category in our system may employ multiple strategies. However, their empirical data on bot transaction patterns provides ground truth for our on-chain indicators (e.g., gas price distributions, transaction timing, bundle composition).

**Flashbots (2021).** The Flashbots project introduced MEV-Boost and proposer-builder separation (PBS) as infrastructure for ordering transactions outside the public mempool. This infrastructure is significant for our taxonomy because it changes the observability of agent behavior: agents submitting through Flashbots produce transaction bundles that are partially opaque until included in a block. Our on-chain indicators must account for both public mempool agents and private order flow agents. Flashbots data (via MEV-Explore and libMEV) provides labeled datasets of MEV extraction that we use for validation.

**Torres, Camino & State (2021).** "Frontrunner Jones and the Raiders of the Dark Forest: An Empirical Study of Frontrunning on the Ethereum Blockchain" (USENIX Security) provides a detailed empirical analysis of frontrunning bot behavior, including displacement, insertion, and suppression attacks. They identify behavioral signatures (gas price escalation patterns, transaction timing relative to victim transactions) that inform our on-chain indicators for the MEV Searcher category. Their work demonstrates that bot categories can be distinguished empirically from transaction data alone, supporting the feasibility of our behavioral mapping approach.

---

## 2.4 On-chain AI Agents

A new generation of systems explicitly markets itself as "AI agents" on blockchain networks. We survey the most prominent examples.

**Autonolas / OLAS.** Autonolas provides an open-source framework for building autonomous agent services that operate off-chain but execute on-chain. Agents are composed of modular skills and coordinate through a multi-agent consensus protocol. Autonolas agents map to our DeFi Management Agent or COLLABORATIVE categories depending on configuration. Their architecture -- off-chain computation with on-chain settlement -- exemplifies our Hybrid environment type. The OLAS token provides economic incentives for agent operation, creating a market dynamic that influences agent behavior (e.g., agents may prioritize revenue-generating tasks).

**AI16Z / ELIZA.** The ELIZA framework enables LLM-powered agents that can trade tokens, post on social media, and interact with DeFi protocols. These agents operate in our Multi-modal environment (combining on-chain transactions with social media activity) with LLM-driven decision models. They represent the purest instantiation of our LLM-Powered Agent category. Key behavioral characteristics include variable transaction latency (reflecting LLM inference time), context-dependent trade sizing, and natural language-influenced parameter selection. The AI16Z DAO and its associated token (ai16z) created a financial ecosystem around these agents.

**Virtuals Protocol.** Virtuals Protocol enables the tokenization of AI agents -- each agent is associated with a token whose value reflects the agent's perceived utility and performance. Agents on Virtuals (e.g., LUNA, GAME) combine LLM reasoning with on-chain actions, mapping to our LLM-Powered Agent category. The tokenization layer adds a financial feedback loop: an agent's behavior affects its token price, which may in turn affect the agent's available capital and behavior. This reflexivity is a unique feature of on-chain AI agents not captured by traditional agent definitions.

**Fetch.ai.** Fetch.ai proposes "Autonomous Economic Agents" (AEAs) that negotiate and transact on behalf of users. Their agent framework emphasizes peer-to-peer communication protocols and multi-agent coordination, mapping to our COLLABORATIVE autonomy level. Fetch.ai agents operate across on-chain and off-chain environments and employ a variety of decision models. Their conceptual framework aligns with our taxonomy but lacks the empirical grounding in observable behavior that we provide.

**He et al. (2025).** "A Survey of AI Agent Protocols" (arXiv:2601.04583) provides a comprehensive survey of AI agent architectures and proposes a threat model for agent security. They categorize agents by capability (perception, reasoning, action, memory) and identify attack surfaces (prompt injection, tool misuse, memory poisoning). Their threat model is complementary to our taxonomy: our categories describe *what* agents are, while their threat model describes *how* agents can be attacked. We note that the attack surface differs systematically by taxonomy category -- deterministic scripts are immune to prompt injection but vulnerable to parameter manipulation, while LLM-powered agents face the full spectrum of LLM-specific attacks.

---

## 2.5 Gaps in Existing Literature

Our survey reveals three critical gaps that our taxonomy addresses:

**Gap 1: No taxonomy maps AI theory to on-chain observability.** Classical agent definitions (Russell & Norvig, Wooldridge & Jennings, Franklin & Graesser) provide abstract characterizations of agency but do not specify how these properties manifest in observable transaction data. The blockchain bot literature (Daian et al., Qin et al.) documents empirical behavior but does not connect it to formal agent theory. Our taxonomy bridges this divide by defining categories in terms of theoretical properties (autonomy, decision model) and validating them against on-chain behavioral indicators.

**Gap 2: No formal criteria for when a bot becomes an agent.** The term "AI agent" is used loosely in both industry and academia. A simple trading bot executing fixed rules is called an "agent" alongside an LLM-powered system that reasons about market conditions and adapts its strategy. This conflation hinders research, regulation, and user understanding. Our taxonomy provides explicit boundary conditions: a system is a Deterministic Script (not an agent) at NONE autonomy, becomes a Simple Trading Bot (minimal agency) at REACTIVE, and progresses through increasing degrees of agency. The decision model dimension further distinguishes: a deterministic reactive system is a "bot," while a statistically adaptive system begins to warrant the "agent" label.

**Gap 3: No empirical validation of agent categories.** Existing taxonomies (Parasuraman et al., 2000; Franklin & Graesser, 1996) are validated conceptually or through thought experiments but not against real-world behavioral data. The blockchain provides a unique opportunity for empirical validation because all actions are permanently recorded on-chain with precise timestamps and full transaction details. Our validation approach -- mapping taxonomy categories to on-chain features and measuring inter-rater agreement -- is the first empirical validation of an agent taxonomy against naturalistic behavioral data.

---

## Summary Table: Key References

| Reference | Year | Key Contribution | Taxonomy Relevance |
|-----------|------|-----------------|-------------------|
| Russell & Norvig | 2020 | Perception-action agent definition; agent architectures | Autonomy dimension, environment dimension |
| Wooldridge & Jennings | 1995 | Autonomy, social ability, reactivity, pro-activeness | Core autonomy levels (REACTIVE-COLLABORATIVE) |
| Franklin & Graesser | 1996 | Property-based agent taxonomy | Formal taxonomy methodology |
| Maes | 1995 | Behavior-based agent definition | Observable behavior principle |
| Park et al. | 2023 | Generative agents with memory and planning | LLM-Powered Agent category |
| Significant Gravitas (AutoGPT) | 2023 | Autonomous LLM agent loop | LLM-Powered Agent category |
| Shneiderman | 2022 | Human-centered AI, control vs. automation | COLLABORATIVE level, NONE level justification |
| Parasuraman et al. | 2000 | 10-level automation taxonomy | Autonomy dimension calibration |
| Amershi et al. | 2019 | 18 guidelines for human-AI interaction | Interaction design per category |
| Horvitz | 1999 | Mixed-initiative interaction principles | PROACTIVE and COLLABORATIVE design |
| Daian et al. | 2020 | Flash Boys 2.0: MEV on DEXs | MEV Searcher category, on-chain indicators |
| Qin et al. | 2022 | Quantifying blockchain extractable value | Validation data, strategy classification |
| Flashbots | 2021 | MEV-Boost and proposer-builder separation | Observability considerations |
| Torres et al. | 2021 | Frontrunning empirical study | Behavioral signatures for MEV Searcher |
| Autonolas/OLAS | 2023-24 | Autonomous agent services framework | DeFi Management Agent exemplar |
| AI16Z/ELIZA | 2024 | LLM-powered trading agents | LLM-Powered Agent exemplar |
| Virtuals Protocol | 2024 | AI agent tokenization | LLM-Powered Agent + token reflexivity |
| Fetch.ai | 2023 | Autonomous Economic Agents | COLLABORATIVE level exemplar |
| He et al. | 2025 | AI agent protocol survey and threat model | Security implications per category |
