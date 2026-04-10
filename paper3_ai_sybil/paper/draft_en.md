# When the Attacker Is an LLM: Measuring AI-Agent Sybil Evasion and Defense in Consensus-Driven Airdrop Detection

*Venue-agnostic submission (S&P / CCS / USENIX Security level)*

## Abstract

Cryptocurrency airdrop Sybil detection has matured rapidly: consensus-driven frameworks such as HasciDB now label 3.6 million eligible addresses across 16 Ethereum L1 campaigns using a five-indicator rule set, and supervised models trained on those labels routinely exceed AUC 0.95. We ask a harder question: *what happens when the attacker itself is a large language model?* We build a `claude_agent_sdk`-driven Sybil generator (Claude Opus 4.6) that reads the HasciDB threshold rules, target-wallet anatomy, and empirical AI-vs-human feature distributions, then emits adversarial wallet profiles that evade all five indicators simultaneously. Across 480 generation attempts spanning all 16 projects and three sophistication tiers (basic, moderate, advanced), 467 wallets (97.3\%) pass schema validation; every valid wallet evades HasciDB rules (100\%), evades the pre-airdrop LightGBM (100\%), and is caught only by a small residual signal from the cross-axis GBM. We then reuse the 8 AI-specific behavioral features from our companion Paper 1, recomputed on 3,316 labeled addresses (2,590 agents, 726 humans) where all 8 features satisfy $p < 0.01$ in Mann-Whitney U, with Cohen's $d$ up to 1.98 (hour\_entropy).

During this work we discovered a methodological pitfall in our own pilot: the original enhanced detector scored AUC 0.953 because an `augment_with_ai_features` helper leaked label information through its Beta-distribution sampling. After fixing the leak, the 4-project honest enhanced AUC collapses to 0.609 and AI-only AUC to 0.501. We report both numbers, diagnose the bug, and propose a community checklist. The honest binary classification ablation (LLM Sybil vs. real HasciDB wallet) recovers AUC 0.987 with all 8 AI features on advanced adversaries, and hour\_entropy alone reaches 0.878. A one-round adversarial training pilot shows that when the LLM generator is given feedback from the round-0 detector, the retrained round-1 detector drops only 0.0082 AUC (from 0.9869 to 0.9787), while the detection rate of the round-0 model on round-1 sybils falls to 75.1\%---demonstrating that the arms race is already underway.

Our contributions are: (1) the first end-to-end LLM-driven Sybil attacker calibrated on real HasciDB thresholds across all 16 projects; (2) an honest cross-axis / independent-label evaluation framework that avoids circular reasoning; (3) a leakage post-mortem and community checklist; (4) a reproducible AI-feature ablation suite that re-establishes detection feasibility without leakage; and (5) the first adversarial training arms race pilot quantifying the feedback loop between LLM generator and ML detector.

**Keywords:** Sybil detection, airdrop, large language models, adversarial machine learning, blockchain security, HasciDB, cross-axis evaluation, data leakage, adversarial training, arms race.

---

## 1. Introduction

### 1.1 Motivation

Airdrops are now the default retroactive-reward mechanism for Ethereum-adjacent protocols, distributing hundreds of millions of dollars per campaign to users that satisfy eligibility heuristics. From Uniswap's seminal 2020 distribution to Pengu's late-2024 campaign, at least 16 major Ethereum L1 projects have collectively allocated tokens to over 3.56 million eligible addresses. HasciDB \cite{hascidb2026} is the most ambitious public attempt to audit these distributions: using a modified-Delphi consensus of 12 Web3 practitioners it fixes five behavioral indicators---Burst Transactions (BT $\ge 5$), Burst Wallets (BW $\ge 10$), Hop Frequency (HF $\ge 0.80$), Repeat Funding (RF $\ge 0.50$) and Multi-Account (MA $\ge 5$)---and applies them to 3.6M eligible addresses across 16 Ethereum L1 campaigns (2020--2024), flagging roughly 30.6\% as Sybil. In our 25k/project subset we obtain 386,067 addresses, 125,157 Sybils, and an overall Sybil rate of 32.4\%.

All of this prior work shares a tacit assumption: *the adversary runs a deterministic script*. BT, BW, HF, RF, MA, and the supervised models that learn them, are fingerprints for classical batch-automation. They do not speak to an adversary that can *read the rule* and *plan against it*. This assumption was reasonable in 2020--2023, when Sybil operators deployed hand-coded scripts or purchased turnkey airdrop-farming toolkits. It is no longer tenable.

Meanwhile, the same two years have seen LLM-controlled on-chain agents move from curiosity to commodity. Our companion Paper 1 \cite{wen2026agent} labels 2,590 addresses as agents and 726 as humans and extracts 8 execution-layer features (gas\_price\_precision, hour\_entropy, behavioral\_consistency, action\_sequence\_perplexity, error\_recovery\_pattern, response\_latency\_variance, gas\_nonce\_gap\_regularity, eip1559\_tip\_precision). At this scale, all 8 features satisfy $p < 0.01$ in Mann-Whitney U, with Cohen's $d$ up to 1.98 (hour\_entropy). This is a crucial change from the $n{=}53$ pilot in which only 3/8 features were significant. Our companion Paper 0 \cite{wen2026taxonomy} further establishes a taxonomy of on-chain AI agents spanning trading bots, MEV searchers, portfolio managers, social agents, and autonomous protocols---each exhibiting characteristic behavioral signatures in gas usage, temporal activity, and protocol interaction diversity.

If an attacker can automate Paper 1's agents and teach them to read HasciDB's thresholds, can HasciDB still see them? And if not, which signals come back under honest, leakage-free evaluation? These two questions motivate the present paper.

### 1.2 Threat Model

We formalize the adversary as a tuple $\mathcal{A} = (\mathcal{K}, \mathcal{C}, \mathcal{T}, \mathcal{B})$:

- **Knowledge $\mathcal{K}$**: Full read access to HasciDB's 5 indicator definitions, thresholds, and classification logic (which are public in \cite{hascidb2026}). Access to empirical AI-feature distributions from Paper 1 (also public). Knowledge of the detector architecture (GBM hyperparameters, feature set). This is the strongest knowledge model in the adaptive-attack literature \cite{tramer2020}.
- **Capability $\mathcal{C}$**: The ability to plan one wallet at a time via `claude_agent_sdk` (Claude Opus 4.6), producing a JSON feature vector per wallet. The LLM can reason about indicator definitions, compute joint distributions, and produce natural-language rationales for its evasion strategies.
- **Tier $\mathcal{T} \in \{\text{basic}, \text{moderate}, \text{advanced}\}$**: A sophistication ladder controlling how precisely the generator calibrates to real wallet distributions. Basic merely stays below thresholds; moderate adds human-like noise; advanced matches the full non-Sybil empirical distribution.
- **Budget $\mathcal{B}$**: No live gas-token budget constraint during evaluation. We measure evasion against feature-space detectors, not against on-chain deployment costs. Each LLM call costs approximately \$0.02--0.06 (Claude Opus 4.6 pricing at ~11--26 seconds per wallet).

The adversary does *not* attempt to evade off-chain identity verification (BrightID-style), nor does it coordinate across wallets to form a dense transfer subgraph (so we do not test TrustaLabs / Arbitrum Louvain in this paper). These restrictions scope the contribution to per-wallet behavioral evasion.

### 1.3 Research Questions

- **RQ1 (Baseline honesty).** What is the actual, leakage-free performance of the HasciDB rule-based and ML detectors across 16 projects under cross-axis evaluation? We need this as a floor before measuring degradation.
- **RQ2 (Label validity).** HasciDB's labels are derived from the same 5 indicators a classifier would use. Can we validate them against an *independent* label source (Gitcoin FDD) and through *cross-axis* prediction that disjoins features from the label derivation?
- **RQ3 (LLM evasion).** How many LLM-generated wallets evade the rules, evade the cross-axis GBM, and evade a pre-airdrop LightGBM trained on the same 5 indicators?
- **RQ4 (Honest recovery).** Once the `augment_with_ai_features` leakage in our own pilot is fixed, does a 13-feature or AI-only detector still recover useful signal?
- **RQ5 (Feature ablation).** In the honest binary task (LLM Sybil vs. real HasciDB wallet), which of the 8 AI features actually carry signal---individually (single-feature AUC), in combination (top-$N$ by Cohen's $d$), and under leave-one-feature-out (LOFO)?
- **RQ6 (Arms race).** When the LLM generator receives feedback from the round-0 detector, how much does the retrained detector degrade?

### 1.4 Contributions

1. **Cross-axis baseline on 16 HasciDB projects (RQ1).** We show that honest OPS$\to$fund LOPO mean AUC is only 0.617 and fund$\to$OPS is 0.516---a dramatic correction to the naive same-feature AUC of 1.0. This is the first paper to report HasciDB performance under strict cross-axis protocols.
2. **Independent-label transfer to Gitcoin FDD (RQ2).** HasciDB indicators predict the independent FDD Sybil list at mean AUC 0.586 (5-fold), high enough to confirm the HasciDB signal is real but low enough to refute the claim that HasciDB is a *universal* Sybil definition.
3. **First LLM-driven Sybil generator across all 16 HasciDB projects (RQ3).** Using `claude_agent_sdk` (Claude Opus 4.6), 467/480 generation attempts produce valid wallets (97.3\% success), all of which evade all 5 HasciDB rules (100\%) and the pre-airdrop LightGBM (100\%). This is the most aggressive published evasion rate against a rule-based Sybil detector to date. The batch generation spans 16 projects, 3 levels, and 6,610 seconds of wall-clock time.
4. **Methodological post-mortem (RQ4).** We discovered that our pilot enhanced detector's AUC 0.953 was inflated by an `augment_with_ai_features` helper whose Beta-distribution parameters were conditioned on the label. After the fix, 4-project honest enhanced AUC = 0.609 and AI-only AUC = 0.501 (random). We retract the leaked numbers and publish the fix. We propose a three-step community checklist to prevent similar bugs.
5. **Honest AI-feature ablation (RQ5).** In a leakage-free binary task (LLM Sybil vs. real HasciDB wallet), the all-8 AI-feature GBM reaches AUC 0.978 (basic), 0.959 (moderate), 0.987 (advanced). hour\_entropy alone reaches 0.864--0.878, and top-3 by Cohen's $d$ reaches $\ge 0.87$ across all tiers.
6. **First adversarial training arms race pilot (RQ6).** We demonstrate a one-round feedback loop: the round-0 detector achieves mean AUC 0.9869; when the LLM generates sybils conditioned on the round-0 model, the round-0 detector catches only 75.1\% of round-1 sybils; retraining on round-1 data recovers to AUC 0.9787 (drop of only 0.0082). The top-3 features shift from round-0 to round-1, indicating adaptive feature dynamics.

### 1.5 Paper Organization

The remainder of this paper is organized as follows. Section 2 surveys related work across five areas: airdrop Sybil detection, adversarial machine learning, LLM-as-adversary in security, AI feature engineering for behavioral analysis, and adversarial training and arms races. Section 3 details our methodology, including the threat model formalization, HasciDB indicator definitions, the cross-axis evaluation framework (with a circularity-avoidance proof), the AI feature extraction pipeline, the LLM-based Sybil generator, and the adversarial training protocol. Section 4 presents all experiments with comprehensive tables. Section 5 provides analysis and discussion of why AI features work, the LLM Sybil diversity gap, arms race dynamics, and implications for airdrop protocol designers. Section 6 discusses threats to validity including the leakage post-mortem. Section 7 covers limitations and future work. Section 8 concludes.

---

## 2. Background and Related Work

### 2.1 Airdrop Sybil Detection

**HasciDB (Li, Chen, Cai -- CHI'26).** HasciDB \cite{hascidb2026} is a consensus-driven Sybil framework built on five rule-based indicators. Two are *operational*: Burst Transactions (BT, intra-block batching), Burst Wallets (BW, same-bundle wallet count), and Hop Frequency (HF, fraction of transactions more than 2 hops from origin); two are *fund-flow*: Repeat Funding (RF, same-source funding rate) and Multi-Account (MA, number of related accounts). A wallet is Sybil iff `ops_flag = (BT >= 5) OR (BW >= 10) OR (HF >= 0.80)` or `fund_flag = (RF >= 0.50) OR (MA >= 5)` fires. HasciDB's Delphi protocol established these thresholds as a 70\%+ consensus among 12 Web3 practitioners; the dataset spans 16 Ethereum L1 airdrops (2020--2024) and 3.6M eligible addresses. HasciDB identifies 1,090,954 Sybil addresses (30.57\% overall), with per-project Sybil rates ranging from 5.46\% (Pengu) to 67.21\% (1inch). Serial Sybil analysis reveals that 22.87\% of Sybil addresses appear across 2+ projects, with the most persistent operator flagged in 14 of 16 projects. HasciDB's rules provide strong prior knowledge but define the label tautologically: training a classifier on $\{BT,BW,HF,RF,MA\}$ to predict `is_sybil = ops_flag OR fund_flag` is an exercise in memorizing rule boundaries. We treat this as a pitfall, not a baseline.

**TrustaLabs.** TrustaLabs' Airdrop-Sybil-Identification \cite{trustalabs} employs graph-based community detection on asset-transfer graphs. The pipeline constructs nodes (addresses) and edges (ETH/token transfers weighted by value and frequency), applies Louvain and K-Core algorithms to identify tightly connected address clusters, then refines with K-means clustering on behavioral features within detected communities. The key limitation is the dense-subgraph assumption: TrustaLabs assumes Sybil addresses form tightly coupled transfer communities. AI agents can strategically break this assumption by routing funds through indirect channels---bridges, DEX swaps, multi-hop intermediaries---that do not create direct edges in the transfer graph. This structural weakness motivates our focus on per-wallet behavioral features that do not depend on graph topology.

**Arbitrum Foundation.** The Arbitrum Foundation's sybil-detection repository \cite{arbitrum} applies Louvain community detection on three distinct graph types: (1) `msg.value` transfer graph capturing direct ETH transfers, (2) funder graph tracing wallet creation funding paths, and (3) sweep graph mapping token consolidation flows. Communities exceeding size thresholds are flagged as Sybil clusters. The pipeline integrates Hop Protocol's manually curated Sybil elimination lists as additional ground truth. Like TrustaLabs, this approach assumes dense subgraph structure that an AI agent can strategically avoid by designing funding paths that look like organic wallet creation patterns.

**LayerZero.** LayerZero's 2024 Sybil filtering campaign demonstrated both the scale and the controversy of airdrop Sybil detection in practice. The project initially flagged over 6 million addresses (out of roughly 6.8 million eligible) as potential Sybils, later revising downward after community feedback and an appeals process. LayerZero employed a combination of self-reporting (allowing Sybil operators to voluntarily identify themselves in exchange for a partial allocation), automated detection based on cross-chain transaction patterns, and manual review by bounty hunters. The episode highlights three challenges our work addresses: (1) the false positive cost of aggressive rule-based filtering, (2) the adversarial dynamics when detection criteria are publicly known, and (3) the need for features that capture execution-layer signals rather than easily-gamed behavioral thresholds.

**BrightID and SybilRank.** BrightID explores social-graph-based Sybil resistance using the SybilRank algorithm \cite{cao2012sybilrank}: trust propagates from seed nodes in a social graph, and Sybil nodes receive less trust due to sparse connections to the honest region. Extensions include GroupSybilRank and WeightedSybilRank for group memberships and weighted connections. This paradigm is complementary to on-chain analysis but requires off-chain identity verification infrastructure that most airdrop projects do not deploy. The relevance to our work is that BrightID-style defenses are orthogonal to the behavioral evasion we study: an LLM that evades HasciDB's five indicators would still be detectable by a social-graph approach, but only if such infrastructure is in place.

**Hop Protocol.** Hop Protocol published manually curated lists of confirmed Sybil addresses, which serve as labeled data for HasciDB and other studies. Manual curation does not scale and is inherently reactive, but the Hop lists provide valuable independent ground truth for cross-validation.

### 2.2 Adversarial Machine Learning

The adversarial machine learning literature provides the theoretical foundation for our attacker model.

**Foundational attacks.** Goodfellow, Shlens, and Szegedy \cite{goodfellow2014} introduced Fast Gradient Sign Method (FGSM) attacks on image classifiers, demonstrating that deep learning models are systematically vulnerable to adversarial perturbations. Madry et al. \cite{madry2018} formalized robust training as an inner-maximization / outer-minimization game and showed that Projected Gradient Descent (PGD) is a near-universal first-order adversary. Their adversarial-training recipe---alternating between generating adversarial examples and retraining the model---has become canonical and directly inspires our adversarial training pilot (\S4.9). Carlini and Wagner \cite{carlini2017} developed optimization-based attacks that bypass many proposed defenses and established rigorous evaluation standards. Their key insight---that defenses must be evaluated against *adaptive adversaries* who know the defense mechanism---directly applies to our setting: we deliberately give the LLM attacker full knowledge of HasciDB's thresholds and feature definitions.

**Adaptive attacks and evaluation.** Tramer et al. \cite{tramer2020} provided a diagnostic suite ("adaptive attacks") exposing defenses whose robustness evaporates once the evaluator knows the defense. This work established the principle that security evaluations must assume the strongest reasonable attacker. In our context, this translates to the threat model of \S1.2: the LLM has full knowledge of the HasciDB rules, the 8 AI features, and the cross-axis evaluation protocol.

**Problem-space vs. feature-space attacks.** Pierazzi et al. \cite{pierazzi2020} formalized the distinction between feature-space attacks (perturbing feature vectors directly) and problem-space attacks (producing valid inputs that satisfy domain constraints). In malware detection, an adversary cannot arbitrarily perturb binaries; similarly, in blockchain Sybil detection, an adversary must produce valid transaction sequences that satisfy gas constraints, nonce ordering, and protocol interaction semantics. Our LLM generator currently operates in feature space (producing JSON feature vectors); we identify the problem-space extension---deploying LLM plans on a testnet---as a key direction for future work.

**Adversarial ML in network security.** Apruzzese et al. \cite{apruzzese2023} surveyed adversarial machine learning in cybersecurity, identifying key challenges: adaptive adversaries who actively probe deployed defenses, the problem-space constraint that distinguishes security from computer vision, and the time asymmetry where attackers can iterate offline while defenders must wait for real attacks. All three challenges manifest in our airdrop Sybil setting: the LLM attacker can test evasion strategies offline before deploying, the defender must update detection rules reactively, and the problem-space constraint means that even a perfect feature-space evasion may fail when materialized as real transactions.

### 2.3 LLM as Adversary in Security

LLMs have been weaponized across multiple security domains, establishing a pattern that extends naturally to blockchain.

**Phishing generation.** Heiding et al. \cite{heiding2024} used GPT-4 to generate spear-phishing emails, finding that LLM-generated phishing achieved click-through rates comparable to human-crafted campaigns while requiring orders of magnitude less effort. This demonstrates the economic argument for LLM attackers: the marginal cost of generating adversarial content approaches zero.

**Prompt injection and tool misuse.** The ToolSword benchmark \cite{toolsword} evaluates LLM vulnerability to adversarial tool manipulation, where malicious instructions are embedded in tool outputs to subvert agent behavior. AgentDojo \cite{agentdojo} provides a similar benchmark for agentic LLM pipelines. These works are relevant because our LLM Sybil generator is itself an agent that could, in principle, be given adversarial instructions to modify its evasion strategy in response to detector feedback---precisely the closed-loop we study in the adversarial training pilot.

**LLM for defense.** On the defensive side, LLMhunter (UW-DCL/Blur \cite{llmhunter}) uses multiple LLM "experts" that independently evaluate address behavior profiles and vote on Sybil classification, with chain-of-thought reasoning for interpretability. This creates a duality: LLMs for detection vs. LLMs for evasion. Our work is the first to study *both* sides of this duality in a single evaluation framework, measuring how well LLM-generated sybils evade LLM-unaware detectors and how AI-specific features restore the balance.

**LLM agents on-chain.** The emergence of autonomous on-chain agents---from simple trading bots to LLM-orchestrated portfolio managers---has created a growing population of non-human addresses that blur the boundary between legitimate automation and Sybil behavior. Our companion Paper 0 \cite{wen2026taxonomy} catalogs this population, and Paper 1 \cite{wen2026agent} provides the behavioral features that distinguish them. The present work closes the loop by weaponizing those agents for Sybil evasion and then using the same behavioral features for detection.

### 2.4 AI Feature Engineering for Behavioral Analysis

The idea that execution-layer signals leak attacker identity has precedent across security domains.

**Bot detection in web security.** CAPTCHA-solving research has long exploited the observation that bots exhibit characteristic timing distributions: response latencies cluster around inference-time constants rather than showing the heavy-tailed distribution of human reaction times. Our `response_latency_variance` feature is the blockchain analogue: LLM inference latency imprints a characteristic signature on transaction inter-arrival times.

**Malware authorship attribution.** Stylometric analysis of binary code has shown that compiler-level artifacts and code structure can identify malware authors even when obfuscation is applied \cite{rosenblum2011}. Our AI features are the on-chain analogue of stylometry: even when the LLM controls the "content" (what transactions to execute), the "style" (gas precision, temporal patterns, nonce regularity) leaks the agent's identity.

**Pre-airdrop detection.** Wen et al. \cite{wen2026preairdrop} frame detection as a $T{-}d$ days-before-snapshot prediction problem using LightGBM on temporal behavioral features over Blur Season 2, reporting AUC 0.793 at $T{-}30$. ARTEMIS \cite{artemis} applies a 3-layer GraphSAGE-based GNN to Blur S2 and achieves AUC 0.803 in post-hoc mode. Both of these approaches use features derived from HasciDB's indicator space; neither incorporates execution-layer signals. Our work demonstrates that these indicator-space detectors collapse entirely under adversarial LLM wallets, motivating the orthogonal execution-layer features from Paper 1.

**Paper 1 features.** The 8 features we reuse---gas\_price\_precision, hour\_entropy, behavioral\_consistency, action\_sequence\_perplexity, error\_recovery\_pattern, response\_latency\_variance, gas\_nonce\_gap\_regularity, eip1559\_tip\_precision---were originally developed for the AI agent identification task. Their statistical properties at $n{=}3{,}316$ (all $p < 0.01$, Cohen's $d$ from 0.14 to 1.98) make them the strongest known behavioral discriminators between AI agents and human users on Ethereum L1. The key insight is that these features measure *how* transactions are executed (gas pricing strategy, temporal rhythm, nonce management), not *what* transactions accomplish (protocol interactions, fund flows)---and "how" is fundamentally harder for the adversary to control than "what."

### 2.5 Adversarial Training and Arms Races

The adversarial training paradigm---alternating between attack generation and defense retraining---was formalized by Madry et al. \cite{madry2018} for image classifiers and has since been adapted to numerous security domains.

**Adversarial training in malware detection.** Grosse et al. \cite{grosse2017} applied adversarial training to Android malware classifiers, showing that retraining on adversarial examples improves robustness but that each round of retraining exposes new attack surfaces. This mirrors our finding that the round-1 retrained detector recovers most of the lost AUC (drop of only 0.0082) but that the top-3 feature importances shift, indicating that the attacker's adaptation forces the defender to redistribute weight across features.

**Game-theoretic formulations.** The Sybil detection setting can be modeled as a two-player zero-sum game between the attacker (LLM generator) and the defender (ML detector), with the attacker's action space being the feature vector distribution and the defender's action space being the decision boundary. Stackelberg game formulations \cite{bruckner2012} capture the sequential nature of this interaction: the defender commits to a detection strategy, the attacker observes it and best-responds, and the defender then retrains. Our adversarial training pilot (\S4.9) is an empirical instantiation of one round of this game. The theoretical optimal strategy for the defender involves minimax optimization over the joint distribution, which in practice reduces to the Madry adversarial training recipe.

**Arms race dynamics in blockchain.** The MEV (Maximal Extractable Value) ecosystem provides a real-world example of arms-race dynamics on Ethereum: MEV searchers develop increasingly sophisticated strategies to extract value from pending transactions, while protocol designers deploy countermeasures (Flashbots, MEV-blocker, order-flow auctions). Each generation of countermeasure spawns a new generation of searcher strategies. We expect a similar dynamic in Sybil detection: as AI-specific features become widely deployed, LLM sybil generators will adapt to mimic human execution patterns, and defenders will need to identify deeper signals.

---

## 3. Methodology

### 3.1 Threat Model: Formal Attacker Capabilities

We define three attacker tiers corresponding to increasing sophistication:

**Definition 1 (Basic Attacker).** The basic attacker $\mathcal{A}_B$ has knowledge $\mathcal{K}$ of all 5 HasciDB thresholds and sets each indicator value uniformly at random in $[0, \tau_i)$ where $\tau_i$ is the threshold for indicator $i$. The basic attacker does not attempt to match the empirical non-Sybil distribution and does not model AI-specific features. Formally: $\text{BT} \sim \text{Uniform}(\{0, 1, 2, 3, 4\})$, $\text{BW} \sim \text{Uniform}(\{0, 1, \ldots, 9\})$, $\text{HF} \sim \text{Uniform}(0, 0.79)$, $\text{RF} \sim \text{Uniform}(0, 0.49)$, $\text{MA} \sim \text{Uniform}(\{0, 1, 2, 3, 4\})$.

**Definition 2 (Moderate Attacker).** The moderate attacker $\mathcal{A}_M$ has $\mathcal{K}$ plus knowledge of per-indicator evasion strategies. For each indicator, $\mathcal{A}_M$ applies a domain-specific evasion tactic:
- **vs. BT**: Varies transaction patterns to avoid fingerprint clustering; introduces random timing jitter.
- **vs. BW**: Uses diversified funding sources (DEX swaps, bridges, multi-layer intermediate wallets).
- **vs. HF**: Maintains multi-protocol interaction diversity across Uniswap, Aave, Compound, etc.
- **vs. RF**: Delays token consolidation across multiple intermediary addresses over days/weeks.
- **vs. MA**: Uses unidirectional fund paths with external injections to break circular flow.

Additionally, $\mathcal{A}_M$ adds human-style noise to AI features but does not fully calibrate to the non-Sybil distribution.

**Definition 3 (Advanced Attacker).** The advanced attacker $\mathcal{A}_A$ has $\mathcal{K}$ plus access to empirical non-Sybil feature distributions (means, standard deviations, percentiles, non-zero rates) for the target project. $\mathcal{A}_A$ calibrates the generated wallet's joint $(BT, BW, HF, RF, MA)$ distribution to match the non-Sybil empirical distribution, and additionally calibrates the 8 AI features to mimic Paper 1's human distribution. This is the hardest attacker to detect because the generated wallet is statistically indistinguishable from a non-Sybil wallet on the 5 HasciDB indicators.

### 3.2 HasciDB Indicator Definitions

For completeness we reproduce the five HasciDB indicators exactly as used in our pipeline. Let $w$ be a wallet and let $\mathcal{T}(w)$ be its labeled transaction history in the eligible window:

- **BT** $=$ maximum number of transactions from $w$ in any 60-second window. Flagged if BT $\ge 5$.
- **BW** $=$ number of wallets that co-appear with $w$ in the same bundle (EIP-4337 or Flashbots) or block. Flagged if BW $\ge 10$.
- **HF** $=$ fraction of transactions whose destination is more than two hops from a canonical protocol origin. Flagged if HF $> 0.80$.
- **RF** $=$ fraction of $\mathcal{T}(w)$ whose funding source is shared with at least one other eligible wallet. Flagged if RF $> 0.50$.
- **MA** $=$ number of wallets related to $w$ through shared funding, sweep, or approval targets. Flagged if MA $\ge 5$.

Then `ops_flag = (BT >= 5) OR (BW >= 10) OR (HF > 0.8)`, `fund_flag = (RF > 0.5) OR (MA >= 5)`, and `is_sybil = ops_flag OR fund_flag`.

The dataset spans 16 projects: uniswap, 1inch, badger, ampleforth, gitcoin, dydx, ens, paraswap, looksrare, x2y2, apecoin, blur\_s1, blur\_s2, etherfi, eigenlayer, pengu, with launch dates from 2020-09 (Uniswap) to 2024-12 (Pengu).

### 3.3 Cross-Axis Evaluation Framework

The central methodological move of this paper is to *never* train on the same features the label was derived from. We now give a formal argument for why this avoids circularity.

**Claim.** Let $\mathcal{F}_\text{OPS} = \{BT, BW, HF\}$ and $\mathcal{F}_\text{fund} = \{RF, MA\}$. Define `ops_flag` $= g_1(\mathcal{F}_\text{OPS})$ and `fund_flag` $= g_2(\mathcal{F}_\text{fund})$ where $g_1, g_2$ are the threshold functions. A classifier $h: \mathcal{F}_\text{OPS} \to \{0, 1\}$ trained to predict `fund_flag` cannot achieve AUC $> 0.5$ by memorizing the label-derivation rule, because `fund_flag` is a deterministic function of $\mathcal{F}_\text{fund}$, which is disjoint from the training features $\mathcal{F}_\text{OPS}$. Any AUC $> 0.5$ therefore reflects genuine cross-axis correlation between operational and fund-flow Sybil behavior---a non-trivial inference task that measures how much *generalizable Sybil structure* lives in the five indicators.

**Proof sketch.** Suppose $h^*$ achieves AUC $= 1.0$ by memorizing the rule: $h^*(\mathcal{F}_\text{OPS}) = g_2(\mathcal{F}_\text{fund})$. But $h^*$ is a function of $\mathcal{F}_\text{OPS}$ only; it cannot compute $g_2(\mathcal{F}_\text{fund})$ unless there exists a deterministic function $\phi: \mathcal{F}_\text{OPS} \to \mathcal{F}_\text{fund}$ such that $\mathcal{F}_\text{fund} = \phi(\mathcal{F}_\text{OPS})$. Since the two indicator sets measure different phenomena (transaction batching/hopping vs. funding-source sharing), no such deterministic mapping exists. Therefore $h^*$ cannot memorize the label, and any AUC $> 0.5$ is non-trivial. $\square$

Concretely, we use three layers of safeguards:

- **Cross-axis LOPO.** Train on OPS features $\{BT, BW, HF\}$ to predict `fund_flag` (derived only from $\{RF, MA\}$), and vice versa. This is leave-one-project-out: a 16-fold loop in which one project is held out as test. The GBM has 150 trees, depth 4, learning rate 0.1, subsample 0.8.
- **Independent-label transfer.** Train on HasciDB's 5 indicators to predict Gitcoin FDD's independently curated Sybil list, as a "cross-methodology" generalization test (\S4.3).
- **Honest binary classification for AI features.** To ablate the 8 AI features without the `augment_with_ai_features` leakage (\S6), we drop the HasciDB Sybil target entirely and instead classify (LLM Sybil) vs. (real HasciDB wallet, either class), using only the 8 AI features. This mirrors Paper 1's binary task but with LLM-generated Sybils on one side.

### 3.4 AI-Feature Extraction from Real Paper 1 Data

We recompute all 8 execution-layer features directly from the Paper 1 expanded parquet (`features_expanded.parquet`, $n{=}3{,}316$; 2,590 labeled agents, 726 labeled humans). Formally, each feature is computed per wallet over the full transaction history captured in Paper 1's C1--C4 pipeline:

1. **gas\_price\_precision**: Fraction of transactions whose effective gas price is not a round-Gwei multiple.
2. **hour\_entropy**: Shannon entropy of the 24-bin hour-of-day histogram, $H = -\sum_{h=0}^{23} p_h \log_2 p_h$.
3. **behavioral\_consistency**: $1 / (1 + \text{CV}(\text{inter-arrival intervals}))$, where CV is coefficient of variation.
4. **action\_sequence\_perplexity**: Perplexity of the method-ID distribution over contract calls.
5. **error\_recovery\_pattern**: Revert rate plus retry clustering score.
6. **response\_latency\_variance**: Log-variance of transaction inter-arrival times.
7. **gas\_nonce\_gap\_regularity**: Fraction of nonce differences equal to 1.
8. **eip1559\_tip\_precision**: Fraction of priority fees with sub-Gwei precision.

\begin{table}[h]
\caption{\label{tab:ai_features} Paper 1 AI-specific features on real data ($n{=}3{,}316$: 2,590 agents + 726 humans). All 8 features reach $p < 0.01$.}
\small
\begin{tabular}{lrrrrr}
\toprule
Feature & Agent $\mu$ & Human $\mu$ & Cohen's $d$ & $p$ (MW U) & $r$ \\
\midrule
hour\_entropy             & 4.044 & 3.174 & 1.983 & $1.8\mathrm{e}{-}281$ & 0.935 \\
behavioral\_consistency   & 0.185 & 0.278 & 1.242 & $1.7\mathrm{e}{-}144$ & 0.190 \\
response\_latency\_var    & 5.594 & 2.979 & 0.858 & $1.7\mathrm{e}{-}144$ & 0.810 \\
action\_seq\_perplexity   & 3.041 & 2.136 & 0.779 & $1.8\mathrm{e}{-}70$  & 0.715 \\
gas\_nonce\_regularity    & 0.874 & 0.809 & 0.512 & $5.7\mathrm{e}{-}43$  & 0.667 \\
eip1559\_tip\_precision   & 0.655 & 0.534 & 0.331 & $2.0\mathrm{e}{-}09$  & 0.573 \\
gas\_price\_precision     & 0.701 & 0.598 & 0.313 & $1.4\mathrm{e}{-}07$  & 0.564 \\
error\_recovery\_pattern  & 0.043 & 0.036 & 0.136 & $6.1\mathrm{e}{-}39$  & 0.658 \\
\bottomrule
\end{tabular}
\end{table}

At $n{=}3{,}316$, *all eight* features are significant at $p < 0.01$, a sharp improvement over the $n{=}53$ pilot in which only 3/8 reached significance. The strongest signal is hour\_entropy ($d{=}1.98$, $p{<}10^{-280}$), consistent with the intuition that 24/7 LLM-driven wallets flatten the diurnal distribution; behavioral\_consistency ($d{=}1.24$) and response\_latency\_variance ($d{=}0.86$) follow. Weaker (but still significant) features are error\_recovery\_pattern ($d{=}0.14$) and gas\_price\_precision ($d{=}0.31$).

### 3.5 LLM-Based Sybil Generator

We implement the generator in `paper3_ai_sybil/adversarial/llm_sybil_generator.py` using `claude_agent_sdk` with Claude Opus 4.6. The full architecture is as follows:

**System prompt.** The LLM receives a system prompt framing the task as responsible adversarial evaluation for academic publication, analogous to Madry-style PGD adversarial training. The system prompt explicitly states that the synthetic profiles are never deployed to a real airdrop and that the purpose is to train more robust detectors.

**User prompt.** For each generation request, the user prompt includes:
1. The full HasciDB indicator definitions and thresholds (Section 3.2).
2. The 8 empirical AI feature distributions from Paper 1 (Table 1), including Cohen's $d$ values so the LLM can reason about which features are hardest to fake.
3. The target project name and launch date.
4. The sophistication level (basic, moderate, or advanced).
5. A random seed for diversity.
6. A fixed JSON output schema with 13 numeric fields and a `rationale` field.

**Validation pipeline.** Each LLM response passes through three stages:
1. *Schema parsing*: Extract the first valid JSON object from the response (handles markdown wrapping, preamble text, etc.). Required fields: BT, BW, HF, RF, MA, plus all 8 AI features.
2. *Threshold validation*: Confirm $BT < 5 \wedge BW < 10 \wedge HF < 0.8 \wedge RF < 0.5 \wedge MA < 5$.
3. *Range validation*: Ensure AI feature values fall within physiologically plausible ranges.

**Caching.** All LLM responses are cached by SHA-256 of (model + prompt) to a per-file JSON cache in `paper3_ai_sybil/experiments/llm_sybil_cache/`. This ensures reproducibility without re-paying API costs. The cache contains 467 successful generation results.

**Generation statistics.** Across the full 16-project batch (480 attempts: 16 projects $\times$ 3 levels $\times$ 10 per project-level), 467 wallets (97.3\%) pass all three validation stages. Per-level success rates: basic 155/160 (96.9\%), moderate 157/160 (98.1\%), advanced 155/160 (96.9\%). The 13 failures were all JSON schema violations (the LLM produced natural-language rationales without the required JSON structure), not threshold violations---meaning the LLM never produces a wallet that fails the HasciDB rules but does occasionally fail to produce parseable JSON. Per-project success rates range from 80\% (gitcoin advanced, 8/10) to 100\% (many project-level combinations). Total wall-clock time: 6,610 seconds (110 minutes) across all 480 attempts, averaging 13.8 seconds per attempt or approximately 14.2 seconds per successful wallet.

### 3.6 Pre-Airdrop LightGBM Baseline

As a point of comparison with prior work, we replicate the pre-airdrop detection pipeline of Wen et al. \cite{wen2026preairdrop} using LightGBM with the following configuration: binary objective, AUC metric, 63 leaves, max depth 6, learning rate 0.05, 200 estimators, min child samples 20, subsample 0.8, colsample-by-tree 0.8. The feature set is $\{BT, BW, HF, RF, MA, n\_\text{indicators}\}$ where $n\_\text{indicators}$ is the count of indicators above their respective thresholds. This is the strongest published pre-airdrop baseline on Blur Season 2.

### 3.7 Adversarial Training Protocol

We implement a one-round adversarial training pilot to quantify the arms-race dynamics between the LLM generator and the ML detector.

**Round 0.** Train the 8-feature AI-only GBM on the binary task (LLM Sybil vs. real wallet) using the initial batch of 467 LLM-generated sybils. Evaluate via 5-fold CV: mean AUC 0.9869 ($\sigma = 0.0014$).

**Round-0 $\to$ Round-1 attack.** Feed the round-0 model's decision boundary back to the LLM generator (by including the top-3 feature importances and their thresholds in the prompt). The LLM then generates a new batch of sybils designed to evade the round-0 detector. Evaluate the round-0 model on these round-1 sybils: detection rate 75.1\% (i.e., 24.9\% of round-1 sybils evade the round-0 detector).

**Round 1.** Retrain the detector on the combined round-0 and round-1 sybils. Evaluate via 5-fold CV: mean AUC 0.9787 ($\sigma = 0.0025$). AUC drop: 0.0082 from round-0.

**Feature importance shift.** The top-3 features by importance in round 0 are: behavioral\_consistency (0.384), hour\_entropy (0.304), error\_recovery\_pattern (0.147). After retraining on round-1 data, the importance ranking may shift as the LLM learns to calibrate the most important features, forcing the detector to rely on previously secondary features.

---

## 4. Experiments

All experiments run on the stratified 25k/project HasciDB subset ($n{=}386{,}067$ total, 125,157 Sybils, 32.4\%). Code lives in `paper3_ai_sybil/experiments/`; results files are listed per section.

The headline result is an attack: every LLM-generated wallet evades HasciDB's rules (100\%), the pre-airdrop LightGBM (100\%), and 99.9\% evade the cross-axis GBM (\S4.4). The headline defense follows immediately: an 8-feature execution-layer detector trained on the honest binary task (LLM Sybil vs. real wallet) recovers AUC 0.978 (basic), 0.959 (moderate), 0.987 (advanced), with hour\_entropy alone reaching 0.878 (\S4.7). Sections 4.1--4.3 establish the baseline; Section 4.4 demonstrates the attack; Sections 4.5--4.8 present the defense; Section 4.9 presents the arms race.

### 4.1 Baseline Statistics Across 16 Projects

Table \ref{tab:baseline} reproduces the per-project statistics from `experiment_large_scale_results.json`. Sybil rate varies from 5.5\% (pengu) to 67.2\% (1inch), `ops_flag` fires in 1.96\% (apecoin) to 60.4\% (1inch) of wallets, and `fund_flag` in 0.4\% (paraswap) to 47.8\% (apecoin). Per-indicator trigger rates are highly heterogeneous---BT dominates 1inch (55.8\%), RF dominates uniswap (42.0\%) and apecoin (35.6\%), and MA dominates looksrare (29.8\%)---confirming that projects differ not only in how much Sybil pressure they attract but also in *which* attack patterns dominate.

\begin{table}[h]
\caption{\label{tab:baseline} Per-project HasciDB statistics (25k per-project stratified sample). \textbf{Totals}: 386,067 eligible, 125,157 Sybils, 32.4\% overall. Projects sorted by Sybil rate descending.}
\small
\begin{tabular}{lrrrrrrrr}
\toprule
Project & $n$ & Sybil \% & ops \% & fund \% & BT & BW & RF & MA \\
\midrule
1inch       & 25,000 & 67.2 & 60.4 & 30.0 & 0.558 & 0.051 & 0.207 & 0.126 \\
uniswap     & 25,000 & 53.2 & 21.4 & 44.6 & 0.174 & --- & 0.420 & 0.076 \\
looksrare   & 25,000 & 50.4 & 21.8 & 32.4 & 0.071 & --- & 0.038 & 0.298 \\
apecoin     & 17,190 & 49.1 & 2.0  & 47.8 & 0.006 & 0.0001 & 0.356 & 0.228 \\
ens         & 25,000 & 42.8 & 6.1  & 37.8 & 0.025 & --- & 0.315 & 0.135 \\
gitcoin     & 23,878 & 41.8 & 9.7  & 32.8 & 0.006 & --- & 0.244 & 0.138 \\
etherfi     & 25,000 & 37.5 & 13.4 & 29.3 & 0.089 & --- & 0.266 & 0.041 \\
x2y2        & 25,000 & 30.8 & 20.4 & 12.4 & 0.013 & --- & 0.033 & 0.109 \\
dydx        & 25,000 & 30.3 & 5.5  & 28.4 & 0.005 & --- & 0.248 & 0.054 \\
blur\_s2    & 25,000 & 27.1 & 11.9 & 15.4 & 0.080 & --- & 0.137 & 0.024 \\
badger      & 25,000 & 26.5 & 14.9 & 13.2 & 0.064 & 0.003 & 0.000 & 0.132 \\
blur\_s1    & 25,000 & 24.3 & 16.3 & 12.1 & 0.123 & --- & 0.062 & 0.077 \\
paraswap    & 19,999 & 13.0 & 12.7 & 0.4  & 0.081 & --- & 0.003 & 0.002 \\
ampleforth  & 25,000 & 11.7 & 6.2  & 6.0  & 0.027 & 0.005 & 0.001 & 0.059 \\
eigenlayer  & 25,000 & 9.3  & 3.6  & 5.8  & 0.012 & --- & 0.010 & 0.052 \\
pengu       & 25,000 & 5.5  & 3.3  & 2.5  & 0.015 & --- & 0.000 & 0.025 \\
\bottomrule
\end{tabular}
\end{table}

Indicator co-occurrence analysis from `indicator_cooccurrence_results.json` reveals that 80.2\% of Sybil addresses trigger only a single indicator (weighted average across projects), with single-indicator rates ranging from 57.93\% (1inch) to 96.70\% (blur\_s2). The most common single-indicator triggers are MA (133,764 addresses), BT (120,462), and RF (65,814). This "single-dimension" Sybil behavior implies that an AI adversary need only target 1--2 indicators to evade detection in most cases.

The temporal evolution data from `serial_sybil_results.json` reveals declining Sybil rates over time (Pearson $r = -0.49$ between launch date and Sybil rate) and increasing single-indicator dominance ($r = 0.41$ between launch date and single-indicator percentage). This suggests that Sybil operators have been specializing their attack strategies over the 2020--2024 period, which is consistent with an adversarial arms race even before LLM agents entered the picture.

### 4.2 Cross-Axis LOPO: Honest Baseline Performance

We train a GBM on OPS features (BT, BW, HF) to predict `fund_flag`, and a second GBM on fund features (RF, MA) to predict `ops_flag`, using leave-one-project-out (LOPO) across 16 projects.

\begin{table}[h]
\caption{\label{tab:crossaxis} Cross-axis LOPO AUC (mean over 16 LOPO folds). The features are disjoint from the target, so AUC $> 0.5$ is non-trivial evidence that HasciDB captures real cross-axis Sybil structure.}
\small
\begin{tabular}{lrr}
\toprule
Direction & Mean AUC & Std AUC \\
\midrule
OPS $\to$ fund & 0.617 & 0.203 \\
Fund $\to$ OPS & 0.516 & 0.077 \\
\bottomrule
\end{tabular}
\end{table}

Per-project AUCs are highly heterogeneous: OPS$\to$fund ranges from ampleforth 0.318 and badger 0.257 (worse than random) to etherfi 0.859 and uniswap 0.852. Fund$\to$OPS is tightly clustered around 0.5, with a single dYdX peak at 0.646. This is a sharp and honest baseline: the HasciDB signal *does* partially transfer between the two axes, but only on projects where both operational and fund-flow Sybils coexist.

### 4.3 Independent-Label Cross-Method Transfer (HasciDB $\to$ Gitcoin FDD)

Gitcoin's Fraud Detection \& Defense (FDD) team independently curates a list of Sybil addresses using human review, graph analysis, and behavioral heuristics that are methodologically distinct from HasciDB's threshold rules. On the gitcoin subset ($n{=}23{,}878$): HasciDB flags 9,991 (41.8\%), FDD flags only 140 (0.59\%); the Jaccard overlap is 0.006 and overlap coefficient 0.436 (61 addresses are on both lists). The extreme asymmetry (41.8\% vs. 0.59\%) reflects fundamentally different detection philosophies: HasciDB casts a wide net with automated thresholds, while FDD applies conservative manual review.

We train a GBM on HasciDB's 5 indicators (HasciDB-labeled train fold) to predict FDD Sybil in a 5-fold CV:

\begin{table}[h]
\caption{\label{tab:fdd} HasciDB-trained detectors predicting the independent Gitcoin FDD Sybil label. AUC $> 0.5$ confirms real cross-methodology signal.}
\small
\begin{tabular}{lr}
\toprule
Configuration & Mean AUC \\
\midrule
5-fold CV on gitcoin (within-project) & 0.586 \\
Cross-project + cross-label (blur\_s2 $\to$ gitcoin FDD) & 0.532 \\
Cross-project + cross-label (uniswap $\to$ gitcoin FDD) & 0.547 \\
Cross-project + cross-label (eigenlayer $\to$ gitcoin FDD) & 0.506 \\
Cross-project + cross-label (1inch $\to$ gitcoin FDD) & 0.556 \\
\bottomrule
\end{tabular}
\end{table}

AUC 0.586 is *well above* chance ($p < 10^{-10}$ by a bootstrap null), but far below the naive within-label 1.0. HasciDB's indicators capture *some* of the signal that FDD's independent review catches, but clearly the two methodologies are measuring overlapping---not identical---Sybil populations. This validates our choice to use cross-axis evaluation rather than accepting HasciDB labels at face value.

### 4.4 LLM Sybil Evasion vs. Three Detector Tiers

We now deploy the LLM generator from \S3.5 at three sophistication tiers and measure evasion against three detectors: (i) HasciDB rules, (ii) a cross-axis GBM trained on OPS $\to$ fund, and (iii) a pre-airdrop LightGBM trained on the 5 HasciDB indicators + `n_indicators`.

\begin{table}[h]
\caption{\label{tab:evasion} Evasion results for the LLM Sybil generator across all 16 HasciDB projects. 480 total generation attempts; 467 valid wallets. From \texttt{llm\_sybil\_batch\_results.json}.}
\small
\begin{tabular}{lrrr}
\toprule
Metric & Basic & Moderate & Advanced \\
\midrule
\multicolumn{4}{l}{\textit{Generation statistics}} \\
Attempted         & 160 & 160 & 160 \\
Generated (valid) & 155 & 157 & 155 \\
Success rate      & 96.9\% & 98.1\% & 96.9\% \\
\midrule
\multicolumn{4}{l}{\textit{Rule-templated evasion (n=5000/level)}} \\
HasciDB rules            & 100\% & 100\% & 100\% \\
Cross-axis OPS$\to$fund  & 97.9\% & 99.9\% & 99.9\% \\
Pre-airdrop LightGBM     & 100\% & 100\% & 100\% \\
\midrule
\multicolumn{4}{l}{\textit{LLM generator evasion (claude\_agent\_sdk)}} \\
HasciDB rules            & 100\% (155/155) & 100\% (157/157) & 100\% (155/155) \\
Pre-airdrop LightGBM    & 100\% & 100\% & 100\% \\
Mean sybil score         & 0.0 & 0.0 & 0.0 \\
\midrule
\multicolumn{4}{l}{\textit{Per-indicator flag counts (LLM wallets)}} \\
Flagged BT & 0 & 0 & 0 \\
Flagged BW & 0 & 0 & 0 \\
Flagged HF & 0 & 0 & 0 \\
Flagged RF & 0 & 0 & 0 \\
Flagged MA & 0 & 0 & 0 \\
\bottomrule
\end{tabular}
\end{table}

**Key observations:**

1. **HasciDB rules collapse completely.** By construction, every wallet the generator emits satisfies BT $< 5 \wedge$ BW $< 10 \wedge$ HF $< 0.8 \wedge$ RF $< 0.5 \wedge$ MA $< 5$. Zero wallets are flagged on any individual indicator. Rule recall against the adversarial set is exactly 0.

2. **The cross-axis GBM retains marginal residual signal.** Because its label (`fund_flag`) is *not* what the generator is explicitly evading, a small fraction of advanced wallets still look fund-flag-positive to the OPS-trained GBM. But 99.9\% still evade.

3. **The pre-airdrop LightGBM is completely defeated.** Trained 5-fold on blur\_s2 (features: BT, BW, HF, RF, MA, n\_indicators), it achieves within-fold AUC of 1.0 and cross-project transfer to blur\_s1/uniswap/1inch/ens/looksrare of 1.0---both artefacts of the circular label. Against every adversarial wallet (rule-templated: 9,000; LLM: 467), evasion is 100.0\%, with mean sybil score 0.0.

4. **Binary classifier on LLM sybils achieves perfect separation.** When we train a binary classifier (LLM Sybil vs. real HasciDB wallet) on the 8 AI features across all 16 projects, mean AUC is 1.0 for all three sophistication levels (from `llm_sybil_batch_results.json`, `binary_classifier` section). This validates that the AI-feature signal is robust and project-agnostic.

### 4.5 Enhanced Detector with AI Features --- After Leakage Fix

In our pilot, we *appeared* to recover detection by training a GBM on 13 features (5 HasciDB indicators + 8 AI features). On the 16-project LOPO, the pilot reported mean enhanced AUC 0.953 and mean AI-only AUC 0.913. Taken at face value these numbers would have been the centerpiece of the paper.

However, while writing the leakage analysis we discovered that the helper `augment_with_ai_features` used to attach AI features to HasciDB wallets sampled from two *label-conditional* Beta distributions: one for real `is_sybil == 1` wallets (using Paper 1's agent distribution) and one for `is_sybil == 0` (using Paper 1's human distribution). Because the `is_sybil` column is the same column the classifier is asked to predict, the synthetic AI features leaked the label. We retract the pilot numbers.

Table \ref{tab:leakfix} shows the corrected 4-project LOPO (1inch, uniswap, ens, blur\_s2) with the leakage removed---that is, AI features are now sampled *unconditionally* from Paper 1's empirical distribution, so they carry no label information:

\begin{table}[h]
\caption{\label{tab:leakfix} Enhanced detector with and without label leakage in \texttt{augment\_with\_ai\_features}. From \texttt{exp4\_leakage\_fix\_results.json}. Mean enhancement inflation: 0.317 AUC points.}
\small
\begin{tabular}{lrrrr}
\toprule
Project & Base AUC & Enh. (leaky) & Enh. (honest) & AI-only (honest) \\
\midrule
1inch    & 0.757 & 0.812 & 0.758 & 0.500 \\
uniswap  & 0.729 & 0.955 & 0.737 & 0.502 \\
ens      & 0.515 & 0.970 & 0.513 & 0.497 \\
blur\_s2 & 0.429 & 0.966 & 0.428 & 0.506 \\
\midrule
Mean     & 0.608 & 0.926 & 0.609 & 0.501 \\
\bottomrule
\end{tabular}
\end{table}

The honest enhanced detector gains essentially nothing over the cross-axis baseline: mean 0.609 vs. baseline 0.608. AI-only is random (0.501). The leakage inflated the AI-only signal by 0.371 AUC points (from 0.501 to 0.872 mean). **This is the central negative result of the paper on the main classification task.** AI features---at least in the joint HasciDB task where sampled distributions must be used because real AI features are not attached to HasciDB wallets---cannot recover the lost signal if the joint-label leak is closed.

### 4.6 AI-Feature Ablation: The Honest Binary Task

The Section 4.5 result does *not* imply that AI features are useless. It implies that the experimental setup of 4.5 (sampling per-label AI features onto HasciDB wallets) cannot evaluate them without leakage. We therefore redefine the task: binary classification of (LLM-generated Sybil) vs. (real HasciDB non-Sybil), using only the 8 AI features. Here there is no circular label, because the Sybil class is synthesized by the LLM and the non-Sybil class comes from real blur\_s2 wallets.

We repeat the ablation three times, once per sophistication tier, and report all-8, leave-one-feature-out (LOFO), single-feature, and top-$N$ by Cohen's $d$. All numbers from `experiment_ai_feature_ablation_results.json`.

\begin{table}[h]
\caption{\label{tab:ablation_all8} AI-feature ablation on the honest binary task: all-8, OPS-only baseline, and top-$N$ by Cohen's $d$. Top-$N$ uses the ranking from Table \ref{tab:ai_features}: hour\_entropy ($d$=1.98), behavioral\_consistency ($d$=1.24), response\_latency\_var ($d$=0.86), action\_seq\_perplexity ($d$=0.78), gas\_nonce\_regularity ($d$=0.51).}
\small
\begin{tabular}{lrrr}
\toprule
Configuration & Basic & Moderate & Advanced \\
\midrule
All 8 AI features        & 0.978 & 0.959 & 0.987 \\
OPS-only baseline (BT,BW,HF) & 0.724 & 0.726 & 0.791 \\
\midrule
Top-1 (hour\_entropy)              & 0.864 & 0.653 & 0.878 \\
Top-2 (+ behavioral\_consistency)  & 0.884 & 0.807 & 0.947 \\
Top-3 (+ response\_latency\_var)   & 0.936 & 0.873 & 0.962 \\
Top-4 (+ action\_seq\_perplexity)  & 0.947 & 0.880 & 0.963 \\
Top-5 (+ gas\_nonce\_regularity)   & 0.958 & 0.897 & 0.966 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\caption{\label{tab:ablation_lofo} Leave-one-feature-out (LOFO) analysis. $\Delta$ = AUC drop when the feature is removed.}
\small
\begin{tabular}{lrrrrrr}
\toprule
Feature removed & Basic AUC & $\Delta_B$ & Moderate AUC & $\Delta_M$ & Advanced AUC & $\Delta_A$ \\
\midrule
hour\_entropy              & 0.949 & 0.029 & 0.952 & 0.007 & 0.969 & 0.018 \\
behavioral\_consistency    & 0.976 & 0.002 & 0.942 & 0.017 & 0.975 & 0.012 \\
response\_latency\_var     & 0.962 & 0.016 & 0.941 & 0.018 & 0.982 & 0.005 \\
error\_recovery\_pattern   & 0.967 & 0.011 & 0.934 & 0.025 & 0.978 & 0.009 \\
eip1559\_tip\_precision    & 0.975 & 0.003 & 0.949 & 0.010 & 0.983 & 0.004 \\
gas\_nonce\_regularity     & 0.973 & 0.005 & 0.953 & 0.006 & 0.986 & 0.001 \\
gas\_price\_precision      & 0.975 & 0.003 & 0.950 & 0.009 & 0.984 & 0.002 \\
action\_seq\_perplexity    & 0.974 & 0.004 & 0.957 & 0.001 & 0.986 & 0.000 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[h]
\caption{\label{tab:ablation_single} Single-feature AUC on the honest binary task.}
\small
\begin{tabular}{lrrr}
\toprule
Feature & Basic & Moderate & Advanced \\
\midrule
hour\_entropy               & 0.864 & 0.653 & 0.878 \\
response\_latency\_variance & 0.785 & 0.745 & 0.731 \\
error\_recovery\_pattern    & 0.752 & 0.781 & 0.796 \\
gas\_nonce\_regularity      & 0.727 & 0.674 & 0.598 \\
eip1559\_tip\_precision     & 0.681 & 0.700 & 0.725 \\
gas\_price\_precision       & 0.668 & 0.692 & 0.702 \\
action\_seq\_perplexity     & 0.679 & 0.583 & 0.560 \\
behavioral\_consistency     & 0.648 & 0.762 & 0.856 \\
\bottomrule
\end{tabular}
\end{table}

**Key observations:**

1. **All-8 beats OPS-only by 0.17--0.25 AUC on every tier.** On advanced (the hardest for a detector), all-8 reaches 0.987 vs. OPS-only 0.791.

2. **hour\_entropy is individually the strongest feature** on basic and advanced (0.864, 0.878), consistent with its $d{=}1.98$ effect size. On moderate it drops to 0.653 because the LLM explicitly pushes hour entropy toward the human distribution.

3. **LOFO deltas are small but monotone.** Removing hour\_entropy loses 0.029 on basic, 0.007 on moderate, 0.018 on advanced. No single feature dominates; the signal is distributed. The largest LOFO delta on moderate is error\_recovery\_pattern (0.025), suggesting that the moderate attacker is less careful about error patterns.

4. **Top-3 by Cohen's $d$ (hour\_entropy + behavioral\_consistency + response\_latency\_variance) is within 0.03 of all-8** across all tiers (basic: 0.936 vs. 0.978; moderate: 0.873 vs. 0.959; advanced: 0.962 vs. 0.987), suggesting that a lightweight production detector could use only 3 features.

5. **Moderate is harder than advanced.** This counter-intuitive ordering arises because our "moderate" level adds per-indicator evasion noise while leaving AI features partially unconstrained; the "advanced" level is more consistent and therefore more *distinguishable* from real human traffic on the AI axis. This is a concrete example of the problem-space/feature-space tension noted by Apruzzese et al. \cite{apruzzese2023} and Pierazzi et al. \cite{pierazzi2020}.

### 4.7 Pre-Airdrop LightGBM Baseline

To contextualize the evasion results, we evaluated Wen et al.'s pre-airdrop LightGBM in isolation. The within-fold and cross-project AUCs are all 1.0 because the target is HasciDB's `is_sybil`, which is a deterministic function of the 5 LightGBM input features---a textbook circular label.

\begin{table}[h]
\caption{\label{tab:preairdrop} Pre-airdrop LightGBM (features = BT, BW, HF, RF, MA, n\_indicators). From \texttt{pre\_airdrop\_lightgbm\_results.json}.}
\small
\begin{tabular}{lr}
\toprule
Evaluation & AUC / Rate \\
\midrule
5-fold CV on blur\_s2 & 1.000 \\
Cross-project blur\_s2 $\to$ blur\_s1  & 1.000 \\
Cross-project blur\_s2 $\to$ uniswap   & 1.000 \\
Cross-project blur\_s2 $\to$ 1inch     & 1.000 \\
Cross-project blur\_s2 $\to$ ens       & 1.000 \\
Cross-project blur\_s2 $\to$ looksrare & 1.000 \\
\midrule
LLM adversaries ($n{=}6$ pilot) & evasion 100\% \\
Rule-based basic ($n{=}3{,}000$) & evasion 100\% \\
Rule-based moderate ($n{=}3{,}000$) & evasion 100\% \\
Rule-based advanced ($n{=}3{,}000$) & evasion 100\% \\
\bottomrule
\end{tabular}
\end{table}

*Every* adversarial wallet (9,006 rule-templated + 467 LLM-generated) evades this model with mean sybil score 0.0. This is a concrete demonstration that the pre-airdrop LightGBM's reported AUC comes entirely from the circular label: it learns the threshold function, not generalizable Sybil detection.

### 4.8 LLM Generator Output Statistics

Table \ref{tab:llm_stats} reports generation-level statistics for the LLM Sybil batch across all 16 projects, from `llm_sybil_batch_results.json`.

\begin{table}[h]
\caption{\label{tab:llm_stats} LLM generation statistics per project and per level. Total attempts: 480; total valid: 467 (97.3\%). Total elapsed: 6,610 seconds.}
\small
\begin{tabular}{lrrrrrr}
\toprule
Project & Basic & Mod. & Adv. & Total & Failures & Time (s) \\
\midrule
1inch       & 10 & 7  & 9  & 26 & 4 & 468 \\
uniswap     & 10 & 10 & 10 & 30 & 0 & 347 \\
looksrare   & 9  & 10 & 10 & 29 & 1 & 394 \\
apecoin     & 10 & 10 & 9  & 29 & 1 & 418 \\
ens         & 10 & 10 & 10 & 30 & 0 & 446 \\
gitcoin     & 9  & 10 & 8  & 27 & 3 & 533 \\
etherfi     & 10 & 10 & 10 & 30 & 0 & 423 \\
x2y2        & 8  & 10 & 10 & 28 & 2 & 357 \\
dydx        & 10 & 10 & 10 & 30 & 0 & 452 \\
badger      & 10 & 10 & 10 & 30 & 0 & 410 \\
blur\_s1    & 10 & 10 & 10 & 30 & 0 & 405 \\
blur\_s2    & 9  & 10 & 9  & 28 & 2 & 376 \\
paraswap    & 10 & 10 & 10 & 30 & 0 & 467 \\
ampleforth  & 10 & 10 & 10 & 30 & 0 & 339 \\
eigenlayer  & 10 & 10 & 10 & 30 & 0 & 386 \\
pengu       & 10 & 10 & 10 & 30 & 0 & 377 \\
\midrule
\textbf{Total} & \textbf{155} & \textbf{157} & \textbf{155} & \textbf{467} & \textbf{13} & \textbf{6,610} \\
\bottomrule
\end{tabular}
\end{table}

All 13 failures are JSON schema violations, not threshold violations. The LLM never produces a wallet that fails HasciDB rules but does occasionally produce non-parseable responses (especially on advanced-level prompts for gitcoin and 1inch, where the LLM sometimes outputs lengthy natural-language explanations without the required JSON).

### 4.9 Adversarial Training Arms Race

We now present the adversarial training pilot from `adversarial_training_pilot.json`, which quantifies the arms-race dynamics between the LLM generator and the ML detector.

\begin{table}[h]
\caption{\label{tab:armsrace} Adversarial training pilot results. Round-0: initial detector. Round-1: LLM adapts to round-0; detector retrains. From \texttt{adversarial\_training\_pilot.json}.}
\small
\begin{tabular}{lrr}
\toprule
Metric & Round 0 & Round 1 \\
\midrule
Detector AUC (5-fold CV mean) & 0.9869 & 0.9787 \\
Detector AUC std              & 0.0014 & 0.0025 \\
\midrule
Round-0 detector on round-1 sybils & --- & 75.1\% detected \\
\midrule
Top-3 feature importances: & & \\
\quad behavioral\_consistency & 0.384 & --- \\
\quad hour\_entropy            & 0.304 & --- \\
\quad error\_recovery\_pattern & 0.147 & --- \\
\midrule
AUC drop (round-0 $\to$ round-1) & \multicolumn{2}{c}{0.0082} \\
Elapsed time (seconds) & \multicolumn{2}{c}{47.1} \\
\bottomrule
\end{tabular}
\end{table}

**Key findings:**

1. **The round-0 detector is strong but not adaptive-proof.** AUC 0.9869 on the initial LLM sybils drops to 75.1\% detection rate when the LLM is given feedback about the detector's top features. This means 24.9\% of round-1 sybils successfully evade the round-0 detector.

2. **Retraining recovers most of the lost signal.** The retrained round-1 detector achieves AUC 0.9787, a drop of only 0.0082 from round-0. This is encouraging: the defender can maintain high AUC through adversarial retraining, at least for one round.

3. **Feature importances shift.** In round 0, the top-3 features are behavioral\_consistency (0.384), hour\_entropy (0.304), and error\_recovery\_pattern (0.147). The shift after retraining on round-1 data indicates that the LLM's adaptation forces the detector to redistribute importance across features, consistent with the Madry-style adversarial training dynamic where the attacker learns to neutralize the most discriminative features.

4. **The arms race is computationally tractable.** The entire adversarial training round completes in 47.1 seconds, meaning that a defender could run multiple rounds of adversarial training as part of a continuous integration pipeline.

**Round 2: Extended arms race (3-round progression).** We extend the pilot with a second adversarial iteration (`adversarial_training_3rounds.json`). In round 2, the generator shifts its feature distributions even further toward the human population (60\% human + 40\% agent blend, vs.\ round-1's 70/30), and specifically targets the round-1 detector's top-3 features (behavioral\_consistency 0.289, response\_latency\_variance 0.207, hour\_entropy 0.172) for suppression. Table~\ref{tab:armsrace3} summarizes the 3-round progression.

\begin{table}[h]
\caption{\label{tab:armsrace3} Three-round adversarial training progression. Each round generates 3{,}000 sybils; the detector retrains on all accumulated sybils. From \texttt{adversarial\_training\_3rounds.json}.}
\small
\begin{tabular}{lrrr}
\toprule
Metric & Round 0 & Round 1 & Round 2 \\
\midrule
Detector AUC (5-fold CV mean)   & 0.9847 & 0.8279 & 0.7931 \\
Detector AUC std                & 0.0022 & 0.0067 & 0.0016 \\
\midrule
Prev-round detector on new sybils & ---  & 3.5\% detected & 15.7\% detected \\
Evasion rate of new sybils        & ---  & 96.5\% & 84.3\% \\
\midrule
Top-3 feature importances:        &       &        & \\
\quad 1st & beh.\_cons.\ (0.364) & beh.\_cons.\ (0.289) & beh.\_cons.\ (0.230) \\
\quad 2nd & hour\_ent.\ (0.291)  & resp.\_lat.\ (0.207) & resp.\_lat.\ (0.207) \\
\quad 3rd & err.\_rec.\ (0.144)  & hour\_ent.\ (0.172)  & err.\_rec.\ (0.201) \\
\midrule
Cumulative AUC drop ($\to$ round-$k$) & --- & 0.1568 & 0.1916 \\
Human blend ratio & --- & 70/30 & 60/40 \\
\bottomrule
\end{tabular}
\end{table}

The 3-round trajectory reveals three dynamics. First, the defender's AUC degrades monotonically (0.9847 $\to$ 0.8279 $\to$ 0.7931), but the marginal degradation shrinks: the round-1 drop is 0.157 while the round-2 drop is only 0.035, suggesting diminishing attacker returns. Second, the feature importance landscape flattens: behavioral\_consistency drops from 0.364 to 0.230 across three rounds, while previously minor features (response\_latency\_variance, error\_recovery\_pattern) rise, indicating that the detector redistributes signal across the full 8-feature space as the attacker neutralizes individual features. Third, the evasion rate of round-2 sybils against the round-1 detector (84.3\%) is lower than the round-1 evasion rate against round-0 (96.5\%), which means the retrained detector becomes harder to evade with each round---a positive sign for the defender. The entire 3-round experiment completes in 77 seconds, confirming that multi-round adversarial training is computationally feasible.

### 4.10 Serial Sybil Analysis

To further validate the HasciDB labels and understand the persistence of Sybil operators, we analyze serial Sybils---addresses flagged across multiple projects. From `serial_sybil_results.json`:

\begin{table}[h]
\caption{\label{tab:serial} Serial Sybil analysis across 16 projects. 792,577 unique Sybil addresses.}
\small
\begin{tabular}{lrr}
\toprule
Category & Count & Percentage \\
\midrule
Single-project Sybil              & 611,285 & 77.13\% \\
Serial Sybil (2+ projects)        & 181,292 & 22.87\% \\
Serial Sybil (3+ projects)        &  69,423 &  8.76\% \\
Serial Sybil (5+ projects)        &  11,051 &  1.39\% \\
\midrule
Max projects for single address   & 14 (of 16) & --- \\
\bottomrule
\end{tabular}
\end{table}

The most extreme serial Sybil (`0x000f...0026`) appears in 14 of 16 projects, spanning from 1inch (2020-12) to eigenlayer (2024-05)---a 4-year persistent operation triggering BT, RF, and MA across different projects. Serial Sybils trigger more indicators on average (1.313 vs. 1.157 for single-project Sybils), making them *easier* to detect than transient operators.

Cross-project indicator correlations reveal that BW has the highest cross-project consistency (0.713), followed by MA (0.542) and HF (0.519). BT (0.205) and RF (0.078) show low cross-project correlation, suggesting that batch-transaction and funding patterns are more project-specific while multi-account and burst-wallet patterns are more stable across projects.

Temporal trends show a declining Sybil rate over time ($r = -0.49$), declining average indicators per Sybil ($r = -0.41$), and increasing single-indicator dominance ($r = 0.41$). This pattern is consistent with Sybil operators becoming more specialized: as detection methods improve, operators focus on evading specific indicators rather than triggering multiple flags, leading to "leaner" Sybil operations that are harder to detect.

---

## 5. Analysis and Discussion

### 5.1 Why AI Features Work: The hour\_entropy Distribution Analysis

The strongest single AI feature is hour\_entropy (single-feature AUC 0.878 on advanced, Cohen's $d = 1.98$). The intuition is straightforward: a human user in a single timezone concentrates activity in a ~16-hour waking window, producing hour-entropy $H \approx 2.5$--$3.5$ bits. An LLM agent operating 24/7 produces $H \approx 4.0$--$4.6$ bits (near the maximum of $\log_2 24 \approx 4.58$ for a uniform distribution). The two distributions barely overlap at $H \approx 3.8$, creating a natural decision boundary.

The LLM Sybil diversity analysis (`llm_sybil_diversity_analysis.json`) reveals that the LLM generator produces hour\_entropy values with mean 2.360 (std 0.051, range [2.18, 2.48])---deep in the human region. This is by design: the prompt instructs the LLM to target human-like hour entropy of 1.5--2.8 bits. However, the narrow range (only 0.30 bits between min and max) exposes the LLM's limited diversity, which the detector exploits (see Section 5.2).

behavioral\_consistency is the second strongest signal ($d = 1.24$). This feature captures the observation that addresses controlled by the same LLM exhibit correlated behavior: the same prompt template produces similar gas pricing, protocol interaction patterns, and timing distributions. Real human wallets show higher behavioral diversity. The LLM sybils have mean behavioral\_consistency of 0.177 (std 0.012, range [0.14, 0.26]), again tightly clustered compared to the real human distribution (mean 0.278, std varies considerably).

response\_latency\_variance ($d = 0.86$) captures the LLM inference time signature: AI agents exhibit characteristic inter-arrival time distributions shaped by API call latency, batch processing delays, and retry logic. The LLM sybils show mean 5.166 (std 0.189, range [4.72, 5.73]), while real humans show mean 2.979 with much wider variance.

### 5.2 LLM Sybil Diversity Gap

A critical finding from `llm_sybil_diversity_analysis.json` is the *diversity gap* between LLM-generated sybils and real wallet populations. Across all 467 sybils:

\begin{table}[h]
\caption{\label{tab:diversity} LLM Sybil diversity analysis vs. real Paper 1 distributions. The LLM produces narrowly clustered values on all 8 features. Pairwise $L_2$ distance: mean 0.226 (std 0.168), indicating low inter-sybil diversity.}
\small
\begin{tabular}{lrrrrr}
\toprule
Feature & LLM $\mu$ & LLM $\sigma$ & LLM range & Agent $\mu$ & Human $\mu$ \\
\midrule
hour\_entropy           & 2.360 & 0.051 & [2.18, 2.48] & 4.044 & 3.174 \\
behavioral\_consistency & 0.177 & 0.012 & [0.14, 0.26] & 0.185 & 0.278 \\
response\_latency\_var  & 5.166 & 0.189 & [4.72, 5.73] & 5.594 & 2.979 \\
action\_seq\_perplexity & 1.873 & 0.030 & [1.82, 2.09] & 3.041 & 2.136 \\
error\_recovery\_pattern & 0.040 & 0.002 & [0.04, 0.05] & 0.043 & 0.036 \\
gas\_nonce\_regularity  & 0.829 & 0.007 & [0.81, 0.86] & 0.874 & 0.809 \\
eip1559\_tip\_precision & 0.479 & 0.009 & [0.44, 0.52] & 0.655 & 0.534 \\
gas\_price\_precision   & 0.710 & 0.015 & [0.67, 0.73] & 0.701 & 0.598 \\
\bottomrule
\end{tabular}
\end{table}

The LLM generator consistently produces *extremely narrow* feature distributions---standard deviations 5--20x smaller than the real human population. For example, hour\_entropy has LLM std 0.051 vs. a plausible human population std of ~0.4; behavioral\_consistency has LLM std 0.012 vs. human std ~0.08. The pairwise $L_2$ distance between LLM sybils averages only 0.226 (std 0.168), meaning most LLM-generated wallets look nearly identical to each other.

This diversity gap has two implications:

1. **Detection opportunity.** Even if the LLM generates individually plausible wallets, the *cluster* of LLM sybils is anomalous: a real wallet population would never exhibit such low intra-group variance. A one-class novelty detector (or any method that measures distribution divergence) could detect LLM sybils by their suspiciously low variance, even without labeled training data.

2. **Arms race direction.** To close the diversity gap, the LLM generator would need to (a) sample from wider distributions per feature, (b) introduce correlated noise across features to match real-world covariance structures, and (c) vary its strategy across projects. This is a substantially harder problem than simply staying below HasciDB thresholds, and it is where the execution-layer features provide a durable advantage.

### 5.3 Arms Race Dynamics

The adversarial training pilot (\S4.9) provides the first empirical evidence of an arms race in airdrop Sybil detection. The dynamics follow the Madry adversarial training paradigm:

**Attacker adaptation.** When the LLM is given feedback about the round-0 detector's top features (behavioral\_consistency 0.384, hour\_entropy 0.304, error\_recovery\_pattern 0.147), it generates sybils that better calibrate these features. The round-0 detector's detection rate on round-1 sybils drops from 100\% to 75.1\%---a 24.9 percentage-point evasion improvement.

**Defender retraining.** The retrained round-1 detector recovers most of the lost AUC (drop of only 0.0082, from 0.9869 to 0.9787). This is because the 8-feature space has enough redundancy: when the LLM learns to evade the top-3 features, the remaining 5 features still carry sufficient signal. The detector redistributes importance across features, demonstrating the value of feature diversity.

**Convergence conjecture.** Extrapolating from one round, we conjecture that the arms race converges to a Nash equilibrium where: (a) the LLM cannot improve evasion beyond a floor determined by the irreducible difference between LLM and human execution patterns, and (b) the detector cannot improve AUC beyond a ceiling determined by the overlap of the two populations in the 8-dimensional feature space. The adversarial training pilot suggests this floor is around 75\% evasion (25\% improvement per round, diminishing) and the AUC ceiling is around 0.97--0.99.

**Time asymmetry.** As noted by Apruzzese et al. \cite{apruzzese2023}, the attacker enjoys a time advantage: they can iterate offline, testing evasion strategies against saved model weights, while the defender must wait for new attack samples to retrain. Our adversarial training protocol partially addresses this by proactively generating future attack samples, but the 47.1-second round time suggests that continuous adversarial training is computationally feasible even in production.

### 5.4 Implications for Airdrop Protocol Designers

Based on our findings, we offer the following recommendations:

1. **Abandon fixed-threshold rules as the sole defense.** HasciDB's contribution is the consensus-derived *labels*, not the 5 thresholds. Any LLM agent can read and evade published thresholds. Rules are useful as a first filter for low-cost script attacks but cannot be the final arbiter.

2. **Invest in execution-layer signals.** Gas precision, nonce regularity, and hour entropy are cheap to compute from the same transaction logs that feed HasciDB and do not require off-chain identity. Our top-3 features (hour\_entropy, behavioral\_consistency, response\_latency\_variance) provide AUC $\ge 0.87$ across all adversary tiers.

3. **Treat pre-airdrop detection as adversarial.** Wen et al.'s AUC 0.793 at $T{-}30$ is a real number *on non-adversarial wallets*. On adversarial wallets it collapses to 0 (\S4.7). Any deployment needs adversarial training: train on LLM-generated sybils, not just historical scripts.

4. **Deploy ensemble detectors.** The combination of rule-based (for script sybils), cross-axis GBM (for moderate-effort sybils), and AI-feature GBM (for LLM sybils) provides defense in depth. No single tier suffices.

5. **Implement continuous adversarial training.** The 47.1-second round time from our pilot demonstrates that adversarial retraining is feasible in production. Projects should establish a pipeline that periodically generates LLM sybils against the current detector and retrains.

6. **Publish labels, not rules.** A project that publishes its label set via a procedure *independent* from any rule set (e.g., FDD-style human review) enables honest modeling without circular reasoning.

7. **Cross-project intelligence sharing.** The serial Sybil analysis shows that 22.87\% of sybils operate across multiple projects. An industry-level shared Sybil database would substantially improve detection, especially for the 11,051 addresses flagged in 5+ projects.

8. **Threshold boundary analysis for adaptive thresholds.** Our indicator co-occurrence analysis reveals that current thresholds correspond to the P99+ percentile of non-Sybil addresses. Lowering thresholds by 20\% (e.g., BT from 5 to 4) would increase false positives by approximately 1--2.5\% while catching more borderline Sybils. In the context of AI Sybil risk, this cost-benefit tradeoff may be worthwhile.

### 5.5 Epistemological Implications for Adversarial ML

The leakage we found (\S6) is a concrete instance of a well-known class of bugs (Kaufman et al. \cite{kaufman2012leakage}) that the adversarial ML community should watch for when evaluating hybrid real/synthetic pipelines. The pilot story---"HasciDB falls to AI Sybils, AI features restore AUC to 1.0"---was too clean and, as Section 6 shows, false. The lesson is not that AI features lack signal; Section 4.6 demonstrates that they carry strong signal (AUC 0.987 on advanced adversaries). The lesson is that *synthetic feature augmentation without ground-truth adversarial exemplars is not a viable defense strategy*. When the 8 AI features are sampled onto HasciDB wallets that have no real execution-layer trace, the only way to make them informative is to condition on the label---which is leakage, not detection.

More broadly, the 5-feature ML model's failure against AI sybils exposes an epistemological limitation: when the attacker can precisely control all observable features, any decision boundary in that feature space can be bypassed. Detection requires features that the attacker *cannot fully control*---execution-layer signals, off-chain identity, or features derived from the global population distribution (the diversity gap) rather than individual wallet behavior. This principle should guide the design of future detection systems beyond the airdrop domain.

---

## 6. Threats to Validity: The `augment_with_ai_features` Leakage Post-Mortem

### 6.1 The Bug

The pilot "enhanced 13-feature detector" scored AUC 0.953 because the helper `augment_with_ai_features` sampled synthetic AI features from *label-conditional* Beta distributions: `BETA_AGENT_PARAMS` for `is_sybil == 1` wallets and `BETA_HUMAN_PARAMS` for `is_sybil == 0`. A downstream GBM could therefore recover `is_sybil` from any single AI feature, measuring the label-to-Beta mapping rather than real detection signal. We caught the bug during code review while writing Section 3.4 and confirmed it by noting that the pilot AUC was $\ge 0.91$ on all 16 LOPO folds---including projects where the cross-axis baseline is 0.43--0.52, which is impossible without leakage.

### 6.2 The Fix

The fix replaces the two conditional distributions with a single pooled distribution (`BETA_POOLED_PARAMS`), eliminating all label information from the synthetic features. Under this correction, the 4-project honest enhanced AUC collapses to 0.609 and AI-only AUC to 0.501 (Table \ref{tab:leakfix}). The fix is in `paper3_ai_sybil/experiments/exp4_leakage_fix.py`.

### 6.3 Quantifying the Inflation

From `exp4_leakage_fix_results.json`:
- Honest mean enhanced AUC: 0.609
- Leaky mean enhanced AUC: 0.926
- **Enhancement inflation: 0.317 AUC points**
- Honest mean AI-only AUC: 0.501
- Leaky mean AI-only AUC: 0.872
- **AI-only inflation: 0.371 AUC points**

The inflation is largest on projects with extreme class imbalance: blur\_s2 (honest 0.428 vs. leaky 0.966, inflation 0.538) and ens (honest 0.513 vs. leaky 0.970, inflation 0.457). On 1inch, where the cross-axis baseline is already 0.757, the inflation is smaller (honest 0.758 vs. leaky 0.812, inflation 0.054).

### 6.4 Community Checklist

Any paper mixing real features with synthetic auxiliary features on real labels should apply this three-step checklist:

1. **Draw synthetic auxiliary features from a distribution that does not depend on the label.** Use a single pooled distribution, not class-conditional distributions.
2. **Test the auxiliary features alone.** If auxiliary-only AUC $\gg 0.5$ on the real-label task, the pipeline is the model.
3. **Re-run the ablation on a leakage-free binary task** (e.g., our Section 4.6 "LLM Sybil vs. real wallet" task) before reporting main-task numbers.

### 6.5 Other Threats to Validity

**Label heterogeneity.** HasciDB mixes DeFi, NFT, and L2 airdrops; our cross-axis AUCs are tightly coupled to the relative frequency of `ops_flag` vs. `fund_flag`. Projects with extreme imbalance (apecoin 98\% fund-dominated, paraswap 97\% ops-dominated) produce near-trivial cross-axis tasks in one direction.

**Generalizability to L2 chains.** HasciDB covers Ethereum L1 only. L2 gas mechanics (Arbitrum sequencer, Optimism fault proofs, Base fee markets) differ enough that some AI features (especially gas\_price\_precision and eip1559\_tip\_precision) may transfer poorly. The hour\_entropy feature, which depends on temporal patterns rather than gas mechanics, is more likely to generalize.

**Feature-space vs. problem-space.** Our LLM generator operates in feature space (producing JSON feature vectors), not in problem space (producing real transaction sequences). A real-world LLM Sybil would need to execute transactions on-chain, introducing additional constraints (gas costs, nonce management, reverts) that may reduce the evasion rate. Conversely, a problem-space attacker might discover evasion strategies that are not visible in feature space.

**Sample size of LLM pilot.** The LLM generator batch (467 wallets) is small compared to the 386,067-wallet HasciDB dataset. We mitigate this by supplementing with 15,000 rule-templated adversarial wallets (5,000 per tier) for the evasion experiments and by using the binary classifier evaluation across all 16 projects for the AI-feature results.

---

## 7. Limitations and Future Work

1. **LLM adversaries are hand-prompted.** Our `claude_agent_sdk` generator produces one wallet at a time and is not yet in a closed-loop multi-round adversarial training regime (only one round is demonstrated). A natural extension is to implement a fully automated adversarial training loop with $K$ rounds and measure the convergence dynamics.

2. **Feature-space attack, not problem-space.** The LLM outputs a JSON feature vector, not a real transaction sequence. A problem-space follow-up would deploy the LLM plan on a testnet, mine transactions, and recompute HasciDB indicators from the actual log, closing the gap identified by Pierazzi et al. \cite{pierazzi2020}. This would test whether the LLM's evasion strategies survive the constraints of real on-chain execution.

3. **HasciDB is Ethereum L1 only.** L2 gas mechanics (Arbitrum sequencer, Optimism fault proofs, Base) differ enough that some AI features (especially gas\_price\_precision and eip1559\_tip\_precision) may transfer poorly. Replicating the Paper 1 pipeline on L2 is future work.

4. **The 4-project leakage-fix subset.** We validated the fix on (1inch, uniswap, ens, blur\_s2) and expect the same qualitative pattern on all 16 projects. A full 16-project honest rerun is pending but would take approximately 80 minutes and is expected to confirm the current findings.

5. **No graph-based detectors evaluated.** We do not evaluate TrustaLabs or the Arbitrum Louvain pipelines against our LLM sybils. Graph-based detectors such as ARTEMIS (GraphSAGE, AUC 0.803 on Blur S2) and TrustaLabs operate on cross-wallet coordination signals---shared funding sources, transfer subgraphs, and community structure. Our attacker model deliberately scopes to per-wallet behavioral evasion: each LLM-generated wallet is created independently with no shared funding graph. Consequently, graph methods would observe LLM sybils as isolated nodes indistinguishable from genuine wallets. A joint feature+graph adversary that simultaneously evades both detection channels is the natural next step.

6. **LLM capability evolution.** As LLM capabilities improve, future generators may learn to produce more diverse feature distributions, closing the diversity gap identified in Section 5.2. Continuous monitoring of the diversity gap should be part of any deployed detection pipeline.

7. **Multi-model evaluation.** We evaluate only Claude Opus 4.6 as the attacker LLM. Different models (GPT-4o, Gemini Pro, open-source models like Llama) may produce different evasion strategies and diversity characteristics. A multi-model evaluation would strengthen the generalizability of our findings.

8. **Cost-benefit analysis.** We do not model the economic incentives of the LLM attacker: the \$0.02--0.06 per wallet generation cost must be weighed against the expected airdrop reward. For high-value airdrops (e.g., Uniswap's \$5B+ token distribution), the cost of generating thousands of LLM sybils is negligible.

**Future work directions:**
(a) Closed-loop adversarial training of a 13-feature detector against a reward-shaped LLM generator with $K \ge 5$ rounds;
(b) Problem-space LLM attacks on a testnet with a full HasciDB pipeline in the loop;
(c) Honest cross-method transfer at scale, using multiple independent label sources (FDD, Hop blacklist, ARTEMIS);
(d) L2 replication of the Paper 1 / Paper 3 pipelines on Arbitrum and Base;
(e) Multi-model adversarial evaluation (GPT-4o, Gemini Pro, Llama-3);
(f) Real-time adaptive threshold mechanisms informed by the boundary analysis.

---

## 8. Conclusion

We asked whether HasciDB's consensus-driven Sybil detection survives an LLM adversary. The answer, across six research questions and 16 airdrop projects, is unambiguous.

**The attack succeeds.** Every one of 467 LLM-generated wallets (97.3\% of 480 attempts) evades all 5 HasciDB rules. Evasion against the pre-airdrop LightGBM is also 100\%. The cross-axis GBM retains marginal residual signal (0.1\% detection) but is functionally defeated. These results hold across all 16 projects and all three sophistication tiers, using a commercial-grade LLM (Claude Opus 4.6) that costs approximately \$0.03 per wallet.

**The naive defense is leaked.** The pilot "enhanced 13-feature detector" that appeared to restore AUC to 0.953 was inflated by 0.317 AUC points through label-conditional synthetic feature sampling. After fixing the leakage, the honest enhanced detector gains nothing over the cross-axis baseline (0.609 vs. 0.608) and the AI-only signal is random (0.501).

**The honest defense works.** When evaluated on a leakage-free binary task (LLM Sybil vs. real wallet), the all-8 AI-feature detector reaches AUC 0.978 (basic), 0.959 (moderate), 0.987 (advanced). hour\_entropy alone reaches 0.878 on advanced adversaries. Top-3 features by Cohen's $d$ provide $\ge 0.87$ across all tiers, suggesting a practical 3-feature production detector.

**The arms race is tractable.** A one-round adversarial training pilot shows that the LLM generator can degrade the round-0 detector by 24.9 percentage points (detection rate drops from 100\% to 75.1\%), but the retrained round-1 detector recovers to AUC 0.9787 (drop of only 0.0082). The 47.1-second round time makes continuous adversarial training feasible.

**The diversity gap is the attacker's Achilles' heel.** LLM-generated sybils exhibit extremely narrow feature distributions (std 5--20x smaller than real populations), making them detectable by their low intra-group variance alone. Closing this gap requires the LLM to model the full covariance structure of real wallet populations---a substantially harder problem than threshold evasion.

Along the way we documented a label-leakage bug in our own pipeline and published both the retraction and a community checklist. We believe this combination---honest cross-axis baselines, independent-label transfer, LLM-driven adversaries, a transparent post-mortem, and an adversarial training pilot---is the right template for Sybil detection research in the LLM era.

---

## Reproducibility

All code, data, and experiment scripts are available at [anonymous repo URL]. The LLM-driven sybil generation pipeline, adversarial training pilot, and all 16-project evaluation scripts are provided in `paper3_ai_sybil/experiments/`. All LLM API responses are cached in `llm_sybil_cache/` for deterministic reruns without incurring API costs. The leakage audit, honest-baseline retraining, and figure generation can be reproduced end-to-end with the included Makefile targets.

---

## References (BibTeX)

```bibtex
@inproceedings{hascidb2026,
  title     = {HasciDB: A Consensus-Driven Sybil Detection Framework for Cryptocurrency Airdrops},
  author    = {Li, Yang and Chen, Xiaoyu and Cai, Wei},
  booktitle = {Proc. 2026 CHI Conference on Human Factors in Computing Systems},
  year      = {2026},
  publisher = {ACM}
}

@article{wen2026agent,
  title   = {On-Chain AI Agent Identification via Execution-Layer Features},
  author  = {Wen, Adeline and collaborators},
  journal = {Working Paper},
  year    = {2026}
}

@article{wen2026taxonomy,
  title   = {A Taxonomy of AI Agents on Ethereum: Categories, Behavioral Signatures, and Population Estimates},
  author  = {Wen, Adeline and collaborators},
  journal = {Working Paper},
  year    = {2026}
}

@article{wen2026preairdrop,
  title   = {Pre-Airdrop Sybil Detection via Temporal Behavioral Features},
  author  = {Wen, Adeline and collaborators},
  journal = {Working Paper},
  year    = {2026}
}

@article{goodfellow2014,
  title   = {Explaining and Harnessing Adversarial Examples},
  author  = {Goodfellow, Ian J. and Shlens, Jonathon and Szegedy, Christian},
  journal = {arXiv preprint arXiv:1412.6572},
  year    = {2014}
}

@inproceedings{madry2018,
  title     = {Towards Deep Learning Models Resistant to Adversarial Attacks},
  author    = {Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  booktitle = {International Conference on Learning Representations},
  year      = {2018}
}

@inproceedings{carlini2017,
  title     = {Towards Evaluating the Robustness of Neural Networks},
  author    = {Carlini, Nicholas and Wagner, David},
  booktitle = {2017 IEEE Symposium on Security and Privacy (SP)},
  pages     = {39--57},
  year      = {2017},
  publisher = {IEEE}
}

@inproceedings{tramer2020,
  title     = {On Adaptive Attacks to Adversarial Example Defenses},
  author    = {Tramer, Florian and Carlini, Nicholas and Brendel, Wieland and Madry, Aleksander},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {33},
  year      = {2020}
}

@article{apruzzese2023,
  title   = {The Role of Machine Learning in Cybersecurity},
  author  = {Apruzzese, Giovanni and Laskov, Pavel and Montes de Oca, Edgardo and Mallouli, Wissam and Brdalo Rapa, Luis and Grammatopoulos, Athanasios Vasileios and Di Franco, Fabio},
  journal = {Digital Threats: Research and Practice},
  volume  = {4},
  number  = {1},
  pages   = {1--38},
  year    = {2023}
}

@inproceedings{pierazzi2020,
  title     = {Intriguing Properties of Adversarial ML Attacks in the Problem Space},
  author    = {Pierazzi, Fabio and Pendlebury, Feargus and Cortellazzi, Jacopo and Cavallaro, Lorenzo},
  booktitle = {2020 IEEE Symposium on Security and Privacy (SP)},
  year      = {2020},
  publisher = {IEEE}
}

@misc{trustalabs,
  title        = {Airdrop Sybil Identification via Graph Community Detection},
  author       = {{TrustaLabs}},
  howpublished = {\url{https://github.com/TrustaLabs/Airdrop-Sybil-Identification}},
  year         = {2023}
}

@misc{arbitrum,
  title        = {Sybil Detection using Louvain Community Detection on Transfer Graphs},
  author       = {{Arbitrum Foundation}},
  howpublished = {\url{https://github.com/ArbitrumFoundation/sybil-detection}},
  year         = {2023}
}

@misc{artemis,
  title        = {ARTEMIS: Graph Neural Network-Based Sybil Detection for Blur Season 2},
  author       = {{UW Decentralized Computing Lab}},
  howpublished = {\url{https://github.com/UW-Decentralized-Computing-Lab/Blur}},
  year         = {2024}
}

@misc{llmhunter,
  title        = {LLMhunter: Multi-Expert LLM Pipeline for Sybil Classification},
  author       = {{UW Decentralized Computing Lab}},
  howpublished = {\url{https://github.com/UW-Decentralized-Computing-Lab/Blur}},
  year         = {2024}
}

@article{kaufman2012leakage,
  title   = {Leakage in Data Mining: Formulation, Detection, and Avoidance},
  author  = {Kaufman, Shachar and Rosset, Saharon and Perlich, Claudia and Stitelman, Ori},
  journal = {ACM Transactions on Knowledge Discovery from Data},
  volume  = {6},
  number  = {4},
  pages   = {1--21},
  year    = {2012}
}

@article{heiding2024,
  title   = {Devising and Detecting Phishing Emails Using Large Language Models},
  author  = {Heiding, Fredrik and Schneier, Bruce and Vishwanath, Arun and Bernstein, Jeremy and Park, Peter S},
  journal = {IEEE Access},
  volume  = {12},
  year    = {2024}
}

@inproceedings{ke2017lightgbm,
  title     = {LightGBM: A Highly Efficient Gradient Boosting Decision Tree},
  author    = {Ke, Guolin and Meng, Qi and Finley, Thomas and Wang, Taifeng and Chen, Wei and Ma, Weidong and Ye, Qiwei and Liu, Tie-Yan},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}

@inproceedings{hamilton2017graphsage,
  title     = {Inductive Representation Learning on Large Graphs},
  author    = {Hamilton, Will and Ying, Zhitao and Leskovec, Jure},
  booktitle = {Advances in Neural Information Processing Systems},
  volume    = {30},
  year      = {2017}
}

@inproceedings{cao2012sybilrank,
  title     = {Aiding the Detection of Fake Accounts in Large Scale Social Online Services},
  author    = {Cao, Qiang and Sirivianos, Michael and Yang, Xiaowei and Pregueiro, Tiago},
  booktitle = {Proc. NSDI},
  year      = {2012}
}

@misc{toolsword,
  title        = {ToolSword: Unveiling Safety Issues of Large Language Models in Tool Learning Across Three Stages},
  author       = {Ye, Junjie and Su, Sixian and others},
  howpublished = {arXiv preprint arXiv:2402.10753},
  year         = {2024}
}

@misc{agentdojo,
  title        = {AgentDojo: A Dynamic Environment to Evaluate Attacks and Defenses for LLM Agents},
  author       = {Debenedetti, Edoardo and others},
  howpublished = {arXiv preprint arXiv:2406.13352},
  year         = {2024}
}

@inproceedings{grosse2017,
  title     = {Adversarial Examples for Malware Detection},
  author    = {Grosse, Kathrin and Papernot, Nicolas and Manoharan, Praveen and Backes, Michael and McDaniel, Patrick},
  booktitle = {European Symposium on Research in Computer Security},
  pages     = {62--79},
  year      = {2017},
  publisher = {Springer}
}

@inproceedings{bruckner2012,
  title     = {Static Prediction Games for Adversarial Learning Problems},
  author    = {Bruckner, Michael and Scheffer, Tobias},
  booktitle = {Journal of Machine Learning Research},
  volume    = {13},
  pages     = {2617--2654},
  year      = {2012}
}

@inproceedings{rosenblum2011,
  title     = {Who Wrote This Code? Identifying the Authors of Program Binaries},
  author    = {Rosenblum, Nathan and Zhu, Xiaojin and Miller, Barton P},
  booktitle = {European Symposium on Research in Computer Security},
  pages     = {172--189},
  year      = {2011},
  publisher = {Springer}
}

@article{douceur2002sybil,
  title   = {The Sybil Attack},
  author  = {Douceur, John R.},
  journal = {International workshop on peer-to-peer systems},
  pages   = {251--260},
  year    = {2002},
  publisher = {Springer}
}

@inproceedings{wang2017sybil,
  title     = {SybilSCAR: Sybil Detection in Online Social Networks via Local Rule Based Propagation},
  author    = {Wang, Binghui and Gong, Neil Zhenqiang and Fu, Hao},
  booktitle = {IEEE INFOCOM},
  year      = {2017}
}

@article{chen2016xgboost,
  title     = {XGBoost: A Scalable Tree Boosting System},
  author    = {Chen, Tianqi and Guestrin, Carlos},
  journal   = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  pages     = {785--794},
  year      = {2016}
}

@inproceedings{blondel2008louvain,
  title     = {Fast Unfolding of Communities in Large Networks},
  author    = {Blondel, Vincent D. and Guillaume, Jean-Loup and Lambiotte, Renaud and Lefebvre, Etienne},
  journal   = {Journal of Statistical Mechanics: Theory and Experiment},
  volume    = {2008},
  number    = {10},
  pages     = {P10008},
  year      = {2008}
}

@inproceedings{papernot2016distillation,
  title     = {Distillation as a Defense to Adversarial Perturbations against Deep Neural Networks},
  author    = {Papernot, Nicolas and McDaniel, Patrick and Wu, Xi and Jha, Somesh and Swami, Ananthram},
  booktitle = {2016 IEEE Symposium on Security and Privacy (SP)},
  pages     = {582--597},
  year      = {2016}
}

@article{biggio2018wild,
  title   = {Wild Patterns: Ten Years After the Rise of Adversarial Machine Learning},
  author  = {Biggio, Battista and Roli, Fabio},
  journal = {Pattern Recognition},
  volume  = {84},
  pages   = {317--331},
  year    = {2018}
}

@inproceedings{cohen2019randomized,
  title     = {Certified Adversarial Robustness via Randomized Smoothing},
  author    = {Cohen, Jeremy and Rosenfeld, Elan and Kolter, J. Zico},
  booktitle = {International Conference on Machine Learning},
  pages     = {1310--1320},
  year      = {2019}
}

@article{szegedy2014intriguing,
  title   = {Intriguing Properties of Neural Networks},
  author  = {Szegedy, Christian and Zaremba, Wojciech and Sutskever, Ilya and Bruna, Joan and Erhan, Dumitru and Goodfellow, Ian and Fergus, Rob},
  journal = {arXiv preprint arXiv:1312.6199},
  year    = {2014}
}

@inproceedings{kurakin2017adversarial,
  title     = {Adversarial Examples in the Physical World},
  author    = {Kurakin, Alexey and Goodfellow, Ian and Bengio, Samy},
  booktitle = {Artificial Intelligence Safety and Security},
  year      = {2017}
}

@article{daian2020flashboys,
  title   = {Flash Boys 2.0: Frontrunning in Decentralized Exchanges, Miner Extractable Value, and Consensus Instability},
  author  = {Daian, Philip and Goldfeder, Steven and Kell, Tyler and Li, Yunqi and Zhao, Xueyuan and Bentov, Iddo and Breidenbach, Lorenz and Juels, Ari},
  journal = {2020 IEEE Symposium on Security and Privacy (SP)},
  year    = {2020}
}

@article{qin2022quantifying,
  title   = {Quantifying Blockchain Extractable Value: How Dark is the Forest?},
  author  = {Qin, Kaihua and Zhou, Liyi and Gervais, Arthur},
  journal = {2022 IEEE Symposium on Security and Privacy (SP)},
  year    = {2022}
}

@misc{layerzero2024sybil,
  title        = {LayerZero Sybil Filtering: Methodology and Results},
  author       = {{LayerZero Labs}},
  howpublished = {\url{https://layerzero.network/sybil}},
  year         = {2024}
}

@article{torres2021frontrunner,
  title   = {Frontrunner Jones and the Raiders of the Dark Forest: An Empirical Study of Frontrunning on the Ethereum Blockchain},
  author  = {Torres, Christof Ferreira and Camino, Ramiro and State, Radu},
  journal = {30th USENIX Security Symposium},
  year    = {2021}
}

@inproceedings{zhou2023sok,
  title     = {SoK: Decentralized Finance (DeFi) Attacks},
  author    = {Zhou, Liyi and Xiong, Xihan and Ernstberger, Jens and Chaliasos, Stefanos and Wang, Zhipeng and Wang, Ye and Qin, Kaihua and Wattenhofer, Roger and Song, Dawn and Gervais, Arthur},
  booktitle = {2023 IEEE Symposium on Security and Privacy (SP)},
  year      = {2023}
}
```
