# 2. Related Work

## 2.1 Sybil Detection in Cryptocurrency Airdrops

### HasciDB (Li, Chen, Cai -- CHI'26)

Li et al. present HasciDB, the first consensus-driven Sybil detection framework for cryptocurrency airdrops. Their key contributions include:

- **Consensus-driven definition**: A modified Delphi method with 12 Web3 practitioners to establish indicator thresholds, moving beyond ad-hoc detection rules used by individual projects.
- **Five-indicator framework**: Two operational indicators (BT: Batch Trading, BW: Batch Wallets, HF: High Frequency) and two fund-flow indicators (RF: Rapid Funds, MA: Multi-Address). Classification follows: `ops_flag = (BT >= 5) OR (BW >= 10) OR (HF >= 0.80)`; `fund_flag = (RF >= 0.50) OR (MA >= 5)`; `is_sybil = ops_flag OR fund_flag`.
- **Scale**: 3.6 million eligible addresses across 16 Ethereum L1 airdrop projects spanning 2020--2024, identifying 1.09 million Sybil addresses (30.6% overall Sybil rate).
- **16 projects**: uniswap, ens, 1inch, blur_s1, blur_s2, gitcoin, looksrare, eigenlayer, x2y2, dydx, apecoin, paraswap, badger, ampleforth, etherfi, pengu.
- **Serial Sybil analysis**: Identification of addresses flagged across multiple projects, revealing persistent Sybil operators.

HasciDB provides the primary baseline and ground truth for our study. Its rule-based detection achieves strong performance against traditional script-based Sybils but, as we demonstrate, is fundamentally vulnerable to AI-agent-driven evasion that keeps all five indicators below their thresholds.

### TrustaLabs: Graph Community Detection

TrustaLabs (Airdrop-Sybil-Identification) employs graph-based community detection on asset transfer graphs:

- **Graph construction**: Nodes are addresses; edges represent ETH/token transfers weighted by value and frequency.
- **Community detection**: Louvain and K-Core algorithms to identify tightly connected address clusters.
- **Refinement**: K-means clustering on behavioral features within detected communities to separate coordinated Sybil clusters from organic activity.
- **Limitation**: Graph-based methods assume Sybil addresses form dense subgraphs. AI agents can break this assumption by using indirect fund flows (bridges, DEX swaps) that do not create direct edges in the transfer graph.

### ArbitrumFoundation: Louvain on Transfer Graphs

The Arbitrum Foundation's Sybil detection approach (sybil-detection repository):

- **Three graph types**: (1) msg.value transfer graph (direct ETH transfers), (2) funder graph (wallet creation funding paths), (3) sweep graph (token consolidation flows).
- **Louvain community detection**: Applied to each graph type independently, with communities exceeding size thresholds flagged as Sybil clusters.
- **Hop Protocol blacklists**: Integration with Hop Protocol's manually curated Sybil elimination lists as additional ground truth.
- **Limitation**: Similar to TrustaLabs, assumes dense subgraph structure that AI agents can strategically avoid.

### BrightID: SybilRank Algorithms

BrightID explores social-graph-based Sybil resistance:

- **SybilRank**: Trust propagation from seed nodes in a social graph; Sybil nodes receive less trust due to sparse connections to the honest region.
- **GroupSybilRank and WeightedSybilRank**: Extensions that account for group memberships and weighted connections.
- **Relevance**: Provides an alternative detection paradigm (social graph vs. transaction graph) that may complement on-chain analysis but requires off-chain identity verification infrastructure.

### Hop Protocol: Sybil Attacker Elimination Lists

Hop Protocol published manually curated lists of confirmed Sybil addresses:

- **Ground truth source**: Used by HasciDB and other studies as labeled data.
- **Methodology**: Manual investigation of suspicious address clusters based on funding patterns, transaction timing, and on-chain behavior.
- **Limitation**: Manual curation does not scale and is reactive rather than proactive.

## 2.2 Pre-Airdrop Detection

### Wen et al. (pre-airdrop-detection)

Wen et al. (Adeline117/pre-airdrop-detection) propose a proactive detection approach that identifies Sybil addresses *before* the airdrop snapshot:

- **Temporal framing**: Train on behavioral features extracted T days before the airdrop snapshot, enabling preemptive detection.
- **Feature engineering**: LightGBM model using temporal behavioral features including transaction frequency, value distributions, protocol interaction patterns, and temporal activity profiles.
- **Performance**: AUC 0.793 at T-30 (30 days before snapshot) on Blur Season 2, demonstrating feasibility of pre-airdrop detection.
- **Dataset**: Blur Season 2 with 53K airdrop recipients, 9.8K confirmed Sybils from HasciDB labels.
- **Relevance to this work**: The pre-airdrop feature set provides a second baseline, and its temporal approach complements our AI-specific features that target agent execution characteristics rather than temporal patterns.

### ARTEMIS (UW-DCL/Blur)

ARTEMIS (UW-Decentralized-Computing-Lab/Blur) applies graph neural networks to Sybil detection:

- **Architecture**: 3-layer GNN combining a custom ArtemisFirstLayerConv with GraphSAGE aggregation on the Blur Season 2 transaction graph.
- **Performance**: AUC 0.803 in post-hoc (after snapshot) detection on the same Blur S2 dataset.
- **Graph features**: Node features derived from transaction patterns; edge features from transfer amounts and frequencies.
- **Limitation**: Post-hoc detection (after airdrop distribution) limits practical utility for prevention. GNN training requires the full transaction graph, which may not be available in real-time.

### LLMhunter (UW-DCL/Blur)

LLMhunter (also from UW-Decentralized-Computing-Lab/Blur) explores LLM-based Sybil judgment:

- **Multi-expert pipeline**: Uses multiple LLM "experts" that independently evaluate address behavior profiles and vote on Sybil classification.
- **Natural language reasoning**: Each expert provides chain-of-thought reasoning about suspicious patterns, enabling interpretable decisions.
- **Relevance**: Demonstrates that LLMs can be used for detection (defensive AI), complementing our study of LLMs being used for evasion (offensive AI). This duality highlights the AI arms race dynamic.

## 2.3 Adversarial Machine Learning

### Foundational Adversarial Attacks

**Goodfellow, Shlens, and Szegedy (2014)** introduced adversarial examples -- imperceptible perturbations to inputs that cause neural networks to misclassify. The Fast Gradient Sign Method (FGSM) demonstrated that deep learning models are systematically vulnerable to adversarial manipulation. This foundational work establishes the theoretical basis for understanding how AI agents can craft evasive on-chain behavior.

**Carlini and Wagner (2017)** developed stronger optimization-based attacks that bypass many proposed defenses, establishing rigorous evaluation standards for adversarial robustness. Their key insight -- that defenses must be evaluated against adaptive adversaries who know the defense mechanism -- directly applies to our setting: AI agents can adapt their evasion strategies based on knowledge of detector architectures.

### Adversarial ML in Security Contexts

**Apruzzese et al. (2023)** survey adversarial machine learning in network security, identifying key challenges:

- **Feature-space vs. problem-space attacks**: In network security (and by extension, blockchain analysis), adversaries must produce valid inputs that satisfy domain constraints, not arbitrary feature vectors.
- **Adaptive adversaries**: Security adversaries actively probe and adapt to deployed defenses, creating an arms race dynamic.
- **Evasion vs. poisoning**: Our AI Sybil generation is primarily an evasion attack (crafting inputs that bypass a trained detector), though training-time poisoning through injected Sybil data is also a concern.

### Adversarial Attacks in Blockchain

Several works have explored adversarial manipulation in blockchain contexts:

- **Transaction graph evasion**: Techniques to restructure fund flows to avoid graph-based detection, analogous to our AI Sybil strategies for evading MA and RF indicators.
- **Bot detection evasion**: MEV bots and arbitrage bots that adopt human-like transaction patterns to avoid detection, a precursor to the AI-agent evasion we study.
- **Smart contract mimicry**: Deploying contracts that mimic legitimate protocol interactions to farm airdrop eligibility.

## 2.4 AI Agent Behavior on Blockchain

### Paper 0: AI Agent Taxonomy

Our companion Paper 0 establishes a taxonomy of AI agents operating on Ethereum:

- **Category classification**: Trading bots, MEV searchers, portfolio managers, social agents, and autonomous protocols.
- **Behavioral signatures**: Each agent category exhibits characteristic on-chain patterns in gas usage, transaction timing, protocol interaction diversity, and error handling.
- **Relevance**: The taxonomy informs our AI Sybil sophistication levels -- basic agents correspond to simple trading bots, while advanced agents leverage LLM-orchestrated behavior that mimics the organic diversity of human users.

### Paper 1: On-Chain AI Agent Identification

Paper 1 develops 8 behavioral features for distinguishing AI agents from human users based on on-chain transaction data:

1. **gas_price_precision**: AI agents compute optimal gas prices with high precision; humans tend to use round numbers or wallet defaults.
2. **hour_entropy**: AI agents operate 24/7 with near-uniform temporal distribution; humans exhibit circadian rhythms.
3. **behavioral_consistency**: Addresses controlled by the same LLM show correlated behavior patterns (same prompt -> similar actions).
4. **action_sequence_perplexity**: LLM-generated action sequences fall within a characteristic perplexity range, distinct from both human randomness and scripted determinism.
5. **error_recovery_pattern**: AI agents exhibit systematic retry/fallback patterns consistent with programmatic error handling.
6. **response_latency_variance**: Transaction inter-arrival times reflect LLM inference latency characteristics.
7. **gas_nonce_gap_regularity**: AI agents maintain regular nonce increments; humans have more variable gaps.
8. **eip1559_tip_precision**: AI agents compute EIP-1559 priority fees with mathematical precision; humans rely on wallet estimates.

These 8 features transfer directly to our enhanced Sybil detector, providing the AI-specific signal that restores detection capability against evasive AI Sybils. The key insight is that while AI agents can carefully control HasciDB's 5 behavioral indicators (which measure *what* the address does), they leak identity through *how* they execute transactions -- an execution-level signal that is fundamentally harder to fake.
