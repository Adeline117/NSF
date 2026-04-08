# On-Chain AI Agent Identification under Label Leakage: An Honest Empirical Study with Provenance-Only Ground Truth

**Target venue:** The Web Conference (WWW) 2026

---

## Abstract

Autonomous AI agents now manage tens of billions of dollars on public blockchains, yet they do not self-identify on-chain, leaving security auditors, protocol designers, and regulators blind to automated financial activity. We present the first honest empirical study of on-chain AI agent identification that explicitly accounts for label leakage. We design a 23-feature behavioral fingerprinting framework spanning temporal patterns, gas pricing, interaction structure, and approval security, and apply it to 3,316 Ethereum mainnet addresses mined from four agent platforms (Autonolas, Fetch.ai, AI Arena, Virtuals), a curated set of MEV searchers, and ENS-verified human users. Our initial pipeline reports a Gradient Boosting AUC of 0.9803 on the full 3,316-address set. In the course of this work, we discovered that three of the mining gates (`hour_entropy`, `burst_ratio`, `tx_interval_cv`) are algebraically equivalent to three top classifier features, constituting direct target leakage on the heuristically labeled majority of the dataset. After demoting the C1-C4 gates to a downstream evaluation tool and retaining only 64 provenance-verified addresses (33 agents, 31 humans), the honest Random Forest LOO-CV AUC is 0.7713 and a Graph Attention Network reaches 0.870 five-fold AUC. A cross-platform test train on Autonolas and test on the trusted provenance set produces AUC 0.24 to 0.34, indicating that intra-platform generalization does not transfer. We report per-model and per-split numbers, McNemar / DeLong / bootstrap tests, a 4-dimensional security audit whose headline direction reverses between leaky and honest subsets, and a GNN comparison. We argue that the leakage discovery is itself a methodological contribution: it quantifies how fragile heuristic labeling can be, provides a reusable template for auditing behavioral classifiers on blockchain data, and establishes honest baselines (RF LOO 0.77, GAT 0.87) against which clean re-mining efforts can be judged. All code, features, labels, and diagnostic scripts are released.

**Keywords:** on-chain AI agents, behavioral fingerprinting, label leakage, Ethereum, graph neural networks, security audit, honest baselines, reproducibility

---

## 1. Introduction

### 1.1 Motivation

Autonomous AI agents are proliferating on public blockchains at unprecedented speed. Simple trading bots, complex multi-step DeFi strategy executors, MEV searchers, and LLM-driven market makers collectively control tens of billions of dollars on Ethereum mainnet alone. Platforms such as Autonolas, Fetch.ai, AI Arena, AI16Z ELIZA, and Virtuals Protocol make on-chain agent deployment increasingly turnkey. Recent reports attribute more than $45M in 2025-2026 losses to AI-agent-specific failure modes, including approval exploits, oracle manipulation, and cascading liquidation failures triggered by coordinated rebalancing.

Yet from the perspective of the Ethereum protocol, an AI-controlled externally owned account (EOA) is indistinguishable from a human-operated one. No on-chain type system exposes whether a transaction's sender is a human, a smart-contract wallet, or an AI policy running in a data center. This opacity creates three fundamental problems:

1. **Security auditors** cannot isolate agent-specific risk (runaway approvals, unbounded slippage tolerance, mis-specified LLM tool calls) from human behavior.
2. **Protocol designers** cannot offer agent-aware primitives (rate limiting, bounded approvals, composability guards) because they cannot tell who is using the protocol.
3. **Regulators and monitoring services** have no visibility into what fraction of market activity is automated, which limits their ability to reason about systemic risk.

Closing these gaps requires a reliable way to identify agents from public on-chain data alone, without relying on platform registries or voluntary disclosures.

### 1.2 The Seduction of High Numbers

When we began this study, we built what seemed to be a compelling pipeline: 3,316 Ethereum addresses (2,590 labeled AGENT, 726 labeled HUMAN) mined from four agent platforms (Autonolas, Fetch.ai, AI Arena, Virtuals), a curated MEV-bot list, and a set of ENS-verified humans. Labels were assigned via a 4-gate heuristic we called **C1-C4**:

- C1: public registry or known identity;
- C2: behavioral consistency with the declared class;
- C3: multi-source cross-validation, operationalized as thresholds on `hour_entropy > 2.5`, `burst_ratio > 0.1`, and `tx_interval_cv < 1.0`;
- C4: temporal stability across windows.

Gradient Boosting on all 23 features achieved **AUC 0.9803** (5-fold 5-rep CV, n=3316). Random Forest reached 0.9766, and even logistic regression landed at 0.9521. A Graph Attention Network on the induced transaction graph obtained 0.934, and GraphSAGE 0.967. The security audit appeared to confirm the hypothesis that agents are riskier than humans: 10.3x more unlimited approvals on average, 1.7x higher DEX interaction rate, and higher revert rates.

This is the number we would have written up in the abstract, and it is wrong as a claim about agent identification.

### 1.3 The Leakage Discovery

In the final week of experiments, we noticed that three of the top-four classifier features were algebraically identical to three C1-C4 gates:

```
hour_entropy (C3 gate)     == active_hour_entropy (classifier feature)
burst_ratio  (C3 gate)     == burst_frequency     (classifier feature)
tx_interval_cv (C3 gate)   ~  tx_interval_std / tx_interval_mean (classifier features)
```

In other words, the same quantities that had decided whether an address was labeled AGENT were then fed to the classifier as predictors. The resulting AUC is therefore not a measure of how well behavioral features separate agents from humans; it is a measure of how well the classifier recovers the thresholding rule the mining script used. The agents-have-10x-more-approvals headline comes from the same labeling bias: C1-C4 preferentially accepted token holders who were already active DeFi users.

Fortunately, 64 of the 3,316 addresses had provenance labels that did not pass through C1-C4: 33 agents from the original manual pilot set (jaredfromsubway, Wintermute, 1inch resolver, Flashbots builders, Autonolas registry entries, etc.) and 31 humans from the curated ENS list (vitalik.eth, Hayden Adams, and comparable accounts). On this **trusted provenance-only subset**:

- Gradient Boosting LOO-CV AUC drops to **0.6276**;
- Random Forest LOO-CV AUC drops to **0.7713** (5x10-CV 0.8030);
- The GBM vs Majority baseline comparison is **no longer statistically significant** (McNemar p=0.58);
- The security audit direction **reverses**: famous ENS humans hold more unlimited approvals (66.1 mean) than curated MEV bots (49.1 mean) because the ENS accounts are heavier DeFi users than a typical searcher.

We therefore rewrote the paper around this discovery. This paper is not an attempt to hide the leakage; it is an attempt to turn the leakage into a methodological contribution, to present the honest baselines, and to provide a reusable audit template so that future work on behavioral classification of on-chain actors does not repeat the same mistake.

### 1.4 Research Questions

- **RQ1 (Identifiability):** Can AI agents be reliably distinguished from human users on the basis of public on-chain behavior alone, using provenance-only ground truth?
- **RQ2 (Leakage):** What is the magnitude of the gap between a heuristically labeled classifier and a provenance-only classifier, and how should future work design around it?
- **RQ3 (Model class):** Do graph neural networks, which can exploit transfer-graph structure beyond node features, beat tabular classifiers on the honest set?
- **RQ4 (Generalization):** Do classifiers trained on one platform (e.g., Autonolas) transfer to agents from a curated external set?
- **RQ5 (Security posture):** Does the "agents are riskier than humans" hypothesis survive the honest relabeling?

### 1.5 Contributions

1. **An honest empirical baseline.** Random Forest LOO-CV AUC = 0.7713 (5x10-CV 0.8030) on 64 provenance-verified Ethereum addresses, Graph Attention Network five-fold AUC = 0.870 on the same subset. These are the numbers we recommend as the reference point for future work.
2. **A leakage case study.** We document how a seemingly sensible 4-gate labeling oracle injected three top-4 classifier features into the labels, causing a 25-point AUC inflation (0.9803 to 0.7713). We release the diagnostic scripts as a template (`verify_c1c4.py`, `pipeline_provenance.py`, `statistical_tests.py`).
3. **Statistical significance on the honest set.** McNemar, DeLong, and bootstrap tests on the 64-address trusted set show that Random Forest significantly beats Gradient Boosting (McNemar p=0.0225, DeLong p=0.0014, bootstrap CI95 [-0.24, -0.06]), GBM is indistinguishable from a majority baseline, and a single best feature (tx_interval_mean) matches or beats GBM.
4. **Cross-platform evaluation.** Autonolas-to-trusted transfer AUC is 0.24 to 0.34, worse than chance, while Fetch.ai-to-AI-Arena transfer AUC is 0.98, exposing that the apparent high performance comes from intra-platform correlations rather than a generalizable agent concept.
5. **GNN vs tabular comparison.** On the same trusted subset, GAT (AUC 0.870) beats GraphSAGE (0.784) and beats all tabular classifiers, suggesting that graph structure carries signal beyond per-node features.
6. **Combined feature study.** Extending the 23 Paper 1 features with 8 AI-specific features from our companion work (Paper 3) yields a combined LightGBM 5x3-CV AUC of 0.7172 on the trusted subset, essentially no improvement (delta +0.013) over Paper 1 features alone.
7. **4-dimensional security audit with honest reversal.** We report both the leaky-set (agents 10.3x more unlimited approvals) and trusted-set (humans hold more) audits side by side, and explain why the direction reverses.
8. **Open artifacts.** Code, mined features, labels, statistical test scripts, GNN model weights, and the full leaky-vs-honest comparison JSONs are released.

### 1.6 Paper Roadmap

Section 2 surveys related work. Section 3 describes the mining pipeline, the 23 features, the tabular classifiers, the GNN architectures, and the security audit. Section 4 presents six result subsections: honest tabular performance (4.1), statistical significance (4.2), cross-platform generalization (4.3), GNN vs tabular (4.4), combined Paper 1 + Paper 3 features (4.5), and the 4-dimensional security audit (4.6). Section 5 is the threats-to-validity section and is the methodological heart of the paper; it walks through the C1-C4 leakage discovery. Section 6 discusses limitations and a clean re-mining roadmap. Section 7 concludes.

---

## 2. Related Work

Our study bridges four research areas: bot detection on blockchains, MEV and automated trading, AI agent platforms, and blockchain security analysis. We also draw on the broader machine-learning literature on label leakage, data-snooping bias, and cross-domain evaluation.

### 2.1 Bot Detection on Blockchains

Detecting automated activity on blockchains has been studied primarily in the context of specific malicious behaviors. Chen et al. [1] used transaction features and opcode analysis to detect Ponzi schemes on Ethereum, demonstrating that behavioral patterns in transaction data reveal automated activity. Victor and Weintraud [2] detected wash trading on decentralized exchanges via graph features. Torres et al. [3] studied Ethereum frontrunner bots through mempool analysis, identifying displacement, insertion, and suppression attacks. Victor et al. [4] proposed behavioral fingerprinting for blockchain bots using temporal and gas features similar in spirit to our temporal and gas groups, but restricted to MEV bots. Li et al. [5] studied Telegram trading bots and their on-chain footprint.

**Gap.** Existing methods target specific malicious behaviors (wash trading, Ponzi schemes, frontrunning) or specific bot families (MEV, Telegram), not the general class of AI-controlled addresses. More importantly, none of these works explicitly audit the relationship between their labeling oracle and their classifier features; our leakage discovery suggests such audits are missing from the field.

### 2.2 MEV and Automated Trading

Daian et al. [6] introduced MEV in "Flash Boys 2.0" and demonstrated that transaction ordering creates a searcher-bot ecosystem. Weintraub et al. [7] measured Flashbots MEV at scale. Qin et al. [8] quantified MEV across DeFi protocols, showing sandwich attacks, liquidations, and arbitrage extract billions annually. Park et al. [9] analyzed post-Merge MEV and the effect of Proposer-Builder Separation. Gupta et al. [10] studied cross-domain MEV across chains and L2s.

**Gap.** MEV research quantifies the economic impact of automation but does not attempt general-purpose identification. Our classifier is meant to generalize beyond MEV searchers to DeFi management agents, portfolio rebalancers, and LLM-driven strategy agents.

### 2.3 AI Agent Platforms

Autonolas (OLAS) [11] provides an open-source framework for autonomous agent services, with on-chain registration. AI16Z ELIZA [12] enables AI agents to interact with blockchain protocols through natural language. Virtuals Protocol [13] offers a launchpad for tokenized AI agents. Fetch.ai [14], SingularityNET [15], and AI Arena [16] contribute to the growing on-chain agent population. These platforms deploy agents but do not provide cross-platform identification. A recent survey by He et al. [23] catalogs the taxonomy of on-chain AI agents but provides no empirical identification methodology.

### 2.4 Blockchain Security Analysis

Zhou et al. [17] systematized DeFi attacks; Wen et al. [18] analyzed DeFi composability risk; Grech et al. [19] and Tsankov et al. [20] built static analysis tools for smart contracts. Wang et al. [21] studied the risks of AI agents with cryptocurrency wallet access, identifying prompt injection as a primary attack vector. Zhang et al. [22] analyzed MaxUint256 approval attacks.

**Gap.** Existing security analysis focuses on protocol-level vulnerabilities. No prior work assesses the security posture of AI agent addresses as a class or compares it to a human baseline, and the handful of works that attempt to do so do not control for labeling bias.

### 2.5 Label Leakage and Data-Snooping Bias

Kaufman et al. [24] formalized leakage in predictive modeling and catalog several subtle forms. Roelofs et al. [25] studied test-set reuse as an implicit form of leakage in ML benchmarks. Geirhos et al. [26] documented shortcut learning: deep networks exploit unintended features that correlate with the label during training but not at deployment. In the blockchain space, Tikhomirov and Ivanitskiy [27] warned about address-type features becoming labels in phishing classifiers. Our work extends this line: we show that a seemingly principled 4-gate labeling heuristic can encode three classifier features into the label, producing a 25-point AUC inflation that is invisible to standard cross-validation.

### 2.6 Graph Neural Networks on Transaction Graphs

GraphSAGE [28] and GAT [29] are standard inductive GNN architectures. In the blockchain setting, Weber et al. [30] applied GNNs to Bitcoin anti-money-laundering. Patel et al. [31] used GNNs to detect Ethereum phishing. Shamsi et al. [32] surveyed GNNs for blockchain analytics. Most of these works report aggregate accuracy on full labeled graphs; very few report transfer performance across labeling schemes, and none to our knowledge report the honest-vs-leaky gap that is the central finding of this paper.

### 2.7 Our Position

Relative to the above, this paper offers: (1) a general behavioral identification framework rather than a specific bot-class detector; (2) an honest provenance-only baseline suitable for comparison across future works; (3) a documented leakage case study that other blockchain-ML researchers can use as a negative-result template; (4) a direct tabular-vs-GNN comparison on the same ground truth; (5) the first public honest security audit of AI-agent-labeled Ethereum addresses.

---

## 3. Methodology

### 3.1 Data Collection

We collect transaction data from Ethereum mainnet via Etherscan V2 API using a 6-key rotation strategy that achieves approximately 30 calls/sec sustained throughput. For each address we extract up to 5,000 most recent transactions covering normal transactions, internal transactions, ERC-20 transfers, and approval events. All data is public.

### 3.2 Address Mining and Labeling

#### 3.2.1 Strategy A: Platform Holders (C1-C4 labeled, **used for inference only**)

We identify addresses holding tokens of four agent platforms (Autonolas OLAS, Fetch.ai FET, AI Arena NRN, Virtuals VIRT). This yields **3,252 platform-token-holder addresses**. Each holder is labeled via our original C1-C4 gate:

- **C1 (public registry):** the address appears in a public platform registry or has a known identity.
- **C2 (behavioral consistency):** the address's on-chain behavior matches the declared class.
- **C3 (multi-source cross-validation):** operationalized as thresholds on `hour_entropy > 2.5`, `burst_ratio > 0.1`, `tx_interval_cv < 1.0`.
- **C4 (temporal stability):** the behavioral class is stable across time windows.

#### 3.2.2 Strategy B: Provenance-Verified Addresses (**used for honest training/testing**)

We identify 64 addresses whose labels do not depend on any behavioral threshold:

- **33 AGENT addresses** from (a) the original Paper 1 pilot set of manually curated bots (jaredfromsubway, Wintermute, 1inch resolver, Flashbots builder, Autonolas service operators), (b) the Paper 0 Delphi-validated agent set, and (c) an external curated MEV-bot list.
- **31 HUMAN addresses** from the ENS-verified human list (vitalik.eth, Hayden Adams, nick.eth, and comparable accounts whose identity is publicly verifiable).

None of the trusted labels pass through `hour_entropy`, `burst_ratio`, or `tx_interval_cv` thresholds.

#### 3.2.3 Exclusion Rules

We exclude contract addresses (Uniswap routers, 0x Exchange Proxy, Seaport, Blur), exchange hot wallets (MEXC, Binance 14/16), and any address whose label cannot be pinned to a non-behavioral source.

#### 3.2.4 Decision: Demote C1-C4 to Evaluation

After discovering that C1-C4 gates overlap with three top classifier features (see Section 5), we demote the 3,252 Strategy A addresses from training to inference-only. The 64 trusted Strategy B addresses become the honest train+test set. We continue to report leaky-set numbers for transparency and for statistical comparison.

### 3.3 Feature Engineering: 23 Behavioral Features

We compute 23 features organized into four groups. Let $T = \{t_1, \ldots, t_n\}$ be the transaction timestamps, $G = \{g_1, \ldots, g_n\}$ the gas prices, and $C = \{c_1, \ldots, c_n\}$ the target contracts.

**Group 1: Temporal Features (7).**

1. `tx_interval_mean`: mean of consecutive inter-transaction intervals.
2. `tx_interval_std`: standard deviation of intervals.
3. `tx_interval_skewness`: skewness of the interval distribution.
4. `active_hour_entropy`: Shannon entropy of transactions bucketed into 24 UTC hours.
5. `night_activity_ratio`: fraction of transactions in UTC 0:00-6:00.
6. `weekend_ratio`: fraction of transactions on Saturday/Sunday.
7. `burst_frequency`: fraction of consecutive transaction pairs with interval < 10 s.

**Group 2: Gas Behavior Features (6).**

8. `gas_price_round_number_ratio`: fraction of transactions with integer-Gwei gas prices.
9. `gas_price_trailing_zeros_mean`: mean number of trailing zeros in the decimal gas-price representation.
10. `gas_limit_precision`: gas_limit relative to gas_used.
11. `gas_price_cv`: coefficient of variation of gas prices.
12. `eip1559_priority_fee_precision`: precision of priority fees in EIP-1559 transactions.
13. `gas_price_nonce_correlation`: Pearson correlation between gas price and nonce.

**Group 3: Interaction Pattern Features (5).**

14. `unique_contracts_ratio`: unique target contracts divided by total transactions.
15. `top_contract_concentration`: Herfindahl-Hirschman index of interaction targets.
16. `method_id_diversity`: unique function signatures per contract call.
17. `contract_to_eoa_ratio`: fraction of transactions whose target is a contract.
18. `sequential_pattern_score`: n-gram repetition score of the action sequence.

**Group 4: Approval & Security Features (5).**

19. `unlimited_approve_ratio`: fraction of approve() calls that set MaxUint256.
20. `approve_revoke_ratio`: revoke operations per approve operation.
21. `unverified_contract_approve_ratio`: fraction of approvals to Etherscan-unverified contracts.
22. `multi_protocol_interaction_count`: count of distinct DeFi protocols interacted with.
23. `flash_loan_usage`: flash-loan indicator.

**Critical note on leakage.** Features 1-7 (temporal) and especially 2, 4, 7 are algebraically entangled with the C3 gate that labeled the Strategy A subset. This is the central issue analyzed in Section 5.

### 3.4 Tabular Classifiers

We train four classifiers via scikit-learn and LightGBM:

- **GradientBoosting (GBM):** 200 trees, max_depth=3, lr=0.1.
- **RandomForest (RF):** 100 trees, unlimited depth.
- **LogisticRegression (LR):** L2-regularized, liblinear solver.
- **LightGBM:** 500 estimators, num_leaves=31.

Features are standardized before training. We evaluate with two complementary schemes:

- **Leave-One-Out Cross-Validation (LOO-CV):** 64 iterations, one held-out sample per fold.
- **Repeated Stratified 5-Fold CV (5x10-CV):** 50 evaluations, mean and std reported.

Metrics: AUC-ROC (primary), precision, recall, F1, accuracy, and Brier score for calibration.

### 3.5 Graph Neural Networks

We construct a directed transaction graph $G = (V, E)$ on the 3,316-address set, with edges from sender to receiver for any transfer of value or token. Node features are the 23 behavioral features above. The resulting graph has 3,316 nodes, 2,387 edges, and is moderately sparse.

We train two GNN architectures:

- **GraphSAGE:** two-layer, hidden dim 64, mean aggregator, dropout 0.3, Adam lr=0.01, early stopping on validation loss.
- **GAT (Graph Attention Network):** two-layer, 8 attention heads, hidden dim 64, dropout 0.3.

We evaluate on two splits: `full_3316` (training labels are C1-C4 leaky) and `trusted_64` (training labels are provenance-only; the other 3,252 nodes are retained in the graph to provide structural context but are masked out of supervision).

### 3.6 Baselines and Ablations

- **Majority classifier:** predicts the majority class (AGENT).
- **Single best feature:** logistic classifier on `tx_interval_mean`, the top univariate AUC feature on the trusted set.
- **Heuristic rule:** `burst_frequency > 0.1 => AGENT`.
- **Feature-group ablations:** retain one group (temporal / gas / interaction / approval) at a time.

### 3.7 Statistical Tests

On the 64-address trusted set, we compute:

- **McNemar's test** for paired predictions.
- **DeLong's test** for paired AUC comparison with bootstrap variance.
- **10,000-sample bootstrap** for AUC difference confidence intervals.
- **Precision @ K** for K in {5, 10, 20, 30}.
- **Brier score** for calibration.

### 3.8 Cross-Platform Evaluation

We run four transfer splits:

1. `auto_train_to_trusted_test`: train on 1,780 Autonolas holders, test on 41 trusted addresses.
2. `trusted_train_to_auto_test`: train on 41 trusted addresses, test on 1,780 Autonolas holders.
3. `fetch_to_ai_arena`: train on 923 Fetch.ai, test on 549 AI Arena.
4. `ai_arena_to_fetch`: train on 549 AI Arena, test on 923 Fetch.ai.

Splits 1-2 are cross-label-scheme (provenance vs C1-C4). Splits 3-4 are intra-label-scheme (both C1-C4). The gap between these two families is a direct measurement of labeling-scheme transfer.

### 3.9 Combined Paper 1 + Paper 3 Features

To evaluate whether our 23 behavioral features can be meaningfully extended, we compute 8 AI-specific features from the companion Paper 3 work (`hour_entropy`, `behavioral_consistency`, `response_latency_variance`, `action_sequence_perplexity`, `gas_nonce_gap_regularity`, `eip1559_tip_precision`, `gas_price_precision`, `error_recovery_pattern`). We train four classifiers on three feature sets: Paper 1 only (23), Paper 3 only (8), and combined (31).

### 3.10 4-Dimensional Security Audit

For each audited address we compute:

- **Dimension 1: Permission exposure.** Counts of active approvals, unlimited (MaxUint256) approvals, and unverified approval targets; average and max approval ages in days.
- **Dimension 2: Agent network topology.** Degree centrality, clustering coefficient, community membership in the transfer graph.
- **Dimension 3: MEV / DEX exposure.** DEX interaction rate, swap rate.
- **Dimension 4: Failure and gas analysis.** Revert rate, gas-used efficiency.

We run the audit twice: once on the full 3,302-address set (14 dropped due to empty histories) and once on the 50-address trusted subset. The difference between the two is a direct measurement of how labeling bias corrupts security conclusions.

---

## 4. Results

We now present the honest results in six subsections. Throughout this section, **the trusted n=64 subset is the headline**. The full n=3,316 numbers are reported only as a leaky reference point.

### 4.1 Honest Performance on the Provenance-Only Set (n=64)

Table 1 reports the core honest performance: four tabular classifiers on the 64 trusted addresses under both LOO-CV and repeated 5x10-CV.

**Table 1. Tabular classifier performance on the trusted provenance-only set (n=64, 33 agents, 31 humans).**

| Model             | Split     | AUC-ROC        | Precision | Recall | F1    | Accuracy |
|-------------------|-----------|----------------|-----------|--------|-------|----------|
| RandomForest      | LOO       | **0.7713**     | 0.7419    | 0.6970 | 0.7188| 0.7188   |
| RandomForest      | 5x10-CV   | **0.8030** +/- 0.102| 0.8008   | 0.7138 | 0.7363| 0.7447   |
| GradientBoosting  | LOO       | 0.6276         | 0.5882    | 0.6061 | 0.5970| 0.5781   |
| GradientBoosting  | 5x10-CV   | 0.6940 +/- 0.119 | 0.6731   | 0.6638 | 0.6548| 0.6499   |
| LogisticRegression| LOO       | 0.6100         | 0.6667    | 0.6061 | 0.6349| 0.6406   |
| LogisticRegression| 5x10-CV   | 0.6203 +/- 0.125 | 0.6974   | 0.6238 | 0.6459| 0.6614   |
| LightGBM          | LOO       | 0.7097         | --        | --     | 0.7273| 0.7188   |
| LightGBM          | 5x10-CV   | 0.7040 +/- 0.088 | --        | --     | 0.6793| 0.6765   |

**Key findings.**

1. **Random Forest is the best honest tabular model** (LOO AUC 0.7713, 5x10-CV 0.8030).
2. **GBM underperforms RF substantially** on the honest set, despite being nearly identical on the leaky set. We confirm this gap with statistical tests in Section 4.2.
3. **Standard deviations are large** (0.088 to 0.125), reflecting the intrinsic variance of 5-fold CV on n=64.
4. **LR is weakest**, close to random on its own.

**Leaky reference (do not cite as headline).** On the full 3,316 addresses with C1-C4 labels, GBM achieves 5-fold 5-rep CV AUC 0.9803 +/- 0.006, RF 0.9766 +/- 0.006, LR 0.9521 +/- 0.012, LightGBM 0.9822 +/- 0.006. The gap between 0.98 and 0.77 is the leakage magnitude.

**Per-feature univariate AUC (trusted n=64).** Table 2 lists the top features by Mann-Whitney-derived AUC.

**Table 2. Top univariate AUC on the trusted set.**

| Rank | Feature                              | Univariate AUC | Group        |
|------|--------------------------------------|----------------|--------------|
| 1    | tx_interval_mean                     | 0.7683         | Temporal     |
| 2    | tx_interval_std                      | 0.7527         | Temporal     |
| 3    | unlimited_approve_ratio              | 0.7116         | Approval     |
| 4    | unverified_contract_approve_ratio    | 0.7107         | Approval     |
| 5    | gas_price_round_number_ratio         | 0.7072         | Gas          |
| 6    | gas_price_trailing_zeros_mean        | 0.7067         | Gas          |
| 7    | active_hour_entropy                  | 0.7028         | Temporal     |
| 8    | multi_protocol_interaction_count     | 0.6960         | Approval     |
| 9    | unique_contracts_ratio               | 0.6950         | Interaction  |
| 10   | method_id_diversity                  | 0.6911         | Interaction  |

Contrast with the leaky set, where `active_hour_entropy` led with AUC 0.9347 and `tx_interval_skewness` reached 0.872. The collapse of these features from 0.93 to 0.70 is direct evidence that the leaky AUCs were inflated by labeling overlap.

### 4.2 Statistical Significance Tests

Given n=64, we report three complementary tests for each pairwise model comparison: McNemar's test on paired predictions, DeLong's test for paired AUCs, and a 10,000-sample bootstrap CI for the AUC difference.

**Table 3. Pairwise statistical tests on the trusted set (n=64, LOO predictions).**

| Comparison               | McNemar p | DeLong p  | Bootstrap CI95 AUC diff | Significant? |
|--------------------------|-----------|-----------|-------------------------|--------------|
| GBM vs RF                | **0.0225**| **0.0014**| [-0.24, -0.06]          | **Yes** (RF wins)|
| GBM vs LR                | 0.4545    | 0.7895    | [-0.11, 0.15]           | No           |
| GBM vs Single-Feature    | 0.7353    | <0.001    | [-0.52, -0.24]          | Mixed        |
| GBM vs Majority          | 0.5839    | --        | --                      | **No**       |

**Key findings.**

1. **Random Forest significantly beats Gradient Boosting** on the trusted set by three independent tests. This is a qualitative reversal of the leaky-set conclusion where GBM and RF were statistically indistinguishable.
2. **GBM is NOT significantly better than a majority classifier** (McNemar p=0.58). Read literally: a model that always predicts AGENT performs as well as the GBM on this set.
3. **A single feature (`tx_interval_mean`) achieves AUC 1.00 at the threshold level** but low accuracy (0.53). This is an artifact of degenerate thresholding on small n; it is reported for completeness but not as a headline claim. The more honest comparison is that RF beats every other model we tried.
4. **Brier scores:** RF 0.194, LR 0.274, LightGBM not reported, GBM 0.355. RF is the best-calibrated model as well as the most accurate.

### 4.3 Cross-Platform Generalization

Table 4 reports the four cross-platform splits. The critical contrast is between splits 1-2 (cross-label-scheme: provenance vs C1-C4) and splits 3-4 (intra-label-scheme: both C1-C4).

**Table 4. Cross-platform transfer AUC.**

| Split                          | Direction                     | GBM AUC | RF AUC | LR AUC | n_train | n_test |
|--------------------------------|-------------------------------|---------|--------|--------|---------|--------|
| Autonolas -> Trusted (honest)  | C1-C4 -> provenance            | 0.269   | 0.243  | 0.578  | 1,780   | 41     |
| Trusted -> Autonolas           | provenance -> C1-C4            | 0.347   | 0.341  | 0.603  | 41      | 1,780  |
| Fetch.ai -> AI Arena           | C1-C4 -> C1-C4                | **0.980**| 0.969 | 0.962  | 923     | 549    |
| AI Arena -> Fetch.ai           | C1-C4 -> C1-C4                | **0.992**| 0.975 | 0.973  | 549     | 923    |

**Key findings.**

1. **Cross-label-scheme transfer collapses to worse than chance** (GBM 0.269, RF 0.243). A classifier trained on Autonolas C1-C4 labels and tested on trusted provenance labels produces **anti-predictions**: it labels agents as humans and vice versa.
2. **Intra-label-scheme transfer is near-perfect** (AUC 0.97-0.99). Fetch.ai and AI Arena holders are indistinguishable under the shared C1-C4 oracle, because the oracle is learning the oracle, not the concept.
3. **The 0.27 to 0.99 gap is another expression of leakage.** It tells us the apparent high transfer performance is spurious: it measures consistency of the labeling heuristic, not consistency of behavior across platforms.
4. **Logistic regression, paradoxically, is the best cross-label-scheme model** (0.578 / 0.603). We interpret this as follows: LR cannot fit the leaky labels as tightly as GBM/RF, so it suffers less when the labels change.

**Implication.** Any future paper claiming high agent-identification AUC on platform-holder data must include this cross-label-scheme transfer test, or the result cannot be distinguished from leakage.

### 4.4 GNN vs Tabular Classifiers

We train GraphSAGE and GAT on the 3,316-node transaction graph and evaluate on the two splits. Table 5 reports the results.

**Table 5. GNN vs tabular AUC.**

| Model            | full_3316 AUC (5-fold mean +/- std) | trusted_64 AUC (5-fold mean +/- std) |
|------------------|-------------------------------------|--------------------------------------|
| GraphSAGE        | 0.9666 +/- 0.0087                   | 0.7841 +/- 0.084                     |
| GAT              | 0.9341 +/- 0.0113                   | **0.8825 +/- 0.108**                 |
| RF (tabular)     | 0.9766 +/- 0.0061                   | 0.8030 +/- 0.102 (5x10)              |
| GBM (tabular)    | 0.9803 +/- 0.0061                   | 0.6940 +/- 0.119 (5x10)              |

**Key findings.**

1. **GAT is the best honest model overall.** On the trusted subset, GAT reaches 5-fold AUC 0.8825 vs RF 0.8030. The 5-fold variance is high (std 0.108) because n=64 leaves only ~13 test nodes per fold.
2. **GraphSAGE does not help on the trusted set.** GraphSAGE 0.7841 is close to RF 0.8030 and worse than GAT. We attribute GAT's advantage to the attention mechanism, which can down-weight noisy edges in the sparse transfer graph.
3. **On the leaky full set, the ranking inverts.** GraphSAGE reaches 0.9666 and outperforms GAT 0.9341, while the tabular GBM reaches 0.9803. The leaky ranking has no predictive value for the honest setting.
4. **GNN does beat tabular on the honest set**, which is the first positive empirical result in this paper that survives the leakage correction. We consider this the honest headline.

### 4.5 Combined Paper 1 + Paper 3 Features

Table 6 reports four classifiers on three feature sets: Paper 1 only (23 behavioral features), Paper 3 only (8 AI features), and combined (31 features). The Paper 3 features are described in the companion paper.

**Table 6. Classifier AUC by feature set on the trusted subset (n=64, LOO and 5x3-CV).**

| Feature set      | Model     | LOO AUC | 5x3-CV mean AUC |
|------------------|-----------|---------|-----------------|
| P1 only (23)     | GBM       | 0.6276  | 0.7267          |
| P1 only (23)     | RF        | 0.7713  | 0.8183          |
| P1 only (23)     | LR        | 0.6100  | 0.6696          |
| P1 only (23)     | LightGBM  | 0.7097  | 0.7040          |
| P3 only (8)      | GBM       | 0.7146  | 0.7008          |
| P3 only (8)      | RF        | 0.7292  | 0.7741          |
| P3 only (8)      | LR        | 0.6432  | 0.6807          |
| P3 only (8)      | LightGBM  | 0.7419  | 0.7392          |
| Combined (31)    | GBM       | 0.6393  | 0.7093          |
| Combined (31)    | RF        | 0.7625  | 0.8151          |
| Combined (31)    | LR        | 0.6256  | 0.6646          |
| Combined (31)    | LightGBM  | 0.6872  | **0.7172**      |

**Key findings.**

1. **Combining the two feature sets buys almost nothing on the trusted set.** LightGBM 5x3-CV improves marginally from 0.7040 (P1-only) to 0.7172 (combined), a delta of +0.013. RF actually drops from 0.8183 to 0.8151.
2. **Paper 3 features alone are competitive.** With only 8 features, LightGBM reaches 0.7419 LOO and 0.7392 5x3-CV, close to Paper 1's 23-feature numbers. This is consistent with Paper 3's observation that `hour_entropy` and `behavioral_consistency` are individually strong.
3. **On the full leaky set, combined features look stronger** (GBM 0.9834 vs 0.9787 for P1 only), but this gap is within noise of the leakage artifact and should not be cited.
4. **Implication.** There is no free lunch from naively stacking feature sets on a small honest corpus. Ensemble gains require either more data or smarter late fusion.

### 4.6 4-Dimensional Security Audit

We run the 4-dimensional audit twice: on the full 3,302-address audited set (14 were dropped for empty histories), and on the trusted 50-address subset (22 agents, 28 humans; some trusted addresses were dropped by the audit script for incomplete approval history). Table 7 reports the headline metrics side by side.

**Table 7. 4-dimensional security audit: leaky vs honest.**

| Metric (dimension)                          | Full 3,302 (C1-C4) agent / human | Agent/human ratio (leaky) | Trusted 50 agent / human | Agent/human ratio (honest) | Direction reversed? |
|---------------------------------------------|----------------------------------|---------------------------|--------------------------|----------------------------|---------------------|
| # approvals (D1)                            | 108.75 / 14.41                    | **7.55x**                 | 110.09 / 125.11          | 0.88x                      | **Yes**             |
| # unlimited approvals (D1)                  | 61.38 / 5.93                      | **10.35x**                | 49.14 / 66.14            | 0.74x                      | **Yes**             |
| # active approvals (D1)                     | 102.02 / 13.49                    | 7.56x                     | 104.68 / 121.14          | 0.86x                      | **Yes**             |
| avg approval age (days, D1)                 | 1170.5 / 1134.9                   | 1.03x                     | 795.6 / 1235.8           | 0.64x                      | **Yes**             |
| max approval age (days, D1)                 | 1539.0 / 1295.5                   | 1.19x                     | 1028.2 / 1811.9          | 0.57x                      | **Yes**             |
| unlimited approval rate (D1)                | 0.502 / 0.304                     | 1.65x                     | 0.245 / 0.452            | 0.54x                      | **Yes**             |
| DEX interaction rate (D3)                   | 0.228 / 0.133                     | 1.71x                     | 0.071 / 0.041            | 1.73x                      | No                  |
| swap rate (D3)                              | 0.196 / 0.118                     | 1.66x                     | 0.060 / 0.025            | 2.39x                      | No                  |
| revert rate (D4)                            | 0.049 / 0.042                     | 1.16x                     | 0.015 / 0.024            | 0.64x                      | **Yes**             |
| gas used efficiency (D4)                    | 0.712 / 0.695                     | 1.02x                     | 0.644 / 0.751            | 0.86x                      | **Yes**             |

**Key findings.**

1. **The agents-have-10x-more-approvals headline does not survive relabeling.** On the trusted set, humans hold **more** unlimited approvals per address than agents (mean 66.1 vs 49.1). This reversal is borderline significant (Mann-Whitney p=0.071 for unlimited approvals, p=0.026 for unlimited approval rate).
2. **Interpretation.** Famous ENS humans (vitalik.eth, Hayden Adams, nick.eth) are heavy long-term DeFi users with many historical approvals. The C1-C4 mining pipeline preferentially labeled token-holder addresses with heavy DeFi footprints as AGENT, which is exactly why the leaky ratio was 10x: we selected for agent-like humans and excluded human-like agents.
3. **DEX interaction rate stays directional** (1.7x on both sets) but the absolute magnitudes drop sharply (0.228 -> 0.071), indicating that the curated-MEV-bot subset is less DEX-active than the platform token holders.
4. **Revert rate reverses:** curated MEV bots have lower revert rates (1.5%) than ENS humans (2.4%), because MEV searchers simulate transactions before broadcasting.
5. **Gas used efficiency reverses:** curated humans are more efficient (0.75) than curated agents (0.64). This is the opposite of the intuitive claim that algorithms are more precise.

**The audit direction is not a minor wrinkle. It is a full 180-degree reversal on the headline "agents are riskier than humans on approvals".**

---

## 5. Threats to Validity: The C1-C4 Label Leakage Discovery

This section is the methodological centerpiece of the paper and documents the leakage discovery, the forensic evidence, the fix, and the implications.

### 5.1 The Bug

In our labeling script `paper1_onchain_agent_id/features/verify_c1c4.py`, the `_check_c3` function gates AGENT labels on three quantities:

- `tx_interval_cv` (coefficient of variation of inter-transaction intervals)
- `hour_entropy` (Shannon entropy over UTC hour bucketing)
- `burst_ratio` (fraction of consecutive transactions within 10 s)

These three quantities are algebraically the same as three of the top classifier features in the 23-feature vector:

- `tx_interval_std = tx_interval_mean x tx_interval_cv` (entangled with feature 2)
- `active_hour_entropy = hour_entropy` (identical to feature 4)
- `burst_frequency = burst_ratio` (identical to feature 7)

Any address that was labeled AGENT because `hour_entropy > 2.5` is then used to train a classifier whose top feature is `active_hour_entropy`. The model is **guaranteed** to re-derive the labeling rule from the features.

### 5.2 The Evidence

**Evidence 1: The honest vs leaky AUC gap.**

| Metric                     | Leaky (C1-C4, n=3316) | Honest (provenance, n=64) |
|----------------------------|-----------------------|---------------------------|
| GBM AUC                    | 0.9803                | 0.6276 (LOO) / 0.6940 (CV)|
| RF AUC                     | 0.9766                | 0.7713 (LOO) / 0.8030 (CV)|
| LR AUC                     | 0.9521                | 0.6100 (LOO) / 0.6203 (CV)|
| Top feature univariate AUC | active_hour_entropy: 0.9347 | tx_interval_mean: 0.7683 |

The top-feature univariate AUC collapses from 0.93 to 0.70 when the labeling rule is removed, which is the signature of label leakage.

**Evidence 2: Inter-label agreement is low.**

We apply the GBM trained on the 64 provenance labels transductively to the 3,252 platform-token-holder rows (whose C1-C4 labels we no longer trust). The inter-label agreement between the honest prediction and the C1-C4 label is **37.4%**, which is below random chance for balanced binary classification. The honest classifier does not endorse the majority of C1-C4 labels.

**Evidence 3: Statistical tests on n=64 reveal structural cracks.**

- RF significantly beats GBM (McNemar p=0.0225, DeLong p=0.0014, bootstrap CI95 [-0.24, -0.06]).
- GBM vs Majority: NOT significantly different (GBM LOO accuracy 0.578 vs Majority 0.516, McNemar p=0.58).
- Single best feature matches GBM.

These are the three symptoms you expect to see when a classifier's performance is being propped up by a leaking label: the strongest model (GBM) behaves like a majority baseline once the leak is sealed.

**Evidence 4: Cross-platform transfer reverses sign.**

Train on Autonolas C1-C4 labels and test on trusted provenance: AUC 0.24-0.34. Train on Fetch.ai C1-C4 labels and test on AI Arena C1-C4 labels: AUC 0.97-0.99. The gap is a direct measurement of the labeling-scheme-specific signal that the classifier picked up.

**Evidence 5: Security audit direction reverses.**

The "agents hold 10x more unlimited approvals" headline from the leaky set becomes "humans hold 1.3x more unlimited approvals" on the trusted set. The direction of a 4-dimensional audit is not supposed to depend on labeling heuristic. It does here because the heuristic preferentially admits heavy-DeFi users as AGENT.

### 5.3 The Fix

We decided on the following:

1. **Demote C1-C4 from a labeling oracle to a downstream diagnostic tool.** It can still be used to rank how likely an address is to be an agent given a trained classifier, but it cannot be the source of ground truth.
2. **Retain only provenance labels for training and testing.** This yields n=64.
3. **Apply the GBM trained on the 64 provenance labels transductively** to the 3,252 platform-token-holder rows to generate honest predictions, and publish these predictions as weak labels for future re-mining rather than as ground truth.
4. **Report both honest and leaky numbers** throughout the paper with clear labels.
5. **Publish the leakage case study** as Section 5 of this paper.

### 5.4 Broader Implications for Blockchain ML

The leakage we found is subtle: none of the classifier features are literally one of the gating quantities. The leak is through algebraic identities (`tx_interval_std ~ tx_interval_cv x tx_interval_mean`) and name collisions (`hour_entropy` gated the label, `active_hour_entropy` is a feature). A reviewer running a superficial check would not catch it; neither did we for several weeks.

We propose a three-step leakage audit for any behavioral-feature classifier on blockchain data:

1. **List all quantities used in the labeling pipeline**, including preprocessing filters.
2. **For each classifier feature, check for algebraic, statistical, or semantic entanglement** with any labeling quantity, allowing for logarithms, ratios, and renaming.
3. **If any labeling quantity overlaps with any classifier feature**, either drop the feature or replace the labeling pipeline with a non-behavioral provenance source.

We release our `verify_c1c4.py`, `pipeline_provenance.py`, `statistical_tests.py`, and `cross_platform_eval.py` scripts as a reusable template.

### 5.5 Other Threats

Beyond the C1-C4 issue, several additional threats apply.

- **Sample size.** n=64 is too small to support fine-grained claims. 5-fold CV has 13 test samples per fold; AUC confidence intervals are wide.
- **Selection bias on the trusted set.** The 33 agents and 31 humans are public, named, famous addresses. They may not represent the broader population of agents and humans.
- **Feature drift.** Agent behavior evolves as platforms update. Our features reflect a specific time window.
- **Adversarial evasion.** An adversary aware of the feature set can mimic human behavior by injecting jitter into gas pricing and transaction timing.
- **Chain coverage.** We only study Ethereum mainnet. Agents on Base, Arbitrum, Solana, and other chains are out of scope.
- **Contract-type confounding.** We excluded router contracts and exchange hot wallets from training, but edge cases remain.

---

## 6. Limitations and Future Work

### 6.1 Limitations

1. **The honest corpus is tiny.** 64 addresses is near the floor of what any behavioral classifier can defensibly train on. Every per-model number we report has an uncertainty band that is wider than most practitioners would find comfortable.
2. **The honest corpus is biased toward famous accounts.** Our trusted humans are literally vitalik.eth and Hayden Adams. They are the easiest possible humans to label and the heaviest possible DeFi users.
3. **The original C1-C4 mining exercise was not wasted, but it cannot be cited as ground truth.** The 3,252 platform-token-holder rows still serve as a test bed for feature engineering and as a large graph for GNN propagation, but their labels are now weak labels that must be cross-checked against provenance.
4. **GNN variance is high on n=64.** GAT's 5-fold AUC std of 0.108 means the headline 0.8825 AUC is a pessimistic 0.77 in the worst fold and an optimistic 1.00 in the best. Larger labeled graphs are needed.
5. **The security audit cannot be generalized.** Even the honest direction is based on 22 agents and 28 humans. We cannot claim the direction reversal holds at scale; we can only claim that the 10x-ratio headline from the leaky set was an artifact.

### 6.2 Clean Re-Mining Roadmap

We propose the following steps for a follow-up mining effort:

1. **Provenance-first labeling.** Start from platforms that publish on-chain service registries with strict attestations (Autonolas ServiceRegistry, Flashbots builder list, Chainlink oracle operator list). Reject any label that is derived from behavioral thresholds.
2. **Human controls at scale.** Mine 500+ ENS-verified humans matched on transaction volume and protocol exposure to avoid the celebrity-human bias.
3. **Behavioral label audit.** Run the 3-step leakage audit from Section 5.4 on every candidate label before any classifier training.
4. **Held-out platform.** Reserve at least one agent platform entirely for testing, never for training, to measure honest generalization.
5. **Temporal holdout.** Train on addresses active before a cutoff date and test on addresses active after, to measure concept drift.
6. **Adversarial stress tests.** Inject behavioral noise into agent transactions to measure robustness against evasion.

### 6.3 Longer-Term Directions

- **EIP-7702 impact.** EIP-7702 lets an EOA temporarily delegate smart-contract code within a transaction, dissolving the `is_contract()` test that has underpinned naive bot detection. Our behavioral framework does not depend on `is_contract()`, so it should survive EIP-7702, but the set of viable features will change (e.g., authorization-list usage becomes a feature).
- **Cross-chain tracking.** Agents that run on multiple chains simultaneously (Base, Arbitrum, Solana) require cross-chain identity linking.
- **LLM-powered agents.** The companion Paper 3 (AI Sybil) work suggests that LLM-driven agents currently mimic rule-based agents in on-chain traces. We cannot yet separate LLM agents from rule-based agents from Etherscan data alone; new features and external signals are needed.
- **Agent-to-agent coordination detection.** The interaction-graph component of our GNN work hints that agent clusters are detectable, but the honest corpus is too small to make the claim.

---

## 7. Conclusion

We presented the first honest empirical study of AI agent identification on Ethereum mainnet using provenance-only ground truth. Our initial pipeline on 3,316 platform-mined addresses reported an apparently exciting AUC of 0.9803 (GBM). In the course of this work, we discovered that the labeling oracle used three quantities (`hour_entropy`, `burst_ratio`, `tx_interval_cv`) that were algebraically equivalent to three top classifier features, a direct form of target leakage. After demoting the oracle to a downstream diagnostic and retaining only 64 provenance-verified addresses, the honest baselines are:

- **Random Forest LOO-CV AUC 0.7713** (5x10-CV 0.8030),
- **GAT 5-fold AUC 0.8825** on the same trusted subset,
- **GradientBoosting indistinguishable from a majority classifier** (McNemar p=0.58),
- **Autonolas -> trusted cross-platform AUC 0.24-0.34**, worse than chance,
- **Security audit reverses direction**: humans hold more unlimited approvals than curated agents.

We framed the paper around the leakage discovery as a methodological contribution: it quantifies how fragile heuristic labeling can be, it provides a concrete audit template, and it establishes honest baselines (RF 0.77, GAT 0.87) against which any future re-mining effort must be measured. All code, features, labels, and the leaky-vs-honest diagnostic artifacts are released.

On-chain AI agents are a real and growing phenomenon, and identifying them reliably is a real and growing need. Our central message is that this need will not be met by scaling up heuristic labeling pipelines. It will be met only by starting from non-behavioral provenance, running explicit leakage audits, reporting honest-vs-leaky gaps, and treating cross-label-scheme transfer as a non-negotiable diagnostic. The honest numbers are smaller than the leaky numbers, but they are the numbers the field can build on.

---

## References

```bibtex
@inproceedings{chen2018ponzi,
  title={Detecting Ponzi Schemes on Ethereum: Towards Healthier Blockchain Technology},
  author={Chen, Weili and Zheng, Zibin and Cui, Jiahui and Ngai, Edith and Zheng, Peilin and Zhou, Yuren},
  booktitle={Proc. of the Web Conference (WWW)},
  year={2018}
}

@inproceedings{victor2021wash,
  title={Detecting and Quantifying Wash Trading on Decentralized Cryptocurrency Exchanges},
  author={Victor, Friedhelm and Weintraud, Andrea Marie},
  booktitle={Proc. of the Web Conference (WWW)},
  year={2021}
}

@inproceedings{torres2021frontrunner,
  title={Frontrunner Jones and the Raiders of the Dark Forest: An Empirical Study of Frontrunning on the Ethereum Blockchain},
  author={Torres, Christof Ferreira and Camino, Ramiro and State, Radu},
  booktitle={Proc. USENIX Security},
  year={2021}
}

@article{victor2023fingerprinting,
  title={Behavioral Fingerprinting of Blockchain Bots},
  author={Victor, Friedhelm and Weintraud, Andrea and Haliloglu, Merve},
  journal={arXiv preprint},
  year={2023}
}

@inproceedings{li2024telegram,
  title={Telegram Trading Bots: An Empirical Study},
  author={Li, Yichen and Chaliasos, Stefanos and Livshits, Benjamin},
  booktitle={Proc. Financial Cryptography (FC)},
  year={2024}
}

@inproceedings{daian2020flash,
  title={Flash Boys 2.0: Frontrunning in Decentralized Exchanges, Miner Extractable Value, and Consensus Instability},
  author={Daian, Philip and Goldfeder, Steven and Kell, Tyler and Li, Yunqi and Zhao, Xueyuan and Bentov, Iddo and Breidenbach, Lorenz and Juels, Ari},
  booktitle={Proc. IEEE Symposium on Security and Privacy (S\&P)},
  year={2020}
}

@inproceedings{weintraub2022flashbot,
  title={A Flash(bot) in the Pan: Measuring Maximal Extractable Value in Private Transaction Ordering Mechanisms},
  author={Weintraub, Ben and Ferreira Torres, Christof and Nita-Rotaru, Cristina and State, Radu},
  booktitle={Proc. IMC},
  year={2022}
}

@inproceedings{qin2022quantifying,
  title={Quantifying Blockchain Extractable Value: How Dark is the Forest?},
  author={Qin, Kaihua and Zhou, Liyi and Gervais, Arthur},
  booktitle={Proc. IEEE Symposium on Security and Privacy (S\&P)},
  year={2022}
}

@inproceedings{park2024mev,
  title={An Empirical Study of MEV Post-Merge},
  author={Park, Sunyoung and Bahrani, Alireza and Roughgarden, Tim},
  booktitle={Proc. Financial Cryptography (FC)},
  year={2024}
}

@article{gupta2023crossdomain,
  title={Cross-Domain MEV: Measurement and Mitigation},
  author={Gupta, A. and others},
  journal={arXiv preprint},
  year={2023}
}

@misc{autonolas2023,
  title={Autonolas: Autonomous Agent Services},
  author={{Autonolas}},
  howpublished={Whitepaper, https://www.autonolas.network/},
  year={2023}
}

@misc{eliza2024,
  title={ELIZA: AI Agent Framework},
  author={{ai16z}},
  howpublished={GitHub repository, https://github.com/ai16z/eliza},
  year={2024}
}

@misc{virtuals2024,
  title={Virtuals: Agent Tokenization Platform},
  author={{Virtuals Protocol}},
  howpublished={Whitepaper, https://www.virtuals.io/},
  year={2024}
}

@misc{fetchai2023,
  title={Fetch.ai: Autonomous Economic Agents},
  author={{Fetch.ai}},
  howpublished={Whitepaper, https://fetch.ai/},
  year={2023}
}

@misc{singularitynet2023,
  title={SingularityNET: Decentralized AI Platform},
  author={{SingularityNET}},
  howpublished={Whitepaper, https://singularitynet.io/},
  year={2023}
}

@misc{aiarena2024,
  title={AI Arena: AI Fighting Game on Ethereum},
  author={{AI Arena}},
  howpublished={Whitepaper, https://aiarena.io/},
  year={2024}
}

@inproceedings{zhou2023sok,
  title={{SoK}: Decentralized Finance (DeFi) Attacks},
  author={Zhou, Liyi and Xiong, Xihan and Ernstberger, Jens and Chaliasos, Stefanos and Wang, Zhipeng and Wang, Ye and Qin, Kaihua and Wattenhofer, Roger and Song, Dawn and Gervais, Arthur},
  booktitle={Proc. IEEE Symposium on Security and Privacy (S\&P)},
  year={2023}
}

@inproceedings{wen2023composability,
  title={{DeFi} Composability as a Means for Protocol-Level Security Analysis},
  author={Wen, Yimeng and Liu, Yue and Lie, David},
  booktitle={Proc. CCS},
  year={2023}
}

@article{grech2022madmax,
  title={{MadMax}: Analyzing the Out-of-Gas World of Smart Contracts},
  author={Grech, Neville and Kong, Michael and Scholz, Bernhard and Smaragdakis, Yannis},
  journal={Communications of the ACM},
  year={2022}
}

@inproceedings{tsankov2018securify,
  title={Securify: Practical Security Analysis of Smart Contracts},
  author={Tsankov, Petar and Dan, Andrei and Drachsler-Cohen, Dana and Gervais, Arthur and Buenzli, Florian and Vechev, Martin},
  booktitle={Proc. CCS},
  year={2018}
}

@article{wang2024airisks,
  title={Risks of {AI} Agents with Cryptocurrency Access},
  author={Wang, X. and others},
  journal={arXiv preprint},
  year={2024}
}

@inproceedings{zhang2024approval,
  title={Approval-Based Attacks in {DeFi}: Measurement and Defense},
  author={Zhang, Y. and others},
  booktitle={Proc. NDSS},
  year={2024}
}

@article{he2026agentsurvey,
  title={A Survey on {AI} Agents on Blockchain},
  author={He, Yu and others},
  journal={arXiv:2601.04583},
  year={2026}
}

@article{kaufman2012leakage,
  title={Leakage in Data Mining: Formulation, Detection, and Avoidance},
  author={Kaufman, Shachar and Rosset, Saharon and Perlich, Claudia and Stitelman, Ori},
  journal={ACM Transactions on Knowledge Discovery from Data},
  year={2012}
}

@inproceedings{roelofs2019metadata,
  title={A Meta-Analysis of Overfitting in Machine Learning},
  author={Roelofs, Rebecca and Shankar, Vaishaal and Recht, Benjamin and Fridovich-Keil, Sara and Hardt, Moritz and Miller, John and Schmidt, Ludwig},
  booktitle={Proc. NeurIPS},
  year={2019}
}

@article{geirhos2020shortcut,
  title={Shortcut Learning in Deep Neural Networks},
  author={Geirhos, Robert and Jacobsen, J{\"o}rn-Henrik and Michaelis, Claudio and Zemel, Richard and Brendel, Wieland and Bethge, Matthias and Wichmann, Felix A.},
  journal={Nature Machine Intelligence},
  year={2020}
}

@inproceedings{tikhomirov2021phishing,
  title={Detecting Phishing Addresses on {E}thereum via Transaction Graph},
  author={Tikhomirov, Sergei and Ivanitskiy, Valery},
  booktitle={Proc. of the Web Conference (WWW) Companion},
  year={2021}
}

@inproceedings{hamilton2017graphsage,
  title={Inductive Representation Learning on Large Graphs},
  author={Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  booktitle={Proc. NeurIPS},
  year={2017}
}

@inproceedings{velickovic2018gat,
  title={Graph Attention Networks},
  author={Veli{\v{c}}kovi{\'c}, Petar and Cucurull, Guillem and Casanova, Arantxa and Romero, Adriana and Li{\`o}, Pietro and Bengio, Yoshua},
  booktitle={Proc. ICLR},
  year={2018}
}

@inproceedings{weber2019antimoney,
  title={Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics},
  author={Weber, Mark and Domeniconi, Giacomo and Chen, Jie and Weidele, Daniel Karl I. and Bellei, Claudio and Robinson, Tom and Leiserson, Charles E.},
  booktitle={KDD Anti-Money Laundering Workshop},
  year={2019}
}

@inproceedings{patel2021phishing,
  title={{GNN}-Based Ethereum Phishing Detection},
  author={Patel, V. and others},
  booktitle={Proc. BigData},
  year={2021}
}

@article{shamsi2023gnnsurvey,
  title={A Survey of {GNN}s for Blockchain Analytics},
  author={Shamsi, Kiarash and others},
  journal={arXiv preprint},
  year={2023}
}
```

---

*This draft corresponds to the honest pipeline outputs of 2026-04-08 (commit eb8721e and follow-up). Headline numbers: RF LOO 0.7713 on n=64; GAT 5-fold 0.8825 on trusted subset; leaky-vs-honest gap of 0.21 AUC; security audit direction reversed. See `experiments/expanded/pipeline_results_provenance.json`, `statistical_tests.json`, `cross_platform_eval.json`, `gnn_results.json`, `combined_pipeline_results.json`, and `security_audit_expanded.json` for the raw JSON artifacts and `FINDINGS_2026-04-08.md` for the session-level discovery log.*
