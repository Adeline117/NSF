# AI Agent-Driven On-Chain Sybil Attacks: Evasion, Detection, and Defense

## Abstract
First empirical study quantifying AI agent capabilities to evade
existing Sybil detectors, with novel AI-specific detection features
that restore detection accuracy. We generate AI-driven Sybil behavior
at three sophistication levels calibrated to HasciDB's 5-indicator
framework (3.6M addresses, 16 projects), show that advanced AI agents
evade 100% of rule-based detection, and demonstrate that 8 AI-specific
behavioral features (gas precision, hour entropy, behavioral consistency,
action sequence perplexity, error recovery, latency variance, nonce
regularity, EIP-1559 tip precision) restore detection AUC to 1.000.
Cross-project transfer via Leave-One-Project-Out validation confirms
generalizability across all 16 HasciDB airdrop projects.

## 1. Introduction
- Traditional Sybil detection achieves AUC 0.946 (HasciDB CHI'26)
- But AI agents can generate human-like behavior
- RQ1: How effectively can AI agents evade existing Sybil detectors?
- RQ2: What AI-specific features can restore detection capability?
- RQ3: Do detection improvements transfer across airdrop projects?

## 2. Related Work
- 2.1 Sybil Detection in Cryptocurrency Airdrops
  - HasciDB (Li et al., CHI'26): 5-indicator framework, 16 projects
  - TrustaLabs: Graph community detection + K-means
  - Arbitrum Foundation: Louvain on transfer graphs
  - BrightID: SybilRank algorithm comparison
- 2.2 Pre-Airdrop Detection
  - Wen et al.: LightGBM temporal features, AUC 0.793 @ T-30
  - ARTEMIS: GNN-based detection (GraphSAGE)
- 2.3 AI-Powered Adversarial Attacks
  - LLM-generated evasion strategies
  - Adversarial ML in security contexts
- 2.4 AI Agent Behavior (connects to Paper 0 & 1)

## 3. Methodology
- 3.1 Data Sources (HasciDB, Blur, pre-airdrop-detection)
  - HasciDB: 3.6M addresses, 1.09M sybils, 16 Ethereum L1 airdrops
  - HasciDB 5 indicators: BT>=5, BW>=10, HF>=0.80, RF>=0.50, MA>=5
  - Classification: ops_flag = BT|BW|HF; fund_flag = RF|MA; is_sybil = ops|fund
  - Blur Season 2: 53K recipients, 9.8K sybils (UW-DCL/Blur repo)
  - pre-airdrop-detection: LightGBM behavioral features on Blur S2
- 3.2 Baseline Detectors (HasciDB rules, HasciDB ML, pre-airdrop LightGBM)
  - HasciDB rule-based: threshold-based classification on 5 indicators
  - HasciDB ML: GradientBoosting on 5 indicator features
  - pre-airdrop LightGBM: temporal behavioral features, AUC 0.793 @ T-30
  - TrustaLabs: Louvain/K-Core community detection on asset transfer graphs
  - ArbitrumFoundation: Louvain on msg.value + funder/sweep graphs
- 3.3 AI Sybil Generation (3 sophistication levels)
  - Basic: Simple parameter randomization
  - Moderate: Strategy-aware evasion with human-like noise
  - Advanced: Full LLM-orchestrated behavior with personality modeling
  - Per-indicator evasion strategies (BT, BW, HF, RF, MA)
- 3.4 AI-Specific Feature Engineering (8 features from Paper 1)
  - gas_price_precision: LLM computes exact gas, humans round
  - hour_entropy: No circadian rhythm for 24/7 agents
  - behavioral_consistency: Cross-address correlation from same LLM
  - action_sequence_perplexity: LLM-generated sequences have characteristic range
  - error_recovery_pattern: Systematic retry/fallback
  - response_latency_variance: LLM inference time signature
  - gas_nonce_gap_regularity: Regular nonce increments
  - eip1559_tip_precision: Precise priority fee calculation
- 3.5 Enhanced Detector Training
  - GradientBoosting (primary): 13 features, 200 trees, depth 5
  - LightGBM (comparison): aligned with pre-airdrop-detection baseline
  - RandomForest (baseline): 200 trees, depth 10
- 3.6 Evaluation Protocol (LOPO, temporal split)
  - Leave-One-Project-Out: train on 15 projects, test on 1
  - Temporal split: train on pre-2024, test on 2024+ projects
  - Protocol transfer: train DeFi, test NFT (and vice versa)

## 4. Results
- 4.1 Evasion Effectiveness (per indicator, per level)
  - Basic: partial evasion, some indicators still triggered
  - Moderate: high evasion rate (>80%)
  - Advanced: 100% evasion of rule-based detection
  - Per-indicator analysis: BT/BW/MA count-based easiest to evade
- 4.2 Baseline Detector Degradation
  - HasciDB rules: AUC drops from 0.946 to ~0.500 vs advanced AI sybils
  - HasciDB ML (5 features): AUC drops significantly
  - pre-airdrop LightGBM: limited effectiveness vs AI sybils
- 4.3 AI-Specific Feature Discriminative Power
  - Individual feature AUC analysis
  - AI features outperform HasciDB features vs AI sybils
  - Top discriminative features: behavioral_consistency, hour_entropy
- 4.4 Enhanced Detector Performance
  - 13-feature GBM recovers AUC to 1.000
  - Significant improvement over all baselines
  - Feature importance: AI features dominate for AI sybil detection
- 4.5 Cross-Project Transfer
  - LOPO: stable AUC across 16 projects
  - Temporal: strong transfer from pre-2024 to 2024 projects
  - Protocol: DeFi->NFT and NFT->DeFi transfer both effective
- 4.6 Feature Importance & Ablation
  - Top features by GBM importance
  - Ablation: removing AI features significantly degrades performance
  - Feature importance stability across LOPO folds (Kendall's tau)

## 5. Discussion
- 5.1 Arms Race Implications
  - AI agents fundamentally change the adversarial landscape
  - Rule-based detection is insufficient against LLM-orchestrated sybils
  - AI-specific features provide a durable detection signal
- 5.2 Recommendations for Airdrop Designers
  - Incorporate AI-specific features into eligibility pipelines
  - Use temporal behavioral analysis (pre-airdrop detection)
  - Consider multi-model ensemble approaches
  - Implement adaptive threshold mechanisms
- 5.3 Integration with Paper 1 Agent Identification
  - Paper 1 features transfer directly to sybil detection
  - Agent taxonomy informs sophistication level classification
  - Unified framework: identify agent first, then classify intent
- 5.4 Limitations
  - Simulated AI sybils vs real-world LLM agents
  - HasciDB covers Ethereum L1 only
  - AI feature extraction requires transaction-level data
  - Evolving LLM capabilities may reduce feature discriminability

## 6. Conclusion
- Advanced AI agents evade 100% of existing rule-based sybil detection
- 8 AI-specific features restore detection AUC to 1.000
- Detection transfers across 16 airdrop projects via LOPO
- Urgent need for AI-aware sybil detection in airdrop design
- Open-source tools and datasets for community adoption
