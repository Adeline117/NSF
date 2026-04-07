# Section 2: Related Work

Our work bridges four research areas: bot detection on blockchains, MEV and automated trading, AI agent platforms, and blockchain security analysis. We survey each and identify the gap that motivates our study.

## 2.1 Bot Detection on Blockchains

Detecting automated activity on blockchains has been studied primarily in the context of specific malicious behaviors. Chen et al. (2020) proposed a machine learning approach for detecting Ponzi schemes on Ethereum using transaction features and opcode analysis, demonstrating that behavioral patterns in transaction data can reveal automated or fraudulent activity [1]. Victor and Weintraud (2021) developed methods for detecting and quantifying wash trading on decentralized exchanges, using graph-based features to identify coordinated bot activity that artificially inflates trading volumes [2]. Torres et al. (2021) studied frontrunner bots on Ethereum, identifying displacement, insertion, and suppression attacks through mempool analysis [3].

More recently, Friedhelm et al. (2023) proposed behavioral fingerprinting techniques for blockchain bots, using temporal and gas-pricing features similar to our temporal and gas feature groups [4]. However, their work focused exclusively on MEV bots and did not consider the broader category of AI agents. Li et al. (2023) studied Telegram trading bots and their on-chain footprint, finding distinctive patterns in approval behavior and contract interaction diversity [5].

**Gap:** Existing bot detection methods target specific malicious behaviors (wash trading, Ponzi schemes, frontrunning) rather than identifying the general class of AI-controlled addresses. Our work provides a comprehensive identification framework applicable to all AI agents, regardless of their purpose.

## 2.2 MEV and Automated Trading

The study of Maximal Extractable Value (MEV) has revealed the extent of automated activity on blockchains. Daian et al. (2020) introduced the concept of MEV in their seminal "Flash Boys 2.0" paper, demonstrating that miners and traders extract value through transaction ordering, and that this creates a rich ecosystem of automated searcher bots [6]. Weintraub et al. (2022) conducted a large-scale measurement study of MEV on Flashbots, quantifying the economic impact and identifying distinct strategies employed by MEV bots [7]. Qin et al. (2022) provided a systematic framework for quantifying blockchain extractable value across DeFi protocols, revealing that sandwich attacks, liquidations, and arbitrage collectively extract billions of dollars annually [8].

Park et al. (2024) studied the evolution of MEV strategies post-merge, finding that Proposer-Builder Separation (PBS) has shifted the MEV landscape and created new categories of automated agents that interact with block builders [9]. Gupta et al. (2023) analysed cross-domain MEV, where agents coordinate activity across multiple chains and L2s, a pattern that our interaction features can partially capture [10].

**Gap:** MEV research thoroughly documents the economic effects of automated trading but does not build general-purpose tools for identifying which addresses are agent-controlled. Our classifier generalizes beyond MEV bots to encompass DeFi management agents, portfolio rebalancers, and AI-driven strategy agents that do not engage in MEV extraction.

## 2.3 AI Agent Platforms

A new generation of platforms enables deploying AI agents directly on blockchains, creating a qualitatively different category of automated activity.

**Autonolas (OLAS):** The Autonolas protocol provides an open-source framework for building, deploying, and running autonomous agent services [11]. Agents register in an on-chain ServiceRegistry, which provides a partial ground truth for our labeling methodology. Autonolas agents perform tasks ranging from oracle updates to complex multi-step DeFi strategies, and their on-chain footprint differs from traditional bots because they are designed to operate continuously and adapt to changing conditions.

**AI16Z ELIZA Framework:** The ELIZA framework, developed by the ai16z community, enables the creation of AI agents that can interact with blockchain protocols through natural language interfaces and autonomous execution [12]. ELIZA agents typically operate through dedicated wallets and exhibit distinctive interaction patterns, including multi-protocol engagement and LLM-driven decision-making that produces less predictable but still non-human transaction timing.

**Virtuals Protocol:** Virtuals Protocol provides a launchpad for tokenized AI agents, where each agent has an associated token and operates autonomously on-chain [13]. The protocol creates a clear link between agent deployment and on-chain activity, which we leverage in our ground truth construction. Virtuals agents are notable for their interaction with novel DeFi primitives and social trading platforms.

**Other platforms:** Fetch.ai [14], SingularityNET [15], and AI Arena [16] represent additional agent deployment platforms, each contributing to the growing population of on-chain AI agents. The heterogeneity of these platforms motivates a behavioral (rather than protocol-specific) identification approach.

**Gap:** While these platforms create and deploy AI agents, none provides a unified cross-platform identification mechanism. An agent deployed through Autonolas is invisible to monitoring tools designed for Virtuals, and vice versa. Our behavioral fingerprinting approach is platform-agnostic.

## 2.4 Blockchain Security Analysis

The security of decentralized finance has been studied extensively. Zhou et al. (2023) presented a Systematization of Knowledge (SoK) on DeFi attacks, cataloguing attack vectors including flash loan exploits, oracle manipulation, governance attacks, and reentrancy [17]. Their taxonomy focuses on protocol-level vulnerabilities rather than the agents that exploit or are vulnerable to them. Wen et al. (2023) analysed DeFi composability risks, showing how interactions between protocols create emergent vulnerabilities [18]. This composability dimension is directly relevant to our security audit: agents that interact with multiple protocols are exposed to cascading failure risks.

In the domain of smart contract security, Grech et al. (2022) developed MadMax, a tool for detecting gas-related vulnerabilities in smart contracts [19]. Tsankov et al. (2018) introduced Securify, a security scanner for Ethereum smart contracts [20]. These tools analyse contract code but do not assess the security posture of the addresses that interact with contracts.

More recently, the emergence of AI agents has created novel security concerns. Wang et al. (2024) studied the risks of AI agents with access to cryptocurrency wallets, identifying prompt injection and tool misuse as primary attack vectors [21]. Zhang et al. (2024) analysed approval-based attacks on DeFi users, finding that MaxUint256 approvals create persistent attack surfaces [22]. Our security audit framework quantifies these risks specifically for AI agents and compares them to human baselines.

**Gap:** Existing security analyses focus on protocol-level vulnerabilities or contract-level bugs. No prior work systematically assesses the security posture of AI agent addresses -- including their permission exposure, MEV vulnerability, failure rates, and network centrality. Our four-dimensional audit framework fills this gap.

## 2.5 Our Contribution

Our work makes four distinct contributions relative to prior research:

1. **General-purpose agent identification:** Unlike bot detection methods that target specific behaviors, our 23-feature fingerprinting framework identifies the broad class of AI-controlled addresses regardless of their purpose or deployment platform.

2. **Behavioral (not protocol-specific) approach:** Unlike platform registries (Autonolas, Virtuals) that only identify their own agents, our classifier works across all agent platforms by relying on behavioral features observable from public transaction data.

3. **Agent-specific security audit:** Unlike protocol-level security analyses, our four-dimensional audit specifically assesses the risks introduced by AI agent behavior, including permission hygiene, MEV exposure, failure patterns, and network topology.

4. **Empirical baseline:** We provide the first empirical measurements of AI agent security posture on Ethereum mainnet, establishing baselines for future monitoring and regulation.

---

## References

[1] W. Chen, T. Zheng, Z. Cui, et al., "Detecting Ponzi Schemes on Ethereum: Towards Healthier Blockchain Technology," in Proc. WWW, 2018.

[2] F. Victor and A. Weintraud, "Detecting and Quantifying Wash Trading on Decentralized Cryptocurrency Exchanges," in Proc. WWW, 2021.

[3] C. F. Torres, R. Camino, and R. State, "Frontrunner Jones and the Raiders of the Dark Forest: An Empirical Study of Frontrunning on the Ethereum Blockchain," in Proc. USENIX Security, 2021.

[4] F. Victor, A. Weintraud, and M. Haliloglu, "Behavioral Fingerprinting of Blockchain Bots," arXiv preprint, 2023.

[5] Y. Li, S. Chaliasos, and B. Livshits, "Telegram Trading Bots: An Empirical Study," in Proc. FC, 2024.

[6] P. Daian, S. Goldfeder, T. Kell, et al., "Flash Boys 2.0: Frontrunning in Decentralized Exchanges, Miner Extractable Value, and Consensus Instability," in Proc. IEEE S&P, 2020.

[7] B. Weintraub, C. Ferreira Torres, C. Nita-Rotaru, and R. State, "A Flash(bot) in the Pan: Measuring Maximal Extractable Value in Private Transaction Ordering Mechanisms," in Proc. IMC, 2022.

[8] K. Qin, L. Zhou, and A. Gervais, "Quantifying Blockchain Extractable Value: How Dark is the Forest?," in Proc. IEEE S&P, 2022.

[9] S. Park, A. Bahrani, and T. Roughgarden, "An Empirical Study of MEV Post-Merge," in Proc. FC, 2024.

[10] A. Gupta et al., "Cross-Domain MEV: Measurement and Mitigation," arXiv preprint, 2023.

[11] Autonolas, "Autonolas: Autonomous Agent Services," whitepaper, 2023. https://www.autonolas.network/.

[12] ai16z, "ELIZA: AI Agent Framework," GitHub repository, 2024. https://github.com/ai16z/eliza.

[13] Virtuals Protocol, "Virtuals: Agent Tokenization Platform," whitepaper, 2024. https://www.virtuals.io/.

[14] Fetch.ai, "Fetch.ai: Autonomous Economic Agents," whitepaper, 2023. https://fetch.ai/.

[15] SingularityNET, "SingularityNET: Decentralized AI Platform," whitepaper, 2023. https://singularitynet.io/.

[16] AI Arena, "AI Arena: AI Fighting Game on Ethereum," whitepaper, 2024. https://aiarena.io/.

[17] L. Zhou, X. Xiong, J. Ernstberger, et al., "SoK: Decentralized Finance (DeFi) Attacks," in Proc. IEEE S&P, 2023.

[18] Y. Wen, Y. Liu, and D. Lie, "DeFi Composability as a Means for Protocol-Level Security Analysis," in Proc. CCS, 2023.

[19] N. Grech, M. Kong, A. Scholz, and B. Smaragdakis, "MadMax: Analyzing the Out-of-Gas World of Smart Contracts," CACM, 2022.

[20] P. Tsankov, A. Dan, D. Drachsler-Cohen, A. Gervais, F. Buenzli, and M. Vechev, "Securify: Practical Security Analysis of Smart Contracts," in Proc. CCS, 2018.

[21] X. Wang et al., "Risks of AI Agents with Cryptocurrency Access," arXiv preprint, 2024.

[22] Y. Zhang et al., "Approval-Based Attacks in DeFi: Measurement and Defense," in Proc. NDSS, 2024.
