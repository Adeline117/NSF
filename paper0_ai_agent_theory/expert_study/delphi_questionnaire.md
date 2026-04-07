# Delphi Expert Survey: AI Agent Definition & Taxonomy for Web3

# 德尔菲专家问卷：Web3 AI Agent 定义与分类体系验证

---

## Study Information / 研究说明

**Title**: Formal Definition and Taxonomy of On-Chain AI Agents
**研究标题**: 链上 AI Agent 的形式化定义与分类体系

**Purpose**: This Delphi survey seeks expert consensus on the first formal definition and taxonomy of AI agents operating on public blockchains.
**目的**: 本德尔菲问卷旨在通过专家共识验证首个面向公链 AI Agent 的形式化定义与分类体系。

**Time required**: ~30 minutes per round (2 rounds total)
**预计耗时**: 每轮约 30 分钟（共 2 轮）

**Anonymity**: Individual responses will not be shared with other panelists. Only aggregated statistics (medians, IQR) will be presented in Round 2.
**匿名性**: 个人回答不会向其他专家公开。第2轮仅展示汇总统计量（中位数、四分位距）。

---

## Round 1: Open Exploration / 第1轮：开放性探索

---

### Part A: Definition Validation / 定义验证

We propose the following formal definition of an on-chain AI Agent:

我们提出以下链上 AI Agent 的形式化定义：

> **Definition 1 (定义1).** An on-chain AI Agent is a software entity that simultaneously satisfies four necessary conditions:
>
> 链上 AI Agent 是同时满足以下四个必要条件的软件实体：
>
> - **C1 (On-chain Actuation / 链上执行能力)**: Controls an Externally Owned Account (EOA), and can autonomously sign and submit transactions to a public blockchain.
>   控制一个外部拥有账户（EOA），能够自主签署并提交交易到公链。
>
> - **C2 (Environmental Perception / 环境感知)**: Perceives on-chain state (e.g., token balances, mempool, oracle feeds) and/or off-chain signals (e.g., price APIs, social media), and incorporates perceived data into its decision process.
>   感知链上状态（如代币余额、内存池、预言机数据）和/或链下信号（如价格API、社交媒体），并将感知数据纳入决策过程。
>
> - **C3 (Autonomous Decision-Making / 自主决策)**: Acts without per-transaction human approval; its decision pipeline includes at least one non-deterministic or adaptive component (e.g., ML model, LLM, RL policy, statistical estimator).
>   无需逐笔交易的人工批准即可行动；其决策流程包含至少一个非确定性或自适应组件（如ML模型、LLM、RL策略、统计估计器）。
>
> - **C4 (Adaptiveness / 适应性)**: Modifies its own behavior (strategy parameters, action selection, or internal state) based on environmental feedback over time.
>   基于环境反馈随时间修改自身行为（策略参数、动作选择或内部状态）。

---

**For each condition C1-C4, please answer: / 对于每个条件 C1-C4，请回答：**

#### Q1: Necessity / 必要性

"Is this condition necessary for defining an AI agent in Web3?"

"该条件对于定义 Web3 AI Agent 是否是必要的？"

| Scale | Label (EN) | 标签（中文） |
|-------|------------|-------------|
| 1 | Strongly Disagree | 非常不同意 |
| 2 | Disagree | 不同意 |
| 3 | Neutral | 中立 |
| 4 | Agree | 同意 |
| 5 | Strongly Agree | 非常同意 |

| Condition | Your Rating (1-5) |
|-----------|-------------------|
| C1: On-chain Actuation / 链上执行 | _____ |
| C2: Environmental Perception / 环境感知 | _____ |
| C3: Autonomous Decision-Making / 自主决策 | _____ |
| C4: Adaptiveness / 适应性 | _____ |

#### Q2: Clarity / 清晰度

"Is this condition clear and unambiguous?"

"该条件的定义是否清晰且无歧义？"

| Scale | Label (EN) | 标签（中文） |
|-------|------------|-------------|
| 1 | Very Unclear | 非常不清晰 |
| 2 | Somewhat Unclear | 较不清晰 |
| 3 | Neutral | 中立 |
| 4 | Somewhat Clear | 较清晰 |
| 5 | Very Clear | 非常清晰 |

| Condition | Your Rating (1-5) |
|-----------|-------------------|
| C1: On-chain Actuation / 链上执行 | _____ |
| C2: Environmental Perception / 环境感知 | _____ |
| C3: Autonomous Decision-Making / 自主决策 | _____ |
| C4: Adaptiveness / 适应性 | _____ |

#### Q3: Suggested Improvements / 改进建议 (Open-ended / 开放式)

For each condition, please suggest any improvements to the wording, scope, or operationalization:

对于每个条件，请提出关于措辞、范围或可操作性的改进建议：

- C1: __________________________________________________________
- C2: __________________________________________________________
- C3: __________________________________________________________
- C4: __________________________________________________________

#### Q4: Joint Sufficiency / 联合充分性 (Open-ended / 开放式)

"Are these four conditions jointly sufficient to define an AI agent in Web3? Are any conditions missing?"

"这四个条件联合起来是否足以定义 Web3 AI Agent？是否有缺失的条件？"

Response / 回答: __________________________________________________________

#### Q5: Redundancy Check / 冗余性检查 (Open-ended / 开放式)

"Should any condition be removed or merged with another? Please explain."

"是否有任何条件应当被删除或与其他条件合并？请说明理由。"

Response / 回答: __________________________________________________________

---

### Part B: Taxonomy Validation / 分类体系验证

We propose an 8-category taxonomy along 3 dimensions: **Autonomy** x **Environment** x **Decision Model**.

我们提出一个沿3个维度（**自主性** x **环境类型** x **决策模型**）构建的8类分类体系。

#### Dimension Definitions / 维度定义

**Dimension 1: Autonomy Level (自主性等级)**

| Level | EN | 中文 | Description |
|-------|----|------|-------------|
| 0 | NONE | 无 | Pure deterministic script; no adaptation |
| 1 | REACTIVE | 反应式 | Responds to events with fixed rules |
| 2 | ADAPTIVE | 自适应 | Adjusts parameters based on environmental feedback |
| 3 | PROACTIVE | 主动式 | Plans and initiates actions independently |
| 4 | COLLABORATIVE | 协作式 | Coordinates with other agents or human governance |

**Dimension 2: Environment Type (环境类型)**

| Type | EN | 中文 | Description |
|------|----|------|-------------|
| onchain | On-chain Only | 纯链上 | Fully on-chain smart contract execution |
| hybrid | Hybrid (Off-chain -> On-chain) | 混合型 | Off-chain compute, on-chain execution |
| cross_chain | Cross-Chain | 跨链 | Operates across multiple blockchain networks |
| multi_modal | Multi-Modal | 多模态 | Combines on-chain + off-chain services (APIs, LLMs, social media) |

**Dimension 3: Decision Model (决策模型)**

| Model | EN | 中文 | Description |
|-------|----|------|-------------|
| deterministic | Deterministic | 确定性 | If-then rules, fully predictable output |
| statistical | Statistical | 统计型 | ML models, probability-based decisions |
| llm | LLM-Driven | LLM驱动 | Large language model reasoning |
| rl | Reinforcement Learning | 强化学习 | Policy optimization via reward signals |
| hybrid | Hybrid | 混合型 | Combination of multiple decision models |

#### The 8 Categories / 八个分类

| # | Category | 中文名 | Tuple (Autonomy, Env, Decision) | Description |
|---|----------|--------|--------------------------------|-------------|
| 1 | Deterministic Script | 确定性脚本 | (NONE, Hybrid, Deterministic) | Fixed-sequence transaction executor; no adaptation (e.g., cron payroll, airdrop scripts) |
| 2 | Simple Trading Bot | 简单交易机器人 | (REACTIVE, Hybrid, Deterministic) | Pre-defined trading rules; responds to market with fixed execution paths (e.g., grid bots, DCA bots) |
| 3 | MEV Searcher | MEV 搜索者 | (ADAPTIVE, Hybrid, Statistical) | Identifies and extracts Maximal Extractable Value via statistical strategies (e.g., sandwich attacks, arbitrage) |
| 4 | Cross-Chain Bridge Agent | 跨链桥代理 | (ADAPTIVE, Cross-chain, Deterministic) | Relays messages and assets across blockchain networks (e.g., LayerZero relayers, Wormhole guardians) |
| 5 | RL Trading Agent | 强化学习交易代理 | (ADAPTIVE, Hybrid, RL) | Uses reinforcement learning to optimize trading strategy with observable learning curves |
| 6 | DeFi Management Agent | DeFi 管理代理 | (PROACTIVE, Hybrid, Hybrid) | Manages DeFi positions across protocols with risk-aware strategies (e.g., Autonolas agents, Yearn vaults) |
| 7 | LLM-Powered Agent | LLM 驱动代理 | (PROACTIVE, Multi-modal, LLM) | Uses LLM reasoning for on-chain actions with multi-modal context (e.g., AI16Z/ELIZA, Virtuals Protocol) |
| 8 | Autonomous DAO Agent | 自治 DAO 代理 | (COLLABORATIVE, On-chain, Hybrid) | Governed by DAO rules; collective decision-making (e.g., Gnosis Safe modules, Governor executors) |

---

**For each of the 8 categories, please answer: / 对于每个分类，请回答：**

#### Q6: How well-defined is this category? / 该分类的定义有多清晰？

| Scale | 1 = Very Poorly Defined (非常不清晰) | 5 = Very Well-Defined (非常清晰) |
|-------|--------------------------------------|----------------------------------|

| Category | Rating (1-5) |
|----------|-------------|
| 1. Deterministic Script | _____ |
| 2. Simple Trading Bot | _____ |
| 3. MEV Searcher | _____ |
| 4. Cross-Chain Bridge Agent | _____ |
| 5. RL Trading Agent | _____ |
| 6. DeFi Management Agent | _____ |
| 7. LLM-Powered Agent | _____ |
| 8. Autonomous DAO Agent | _____ |

#### Q7: How distinguishable is this category from the other 7? / 该分类与其他7类的可区分度如何？

| Scale | 1 = Highly Overlapping (高度重叠) | 5 = Clearly Distinct (明显可区分) |
|-------|----------------------------------|----------------------------------|

| Category | Rating (1-5) |
|----------|-------------|
| 1. Deterministic Script | _____ |
| 2. Simple Trading Bot | _____ |
| 3. MEV Searcher | _____ |
| 4. Cross-Chain Bridge Agent | _____ |
| 5. RL Trading Agent | _____ |
| 6. DeFi Management Agent | _____ |
| 7. LLM-Powered Agent | _____ |
| 8. Autonomous DAO Agent | _____ |

#### Q8: Can you provide real-world examples we missed? / 是否有我们遗漏的真实案例？ (Open-ended / 开放式)

| Category | Your Examples |
|----------|--------------|
| 1. Deterministic Script | _____ |
| 2. Simple Trading Bot | _____ |
| 3. MEV Searcher | _____ |
| 4. Cross-Chain Bridge Agent | _____ |
| 5. RL Trading Agent | _____ |
| 6. DeFi Management Agent | _____ |
| 7. LLM-Powered Agent | _____ |
| 8. Autonomous DAO Agent | _____ |

#### Q9: Should this category be split, merged, or removed? / 该分类是否应拆分、合并或删除？

For each category, select one and provide explanation if applicable:

| Category | Keep as-is / Split / Merge / Remove / Redefine | Explanation |
|----------|-------------------------------------------------|-------------|
| 1. Deterministic Script | _____ | _____ |
| 2. Simple Trading Bot | _____ | _____ |
| 3. MEV Searcher | _____ | _____ |
| 4. Cross-Chain Bridge Agent | _____ | _____ |
| 5. RL Trading Agent | _____ | _____ |
| 6. DeFi Management Agent | _____ | _____ |
| 7. LLM-Powered Agent | _____ | _____ |
| 8. Autonomous DAO Agent | _____ | _____ |

---

### Part C: Classification Exercise / 分类练习

Please classify each of the following 10 real-world entities into one of the 8 categories above (or "Not an Agent").

请将以下10个真实实体分类到上述8个类别之一（或"非Agent"）。

| # | Entity | Description |
|---|--------|-------------|
| 1 | **jaredfromsubway.eth** | Ethereum's most prolific MEV extractor. Conducts sandwich attacks on DEX trades by monitoring the mempool, front-running victim transactions, and back-running to capture price impact. Operates via Flashbots bundles. |
| 2 | **Wintermute** | Major crypto market maker providing liquidity across CEXes and DEXes. Algorithmic trading with high frequency and tight spreads. Interacts primarily with DEX router contracts. |
| 3 | **Autonolas OLAS agent** | Registered in Autonolas ServiceRegistry. Manages DeFi positions across Aave, Uniswap, Yearn. Hybrid decision model: statistical risk assessment + rule-based execution. |
| 4 | **Uniswap V2 Router** | Smart contract that routes token swaps through Uniswap V2 pools. Deterministic swap logic (constant product formula). No off-chain component or autonomous decision-making. |
| 5 | **Personal wallet (manual trades)** | Human-controlled EOA. Manual trades via Uniswap web app. Shows circadian rhythm, irregular intervals, round-number gas prices. |
| 6 | **Aave liquidation bot** | Monitors Aave positions; triggers liquidations when health factors drop below thresholds. Uses statistical models for gas cost and profitability estimation. Adapts gas bidding based on competition. |
| 7 | **LayerZero relayer** | Monitors source-chain events; submits proof transactions on destination chains (Ethereum, Arbitrum, Optimism, Polygon). Deterministic relay logic with adaptive gas pricing. |
| 8 | **AI16Z ELIZA trading agent** | Uses GPT-4 to analyze social media sentiment, news, and on-chain data. Executes token swaps on Uniswap based on LLM reasoning. Posts updates on Twitter. |
| 9 | **Cron payroll script** | Runs every 2 weeks; sends fixed USDC amounts to predefined employee addresses. No conditional logic, no adaptation, identical calldata every execution. |
| 10 | **Chainlink oracle node** | Submits price feed updates to Chainlink aggregator contracts. Deterministic submission logic triggered by price deviation thresholds. Operates within Chainlink consensus protocol. |

**For each entity, provide: / 对于每个实体，请提供：**

| # | Entity | Category (from 8 above, or "Not an Agent") | Satisfies C1? | Satisfies C2? | Satisfies C3? | Satisfies C4? | Confidence (1-5) | Rationale |
|---|--------|---------------------------------------------|---------------|---------------|---------------|---------------|-------------------|-----------|
| 1 | jaredfromsubway.eth | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 2 | Wintermute | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 3 | Autonolas OLAS agent | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 4 | Uniswap V2 Router | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 5 | Personal wallet | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 6 | Aave liquidation bot | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 7 | LayerZero relayer | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 8 | AI16Z ELIZA agent | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 9 | Cron payroll script | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |
| 10 | Chainlink oracle node | _____ | Y/N | Y/N | Y/N | Y/N | _____ | _____ |

---

### Part D: Framework Comparison / 框架对比

Rate the usefulness of each framework for classifying AI agents in the Web3/blockchain context:

请评价以下每个框架对 Web3/区块链场景中 AI Agent 分类的有用性：

| Scale | 1 = Not at all useful (完全无用) | 5 = Extremely useful (非常有用) |
|-------|----------------------------------|-------------------------------|

| # | Framework | Key Idea | Rating (1-5) |
|---|-----------|----------|-------------|
| 1 | Russell & Norvig (2020) | 5-type hierarchy: Simple reflex -> Model-based -> Goal-based -> Utility-based -> Learning | _____ |
| 2 | Wooldridge & Jennings (1995) | 4 properties: Autonomy + Social ability + Reactivity + Pro-activeness | _____ |
| 3 | Franklin & Graesser (1996) | 13-property combinatorial taxonomy | _____ |
| 4 | Parasuraman et al. (2000) | 10-level automation scale across 4 information-processing stages | _____ |
| 5 | He et al. (2025) arXiv 2601.04583 | AI agent survey: perception-reasoning-action-learning loop | _____ |
| 6 | **Our C1-C4 + 8-category taxonomy** | 4 necessary conditions + 3-dimensional (Autonomy x Environment x Decision) 8-category classification | _____ |

#### Q-D1: What does our framework capture that others miss? / 我们的框架捕捉了哪些其他框架遗漏的内容？

Response / 回答: __________________________________________________________

#### Q-D2: What do other frameworks capture that ours misses? / 其他框架捕捉了哪些我们遗漏的内容？

Response / 回答: __________________________________________________________

#### Q-D3: Overall assessment / 总体评价 (Open-ended / 开放式)

"Considering all frameworks, how would you rank our taxonomy's contribution to the field?"

"综合考虑所有框架，您如何评价我们分类体系对该领域的贡献？"

Response / 回答: __________________________________________________________

---

## Round 2: Convergence / 第2轮：共识收敛

*(This section will be populated after Round 1 analysis / 本部分将在第1轮分析后填充)*

### Instructions / 说明

In Round 1, you and your fellow experts independently rated the definition conditions, taxonomy categories, and comparison frameworks. Below we present the group's aggregated results.

在第1轮中，您和其他专家独立评价了定义条件、分类类别和对比框架。以下展示小组汇总结果。

**Consensus criterion / 共识标准**: IQR (Interquartile Range) <= 1.0

### For each item below / 对于以下每个条目:

- If **consensus reached** (IQR <= 1): No re-rating needed unless you have strong objections.
  如果**已达成共识**（IQR <= 1）：无需重新评分，除非您有强烈异议。

- If **consensus NOT reached** (IQR > 1): Please re-rate considering the group median and summarized feedback.
  如果**未达成共识**（IQR > 1）：请在参考小组中位数和汇总反馈后重新评分。

### Template for Round 2 Items / 第2轮条目模板

| Question ID | Original Median | IQR | Consensus? | Summarized Feedback | Your New Rating (1-5) |
|-------------|-----------------|-----|------------|---------------------|----------------------|
| [To be filled after Round 1] | | | | | |

### Additional Round 2 Questions / 第2轮附加问题

Q-R2-1: "After seeing the group's feedback, has your view on any condition (C1-C4) changed? If so, how?"

"在看到小组反馈后，您对任何条件（C1-C4）的看法是否有变化？如有，请说明。"

Response / 回答: __________________________________________________________

Q-R2-2: "Are there any new suggestions for categories or modifications that emerged from the group feedback?"

"小组反馈中是否出现了新的分类建议或修改意见？"

Response / 回答: __________________________________________________________

Q-R2-3: "Final overall assessment: Is the C1-C4 definition + 8-category taxonomy ready for publication?" (1-5)

"最终总体评价：C1-C4定义 + 8类分类体系是否已准备好发表？"（1-5）

Rating / 评分: _____

---

## Expert Demographics / 专家背景信息

*(Collected once, before Round 1 / 在第1轮前一次性收集)*

| Field | Response |
|-------|----------|
| Name (optional) / 姓名（可选） | _____ |
| Affiliation / 所属机构 | _____ |
| Primary expertise area / 主要专业领域 | [ ] AI/ML [ ] Blockchain/DeFi [ ] Security [ ] HCI [ ] Other: _____ |
| Years in field / 从业年数 | _____ |
| # of publications in this area / 该领域发表论文数 | _____ |
| Familiarity with on-chain agents / 对链上Agent的熟悉程度 (1-5) | _____ |
| Familiarity with AI agent theory / 对AI Agent理论的熟悉程度 (1-5) | _____ |

---

*Thank you for your participation. Your expertise is essential for validating this framework.*

*感谢您的参与。您的专业知识对验证此框架至关重要。*
