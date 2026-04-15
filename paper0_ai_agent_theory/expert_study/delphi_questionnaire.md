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

**Consensus criterion**: An item reaches consensus when the interquartile range (IQR) of expert ratings is <= 1.0 on the 7-point scale.
**共识标准**: 当专家评分的四分位距（IQR）在7分量表上 <= 1.0 时，该条目达成共识。

**Sample size**: N = 12-15 experts (minimum N >= 12 for the full study; N >= 5 acceptable for a pilot round). Justification: Delphi panels of 10-18 experts produce stable consensus estimates (Hallowell & Gambatese, 2010; Skulmoski, Hartman & Krahn, 2007).
**样本量**: N = 12-15 位专家（完整研究至少 N >= 12；试点轮 N >= 5 可接受）。依据：10-18 人的德尔菲面板可产生稳定的共识估计（Hallowell & Gambatese, 2010; Skulmoski, Hartman & Krahn, 2007）。

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
| 3 | Somewhat Disagree | 有些不同意 |
| 4 | Neutral | 中立 |
| 5 | Somewhat Agree | 有些同意 |
| 6 | Agree | 同意 |
| 7 | Strongly Agree | 非常同意 |

| Condition | Your Rating (1-7) |
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
| 2 | Unclear | 不清晰 |
| 3 | Somewhat Unclear | 有些不清晰 |
| 4 | Neutral | 中立 |
| 5 | Somewhat Clear | 有些清晰 |
| 6 | Clear | 清晰 |
| 7 | Very Clear | 非常清晰 |

| Condition | Your Rating (1-7) |
|-----------|-------------------|
| C1: On-chain Actuation / 链上执行 | _____ |
| C2: Environmental Perception / 环境感知 | _____ |
| C3: Autonomous Decision-Making / 自主决策 | _____ |
| C4: Adaptiveness / 适应性 | _____ |

#### Q2b: Operationalizability / 可操作性

"Can this condition be reliably measured or verified from on-chain transaction data alone?"

"仅从链上交易数据是否能可靠地测量或验证该条件？"

| Scale | Label (EN) | 标签（中文） |
|-------|------------|-------------|
| 1 | Not at all measurable | 完全不可测量 |
| 2 | Mostly not measurable | 大部分不可测量 |
| 3 | Somewhat not measurable | 有些不可测量 |
| 4 | Neutral | 中立 |
| 5 | Somewhat measurable | 有些可测量 |
| 6 | Mostly measurable | 大部分可测量 |
| 7 | Fully measurable | 完全可测量 |

| Condition | Your Rating (1-7) |
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

#### Q4: Joint Sufficiency / 联合充分性

"Are these four conditions jointly sufficient to define an AI agent in Web3? Are any conditions missing?"

"这四个条件联合起来是否足以定义 Web3 AI Agent？是否有缺失的条件？"

Rate on a 1-7 scale: 请在1-7量表上评分：

| Scale | 1 = Definitely insufficient (明显不充分) | 7 = Fully sufficient (完全充分) |
|-------|-------------------------------------------|---------------------------------|

Rating / 评分: _____

If you rated <= 4, what additional condition(s) would you add? / 如果您评分 <= 4，您会增加什么条件？

Response / 回答: __________________________________________________________

#### Q5: Redundancy Check / 冗余性检查 (Open-ended / 开放式)

"Should any condition be removed or merged with another? Please explain."

"是否有任何条件应当被删除或与其他条件合并？请说明理由。"

Response / 回答: __________________________________________________________

#### Q5b: Boundary Cases / 边界案例

"For each boundary case below, indicate whether it should be classified as an AI agent under C1-C4. If the definition produces a counter-intuitive result, explain why."

"对于以下每个边界案例，请判断其在 C1-C4 定义下是否应被归类为 AI Agent。如果定义产生了违反直觉的结果，请说明原因。"

| # | Boundary Case | Agent under C1-C4? (Y/N/Borderline) | Which condition(s) are ambiguous? | Your reasoning |
|---|---------------|--------------------------------------|-----------------------------------|----------------|
| 1 | **Account Abstraction (ERC-4337) wallet** with automated bundler: user delegates transaction signing to a bundler that optimizes gas and batches operations. | _____ | _____ | _____ |
| 2 | **Cross-chain bridge relayer** that monitors events on Chain A and submits proofs on Chain B, using adaptive gas pricing but deterministic relay logic. | _____ | _____ | _____ |
| 3 | **MEV bot with frozen strategy**: deployed once, never updated, but uses statistical models for real-time arbitrage decisions. Satisfies C1-C3 but arguably not C4. | _____ | _____ | _____ |
| 4 | **Multi-sig wallet with Gnosis Safe module**: automated execution triggered by governance vote outcomes. The module itself is deterministic. | _____ | _____ | _____ |
| 5 | **Chainlink oracle node**: submits price updates when deviation thresholds are crossed. Deterministic submission logic, but adaptive heartbeat intervals. | _____ | _____ | _____ |
| 6 | **Intent-based protocol solver** (e.g., CoW Protocol solver): fills user intents by finding optimal routes. Uses optimization algorithms but operates within a permissioned solver set. | _____ | _____ | _____ |

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

| Scale | 1 = Very Poorly Defined (非常不清晰) | 4 = Neutral (中立) | 7 = Very Well-Defined (非常清晰) |
|-------|--------------------------------------|--------------------|---------------------------------|

| Category | Rating (1-7) |
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

| Scale | 1 = Highly Overlapping (高度重叠) | 4 = Neutral (中立) | 7 = Clearly Distinct (明显可区分) |
|-------|----------------------------------|--------------------|---------------------------------|

| Category | Rating (1-7) |
|----------|-------------|
| 1. Deterministic Script | _____ |
| 2. Simple Trading Bot | _____ |
| 3. MEV Searcher | _____ |
| 4. Cross-Chain Bridge Agent | _____ |
| 5. RL Trading Agent | _____ |
| 6. DeFi Management Agent | _____ |
| 7. LLM-Powered Agent | _____ |
| 8. Autonomous DAO Agent | _____ |

#### Q7b: Taxonomy-Level Assessment / 分类体系整体评估

**Exhaustiveness**: "Does this 8-category taxonomy cover all meaningful types of on-chain AI agents you are aware of?"

**穷尽性**: "该8类分类体系是否覆盖了您所知的所有有意义的链上 AI Agent 类型？"

| Scale | 1 = Major types missing (有重要类型缺失) | 4 = Neutral (中立) | 7 = Fully exhaustive (完全穷尽) |
|-------|-------------------------------------------|--------------------|--------------------------------|

Rating / 评分: _____

If you rated <= 4, what categories are missing? / 如果您评分 <= 4，缺失了哪些类别？

Response / 回答: __________________________________________________________

**Mutual Exclusivity**: "Can a real-world agent be unambiguously assigned to exactly one of the 8 categories?"

**互斥性**: "一个真实世界的 Agent 是否可以被无歧义地归入恰好一个分类？"

| Scale | 1 = Frequent overlap (经常重叠) | 4 = Neutral (中立) | 7 = Always unambiguous (总是无歧义) |
|-------|--------------------------------|--------------------|------------------------------------|

Rating / 评分: _____

If you rated <= 4, which category pairs overlap most? / 如果您评分 <= 4，哪些类别对最容易重叠？

Response / 回答: __________________________________________________________

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

*Confidence scale: 1 = Pure guess, 4 = Moderate confidence, 7 = Completely certain*
*信心量表：1 = 纯粹猜测，4 = 中等信心，7 = 完全确定*

| # | Entity | Category (from 8 above, or "Not an Agent") | Satisfies C1? | Satisfies C2? | Satisfies C3? | Satisfies C4? | Confidence (1-7) | Rationale |
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

| Scale | 1 = Not at all useful (完全无用) | 4 = Neutral (中立) | 7 = Extremely useful (非常有用) |
|-------|----------------------------------|--------------------|---------------------------------|

| # | Framework | Key Idea | Rating (1-7) |
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

### Convergence Protocol / 收敛协议

**Step 1**: After Round 1 closes, the research team computes descriptive statistics for every Likert item: median, IQR, mean, SD, and frequency distribution.

**第1步**: 第1轮结束后，研究团队计算每个李克特条目的描述性统计量：中位数、IQR、均值、标准差和频率分布。

**Step 2**: Items are classified into three categories:
- **Consensus reached (IQR <= 1.0)**: Retained as-is. Experts may optionally provide dissenting comments but do not re-rate.
- **Near-consensus (1.0 < IQR <= 1.5)**: Presented with group median and a blinded summary of the two most divergent rationales. Experts re-rate on the same 1-7 scale.
- **No consensus (IQR > 1.5)**: Presented with full distribution histogram, group median, and all anonymized open-ended comments. Experts re-rate and provide a revised rationale.

**第2步**: 条目分为三类：
- **已达成共识（IQR <= 1.0）**：保留原状。专家可选择性提供异议意见，但无需重新评分。
- **接近共识（1.0 < IQR <= 1.5）**：展示小组中位数和两个最分歧理由的匿名摘要。专家在同一1-7量表上重新评分。
- **未达成共识（IQR > 1.5）**：展示完整分布直方图、小组中位数和所有匿名开放式评论。专家重新评分并提供修订后的理由。

**Step 3**: After Round 2, final consensus is assessed. Items still with IQR > 1.5 after Round 2 are reported as "unresolved disagreements" in the paper, with the distribution of expert opinion described transparently.

**第3步**: 第2轮后，进行最终共识评估。第2轮后IQR仍 > 1.5的条目在论文中报告为"未解决分歧"，并透明描述专家意见分布。

**Stopping rule**: The study uses a maximum of 2 rounds. A third round is triggered only if fewer than 70% of items reach consensus AND at least 80% of panelists agree to continue.

**终止规则**: 本研究最多进行2轮。仅当少于70%的条目达成共识且至少80%的专家同意继续时，才启动第3轮。

### Instructions / 说明

In Round 1, you and your fellow experts independently rated the definition conditions, taxonomy categories, and comparison frameworks. Below we present the group's aggregated results. You provided your original rating of [YOUR_SCORE]; the group results are shown below.

在第1轮中，您和其他专家独立评价了定义条件、分类类别和对比框架。以下展示小组汇总结果。您的原始评分为 [YOUR_SCORE]；小组结果如下所示。

### Template for Round 2 Items (Near-consensus and No-consensus only) / 第2轮条目模板（仅限接近共识和未达共识条目）

| Question ID | Your R1 Score | Group Median | IQR | Status | Summarized Feedback | Your New Rating (1-7) | Revised Rationale (if changed) |
|-------------|---------------|--------------|-----|--------|---------------------|----------------------|-------------------------------|
| [To be filled after Round 1] | | | | | | | |

### Additional Round 2 Questions / 第2轮附加问题

Q-R2-1: "After seeing the group's feedback, has your view on any condition (C1-C4) changed? If so, how?"

"在看到小组反馈后，您对任何条件（C1-C4）的看法是否有变化？如有，请说明。"

Response / 回答: __________________________________________________________

Q-R2-2: "Are there any new suggestions for categories or modifications that emerged from the group feedback?"

"小组反馈中是否出现了新的分类建议或修改意见？"

Response / 回答: __________________________________________________________

Q-R2-3: "The group identified the following boundary cases as ambiguous: [LIST]. For each, please provide your revised classification and reasoning."

"小组认为以下边界案例存在歧义：[列表]。请对每个案例提供您修订后的分类和理由。"

Response / 回答: __________________________________________________________

Q-R2-4: "Final overall assessment: Is the C1-C4 definition + 8-category taxonomy ready for publication?"

"最终总体评价：C1-C4定义 + 8类分类体系是否已准备好发表？"

| Scale | 1 = Needs major revision (需要大幅修订) | 4 = Needs minor revision (需要小幅修订) | 7 = Ready as-is (可直接发表) |
|-------|----------------------------------------|----------------------------------------|------------------------------|

Rating / 评分: _____

Explanation / 说明: __________________________________________________________

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
| Familiarity with on-chain agents / 对链上Agent的熟悉程度 (1-7) | _____ |
| Familiarity with AI agent theory / 对AI Agent理论的熟悉程度 (1-7) | _____ |
| Familiarity with agent taxonomies / 对Agent分类体系的熟悉程度 (1-7) | _____ |
| Have you personally built, deployed, or audited an on-chain agent? / 您是否亲自构建、部署或审计过链上Agent？ | [ ] Yes / [ ] No |

---

*Thank you for your participation. Your expertise is essential for validating this framework.*

*感谢您的参与。您的专业知识对验证此框架至关重要。*
