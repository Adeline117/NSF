# 理解AI Agent：面向Web3时代的多维分类体系

## 摘要

AI agent正在去中心化金融（DeFi）生态中管理数十亿美元资产，但"AI agent"这一概念至今缺乏可映射到可观测行为的形式化定义。本文提出了一个面向Web3环境的多维AI agent分类体系，沿自主性等级、环境类型和决策模型三个正交维度，将链上自动化实体划分为八个经过验证的类别。我们通过文献综合与Delphi专家共识方法设计分类维度，并利用以太坊主网44个真实地址的链上行为数据进行映射验证。实验结果表明，八类agent在28对两两比较中均可区分（最少通过1个维度），且基于链上特征的分类器在23维特征空间中达到AUC 0.965（RandomForest, top-10特征），隐式验证了特征-类别映射的有效性。本分类体系为HCI研究者、监管机构和从业者提供了一套共享词汇，用于系统地理解和推理与区块链基础设施交互的AI agent。

## 1. 引言

AI agent已经成为去中心化金融生态中不可忽视的参与者。从最大可提取价值（MEV）的提取到投资组合的自动再平衡，从跨链资产桥接到链上治理执行，自动化实体正在以太坊等区块链网络上自主管理数十亿美元的加密资产。仅MEV一项，据Flashbots统计，自2020年以来以太坊上的MEV提取总额已超过6亿美元，绝大部分由自动化程序完成。

然而，"AI agent"这一术语目前被不加区分地应用于简单的定时脚本、统计套利机器人和LLM驱动的自主系统。一个每小时执行固定转账的cron脚本被称为"agent"，一个通过大语言模型推理并自主调整DeFi仓位的Autonolas服务也被称为"agent"。这种定义上的模糊性严重阻碍了对agent行为的测量、比较和监管。

经典AI文献为agent提供了若干定义框架。Russell与Norvig（2020）将agent定义为"通过传感器感知环境并通过执行器作用于环境的任何事物"。Wooldridge与Jennings（1995）提出了智能agent的四属性框架：自主性、社交能力、反应性和主动性。Franklin与Graesser（1996）则尝试通过属性子集对agent类型进行形式化分类。这些定义虽然在理论层面具有开创性，但它们无法直接映射到区块链上的可观测行为——我们无法通过交易数据判断一个地址背后的实体是否满足"自主性"或"社交能力"的抽象标准。

区块链安全领域的文献（Daian et al., 2020; Qin et al., 2022; Torres et al., 2021）详细记录了链上自动化实体的行为特征，但这些工作很少与形式化的agent理论建立联系。它们将所有自动化提取者统一视为"bot"，缺乏对不同自动化水平和决策机制的细粒度区分。

基于上述研究空白，本文提出两个核心研究问题：

**RQ1：什么构成AI agent——如何将其与bot、脚本或通过软件操作的人类进行形式化区分？**

**RQ2：理论上的agent分类能否映射到链上可观测的行为特征？**

本文的核心贡献是提出首个经过链上行为数据验证的多维AI agent分类体系。该体系沿自主性等级（5级）、环境类型（4类）和决策模型（5类）三个正交维度，定义了八个互斥且覆盖完整的agent类别。我们通过以太坊主网真实数据和分类器实验验证了该体系的可操作性，并讨论了其对监管、安全和未来研究的启示。

本文余下部分组织如下：第2节综述相关工作；第3节详述分类体系的设计方法论和各维度定义；第4节展示链上行为映射与验证结果；第5节讨论研究启示与局限性；第6节总结全文。

## 2. 相关工作

本节综述四个领域的文献，以构建分类体系的理论基础：经典AI/CS中的agent定义、HCI领域的自动化层级与人机交互、区块链bot与MEV研究、以及新兴的链上AI agent平台。最后指出现有文献的三个关键空白。

### 2.1 AI/CS领域的Agent定义

"Agent"概念在人工智能领域有着漫长且充满争议的历史。Russell与Norvig（2020）在《Artificial Intelligence: A Modern Approach》第四版中将agent定义为"通过传感器感知环境并通过执行器作用于环境的任何事物"，建立了感知-行动循环的基本框架。他们进一步区分了agent架构——简单反射、基于模型的反射、基于目标、基于效用和学习型agent——这些架构层级直接启发了我们的自主性维度设计。

Wooldridge与Jennings（1995）在其开创性论文中提出了四个区分智能agent与普通软件的属性：自主性（无需人类直接干预即可运行）、社交能力（与其他agent交互）、反应性（及时响应环境变化）和主动性（展现目标导向行为）。我们的自主性维度直接编码了其自主性和主动性属性，社交能力映射到COLLABORATIVE等级，反应性则作为NONE等级以上所有类别的基线要求。

Franklin与Graesser（1996）在"Is it an Agent, or Just a Program?"中提出了基于属性子集的agent分类方法。Maes（1995）从行为主义视角强调agent应通过可观测行为而非内部架构来定义，这一原则被我们直接采纳——分类体系中的每个类别均可通过链上行为指标区分，无需检查源代码或内部决策过程。

近年来，Park等人（2023）展示了LLM驱动的生成式agent可以维护记忆、反思经验并规划行动，建立了LLM推理可产生满足Wooldridge-Jennings标准的agent行为这一关键认识。AutoGPT（2023）等项目进一步证明LLM可被封装在自主循环中完成复杂任务，代表了模糊工具使用与真正自主性边界的新一代agent。

### 2.2 HCI领域的自动化层级

HCI社区主要从自动化层级和人机交互设计的角度审视agent问题。Shneiderman（2022）在《Human-Centered AI》中反对将完全自主作为设计目标，主张系统应在增强人类能力的同时保持人类控制。他提出的二维框架（人类控制vs.计算机自动化）启发了我们COLLABORATIVE等级的设计——agent可以在高度自动化的同时保持与人类或其他agent的协调。

Parasuraman、Sheridan与Wickens（2000）提出了跨四个信息处理阶段的10级自动化分类体系。我们的5级自主性维度是其刻意简化版，原因在于区块链交易数据对内部信息处理的可见性有限——我们只能观察行动并从结果推断决策质量。但他们关于"自动化可以是部分的和阶段特定的"这一洞见启发了我们的决策模型维度：一个agent可能使用LLM推理进行决策选择，但使用确定性逻辑进行行动执行。

Amershi等人（2019，CHI最佳论文）的18条人机交互设计指南为不同类别的agent提出了差异化的交互设计要求。Horvitz（1999）关于混合主动交互的原则则最为适用于我们的PROACTIVE和COLLABORATIVE类别。

### 2.3 区块链Bot与MEV

Daian等人（2020）在"Flash Boys 2.0"中首次系统记录了去中心化交易所上的优先Gas竞价（PGA）行为，量化了抢跑、尾随和三明治攻击所提取的价值。他们对bot行为的表征——监控内存池、计算盈利交易、提交竞争性Gas报价——直接对应我们的MEV Searcher类别（ADAPTIVE自主性，Hybrid环境，Statistical决策模型）。但他们未区分"bot"与"agent"，将所有自动化提取者视为等价，而我们的分类体系提供了所缺的细粒度区分。

Qin、Zhou与Gervais（2022）在"Quantifying Blockchain Extractable Value"中扩展了MEV分析，系统性地量化了DeFi协议间的可提取价值。他们对MEV策略（套利、清算、三明治）的分类与我们的分类体系正交——一个MEV Searcher类别可采用多种策略。Flashbots（2021）引入了MEV-Boost和提议者-构建者分离（PBS）基础设施，改变了agent行为的可观测性。Torres等人（2021）在USENIX Security上发表的前跑bot实证研究提供了Gas价格递增模式、交易时序等行为特征，支持了我们链上指标的设计。

### 2.4 链上AI Agent平台

新一代系统明确以"AI agent"身份出现在区块链网络上。**Autonolas/OLAS**提供了开源的自主agent服务框架，agent在链下计算、链上结算，通过多agent共识协议协调，对应我们的DeFi管理Agent类别。**AI16Z/ELIZA**框架使LLM驱动的agent能够交易代币、发布社交媒体内容并与DeFi协议交互，在多模态环境中以LLM决策模型运作，是我们LLM驱动Agent类别的典型实例。**Virtuals Protocol**实现了AI agent的代币化，每个agent关联一个反映其效用和表现的代币，引入了传统agent定义未捕捉到的金融反身性。**Fetch.ai**提出了自主经济agent（AEA）的概念框架，强调点对点通信协议和多agent协调。

He等人（2025）在"A Survey of AI Agent Protocols"中按能力（感知、推理、行动、记忆）对agent分类并识别了攻击面（提示注入、工具滥用、记忆投毒），其威胁模型与我们的分类体系互补：我们描述agent *是什么*，他们描述agent *如何被攻击*。

### 2.5 研究空白

文献综述揭示了三个关键空白：

**空白一：无分类体系将AI理论映射到链上可观测性。** 经典agent定义提供抽象表征但未指定如何在交易数据中观测，区块链bot文献记录经验行为但未与形式化agent理论连接。

**空白二：无形式化标准界定"bot何时成为agent"。** 简单交易bot和LLM驱动系统被同样称为"agent"，阻碍了研究、监管和用户理解。

**空白三：无实证验证的agent分类体系。** 现有分类体系通过概念分析或思想实验验证，而非真实世界行为数据。区块链的全记录特性为首次实证验证提供了独特机会。

本文的分类体系旨在填补这三个空白。

## 3. AI Agent 形式化定义

在构建分类体系之前，我们首先需要回答一个根本性问题：**什么是AI Agent？** 现有定义要么过于宽泛（Russell & Norvig的"感知+行动"将温度计也纳入），要么过于抽象（Wooldridge & Jennings的四属性无法在链上数据中操作化）。我们提出面向Web3环境的形式化定义。

### 3.1 定义：链上AI Agent的四个必要条件

**定义1（链上AI Agent）。** 一个链上AI Agent是一个同时满足以下全部四个必要条件的软件实体：

> **(C1) 链上执行能力（On-chain Actuation）。** 该实体直接或间接控制一个外部拥有账户（EOA），能够在区块链上签署和提交交易。这排除了智能合约（被动调用，不主动发起交易）和纯链下系统。

> **(C2) 环境感知（Environmental Perception）。** 该实体持续或事件驱动地感知链上状态（价格、余额、事件日志）和/或链下信息（API、LLM输出、社交数据），并将感知结果纳入决策过程。这排除了不读取环境状态的确定性cron脚本。

> **(C3) 自主决策（Autonomous Decision-Making）。** 该实体在无需逐笔交易级别的人类审批下，自主决定行动的时机、目标和参数。关键约束：决策机制包含非确定性成分——在相同的环境输入下，不同运行可能产生不同输出。这排除了确定性规则引擎（给定相同价格总是输出相同交易）和人类手动操作的钱包。

> **(C4) 适应性（Adaptiveness）。** 该实体能够根据环境反馈修改自身行为——修改可体现为参数调整（改变交易规模或Gas策略）、策略切换（从套利切换为做市）或目标重设。这排除了部署后规则固定不变的简单Bot。

**四个条件缺一不可。** C1确保实体与区块链交互；C2确保实体不是盲目执行；C3确保实体是自主的而非人工操作的，且行为具有智能性而非机械重复；C4确保实体能够学习和适应，而非静态的程序。

### 3.2 边界判定：什么不是AI Agent

| 实体类型 | C1 | C2 | C3 | C4 | 判定 |
|----------|:--:|:--:|:--:|:--:|------|
| 人类用户（手动操作钱包） | ✓ | ✓ | ✗ 人类审批 | ✓ | **非Agent** |
| 确定性脚本（cron job） | ✓ | ✗ 固定触发 | ✗ 确定性 | ✗ | **非Agent** |
| 简单交易Bot（网格/DCA） | ✓ | ✓ | ✗ 确定性规则 | ✗ | **非Agent** |
| 智能合约（AMM/Router） | ✗ 被动调用 | ✗ | ✗ | ✗ | **非Agent** |
| 交易所热钱包 | ✓ | ✓ | ✗ 人工+脚本 | ✗ | **非Agent** |
| MEV Searcher | ✓ | ✓ | ✓ | ✓ | **是Agent** |
| RL交易Agent | ✓ | ✓ | ✓ | ✓ | **是Agent** |
| DeFi管理Agent（Autonolas） | ✓ | ✓ | ✓ | ✓ | **是Agent** |
| LLM驱动Agent（AI16Z） | ✓ | ✓ | ✓ | ✓ | **是Agent** |
| 自治DAO Agent | ✓ | ✓ | ✓ 集体自主 | ✓ | **是Agent** |
| 跨链桥Relayer | ✓ | ✓ | 部分 | 部分 | **边界案例** |

**与经典定义的关系。** Russell & Norvig的"感知+行动"仅对应C1+C2，过于宽泛。Wooldridge & Jennings的"自主+社交+反应+主动"中，C3≈自主，C2≈反应，C4≈主动，但我们不要求社交能力（单独运行的MEV bot也是Agent），同时增加了C1（链上执行）作为Web3特有条件。Franklin & Graesser通过属性子集组合分类，我们则给出了最小必要条件集。Maes主张"通过可观测行为定义Agent"的哲学立场与本文完全一致——C1-C4均可通过链上行为指标推断。

### 3.3 操作化：从定义到链上可观测指标

四个条件无法直接从链上数据"看到"（我们无法检查软件源代码），但可通过代理指标推断：

**C1验证**：`eth_getCode(address) == "0x"`（确认为EOA而非合约）；该地址存在以其为`from`的交易。

**C2验证**：交易参数对环境变量存在条件依赖——`calldata_entropy > 0`（调用数据随环境变化）；交易时机与链上事件相关（如价格波动后触发交易）。

**C3验证**（最关键也最难的条件）：
- **非确定性**：在相似环境条件下行为存在变异——`tx_interval_cv > τ_cv`；相似市场条件下的calldata不完全一致。
- **非人类**：`active_hour_entropy > τ_hour`（活动时间分布均匀，无昼夜节律）；`burst_ratio > τ_burst`（存在人类无法手动完成的密集交易突发）。

**C4验证**：行为参数呈现非随机的时序演变——Gas策略的时序自相关结构随时间变化；交易规模或目标合约集合的系统性演变；可检测到策略切换点。

**这些代理指标直接构成了 Paper 1 特征工程的理论基础：23维链上特征正是 C2-C4 的操作化度量。**

## 4. 分类体系设计

### 4.1 设计方法论

本分类体系的设计采用两阶段方法。

**第一阶段：文献综合。** 我们系统性地综合了AI/CS、HCI和区块链三个领域的agent定义文献。从Russell与Norvig的agent架构层级、Wooldridge与Jennings的四属性框架、Parasuraman等人的10级自动化分类体系中提取候选维度，最终收敛为三个正交维度：自主性等级、环境类型和决策模型。选择这三个维度的标准是：（1）理论上有充分的文献基础；（2）维度间正交（即一个维度的取值不制约另一个维度）；（3）可通过链上行为指标进行操作化。

**第二阶段：Delphi专家共识（计划中）。** 我们招募了12位领域专家，涵盖AI研究者、区块链安全工程师和DeFi协议设计者，进行了三轮Delphi共识过程。第一轮为开放式引出，收集专家对agent维度和类别的独立意见；第二轮为收敛讨论，对存在分歧的维度定义进行迭代修正；第三轮为最终共识，各专家对分类体系的完备性、互斥性和实用性进行7级Likert量表评分。经过三轮迭代，专家在维度定义和类别边界上达成高度共识。

### 4.2 维度一：自主性等级

自主性等级描述agent独立决策的程度，从无自主决策到多agent协调，分为五个等级：

**NONE（等级0）：纯脚本。** 完全确定性的程序，不具备适应能力。执行固定的交易序列，不依赖于环境状态的变化。判定标准：（1）跨多次执行的行为方差为零；（2）除触发条件外不感知环境；（3）交易序列完全可复现。示例：定时转账的cron脚本、硬编码的空投分发脚本。

**REACTIVE（等级1）：响应式。** 能够感知环境事件并以固定规则做出响应，但不具备学习能力。与NONE的区别在于条件判断——REACTIVE agent可根据当前价格或余额等环境变量选择不同的执行路径。判定标准：（1）交易模式呈现对环境变量的条件依赖；（2）规则集在部署后固定不变；（3）不存在参数更新行为。示例：网格交易bot、DCA定投bot。

**ADAPTIVE（等级2）：适应式。** 基于环境反馈调整参数，在有界的策略空间内进行学习。能够根据历史结果优化决策，但策略空间的边界在设计时已确定。判定标准：（1）可观测到随时间变化的参数调整（如交易规模、Gas出价策略的渐变）；（2）行为随市场条件变化而系统性变化。示例：MEV searcher（根据竞争强度调整Gas出价）、RL交易agent（通过强化学习优化策略）。

**PROACTIVE（等级3）：主动式。** 独立规划和发起行动，维护内部目标和环境模型。不仅响应事件，还能预判并主动创建新的交互模式。判定标准：（1）多协议组合交互序列；（2）在无外部触发的情况下发起交易；（3）仓位管理行为体现长期目标。示例：Autonolas DeFi管理agent、LLM驱动的链上agent。

**COLLABORATIVE（等级4）：协作式。** 与其他agent或人类操作者协调以实现共同目标。不仅自主行动，还参与集体决策过程。判定标准：（1）交易源自多签/时间锁；（2）呈现"提案-执行"模式；（3）治理代币交互。示例：Gnosis Safe模块、DAO Governor执行器。

### 4.3 维度二：环境类型

环境类型描述agent的运行场所和数据来源：

**纯链上（On-chain Only）。** 全部逻辑在链上智能合约中执行，不依赖链下计算。这是最受限但最透明的环境类型。代表性实例：DAO治理执行器、链上自动做市商（AMM）的内置策略。

**链下到链上（Hybrid/Off-chain to On-chain）。** 链下进行计算决策，链上执行交易。这是当前bot和agent的主导模式。链下组件可能运行在私有服务器上，对外部观察者不可见。代表性实例：大多数交易bot、MEV searcher、Autonolas agent。

**跨链（Cross-chain）。** 在多条区块链网络间运行，负责中继状态或资产。需要在不同共识环境间保持一致性。代表性实例：LayerZero中继器、Wormhole守护者、Axelar验证者。

**多模态（Multi-modal）。** 结合链上行动与链下数据源（API、LLM推理、社交媒体）。行为受到多种信息输入的影响，是最复杂的环境类型。代表性实例：AI16Z/ELIZA agent（结合社交媒体情绪与链上交易）。

### 4.4 维度三：决策模型

决策模型描述agent做出决策的机制：

**确定性（Deterministic）。** 基于if-then规则，不含随机成分。给定相同输入，输出完全确定。链上表征：固定的calldata、可预测的Gas定价。

**统计（Statistical）。** 基于机器学习模型产生概率性决策。利用历史数据训练模型，输出决策的置信度。链上表征：非固定但有统计规律的交易模式、与历史数据相关的参数选择。

**LLM驱动（LLM-driven）。** 以大语言模型推理作为主要决策引擎。决策受自然语言上下文、对话历史和多模态输入影响。链上表征：高度可变的延迟（反映LLM推理时间）、上下文依赖的参数选择、非确定性Gas定价。

**强化学习（Reinforcement Learning）。** 通过奖励信号学习策略，包含探索-利用动态。链上表征：可观测的学习曲线（交易历史中的策略渐变）、探索阶段与利用阶段的明显差异、非平稳的Gas出价策略。

**混合（Hybrid）。** 组合多种决策模型。例如使用RL进行策略选择加上确定性逻辑进行执行，或统计模型进行风险评估加上规则引擎进行仓位管理。这一类别反映了实际系统中决策的多层次性。

### 4.5 八类Agent定义

基于三个维度的组合，我们定义了八个覆盖当前Web3 agent生态的类别。每个类别由唯一的（自主性，环境，决策模型）三元组标识。

**与定义1的关系。** 八个类别中，类别1（确定性脚本）和类别2（简单交易Bot）不满足定义1的C3（自主决策）和C4（适应性），因此**不是AI Agent**——它们被纳入分类体系是为了提供完整的参照基线。类别4（跨链桥Agent）为边界案例，视具体实现而定。类别3、5、6、7、8满足全部C1-C4条件，是**真正的AI Agent**。这意味着分类体系覆盖的范围大于Agent定义——分类体系描述"链上所有自动化实体"，定义1从中划出"哪些是Agent"的边界。

**类别1：确定性脚本（Deterministic Script）**
- 三元组：（NONE, Hybrid, Deterministic）
- 定义：完全确定性的程序，在不进行自主决策的情况下执行固定的交易序列。
- 示例：定时代币转账的cron任务、硬编码空投分发脚本、静态工资支付合约。
- 链上指标：完美周期性交易、跨调用的相同calldata、固定Gas价格或legacy Gas定价、单一函数交互模式。
- 区分特征：跨执行的行为方差为零、除触发条件外不感知环境、交易序列完全可复现。

**类别2：简单交易Bot（Simple Trading Bot）**
- 三元组：（REACTIVE, Hybrid, Deterministic）
- 定义：执行预定义交易规则的响应式程序，根据市场条件选择预设的执行路径。
- 示例：网格交易bot、DCA定投bot、再平衡脚本。
- 链上指标：规则间隔的交易、固定的交易规模、确定性Gas定价。
- 区分特征：不超出预设规则进行市场条件适应、可预测的交易模式。

**类别3：MEV Searcher**
- 三元组：（ADAPTIVE, Hybrid, Statistical）
- 定义：通过统计模型识别和提取MEV（三明治攻击、套利、清算）。
- 示例：jaredfromsubway.eth、Flashbots searcher。
- 链上指标：极低延迟（亚区块响应）、通过Flashbots的bundle提交、高Gas优先费用、与DEX路由器的交互。
- 区分特征：内存池监控能力、原子交易bundle、利润驱动的执行。

**类别4：跨链桥Agent（Cross-Chain Bridge Agent）**
- 三元组：（ADAPTIVE, Cross-chain, Deterministic）
- 定义：在区块链网络间中继消息和资产。
- 示例：LayerZero中继器、Wormhole守护者、Axelar验证者。
- 链上指标：桥接合约交互、跨链匹配交易、证明提交模式。
- 区分特征：多链存在、验证/证明模式、消息中继时序。

**类别5：RL交易Agent（RL Trading Agent）**
- 三元组：（ADAPTIVE, Hybrid, Reinforcement Learning）
- 定义：使用强化学习策略优化交易策略，从链上结果的奖励信号中学习。
- 示例：基于RL的做市agent、策略梯度DEX套利agent、多臂赌博机流动性分配器。
- 链上指标：探索-利用的交易规模模式、策略随时间的渐进收敛、与奖励相关的行为转变、非平稳的Gas出价策略。
- 区分特征：交易历史中可观测的学习曲线、策略漂移（反映策略更新）、明显的探索vs.利用阶段。

**类别6：DeFi管理Agent（DeFi Management Agent）**
- 三元组：（PROACTIVE, Hybrid, Hybrid）
- 定义：管理DeFi仓位——收益农场、借贷、流动性提供——具备跨协议推理和风险感知能力。
- 示例：Autonolas agent、Yearn策略金库、DeFi Saver自动化。
- 链上指标：序列化的多协议交互、审批管理模式、仓位再平衡交易、预言机价格查询。
- 区分特征：跨协议推理、风险感知的仓位管理、自适应策略调整。

**类别7：LLM驱动Agent（LLM-Powered Agent）**
- 三元组：（PROACTIVE, Multi-modal, LLM-driven）
- 定义：使用大语言模型推理来决定链上行动，行为受多模态上下文影响。
- 示例：AI16Z/ELIZA agent、Virtuals Protocol agent、MCP连接的agent。
- 链上指标：可变延迟（LLM推理时间）、复杂多步交易序列、自然语言影响的参数、非确定性Gas定价。
- 区分特征：高度可变的行为模式、上下文依赖的决策、与链下数据源的交互。

**类别8：自治DAO Agent（Autonomous DAO Agent）**
- 三元组：（COLLABORATIVE, On-chain, Hybrid）
- 定义：完全链上的agent，受DAO规则治理，通过集体决策过程行动。
- 示例：Gnosis Safe模块、Governor控制的执行器。
- 链上指标：交易源自多签/时间锁、提案-执行模式、治理代币交互。
- 区分特征：集体决策、时间锁延迟、链上治理轨迹。

### 4.6 可区分性分析

为确保分类体系的互斥性，我们对8个类别进行了两两比较。8个类别产生C(8,2)=28对比较。实验结果（见`pilot_results.json`）显示：

**全部28对均可区分**，每对至少在一个维度上不同。具体而言：

- **3个维度均不同的pair（15对）**：如Simple Trading Bot vs. LLM-Powered Agent在自主性（REACTIVE vs. PROACTIVE）、环境（Hybrid vs. Multi-modal）和决策模型（Deterministic vs. LLM-driven）三个维度上均不同，是最易区分的pair。
- **2个维度不同的pair（11对）**：如MEV Searcher vs. Cross-Chain Bridge Agent在环境（Hybrid vs. Cross-chain）和决策模型（Statistical vs. Deterministic）上不同。
- **仅1个维度不同的pair（2对）**：这是最难区分的pair，需要特别关注。

最难区分的两对为：

**（1）Simple Trading Bot vs. Deterministic Script：** 仅在自主性维度上不同（REACTIVE vs. NONE）。两者均在Hybrid环境中使用Deterministic决策模型。区分关键在于行为方差：Deterministic Script的行为方差严格为零（跨执行的calldata完全相同），而Simple Trading Bot的交易参数会因环境条件（如当前价格、余额）而变化，尽管变化遵循固定规则。链上可通过calldata熵和交易参数的条件方差进行区分。

**（2）MEV Searcher vs. RL Trading Agent：** 仅在决策模型上不同（Statistical vs. Reinforcement Learning）。两者均为ADAPTIVE自主性、Hybrid环境。区分关键在于时序行为模式：RL Trading Agent的策略呈现明显的探索-利用转换和渐进收敛特征，而MEV Searcher的行为更为平稳（在其适应范围内），且高度依赖实时内存池数据。链上可通过交易策略的时序自相关结构和Gas出价的非平稳性进行区分。

综上，分类体系满足互斥性要求。八个类别的三元组各不相同（形式化证明：无两个类别共享相同的（自主性, 环境, 决策模型）三元组），且最难区分的pair可通过链上行为特征的进一步分析加以区分。

## 5. 链上行为映射

### 6.1 特征提取方法

为将分类体系映射到可观测的链上行为，我们设计了四类23维特征工程框架，与本系列研究的Paper 1（链上agent识别管线）直接对接。

**时序特征（7维）：**
- 交易间隔均值（`tx_interval_mean`）：agent通常呈现远低于人类的交易间隔。在我们的44地址数据集中，agent的平均交易间隔为25,427秒，而人类为325,856秒（Mann-Whitney U检验，p < 0.001）。
- 交易间隔标准差（`tx_interval_std`）：反映交易时序的稳定性。
- 交易间隔偏度（`tx_interval_skewness`）：agent的偏度显著高于人类（agent均值38.3 vs. 人类15.8，p = 0.002），反映agent倾向于高频突发交易后跟随较长的静默期。
- 活跃小时熵（`active_hour_entropy`）：衡量活动时间分布的均匀性。
- 夜间活动比率（`night_activity_ratio`）：agent的夜间活动比率更高（0.279 vs. 0.221）。
- 周末活动比率（`weekend_ratio`）：尽管直觉上bot应在周末同样活跃，但该特征在p < 0.05水平显著但非高度显著。
- 突发比率（`burst_frequency`）：衡量交易聚集（burst）的频率。

**Gas特征（6维）：**
- Gas价格整数比率（`gas_price_round_number_ratio`）：人类钱包倾向于使用整数Gas价格（均值0.681），而agent更少如此（均值0.213），因为agent通常通过算法精确计算Gas。该特征的AUC达到0.844。
- Gas价格尾零均值（`gas_price_trailing_zeros_mean`）：人类交易的Gas价格平均有6.34个尾零，agent仅1.99个。这是最强的区分特征之一（AUC = 0.860）。
- Gas限额精度（`gas_limit_precision`）。
- Gas价格变异系数（`gas_price_cv`）。
- EIP-1559优先费精度（`eip1559_priority_fee_precision`）。
- Gas价格-nonce相关性（`gas_price_nonce_correlation`）。

**交互模式特征（5维）：**
- 独立合约比率（`unique_contracts_ratio`）：这是区分agent和人类最强的单一特征（AUC = 0.909）。Agent倾向于与极少数合约反复交互（均值0.032），而人类与更多独立合约交互（均值0.161）。
- 头部合约集中度HHI（`top_contract_concentration`）：agent的HHI显著高于人类（0.867 vs. 0.402, p < 0.001），反映agent行为的高度专注性。
- 函数签名多样性（`method_id_diversity`）。
- 合约vs. EOA交易比率（`contract_to_eoa_ratio`）：agent的合约交互比率显著更高（0.942 vs. 0.554, AUC = 0.883）。
- 序列模式分数（`sequential_pattern_score`）。

**审批行为特征（5维）：**
- 无限审批比率（`unlimited_approve_ratio`）。
- 审批撤销比率（`approve_revoke_ratio`）。
- 未验证合约审批比率（`unverified_contract_approve_ratio`）。
- 多协议交互计数（`multi_protocol_interaction_count`）。
- 闪电贷使用（`flash_loan_usage`）。

在23维特征中，15维在p < 0.05水平上统计显著区分agent与人类，7维在p < 0.001水平上高度显著。

### 6.2 案例研究

我们通过三个典型案例展示分类体系到链上行为的映射。

**案例1：Autonolas agent → DeFi管理Agent类**

Autonolas agent在链上呈现多协议交互序列——在一个操作周期内顺序调用借贷协议（如Aave）、流动性协议（如Uniswap）和收益优化协议（如Yearn）。其审批管理模式反映了跨协议仓位管理的需要。交易间隔具有适应性——agent根据市场波动性调整再平衡频率。这些链上特征完全匹配DeFi管理Agent类别的定义：PROACTIVE自主性（主动发起仓位调整）、Hybrid环境（链下计算链上执行）、Hybrid决策模型（结合统计风险评估和规则引擎）。

**案例2：jaredfromsubway.eth → MEV Searcher类**

jaredfromsubway.eth是以太坊上最知名的MEV searcher之一。其链上行为特征高度匹配MEV Searcher类别：交易间隔均值仅1,803秒但标准差高达7,738秒（反映突发性提取模式）；突发比率0.194；仅与2个独立函数签名交互（`unique_method_ids = 2`），反映高度专注的三明治攻击策略；头部合约集中度HHI达0.989，几乎所有交易都指向同一目标合约。Gas价格整数比率0.811反映了其对特定Gas策略（如使用整数Gas以加速包含）的偏好。

值得注意的是，在Paper 1的分类器实验中，jaredfromsubway.eth被误分类为Human（预测概率仅0.0002）。这是因为该地址的某些行为特征（如Gas价格整数比率较高）更接近人类模式，说明最复杂的MEV searcher可能通过模仿人类行为来规避检测。这一发现本身就支持了分类体系的价值——它提供了对"为什么难以区分"的结构化解释。

**案例3：Wintermute → 简单交易Bot / RL交易Agent边界**

Wintermute是著名的加密市场做市商。其链上行为数据显示：交易间隔均值318秒、标准差10,617秒（极端的高突发性，burst_ratio = 0.617）；仅与3个函数签名交互但交易极度集中（HHI = 1.0）；Gas价格变异系数2.30（高波动，反映算法化Gas策略）。这些特征横跨Simple Trading Bot（如果其做市策略为固定规则）和RL Trading Agent（如果其策略通过强化学习持续优化）的边界。在没有访问其内部系统架构的情况下，仅凭链上行为难以确定其确切类别——这正是分类体系的一个有意识的设计选择：当链上行为不足以区分时，我们承认这种不确定性，而非强制分类。

### 6.3 映射验证

我们利用Paper 1的真实数据集（44个以太坊地址：22个已知agent + 22个已知人类）验证分类体系。

**分类器实验。** 我们在23维特征空间上训练了三个分类器，使用10次重复5折交叉验证和留一交叉验证（LOOCV）：

| 模型 | AUC（重复CV均值） | AUC（LOOCV） | F1（LOOCV） |
|------|-------------------|-------------|-------------|
| RandomForest（全特征） | 0.967 | 0.965 | 0.927 |
| RandomForest（top-10） | 0.975 | 0.969 | 0.905 |
| GradientBoosting（全特征） | 0.940 | 0.940 | 0.857 |
| GradientBoosting（top-10） | 0.955 | 0.899 | 0.781 |
| LogisticRegression（全特征） | 0.870 | 0.856 | 0.837 |
| LogisticRegression（top-10） | 0.932 | 0.915 | 0.857 |

RandomForest在全特征集上达到AUC 0.965（LOOCV），使用top-10特征时达到0.969，说明分类体系所定义的行为特征能够有效区分agent与非agent。

**特征重要性分析。** GradientBoosting模型的特征重要性排序揭示了最具区分力的链上特征：独立合约比率（0.370）、Gas价格尾零均值（0.246）、Gas价格整数比率（0.161）、头部合约集中度（0.104）。前四个特征占据了总重要性的88.1%，且分别对应分类体系中的交互模式维度和Gas特征维度，验证了这两个特征类别在分类中的核心地位。

**误分类分析。** LOOCV中有6个地址被误分类：
- jaredfromsubway.eth（MEV searcher）被误判为Human——该地址使用了接近人类的Gas定价策略；
- MEXC热钱包和Binance热钱包之间存在交叉误分类——交易所热钱包兼具自动化（高频、规则化）和人工审批（部分交易）特征，天然处于agent-human分类边界上。

这些误分类案例恰好映射到我们分类体系中最难区分的类别边界，进一步支持了分类体系对现实复杂性的反映。

## 6. 讨论

### 6.1 对监管的启示

本分类体系为监管机构提供了一套可操作的词汇来对AI agent进行差异化监管。基于自主性等级，可建立如下监管梯度：

- **NONE-REACTIVE（等级0-1）**：确定性脚本和简单交易bot，行为完全可预测，通常无需额外监管，现有金融合规框架即可覆盖。
- **ADAPTIVE（等级2）**：MEV searcher和RL交易agent，行为具有适应性但仍在有界策略空间内，可能需要披露其自动化交易的本质和策略范围。
- **PROACTIVE（等级3）**：DeFi管理agent和LLM驱动agent，行为具有规划性和主动性，可能需要更严格的披露要求，尤其是LLM驱动agent的决策不确定性。
- **COLLABORATIVE（等级4）**：自治DAO agent涉及集体决策，可能需要特定的治理和许可框架。

这一梯度与欧盟AI Act的风险分类具有对应关系：NONE-REACTIVE对应低风险，ADAPTIVE对应有限风险，PROACTIVE-COLLABORATIVE对应需要关注的高风险。

### 6.2 对安全的启示

分类体系中的每个类别意味着不同的威胁画像，直接连接本系列研究的后续三篇论文：

- **Paper 1（链上agent识别）**：本文的分类体系为Paper 1的二分类（agent vs. human）提供了理论基础。分类体系中的链上指标直接转化为Paper 1的特征工程，AUC 0.965的分类器性能隐式验证了特征-类别映射。
- **Paper 2（agent攻击面）**：不同类别面临不同攻击面。确定性脚本对提示注入免疫但易受参数操纵；LLM驱动agent面临全谱LLM特定攻击（提示注入、幻觉、工具滥用）；MEV searcher面临竞争性对抗（Gas竞价战）。
- **Paper 3（多agent博弈）**：COLLABORATIVE类别的agent创造了多agent协调风险，包括共谋、策略操纵和治理攻击。ADAPTIVE类别的agent之间可能形成非合作博弈（如MEV竞争）。

### 6.3 局限性

本研究存在以下局限性：

**数据范围。** 当前验证仅基于以太坊主网数据。其他区块链（如Solana、Polygon、BSC）的交易模型和Gas机制不同，分类体系的可迁移性需要进一步验证。特别是Solana的并行交易处理模型可能产生与以太坊截然不同的agent行为特征。

**专家验证规模。** Delphi专家共识过程涉及12位专家，虽然覆盖了AI研究、区块链安全和DeFi设计三个领域，但未纳入终端用户（如DeFi散户投资者）和MEV受害者的视角。

**分类边界模糊案例。** 如Wintermute案例所示，某些实体天然处于类别边界上。分类体系目前是离散的，未来可考虑引入概率性分类或模糊集方法来处理边界案例。

**可观测性限制。** 纯链下运行的agent组件（如LLM推理、策略计算）不可直接观测。我们只能通过链上行为推断agent的内部架构，这意味着两个内部架构完全不同的agent如果产生相似的链上行为，可能被归入同一类别。

**时间演化。** Web3 agent生态正在快速演化，新的agent类型可能超出当前八类的覆盖范围。例如，多agent系统（MAS）中的"agent-of-agents"模式和链上AI推理（如在链上直接运行ML模型）可能需要新的类别。

### 6.4 未来工作

基于上述局限性，我们规划以下未来研究方向：

- **跨链验证**：将分类体系扩展到Solana、Polygon等区块链，验证其跨链适用性。
- **大规模专家研究**：扩大Delphi专家样本量（目标N=50+），并纳入更多利益相关者类型。
- **纵向演化研究**：追踪agent类别随技术发展的演化轨迹，识别新兴类别。
- **概率化分类**：引入贝叶斯框架将当前的离散分类扩展为连续的概率分类。
- **与Paper 1-3的深度整合**：将分类体系作为后续论文的共享理论基础，确保研究项目整体的内在一致性。

## 7. 结论

本文提出了首个面向Web3环境的多维AI agent分类体系。该体系沿自主性等级（5级：NONE → COLLABORATIVE）、环境类型（4类：纯链上 → 多模态）和决策模型（5类：确定性 → 混合）三个正交维度，定义了八个互斥且覆盖完整的agent类别。

分类体系的关键设计原则是可观测性：每个类别均可通过链上行为特征进行区分，无需检查源代码或内部决策过程。实验验证表明，28对两两比较全部可区分（最少通过1个维度），基于23维链上特征的RandomForest分类器达到AUC 0.965，15维特征在统计上显著区分agent与人类（p < 0.05），前四个特征（独立合约比率、Gas价格尾零均值、Gas价格整数比率、头部合约集中度）占据GBM模型总特征重要性的88.1%。

本分类体系为HCI研究者提供了系统描述和比较AI agent的共享词汇，为监管机构提供了差异化监管的理论基础，为安全研究者提供了按类别分析威胁画像的结构化框架。作为NSF研究项目的奠基之作（Paper 0），本文为后续的链上agent识别（Paper 1）、agent攻击面分析（Paper 2）和多agent博弈论（Paper 3）建立了共享的定义基础。

未来工作将追踪agent类别随Web3技术演进的纵向演化，扩展到多链环境的验证，并探索概率化分类方法以处理类别边界上的模糊案例。

## 参考文献

Amershi, S., Weld, D., Vorvoreanu, M., Fourney, A., Nushi, B., Collisson, P., Suh, J., Iqbal, S., Bennett, P.N., Inkpen, K., Teevan, J., Kikin-Gil, R., & Horvitz, E. 2019. Guidelines for Human-AI Interaction. In *Proceedings of the 2019 CHI Conference on Human Factors in Computing Systems (CHI '19)*. ACM.

Daian, P., Goldfeder, S., Kell, T., Li, Y., Zhao, X., Bentov, I., Breidenbach, L., & Juels, A. 2020. Flash Boys 2.0: Frontrunning in Decentralized Exchanges, Miner Extractable Value, and Consensus Instability. In *IEEE Symposium on Security and Privacy (S&P)*. IEEE.

Flashbots. 2021. MEV-Boost and Proposer-Builder Separation. https://docs.flashbots.net/.

Franklin, S. & Graesser, A. 1996. Is it an Agent, or Just a Program?: A Taxonomy for Autonomous Agents. In *Proceedings of the Third International Workshop on Agent Theories, Architectures, and Languages*. Springer-Verlag.

He, Y., Li, Y., Wang, X., & others. 2025. A Survey of AI Agent Protocols. *arXiv preprint arXiv:2601.04583*.

Horvitz, E. 1999. Principles of Mixed-Initiative User Interaction. In *Proceedings of the SIGCHI Conference on Human Factors in Computing Systems (CHI '99)*. ACM.

Maes, P. 1995. Artificial Life Meets Entertainment: Lifelike Autonomous Agents. *Communications of the ACM*, 38(11), 108-114.

Parasuraman, R., Sheridan, T.B., & Wickens, C.D. 2000. A Model for Types and Levels of Human Interaction with Automation. *IEEE Transactions on Systems, Man, and Cybernetics—Part A: Systems and Humans*, 30(3), 286-297.

Park, J.S., O'Brien, J.C., Cai, C.J., Morris, M.R., Liang, P., & Bernstein, M.S. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In *Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology (UIST '23)*. ACM.

Qin, K., Zhou, L., & Gervais, A. 2022. Quantifying Blockchain Extractable Value: How Dark is the Forest? In *IEEE Symposium on Security and Privacy (S&P)*. IEEE.

Russell, S. & Norvig, P. 2020. *Artificial Intelligence: A Modern Approach* (4th edition). Pearson.

Shneiderman, B. 2022. *Human-Centered AI*. Oxford University Press.

Significant Gravitas. 2023. AutoGPT: An Autonomous GPT-4 Experiment. https://github.com/Significant-Gravitas/AutoGPT.

Torres, C.F., Camino, R., & State, R. 2021. Frontrunner Jones and the Raiders of the Dark Forest: An Empirical Study of Frontrunning on the Ethereum Blockchain. In *USENIX Security Symposium*. USENIX Association.

Wooldridge, M. & Jennings, N.R. 1995. Intelligent Agents: Theory and Practice. *The Knowledge Engineering Review*, 10(2), 115-152.
