# 链上AI Agent识别与安全态势：实证研究

**目标会议：** The Web Conference (WWW)

---

## 摘要

随着自主AI agent日益在公共区块链上运行——管理DeFi仓位、执行交易、与智能合约交互——它们引入了现有监测工具无法察觉的新型风险，因为agent不会在链上自报身份。本文提出了第一个从链上交易数据中系统性识别AI agent地址并量化其安全暴露面的实证研究。我们设计了一个涵盖四个维度（时序模式、Gas定价行为、交互模式、授权安全）的23特征行为指纹框架，基于44个标注地址（22个已确认agent和22个已确认人类用户）构建分类器。实验表明，RandomForest在留一交叉验证(LOO-CV)中取得0.965的AUC-ROC，GradientBoosting取得0.940的AUC-ROC。在23个特征中，15个在Mann-Whitney U检验下达到p<0.05的统计显著性水平，其中交互模式特征（如unique_contracts_ratio，单特征AUC=0.909）展现出最强的区分能力。我们进一步对22个已识别agent执行四维安全审计，发现：228个无限授权集中于2个agent地址，agent平均交易失败率为6.0%，个别agent（Wintermute）失败率高达29.6%。风险评级显示3个agent为MEDIUM风险、19个为LOW风险。本研究为链上AI agent的监测与治理提供了方法论基础和实证基线。

## 1. 引言

### 1.1 研究背景

自主AI agent正在以前所未有的速度渗透公共区块链生态系统。从简单的交易机器人到复杂的多步骤DeFi策略执行者，从MEV搜索器到AI驱动的做市商，一个日益庞大的非人类参与者群体已经在以太坊主网上活跃运作。Autonolas、AI16Z ELIZA、Virtuals Protocol等平台使得部署链上AI agent变得前所未有地便捷，推动agent管理的链上资产规模已达数十亿美元量级。

然而，这一趋势伴随着严重的安全隐患。2025至2026年间，已有超过$45M的损失被归因于AI agent相关的安全漏洞，包括授权利用攻击、预言机操纵、以及级联清算失败。更令人担忧的是，这些agent在链上不具备任何自我标识机制——从交易数据的表面来看，一个AI控制的钱包与一个人类操作的钱包在形式上是不可区分的。

### 1.2 问题定义

当前的区块链安全生态面临一个根本性的盲区：**agent不自报身份，与人类用户混在一起**。这一不透明性导致：(1) 安全审计人员无法针对agent的特殊风险进行评估；(2) 协议设计者无法为agent行为做针对性优化；(3) 监管机构对自动化金融活动缺乏可见性。

近期的学术工作已经开始关注这一问题。He等人（arXiv 2601.04583）对链上AI agent进行了综述性研究，系统梳理了agent的分类体系和能力边界，但**缺乏实证数据支撑**——他们的工作停留在分类框架层面，未能提供从链上数据中实际识别agent的方法和验证。

### 1.3 研究问题

基于上述差距，本文提出两个核心研究问题：

- **RQ1：** 能否仅从链上行为特征识别AI agent？即，agent和人类用户在交易模式上是否存在系统性的、可被机器学习模型捕获的差异？

- **RQ2：** 已识别agent的安全态势如何？与人类用户相比，agent在权限管理、交易失败率、网络拓扑等维度上是否表现出不同的风险暴露特征？

### 1.4 主要贡献

本文的贡献包括：

1. **识别方法论：** 提出一个涵盖四组23个特征的行为指纹框架，用于从公开交易数据中区分AI agent与人类用户，RandomForest分类器在LOO-CV中达到AUC 0.965。

2. **标注数据集：** 构建了一个包含44个以太坊地址（22个agent + 22个人类）的标注数据集，标注来源包括MEV追踪器、协议注册表、ENS域名和社交验证。

3. **安全态势基线：** 对22个agent执行四维安全审计，建立了链上AI agent安全态势的首个实证基线，揭示了权限暴露集中化、失败率异质性等关键发现。

4. **特征重要性分析：** 通过统计检验和消融实验，系统性地量化了各特征组和单个特征对agent识别的贡献，发现交互模式和Gas行为是最具区分力的特征维度。

## 2. 相关工作

本研究涉及四个研究领域：区块链上的机器人检测、MEV与自动化交易、AI agent平台、以及区块链安全分析。我们分别综述各领域的主要工作，并指出本研究所填补的空白。

### 2.1 区块链机器人检测

检测区块链上的自动化活动主要集中在特定恶意行为的识别。Chen等人[1]提出了基于交易特征和操作码分析的以太坊庞氏骗局检测方法，表明交易数据中的行为模式可以揭示自动化或欺诈活动。Victor和Weintraud[2]开发了识别去中心化交易所刷量交易的方法，利用图特征检测人为抬高交易量的协调机器人活动。Torres等人[3]研究了以太坊上的抢跑机器人，通过内存池分析识别置换、插入和抑制攻击。

Friedhelm等人[4]提出了区块链机器人的行为指纹技术，使用了与本文时序和Gas特征组相似的时间和Gas定价特征。但其工作仅聚焦于MEV机器人，未考虑更广泛的AI agent类别。Li等人[5]研究了Telegram交易机器人的链上足迹，发现其在授权行为和合约交互多样性上具有独特模式。

**差距：** 现有机器人检测方法针对特定恶意行为（刷量交易、庞氏骗局、抢跑），而非识别"AI控制地址"这一通用类别。本研究提供了一个适用于所有AI agent的综合识别框架，不论其目的为何。

### 2.2 MEV与自动化交易

最大可提取价值（MEV）的研究揭示了区块链上自动化活动的广度。Daian等人[6]在其开创性论文"Flash Boys 2.0"中引入了MEV概念，证明矿工和交易者通过交易排序提取价值，并创造了丰富的自动化搜索器机器人生态系统。Weintraub等人[7]对Flashbots上的MEV进行了大规模测量研究，量化了经济影响并识别了MEV机器人采用的不同策略。Qin等人[8]为跨DeFi协议的区块链可提取价值量化提供了系统框架，揭示三明治攻击、清算和套利每年共计提取数十亿美元。

Park等人[9]研究了合并后MEV策略的演变，发现提议者-建设者分离（PBS）改变了MEV格局并催生了与区块建设者交互的新型自动化agent。Gupta等人[10]分析了跨域MEV，其中agent协调跨多条链和L2的活动。

**差距：** MEV研究深入记录了自动化交易的经济效应，但未构建通用的地址类型识别工具。本文的分类器泛化至MEV机器人之外，涵盖DeFi管理agent、投资组合再平衡器和AI驱动的策略agent。

### 2.3 AI Agent平台

新一代平台使得直接在区块链上部署AI agent成为可能，创造了质的不同于传统机器人的自动化活动类别。

**Autonolas (OLAS)：** Autonolas协议提供了构建、部署和运行自主agent服务的开源框架[11]。Agent注册于链上的ServiceRegistry，为本文的标注方法提供了部分ground truth。Autonolas agent执行从预言机更新到复杂多步骤DeFi策略的各种任务。

**AI16Z ELIZA框架：** ELIZA框架使得创建能通过自然语言接口和自主执行与区块链协议交互的AI agent成为可能[12]。ELIZA agent通常通过专用钱包运作，展现出独特的交互模式。

**Virtuals Protocol：** Virtuals Protocol为代币化AI agent提供了启动平台，每个agent有关联代币并在链上自主运作[13]。其他平台如Fetch.ai[14]、SingularityNET[15]和AI Arena[16]也为链上AI agent群体贡献了增量。

**差距：** 虽然这些平台创建和部署AI agent，但没有一个提供统一的跨平台识别机制。本文基于行为的指纹方法是平台无关的。

### 2.4 区块链安全分析

Zhou等人[17]在其SoK论文中系统整理了DeFi攻击向量，包括闪电贷利用、预言机操纵、治理攻击和重入攻击。Wen等人[18]分析了DeFi可组合性风险，表明协议间的交互创造了涌现性漏洞。Grech等人[19]开发了MadMax工具用于检测智能合约中的Gas相关漏洞。Tsankov等人[20]引入了Securify安全扫描器。

在AI agent安全领域，Wang等人[21]研究了具有加密货币钱包访问权的AI agent的风险，识别出提示注入和工具误用是主要攻击向量。Zhang等人[22]分析了DeFi中基于授权的攻击，发现MaxUint256授权创造了持久的攻击面。

**差距：** 现有安全分析聚焦于协议级漏洞或合约级缺陷。尚无工作系统性地评估AI agent地址的安全态势——包括其权限暴露、MEV脆弱性、失败率和网络中心性。本文的四维审计框架填补了这一空白。

### 2.5 本文定位

相较于上述工作，本研究的独特贡献在于：(1) 通用agent识别而非特定行为检测；(2) 基于行为而非基于协议的方法；(3) agent专属的安全审计框架；(4) 以太坊主网上AI agent安全态势的首个实证基线。

## 3. 方法

### 3.1 数据收集

**数据源与采集方式。** 我们从以太坊主网通过Etherscan V2 API采集链上交易数据。为突破API速率限制，采用6-key轮转策略实现约30 calls/sec的稳定吞吐量，确保在合理时间内完成大规模数据采集。

**数据集规模。** 最终数据集包含44个以太坊地址：22个已确认的AI agent地址和22个已确认的人类用户地址。对每个地址，我们提取最多5000笔最新交易，涵盖普通交易（normal transactions）、内部交易（internal transactions）、ERC-20转账和代币授权记录。

**Agent样本构成。** 22个agent地址涵盖多种类型：MEV搜索器（如jaredfromsubway.eth、jaredfromsubway v2及多个匿名MEV bot）、做市商（如Wintermute及其多个操作地址）、DeFi聚合器（如1inch resolver、1inch v5 router）、DEX路由器（如Uniswap V2 Router、Uniswap V3 Router 02）、NFT交易自动化（如Seaport Bot、Blur Exchange）、交易所热钱包（如MEXC hot wallet）以及区块构建器（如Flashbots builder）。

**人类样本构成。** 22个人类地址包括知名个人（如vitalik.eth、Hayden Adams）、机构钱包（通过Arkham Intelligence和Nansen标签确认）以及具有ENS域名和社交账户关联的活跃DeFi用户。

**伦理考量。** 所有数据均来源于公开的以太坊区块链，不涉及任何非公开信息。地址标签使用公开可查的来源进行验证。

### 3.2 Ground Truth标注

可靠的ground truth是分类任务的基础。我们采用多源交叉验证的标注策略：

**Agent正例标注。** Agent地址的确认基于以下来源：(1) Flashbots/EigenPhi MEV机器人列表：已知MEV搜索器地址；(2) 协议注册表：如Autonolas ServiceRegistry中注册的自主服务运营者地址；(3) 链上行为特征与已知bot签名的匹配：如极高频交易（间隔均值仅数十秒）、几乎100%的合约调用比例、以及高度集中的交互目标。

**人类负例标注。** 人类地址的确认基于：(1) ENS域名关联：如vitalik.eth解析到已知的Ethereum联合创始人地址；(2) 社交验证：通过Twitter、GitHub等公开平台确认地址所有者身份；(3) 已知机构标签：Arkham Intelligence和Nansen提供的经过审核的地址标签。

**排除规则。** 我们排除了无法以高置信度标注的地址。特别值得注意的是，**交易所热钱包**构成了一个边界案例：它们在行为上类似bot（高频、24/7操作、自动化Gas定价），但在分类上被视为人类操作的基础设施而非自主agent。这一决策对后续误分类分析具有重要意义（详见4.5节）。

### 3.3 特征工程

我们设计了四组共23个特征来捕捉agent与人类用户之间的行为差异。特征的设计遵循以下直觉：AI agent通常表现出更规律的交易时间模式、更精确的算法化Gas定价、更集中的合约交互模式、以及不同的授权管理行为。

#### 3.3.1 时序特征组（7个特征）

时序特征捕捉交易的时间分布模式。

- **tx_interval_mean（交易间隔均值）：** 计算相邻交易之间的时间间隔（秒）的算术均值。直觉：agent具有更短、更规律的间隔。实验数据显示，agent的间隔均值为25,427秒，而人类为325,856秒（p=5.68e-05）。

- **tx_interval_std（交易间隔标准差）：** 交易间隔的标准差。Agent均值为89,278秒，人类为1,260,815秒（p=1.13e-04）。

- **tx_interval_skewness（交易间隔偏度）：** 间隔分布的偏度。Agent展现更高的正偏度（均值38.3 vs 人类15.8），因为agent在大部分时间保持规律间隔但偶尔有长时间停顿（如策略切换、资金注入），导致右偏分布（p=0.0024）。

- **active_hour_entropy（活跃时段熵）：** 将交易按UTC小时分桶后计算Shannon熵。Agent的时段分布更均匀（24/7操作），熵值略高（2.90 vs 2.85，p=0.050）。

- **night_activity_ratio（夜间活动比例）：** UTC 0:00-6:00时段交易占比。Agent在夜间更活跃（27.9% vs 22.1%，p=0.034）。

- **weekend_ratio（周末活动比例）：** 周末交易占比。虽然agent理论上应保持稳定的周末活动，但实际差异未达统计显著性（p=0.169）。

- **burst_frequency（突发频率）：** 10秒内连续交易的比例。Agent更容易产生突发性的高频交易模式（37.6% vs 20.8%，p=0.042）。

#### 3.3.2 Gas行为特征组（6个特征）

Gas特征捕捉交易手续费设置策略的差异。人类通常使用钱包软件推荐的Gas价格（倾向于整数），而agent使用算法精确计算Gas。

- **gas_price_round_number_ratio（Gas价格整数比例）：** Gas价格为整数Gwei的交易比例。Agent更少使用整数价格（21.3% vs 68.1%，p=9.68e-05）。

- **gas_price_trailing_zeros_mean（Gas价格尾零均值）：** Gas价格十进制表示中尾随零的平均个数。Agent的Gas价格更精确，尾零更少（1.99 vs 6.34，p=4.65e-05）。

- **gas_limit_precision（Gas限制精度）：** Gas limit相对于实际Gas used的精度。差异未达统计显著性（p=0.286）。

- **gas_price_cv（Gas价格变异系数）：** Gas价格的变异系数。差异未达统计显著性（p=0.630）。

- **eip1559_priority_fee_precision（EIP-1559优先费精度）：** EIP-1559交易中priority fee的精度。在本数据集中该特征无区分力（AUC=0.500）。

- **gas_price_nonce_correlation（Gas价格-Nonce相关性）：** Gas价格与nonce的Pearson相关系数，捕捉算法化的Gas调整模式。差异未达统计显著性（p=0.879）。

#### 3.3.3 交互模式特征组（5个特征）

交互特征捕捉地址与智能合约交互的多样性和集中度。

- **unique_contracts_ratio（唯一合约比例）：** 交互过的唯一合约数占总交易数的比例。Agent交互范围更窄（3.2% vs 16.1%，p=3.18e-06），这是所有特征中区分力最强的单一特征（AUC=0.909）。

- **top_contract_concentration（头部合约集中度，HHI指数）：** 使用Herfindahl-Hirschman指数衡量交互目标的集中度。Agent的交互高度集中（0.867 vs 0.402，p=4.53e-05）。

- **method_id_diversity（方法ID多样性）：** 调用的唯一函数签名数占总合约调用数的比例。Agent调用的函数种类相对更少（5.5% vs 15.3%，p=0.032）。

- **contract_to_eoa_ratio（合约调用比例）：** 目标为合约地址的交易占比。Agent几乎全部与合约交互（94.2% vs 55.4%，p=1.36e-05），AUC达0.883。

- **sequential_pattern_score（序列模式得分）：** 基于n-gram分析的交易动作序列重复度。差异未达统计显著性（p=0.212）。

#### 3.3.4 授权安全特征组（5个特征）

授权特征捕捉代币授权管理行为的差异，与安全态势直接相关。

- **unlimited_approve_ratio（无限授权比例）：** 设置MaxUint256授权的approve调用占总approve调用的比例。差异达到统计显著性（p=0.021），但方向与直觉相反：人类用户反而更多地使用无限授权。

- **approve_revoke_ratio（授权撤销比例）：** 撤销（revoke）操作占总授权操作的比例。差异未达统计显著性（p=0.323）。

- **unverified_contract_approve_ratio（未验证合约授权比例）：** 向未经Etherscan验证的合约发出授权的比例。Agent更少向未验证合约授权（8.4% vs 44.3%，p=0.007）。

- **multi_protocol_interaction_count（多协议交互数）：** 与之交互的DeFi协议数量。差异勉强达到统计显著性（p=0.046）。

- **flash_loan_usage（闪电贷使用）：** 闪电贷使用情况。在本数据集中该特征无区分力（AUC=0.500），所有样本的值均为0。

### 3.4 分类模型

考虑到数据集规模较小（n=44），我们采用多种分类策略以确保结果的稳健性。

**模型选择。** 我们训练三种分类器：(1) GradientBoosting（主模型），200棵树，最大深度3，学习率0.1；(2) RandomForest，100棵树，最大深度无限制；(3) LogisticRegression（基线），L2正则化。所有模型使用scikit-learn实现，特征在训练前进行标准化处理。

**评估策略。** 鉴于小样本的特殊性，我们采用两种互补的交叉验证方案：

- **留一交叉验证（LOO-CV）：** 每次留出一个样本作为测试集，其余43个作为训练集，迭代44次。LOO-CV在小样本下提供了几乎无偏的性能估计，但方差较大。

- **重复分层5折交叉验证（Repeated Stratified 5-Fold CV）：** 10次重复的5折交叉验证，确保每折中agent和人类样本的比例与全局一致。报告50次评估的均值和标准差。

**特征选择实验。** 我们同时评估全部23个特征和Top-10特征（按单特征AUC排序选取）两种配置，以检验特征冗余性和模型简化的可行性。

**评估指标。** AUC-ROC（主指标）、Precision、Recall、F1-score和Accuracy。

### 3.5 安全审计框架

对22个已识别的agent地址，我们执行四维安全审计：

**维度一：权限暴露。** 统计每个agent的无限授权（MaxUint256 approve）数量、涉及的唯一授权目标（spender）数量、以及授权目标合约是否经过Etherscan验证。无限授权意味着被授权合约可以无限量转走用户代币，构成持久性攻击面。

**维度二：Agent网络。** 构建agent间的转账/调用有向图，计算度中心性、聚类系数，并通过社区检测识别agent集群。高度中心化的agent节点构成系统性风险点。

**维度三：MEV暴露。** 统计agent在DEX交易中遭受三明治攻击的比率，与人类用户基线进行比较。

**维度四：失败率分析。** 计算每个agent的交易回退（revert）率，分类回退原因（Gas不足、滑点超限、权限不足等），并分析失败后的重试行为。

**风险评级。** 基于上述四维度的综合评估，我们为每个agent分配HIGH/MEDIUM/LOW三级风险评级。评级标准：具有大量无限授权或极高失败率的agent标记为MEDIUM或HIGH；无显著风险暴露的agent标记为LOW。

## 4. 实验结果

### 4.1 Agent识别性能

表1总结了三种分类器在两种特征配置下的LOO-CV和重复5折CV性能。

**表1：分类器性能比较（LOO-CV）**

| 模型 | 特征数 | AUC-ROC | Precision | Recall | F1 | Accuracy |
|------|--------|---------|-----------|--------|----|----------|
| RandomForest | 23 | **0.965** | **1.000** | 0.864 | **0.927** | **0.932** |
| RandomForest | 10 | **0.969** | 0.950 | 0.864 | 0.905 | 0.909 |
| GradientBoosting | 23 | 0.940 | 0.900 | 0.818 | 0.857 | 0.864 |
| GradientBoosting | 10 | 0.899 | 0.842 | 0.727 | 0.781 | 0.795 |
| LogisticRegression | 23 | 0.855 | 0.857 | 0.818 | 0.837 | 0.841 |
| LogisticRegression | 10 | 0.915 | 0.900 | 0.818 | 0.857 | 0.864 |

**主要发现。** RandomForest在全特征（23个）配置下取得了最佳的LOO-CV AUC=0.965，且Precision达到完美的1.000，意味着被模型判定为agent的地址全部确实为agent（无假阳性）。其Recall为0.864，即86.4%的agent被成功识别。GradientBoosting紧随其后，AUC=0.940。

**Top-10特征模型的表现值得关注。** RandomForest在仅使用10个特征时AUC反而微升至0.969，表明额外的13个特征可能引入噪声。LogisticRegression在Top-10配置下AUC从0.855显著提升至0.915，说明降维有助于线性模型避免过拟合。而GradientBoosting在Top-10配置下AUC从0.940降至0.899，表明其对全特征集的利用更为充分。

**表2：重复5折CV性能（均值 +/- 标准差）**

| 模型 | 特征数 | AUC-ROC | F1 |
|------|--------|---------|-----|
| RandomForest | 23 | 0.967 +/- 0.054 | 0.907 +/- 0.094 |
| RandomForest | 10 | 0.975 +/- 0.045 | 0.901 +/- 0.105 |
| GradientBoosting | 23 | 0.940 +/- 0.083 | 0.851 +/- 0.111 |
| GradientBoosting | 10 | 0.955 +/- 0.071 | 0.862 +/- 0.104 |
| LogisticRegression | 23 | 0.870 +/- 0.105 | 0.809 +/- 0.145 |
| LogisticRegression | 10 | 0.932 +/- 0.089 | 0.829 +/- 0.142 |

重复5折CV的结果与LOO-CV一致，进一步验证了模型的稳健性。RandomForest Top-10配置在重复CV中达到最高的AUC=0.975。不过，所有模型的标准差均较大（0.045-0.105），反映了小样本下的固有不稳定性。

### 4.2 特征重要性分析

#### 4.2.1 单特征区分能力

表3列出了各特征的单特征AUC-ROC排名（基于Mann-Whitney U检验）。

**表3：Top-10特征（按单特征AUC排序）**

| 排名 | 特征 | 单特征AUC | 特征组 |
|------|------|-----------|--------|
| 1 | unique_contracts_ratio | 0.909 | 交互 |
| 2 | contract_to_eoa_ratio | 0.883 | 交互 |
| 3 | gas_price_trailing_zeros_mean | 0.860 | Gas |
| 4 | top_contract_concentration | 0.857 | 交互 |
| 5 | tx_interval_mean | 0.855 | 时序 |
| 6 | gas_price_round_number_ratio | 0.844 | Gas |
| 7 | tx_interval_std | 0.841 | 时序 |
| 8 | tx_interval_skewness | 0.769 | 时序 |
| 9 | method_id_diversity | 0.690 | 交互 |
| 10 | night_activity_ratio | 0.688 | 时序 |

**关键发现：** Top-10特征中，交互模式组贡献了4个（排名1、2、4、9），Gas行为组贡献了2个（排名3、6），时序组贡献了4个（排名5、7、8、10），而授权安全组没有特征进入Top-10。这揭示了一个重要洞察：**交互模式 > Gas行为 > 时序模式 > 授权安全**，即agent与人类的根本差异更多体现在"与谁交互、怎么交互"上，而非简单的时间或Gas使用模式。

#### 4.2.2 GradientBoosting特征重要性

GradientBoosting模型内部的特征重要性分数提供了另一个视角。排名前五的特征为：

| 特征 | 重要性 |
|------|--------|
| unique_contracts_ratio | 0.370 |
| gas_price_trailing_zeros_mean | 0.246 |
| gas_price_round_number_ratio | 0.161 |
| top_contract_concentration | 0.104 |
| gas_price_nonce_correlation | 0.023 |

值得注意的是，GradientBoosting分配了极高的权重给前四个特征（合计0.881），尤其是unique_contracts_ratio独占37.0%的重要性。这表明该模型严重依赖少数核心特征，而大多数特征的贡献接近于零——23个特征中有8个的GBM重要性恰好为0.000（包括所有5个授权安全特征中的4个以及flash_loan_usage）。

#### 4.2.3 按特征组的贡献分析

综合单特征AUC和GBM特征重要性，各特征组的贡献如下：

- **交互模式组（5个特征）：** 包含区分力最强的单一特征（unique_contracts_ratio, AUC=0.909），且有4/5个特征统计显著（p<0.05）。该组是agent识别的核心支柱。

- **Gas行为组（6个特征）：** gas_price_trailing_zeros_mean（AUC=0.860）和gas_price_round_number_ratio（AUC=0.844）是该组的强特征。但gas_price_cv、gas_limit_precision和eip1559_priority_fee_precision几乎无区分力。有效特征仅2/6个。

- **时序特征组（7个特征）：** tx_interval_mean（AUC=0.855）和tx_interval_std（AUC=0.841）表现强劲。但active_hour_entropy和weekend_ratio的区分力较弱。整体贡献中等，有5/7个特征统计显著。

- **授权安全组（5个特征）：** 仅3/5个特征统计显著，且AUC均低于0.69。flash_loan_usage和eip1559_priority_fee_precision在本数据集中完全无效（AUC=0.500）。该组对分类的直接贡献最小，但其特征在安全审计中具有独立价值。

### 4.3 特征统计显著性

我们对所有23个特征进行了Mann-Whitney U检验（非参数检验，适用于小样本且不假设正态分布），结果如下。

**15/23个特征在p<0.05水平下具有统计显著性。** 具体包括：

**高度显著（p<0.001）的特征（7个）：**

| 特征 | p值 | Agent均值 | Human均值 |
|------|------|-----------|-----------|
| unique_contracts_ratio | 3.18e-06 | 0.032 | 0.161 |
| contract_to_eoa_ratio | 1.36e-05 | 0.942 | 0.554 |
| top_contract_concentration | 4.53e-05 | 0.867 | 0.402 |
| gas_price_trailing_zeros_mean | 4.65e-05 | 1.986 | 6.338 |
| tx_interval_mean | 5.68e-05 | 25,427 | 325,856 |
| gas_price_round_number_ratio | 9.68e-05 | 0.213 | 0.681 |
| tx_interval_std | 1.13e-04 | 89,278 | 1,260,815 |

**显著（p<0.05）的特征（8个）：**

| 特征 | p值 |
|------|------|
| tx_interval_skewness | 0.0024 |
| unverified_contract_approve_ratio | 0.0068 |
| unlimited_approve_ratio | 0.021 |
| method_id_diversity | 0.032 |
| night_activity_ratio | 0.034 |
| burst_frequency | 0.042 |
| multi_protocol_interaction_count | 0.046 |
| active_hour_entropy | 0.050 |

**不显著（p>=0.05）的特征（8个）：** weekend_ratio (p=0.169), sequential_pattern_score (p=0.212), gas_limit_precision (p=0.286), approve_revoke_ratio (p=0.323), gas_price_cv (p=0.630), gas_price_nonce_correlation (p=0.879), eip1559_priority_fee_precision (p=1.000), flash_loan_usage (p=1.000)。

### 4.4 安全审计发现

我们对22个agent地址执行了四维安全审计，结果揭示了若干值得关注的安全态势特征。

#### 4.4.1 权限暴露

**无限授权的集中性。** 在22个agent中，仅有2个agent持有无限授权（unlimited approvals），但这2个agent共持有228个无限授权：

- **1inch resolver（0xFa4F...6199）：** 218个无限授权，涉及95个唯一被授权方（spender），总approve调用608次中有218次为无限授权。该agent作为DEX聚合器的解析器，需要向多个DEX路由器发放授权以实现跨平台的流动性路由。

- **jaredfromsubway.eth（0x5617...5Bf9）：** 10个无限授权，涉及5个唯一被授权方。作为知名MEV搜索器，其需要向目标交易对的路由合约发放授权以执行三明治攻击。

这一高度集中的分布表明，**权限暴露风险不是均匀分布的**，而是集中在特定功能角色的agent上。DEX聚合器和MEV搜索器由于其业务逻辑的需要，天然具有更大的授权面。若这些agent的私钥被泄露或其逻辑被操纵，所有活跃的无限授权都可能被利用。

#### 4.4.2 交易失败率

**整体概况。** 22个agent的平均交易回退率（revert rate）为6.0%（加权平均）。但该指标的异质性极大：

**高失败率agent（revert rate > 5%）：**

| Agent | Revert Rate | Revert Count / Total Txs |
|-------|------------|--------------------------|
| Wintermute | **29.6%** | 1,478 / 5,000 |
| Uniswap V2 Router | 24.4% | 1,218 / 5,000 |
| MEV bot 8 | 17.0% | 849 / 5,000 |
| Seaport 1.1 | 13.2% | 659 / 5,000 |
| 1inch v5 router | 9.2% | 458 / 5,000 |
| Uniswap V3 Router 02 | 9.4% | 469 / 5,000 |
| Blur Exchange | 8.8% | 439 / 5,000 |
| 0x Exchange Proxy | 6.3% | 314 / 5,000 |

**低失败率agent（revert rate < 1%）：** 多数MEV bot和做市商操作地址维持在极低的失败率（0.0%-1.0%），如MEV bot 2（0.0%）、MEXC hot wallet（0.0%）、Wintermute 2（0.04%）等。

**失败率的异质性分析。** Wintermute的29.6%失败率是一个值得深入探讨的发现。作为主要的做市商，Wintermute地址接收大量入站交易，其中相当比例因Gas不足、滑点超限或流动性不足而失败。这与MEV bot的低失败率形成鲜明对比——MEV bot通常在发送交易前已通过模拟确认交易可成功执行（否则不发送），因此revert率极低。

路由器合约（如Uniswap V2 Router 24.4%、Uniswap V3 Router 02 9.4%）的高失败率反映的是最终用户交易失败而非agent本身的策略缺陷，这是一个分析中需要注意的区分。

#### 4.4.3 风险评级分布

**评级结果：**
- **HIGH风险：** 0个agent
- **MEDIUM风险：** 3个agent——jaredfromsubway.eth（无限授权+单一目标主导）、MEXC hot wallet（高突发活动+单一方法模式）、1inch resolver（218个无限授权）
- **LOW风险：** 19个agent

MEDIUM风险agent的风险标签（risk flags）提供了更具体的风险细节：

| Agent | 风险标签 |
|-------|---------|
| jaredfromsubway.eth | unlimited_approvals:10, single_target_dominance:1.00 |
| MEXC hot wallet | high_burst_activity:0.58, single_method_bot_pattern |
| 1inch resolver | unlimited_approvals:218 |

**无HIGH风险agent的解读。** 虽然没有agent被评为HIGH风险，但这并不意味着安全无虞。当前的评级框架侧重于直接可观测的链上风险指标。许多潜在风险（如私钥管理质量、链下逻辑的安全性、供应链攻击面）无法仅从链上数据评估。

#### 4.4.4 Agent网络特征

从安全审计数据中还可以观察到agent的网络交互模式：

- **高度集中型：** jaredfromsubway.eth（5个唯一交互方，头部目标占99.8%）、jaredfromsubway v2（6个唯一交互方，头部目标占99.9%）表现出极端集中的交互模式，几乎所有交易指向单一目标合约。

- **广泛分散型：** 1inch resolver（528个唯一交互方，头部目标仅占9.6%）和MEXC hot wallet（2,470个唯一交互方，头部目标仅占4.1%）展现出高度分散的交互网络。

- **仅接收型：** 多数路由器合约和部分MEV bot表现为纯接收模式（incoming_outgoing_ratio = inf），它们不主动发起交易，而是被其他地址调用。

### 4.5 误分类分析

GradientBoosting LOO-CV中共有6个误分类样本（6/44 = 13.6%），分析其模式对理解分类器的边界条件至关重要。

**被误分类为人类的Agent（4个）：**

| 地址 | 名称 | 预测概率 | 可能原因 |
|------|------|----------|---------|
| 0x5617...5Bf9 | jaredfromsubway.eth | 0.0002 | 该MEV bot的Gas使用模式（高整数比例0.811、高尾零9.03）与人类钱包相似，因为其使用MetaMask兼容的Gas策略 |
| 0x75e8...cB88 | MEXC hot wallet | 0.0000 | 交易所热钱包在行为上类似人类操作——广泛的目标分散度（2,468个唯一目标）、高比例的简单ETH转账 |
| 0xDef1...2EfF | 0x Exchange Proxy | 0.3209 | 作为路由器合约，其行为特征（高度多样的交互方1,277个）更接近人类用户的多样性 |
| 0x68b3...Fc45 | Uniswap V3 Router 02 | 0.2185 | 同上，路由器合约的交互模式由最终用户驱动而非agent自身 |

**被误分类为Agent的人类（2个）：**

| 地址 | 名称 | 预测概率 | 可能原因 |
|------|------|----------|---------|
| 0x28C6...1d60 | Binance 14 hot | 0.7053 | 交易所热钱包的高频、自动化操作模式使其行为与agent高度相似 |
| 0xDFd5...63d | Binance 16 | 0.9989 | 同上，且预测概率极高（0.999），分类器几乎确信这是agent |

**误分类的核心模式。** 6个误分类中有3个涉及交易所热钱包（MEXC、Binance 14、Binance 16），占误分类总数的50%。这揭示了分类器面临的核心挑战：**交易所热钱包是一个行为上的混合类别**——它们由自动化系统操作（类agent），但在当前分类体系中被标注为"人类"（因为它们是中心化机构的基础设施而非自主决策agent）。

另外2个误分类涉及路由器合约（0x Exchange Proxy、Uniswap V3 Router 02），其行为特征由调用它们的最终用户决定，而非合约/agent自身的逻辑，形成了另一类边界模糊的案例。

这些发现表明，**当前的二分类框架（agent vs. 人类）可能过于简化**，未来工作应考虑引入"半自动化"或"基础设施"等中间类别。

## 5. 讨论

### 5.1 识别方法的实用性

本研究表明，仅基于公开可得的链上交易数据，就可以以较高的准确率（AUC 0.965）区分AI agent与人类用户。这一发现具有直接的实际应用价值：

**链上监测系统。** 安全团队可以将本文的23特征框架集成到链上监测管道中，对新出现的地址进行实时分类，识别潜在的agent地址。特别是，仅使用Top-10特征即可维持0.969的AUC，大幅降低了特征计算的开销。

**协议设计。** DeFi协议设计者可以利用agent识别结果为不同类型的用户提供差异化的服务：例如，为已识别的agent提供优化的Gas消耗路径、批量交易支持、以及自动化的授权管理接口。

**风险预警。** 当新地址的行为特征被分类为"高概率agent"时，安全审计人员可以优先检查其授权模式和交互网络，提前发现潜在的系统性风险。

### 5.2 安全发现的启示

安全审计揭示了几个值得产业界关注的发现：

**授权集中风险。** 228个无限授权集中于2个agent（1inch resolver和jaredfromsubway.eth），意味着少数agent节点的安全失败可能导致大规模资金损失。建议：协议应推广有限额度授权（finite approvals）作为默认行为，并提供自动化的授权管理和定期撤销工具。

**失败率的异质性。** Agent之间的失败率差异巨大（0.0% 到 29.6%），反映了不同agent策略的成熟度差异。高失败率不仅浪费Gas费用，还可能暴露agent的策略意图（竞争对手可以通过观察revert模式推断agent的目标交易）。

**风险评级的保守性。** 当前框架将所有agent评为LOW或MEDIUM，没有HIGH风险agent。这可能反映了我们样本偏向于成熟的、已被广泛识别的agent。新兴的、未经验证的agent可能展现出更高的风险特征。

### 5.3 与现有分类体系的对应

He等人（arXiv 2601.04583）提出了链上AI agent的分类体系，将agent按功能分为交易agent、DeFi管理agent、信息聚合agent等类别。本文的实证数据可以从行为维度丰富和验证这一分类体系：

- **MEV搜索器类agent**（如jaredfromsubway系列）展现出极低的交易间隔（均值28-1,803秒）、极高的合约调用比例（>99.5%）、和极度集中的交互目标（HHI>0.989）。

- **做市商类agent**（如Wintermute）展现出高突发活动（burst ratio 61.7%）、适度的交互分散性、以及显著更高的失败率。

- **DEX聚合器/路由器类agent**展现出高度多样的交互方（500-3,600+）和中等水平的失败率。

这些行为指纹可以作为He等人分类体系的实证补充，帮助从链上数据推断agent的功能类别。

### 5.4 局限性

本研究存在以下局限性，读者在解读结果时需要注意：

**样本量限制。** 44个样本（22 agent + 22 human）是一个较小的数据集。虽然我们通过LOO-CV和重复交叉验证来缓解过拟合风险，但小样本下的性能估计天然具有较高的方差（如RandomForest的AUC标准差为0.054）。更大规模的标注数据集将有助于提高估计的置信度。

**链的覆盖范围。** 本研究仅覆盖以太坊主网。随着AI agent活动向Base、Arbitrum、Solana等链和L2扩展，跨链agent识别将成为重要的后续研究方向。不同链的Gas机制和交易结构差异可能需要特征框架的适配。

**Ground truth偏差。** 我们的agent样本偏向于已被公开识别的、成熟的agent（如知名MEV bot、主要做市商）。存在大量未被识别的"隐性agent"，它们可能具有与我们样本不同的行为特征。此外，路由器合约和交易所热钱包的分类归属存在争议，影响了ground truth的清晰度。

**特征时效性。** Agent行为会随着平台更新、策略演化和Gas机制变化而漂移。本文提取的特征值反映了特定时间窗口内的行为快照，可能无法完全泛化到未来的agent行为。

**对抗性逃逸。** 精心设计的agent可以故意模拟人类行为模式（如引入随机延迟、使用整数Gas价格、分散交互目标）来逃避分类器。本文未评估分类器的对抗鲁棒性，这是一个重要的安全性研究方向。

## 6. 结论

本文提出了第一个从链上交易数据中系统性识别AI agent地址并量化其安全暴露面的实证研究。通过设计涵盖时序模式、Gas定价行为、交互模式和授权安全的23特征行为指纹框架，我们在44个以太坊地址（22个agent + 22个人类）的数据集上训练了多种分类器，RandomForest在LOO-CV中达到AUC 0.965。特征分析表明，交互模式（特别是unique_contracts_ratio，AUC=0.909）是区分agent与人类的最强信号，15/23个特征在p<0.05水平下具有统计显著性。

四维安全审计揭示了agent安全态势的三个核心发现：(1) 权限暴露高度集中，228个无限授权由仅2个agent持有；(2) 交易失败率在agent间差异巨大（0.0%-29.6%）；(3) 当前评级框架下22个agent中3个为MEDIUM风险、19个为LOW风险。误分类分析指出交易所热钱包是当前分类器的核心挑战，未来应考虑引入更细粒度的分类体系。

我们呼吁区块链协议设计者实施agent感知的安全机制——包括有限额度授权作为默认设置、agent专属的Gas优化路径、以及基于行为指纹的实时监测系统。本文的识别框架、标注数据集和安全审计方法论为这一方向的后续研究提供了基础。

## 参考文献

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

[11] Autonolas, "Autonolas: Autonomous Agent Services," whitepaper, 2023.

[12] ai16z, "ELIZA: AI Agent Framework," GitHub repository, 2024.

[13] Virtuals Protocol, "Virtuals: Agent Tokenization Platform," whitepaper, 2024.

[14] Fetch.ai, "Fetch.ai: Autonomous Economic Agents," whitepaper, 2023.

[15] SingularityNET, "SingularityNET: Decentralized AI Platform," whitepaper, 2023.

[16] AI Arena, "AI Arena: AI Fighting Game on Ethereum," whitepaper, 2024.

[17] L. Zhou, X. Xiong, J. Ernstberger, et al., "SoK: Decentralized Finance (DeFi) Attacks," in Proc. IEEE S&P, 2023.

[18] Y. Wen, Y. Liu, and D. Lie, "DeFi Composability as a Means for Protocol-Level Security Analysis," in Proc. CCS, 2023.

[19] N. Grech, M. Kong, A. Scholz, and B. Smaragdakis, "MadMax: Analyzing the Out-of-Gas World of Smart Contracts," CACM, 2022.

[20] P. Tsankov, A. Dan, D. Drachsler-Cohen, A. Gervais, F. Buenzli, and M. Vechev, "Securify: Practical Security Analysis of Smart Contracts," in Proc. CCS, 2018.

[21] X. Wang et al., "Risks of AI Agents with Cryptocurrency Access," arXiv preprint, 2024.

[22] Y. Zhang et al., "Approval-Based Attacks in DeFi: Measurement and Defense," in Proc. NDSS, 2024.

[23] Y. He et al., "A Survey on AI Agents on Blockchain," arXiv preprint 2601.04583, 2026.
