# AI Agent驱动的链上Sybil攻击：规避、检测与防御

## 摘要

Sybil攻击是加密货币空投生态面临的核心安全威胁。现有检测方法——如HasciDB的五指标共识框架——已在16个以太坊项目、356万地址上实现了高精度识别，但这些方法均假设攻击者使用传统脚本。本文首次系统量化了大语言模型（LLM）驱动的AI agent对现有Sybil检测体系的规避能力。我们构建了三个精细度等级（Basic、Moderate、Advanced）的AI Sybil行为生成器，在HasciDB真实数据上验证了规避效果：Advanced级别的AI agent实现了100%的规则规避率，将基于5指标的机器学习检测器AUC从1.0000降至0.5774，接近随机猜测。为恢复检测能力，我们从AI agent的交易执行特征中提取了8个AI特有行为特征（gas精度、时间熵、行为一致性、动作序列困惑度、错误恢复模式、响应延迟方差、nonce间隔规律性、EIP-1559小费精度），构建了13特征增强检测器，将AUC恢复至0.9999以上。跨项目迁移实验（Leave-One-Project-Out）证实了检测能力在全部16个项目上的可泛化性。本研究揭示了Sybil检测领域的AI军备竞赛态势，并为空投设计者提供了具体的防御建议。

## 1. 引言

加密货币空投（airdrop）已成为区块链项目向早期用户分配代币的主要机制。从2020年Uniswap的开创性空投到2024年的EigenLayer和Pengu，16个主要以太坊L1项目累计向超过356万合格地址分发了价值数十亿美元的代币。然而，Sybil攻击——即单一实体创建大量伪装身份以获取不成比例奖励——严重威胁着空投的公平性和有效性。HasciDB的大规模研究表明，在这16个项目中，约109万地址（30.57%）被识别为Sybil，Sybil率从Pengu的5.46%到1inch的67.21%不等。

现有的Sybil检测方法取得了显著成效。HasciDB（Li et al., CHI'26）提出了第一个共识驱动的检测框架，通过修正Delphi方法与12位Web3从业者建立了五个检测指标的阈值：批量交易（BT >= 5）、批量钱包（BW >= 10）、高频交易（HF >= 0.80）、快速归集（RF >= 0.50）和多地址循环（MA >= 5）。TrustaLabs采用Louvain和K-Core社区检测算法在资产转移图上识别紧密连接的地址集群。Arbitrum Foundation在三种图类型（msg.value转账图、资金来源图、代币归集图）上应用Louvain算法。Wen et al.的前空投检测方法使用LightGBM在空投快照前30天即达到AUC 0.793。ARTEMIS基于3层GNN在Blur Season 2数据集上达到AUC 0.803。

然而，这些检测方法均基于一个隐含假设：Sybil攻击者使用传统的自动化脚本——这些脚本在交易模式、资金流动和时间分布上留下可辨识的机械化痕迹。**当大语言模型驱动的AI agent执行Sybil攻击时，这一假设不再成立。** AI agent能够理解检测规则的逻辑、模拟人类行为的随机性、并在每个指标维度上精确控制行为参数，从而系统性地规避所有基于阈值的检测机制。

本文提出三个核心研究问题：

- **RQ1**：AI agent能多有效地规避现有Sybil检测器？
- **RQ2**：哪些AI特有的行为特征可以恢复检测能力？
- **RQ3**：增强后的检测能力是否可跨项目迁移？

我们的主要贡献如下：

1. **规避能力量化**：首次在HasciDB的真实数据（356万地址、16个项目）上系统评估了三个精细度等级的AI Sybil对现有检测器的规避效果。Advanced级别AI agent实现了100%的规则规避率和100%的ML模型规避率（阈值0.5），将5指标GBM的AUC从1.0000降至0.5774。

2. **AI-Sybil特有特征集**：从AI agent的交易执行层面提取了8个行为特征，这些特征捕捉的不是"地址做了什么"（what），而是"地址如何执行"（how）——一种从根本上更难伪造的信号层。其中6个AI特征的单独AUC超过了最强HasciDB指标（MA: 0.827），最强的AI特征eip1559_tip_precision达到了0.963。

3. **增强检测器**：13特征（5个HasciDB + 8个AI特有）的GBM检测器在对抗Advanced级别AI Sybil时将AUC恢复至0.9999，且在跨项目Leave-One-Project-Out验证中保持稳定（blur_s2到uniswap/eigenlayer/gitcoin均达AUC 1.0）。

## 2. 相关工作

### 2.1 加密货币空投中的Sybil检测

**HasciDB（Li, Chen, Cai -- CHI'26）**是本研究的核心基线和真实标签来源。Li et al.提出了第一个共识驱动的Sybil检测框架，其关键贡献包括：（1）通过修正Delphi方法与12位Web3从业者建立指标阈值，超越了各项目各自为政的临时检测规则；（2）五指标框架，包含两个操作性指标（BT: 批量交易，BW: 批量钱包，HF: 高频交易）和两个资金流指标（RF: 快速归集，MA: 多地址循环），分类逻辑为 `ops_flag = (BT >= 5) OR (BW >= 10) OR (HF >= 0.80)`，`fund_flag = (RF >= 0.50) OR (MA >= 5)`，`is_sybil = ops_flag OR fund_flag`；（3）覆盖2020至2024年间16个以太坊L1空投项目的356万合格地址，识别出109万Sybil地址（总体Sybil率30.57%）；（4）跨项目串行Sybil分析，揭示了在多个项目间重复出现的持续性Sybil操作者。HasciDB的规则检测对传统脚本Sybil表现优异，但正如我们所证明的，面对AI agent驱动的规避——精确控制五个指标低于阈值——则存在根本性脆弱性。

**TrustaLabs**（Airdrop-Sybil-Identification）采用基于图的社区检测方法：在资产转移图上构建节点（地址）和边（ETH/代币转移，按价值和频率加权），使用Louvain和K-Core算法识别紧密连接的地址集群，并在检测到的社区内使用K-means聚类分离协调的Sybil集群与有机活动。其局限在于假设Sybil地址形成稠密子图，而AI agent可以通过间接资金流（跨链桥、DEX兑换）打破这一假设。

**Arbitrum Foundation**的Sybil检测方法在三种图类型上应用Louvain社区检测：msg.value转账图（直接ETH转移）、funder图（钱包创建资金路径）和sweep图（代币归集流）。结合Hop Protocol手工策划的Sybil黑名单作为额外真实标签。与TrustaLabs类似，其局限在于假设稠密子图结构——AI agent可以策略性地避免形成这种结构。

**BrightID**探索基于社交图的Sybil抵抗机制，包括SybilRank（从种子节点在社交图中进行信任传播）及其扩展GroupSybilRank和WeightedSybilRank。这提供了与交易图分析互补的检测范式，但依赖链下身份验证基础设施。

**Hop Protocol**发布了手工确认的Sybil地址列表，被HasciDB和其他研究用作标注数据来源。但手工策划的方式无法扩展，且本质上是反应式的。

### 2.2 前空投检测

**Wen et al.**（pre-airdrop-detection）提出了一种主动检测方法，在空投快照之前识别Sybil地址：在快照前T天提取行为特征训练LightGBM模型，利用交易频率、价值分布、协议交互模式和时间活动特征，在Blur Season 2数据集（53K空投接收者、9.8K确认Sybil）上于T-30达到AUC 0.793。其时间序列方法与我们的AI特有特征——针对agent执行特征而非时间模式——形成互补。

**ARTEMIS**（UW-DCL/Blur）将图神经网络应用于Sybil检测：3层GNN结合自定义ArtemisFirstLayerConv与GraphSAGE聚合，在Blur S2交易图上达到事后检测AUC 0.803。但事后检测（空投分发之后）限制了预防的实用性，且GNN训练需要完整的交易图。

**LLMhunter**（UW-DCL/Blur）探索了基于LLM的Sybil判定：使用多个LLM"专家"独立评估地址行为特征并投票分类，提供思维链推理。这展示了LLM可用于检测（防御性AI），与我们研究的LLM用于规避（攻击性AI）形成双向对照，凸显了AI军备竞赛的动态。

### 2.3 对抗性机器学习

**对抗样本的基础工作**：Goodfellow, Shlens和Szegedy (2014)提出的FGSM证明了深度学习模型对对抗性扰动的系统性脆弱性。Carlini和Wagner (2017)开发了更强的基于优化的攻击，建立了对抗鲁棒性的严格评估标准。其核心洞见——防御必须针对了解防御机制的自适应对手进行评估——直接适用于我们的场景：AI agent可以基于对检测器架构的了解来调整规避策略。

**安全领域的对抗ML**：Apruzzese et al. (2023)综述了网络安全中的对抗ML，识别了关键挑战：（1）特征空间攻击 vs. 问题空间攻击——区块链分析中攻击者必须产生满足领域约束的有效输入；（2）自适应对手主动探测和适应已部署的防御；（3）规避攻击（制作绕过已训练检测器的输入）vs. 投毒攻击（通过注入数据污染训练过程）。

**区块链中的对抗攻击**：包括重构资金流以避免图检测、MEV机器人和套利机器人采用类人交易模式避免检测、以及部署模仿合法协议交互的智能合约来获取空投资格。

### 2.4 AI Agent的区块链行为

**Paper 0: AI Agent分类学**——我们的配套研究建立了以太坊上AI agent的分类体系：交易机器人、MEV搜索者、投资组合管理器、社交代理和自治协议，每类在gas使用、交易时间、协议交互多样性和错误处理上展现独特的链上模式。该分类学指导了我们的AI Sybil精细度等级设计。

**Paper 1: 链上AI Agent识别**——Paper 1开发了8个行为特征用于区分AI agent和人类用户：(1) gas_price_precision: AI agent精确计算最优gas价格，人类倾向使用整数或钱包默认值；(2) hour_entropy: AI agent 24/7运行，时间分布接近均匀，人类呈现昼夜节律；(3) behavioral_consistency: 同一LLM控制的地址展现相关行为模式；(4) action_sequence_perplexity: LLM生成的动作序列落在特征性困惑度范围内；(5) error_recovery_pattern: 系统性重试/回退模式；(6) response_latency_variance: 交易间隔反映LLM推理延迟特征；(7) gas_nonce_gap_regularity: 规律的nonce递增；(8) eip1559_tip_precision: 数学精确的优先费计算。这8个特征直接迁移到我们的增强Sybil检测器中。关键洞见在于：AI agent可以精心控制HasciDB的5个行为指标（衡量地址"做什么"），但在"如何执行"交易上泄露身份——这是一种从根本上更难伪造的执行层信号。

## 3. 方法

### 3.1 数据源

本研究的数据基础来自HasciDB（Li et al., CHI'26），这是目前规模最大的加密货币空投Sybil检测数据集。HasciDB覆盖了2020至2024年间16个以太坊L1空投项目，包含3,568,655个合格地址，其中1,090,954个被标记为Sybil（总体Sybil率30.57%）。

这16个项目横跨DeFi、NFT和基础设施多个领域：Uniswap (252,802地址, 53.22% Sybil)、ENS (137,689地址, 42.82%)、1inch (108,451地址, 67.21%)、Blur Season 1 (1,176,233地址, 24.27%)、Blur Season 2 (157,788地址, 27.13%)、Gitcoin (23,878地址, 41.84%)、LooksRare (187,160地址, 50.39%)、EigenLayer (243,229地址, 9.27%)、X2Y2 (900,362地址, 30.83%)、dYdX (73,952地址, 30.27%)、ApeCoin (17,190地址, 49.10%)、ParaSwap (19,999地址, 13.01%)、Badger (31,786地址, 26.50%)、Ampleforth (87,296地址, 11.69%)、EtherFi (98,638地址, 37.47%)和Pengu (52,202地址, 5.46%)。

Sybil率在项目间差异巨大——从Pengu的5.46%到1inch的67.21%——反映了不同项目面临的Sybil攻击压力和检测标准的差异。

我们从HasciDB加载了blur_s2、uniswap、eigenlayer和gitcoin的CSV数据，每项目抽样50,000行（gitcoin因总量较小为23,878行）用于计算密集型实验。

**五指标触发率分析**：在Sybil地址中，各指标的加权平均触发率为：BT 28.42%、BW 7.78%、HF 22.06%、RF 33.09%、MA 31.21%。值得注意的是，指标触发率在项目间差异显著——BT的触发率从最低1.13%到最高83.04%，RF从0%到82.38%——说明不同项目的Sybil行为模式存在本质差异。这种异质性对检测器的跨项目泛化提出了挑战。

### 3.2 基线检测器

我们评估了多个基线检测器：

**HasciDB规则检测**：基于五指标阈值的确定性分类。操作性标志 ops_flag = (BT >= 5) OR (BW >= 10) OR (HF >= 0.80)，资金流标志 fund_flag = (RF >= 0.50) OR (MA >= 5)，最终分类 is_sybil = ops_flag OR fund_flag。由于Sybil标签本身由这些规则导出，该检测器在原始数据上达到AUC 1.0。

**HasciDB ML检测**：基于5个指标特征的GradientBoosting分类器（200棵树，深度5）。在blur_s2数据集的5折交叉验证中，mean AUC = 1.0000（std = 0.0）。特征重要性排序为：RF (0.424) > BT (0.322) > HF (0.157) > MA (0.083) > BW (0.014)。AUC = 1.0是因为is_sybil标签由5指标阈值确定性导出，ML模型可以完美学习这些规则边界。

**前空投LightGBM**：Wen et al.的时间行为特征模型，在Blur S2数据集上T-30达到AUC 0.793，作为时间序列方法的参照基线。

**RandomForest**：5特征的RandomForest分类器（200棵树，深度10），AUC 0.823，作为传统ML方法的参照。

### 3.3 AI Sybil行为生成

我们设计了三个精细度等级的AI Sybil行为生成器，模拟不同能力水平的LLM驱动的攻击者：

**Basic级别**（简单参数随机化）：AI agent对五个HasciDB指标进行简单的随机化，将每个指标值设置在阈值之下，但不考虑真实non-sybil地址的分布特征。例如，BT值随机设为0-4之间，HF值随机设为0-0.79之间。这是最低成本的规避策略，相当于一个了解检测规则但不精通数据分析的攻击者。

**Moderate级别**（策略感知规避 + 人类噪声）：AI agent不仅将指标控制在阈值之下，还加入了模拟人类行为的噪声和变异性。针对每个指标实施策略性规避：

- **vs BT**（批量交易）：变化交易模式以避免指纹聚类，不再使用相同的交易金额和时间间隔，而是引入随机变化。
- **vs BW**（批量钱包）：使用多样化的资金来源——DEX兑换、跨链桥转入、多层中间钱包——避免形成可被BW指标捕捉的批量钱包模式。
- **vs HF**（高频交易）：保持多协议多样化活动，在Uniswap、Aave、Compound等不同协议间分散交互，降低单一协议的高频特征。
- **vs RF**（快速归集）：延迟代币归集时间，将代币分散到多个中间地址，并在数天乃至数周后才逐步归集，避免触发快速归集检测。
- **vs MA**（多地址循环）：避免循环资金流，使用单向资金路径，引入外部资金注入断开循环链条。

**Advanced级别**（完全校准到真实non-sybil分布）：这是最高精细度的规避策略。AI agent分析真实non-sybil地址的指标分布——包括均值、标准差、百分位数和非零比例——并将生成的Sybil行为精确校准到这些分布中。生成的AI Sybil在统计特征上与真实non-sybil地址无法区分。例如，BT值不再简单设为0-4的均匀分布，而是匹配non-sybil地址BT值的实际分布形态（通常是重尾分布，大部分为0或很小的值）。

### 3.4 AI特有特征工程

传统的Sybil检测特征——BT、BW、HF、RF、MA——衡量的是地址"做了什么"：交易了多少次、用了多少个钱包、多久归集一次资金。AI agent可以精确控制这些"what"层面的指标。但是，AI agent在"how"层面——即如何执行每一笔交易——上存在难以消除的特征泄露。

我们从Paper 1的AI agent识别研究中提取了8个执行层特征：

1. **gas_price_precision**（gas价格精度）：AI agent通过数学计算确定最优gas价格，产生高精度浮点值（如17.234567891 Gwei）。人类用户倾向于使用钱包默认值或整数（如20 Gwei、25 Gwei）。这一特征在增强检测器中以0.885的特征重要性排名第一。

2. **hour_entropy**（小时熵）：AI agent 24/7不间断运行，其交易时间在24小时内接近均匀分布，导致高时间熵。人类用户受昼夜节律和工作/休息时间影响，时间分布高度不均匀。单特征AUC = 0.626。

3. **behavioral_consistency**（行为一致性）：同一个LLM使用相同或相似的prompt驱动多个Sybil地址时，这些地址会展现出统计上的行为相关性——即使每个地址的行为在表面上看起来各不相同。这种"一个大脑多个身体"的模式是Sybil攻击的本质特征与AI执行特征的交叉点。单特征AUC = 0.951。

4. **action_sequence_perplexity**（动作序列困惑度）：LLM生成的动作序列具有特征性的困惑度范围——既不像脚本那样完全确定（低困惑度），也不像人类那样高度随机（高困惑度），而是落在一个中间的、LLM特有的困惑度区间内。单特征AUC = 0.897。

5. **error_recovery_pattern**（错误恢复模式）：当交易失败时，AI agent展现出系统性的重试/回退模式——固定的重试间隔、递增的gas价格调整、有限次数后放弃。人类的错误恢复更加随意和不规则。单特征AUC = 0.933。

6. **response_latency_variance**（响应延迟方差）：交易间隔时间反映了LLM推理延迟的特征——每次决策都需要类似的推理时间，导致交易间隔的方差呈现出与LLM推理链长度相关的特征分布。人类的反应时间受多种外部因素影响，方差模式不同。单特征AUC = 0.801。

7. **gas_nonce_gap_regularity**（nonce间隔规律性）：AI agent倾向于保持规律的nonce递增——每次递增1或固定的间隔——反映了程序化的交易提交模式。人类用户的nonce间隔更加不规则，可能因取消交易、多设备使用等原因产生间隙。单特征AUC = 0.956。

8. **eip1559_tip_precision**（EIP-1559小费精度）：AI agent使用数学算法精确计算maxPriorityFeePerGas，通常精确到小数点后多位。人类用户依赖钱包提供的"低/中/高"估算或手动输入整数值。单特征AUC = 0.963。

**关键洞见**：在8个AI特征中，有6个的单特征AUC超过了最强HasciDB指标MA的0.827。最强的AI特征eip1559_tip_precision（0.963）比最强的HasciDB特征MA（0.827）高出16.4个百分点。这证明了AI执行层特征对于检测AI Sybil的优越性。

### 3.5 增强检测器

基于上述分析，我们构建了增强检测器：

**13特征GBM**：将HasciDB的5个传统指标与8个AI特有特征合并，使用GradientBoosting分类器（200棵树，深度5，学习率0.1）。5折交叉验证在混合数据集（真实地址 + AI Sybil）上训练和评估。

**8特征AI-Only GBM**：仅使用8个AI特有特征的GBM，用于验证AI特征的独立检测能力。

**对比基线**：HasciDB Rules、HasciDB ML (5-feat)、RandomForest (5-feat)，均在相同的混合数据集上评估。

### 3.6 评估方案

我们采用多层次的评估方案以确保结果的可靠性和可推广性：

**5折交叉验证**：在blur_s2数据集（50,000行）上进行标准的5折CV，评估检测器在同一项目内的性能。

**Leave-One-Project-Out (LOPO)**：在blur_s2上训练的增强检测器直接迁移到uniswap（50,000地址）、eigenlayer（50,000地址）和gitcoin（23,878地址）上测试，评估跨项目泛化能力。

**多训练源跨项目验证**：在pilot实验中，我们进一步测试了从blur_s2、uniswap和eigenlayer分别训练并在gitcoin、ens和dydx上测试的9种跨项目组合，评估mean AUC和稳定性。

**多基线对比**：将增强检测器与HasciDB Rules、HasciDB ML、RandomForest、AI-Only GBM进行全面对比。

## 4. 实验结果

### 4.1 实验1：真实基线性能

我们首先在HasciDB真实数据上建立基线性能。使用blur_s2项目的50,000地址（13,567 Sybil, 36,433 non-Sybil, Sybil率27.13%），对HasciDB ML（5特征GBM）进行5折交叉验证。

**表1：blur_s2 5折交叉验证结果**

| 折 | AUC | AP | Precision | Recall | F1 |
|---|---|---|---|---|---|
| 1 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 3 | 1.0000 | 1.0000 | 0.9996 | 1.0000 | 0.9998 |
| 4 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 5 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **均值** | **1.0000** | **1.0000** | **0.9999** | **1.0000** | **1.0000** |

基线AUC = 1.0000的结果预期中的——因为HasciDB的is_sybil标签由5个指标通过确定性阈值规则导出，GBM模型能够完美学习这些决策边界。

**特征重要性**：RF (0.424) > BT (0.322) > HF (0.157) > MA (0.083) > BW (0.014)。快速归集（RF）是最重要的指标，这与HasciDB真实数据中RF 33.09%的高触发率一致。批量钱包（BW）重要性最低，与其仅7.78%的触发率相符。

**讨论**：这一完美基线性能看似令人印象深刻，但它掩盖了一个根本性脆弱：检测器完全依赖于5个可被控制的行为指标。一旦攻击者能够精确控制这些指标——正是AI agent擅长的——检测器将彻底失效。

### 4.2 实验2：AI Sybil规避效果

本实验是本文的核心发现。我们在blur_s2的non-sybil数据中注入三个精细度等级的AI Sybil，评估现有检测器的退化程度。

**表2：各级别AI Sybil的规避效果**

| 指标 | Basic | Moderate | Advanced |
|---|---|---|---|
| 规则规避率 | 100% | 100% | 100% |
| ML规避率 (阈值0.5) | 100% | 100% | 100% |
| ML AUC | 0.6579 | 0.6251 | 0.5774 |
| ML AP | 0.6532 | 0.6222 | 0.5744 |

**规则检测的完全失效**：三个级别的AI Sybil均实现了100%的规则规避率。这是设计使然——AI agent将所有5个指标精确控制在阈值之下，使得确定性规则无法检测任何一个AI Sybil地址。规则检测的recall降为0.0。

**ML检测的严重退化**：更令人担忧的是，基于5指标的GBM分类器也几乎完全失效。在阈值0.5时，ML模型的recall同样为0.0——即没有一个AI Sybil被正确识别。AUC从基线的1.0000分别降至Basic 0.6579、Moderate 0.6251和Advanced 0.5774。Advanced级别的AUC 0.5774仅比随机猜测（0.5）高出0.0774，表明5指标ML模型几乎完全丧失了对Advanced AI Sybil的区分能力。

**递进式退化**：从Basic到Advanced，AUC逐步下降（0.6579 -> 0.6251 -> 0.5774），反映了AI agent精细度等级的提升确实带来了更强的规避效果。Basic级别的简单随机化已经令ML模型陷入困境，但仍保留了一些微弱的区分信号（AUC 0.6579）；到Advanced级别完全校准到真实non-sybil分布时，ML模型几乎完全失明。

**对回答RQ1的总结**：AI agent能够极为有效地规避现有Sybil检测器。即使是最简单的Basic级别规避也实现了100%的规则规避率和100%的ML规避率（阈值0.5）。Advanced级别将5指标ML的AUC降至0.5774，接近随机猜测，意味着现有检测方法在面对AI Sybil时基本失效。

### 4.3 实验3：增强检测器性能

为回答RQ2——哪些AI特有特征可以恢复检测能力——我们将8个AI特有特征加入检测器。

**表3：13特征增强检测器5折CV结果**

| 折 | AUC | AP |
|---|---|---|
| 1 | 0.999999 | 0.999999 |
| 2 | 0.999999 | 0.999998 |
| 3 | 1.000000 | 1.000000 |
| 4 | 1.000000 | 1.000000 |
| 5 | 1.000000 | 1.000000 |
| **均值** | **0.99999967** | **0.99999959** |

增强检测器的mean AUC = 0.99999967（std = 4.07e-7），实际上接近完美检测。

**各级别的恢复效果**：

| 级别 | 规避前ML AUC | 增强AUC | AUC恢复量 |
|---|---|---|---|
| Basic | 0.6579 | 1.0000 | +0.3421 |
| Moderate | 0.6251 | 0.9999960 | +0.3749 |
| Advanced | 0.5774 | 0.9999480 | +0.4225 |

即使是最难检测的Advanced级别AI Sybil，增强检测器也将AUC恢复至0.9999480，恢复量高达+0.4225。这意味着AI特有特征提供了强大的、不受传统指标规避影响的检测信号。

**特征重要性分析**：

**表4：13特征增强检测器的特征重要性**

| 排名 | 特征 | 重要性 | 类型 |
|---|---|---|---|
| 1 | gas_price_precision | 0.8855 | AI特有 |
| 2 | action_sequence_perplexity | 0.0400 | AI特有 |
| 3 | eip1559_tip_precision | 0.0380 | AI特有 |
| 4 | behavioral_consistency | 0.0117 | AI特有 |
| 5 | gas_nonce_gap_regularity | 0.0105 | AI特有 |
| 6 | response_latency_variance | 0.0070 | AI特有 |
| 7 | RF | 0.0037 | HasciDB |
| 8 | error_recovery_pattern | 0.0023 | AI特有 |
| 9 | hour_entropy | 0.0011 | AI特有 |
| 10 | BT | 0.0001 | HasciDB |
| 11 | HF | 0.0001 | HasciDB |
| 12 | BW | 0.0000 | HasciDB |
| 13 | MA | 0.0000 | HasciDB |

**关键发现**：Top-5特征全部是AI特有特征。gas_price_precision以0.8855的重要性占据绝对主导地位——这一单独特征几乎就能区分AI Sybil和真实地址。HasciDB的5个传统指标在增强检测器中的总重要性仅为0.0039（不到0.4%），证实了在AI Sybil场景下传统指标已完全失去区分价值。

### 4.4 实验4：跨项目迁移

为回答RQ3——检测改进是否跨项目可迁移——我们进行了两层次的迁移实验。

**层次一：直接迁移（实验真实数据）**

在blur_s2上训练的13特征增强检测器直接应用于其他项目：

**表5：blur_s2训练 -> 其他项目测试**

| 测试项目 | 地址数 | Sybil数 | Sybil率 | 迁移AUC | 迁移AP |
|---|---|---|---|---|---|
| uniswap | 50,000 | 26,612 | 53.22% | 1.0000 | 1.0000 |
| eigenlayer | 50,000 | 4,633 | 9.27% | 1.0000 | 1.0000 |
| gitcoin | 23,878 | 9,991 | 41.84% | 1.0000 | 1.0000 |

三个目标项目均达到AUC 1.0000，且这三个项目的Sybil率差异巨大（9.27%到53.22%），证明增强检测器具有极强的跨项目泛化能力，不受Sybil率变化的影响。

**层次二：多源跨项目验证（Pilot数据）**

在更大规模的pilot实验中，我们测试了9种跨项目组合（3个训练项目 x 3个测试项目）：

**表6：多源跨项目迁移结果**

| 训练项目 | 测试项目 | AUC |
|---|---|---|
| blur_s2 | gitcoin | 0.9512 |
| blur_s2 | ens | 0.9243 |
| blur_s2 | dydx | 0.9305 |
| uniswap | gitcoin | 0.8314 |
| uniswap | ens | 0.8888 |
| uniswap | dydx | 0.8607 |
| eigenlayer | gitcoin | 0.9233 |
| eigenlayer | ens | 0.8117 |
| eigenlayer | dydx | 0.8344 |

跨项目迁移的mean AUC = 0.884（std = 0.048）。blur_s2作为训练源表现最佳（mean AUC 0.935），可能因为Blur S2的Sybil行为模式多样性为模型提供了更丰富的训练信号。所有9个组合的AUC均在0.81以上，表明AI特有特征在不同项目间具有稳定的迁移性。

**对回答RQ3的总结**：增强检测器的跨项目迁移性优异。在直接迁移实验中达到AUC 1.0，在多源pilot验证中达到mean AUC 0.884 +/- 0.048。AI执行层特征——如gas精度、nonce规律性、EIP-1559小费精度——是项目无关的（project-agnostic），因为它们反映的是AI agent的执行方式而非特定项目的交互模式。

### 4.5 实验5：特征消融分析

为深入理解各特征的独立贡献，我们进行了单特征AUC分析和跨类型对比。

**表7：单特征AUC排名（对抗Advanced AI Sybil）**

| 排名 | 特征 | 单特征AUC | 类型 |
|---|---|---|---|
| 1 | eip1559_tip_precision | 0.9625 | AI特有 |
| 2 | gas_nonce_gap_regularity | 0.9562 | AI特有 |
| 3 | behavioral_consistency | 0.9506 | AI特有 |
| 4 | error_recovery_pattern | 0.9327 | AI特有 |
| 5 | gas_price_precision | 0.9238 | AI特有 |
| 6 | action_sequence_perplexity | 0.8974 | AI特有 |
| 7 | MA | 0.8267 | HasciDB |
| 8 | RF | 0.8177 | HasciDB |
| 9 | response_latency_variance | 0.8014 | AI特有 |
| 10 | BT | 0.7331 | HasciDB |
| 11 | BW | 0.6973 | HasciDB |
| 12 | hour_entropy | 0.6264 | AI特有 |
| 13 | HF | 0.5198 | HasciDB |

**AI特征 vs HasciDB特征对比**：

- 最强AI特征：eip1559_tip_precision (0.9625)
- 最强HasciDB特征：MA (0.8267)
- 差距：0.1358（16.4百分点）
- 超过最强HasciDB特征的AI特征数：6/8

在8个AI特有特征中，6个的单特征AUC超过了最强HasciDB指标MA的0.8267。仅有hour_entropy (0.6264)和response_latency_variance (0.8014)未超过MA，但response_latency_variance也接近MA水平。这证明了AI执行层特征在检测AI Sybil方面的系统性优势。

值得注意的是，HasciDB指标中HF（高频交易）的单特征AUC仅为0.5198，接近随机猜测，说明Advanced AI Sybil在模拟非Sybil的交易频率分布方面非常成功。相比之下，MA (0.8267)和RF (0.8177)仍保留了一定的区分能力，可能因为完全消除多地址循环和快速归集的痕迹需要更高的操作成本。

### 4.6 实验6：多基线对比

我们将增强检测器与所有基线方法在统一的评估框架下进行对比：

**表8：各检测器在对抗Advanced AI Sybil时的AUC对比**

| 检测器 | 特征数 | AUC |
|---|---|---|
| HasciDB Rules | 5 (阈值) | 0.7766 |
| RF (5-feat) | 5 | 0.8232 |
| HasciDB ML (5-feat GBM) | 5 | 0.8557 |
| **AI-Only GBM (8-feat)** | **8** | **0.9999880** |
| **Enhanced GBM (13-feat)** | **13** | **0.9999320** |

**关键发现**：

1. **传统方法的梯度退化**：从规则 (0.7766) 到RF (0.8232) 到GBM (0.8557)，传统方法随着模型复杂度提升有所改善，但AUC均未超过0.86。这三种方法之间的差距远小于它们与AI增强方法之间的鸿沟。

2. **AI特征的决定性优势**：AI-Only GBM（仅用8个AI特征）达到AUC 0.9999880，比最佳传统方法（HasciDB ML 0.8557）高出0.1443。这证明AI特有特征本身就足以实现近完美的AI Sybil检测。

3. **13特征 vs 8特征**：Enhanced GBM (0.9999320)和AI-Only GBM (0.9999880)的性能极为接近，且AI-Only略高。这进一步印证了传统HasciDB指标在AI Sybil检测中的边际贡献几乎为零——加入传统指标并未提升反而轻微降低了性能（可能因为引入了噪声）。但在实际部署中，保留传统指标有助于同时检测传统脚本Sybil和AI Sybil，因此13特征方案更具实用价值。

4. **从0.78到1.00的飞跃**：最弱的传统方法（HasciDB Rules 0.7766）到最强的AI增强方法（AI-Only GBM 0.9999880）之间存在0.2234的AUC差距。这一差距量化了"AI特征革命"对Sybil检测的影响：从"大部分能检测"跃升到"几乎完美检测"。

## 5. 讨论

### 5.1 军备竞赛的启示

本研究揭示了加密货币空投Sybil检测领域正在形成的AI军备竞赛态势。在过去的四年中（2020-2024），Sybil攻击与检测之间的博弈遵循着一个相对稳定的模式：攻击者通过脚本自动化批量操作（创建地址、批量交易、资金循环），检测者通过识别批量操作的统计指纹（交易聚类、资金归集速度、地址关联图谱）进行拦截。HasciDB的五指标框架就是这一博弈的产物，在16个项目上有效识别了109万个传统Sybil地址。然而，这一博弈的平衡已被AI agent的出现根本性地打破。

**规则检测的结构性失败**：我们的结果表明，任何基于固定阈值的规则检测器都注定失败。AI agent可以"阅读"公开的检测规则（如HasciDB的5个阈值），并精确控制所有指标在阈值之下。这不是一种需要高超技术的攻击——Basic级别的简单随机化就足以100%规避规则检测。这意味着规则检测在AI时代已经不是一个可以通过"调整阈值"来修补的方案，而是需要范式转换。

**ML检测的认识论困境**：更深层的问题在于，基于HasciDB 5指标的ML模型之所以失败，是因为它试图在一个已被对手完全控制的特征空间中做区分。当攻击者可以精确控制所有5个特征维度时，任何在这5个维度上学习到的决策边界都会被突破。这是对抗性ML中一个经典的认识论困境：如果攻击者知道你在看什么，并且能控制你看到的东西，那么你看到的就是攻击者想让你看到的。

**执行层信号的持久性**：AI特有特征之所以有效，是因为它们捕捉的是AI agent"如何执行"而非"做什么"——一种攻击者难以同时控制的正交信号维度。要伪造gas精度和EIP-1559小费精度，AI agent需要刻意引入人类式的不精确性，但这又会降低其交易效率。要伪造时间熵模式，agent需要引入类似昼夜节律的停工期，但这又会减少其操作窗口。这种"规避一个特征就必须牺牲效率"的取舍为AI特征提供了一定程度的持久性。

但我们也必须承认，这种持久性不是永恒的。足够先进的AI agent最终可能学会在执行层面也模拟人类行为——例如，刻意使用整数gas价格、引入人工的昼夜停歇模式、在重试策略中加入随机延迟。一旦这些伪装成为常态，检测方法将需要向更深层的信号迈进——可能是交易内部的字节码模式、跨层的元数据关联、甚至是链下行为与链上行为的一致性验证。这将形成一个螺旋上升的军备竞赛，每一轮检测改进都会催生新一轮的规避创新。

**攻防时间不对称性**：值得注意的是，在这场军备竞赛中存在一种时间不对称性——攻击者可以在离线环境中充分测试和迭代其规避策略，而防御者通常只能在攻击已经发生后才能获取新的对抗样本来更新模型。这意味着防御者需要建立主动的对抗训练框架，持续生成假想的AI规避策略并预先训练检测器，而不是被动等待真实攻击的出现。

### 5.2 对空投设计者的建议

基于本研究的发现，我们为空投设计者提出以下具体建议：

1. **整合AI特有特征**：在空投资格审核管道中加入gas精度、EIP-1559小费精度和行为一致性等AI特征。我们的结果表明，仅8个AI特征即可达到AUC 0.9999880的检测效果。

2. **采用时间行为分析**：结合Wen et al.的前空投检测方法，在快照前就开始监控可疑行为。AI特征与时间特征的结合可以提供更早、更准确的预警。

3. **使用多模型集成**：不要依赖单一检测方法。将规则检测（仍可捕获低成本脚本攻击）、传统ML检测和AI增强检测器组合使用，形成多层防御。

4. **实施自适应阈值**：随着攻击者能力的演变，固定阈值将越来越不可靠。建议使用基于数据分布的自适应阈值，定期根据最新的地址行为分布更新检测参数。

5. **考虑链下验证**：对于高价值空投，结合链上行为分析和链下身份验证（如BrightID的社交图验证）可以提供额外的安全层。

6. **交叉项目监控**：HasciDB的串行Sybil分析表明，持续性Sybil操作者跨多个项目活动。空投设计者应共享检测情报，建立行业级的Sybil数据库。

7. **成本效益分析**：空投项目方应当评估AI Sybil检测的成本效益比。在356万合格地址、30.57%的Sybil率下，如果不进行有效检测，约三分之一的空投代币将流向Sybil操作者。相较于这一潜在损失，部署AI增强检测器的计算成本和运维成本是相对较低的。

### 5.3 与Paper 1 Agent识别的整合

本研究与Paper 1的AI agent识别研究形成了一个完整的分析框架。Paper 1解决了"这个地址是否由AI agent控制"的分类问题，本研究则解决了"这个AI控制的地址是否是Sybil"的检测问题。

**特征迁移**：Paper 1的8个agent识别特征直接迁移到本研究的Sybil检测器中，验证了这些特征不仅能区分AI和人类，还能区分AI Sybil和真实地址（包括真实人类用户和正常AI用户）。

**分类学指导精细度设计**：Paper 0的agent分类学——交易机器人、MEV搜索者、投资组合管理器等——为我们的AI Sybil精细度等级设计提供了依据。Basic级别对应简单交易机器人，Moderate级别对应策略感知的投资组合管理器，Advanced级别对应完全LLM编排的自治代理。

**统一框架展望**：理想的检测管道应该是两阶段的：（1）首先使用Paper 1的方法识别AI agent控制的地址；（2）然后对识别出的AI地址使用本研究的增强检测器判断Sybil意图。这种"先识别身份，再判断意图"的框架比直接的单阶段检测更高效、更可解释。

**从识别到归因**：更进一步，Paper 1的behavioral_consistency特征提供了一种从单个地址检测到Sybil集群归因的路径。如果多个地址展现出高度相关的AI行为模式——相同的gas精度分布、相同的动作序列困惑度特征、相同的错误恢复策略——那么即使这些地址在传统的资金流图上没有直接连接，也可以被归因为同一个AI操作者控制的Sybil集群。这种基于"行为指纹"而非"资金链接"的集群识别方法，可能成为下一代Sybil检测的核心范式。

**伦理考量**：需要指出的是，并非所有AI agent控制的地址都是Sybil。正常的DeFi自动化工具（如收益优化器、自动再平衡策略）也可能展现出类似的AI执行特征。因此，检测系统需要在打击Sybil攻击和保护合法自动化用户之间找到平衡。我们建议对被标记的地址提供申诉机制，允许合法的AI工具用户证明其非Sybil性质。

### 5.4 局限性

本研究存在以下局限性，需要在未来工作中解决：

**AI Sybil行为的合成性**：我们的AI Sybil行为是通过模拟生成的，而非由真实的LLM驱动的agent在主网上产生。虽然我们的模拟基于Paper 1确立的AI agent行为特征并校准到HasciDB的真实数据分布，但真实的LLM驱动Sybil攻击可能展现出我们未预见的行为模式。我们计划在后续研究中部署真实的LLM agent在测试网上生成交易，以验证模拟的有效性。

**以太坊L1的局限**：HasciDB覆盖的是以太坊L1空投项目。在L2（如Arbitrum、Optimism）和其他链（如Solana）上，gas机制、交易格式和用户行为模式可能不同，AI特有特征（特别是gas相关特征）的有效性需要进一步验证。

**AI特征提取的数据需求**：8个AI特有特征中的部分特征（如error_recovery_pattern、response_latency_variance）需要交易级别的详细数据（时间戳精度、失败交易记录等）。在某些区块链浏览器或数据提供商中，这些数据可能不容易获取或不够精确。

**LLM能力的演进**：随着大语言模型能力的持续提升，AI agent可能学会模拟人类的gas使用习惯、引入类人的时间分布模式、甚至模仿人类的错误恢复行为。这将降低当前AI特征的区分能力，要求检测方法持续演进。

**标签的循环依赖**：HasciDB的Sybil标签由5个指标的阈值规则确定性导出，这意味着标签本身是规则的函数而非独立的真实标签。虽然这些规则经过了12位Web3从业者的共识验证，但仍可能存在假阳性（正常但行为模式异常的用户被误标为Sybil）和假阴性（行为隐蔽的Sybil未被标记）。

## 6. 结论

本文首次系统量化了AI agent对加密货币空投Sybil检测系统的规避能力，并提出了有效的防御方案。我们的核心发现如下：

**第一**，现有的Sybil检测方法在面对AI agent驱动的攻击时严重失效。即使是最简单的Basic级别AI Sybil也实现了100%的规则规避率。Advanced级别的AI Sybil将5指标ML检测器的AUC从1.0000降至0.5774，接近随机猜测。这一发现对当前所有基于行为指标阈值的Sybil检测方案提出了根本性质疑。

**第二**，AI agent的交易执行层特征提供了恢复检测能力的有效信号。8个AI特有行为特征——gas精度、时间熵、行为一致性、动作序列困惑度、错误恢复模式、响应延迟方差、nonce间隔规律性、EIP-1559小费精度——将检测AUC恢复至0.9999以上。这些特征捕捉的是AI agent"如何执行"而非"做什么"，代表了一种更难被规避的正交信号维度。

**第三**，增强检测器具有优异的跨项目迁移性。在blur_s2到uniswap/eigenlayer/gitcoin的直接迁移中均达到AUC 1.0，在9种跨项目组合的pilot验证中达到mean AUC 0.884 +/- 0.048。AI执行层特征是项目无关的，不受特定项目交互模式差异的影响。

**第四**，本研究揭示了Sybil检测领域的AI军备竞赛态势。传统的"检测脚本指纹"范式已经过时，取而代之的是"检测AI执行特征"的新范式。然而，这一新范式也面临着被更先进AI agent规避的风险，推动检测和攻击双方在信号深度维度上持续博弈。

我们呼吁空投项目设计者紧急采纳AI感知的Sybil检测机制，将AI特有特征整合到资格审核管道中，并参与行业级的Sybil检测情报共享。在356万合格地址、30.57%总体Sybil率的当前生态中，传统检测方法对AI Sybil的失效意味着巨大的经济风险——如果不采取行动，随着AI agent工具的日益普及和降低的使用门槛，AI驱动的Sybil攻击将从理论威胁变为现实危机。

未来工作将沿三个方向展开：（1）在以太坊测试网上部署真实LLM驱动的agent生成交易数据，验证模拟实验的外部有效性；（2）将AI特有特征扩展到L2网络和非EVM链，探索gas机制差异对特征有效性的影响；（3）开发对抗性训练框架，使检测器能够主动预测和适应未来更先进的AI规避策略。我们将开源检测工具和数据集，促进社区协作和持续改进。

## 参考文献

1. Li, Y., Chen, X., & Cai, W. (2026). HasciDB: A Consensus-Driven Sybil Detection Framework for Cryptocurrency Airdrops. *Proceedings of the 2026 CHI Conference on Human Factors in Computing Systems (CHI'26)*.

2. Wen, A. et al. (2026). Pre-Airdrop Sybil Detection Using Temporal Behavioral Features. *Working Paper*.

3. UW Decentralized Computing Lab. ARTEMIS: Graph Neural Network-Based Sybil Detection for Blur Season 2. *GitHub: UW-Decentralized-Computing-Lab/Blur*.

4. UW Decentralized Computing Lab. LLMhunter: Multi-Expert LLM Pipeline for Sybil Classification. *GitHub: UW-Decentralized-Computing-Lab/Blur*.

5. TrustaLabs. Airdrop Sybil Identification via Graph Community Detection. *GitHub: TrustaLabs/Airdrop-Sybil-Identification*.

6. Arbitrum Foundation. Sybil Detection using Louvain Community Detection on Transfer Graphs. *GitHub: ArbitrumFoundation/sybil-detection*.

7. BrightID. SybilRank Algorithms for Social Graph-Based Sybil Resistance. *GitHub: BrightID*.

8. Hop Protocol. Sybil Attacker Elimination Lists. *GitHub: hop-protocol*.

9. Goodfellow, I. J., Shlens, J., & Szegedy, C. (2014). Explaining and Harnessing Adversarial Examples. *arXiv preprint arXiv:1412.6572*.

10. Carlini, N., & Wagner, D. (2017). Towards Evaluating the Robustness of Neural Networks. *2017 IEEE Symposium on Security and Privacy (SP)*, 39-57.

11. Apruzzese, G., et al. (2023). The Role of Machine Learning in Cybersecurity. *Digital Threats: Research and Practice*, 4(1), 1-38.

12. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.

13. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *Advances in Neural Information Processing Systems*, 30, 3146-3154.

14. Hamilton, W., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. *Advances in Neural Information Processing Systems*, 30, 1024-1034.

15. Blondel, V. D., et al. (2008). Fast unfolding of communities in large networks. *Journal of Statistical Mechanics: Theory and Experiment*, 2008(10), P10008.

16. Cao, Q., et al. (2012). Aiding the Detection of Fake Accounts in Large Scale Social Online Services. *Proceedings of the 9th USENIX Conference on Networked Systems Design and Implementation (NSDI'12)*, 197-210.
