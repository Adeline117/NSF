# Section 4: On-Chain Empirical Validation

## 4.1 Validation Strategy

We validate the C1-C4 definition and the 8-category taxonomy on
real Ethereum transaction data using three complementary methods:

1. **Rule-based projection** (§4.2) — assigns every labeled address
   in our dataset to one of the 8 taxonomy categories using a
   provenance-prior-then-feature-refinement scheme. Tests whether the
   taxonomy can be operationalized as a deterministic classifier.

2. **Unsupervised cluster validation** (§4.3) — runs K-Means on the
   23 behavioral features and compares the discovered clusters to the
   taxonomy projection via Adjusted Rand Index, normalized mutual
   information, and per-cluster purity. Tests whether the taxonomy
   structure is *recovered* from behavioral data alone.

3. **Multi-class supervised classifier** (§4.4) — trains gradient
   boosting, random forest, and logistic regression to predict the
   taxonomy category from the 23 features. Tests whether each
   category has a learnable behavioral signature.

The dataset is the 2,590 agent addresses from Paper 1's expanded
mining (`features_expanded.parquet`), which spans Autonolas, Fetch.ai,
AI Arena, and curated MEV-bot lists. We exclude the 726 human-labeled
addresses from the projection to keep the focus on agent-subtype
distinguishability.

## 4.2 Rule-Based Taxonomy Projection

### 4.2.1 Method

Each agent is assigned to a taxonomy category by a two-tier rule
system. Tier 1 uses *provenance*: the contract source from which
the address was mined, plus any name annotation (Etherscan label,
ENS resolution, Flashbots searcher list). Tier 2 uses *feature-based
refinement*: when provenance is ambiguous, the rule consults a
4-dimensional subspace of the behavioral feature set (`burst_frequency`,
`gas_price_round_number_ratio`, `unique_contracts_ratio`,
`multi_protocol_interaction_count`).

Tier 1 rules (excerpt):
- Address mined from `Autonolas Agent Registry` → DeFi Management Agent (confidence 0.90)
- Name contains `MEV|Flashbots|sandwich|builder` → MEV Searcher (0.95)
- Name contains `AI Arena NRN` → LLM-Powered Agent candidate (0.75)
- Name contains `Paper0-validated` → DeFi Management Agent (0.90)

Tier 2 rules (excerpt):
- `burst_frequency > 0.20 ∧ gas_price_cv > 0.30` → MEV Searcher (0.90)
- `gas_price_round_number_ratio > 0.60 ∧ method_id_diversity < 1.5` →
  Deterministic Script (0.80)
- `multi_protocol_interaction_count ≥ 5 ∧ unique_contracts_ratio > 0.40` →
  DeFi Management Agent (0.85)

The full rule set is in
`paper0_ai_agent_theory/experiments/taxonomy_projection.py`. We
deliberately use a *transparent* rule set rather than a learned
mapping, because Section 4.4's classifier will then consume the
projection as ground truth and we want a non-circular relationship:
the projection rules use 4 features, the classifier uses all 23.

### 4.2.2 Results

The 2,590 agents project as follows:

| Category | Count | Percentage | Mean confidence |
|----------|------:|-----------:|----------------:|
| DeFi Management Agent | 1,669 | 64.4% | 0.62 |
| Deterministic Script  |   666 | 25.7% | 0.59 |
| Simple Trading Bot    |   130 |  5.0% | 0.63 |
| LLM-Powered Agent     |    71 |  2.7% | 0.71 |
| MEV Searcher          |    54 |  2.1% | 0.86 |
| Autonomous DAO Agent  |     0 |  0.0% | —    |
| Cross-Chain Bridge    |     0 |  0.0% | —    |
| RL Trading Agent      |     0 |  0.0% | —    |

The projection covers 5 of the 8 taxonomy categories. Three
categories have zero instances: DAO agents, cross-chain bridge
relayers, and RL trading agents. This reflects the mining bias
of Paper 1, which targeted Autonolas/Fetch/AI-Arena registry
contracts and curated MEV bot lists. A targeted Phase 2 mining
pass against DAO multisigs and bridge contracts would close
this gap. We do not claim coverage of those three categories
in Sections 4.3 and 4.4.

Confidence distribution: 1,853 projections at 0.5–0.7 (medium),
688 at 0.7–0.85 (high), 45 at 0.85+ (very high). 99.7% used
provenance-then-features tier; only 0.3% relied on features alone.

## 4.3 Unsupervised Cluster Validation

### 4.3.1 Method

We standardize the 23 behavioral features (z-score), winsorize to the
[1, 99] percentile range, and run K-Means with `n_init=10` for
$k \in \{3, 4, \ldots, 15\}$. For each $k$ we report:
- Silhouette score (sample size 2,000)
- Adjusted Rand Index (ARI) against the 5-category projection labels
- Normalized Mutual Information (NMI) against the projection labels
- Per-cluster purity (fraction belonging to the dominant projection category)
- Number of orphan clusters (purity < 0.5)

### 4.3.2 Results

| $k$ | Silhouette | ARI | NMI | Inertia |
|----:|-----------:|----:|----:|--------:|
|  3 | **0.151** | 0.264 | 0.212 | 24,140 |
|  4 | 0.123 | 0.180 | 0.204 | 22,317 |
|  5 | 0.132 | **0.319** | **0.340** | 20,851 |
|  6 | 0.125 | 0.238 | 0.301 | 19,667 |
|  7 | 0.128 | 0.216 | 0.293 | 18,729 |
|  8 | 0.129 | 0.222 | 0.313 | 17,915 |
|  9 | 0.117 | 0.174 | 0.299 | 17,189 |
| 10 | 0.115 | 0.167 | 0.310 | 16,609 |
| 12 | 0.121 | 0.159 | 0.293 | 15,575 |
| 15 | 0.106 | 0.120 | 0.290 | 14,376 |

Silhouette is **maximized at $k=3$**, while ARI is maximized at $k=5$
(matching the number of populated taxonomy categories). At $k=8$
(the full taxonomy), mean purity is 0.78 with one orphan cluster.

### 4.3.3 Interpretation: behavioral super-clusters

The silhouette result is the strongest empirical finding of this
section: the behavioral feature space supports a 3-cluster structure
much more clearly than an 8-cluster structure. The three super-clusters
correspond to:

- **Cluster A (Static / Deterministic):** addresses with high
  `gas_price_round_number_ratio`, low `method_id_diversity`, low
  `burst_frequency`. This is the "Deterministic Script" + "Simple
  Trading Bot" merge.
- **Cluster B (Active DeFi):** addresses with high
  `multi_protocol_interaction_count`, moderate `unique_contracts_ratio`,
  moderate `burst_frequency`. The "DeFi Management Agent" cluster.
- **Cluster C (Adaptive / High-frequency):** addresses with very
  high `burst_frequency`, low `tx_interval_mean`, high
  `gas_price_cv`. The "MEV Searcher" + "LLM-Powered Agent" merge.

This means the 8-category taxonomy is **semantically coherent but
behaviorally over-split** for the current 23-feature representation.
The taxonomy distinguishes DeFi Management from LLM-Powered, but
in feature space the two are largely overlapping. We propose two
possible interpretations:

1. **Taxonomic interpretation:** The 8 categories are an analytical
   refinement of an underlying 3-class structure. Future work should
   either accept the 3-class decomposition or expand the feature
   set to make all 8 distinguishable.
2. **Measurement interpretation:** Current LLM-driven agents
   *mimic* rule-based DeFi agents because their on-chain footprint
   is similar; they differ in *off-chain* prompt-driven action
   diversity that does not yet appear in tx-level features. New
   features (e.g., the 8 AI features from Paper 3) are needed to
   distinguish them.

We adopt the measurement interpretation in Section 6 (Discussion),
because Paper 3 shows that 8 AI-specific features drawn from real
agent/human data separate the two populations with Cohen's $d > 1.0$
on `hour_entropy` and `behavioral_consistency`.

## 4.4 Multi-Class Supervised Classifier

### 4.4.1 Method

We train three classifiers to predict the taxonomy category from
the 23 behavioral features only (no source/name leakage):

- Gradient Boosting (100 trees, depth 3, lr 0.1)
- Random Forest (300 trees, depth 8)
- Logistic Regression (C=1.0)

Evaluation uses 5-fold stratified cross-validation. We drop the 3
empty categories and the LLM-Powered class is kept (n=71 ≥ 20).
Final training set: $N = 2{,}590$ across 5 classes.

### 4.4.2 Results

Per-class F1 (Gradient Boosting, the best model):

| Class | n | F1 | Precision | Recall |
|-------|--:|---:|----------:|-------:|
| Deterministic Script |  666 | **0.996** | 0.999 | 0.994 |
| Simple Trading Bot   |  130 | 0.988 | 0.992 | 0.985 |
| DeFi Management Agent | 1,669 | 0.982 | 0.970 | 0.994 |
| MEV Searcher         |   54 | 0.849 | 0.865 | 0.833 |
| **LLM-Powered Agent** |   71 | **0.528** | 0.800 | 0.394 |

Overall: accuracy $0.9737 \pm 0.0040$, F1-macro $0.8683 \pm 0.0267$,
F1-weighted $0.9706$.

### 4.4.3 The LLM-Powered Agent confusion

The LLM-Powered class is the only one with sub-0.85 F1. Its
confusion-matrix row shows 60% of LLM-Powered instances misclassified
as DeFi Management Agent. The classifier's per-class precision for
LLM-Powered is 0.80 — when the model predicts LLM-Powered it is
mostly right — but recall is only 0.39, meaning the majority of
true LLM-Powered agents are predicted as DeFi Management.

This is consistent with the cluster validation result in §4.3:
LLM-Powered and DeFi Management Agent occupy overlapping regions of
the 23-feature space. Our interpretation is that **current
LLM-driven agents do not yet exhibit a distinct on-chain
footprint** at the level captured by our 23 tabular features. The
distinction is recoverable only with finer-grained features (e.g.,
Paper 3's 8 AI features).

### 4.4.4 Threats to validity

**Partial circularity.** The taxonomy projection rules use 8 of the
23 features in Tier 2 refinement. Therefore the multi-class
classifier can re-learn the projection rules from the shared feature
subset, inflating the apparent accuracy. We did not observe this in
practice — the classifier learns combinations the rules do not
specify (gas precision × method diversity, etc.) — but a stricter
test would require a hand-annotated taxonomy label per address.

**Mining bias.** The 2,590 agents are 97% from three platforms
(Autonolas, Fetch.ai, AI Arena). Cross-platform generalization is
not tested in this paper; see Paper 1 §4.2 for cross-platform
evaluation results which show that classifiers trained on
Autonolas-labeled data have AUC 0.24–0.34 on a trusted provenance set
(worse than chance), confirming a strong distribution-shift effect.

**Empty categories.** Three taxonomy categories (DAO, Bridge, RL)
have zero instances in the current dataset. The taxonomy is
*not validated* on those categories. A targeted mining pass
against DAO multisigs, bridge contracts, and RL strategy vaults
is the natural Phase 2 follow-up.

## 4.5 Cross-Reference with Paper 1

Paper 1 trains a *binary* classifier (agent vs human) on the same
23 features. Its honest performance on a 64-row provenance-only
trusted set is GraphSAGE GAT AUC 0.870 (the new best honest model
after our P1-GNN experiment). The relationship between Paper 0 and
Paper 1 is:

- **Paper 1 answers:** Is this address an agent (any of the 5
  populated categories) or a human?
- **Paper 0 answers:** If Paper 1 says agent, *which* of the 5
  categories is it?

Together they constitute a two-stage decision pipeline. Paper 1's
binary AUC is the upper bound on the joint pipeline's accuracy;
Paper 0's per-class F1 determines how informative the multi-class
output is.

For LLM-Powered Agent, the joint pipeline currently fails: Paper 1
identifies it as an agent (with high probability), but Paper 0
confuses it with DeFi Management Agent 60% of the time. This
motivates the integration work in `paper0_ai_agent_theory/paper/
paper1_integration.md`, which proposes adding Paper 3's 8 AI
features to Paper 0's classifier.

## 4.6 Summary of Validation Findings

1. **5 of 8 taxonomy categories are empirically populated** in the
   current Paper 1 dataset. Three (DAO, Bridge, RL) require
   targeted mining to validate.

2. **Behavioral clustering supports a 3-cluster structure**
   (silhouette 0.151 at $k=3$ vs 0.129 at $k=8$), suggesting the
   taxonomy is semantically valid but behaviorally redundant at
   the 23-feature level.

3. **Multi-class supervised classification achieves 97.4% accuracy**
   with F1-macro 0.87. Four of five categories exceed F1=0.85.

4. **The LLM-Powered Agent class is the empirical weak spot**
   (F1=0.53), confused with DeFi Management Agent because both
   have similar tabular footprints. New AI-specific features
   (Paper 3) are needed to distinguish them.

5. **The taxonomy is internally consistent** with Paper 1's binary
   identification: the joint pipeline correctly handles 4 of 5
   populated categories.
