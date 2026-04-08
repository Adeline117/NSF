# Formal Statement of the C1-C4 Agent Definition

This document formalizes the four necessary and sufficient conditions
from `shared/definition.md` and provides non-redundancy arguments
(counterexample-based) for each condition.

## 1. Formal definition

**Definition (On-chain AI Agent).** Let $E$ be a software entity and let
$B$ be the sequence of blockchain events associated with $E$ over a
time window $[t_0, t_1]$. Then $E$ is an **on-chain AI agent** iff it
satisfies all four conditions:

$$
\text{Agent}(E) \iff C_1(E) \land C_2(E) \land C_3(E) \land C_4(E).
$$

### C1 (On-chain actuation)

$$
C_1(E) \iff \text{Control}(E, \alpha) \text{ where } \alpha \text{ is an EOA and } \exists\, \text{tx} \in B : \text{tx.from} = \alpha.
$$

$E$ directly or indirectly controls an externally-owned account
$\alpha$ and $\alpha$ initiates at least one transaction in the
observation window.

### C2 (Environmental perception)

$$
C_2(E) \iff I(\text{state}_{t}; \text{action}_{t+1}) > 0
$$

where $I$ is the mutual information between the chain state at time
$t$ (and any off-chain signals observable by $E$) and the action
parameters of the next transaction. Operationally:
- `calldata_entropy > 0` (diverse method calls → reaction to environment)
- `event_driven_latency < scheduled_latency` (faster reaction than a
  fixed cron) OR
- `correlation(tx_time, on_chain_event_time) > 0`

### C3 (Autonomous decision-making)

$$
C_3(E) \iff H(\text{action} \mid \text{environment}) > 0 \land \neg\text{human-gated}(E).
$$

The conditional entropy of actions given environment is positive
(non-deterministic) AND each individual transaction is NOT preceded
by a per-transaction human approval signal. Operationalized as:
- `tx_interval_cv > threshold_cv` (non-deterministic timing)
- `active_hour_entropy > 2.5` OR `burst_ratio > 0.05`
  (non-circadian, or humanly-impossible burst patterns)

### C4 (Adaptiveness)

$$
C_4(E) \iff \exists\, \text{window } w : \text{KS-dist}(\text{params}_{t \leq w}, \text{params}_{t > w}) > \delta.
$$

Some behavioral-parameter distribution shifts significantly between
two time windows (Kolmogorov-Smirnov distance above threshold).
Operationalized via:
- `gas_strategy_drift` > 0.15 between early and late windows
- `target_set_jaccard` shows > 0.2 change between windows
- `strategy_switch_points` detectable via changepoint detection

## 2. Non-redundancy arguments

Each $C_i$ excludes entities that satisfy the other three.

### $C_1$ is non-redundant: smart contract

A pure AMM smart contract (e.g., Uniswap V3 Pool) satisfies:
- $C_2$: reacts to calldata from external callers
- $C_3$: its pricing curve has non-deterministic-looking outputs
  given varied input
- $C_4$: fee tier can be governance-adjusted

But it fails $C_1$: the contract does not control an EOA; it is
only callable. Without $C_1$, the definition would include every
non-trivial contract.

### $C_2$ is non-redundant: fully off-chain entity relaying on-chain
transactions with no awareness

An entity that signs pre-computed batches and submits them at fixed
intervals without reading chain state could satisfy:
- $C_1$: has an EOA and submits txs
- $C_3$: tx intervals vary around a mean (non-zero CV) because of
  mempool latency
- $C_4$: its tx sizes drift due to external queue dynamics

But fails $C_2$ because the tx parameters are not conditioned on
chain state. Without $C_2$, every scheduled transaction relayer
would be classified as an agent.

### $C_3$ is non-redundant: deterministic rule-based trading bot

A grid trading bot (e.g., fixed-interval DCA) satisfies:
- $C_1$: EOA submitting txs
- $C_2$: reads price before each trade
- $C_4$: grid width can be parameter-updated by owner

But fails $C_3$ because given the same price, it always executes
the same trade (no non-determinism) and it may be gated by a
human-approved parameter update. Without $C_3$, DCA bots would
be classified as agents.

### $C_4$ is non-redundant: stateless MEV sandwich bot that never
updates its strategy

A bot that executes the exact same sandwich logic from deployment
to deprecation satisfies:
- $C_1$: EOA
- $C_2$: reads mempool events (environment)
- $C_3$: reacts within a block, varying which mempool tx to target
  (non-deterministic selection from the mempool stream)

But fails $C_4$ if its sandwich parameters (slippage, gas,
target selection algorithm) never change over the observation
window. Without $C_4$, frozen-strategy bots would be classified
as agents, which misses a key characteristic of *intelligent*
agency — the ability to learn from experience.

## 3. Observability and label quality

Each condition is observable from on-chain data with varying levels
of robustness:

| Condition | Robustness | Observability proxy | False positive sources |
|-----------|-----------|---------------------|-----------------------|
| $C_1$ | high | `eth_getCode`, outgoing tx count | multisig-controlled contracts |
| $C_2$ | medium | calldata entropy, event-driven latency | pre-computed batched txs |
| $C_3$ | low | tx_interval_cv, hour_entropy, burst_ratio | heavy human DeFi users |
| $C_4$ | low | distributional drift, KS-distance | external market regime shifts |

**Critical caveat — label leakage.** In Paper 1 we discovered that
using the $C_3$ indicators (`tx_interval_cv`, `hour_entropy`,
`burst_ratio`) as labeling gates AND as classifier features produces
direct target leakage. The formal definition above is independent of
this: $C_3$ requires the CONCEPT of non-determinism and non-human-
gating, not the specific proxies. For Paper 1 training, labels must
be derived from **external provenance** (platform registry membership,
Flashbots list, ENS name resolution) and the 23 classifier features
should be used only for TESTING how well the concept is recoverable
from behavior, not for DEFINING the label.

## 4. Relation to observed clusters

Paper 0's cluster validation (P0-A) finds that on the 23-feature
behavioral matrix, K-means at $k=8$ has mean purity 0.78 vs the
rule-based taxonomy labels, while $k=3$ gives the best silhouette.
The coarse 3-cluster structure corresponds roughly to:

1. **Deterministic script / Simple bot** ($C_1 \land C_2 \land \neg C_3 \land \neg C_4$)
2. **Active but non-adaptive** ($C_1 \land C_2 \land C_3 \land \neg C_4$)
3. **Fully adaptive agent** ($C_1 \land C_2 \land C_3 \land C_4$)

The 8-category taxonomy refines cluster 3 into MEV-searcher /
DeFi-management / LLM-powered / DAO / RL subtypes by semantic
provenance, not by behavioral distinguishability. This is itself
a publishable finding: **behavioral taxonomies saturate at 3 classes;
semantic taxonomies require external provenance signals**.
