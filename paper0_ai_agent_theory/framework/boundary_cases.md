# Boundary Cases for the C1-C4 Agent Definition

This document catalogues entity types that sit on or near the boundary
of the agent definition, explaining which conditions they satisfy and
why they land inside or outside the definition.

## Clear non-agents

### Smart contracts (AMMs, routers, vaults)
- $C_1$: **FAIL** — a smart contract cannot initiate transactions (only
  be called); it is not an EOA.
- All other conditions moot.

### Human users (wallet-operated)
- $C_1$: PASS
- $C_2$: PASS (humans observe prices, etc.)
- $C_3$: **FAIL** — every transaction is approved by a human via
  wallet prompt; behavior follows circadian rhythm.
- $C_4$: Variable (humans adapt strategies over time).

### Simple deterministic bots (DCA, grid bots)
- $C_1$: PASS
- $C_2$: PASS (reads price before each trade)
- $C_3$: **FAIL** — given the same input the bot always executes the
  same output; no randomization.
- $C_4$: Often fails because parameters are hand-tuned by owner
  between deploys.

### Exchange hot wallets
- $C_1$: PASS (EOA submitting withdrawals)
- $C_2$: PASS (reacts to deposits/withdrawals)
- $C_3$: **FAIL** — each transaction is human- or policy-gated
  (withdrawal approvals, batching scripts).
- $C_4$: Variable.

### Cron-triggered deterministic scripts
- $C_1$: PASS
- $C_2$: **FAIL** if the script only reads the clock, not chain state.
  PASS if it reads price.
- $C_3$: **FAIL** — fixed schedule, no variability.
- $C_4$: **FAIL** — parameters frozen.

## Clear agents

### MEV searchers (jaredfromsubway, Flashbots searchers)
- $C_1$: PASS
- $C_2$: PASS (mempool monitoring is intense environmental perception)
- $C_3$: PASS (decision depends on mempool state, which is
  inherently unpredictable)
- $C_4$: PASS (tactics evolve — sandwich → atomic arb → liquidation
  over weeks as markets change)

### Autonolas DeFi management agents
- $C_1$: PASS (service EOAs registered in AgentRegistry)
- $C_2$: PASS (reads price oracles, TVL, APR)
- $C_3$: PASS (LLM or policy network chooses next action)
- $C_4$: PASS (strategies update via agent lifecycle)

### LLM-powered agents (AI16Z/Eliza, Virtuals)
- $C_1$: PASS
- $C_2$: PASS (prompt context includes chain state + news)
- $C_3$: PASS (LLM output is non-deterministic by construction)
- $C_4$: PASS (fine-tuning, memory, prompt evolution)

### RL trading agents (Giza, Sanctum)
- $C_1$: PASS
- $C_2$: PASS (policy observes state)
- $C_3$: PASS (stochastic policy)
- $C_4$: PASS (policy updates via gradient descent)

### Autonomous DAO agents (collectively controlled)
- $C_1$: PASS (multisig-gated EOA OR contract with execution module)
- $C_2$: PASS (reads governance proposals, treasury state)
- $C_3$: Collective autonomy — DAO decides by vote, no single human
  approves each tx
- $C_4$: PASS (governance changes over time)

## True boundary cases

### Cross-chain bridge relayers
- $C_1$: PASS (EOA submitting cross-chain transfers)
- $C_2$: PASS (monitors both chains)
- $C_3$: **PARTIAL** — deterministic relay logic (if finalized, relay)
  but timing can be stochastic
- $C_4$: **PARTIAL** — parameter tuning happens via off-chain operator
  updates

Verdict: Boundary. Paper 0 labels these as category 5 (Cross-Chain
Bridge Agent) with a 0.5 confidence weight.

### Account abstraction (ERC-4337) smart wallets with automation
- $C_1$: TRICKY — the smart wallet is NOT an EOA, but the bundler
  that submits on behalf of it IS
- $C_2$: PASS (reads chain state to decide when to execute)
- $C_3$: PASS if the automation logic uses learned policy
- $C_4$: Variable

Verdict: Re-examine $C_1$. The **signer of the user operation** is an
EOA (the bundler or a session key), so $C_1$ is satisfied via the
bundler. The smart wallet itself is a tool used BY the agent, not the
agent itself.

### Flashloan-based arbitrage bots
- $C_1$: PASS
- $C_2$: PASS
- $C_3$: PASS (opportunistic)
- $C_4$: TRICKY — many use the same exact contract template deployed
  dozens of times. Each instance is "frozen" but the pool of instances
  evolves.

Verdict: Agent, if the $C_4$ window is wide enough to include the
evolution of the instance pool.

### Governance voting bots (Snapshot vote delegators, auto-voters)
- $C_1$: PASS
- $C_2$: PASS (reads proposals)
- $C_3$: PARTIAL — many follow a fixed delegation (vote with X) which
  makes the decision fully determined by X's choice
- $C_4$: Usually fails — delegation doesn't change

Verdict: Non-agent if it just auto-forwards a delegate's choice;
agent if it has its own voting policy with learning.

### Serial Sybil bots (one operator, many EOAs)
- $C_1$: PASS per EOA
- $C_2$: PASS (reads airdrop criteria)
- $C_3$: **FAIL usually** — all EOAs execute the same playbook from
  the same operator
- $C_4$: **FAIL** — playbook is static

Verdict: NOT an agent in our definition. The coordinating operator
may be an agent, but each individual Sybil EOA is a scripted entity.
**This is important for Paper 3: AI sybils violate C3 because their
behavior is coordinated by a single controller.**

### Liquidation bots
- $C_1$: PASS
- $C_2$: PASS
- $C_3$: PASS (race condition: only one bot wins each liquidation,
  so the outcome depends on gas bidding strategy)
- $C_4$: PASS in practice (gas strategies evolve)

Verdict: Agent.

## Explicit exclusions in this paper's scope

The following entities are intentionally excluded from our Paper 1
training corpus even though they could nominally satisfy C1-C4:
- **Exchange wallets**: high tx volume distorts feature stats
- **Multisig signers**: $C_3$ is collective, hard to assess from a
  single signer's perspective
- **Deployer addresses**: typically used for 1-2 txs, insufficient
  data for $C_2$-$C_4$ measurement
- **Vanity addresses, airdrop-only accounts**: $C_4$ cannot be
  measured on $< 20$ txs

## Impact on labeling

Any training set for Paper 1 should **explicitly exclude** the
following categories:
1. Smart contract addresses (fail $C_1$, detected via `eth_getCode`)
2. Known exchange hot wallets (Binance, Coinbase, etc. — maintain
   a blocklist in `labeling_config.py`)
3. Addresses with fewer than 20 transactions
4. Addresses with only inbound txs (fail $C_1$ operationally)
5. Cross-chain bridges when analyzing them on a single chain

This matches the `EXCLUDE` bucket in Paper 1's labeling pipeline.
