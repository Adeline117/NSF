# Prior Art Audit: Tool-Chain Prompt Injection (TCPI)

**Audit date:** 2026-04-15
**Conclusion:** No prior work formalizes, proves undecidability for, *or* demonstrates on-chain the exact class of attack we call **Tool-Chain Prompt Injection (TCPI)** — namely, the attacker-controlled flow from tool `T_A`'s *output* into tool `T_B`'s *parameters* through an LLM orchestrator, treated as a formal security property of multi-tool agent systems. Existing work is adjacent but each misses at least one of: (i) the specific Output→Parameter threat model, (ii) a decidability-theoretic lower bound, (iii) a physically-executed on-chain demonstration.

The table below records what each closest work does, and the gap TCPI fills.

## Closest prior work, item-by-item

| Work | Threat model | Multi-hop Output→Parameter? | Formal theorem? | On-chain demo? | Gap TCPI fills |
|---|---|---|---|---|---|
| Greshake et al. 2023 "Not what you've signed up for" (2302.12173) | Attacker poisons **retrieved documents** that land in context window | Single-hop: document → LLM behavior | No | No | Does not model tool→tool parameter flow; no formal detectability result |
| Zhan et al. 2024 "InjecAgent" (2403.02691, ACL Findings 2024) | Empirical benchmark of 1,054 IPI cases; attacker tool returns instructions | Partial: attacker tool → "direct harm" / "data exfil" tool, but the second tool is also attacker-picked, not a benign tool whose parameters are hijacked | No (empirical only) | No | No theorem; no formal Output→Parameter separation from single-hop; no on-chain |
| Debenedetti et al. 2024 "AgentDojo" (2406.13352, NeurIPS D&B 2024) | Dynamic eval environment, 97 tasks, 629 cases; data returned by external tools "hijacks" the agent | Mixed: framework supports it but no formalization of the specific flow class | No | No | Benchmark, not theory; no undecidability result; no on-chain contracts |
| Invariant Labs "Tool Poisoning Attacks" (2025 blog) | Malicious instructions embedded in **tool descriptions / metadata** | No — description channel, not output channel | No | No | Different attack class entirely (describes T_A's *signature*, not its *output*) |
| Invariant Labs "Toxic Flow Analysis" (2025 blog) | "Lethal trifecta": untrusted instructions + sensitive data + exfil channel | Yes informally: models "flows" across tools for the trifecta pattern | No — explicitly an "early preview" blog, no theorems | No | **Closest competitor.** TFA is informal/engineering-oriented. TCPI contributes (a) formal definition, (b) undecidability theorem, (c) variant taxonomy beyond the trifecta, (d) on-chain demo |
| Shi et al. 2025 "ToolHijacker" (2504.19793) | Inject malicious **tool document** into tool library to steer tool **selection** | No — hijacks which tool is *chosen*, not a benign tool's parameters | No | No | Selection channel, not parameter channel |
| Lee et al. 2024 "Prompt Infection" (2410.07283) | **LLM-to-LLM** viral propagation in multi-AGENT systems | Agent-to-agent, not tool-to-tool | No | No | Different system model (many LLMs, one tool each) vs. TCPI's (one LLM, many tools) |
| Lin et al. 2025 "STAC" (2509.25624) | Sequential tool attack chains — innocuous steps collectively harmful | User-driven multi-turn, not output-driven parameter flow | No (pipeline + 483 cases) | No | User chooses the sequence; TCPI's attacker controls it by writing T_A's output |
| WithSecure Labs "Multi-Chain Prompt Injection" | **LangChain** pipelines (prompt → LLM → parse → prompt) | Chain = LangChain stages, not tool-to-tool | No | No | Different "chain": LLM-pipeline stages, not tool calls |
| AgentArmor (2508.01249) | Runtime PDG over agent traces; detects cross-resource 2-hop pollution | Yes — detects some Output→Parameter flows via runtime PDG + taint | No theorem; empirical 3% ASR | No | **Rebuttal target.** TCPI's theorem shows that *per-tool static analysis* cannot detect TCPI in general. AgentArmor is *runtime PDG*, not per-tool static, and relies on observing the attack; TCPI theorem does not contradict it but clarifies why static pre-deployment scanning is insufficient |
| Causality Laundering (2604.04035) | Denial-feedback leakage via cross-resource laundering | Adjacent multi-step laundering | No theorem | No | Different semantic channel (denials); also empirical |
| CrAIBench (referenced) | Web3 benchmark, 150 tasks, 500 attack cases, context manipulation | Empirical | No | Uses simulated web3 tooling, but not a live testnet ERC20 transfer demo | No theorem; no live testnet settlement |
| A1 / EVMbench / "Prompt-to-Pwn" | LLM agents generating **smart contract exploits** | Different direction: LLM *attacks* contracts | No theorem about agent-tool-injection | Yes, for contract exploitation | Orthogonal: these weaponize LLMs against contracts; TCPI weaponizes contracts/tools against LLMs |

## Consolidated gap statement

Every prior line of work falls into exactly one of five buckets:

1. **Description-channel** (Invariant tool poisoning, ToolHijacker) — wrong channel.
2. **Data-channel into user context** (Greshake, AgentDojo single-hop) — single-hop, not tool-to-tool parameter flow.
3. **User-driven sequential chains** (STAC) — the *user* or attacker-as-user picks the sequence, not a tool's output.
4. **Multi-agent LLM-to-LLM** (Prompt Infection) — different system shape.
5. **Empirical-only multi-hop** (InjecAgent, AgentDojo multi-hop subset, AgentArmor, Toxic Flow Analysis) — correct shape, but no formal model, no undecidability theorem, no on-chain settlement demo.

TCPI is the first work to simultaneously:

- (A) **Formally isolate** the Output→Parameter class as distinct from (1)-(4).
- (B) **Prove** it is undetectable by any per-tool static analyzer via reduction from program equivalence.
- (C) **Enumerate** seven structurally distinct variants (Output→Parameter, Output→Tool-Selection, Output→State, Cross-Model Handoff, Multi-Turn Persistence, Reflexive, Schema-Shaped).
- (D) **Demonstrate** end-to-end on a public testnet (Base Sepolia) with real ERC-20 settlement — no prior attack paper in this family has moved from simulated sandboxes to a live on-chain demo where funds actually move.

The closest competitor is **Invariant Labs' Toxic Flow Analysis**; TCPI supersedes it by providing the formal-semantic backbone (definition + theorem) that TFA explicitly lacks, and by exhibiting a live on-chain demo that makes the impact physical rather than hypothetical.

## Citations captured for related work section

- arXiv:2302.12173 — Greshake et al., "Not what you've signed up for"
- arXiv:2403.02691 — Zhan et al., "InjecAgent" (ACL Findings 2024)
- arXiv:2406.13352 — Debenedetti et al., "AgentDojo" (NeurIPS D&B 2024)
- arXiv:2504.19793 — "ToolHijacker"
- arXiv:2410.07283 — "Prompt Infection"
- arXiv:2509.25624 — "STAC"
- arXiv:2508.01249 — "AgentArmor"
- arXiv:2604.04035 — "Causality Laundering"
- arXiv:2507.05558 — "AI Agent Smart Contract Exploit Generation" (A1)
- Invariant Labs blog — "MCP Security Notification: Tool Poisoning Attacks"
- Invariant Labs blog — "Toxic Flow Analysis"
- WithSecure Labs — "Multi-Chain Prompt Injection Attacks"
- OWASP — MCP03:2025 Tool Poisoning; LLM01:2025 Prompt Injection
