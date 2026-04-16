# TCPI — Tool-Chain Prompt Injection

P2 greatness-path artifact for the paper2 line. This directory formalizes a
specific prompt-injection class — **Tool-Chain Prompt Injection (TCPI)** —
and establishes it as novel, undecidable-for-per-tool-static-analysis, and
live-on-testnet demonstrable.

## Contents

| File | What it contains |
|---|---|
| `prior_art.md` | Full prior-art audit. Matrix of 12 closest works, gap analysis, citation list. |
| `formal_definition.md` | System model Σ = (M, T, S, U, Π, Ω); adversary taxonomy; Definition 1 of TCPI; distinctness from six adjacent classes. |
| `theorem_undecidability.md` | **Theorem 1**: TCPI-SOURCE is undecidable per-tool. Proof by reduction from Rice / program equivalence. |
| `variants.md` | V1–V7: seven structurally distinct TCPI patterns with a dimension matrix and pairwise non-reducibility. |
| `harness.py` | Runnable test harness. Implements V1–V7 as minimal scenarios; mock + Anthropic backends; TCPI vs. single-hop vs. null baselines. |
| `demo_design.md` | Design for the on-chain Base Sepolia demo: `MaliciousOracle.sol` + `TokenSwap.sol` + MCP server + Claude client. |
| `tcpi_results.json` | Harness output (mock backend, 200 rollouts/cell). |

## Headline claims

1. **Novelty.** No prior work combines (a) a formal Output→Parameter threat model, (b) a decidability-theoretic lower bound, (c) an on-chain executable demo. Closest prior works — Invariant Labs' Toxic Flow Analysis, AgentArmor, InjecAgent, AgentDojo — each miss at least one of these. See `prior_art.md`.

2. **Undecidability.** Theorem 1: any per-tool static analyzer that decides whether a given tool can act as the output-source in a successful TCPI attack would decide program equivalence, which is undecidable. See `theorem_undecidability.md`.

3. **Seven variants, all distinct.** V1 Output→Parameter, V2 Output→Tool-Selection, V3 Output→State, V4 Cross-Model Handoff, V5 Multi-Turn Persistence, V6 Reflexive, V7 Schema-Shaped. See `variants.md`.

4. **Reproducible harness.** `harness.py` drives mock and Anthropic backends against all seven variants. With the mock backend (calibrated to the ASR ordering reported by InjecAgent / AgentDojo) TCPI ASR is 44–75% while single-hop baseline is ≤4% — the expected qualitative signature of TCPI.

## Harness output (mock backend, n=200, seed=7)

| variant | TCPI ASR | single-hop ASR | null ASR | Δ |
|---|---|---|---|---|
| V1 Output→Param       | 0.660 | 0.000 | 0.000 | +0.660 |
| V2 Output→Tool-Sel    | 0.535 | 0.000 | 0.000 | +0.535 |
| V3 Output→State       | 0.605 | 0.040 | 0.000 | +0.565 |
| V4 Cross-Model        | 0.750 | 0.000 | 0.000 | +0.750 |
| V5 Multi-Turn         | 0.465 | 0.040 | 0.000 | +0.425 |
| V6 Reflexive          | 0.440 | 0.000 | 0.000 | +0.440 |
| V7 Schema-Shaped      | 0.610 | 0.000 | 0.000 | +0.610 |

## Reproduce

```bash
cd paper2_agent_tool_security/tcpi
python3 harness.py --n 200 --seed 7            # mock backend
python3 harness.py --backend anthropic --model claude-opus-4-5 --n 20
```

## What's novel vs. what's prior art

**Novel (TCPI contribution).**
- Formal isolation of the Output→Parameter class as distinct from description poisoning, user-prompt injection, tool-selection hijack, and LLM-to-LLM infection (`formal_definition.md`, §5).
- Undecidability theorem for per-tool static analysis (`theorem_undecidability.md`).
- Variants V4 (Cross-Model Handoff), V6 (Reflexive), V7 (Schema-Shaped) appear to be new threat surfaces not named in any prior work we found.
- On-chain settlement demo design (`demo_design.md`) — no prior TCPI-family paper has exhibited a live public-testnet attack where funds actually move driven by a benign-prompt agent + poisoned tool output.

**Prior art acknowledged.**
- Greshake 2023 established indirect prompt injection as a concept (single-hop, via retrieved documents).
- InjecAgent (Zhan 2024) and AgentDojo (Debenedetti 2024) built benchmarks covering adjacent empirical territory.
- Invariant Labs' "Toxic Flow Analysis" blog informally gestures at multi-tool flows (the "lethal trifecta") but without formal definition, theorem, or on-chain demo.
- AgentArmor provides a runtime PDG-based defense that can catch some 2-hop pollution; Theorem 1 does *not* rule out runtime defenses, only per-tool static ones — AgentArmor and TCPI are compatible.
- STAC (2509.25624) introduced sequential tool attack chains; those are user-driven rather than output-driven. Different threat model.

## Next steps (not executed)

- Swap `MockPolicyBackend` for `AnthropicBackend` and run Opus / Sonnet / Haiku head-to-head on V1–V7 with n ≥ 100. Expected: V4 ASR varies with the downstream-tier gap; V7 stays high even for the strongest model because it abuses the structured extractor, not the safety classifier.
- Deploy `MaliciousOracle.sol` + `TokenSwap.sol` to Base Sepolia per `demo_design.md` §8. Capture the two tx hashes (attack, control) for the paper headline figure.
- Submit TCPI to CCS'26 (main track, security of ML systems) or USENIX Security '26.
