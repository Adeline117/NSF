# Tool-Chain Prompt Injection (TCPI): Formal Definition

## 1. System model

Let an LLM-orchestrated agent system be the tuple

```
Σ = (M, T, S, U, Π, Ω)
```

where

- **M** — a language model, modeled as a conditional distribution `M(· | h)` over next-action tokens given history `h`.
- **T = {T_1, T_2, …, T_n}** — a finite set of *tools*. Each tool `T_i : Θ_i × S → O_i × S` takes a parameter vector `θ_i ∈ Θ_i` and an external state `s ∈ S`, returns an output `o_i ∈ O_i` and a next state `s' ∈ S`.
- **S** — the (possibly unbounded) external-world state; e.g., files, RPC-reachable blockchain state, API mailboxes.
- **U** — the user prompt space.
- **Π** — the agent policy that, given history `h`, emits a *tool-call record* `(i, θ_i)` or a *final message*. Operationally, `Π` is realized by `M` under a tool-use scaffold (ReAct, JSON schema, MCP).
- **Ω** — the observable trace: a finite sequence

  ```
  ω = ⟨u, (i_1, θ_1, o_1), (i_2, θ_2, o_2), …, (i_k, θ_k, o_k), m⟩
  ```

  of the user prompt, tool-call records, and the final assistant message `m`.

## 2. Adversary model

An adversary `A` is characterized by the subset of the system it controls.

| Adversary | Controls | Known as |
|---|---|---|
| **A_user** | the prompt `u ∈ U` | direct prompt injection (single-hop) |
| **A_desc** | `T_i.description` seen by `M` before the call | tool poisoning (Invariant Labs) |
| **A_doc** | retrieved documents that land in `M`'s context | indirect prompt injection (Greshake 2023) |
| **A_out** *(this paper)* | the output `o_A` of some tool `T_A ∈ T` | **tool-chain injection source** |

The TCPI adversary controls **only `o_A`**: it cannot rewrite `u`, cannot alter any tool description, cannot read `M`'s weights, cannot choose which tools the developer exposed. This is realistic: `T_A` can be an oracle API, a web-search result, a smart-contract view function, an email, an issue comment, a database cell — any data source whose *contents* an outside party can write to but whose interface signature the agent developer fixed at deployment.

## 3. Victim workflow

The canonical TCPI victim workflow is a two-tool chain:

```
u  ──► M  ──► (A, θ_A)  ──► T_A  ──► o_A
                                       │
                                       ▼
                                       M  (sees o_A in its context)
                                       │
                                       ▼
                                 (B, θ_B)  ──► T_B  ──► effect on S
```

Here `T_A` is typically a **read-only** tool (oracle, search, read-mail, view-balance) and `T_B` is a **state-mutating** tool (transfer, write-file, send-email, approve-allowance, execute-SQL). The agent's policy `Π` picks `θ_B` as a function of `o_A`:

```
θ_B = f_M(u, o_A, h)
```

where `f_M` is the (stochastic) parameter-synthesis function implicit in `M`.

## 4. TCPI attack — formal definition

> **Definition 1 (TCPI attack).** A trace `ω` constitutes a *tool-chain prompt injection* against deployment `Σ` if there exist tools `T_A, T_B ∈ T` with `T_A ≠ T_B`, a user prompt `u` that does not itself encode attacker intent, and an attacker-chosen output `o_A* ∈ O_A` such that:
>
> 1. **(Benign baseline)** When `T_A` returns its *honest* output `o_A^honest`, the resulting θ_B^honest satisfies the user's intent and does not transfer control/resources to the attacker:
>    `safe(θ_B^honest) = 1`.
> 2. **(Attacker-chosen deviation)** When `T_A` returns `o_A*`, the resulting parameter θ_B* satisfies an attacker predicate `φ_A`:
>    `φ_A(θ_B*) = 1`, where `φ_A` is chosen by the attacker (e.g., `recipient = 0xATTACKER`).
> 3. **(No other channel)** The attacker modifies *only* `o_A`. Formally, `u`, `T_A.description`, `T_B.description`, `T_B.code`, and `M`'s weights are identical in the benign and attack executions.
> 4. **(LLM-derivation)** The flow from `o_A*` to `θ_B*` goes through `M`: there is no direct programmatic pipe `o_A → θ_B` in the harness. `θ_B*` is *synthesized by M* from `o_A*` via `f_M`.

Condition (4) is what separates TCPI from ordinary **data-flow** vulnerabilities: the attacker is not exploiting a hard-coded taint flow in glue code; it is exploiting the model's natural-language *reasoning* that decides `θ_B` from what `o_A` says.

## 5. What TCPI is **not**

| Class | Attacker controls | Why it's not TCPI |
|---|---|---|
| Single-hop direct injection | `u` | TCPI requires `u` benign and the attack carried in `o_A` |
| Tool poisoning | `T_A.description` | TCPI requires `description` unchanged; only `o_A` moves |
| Indirect prompt injection (Greshake) | retrieved document shown to `M` | Document becomes part of `M`'s context but does not pass through a tool signature; no parameter-synthesis step |
| Tool selection hijack (ToolHijacker) | tool registry | Attacker flips *which* tool is called; TCPI flips *parameters of* an already-chosen benign tool |
| Direct injection with `T_A = T_B` | `o_A`, but only one tool | Degenerate case; equivalent to the tool being locally malicious |
| LLM-to-LLM prompt infection | inter-agent messages | Different system (many M's); TCPI is one M, many T's |
| User-chosen STAC sequences | the multi-turn user prompts | User (not `o_A`) drives the chain |

## 6. Attack success rate (ASR) — operational metric

For a scenario family `F` and model `M`, the TCPI ASR is

```
ASR(F, M) = Pr_{u ∼ U_F, o_A* ∼ Adv_F}  [ φ_A(θ_B*) = 1  ∧  ω is well-formed ]
```

estimated by Monte Carlo over `N` rollouts. We compare to:

- **Single-hop baseline**: same `φ_A` but attacker injects into `u` directly. Modern safety-trained models drive this near zero.
- **Null baseline**: `o_A = o_A^honest`. This must yield ≈0 attacker satisfaction.

TCPI is *interesting* precisely when ASR_TCPI ≫ ASR_single-hop, because it reveals that the safety training invested against `A_user` did not transfer to `A_out`.

## 7. Distinctness proof sketch (informal)

TCPI is **strictly** distinct from each class in §5 because for each class X there exists a system `Σ_X` vulnerable to X and safe against TCPI, and vice versa; the parameter-synthesis step (§4, condition 4) is a necessary witness for TCPI but absent from all others. A formal proof is by constructing the four corners of a 2×2 table over {Σ safe/vulnerable to TCPI} × {safe/vulnerable to X} and exhibiting a non-empty witness in each cell — all witnesses are straightforward and are deferred to Appendix A of the paper.
