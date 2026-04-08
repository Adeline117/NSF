# Comparative Matrix of AI Agent Definitions

This matrix compares major AI-agent definitions across CS/HCI/philosophy
and evaluates their coverage of the C1-C4 conditions used in this
paper. Source: NSF project `shared/definition.md`.

| Work | Year | Discipline | Definition type | Covers C1 | C2 | C3 | C4 | Web3 applicable? |
|------|------|-----------|----------------|-----------|----|----|----|------------------|
| Russell & Norvig, *AI: A Modern Approach* | 2020 | AI textbook | Rational agent = perceive + act | — | ✓ | partial | — | ✗ (too broad: thermometers qualify) |
| Wooldridge & Jennings, "Intelligent Agents: Theory and Practice" | 1995 | Multi-agent systems | 4 properties: autonomy, social ability, reactivity, pro-activeness | — | ✓ | ✓ | ✓ | ✗ (no actuation requirement) |
| Franklin & Graesser, "Is it an agent or just a program?" | 1996 | Conceptual taxonomy | Property enumeration (reactive, autonomous, goal-oriented, temporally-continuous, communicative, learning, mobile, flexible, character) | — | ✓ | ✓ | ✓ | ✗ (no commitment to necessity) |
| Maes, "Modeling adaptive autonomous agents" | 1995 | Artificial life | "Computational systems that inhabit some complex dynamic environment, sense and act autonomously" | — | ✓ | ✓ | ✓ | ✗ (no persistence criterion) |
| Shoham, "Agent-oriented programming" | 1993 | Programming languages | Mentalistic attitudes (beliefs, commitments) | — | — | ✓ | — | ✗ (requires BDI layer) |
| Russell, "Rationality and intelligence" | 1997 | AI theory | Bounded rationality under resource constraints | — | ✓ | ✓ | — | ✗ |
| Crabtree et al., "Agency in the age of the machine" | 2016 | HCI | Sociomaterial framing: agency as distributed across humans + systems | — | ✓ | ✓ | ✓ | partial (relational view) |
| Chopra & Singh, "Agent communication languages" | 2000 | MAS | Communication protocols as definitional | — | ✓ | — | — | ✗ |
| Dennett, "The intentional stance" | 1987 | Philosophy | Intentional-level interpretability | — | ✓ | ✓ | — | ✗ (observer-dependent) |
| Bratman, "Intention, plans, and practical reason" | 1987 | Philosophy | BDI: beliefs, desires, intentions | — | ✓ | ✓ | — | ✗ |
| **This paper** | 2026 | Web3 systems | C1-C4 necessary+sufficient, chain-observable | ✓ | ✓ | ✓ | ✓ | ✓ (native) |
| Park et al., "Generative agents" | 2023 | NLP / HCI | LLM-powered simulacra with memory | — | ✓ | ✓ | ✓ | ✗ (text-only) |
| Chan et al., "Harms from AI agents" | 2023 | AI safety | Causal / counterfactual influence on environment | — | ✓ | ✓ | — | partial |
| Shavit et al., "Practices for governing agentic AI systems" | 2023 | AI governance | Agent = system pursuing goals with little supervision | — | ✓ | ✓ | — | ✗ (no actuation spec) |
| Autonolas Whitepaper | 2022 | Web3 engineering | Multi-agent services in FSM + proof-of-activity | ✓ | ✓ | ✓ | ✓ | ✓ |
| Eliza (AI16Z) docs | 2024 | Web3 engineering | "Autonomous agents with on-chain persona" | ✓ | ✓ | ✓ | ✓ | ✓ |

## Gaps in the literature

1. **No existing definition ties agent-ness to on-chain observability.**
   All CS/HCI definitions are architectural (BDI, FSM, RL policy)
   rather than behavioral. This paper's contribution is to translate
   the concept into something measurable from public ledger data.

2. **Existing definitions conflate "has a policy" with "adapts its
   policy".** Wooldridge's "pro-activeness" and Franklin & Graesser's
   "goal-oriented" both admit frozen deterministic bots. Our $C_4$
   (adaptiveness) separates learning agents from static scripts.

3. **Actuation (C1) is assumed, not required.** Most definitions take
   for granted that the entity can act; in Web3, this is non-trivial
   because smart contracts cannot initiate transactions. Our $C_1$
   makes actuation explicit.

4. **No existing framework has a counterexample-based non-redundancy
   argument.** We provide one for each $C_i$ (see
   `framework/c1c4_formalization.md`).

## Taxonomy comparison

The 8-category pilot taxonomy is roughly compatible with:

- Franklin & Graesser's property-subset classifications (each
  category = different subset of properties)
- Wooldridge's 4-property matrix (with C1 added for Web3)
- Shavit et al.'s governance categories ("narrow AI assistants",
  "general agents", "autonomous systems")

It adds:
- **Explicit on-chain actuation**
- **Behavioral distinguishability** (each category has measurable
  on-chain indicators)
- **Boundary case treatment** (MEV searchers, liquidation bots,
  cross-chain bridges)

## Recommended citations for Section 2

```bibtex
@book{russellnorvig2020,
  author = {Russell, Stuart and Norvig, Peter},
  title = {Artificial Intelligence: A Modern Approach},
  edition = {4},
  publisher = {Pearson},
  year = {2020}
}

@article{wooldridge1995intelligent,
  title={Intelligent agents: Theory and practice},
  author={Wooldridge, Michael and Jennings, Nicholas R},
  journal={The Knowledge Engineering Review},
  volume={10},
  number={2},
  pages={115--152},
  year={1995}
}

@inproceedings{franklin1996agent,
  title={Is it an agent, or just a program?: A taxonomy for autonomous agents},
  author={Franklin, Stan and Graesser, Art},
  booktitle={ATAL},
  year={1996}
}

@incollection{maes1995modeling,
  title={Modeling adaptive autonomous agents},
  author={Maes, Pattie},
  booktitle={Artificial Life: An Overview},
  year={1995}
}

@article{park2023generative,
  title={Generative agents: Interactive simulacra of human behavior},
  author={Park, Joon Sung and O'Brien, Joseph C and Cai, Carrie J
           and Morris, Meredith Ringel and Liang, Percy and Bernstein, Michael S},
  journal={UIST},
  year={2023}
}

@article{chan2023harms,
  title={Harms from increasingly agentic algorithmic systems},
  author={Chan, Alan and others},
  journal={FAccT},
  year={2023}
}
```
