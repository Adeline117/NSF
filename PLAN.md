# NSF Research Execution Plan

## Timeline Overview

| Phase | Paper 0 (CHI) | Paper 1 (WWW) | Paper 2 (S&P) | Paper 3 (GLOBECOM) |
|-------|---------------|---------------|---------------|-------------------|
| Phase 1: Foundation | Taxonomy + Lit Review | Feature pipeline | Threat model | HasciDB integration |
| Phase 2: Data | Case studies | Etherscan extraction | GitHub server scan | Real sybil labels |
| Phase 3: Experiments | Validation | Classifier + audit | Static + dynamic | Evasion + detection |
| Phase 4: Writing | Draft | Draft | Draft | Draft |

## Paper 0: Understanding AI Agents (CHI)

### Phase 1 - NOW
- [x] Pilot: 6-category taxonomy validated
- [ ] Expand taxonomy: add NONE autonomy + RL decision model
- [ ] Literature review: survey AI agent definitions across CS/HCI/Philosophy
- [ ] Formalize agent definition framework (necessary & sufficient conditions)
- [ ] Write Section 2: Background & Related Work

### Phase 2
- [ ] Map taxonomy to real-world case studies (Autonolas, AI16Z, Virtuals)
- [ ] Interview/survey design for expert validation (Delphi method)
- [ ] Write Section 3: Taxonomy Design

### Phase 3
- [ ] Validate taxonomy completeness with real on-chain data
- [ ] Cross-reference with Paper 1 agent identification results
- [ ] Write Section 4: Validation

### Phase 4
- [ ] Full draft
- [ ] Internal review cycle

## Paper 1: On-Chain AI Agent ID & Security (WWW)

### Phase 1 - NOW
- [x] Pilot: 4 feature groups show signal (synthetic)
- [ ] Formalize feature extraction pipeline (production-grade)
- [ ] Define ground truth labeling methodology
- [ ] Query Autonolas registry for confirmed agent addresses
- [ ] Write Section 2: Related Work
- [ ] Write Section 3: Methodology

### Phase 2
- [ ] Etherscan API: extract features for known agents + humans
- [ ] Dune Analytics: large-scale gas/timing distributions
- [ ] Build labeled dataset (agent vs human)

### Phase 3
- [ ] Train classifier (LightGBM + RF + GNN)
- [ ] Four-dimensional security audit on identified agents
- [ ] Statistical significance tests

### Phase 4
- [ ] Full draft
- [ ] Open-source tool release

## Paper 2: Agent Tool Interface Security (S&P/USENIX)

### Phase 1 - NOW
- [x] Pilot: 4 protocol families, 63 findings, 18/20 categories
- [ ] Formalize unified vulnerability taxonomy with CWE mapping
- [ ] Write threat model (attacker capabilities, attack surfaces)
- [ ] Enumerate real servers from GitHub/npm for all 4 protocols
- [ ] Write Section 2: Background & Threat Model

### Phase 2
- [ ] Clone top 50 servers per protocol family
- [ ] Run static analysis at scale
- [ ] Aggregate cross-protocol statistics

### Phase 3
- [ ] Dynamic testing: tool poisoning, prompt injection, key extraction
- [ ] Harness gap analysis
- [ ] Risk scoring validation

### Phase 4
- [ ] Full draft
- [ ] Responsible disclosure
- [ ] Open-source scanner release

## Paper 3: AI Agent Sybil Attacks (GLOBECOM/CHI)

### Phase 1 - NOW
- [x] Pilot v2: HasciDB-calibrated, 100% evasion confirmed
- [ ] Clone HasciDB repo, build SQLite database
- [ ] Load real indicator distributions from 16 projects
- [ ] Integrate pre-airdrop-detection LightGBM as baseline
- [ ] Write Section 2: Related Work (HasciDB CHI'26, TrustaLabs, Arbitrum)

### Phase 2
- [ ] Design LLM-based Sybil generator (GPT-4/Claude)
- [ ] Generate AI Sybil transaction sequences
- [ ] Run against HasciDB + pre-airdrop baselines

### Phase 3
- [ ] Train enhanced detector on real data
- [ ] Cross-project transfer experiments
- [ ] Ablation study on AI-specific features

### Phase 4
- [ ] Full draft
- [ ] Dataset release
