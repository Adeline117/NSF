# Datasheet for the OnChainAgentID Benchmark

Following Gebru et al. (2018), *"Datasheets for Datasets,"* this document
answers the seven standard sections. Dataset version: **v4 (2026-04-09)**.

-----------------------------------------------------------------

## 1. Motivation

### Why was the dataset created?

To enable **audits of on-chain agent-vs-human classifiers** for label
leakage and cross-platform generalisation. The authors' own prior pipeline
reported AUC > 0.98, which collapsed to ~0.68 once three features that were
algebraically identical to the mining rule's gate variables were audited
away. This benchmark packages the corrected pipeline, its splits, and a
reproducible audit harness so others can avoid the same failure mode.

### Who created it and for whom?

Created by the NSF project authors at UW and collaborators, for the
account-level blockchain analytics community and for adjacent research
areas that rely on weakly-supervised or rule-mined labels
(MEV/anti-fraud, bot detection, trading-behaviour analysis).

### Funding

No commercial funding. Data collection used the authors' own Etherscan
and Flashbots relay API quotas.

-----------------------------------------------------------------

## 2. Composition

### What does each instance represent?

Each instance is a **single Ethereum externally-owned account (EOA) or
contract address** together with:

* 23 numerical behavioural features (temporal, gas, interaction,
  approval-security groups),
* a binary label (`1 = agent`, `0 = human`),
* provenance metadata (`provenance_source`, `label_provenance ∈ {0, 1}`,
  `category`),
* the raw Etherscan transaction history (available separately at
  `data/raw/<address>.parquet`).

### How many instances are there?

| Set | N | Notes |
|-----|---:|-------|
| Full benchmark (v4) | 1,147 | 533 agents, 614 humans |
| Strict curated core (Level 4) | 70 | 35 agents, 35 humans |
| Raw transaction files | 4,854 | Superset from full mining pipeline |

### Source categories (18)

Agents (`label_provenance = 1`, total 533):

| Category | N | Provenance source |
|----------|---:|-------------------|
| `defi_hf_trader` | 199 | On-chain rule: >100 tx/week for >4 weeks |
| `chainlink_keeper` | 108 | On-chain: Chainlink Automation Registry interactions |
| `compound_v3_liquidator` | 97 | On-chain: Compound V3 liquidation function calls |
| `gelato_executor` | 47 | On-chain: Gelato Network executor contracts |
| `keep3r_executor` | 46 | On-chain: Keep3r Network job execution |
| `pilot_agent` | 15 | Manual curation (pilot) |
| `curated_mev_bot` | 10 | Etherscan/Arkham MEV labels + Flashbots relay |
| `expanded_mev_bot` | 10 | Manual MEV curation (expanded pilot) |
| `flash_loan_user` | 1 | On-chain: Aave/dYdX flash loan function calls |

Humans (`label_provenance = 0`, total 614):

| Category | N | Provenance source |
|----------|---:|-------------------|
| `human_ens_interaction` | 167 | On-chain: ENS registration/transfer |
| `gitcoin_donor` | 146 | On-chain: Gitcoin Grants donation events |
| `ens_reverse_setter` | 144 | On-chain: ENS reverse registrar |
| `pooltogether_depositor` | 89 | On-chain: PoolTogether V4 deposits |
| `human_exchange_depositor` | 28 | On-chain: known CEX deposit addresses |
| `expanded_human` | 25 | Manual curation (pilot) |
| `curated_human` | 7 | ENS + Twitter verified |
| `human_ens_interaction_governance_voter` | 5 | ENS + DAO vote participant |
| `pilot_human` | 3 | Manual curation (pilot) |

### Is there a target variable?

Yes — `label` (binary). The "level of confidence" is encoded by
`label_provenance`: 1 = at least one off-chain attestation (relay data,
Etherscan label, Arkham label, ENS verification) or a hand-curated
inclusion; 0 = label derived from an on-chain heuristic rule only.

### Any recommended data splits?

Yes — four canonical splits are shipped in `benchmark/splits/*.json`:

1. **Level 2 random 10-fold CV** (seed=42)
2. **Level 3 temporal holdout** (split at median first-seen block)
3. **Level 4 strict curated core** (N=70, curated sources only)
4. **Level 5 LOPO** (14 platform folds, each holds out one category)

Users should report all four levels. A paper that reports Level 2 alone
is not defensible on this benchmark.

### Known errors, noise, or redundancies

- **Known leakage.** Three of the top-4 classifier features
  (`active_hour_entropy`, `burst_frequency`, `tx_interval_std`) are
  algebraically equivalent to three gate variables of the C1-C4 agent
  mining rule. `onchain_audit.check_label_feature_overlap` will flag
  this.
- **Platform imbalance.** `defi_hf_trader` = 199 / 533 agent set = 37%,
  and its mining rule is "more than 100 tx/week for >4 weeks." Models
  relying on raw transaction frequency will look artificially strong.
- **Mixed time windows.** Addresses were collected between 2023-11 and
  2026-04. First-seen blocks span roughly block 18,000,000–22,000,000.
- **Case-insensitive duplicates** were resolved by lower-casing during
  matching; index strings preserve the original casing.

-----------------------------------------------------------------

## 3. Collection process

### How was the data acquired?

1. **Agent curation (tier 1, 35 addresses).** Started from Flashbots
   relay public API, Etherscan agent tags (MEV bot iota, jaredfromsubway,
   Wintermute), and Arkham entity labels.
2. **Human curation (tier 1, 35 addresses).** Started from ENS
   verification (names like `vitalik.eth`), Twitter-verified ENS
   resolvers, and DAO governance voters.
3. **On-chain heuristic mining (tier 2, ~1,077 addresses).** For each
   platform (Chainlink, Compound, Gelato, Keep3r, Gitcoin, PoolTogether,
   ENS) we wrote a SQL-like query against the Etherscan tx API that
   matches the platform's canonical interaction pattern (e.g., "called
   `performUpkeep` on `0x...` ≥ 3 times in 30 days").
4. **Feature extraction.** For each address we fetch up to 10,000
   historical transactions from Etherscan and compute 23 behavioural
   features (`features/feature_pipeline.py`).

### Who was involved?

Three co-authors of the WWW '26 submission. No MTurk or crowdsourcing.

### Time frame

Mining window 2026-04-07 → 2026-04-09. Transactions themselves span
2023-11 (earliest first-seen block) → 2026-04 (snapshot date).

### Ethical review

No IRB required (no personal data; only pseudonymous Ethereum
addresses). No addresses were linked to real identities except where
already publicly self-attested (e.g., `vitalik.eth`).

-----------------------------------------------------------------

## 4. Preprocessing / cleaning / labeling

### What was done?

1. Raw transactions fetched via Etherscan API.
2. Features computed per address (`features/feature_pipeline.py`).
3. Labels assigned by a per-category rule (see §3).
4. NaN → column median imputation; 1st/99th percentile clipping
   per feature (applied at train time, not stored pre-clipped).

### Is the raw data preserved?

Yes — `data/raw/<address>.parquet` retains every transaction field
returned by Etherscan (blockNumber, timeStamp, hash, from, to, value,
gas, gasPrice, gasUsed, isError, input, methodId, functionName).

-----------------------------------------------------------------

## 5. Uses

### Intended uses

- **Benchmarking** agent-vs-human classifiers under the four-level
  protocol.
- **Auditing** your own classifier for label leakage via the
  `onchain_audit` package (see `../notebooks/tutorial.ipynb`).
- **Downstream** studies of agent behaviour (e.g., approval security,
  gas pricing, cross-platform presence).

### Tasks for which this dataset is inappropriate

- **Deanonymisation.** The benchmark tags only behavioural archetypes
  ("MEV bot", "Gitcoin donor"), never real-world identities.
- **Fraud detection in production.** Labels are research-grade. A
  production system would need human-in-the-loop review.
- **Claims about LLM-powered agents specifically.** Paper 0 finds that
  LLM-powered agents are behaviourally indistinguishable from DeFi
  management agents using this feature set (F1 = 0.53). This benchmark
  is **not** a good test bed for LLM-specific agent detection.

### Biases to be aware of

- **Platform bias.** DeFi high-frequency traders dominate the agent
  label. Non-DeFi agents (bridge relayers, gaming NFT bots) are absent.
- **Survivor bias.** Addresses with fewer than 10 transactions were
  filtered out during feature extraction.
- **Time-window bias.** Addresses that first appeared pre-2023-11 are
  not in the dataset even if still active.

-----------------------------------------------------------------

## 6. Distribution

### How is it distributed?

- **Primary:** this Git repository
  (https://github.com/adelinewen/NSF/tree/main/paper1_onchain_agent_id).
- **Archive:** a tagged Zenodo release will issue a DOI; see
  `/.zenodo.json` and `/CITATION.cff`.

### License

- **Data:** Creative Commons Attribution 4.0 (CC BY 4.0) — see
  `benchmark/LICENSE`.
- **Code:** MIT — see repository-level `LICENSE`.

### Restrictions

None beyond attribution. No IP or contractual restrictions on Etherscan
data relevant here (public blockchain).

-----------------------------------------------------------------

## 7. Maintenance

### Who maintains it?

Adeline Wen (lead) and co-authors. Contact via the GitHub repo's issue
tracker.

### Update policy

- **Major releases** (new addresses, new categories): versioned `vN`
  with a new `features_provenance_vN.parquet`. Old versions are frozen
  and stay in the repo for reproducibility.
- **Minor releases** (bug fixes, clarifications): in-place on `main`
  with a dated `FINDINGS_YYYY-MM-DD.md`.

### Erratum policy

If we discover a label error or a leakage artefact, we publish a
`FINDINGS_YYYY-MM-DD.md` at the repository root and link it from the
benchmark README and `CITATION.cff`. The current erratum is
`/FINDINGS_2026-04-08.md`, documenting the C1-C4 leakage and the
Paper 3 synthetic AI feature leakage.

### Planned extensions

- **Polygon extension** (~N=300) — cross-chain replication, see
  `paper3_ai_sybil/experiments/polygon_replication.py`.
- **LLM-agent subset** — direct labels from the ElizaOS / Virtuals
  Protocol ecosystems once on-chain enough.

-----------------------------------------------------------------

*Datasheet last updated: 2026-04-15.*
