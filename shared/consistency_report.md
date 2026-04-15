# Cross-Paper Consistency Report

**NSF Project: On-Chain AI Agent Characterization**
**Date:** 2026-04-14
**Scope:** Paper 0, Paper 1, Paper 3

---

## 1. Address Count Inconsistencies

### 1.1 Paper 0: "2,744" vs actual "2,590"

| Location | Claimed number |
|----------|---------------|
| P0 title (original task description) | 2,744 |
| P0 title (actual main.tex, line 9) | **2,590** |
| P0 abstract (line 14) | 2,590 |
| P0 Section 4.1 (line 218) | 2,590 |
| P0 conclusion (line 485) | 2,590 |

**Finding:** The task description referenced "2,744 addresses" but Paper 0 consistently uses **2,590** throughout the manuscript. There is no internal inconsistency within Paper 0. The 2,744 figure does not appear anywhere in the current drafts.

### 1.2 Paper 1: address counts

| Subset | P1 claim | Location |
|--------|----------|----------|
| Full (leaky) C1-C4 set | 3,316 (2,590 agents + 726 humans) | abstract, S1.2, S3.2 |
| Agent addresses used by P0 | 2,590 | S1.2 ("2,590 labeled AGENT") |
| Full provenance v4 | 1,147 (533 agents + 614 humans) | S1.3, S3.2.3, headline |
| Strict provenance | 64 (33 agents + 31 humans) | S3.2.2 |

**Finding:** Internally consistent. The 2,590 agent addresses are the same set that Paper 0 uses for taxonomy validation.

### 1.3 Paper 3: address counts

| Dataset | P3 claim | Location |
|---------|----------|----------|
| HasciDB total | 3.6M eligible across 16 campaigns | S1.1 |
| Stratified subset | 386,067 (125,157 Sybils, 32.4%) | S1.1, Table 1 |
| Paper 1 data reused for AI features | 3,316 (2,590 agents + 726 humans) | S3.2 |

**Finding:** Internally consistent. Paper 3 correctly cites the Paper 1 agent/human split.

### 1.4 Cross-paper address count verdict

Paper 0's address count (2,590) is consistent with Paper 1's agent subset. Paper 3 correctly references Paper 1's full dataset (3,316). **No cross-paper numeric conflicts found.**

---

## 2. Feature Set Consistency (23 Features)

### 2.1 Paper 0 feature list (Table 3, lines 220-235)

**Timing (7):** tx_interval_mean, tx_interval_std, tx_interval_skewness, active_hour_entropy, night_activity_ratio, weekend_ratio, burst_frequency

**Gas (6):** gas_price_round_number_ratio, gas_price_trailing_zeros_mean, gas_limit_precision, gas_price_cv, eip1559_priority_fee_precision, gas_price_nonce_correlation

**Interaction (5):** unique_contracts_ratio, top_contract_concentration, method_id_diversity, contract_to_eoa_ratio, sequential_pattern_score

**Approval (5):** unlimited_approve_ratio, approve_revoke_ratio, unverified_contract_approve_ratio, multi_protocol_interaction_count, flash_loan_usage

### 2.2 Paper 1 feature list (Section 3.3, lines 247-285)

**Temporal (7):** tx_interval_mean, tx_interval_std, tx_interval_skewness, active_hour_entropy, night_activity_ratio, weekend_ratio, burst_frequency

**Gas (6):** gas_price_round_number_ratio, gas_price_trailing_zeros_mean, gas_limit_precision, gas_price_cv, eip1559_priority_fee_precision, gas_price_nonce_correlation

**Interaction (5):** unique_contracts_ratio, top_contract_concentration, method_id_diversity, contract_to_eoa_ratio, sequential_pattern_score

**Approval & Security (5):** unlimited_approve_ratio, approve_revoke_ratio, unverified_contract_approve_ratio, multi_protocol_interaction_count, flash_loan_usage

### 2.3 Paper 3 feature reference

Paper 3 does not independently list the 23 features; it references "the Paper 1 expanded parquet" and uses the 8 AI-specific features. The 8 AI features are:

1. gas_price_precision
2. hour_entropy
3. behavioral_consistency
4. action_sequence_perplexity
5. error_recovery_pattern
6. response_latency_variance
7. gas_nonce_gap_regularity
8. eip1559_tip_precision

### 2.4 Feature set verdict

**The 23 features are identical across Paper 0 and Paper 1.** The only cosmetic difference is the group name: Paper 0 calls group 1 "Timing" while Paper 1 calls it "Temporal", and Paper 0 calls group 4 "Approval" while Paper 1 calls it "Approval & Security". The feature names and ordering match exactly.

**Minor naming inconsistency:** Paper 1 Section 3.9 lists the 8 AI features using slightly different names than Paper 3 Table 1:
- P1: `response_latency_variance` vs P3: `response_latency_var` (abbreviation only)
- P1: `action_sequence_perplexity` vs P3: `action_seq_perplexity` (abbreviation only)
- P1: `gas_nonce_gap_regularity` vs P3: `gas_nonce_regularity` (abbreviation only)

**Recommendation:** Standardize AI feature names across papers to their full form for clarity.

---

## 3. C1-C4 Definition Consistency

### 3.1 Paper 0 formal definition (Section 3.1, lines 124-152)

- **C1 (On-chain actuation):** E controls an EOA alpha, and alpha initiates >= 1 tx.
- **C2 (Environmental perception):** Mutual information between chain state at t and action at t+1 is > 0.
- **C3 (Autonomous decision-making):** Conditional entropy of actions given environment > 0 AND not human-gated.
- **C4 (Adaptiveness):** KS-distance between early/late behavioral parameters exceeds threshold delta.

### 3.2 Paper 1 references to C1-C4 (Section 1.2, lines 47-53)

Paper 1 redefines C1-C4 as a **labeling heuristic** with operational thresholds:
- C1: public registry or known identity
- C2: behavioral consistency with the declared class
- C3: thresholds on hour_entropy > 2.5, burst_ratio > 0.1, tx_interval_cv < 1.0
- C4: temporal stability across windows

### 3.3 Paper 3 references to C1-C4

Paper 3 Section 3.2 references "Paper 1's C1-C4 pipeline" for the 3,316 labeled addresses and uses the operational definitions, not the formal definitions.

### 3.4 C1-C4 consistency verdict

**INCONSISTENCY FOUND.** Paper 0 defines C1-C4 as abstract formal conditions grounded in information theory (mutual information, conditional entropy, KS-distance). Papers 1 and 3 redefine C1-C4 as a concrete labeling heuristic with threshold-based rules. This is by design -- Paper 1 explicitly documents this repurposing and the resulting leakage problem -- but the shared label "C1-C4" is used for two different things:

1. **Paper 0's C1-C4:** A formal definition of on-chain AI agency (theoretical).
2. **Paper 1's C1-C4:** A 4-gate labeling heuristic (operational, subsequently demoted).

Paper 0 Section 3.3 acknowledges this gap: "The four conditions are latent -- we cannot inspect the source code." Paper 1 Section 5 documents the consequences of operationalizing them as labeling gates.

**Recommendation:** All three papers should explicitly distinguish between "C1-C4 (formal definition)" and "C1-C4 (operational heuristic)" wherever C1-C4 is mentioned. Paper 0 already does this in Section 3.3 line 197 ("A critical caveat on label leakage"), but the distinction should be more prominent.

---

## 4. Cross-Paper Numerical Claims

### 4.1 Paper 0's claims about Paper 1

| Claim in P0 | Location | P1 actual | Match? |
|-------------|----------|-----------|--------|
| P1 trains binary classifier on same 23 features | S4.7 | Yes | YES |
| P1 honest AUC 0.883 (GAT on n=64) | S4.7 | 0.8825 +/- 0.108 (5-fold) | CLOSE (0.883 vs 0.8825; rounding) |
| P1 identifies agents, P0 sub-classifies | S4.7 | Confirmed conceptually | YES |

### 4.2 Paper 0's claims about Paper 3

| Claim in P0 | Location | P3 actual | Match? |
|-------------|----------|-----------|--------|
| P3 has 8 AI-specific features with Cohen's d > 1.0 on hour_entropy | S4.4, S5.1 | d = 1.98 for hour_entropy, d = 1.24 for behavioral_consistency | YES |
| Adding P3's 8 features should close LLM-Powered Agent F1 gap | S5.1 | P1 S4.5 shows delta = +0.013 on trusted set | PARTIALLY CONTRADICTED (see 4.5) |

### 4.3 Paper 3's claims about Paper 1

| Claim in P3 | Location | P1 actual | Match? |
|-------------|----------|-----------|--------|
| Paper 1 labels 2,590 agents and 726 humans | S1.1 | 2,590 + 726 = 3,316 | YES |
| 8 AI features from Paper 1 all satisfy p < 0.01 at n=3,316 | S3.2, Table 1 | Confirmed in P3 Table 1 | YES |
| hour_entropy Cohen's d = 1.98 | Table 1 | 1.983 | YES |

### 4.4 Paper 1's claims about Paper 0

| Claim in P1 | Location | P0 actual | Match? |
|-------------|----------|-----------|--------|
| P0 defines C1-C4 | S1.2 | S3.1 | YES |
| 23 features shared with P0 | S3.3 | Identical list | YES |

### 4.5 Potential tension: P0 predicts AI features will fix LLM-Powered Agent F1

Paper 0 Section 5.1 predicts: "adding Paper 3's eight AI-specific features [...] the LLM-Powered Agent F1 will climb above 0.80 without harming the other four classes."

Paper 1 Section 4.5 Table 7 shows: combining 23 P1 features with 8 P3 features yields LightGBM 5x3-CV AUC 0.7172 on the **binary** task (trusted n=64), a delta of only +0.013 over P1 alone. RF actually drops from 0.8183 to 0.8151.

**TENSION:** Paper 0 makes a strong prediction about the multi-class LLM-Powered Agent F1 improvement. Paper 1 shows that the combined features provide essentially no improvement on the binary task. These are different tasks (multi-class sub-typing vs binary agent/human), so the results are not directly contradictory, but Paper 0's optimism is not supported by the Paper 1 evidence that exists so far. Paper 0 should temper this prediction or note that it remains untested.

---

## 5. Bibliography Cross-Reference Issues

### 5.1 Duplicate entries for the same paper

| Paper | P0 key | P1 key | Same paper? |
|-------|--------|--------|-------------|
| Daian et al. 2020 (Flash Boys 2.0) | `daian2020flashboys` | `daian2020flash` | YES -- different keys |
| Torres et al. 2021 (Frontrunner Jones) | `torres2021frontrunner` | `torres2021frontrunner` | YES -- same key |
| Qin et al. 2022 (Quantifying BEV) | `qin2022quantifying` | `qin2022quantifying` | YES -- same key |
| Autonolas whitepaper | `autonolas2022` (year 2022) | `autonolas2023` (year 2023) | SAME platform, different years |
| He et al. agent survey | `he2025survey` (year 2025) | `he2026agentsurvey` (year 2026) | SAME arXiv ID (2601.04583), year discrepancy |
| Kaufman et al. 2012 (Leakage) | not in P0 | `kaufman2012leakage` (P1) | Also in P3 with more metadata |
| Hamilton et al. 2017 (GraphSAGE) | not in P0 | `hamilton2017graphsage` (P1) | Also in P3, slightly different author list |
| MCP 2024 | `mcp2024` (P0) | not in P1 | Also in shared as `anthropic2024mcp` |

### 5.2 Inconsistent year for He et al. arXiv:2601.04583

- Paper 0: `he2025survey` (year=2025)
- Paper 1: `he2026agentsurvey` (year=2026)
- Same arXiv paper: 2601.04583

**Recommendation:** Standardize to one year. The arXiv ID prefix 2601 suggests January 2026, so `He2026` is more accurate.

### 5.3 Autonolas year discrepancy

- Paper 0: `autonolas2022` -- "Autonolas Whitepaper" (2022)
- Paper 1: `autonolas2023` -- "Autonolas: Autonomous Agent Services" (2023)

These may be different documents (whitepaper versions). If they are the same document, standardize the year.

### 5.4 Hamilton et al. 2017 author list discrepancy

- Paper 1 (P1): "Hamilton, William L. and Ying, Rex and Leskovec, Jure"
- Paper 3 (P3): "Hamilton, Will and Ying, Zhitao and Leskovec, Jure"

The canonical NeurIPS paper lists "William L. Hamilton, Rex Ying, Jure Leskovec". Paper 3's "Zhitao" is incorrect; it should be "Rex".

**Action required:** Fix the author field in Paper 3's bib entry.

---

## 6. Structural / Conceptual Consistency

### 6.1 Pipeline flow is coherent

The stated pipeline P0 -> P1 -> P3 is well-supported:

- **P0 -> P1:** Paper 0 defines what an agent is; Paper 1 builds a classifier to identify them. Both share the 23-feature set. Paper 0's taxonomy uses Paper 1's 2,590 agent addresses as the validation dataset.
- **P1 -> P3:** Paper 3 reuses the 8 AI-specific features computed on Paper 1's 3,316-address dataset. Paper 3 also reuses the agent/human labels for feature validation.
- **P3 -> P0 (feedback):** Paper 3's finding that LLM-generated wallets are hard to distinguish from real wallets motivates Paper 0's discussion of the LLM-Powered Agent F1 gap.

### 6.2 The "leakage" theme is consistent

All three papers discuss methodological leakage:
- **Paper 0:** Warns about partial circularity between taxonomy projection rules and classifier features (Section 4.6).
- **Paper 1:** Central contribution is the C1-C4 label leakage discovery (Section 5).
- **Paper 3:** Discovers a second leakage bug in `augment_with_ai_features` (Section 5).

This is a strength: the project treats leakage as a first-class research concern across all papers.

### 6.3 C3 boundary case for Sybils

Paper 0 Section 3.4 (line 207): "AI Sybils violate C3 because their behavior is coordinated by a single controller, even though the individual EOAs look noisy."

Paper 3 abstract and Section 1.1: The attacker is "an LLM-powered wallet operator" that creates individual wallets. Each wallet is generated independently (no shared funding graph).

**Mild tension:** Paper 0 says AI Sybils violate C3 because of coordination, but Paper 3's LLM generator creates wallets independently. If each wallet independently has non-deterministic behavior and is not human-gated, individual wallets could satisfy C3. The violation argument depends on whether "coordinated by a single controller" means the wallets share a policy (which they do -- the same LLM prompt), not that they coordinate on-chain.

**Recommendation:** Paper 0 should clarify that C3 violation for Sybils is about shared decision-making origin (same prompt/controller), not about on-chain coordination.

---

## 7. Summary of Actions Required

### Critical (must fix before submission)

1. **Hamilton et al. author fix:** Paper 3 bib lists "Ying, Zhitao" instead of "Ying, Rex" for the GraphSAGE paper.

2. **He et al. year standardization:** Reconcile `he2025survey` (P0) vs `he2026agentsurvey` (P1) -- both cite arXiv:2601.04583. Use 2026 consistently.

### Important (should fix)

3. **Temper Paper 0's prediction:** P0 Section 5.1 predicts that adding 8 AI features will lift LLM-Powered Agent F1 above 0.80. P1 Section 4.5 shows minimal improvement on the binary task. Note that the multi-class test has not been run yet.

4. **Standardize AI feature names:** Use full names (response_latency_variance, action_sequence_perplexity, gas_nonce_gap_regularity) consistently across P1 and P3.

5. **Clarify C1-C4 dual meaning:** Whenever Papers 1 or 3 reference "C1-C4", distinguish the formal definition (P0) from the operational heuristic (P1).

### Minor (cosmetic)

6. **Feature group naming:** Paper 0 says "Timing", Paper 1 says "Temporal". Pick one.

7. **Daian et al. bib key:** Paper 0 uses `daian2020flashboys`, Paper 1 uses `daian2020flash`. If switching to unified.bib, use `Daian2020`.

8. **Autonolas year:** Verify whether `autonolas2022` (P0) and `autonolas2023` (P1) reference the same or different documents.

---

## 8. Unified Bibliography Status

A merged and deduplicated bibliography has been created at `shared/unified.bib` with:
- **70 unique entries** consolidated from all three paper bib files and the existing shared/references.bib
- AuthorYear naming convention (e.g., `Daian2020`, `Hamilton2017`)
- Alias comments mapping old keys to new canonical keys
- Duplicate entries (same paper, different keys) consolidated under a single entry
- Entries organized by thematic section for maintainability

To adopt: each paper can either `\bibliography{../../shared/unified}` or continue with its own main.bib with synchronized keys.

---

## 9. Pipeline Figure Status

A standalone TikZ pipeline figure has been created at `shared/figures/fig_pipeline.tex`. It includes:
- Three paper boxes (P0, P1, P3) with key outputs and metrics
- Forward arrows showing data/concept flow
- Feedback arrow from P3 to P0 (LLM-Powered Agent gap)
- Shared data layer (23 features + Ethereum transaction data)
- Caption with summary of inter-paper relationships

To include in any paper: `\input{../../shared/figures/fig_pipeline.tex}` (adjust relative path as needed). Requires `\usepackage{tikz}` in the preamble.
