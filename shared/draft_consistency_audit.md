# Draft Consistency Audit: Numbers in draft_en.md vs Source JSON

**Audit date:** 2026-04-08
**Auditor:** Claude Opus 4.6 (automated)

---

## Discrepancies Found

| Paper | Claim location | Draft value | Source file | Source value | Match? |
|-------|---------------|-------------|-------------|--------------|--------|
| **Paper 0** | Table 4, DeFi Management Agent mean confidence | 0.62 | `paper0_ai_agent_theory/experiments/taxonomy_projection_results.json` -> `category_examples.DeFi Management Agent.mean_confidence` | **0.603** | NO |
| **Paper 0** | Table 4, Deterministic Script mean confidence | 0.59 | same file -> `category_examples.Deterministic Script.mean_confidence` | **0.8** | NO |
| **Paper 0** | Table 4, Simple Trading Bot mean confidence | 0.63 | same file -> `category_examples.Simple Trading Bot.mean_confidence` | **0.7** | NO |
| **Paper 0** | Table 4, LLM-Powered Agent mean confidence | 0.71 | same file -> `category_examples.LLM-Powered Agent.mean_confidence` | **0.6** | NO |
| **Paper 0** | Table 4, MEV Searcher mean confidence | 0.86 | same file -> `category_examples.MEV Searcher.mean_confidence` | **0.85** | NO |
| **Paper 0** | Section 4.7, "AUC 0.870 under a GraphSAGE-GAT architecture" (cross-ref to Paper 1) | 0.870 | `paper1_onchain_agent_id/experiments/expanded/gnn_results.json` -> `splits.trusted_64.GAT.mean_auc` | **0.8825** | NO |
| **Paper 1** | Abstract, "Graph Attention Network reaches 0.870 five-fold AUC" | 0.870 | `paper1_onchain_agent_id/experiments/expanded/gnn_results.json` -> `splits.trusted_64.GAT.mean_auc` | **0.8825** | NO |
| **Paper 1** | Contribution 1 (line 73), "Graph Attention Network five-fold AUC = 0.870" | 0.870 | same file | **0.8825** | NO |
| **Paper 1** | Contribution 5 (line 77), "GAT (AUC 0.870) beats GraphSAGE (0.784)" | 0.870 | same file | **0.8825** | NO |

---

## Notes on Paper 1 GAT AUC

The draft is internally inconsistent. The abstract and contributions sections report GAT honest AUC as **0.870**, but later in the same draft:
- Table 5 (line 375) correctly reports **0.8825 +/- 0.108**
- Section 4.4 text (line 381) correctly says "GAT reaches 5-fold AUC **0.8825**"
- Section 6.1 (line 540) correctly says "the headline **0.8825** AUC"
- Section 7 (line 568) correctly says "GAT 5-fold AUC **0.8825**"
- The conclusion (line 573) uses the rounded **0.87**, which is acceptable.

The source JSON (`gnn_results.json`) unambiguously gives `trusted_64.GAT.mean_auc = 0.8825`. The value 0.870 does not appear in any source file and appears to be a transcription error in the abstract/contributions that was later corrected in the body but not propagated back.

## Notes on Paper 0 Mean Confidence Values

All five mean confidence values in Table 4 disagree with `taxonomy_projection_results.json`. The discrepancies are large (e.g., Deterministic Script: draft 0.59 vs source 0.8; LLM-Powered Agent: draft 0.71 vs source 0.6). These are not rounding differences; the values are categorically different. Possible explanations:
1. The draft was written from an earlier run whose output was later overwritten.
2. The mean confidence in the JSON reflects per-category example confidence, while the draft reports a differently weighted average. However, the JSON field is named `mean_confidence` and matches the column header in the draft ("Mean confidence").

This should be investigated and reconciled.

---

## Verified (All Match)

The following claims were checked and confirmed to match their source JSON files:

### Paper 0
- 2,590 agents, 5/8 categories populated
- Per-category counts: 1,669 / 666 / 130 / 71 / 54
- Silhouette at k=3: 0.1509
- ARI at k=5: 0.3189
- GBM accuracy: 0.9737
- F1-macro: 0.8683
- Per-class F1: 0.996 / 0.988 / 0.982 / 0.849 / 0.528
- Full K-Means sweep table (all k=3..15 values)

### Paper 1
- GBM leaky AUC: 0.9803
- RF honest LOO AUC: 0.7713
- RF honest 5x10-CV AUC: 0.8030
- McNemar p (GBM vs RF): 0.0225 (source: 0.022461)
- DeLong p (GBM vs RF): 0.0014 (source: 0.001358)
- All cross-platform AUCs (4 splits x 3 models = 12 values)
- GraphSAGE trusted_64 AUC: 0.784 (source: 0.7841)
- GAT full_3316 AUC: 0.9341
- GNN combined results (all values)
- Security audit ratios and Cohen's d values (full and trusted subsets)
- Combined pipeline results (all feature sets, all models)

### Paper 2
- 27 patterns, 21 categories, 16 CWEs (from taxonomy.json)
- 138 servers scanned (from full_catalog_scan_results.json metadata)
- Total findings: 5,641 (637 critical, 1,434 high, 3,570 medium)
- LLM harness: 1/8 attack success, 3 refused, 4 safe-proceed
- Stratification targets (50/50/40/12) and actuals (50/50/28/10) are both correctly reported
- Recalibrated risk score stats from pilot (62 repos): all match recalibrated_risk_scores.json

### Paper 3
- 386,067 addresses, 125,157 sybils (32.4%)
- OPS->fund mean AUC: 0.617 (source: 0.6175)
- Fund->OPS mean AUC: 0.516 (source: 0.5159)
- AI-only honest AUC: 0.501 (source: 0.5012)
- Enhanced honest AUC: 0.609 (source: 0.6089)
- Ablation all-8 AUC per level: basic 0.978, moderate 0.959, advanced 0.987 (all match after rounding)
- Cohen's d values: hour_entropy 1.98 (source: 1.983), behavioral_consistency 1.24 (source: 1.242)
- hour_entropy single-feature AUC range 0.864-0.878 (source: basic 0.8638, advanced 0.8776)
- Per-project baseline statistics (spot-checked)
- Pre-airdrop LightGBM: all AUCs = 1.0, all evasion = 100%
