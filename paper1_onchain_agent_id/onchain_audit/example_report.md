# OnChainAudit report — `OnChainAgentID v4 (N=1,147)`

_Generated 2026-04-16T06:03:17Z_

**Classifier:** `RandomForestClassifier`
**N addresses:** 1147
**N agents:** 533
**N humans:** 614

## Summary verdict

FAIL — at least one leakage indicator triggered:
- Step 1 flagged direct rule↔feature overlap.
- Step 3 flagged a cross-scheme transfer gap.
Investigate before claiming the reported AUC.

---

### Step 1 — Label/feature overlap

**Risk flags:**
- Rule `c1c4_c2_gas_precision` has Jaccard=0.512 with feature `tx_interval_skewness` (>= 0.5) — supports nearly coincide.
- Rule `c1c4_c3_hour_entropy` has |Pearson|=0.697 with feature `active_hour_entropy` (>= 0.6) — suspected direct leakage.
- Rule `c1c4_c3_hour_entropy` has Jaccard=0.572 with feature `active_hour_entropy` (>= 0.5) — supports nearly coincide.
- Rule `c1c4_c4_contract_diversity` has Jaccard=0.528 with feature `tx_interval_skewness` (>= 0.5) — supports nearly coincide.

**Max |Pearson| between each mining rule and any feature:**

| Rule | max |Pearson| | max Jaccard |
|------|---------------:|------------:|
| `c1c4_c1_tx_interval_cv` | 0.135 | 0.499 |
| `c1c4_c2_gas_precision` | 0.402 | 0.512 |
| `c1c4_c3_hour_entropy` | 0.697 | 0.572 |
| `c1c4_c4_contract_diversity` | 0.528 | 0.528 |
| `rule_chainlink_keeper` | 0.256 | 0.133 |
| `rule_defi_hf_trader` | 0.395 | 0.313 |
| `rule_ens_interaction` | 0.213 | 0.216 |

---

### Step 2 — Label-purity tier comparison

| Tier | N | Agents | Humans | AUC (mean) | AUC (std) |
|------|---:|------:|------:|----------:|----------:|
| all_mined_n1147 | 1147 | 533 | 614 | 0.807 | 0.018 |
| strict_curated | 70 | 35 | 35 | 0.767 | 0.103 |

**Drop vs best tier:** 0.039

---

### Step 3 — Cross-scheme transfer

| Direction | AUC |
|-----------|----:|
| Scheme A internal (CV) | 0.870 |
| Scheme B internal (CV) | 0.817 |
| Train A → Test B | 0.623 |
| Train B → Test A | 0.653 |

**Transfer gap (internal - transferred):** 0.205

**Warning:** Cross-scheme transfer drops by 0.205 below internal CV. The classifier learned scheme-specific cues, not a behavioural definition of 'agent' that generalises across label sources.
