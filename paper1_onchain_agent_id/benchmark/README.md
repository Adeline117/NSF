# OnChainAgentID Benchmark

A provenance-verified benchmark for auditing on-chain
agent-vs-human classifiers. Released alongside Wen et al., *"Auditing
on-chain agent-vs-human classifiers for label leakage,"* WWW '26
(submitted).

This benchmark packages 1,147 Ethereum addresses with feature vectors,
provenance-tagged labels, and **four fixed evaluation splits** chosen to
expose label/feature leakage at increasing levels of rigour.

The benchmark's core contribution is not "here is another agent dataset" —
it is **a reproducible audit harness** (the `onchain_audit` Python package)
that lets any user plug in their own classifier and labels, and ask: is the
reported AUC driven by behavioural signal, or by the label-generating
rule?

-----------------------------------------------------------------

## 1. What's in the box

```
benchmark/
├── README.md               # this file
├── datasheet.md            # Gebru et al. (2018) style datasheet
├── LICENSE                 # CC BY 4.0 (data)
├── CITATION.cff            # (lives at repo root)
├── .zenodo.json            # (lives at repo root)
└── splits/
    ├── all_addresses.json           # full 1,147 index
    ├── level2_random_10fold.json    # random stratified 10-fold CV
    ├── level3_temporal_holdout.json # split by first-seen block
    ├── level4_strict_core.json      # hand-curated core (N=70)
    ├── level5_lopo.json             # leave-one-platform-out (14 folds)
    └── generate_splits.py           # reproducer
```

Features (not copied here, shared with the experiments code):

* `../data/features_provenance_v4.parquet` — 1,147 addresses × 23
  behavioural features + metadata columns.
* `../data/labels_provenance_v4.json` — per-address labels with
  `provenance_source`, `label_provenance ∈ {0, 1}` and category.
* `../data/raw/*.parquet` — 4,854 raw transaction files (one per address;
  used only by the temporal split and by feature re-extraction).

-----------------------------------------------------------------

## 2. The five evaluation levels

| Level | Name | N | Purpose |
|------:|------|---:|---------|
| 1 | In-sample fit | 1,147 | Sanity check on a trained model (not a split). |
| 2 | Random 10-fold CV | 1,147 | Random stratified split. **Upper bound only.** |
| 3 | Temporal holdout | 1,147 | Past → future. Tests drift robustness. |
| 4 | Strict curated core | 70 | Hand-curated labels only, no mining rules. The honest AUC ceiling. |
| 5 | Leave-One-Platform-Out (LOPO) | 14 folds | Train on 13 platforms, test on the 14th. The strongest generalisation test. |

A classifier that looks impressive at Level 2 but degrades sharply at
Levels 3–5 is picking up on the label-generating process rather than an
intrinsic behavioural signal. **Our own paper's main finding is exactly
this**: GBM AUC 0.98 at Level 2 → 0.63 at Level 4 → ~random at Level 5
for the hardest held-out platforms.

See `../FINDINGS_2026-04-08.md` for the full leakage post-mortem.

-----------------------------------------------------------------

## 3. Loading the splits

```python
import json
import pandas as pd
from pathlib import Path

root = Path("paper1_onchain_agent_id")
features = pd.read_parquet(root / "data" / "features_provenance_v4.parquet")

# Level 2 (random CV)
with open(root / "benchmark" / "splits" / "level2_random_10fold.json") as f:
    level2 = json.load(f)

fold0 = level2["folds"][0]
X_train = features.loc[fold0["train_addresses"]]
X_test  = features.loc[fold0["test_addresses"]]
y_train = X_train["label"].astype(int)
y_test  = X_test["label"].astype(int)

# Level 4 (strict core)
with open(root / "benchmark" / "splits" / "level4_strict_core.json") as f:
    level4 = json.load(f)
X_strict = features.loc[level4["addresses"]]
y_strict = pd.Series(level4["labels"], index=level4["addresses"])
```

-----------------------------------------------------------------

## 4. Using the audit harness

Audit any binary classifier on this (or your own) dataset:

```python
from sklearn.ensemble import RandomForestClassifier
from onchain_audit import generate_audit_report
from onchain_audit.example import load_example_dataset

clf = RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42)
dataset = load_example_dataset()           # or your own AuditDataset
print(generate_audit_report(clf, dataset)) # Markdown, ~1 min on laptop
```

The report emits three sections:

1. **Label/feature overlap** — Jaccard + Pearson between your mining rules
   and your model's features.
2. **Purity-tier comparison** — AUC across tiers of increasing label purity.
3. **Cross-scheme transfer** — train on scheme A, evaluate on scheme B.

See `../onchain_audit/example.py` for the reference wiring, and
`../notebooks/tutorial.ipynb` for a 10-minute end-to-end walkthrough.

-----------------------------------------------------------------

## 5. Reproducing headline numbers

Provenance-only pipeline (Level 2):

```bash
cd paper1_onchain_agent_id
python experiments/run_provenance_pipeline.py
```

Temporal holdout (Level 3):

```bash
python experiments/run_temporal_holdout.py
```

Leave-One-Platform-Out (Level 5):

```bash
python experiments/leave_one_platform_out.py
```

Regenerate splits from scratch:

```bash
python benchmark/splits/generate_splits.py
```

-----------------------------------------------------------------

## 6. Citing

If you use the OnChainAgentID benchmark or the `onchain_audit` toolkit,
please cite using the entry in `/CITATION.cff`.

```bibtex
@inproceedings{wen2026onchainagentid,
  title     = {OnChainAgentID: a provenance-verified benchmark for auditing
               on-chain agent-vs-human classifiers},
  author    = {Wen, Adeline and others},
  booktitle = {Proceedings of the ACM Web Conference (WWW '26)},
  year      = {2026},
  note      = {Benchmark and audit toolkit at
               \url{https://github.com/adelinewen/NSF/tree/main/paper1_onchain_agent_id}},
}
```

-----------------------------------------------------------------

## 7. Known limitations

- **Ethereum only.** All 1,147 addresses are on Ethereum mainnet. A
  Polygon extension exists in `../../paper3_ai_sybil/experiments/` but
  is not part of this benchmark.
- **18 operational categories.** Six categories have fewer than 10
  addresses and are excluded from the Level 5 LOPO folds. These are
  retained in Levels 2–4 but under-represent the long tail of agent
  archetypes.
- **Platform skew.** The largest single platform (`defi_hf_trader`,
  17.4%) is an on-chain heuristic (`> 100 tx/week`). Classifiers that
  rely on raw transaction frequency will look artificially strong.
- **Label provenance is not ground truth.** `label_provenance = 1`
  signals that an address has an on-chain or off-chain attestation
  (Flashbots relay, Etherscan label, Arkham label, contract source
  code). It does **not** guarantee the underlying entity is an agent in
  the strong sense used by Paper 0's taxonomy.

See `datasheet.md` §6 (Uses) and §7 (Distribution) for more.

-----------------------------------------------------------------

## 8. License

Data: CC BY 4.0 (see `LICENSE`).
Code (`onchain_audit/`, `experiments/`): MIT (see repository-level
`LICENSE`).
