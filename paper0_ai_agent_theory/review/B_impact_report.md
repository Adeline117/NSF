# Agent B: Experiment Tracker Impact Report

**Date:** 2026-04-20
**Scope:** Paper 0 NatMI (`paper0_ai_agent_theory/natmi/main.tex`)

## Summary

| Metric | Value |
|--------|-------|
| Macros defined in `macros_results.tex` | 61 |
| Macros actively wired into `main.tex` | 25 |
| Total macro invocations in body text | 125 |
| FLUID markers added | 23 |
| Macros reserved (defined but unused) | 36 |

## Compilation Status

**Current state: compilable, all macros resolved**

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` exits cleanly.
- No "undefined control sequence" warnings in `main.log`.
- Output: 20 pages, 323,803 bytes.
- Bibliography resolves via `main.bbl` (no missing citations).

## Compilation Issues Encountered and Fixed

None. The paper compiled successfully on the first attempt with no errors or undefined macros.

## Macro Coverage Details

The 25 actively-used macros span all major result sections:

- **Corpus descriptors** (4): `\nSamples`, `\nFeaturesFull`, `\nFeaturesBase`, `\nClasses`
- **Classifier performance** (3): `\gbmAccuracyfeat`, `\gbmFMacrofeat`, `\dpBaselinefeatAccuracy`
- **Information-theoretic ceiling** (7): `\fanoMiJointBitsfeat`, `\fanoAccuracyCeilingfeat`, `\fanoAccuracyCeilingfeatB`, `\fanoErrorLowerBoundfeat`, `\fanoHeadroomPpfeat`, `\fanoHeadroomPpfeatB`, `\dpfeatAccuracy`
- **Reaction-time analysis** (6): `\rtCohensDLlmVsDefiMedian`, `\rtTotalDefiTxs`, `\rtLlmMedianReactionS`, `\rtRuleMedianReactionS`, `\rtLlmDefiExecMedianS`, `\rtRuleDefiExecMedianS`
- **Clustering** (4): `\clusterHdbscanNClusters`, `\clusterKmeansOptimalSilhouetteK`, `\clusterBestK`, `\ablationKmeansKSilhouette`
- **Decision process** (1): `\dpLlmFGain`

The 36 unused macros are reserved for supplementary materials (`supplementary.tex`) or future expansions (e.g., onchain validation, taxonomy comparison, ablation drops).

## FLUID Marker Distribution

FLUID markers annotate every section where numerical claims depend on `manifest.json`:

- Abstract (1)
- Section 2 Data & Methods (2)
- Section 3 Results: ceiling band (3), classifier table (2), clustering (3), MI analysis (1), feature importance (1), cross-chain (1)
- Section 4 Reaction Time (3)
- Section 5 Decision Process (2)
- Section 6 Discussion (3)
- Section 7 Related Work (1)

All markers follow the format `% FLUID: <dependency description>`, enabling automated staleness detection.
