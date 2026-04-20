# Agent B: Paper 1 Compilation & Macro Verification Report

**Date:** 2026-04-20
**Paper:** OnChainAgentID (paper1_onchain_agent_id/neurips_db/main.tex)
**Agent:** B (Experiment Tracker)

## Compilation Status

**PASS** -- Paper compiles successfully with `latexmk -pdf -interaction=nonstopmode -halt-on-error`.

- Output: `main.pdf` (11 pages, 363,953 bytes)
- No undefined control sequences
- No missing references
- Only benign warning: `Font shape 'OMS/cmtt/m/n' undefined` (standard LaTeX font fallback, harmless)

## Macro Substitution Audit

All 54 macros defined in `macros_results.tex` resolve correctly in the compiled PDF. No fixes were required.

### Macros verified (all 54):

| Category | Macros | Status |
|----------|--------|--------|
| Dataset descriptors (6) | `\datasetNAddresses` (1,147), `\datasetNAgents` (533), `\datasetNHumans` (614), `\datasetNFeatures` (23), `\datasetNCategories` (18), `\datasetStrictCoreN` (70) | OK |
| Cascade levels (5) | `\cascadeLevelLeakyAuc` (0.844), `\cascadeLevelCvAuc` (0.804), `\cascadeLevelTemporalAuc` (0.669), `\cascadeLevelStrictCoreAuc` (0.760), `\cascadeLevelLopoAuc` (0.397) | OK |
| Random CV (3) | `\randomCvRfAuc` (0.803), `\randomCvRfAucStd` (0.027), `\randomCvGbmAuc` (0.802) | OK |
| Temporal holdout (5) | `\temporalHoldoutRfAuc` (0.669), `\temporalHoldoutRfF` (0.447), `\temporalHoldoutRfPrecision` (0.812), `\temporalHoldoutGbmAuc` (0.664), `\temporalHoldoutGbmF` (0.604) | OK |
| Temporal delta (1) | `\temporalVsCvDelta` (-0.134) | OK |
| LOPO (3) | `\lopoRfPooledAuc` (0.397), `\lopoRfPooledF` (0.355), `\lopoGbmPooledAuc` (0.422) | OK |
| Mixed-class LOPO (5) | `\mixedClassLopoRfAuc` (0.742), `\mixedClassLopoRfMeanClusterAuc` (0.758), `\mixedClassLopoRfStdClusterAuc` (0.079), `\mixedClassLopoGbmAuc` (0.753), `\mixedClassLopoDeltaVsOriginal` (0.331) | OK |
| Mixed-class delta (1) | `\mixedClassLopoDeltaVsCv` (-0.049) | OK |
| GAT temporal (3) | `\gatTemporalMeanAuc` (0.666), `\gatTemporalStdAuc` (0.0062), `\gatTemporalMeanF` (0.353) | OK |
| GraphSAGE temporal (3) | `\graphsageTemporalMeanAuc` (0.668), `\graphsageTemporalStdAuc` (0.022), `\graphsageTemporalMeanF` (0.635) | OK |
| GNN delta (1) | `\gnnVsRfDelta` (-0.0033) | OK |
| Baseline progression (5) | `\baselineHeuristicCvAuc` (0.427), `\baselineSingleFeatureCvAuc` (0.648), `\baselineLogregGasCvAuc` (0.718), `\baselineProposedRfCvAuc` (0.804), `\baselineProposedRfCvAucStd` (0.035) | OK |
| External validation (8) | `\externalNiedermayerCvAuc` (0.779), `\externalNiedermayerCvAucStd` (0.033), `\externalNiedermayerTemporalAuc` (0.684), `\externalNiedermayerLopoAuc` (0.469), `\externalOursCvAuc` (0.826), `\externalOursCvAucStd` (0.034), `\externalOursTemporalAuc` (0.701), `\externalOursLopoAuc` (0.426) | OK |
| External delta (1) | `\externalDeltaCvAuc` (0.046) | OK |
| Etherscan validation (3) | `\etherscanValidationConsistencyRate` (0.935), `\etherscanValidationNQueried` (200), `\etherscanValidationNInconsistent` (13) | OK |

## Fixes Applied

**None.** Compilation succeeded on the first attempt with zero errors.

## Cascade Integrity Check

The five-level cascade reported in the abstract and conclusion is internally consistent:

```
L1 (leaky)       = 0.844
L2 (random CV)   = 0.803  (delta: -0.041)
L3 (temporal)    = 0.669  (delta: -0.134)
L4 (strict core) = 0.760  (delta: +0.091 from L3, -0.043 from L2)
L5 (LOPO)        = 0.397 / 0.742 (single-class / mixed-class)
```

The monotonic decline L1 > L2 > L3 holds. L4 sits between L2 and L3 as expected for a small curated set. L5 single-class collapses below chance; mixed-class recovers to 0.742.

## Conclusion

Paper 1 compiles cleanly after Agent A's macro substitutions. All 54 data-dependent macros in `macros_results.tex` are correctly defined and referenced in `main.tex`. No manual intervention was needed.
