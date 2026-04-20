# Project Status — 2026-04-20

## T-Minus
| Paper | Venue | Deadline | T-minus | Risk |
|-------|-------|----------|---------|------|
| P0 | Nature Machine Intelligence | **TBD** | ? | YELLOW — no deadline set |
| P1 | NeurIPS D&B 2026 | **TBD** | ? | YELLOW — no deadline set |
| P2 | IEEE S&P | **TBD** | ? | YELLOW — no deadline set |
| P3 | ACM CCS 2026 | **TBD** | ? | YELLOW — no deadline set |

## Since Last Commit (319ff41, Apr 17)
- **TODAY**: Full orchestration system deployed
  - 4× `results/manifest.json` (single source of truth)
  - 4× `macros_results.tex` (auto-generated LaTeX macros)
  - 4× Agent A ran: hardcoded numbers → macro references (176+ substitutions total)
  - 4× Agent B verified: all compile, 0 fixes needed
  - Scripts: `generate_macros.py`, `update_manifest.py`, `freeze_audit.py`
  - `ORCHESTRATION.md` protocol reference

## Experiment State
| Paper | Latest Result | Staleness | Running Now? |
|-------|--------------|-----------|--------------|
| P0 | reaction_time_full, decision_process | current | No |
| P1 | external_baseline_niedermayer | current | No |
| P2 | full_catalog_scan, precision_validation | current | No |
| P3 | experiment_graph_detector | current | No |

**No experiments currently running. No new results since manifests created.**

## Paper Readiness
| Paper | Pages | Compiles | Macros Wired | FLUID Markers | Blocking Issues |
|-------|-------|----------|-------------|---------------|----------------|
| P0 NatMI | 20 | YES | 25/61 active | 23 | 0 |
| P1 NeurIPS | 11 | YES | 38 active | 11 | 0 |
| P2 S&P | 13 | YES | 43 active | 22 | 0 |
| P3 CCS | 13 | YES | 55 active | 19 | 0 |

## Uncommitted Work
- 28 modified files + 32 new files (orchestration infra + macro substitutions)
- Needs commit + push NOW

## Next Actions (priority order)
1. **COMMIT**: orchestration infrastructure + Agent A macro work
2. **DECISION NEEDED**: Set actual deadlines for all 4 papers
3. **Stale figures**: All papers have figures 7-10 days old vs latest results
4. **P0**: NatMI word count re-check after macro substitution (was 4,883, limit 5,000)
5. **P3**: `fig_best_response_equilibrium.pdf` untracked — needs integration
