# Living Paper Orchestration Protocol

## Architecture

```
paper{N}/
├── results/
│   └── manifest.json          ← SINGLE SOURCE OF TRUTH for all numbers
├── review/
│   ├── B_impact_report.md     ← Agent B output (what changed)
│   ├── C_major_changes.md     ← Agent C escalation (direction flips)
│   └── D_freeze_audit_*.md    ← Agent D pre-deadline audits
├── latex/ (or natmi/, neurips_db/)
│   ├── main.tex               ← paper body
│   └── macros_results.tex     ← AUTO-GENERATED, \input{} this
└── experiments/
    └── *_results.json         ← raw experiment outputs
```

## Layer Protocol

### Layer 1 (Stable) — write once, don't touch unless structure changes
- Abstract structure (numbers via `\macroname`)
- Introduction motivation + problem statement
- Related Work
- Method/algorithm description
- Notation, definitions
- Experimental setup framework
- Design-level limitations
- Conclusion framing

### Layer 2 (Fluid) — managed via manifest + macros
- All specific numbers → `\macroname` from `macros_results.tex`
- Table data rows → macros or auto-generated subtables
- Figures → regenerated from experiment scripts
- Results claims → hedged, direction-agnostic in Layer 1; specifics via macros
- Mark fluid sections: `% FLUID: [description]`

## Agent Commands

| Command | Triggers |
|---------|----------|
| "新结果出了 [exp]" | Agent B: update manifest → regenerate macros → impact report |
| "锁一下 [section]" | Lock: replace all FLUID markers in section with hardcoded values |
| "解锁 [section]" | Unlock: re-fluid-ize section, restore macro references |
| "做一次 freeze audit" | Agent D: 72h-level audit |
| "Adversarial review 一下" | Agent D: adversarial sub-mode |
| "show me orphaned" | List all `% ORPHANED` markers |

## Human Intervention Points (MUST stop and ask)

1. Layer classification ambiguous (L1 vs L2)
2. Manifest key collision
3. `C_major_changes.md` is non-empty (claim direction flipped)
4. Agent D Blocking list non-empty
5. Main thesis threatened by new results
6. Fluid placeholder has no data at deadline

## Paper Status

| Paper | Venue | Deadline | Status |
|-------|-------|----------|--------|
| P0 | Nature Machine Intelligence | TBD | Experiments complete, drafting NatMI version |
| P1 | NeurIPS D&B 2026 | TBD | Full draft, all 5 cascade levels done |
| P2 | IEEE S&P | TBD | Full draft, TCPI empirical + scan complete |
| P3 | ACM CCS 2026 | TBD | Full draft, 10-round equilibrium + external validation done |

## Scripts

```bash
# Regenerate all macros from manifests
python3 scripts/generate_macros.py

# Regenerate single paper
python3 scripts/generate_macros.py paper0

# Update manifest after new experiment (Agent B logic)
python3 scripts/update_manifest.py paper3 adversarial_best_response
```
