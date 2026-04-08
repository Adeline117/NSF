# Agent Tool Scanner

Static analysis scanner for AI agent tool interface security. Detects vulnerabilities across four protocol families:

- **MCP** (Model Context Protocol)
- **OpenAI function calling**
- **LangChain tools**
- **Web3-native** (Safe modules, ERC-4337 plugins)

27 pattern detectors mapped to 16 CWEs and 2 OWASP LLM Top-10 categories.

## Install

```bash
pip install -e .
```

No non-stdlib Python runtime dependencies.

## Quick start

```bash
# Discover servers on GitHub
agent-tool-scanner enumerate --protocol mcp

# Scan a single cloned repo
agent-tool-scanner scan /path/to/repo

# Scan a directory of cloned repos
agent-tool-scanner batch ~/tmp/cloned_repos --output results.json

# Export the vulnerability taxonomy
agent-tool-scanner taxonomy
```

## Architecture

```
scanner/
  enumerate_servers.py     # GitHub search → server_catalog.json
  cli.py                   # unified CLI entrypoint
static_analysis/
  analyzer.py              # 27 VulnPatterns, risk scoring, harness detection
scripts/
  export_taxonomy.py       # taxonomy.json / taxonomy.md generation
  recalibrate_risk_score.py # p90-based risk score calibration
data/
  taxonomy.json            # machine-readable vulnerability catalog
  server_catalog.json      # discovered server inventory
paper/
  taxonomy.md              # human-readable Appendix A
  risk_score_calibration.md # Appendix C
  threat_model.md          # formal threat model
```

## Taxonomy summary

| Attack surface | Patterns | Categories |
|----------------|----------|------------|
| S1 Tool Definition | 4 | tool_poisoning, excessive_permissions, skill_scope_leak |
| S2 Input Construction | 6 | prompt_injection, command_injection, data_leakage, ... |
| S3 Tool Execution | 6 | privilege_escalation, private_key_exposure, ... |
| S4 Output Handling | 5 | state_confusion, no_output_validation, ... |
| S5 Tool Chain | 6 | tool_chain_escalation, missing_harness, ... |

See `paper/taxonomy.md` for the full catalog with CWE mappings.

## Risk scoring

Risk scores are calibrated to p90 of raw base scores across a real 62-repo scan set (`expected_max = 1460.7`). See `paper/risk_score_calibration.md` for derivation.

## Citation

```
@inproceedings{nsf2026agent,
  title   = {Agent Tool Interface Security: A Cross-Protocol Taxonomy},
  author  = {NSF Project Team},
  year    = {2026},
  booktitle = {TBD},
}
```

## License

Apache-2.0
