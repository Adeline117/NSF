# NSF: On-Chain AI Agent Security Research

An open-source framework for identifying on-chain AI agents, auditing their security posture, and scanning Web3 MCP server vulnerabilities.

## Research Papers

| Paper | Layer | Topic | Target Venue |
|-------|-------|-------|-------------|
| Paper 0 | Theory | Understanding AI Agents | CHI |
| Paper 1 | Foundation | On-chain AI Agent Identification & Security Posture | WWW |
| Paper 2 | Protocol | Web3 MCP Server Security Audit | S&P/USENIX |
| Paper 3 | Application | AI Agent-driven Sybil Attacks | GLOBECOM/CHI |

## Architecture

```
Paper 1 (Foundation Layer)
├── On-chain AI agent identification methodology
├── Labeled agent address dataset
└── Security posture quantification baseline
    ├── Permission exposure (unlimited approvals)
    ├── MEV exposure (sandwich attack rates)
    ├── Failure rates (revert analysis)
    └── Agent network topology

Paper 2 (Protocol Layer)          Paper 3 (Application Layer)
├── Web3 MCP server scanning      ├── AI Sybil evasion capability
├── Tool poisoning detection       ├── LLM behavioral fingerprinting
├── Private key extraction tests   ├── Cross-address correlation
└── Risk scoring framework         └── Enhanced detection features
```

## Project Structure

```
NSF/
├── paper0_ai_agent_theory/    # Paper 0: Theoretical framework
├── paper1_onchain_agent_id/   # Paper 1: On-chain agent identification
├── paper2_mcp_security/       # Paper 2: MCP server security audit
├── paper3_ai_sybil/           # Paper 3: AI Sybil attacks
├── shared/                    # Shared utilities and data collection
└── framework/                 # Open-source security audit framework
```

## Setup

```bash
pip install -r requirements.txt
cp shared/configs/config.example.yaml shared/configs/config.yaml
# Add your API keys (Etherscan, Dune, etc.)
```

## License

MIT
