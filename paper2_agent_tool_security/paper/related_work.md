# Related Work

## 2.1 MCP Security

The Model Context Protocol (MCP) was introduced by Anthropic in late 2024
as a standardized interface for connecting LLMs to external tools and data
sources [Anthropic 2024]. The protocol defines three primitives -- tools,
resources, and prompts -- communicated over stdio or SSE transports. The
MCP specification acknowledges security considerations including transport
security and tool sandboxing but does not mandate specific enforcement
mechanisms.

Invariant Labs [2025] published the first systematic security analysis of
MCP, identifying tool poisoning attacks where malicious tool descriptions
embed directives that manipulate agent behavior. Their work demonstrated
that MCP tool descriptions are treated as trusted instructions by the LLM,
enabling a malicious tool provider to hijack the agent's actions. They also
identified cross-tool information leakage via the shared LLM context
window. However, their analysis was limited to MCP only and did not
examine whether these vulnerabilities are protocol-specific or systemic.
Our work extends this by demonstrating that 18 of 20 vulnerability
categories manifest across all four protocol families, establishing that
the problems are architectural rather than protocol-specific.

Hou et al. [2025] analyzed the security of MCP in the context of
multi-agent systems, identifying risks of tool-to-tool escalation when
MCP servers expose `callTool` capabilities. Their threat model focused on
the composition surface (our S5), while our analysis covers all five
attack surfaces.

## 2.2 LLM Tool Use Security

Greshake et al. [2023] introduced the concept of indirect prompt injection,
demonstrating that adversarial content in data processed by an LLM
(web pages, emails, documents) can hijack the model's behavior.
Their work established that tool-augmented LLMs are particularly
vulnerable because injected instructions can trigger tool invocations
that the user did not request. We extend this work to the tool interface
layer specifically, showing that even without prompt injection, the tool
interface itself introduces critical vulnerabilities through tool
poisoning, excessive permissions, and missing input validation.

Zhan et al. [2024] systematically categorized attacks against LLM-based
agents, including tool-level attacks. Their taxonomy distinguishes between
attacks on the agent's reasoning (jailbreaking, manipulation) and attacks
via the tool interface (data poisoning, parameter injection). Our
vulnerability taxonomy (20 categories) is more granular and grounded in
static analysis patterns with CWE mappings, enabling automated detection.

The OWASP Top 10 for LLM Applications [2025] identifies "Excessive
Agency" (LLM06) as a top risk, where LLM-based systems are granted
excessive functionality, permissions, or autonomy. Our harness gap
analysis quantifies this risk empirically: we measure the percentage of
real tool servers that implement permission scoping, sandboxing,
transaction validation, and other controls. We find that the vast
majority lack any form of security harness.

Debenedetti et al. [2024] demonstrated AgentDojo, a framework for
evaluating the adversarial robustness of AI agents. Their work focused
on benchmark tasks where agents must resist prompt injection while
completing legitimate tool-use tasks. Our work is complementary: while
AgentDojo tests the agent's resilience, we analyze the tool servers
themselves for inherent vulnerabilities regardless of agent robustness.

## 2.3 Smart Contract Security

Smart contract security research provides essential context for the
Web3-specific vulnerabilities in our taxonomy.

Atzei, Bartoletti, and Cimoli [2017] published the seminal survey of
Ethereum smart contract vulnerabilities, categorizing attacks into
call-depth, reentrancy, gasless-send, type-confusion, and access-control
categories. Several of these map directly to our Web3-native tool
patterns: reentrancy through unchecked external calls (our S3-UC-001),
access control through missing authorization on module operations
(our S3-PE-001), and type confusion through missing input validation
(our S2-IV-001).

Chen et al. [2020] used machine learning to detect smart contract
vulnerabilities at scale, analyzing thousands of deployed contracts.
Their approach of pattern-based static analysis followed by
classification inspired our methodology. However, they focused on
standalone contracts, whereas our Web3-native analysis targets
modular smart account components (Safe modules, ERC-4337 plugins,
ERC-7579 modules) that introduce a new attack surface: the module
can act on behalf of the account with delegated authority.

Tsankov et al. [2018] developed Securify, a security analyzer for
Ethereum smart contracts based on semantic patterns. Our static
analysis patterns for Solidity (S3-DC-001 delegatecall abuse,
S3-AI-001 approval injection, W3-EP-001 excessive permissions) draw
on similar pattern-based detection but are specifically designed for
the intersection of AI agents and smart contracts, a combination
not addressed by existing tools.

The Gnosis Safe (now Safe) module system [Safe 2023] enables third-party
modules to execute transactions on behalf of a multi-signature wallet.
This architecture is analogous to the agent-tool interface: the module
(tool) acts with the Safe's (agent's) authority. Our Web3-native analysis
is the first to systematically assess the security of Safe modules and
ERC-4337 plugins as AI agent tool interfaces.

## 2.4 API Security

OWASP API Security Top 10 [2023] identifies common API vulnerabilities
including broken object-level authorization (API1), broken authentication
(API2), and excessive data exposure (API3). These map to our attack
surfaces: tool definition surface (API1 -- improper object access),
input construction surface (API2 -- broken authentication flow), and
output handling surface (API3 -- excessive data exposure). We argue
that AI agent tool interfaces are a specialized class of APIs with
additional attack surfaces not captured by the OWASP API Top 10,
specifically the tool definition surface (S1) where descriptions
influence LLM behavior, and the cross-tool surface (S5) where
information flows through the LLM context.

REST API security research [Atlidakis et al. 2019] has developed
automated testing tools (RESTler) for API endpoint fuzzing. Our
dynamic testing methodology draws on this approach but adapts it
for the LLM-mediated interaction model, where the "client" (agent)
interprets natural language descriptions rather than API documentation.

## 2.5 AI Agent Security Surveys

Xi et al. [2025] (arXiv 2601.04583) provide a comprehensive survey of
AI agent architectures, capabilities, and security considerations. They
identify tool use as a core capability and discuss associated risks at
a high level. Our work provides the empirical grounding that their survey
calls for: we quantify the prevalence of specific vulnerability categories
across real tool server implementations.

Wang et al. [2024] surveyed LLM-based autonomous agents, categorizing
them by architecture (single-agent, multi-agent) and capability
(reasoning, planning, tool use). Their security discussion focuses on
alignment and safety rather than the tool interface layer. Our
work addresses this gap.

Ruan et al. [2024] proposed a taxonomy of risks in AI agent systems,
distinguishing between risks from agent failures (misuse, hallucination)
and risks from the environment (adversarial inputs, tool failures). Our
five attack surfaces map to their environmental risk category, but we
provide a more detailed decomposition specific to the tool interface.

## 2.6 Our Contribution in Context

Our work makes several contributions not addressed by prior work:

1. **Cross-protocol scope.** Prior analyses focus on a single protocol
   (MCP, OpenAI, or smart contracts). We are the first to analyze all
   four protocol families under a unified framework, enabling cross-
   protocol comparison and identification of systemic vulnerabilities.

2. **Empirical scale.** Prior security analyses of AI agent tools are
   primarily conceptual or limited to a small number of case studies.
   We analyze 138 real tool servers from GitHub, providing statistical
   evidence of vulnerability prevalence.

3. **Unified taxonomy.** Our 21-category vulnerability taxonomy with
   CWE mappings, 5 attack surfaces, and 3 attacker types provides a
   reusable framework for future research. Existing taxonomies are
   either too general (OWASP LLM Top 10) or too narrow (MCP-specific).

4. **Harness gap analysis.** We introduce the "harness" concept as a
   measurable security property and empirically quantify the gap between
   what should exist (6 requirement categories) and what does exist in
   real deployments.

5. **Web3-specific vectors.** We identify four attack vectors specific to
   the intersection of AI agents and blockchain systems, where
   transaction finality makes vulnerabilities irreversible. This vertical
   analysis demonstrates the amplified impact of tool interface
   vulnerabilities in high-stakes domains.

6. **Reproducibility.** We release our static analysis tool, vulnerability
   patterns, and anonymized dataset to enable reproduction and extension.

---

## References

- Anthropic. "Model Context Protocol Specification." 2024.
  https://modelcontextprotocol.io/specification
- Atlidakis, V., Godefroid, P., and Polishchuk, M. "RESTler: Stateful
  REST API Fuzzing." ICSE 2019.
- Atzei, N., Bartoletti, M., and Cimoli, T. "A Survey of Attacks on
  Ethereum Smart Contracts (SoK)." POST 2017.
- Chen, J., Xia, X., Lo, D., Grundy, J., Luo, X., and Chen, T.
  "Defining Smart Contract Defects on Ethereum." IEEE TSE 2020.
- Debenedetti, E., et al. "AgentDojo: A Dynamic Environment to Assess
  Attacks and Defenses for LLM Agents." arXiv 2406.13352, 2024.
- Greshake, K., Abdelnabi, S., Mishra, S., Endres, C., Holz, T., and
  Fritz, M. "Not What You've Signed Up For: Compromising Real-World
  LLM-Integrated Applications with Indirect Prompt Injection."
  AISec@CCS 2023.
- Hou, X., et al. "Security Analysis of MCP-Based Multi-Agent Systems."
  arXiv 2025.
- Invariant Labs. "MCP Security Analysis: Tool Poisoning Attacks." 2025.
  https://invariantlabs.ai/blog/mcp-security-analysis
- OWASP. "OWASP API Security Top 10 -- 2023."
  https://owasp.org/API-Security/
- OWASP. "OWASP Top 10 for Large Language Model Applications." 2025.
  https://owasp.org/www-project-top-10-for-large-language-model-applications/
- Ruan, J., et al. "Identifying the Risks of LM Agents with an
  LM-Emulated Sandbox." arXiv 2309.15817, 2024.
- Safe. "Safe Modules Documentation." 2023.
  https://docs.safe.global/advanced/smart-account-modules
- Tsankov, P., Dan, A., Drachsler-Cohen, D., Gervais, A., Buenzli, F.,
  and Vechev, M. "Securify: Practical Security Analysis of Smart
  Contracts." CCS 2018.
- Wang, L., et al. "A Survey on Large Language Model Based Autonomous
  Agents." arXiv 2308.11432, 2024.
- Xi, Z., et al. "AI Agent: A Comprehensive Survey." arXiv 2601.04583,
  2025.
- Zhan, Q., et al. "Systematic Analysis of Attacks Against LLM-Based
  Agents." arXiv 2402.08276, 2024.
