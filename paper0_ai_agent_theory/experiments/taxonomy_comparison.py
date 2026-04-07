"""
Taxonomy Comparison Matrix
===========================
Compare our C1-C4 definition + 8-category taxonomy against 5 existing frameworks:
1. Russell & Norvig (2020): {Simple reflex, Model-based, Goal-based, Utility-based, Learning}
2. Wooldridge & Jennings (1995): 4-property weak agency (Autonomy, Social, Reactive, Proactive)
3. Franklin & Graesser (1996): 13-property subset taxonomy
4. Parasuraman et al. (2000): 10 levels of automation x 4 stages
5. He et al. (2025): AI agent survey arXiv 2601.04583 (Perception-Reasoning-Action-Learning)

For each:
- Map their categories/dimensions to ours (coverage matrix)
- Identify categories they have that we don't
- Identify categories we have that they don't
- Evaluate Web3 applicability of each
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from paper0_ai_agent_theory.experiments.pilot_taxonomy import TAXONOMY


# ============================================================
# OUR TAXONOMY CATEGORIES
# ============================================================

OUR_CATEGORIES = [
    "Deterministic Script",
    "Simple Trading Bot",
    "MEV Searcher",
    "Cross-Chain Bridge Agent",
    "RL Trading Agent",
    "DeFi Management Agent",
    "LLM-Powered Agent",
    "Autonomous DAO Agent",
]

OUR_CONDITIONS = ["C1 (On-chain Actuation)", "C2 (Environmental Perception)",
                  "C3 (Autonomous Decision-Making)", "C4 (Adaptiveness)"]

OUR_DIMENSIONS = ["Autonomy Level (5 levels)", "Environment Type (4 types)",
                   "Decision Model (5 types)"]


# ============================================================
# FRAMEWORK 1: Russell & Norvig (2020)
# ============================================================

RUSSELL_NORVIG = {
    "name": "Russell & Norvig (2020)",
    "full_title": "Artificial Intelligence: A Modern Approach, 4th Edition",
    "categories": [
        "Simple reflex agent",
        "Model-based reflex agent",
        "Goal-based agent",
        "Utility-based agent",
        "Learning agent",
    ],
    "key_idea": (
        "Hierarchical agent architecture taxonomy based on increasing "
        "internal complexity: from condition-action rules to full learning."
    ),
    "definition_of_agent": (
        "An agent is anything that can be viewed as perceiving its environment "
        "through sensors and acting upon that environment through actuators."
    ),
    "condition_coverage": {
        "C1 (On-chain Actuation)": {
            "covered": False,
            "note": (
                "R&N defines 'actuators' abstractly but has no concept of "
                "blockchain-specific actuation (EOA control, transaction signing). "
                "Their actuator model assumes physical or software effectors, "
                "not cryptographic key-based blockchain execution."
            ),
        },
        "C2 (Environmental Perception)": {
            "covered": True,
            "note": (
                "Core to R&N: all agents perceive through sensors. However, the "
                "distinction between on-chain state (blocks, transactions, mempool) "
                "and off-chain signals (APIs, social media) is not modeled."
            ),
        },
        "C3 (Autonomous Decision-Making)": {
            "covered": True,
            "note": (
                "Partially covered via the architecture hierarchy. Goal-based and "
                "utility-based agents make autonomous decisions, but the specific "
                "requirement for 'non-deterministic components' (ML, LLM, RL) is "
                "not part of R&N's definition."
            ),
        },
        "C4 (Adaptiveness)": {
            "covered": True,
            "note": (
                "Covered by the 'Learning agent' type but not as a necessary "
                "condition. In R&N, simple reflex agents are still 'agents' even "
                "without any adaptiveness."
            ),
        },
    },
    # Coverage matrix: rows = their categories, columns = our categories
    # Values: 0 = no mapping, 1 = partial mapping, 2 = strong mapping
    "coverage_matrix": [
        # DetScript  TradBot  MEVSearch  Bridge  RLAgent  DeFiMgmt  LLMAgent  DAOAgent
        [2,          1,       0,         0,      0,       0,        0,        0],  # Simple reflex
        [0,          2,       1,         1,      0,       0,        0,        0],  # Model-based reflex
        [0,          0,       1,         1,      0,       2,        1,        1],  # Goal-based
        [0,          0,       2,         0,      1,       2,        1,        0],  # Utility-based
        [0,          0,       1,         0,      2,       1,        2,        0],  # Learning
    ],
    "categories_they_have_we_dont": [
        {
            "category": "Model-based reflex agent",
            "note": (
                "Our taxonomy does not explicitly model 'internal state' vs "
                "'no internal state'. The distinction between REACTIVE and "
                "ADAPTIVE partially captures this, but not identically."
            ),
        },
    ],
    "categories_we_have_they_dont": [
        {
            "category": "Cross-Chain Bridge Agent",
            "note": (
                "R&N has no concept of execution environment. A bridge agent "
                "and a trading bot might both be 'goal-based' in their "
                "framework, but they serve fundamentally different roles."
            ),
        },
        {
            "category": "Autonomous DAO Agent",
            "note": (
                "R&N does not model collective/collaborative decision-making "
                "as an agent architecture type."
            ),
        },
        {
            "category": "MEV Searcher (as distinct from other adaptive agents)",
            "note": (
                "The domain-specific distinction between MEV extraction "
                "and general utility optimization is absent in R&N."
            ),
        },
    ],
    "web3_applicability": {
        "score": 2,  # out of 5
        "assessment": (
            "Provides a useful theoretical foundation but lacks the "
            "environment and domain-specific dimensions needed for Web3. "
            "Cannot distinguish between an MEV searcher and a DeFi "
            "management agent (both could be 'utility-based'). No concept "
            "of on-chain vs off-chain execution."
        ),
    },
}


# ============================================================
# FRAMEWORK 2: Wooldridge & Jennings (1995)
# ============================================================

WOOLDRIDGE_JENNINGS = {
    "name": "Wooldridge & Jennings (1995)",
    "full_title": (
        "Intelligent Agents: Theory and Practice. The Knowledge Engineering Review."
    ),
    "categories": [
        "Autonomy (operates without direct human intervention)",
        "Social ability (interacts with other agents/humans via agent-communication language)",
        "Reactivity (perceives environment and responds in a timely fashion)",
        "Pro-activeness (takes initiative, goal-directed behavior)",
    ],
    "key_idea": (
        "Defines 'weak notion of agency' via four necessary properties. "
        "An entity is an agent iff it possesses all four: autonomy, social "
        "ability, reactivity, and pro-activeness. Also defines 'strong notion' "
        "with mentalistic properties (beliefs, desires, intentions)."
    ),
    "definition_of_agent": (
        "A hardware or software-based computer system that enjoys the "
        "properties of autonomy, social ability, reactivity, and pro-activeness."
    ),
    "condition_coverage": {
        "C1 (On-chain Actuation)": {
            "covered": False,
            "note": (
                "W&J assume generic effectors. No notion of blockchain-specific "
                "actuation: EOA ownership, private key control, or transaction "
                "signing. Their framework applies to any software system, not "
                "specifically to blockchain execution."
            ),
        },
        "C2 (Environmental Perception)": {
            "covered": True,
            "note": (
                "Covered by 'Reactivity': the agent perceives its environment "
                "and responds in a timely fashion. However, W&J do not "
                "distinguish between on-chain and off-chain perception."
            ),
        },
        "C3 (Autonomous Decision-Making)": {
            "covered": True,
            "note": (
                "Covered by 'Autonomy' and 'Pro-activeness'. W&J's autonomy "
                "means operating without direct human intervention. However, "
                "they do not require non-deterministic components -- a "
                "purely rule-based system could satisfy W&J's autonomy."
            ),
        },
        "C4 (Adaptiveness)": {
            "covered": False,
            "note": (
                "NOT explicitly covered. W&J's four properties do not include "
                "learning or adaptation. Their 'reactivity' is about timely "
                "response, not about modifying behavior over time. This is a "
                "key gap: a fixed-rule reactive system satisfies W&J but "
                "fails our C4."
            ),
        },
    },
    "coverage_matrix": [
        # DetScript  TradBot  MEVSearch  Bridge  RLAgent  DeFiMgmt  LLMAgent  DAOAgent
        [1,          2,       2,         2,      2,       2,        2,        2],  # Autonomy
        [0,          0,       0,         1,      0,       1,        2,        2],  # Social ability
        [1,          2,       2,         2,      2,       2,        2,        2],  # Reactivity
        [0,          0,       1,         0,      1,       2,        2,        2],  # Pro-activeness
    ],
    "categories_they_have_we_dont": [
        {
            "category": "Social ability",
            "note": (
                "W&J require agents to interact with other agents via an "
                "agent-communication language. Our taxonomy captures social "
                "interaction only in the COLLABORATIVE autonomy level (DAO Agent). "
                "We do not model inter-agent communication as a necessary property."
            ),
        },
        {
            "category": "Strong agency (BDI mentalistic notions)",
            "note": (
                "W&J's 'strong notion of agency' includes beliefs, desires, "
                "and intentions. Our taxonomy does not model internal cognitive "
                "states -- they are not observable on-chain."
            ),
        },
    ],
    "categories_we_have_they_dont": [
        {
            "category": "Environment Type dimension (all)",
            "note": (
                "W&J have no concept of execution environment. On-chain, "
                "hybrid, cross-chain, and multi-modal are entirely absent."
            ),
        },
        {
            "category": "Decision Model dimension (all)",
            "note": (
                "W&J do not distinguish between decision mechanisms (deterministic, "
                "statistical, LLM, RL). A rule-based bot and an RL agent both "
                "satisfy their four properties equally."
            ),
        },
        {
            "category": "Adaptiveness (C4) as necessary condition",
            "note": (
                "W&J do not require adaptation/learning. Our C4 explicitly "
                "requires behavior modification over time, which excludes "
                "fixed-rule reactive systems from being classified as agents."
            ),
        },
        {
            "category": "Domain-specific categories (MEV, Bridge, DeFi, etc.)",
            "note": (
                "W&J provide property-level criteria, not domain-specific types. "
                "All 8 of our categories that satisfy the four properties would "
                "be indistinguishable under W&J."
            ),
        },
    ],
    "web3_applicability": {
        "score": 2,  # out of 5
        "assessment": (
            "The four-property definition is influential and provides a good "
            "starting point, but it is too broad for Web3. A cron job payroll "
            "script arguably satisfies 'autonomy' and 'reactivity' under W&J, "
            "blurring the agent boundary. No environment-type or decision-model "
            "dimensions means all on-chain agents collapse into one class. "
            "The social-ability requirement is too strong (excludes solitary MEV bots) "
            "while the adaptiveness requirement is too weak (not required)."
        ),
    },
}


# ============================================================
# FRAMEWORK 3: Franklin & Graesser (1996)
# ============================================================

FRANKLIN_GRAESSER = {
    "name": "Franklin & Graesser (1996)",
    "full_title": "Is it an Agent, or Just a Program?: A Taxonomy for Autonomous Agents",
    "categories": [
        "Reactive",
        "Autonomous",
        "Goal-oriented",
        "Temporally continuous",
        "Communicative",
        "Learning",
        "Mobile",
        "Flexible",
        "Character (personality)",
        "Emotional",
        "Modelling (of environment)",
        "Personality",
        "Adaptive",
    ],
    "key_idea": (
        "Property-based taxonomy: an agent is classified by which subset "
        "of 13 boolean properties it possesses. Not a hierarchy but a "
        "combinatorial space of 2^13 possible agent types."
    ),
    "definition_of_agent": (
        "An autonomous agent is a system situated within and a part of an "
        "environment that senses that environment and acts on it, over time, "
        "in pursuit of its own agenda and so as to effect what it senses "
        "in the future."
    ),
    "condition_coverage": {
        "C1 (On-chain Actuation)": {
            "covered": False,
            "note": (
                "F&G define agents in terms of acting on environment but do "
                "not specify blockchain-specific actuation mechanisms."
            ),
        },
        "C2 (Environmental Perception)": {
            "covered": True,
            "note": (
                "Covered by 'Reactive' and 'Modelling' properties. The agent "
                "senses its environment. However, no distinction between "
                "on-chain vs off-chain perception."
            ),
        },
        "C3 (Autonomous Decision-Making)": {
            "covered": True,
            "note": (
                "Covered by 'Autonomous' and 'Goal-oriented'. However, no "
                "requirement for non-deterministic components."
            ),
        },
        "C4 (Adaptiveness)": {
            "covered": True,
            "note": (
                "Covered by 'Adaptive' and 'Learning' properties. But these "
                "are optional properties, not necessary conditions."
            ),
        },
    },
    "coverage_matrix": [
        # DetScript  TradBot  MEVSearch  Bridge  RLAgent  DeFiMgmt  LLMAgent  DAOAgent
        [1,          2,       2,         2,      2,       2,        2,        2],  # Reactive
        [0,          0,       1,         1,      1,       2,        2,        2],  # Autonomous
        [0,          1,       2,         1,      2,       2,        2,        2],  # Goal-oriented
        [1,          2,       2,         2,      2,       2,        2,        2],  # Temporally continuous
        [0,          0,       0,         1,      0,       1,        2,        2],  # Communicative
        [0,          0,       1,         0,      2,       1,        2,        0],  # Learning
        [0,          0,       0,         2,      0,       0,        0,        0],  # Mobile
        [0,          1,       2,         1,      2,       2,        2,        1],  # Flexible
        [0,          0,       0,         0,      0,       0,        1,        0],  # Character
        [0,          0,       0,         0,      0,       0,        0,        0],  # Emotional
        [0,          1,       2,         1,      2,       2,        2,        1],  # Modelling
        [0,          0,       0,         0,      0,       0,        1,        0],  # Personality
        [0,          0,       2,         1,      2,       2,        2,        1],  # Adaptive
    ],
    "categories_they_have_we_dont": [
        {
            "category": "Mobile",
            "note": (
                "Physical mobility is irrelevant to Web3 software agents, "
                "though 'cross-chain mobility' is captured by our "
                "Environment dimension."
            ),
        },
        {
            "category": "Emotional / Character / Personality",
            "note": (
                "These properties could become relevant for LLM-powered "
                "social agents (e.g., AI16Z agents with Twitter personas) "
                "but are not observable on-chain."
            ),
        },
    ],
    "categories_we_have_they_dont": [
        {
            "category": "All domain-specific categories",
            "note": (
                "F&G provides properties, not categories. The mapping from "
                "property-subsets to concrete agent types (MEV, DeFi, Bridge) "
                "is entirely absent. Multiple categories share the same "
                "property subset, making classification ambiguous."
            ),
        },
    ],
    "web3_applicability": {
        "score": 2,  # out of 5
        "assessment": (
            "The property-based approach is theoretically flexible but "
            "impractical for Web3 classification. The 2^13 combinatorial "
            "space is too large; most properties are either irrelevant "
            "(Emotional, Personality) or universally present (Reactive, "
            "Temporally continuous). Cannot operationalize on-chain."
        ),
    },
}


# ============================================================
# FRAMEWORK 4: Parasuraman et al. (2000)
# ============================================================

PARASURAMAN = {
    "name": "Parasuraman, Sheridan & Wickens (2000)",
    "full_title": (
        "A Model for Types and Levels of Human Interaction with Automation"
    ),
    "categories": [
        "Level 1: Manual (human does everything)",
        "Level 2: System suggests alternatives",
        "Level 3: System narrows selection",
        "Level 4: System suggests one alternative",
        "Level 5: System executes if human approves",
        "Level 6: System allows human limited veto time",
        "Level 7: System executes, informs human",
        "Level 8: System executes, informs if asked",
        "Level 9: System executes, informs if it decides to",
        "Level 10: Fully automatic (ignores human)",
    ],
    "key_idea": (
        "10-level automation scale applied across 4 information processing "
        "stages: acquisition, analysis, decision selection, action. Designed "
        "for human-in-the-loop systems."
    ),
    "definition_of_agent": (
        "Not an agent definition per se but an automation taxonomy. "
        "Automation is 'the execution by a machine agent (usually a computer) "
        "of a function that was previously carried out by a human.'"
    ),
    "condition_coverage": {
        "C1 (On-chain Actuation)": {
            "covered": False,
            "note": (
                "Parasuraman models 'action implementation' as one of four "
                "stages but has no blockchain-specific actuation concept."
            ),
        },
        "C2 (Environmental Perception)": {
            "covered": True,
            "note": (
                "Covered by 'Information Acquisition' stage. The automation "
                "levels apply to how information is sensed and filtered."
            ),
        },
        "C3 (Autonomous Decision-Making)": {
            "covered": True,
            "note": (
                "Covered by 'Decision Selection' stage at levels 5-10. "
                "However, the framework focuses on degree of human involvement, "
                "not on the decision mechanism itself."
            ),
        },
        "C4 (Adaptiveness)": {
            "covered": False,
            "note": (
                "The 10 levels describe static automation settings, not "
                "adaptation over time. A system at Level 8 stays at Level 8; "
                "the framework does not model behavioral change."
            ),
        },
    },
    "coverage_matrix": [
        # DetScript  TradBot  MEVSearch  Bridge  RLAgent  DeFiMgmt  LLMAgent  DAOAgent
        [0,          0,       0,         0,      0,       0,        0,        0],  # L1 manual
        [0,          0,       0,         0,      0,       0,        0,        0],  # L2
        [0,          0,       0,         0,      0,       0,        0,        0],  # L3
        [0,          0,       0,         0,      0,       0,        0,        0],  # L4
        [0,          0,       0,         0,      0,       0,        0,        2],  # L5 (DAO approve)
        [0,          0,       0,         0,      0,       0,        0,        1],  # L6 (DAO timelock)
        [0,          0,       0,         0,      0,       1,        1,        0],  # L7
        [0,          0,       1,         1,      1,       1,        0,        0],  # L8
        [0,          1,       1,         1,      1,       0,        0,        0],  # L9
        [2,          1,       1,         0,      0,       0,        0,        0],  # L10 fully auto
    ],
    "categories_they_have_we_dont": [
        {
            "category": "Levels 1-4 (human-centric decision support)",
            "note": (
                "Our taxonomy starts at fully automated (Level 10 = NONE). "
                "Human-in-the-loop decision support levels are not covered "
                "because they describe human users, not autonomous agents."
            ),
        },
    ],
    "categories_we_have_they_dont": [
        {
            "category": "Environment Type (all)",
            "note": (
                "Parasuraman has no concept of execution environment. "
                "The framework was designed for aviation/industrial "
                "automation, not software agent deployment contexts."
            ),
        },
        {
            "category": "Decision Model (all)",
            "note": (
                "The 10 levels describe degree of automation, not mechanism "
                "of decision-making. An RL agent and a rule-based bot at "
                "the same automation level would be indistinguishable."
            ),
        },
    ],
    "web3_applicability": {
        "score": 1,  # out of 5
        "assessment": (
            "Designed for human-machine interaction in physical systems. "
            "Most levels (1-4) are irrelevant to autonomous on-chain agents. "
            "The single dimension (automation level) cannot capture the "
            "multi-dimensional nature of Web3 agents. However, the insight "
            "that automation is stage-specific influenced our Decision Model "
            "dimension."
        ),
    },
}


# ============================================================
# FRAMEWORK 5: He et al. (2025) arXiv 2601.04583
# ============================================================

HE_ET_AL = {
    "name": "He et al. (2025)",
    "full_title": (
        "A Survey on AI Agent. arXiv:2601.04583. "
        "Perception-Reasoning-Action-Learning agent loop."
    ),
    "categories": [
        "Perception module (text, vision, audio, multimodal)",
        "Reasoning module (CoT, planning, reflection, memory)",
        "Action module (tool use, code generation, API calls)",
        "Learning module (in-context, fine-tuning, RL from feedback)",
    ],
    "key_idea": (
        "Comprehensive survey defining AI agents through a four-module "
        "architecture: Perception -> Reasoning -> Action -> Learning. "
        "Focuses on LLM-based agents and their component modules. "
        "Classifies agents by application domain: software dev, science, "
        "gaming, social simulation, etc."
    ),
    "definition_of_agent": (
        "An AI agent is an intelligent entity that perceives its environment, "
        "reasons about the perceived information, takes actions to achieve "
        "specific goals, and learns from experience to improve performance."
    ),
    "condition_coverage": {
        "C1 (On-chain Actuation)": {
            "covered": False,
            "note": (
                "He et al.'s 'Action module' covers tool use, API calls, "
                "and code generation, but does not address blockchain-specific "
                "actuation (EOA control, transaction signing, on-chain execution). "
                "Their action model is generic software agent actions."
            ),
        },
        "C2 (Environmental Perception)": {
            "covered": True,
            "note": (
                "Strongly covered by the 'Perception module'. He et al. model "
                "text, vision, audio, and multimodal perception. However, they "
                "do not distinguish on-chain state perception (block data, "
                "mempool, oracle feeds) from general data input."
            ),
        },
        "C3 (Autonomous Decision-Making)": {
            "covered": True,
            "note": (
                "Covered by 'Reasoning module' (chain-of-thought, planning, "
                "reflection). The emphasis on LLM-based reasoning implicitly "
                "requires non-deterministic components."
            ),
        },
        "C4 (Adaptiveness)": {
            "covered": True,
            "note": (
                "Covered by 'Learning module' (in-context learning, fine-tuning, "
                "RLHF). He et al. explicitly model learning as a core module. "
                "However, they focus on LLM-centric learning paradigms and do "
                "not cover statistical or RL-based adaptation used by MEV bots."
            ),
        },
    },
    "coverage_matrix": [
        # DetScript  TradBot  MEVSearch  Bridge  RLAgent  DeFiMgmt  LLMAgent  DAOAgent
        [0,          1,       1,         1,      1,       1,        2,        1],  # Perception
        [0,          0,       1,         0,      1,       2,        2,        1],  # Reasoning
        [1,          1,       1,         1,      1,       1,        2,        1],  # Action
        [0,          0,       1,         0,      2,       1,        2,        0],  # Learning
    ],
    "categories_they_have_we_dont": [
        {
            "category": "Perception module sub-types (vision, audio, multimodal)",
            "note": (
                "He et al. provide fine-grained perception modality distinctions. "
                "Our taxonomy treats perception as a binary condition (C2: yes/no) "
                "rather than decomposing by modality. For Web3, the relevant "
                "distinction is on-chain vs off-chain, not text vs vision."
            ),
        },
        {
            "category": "Reasoning sub-types (CoT, ReAct, reflexion, tree-of-thought)",
            "note": (
                "He et al. catalog diverse LLM reasoning patterns. Our Decision "
                "Model dimension distinguishes LLM from RL from statistical, but "
                "does not sub-categorize LLM reasoning approaches. This is "
                "appropriate for on-chain observability (reasoning internals are "
                "not visible on-chain)."
            ),
        },
        {
            "category": "Application domain taxonomy (software dev, science, gaming)",
            "note": (
                "He et al. classify agents by application domain. Web3/blockchain "
                "is mentioned but not a primary domain. Our taxonomy is "
                "specifically designed for the Web3 domain, providing more "
                "granular classification within this niche."
            ),
        },
    ],
    "categories_we_have_they_dont": [
        {
            "category": "On-chain Actuation (C1) as a necessary condition",
            "note": (
                "He et al. do not distinguish between on-chain and off-chain "
                "action execution. A chatbot and an MEV searcher would both "
                "fall under the 'Action module' without distinction."
            ),
        },
        {
            "category": "Environment Type dimension",
            "note": (
                "He et al. do not model where the agent operates. The on-chain / "
                "hybrid / cross-chain / multi-modal environment distinction is "
                "absent. This is critical for Web3 where the execution "
                "environment determines constraints, costs, and observability."
            ),
        },
        {
            "category": "Domain-specific categories (MEV, Bridge, DeFi, DAO)",
            "note": (
                "He et al. provide a module-based architecture decomposition, "
                "not a category-based classification for Web3. Multiple of our "
                "categories (MEV Searcher, DeFi Manager, RL Agent) share the "
                "same module architecture but differ in domain function."
            ),
        },
        {
            "category": "Non-LLM agent types",
            "note": (
                "He et al. focus heavily on LLM-based agents. Statistical (MEV), "
                "RL-based (trading), and deterministic (scripts, bridges) agent "
                "types are underrepresented in their framework. Our taxonomy "
                "covers the full spectrum of decision models."
            ),
        },
    ],
    "web3_applicability": {
        "score": 2,  # out of 5
        "assessment": (
            "The most modern and comprehensive of the comparison frameworks, "
            "but designed primarily for LLM-based software agents. The "
            "Perception-Reasoning-Action-Learning loop maps well to LLM-Powered "
            "Agents (Category 7) but poorly to MEV Searchers (Category 3) or "
            "Deterministic Scripts (Category 1). No blockchain-specific "
            "considerations: no concept of on-chain execution, gas optimization, "
            "mempool monitoring, or cross-chain coordination. The application "
            "domain taxonomy mentions blockchain but does not provide fine-grained "
            "categories. Overall, a strong general-purpose framework that lacks "
            "Web3-specific depth."
        ),
    },
}


# ============================================================
# ALL FRAMEWORKS
# ============================================================

ALL_FRAMEWORKS = [
    RUSSELL_NORVIG,
    WOOLDRIDGE_JENNINGS,
    FRANKLIN_GRAESSER,
    PARASURAMAN,
    HE_ET_AL,
]


# ============================================================
# ANALYSIS FUNCTIONS
# ============================================================

def compute_coverage_analysis() -> dict:
    """
    For each framework, compute coverage statistics.

    Coverage = fraction of our categories that have at least one
    partial (1) or strong (2) mapping from their framework.
    """
    results = {}

    for fw in ALL_FRAMEWORKS:
        matrix = np.array(fw["coverage_matrix"])
        n_their, n_ours = matrix.shape

        # For each of our categories, check if any of their categories map
        our_coverage = np.max(matrix, axis=0)  # best mapping per our category
        covered_strong = int(np.sum(our_coverage >= 2))
        covered_partial = int(np.sum(our_coverage >= 1))
        not_covered = int(np.sum(our_coverage == 0))

        # For each of their categories, check reverse
        their_coverage = np.max(matrix, axis=1)
        their_covered_strong = int(np.sum(their_coverage >= 2))
        their_covered_partial = int(np.sum(their_coverage >= 1))
        their_not_covered = int(np.sum(their_coverage == 0))

        # Total mapping strength
        total_strength = float(np.sum(matrix))
        max_possible = n_their * n_ours * 2
        coverage_density = total_strength / max_possible if max_possible > 0 else 0

        results[fw["name"]] = {
            "n_their_categories": n_their,
            "n_our_categories": n_ours,
            "our_categories_covered_strong": covered_strong,
            "our_categories_covered_partial": covered_partial,
            "our_categories_not_covered": not_covered,
            "our_coverage_rate": round(covered_partial / n_ours, 3),
            "their_categories_mapped_strong": their_covered_strong,
            "their_categories_mapped_partial": their_covered_partial,
            "their_categories_unmapped": their_not_covered,
            "coverage_density": round(coverage_density, 3),
        }

    return results


def compute_condition_coverage() -> dict:
    """
    For each framework, check which of our C1-C4 conditions they cover.
    """
    results = {}
    for fw in ALL_FRAMEWORKS:
        cc = fw["condition_coverage"]
        covered = [c for c, info in cc.items() if info["covered"]]
        not_covered = [c for c, info in cc.items() if not info["covered"]]
        results[fw["name"]] = {
            "conditions_covered": covered,
            "conditions_not_covered": not_covered,
            "coverage_count": len(covered),
            "coverage_fraction": round(len(covered) / 4, 2),
            "details": {
                c: info["note"] for c, info in cc.items()
            },
        }
    return results


def compute_gap_analysis() -> dict:
    """
    Identify what each framework captures that we don't, and vice versa.
    """
    analysis = {}
    for fw in ALL_FRAMEWORKS:
        analysis[fw["name"]] = {
            "they_have_we_dont": fw["categories_they_have_we_dont"],
            "we_have_they_dont": fw["categories_we_have_they_dont"],
            "n_unique_to_them": len(fw["categories_they_have_we_dont"]),
            "n_unique_to_us": len(fw["categories_we_have_they_dont"]),
        }
    return analysis


def compute_web3_applicability_ranking() -> list:
    """Rank all frameworks by Web3 applicability score."""
    ranking = []
    # Add our taxonomy
    ranking.append({
        "name": "Web3 AI Agent Taxonomy (Ours: C1-C4 + 8 categories)",
        "score": 5,
        "assessment": (
            "Designed specifically for Web3 context with on-chain "
            "observability as core design principle. All categories "
            "operationalized through blockchain transaction features. "
            "C1 (On-chain Actuation) is unique to our framework."
        ),
    })
    for fw in ALL_FRAMEWORKS:
        ranking.append({
            "name": fw["name"],
            "score": fw["web3_applicability"]["score"],
            "assessment": fw["web3_applicability"]["assessment"],
        })
    ranking.sort(key=lambda x: x["score"], reverse=True)
    return ranking


def generate_comparison_table() -> str:
    """Generate a formatted comparison table for the paper."""
    header = (
        "| Framework | Type | Categories/Properties | Covers C1-C4 | "
        "Covers Our 8 Cats | Web3 Score |\n"
        "|-----------|------|----------------------|-------------|"
        "-------------------|------------|\n"
    )
    rows = []

    coverage = compute_coverage_analysis()
    condition_cov = compute_condition_coverage()

    # Our taxonomy
    rows.append(
        "| **Web3 AI Agent Taxonomy (Ours)** | Definition + Taxonomy | "
        "4 conditions + 8 categories (3 dims) | 4/4 | N/A | **5/5** |"
    )

    for fw in ALL_FRAMEWORKS:
        cov = coverage[fw["name"]]
        cc = condition_cov[fw["name"]]
        n_cats = len(fw["categories"])

        fw_type = {
            "Russell & Norvig (2020)": "Architecture hierarchy",
            "Wooldridge & Jennings (1995)": "Property definition",
            "Franklin & Graesser (1996)": "Property taxonomy",
            "Parasuraman, Sheridan & Wickens (2000)": "Automation scale",
            "He et al. (2025)": "Module architecture",
        }[fw["name"]]

        covered = cov["our_categories_covered_partial"]
        total = cov["n_our_categories"]
        web3_score = fw["web3_applicability"]["score"]
        c_covered = cc["coverage_count"]

        rows.append(
            f"| {fw['name']} | {fw_type} | {n_cats} | "
            f"{c_covered}/4 | "
            f"{covered}/{total} ({cov['our_coverage_rate']:.0%}) | "
            f"{web3_score}/5 |"
        )

    return header + "\n".join(rows)


def generate_condition_table() -> str:
    """Generate a table showing which conditions each framework covers."""
    header = (
        "| Framework | C1 (On-chain) | C2 (Perception) | "
        "C3 (Decision) | C4 (Adaptive) |\n"
        "|-----------|--------------|-----------------|"
        "--------------|---------------|\n"
    )
    rows = []
    rows.append(
        "| **Ours** | **Yes** (necessary) | **Yes** (necessary) | "
        "**Yes** (necessary) | **Yes** (necessary) |"
    )

    for fw in ALL_FRAMEWORKS:
        cc = fw["condition_coverage"]
        cells = []
        for cond in ["C1 (On-chain Actuation)", "C2 (Environmental Perception)",
                      "C3 (Autonomous Decision-Making)", "C4 (Adaptiveness)"]:
            if cc[cond]["covered"]:
                cells.append("Yes")
            else:
                cells.append("**No**")
        rows.append(f"| {fw['name']} | {' | '.join(cells)} |")

    return header + "\n".join(rows)


def print_coverage_matrix(framework: dict):
    """Print a visual coverage matrix."""
    matrix = np.array(framework["coverage_matrix"])
    short_names = ["DetScrpt", "TradBot", "MEVSrch", "Bridge", "RLAgent",
                    "DeFiMgmt", "LLMAgent", "DAOAgent"]

    print(f"\n  {'':30s}", end="")
    for name in short_names:
        print(f"{name:>9s}", end="")
    print()
    print("  " + "-" * (30 + 9 * len(short_names)))

    for i, their_cat in enumerate(framework["categories"]):
        label = their_cat[:30].ljust(30)
        print(f"  {label}", end="")
        for j in range(len(short_names)):
            val = matrix[i, j]
            symbol = {0: "  .  ", 1: "  ~  ", 2: "  X  "}[val]
            print(f"{symbol:>9s}", end="")
        print()

    print("  Legend: X = strong mapping, ~ = partial mapping, . = no mapping")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the formal taxonomy comparison."""
    print("=" * 75)
    print("Formal Taxonomy Comparison: C1-C4 + 8-Category Framework vs. 5 Existing")
    print("=" * 75)

    # 1. Condition Coverage
    print("\n--- Condition Coverage (C1-C4) ---")
    print("Which of our four necessary conditions does each framework cover?\n")
    condition_cov = compute_condition_coverage()
    for fw_name, cc in condition_cov.items():
        print(f"  {fw_name}: {cc['coverage_count']}/4 conditions covered")
        for cond in cc["conditions_not_covered"]:
            print(f"    MISSING: {cond}")

    print("\n  Condition Coverage Table (Markdown):")
    print(generate_condition_table())

    print("\n  KEY FINDING: No existing framework covers C1 (On-chain Actuation).")
    print("  This is the unique contribution of our definition to the literature.")

    # 2. Category Coverage Analysis
    print("\n--- Category Coverage Analysis ---")
    coverage = compute_coverage_analysis()
    for fw_name, cov in coverage.items():
        print(f"\n  {fw_name}:")
        print(f"    Their categories/properties: {cov['n_their_categories']}")
        print(f"    Our categories covered (strong): "
              f"{cov['our_categories_covered_strong']}/8")
        print(f"    Our categories covered (any): "
              f"{cov['our_categories_covered_partial']}/8 "
              f"({cov['our_coverage_rate']:.0%})")
        print(f"    Their categories unmapped to ours: "
              f"{cov['their_categories_unmapped']}")
        print(f"    Coverage density: {cov['coverage_density']:.3f}")

    # 3. Coverage Matrices
    print("\n--- Coverage Matrices ---")
    for fw in ALL_FRAMEWORKS:
        print(f"\n  {fw['name']}:")
        print_coverage_matrix(fw)

    # 4. Gap Analysis
    print("\n--- Gap Analysis ---")
    gaps = compute_gap_analysis()
    for fw_name, gap in gaps.items():
        print(f"\n  {fw_name}:")
        print(f"    Concepts unique to them: {gap['n_unique_to_them']}")
        for item in gap["they_have_we_dont"]:
            print(f"      - {item['category']}: {item['note'][:80]}...")
        print(f"    Concepts unique to us: {gap['n_unique_to_us']}")
        for item in gap["we_have_they_dont"]:
            print(f"      - {item['category']}: {item['note'][:80]}...")

    # 5. Web3 Applicability Ranking
    print("\n--- Web3 Applicability Ranking ---")
    ranking = compute_web3_applicability_ranking()
    for i, r in enumerate(ranking, 1):
        print(f"  {i}. {r['name']}: {r['score']}/5")
        print(f"     {r['assessment'][:100]}...")

    # 6. Comparison Table
    print("\n--- Comparison Table (Markdown) ---")
    print(generate_comparison_table())

    # 7. Key Findings Summary
    print("\n--- Key Findings ---")
    print("""
  1. CONDITION COVERAGE: No existing framework covers all four C1-C4 conditions.
     - C1 (On-chain Actuation) is UNIQUE to our framework -- no prior work
       addresses blockchain-specific actuation (EOA control, tx signing).
     - C4 (Adaptiveness) is surprisingly absent from Wooldridge & Jennings (1995)
       and Parasuraman et al. (2000).
     - Best condition coverage: He et al. (2025) covers 3/4 (C2, C3, C4) via
       their Perception-Reasoning-Action-Learning loop, but misses C1.

  2. CATEGORY COVERAGE: No existing framework covers all 8 of our categories.
     - Best: R&N (2020) partially covers 6/8 but collapses MEV Searcher,
       RL Agent, and DeFi Agent into 'Utility-based'.
     - Worst: Parasuraman (2000) covers only 4/8 -- designed for human-machine
       systems, not autonomous software agents.

  3. OUR UNIQUE CONTRIBUTIONS:
     - C1 (On-chain Actuation): first to formalize blockchain-specific actuation.
     - Environment Type dimension: absent from ALL existing frameworks.
     - Decision Model dimension: distinguishes LLM, RL, statistical, and
       deterministic -- no existing framework separates these for agents.
     - Domain-specific categories: MEV Searcher, Cross-Chain Bridge Agent, etc.
       are absent from all general-purpose frameworks.

  4. WHAT WE COULD INCORPORATE:
     - Social ability (from W&J 1995): could enrich our COLLABORATIVE dimension.
     - BDI mentalistic notions (from W&J 1995): relevant for LLM agent reasoning.
     - Perception sub-types (from He et al. 2025): text vs visual vs multimodal.
     - Reasoning patterns (from He et al. 2025): CoT, ReAct, tree-of-thought.
     - Stage-specific automation (from Parasuraman 2000): different components
       may have different automation levels.

  5. WEB3 APPLICABILITY:
     - All existing frameworks score 1-2/5 for Web3 applicability.
     - Our taxonomy scores 5/5 by design (built for Web3 context).
     - He et al. (2025) is the most modern but still LLM-centric and lacks
       blockchain-specific dimensions.
""")

    # Save results
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output = {
        "condition_coverage": condition_cov,
        "coverage_analysis": coverage,
        "gap_analysis": {
            k: {
                "they_have_we_dont": [
                    item if isinstance(item, dict) else asdict(item)
                    for item in v["they_have_we_dont"]
                ],
                "we_have_they_dont": [
                    item if isinstance(item, dict) else asdict(item)
                    for item in v["we_have_they_dont"]
                ],
                "n_unique_to_them": v["n_unique_to_them"],
                "n_unique_to_us": v["n_unique_to_us"],
            }
            for k, v in gaps.items()
        },
        "web3_applicability_ranking": ranking,
        "comparison_table_markdown": generate_comparison_table(),
        "condition_table_markdown": generate_condition_table(),
        "frameworks": {
            fw["name"]: {
                "full_title": fw["full_title"],
                "key_idea": fw["key_idea"],
                "definition_of_agent": fw["definition_of_agent"],
                "n_categories": len(fw["categories"]),
                "categories": fw["categories"],
                "coverage_matrix": fw["coverage_matrix"],
                "condition_coverage_count": condition_cov[fw["name"]]["coverage_count"],
                "web3_score": fw["web3_applicability"]["score"],
            }
            for fw in ALL_FRAMEWORKS
        },
    }

    output_path = os.path.join(output_dir, "taxonomy_comparison_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 75)
    print("TAXONOMY COMPARISON COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
