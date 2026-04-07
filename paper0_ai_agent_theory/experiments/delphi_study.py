"""
Delphi Expert Validation for AI Agent Taxonomy
================================================
A 2-round Delphi study to validate the proposed 8-category taxonomy.

Round 1: Open-ended exploration
- Present each taxonomy dimension and category
- Ask experts to rate relevance (1-5 Likert)
- Ask for missing categories or dimensions
- Collect free-text feedback

Round 2: Convergence
- Present Round 1 results with median ratings
- Ask experts to re-rate considering group feedback
- Measure consensus (IQR ≤ 1 = consensus reached)
"""

import json
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Import taxonomy from pilot
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from paper0_ai_agent_theory.experiments.pilot_taxonomy import (
    TAXONOMY,
    AutonomyLevel,
    EnvironmentType,
    DecisionModel,
)


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class LikertQuestion:
    """A single Likert-scale question (1-5)."""
    id: str
    text: str
    scale_min: int = 1
    scale_max: int = 5
    anchor_low: str = "Strongly Disagree"
    anchor_high: str = "Strongly Agree"


@dataclass
class OpenEndedQuestion:
    """A free-text response question."""
    id: str
    text: str


@dataclass
class MultipleChoiceQuestion:
    """A multiple-choice question with optional open-ended follow-up."""
    id: str
    text: str
    options: list[str] = field(default_factory=list)
    allow_other: bool = True


@dataclass
class ClassificationItem:
    """An entity to be classified by experts."""
    id: str
    name: str
    description: str
    expected_category: str
    confidence_question_id: str


@dataclass
class RankingQuestion:
    """A ranking question where experts order items."""
    id: str
    text: str
    items_to_rank: list[str] = field(default_factory=list)


# ============================================================
# PART A: DIMENSION VALIDATION
# ============================================================

DIMENSIONS = [
    {
        "name": "Autonomy Level (自主性等级)",
        "definition": (
            "Degree of autonomous decision-making, ranging from NONE "
            "(pure deterministic script) through REACTIVE (fixed-rule "
            "responses), ADAPTIVE (parameter adjustment based on feedback), "
            "PROACTIVE (independent planning and goal pursuit), to "
            "COLLABORATIVE (multi-agent coordination)."
        ),
        "levels": [
            "NONE (Level 0): Pure script, deterministic, no adaptation",
            "REACTIVE (Level 1): Responds to events with fixed rules",
            "ADAPTIVE (Level 2): Adjusts parameters based on environment",
            "PROACTIVE (Level 3): Plans and initiates actions independently",
            "COLLABORATIVE (Level 4): Coordinates with other agents/humans",
        ],
        "literature_basis": (
            "Derived from Russell & Norvig (2020) agent architecture "
            "hierarchy and Wooldridge & Jennings (1995) four-property "
            "framework. Simplified from Parasuraman et al. (2000) 10-level "
            "automation to 5 levels due to limited observability through "
            "blockchain transaction data."
        ),
    },
    {
        "name": "Environment Type (环境类型)",
        "definition": (
            "Where the agent operates and what data sources it accesses. "
            "On-chain Only (fully on-chain smart contract logic), "
            "Hybrid/Off-chain-to-On-chain (off-chain compute, on-chain "
            "execution), Cross-chain (operates across multiple blockchains), "
            "Multi-modal (combines on-chain actions with off-chain data "
            "sources such as APIs, LLMs, social media)."
        ),
        "levels": [
            "ON-CHAIN ONLY: Fully on-chain smart contract execution",
            "HYBRID: Off-chain compute, on-chain execution",
            "CROSS-CHAIN: Operates across multiple blockchain networks",
            "MULTI-MODAL: Combines on-chain + off-chain services (APIs, LLMs)",
        ],
        "literature_basis": (
            "Novel dimension specific to Web3 context. Traditional agent "
            "taxonomies do not distinguish environment types at this "
            "granularity. Motivated by the unique transparency and "
            "constraints of blockchain execution environments."
        ),
    },
    {
        "name": "Decision Model (决策模型)",
        "definition": (
            "The mechanism by which the agent makes decisions. "
            "Deterministic (if-then rules), Statistical (ML/probability-"
            "based), LLM-driven (large language model reasoning), "
            "Reinforcement Learning (policy optimization via reward signals), "
            "Hybrid (combination of multiple models)."
        ),
        "levels": [
            "DETERMINISTIC: If-then rules, fully predictable output",
            "STATISTICAL: ML models, probability-based decisions",
            "LLM-DRIVEN: Large language model reasoning",
            "REINFORCEMENT LEARNING: RL-based policy optimization",
            "HYBRID: Combination of multiple decision models",
        ],
        "literature_basis": (
            "Extends Russell & Norvig (2020) agent architectures with "
            "modern AI decision paradigms (LLM, RL). The distinction "
            "between Statistical and LLM-driven is novel and reflects "
            "the emergence of foundation-model-based agents."
        ),
    },
]


def build_dimension_questions() -> dict:
    """Build Part A questions for each dimension."""
    questions = {"part_a_dimension_validation": []}

    for i, dim in enumerate(DIMENSIONS, 1):
        dim_block = {
            "dimension": dim["name"],
            "definition": dim["definition"],
            "levels": dim["levels"],
            "literature_basis": dim["literature_basis"],
            "questions": [
                asdict(LikertQuestion(
                    id=f"A{i}_necessity",
                    text=(
                        f"Is the dimension '{dim['name']}' necessary for "
                        "distinguishing AI agents from other automated systems "
                        "in the Web3 context?"
                    ),
                    anchor_low="Not at all necessary",
                    anchor_high="Absolutely necessary",
                )),
                asdict(OpenEndedQuestion(
                    id=f"A{i}_sufficiency",
                    text=(
                        f"Is the dimension '{dim['name']}' sufficient as "
                        "defined, or are additional sub-dimensions or "
                        "modifications needed? Please elaborate."
                    ),
                )),
                asdict(LikertQuestion(
                    id=f"A{i}_clarity",
                    text=(
                        f"Rate the clarity of the definitions for "
                        f"'{dim['name']}' and its levels."
                    ),
                    anchor_low="Very unclear",
                    anchor_high="Very clear",
                )),
                asdict(LikertQuestion(
                    id=f"A{i}_completeness",
                    text=(
                        f"Rate the completeness of the levels within "
                        f"'{dim['name']}'. Are all meaningful levels covered?"
                    ),
                    anchor_low="Major gaps exist",
                    anchor_high="Fully complete",
                )),
                asdict(OpenEndedQuestion(
                    id=f"A{i}_missing",
                    text=(
                        "Are there additional dimensions not captured by "
                        "our three-dimension framework that are essential "
                        "for classifying Web3 AI agents? If so, please "
                        "describe them."
                    ),
                )),
            ],
        }
        questions["part_a_dimension_validation"].append(dim_block)

    return questions


# ============================================================
# PART B: CATEGORY VALIDATION
# ============================================================

CATEGORIES = [
    {
        "name": "Deterministic Script",
        "tuple": "(NONE, Hybrid, Deterministic)",
        "definition": (
            "Fully deterministic program that executes a fixed sequence "
            "of transactions without any autonomous decision-making."
        ),
        "examples": [
            "Cron-scheduled token transfers",
            "Hardcoded airdrop distribution scripts",
            "Static payroll contracts",
        ],
        "on_chain_indicators": [
            "Perfectly periodic transactions",
            "Identical calldata across invocations",
            "Fixed gas price or legacy gas pricing",
            "Single-function interaction pattern",
        ],
    },
    {
        "name": "Simple Trading Bot",
        "tuple": "(REACTIVE, Hybrid, Deterministic)",
        "definition": (
            "Executes pre-defined trading rules; responds to market "
            "conditions with fixed execution paths."
        ),
        "examples": [
            "Grid trading bots",
            "DCA (Dollar Cost Averaging) bots",
            "Rebalancing scripts",
        ],
        "on_chain_indicators": [
            "Regular interval transactions",
            "Fixed trade sizes",
            "Deterministic gas pricing",
        ],
    },
    {
        "name": "MEV Searcher",
        "tuple": "(ADAPTIVE, Hybrid, Statistical)",
        "definition": (
            "Identifies and extracts Maximal Extractable Value through "
            "strategies like sandwich attacks, arbitrage, and liquidations."
        ),
        "examples": [
            "jaredfromsubway.eth",
            "Flashbots searchers",
        ],
        "on_chain_indicators": [
            "Extremely low latency (sub-block response)",
            "Bundle submissions via Flashbots",
            "High gas priority fees",
            "Interaction with DEX routers",
        ],
    },
    {
        "name": "Cross-Chain Bridge Agent",
        "tuple": "(ADAPTIVE, Cross-chain, Deterministic)",
        "definition": (
            "Relays messages and assets across blockchain networks, "
            "maintaining consistency between different consensus environments."
        ),
        "examples": [
            "LayerZero relayers",
            "Wormhole guardians",
            "Axelar validators",
        ],
        "on_chain_indicators": [
            "Bridge contract interactions",
            "Matching transactions across chains",
            "Proof submission patterns",
        ],
    },
    {
        "name": "RL Trading Agent",
        "tuple": "(ADAPTIVE, Hybrid, Reinforcement Learning)",
        "definition": (
            "Uses reinforcement learning policies to optimize trading "
            "strategy; learns from reward signals derived from on-chain "
            "outcomes."
        ),
        "examples": [
            "RL-based market-making agents",
            "Policy-gradient DEX arbitrage agents",
            "Multi-armed bandit liquidity allocators",
        ],
        "on_chain_indicators": [
            "Exploration-exploitation trade size patterns",
            "Gradual strategy convergence over time",
            "Reward-correlated behavioral shifts",
            "Non-stationary gas bidding strategies",
        ],
    },
    {
        "name": "DeFi Management Agent",
        "tuple": "(PROACTIVE, Hybrid, Hybrid)",
        "definition": (
            "Manages DeFi positions with cross-protocol reasoning and "
            "risk-aware capability: yield farming, lending, liquidity "
            "provision."
        ),
        "examples": [
            "Autonolas agents",
            "Yearn strategy vaults",
            "DeFi Saver automation",
        ],
        "on_chain_indicators": [
            "Multi-protocol interactions in sequences",
            "Approval management patterns",
            "Position rebalancing transactions",
            "Oracle price consultation",
        ],
    },
    {
        "name": "LLM-Powered Agent",
        "tuple": "(PROACTIVE, Multi-modal, LLM-driven)",
        "definition": (
            "Uses large language model reasoning to decide on-chain "
            "actions; behavior influenced by multi-modal context."
        ),
        "examples": [
            "AI16Z/ELIZA agents",
            "Virtuals Protocol agents",
            "MCP-connected agents",
        ],
        "on_chain_indicators": [
            "Variable latency (LLM inference time)",
            "Complex multi-step transaction sequences",
            "Natural language-influenced parameters",
            "Non-deterministic gas pricing",
        ],
    },
    {
        "name": "Autonomous DAO Agent",
        "tuple": "(COLLABORATIVE, On-chain, Hybrid)",
        "definition": (
            "Fully on-chain agent governed by DAO rules; acts through "
            "collective decision-making processes."
        ),
        "examples": [
            "Gnosis Safe modules",
            "Governor-controlled executors",
        ],
        "on_chain_indicators": [
            "Transactions originate from multisig/timelock",
            "Proposal-execution pattern",
            "Governance token interactions",
        ],
    },
]


def build_category_questions() -> dict:
    """Build Part B questions for each category."""
    questions = {"part_b_category_validation": []}

    for i, cat in enumerate(CATEGORIES, 1):
        cat_block = {
            "category": cat["name"],
            "taxonomy_tuple": cat["tuple"],
            "definition": cat["definition"],
            "examples": cat["examples"],
            "on_chain_indicators": cat["on_chain_indicators"],
            "questions": [
                asdict(LikertQuestion(
                    id=f"B{i}_well_defined",
                    text=(
                        f"Rate how well-defined the category "
                        f"'{cat['name']}' is."
                    ),
                    anchor_low="Very poorly defined",
                    anchor_high="Very well-defined",
                )),
                asdict(LikertQuestion(
                    id=f"B{i}_distinguishable",
                    text=(
                        f"Rate how distinguishable '{cat['name']}' is from "
                        f"the other 7 categories."
                    ),
                    anchor_low="Highly overlapping",
                    anchor_high="Clearly distinct",
                )),
                asdict(LikertQuestion(
                    id=f"B{i}_examples_representative",
                    text=(
                        f"Rate how representative the provided examples are "
                        f"for '{cat['name']}'."
                    ),
                    anchor_low="Not representative",
                    anchor_high="Highly representative",
                )),
                asdict(OpenEndedQuestion(
                    id=f"B{i}_missing_examples",
                    text=(
                        "Can you provide real-world examples that belong to "
                        f"'{cat['name']}' that we missed?"
                    ),
                )),
                asdict(MultipleChoiceQuestion(
                    id=f"B{i}_structural",
                    text=(
                        f"Should the category '{cat['name']}' be modified?"
                    ),
                    options=[
                        "Keep as-is",
                        "Split into multiple categories",
                        "Merge with another category",
                        "Remove entirely",
                        "Redefine boundaries",
                    ],
                    allow_other=True,
                )),
                asdict(OpenEndedQuestion(
                    id=f"B{i}_structural_explanation",
                    text=(
                        "If you selected anything other than 'Keep as-is', "
                        "please explain your reasoning."
                    ),
                )),
            ],
        }
        questions["part_b_category_validation"].append(cat_block)

    return questions


# ============================================================
# PART C: CLASSIFICATION EXERCISE
# ============================================================

CLASSIFICATION_ENTITIES = [
    ClassificationItem(
        id="C1",
        name="jaredfromsubway.eth",
        description=(
            "One of Ethereum's most prolific MEV extractors. Conducts "
            "sandwich attacks on DEX trades by monitoring the mempool, "
            "front-running victim transactions, and back-running to "
            "capture price impact profit. Operates via Flashbots bundles."
        ),
        expected_category="MEV Searcher",
        confidence_question_id="C1_confidence",
    ),
    ClassificationItem(
        id="C2",
        name="Wintermute",
        description=(
            "A major crypto market maker providing liquidity across "
            "centralized and decentralized exchanges. Operates algorithmic "
            "trading strategies with high frequency and tight spreads. "
            "Interacts primarily with DEX router contracts."
        ),
        expected_category="Simple Trading Bot / RL Trading Agent (boundary)",
        confidence_question_id="C2_confidence",
    ),
    ClassificationItem(
        id="C3",
        name="Autonolas OLAS agent",
        description=(
            "An autonomous agent registered in the Autonolas ServiceRegistry. "
            "Manages DeFi positions across multiple protocols (Aave, "
            "Uniswap, Yearn), rebalances based on market conditions, and "
            "uses a hybrid decision model combining statistical risk "
            "assessment with rule-based execution."
        ),
        expected_category="DeFi Management Agent",
        confidence_question_id="C3_confidence",
    ),
    ClassificationItem(
        id="C4",
        name="Uniswap V2 Router",
        description=(
            "A smart contract that routes token swaps through Uniswap V2 "
            "liquidity pools. Executes deterministic swap logic based on "
            "the constant product formula. Has no off-chain component or "
            "autonomous decision-making."
        ),
        expected_category="Deterministic Script (or not an agent at all)",
        confidence_question_id="C4_confidence",
    ),
    ClassificationItem(
        id="C5",
        name="A personal wallet doing manual trades",
        description=(
            "An externally owned account (EOA) controlled by a human "
            "user who manually initiates DEX trades via a frontend (e.g., "
            "Uniswap web app). Transactions show circadian rhythm, "
            "irregular intervals, and round-number gas prices."
        ),
        expected_category="Not an agent (Human)",
        confidence_question_id="C5_confidence",
    ),
    ClassificationItem(
        id="C6",
        name="An Aave liquidation bot",
        description=(
            "A bot that monitors Aave lending positions and triggers "
            "liquidation transactions when health factors drop below "
            "thresholds. Uses statistical models to estimate gas costs "
            "and profitability. Adapts gas bidding strategy based on "
            "competition."
        ),
        expected_category="MEV Searcher",
        confidence_question_id="C6_confidence",
    ),
    ClassificationItem(
        id="C7",
        name="LayerZero relayer",
        description=(
            "A relayer node that monitors source-chain events and submits "
            "proof transactions on destination chains. Operates across "
            "Ethereum, Arbitrum, Optimism, and Polygon. Deterministic "
            "relay logic with adaptive gas pricing."
        ),
        expected_category="Cross-Chain Bridge Agent",
        confidence_question_id="C7_confidence",
    ),
    ClassificationItem(
        id="C8",
        name="AI16Z ELIZA trading agent",
        description=(
            "An agent built on the ELIZA framework that uses GPT-4 to "
            "analyze social media sentiment, news, and on-chain data "
            "to make trading decisions. Executes token swaps on Uniswap "
            "based on LLM reasoning. Posts updates on Twitter."
        ),
        expected_category="LLM-Powered Agent",
        confidence_question_id="C8_confidence",
    ),
    ClassificationItem(
        id="C9",
        name="A cron job that distributes payroll",
        description=(
            "A cron-scheduled script that runs every two weeks and sends "
            "fixed amounts of USDC to a predefined list of employee "
            "addresses. No conditional logic, no adaptation, identical "
            "calldata every execution."
        ),
        expected_category="Deterministic Script",
        confidence_question_id="C9_confidence",
    ),
    ClassificationItem(
        id="C10",
        name="A Chainlink oracle node",
        description=(
            "A node operator that submits price feed updates to Chainlink "
            "aggregator contracts. Uses deterministic submission logic "
            "triggered by price deviation thresholds. Operates within the "
            "Chainlink consensus protocol alongside other oracle nodes."
        ),
        expected_category="Cross-Chain Bridge Agent / Deterministic Script (boundary)",
        confidence_question_id="C10_confidence",
    ),
]

CATEGORY_OPTIONS = [
    "Deterministic Script",
    "Simple Trading Bot",
    "MEV Searcher",
    "Cross-Chain Bridge Agent",
    "RL Trading Agent",
    "DeFi Management Agent",
    "LLM-Powered Agent",
    "Autonomous DAO Agent",
    "Not an agent (Human / Smart Contract / Other)",
]


def build_classification_exercise() -> dict:
    """Build Part C classification exercise."""
    items = []
    for entity in CLASSIFICATION_ENTITIES:
        items.append({
            "entity": {
                "id": entity.id,
                "name": entity.name,
                "description": entity.description,
            },
            "questions": [
                asdict(MultipleChoiceQuestion(
                    id=f"{entity.id}_classify",
                    text=f"Which category does '{entity.name}' belong to?",
                    options=CATEGORY_OPTIONS,
                    allow_other=True,
                )),
                asdict(LikertQuestion(
                    id=entity.confidence_question_id,
                    text=(
                        "How confident are you in your classification of "
                        f"'{entity.name}'?"
                    ),
                    anchor_low="Not at all confident",
                    anchor_high="Completely confident",
                )),
                asdict(OpenEndedQuestion(
                    id=f"{entity.id}_rationale",
                    text=(
                        "Briefly explain your classification rationale for "
                        f"'{entity.name}'."
                    ),
                )),
            ],
            "_expected_category": entity.expected_category,
        })

    return {"part_c_classification_exercise": items}


# ============================================================
# PART D: TAXONOMY COMPARISON
# ============================================================

COMPARISON_FRAMEWORKS = [
    {
        "name": "Russell & Norvig (2020)",
        "source": "Artificial Intelligence: A Modern Approach, 4th ed.",
        "categories": [
            "Simple reflex agents",
            "Model-based reflex agents",
            "Goal-based agents",
            "Utility-based agents",
            "Learning agents",
        ],
        "strengths": [
            "Well-established theoretical foundation",
            "Clear hierarchy of increasing capability",
            "Widely taught and recognized",
        ],
        "limitations_for_web3": [
            "No distinction for execution environment (on-chain vs off-chain)",
            "Does not account for blockchain-specific constraints",
            "Cannot distinguish MEV searchers from DeFi management agents",
        ],
    },
    {
        "name": "Franklin & Graesser (1996)",
        "source": "Is it an Agent, or Just a Program?",
        "categories": [
            "Property-based classification with 13 properties:",
            "Reactive, Autonomous, Goal-oriented, Temporally continuous,",
            "Communicative, Learning, Mobile, Flexible, Character,",
            "Emotional, Modelling, Personality, Adaptive",
        ],
        "strengths": [
            "Fine-grained property-based decomposition",
            "Allows partial agent classification",
            "Addresses the 'what is an agent' question directly",
        ],
        "limitations_for_web3": [
            "Too many dimensions for practical classification",
            "Properties like 'Emotional' and 'Personality' irrelevant to Web3",
            "No operationalization for blockchain observability",
        ],
    },
    {
        "name": "Parasuraman, Sheridan & Wickens (2000)",
        "source": "A Model for Types and Levels of Human Interaction with Automation",
        "categories": [
            "10 levels of automation from fully manual to fully automatic,",
            "applied across 4 information processing stages:",
            "Information Acquisition, Information Analysis,",
            "Decision Selection, Action Implementation",
        ],
        "strengths": [
            "Granular 10-level automation scale",
            "Stage-specific automation levels",
            "Well-validated in HCI research",
        ],
        "limitations_for_web3": [
            "Designed for human-machine systems, not autonomous software agents",
            "10 levels too granular for blockchain observability",
            "Does not account for multi-agent coordination",
            "No concept of execution environment",
        ],
    },
    {
        "name": "Castelfranchi (1995)",
        "source": "Guarantees for Autonomy in Cognitive Agent Architecture (Social Commitment Agents)",
        "categories": [
            "Autonomous agents with social commitments",
            "Obligation-based reasoning",
            "Delegation and trust models",
            "Normative agent architectures",
        ],
        "strengths": [
            "Addresses social aspects of agent behavior",
            "Relevant to DAO governance and collective decision-making",
            "Models trust and delegation",
        ],
        "limitations_for_web3": [
            "Focused on social/normative aspects, not operational classification",
            "No observable on-chain indicators defined",
            "Limited applicability to purely technical agent types (MEV, trading)",
        ],
    },
    {
        "name": "Jennings, Sycara & Wooldridge (1998)",
        "source": "A Roadmap of Agent Research and Development",
        "categories": [
            "Deliberative agents",
            "Reactive agents",
            "Hybrid agents",
            "Multi-agent systems (MAS)",
            "Collaborative agents",
        ],
        "strengths": [
            "Broad coverage of agent paradigms",
            "Includes multi-agent considerations",
            "Establishes BDI (Belief-Desire-Intention) framework",
        ],
        "limitations_for_web3": [
            "Too abstract for operational classification",
            "BDI model hard to observe on-chain",
            "Does not distinguish between blockchain execution environments",
            "No decision-model dimension",
        ],
    },
]


def build_comparison_questions() -> dict:
    """Build Part D taxonomy comparison questions."""
    framework_summaries = []
    for fw in COMPARISON_FRAMEWORKS:
        framework_summaries.append({
            "framework": fw["name"],
            "source": fw["source"],
            "categories": fw["categories"],
            "strengths": fw["strengths"],
            "limitations_for_web3": fw["limitations_for_web3"],
        })

    questions = {
        "part_d_taxonomy_comparison": {
            "instructions": (
                "Below we present our 8-category taxonomy side-by-side "
                "with 5 existing agent classification frameworks. Please "
                "review each and answer the following questions."
            ),
            "our_taxonomy": {
                "name": "Web3 AI Agent Taxonomy (This Paper)",
                "dimensions": ["Autonomy Level (5)", "Environment Type (4)", "Decision Model (5)"],
                "categories": [cat["name"] for cat in CATEGORIES],
                "design_principles": [
                    "Observability: every category distinguishable via on-chain behavior",
                    "Mutual exclusivity: unique (autonomy, environment, decision) triple per category",
                    "Exhaustiveness: covers all known Web3 agent types",
                ],
            },
            "comparison_frameworks": framework_summaries,
            "questions": [
                asdict(RankingQuestion(
                    id="D1_ranking",
                    text=(
                        "Rank the following taxonomies from most useful (1) "
                        "to least useful (6) for classifying AI agents in "
                        "the Web3/blockchain context."
                    ),
                    items_to_rank=[
                        "Web3 AI Agent Taxonomy (This Paper)",
                        "Russell & Norvig (2020)",
                        "Franklin & Graesser (1996)",
                        "Parasuraman et al. (2000)",
                        "Castelfranchi (1995)",
                        "Jennings et al. (1998)",
                    ],
                )),
                asdict(OpenEndedQuestion(
                    id="D2_unique_contribution",
                    text=(
                        "What does our taxonomy capture that the other "
                        "frameworks miss? Please be specific."
                    ),
                )),
                asdict(OpenEndedQuestion(
                    id="D3_missing_from_ours",
                    text=(
                        "What do the other frameworks capture that our "
                        "taxonomy misses? Should we incorporate any of "
                        "these elements?"
                    ),
                )),
                asdict(LikertQuestion(
                    id="D4_web3_applicability",
                    text=(
                        "Rate the overall applicability of our taxonomy "
                        "to the Web3 context compared to existing frameworks."
                    ),
                    anchor_low="Much less applicable",
                    anchor_high="Much more applicable",
                )),
                asdict(LikertQuestion(
                    id="D5_generalizability",
                    text=(
                        "Rate the potential generalizability of our taxonomy "
                        "beyond Web3 to other AI agent domains."
                    ),
                    anchor_low="Not generalizable",
                    anchor_high="Highly generalizable",
                )),
            ],
        }
    }

    return questions


# ============================================================
# ROUND 2: CONVERGENCE INSTRUMENT
# ============================================================

def build_round2_instrument(round1_results: dict) -> dict:
    """
    Build Round 2 instrument based on Round 1 results.

    In Round 2, experts see the group's median ratings and IQR for each
    question, along with summarized free-text feedback, and are asked to
    re-rate items where consensus was not reached (IQR > 1).

    Args:
        round1_results: Dictionary mapping question_id -> list of ratings.

    Returns:
        Round 2 instrument with group feedback.
    """
    round2 = {
        "round": 2,
        "instructions": (
            "In Round 1, you and your fellow experts independently rated "
            "the taxonomy dimensions, categories, and comparison frameworks. "
            "Below we present the group's median ratings and spread (IQR). "
            "For items where consensus was NOT reached (IQR > 1), please "
            "re-rate considering the group feedback. For items where "
            "consensus WAS reached (IQR <= 1), no re-rating is needed "
            "unless you have strong objections."
        ),
        "consensus_threshold": "IQR <= 1.0",
        "items_requiring_re_rating": [],
        "items_with_consensus": [],
    }

    for qid, ratings in round1_results.items():
        ratings_arr = np.array(ratings, dtype=float)
        median = float(np.median(ratings_arr))
        q25 = float(np.percentile(ratings_arr, 25))
        q75 = float(np.percentile(ratings_arr, 75))
        iqr = q75 - q25
        mean = float(np.mean(ratings_arr))
        std = float(np.std(ratings_arr))

        item = {
            "question_id": qid,
            "n_respondents": len(ratings),
            "median": round(median, 2),
            "mean": round(mean, 2),
            "std": round(std, 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "consensus_reached": iqr <= 1.0,
        }

        if iqr <= 1.0:
            round2["items_with_consensus"].append(item)
        else:
            item["re_rate_question"] = {
                "text": (
                    f"The group median for this item was {median:.1f} "
                    f"(IQR = {iqr:.1f}). Considering this, please re-rate."
                ),
                "your_round1_rating": "{{PERSONALIZED}}",
                "scale": "1-5",
            }
            round2["items_requiring_re_rating"].append(item)

    return round2


# ============================================================
# CONSENSUS METRICS
# ============================================================

def compute_krippendorff_alpha(
    classification_matrix: list[list[int]],
    n_categories: int = 9,
) -> float:
    """
    Compute Krippendorff's alpha for inter-rater agreement on the
    classification exercise (Part C).

    Args:
        classification_matrix: List of raters, each containing a list of
            category assignments (integer indices) for the 10 entities.
            Shape: (n_raters, n_entities).
        n_categories: Number of possible category choices.

    Returns:
        Krippendorff's alpha (nominal data).
    """
    matrix = np.array(classification_matrix)
    n_raters, n_entities = matrix.shape

    if n_raters < 2:
        return float("nan")

    # Build coincidence matrix
    coincidence = np.zeros((n_categories, n_categories))

    for entity_idx in range(n_entities):
        ratings = matrix[:, entity_idx]
        # Count pairs of raters who agree/disagree
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                c_i = ratings[i]
                c_j = ratings[j]
                coincidence[c_i, c_j] += 1
                coincidence[c_j, c_i] += 1

    # Observed disagreement
    n_pairs_total = n_entities * n_raters * (n_raters - 1) / 2
    observed_disagreement = 0.0
    for entity_idx in range(n_entities):
        ratings = matrix[:, entity_idx]
        for i in range(n_raters):
            for j in range(i + 1, n_raters):
                if ratings[i] != ratings[j]:
                    observed_disagreement += 1

    d_o = observed_disagreement / n_pairs_total if n_pairs_total > 0 else 0

    # Expected disagreement (based on marginal frequencies)
    marginals = np.sum(coincidence, axis=1)
    total_assignments = np.sum(marginals)
    expected_disagreement = 0.0
    for c in range(n_categories):
        for k in range(n_categories):
            if c != k:
                expected_disagreement += marginals[c] * marginals[k]
    d_e = expected_disagreement / (total_assignments * (total_assignments - 1)) if total_assignments > 1 else 0

    if d_e == 0:
        return 1.0  # Perfect agreement

    alpha = 1.0 - (d_o / d_e)
    return alpha


def compute_consensus_metrics(ratings_dict: dict) -> dict:
    """
    Compute consensus metrics for all Likert-scale questions.

    Args:
        ratings_dict: Mapping from question_id to list of ratings.

    Returns:
        Dictionary with per-question and overall consensus metrics.
    """
    metrics = {
        "per_question": {},
        "summary": {},
    }

    consensus_count = 0
    total_likert = 0

    for qid, ratings in ratings_dict.items():
        arr = np.array(ratings, dtype=float)
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        iqr = q75 - q25

        item_metrics = {
            "n": len(ratings),
            "median": round(float(np.median(arr)), 2),
            "mean": round(float(np.mean(arr)), 2),
            "std": round(float(np.std(arr)), 2),
            "q25": round(q25, 2),
            "q75": round(q75, 2),
            "iqr": round(iqr, 2),
            "consensus": iqr <= 1.0,
        }
        metrics["per_question"][qid] = item_metrics

        total_likert += 1
        if iqr <= 1.0:
            consensus_count += 1

    metrics["summary"] = {
        "total_likert_questions": total_likert,
        "consensus_reached": consensus_count,
        "consensus_rate": (
            round(consensus_count / total_likert, 3) if total_likert > 0 else 0
        ),
        "consensus_threshold": "IQR <= 1.0",
    }

    return metrics


# ============================================================
# MAIN: GENERATE INSTRUMENT & RUN SIMULATED VALIDATION
# ============================================================

def generate_full_instrument() -> dict:
    """Assemble the complete 2-round Delphi instrument."""
    instrument = {
        "study_title": (
            "Delphi Expert Validation for Web3 AI Agent Taxonomy"
        ),
        "version": "1.0",
        "n_rounds": 2,
        "target_experts": {
            "n_target": "12-20",
            "expertise_areas": [
                "AI/ML researchers with agent systems experience",
                "Blockchain security engineers",
                "DeFi protocol designers",
                "HCI researchers specializing in automation/AI",
                "MEV researchers",
            ],
            "recruitment_criteria": [
                "Minimum 3 years experience in relevant domain",
                "Published at least 1 peer-reviewed paper in related area",
                "Active involvement in Web3/blockchain ecosystem preferred",
            ],
        },
        "round_1": {
            "description": (
                "Open-ended exploration: experts independently rate "
                "dimensions and categories, classify real-world entities, "
                "and compare against existing frameworks."
            ),
            "estimated_completion_time": "45-60 minutes",
            **build_dimension_questions(),
            **build_category_questions(),
            **build_classification_exercise(),
            **build_comparison_questions(),
        },
        "round_2_template": {
            "description": (
                "Convergence round: experts see group medians and IQR "
                "from Round 1, re-rate items where consensus not reached "
                "(IQR > 1), confirm or adjust their classifications."
            ),
            "estimated_completion_time": "20-30 minutes",
            "note": (
                "Round 2 instrument is generated dynamically based on "
                "Round 1 results. See build_round2_instrument()."
            ),
        },
    }
    return instrument


def simulate_expert_responses(n_experts: int = 15) -> dict:
    """
    Simulate expert responses for demonstration purposes.

    Generates plausible Likert ratings with moderate-to-high agreement
    to demonstrate the consensus analysis pipeline.
    """
    np.random.seed(42)
    simulated = {
        "likert_ratings": {},
        "classifications": [],
    }

    # Simulate Part A: Dimension validation
    # Experts generally agree dimensions are necessary and clear
    for i in range(1, 4):
        simulated["likert_ratings"][f"A{i}_necessity"] = (
            np.clip(np.random.normal(4.2, 0.7, n_experts), 1, 5)
            .round().astype(int).tolist()
        )
        simulated["likert_ratings"][f"A{i}_clarity"] = (
            np.clip(np.random.normal(3.8, 0.9, n_experts), 1, 5)
            .round().astype(int).tolist()
        )
        simulated["likert_ratings"][f"A{i}_completeness"] = (
            np.clip(np.random.normal(3.6, 1.0, n_experts), 1, 5)
            .round().astype(int).tolist()
        )

    # Simulate Part B: Category validation
    for i in range(1, 9):
        simulated["likert_ratings"][f"B{i}_well_defined"] = (
            np.clip(np.random.normal(3.9, 0.8, n_experts), 1, 5)
            .round().astype(int).tolist()
        )
        simulated["likert_ratings"][f"B{i}_distinguishable"] = (
            np.clip(np.random.normal(3.7, 0.9, n_experts), 1, 5)
            .round().astype(int).tolist()
        )
        simulated["likert_ratings"][f"B{i}_examples_representative"] = (
            np.clip(np.random.normal(4.0, 0.7, n_experts), 1, 5)
            .round().astype(int).tolist()
        )

    # Simulate Part C: Classification exercise
    # Expected category indices (0-indexed into CATEGORY_OPTIONS)
    expected_map = {
        "C1": 2,   # MEV Searcher
        "C2": 1,   # Simple Trading Bot (boundary case)
        "C3": 5,   # DeFi Management Agent
        "C4": 0,   # Deterministic Script
        "C5": 8,   # Not an agent
        "C6": 2,   # MEV Searcher
        "C7": 3,   # Cross-Chain Bridge Agent
        "C8": 6,   # LLM-Powered Agent
        "C9": 0,   # Deterministic Script
        "C10": 3,  # Cross-Chain Bridge Agent (boundary)
    }

    for expert_idx in range(n_experts):
        expert_classifications = []
        for entity_id in [f"C{i}" for i in range(1, 11)]:
            expected = expected_map[entity_id]
            # Most experts agree with expected, some deviate
            if np.random.random() < 0.75:
                choice = expected
            else:
                # Random deviation for boundary cases
                choice = np.random.choice(len(CATEGORY_OPTIONS))
            expert_classifications.append(choice)
        simulated["classifications"].append(expert_classifications)

    # Simulate Part D: Comparison
    simulated["likert_ratings"]["D4_web3_applicability"] = (
        np.clip(np.random.normal(4.3, 0.6, n_experts), 1, 5)
        .round().astype(int).tolist()
    )
    simulated["likert_ratings"]["D5_generalizability"] = (
        np.clip(np.random.normal(3.5, 1.0, n_experts), 1, 5)
        .round().astype(int).tolist()
    )

    # Simulate confidence ratings for classification
    for i in range(1, 11):
        # Higher confidence for clear cases, lower for boundary
        base_conf = 4.0 if i not in [2, 4, 10] else 3.0
        simulated["likert_ratings"][f"C{i}_confidence"] = (
            np.clip(np.random.normal(base_conf, 0.8, n_experts), 1, 5)
            .round().astype(int).tolist()
        )

    return simulated


def main():
    """Generate the Delphi instrument and run simulated validation."""
    print("=" * 70)
    print("Delphi Expert Validation for Web3 AI Agent Taxonomy")
    print("=" * 70)

    # 1. Generate full instrument
    print("\n--- Generating Delphi Instrument ---")
    instrument = generate_full_instrument()

    n_questions_a = sum(
        len(d["questions"])
        for d in instrument["round_1"]["part_a_dimension_validation"]
    )
    n_questions_b = sum(
        len(c["questions"])
        for c in instrument["round_1"]["part_b_category_validation"]
    )
    n_entities_c = len(
        instrument["round_1"]["part_c_classification_exercise"]
    )
    n_questions_d = len(
        instrument["round_1"]["part_d_taxonomy_comparison"]["questions"]
    )

    print(f"Part A (Dimension Validation): {n_questions_a} questions "
          f"across {len(DIMENSIONS)} dimensions")
    print(f"Part B (Category Validation): {n_questions_b} questions "
          f"across {len(CATEGORIES)} categories")
    print(f"Part C (Classification Exercise): {n_entities_c} entities "
          f"to classify")
    print(f"Part D (Taxonomy Comparison): {n_questions_d} questions "
          f"across {len(COMPARISON_FRAMEWORKS)} frameworks")
    total_q = n_questions_a + n_questions_b + n_entities_c * 3 + n_questions_d
    print(f"Total questions: {total_q}")

    # 2. Simulate expert responses
    print("\n--- Simulating Expert Responses (N=15) ---")
    simulated = simulate_expert_responses(n_experts=15)

    # 3. Compute consensus metrics
    print("\n--- Consensus Analysis ---")
    consensus = compute_consensus_metrics(simulated["likert_ratings"])
    print(f"Total Likert questions analyzed: "
          f"{consensus['summary']['total_likert_questions']}")
    print(f"Consensus reached (IQR <= 1): "
          f"{consensus['summary']['consensus_reached']} / "
          f"{consensus['summary']['total_likert_questions']} "
          f"({consensus['summary']['consensus_rate']:.1%})")

    # Show per-question details for Part A
    print("\nPart A - Dimension Necessity Ratings:")
    for dim_name, qid in zip(
        ["Autonomy", "Environment", "Decision Model"],
        ["A1_necessity", "A2_necessity", "A3_necessity"],
    ):
        m = consensus["per_question"][qid]
        print(f"  {dim_name}: median={m['median']}, IQR={m['iqr']}, "
              f"consensus={'YES' if m['consensus'] else 'NO'}")

    print("\nPart B - Category Well-Defined Ratings:")
    for i, cat in enumerate(CATEGORIES, 1):
        qid = f"B{i}_well_defined"
        m = consensus["per_question"][qid]
        print(f"  {cat['name']}: median={m['median']}, IQR={m['iqr']}, "
              f"consensus={'YES' if m['consensus'] else 'NO'}")

    # 4. Compute inter-rater agreement
    print("\n--- Inter-Rater Agreement (Classification Exercise) ---")
    alpha = compute_krippendorff_alpha(
        simulated["classifications"],
        n_categories=len(CATEGORY_OPTIONS),
    )
    print(f"Krippendorff's alpha: {alpha:.3f}")

    if alpha >= 0.8:
        interpretation = "Good agreement"
    elif alpha >= 0.667:
        interpretation = "Acceptable agreement"
    else:
        interpretation = "Low agreement - categories may need refinement"
    print(f"Interpretation: {interpretation}")

    # Compute per-entity agreement rate
    print("\nPer-Entity Agreement:")
    classifications = np.array(simulated["classifications"])
    for i, entity in enumerate(CLASSIFICATION_ENTITIES):
        entity_ratings = classifications[:, i]
        modal_category = int(np.argmax(np.bincount(entity_ratings, minlength=len(CATEGORY_OPTIONS))))
        agreement_rate = np.mean(entity_ratings == modal_category)
        print(f"  {entity.name}: {agreement_rate:.0%} agree on "
              f"'{CATEGORY_OPTIONS[modal_category]}' "
              f"(expected: {entity.expected_category})")

    # 5. Generate Round 2 instrument
    print("\n--- Round 2 Instrument Generation ---")
    round2 = build_round2_instrument(simulated["likert_ratings"])
    n_rerating = len(round2["items_requiring_re_rating"])
    n_consensus = len(round2["items_with_consensus"])
    print(f"Items with consensus (no re-rating needed): {n_consensus}")
    print(f"Items requiring re-rating: {n_rerating}")

    # 6. Part D summary
    print("\n--- Taxonomy Comparison Summary ---")
    d4 = consensus["per_question"]["D4_web3_applicability"]
    d5 = consensus["per_question"]["D5_generalizability"]
    print(f"Web3 applicability of our taxonomy: median={d4['median']}, "
          f"IQR={d4['iqr']}")
    print(f"Generalizability beyond Web3: median={d5['median']}, "
          f"IQR={d5['iqr']}")

    # Save outputs
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output = {
        "instrument": instrument,
        "simulated_results": {
            "n_experts": 15,
            "consensus_metrics": consensus,
            "krippendorff_alpha": round(alpha, 4),
            "round2_instrument": round2,
        },
    }

    output_path = os.path.join(output_dir, "delphi_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("DELPHI STUDY INSTRUMENT GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
