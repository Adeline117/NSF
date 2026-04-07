"""
Delphi Study Analysis Plan
===========================
Process Round 1 and Round 2 Delphi responses:
- Compute median and IQR for each Likert question
- Identify consensus (IQR <= 1) vs non-consensus items
- Compute Krippendorff's alpha for classification exercise
- Generate summary tables and visualizations
- Flag items needing Round 2 discussion
"""

import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from itertools import combinations

import numpy as np

# Optional imports for visualization
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class LikertResult:
    """Aggregated result for a single Likert question."""
    question_id: str
    question_text: str
    n_respondents: int
    ratings: list[int]
    median: float
    mean: float
    std: float
    q25: float
    q75: float
    iqr: float
    consensus_reached: bool


@dataclass
class ClassificationResult:
    """Aggregated result for a single classification item."""
    entity_id: str
    entity_name: str
    expected_category: str
    responses: list[str]
    modal_category: str
    agreement_rate: float
    confidence_ratings: list[int]
    confidence_median: float
    c1_votes: dict  # {"Y": n, "N": m}
    c2_votes: dict
    c3_votes: dict
    c4_votes: dict


# ============================================================
# CONSENSUS ANALYSIS
# ============================================================

CONSENSUS_THRESHOLD = 1.0  # IQR <= 1.0 means consensus


def compute_likert_stats(
    question_id: str,
    question_text: str,
    ratings: list[int],
) -> LikertResult:
    """Compute descriptive statistics for a single Likert question."""
    arr = np.array(ratings, dtype=float)
    n = len(arr)
    median = float(np.median(arr))
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    q25 = float(np.percentile(arr, 25))
    q75 = float(np.percentile(arr, 75))
    iqr = q75 - q25

    return LikertResult(
        question_id=question_id,
        question_text=question_text,
        n_respondents=n,
        ratings=ratings,
        median=round(median, 2),
        mean=round(mean, 2),
        std=round(std, 2),
        q25=round(q25, 2),
        q75=round(q75, 2),
        iqr=round(iqr, 2),
        consensus_reached=iqr <= CONSENSUS_THRESHOLD,
    )


def analyze_all_likert(responses: dict[str, list[int]], question_texts: dict[str, str]) -> list[LikertResult]:
    """
    Analyze all Likert questions from Round 1 or Round 2.

    Args:
        responses: {question_id: [list of integer ratings from all experts]}
        question_texts: {question_id: "question text"}

    Returns:
        List of LikertResult objects.
    """
    results = []
    for qid, ratings in responses.items():
        text = question_texts.get(qid, qid)
        result = compute_likert_stats(qid, text, ratings)
        results.append(result)
    return results


def identify_non_consensus_items(results: list[LikertResult]) -> list[LikertResult]:
    """Identify items that need Round 2 re-rating (IQR > 1)."""
    return [r for r in results if not r.consensus_reached]


def identify_consensus_items(results: list[LikertResult]) -> list[LikertResult]:
    """Identify items where consensus was reached (IQR <= 1)."""
    return [r for r in results if r.consensus_reached]


# ============================================================
# KRIPPENDORFF'S ALPHA
# ============================================================

def _build_coincidence_matrix(
    data: list[list[str | None]],
    categories: list[str],
) -> np.ndarray:
    """
    Build a coincidence matrix for Krippendorff's alpha.

    Args:
        data: List of raters' responses. data[r][i] = category assigned by
              rater r to item i. None if rater did not code that item.
        categories: List of unique category labels.

    Returns:
        Coincidence matrix of shape (n_categories, n_categories).
    """
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)
    coincidence = np.zeros((n_cats, n_cats), dtype=float)

    n_raters = len(data)
    n_items = len(data[0]) if data else 0

    for i in range(n_items):
        # Collect all non-None responses for this item
        item_codes = []
        for r in range(n_raters):
            if data[r][i] is not None:
                item_codes.append(data[r][i])

        m_i = len(item_codes)
        if m_i < 2:
            continue

        # Add to coincidence matrix
        for a, b in combinations(range(m_i), 2):
            c_a = cat_to_idx.get(item_codes[a])
            c_b = cat_to_idx.get(item_codes[b])
            if c_a is not None and c_b is not None:
                weight = 1.0 / (m_i - 1)
                coincidence[c_a, c_b] += weight
                coincidence[c_b, c_a] += weight

    return coincidence


def krippendorff_alpha_nominal(data: list[list[str | None]]) -> float:
    """
    Compute Krippendorff's alpha for nominal data.

    Args:
        data: List of raters' category assignments.
              data[r][i] = category string or None.

    Returns:
        Krippendorff's alpha (float between -1 and 1).
        Values > 0.667 are typically considered acceptable.
    """
    # Collect all categories
    all_cats = set()
    for rater_data in data:
        for val in rater_data:
            if val is not None:
                all_cats.add(val)
    categories = sorted(all_cats)

    if len(categories) < 2:
        return 1.0  # Perfect agreement if only one category used

    coincidence = _build_coincidence_matrix(data, categories)
    n_cats = len(categories)

    # Observed disagreement (D_o)
    n_total = np.sum(coincidence)
    if n_total == 0:
        return 0.0

    d_o = 1.0 - np.trace(coincidence) / n_total

    # Expected disagreement (D_e)
    marginals = np.sum(coincidence, axis=1)
    n_total_pairs = np.sum(marginals)

    d_e = 0.0
    for c in range(n_cats):
        for k in range(n_cats):
            if c != k:
                d_e += marginals[c] * marginals[k]
    d_e /= (n_total_pairs * (n_total_pairs - 1)) if n_total_pairs > 1 else 1

    if d_e == 0:
        return 1.0

    alpha = 1.0 - d_o / d_e
    return round(alpha, 4)


def analyze_classification_exercise(
    expert_responses: list[dict],
    entities: list[dict],
) -> tuple[list[ClassificationResult], float]:
    """
    Analyze Part C classification exercise.

    Args:
        expert_responses: List of dicts, one per expert.
            Each dict maps entity_id -> {
                "category": str,
                "c1": "Y"/"N",
                "c2": "Y"/"N",
                "c3": "Y"/"N",
                "c4": "Y"/"N",
                "confidence": int (1-5),
            }
        entities: List of entity dicts with "id", "name", "expected_category".

    Returns:
        Tuple of (list of ClassificationResult, Krippendorff's alpha).
    """
    results = []

    for entity in entities:
        eid = entity["id"]
        categories_assigned = []
        confidences = []
        c1_votes = {"Y": 0, "N": 0}
        c2_votes = {"Y": 0, "N": 0}
        c3_votes = {"Y": 0, "N": 0}
        c4_votes = {"Y": 0, "N": 0}

        for expert in expert_responses:
            if eid in expert:
                resp = expert[eid]
                categories_assigned.append(resp["category"])
                confidences.append(resp["confidence"])
                c1_votes[resp.get("c1", "N")] += 1
                c2_votes[resp.get("c2", "N")] += 1
                c3_votes[resp.get("c3", "N")] += 1
                c4_votes[resp.get("c4", "N")] += 1

        if categories_assigned:
            counter = Counter(categories_assigned)
            modal = counter.most_common(1)[0][0]
            agreement = counter.most_common(1)[0][1] / len(categories_assigned)
        else:
            modal = "N/A"
            agreement = 0.0

        results.append(ClassificationResult(
            entity_id=eid,
            entity_name=entity["name"],
            expected_category=entity["expected_category"],
            responses=categories_assigned,
            modal_category=modal,
            agreement_rate=round(agreement, 3),
            confidence_ratings=confidences,
            confidence_median=round(float(np.median(confidences)), 2) if confidences else 0.0,
            c1_votes=c1_votes,
            c2_votes=c2_votes,
            c3_votes=c3_votes,
            c4_votes=c4_votes,
        ))

    # Compute Krippendorff's alpha across all classification items
    n_experts = len(expert_responses)
    n_items = len(entities)
    data_matrix = []

    for expert in expert_responses:
        rater_codes = []
        for entity in entities:
            eid = entity["id"]
            if eid in expert:
                rater_codes.append(expert[eid]["category"])
            else:
                rater_codes.append(None)
        data_matrix.append(rater_codes)

    alpha = krippendorff_alpha_nominal(data_matrix)

    return results, alpha


# ============================================================
# ROUND COMPARISON (Round 1 vs Round 2)
# ============================================================

def compare_rounds(
    round1_results: list[LikertResult],
    round2_results: list[LikertResult],
) -> dict:
    """
    Compare Round 1 and Round 2 results to measure convergence.

    Returns:
        Dict with convergence statistics.
    """
    r1_map = {r.question_id: r for r in round1_results}
    r2_map = {r.question_id: r for r in round2_results}

    common_ids = set(r1_map.keys()) & set(r2_map.keys())

    convergence_stats = {
        "total_items": len(common_ids),
        "r1_consensus_count": 0,
        "r2_consensus_count": 0,
        "newly_converged": [],
        "still_divergent": [],
        "median_shifts": [],
        "iqr_reductions": [],
    }

    for qid in sorted(common_ids):
        r1 = r1_map[qid]
        r2 = r2_map[qid]

        if r1.consensus_reached:
            convergence_stats["r1_consensus_count"] += 1
        if r2.consensus_reached:
            convergence_stats["r2_consensus_count"] += 1

        if not r1.consensus_reached and r2.consensus_reached:
            convergence_stats["newly_converged"].append(qid)
        elif not r1.consensus_reached and not r2.consensus_reached:
            convergence_stats["still_divergent"].append(qid)

        convergence_stats["median_shifts"].append({
            "question_id": qid,
            "r1_median": r1.median,
            "r2_median": r2.median,
            "shift": round(r2.median - r1.median, 2),
        })

        convergence_stats["iqr_reductions"].append({
            "question_id": qid,
            "r1_iqr": r1.iqr,
            "r2_iqr": r2.iqr,
            "reduction": round(r1.iqr - r2.iqr, 2),
        })

    convergence_stats["convergence_rate"] = round(
        convergence_stats["r2_consensus_count"]
        / convergence_stats["total_items"],
        3,
    ) if convergence_stats["total_items"] > 0 else 0

    return convergence_stats


# ============================================================
# SUMMARY TABLE GENERATION
# ============================================================

def generate_summary_table(results: list[LikertResult]) -> str:
    """Generate a Markdown summary table of all Likert results."""
    lines = [
        "| Question ID | Median | Mean | SD | Q25 | Q75 | IQR | Consensus? | N |",
        "|-------------|--------|------|----|-----|-----|-----|------------|---|",
    ]
    for r in results:
        consensus_str = "Yes" if r.consensus_reached else "**NO**"
        lines.append(
            f"| {r.question_id} | {r.median} | {r.mean} | {r.std} | "
            f"{r.q25} | {r.q75} | {r.iqr} | {consensus_str} | {r.n_respondents} |"
        )
    return "\n".join(lines)


def generate_classification_table(results: list[ClassificationResult]) -> str:
    """Generate a Markdown table of classification results."""
    lines = [
        "| Entity | Expected | Modal Response | Agreement | Confidence (Md) | C1 | C2 | C3 | C4 |",
        "|--------|----------|----------------|-----------|-----------------|----|----|----|----|",
    ]
    for r in results:
        c1_pct = r.c1_votes["Y"] / max(sum(r.c1_votes.values()), 1) * 100
        c2_pct = r.c2_votes["Y"] / max(sum(r.c2_votes.values()), 1) * 100
        c3_pct = r.c3_votes["Y"] / max(sum(r.c3_votes.values()), 1) * 100
        c4_pct = r.c4_votes["Y"] / max(sum(r.c4_votes.values()), 1) * 100
        lines.append(
            f"| {r.entity_name} | {r.expected_category[:25]} | "
            f"{r.modal_category[:25]} | {r.agreement_rate:.0%} | "
            f"{r.confidence_median} | {c1_pct:.0f}% | {c2_pct:.0f}% | "
            f"{c3_pct:.0f}% | {c4_pct:.0f}% |"
        )
    return "\n".join(lines)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_likert_summary(
    results: list[LikertResult],
    title: str = "Delphi Round 1: Likert Ratings Summary",
    output_path: str | None = None,
):
    """Plot a horizontal bar chart of median ratings with IQR error bars."""
    if not HAS_MATPLOTLIB:
        print("  [SKIP] matplotlib not available; skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, max(6, len(results) * 0.4)))

    qids = [r.question_id for r in results]
    medians = [r.median for r in results]
    iqr_lower = [r.median - r.q25 for r in results]
    iqr_upper = [r.q75 - r.median for r in results]
    colors = ["#2ecc71" if r.consensus_reached else "#e74c3c" for r in results]

    y_pos = range(len(results))
    ax.barh(y_pos, medians, xerr=[iqr_lower, iqr_upper],
            color=colors, edgecolor="black", linewidth=0.5,
            capsize=3, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(qids, fontsize=8)
    ax.set_xlabel("Rating (1-5)")
    ax.set_xlim(0.5, 5.5)
    ax.axvline(x=3, color="gray", linestyle="--", alpha=0.5, label="Neutral")
    ax.set_title(title)
    ax.legend(["Neutral line", "Consensus (IQR<=1)", "No consensus (IQR>1)"],
              loc="lower right", fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {output_path}")
    plt.close()


def plot_classification_agreement(
    results: list[ClassificationResult],
    output_path: str | None = None,
):
    """Plot a bar chart of classification agreement rates."""
    if not HAS_MATPLOTLIB:
        print("  [SKIP] matplotlib not available; skipping visualization")
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    names = [r.entity_name[:20] for r in results]
    agreements = [r.agreement_rate * 100 for r in results]
    colors = ["#2ecc71" if a >= 75 else "#f39c12" if a >= 50 else "#e74c3c"
              for a in agreements]

    bars = ax.bar(range(len(results)), agreements, color=colors,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Agreement Rate (%)")
    ax.set_ylim(0, 105)
    ax.axhline(y=75, color="green", linestyle="--", alpha=0.5, label="75% threshold")
    ax.axhline(y=50, color="orange", linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_title("Classification Exercise: Expert Agreement")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"  Saved plot to {output_path}")
    plt.close()


# ============================================================
# FLAGGING ITEMS FOR ROUND 2
# ============================================================

def flag_round2_items(
    likert_results: list[LikertResult],
    classification_results: list[ClassificationResult],
    alpha: float,
) -> dict:
    """
    Flag items needing Round 2 discussion.

    Criteria:
    - Likert items with IQR > 1 (no consensus)
    - Classification items with agreement < 50%
    - Krippendorff's alpha < 0.667 (overall classification reliability)
    """
    flags = {
        "likert_non_consensus": [],
        "classification_low_agreement": [],
        "overall_classification_reliability": {
            "krippendorff_alpha": alpha,
            "acceptable": alpha >= 0.667,
            "interpretation": (
                "Good reliability" if alpha >= 0.8
                else "Acceptable reliability" if alpha >= 0.667
                else "Low reliability -- taxonomy may need revision"
            ),
        },
        "summary": {},
    }

    for r in likert_results:
        if not r.consensus_reached:
            flags["likert_non_consensus"].append({
                "question_id": r.question_id,
                "median": r.median,
                "iqr": r.iqr,
                "action": "Include in Round 2 for re-rating",
            })

    for r in classification_results:
        if r.agreement_rate < 0.5:
            flags["classification_low_agreement"].append({
                "entity": r.entity_name,
                "agreement_rate": r.agreement_rate,
                "modal_category": r.modal_category,
                "expected_category": r.expected_category,
                "action": "Discuss in Round 2; may indicate ambiguous boundary",
            })

    flags["summary"] = {
        "total_likert_items": len(likert_results),
        "consensus_reached": len([r for r in likert_results if r.consensus_reached]),
        "needs_re_rating": len(flags["likert_non_consensus"]),
        "classification_items": len(classification_results),
        "high_agreement_items": len([r for r in classification_results if r.agreement_rate >= 0.75]),
        "low_agreement_items": len(flags["classification_low_agreement"]),
    }

    return flags


# ============================================================
# FULL ANALYSIS PIPELINE
# ============================================================

def run_full_analysis(
    likert_responses: dict[str, list[int]],
    question_texts: dict[str, str],
    classification_responses: list[dict],
    entities: list[dict],
    output_dir: str,
) -> dict:
    """
    Run the full Delphi analysis pipeline.

    Args:
        likert_responses: {question_id: [list of ratings]}
        question_texts: {question_id: "text"}
        classification_responses: List of expert response dicts
        entities: List of entity dicts
        output_dir: Directory to save outputs

    Returns:
        Complete analysis results dict.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("Delphi Expert Study: Full Analysis Pipeline")
    print("=" * 70)

    # 1. Likert analysis
    print("\n--- Part 1: Likert Question Analysis ---")
    likert_results = analyze_all_likert(likert_responses, question_texts)
    consensus = identify_consensus_items(likert_results)
    non_consensus = identify_non_consensus_items(likert_results)

    print(f"  Total Likert items: {len(likert_results)}")
    print(f"  Consensus reached (IQR <= 1): {len(consensus)}")
    print(f"  No consensus (IQR > 1): {len(non_consensus)}")

    for r in non_consensus:
        print(f"    - {r.question_id}: median={r.median}, IQR={r.iqr}")

    # Print summary table
    print("\n  Summary Table:")
    print(generate_summary_table(likert_results))

    # 2. Classification analysis
    print("\n--- Part 2: Classification Exercise Analysis ---")
    class_results, alpha = analyze_classification_exercise(
        classification_responses, entities,
    )

    print(f"  Krippendorff's alpha: {alpha:.4f}")
    if alpha >= 0.8:
        print("  Interpretation: Good reliability")
    elif alpha >= 0.667:
        print("  Interpretation: Acceptable reliability")
    else:
        print("  Interpretation: LOW reliability -- taxonomy needs revision")

    print("\n  Classification Table:")
    print(generate_classification_table(class_results))

    # 3. Flags for Round 2
    print("\n--- Part 3: Round 2 Flags ---")
    flags = flag_round2_items(likert_results, class_results, alpha)
    print(f"  Items needing re-rating: {flags['summary']['needs_re_rating']}")
    print(f"  Low-agreement classifications: {flags['summary']['low_agreement_items']}")
    print(f"  Classification reliability: {flags['overall_classification_reliability']['interpretation']}")

    # 4. Visualizations
    print("\n--- Part 4: Generating Visualizations ---")
    plot_likert_summary(
        likert_results,
        output_path=os.path.join(output_dir, "likert_summary.png"),
    )
    plot_classification_agreement(
        class_results,
        output_path=os.path.join(output_dir, "classification_agreement.png"),
    )

    # 5. Save results
    output = {
        "likert_results": [asdict(r) for r in likert_results],
        "classification_results": [asdict(r) for r in class_results],
        "krippendorff_alpha": alpha,
        "round2_flags": flags,
        "summary_table_markdown": generate_summary_table(likert_results),
        "classification_table_markdown": generate_classification_table(class_results),
    }

    def _json_serializer(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    output_path = os.path.join(output_dir, "delphi_analysis_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=_json_serializer)
    print(f"\n  Results saved to {output_path}")

    return output


# ============================================================
# DEMO WITH SIMULATED DATA
# ============================================================

def generate_simulated_data() -> tuple:
    """
    Generate simulated expert responses for demonstration.
    In production, this would be replaced by actual survey data.
    """
    np.random.seed(42)
    n_experts = 12

    # Question texts
    question_texts = {
        # Part A: Definition validation (C1-C4 x {necessity, clarity})
        "A_C1_necessity": "Is C1 (On-chain Actuation) necessary?",
        "A_C1_clarity": "Is C1 clearly defined?",
        "A_C2_necessity": "Is C2 (Environmental Perception) necessary?",
        "A_C2_clarity": "Is C2 clearly defined?",
        "A_C3_necessity": "Is C3 (Autonomous Decision-Making) necessary?",
        "A_C3_clarity": "Is C3 clearly defined?",
        "A_C4_necessity": "Is C4 (Adaptiveness) necessary?",
        "A_C4_clarity": "Is C4 clearly defined?",
        # Part B: Category validation (8 categories x {well-defined, distinguishable})
        "B1_well_defined": "Deterministic Script: well-defined?",
        "B1_distinguishable": "Deterministic Script: distinguishable?",
        "B2_well_defined": "Simple Trading Bot: well-defined?",
        "B2_distinguishable": "Simple Trading Bot: distinguishable?",
        "B3_well_defined": "MEV Searcher: well-defined?",
        "B3_distinguishable": "MEV Searcher: distinguishable?",
        "B4_well_defined": "Cross-Chain Bridge Agent: well-defined?",
        "B4_distinguishable": "Cross-Chain Bridge Agent: distinguishable?",
        "B5_well_defined": "RL Trading Agent: well-defined?",
        "B5_distinguishable": "RL Trading Agent: distinguishable?",
        "B6_well_defined": "DeFi Management Agent: well-defined?",
        "B6_distinguishable": "DeFi Management Agent: distinguishable?",
        "B7_well_defined": "LLM-Powered Agent: well-defined?",
        "B7_distinguishable": "LLM-Powered Agent: distinguishable?",
        "B8_well_defined": "Autonomous DAO Agent: well-defined?",
        "B8_distinguishable": "Autonomous DAO Agent: distinguishable?",
        # Part D: Framework comparison
        "D_web3_applicability": "Our taxonomy's Web3 applicability?",
        "D_generalizability": "Our taxonomy's generalizability?",
    }

    # Simulate Likert responses (generally positive with some variance)
    # C1 and C2 should have strong consensus; C3 and C4 may be more debated
    likert_responses = {}

    # High consensus items (strong agreement)
    for qid in ["A_C1_necessity", "A_C2_necessity", "A_C3_necessity"]:
        likert_responses[qid] = list(np.random.choice([4, 5], size=n_experts, p=[0.3, 0.7]))

    # Lower consensus on C4 (adaptiveness -- debatable)
    likert_responses["A_C4_necessity"] = list(np.random.choice([3, 4, 5], size=n_experts, p=[0.2, 0.4, 0.4]))

    # Clarity ratings
    for qid in ["A_C1_clarity", "A_C2_clarity", "A_C3_clarity"]:
        likert_responses[qid] = list(np.random.choice([3, 4, 5], size=n_experts, p=[0.1, 0.4, 0.5]))
    likert_responses["A_C4_clarity"] = list(np.random.choice([2, 3, 4, 5], size=n_experts, p=[0.1, 0.3, 0.3, 0.3]))

    # Category well-definedness (some categories more controversial)
    easy_cats = ["B1", "B3", "B7", "B8"]  # Clear categories
    hard_cats = ["B2", "B4", "B5", "B6"]  # Boundary-ambiguous

    for cat_id in easy_cats:
        likert_responses[f"{cat_id}_well_defined"] = list(np.random.choice([4, 5], size=n_experts, p=[0.4, 0.6]))
        likert_responses[f"{cat_id}_distinguishable"] = list(np.random.choice([3, 4, 5], size=n_experts, p=[0.1, 0.4, 0.5]))

    for cat_id in hard_cats:
        likert_responses[f"{cat_id}_well_defined"] = list(np.random.choice([3, 4, 5], size=n_experts, p=[0.3, 0.4, 0.3]))
        likert_responses[f"{cat_id}_distinguishable"] = list(np.random.choice([2, 3, 4], size=n_experts, p=[0.2, 0.5, 0.3]))

    # Framework comparison
    likert_responses["D_web3_applicability"] = list(np.random.choice([4, 5], size=n_experts, p=[0.3, 0.7]))
    likert_responses["D_generalizability"] = list(np.random.choice([3, 4, 5], size=n_experts, p=[0.3, 0.4, 0.3]))

    # Convert numpy int64 to native int
    for k in likert_responses:
        likert_responses[k] = [int(v) for v in likert_responses[k]]

    # Classification exercise
    categories = [
        "Deterministic Script",
        "Simple Trading Bot",
        "MEV Searcher",
        "Cross-Chain Bridge Agent",
        "RL Trading Agent",
        "DeFi Management Agent",
        "LLM-Powered Agent",
        "Autonomous DAO Agent",
        "Not an Agent",
    ]

    entities = [
        {"id": "C1", "name": "jaredfromsubway.eth", "expected_category": "MEV Searcher"},
        {"id": "C2", "name": "Wintermute", "expected_category": "RL Trading Agent"},
        {"id": "C3", "name": "Autonolas OLAS agent", "expected_category": "DeFi Management Agent"},
        {"id": "C4", "name": "Uniswap V2 Router", "expected_category": "Not an Agent"},
        {"id": "C5", "name": "Personal wallet", "expected_category": "Not an Agent"},
        {"id": "C6", "name": "Aave liquidation bot", "expected_category": "MEV Searcher"},
        {"id": "C7", "name": "LayerZero relayer", "expected_category": "Cross-Chain Bridge Agent"},
        {"id": "C8", "name": "AI16Z ELIZA agent", "expected_category": "LLM-Powered Agent"},
        {"id": "C9", "name": "Cron payroll script", "expected_category": "Deterministic Script"},
        {"id": "C10", "name": "Chainlink oracle node", "expected_category": "Cross-Chain Bridge Agent"},
    ]

    # Simulate classification with mostly correct but some disagreement
    classification_responses = []
    for expert_idx in range(n_experts):
        expert = {}
        for entity in entities:
            expected = entity["expected_category"]
            # 70% chance correct, 30% chance of plausible alternative
            if np.random.random() < 0.70:
                chosen = expected
            else:
                # Pick a plausible alternative
                alternatives = {
                    "MEV Searcher": ["Simple Trading Bot", "RL Trading Agent"],
                    "RL Trading Agent": ["MEV Searcher", "Simple Trading Bot"],
                    "DeFi Management Agent": ["RL Trading Agent", "LLM-Powered Agent"],
                    "Not an Agent": ["Deterministic Script"],
                    "Cross-Chain Bridge Agent": ["Deterministic Script", "Simple Trading Bot"],
                    "LLM-Powered Agent": ["DeFi Management Agent"],
                    "Deterministic Script": ["Simple Trading Bot", "Not an Agent"],
                }
                alts = alternatives.get(expected, categories[:3])
                chosen = np.random.choice(alts)

            # C1-C4 votes
            is_agent = expected not in ["Not an Agent"]
            expert[entity["id"]] = {
                "category": chosen,
                "c1": "Y" if is_agent and np.random.random() < 0.9 else "N",
                "c2": "Y" if is_agent and np.random.random() < 0.85 else "N",
                "c3": "Y" if is_agent and np.random.random() < 0.8 else "N",
                "c4": "Y" if is_agent and expected != "Deterministic Script" and np.random.random() < 0.75 else "N",
                "confidence": int(np.random.choice([3, 4, 5], p=[0.2, 0.5, 0.3])),
            }

        classification_responses.append(expert)

    return likert_responses, question_texts, classification_responses, entities


# ============================================================
# MAIN
# ============================================================

def main():
    """Run the analysis pipeline with simulated data."""
    output_dir = os.path.dirname(os.path.abspath(__file__))

    # Generate simulated data
    print("Generating simulated expert responses for demonstration...")
    (
        likert_responses,
        question_texts,
        classification_responses,
        entities,
    ) = generate_simulated_data()

    # Run full analysis
    results = run_full_analysis(
        likert_responses=likert_responses,
        question_texts=question_texts,
        classification_responses=classification_responses,
        entities=entities,
        output_dir=output_dir,
    )

    # Final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"""
Key findings from simulated data:
  - Likert consensus rate: {results['round2_flags']['summary']['consensus_reached']}/{results['round2_flags']['summary']['total_likert_items']}
  - Items needing Round 2: {results['round2_flags']['summary']['needs_re_rating']}
  - Krippendorff's alpha: {results['krippendorff_alpha']:.4f}
  - Classification reliability: {results['round2_flags']['overall_classification_reliability']['interpretation']}
  - High-agreement entities: {results['round2_flags']['summary']['high_agreement_items']}/{results['round2_flags']['summary']['classification_items']}

NOTE: These results use SIMULATED data for demonstration.
      Replace generate_simulated_data() with actual survey responses
      when real data is collected.
""")


if __name__ == "__main__":
    main()
