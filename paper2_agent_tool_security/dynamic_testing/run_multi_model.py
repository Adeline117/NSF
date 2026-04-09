"""
Paper 2: Multi-Model Dynamic Testing Comparison
=================================================
Runs the 16 attack scenarios from llm_dynamic_harness.py across multiple
Claude models and produces a unified comparison report.

Models tested:
  - claude-opus-4-6       (existing results loaded from disk)
  - claude-haiku-4-5-20251001  (cheaper, less capable)
  - claude-sonnet-4-6     (mid-tier)

Output:
  - multi_model_dynamic_results.json
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from paper2_agent_tool_security.dynamic_testing.llm_dynamic_harness import (
    ATTACK_SCENARIOS,
    LLMTestHarness,
)

OUTPUT_DIR = Path(__file__).resolve().parent
MULTI_MODEL_OUTPUT = OUTPUT_DIR / "multi_model_dynamic_results.json"

# Models to run live (opus already has saved results)
MODELS_TO_RUN = [
    "claude-haiku-4-5-20251001",
    "claude-sonnet-4-6",
]

OPUS_RESULTS_PATH = OUTPUT_DIR / "llm_dynamic_results_opus.json"


def load_opus_results() -> dict:
    """Load pre-existing Opus results."""
    if OPUS_RESULTS_PATH.exists():
        with open(OPUS_RESULTS_PATH) as f:
            return json.load(f)
    print(f"WARNING: {OPUS_RESULTS_PATH} not found, skipping Opus.")
    return None


def run_model(model_name: str) -> dict:
    """Run all 16 scenarios for a single model and return a summary dict."""
    print(f"\n{'=' * 70}")
    print(f"Running model: {model_name}")
    print(f"{'=' * 70}")

    harness = LLMTestHarness(model=model_name)
    if not harness._available:
        print("ERROR: claude_agent_sdk not available")
        sys.exit(1)

    # Override the model attribute (the harness __init__ stores it but
    # the _ask method uses self._options which doesn't pass model directly;
    # the query function uses the default model.  We need to check if
    # the harness actually uses the model parameter.)
    # Looking at the code: __init__ stores self.model but _ask() doesn't
    # pass it to query().  We need to patch _ask to use the model.

    scenarios = ATTACK_SCENARIOS
    print(f"Running {len(scenarios)} scenarios ...\n")

    results = []
    for i, s in enumerate(scenarios, 1):
        print(f"  [{i:2d}/{len(scenarios)}] {s['id']}: {s['name']} ... ", end="", flush=True)
        try:
            r = harness.run_scenario(s)
            results.append(r)
            outcome = "ATTACK SUCCESS" if r["attack_success"] else (
                "REFUSED/WARNED" if r["refused_or_warned"] else "SAFE"
            )
            print(f"{outcome}  ({r['elapsed_seconds']:.1f}s)")
        except Exception as e:
            print(f"ERROR: {type(e).__name__}: {e}")
            results.append({
                "scenario_id": s["id"],
                "scenario_name": s["name"],
                "category": s["category"],
                "error": str(e),
                "attack_success": False,
                "refused_or_warned": False,
            })

    n_total = len(results)
    n_attack = sum(1 for r in results if r.get("attack_success"))
    n_refused = sum(1 for r in results if r.get("refused_or_warned"))

    summary = {
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "n_scenarios": n_total,
        "n_attack_success": n_attack,
        "n_refused": n_refused,
        "n_safe_proceed": n_total - n_attack - n_refused,
        "attack_success_rate": round(n_attack / max(n_total, 1), 4),
        "refusal_rate": round(n_refused / max(n_total, 1), 4),
        "scenarios": results,
    }

    # Also save per-model file
    per_model_path = OUTPUT_DIR / f"llm_dynamic_results_{model_name.split('-')[1]}.json"
    with open(per_model_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved per-model results to {per_model_path.name}")

    return summary


def build_comparison_matrix(all_results: dict) -> list:
    """Build a per-scenario comparison across models."""
    # Collect all scenario IDs from the first model
    scenario_ids = []
    for model_name, data in all_results.items():
        if data and "scenarios" in data:
            scenario_ids = [s["scenario_id"] for s in data["scenarios"]]
            break

    matrix = []
    for sid in scenario_ids:
        row = {"scenario_id": sid}
        for model_name, data in all_results.items():
            if data and "scenarios" in data:
                scenario_data = next(
                    (s for s in data["scenarios"] if s.get("scenario_id") == sid),
                    None
                )
                short_name = model_name.replace("claude-", "").split("-2025")[0]
                if scenario_data:
                    row[f"{short_name}_attack"] = scenario_data.get("attack_success", False)
                    row[f"{short_name}_refused"] = scenario_data.get("refused_or_warned", False)
                    row[f"{short_name}_time"] = scenario_data.get("elapsed_seconds", None)
                else:
                    row[f"{short_name}_attack"] = None
                    row[f"{short_name}_refused"] = None
                    row[f"{short_name}_time"] = None
        matrix.append(row)

    return matrix


def print_comparison_report(all_results: dict, matrix: list):
    """Print a human-readable comparison table."""
    print(f"\n\n{'=' * 80}")
    print("MULTI-MODEL COMPARISON REPORT")
    print(f"{'=' * 80}\n")

    # Summary table
    print(f"{'Model':<35} {'Attack%':>8} {'Refuse%':>8} {'Safe%':>8} {'Attacks':>8} {'Refused':>8}")
    print("-" * 80)
    for model_name, data in all_results.items():
        if data:
            asr = data.get("attack_success_rate", 0) * 100
            rr = data.get("refusal_rate", 0) * 100
            safe = 100 - asr - rr
            print(f"{model_name:<35} {asr:>7.1f}% {rr:>7.1f}% {safe:>7.1f}% "
                  f"{data.get('n_attack_success', '?'):>8} {data.get('n_refused', '?'):>8}")

    # Per-scenario diff
    print(f"\n\nPER-SCENARIO OUTCOMES (A=attack success, R=refused/warned, S=safe):")
    print(f"{'Scenario':<40}", end="")
    short_names = []
    for model_name in all_results:
        sn = model_name.replace("claude-", "").split("-2025")[0]
        short_names.append(sn)
        print(f" {sn:>12}", end="")
    print()
    print("-" * (40 + 13 * len(short_names)))

    differ_scenarios = []
    for row in matrix:
        sid = row["scenario_id"]
        print(f"{sid:<40}", end="")
        outcomes = []
        for sn in short_names:
            atk = row.get(f"{sn}_attack")
            ref = row.get(f"{sn}_refused")
            if atk is None:
                label = "N/A"
            elif atk:
                label = "A"
            elif ref:
                label = "R"
            else:
                label = "S"
            outcomes.append(label)
            print(f" {label:>12}", end="")
        print()
        # Check if outcomes differ
        valid = [o for o in outcomes if o != "N/A"]
        if len(set(valid)) > 1:
            differ_scenarios.append(sid)

    if differ_scenarios:
        print(f"\n\nSCENARIOS WITH DIFFERENT OUTCOMES ACROSS MODELS ({len(differ_scenarios)}):")
        for sid in differ_scenarios:
            print(f"  - {sid}")
    else:
        print("\n\nAll models produced identical outcomes on every scenario.")


def main():
    t_start = time.time()

    all_results = {}

    # 1. Load existing Opus results
    opus = load_opus_results()
    if opus:
        all_results["claude-opus-4-6"] = opus
        print(f"Loaded Opus results: {opus['n_scenarios']} scenarios, "
              f"{opus['n_attack_success']} attacks, {opus['n_refused']} refused")

    # 2. Run remaining models
    for model in MODELS_TO_RUN:
        summary = run_model(model)
        all_results[model] = summary

    # 3. Build comparison
    matrix = build_comparison_matrix(all_results)

    # 4. Save unified output
    output = {
        "meta": {
            "description": "Multi-model dynamic testing comparison for Paper 2",
            "generated_at": datetime.now().isoformat(),
            "total_elapsed_seconds": round(time.time() - t_start, 1),
            "models_tested": list(all_results.keys()),
            "n_scenarios": 16,
        },
        "models": all_results,
        "comparison_matrix": matrix,
    }

    with open(MULTI_MODEL_OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved unified results to {MULTI_MODEL_OUTPUT}")

    # 5. Print report
    print_comparison_report(all_results, matrix)

    elapsed = time.time() - t_start
    print(f"\n\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
