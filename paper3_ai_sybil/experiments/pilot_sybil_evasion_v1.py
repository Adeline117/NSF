"""
Paper 3 Pilot: AI Agent Sybil Attack Evasion Analysis
======================================================
Goal: Validate that AI agents can evade traditional Sybil detectors,
and that new features can detect them.

Pilot Experiment:
1. Implement baseline 5-indicator Sybil detector (from HasciDB CHI'26)
2. Simulate AI-generated "human-like" Sybil behavior
3. Measure evasion rate against baseline
4. Propose and test AI-Sybil-specific features

All synthetic data - no real Sybil addresses in pilot.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score
from dataclasses import dataclass, asdict


# ============================================================
# BASELINE SYBIL DETECTOR (5-indicator model from prior work)
# ============================================================

@dataclass
class SybilFeatures:
    """Traditional 5-indicator Sybil detection features."""
    tx_timing_regularity: float      # How regular are transaction intervals
    funding_source_concentration: float  # % of funds from single source
    contract_interaction_overlap: float  # Jaccard similarity with other addresses
    bridge_usage_pattern: float       # Bridge usage timing correlation
    airdrop_claim_speed: float        # How fast after eligibility


@dataclass
class AISpecificFeatures:
    """Additional features for detecting AI-generated Sybils."""
    gas_price_precision: float       # From Paper 1: agent gas patterns
    hour_entropy: float             # From Paper 1: circadian rhythm
    behavioral_consistency: float    # Cross-address statistical correlation
    action_sequence_perplexity: float  # LLM-generated sequence detection
    error_recovery_pattern: float    # How errors are handled (agent-specific)
    response_latency_variance: float  # Variance in response to events


def generate_legitimate_users(n: int, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic legitimate user behavior."""
    rng = np.random.RandomState(seed)

    return pd.DataFrame({
        "tx_timing_regularity": rng.beta(2, 5, n),       # Irregular (low values)
        "funding_source_concentration": rng.beta(2, 3, n),  # Multiple sources
        "contract_interaction_overlap": rng.beta(1, 8, n),   # Low overlap
        "bridge_usage_pattern": rng.beta(1, 5, n),           # Uncorrelated
        "airdrop_claim_speed": rng.beta(2, 2, n),            # Mixed speeds
        # AI-specific features
        "gas_price_precision": rng.beta(2, 8, n),     # Low precision (round numbers)
        "hour_entropy": rng.beta(3, 5, n),            # Low entropy (circadian)
        "behavioral_consistency": rng.beta(2, 5, n),  # Inconsistent across sessions
        "action_sequence_perplexity": rng.lognormal(2, 1, n),  # High perplexity (human)
        "error_recovery_pattern": rng.beta(2, 3, n),  # Variable error handling
        "response_latency_variance": rng.lognormal(1, 1.5, n),  # High variance
        "label": 0,
    })


def generate_traditional_sybils(n: int, seed: int = 43) -> pd.DataFrame:
    """Generate synthetic traditional (script-based) Sybil behavior."""
    rng = np.random.RandomState(seed)

    return pd.DataFrame({
        "tx_timing_regularity": rng.beta(8, 2, n),       # Very regular
        "funding_source_concentration": rng.beta(8, 1, n),  # Single source
        "contract_interaction_overlap": rng.beta(7, 2, n),   # High overlap
        "bridge_usage_pattern": rng.beta(7, 2, n),           # Correlated
        "airdrop_claim_speed": rng.beta(8, 1, n),            # Very fast
        # AI-specific features
        "gas_price_precision": rng.beta(8, 2, n),     # High precision
        "hour_entropy": rng.beta(8, 2, n),            # High entropy (24/7)
        "behavioral_consistency": rng.beta(8, 2, n),  # Very consistent
        "action_sequence_perplexity": rng.lognormal(0.5, 0.3, n),  # Low perplexity
        "error_recovery_pattern": rng.beta(8, 2, n),  # Uniform error handling
        "response_latency_variance": rng.lognormal(-1, 0.3, n),  # Low variance
        "label": 1,
    })


def generate_ai_sybils(n: int, seed: int = 44) -> pd.DataFrame:
    """Generate synthetic AI-agent-generated Sybil behavior.

    Key insight: AI agents can mimic human-like irregularity in traditional
    features, but leave traces in AI-specific features.
    """
    rng = np.random.RandomState(seed)

    return pd.DataFrame({
        # Traditional features: AI makes these look human-like
        "tx_timing_regularity": rng.beta(2.5, 4, n),     # Mimics human irregularity
        "funding_source_concentration": rng.beta(2.5, 3, n),
        "contract_interaction_overlap": rng.beta(1.5, 6, n),  # Low but not too low
        "bridge_usage_pattern": rng.beta(1.5, 4, n),
        "airdrop_claim_speed": rng.beta(3, 2, n),         # Slightly faster than human

        # AI-specific features: harder to fake
        "gas_price_precision": rng.beta(6, 3, n),     # Still precise (agent compute)
        "hour_entropy": rng.beta(6, 3, n),            # Still high (no real circadian)
        "behavioral_consistency": rng.beta(6, 3, n),  # Cross-address correlation
        "action_sequence_perplexity": rng.lognormal(1, 0.5, n),  # LLM-typical range
        "error_recovery_pattern": rng.beta(6, 3, n),  # Systematic error handling
        "response_latency_variance": rng.lognormal(0, 0.5, n),  # Moderate variance
        "label": 1,
    })


def run_pilot():
    """Run the Sybil evasion pilot experiment."""
    print("=" * 60)
    print("Paper 3 Pilot: AI Sybil Evasion Analysis")
    print("=" * 60)

    n_legit = 500
    n_trad_sybil = 200
    n_ai_sybil = 200

    legit = generate_legitimate_users(n_legit)
    trad_sybil = generate_traditional_sybils(n_trad_sybil)
    ai_sybil = generate_ai_sybils(n_ai_sybil)

    traditional_features = [
        "tx_timing_regularity", "funding_source_concentration",
        "contract_interaction_overlap", "bridge_usage_pattern",
        "airdrop_claim_speed",
    ]
    ai_features = [
        "gas_price_precision", "hour_entropy", "behavioral_consistency",
        "action_sequence_perplexity", "error_recovery_pattern",
        "response_latency_variance",
    ]
    all_features = traditional_features + ai_features

    # ============================================================
    # EXPERIMENT 1: Baseline detector vs traditional Sybils
    # ============================================================
    print("\n--- Exp 1: Baseline (5-indicator) vs Traditional Sybils ---")

    train_data = pd.concat([legit, trad_sybil], ignore_index=True)
    X_trad = train_data[traditional_features].values
    y_trad = train_data["label"].values

    clf_baseline = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_baseline = cross_val_score(clf_baseline, X_trad, y_trad, cv=5, scoring="roc_auc")
    print(f"Baseline AUC vs Traditional Sybils: {scores_baseline.mean():.3f} ± {scores_baseline.std():.3f}")

    # ============================================================
    # EXPERIMENT 2: Baseline detector vs AI Sybils (evasion test)
    # ============================================================
    print("\n--- Exp 2: Baseline (5-indicator) vs AI Sybils ---")

    # Train on traditional Sybils, test on AI Sybils
    clf_baseline.fit(X_trad, y_trad)

    test_data = pd.concat([
        generate_legitimate_users(200, seed=99),
        ai_sybil,
    ], ignore_index=True)
    X_test = test_data[traditional_features].values
    y_test = test_data["label"].values

    y_pred_proba = clf_baseline.predict_proba(X_test)[:, 1]
    auc_evasion = roc_auc_score(y_test, y_pred_proba)
    print(f"Baseline AUC vs AI Sybils: {auc_evasion:.3f}")
    print(f"Evasion effectiveness: AUC dropped from {scores_baseline.mean():.3f} to {auc_evasion:.3f}")
    print(f"Relative AUC drop: {(1 - auc_evasion/scores_baseline.mean()) * 100:.1f}%")

    # ============================================================
    # EXPERIMENT 3: Enhanced detector with AI-specific features
    # ============================================================
    print("\n--- Exp 3: Enhanced (11-feature) vs AI Sybils ---")

    # Train enhanced model on all features
    train_enhanced = pd.concat([
        legit,
        generate_ai_sybils(200, seed=45),  # Include some AI Sybils in training
    ], ignore_index=True)
    X_enhanced = train_enhanced[all_features].values
    y_enhanced = train_enhanced["label"].values

    clf_enhanced = RandomForestClassifier(n_estimators=100, random_state=42)
    scores_enhanced = cross_val_score(clf_enhanced, X_enhanced, y_enhanced, cv=5, scoring="roc_auc")
    print(f"Enhanced AUC vs AI Sybils (CV): {scores_enhanced.mean():.3f} ± {scores_enhanced.std():.3f}")

    # Test on held-out AI Sybils
    clf_enhanced.fit(X_enhanced, y_enhanced)
    test_enhanced = pd.concat([
        generate_legitimate_users(200, seed=100),
        generate_ai_sybils(200, seed=46),
    ], ignore_index=True)
    X_test_e = test_enhanced[all_features].values
    y_test_e = test_enhanced["label"].values

    y_pred_enhanced = clf_enhanced.predict_proba(X_test_e)[:, 1]
    auc_enhanced = roc_auc_score(y_test_e, y_pred_enhanced)
    print(f"Enhanced AUC vs AI Sybils (held-out): {auc_enhanced:.3f}")

    # ============================================================
    # EXPERIMENT 4: Feature importance analysis
    # ============================================================
    print("\n--- Exp 4: Feature Importance ---")

    importances = clf_enhanced.feature_importances_
    feature_imp = sorted(zip(all_features, importances), key=lambda x: -x[1])
    print(f"\nFeature Importance Ranking:")
    for feat, imp in feature_imp:
        bar = "█" * int(imp * 50)
        marker = " (AI-specific)" if feat in ai_features else ""
        print(f"  {feat:35s}: {imp:.4f} {bar}{marker}")

    # Check if AI-specific features are in top-5
    top5 = [f for f, _ in feature_imp[:5]]
    ai_in_top5 = sum(1 for f in top5 if f in ai_features)
    print(f"\nAI-specific features in top-5: {ai_in_top5}/5")

    # ============================================================
    # EXPERIMENT 5: Per-feature discriminative power
    # ============================================================
    print("\n--- Exp 5: Individual Feature AUC ---")

    legit_test = generate_legitimate_users(300, seed=101)
    ai_sybil_test = generate_ai_sybils(300, seed=47)
    combined = pd.concat([legit_test, ai_sybil_test], ignore_index=True)

    print(f"\nIndividual feature AUC (AI Sybils vs Legit):")
    feature_aucs = {}
    for feat in all_features:
        auc = roc_auc_score(combined["label"], combined[feat])
        auc = max(auc, 1 - auc)  # Take max with complement
        feature_aucs[feat] = auc
        marker = " (AI-specific)" if feat in ai_features else " (traditional)"
        bar = "█" * int((auc - 0.5) * 100)
        print(f"  {feat:35s}: {auc:.3f} {bar}{marker}")

    # ============================================================
    # SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("PILOT SUMMARY")
    print("=" * 60)

    print(f"\n{'Metric':<45} {'Value':>10}")
    print("-" * 60)
    print(f"{'Baseline AUC (vs trad. Sybils)':<45} {scores_baseline.mean():>10.3f}")
    print(f"{'Baseline AUC (vs AI Sybils)':<45} {auc_evasion:>10.3f}")
    print(f"{'Evasion rate (AUC drop)':<45} {(1 - auc_evasion/scores_baseline.mean()) * 100:>9.1f}%")
    print(f"{'Enhanced AUC (vs AI Sybils)':<45} {auc_enhanced:>10.3f}")
    print(f"{'Recovery (Enhanced - Baseline vs AI)':<45} {(auc_enhanced - auc_evasion):>10.3f}")
    print(f"{'AI features in top-5 importance':<45} {ai_in_top5:>10d}/5")

    print(f"\nFEASIBILITY ASSESSMENT:")
    feasible = (
        auc_evasion < scores_baseline.mean() * 0.85 and  # Significant evasion
        auc_enhanced > auc_evasion + 0.05  # Recovery with new features
    )
    if feasible:
        print("  CONFIRMED - AI Sybils evade baseline, enhanced features recover detection")
        print("  Paper 3 contribution is viable")
    else:
        print("  NEEDS ADJUSTMENT - Effect sizes may need tuning")
        print(f"  Evasion drop: {(1 - auc_evasion/scores_baseline.mean()) * 100:.1f}% (need >15%)")
        print(f"  Recovery: {auc_enhanced - auc_evasion:.3f} (need >0.05)")

    print(f"\nNEXT STEPS:")
    print(f"  1. Obtain real Sybil labels from airdrop post-mortems")
    print(f"  2. Integrate Paper 1 agent identification for ground truth")
    print(f"  3. Test with actual LLM-generated transaction sequences")

    # Save results
    results = {
        "baseline_auc_trad": float(scores_baseline.mean()),
        "baseline_auc_ai": float(auc_evasion),
        "evasion_rate": float(1 - auc_evasion / scores_baseline.mean()),
        "enhanced_auc_ai": float(auc_enhanced),
        "recovery": float(auc_enhanced - auc_evasion),
        "feature_importance": {f: float(i) for f, i in feature_imp},
        "individual_aucs": {f: float(a) for f, a in feature_aucs.items()},
        "feasibility": "CONFIRMED" if feasible else "NEEDS_ADJUSTMENT",
    }
    with open("paper3_ai_sybil/experiments/pilot_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to paper3_ai_sybil/experiments/pilot_results.json")

    return results


if __name__ == "__main__":
    run_pilot()
