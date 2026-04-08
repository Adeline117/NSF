"""
Paper 3 — Fix Circular Reasoning with Independent-Label Experiments
====================================================================
The original experiment trains GBM on {BT, BW, HF, RF, MA} to predict
is_sybil, but is_sybil is a deterministic function of those same
indicators (ops_flag OR fund_flag). This produces trivially AUC=1.0.

This script replaces that circular setup with THREE independent-label
approaches:

Approach A: Cross-Axis Prediction
  - Train on OPS indicators (BT, BW, HF) to predict FUND flag
  - Train on FUND indicators (RF, MA) to predict OPS flag
  - Tests whether the two indicator axes are correlated

Approach B: Cross-Project Transfer (truly independent labels)
  - Train on project A, test on project B
  - The test project's labels were never seen during training
  - Measures how well sybil patterns transfer across airdrops

Approach C: Independent Ground Truth (Gitcoin FDD/SAD)
  - Use Gitcoin's separately curated sybil list as INDEPENDENT labels
  - Compare HasciDB labels vs Gitcoin FDD labels
  - Train HasciDB model, evaluate on FDD labels (and vice versa)

Data: paper3_ai_sybil/data/HasciDB/data/sybil_results/{project}_chi26_v3.csv
      paper3_ai_sybil/data/HasciDB/data/official_sybils/gitcoin_official_sybils.csv

Usage:
    python3 paper3_ai_sybil/experiments/experiment_independent_labels.py
"""

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ============================================================
# PATHS
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
HASCIDB_DIR = (
    PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "sybil_results"
)
OFFICIAL_DIR = (
    PROJECT_ROOT / "paper3_ai_sybil" / "data" / "HasciDB" / "data" / "official_sybils"
)
OUTPUT_FILE = SCRIPT_DIR / "experiment_independent_labels_results.json"

# ============================================================
# CONSTANTS
# ============================================================

INDICATOR_COLS = ["BT", "BW", "HF", "RF", "MA"]
OPS_COLS = ["BT", "BW", "HF"]
FUND_COLS = ["RF", "MA"]

THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}

PROJECTS = [
    "1inch", "uniswap", "looksrare", "apecoin", "ens", "gitcoin",
    "etherfi", "x2y2", "dydx", "badger", "blur_s1", "blur_s2",
    "paraswap", "ampleforth", "eigenlayer", "pengu",
]

# Maximum rows per project for speed
MAX_ROWS = 80_000

# GBM hyperparams
GBM_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.1,
    subsample=0.8,
    random_state=42,
)


# ============================================================
# DATA LOADING
# ============================================================


def load_project(project: str, max_rows: int = MAX_ROWS) -> pd.DataFrame:
    """Load a HasciDB CSV with stratified sampling if needed."""
    csv_path = HASCIDB_DIR / f"{project}_chi26_v3.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    df = pd.read_csv(csv_path)

    # Normalize columns
    for col in INDICATOR_COLS:
        if col not in df.columns:
            for alt in [col.lower(), f"{col.lower()}_score"]:
                if alt in df.columns:
                    df[col] = df[alt]
                    break
            else:
                df[col] = 0

    # Ensure derived columns exist
    if "ops_flag" not in df.columns:
        df["ops_flag"] = (
            (df["BT"] >= THRESHOLDS["BT"]) |
            (df["BW"] >= THRESHOLDS["BW"]) |
            (df["HF"] >= THRESHOLDS["HF"])
        ).astype(int)
    if "fund_flag" not in df.columns:
        df["fund_flag"] = (
            (df["RF"] >= THRESHOLDS["RF"]) |
            (df["MA"] >= THRESHOLDS["MA"])
        ).astype(int)
    if "is_sybil" not in df.columns:
        df["is_sybil"] = ((df["ops_flag"] == 1) | (df["fund_flag"] == 1)).astype(int)

    # Stratified sample if too large
    if len(df) > max_rows:
        sybils = df[df["is_sybil"] == 1]
        non_sybils = df[df["is_sybil"] == 0]
        rate = len(sybils) / len(df)
        n_s = min(int(max_rows * rate), len(sybils))
        n_ns = min(max_rows - n_s, len(non_sybils))
        df = pd.concat([
            sybils.sample(n=n_s, random_state=42),
            non_sybils.sample(n=n_ns, random_state=42),
        ], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    df["project"] = project
    return df


def load_gitcoin_fdd_labels() -> set:
    """Load Gitcoin FDD/SAD official sybil addresses as independent labels."""
    fdd_path = OFFICIAL_DIR / "gitcoin_official_sybils.csv"
    if not fdd_path.exists():
        return set()

    df = pd.read_csv(fdd_path)
    # Filter to actual sybil addresses (not N/A rows)
    sybils = df[df["label"] == "sybil"]
    return set(sybils["address"].str.lower())


# ============================================================
# EVALUATION HELPERS
# ============================================================


def evaluate_model(y_true, y_prob, y_pred=None) -> dict:
    """Compute standard classification metrics."""
    if y_pred is None:
        y_pred = (y_prob >= 0.5).astype(int)

    metrics = {}
    try:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        metrics["auc_roc"] = float("nan")
    try:
        metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        metrics["avg_precision"] = float("nan")
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics["tn"] = int(cm[0, 0])
    metrics["fp"] = int(cm[0, 1])
    metrics["fn"] = int(cm[1, 0])
    metrics["tp"] = int(cm[1, 1])

    return metrics


# ============================================================
# APPROACH A: CROSS-AXIS PREDICTION
# ============================================================


def experiment_cross_axis(projects_data: dict[str, pd.DataFrame]) -> dict:
    """Train on OPS indicators to predict FUND flag, and vice versa.

    This is NOT circular because:
    - OPS indicators (BT, BW, HF) are independent features from FUND indicators (RF, MA)
    - The label (fund_flag / ops_flag) is NOT a function of the input features
    - If AUC > 0.5, it means the two sybil axes are correlated
    """
    print("\n" + "=" * 80)
    print("APPROACH A: CROSS-AXIS PREDICTION")
    print("=" * 80)

    results = {}

    for project_name, df in projects_data.items():
        if len(df) < 100:
            continue

        project_results = {"n_total": len(df)}

        # A1: OPS features → predict fund_flag
        X_ops = df[OPS_COLS].values
        y_fund = df["fund_flag"].values.astype(int)
        n_fund_pos = y_fund.sum()
        project_results["fund_flag_rate"] = float(y_fund.mean())

        if n_fund_pos >= 10 and (len(y_fund) - n_fund_pos) >= 10:
            clf = GradientBoostingClassifier(**GBM_PARAMS)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_aucs = []
            for train_idx, test_idx in skf.split(X_ops, y_fund):
                clf.fit(X_ops[train_idx], y_fund[train_idx])
                probs = clf.predict_proba(X_ops[test_idx])[:, 1]
                fold_aucs.append(roc_auc_score(y_fund[test_idx], probs))
            project_results["ops_to_fund"] = {
                "features": OPS_COLS,
                "target": "fund_flag",
                "cv_aucs": [float(a) for a in fold_aucs],
                "mean_auc": float(np.mean(fold_aucs)),
                "std_auc": float(np.std(fold_aucs)),
            }
            print(f"  {project_name}: OPS→FUND AUC={np.mean(fold_aucs):.4f} (+/-{np.std(fold_aucs):.4f})")
        else:
            project_results["ops_to_fund"] = {"skipped": True, "reason": "insufficient fund_flag positives"}
            print(f"  {project_name}: OPS→FUND skipped (insufficient positives: {n_fund_pos})")

        # A2: FUND features → predict ops_flag
        X_fund = df[FUND_COLS].values
        y_ops = df["ops_flag"].values.astype(int)
        n_ops_pos = y_ops.sum()
        project_results["ops_flag_rate"] = float(y_ops.mean())

        if n_ops_pos >= 10 and (len(y_ops) - n_ops_pos) >= 10:
            clf = GradientBoostingClassifier(**GBM_PARAMS)
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_aucs = []
            for train_idx, test_idx in skf.split(X_fund, y_ops):
                clf.fit(X_fund[train_idx], y_ops[train_idx])
                probs = clf.predict_proba(X_fund[test_idx])[:, 1]
                fold_aucs.append(roc_auc_score(y_ops[test_idx], probs))
            project_results["fund_to_ops"] = {
                "features": FUND_COLS,
                "target": "ops_flag",
                "cv_aucs": [float(a) for a in fold_aucs],
                "mean_auc": float(np.mean(fold_aucs)),
                "std_auc": float(np.std(fold_aucs)),
            }
            print(f"  {project_name}: FUND→OPS AUC={np.mean(fold_aucs):.4f} (+/-{np.std(fold_aucs):.4f})")
        else:
            project_results["fund_to_ops"] = {"skipped": True, "reason": "insufficient ops_flag positives"}
            print(f"  {project_name}: FUND→OPS skipped (insufficient positives: {n_ops_pos})")

        results[project_name] = project_results

    return results


# ============================================================
# APPROACH B: CROSS-PROJECT TRANSFER
# ============================================================


def experiment_cross_project_transfer(projects_data: dict[str, pd.DataFrame]) -> dict:
    """Train on one project, test on another.

    This is NOT circular because:
    - The test project's labels were never seen during training
    - Different projects have different sybil populations
    - AUC < 1.0 expected due to distribution shift
    """
    print("\n" + "=" * 80)
    print("APPROACH B: CROSS-PROJECT TRANSFER")
    print("=" * 80)

    results = {}

    # Select projects with sufficient sybils for training
    trainable = {}
    for name, df in projects_data.items():
        n_sybil = df["is_sybil"].sum()
        n_nonsybil = len(df) - n_sybil
        if n_sybil >= 100 and n_nonsybil >= 100:
            trainable[name] = df

    project_names = sorted(trainable.keys())
    print(f"  Trainable projects: {len(project_names)}")

    # Pairwise transfer matrix
    transfer_matrix = {}

    for train_proj in project_names:
        train_df = trainable[train_proj]
        X_train = train_df[INDICATOR_COLS].values
        y_train = train_df["is_sybil"].values.astype(int)

        clf = GradientBoostingClassifier(**GBM_PARAMS)
        clf.fit(X_train, y_train)

        row = {}
        for test_proj in project_names:
            if test_proj == train_proj:
                # Within-project CV for reference
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                cv_aucs = []
                for tr_idx, te_idx in skf.split(X_train, y_train):
                    clf_cv = GradientBoostingClassifier(**GBM_PARAMS)
                    clf_cv.fit(X_train[tr_idx], y_train[tr_idx])
                    probs = clf_cv.predict_proba(X_train[te_idx])[:, 1]
                    try:
                        cv_aucs.append(roc_auc_score(y_train[te_idx], probs))
                    except ValueError:
                        pass
                row[test_proj] = {
                    "type": "within_project_cv",
                    "mean_auc": float(np.mean(cv_aucs)) if cv_aucs else float("nan"),
                    "note": "CIRCULAR (same project, same label definition)"
                }
                continue

            test_df = trainable[test_proj]
            X_test = test_df[INDICATOR_COLS].values
            y_test = test_df["is_sybil"].values.astype(int)

            probs = clf.predict_proba(X_test)[:, 1]
            try:
                auc = roc_auc_score(y_test, probs)
            except ValueError:
                auc = float("nan")

            preds = (probs >= 0.5).astype(int)
            metrics = evaluate_model(y_test, probs, preds)
            row[test_proj] = {
                "type": "cross_project",
                "auc_roc": metrics["auc_roc"],
                "avg_precision": metrics["avg_precision"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "n_test": len(test_df),
                "n_test_sybil": int(y_test.sum()),
            }

        transfer_matrix[train_proj] = row

    # Print summary matrix
    print("\n  Cross-project AUC matrix (rows=train, cols=test):")
    header = f"{'':>14}" + "".join(f"{p[:8]:>10}" for p in project_names)
    print(f"  {header}")
    for train_p in project_names:
        vals = []
        for test_p in project_names:
            entry = transfer_matrix[train_p][test_p]
            auc = entry.get("auc_roc", entry.get("mean_auc", float("nan")))
            vals.append(f"{auc:>10.4f}")
        print(f"  {train_p[:14]:>14}" + "".join(vals))

    # Aggregate cross-project stats (excluding self)
    cross_aucs = []
    for train_p in project_names:
        for test_p in project_names:
            if train_p != test_p:
                entry = transfer_matrix[train_p][test_p]
                auc = entry.get("auc_roc", float("nan"))
                if not np.isnan(auc):
                    cross_aucs.append(auc)

    results["transfer_matrix"] = transfer_matrix
    results["summary"] = {
        "n_projects": len(project_names),
        "n_cross_pairs": len(cross_aucs),
        "cross_project_mean_auc": float(np.mean(cross_aucs)) if cross_aucs else float("nan"),
        "cross_project_std_auc": float(np.std(cross_aucs)) if cross_aucs else float("nan"),
        "cross_project_min_auc": float(np.min(cross_aucs)) if cross_aucs else float("nan"),
        "cross_project_max_auc": float(np.max(cross_aucs)) if cross_aucs else float("nan"),
    }

    print(f"\n  Cross-project AUC: {results['summary']['cross_project_mean_auc']:.4f} "
          f"+/- {results['summary']['cross_project_std_auc']:.4f}")
    print(f"  Range: [{results['summary']['cross_project_min_auc']:.4f}, "
          f"{results['summary']['cross_project_max_auc']:.4f}]")

    return results


# ============================================================
# APPROACH C: INDEPENDENT GROUND TRUTH (Gitcoin FDD)
# ============================================================


def experiment_independent_ground_truth(projects_data: dict[str, pd.DataFrame]) -> dict:
    """Use Gitcoin FDD/SAD independently curated sybil list as ground truth.

    This is NOT circular because:
    - Gitcoin FDD used a DIFFERENT methodology (SAD model, not HasciDB indicators)
    - The FDD labels are independent of BT/BW/HF/RF/MA
    - We evaluate HasciDB indicators' ability to detect FDD-identified sybils
    """
    print("\n" + "=" * 80)
    print("APPROACH C: INDEPENDENT GROUND TRUTH (Gitcoin FDD/SAD)")
    print("=" * 80)

    fdd_sybils = load_gitcoin_fdd_labels()
    if not fdd_sybils:
        print("  WARNING: No Gitcoin FDD sybil list found. Skipping Approach C.")
        return {"skipped": True, "reason": "No Gitcoin FDD sybil list"}

    print(f"  Loaded {len(fdd_sybils)} Gitcoin FDD/SAD sybil addresses")

    # Load Gitcoin HasciDB data
    if "gitcoin" not in projects_data:
        print("  WARNING: Gitcoin project data not loaded. Skipping.")
        return {"skipped": True, "reason": "Gitcoin HasciDB data not loaded"}

    gitcoin_df = projects_data["gitcoin"].copy()
    gitcoin_df["address_lower"] = gitcoin_df["address"].str.lower()

    # Create independent FDD label
    gitcoin_df["fdd_sybil"] = gitcoin_df["address_lower"].isin(fdd_sybils).astype(int)

    n_total = len(gitcoin_df)
    n_hasci_sybil = gitcoin_df["is_sybil"].sum()
    n_fdd_sybil = gitcoin_df["fdd_sybil"].sum()

    print(f"  Gitcoin addresses: {n_total}")
    print(f"  HasciDB sybils: {n_hasci_sybil} ({n_hasci_sybil/n_total*100:.1f}%)")
    print(f"  FDD sybils (matched): {n_fdd_sybil} ({n_fdd_sybil/n_total*100:.1f}%)")

    results = {
        "n_total": n_total,
        "n_hasci_sybil": int(n_hasci_sybil),
        "n_fdd_sybil": int(n_fdd_sybil),
        "hasci_sybil_rate": float(n_hasci_sybil / n_total),
        "fdd_sybil_rate": float(n_fdd_sybil / n_total),
    }

    # ----------------------------------------------------------
    # C1: Agreement analysis (Jaccard, overlap)
    # ----------------------------------------------------------
    hasci_set = set(gitcoin_df[gitcoin_df["is_sybil"] == 1]["address_lower"])
    fdd_set = set(gitcoin_df[gitcoin_df["fdd_sybil"] == 1]["address_lower"])

    intersection = hasci_set & fdd_set
    union = hasci_set | fdd_set

    results["agreement"] = {
        "jaccard": float(len(intersection) / len(union)) if union else 0.0,
        "overlap_coeff": float(len(intersection) / min(len(hasci_set), len(fdd_set))) if min(len(hasci_set), len(fdd_set)) > 0 else 0.0,
        "hasci_only": int(len(hasci_set - fdd_set)),
        "fdd_only": int(len(fdd_set - hasci_set)),
        "both": int(len(intersection)),
        "neither": int(n_total - len(union)),
    }

    print(f"\n  Agreement:")
    print(f"    Jaccard: {results['agreement']['jaccard']:.4f}")
    print(f"    Overlap coefficient: {results['agreement']['overlap_coeff']:.4f}")
    print(f"    Both methods flag: {len(intersection)}")
    print(f"    HasciDB only: {len(hasci_set - fdd_set)}")
    print(f"    FDD only: {len(fdd_set - hasci_set)}")

    # ----------------------------------------------------------
    # C2: HasciDB indicators as features → predict FDD labels
    #     (NOT circular: FDD labels are from a different methodology)
    # ----------------------------------------------------------
    if n_fdd_sybil >= 20 and (n_total - n_fdd_sybil) >= 20:
        X = gitcoin_df[INDICATOR_COLS].values
        y_fdd = gitcoin_df["fdd_sybil"].values.astype(int)

        clf = GradientBoostingClassifier(**GBM_PARAMS)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []

        for train_idx, test_idx in skf.split(X, y_fdd):
            clf.fit(X[train_idx], y_fdd[train_idx])
            probs = clf.predict_proba(X[test_idx])[:, 1]
            preds = (probs >= 0.5).astype(int)
            fold_metrics.append(evaluate_model(y_fdd[test_idx], probs, preds))

        results["hasci_features_predict_fdd"] = {
            "features": INDICATOR_COLS,
            "target": "fdd_sybil (independent label)",
            "cv_folds": fold_metrics,
            "mean_auc": float(np.mean([m["auc_roc"] for m in fold_metrics])),
            "std_auc": float(np.std([m["auc_roc"] for m in fold_metrics])),
            "mean_f1": float(np.mean([m["f1"] for m in fold_metrics])),
            "note": "NOT circular: FDD labels are independently derived",
        }
        print(f"\n  HasciDB indicators → FDD labels: "
              f"AUC={results['hasci_features_predict_fdd']['mean_auc']:.4f} "
              f"F1={results['hasci_features_predict_fdd']['mean_f1']:.4f}")
    else:
        results["hasci_features_predict_fdd"] = {
            "skipped": True,
            "reason": f"Insufficient FDD sybils in HasciDB data ({n_fdd_sybil})",
        }
        print(f"  HasciDB → FDD: skipped (only {n_fdd_sybil} FDD sybils matched)")

    # ----------------------------------------------------------
    # C3: Cross-method transfer: train on other projects, test on
    #     Gitcoin with FDD labels
    # ----------------------------------------------------------
    if n_fdd_sybil >= 20:
        # Train on blur_s2 (largest project), test on Gitcoin FDD labels
        train_candidates = ["blur_s2", "uniswap", "eigenlayer", "1inch"]
        cross_method_results = {}

        for train_proj in train_candidates:
            if train_proj not in projects_data:
                continue
            train_df = projects_data[train_proj]
            if train_df["is_sybil"].sum() < 50:
                continue

            X_train = train_df[INDICATOR_COLS].values
            y_train = train_df["is_sybil"].values.astype(int)

            X_test = gitcoin_df[INDICATOR_COLS].values
            y_test_fdd = gitcoin_df["fdd_sybil"].values.astype(int)

            clf = GradientBoostingClassifier(**GBM_PARAMS)
            clf.fit(X_train, y_train)

            probs = clf.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.5).astype(int)
            metrics = evaluate_model(y_test_fdd, probs, preds)

            cross_method_results[train_proj] = {
                "train_project": train_proj,
                "train_label": "HasciDB is_sybil",
                "test_label": "Gitcoin FDD sybil",
                "metrics": metrics,
                "note": "DOUBLY independent: different project + different label methodology",
            }

            print(f"  Train {train_proj} (HasciDB) → Test Gitcoin (FDD): "
                  f"AUC={metrics['auc_roc']:.4f} F1={metrics['f1']:.4f}")

        results["cross_method_transfer"] = cross_method_results

    # ----------------------------------------------------------
    # C4: Per-indicator analysis against FDD labels
    # ----------------------------------------------------------
    if n_fdd_sybil >= 20:
        per_indicator = {}
        for ind in INDICATOR_COLS:
            threshold = THRESHOLDS[ind]
            triggered = gitcoin_df[ind] >= threshold
            y_fdd = gitcoin_df["fdd_sybil"].values.astype(int)

            tp = int((triggered & (y_fdd == 1)).sum())
            fp = int((triggered & (y_fdd == 0)).sum())
            fn = int((~triggered & (y_fdd == 1)).sum())
            tn = int((~triggered & (y_fdd == 0)).sum())

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0

            per_indicator[ind] = {
                "threshold": threshold,
                "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                "precision_vs_fdd": float(prec),
                "recall_vs_fdd": float(rec),
                "f1_vs_fdd": float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0,
            }
            print(f"  {ind} >= {threshold} vs FDD: prec={prec:.4f} rec={rec:.4f}")

        results["per_indicator_vs_fdd"] = per_indicator

    return results


# ============================================================
# APPROACH D: SERIAL SYBIL CROSS-PROJECT
# ============================================================


def experiment_serial_sybil(projects_data: dict[str, pd.DataFrame]) -> dict:
    """Find addresses that appear in multiple projects and test consistency.

    If an address is labeled sybil in project A, is it also sybil in project B?
    This tests label robustness using cross-project appearance as natural holdout.
    """
    print("\n" + "=" * 80)
    print("APPROACH D: SERIAL SYBIL ANALYSIS (Cross-Project Consistency)")
    print("=" * 80)

    # Build address→project appearances
    address_projects = {}
    for proj_name, df in projects_data.items():
        for _, row in df.iterrows():
            addr = str(row.get("address", "")).lower()
            if not addr or addr == "nan":
                continue
            if addr not in address_projects:
                address_projects[addr] = {}
            address_projects[addr][proj_name] = int(row["is_sybil"])

    # Filter to addresses appearing in 2+ projects
    multi_project = {addr: projs for addr, projs in address_projects.items()
                     if len(projs) >= 2}

    print(f"  Total unique addresses: {len(address_projects):,}")
    print(f"  Addresses in 2+ projects: {len(multi_project):,}")

    if len(multi_project) < 10:
        print("  Not enough multi-project addresses for analysis.")
        return {"skipped": True, "n_multi_project": len(multi_project)}

    # Consistency analysis
    consistent = 0
    inconsistent = 0
    always_sybil = 0
    always_clean = 0
    mixed = 0

    for addr, projs in multi_project.items():
        labels = list(projs.values())
        if all(l == 1 for l in labels):
            always_sybil += 1
            consistent += 1
        elif all(l == 0 for l in labels):
            always_clean += 1
            consistent += 1
        else:
            mixed += 1
            inconsistent += 1

    total = len(multi_project)
    results = {
        "n_multi_project": total,
        "consistency_rate": float(consistent / total) if total > 0 else 0,
        "always_sybil": {"count": always_sybil, "pct": float(always_sybil / total * 100)},
        "always_clean": {"count": always_clean, "pct": float(always_clean / total * 100)},
        "mixed": {"count": mixed, "pct": float(mixed / total * 100)},
    }

    print(f"\n  Consistency rate: {results['consistency_rate']*100:.1f}%")
    print(f"  Always sybil: {always_sybil} ({results['always_sybil']['pct']:.1f}%)")
    print(f"  Always clean: {always_clean} ({results['always_clean']['pct']:.1f}%)")
    print(f"  Mixed (inconsistent): {mixed} ({results['mixed']['pct']:.1f}%)")

    # For mixed addresses, analyze which project pairs disagree most
    if mixed > 0:
        pair_disagreements = {}
        for addr, projs in multi_project.items():
            labels = list(projs.values())
            if not (all(l == 1 for l in labels) or all(l == 0 for l in labels)):
                proj_names = list(projs.keys())
                for i in range(len(proj_names)):
                    for j in range(i + 1, len(proj_names)):
                        p1, p2 = sorted([proj_names[i], proj_names[j]])
                        pair = f"{p1}|{p2}"
                        if pair not in pair_disagreements:
                            pair_disagreements[pair] = {"agree": 0, "disagree": 0}
                        if projs[proj_names[i]] != projs[proj_names[j]]:
                            pair_disagreements[pair]["disagree"] += 1
                        else:
                            pair_disagreements[pair]["agree"] += 1

        # Top disagreeing pairs
        sorted_pairs = sorted(
            pair_disagreements.items(),
            key=lambda x: x[1]["disagree"],
            reverse=True,
        )[:10]

        results["top_disagreeing_pairs"] = {
            pair: {
                "agree": data["agree"],
                "disagree": data["disagree"],
                "disagree_rate": float(data["disagree"] / (data["agree"] + data["disagree"]))
                if (data["agree"] + data["disagree"]) > 0 else 0,
            }
            for pair, data in sorted_pairs
        }

        print("\n  Top disagreeing project pairs:")
        for pair, data in sorted_pairs[:5]:
            total_p = data["agree"] + data["disagree"]
            rate = data["disagree"] / total_p if total_p > 0 else 0
            print(f"    {pair}: {data['disagree']}/{total_p} disagree ({rate*100:.1f}%)")

    # Cross-project prediction using serial sybils
    # Train on project A's labels for shared addresses, test on project B's labels
    if mixed >= 20:
        project_names = sorted(projects_data.keys())
        cross_pred_results = {}

        for train_proj in ["blur_s2", "uniswap", "1inch"]:
            if train_proj not in projects_data:
                continue
            for test_proj in ["eigenlayer", "gitcoin", "ens"]:
                if test_proj not in projects_data or test_proj == train_proj:
                    continue

                # Find shared addresses
                train_addrs = set(projects_data[train_proj]["address"].str.lower())
                test_addrs = set(projects_data[test_proj]["address"].str.lower())
                shared = train_addrs & test_addrs

                if len(shared) < 20:
                    continue

                train_df = projects_data[train_proj]
                train_df_shared = train_df[train_df["address"].str.lower().isin(shared)]
                test_df = projects_data[test_proj]
                test_df_shared = test_df[test_df["address"].str.lower().isin(shared)]

                if train_df_shared["is_sybil"].nunique() < 2 or test_df_shared["is_sybil"].nunique() < 2:
                    continue

                X_train = train_df_shared[INDICATOR_COLS].values
                y_train = train_df_shared["is_sybil"].values

                # Test features come from test project, test labels from test project
                X_test = test_df_shared[INDICATOR_COLS].values
                y_test = test_df_shared["is_sybil"].values

                clf = GradientBoostingClassifier(**GBM_PARAMS)
                clf.fit(X_train, y_train)
                probs = clf.predict_proba(X_test)[:, 1]

                try:
                    auc = roc_auc_score(y_test, probs)
                except ValueError:
                    auc = float("nan")

                key = f"{train_proj}→{test_proj}"
                cross_pred_results[key] = {
                    "n_shared": len(shared),
                    "auc": float(auc),
                }
                print(f"  Serial transfer {key}: {len(shared)} shared, AUC={auc:.4f}")

        results["serial_cross_prediction"] = cross_pred_results

    return results


# ============================================================
# MAIN
# ============================================================


def main():
    start_time = time.time()

    print("=" * 80)
    print("EXPERIMENT: INDEPENDENT-LABEL SYBIL DETECTION")
    print("Fixes circular reasoning in Paper 3 baseline")
    print("=" * 80)

    # Load all projects
    print("\nLoading HasciDB data...")
    projects_data = {}
    for proj in PROJECTS:
        print(f"  {proj}...", end=" ")
        df = load_project(proj)
        if df.empty:
            print("SKIPPED")
            continue
        projects_data[proj] = df
        n_s = df["is_sybil"].sum()
        print(f"OK ({len(df):,} rows, {n_s:,} sybils, {n_s/len(df)*100:.1f}%)")

    print(f"\nLoaded {len(projects_data)} projects")

    # Run experiments
    all_results = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "projects_loaded": list(projects_data.keys()),
            "max_rows_per_project": MAX_ROWS,
            "indicator_cols": INDICATOR_COLS,
            "thresholds": THRESHOLDS,
            "gbm_params": GBM_PARAMS,
        }
    }

    # Approach A: Cross-axis
    all_results["approach_a_cross_axis"] = experiment_cross_axis(projects_data)

    # Approach B: Cross-project transfer
    all_results["approach_b_cross_project"] = experiment_cross_project_transfer(projects_data)

    # Approach C: Independent ground truth
    all_results["approach_c_independent_gt"] = experiment_independent_ground_truth(projects_data)

    # Approach D: Serial sybil consistency
    all_results["approach_d_serial_sybil"] = experiment_serial_sybil(projects_data)

    elapsed = time.time() - start_time
    all_results["metadata"]["elapsed_seconds"] = round(elapsed, 1)

    # Save results
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'=' * 80}")
    print(f"Results saved to {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed:.1f}s")

    # Summary
    print(f"\n{'=' * 80}")
    print("SUMMARY: Why these experiments are NOT circular")
    print("=" * 80)
    print("""
  OLD (circular):
    Train GBM on {BT,BW,HF,RF,MA} → predict is_sybil
    is_sybil = f(BT,BW,HF,RF,MA)  ← DETERMINISTIC ← AUC=1.0 trivially

  NEW (independent):
    A) Cross-Axis: Train on {BT,BW,HF} → predict fund_flag (derived from {RF,MA})
       Input features are DISJOINT from label derivation

    B) Cross-Project: Train on blur_s2 → test on uniswap
       Test labels never seen during training

    C) Independent GT: Train on HasciDB indicators → predict Gitcoin FDD labels
       FDD used a COMPLETELY DIFFERENT methodology (SAD model, not HasciDB)

    D) Serial Sybils: Same address across projects
       Cross-project appearance provides natural holdout
""")


if __name__ == "__main__":
    main()
