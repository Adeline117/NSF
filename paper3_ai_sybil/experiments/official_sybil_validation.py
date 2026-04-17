#!/usr/bin/env python3
"""
Validate HasciDB sybil detection against official project sybil lists.

For each project in data/HasciDB/data/official_sybils/:
  - If real addresses exist: cross-reference with HasciDB is_sybil flag
  - If N/A / NO_OFFICIAL_SYBIL_LIST: note no ground truth available

Saves results to experiments/official_sybil_validation_results.json
"""

import json
import os
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OFFICIAL_DIR = BASE / "data" / "HasciDB" / "data" / "official_sybils"
RESULTS_DIR = BASE / "data" / "HasciDB" / "data" / "sybil_results"
OUTPUT = Path(__file__).resolve().parent / "official_sybil_validation_results.json"

NO_LIST_LABELS = {"NO_OFFICIAL_SYBIL_LIST", "N/A"}


def project_name(csv_name: str) -> str:
    """Extract project key from official sybil filename."""
    return csv_name.replace("_official_sybils.csv", "")


def load_official(path: Path):
    """Load an official sybil CSV. Returns (has_real_list, df, note)."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    # Check if first row address is N/A => no real list
    first_addr = str(df.iloc[0]["address"]).strip() if len(df) > 0 else "N/A"
    first_label = str(df.iloc[0]["label"]).strip() if len(df) > 0 else ""
    if first_addr == "N/A" or first_label in NO_LIST_LABELS:
        note = str(df.iloc[0].get("note", ""))
        return False, None, note
    # Real list — normalise addresses to lowercase
    df["address"] = df["address"].str.strip().str.lower()
    return True, df, None


def find_hascidb_file(project: str) -> Path | None:
    """Find the matching chi26_v3 results file."""
    candidate = RESULTS_DIR / f"{project}_chi26_v3.csv"
    if candidate.exists():
        return candidate
    return None


def validate_project(project: str, official_path: Path) -> dict:
    """Run validation for one project."""
    has_list, off_df, note = load_official(official_path)

    result = {"project": project, "official_csv": official_path.name}

    if not has_list:
        result["has_official_list"] = False
        result["status"] = "no ground truth available"
        result["note"] = note if note else ""
        return result

    result["has_official_list"] = True
    official_addrs = set(off_df["address"])
    result["official_sybil_count"] = len(official_addrs)

    # Head-3 preview
    result["official_head3"] = off_df.head(3)[["address", "label"]].to_dict(orient="records")

    # Load HasciDB results
    hasci_path = find_hascidb_file(project)
    if hasci_path is None:
        result["status"] = "no HasciDB results file found"
        return result

    hdf = pd.read_csv(hasci_path, dtype={"address": str})
    hdf["address"] = hdf["address"].str.strip().str.lower()
    hasci_sybils = set(hdf.loc[hdf["is_sybil"] == 1, "address"])
    hasci_all = set(hdf["address"])

    result["hascidb_total_addresses"] = len(hasci_all)
    result["hascidb_sybil_count"] = len(hasci_sybils)

    # --- Agreement metrics ---
    # Official addresses that appear in HasciDB dataset at all
    overlap_universe = official_addrs & hasci_all
    result["official_in_hascidb_universe"] = len(overlap_universe)

    # Of those, how many does HasciDB also flag?
    agreed_sybils = official_addrs & hasci_sybils
    result["agreed_sybils"] = len(agreed_sybils)

    if len(overlap_universe) > 0:
        recall = len(agreed_sybils) / len(overlap_universe)
    else:
        recall = None
    result["recall_on_overlapping"] = round(recall, 6) if recall is not None else None

    # Overall recall against full official list
    overall_recall = len(agreed_sybils) / len(official_addrs) if len(official_addrs) > 0 else None
    result["recall_overall"] = round(overall_recall, 6) if overall_recall is not None else None

    # HasciDB sybils NOT in official list (false positives relative to official)
    hasci_only = hasci_sybils - official_addrs
    result["hascidb_only_sybils"] = len(hasci_only)
    if len(hasci_sybils) > 0:
        fp_rate = len(hasci_only) / len(hasci_sybils)
    else:
        fp_rate = None
    result["hascidb_sybil_pct_not_in_official"] = round(fp_rate, 6) if fp_rate is not None else None

    # Official sybils missed by HasciDB
    missed = overlap_universe - hasci_sybils
    result["official_missed_by_hascidb"] = len(missed)

    result["status"] = "validated"
    return result


def main():
    csvs = sorted(OFFICIAL_DIR.glob("*.csv"))
    print(f"Found {len(csvs)} official sybil CSVs\n")

    results = []
    summary = {"total_projects": len(csvs), "with_official_list": 0, "without_official_list": 0}

    for csv_path in csvs:
        proj = project_name(csv_path.name)
        print(f"--- {proj} ---")

        # Show head-3
        with open(csv_path) as f:
            lines = [f.readline().rstrip() for _ in range(3)]
        for l in lines:
            print(f"  {l}")

        r = validate_project(proj, csv_path)
        results.append(r)

        if r["has_official_list"]:
            summary["with_official_list"] += 1
            print(f"  Official sybils:        {r['official_sybil_count']}")
            print(f"  In HasciDB universe:    {r.get('official_in_hascidb_universe', 'N/A')}")
            print(f"  Agreed (both flag):     {r.get('agreed_sybils', 'N/A')}")
            print(f"  Recall (overlapping):   {r.get('recall_on_overlapping', 'N/A')}")
            print(f"  Recall (overall):       {r.get('recall_overall', 'N/A')}")
            print(f"  HasciDB-only sybils:    {r.get('hascidb_only_sybils', 'N/A')}")
            print(f"  HasciDB sybil % not in official: {r.get('hascidb_sybil_pct_not_in_official', 'N/A')}")
        else:
            summary["without_official_list"] += 1
            print(f"  => {r['status']}")
        print()

    output = {"summary": summary, "projects": results}
    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {OUTPUT}")


if __name__ == "__main__":
    main()
