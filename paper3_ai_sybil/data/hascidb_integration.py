"""
HasciDB Data Integration for Paper 3
=====================================
Utilities for fetching and processing real Sybil data from HasciDB.

Data Sources:
- HasciDB REST API: hascidb.org/v1/*
- HasciDB SQLite: UW-Decentralized-Computing-Lab/HasciDB (data/hascidb.sqlite, ~1.4GB)
- HasciDB CSVs: data/sybil_results/{project}_chi26_v3.csv

Indicator Schema (sybil_results table):
    address TEXT, BT REAL, BW REAL, HF REAL, RF REAL, MA REAL,
    ops_flag INT, fund_flag INT, is_sybil INT, n_indicators INT

Classification:
    ops_flag  = (BT >= 5) OR (BW >= 10) OR (HF >= 0.80)
    fund_flag = (RF >= 0.50) OR (MA >= 5)
    is_sybil  = ops_flag OR fund_flag

16 Projects (2020-2024):
    uniswap, ens, 1inch, blur_s1, blur_s2, gitcoin, looksrare,
    eigenlayer, x2y2, dydx, apecoin, paraswap, badger, ampleforth,
    etherfi, pengu

Related Repos:
- UW-Decentralized-Computing-Lab/HasciDB: database + API
- UW-Decentralized-Computing-Lab/Blur: GNN (ARTEMIS) + LLMhunter + labeled data
- Adeline117/pre-airdrop-detection: LightGBM pre-airdrop features
- TrustaLabs/Airdrop-Sybil-Identification: graph community detection
- ArbitrumFoundation/sybil-detection: Louvain on transfer graphs
"""

import os
import json
import sqlite3
import requests
import pandas as pd
from typing import Optional
from pathlib import Path


HASCIDB_API = "https://hascidb.org"
HASCIDB_PROJECTS = [
    "uniswap", "ens", "1inch", "blur_s1", "blur_s2", "gitcoin",
    "looksrare", "eigenlayer", "x2y2", "dydx", "apecoin",
    "paraswap", "badger", "ampleforth", "etherfi", "pengu",
]

# HasciDB indicator thresholds (CHI'26 Table 13, Delphi Round 2)
THRESHOLDS = {"BT": 5, "BW": 10, "HF": 0.80, "RF": 0.50, "MA": 5}


class HasciDBLocal:
    """Query local HasciDB SQLite database.

    Setup:
        git clone https://github.com/UW-Decentralized-Computing-Lab/HasciDB
        cd HasciDB && python scripts/build_db.py
    """

    def __init__(self, db_path: str):
        if not os.path.exists(db_path):
            raise FileNotFoundError(
                f"HasciDB not found at {db_path}. "
                "Clone and build: git clone UW-Decentralized-Computing-Lab/HasciDB && "
                "cd HasciDB && python scripts/build_db.py"
            )
        self.db_path = db_path

    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA cache_size=-64000")
        return conn

    def get_project_sybils(self, project: str) -> pd.DataFrame:
        """Get all sybil results for a project."""
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM sybil_results WHERE project = ? AND is_sybil = 1",
                conn, params=(project,),
            )

    def get_project_legitimate(self, project: str) -> pd.DataFrame:
        """Get non-sybil eligible addresses for a project."""
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT * FROM sybil_results WHERE project = ? AND is_sybil = 0",
                conn, params=(project,),
            )

    def get_indicator_distributions(self, project: str) -> dict:
        """Get indicator value distributions for sybils vs non-sybils."""
        with self._conn() as conn:
            sybils = pd.read_sql_query(
                "SELECT BT, BW, HF, RF, MA FROM sybil_results "
                "WHERE project = ? AND is_sybil = 1", conn, params=(project,),
            )
            legit = pd.read_sql_query(
                "SELECT BT, BW, HF, RF, MA FROM sybil_results "
                "WHERE project = ? AND is_sybil = 0", conn, params=(project,),
            )
        return {
            "sybil": sybils.describe().to_dict(),
            "legitimate": legit.describe().to_dict(),
        }

    def get_serial_sybils(self) -> pd.DataFrame:
        """Get addresses flagged as sybil across multiple projects."""
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT address, COUNT(DISTINCT project) as n_projects, "
                "GROUP_CONCAT(project) as projects "
                "FROM sybil_results WHERE is_sybil = 1 "
                "GROUP BY address HAVING n_projects >= 2 "
                "ORDER BY n_projects DESC",
                conn,
            )

    def get_indicator_trigger_rates(self) -> pd.DataFrame:
        """Get per-project indicator trigger rates."""
        with self._conn() as conn:
            return pd.read_sql_query(
                "SELECT project, "
                "AVG(CASE WHEN BT >= 5 THEN 1 ELSE 0 END) as bt_rate, "
                "AVG(CASE WHEN BW >= 10 THEN 1 ELSE 0 END) as bw_rate, "
                "AVG(CASE WHEN HF >= 0.80 THEN 1 ELSE 0 END) as hf_rate, "
                "AVG(CASE WHEN RF >= 0.50 THEN 1 ELSE 0 END) as rf_rate, "
                "AVG(CASE WHEN MA >= 5 THEN 1 ELSE 0 END) as ma_rate, "
                "AVG(is_sybil) as sybil_rate, "
                "COUNT(*) as total "
                "FROM sybil_results GROUP BY project",
                conn,
            )


class HasciDBAPI:
    """Query HasciDB via REST API."""

    def __init__(self, base_url: str = HASCIDB_API):
        self.base_url = base_url
        self.session = requests.Session()

    def scan(self, address: str) -> dict:
        resp = self.session.post(f"{self.base_url}/v1/scan",
                                json={"address": address}, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def batch_scan(self, addresses: list[str]) -> dict:
        resp = self.session.post(f"{self.base_url}/v1/batch",
                                json={"addresses": addresses}, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def stats(self) -> dict:
        resp = self.session.get(f"{self.base_url}/v1/stats", timeout=15)
        resp.raise_for_status()
        return resp.json()

    def projects(self) -> list[dict]:
        resp = self.session.get(f"{self.base_url}/v1/projects", timeout=15)
        resp.raise_for_status()
        return resp.json()


def load_blur_labels(blur_repo_path: str) -> pd.DataFrame:
    """Load Blur Season 2 labeled data from UW-DCL/Blur repo.

    Expected file: database/airdrop_targets_behavior_flags.csv
    Columns: address, bw_flag, ml_flag, fd_flag, hf_flag

    Mapping to HasciDB indicators:
        bw_flag → BW (batch wallets)
        ml_flag → MA (multi-address / loop cycles)
        fd_flag → RF (rapid fund consolidation)
        hf_flag → HF (high frequency)
    """
    csv_path = os.path.join(blur_repo_path, "database",
                            "airdrop_targets_behavior_flags.csv")
    df = pd.read_csv(csv_path)
    df["is_sybil"] = (df[["bw_flag", "ml_flag", "fd_flag", "hf_flag"]].max(axis=1) > 0).astype(int)
    return df


def load_pre_airdrop_features(repo_path: str) -> pd.DataFrame:
    """Load pre-airdrop-detection features from Adeline117 repo.

    Expected output from 00_build_nft_feats_labeled.py
    """
    # Look for the labeled feature file
    candidates = [
        "data/nft_features_labeled.csv",
        "data/features_labeled.csv",
        "output/features.csv",
    ]
    for candidate in candidates:
        path = os.path.join(repo_path, candidate)
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError(f"Feature file not found in {repo_path}")
