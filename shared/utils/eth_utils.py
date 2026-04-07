"""Shared Ethereum data collection utilities."""

import os
import time
import json
import requests
import pandas as pd
from typing import Optional


class EtherscanClient:
    """Lightweight Etherscan API client with rate limiting."""

    BASE_URL = "https://api.etherscan.io/api"
    RATE_LIMIT = 0.21  # 5 calls/sec for free tier

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("ETHERSCAN_API_KEY", "")
        self._last_call = 0

    def _rate_limit(self):
        elapsed = time.time() - self._last_call
        if elapsed < self.RATE_LIMIT:
            time.sleep(self.RATE_LIMIT - elapsed)
        self._last_call = time.time()

    def _get(self, params: dict) -> dict:
        self._rate_limit()
        params["apikey"] = self.api_key
        resp = requests.get(self.BASE_URL, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("status") == "0" and data.get("message") != "No transactions found":
            raise ValueError(f"Etherscan error: {data.get('result', 'Unknown')}")
        return data

    def get_normal_txs(self, address: str, start_block: int = 0,
                       end_block: int = 99999999, page: int = 1,
                       offset: int = 100) -> pd.DataFrame:
        """Get normal transactions for an address."""
        data = self._get({
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "page": page,
            "offset": offset,
            "sort": "asc",
        })
        txs = data.get("result", [])
        if not txs or isinstance(txs, str):
            return pd.DataFrame()
        return pd.DataFrame(txs)

    def get_internal_txs(self, address: str, start_block: int = 0,
                         end_block: int = 99999999) -> pd.DataFrame:
        """Get internal transactions for an address."""
        data = self._get({
            "module": "account",
            "action": "txlistinternal",
            "address": address,
            "startblock": start_block,
            "endblock": end_block,
            "sort": "asc",
        })
        txs = data.get("result", [])
        if not txs or isinstance(txs, str):
            return pd.DataFrame()
        return pd.DataFrame(txs)

    def get_erc20_transfers(self, address: str) -> pd.DataFrame:
        """Get ERC20 token transfers."""
        data = self._get({
            "module": "account",
            "action": "tokentx",
            "address": address,
            "sort": "asc",
        })
        txs = data.get("result", [])
        if not txs or isinstance(txs, str):
            return pd.DataFrame()
        return pd.DataFrame(txs)

    def get_contract_abi(self, address: str) -> Optional[list]:
        """Get contract ABI if verified."""
        data = self._get({
            "module": "contract",
            "action": "getabi",
            "address": address,
        })
        result = data.get("result", "")
        if result and result != "Contract source code not verified":
            return json.loads(result)
        return None

    def is_contract(self, address: str) -> bool:
        """Check if address is a contract."""
        data = self._get({
            "module": "proxy",
            "action": "eth_getCode",
            "address": address,
            "tag": "latest",
        })
        code = data.get("result", "0x")
        return code != "0x"


class KnownAgentAddresses:
    """Known AI agent protocol addresses for ground truth labeling."""

    # Verified AI agent protocols/platforms on Ethereum mainnet
    KNOWN_AGENT_PROTOCOLS = {
        # Autonolas/OLAS - autonomous agent services
        "olas": {
            "registry": "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2",
            "service_registry": "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2",
            "description": "Autonolas autonomous agent service registry",
        },
        # Fetch.ai agents
        "fetch_ai": {
            "token": "0xaea46A60368A7bD060eec7DF8CBa43b7EF41Ad85",
            "description": "Fetch.ai FET token contract",
        },
        # AI Arena
        "ai_arena": {
            "token": "0x6De037ef9aD2725EB40118Bb1702EBb27e4Aeb24",
            "description": "AI Arena NRN token",
        },
    }

    # Addresses confirmed as AI agent operators via on-chain evidence
    # (e.g., registered in Autonolas service registry, known bot operators)
    CONFIRMED_AGENT_ADDRESSES: list[str] = [
        # These would be populated from Autonolas registry queries
        # and manual verification
    ]

    # Known MEV bot addresses (subset that uses AI/ML strategies)
    KNOWN_MEV_BOTS: list[str] = [
        "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9",  # jaredfromsubway.eth
        "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80",  # Known MEV bot
    ]
