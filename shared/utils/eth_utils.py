"""Shared Ethereum data collection utilities."""

import os
import time
import json
import yaml
import itertools
import requests
import pandas as pd
from typing import Optional
from pathlib import Path


def load_config(config_path: Optional[str] = None) -> dict:
    """Load config from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


class EtherscanClient:
    """Etherscan API client with multi-key rotation.

    With 6 API keys, achieves ~30 calls/sec (5 per key).
    Keys are rotated round-robin to maximize throughput.
    """

    BASE_URL_V1 = "https://api.etherscan.io/api"
    BASE_URL_V2 = "https://api.etherscan.io/v2/api"
    RATE_LIMIT_PER_KEY = 0.21  # 5 calls/sec per key

    def __init__(self, api_keys: Optional[list[str]] = None):
        if api_keys is None:
            try:
                cfg = load_config()
                api_keys = cfg.get("etherscan", {}).get("api_keys", [])
            except FileNotFoundError:
                api_keys = []
            # Fallback to env var
            if not api_keys:
                env_key = os.getenv("ETHERSCAN_API_KEY", "")
                api_keys = [env_key] if env_key else []

        self.api_keys = api_keys
        self._key_cycle = itertools.cycle(api_keys) if api_keys else None
        self._key_last_used: dict[str, float] = {k: 0 for k in api_keys}
        self._total_calls = 0

    @property
    def num_keys(self) -> int:
        return len(self.api_keys)

    def _get_next_key(self) -> str:
        """Get next available API key with per-key rate limiting."""
        if not self._key_cycle:
            return ""
        key = next(self._key_cycle)
        elapsed = time.time() - self._key_last_used[key]
        if elapsed < self.RATE_LIMIT_PER_KEY:
            time.sleep(self.RATE_LIMIT_PER_KEY - elapsed)
        self._key_last_used[key] = time.time()
        return key

    def _get(self, params: dict, retries: int = 3) -> dict:
        key = self._get_next_key()
        params["apikey"] = key
        # V2 API requires chainid parameter for Ethereum mainnet
        if "chainid" not in params:
            params["chainid"] = 1
        for attempt in range(retries):
            try:
                resp = requests.get(self.BASE_URL_V2, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                # Rate limit hit → wait and retry with different key
                if data.get("result") == "Max rate limit reached":
                    time.sleep(1)
                    key = self._get_next_key()
                    params["apikey"] = key
                    continue
                if data.get("status") == "0" and data.get("message") != "No transactions found":
                    raise ValueError(f"Etherscan error: {data.get('result', 'Unknown')}")
                self._total_calls += 1
                return data
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise
        return {"status": "0", "result": []}

    def get_normal_txs(self, address: str, start_block: int = 0,
                       end_block: int = 99999999, page: int = 1,
                       offset: int = 1000) -> pd.DataFrame:
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

    def get_all_txs(self, address: str, max_pages: int = 10) -> pd.DataFrame:
        """Get all transactions with pagination."""
        all_txs = []
        for page in range(1, max_pages + 1):
            df = self.get_normal_txs(address, page=page, offset=10000)
            if df.empty:
                break
            all_txs.append(df)
            if len(df) < 10000:
                break
        return pd.concat(all_txs, ignore_index=True) if all_txs else pd.DataFrame()

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

    KNOWN_AGENT_PROTOCOLS = {
        "olas": {
            "registry": "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2",
            "service_registry": "0x48b6af7B12C71f09e2fC8aF4855De4Ff54e002c2",
            "description": "Autonolas autonomous agent service registry",
        },
        "fetch_ai": {
            "token": "0xaea46A60368A7bD060eec7DF8CBa43b7EF41Ad85",
            "description": "Fetch.ai FET token contract",
        },
        "ai_arena": {
            "token": "0x6De037ef9aD2725EB40118Bb1702EBb27e4Aeb24",
            "description": "AI Arena NRN token",
        },
    }

    CONFIRMED_AGENT_ADDRESSES: list[str] = []

    KNOWN_MEV_BOTS: list[str] = [
        "0x56178a0d5F301bAf6CF3e1Cd53d9863437345Bf9",  # jaredfromsubway.eth
        "0x6b75d8AF000000e20B7a7DDf000Ba900b4009A80",  # Known MEV bot
    ]
