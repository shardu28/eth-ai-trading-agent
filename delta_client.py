# delta_client.py
import os
import time
import hmac
import hashlib
import requests
from config import DELTA_BASE_URL

UA = {"User-Agent": "python-backtest-client"}

class DeltaClient:
    def __init__(self):
        self.base_url = DELTA_BASE_URL
        self.api_key = os.getenv("DELTA_API_KEY")
        self.api_secret = os.getenv("DELTA_API_SECRET")

        if not self.api_key or not self.api_secret:
            self.api_key = None
            self.api_secret = None

    def _signed_get(self, path, timeout=(5, 30)):
        timestamp = str(int(time.time()))
        message = timestamp + "GET" + f"/v2{path}"

        signature = hmac.new(
            self.api_secret.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

        headers = {
            "api-key": self.api_key,
            "timestamp": timestamp,
            "signature": signature,
            **UA
        }

        url = f"{self.base_url}/v2{path}"
        r = requests.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def get_wallet_balance(self):
        """
        Returns wallet balances from Delta Exchange.
        """
        data = self._signed_get("/wallet/balances")
        return data

    def get(self, path, params=None, timeout=(5, 30)):
        """
        Perform GET request to Delta API.
        Example: client.get("/history/candles", {...})
        """
        # Always prefix with /v2
        url = f"{self.base_url}/v2{path}"
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        r.raise_for_status()
        return r.json()
