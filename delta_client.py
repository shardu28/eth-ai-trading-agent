# delta_client.py
import requests
from config import DELTA_BASE_URL

UA = {"User-Agent": "python-backtest-client"}

class DeltaClient:
    def __init__(self):
        self.base_url = DELTA_BASE_URL

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
