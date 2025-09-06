# delta_client.py
import requests
from config import DELTA_BASE_URL

UA = {"User-Agent": "python-backtest-client"}

class DeltaClient:
    def __init__(self):
        self.base_url = DELTA_BASE_URL

    def get(self, path, params=None, timeout=(5, 30)):
        url = f"{self.base_url}{path}"
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
        r.raise_for_status()
        return r.json()

    def get_candles(self, symbol, resolution, start, end):
        return self.get("/history/candles", {
            "symbol": symbol,
            "resolution": resolution,
            "start": start,
            "end": end,
        })

    def get_funding(self, symbol, resolution, start, end):
        return self.get("/history/candles", {
            "symbol": f"FUNDING:{symbol}",
            "resolution": resolution,
            "start": start,
            "end": end,
        })
