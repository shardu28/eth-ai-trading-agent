# candles.py
import os
import time
import pandas as pd
from datetime import datetime, timezone
from delta_client import DeltaClient

client = DeltaClient()

def fetch_chunked_candles(symbol, resolution, start, end, chunk_days=90):
    """
    Fetch candles from Delta in chunks to avoid rate limits.
    Returns a list of raw candle dicts.
    """
    all_candles = []
    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + chunk_days * 24 * 3600, end)
        raw = client.get("/history/candles", {
            "symbol": symbol,
            "resolution": resolution,
            "start": cur_start,
            "end": cur_end,
        })
        chunk = raw.get("result", [])
        all_candles.extend(chunk)
        cur_start = cur_end
        time.sleep(0.2)  # avoid hitting API rate limit
    all_candles.sort(key=lambda r: r["time"])
    return all_candles

def fetch_and_save_candles(symbol, resolution, start, end, out_file):
    """
    Fetch candles from Delta and save to CSV.
    Columns: time_utc, open, high, low, close, volume
    """
    print(f"Fetching candles for {symbol} {resolution} from {start} to {end}")
    candles = fetch_chunked_candles(symbol, resolution, start, end)

    if not candles:
        print("No candles returned.")
        return

    df = pd.DataFrame(candles)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time_utc", "open", "high", "low", "close", "volume"]]
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    tmp = out_file + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out_file)

    print(f"Saved {len(df)} candles to {out_file}")
    return df
