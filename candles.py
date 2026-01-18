# candles.py
import os
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from delta_client import DeltaClient
from config import PRODUCT_SYMBOL, CANDLE_RESOLUTION
import requests

CANDLES_FILE = "candles.csv"
client = DeltaClient()

EXPECTED_COLUMNS = ["time_utc", "open", "high", "low", "close", "volume"]

# -------------------- Utilities --------------------
def atomic_write(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

# -------------------- Retry-safe GET --------------------
def safe_get(client, path, params, retries=5, delay=5):
    for attempt in range(1, retries + 1):
        try:
            return client.get(path, params)
        except requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status and 500 <= status < 600:
                print(f"⚠️ Delta API {status} (attempt {attempt}/{retries})")
                if attempt < retries:
                    time.sleep(delay)
                    continue
            raise
    return None

# -------------------- Fetching --------------------
def fetch_chunked_candles(symbol, resolution, start, end, chunk_days=90):
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
        all_candles.extend(raw.get("result", []))
        cur_start = cur_end
        time.sleep(0.2)

    all_candles.sort(key=lambda r: r["time"])
    return all_candles

def fetch_and_save_candles(symbol, resolution, start, end, out_file):
    print(f"Fetching candles for {symbol} {resolution} from {start} to {end}")
    candles = fetch_chunked_candles(symbol, resolution, start, end)

    if not candles:
        print("No candles returned.")
        return

    df = pd.DataFrame(candles)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)

    df = df[EXPECTED_COLUMNS]
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float
    })

    atomic_write(df, out_file)
    print(f"Saved {len(df)} candles to {out_file}")

# -------------------- Maintenance --------------------
def ensure_candles():
    if not os.path.exists(CANDLES_FILE):
        print("candles.csv missing. Bootstrapping 30 days of data.")
        end = int(time.time())
        start = end - 30 * 24 * 3600
        fetch_and_save_candles(
            PRODUCT_SYMBOL,
            CANDLE_RESOLUTION,
            start,
            end,
            CANDLES_FILE
        )

def append_new_candle(client):
    if not os.path.exists(CANDLES_FILE):
        print("candles.csv missing. Running ensure_candles first.")
        ensure_candles()

    df = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])

    if df.empty:
        print("Candles file empty. Skipping append.")
        return

    # 🔒 Enforce schema immediately (kills legacy column drift)
    df = df[EXPECTED_COLUMNS]

    last_ts = df["time_utc"].max().to_pydatetime().replace(tzinfo=timezone.utc)

    if CANDLE_RESOLUTION.endswith("h"):
        step = timedelta(hours=int(CANDLE_RESOLUTION[:-1]))
    elif CANDLE_RESOLUTION.endswith("m"):
        step = timedelta(minutes=int(CANDLE_RESOLUTION[:-1]))
    else:
        step = timedelta(hours=1)

    expected_next = last_ts + step
    now = datetime.now(timezone.utc)

    if now < expected_next + timedelta(minutes=5):
        print("No finalized candle yet.")
        return

    raw = safe_get(client, "/history/candles", {
        "symbol": PRODUCT_SYMBOL,
        "resolution": CANDLE_RESOLUTION,
        "start": int(expected_next.timestamp()),
        "end": int(now.timestamp())
    })

    if not raw or "result" not in raw:
        print("❌ Failed to fetch new candle after retries. Skipping this run.")
        return

    rows = raw.get("result", [])
    if not rows:
        print("No new candles returned.")
        return

    new = pd.DataFrame(rows)
    new["time_utc"] = pd.to_datetime(new["time"], unit="s", utc=True)

    new = new[EXPECTED_COLUMNS]

    for c in ["open", "high", "low", "close", "volume"]:
        new[c] = new[c].astype(float)

    # allow multiple missed candles, zero-volume candles allowed
    new = new[new["time_utc"] > last_ts]

    if new.empty:
        print("No strictly new candles.")
        return

    df = (
        pd.concat([df, new])
        .drop_duplicates(subset=["time_utc"])
        .sort_values("time_utc")
    )

    # 🔒 Enforce schema again before write (paranoia justified)
    df = df[EXPECTED_COLUMNS]

    atomic_write(df, CANDLES_FILE)
    print(f"Appended {len(new)} new candle(s).")

# -------------------- Entrypoint --------------------
def run_candles():
    ensure_candles()
    append_new_candle(client)

if __name__ == "__main__":
    run_candles()
