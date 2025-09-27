# test_csv.py
import pandas as pd
import time
from datetime import datetime, timedelta, timezone
from delta_client import DeltaClient
from config import PRODUCT_SYMBOL, CANDLE_RESOLUTION  # dynamic imports

client = DeltaClient()

def fetch_yesterday_candles(symbol=PRODUCT_SYMBOL, resolution=CANDLE_RESOLUTION, out_file="test.csv"):
    """
    Fetch all 1H candles from yesterday (00:00 to 23:59 UTC) and save to test.csv
    """
    now = datetime.now(timezone.utc)
    # define yesterday's UTC window
    yesterday = now.date() - timedelta(days=1)
    start = datetime.combine(yesterday, datetime.min.time()).replace(tzinfo=timezone.utc)
    end = datetime.combine(yesterday, datetime.max.time()).replace(tzinfo=timezone.utc)

    print(f"📥 Fetching candles for {symbol} {resolution} from {start} to {end}")

    raw = client.get("/history/candles", {
        "symbol": symbol,
        "resolution": resolution,
        "start": int(start.timestamp()),
        "end": int(end.timestamp())
    })

    rows = raw.get("result", [])
    if not rows:
        print("❌ No candles returned")
        return

    df = pd.DataFrame(rows)
    df["time_utc"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df[["time_utc", "open", "high", "low", "close", "volume"]]

    # enforce float dtypes
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df.to_csv(out_file, index=False)
    print(f"✅ Saved {len(df)} candles to {out_file}")
    print(df.head())
    print(df.tail())

if __name__ == "__main__":
    fetch_yesterday_candles()
