# sentiment.py
import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from delta_client import DeltaClient
from candles import fetch_and_save_candles  # your candles.py provides this

CANDLES_FILE = "candles.csv"
SENTIMENT_FILE = "sentiment.csv"
SENTIMENT_AGG_FILE = "sentiment_agg.json"

PRODUCT_SYMBOL = "ETHUSD"
L2_DEPTH = 5
TRADES_LOOKBACK = 200

client = DeltaClient()

def atomic_write(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)

def load_existing(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

def fetch_l2_sentiment(symbol=PRODUCT_SYMBOL, depth=L2_DEPTH):
    ob = client.get("/v2/l2orderbook", {"symbol": symbol, "depth": depth})
    bids = sum([b["size"] for b in ob["buy_levels"]])
    asks = sum([a["size"] for a in ob["sell_levels"]])
    imbalance = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
    return imbalance

def fetch_trades_sentiment(symbol=PRODUCT_SYMBOL, lookback=TRADES_LOOKBACK):
    trades = client.get("/v2/public/trades", {"symbol": symbol, "limit": lookback})
    signed = sum([t["size"] if t["side"] == "buy" else -t["size"] for t in trades["trades"]])
    total = sum([t["size"] for t in trades["trades"]])
    flow = signed / total if total > 0 else 0
    return flow

def ensure_candles():
    """On first run, fetch 1 month of 1H candles if not present."""
    if not os.path.exists(CANDLES_FILE):
        print("Fetching 1 month of historical candles...")
        end = int(time.time())
        start = end - 30 * 24 * 3600
        fetch_and_save_candles(PRODUCT_SYMBOL, "1h", start, end, CANDLES_FILE)

def append_new_candle():
    """Check if a new hourly candle formed, append if needed."""
    df = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])
    last_ts = df["time_utc"].max().to_pydatetime().replace(tzinfo=timezone.utc)
    expected_next = last_ts + timedelta(hours=1)
    now = datetime.now(timezone.utc)

    if now >= expected_next + timedelta(minutes=5):  # allow API to finalize
        raw = client.get("/history/candles", {
            "symbol": PRODUCT_SYMBOL,
            "resolution": "1h",
            "start": int(expected_next.timestamp()),
            "end": int(now.timestamp())
        })
        rows = raw.get("result", [])
        if rows:
            new = pd.DataFrame(rows)
            new["time_utc"] = pd.to_datetime(new["time"], unit="s", utc=True)
            new = new[["time_utc", "open", "high", "low", "close", "volume"]]
            df = pd.concat([df, new]).drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
            atomic_write(df, CANDLES_FILE)
            print(f"Appended {len(new)} new candles.")

def run_sentiment():
    ensure_candles()
    append_new_candle()

    run_time_utc = datetime.now(timezone.utc)
    run_time_ist = run_time_utc.astimezone(timezone(timedelta(hours=5, minutes=30)))

    imbalance = fetch_l2_sentiment()
    flow = fetch_trades_sentiment()
    score = 0.6 * imbalance + 0.4 * flow

    # log to sentiment.csv
    existing = load_existing(SENTIMENT_FILE)
    new_row = pd.DataFrame([{
        "run_time_utc": run_time_utc,
        "run_time_ist": run_time_ist,
        "imbalance": imbalance,
        "trade_flow": flow,
        "sentiment_score": score
    }])
    updated = pd.concat([existing, new_row], ignore_index=True)
    atomic_write(updated, SENTIMENT_FILE)

    # compute rolling avg last 4 samples
    last4 = updated.tail(4)
    avg_score = last4["sentiment_score"].mean() if not last4.empty else score
    agg = {
        "last_run": run_time_utc.isoformat(),
        "avg_sentiment_last_4": avg_score,
        "n_samples": len(last4),
    }
    atomic_json(agg, SENTIMENT_AGG_FILE)

    print(f"[{run_time_ist}] Imbalance={imbalance:.3f}, Flow={flow:.3f}, Score={score:.3f}, Avg4={avg_score:.3f}")

if __name__ == "__main__":
    run_sentiment()
