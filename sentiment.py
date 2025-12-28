# sentiment.py
import os
import json
import pandas as pd
from datetime import datetime, timedelta, timezone
from delta_client import DeltaClient
from config import PRODUCT_SYMBOL
from config import DELTA_API_GLOBAL

SENTIMENT_FILE = "sentiment.csv"
SENTIMENT_AGG_FILE = "sentiment_agg.json"

L2_DEPTH = 5
TRADES_LOOKBACK = 200
ROLLING_WINDOW = 5

client = DeltaClient()
client_l2 = DeltaClient(base_url=DELTA_API_GLOBAL)

# -------------------- Utilities --------------------
def atomic_write(df, path):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def load_existing(path):
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["run_time_utc", "run_time_ist"])
    return pd.DataFrame()

def now_ist():
    return datetime.now(timezone.utc).astimezone(
        timezone(timedelta(hours=5, minutes=30))
    )

# -------------------- Sentiment Fetch --------------------

def fetch_l2_sentiment(symbol=PRODUCT_SYMBOL, depth=L2_DEPTH):
    try:
        ob = client_l2.get("/l2orderbook", {
            "symbol": symbol,
            "depth": depth
        })

        buy = sum(b.get("size", 0) for b in ob.get("buy_levels", []))
        sell = sum(a.get("size", 0) for a in ob.get("sell_levels", []))

        return (buy - sell) / (buy + sell) if buy + sell > 0 else 0.0

    except Exception as e:
        print(f"⚠️ L2 unavailable (global): {e}")
        return 0.0

def fetch_trades_sentiment(symbol=PRODUCT_SYMBOL, lookback=TRADES_LOOKBACK):
    trades = client.get("/public/trades", {"symbol": symbol, "limit": lookback})
    signed = sum(
        t["size"] if t["side"] == "buy" else -t["size"]
        for t in trades.get("trades", [])
    )
    total = sum(t["size"] for t in trades.get("trades", []))
    return signed / total if total > 0 else 0.0

# -------------------- Core --------------------
def run_sentiment():
    run_time_utc = datetime.now(timezone.utc)
    run_time_ist = now_ist()

    imbalance = fetch_l2_sentiment()
    flow = fetch_trades_sentiment()
    score = 0.6 * imbalance + 0.4 * flow

    existing = load_existing(SENTIMENT_FILE)

    new_row = pd.DataFrame([{
        "run_time_utc": run_time_utc,
        "run_time_ist": run_time_ist,
        "imbalance": float(imbalance),
        "trade_flow": float(flow),
        "sentiment_score": float(score),
    }])

    updated = pd.concat([existing, new_row], ignore_index=True)
    atomic_write(updated, SENTIMENT_FILE)

    # Rolling aggregate
    last_n = updated.tail(ROLLING_WINDOW)
    avg_score = last_n["sentiment_score"].mean()

    agg = {
        "last_run_utc": run_time_utc.isoformat(),
        "last_run_ist": run_time_ist.isoformat(),
        "avg_sentiment_last_n": avg_score,
        "n_samples": len(last_n),
    }

    atomic_json(agg, SENTIMENT_AGG_FILE)

    print(
        f"[{run_time_ist.strftime('%Y-%m-%d %H:%M')}] "
        f"imb={imbalance:.3f}, flow={flow:.3f}, "
        f"score={score:.3f}, avg{ROLLING_WINDOW}={avg_score:.3f}"
    )

# -------------------- Public API --------------------
def get_latest_sentiment():
    if os.path.exists(SENTIMENT_AGG_FILE):
        with open(SENTIMENT_AGG_FILE) as f:
            return float(json.load(f).get("avg_sentiment_last_n", 0.0))
    return 0.0

if __name__ == "__main__":
    run_sentiment()
