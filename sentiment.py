import os
import json
import time
import pandas as pd
from delta_client import DeltaClient

SENTIMENT_FILE = "sentiment.csv"
SENTIMENT_AGG_FILE = "sentiment_agg.json"

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
    
def is_first_run_today():
    """Return True if current time is around 8:00 AM IST (first run of the day)."""
    now_ist = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30)))
    return now_ist.hour == 8 and now_ist.minute < 15  # run audit only in 8:00–8:15 window
                
def fetch_l2_sentiment(symbol=PRODUCT_SYMBOL, depth=L2_DEPTH):
    ob = client.get("/l2orderbook", {"symbol": symbol, "depth": depth})
    bids = sum([b["size"] for b in ob["buy_levels"]])
    asks = sum([a["size"] for a in ob["sell_levels"]])
    imbalance = (bids - asks) / (bids + asks) if (bids + asks) > 0 else 0
    return imbalance

def fetch_trades_sentiment(symbol=PRODUCT_SYMBOL, lookback=TRADES_LOOKBACK):
    trades = client.get("/public/trades", {"symbol": symbol, "limit": lookback})
    signed = sum([t["size"] if t["side"] == "buy" else -t["size"] for t in trades["trades"]])
    total = sum([t["size"] for t in trades["trades"]])
    flow = signed / total if total > 0 else 0
    return flow


def run_sentiment():
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

# --- NEW: helper for main.py ---
def get_latest_sentiment(sentiment_file=SENTIMENT_FILE, agg_file=SENTIMENT_AGG_FILE):
    """
    Returns the most recent averaged sentiment score.
    Priority: sentiment_agg.json avg -> latest sentiment.csv row -> 0 fallback.
    """
    if os.path.exists(agg_file):
        with open(agg_file, "r") as f:
            agg = json.load(f)
            if "avg_sentiment_last_4" in agg:
                return float(agg["avg_sentiment_last_4"])

    if os.path.exists(sentiment_file):
        df = pd.read_csv(sentiment_file)
        if not df.empty and "sentiment_score" in df.columns:
            return float(df["sentiment_score"].iloc[-1])

    return 0.0

if __name__ == "__main__":
    run_sentiment()
