import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta, timezone
from delta_client import DeltaClient
from candles import fetch_and_save_candles  # your candles.py provides this
from config import PRODUCT_SYMBOL, CANDLE_RESOLUTION  # dynamic import

CANDLES_FILE = "candles.csv"
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
    
def audit_midnight_candles():
    """
    Ensure that candles.csv contains the overnight candles up to 7:00 AM IST.
    This acts as a fallback in case midnight_candles.py failed or was delayed.
    """
    if not os.path.exists(CANDLES_FILE):
        print("❌ candles.csv not found, cannot audit midnight candles.")
        return

    df = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])
    if df.empty:
        print("❌ candles.csv is empty, fetching fresh history...")
        # Fallback: fetch last 30 days (same as ensure_candles)
        end = int(time.time())
        start = end - 30 * 24 * 3600
        fetch_and_save_candles(PRODUCT_SYMBOL, CANDLE_RESOLUTION, start, end, CANDLES_FILE)
        return

    last_ts = df["time_utc"].max()

    # Convert last_ts to IST
    last_ts_ist = last_ts.tz_convert("Asia/Kolkata")

    # Build today's expected 7:00 AM IST candle close time
    today = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=5, minutes=30))).date()
    expected_7am_ist = datetime.combine(today, datetime.min.time()).replace(hour=7, tzinfo=timezone(timedelta(hours=5, minutes=30)))

    if last_ts_ist >= expected_7am_ist:
        print(f"✅ Audit: candles.csv already has data up to {last_ts_ist.strftime('%Y-%m-%d %H:%M %Z')}")
    else:
        print(f"⚠️ Audit: candles.csv missing overnight candles. Fetching from midnight to 7 AM IST...")
        start = int(last_ts.timestamp()) + 1
        end = int(expected_7am_ist.astimezone(timezone.utc).timestamp())
        raw = client.get("/history/candles", {
            "symbol": PRODUCT_SYMBOL,
            "resolution": CANDLE_RESOLUTION,
            "start": start,
            "end": end
        })
        rows = raw.get("result", [])
        if rows:
            new = pd.DataFrame(rows)
            new["time_utc"] = pd.to_datetime(new["time"], unit="s", utc=True)
            new = new[["time_utc", "open", "high", "low", "close", "volume"]]
            df = pd.concat([df, new]).drop_duplicates(subset=["time_utc"]).sort_values("time_utc")
            atomic_write(df, CANDLES_FILE)
            print(f"✅ Audit: appended {len(new)} overnight candles.")
        else:
            print("⚠️ Audit: no overnight candles returned from API.")
            
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

def ensure_candles():
    """On first run, fetch 1 month of 1H candles if not present."""
    if not os.path.exists(CANDLES_FILE):
        print("Fetching 1 month of historical candles...")
        end = int(time.time())
        start = end - 30 * 24 * 3600
        fetch_and_save_candles(PRODUCT_SYMBOL, CANDLE_RESOLUTION, start, end, CANDLES_FILE)

def append_new_candle(client):
    df = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])

    if df.empty:
        print("Candles file empty. Skipping append.")
        return

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

    raw = client.get("/history/candles", {
        "symbol": PRODUCT_SYMBOL,
        "resolution": CANDLE_RESOLUTION,
        "start": int(expected_next.timestamp()),
        "end": int(now.timestamp())
    })

    rows = raw.get("result", [])
    if not rows:
        print("No new candles returned by API.")
        return

    new = pd.DataFrame(rows)
    new["time_utc"] = pd.to_datetime(new["time"], unit="s", utc=True)
    new = new[["time_utc", "open", "high", "low", "close", "volume"]]

    for c in ["open", "high", "low", "close", "volume"]:
        new[c] = new[c].astype(float)

    new = new[new["time_utc"] > last_ts]

    if new.empty:
        print("No strictly new candles after filtering.")
        return

    df = (
        pd.concat([df, new])
        .drop_duplicates(subset=["time_utc"])
        .sort_values("time_utc")
    )

    atomic_write(df, CANDLES_FILE)
    print(f"Appended {len(new)} new candles.")

def run_sentiment():
    if is_first_run_today():
    audit_midnight_candles() # only on first run of the day
        
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
