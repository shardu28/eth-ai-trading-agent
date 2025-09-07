# pseudo_live.py — pseudo-live test for 09:00-11:00 IST window using L2 + public trades sentiment
import csv
import time
from datetime import datetime, date, timedelta, timezone
import pandas as pd
import numpy as np
import ta

# assumes your repo already has DeltaClient, indicators, features, sentiment etc.
from delta_client import DeltaClient
from indicators import ema, vwma, roc
from features import ema_cross_signal
from sentiment import funding_signal, momentum_signal
from config import PRODUCT_SYMBOL

OUTPUT_CSV = "pseudo_live_signal.csv"

# ---- helper functions (same logic as backtest, adapted) ----
def compute_averaged_vwma_from_df(df, avg_window=3):
    avg_close = df["close"].rolling(window=avg_window, min_periods=1).mean()
    avg_vwma = (df["close"] * df["volume"]).rolling(window=avg_window, min_periods=1).sum() / \
               df["volume"].rolling(window=avg_window, min_periods=1).sum()
    return avg_close, avg_vwma

def compute_volume_profile_node_array(closes, volumes, window=50, price_precision=2):
    # returns array aligned with closes length (np.nan for early indexes)
    n = len(closes)
    nodes = np.full(n, np.nan)
    for i in range(n):
        if i < window - 1:
            continue
        start = i - window + 1
        slice_prices = np.round(closes[start:i+1], price_precision)
        slice_vols = volumes[start:i+1]
        uniq, inv = np.unique(slice_prices, return_inverse=True)
        vol_sums = np.zeros(len(uniq), dtype=float)
        np.add.at(vol_sums, inv, slice_vols)
        nodes[i] = float(uniq[int(np.argmax(vol_sums))])
    return nodes

def normalize(val, min_abs_clip=1e-9):
    # simple scale to -1..1 by dividing by absolute max (or use tanh)
    return np.tanh(val)

# ---- sentiment calculations ----
def orderbook_imbalance_from_l2(l2_json, top_n=5):
    """
    l2_json expected format per docs:
    {"success":true, "result": {"buy": [{"price":"..","size":...},...], "sell":[...], ...}}
    returns imbalance in [-1,1] = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    summing top_n levels on each side.
    """
    buy = l2_json.get("result", {}).get("buy", [])[:top_n]
    sell = l2_json.get("result", {}).get("sell", [])[:top_n]
    bid_qty = sum(float(b.get("size", 0)) for b in buy)
    ask_qty = sum(float(s.get("size", 0)) for s in sell)
    if (bid_qty + ask_qty) == 0:
        return 0.0
    imb = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    return float(imb)

def trade_flow_from_trades(trades_json, lookback=100):
    """
    trades_json expected per docs:
    {"success":true, "result": {"trades":[{"side":"buy","size":...,"price":"...","timestamp":...}, ...]}}
    return normalized trade flow in [-1,1] = sum(sign*size)/sum(size)
    """
    trades = trades_json.get("result", {}).get("trades", [])[:lookback]
    if not trades:
        return 0.0
    signed = 0.0
    total = 0.0
    for t in trades:
        side = t.get("side", "").lower()
        size = float(t.get("size", 0) or 0)
        sign = 1.0 if side == "buy" else -1.0
        signed += sign * size
        total += size
    if total == 0:
        return 0.0
    return float(signed / total)

# ---- orchestration: fetch data for today 09:00-11:00 IST ----
def run_pseudo_live(session_start_ist=9, session_end_ist=11,
                    avg_window=5, vp_window=50, rvi_period=10,
                    atr_mult_sl=1.5, atr_mult_tp=2.5, risk_fraction=0.01,
                    l2_top_n=5, trades_lookback=200):
    """
    session_start_ist and session_end_ist are inclusive hours (IST).
    This function:
      - computes UTC timestamps for today's session
      - fetches 1h candles for that window
      - fetches l2 orderbook snapshot and recent public trades
      - computes sentiment and indicator confirmations and prints a 'pseudo-live' decision
    """
    client = DeltaClient()

    # determine today's IST dates (use current date in Asia/Kolkata)
    # we'll compute start and end as utc epoch seconds
    now_utc = datetime.now(timezone.utc)
    today_utc = now_utc.astimezone(timezone.utc).date()

    # Find today's date in IST: convert now_utc to Asia/Kolkata local date
    # pandas helpful for tz conversions
    now_ts = pd.Timestamp(now_utc)
    now_ist = now_ts.tz_convert("Asia/Kolkata")
    today_ist = now_ist.date()

    # compose start and end in IST for this date
    start_ist = pd.Timestamp(datetime(today_ist.year, today_ist.month, today_ist.day, session_start_ist, 0), tz="Asia/Kolkata")
    end_ist   = pd.Timestamp(datetime(today_ist.year, today_ist.month, today_ist.day, session_end_ist, 0), tz="Asia/Kolkata")

    # convert to UTC unix seconds (Delta history endpoint expects seconds)
    start_utc = int(start_ist.tz_convert("UTC").timestamp())
    end_utc   = int(end_ist.tz_convert("UTC").timestamp())

    print(f"Fetching candles for {PRODUCT_SYMBOL} from {start_ist} to {end_ist} (IST) -> {start_utc}..{end_utc} (UTC epoch secs)")

    # get 1h candles
    raw = client.get("/history/candles", {
        "symbol": PRODUCT_SYMBOL,
        "resolution": "1h",
        "start": start_utc,
        "end": end_utc,
    })
    candles = raw.get("result", [])
    if not candles:
        print("No candles returned for window. Aborting pseudo-live test.")
        return None

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    # compute ATR (14) on our tiny window — if too short, ATR will use min_periods
    df["atr"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()

    # compute VWMA averaged, VP node, RVI (same as backtest)
    df["avg_close"], df["avg_vwma"] = compute_averaged_vwma_from_df(df, avg_window=avg_window)
    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1

    closes = df["close"].values
    vols = df["volume"].values
    vp_nodes = compute_volume_profile_node_array(closes, vols, window=vp_window)
    df["vp_node"] = vp_nodes

    # approximate RVI
    df["rvi"] = (df["close"] - df["open"]).rolling(window=rvi_period, min_periods=1).mean() / \
                ( (df["high"] - df["low"]).rolling(window=rvi_period, min_periods=1).mean().abs() + 1e-9)
    # map to 0..100 like prior
    df["rvi_scale"] = 50 * (1 + df["rvi"].clip(-1,1))
    df["rvi_signal"] = df["rvi_scale"].apply(lambda x: 1 if x > 50 else -1)

    # Fetch L2 orderbook snapshot (public endpoint per docs)
    # Endpoint: GET /l2orderbook/{symbol} (no auth required). See docs. :contentReference[oaicite:4]{index=4}
    l2_json = client.get(f"/l2orderbook/{PRODUCT_SYMBOL}", params={})  # wrapper may accept (path, params)
    imb = orderbook_imbalance_from_l2(l2_json, top_n=l2_top_n)
    imb_norm = normalize(imb)

    # Fetch recent public trades: GET /trades/{symbol} (per docs) :contentReference[oaicite:5]{index=5}
    trades_json = client.get(f"/trades/{PRODUCT_SYMBOL}", params={})
    tf = trade_flow_from_trades(trades_json, lookback=trades_lookback)
    tf_norm = normalize(tf)

    # Combine sentiment: weighted average (you can tune weights)
    w_imb, w_tf = 0.6, 0.4
    sentiment_score = w_imb * imb_norm + w_tf * tf_norm  # in roughly -1..1

    # Indicator checks on the most recent candle in df
    last_idx = len(df) - 1
    vwma_sig = int(df["vwma_signal"].iloc[last_idx])
    rvi_sig = int(df["rvi_signal"].iloc[last_idx])
    vp_node = float(df["vp_node"].iloc[last_idx]) if not np.isnan(df["vp_node"].iloc[last_idx]) else None
    last_close = float(df["close"].iloc[last_idx])
    last_atr = float(df["atr"].iloc[last_idx]) if not np.isnan(df["atr"].iloc[last_idx]) else None

    # VP breakout check (if vp_node available)
    vp_ok = False
    if vp_node and not np.isnan(vp_node):
        vp_ok = last_close > vp_node if vwma_sig == 1 else last_close < vp_node if vwma_sig == -1 else False

    # Voting / final rule:
    # require: vwma_sig == direction AND (rvi agrees OR vp_ok) AND sentiment_score aligned (threshold)
    final_dir = 0
    if vwma_sig != 0:
        votes = 0
        votes += 1 if rvi_sig == vwma_sig else 0
        votes += 1 if vp_ok else 0
        # require at least 1 of 2 confirmations
        conf_ok = votes >= 1
        # sentiment alignment threshold (tunable)
        sent_thresh = 0.15
        sent_ok = (sentiment_score > sent_thresh and vwma_sig == 1) or (sentiment_score < -sent_thresh and vwma_sig == -1)
        # final: both conf_ok and sentiment aligned
        if conf_ok and sent_ok:
            final_dir = vwma_sig

    # Build result object
    result = {
        "window_start_ist": str(start_ist),
        "window_end_ist": str(end_ist),
        "candles_count": len(df),
        "vwma_sig": vwma_sig,
        "rvi_sig": rvi_sig,
        "vp_node": float(vp_node) if vp_node is not None and not np.isnan(vp_node) else None,
        "orderbook_imbalance": imb,
        "trade_flow": tf,
        "sentiment_score": sentiment_score,
        "final_signal": "long" if final_dir == 1 else "short" if final_dir == -1 else "neutral"
    }

    # If we have a direction, build suggested entry / TP / SL and dynamic size (ATR-based risk_fraction %)
    if final_dir != 0 and last_atr and last_atr > 0:
        side = "buy" if final_dir == 1 else "sell"
        entry = last_close
        sl = entry - atr_mult_sl * last_atr if side == "buy" else entry + atr_mult_sl * last_atr
        tp = entry + atr_mult_tp * last_atr if side == "buy" else entry - atr_mult_tp * last_atr

        # dynamic size: risk_fraction of current equity (assume start equity 1000 or you can pass live equity)
        assumed_equity = 1000.0
        risk_capital = assumed_equity * risk_fraction
        size = risk_capital / (abs(entry - sl)) if abs(entry - sl) > 0 else 0

        result.update({
            "suggested_side": side,
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "suggested_size": size,
            "risk_capital": risk_capital
        })

    # Save CSV summary
    fields = list(result.keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(result)

    # Print report
    print("---- PSEUDO-LIVE SIGNAL REPORT ----")
    for k, v in result.items():
        print(f"{k}: {v}")
    print("-----------------------------------")
    return result

if __name__ == "__main__":
    # using your sweet-spot parameters by default
    res = run_pseudo_live(session_start_ist=9, session_end_ist=11,
                          avg_window=5, vp_window=50, rvi_period=10,
                          atr_mult_sl=1.5, atr_mult_tp=2.5, risk_fraction=0.01,
                          l2_top_n=5, trades_lookback=200)
