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
    return np.tanh(val)

# ---- sentiment calculations ----
def orderbook_imbalance_from_l2(l2_json, top_n=5):
    buy = l2_json.get("result", {}).get("buy", [])[:top_n]
    sell = l2_json.get("result", {}).get("sell", [])[:top_n]
    bid_qty = sum(float(b.get("size", 0)) for b in buy)
    ask_qty = sum(float(s.get("size", 0)) for s in sell)
    if (bid_qty + ask_qty) == 0:
        return 0.0
    imb = (bid_qty - ask_qty) / (bid_qty + ask_qty)
    return float(imb)

def trade_flow_from_trades(trades_json, lookback=100):
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
                    l2_top_n=5, trades_lookback=200, days_history=1):
    """
    session_start_ist and session_end_ist are inclusive hours (IST).
    """
    client = DeltaClient()

    now_utc = datetime.now(timezone.utc)
    now_ts = pd.Timestamp(now_utc)
    now_ist = now_ts.tz_convert("Asia/Kolkata")
    today_ist = now_ist.date()

    # session start and end in IST
    session_start = pd.Timestamp(datetime(today_ist.year, today_ist.month, today_ist.day, session_start_ist, 0), tz="Asia/Kolkata")
    session_end   = pd.Timestamp(datetime(today_ist.year, today_ist.month, today_ist.day, session_end_ist, 0), tz="Asia/Kolkata")

    # add extra history before session start
    hist_start = session_start - timedelta(days=days_history)

    start_utc = int(hist_start.tz_convert("UTC").timestamp())
    end_utc   = int(session_end.tz_convert("UTC").timestamp())

    print(f"Fetching candles for {PRODUCT_SYMBOL} from {hist_start} to {session_end} (IST) -> {start_utc}..{end_utc} (UTC)")

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
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    for c in ["open","high","low","close","volume"]:
        df[c] = df[c].astype(float)

    # indicators use full df
    df["atr"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    df["avg_close"], df["avg_vwma"] = compute_averaged_vwma_from_df(df, avg_window=avg_window)
    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1
    df["vp_node"] = compute_volume_profile_node_array(df["close"].values, df["volume"].values, window=vp_window)
    df["rvi"] = (df["close"] - df["open"]).rolling(window=rvi_period, min_periods=1).mean() / \
                ((df["high"] - df["low"]).rolling(window=rvi_period, min_periods=1).mean().abs() + 1e-9)
    df["rvi_scale"] = 50 * (1 + df["rvi"].clip(-1,1))
    df["rvi_signal"] = df["rvi_scale"].apply(lambda x: 1 if x > 50 else -1)

    # trim to session window
    df = df[(df["time"] >= session_start) & (df["time"] <= session_end)].copy()
    if df.empty:
        print("No candles in session window after trimming.")
        return None

    # L2 + trades
    l2_json = client.get(f"/l2orderbook/{PRODUCT_SYMBOL}", params={})
    imb = orderbook_imbalance_from_l2(l2_json, top_n=l2_top_n)
    imb_norm = normalize(imb)

    trades_json = client.get(f"/trades/{PRODUCT_SYMBOL}", params={})
    tf = trade_flow_from_trades(trades_json, lookback=trades_lookback)
    tf_norm = normalize(tf)

    sentiment_score = 0.6 * imb_norm + 0.4 * tf_norm

    # last candle
    last_idx = len(df) - 1
    vwma_sig = int(df["vwma_signal"].iloc[last_idx])
    rvi_sig = int(df["rvi_signal"].iloc[last_idx])
    vp_node = float(df["vp_node"].iloc[last_idx]) if not np.isnan(df["vp_node"].iloc[last_idx]) else None
    last_close = float(df["close"].iloc[last_idx])
    last_atr = float(df["atr"].iloc[last_idx]) if not np.isnan(df["atr"].iloc[last_idx]) else None

    vp_ok = False
    if vp_node and not np.isnan(vp_node):
        vp_ok = last_close > vp_node if vwma_sig == 1 else last_close < vp_node if vwma_sig == -1 else False

    final_dir = 0
    if vwma_sig != 0:
        votes = 0
        votes += 1 if rvi_sig == vwma_sig else 0
        votes += 1 if vp_ok else 0
        conf_ok = votes >= 1
        sent_thresh = 0.15
        sent_ok = (sentiment_score > sent_thresh and vwma_sig == 1) or (sentiment_score < -sent_thresh and vwma_sig == -1)
        if conf_ok and sent_ok:
            final_dir = vwma_sig

    result = {
        "window_start_ist": str(session_start),
        "window_end_ist": str(session_end),
        "candles_count": len(df),
        "vwma_sig": vwma_sig,
        "rvi_sig": rvi_sig,
        "vp_node": float(vp_node) if vp_node is not None and not np.isnan(vp_node) else None,
        "orderbook_imbalance": imb,
        "trade_flow": tf,
        "sentiment_score": sentiment_score,
        "final_signal": "long" if final_dir == 1 else "short" if final_dir == -1 else "neutral"
    }

    if final_dir != 0 and last_atr and last_atr > 0:
        side = "buy" if final_dir == 1 else "sell"
        entry = last_close
        sl = entry - atr_mult_sl * last_atr if side == "buy" else entry + atr_mult_sl * last_atr
        tp = entry + atr_mult_tp * last_atr if side == "buy" else entry - atr_mult_tp * last_atr
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

    fields = list(result.keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(result)

    print("---- PSEUDO-LIVE SIGNAL REPORT ----")
    for k, v in result.items():
        print(f"{k}: {v}")
    print("-----------------------------------")
    return result

if __name__ == "__main__":
    res = run_pseudo_live(session_start_ist=9, session_end_ist=11,
                          avg_window=5, vp_window=50, rvi_period=10,
                          atr_mult_sl=1.5, atr_mult_tp=2.5, risk_fraction=0.01,
                          l2_top_n=5, trades_lookback=200)
