# backtest.py  (SOLUSD standalone backtest, tweaked params)
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import ta
import os

from delta_client import DeltaClient
from config import PRODUCT_SYMBOL

OUTPUT_CSV = "backtest_results.csv"
EQUITY_CURVE_IMG = "equity_curve.png"

# 🔹 Parity-testing exports
BACKTEST_CANDLES_CSV = "backtest_candles.csv"
BACKTEST_SIGNALS_CSV = "backtest_signals.csv"

# ----------------- Fixed Backtest Params -----------------
DAYS = 365
START_EQUITY = 100.0
LEVERAGE = 20.0
ROUND_TRIP_FEE = 0.001

ATR_MULT_SL = 1.3
ATR_MULT_TP = 3.7
ADX_THRESH = 29
AVG_WINDOW = 5
VP_WINDOW = 50
RVI_PERIOD = 10
RISK_FRACTION = 0.02

# 🔹 Session filter (IST, minute-based, midnight-safe)
SESSION_START_MIN = 12 * 60 + 30   # 12:30 IST
SESSION_END_MIN   = 1 * 60 + 30    # 01:30 IST

ATR_PCT_MIN = 0.005
CHUNK_DAYS = 90
POLITE_DELAY = 0.2

# ----------------- Helpers -----------------
def fetch_chunked_candles(client, symbol, resolution, start, end):
    out = []
    cur = start
    while cur < end:
        nxt = min(cur + CHUNK_DAYS * 86400, end)
        raw = client.get("/history/candles", {
            "symbol": symbol,
            "resolution": resolution,
            "start": cur,
            "end": nxt,
        })
        out.extend(raw.get("result", []))
        cur = nxt
        time.sleep(POLITE_DELAY)
    return sorted(out, key=lambda x: x["time"])

def compute_avg_vwma(close, vol, window):
    return (close * vol).rolling(window, 1).sum() / vol.rolling(window, 1).sum()

def compute_vp_node(close, window, precision=2):
    def vp(x):
        r = x.round(precision).astype(str)
        return float(pd.Series(r).mode().iloc[0])
    return close.rolling(window, 1).apply(vp, raw=False)

def rvi_approx(c, o, h, l, p):
    num = (c - o).rolling(p, 1).mean()
    den = (h - l).rolling(p, 1).mean().abs() + 1e-9
    return 50 * (1 + (num / den).clip(-1, 1))

# ----------------- Backtest -----------------
def run_backtest():
    client = DeltaClient()
    end_ts = int(time.time())
    start_ts = end_ts - DAYS * 86400

    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start_ts, end_ts)
    df = pd.DataFrame(candles)

    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ist"] = df["time"].dt.tz_convert("Asia/Kolkata")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    df["atr"] = ta.volatility.AverageTrueRange(
        df["high"], df["low"], df["close"], 14
    ).average_true_range()

    df["adx"] = ta.trend.ADXIndicator(
        df["high"], df["low"], df["close"], 14
    ).adx()

    df["avg_close"] = df["close"].rolling(AVG_WINDOW, 1).mean()
    df["avg_vwma"] = compute_avg_vwma(df["close"], df["volume"], AVG_WINDOW)

    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1

    df["rvi"] = rvi_approx(df["close"], df["open"], df["high"], df["low"], RVI_PERIOD)
    mean_rvi = df["rvi"].mean()
    df["rvi_signal"] = df["rvi"].apply(lambda x: 1 if x > mean_rvi else -1)

    df["vp_node"] = compute_vp_node(df["close"], VP_WINDOW)

    df.to_csv(BACKTEST_CANDLES_CSV, index=False)

    equity = START_EQUITY
    trades = []
    equity_curve, times = [], []
    trades_today = {}
    signal_rows = []

    for i, r in df.iterrows():
        ts, ts_ist = r["time"], r["time_ist"]
        close, atr, adx = r["close"], r["atr"], r["adx"]
        vw, rv, vp = r["vwma_signal"], r["rvi_signal"], r["vp_node"]

        signal_rows.append({
            "time": ts, "close": close, "atr": atr,
            "adx": adx, "vwma_signal": vw,
            "rvi_signal": rv, "vp_node": vp
        })

        # -------- EXIT ALWAYS --------
        if trades and trades[-1]["status"] == "open":
            t = trades[-1]
            hit = None
            if t["side"] == "buy":
                if close >= t["tp"]: hit = ("tp", t["tp"])
                elif close <= t["sl"]: hit = ("sl", t["sl"])
            else:
                if close <= t["tp"]: hit = ("tp", t["tp"])
                elif close >= t["sl"]: hit = ("sl", t["sl"])

            if hit:
                result, px = hit
                pnl = ((px - t["entry"]) if t["side"] == "buy" else (t["entry"] - px)) * t["size"]
                fees = ROUND_TRIP_FEE * px * t["size"]
                equity += pnl - fees
                t.update({"exit": px, "result": result, "pnl": pnl - fees, "equity": equity, "status": result})

        if trades and trades[-1]["status"] == "open":
            equity_curve.append(equity); times.append(ts); continue

        # -------- ENTRY SESSION FILTER (minute-based) --------
        cur_min = ts_ist.hour * 60 + ts_ist.minute
        if SESSION_START_MIN <= SESSION_END_MIN:
            in_session = SESSION_START_MIN <= cur_min <= SESSION_END_MIN
        else:
            in_session = cur_min >= SESSION_START_MIN or cur_min <= SESSION_END_MIN

        if not in_session:
            equity_curve.append(equity); times.append(ts); continue

        if i < 30 or adx <= ADX_THRESH or atr / close < ATR_PCT_MIN:
            equity_curve.append(equity); times.append(ts); continue

        vp_ok = (close > vp) if vw == 1 else (close < vp)
        if vw != 0 and ((rv == vw) or vp_ok):
            day = ts_ist.date()
            if trades_today.get(day, 0) >= 3:
                equity_curve.append(equity); times.append(ts); continue

            risk = equity * RISK_FRACTION
            size = risk / (ATR_MULT_SL * atr)
            if (size * close) / LEVERAGE > equity:
                equity_curve.append(equity); times.append(ts); continue

            side = "buy" if vw == 1 else "sell"
            trades.append({
                "time": ts, "time_ist": ts_ist, "side": side,
                "entry": close,
                "tp": close + ATR_MULT_TP * atr if side == "buy" else close - ATR_MULT_TP * atr,
                "sl": close - ATR_MULT_SL * atr if side == "buy" else close + ATR_MULT_SL * atr,
                "size": size, "status": "open"
            })
            trades_today[day] = trades_today.get(day, 0) + 1

        equity_curve.append(equity); times.append(ts)

    pd.DataFrame(signal_rows).to_csv(BACKTEST_SIGNALS_CSV, index=False)

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, trades[0].keys())
        w.writeheader()
        w.writerows(trades)

    plt.figure(figsize=(10,5))
    plt.plot(times, equity_curve)
    plt.title(f"Equity Curve - {PRODUCT_SYMBOL}")
    plt.grid(True)
    plt.savefig(EQUITY_CURVE_IMG)

    print(f"Closed trades: {len([t for t in trades if t['status'] in ('tp','sl')])}")
    print(f"Final equity: {equity:.2f}")

if __name__ == "__main__":
    run_backtest()
