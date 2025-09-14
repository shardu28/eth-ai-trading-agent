# backtest.py  (standalone for SOL backtest)
import csv
import time
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt
import pandas as pd
import ta
import math
import os

from delta_client import DeltaClient
from config import PRODUCT_SYMBOL  # ensure you've changed this to "SOLUSD" in config.py

OUTPUT_CSV = "backtest_results.csv"
EQUITY_CURVE_IMG = "equity_curve.png"

# ----------------- Backtest params (user-specified) -----------------
DAYS = 365  # 1 year
START_EQUITY = 100.0
LEVERAGE = 20.0

# new params according to your last message
ATR_MULT_SL = 1.5
ATR_MULT_TP = 2.5
ADX_THRESH = 25
AVG_WINDOW = 5
VP_WINDOW = 50
RVI_PERIOD = 10
RISK_FRACTION = 0.01  # percent of equity risked per trade (1%)

# time window in IST
SESSION_START_IST = 8   # 08:00 IST
SESSION_END_IST = 23    # 23:00 IST

# volatility filter: skip trading if ATR% (ATR / price) is below this threshold
# small, but avoids dead sessions. You can tweak later.
ATR_PCT_MIN = 0.005  # 0.5%

# API chunking
CHUNK_DAYS = 90
POLITE_DELAY = 0.2

# ----------------- helpers -----------------
def fetch_chunked_candles(client, symbol, resolution, start, end, chunk_days=CHUNK_DAYS):
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
        time.sleep(POLITE_DELAY)
    all_candles.sort(key=lambda r: r["time"])
    return all_candles

def compute_avg_vwma(series_close, series_volume, window):
    # returns pandas.Series aligned with series_close (NaN for early values)
    num = (series_close * series_volume).rolling(window=window, min_periods=1).sum()
    den = series_volume.rolling(window=window, min_periods=1).sum()
    return num / den

def compute_vp_node(series_close, window, precision=2):
    # returns a Series with the dominant price node (NaN for early rows)
    def vp_mode(x):
        if len(x) == 0:
            return float("nan")
        rounded = (x.round(precision)).astype(str)
        return float(pd.Series(rounded).mode().iloc[0])
    return series_close.rolling(window=window, min_periods=1).apply(lambda x: vp_mode(x), raw=False)

def rvi_approx(series_close, series_open, series_high, series_low, period):
    # quick rough RVI-like measure: normalized momentum / volatility (0..100-ish)
    # We'll use rolling mean of (close - open) divided by rolling mean of (high-low)
    num = (series_close - series_open).rolling(window=period, min_periods=1).mean()
    den = (series_high - series_low).rolling(window=period, min_periods=1).mean().abs() + 1e-9
    rvi = num / den
    # scale to 0..100-ish as previous code used 50*(1 + rvi)
    return 50.0 * (1 + rvi.clip(-1, 1))

# ----------------- main backtest -----------------
def run_backtest():
    client = DeltaClient()

    end_ts = int(time.time())
    start_ts = end_ts - DAYS * 24 * 3600

    print(f"Fetching {DAYS} days of 1H candles for {PRODUCT_SYMBOL}...")
    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start_ts, end_ts)
    if not candles:
        raise RuntimeError("No candle data returned from API")

    df = pd.DataFrame(candles)
    # convert time (assumes 'time' field is epoch seconds)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    # IST column for session filtering
    df["time_ist"] = df["time"].dt.tz_convert("Asia/Kolkata")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # indicators
    df["atr"] = ta.volatility.AverageTrueRange(high=df["high"], low=df["low"], close=df["close"], window=14).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"], window=14).adx()
    df["avg_close"] = df["close"].rolling(window=AVG_WINDOW, min_periods=1).mean()
    df["avg_vwma"] = compute_avg_vwma(df["close"], df["volume"], AVG_WINDOW)
    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1

    df["rvi"] = rvi_approx(df["close"], df["open"], df["high"], df["low"], RVI_PERIOD)
    mean_rvi = df["rvi"].mean()
    df["rvi_signal"] = df["rvi"].apply(lambda x: 1 if x > mean_rvi else -1)

    df["vp_node"] = compute_vp_node(df["close"], VP_WINDOW)

    # Backtest loop
    equity = START_EQUITY
    trades = []
    equity_curve = []
    times = []
    trades_today = {}

    # funding rates (optional) — fetch funding history for analysis (not used in logic here)
    try:
        frates = fetch_chunked_funding(client, PRODUCT_SYMBOL, "1h", start_ts, end_ts)
    except Exception:
        frates = []

    for i, row in df.iterrows():
        ts = row["time"]
        ts_ist = row["time_ist"]
        close = row["close"]
        atr = row["atr"]
        adx = row["adx"]
        vwma_sig = int(row["vwma_signal"]) if not pd.isna(row["vwma_signal"]) else 0
        rvi_sig = int(row["rvi_signal"]) if not pd.isna(row["rvi_signal"]) else 0
        vp_node = row["vp_node"]

        # session time filter (IST)
        hour_ist = int(ts_ist.hour)
        if hour_ist < SESSION_START_IST or hour_ist > SESSION_END_IST:
            equity_curve.append(equity); times.append(ts); continue

        # Check active trade exits
        if trades and trades[-1][4] == "open":
            last = trades[-1]
            entry = float(last[2]); side = last[1]; tp = float(last[7]); sl = float(last[8]); size = float(last[9])
            exit_price, result = None, None
            if side == "buy":
                if close >= tp: exit_price, result = tp, "tp"
                elif close <= sl: exit_price, result = sl, "sl"
            else:
                if close <= tp: exit_price, result = tp, "tp"
                elif close >= sl: exit_price, result = sl, "sl"
            if result:
                pnl = (exit_price - entry) * size if side == "buy" else (entry - exit_price) * size
                equity += pnl
                trades[-1][3] = exit_price
                trades[-1][4] = result
                trades[-1][5] = pnl
                trades[-1][6] = equity

        # if still open after exit-check, skip new entries
        if trades and trades[-1][4] == "open":
            equity_curve.append(equity); times.append(ts); continue

        # require minimal history
        if i < 30 or pd.isna(adx) or pd.isna(atr):
            equity_curve.append(equity); times.append(ts); continue

        # volatility filter (skip if ATR% below threshold)
        atr_pct = atr / close if (atr and close) else 0.0
        if atr_pct < ATR_PCT_MIN:
            equity_curve.append(equity); times.append(ts); continue

        # ADX filter
        if adx <= ADX_THRESH:
            equity_curve.append(equity); times.append(ts); continue

        # indicator + confirmations
        ind_sig = vwma_sig
        vp_ok = False
        if not pd.isna(vp_node):
            vp_ok = (close > vp_node) if ind_sig == 1 else (close < vp_node)

        extra_confirmation = (rvi_sig == ind_sig) or vp_ok
        if ind_sig != 0 and extra_confirmation:
            day = ts_ist.date()
            trades_today[day] = trades_today.get(day, 0)
            if trades_today[day] >= 2:
                equity_curve.append(equity); times.append(ts); continue

            # sizing: risk_capital = equity * risk_fraction (absolute USD)
            risk_capital = equity * RISK_FRACTION
            # size = how many units of SOL to buy such that a stop (atr_mult_sl * atr) move costs risk_capital
            denom = (ATR_MULT_SL * atr) if atr and atr > 0 else None
            if not denom or denom == 0:
                equity_curve.append(equity); times.append(ts); continue
            size = risk_capital / denom

            # margin check: required margin = (size * entry) / leverage
            required_margin = (size * close) / LEVERAGE
            if required_margin > equity:
                # can't afford margin with current equity & leverage, skip
                equity_curve.append(equity); times.append(ts); continue

            side = "buy" if ind_sig == 1 else "sell"
            entry = close
            sl = entry - ATR_MULT_SL * atr if side == "buy" else entry + ATR_MULT_SL * atr
            tp = entry + ATR_MULT_TP * atr if side == "buy" else entry - ATR_MULT_TP * atr

            trades.append([ts, side, entry, None, "open", 0.0, equity, tp, sl, size])
            trades_today[day] += 1

        equity_curve.append(equity)
        times.append(ts)

    # finalize outputs
    closed_trades = [t for t in trades if t[4] in ("tp", "sl")]
    wins = [t for t in closed_trades if t[4] == "tp"]
    losses = [t for t in closed_trades if t[4] == "sl"]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
    avg_win = sum(t[5] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t[5] for t in losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # write out trades
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time","side","entry","exit","result","pnl_usd","equity","tp","sl","size"])
        writer.writerows(trades)

    # equity curve
    plt.figure(figsize=(10,5))
    plt.plot(times, equity_curve, label="Equity")
    plt.title(f"Equity Curve - {PRODUCT_SYMBOL}")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(EQUITY_CURVE_IMG)
    print(f"Saved equity curve -> {EQUITY_CURVE_IMG}")

    print(f"Trades total: {len(trades)} | Closed trades: {len(closed_trades)} | Win rate: {win_rate:.2%} | Expectancy: {expectancy:.2f} | Final equity: {equity:.2f}")

# small helper used above
def fetch_chunked_funding(client, symbol, resolution, start, end, chunk_days=CHUNK_DAYS):
    all_rates = []
    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + chunk_days * 24 * 3600, end)
        raw = client.get("/history/candles", {
            "symbol": f"FUNDING:{symbol}",
            "resolution": resolution,
            "start": cur_start,
            "end": cur_end,
        })
        chunk = raw.get("result", [])
        all_rates.extend([float(r["close"]) for r in chunk])
        cur_start = cur_end
        time.sleep(POLITE_DELAY)
    return all_rates

if __name__ == "__main__":
    run_backtest()
