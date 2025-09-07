# backtest.py  -- 5-year final backtest with your chosen params
import csv
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ta

from delta_client import DeltaClient
from indicators import ema, vwma, roc
from features import ema_cross_signal
from sentiment import funding_signal, momentum_signal
from config import PRODUCT_SYMBOL

OUTPUT_CSV = "backtest_results.csv"


def fetch_chunked_candles(client, symbol, resolution, start, end, chunk_days=30):
    """
    Fetch historical candles in chunks of `chunk_days` length.
    This protects us from API range limits and request throttling.
    """
    all_candles = []
    cur_start = start
    polite_delay = 0.25  # seconds between requests to be polite to the API
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
        time.sleep(polite_delay)
    all_candles.sort(key=lambda r: r["time"])
    return all_candles


def fetch_chunked_funding(client, symbol, resolution, start, end, chunk_days=30):
    all_rates = []
    cur_start = start
    polite_delay = 0.25
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
        time.sleep(polite_delay)
    return all_rates


def compute_averaged_vwma(df, avg_window=3):
    avg_close = df["close"].rolling(window=avg_window, min_periods=1).mean()
    avg_vwma = (df["close"] * df["volume"]).rolling(window=avg_window, min_periods=1).sum() / \
               df["volume"].rolling(window=avg_window, min_periods=1).sum()
    return avg_close, avg_vwma


def compute_volume_profile_node(df, window=50, price_precision=2):
    n = len(df)
    nodes = [np.nan] * n
    closes = df["close"].values
    vols = df["volume"].values
    for i in range(n):
        if i < window - 1:
            nodes[i] = np.nan
            continue
        start = i - window + 1
        prices = np.round(closes[start:i+1].astype(float), price_precision)
        volumes = vols[start:i+1].astype(float)
        uniq_prices, idxs = np.unique(prices, return_inverse=True)
        vol_sums = np.zeros(len(uniq_prices), dtype=float)
        np.add.at(vol_sums, idxs, volumes)
        max_idx = int(np.argmax(vol_sums))
        nodes[i] = float(uniq_prices[max_idx])
    return pd.Series(nodes, index=df.index)


def compute_rvi_simple(df, period=10):
    eps = 1e-9
    num = (df["close"] - df["open"]).rolling(window=period, min_periods=1).mean()
    den = (df["high"] - df["low"]).rolling(window=period, min_periods=1).mean().abs() + eps
    raw = num / den
    clipped = raw.clip(-1.0, 1.0)
    rvi = 50 * (1 + clipped)  # maps -1..1 -> 0..100
    return rvi


def run_backtest(days=1825, start_equity=1000.0,
                 atr_mult_sl=1.5, atr_mult_tp=2.5, adx_thresh=25,
                 avg_window=5, vp_window=50, rvi_period=10,
                 risk_fraction=0.01, session_start=9, session_end=22,
                 chunk_days=30):
    """
    Run a single backtest over `days` days (default 5 years).
    The indicator/risk/session params are provided and hardcoded where you requested.
    chunk_days controls how large each candle fetch chunk is.
    """
    client = DeltaClient()

    # time range
    end = int(time.time())
    start = end - int(days * 24 * 3600)

    # fetch data in safe chunks
    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start, end, chunk_days=chunk_days)
    candles_4h = fetch_chunked_candles(client, PRODUCT_SYMBOL, "4h", start, end, chunk_days=chunk_days)
    frates = fetch_chunked_funding(client, PRODUCT_SYMBOL, "1h", start, end, chunk_days=chunk_days)

    if not candles:
        print("No candles returned. Aborting.")
        return

    df = pd.DataFrame(candles)
    # utc -> keep tz-naive for backtest but interpret epoch seconds
    df["time"] = pd.to_datetime(df["time"], unit="s")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # time filter: create an IST-based column for session filtering
    df["time_utc"] = pd.to_datetime(df["time"], unit=None, utc=True)
    df["time_ist"] = df["time_utc"].dt.tz_convert("Asia/Kolkata")
    df["hour_ist"] = df["time_ist"].dt.hour

    # indicators and features
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()
    df["vol_ma"] = df["volume"].rolling(20).mean()

    # VWMA averaged + VP + RVI (as used in your validated config)
    df["avg_close"], df["avg_vwma"] = compute_averaged_vwma(df, avg_window=avg_window)
    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1

    df["vp_node"] = compute_volume_profile_node(df, window=vp_window)
    df["rvi"] = compute_rvi_simple(df, period=rvi_period)
    df["rvi_signal"] = (df["rvi"] > 50).astype(int).replace({0: -1})

    # 4h bias map (EMA50 slope)
    df4 = pd.DataFrame(candles_4h)
    df4["time"] = pd.to_datetime(df4["time"], unit="s")
    df4["close"] = df4["close"].astype(float)
    df4["ema50"] = df4["close"].ewm(span=50, adjust=False).mean()
    df4["ema50_slope"] = df4["ema50"].diff().fillna(0)
    bias_map = {}
    for i, row in df4.iterrows():
        bias_map[row["time"]] = 1 if row["ema50_slope"] > 0 else -1
    # map bias to 1h rows safely
    sorted_bias_times = sorted(bias_map.keys())
    def get_bias(ts):
        # find the most recent bias_time <= ts
        candidates = [bt for bt in sorted_bias_times if bt <= ts]
        if not candidates:
            return 0
        return bias_map[max(candidates)]
    df["bias"] = df["time"].apply(get_bias)

    # Backtest loop variables
    equity = start_equity
    trades = []
    equity_curve = []
    times = []

    closes = []
    vols = []
    trades_today = {}

    for i, row in df.iterrows():
        ts = row["time"]
        ts_ist = row["time_ist"]
        hour_ist = int(row["hour_ist"])
        close = row["close"]
        vol = row["volume"]
        atr = row["atr"]
        adx = row["adx"]
        vol_ma = row["vol_ma"]
        st_dummy = 0  # we are not using supertrend trailing in this final test
        bias = int(row["bias"])
        vwma_sig = int(row["vwma_signal"])
        rvi_sig = int(row["rvi_signal"])
        vp_node = row["vp_node"] if not pd.isna(row["vp_node"]) else None

        closes.append(close)
        vols.append(vol)

        # Check exits for open trade
        if trades and trades[-1][4] == "open":
            last = trades[-1]
            entry = last[2]
            side = last[1]
            tp = last[7]
            sl = last[8]
            size = last[9] if len(last) > 9 else 1.0
            exit_price, result = None, None
            if side == "buy":
                if close >= tp:
                    exit_price, result = tp, "tp"
                elif close <= sl:
                    exit_price, result = sl, "sl"
            else:
                if close <= tp:
                    exit_price, result = tp, "tp"
                elif close >= sl:
                    exit_price, result = sl, "sl"
            if result:
                pnl = (exit_price - entry) * size if side == "buy" else (entry - exit_price) * size
                equity += pnl
                trades[-1][3] = exit_price
                trades[-1][4] = result
                trades[-1][5] = pnl
                trades[-1][6] = equity

        # If there's an open trade, continue to next bar (we only have one concurrent trade)
        if trades and trades[-1][4] == "open":
            equity_curve.append(equity)
            times.append(ts)
            continue

        # Session filter (IST)
        if hour_ist < session_start or hour_ist > session_end:
            equity_curve.append(equity)
            times.append(ts)
            continue

        # Basic data sufficiency / filters
        if i < 50 or pd.isna(vol_ma) or pd.isna(adx) or pd.isna(atr):
            equity_curve.append(equity)
            times.append(ts)
            continue
        if vol <= vol_ma or adx <= adx_thresh:
            equity_curve.append(equity)
            times.append(ts)
            continue

        # Indicator voting and sentiment substitutes (we keep the same voting logic)
        ind_sig = vwma_sig
        # vp_ok if breakout relative to vp_node
        vp_ok = False
        if vp_node is not None and not pd.isna(vp_node):
            if ind_sig == 1 and close > vp_node:
                vp_ok = True
            if ind_sig == -1 and close < vp_node:
                vp_ok = True

        extra_confirmation = (rvi_sig == ind_sig) or vp_ok

        # entry condition: VWMA indicator plus extra confirmation and bias match
        if ind_sig and extra_confirmation and bias == ind_sig:
            # limit to max 2 trades per IST-day
            day = ts_ist.date()
            trades_today[day] = trades_today.get(day, 0)
            if trades_today[day] >= 2:
                equity_curve.append(equity)
                times.append(ts)
                continue

            # dynamic position sizing using ATR-based risk fraction
            risk_capital = equity * risk_fraction
            # risk per point = atr_mult_sl * atr
            if atr and atr > 0:
                size = risk_capital / (atr_mult_sl * atr)
            else:
                size = 0

            if size <= 0:
                equity_curve.append(equity)
                times.append(ts)
                continue

            side = "buy" if ind_sig == 1 else "sell"
            entry = close
            sl = entry - atr_mult_sl * atr if side == "buy" else entry + atr_mult_sl * atr
            tp = entry + atr_mult_tp * atr if side == "buy" else entry - atr_mult_tp * atr

            # trade record: [time, side, entry, exit, result, pnl, equity, tp, sl, size]
            trades.append([ts, side, entry, None, "open", 0.0, equity, tp, sl, size])
            trades_today[day] += 1

        equity_curve.append(equity)
        times.append(ts)

    # Save trades to CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "side", "entry", "exit", "result", "pnl", "equity", "tp", "sl", "size"])
        writer.writerows(trades)

    # Compute metrics
    closed_trades = [t for t in trades if t[4] in ("tp", "sl")]
    wins = [t for t in closed_trades if t[4] == "tp"]
    losses = [t for t in closed_trades if t[4] == "sl"]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0.0
    avg_win = (sum(t[5] for t in wins) / len(wins)) if wins else 0.0
    avg_loss = (sum(t[5] for t in losses) / len(losses)) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    print(f"Trades: {len(closed_trades)} | Win rate: {win_rate:.2%} | Expectancy: {expectancy:.2f} USD")
    print(f"Final equity: {equity:.2f}")

    # Equity curve plot
    plt.figure(figsize=(12, 6))
    plt.plot(times, equity_curve, label="Equity")
    plt.title(f"Equity Curve ({days/365:.1f} years backtest)")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.show()

    return {
        "trades": trades,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "final_equity": equity,
        "params": {
            "atr_mult_sl": atr_mult_sl, "atr_mult_tp": atr_mult_tp,
            "adx_thresh": adx_thresh, "avg_window": avg_window,
            "vp_window": vp_window, "rvi_period": rvi_period,
            "risk_fraction": risk_fraction, "session_start": session_start, "session_end": session_end
        }
    }


if __name__ == "__main__":
    # Hardcoded "sweet spot" params (you requested these)
    result = run_backtest(
        days=1825,
        start_equity=1000.0,
        atr_mult_sl=1.5,
        atr_mult_tp=2.5,
        adx_thresh=25,
        avg_window=5,
        vp_window=50,
        rvi_period=10,
        risk_fraction=0.01,
        session_start=9,
        session_end=22,
        chunk_days=30
    )
