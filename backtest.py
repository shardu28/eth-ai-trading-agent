# backtest.py
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ta
import pytz

from delta_client import DeltaClient
from indicators import ema, vwma, roc
from features import ema_cross_signal
from sentiment import funding_signal, momentum_signal
from config import PRODUCT_SYMBOL

OUTPUT_CSV = "backtest_results.csv"


def fetch_chunked_candles(client, symbol, resolution, start, end, chunk_days=90):
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
        time.sleep(0.2)
    all_candles.sort(key=lambda r: r["time"])
    return all_candles


def compute_averaged_vwma_from_df(df, avg_window=5):
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


def run_backtest(days=365, start_equity=1000.0,
                 atr_mult_sl=1.5, atr_mult_tp=2.5,
                 adx_thresh=25, avg_window=5, vp_window=50, rvi_period=10,
                 risk_fraction=0.01, session_start=9, session_end=22):
    client = DeltaClient()

    end = int(time.time())
    start = end - days * 24 * 3600

    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start, end)

    # Build DataFrame
    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # Indicators
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()
    df["avg_close"], df["avg_vwma"] = compute_averaged_vwma_from_df(df, avg_window=avg_window)
    closes = df["close"].values
    vols = df["volume"].values
    df["vp_node"] = compute_volume_profile_node_array(closes, vols, window=vp_window)
    df["rvi"] = (df["close"] - df["open"]).rolling(window=rvi_period, min_periods=1).mean() / \
                ((df["high"] - df["low"]).rolling(window=rvi_period, min_periods=1).mean().abs() + 1e-9)
    df["rvi_scale"] = 50 * (1 + df["rvi"].clip(-1, 1))
    df["rvi_signal"] = df["rvi_scale"].apply(lambda x: 1 if x > 50 else -1)
    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1

    equity = start_equity
    trades = []
    equity_curve, times = [], []

    for i, row in df.iterrows():
        ts = row["time"]
        ts_ist = ts.tz_convert("Asia/Kolkata")

        # ---- IST session filter ----
        if ts_ist.hour < session_start or ts_ist.hour >= session_end:
            equity_curve.append(equity)
            times.append(ts)
            continue

        close, atr, adx = row["close"], row["atr"], row["adx"]
        vwma_sig, rvi_sig, vp_node = row["vwma_signal"], row["rvi_signal"], row["vp_node"]

        # Active trade management
        if trades and trades[-1][4] == "open":
            last = trades[-1]
            entry, side, tp, sl = last[2], last[1], last[7], last[8]
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
                pnl = (exit_price - entry) if side == "buy" else (entry - exit_price)
                risk_capital = equity * risk_fraction
                pnl_usd = (pnl / entry) * (risk_capital / (abs(entry - sl)) if abs(entry - sl) > 0 else 1)
                equity += pnl_usd
                trades[-1][3] = exit_price
                trades[-1][4] = result
                trades[-1][5] = pnl_usd
                trades[-1][6] = equity

        # Skip if trade is still open
        if trades and trades[-1][4] == "open":
            equity_curve.append(equity)
            times.append(ts)
            continue

        # Filters
        if i < 30 or pd.isna(adx) or pd.isna(atr):
            equity_curve.append(equity)
            times.append(ts)
            continue
        if adx <= adx_thresh:
            equity_curve.append(equity)
            times.append(ts)
            continue

        # Voting logic
        vp_ok = False
        if not np.isnan(vp_node):
            vp_ok = close > vp_node if vwma_sig == 1 else close < vp_node if vwma_sig == -1 else False

        final_dir = 0
        if vwma_sig != 0:
            votes = 0
            votes += 1 if rvi_sig == vwma_sig else 0
            votes += 1 if vp_ok else 0
            if votes >= 1:
                final_dir = vwma_sig

        if final_dir != 0:
            side = "buy" if final_dir == 1 else "sell"
            entry = close
            sl = entry - atr_mult_sl * atr if side == "buy" else entry + atr_mult_sl * atr
            tp = entry + atr_mult_tp * atr if side == "buy" else entry - atr_mult_tp * atr

            risk_capital = equity * risk_fraction
            size = risk_capital / (abs(entry - sl)) if abs(entry - sl) > 0 else 0

            trades.append([ts, side, entry, None, "open", 0.0, equity, tp, sl, size])

        equity_curve.append(equity)
        times.append(ts)

    # Save trades
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "side", "entry", "exit", "result", "pnl_usd", "equity", "tp", "sl", "size"])
        writer.writerows(trades)

    closed_trades = [t for t in trades if t[4] in ("tp", "sl")]
    wins = [t for t in closed_trades if t[4] == "tp"]
    losses = [t for t in closed_trades if t[4] == "sl"]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0
    avg_win = sum(t[5] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t[5] for t in losses) / len(losses) if losses else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    print(f"Trades: {len(closed_trades)} | Win rate: {win_rate:.2%} | Expectancy: {expectancy:.2f} USD")
    print(f"Final equity: {equity:.2f}")

    plt.figure(figsize=(10, 5))
    plt.plot(times, equity_curve, label="Equity")
    plt.title("Equity Curve (Backtest) — 9 AM to 10 PM IST")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.show()


if __name__ == "__main__":
    run_backtest()
