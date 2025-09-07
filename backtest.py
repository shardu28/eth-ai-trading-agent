# backtest.py
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import ta

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


def fetch_chunked_funding(client, symbol, resolution, start, end, chunk_days=90):
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
        time.sleep(0.2)
    return all_rates


def heikin_ashi(df):
    ha_df = df.copy()
    ha_df["HA_close"] = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
    ha_open = []
    for i in range(len(df)):
        if i == 0:
            ha_open.append((df["open"].iloc[0] + df["close"].iloc[0]) / 2)
        else:
            ha_open.append((ha_open[-1] + ha_df["HA_close"].iloc[i - 1]) / 2)
    ha_df["HA_open"] = ha_open
    ha_df["HA_high"] = ha_df[["high", "HA_open", "HA_close"]].max(axis=1)
    ha_df["HA_low"] = ha_df[["low", "HA_open", "HA_close"]].min(axis=1)
    return ha_df[["time", "HA_open", "HA_high", "HA_low", "HA_close", "volume"]].rename(
        columns={"HA_open": "open", "HA_high": "high", "HA_low": "low", "HA_close": "close"}
    )


def run_backtest(days=365, start_equity=1000.0,
                 atr_mult_sl=1.5, atr_mult_tp=2.5, adx_thresh=25,
                 avg_window=5, vp_window=50, rvi_period=10,
                 risk_fraction=0.01, session_start=9, session_end=22):
    client = DeltaClient()

    end = int(time.time())
    start = end - days * 24 * 3600

    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start, end)
    frates = fetch_chunked_funding(client, PRODUCT_SYMBOL, "1h", start, end)

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ist"] = df["time"].dt.tz_convert("Asia/Kolkata")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # --- convert to Heikin Ashi candles ---
    df = heikin_ashi(df)
    df["time_ist"] = df["time"].dt.tz_convert("Asia/Kolkata")

    # --- indicators ---
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()

    df["avg_close"] = df["close"].rolling(window=avg_window).mean()
    df["avg_vwma"] = (df["close"] * df["volume"]).rolling(window=avg_window).sum() / \
                     df["volume"].rolling(window=avg_window).sum()
    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1

    df["rvi"] = df["close"].pct_change().rolling(rvi_period).std()
    mean_rvi = df["rvi"].mean()
    df["rvi_signal"] = df["rvi"].apply(lambda x: 1 if x > mean_rvi else -1)

    df["vp_node"] = df["close"].rolling(vp_window).apply(
        lambda x: x.value_counts().idxmax() if len(x) > 0 else 0, raw=False
    )

    equity = start_equity
    trades, equity_curve, times = [], [], []
    trades_today = {}

    for i, row in df.iterrows():
        ts = row["time"]
        ts_ist = row["time_ist"]
        close, atr, adx = row["close"], row["atr"], row["adx"]
        vwma_sig, rvi_sig = row["vwma_signal"], row["rvi_signal"]
        vp_ok = close > row["vp_node"]

        # time filter
        if ts_ist.hour < session_start or ts_ist.hour > session_end:
            equity_curve.append(equity)
            times.append(ts)
            continue

        # exits
        if trades and trades[-1][4] == "open":
            last = trades[-1]
            entry, side, tp, sl, size = last[2], last[1], last[7], last[8], last[9]
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

        if trades and trades[-1][4] == "open":
            equity_curve.append(equity)
            times.append(ts)
            continue

        if i < 30 or pd.isna(adx) or pd.isna(atr):
            equity_curve.append(equity)
            times.append(ts)
            continue
        if adx <= adx_thresh:
            equity_curve.append(equity)
            times.append(ts)
            continue

        ind_sig = vwma_sig
        extra_confirmation = (rvi_sig == ind_sig) or vp_ok
        if ind_sig != 0 and extra_confirmation:
            day = ts_ist.date()
            trades_today[day] = trades_today.get(day, 0)
            if trades_today[day] >= 2:
                equity_curve.append(equity)
                times.append(ts)
                continue

            risk_capital = equity * risk_fraction
            size = risk_capital / (atr_mult_sl * atr) if atr > 0 else 0
            if size <= 0:
                continue

            side = "buy" if ind_sig == 1 else "sell"
            entry = close
            sl = entry - atr_mult_sl * atr if side == "buy" else entry + atr_mult_sl * atr
            tp = entry + atr_mult_tp * atr if side == "buy" else entry - atr_mult_tp * atr
            trades.append([ts, side, entry, None, "open", 0.0, equity, tp, sl, size])
            trades_today[day] += 1

        equity_curve.append(equity)
        times.append(ts)

    closed_trades = [t for t in trades if t[4] in ("tp", "sl")]
    wins = [t for t in closed_trades if t[4] == "tp"]
    losses = [t for t in closed_trades if t[4] == "sl"]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0
    avg_win = sum(t[5] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t[5] for t in losses) / len(losses) if losses else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    trades_out = trades
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time","side","entry","exit","result","pnl_usd","equity","tp","sl","size"])
        writer.writerows(trades_out)

    times, equity_curve = times, equity_curve
    plt.figure(figsize=(10,5))
    plt.plot(times, equity_curve, label="Equity")
    plt.title("Equity Curve (Heikin Ashi)")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("equity_curve.png")

    print(f"Trades: {len(trades_out)} | Win rate: {win_rate:.2%} | Expectancy: {expectancy:.2f} | Final equity: {equity:.2f}")


if __name__ == "__main__":
    run_backtest()
