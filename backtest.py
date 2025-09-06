# backtest.py
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import ta
import numpy as np

from delta_client import DeltaClient
from indicators import ema, vwma, roc
from features import ema_cross_signal
from sentiment import funding_signal, momentum_signal
from config import PRODUCT_SYMBOL

OUTPUT_CSV = "backtest_results.csv"

# --- New helpers for Averaged VWAP, Volume Profile, and RVI ---

def compute_averaged_vwap(df, vwap_window=20, avg_window=3):
    vwap = (df["close"] * df["volume"]).rolling(window=vwap_window, min_periods=1).sum() / \
           df["volume"].rolling(window=vwap_window, min_periods=1).sum()
    avg_close = df["close"].rolling(window=avg_window, min_periods=1).mean()
    avg_vwap = vwap.rolling(window=avg_window, min_periods=1).mean()
    return avg_close, avg_vwap

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

# --- Candle fetchers ---

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

def supertrend(df, period=10, multiplier=2):
    hl2 = (df["high"] + df["low"]) / 2
    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=period
    ).average_true_range()
    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr
    st = pd.Series(index=df.index, dtype=float)
    direction = 1
    for i in range(len(df)):
        if df["close"].iloc[i] > upperband.iloc[i - 1] if i > 0 else True:
            direction = 1
        elif df["close"].iloc[i] < lowerband.iloc[i - 1] if i > 0 else False:
            direction = -1
        st.iloc[i] = 1 if direction == 1 else -1
    return st

# --- Main backtest ---

def run_backtest(days=365, start_equity=1000.0, risk_per_trade=10.0):
    client = DeltaClient()

    end = int(time.time())
    start = end - days * 24 * 3600

    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start, end)
    candles_4h = fetch_chunked_candles(client, PRODUCT_SYMBOL, "4h", start, end)
    frates = fetch_chunked_funding(client, PRODUCT_SYMBOL, "1h", start, end)

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()
    df["vol_ma"] = df["volume"].rolling(20).mean()
    df["supertrend"] = supertrend(df)

    # --- New indicator columns ---
    df["avg_close"], df["avg_vwap"] = compute_averaged_vwap(df, vwap_window=20, avg_window=3)
    df["vp_node"] = compute_volume_profile_node(df, window=50, price_precision=2)
    df["rvi"] = compute_rvi_simple(df, period=10)
    df["vwav_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwap"], "vwav_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwap"], "vwav_signal"] = -1
    df["rvi_signal"] = (df["rvi"] > 50).astype(int).replace({0: -1})

    # 4h EMA50 slope for bias
    df4 = pd.DataFrame(candles_4h)
    df4["time"] = pd.to_datetime(df4["time"], unit="s")
    df4["close"] = df4["close"].astype(float)
    df4["ema50"] = df4["close"].ewm(span=50, adjust=False).mean()
    df4["ema50_slope"] = df4["ema50"].diff()
    bias_map = {}
    for i, row in df4.iterrows():
        bias_map[row["time"]] = 1 if row["ema50_slope"] > 0 else -1
    df["bias"] = df["time"].apply(lambda t: bias_map.get(max([bt for bt in bias_map if bt <= t]), 0) if any(bt <= t for bt in bias_map) else 0)

    equity = start_equity
    trades = []
    equity_curve, times = [], []

    closes, vols = [], []
    trades_today = {}

    for i, row in df.iterrows():
        ts = row["time"]
        close = row["close"]
        vol = row["volume"]
        atr = row["atr"]
        adx = row["adx"]
        vol_ma = row["vol_ma"]
        st_sig = row["supertrend"]
        bias = row["bias"]
        vwav_sig = row["vwav_signal"]
        rvi_sig = row["rvi_signal"]
        vp_node = row["vp_node"]

        closes.append(close)
        vols.append(vol)

        # check exits
        if trades and trades[-1][4] == "open":
            last = trades[-1]
            entry, side = last[2], last[1]
            tp1, tp2, sl = last[7], last[9], last[8]
            if side == "buy":
                if close >= tp1 and last[10] == 0:
                    pnl = (tp1 - entry) / entry * risk_per_trade * 0.4
                    equity += pnl
                    trades[-1][5] += pnl
                    trades[-1][6] = equity
                    trades[-1][10] = 1
                elif close >= tp2:
                    pnl = (tp2 - entry) / entry * risk_per_trade * 0.6
                    equity += pnl
                    trades[-1][3], trades[-1][4], trades[-1][5], trades[-1][6] = tp2, "tp", trades[-1][5] + pnl, equity
                elif close <= sl:
                    pnl = (sl - entry) / entry * risk_per_trade
                    equity += pnl
                    trades[-1][3], trades[-1][4], trades[-1][5], trades[-1][6] = sl, "sl", trades[-1][5] + pnl, equity
            else:
                if close <= tp1 and last[10] == 0:
                    pnl = (entry - tp1) / entry * risk_per_trade * 0.4
                    equity += pnl
                    trades[-1][5] += pnl
                    trades[-1][6] = equity
                    trades[-1][10] = 1
                elif close <= tp2:
                    pnl = (entry - tp2) / entry * risk_per_trade * 0.6
                    equity += pnl
                    trades[-1][3], trades[-1][4], trades[-1][5], trades[-1][6] = tp2, "tp", trades[-1][5] + pnl, equity
                elif close >= sl:
                    pnl = (entry - sl) / entry * risk_per_trade
                    equity += pnl
                    trades[-1][3], trades[-1][4], trades[-1][5], trades[-1][6] = sl, "sl", trades[-1][5] + pnl, equity

        if trades and trades[-1][4] == "open":
            equity_curve.append(equity)
            times.append(ts)
            continue

        # filters
        if i < 30 or pd.isna(vol_ma) or pd.isna(adx) or pd.isna(atr) or pd.isna(vp_node):
            equity_curve.append(equity)
            times.append(ts)
            continue
        if vol <= vol_ma or adx <= 25:
            equity_curve.append(equity)
            times.append(ts)
            continue

        # indicators
        e9 = ema(closes, 9)
        e21 = ema(closes, 21)
        v3 = vwma(closes, vols, 3)
        momentum = roc(closes, 3)

        ind_sig = ema_cross_signal(closes, e9, e21, v3)
        mom_sig = momentum_signal(momentum[-1], e9)
        lookback_rates = frates[max(0, i-24):i] if frates else []
        f_sig = funding_signal(lookback_rates) if lookback_rates else 0

        votes = [s for s in (mom_sig, f_sig) if s != 0]
        dirn = 1 if votes.count(1) > votes.count(-1) else -1 if votes else 0
        match_rate = len([v for v in votes if v == dirn]) / len(votes) if votes else 0

        # --- Extra confirmation: VWAP + RVI + VP breakout ---
        vp_ok = (ind_sig == 1 and close > vp_node) or (ind_sig == -1 and close < vp_node)
        extra_confirmation = (vwav_sig == ind_sig) and (rvi_sig == ind_sig) and vp_ok

        if ind_sig and ind_sig == dirn and match_rate >= 0.5 and st_sig == ind_sig and bias == ind_sig and extra_confirmation:
            day = ts.date()
            trades_today[day] = trades_today.get(day, 0)
            if trades_today[day] >= 2:
                equity_curve.append(equity)
                times.append(ts)
                continue
            side = "buy" if ind_sig == 1 else "sell"
            entry = close
            sl = entry - 1.5 * atr if side == "buy" else entry + 1.5 * atr
            tp1 = entry + 1.2 * atr if side == "buy" else entry - 1.2 * atr
            tp2 = entry + 2.5 * atr if side == "buy" else entry - 2.5 * atr
            trades.append([ts, side, entry, None, "open", 0.0, equity, tp1, sl, tp2, 0])
            trades_today[day] += 1

        equity_curve.append(equity)
        times.append(ts)

    # save trades
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "side", "entry", "exit", "result", "pnl_usd", "equity", "tp1", "sl", "tp2", "partial"])
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
    plt.title("Equity Curve (Backtest)")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.show()

if __name__ == "__main__":
    run_backtest()
