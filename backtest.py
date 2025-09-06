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


def supertrend(df, period=10, multiplier=3):
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


def run_backtest(days=365, start_equity=1000.0, risk_per_trade=10.0):
    client = DeltaClient()

    end = int(time.time())
    start = end - days * 24 * 3600

    # 1h candles for main logic
    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start, end)
    frates = fetch_chunked_funding(client, PRODUCT_SYMBOL, "1h", start, end)

    # Build DataFrame (1h)
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
    df["supertrend"] = supertrend(df, period=10, multiplier=3)

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

        closes.append(close)
        vols.append(vol)

        # --- Manage open trade ---
        if trades and trades[-1][4] == "open":
            last = trades[-1]
            entry, side = last[2], last[1]
            tp, sl, partial, hit_partial = last[7], last[8], last[9], last[10]

            # Partial exit at 1 ATR (30%)
            if not hit_partial:
                if side == "buy" and close >= partial:
                    pnl = (partial - entry) / entry * risk_per_trade * 0.3
                    equity += pnl
                    last[6] = equity
                    last[10] = True
                    last[8] = entry  # move SL to breakeven
                elif side == "sell" and close <= partial:
                    pnl = (entry - partial) / entry * risk_per_trade * 0.3
                    equity += pnl
                    last[6] = equity
                    last[10] = True
                    last[8] = entry

            # Exit conditions
            exit_price, result = None, None
            if side == "buy":
                if close >= tp:
                    exit_price, result = tp, "tp"
                elif close <= last[8]:
                    exit_price, result = last[8], "sl"
                elif st_sig == -1:  # supertrend flip
                    exit_price, result = close, "st_exit"
            else:
                if close <= tp:
                    exit_price, result = tp, "tp"
                elif close >= last[8]:
                    exit_price, result = last[8], "sl"
                elif st_sig == 1:  # supertrend flip
                    exit_price, result = close, "st_exit"

            if result:
                # scale PnL: if partial hit, remaining 70%; else full
                size = 0.7 if hit_partial else 1.0
                pnl = (exit_price - entry) if side == "buy" else (entry - exit_price)
                pnl_usd = (pnl / entry) * risk_per_trade * size
                equity += pnl_usd
                last[3] = exit_price
                last[4] = result
                last[5] = pnl_usd
                last[6] = equity

        # Skip if trade still open
        if trades and trades[-1][4] == "open":
            equity_curve.append(equity)
            times.append(ts)
            continue

        # --- Entry filters ---
        if i < 30 or pd.isna(vol_ma) or pd.isna(adx) or pd.isna(atr):
            equity_curve.append(equity)
            times.append(ts)
            continue
        if vol <= vol_ma or adx <= 20:
            equity_curve.append(equity)
            times.append(ts)
            continue

        # Indicators
        e9, e21, v3 = ema(closes, 9), ema(closes, 21), vwma(closes, vols, 3)
        momentum = roc(closes, 3)

        ind_sig = ema_cross_signal(closes, e9, e21, v3)
        mom_sig = momentum_signal(momentum[-1], e9)
        lookback_rates = frates[max(0, i-24):i] if frates else []
        f_sig = funding_signal(lookback_rates) if lookback_rates else 0

        votes = [s for s in (mom_sig, f_sig) if s != 0]
        dirn = 1 if votes.count(1) > votes.count(-1) else -1 if votes else 0
        match_rate = len([v for v in votes if v == dirn]) / len(votes) if votes else 0

        # Entry condition (bias filter removed)
        if ind_sig and ind_sig == dirn and match_rate >= 0.5 and st_sig == ind_sig:
            day = ts.date()
            trades_today[day] = trades_today.get(day, 0)
            if trades_today[day] >= 2:
                equity_curve.append(equity)
                times.append(ts)
                continue

            side = "buy" if ind_sig == 1 else "sell"
            entry = close
            sl = entry - 1.5 * atr if side == "buy" else entry + 1.5 * atr
            tp = entry + 2 * atr if side == "buy" else entry - 2 * atr
            partial = entry + 1 * atr if side == "buy" else entry - 1 * atr
            trades.append([ts, side, entry, None, "open", 0.0, equity, tp, sl, partial, False])
            trades_today[day] += 1

        equity_curve.append(equity)
        times.append(ts)

    # Save trades
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["time", "side", "entry", "exit", "result", "pnl_usd", "equity", "tp", "sl", "partial", "partial_hit"]
        )
        writer.writerows(trades)

    # Metrics
    closed_trades = [t for t in trades if t[4] in ("tp", "sl", "st_exit")]
    wins = [t for t in closed_trades if t[4] == "tp"]
    losses = [t for t in closed_trades if t[4] == "sl"]
    others = [t for t in closed_trades if t[4] == "st_exit"]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0
    avg_win = sum(t[5] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t[5] for t in losses) / len(losses) if losses else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    print(f"Trades: {len(closed_trades)} | Win rate: {win_rate:.2%} | Expectancy: {expectancy:.2f} USD")
    print(f"Final equity: {equity:.2f}")

    # Equity curve plot
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
