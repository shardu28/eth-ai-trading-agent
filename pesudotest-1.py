# pesudotest-1.py
# Pseudo backtest: Candles + Sentiment (read-only)

import csv
from datetime import datetime
import pandas as pd
import ta
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- PATH RESOLUTION -----------------
REPO_ROOT = Path(__file__).resolve().parent.parent

def require_file(relative_path: str) -> Path:
    path = REPO_ROOT / relative_path
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path

# ----------------- CONFIG -----------------
CANDLES_CSV = require_file("candles-data/candles.csv")
SENTIMENT_CSV = require_file("sentiment-data/sentiment.csv")

OUTPUT_CSV = "pesudotest_results.csv"
EQUITY_CURVE_IMG = "pesudotest_equity_curve.png"
SIGNALS_CSV = "pesudotest_signals.csv"

START_EQUITY = 100.0
LEVERAGE = 20.0
ROUND_TRIP_FEE = 0.001

ATR_MULT_SL = 1.3
ATR_MULT_TP = 3.7
ADX_THRESH = 29
AVG_WINDOW = 5
VP_WINDOW = 50
RVI_PERIOD = 10
RISK_FRACTION = 0.03
ATR_PCT_MIN = 0.005

SESSION_START_IST = 10
SESSION_END_IST = 0   # 0 = midnight cutoff, not "whole day"

SENTIMENT_WINDOW = 5

SENTIMENT_START = pd.Timestamp("2025-12-28 06:36:46.525457+00:00")
SENTIMENT_END   = pd.Timestamp("2026-01-25 23:46:30.606611+00:00")

CANDLE_START = pd.Timestamp("2024-12-28 07:00:00+00:00")
CANDLE_END   = pd.Timestamp("2026-01-26 00:00:00+00:00")

# ----------------- Helpers -----------------
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

# ----------------- Core Logic -----------------
def run_pseudotest():

    candles = pd.read_csv(CANDLES_CSV, parse_dates=["time_utc"])
    sentiment = pd.read_csv(SENTIMENT_CSV, parse_dates=["run_time_utc"])

    candles = candles[(candles["time_utc"] >= CANDLE_START) &
                      (candles["time_utc"] <= CANDLE_END)].copy()

    sentiment = sentiment[(sentiment["run_time_utc"] >= SENTIMENT_START) &
                          (sentiment["run_time_utc"] <= SENTIMENT_END)].copy()

    candles["time_ist"] = candles["time_utc"].dt.tz_convert("Asia/Kolkata")

    # -------- SENTIMENT ALIGNMENT (FIXED) --------
    candles["hour"] = candles["time_utc"].dt.floor("1h")
    sentiment["hour"] = sentiment["run_time_utc"].dt.floor("1h")

    sent_agg = (
        sentiment
        .groupby("hour", as_index=False)["sentiment_score"]
        .mean()
        .assign(
            sentiment_mean=lambda x:
            x["sentiment_score"]
            .rolling(SENTIMENT_WINDOW, min_periods=1)
            .mean()
        )
        .drop(columns="sentiment_score")
    )

    candles = candles.merge(sent_agg, on="hour", how="left")
    candles["sentiment_mean"] = candles["sentiment_mean"].fillna(0.0)

    if candles["sentiment_mean"].abs().sum() == 0:
        raise RuntimeError("Sentiment merge failed: all sentiment values are zero")

    # -------- Indicators --------
    for c in ["open", "high", "low", "close", "volume"]:
        candles[c] = candles[c].astype(float)

    candles["atr"] = ta.volatility.AverageTrueRange(
        candles["high"], candles["low"], candles["close"], 14
    ).average_true_range()

    candles["adx"] = ta.trend.ADXIndicator(
        candles["high"], candles["low"], candles["close"], 14
    ).adx()

    candles["avg_close"] = candles["close"].rolling(AVG_WINDOW, 1).mean()
    candles["avg_vwma"] = compute_avg_vwma(candles["close"], candles["volume"], AVG_WINDOW)

    candles["vwma_signal"] = 0
    candles.loc[candles["avg_close"] > candles["avg_vwma"], "vwma_signal"] = 1
    candles.loc[candles["avg_close"] < candles["avg_vwma"], "vwma_signal"] = -1

    candles["rvi"] = rvi_approx(
        candles["close"], candles["open"],
        candles["high"], candles["low"], RVI_PERIOD
    )
    mean_rvi = candles["rvi"].mean()
    candles["rvi_signal"] = candles["rvi"].apply(lambda x: 1 if x > mean_rvi else -1)

    candles["vp_node"] = compute_vp_node(candles["close"], VP_WINDOW)

    # -------- Backtest Loop --------
    equity = START_EQUITY
    trades = []
    equity_curve, times = [], []
    signal_rows = []

    for i, r in candles.iterrows():
        ts, ts_ist = r["time_utc"], r["time_ist"]
        hour = ts_ist.hour

        # -------- SESSION FILTER (FIXED) --------
        if SESSION_END_IST == 0:
            in_session = hour >= SESSION_START_IST
        elif SESSION_START_IST < SESSION_END_IST:
            in_session = SESSION_START_IST <= hour < SESSION_END_IST
        else:
            in_session = hour >= SESSION_START_IST or hour < SESSION_END_IST

        if not in_session:
            equity_curve.append(equity)
            times.append(ts)
            continue

        close, atr, adx = r["close"], r["atr"], r["adx"]
        vw, rv, vp = r["vwma_signal"], r["rvi_signal"], r["vp_node"]
        sent = r["sentiment_mean"]

        signal_rows.append({
            "time": ts,
            "close": close,
            "vwma": vw,
            "rvi": rv,
            "sentiment": sent
        })

        if i < 30 or adx <= ADX_THRESH or atr / close < ATR_PCT_MIN:
            equity_curve.append(equity)
            times.append(ts)
            continue

        if vw == 1 and sent <= 0:
            continue
        if vw == -1 and sent >= 0:
            continue

        vp_ok = (close > vp) if vw == 1 else (close < vp)
        if vw != 0 and ((rv == vw) or vp_ok):
            risk = equity * RISK_FRACTION
            size = int(risk / (ATR_MULT_SL * atr))
            if size < 1:
                continue
            if (size * close) / LEVERAGE > equity:
                continue

            side = "buy" if vw == 1 else "sell"
            trades.append({
                "time": ts,
                "side": side,
                "entry": close,
                "tp": close + ATR_MULT_TP * atr if side == "buy" else close - ATR_MULT_TP * atr,
                "sl": close - ATR_MULT_SL * atr if side == "buy" else close + ATR_MULT_SL * atr,
                "size": size,
                "status": "open"
            })

        equity_curve.append(equity)
        times.append(ts)

    pd.DataFrame(trades).to_csv(OUTPUT_CSV, index=False)
    pd.DataFrame(signal_rows).to_csv(SIGNALS_CSV, index=False)

    plt.figure(figsize=(10, 5))
    plt.plot(times, equity_curve)
    plt.grid(True)
    plt.savefig(EQUITY_CURVE_IMG)

    print("Trades:", len(trades))
    print(f"Final equity: {equity:.2f}")
    print(f"Return %: {((equity / START_EQUITY) - 1) * 100:.2f}")

# ----------------- Entry Point -----------------
def main():
    run_pseudotest()

if __name__ == "__main__":
    main()
