# pesudotest-1.py
# Indicator-only pseudo backtest (NO sentiment, sane execution)

import csv
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

# ----------------- FILES -----------------
CANDLES_CSV = require_file("candles-data/candles.csv")

OUTPUT_CSV = "pesudotest_results.csv"
SIGNALS_CSV = "pesudotest_signals.csv"
EQUITY_CURVE_IMG = "pesudotest_equity_curve.png"

# ----------------- PARAMS -----------------
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
SESSION_END_IST = 0

# ----------------- HELPERS -----------------
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

# ----------------- CORE -----------------
def run_pseudotest():

    df = pd.read_csv(CANDLES_CSV, parse_dates=["time_utc"])
    df = df.sort_values("time_utc").reset_index(drop=True)

    df["time"] = df["time_utc"]
    df["time_ist"] = df["time"].dt.tz_convert("Asia/Kolkata")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # -------- INDICATORS (PARITY) --------
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

    # -------- BACKTEST LOOP --------
    equity = START_EQUITY
    trade = None
    trades = []
    equity_curve, times = [], []
    signal_rows = []

    for i in range(len(df) - 1):
        r = df.iloc[i]
        nxt = df.iloc[i + 1]

        ts, ts_ist = r["time"], r["time_ist"]
        close, high, low = r["close"], r["high"], r["low"]

        signal_rows.append({
            "time": ts,
            "close": close,
            "atr": r["atr"],
            "adx": r["adx"],
            "vwma_signal": r["vwma_signal"],
            "rvi_signal": r["rvi_signal"],
            "vp_node": r["vp_node"]
        })

        # -------- EXIT (ONLY AFTER ACTIVATION) --------
        if trade and trade["active"]:
            hit = None
            if trade["side"] == "buy":
                if low <= trade["sl"]:
                    hit = trade["sl"]
                elif high >= trade["tp"]:
                    hit = trade["tp"]
            else:
                if high >= trade["sl"]:
                    hit = trade["sl"]
                elif low <= trade["tp"]:
                    hit = trade["tp"]

            if hit:
                pnl = ((hit - trade["entry"]) if trade["side"] == "buy"
                       else (trade["entry"] - hit)) * trade["size"]
                fees = ROUND_TRIP_FEE * hit * trade["size"]
                equity += pnl - fees
                trade.update({
                    "exit": hit,
                    "exit_time": ts,
                    "pnl": pnl - fees,
                    "equity": equity,
                    "status": "closed"
                })
                trades.append(trade)
                trade = None
                equity_curve.append(equity)
                times.append(ts)
                continue

        # -------- ACTIVATE TRADE (NEXT CANDLE) --------
        if trade and not trade["active"]:
            trade["active"] = True

        if trade:
            equity_curve.append(equity)
            times.append(ts)
            continue

        # -------- SESSION FILTER --------
        start = ts_ist.replace(hour=SESSION_START_IST, minute=0, second=0)
        end = ts_ist.replace(hour=23, minute=59, second=59) if SESSION_END_IST == 0 \
            else ts_ist.replace(hour=SESSION_END_IST, minute=0, second=0)

        if not (start <= ts_ist <= end):
            equity_curve.append(equity)
            times.append(ts)
            continue

        if i < 30 or r["adx"] <= ADX_THRESH or r["atr"] / close < ATR_PCT_MIN:
            equity_curve.append(equity)
            times.append(ts)
            continue

        vw, rv, vp = r["vwma_signal"], r["rvi_signal"], r["vp_node"]
        vp_ok = (close > vp) if vw == 1 else (close < vp)

        if vw != 0 and ((rv == vw) or vp_ok):
            risk = equity * RISK_FRACTION
            size = int(risk / (ATR_MULT_SL * r["atr"]))
            if size < 1:
                equity_curve.append(equity)
                times.append(ts)
                continue

            if (size * close) / LEVERAGE > equity:
                equity_curve.append(equity)
                times.append(ts)
                continue

            side = "buy" if vw == 1 else "sell"
            trade = {
                "entry_time": nxt["time"],
                "side": side,
                "entry": nxt["open"],   # NEXT candle open
                "tp": nxt["open"] + ATR_MULT_TP * r["atr"] if side == "buy"
                      else nxt["open"] - ATR_MULT_TP * r["atr"],
                "sl": nxt["open"] - ATR_MULT_SL * r["atr"] if side == "buy"
                      else nxt["open"] + ATR_MULT_SL * r["atr"],
                "size": size,
                "active": False,
                "status": "open"
            }

        equity_curve.append(equity)
        times.append(ts)

    pd.DataFrame(signal_rows).to_csv(SIGNALS_CSV, index=False)
    pd.DataFrame(trades).to_csv(OUTPUT_CSV, index=False)

    plt.figure(figsize=(10,5))
    plt.plot(times, equity_curve)
    plt.grid(True)
    plt.savefig(EQUITY_CURVE_IMG)

    print(f"Closed trades: {len(trades)}")
    print(f"Final equity: {equity:.2f}")

# ----------------- ENTRY -----------------
if __name__ == "__main__":
    run_pseudotest()
