# pesudotest-1.py
# Indicator-only pseudo backtest (CLOSE-ONLY exits, parity with backtest.py)

import csv
import pandas as pd
import ta
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------- PATH -----------------
REPO_ROOT = Path(__file__).resolve().parent.parent

def require_file(p):
    path = REPO_ROOT / p
    if not path.exists():
        raise FileNotFoundError(path)
    return path

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
SESSION_END_IST = 0  # midnight

# ----------------- HELPERS -----------------
def compute_avg_vwma(close, vol, w):
    return (close * vol).rolling(w, 1).sum() / vol.rolling(w, 1).sum()

def compute_vp_node(close, w, p=2):
    def vp(x):
        return float(pd.Series(x.round(p).astype(str)).mode().iloc[0])
    return close.rolling(w, 1).apply(vp, raw=False)

def rvi_approx(c, o, h, l, p):
    num = (c - o).rolling(p, 1).mean()
    den = (h - l).rolling(p, 1).mean().abs() + 1e-9
    return 50 * (1 + (num / den).clip(-1, 1))

# ----------------- CORE -----------------
def run_pseudotest():
    df = pd.read_csv(CANDLES_CSV, parse_dates=["time_utc"]).sort_values("time_utc")

    df["time"] = df["time_utc"]
    df["time_ist"] = df["time"].dt.tz_convert("Asia/Kolkata")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # -------- INDICATORS --------
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

    # -------- BACKTEST --------
    equity = START_EQUITY
    trades, equity_curve, times = [], [], []
    trades_today = {}
    signals = []

    for i, r in df.iterrows():
        ts, ts_ist = r["time"], r["time_ist"]
        close, atr, adx = r["close"], r["atr"], r["adx"]
        vw, rv, vp = r["vwma_signal"], r["rvi_signal"], r["vp_node"]

        signals.append({
            "time": ts, "close": close, "atr": atr,
            "adx": adx, "vwma": vw, "rvi": rv, "vp_node": vp
        })

        # -------- EXIT (CLOSE-ONLY, PARITY) --------
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
                res, px = hit
                pnl = ((px - t["entry"]) if t["side"] == "buy"
                       else (t["entry"] - px)) * t["size"]
                fees = ROUND_TRIP_FEE * px * t["size"]
                equity += pnl - fees
                t.update({
                    "exit": px, "exit_time": ts,
                    "pnl": pnl - fees,
                    "equity": equity,
                    "status": res
                })

        if trades and trades[-1]["status"] == "open":
            equity_curve.append(equity); times.append(ts); continue

        # -------- SESSION FILTER --------
        start = ts_ist.replace(hour=SESSION_START_IST, minute=0, second=0)
        end = ts_ist.replace(hour=23, minute=59, second=59) if SESSION_END_IST == 0 \
              else ts_ist.replace(hour=SESSION_END_IST, minute=0, second=0)

        if not (start <= ts_ist <= end):
            equity_curve.append(equity); times.append(ts); continue

        if i < 30 or adx <= ADX_THRESH or atr / close < ATR_PCT_MIN:
            equity_curve.append(equity); times.append(ts); continue

        vp_ok = (close > vp) if vw == 1 else (close < vp)
        if vw != 0 and ((rv == vw) or vp_ok):
            day = ts_ist.date()
            if trades_today.get(day, 0) >= 3:
                equity_curve.append(equity); times.append(ts); continue

            risk = equity * RISK_FRACTION
            size = int(risk / (ATR_MULT_SL * atr))
            if size < 1 or (size * close) / LEVERAGE > equity:
                equity_curve.append(equity); times.append(ts); continue

            side = "buy" if vw == 1 else "sell"
            trades.append({
                "time": ts, "time_ist": ts_ist,
                "side": side, "entry": close,
                "tp": close + ATR_MULT_TP * atr if side == "buy" else close - ATR_MULT_TP * atr,
                "sl": close - ATR_MULT_SL * atr if side == "buy" else close + ATR_MULT_SL * atr,
                "size": size, "status": "open"
            })
            trades_today[day] = trades_today.get(day, 0) + 1

        equity_curve.append(equity); times.append(ts)

    pd.DataFrame(signals).to_csv(SIGNALS_CSV, index=False)

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, trades[0].keys())
        w.writeheader()
        w.writerows(trades)

    plt.figure(figsize=(10,5))
    plt.plot(times, equity_curve)
    plt.grid(True)
    plt.savefig(EQUITY_CURVE_IMG)

    print(f"Closed trades: {len([t for t in trades if t['status'] in ('tp','sl')])}")
    print(f"Final equity: {equity:.2f}")

if __name__ == "__main__":
    run_pseudotest()
