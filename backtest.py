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

# ----------------- Backtest params (defaults) -----------------
DAYS = 365  # 1 year
START_EQUITY = 100.0
LEVERAGE = 20.0

# defaults (can be overridden in run_backtest call)
ATR_MULT_SL = 1.5
ATR_MULT_TP = 2.5
ADX_THRESH = 25
AVG_WINDOW = 5
VP_WINDOW = 50
RVI_PERIOD = 10
RISK_FRACTION = 0.01

SESSION_START_IST = 8
SESSION_END_IST = 23

ATR_PCT_MIN = 0.005  # 0.5%

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
    num = (series_close * series_volume).rolling(window=window, min_periods=1).sum()
    den = series_volume.rolling(window=window, min_periods=1).sum()
    return num / den


def compute_vp_node(series_close, window, precision=2):
    def vp_mode(x):
        if len(x) == 0:
            return float("nan")
        rounded = (x.round(precision)).astype(str)
        return float(pd.Series(rounded).mode().iloc[0])
    return series_close.rolling(window=window, min_periods=1).apply(lambda x: vp_mode(x), raw=False)


def rvi_approx(series_close, series_open, series_high, series_low, period):
    num = (series_close - series_open).rolling(window=period, min_periods=1).mean()
    den = (series_high - series_low).rolling(window=period, min_periods=1).mean().abs() + 1e-9
    rvi = num / den
    return 50.0 * (1 + rvi.clip(-1, 1))


# ----------------- backtest core -----------------
def run_backtest(atr_mult_sl=ATR_MULT_SL, atr_mult_tp=ATR_MULT_TP,
                 adx_thresh=ADX_THRESH, avg_window=AVG_WINDOW,
                 vp_window=VP_WINDOW, rvi_period=RVI_PERIOD,
                 start_equity=START_EQUITY):

    client = DeltaClient()
    end_ts = int(time.time())
    start_ts = end_ts - DAYS * 24 * 3600

    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start_ts, end_ts)
    if not candles:
        raise RuntimeError("No candle data returned from API")

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ist"] = df["time"].dt.tz_convert("Asia/Kolkata")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # indicators
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()
    df["avg_close"] = df["close"].rolling(window=avg_window, min_periods=1).mean()
    df["avg_vwma"] = compute_avg_vwma(df["close"], df["volume"], avg_window)
    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1
    df["rvi"] = rvi_approx(df["close"], df["open"], df["high"], df["low"], rvi_period)
    mean_rvi = df["rvi"].mean()
    df["rvi_signal"] = df["rvi"].apply(lambda x: 1 if x > mean_rvi else -1)
    df["vp_node"] = compute_vp_node(df["close"], vp_window)

    equity = start_equity
    trades, equity_curve, times = [], [], []
    trades_today = {}

    for i, row in df.iterrows():
        ts, ts_ist, close, atr, adx = row["time"], row["time_ist"], row["close"], row["atr"], row["adx"]
        vwma_sig, rvi_sig, vp_node = row["vwma_signal"], row["rvi_signal"], row["vp_node"]

        # session filter
        if ts_ist.hour < SESSION_START_IST or ts_ist.hour > SESSION_END_IST:
            equity_curve.append(equity); times.append(ts); continue

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
                trades[-1][3], trades[-1][4], trades[-1][5], trades[-1][6] = exit_price, result, pnl, equity
        if trades and trades[-1][4] == "open":
            equity_curve.append(equity); times.append(ts); continue

        # history / filters
        if i < 30 or pd.isna(adx) or pd.isna(atr):
            equity_curve.append(equity); times.append(ts); continue
        atr_pct = atr / close if close else 0.0
        if atr_pct < ATR_PCT_MIN:
            equity_curve.append(equity); times.append(ts); continue
        if adx <= adx_thresh:
            equity_curve.append(equity); times.append(ts); continue

        # entry
        ind_sig = vwma_sig
        vp_ok = (close > vp_node) if (ind_sig == 1 and not pd.isna(vp_node)) else \
                (close < vp_node) if (ind_sig == -1 and not pd.isna(vp_node)) else False
        extra_confirmation = (rvi_sig == ind_sig) or vp_ok
        if ind_sig != 0 and extra_confirmation:
            day = ts_ist.date()
            trades_today[day] = trades_today.get(day, 0)
            if trades_today[day] >= 2:
                equity_curve.append(equity); times.append(ts); continue

            risk_capital = equity * RISK_FRACTION
            denom = atr_mult_sl * atr if atr else None
            if not denom or denom == 0:
                equity_curve.append(equity); times.append(ts); continue
            size = risk_capital / denom
            required_margin = (size * close) / LEVERAGE
            if required_margin > equity:
                equity_curve.append(equity); times.append(ts); continue

            side = "buy" if ind_sig == 1 else "sell"
            entry = close
            sl = entry - atr_mult_sl * atr if side == "buy" else entry + atr_mult_sl * atr
            tp = entry + atr_mult_tp * atr if side == "buy" else entry - atr_mult_tp * atr
            trades.append([ts, side, entry, None, "open", 0.0, equity, tp, sl, size])
            trades_today[day] += 1

        equity_curve.append(equity); times.append(ts)

    closed_trades = [t for t in trades if t[4] in ("tp", "sl")]
    wins = [t for t in closed_trades if t[4] == "tp"]
    losses = [t for t in closed_trades if t[4] == "sl"]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0
    avg_win = sum(t[5] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t[5] for t in losses) / len(losses) if losses else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "trades": trades,
        "equity_curve": (times, equity_curve),
        "final_equity": equity,
        "win_rate": win_rate,
        "expectancy": expectancy,
        "params": {
            "atr_mult_sl": atr_mult_sl, "atr_mult_tp": atr_mult_tp,
            "adx_thresh": adx_thresh, "avg_window": avg_window,
            "vp_window": vp_window, "rvi_period": rvi_period
        }
    }


# ----------------- optimizer -----------------
def optimize():
    best = None
    for sl in [1.5, 1.8, 2.0]:
        for tp in [2.2, 2.5, 2.8]:
            for adx in [22, 25, 28]:
                for avg in [5, 7]:
                    for vp in [35, 45, 50]:
                        result = run_backtest(atr_mult_sl=sl, atr_mult_tp=tp,
                                              adx_thresh=adx, avg_window=avg,
                                              vp_window=vp)
                        if best is None or result["expectancy"] > best["expectancy"]:
                            best = result
    # save best equity curve
    times, eq_curve = best["equity_curve"]
    plt.figure(figsize=(10,5))
    plt.plot(times, eq_curve, label="Equity")
    plt.title(f"Best Equity Curve - {PRODUCT_SYMBOL}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(EQUITY_CURVE_IMG)
    print("✅ Best Params:", best["params"])
    print(f"Trades: {len(best['trades'])} | Win rate: {best['win_rate']:.2%} | "
          f"Expectancy: {best['expectancy']:.2f} | Final equity: {best['final_equity']:.2f}")


if __name__ == "__main__":
    optimize()
