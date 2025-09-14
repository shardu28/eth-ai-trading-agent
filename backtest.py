# backtest.py (optimized standalone for SOL backtest grid search)
import csv
import time
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
import ta

from delta_client import DeltaClient
from config import PRODUCT_SYMBOL  # set PRODUCT_SYMBOL = "SOLUSD" in config.py

OUTPUT_CSV = "backtest_results.csv"
EQUITY_CURVE_IMG = "equity_curve.png"

# ----------------- Backtest params -----------------
DAYS = 365
START_EQUITY = 100.0
LEVERAGE = 20.0
RISK_FRACTION = 0.01

SESSION_START_IST = 8
SESSION_END_IST = 23
ATR_PCT_MIN = 0.005

CHUNK_DAYS = 90
POLITE_DELAY = 0.2


# ----------------- Helpers -----------------
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


def prepare_data():
    client = DeltaClient()
    end_ts = int(time.time())
    start_ts = end_ts - DAYS * 24 * 3600

    print(f"Fetching {DAYS} days of 1H candles for {PRODUCT_SYMBOL}...")
    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start_ts, end_ts)
    if not candles:
        raise RuntimeError("No candle data returned from API")

    df = pd.DataFrame(candles)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df["time_ist"] = df["time"].dt.tz_convert("Asia/Kolkata")
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)

    # static indicators
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()

    return df


# ----------------- Backtest core -----------------
def run_backtest(df, atr_mult_sl, atr_mult_tp, adx_thresh,
                 avg_window, vp_window, rvi_period,
                 start_equity=START_EQUITY):

    df = df.copy()

    # recompute param-dependent indicators
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
        ts, ts_ist = row["time"], row["time_ist"]
        close, atr, adx = row["close"], row["atr"], row["adx"]
        vwma_sig, rvi_sig, vp_node = row["vwma_signal"], row["rvi_signal"], row["vp_node"]

        # session time filter
        if ts_ist.hour < SESSION_START_IST or ts_ist.hour > SESSION_END_IST:
            equity_curve.append(equity); times.append(ts); continue

        # exit check
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

        # skip if still in open trade
        if trades and trades[-1][4] == "open":
            equity_curve.append(equity); times.append(ts); continue

        if i < 30 or pd.isna(adx) or pd.isna(atr):
            equity_curve.append(equity); times.append(ts); continue

        if atr/close < ATR_PCT_MIN:  # volatility filter
            equity_curve.append(equity); times.append(ts); continue

        if adx <= adx_thresh:
            equity_curve.append(equity); times.append(ts); continue

        ind_sig = vwma_sig
        vp_ok = (close > vp_node) if (ind_sig == 1 and not pd.isna(vp_node)) else \
                (close < vp_node) if (ind_sig == -1 and not pd.isna(vp_node)) else False
        extra_conf = (rvi_sig == ind_sig) or vp_ok

        if ind_sig != 0 and extra_conf:
            day = ts_ist.date()
            trades_today[day] = trades_today.get(day, 0)
            if trades_today[day] >= 2:
                equity_curve.append(equity); times.append(ts); continue

            risk_capital = equity * RISK_FRACTION
            denom = atr_mult_sl * atr if atr > 0 else None
            if not denom: continue
            size = risk_capital / denom
            if (size * close) / LEVERAGE > equity:  # margin check
                equity_curve.append(equity); times.append(ts); continue

            side = "buy" if ind_sig == 1 else "sell"
            entry = close
            sl = entry - atr_mult_sl * atr if side == "buy" else entry + atr_mult_sl * atr
            tp = entry + atr_mult_tp * atr if side == "buy" else entry - atr_mult_tp * atr

            trades.append([ts, side, entry, None, "open", 0.0, equity, tp, sl, size])
            trades_today[day] += 1

        equity_curve.append(equity)
        times.append(ts)

    closed = [t for t in trades if t[4] in ("tp", "sl")]
    wins = [t for t in closed if t[4] == "tp"]
    losses = [t for t in closed if t[4] == "sl"]
    win_rate = len(wins) / len(closed) if closed else 0.0
    avg_win = sum(t[5] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t[5] for t in losses) / len(losses) if losses else 0.0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    return {
        "params": (atr_mult_sl, atr_mult_tp, adx_thresh, avg_window, vp_window, rvi_period),
        "trades": trades,
        "equity_curve": (times, equity_curve),
        "win_rate": win_rate,
        "expectancy": expectancy,
        "final_equity": equity
    }


# ----------------- Optimization -----------------
def optimize():
    df = prepare_data()
    best = None

    for sl in [1.5, 1.8, 2.0]:
        for tp in [2.2, 2.5, 2.8]:
            for adx in [22, 25, 28]:
                for avg in [5, 7]:
                    for vp in [35, 45, 50]:
                        result = run_backtest(
                            df, atr_mult_sl=sl, atr_mult_tp=tp,
                            adx_thresh=adx, avg_window=avg,
                            vp_window=vp, rvi_period=10
                        )
                        if best is None or result["expectancy"] > best["expectancy"]:
                            best = result

    # write trades
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time","side","entry","exit","result","pnl_usd","equity","tp","sl","size"])
        writer.writerows(best["trades"])

    # plot equity curve
    times, curve = best["equity_curve"]
    plt.figure(figsize=(10,5))
    plt.plot(times, curve, label="Equity")
    plt.title(f"Best Equity Curve - {PRODUCT_SYMBOL}")
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(EQUITY_CURVE_IMG)

    print("✅ Best Params:", best["params"])
    print(f"Trades: {len(best['trades'])} | Win rate: {best['win_rate']:.2%} | "
          f"Expectancy: {best['expectancy']:.2f} | Final equity: {best['final_equity']:.2f}")


if __name__ == "__main__":
    optimize()
