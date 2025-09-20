import pandas as pd
import ta
from datetime import datetime
import csv

# --- your existing utilities remain unchanged ---
def ema(series, length):
    if not series or length <= 0 or len(series) < length:
        return [None] * len(series)
    k = 2 / (length + 1)
    out = [None] * len(series)
    sma = sum(series[:length]) / length
    out[length - 1] = sma
    prev = sma
    for i in range(length, len(series)):
        prev = series[i] * k + prev * (1 - k)
        out[i] = prev
    return out

def vwma(closes, volumes, length):
    out = [None] * len(closes)
    for i in range(len(closes)):
        if i + 1 < length:
            continue
        wsum = 0.0
        vsum = 0.0
        for j in range(i - length + 1, i + 1):
            wsum += closes[j] * volumes[j]
            vsum += volumes[j]
        out[i] = (wsum / vsum) if vsum else None
    return out

def roc(series, n):
    out = [None] * len(series)
    for i in range(len(series)):
        if i < n:
            continue
        prev = series[i - n]
        out[i] = (series[i] - prev) / prev if prev else None
    return out

# ---- Backtest-identical indicator helpers ----
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

# ---- Dynamic Equity Helper ----
def get_latest_equity(tradebook_csv="tradebook.csv", default_equity=100.0):
    if os.path.exists(tradebook_csv):
        df = pd.read_csv(tradebook_csv)
        if not df.empty and "equity" in df.columns:
            return float(df["equity"].iloc[-1])
    return default_equity

# ---- Indicator Calculations ----
def compute_indicators(df,
                       atr_mult_sl=1.3, atr_mult_tp=3.7, adx_thresh=29,
                       avg_window=5, vp_window=50, rvi_period=10):
    # ATR & ADX
    df["atr"] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range()
    df["adx"] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).adx()

    # VWMA vs close
    df["avg_close"] = df["close"].rolling(window=avg_window, min_periods=1).mean()
    df["avg_vwma"] = compute_avg_vwma(df["close"], df["volume"], avg_window)
    df["vwma_signal"] = 0
    df.loc[df["avg_close"] > df["avg_vwma"], "vwma_signal"] = 1
    df.loc[df["avg_close"] < df["avg_vwma"], "vwma_signal"] = -1

    # RVI identical to backtest
    df["rvi"] = rvi_approx(df["close"], df["open"], df["high"], df["low"], rvi_period)
    mean_rvi = df["rvi"].mean()
    df["rvi_signal"] = df["rvi"].apply(lambda x: 1 if x > mean_rvi else -1)

    # Volume profile node
    df["vp_node"] = compute_vp_node(df["close"], vp_window)

    return df
                           
def generate_signal(candles_csv="candles.csv",
                    atr_mult_sl=1.3, atr_mult_tp=3.7,
                    adx_thresh=29, avg_window=5,
                    vp_window=50, rvi_period=10,
                    risk_fraction=0.02):
    """
    Reads candles.csv, computes indicators, and generates final trading signal.
    Returns dict with signal, entry, TP, SL, and size.
    Also appends signal to signal.csv for logging.
    """

    # ---- Load candles ----
    df = pd.read_csv(candles_csv)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    # ---- Compute indicators ----
    df = compute_indicators(df,
                            atr_mult_sl=atr_mult_sl,
                            atr_mult_tp=atr_mult_tp,
                            adx_thresh=adx_thresh,
                            avg_window=avg_window,
                            vp_window=vp_window,
                            rvi_period=rvi_period)

    # ---- Last row for signal ----
    last = df.iloc[-1]
    adx, atr, close = last["adx"], last["atr"], last["close"]
    vwma_sig, rvi_sig = last["vwma_signal"], last["rvi_signal"]

    # VP confirmation
    vp_ok = (close > last["vp_node"] if vwma_sig == 1 else close < last["vp_node"])

    signal, entry, sl, tp, size = "neutral", None, None, None, None
    if pd.notna(adx) and pd.notna(atr) and adx > adx_thresh:
        ind_sig = vwma_sig
        extra_confirmation = (rvi_sig == ind_sig) or vp_ok
        if ind_sig != 0 and extra_confirmation:
            side = "buy" if ind_sig == 1 else "sell"
            entry = close
            sl = entry - atr_mult_sl * atr if side == "buy" else entry + atr_mult_sl * atr
            tp = entry + atr_mult_tp * atr if side == "buy" else entry - atr_mult_tp * atr

            # 🔑 Dynamic equity from tradebook
            equity = get_latest_equity()
            risk_capital = equity * risk_fraction  # fixed equity assumption (can be dynamic later)
            size = risk_capital / (atr_mult_sl * atr) if atr > 0 else 0
            signal = side

    result = {
        "timestamp": datetime.utcnow().isoformat(),
        "signal": signal,
        "entry": entry,
        "stop_loss": sl,
        "take_profit": tp,
        "size": size,
        "adx": adx,
        "atr": atr,
        "vwma_sig": vwma_sig,
        "rvi_sig": rvi_sig,
        "vp_ok": vp_ok
    }

    # ---- Append to signal.csv ----
    with open("signal.csv", "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        if f.tell() == 0:  # write header if file is empty
            writer.writeheader()
        writer.writerow(result)

    return result
