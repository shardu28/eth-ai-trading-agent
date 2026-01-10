# parity_test.py
import pandas as pd
import numpy as np

from indicators import compute_indicators

# ---------------- CONFIG ----------------
CANDLES_CSV = "backtest_candles.csv"
SIGNALS_CSV = "backtest_signals.csv"

FLOAT_TOL = 1e-8
VP_NODE_TOL = 0.01
# --------------------------------------


def assert_series_close(name, a, b, tol):
    diff = (a - b).abs()
    max_diff = diff.max()

    if max_diff > tol:
        bad_idx = diff.idxmax()
        raise AssertionError(
            f"{name} mismatch\n"
            f"Max diff: {max_diff}\n"
            f"At index: {bad_idx}\n"
            f"Time: {merged.loc[bad_idx, 'time']}"
        )
    print(f"✔ {name} parity OK (max diff {max_diff})")


print("Loading backtest candles...")
candles = pd.read_csv(CANDLES_CSV, parse_dates=["time"])

print("Loading backtest signals (ground truth)...")
truth = pd.read_csv(SIGNALS_CSV, parse_dates=["time"])

print("Running indicators.py...")
calc = compute_indicators(candles.copy())

calc_cols = [
    "time",
    "atr",
    "adx",
    "vwma_signal",
    "rvi_signal",
    "vp_node",
]

calc = calc[calc_cols]

print("Merging calculated vs backtest data...")
merged = calc.merge(
    truth,
    on="time",
    suffixes=("_calc", "_truth"),
    how="inner"
)

if merged.empty:
    raise RuntimeError("Merged dataframe is empty. Time alignment is broken.")

# ---------------- NUMERIC PARITY ----------------
assert_series_close(
    "ATR",
    merged["atr_calc"],
    merged["atr_truth"],
    FLOAT_TOL
)

assert_series_close(
    "ADX",
    merged["adx_calc"],
    merged["adx_truth"],
    FLOAT_TOL
)

assert_series_close(
    "VP Node",
    merged["vp_node_calc"],
    merged["vp_node_truth"],
    VP_NODE_TOL
)

# ---------------- SIGNAL PARITY ----------------
if not (merged["vwma_signal_calc"] == merged["vwma_signal_truth"]).all():
    bad = merged[merged["vwma_signal_calc"] != merged["vwma_signal_truth"]].iloc[0]
    raise AssertionError(
        f"VWMA signal mismatch at {bad['time']}: "
        f"{bad['vwma_signal_calc']} vs {bad['vwma_signal_truth']}"
    )
print("✔ VWMA signal parity OK")

if not (merged["rvi_signal_calc"] == merged["rvi_signal_truth"]).all():
    bad = merged[merged["rvi_signal_calc"] != merged["rvi_signal_truth"]].iloc[0]
    raise AssertionError(
        f"RVI signal mismatch at {bad['time']}: "
        f"{bad['rvi_signal_calc']} vs {bad['rvi_signal_truth']}"
    )
print("✔ RVI signal parity OK")

print("\n✅ ALL INDICATORS MATCH BACKTEST EXACTLY")
