# tester.py
import os
import pandas as pd
from datetime import datetime, timezone

from candles import fetch_and_save_candles
from sentiment import append_new_candle
from config import PRODUCT_SYMBOL, CANDLE_RESOLUTION
from delta_client import DeltaClient

CANDLES_FILE = "candles.csv"
client = DeltaClient()

# -------------------- Helpers --------------------
def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


# -------------------- Tests --------------------
def test_append_24h_behavior(runs=24):
    print("🧪 Test 1: 24-run append behavior")

    assert_true(os.path.exists(CANDLES_FILE), "candles.csv does not exist")

    df_start = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])
    start_len = len(df_start)
    last_ts = df_start["time_utc"].max()

    print(f"Initial candles: {start_len}")
    print(f"Last candle timestamp: {last_ts}")

    for i in range(1, runs + 1):
        print(f"→ Append run {i}")
        append_new_candle(client)

        df = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])

        # Core append invariants
        assert_true(len(df) >= start_len, "Candle count decreased")
        assert_true(df["time_utc"].is_unique, "Duplicate timestamps detected")
        assert_true(
            df["time_utc"].is_monotonic_increasing,
            "Timestamps not strictly increasing"
        )

        # Timestamp sanity
        new_last_ts = df["time_utc"].max()
        assert_true(new_last_ts >= last_ts, "Last timestamp moved backwards")

        last_ts = new_last_ts
        start_len = len(df)

    print("✅ 24-run append behavior OK")
    return df


def test_data_integrity(df):
    print("🧪 Test 2: Data integrity")

    expected_cols = ["time_utc", "open", "high", "low", "close", "volume"]
    assert_true(list(df.columns) == expected_cols, "Column headers mismatch")

    assert_true(df.isnull().sum().sum() == 0, "Null values detected")

    assert_true(
        (df["high"] >= df[["open", "close", "low"]].max(axis=1)).all(),
        "Invalid OHLC: high < open/close/low"
    )

    assert_true(
        (df["low"] <= df[["open", "close", "high"]].min(axis=1)).all(),
        "Invalid OHLC: low > open/close/high"
    )

    assert_true((df["volume"] >= 0).all(), "Negative volume detected")

    if (df["volume"] == 0).any():
        print("⚠️ Warning: Zero-volume candles detected (acceptable)")

    # Ensure hourly spacing
    min_delta = df["time_utc"].diff().dropna().min().total_seconds()
    assert_true(min_delta >= 3600, "Non-hourly candles detected")

    print("✅ Data integrity OK")


# -------------------- Runner --------------------
def main():
    print("🚀 Running candles append sandbox tests")

    df = test_append_24h_behavior(runs=24)
    test_data_integrity(df)

    print("🎉 candles append logic PASSED stress test")


if __name__ == "__main__":
    main()
