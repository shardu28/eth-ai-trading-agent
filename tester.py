# tester.py
import os
import pandas as pd
from datetime import datetime, timezone

from candles import fetch_and_save_candles
from candles import append_new_candle
from config import PRODUCT_SYMBOL, CANDLE_RESOLUTION
from delta_client import DeltaClient

CANDLES_FILE = "candles.csv"
client = DeltaClient()

# -------------------- Helpers --------------------
def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


# -------------------- Tests --------------------
def test_append_24h_behavior():
    if not os.path.exists(CANDLES_FILE):
        print("⚠️ candles.csv missing, bootstrapping historical data")
        end = int(datetime.now(timezone.utc).timestamp())
        start = end - 365 * 24 * 3600

        fetch_and_save_candles(
            PRODUCT_SYMBOL,
            CANDLE_RESOLUTION,
            start,
            end,
            CANDLES_FILE
        )

    print("🧪 Test 1: Append behavior (single-run, production-style)")

    assert_true(os.path.exists(CANDLES_FILE), "candles.csv does not exist")

    df_before = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])
    prev_len = len(df_before)
    prev_last_ts = df_before["time_utc"].max()

    print(f"Initial candles: {prev_len}")
    print(f"Last candle timestamp: {prev_last_ts}")

    print("→ Checking for new 1h candle")
    append_new_candle(client)

    df_after = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])

    # No data loss
    assert_true(len(df_after) >= prev_len, "Candle count decreased")

    # No duplicates
    assert_true(df_after["time_utc"].is_unique, "Duplicate timestamps detected")

    # Strict ordering (gaps allowed)
    sorted_ts = df_after["time_utc"].sort_values()
    assert_true(
        sorted_ts.equals(df_after["time_utc"]),
        "Timestamps not strictly ordered"
    )

    new_last_ts = df_after["time_utc"].max()

    if new_last_ts > prev_last_ts:
        appended = len(df_after) - prev_len
        print(f"✅ {appended} new candle(s) appended")
    else:
        print("ℹ️ No new 1h candle formed yet (expected behavior)")

    print("✅ Append behavior verified for this run")
    return df_after


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

    df = test_append_24h_behavior()
    test_data_integrity(df)

    print("🎉 candles append logic PASSED for this hourly run")


if __name__ == "__main__":
    main()
