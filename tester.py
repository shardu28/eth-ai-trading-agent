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

def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)

def test_candles_creation():
    print("🧪 Test 1: Initial candle fetch")

    if os.path.exists(CANDLES_FILE):
        os.remove(CANDLES_FILE)

    end = int(datetime.now(timezone.utc).timestamp())
    start = end - 365 * 24 * 3600

    fetch_and_save_candles(PRODUCT_SYMBOL, CANDLE_RESOLUTION, start, end, CANDLES_FILE)

    assert_true(os.path.exists(CANDLES_FILE), "candles.csv was not created")

    df = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])
    assert_true(len(df) > 1000, "candles.csv has insufficient data")

    expected_cols = ["time_utc", "open", "high", "low", "close", "volume"]
    assert_true(list(df.columns) == expected_cols, "Column headers mismatch")

    print("✅ Initial candle fetch OK")
    return df

def test_append_behavior(df_before):
    print("🧪 Test 2: Append behavior")

    rows_before = len(df_before)
    last_ts_before = df_before["time_utc"].max()

    append_new_candle(client)

    df_after = pd.read_csv(CANDLES_FILE, parse_dates=["time_utc"])
    rows_after = len(df_after)
    last_ts_after = df_after["time_utc"].max()

    assert_true(rows_after >= rows_before, "Row count decreased after append")
    assert_true(
        rows_after == rows_before or rows_after == rows_before + 1,
        "Unexpected row count change"
    )

    assert_true(df_after["time_utc"].is_unique, "Duplicate timestamps detected")

    print("✅ Append behavior OK")
    return df_after

def test_data_integrity(df):
    print("🧪 Test 3: Data integrity")

    assert_true(df.isnull().sum().sum() == 0, "Null values found")

    assert_true((df["high"] >= df[["open", "close"]].max(axis=1)).all(), "Invalid high price")
    assert_true((df["low"] <= df[["open", "close"]].min(axis=1)).all(), "Invalid low price")
    assert_true((df["volume"] > 0).all(), "Invalid volume detected")

    assert_true(
        df["time_utc"].diff().dropna().min().total_seconds() >= 3600,
        "Non-hourly candles detected"
    )

    print("✅ Data integrity OK")

def main():
    print("🚀 Running candles.py sandbox tests")
    df1 = test_candles_creation()
    df2 = test_append_behavior(df1)
    test_data_integrity(df2)
    print("🎉 candles.py PASSED all tests")

if __name__ == "__main__":
    main()
