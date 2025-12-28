# sentiment_tester.py
import os
import pandas as pd
from sentiment import run_sentiment

SENTIMENT_FILE = "sentiment.csv"

def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)

def test_sentiment_append():
    prev_len = 0
    if os.path.exists(SENTIMENT_FILE):
        prev_len = len(pd.read_csv(SENTIMENT_FILE))

    run_sentiment()

    assert_true(os.path.exists(SENTIMENT_FILE), "sentiment.csv missing")

    df = pd.read_csv(SENTIMENT_FILE, parse_dates=["run_time_utc", "run_time_ist"])
    assert_true(len(df) >= prev_len + 1, "Sentiment row not appended")

    expected_cols = [
        "run_time_utc",
        "run_time_ist",
        "imbalance",
        "trade_flow",
        "sentiment_score",
    ]
    assert_true(list(df.columns) == expected_cols, "Column mismatch")

    assert_true(df["sentiment_score"].between(-1, 1).all(), "Invalid score range")

    print("✅ Sentiment append test passed")

if __name__ == "__main__":
    test_sentiment_append()
