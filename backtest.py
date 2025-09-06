# backtest.py
import csv
import time
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

from delta_client import DeltaClient
from indicators import ema, vwma, roc
from features import ema_cross_signal
from sentiment import funding_signal, momentum_signal
from trade_manager import TradeManager
from config import PRODUCT_SYMBOL

OUTPUT_CSV = "backtest_results.csv"

def fetch_chunked_candles(client, symbol, resolution, start, end, chunk_days=90):
    """Fetch historical candles in chunks to bypass API limits."""
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
        time.sleep(0.2)  # polite delay
    all_candles.sort(key=lambda r: r["time"])
    return all_candles

def fetch_chunked_funding(client, symbol, resolution, start, end, chunk_days=90):
    """Fetch funding history in chunks (same pattern as candles)."""
    all_rates = []
    cur_start = start
    while cur_start < end:
        cur_end = min(cur_start + chunk_days * 24 * 3600, end)
        raw = client.get("/history/candles", {
            "symbol": f"FUNDING:{symbol}",
            "resolution": resolution,
            "start": cur_start,
            "end": cur_end,
        })
        chunk = raw.get("result", [])
        all_rates.extend([float(r["close"]) for r in chunk])
        cur_start = cur_end
        time.sleep(0.2)
    return all_rates

def run_backtest(days=365, start_equity=1000.0, risk_per_trade=10.0):
    client = DeltaClient()
    tm = TradeManager()

    # 1 year range
    end = int(time.time())
    start = end - days * 24 * 3600

    # Fetch candles and funding in chunks
    candles = fetch_chunked_candles(client, PRODUCT_SYMBOL, "1h", start, end)
    frates = fetch_chunked_funding(client, PRODUCT_SYMBOL, "1h", start, end)

    equity = start_equity
    trades = []
    closes, vols, equity_curve, times = [], [], [], []

    for i, c in enumerate(candles):
        ts = c["time"]
        close = float(c["close"])
        vol = float(c.get("volume", 0))
        closes.append(close)
        vols.append(vol)

        # check exits
        hit = tm.check_close(close)
        if hit:
            exit_price = close
            side = tm.state["side"]
            entry = tm.state["entry"]
            pnl = (exit_price - entry) if side == "buy" else (entry - exit_price)
            pnl_usd = (pnl / entry) * risk_per_trade
            equity += pnl_usd
            trades.append([datetime.fromtimestamp(ts), side, entry, exit_price, hit, pnl_usd, equity])
            tm.reset()

        if i < 30 or not tm.can_issue():
            equity_curve.append(equity)
            times.append(datetime.fromtimestamp(ts))
            continue

        # indicators
        e9 = ema(closes, 9)
        e21 = ema(closes, 21)
        v3 = vwma(closes, vols, 3)
        momentum = roc(closes, 3)

        ind_sig = ema_cross_signal(closes, e9, e21, v3)
        mom_sig = momentum_signal(momentum[-1], e9)

        # funding signal (use last 24 values up to index i if available)
        lookback_rates = frates[max(0, i-24):i] if frates else []
        f_sig = funding_signal(lookback_rates) if lookback_rates else 0

        # sentiment = majority of [mom_sig, f_sig]
        votes = [s for s in (mom_sig, f_sig) if s != 0]
        if votes:
            dirn = 1 if votes.count(1) > votes.count(-1) else -1
            match_rate = len([v for v in votes if v == dirn]) / len(votes)

            if ind_sig and ind_sig == dirn and match_rate >= 0.5:
                side = "buy" if ind_sig == 1 else "sell"
                entry = close
                tp, sl = tm.open(side, entry)
                trades.append([datetime.fromtimestamp(ts), side, entry, None, "open", 0.0, equity])

        equity_curve.append(equity)
        times.append(datetime.fromtimestamp(ts))

    # Save trades
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "side", "entry", "exit", "result", "pnl_usd", "equity"])
        writer.writerows(trades)

    # Metrics
    closed_trades = [t for t in trades if t[4] in ("tp", "sl")]
    wins = [t for t in closed_trades if t[4] == "tp"]
    losses = [t for t in closed_trades if t[4] == "sl"]
    win_rate = len(wins) / len(closed_trades) if closed_trades else 0
    avg_win = sum(t[5] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t[5] for t in losses) / len(losses) if losses else 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    print(f"Trades: {len(closed_trades)} | Win rate: {win_rate:.2%} | Expectancy: {expectancy:.2f} USD")
    print(f"Final equity: {equity:.2f}")

    # Equity curve plot
    plt.figure(figsize=(10, 5))
    plt.plot(times, equity_curve, label="Equity")
    plt.title("Equity Curve (Backtest)")
    plt.xlabel("Time")
    plt.ylabel("Equity (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("equity_curve.png")
    plt.show()

if __name__ == "__main__":
    run_backtest()
