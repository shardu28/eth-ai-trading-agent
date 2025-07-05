Python 3.9.0 (tags/v3.9.0:9cf6752, Oct  5 2020, 15:34:40) [MSC v.1927 64 bit (AMD64)] on win32
Type "help", "copyright", "credits" or "license()" for more information.
>>> from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import ccxt
import pandas as pd
import schedule
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email Configuration (Replace with your credentials)
EMAIL_ADDRESS = 'shardulkshirsagar22@gmail.com'
EMAIL_PASSWORD = 'eobs szql wkfj fazq'
RECIPIENT_EMAIL = 'shardulkshirsagar22@gmail.com'

# Initialize Exchange
exchange = ccxt.binance()
symbol = 'ETH/USDT'

def fetch_ohlcv():
    data = exchange.fetch_ohlcv(symbol, timeframe='1d', limit=100)
    df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def add_indicators(df):
    df['ema_50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()
    df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
    df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14).average_true_range()
    return df

def generate_trade_idea(df):
    latest = df.iloc[-1]
    signal = None

    if latest['close'] > latest['ema_50'] and latest['rsi'] > 55:
        entry = round(latest['close'], 2)
        sl = round(entry - latest['atr'], 2)
        tp = round(entry + 3 * (entry - sl), 2)
        signal = {
            "symbol": symbol,
            "direction": "BUY",
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "risk_reward": "1:3",
            "reason": "Close > EMA50, RSI > 55, ATR-based RRR"
        }

    return signal

def send_email(signal):
    subject = f"[Trade Signal] {signal['symbol']} - {signal['direction']}"
    body = "\n".join([f"{key}: {value}" for key, value in signal.items()])

    msg = MIMEMultipart()
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {e}")

def run_agent():
    df = fetch_ohlcv()
    df = add_indicators(df)
    signal = generate_trade_idea(df)
    if signal:
        print("\n--- Trade Signal ---")
        for key, value in signal.items():
            print(f"{key}: {value}")
        send_email(signal)
    else:
        print("No valid trade signal today.")

# Schedule to run every day at 9:00 UTC
schedule.every().day.at("09:00").do(run_agent)

print("[Agent Initialized] Running ETH trade signal daily at 09:00 UTC...")
while True:
    schedule.run_pending()
    time.sleep(60)