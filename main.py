from ta.trend import EMAIndicator, ADXIndicator
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange
import ccxt
import pandas as pd
import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email Configuration (Read securely from GitHub Actions secrets)
EMAIL_ADDRESS = os.getenv('EMAIL_ADDRESS')
EMAIL_PASSWORD = os.getenv('EMAIL_PASSWORD')
RECIPIENT_EMAIL = os.getenv('RECIPIENT_EMAIL')

# Initialize Exchange
exchange = ccxt.kucoin()
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
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    adx = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx.adx()
    return df

def detect_rsi_divergence(df):
    # Detect basic bullish or bearish RSI divergence over last 5 candles
    lows = df['low'].iloc[-5:]
    highs = df['high'].iloc[-5:]
    rsi = df['rsi'].iloc[-5:]

    # Bullish Divergence
    if lows.iloc[-1] < lows.iloc[-2] and rsi.iloc[-1] > rsi.iloc[-2]:
        return "Bullish"

    # Bearish Divergence
    if highs.iloc[-1] > highs.iloc[-2] and rsi.iloc[-1] < rsi.iloc[-2]:
        return "Bearish"

    return None

def generate_trade_idea(df):
    latest = df.iloc[-1]
    signal = None

    # Filters
    volume_ok = latest['volume'] > latest['volume_sma_20']
    adx_ok = latest['adx'] > 20
    divergence = detect_rsi_divergence(df)

    if not volume_ok or not adx_ok:
        return None  # Skip trade if filters fail

    rrr = 1.5  # Risk-to-reward ratio
    entry = round(latest['close'], 2)
    atr = latest['atr']

    # BUY Signal
    if latest['close'] > latest['ema_50'] and latest['rsi'] > 55 and divergence == "Bullish":
        sl = round(entry - atr, 2)
        tp = round(entry + rrr * (entry - sl), 2)
        signal = {
            "symbol": symbol,
            "direction": "BUY",
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "risk_reward": "1:1.5",
            "reason": "Volume > avg, ADX > 20, Bullish RSI divergence, Close > EMA50, RSI > 55"
        }

    # SELL Signal
    elif latest['close'] < latest['ema_50'] and latest['rsi'] < 45 and divergence == "Bearish":
        sl = round(entry + atr, 2)
        tp = round(entry - rrr * (sl - entry), 2)
        signal = {
            "symbol": symbol,
            "direction": "SELL",
            "entry": entry,
            "stop_loss": sl,
            "take_profit": tp,
            "risk_reward": "1:1.5",
            "reason": "Volume > avg, ADX > 20, Bearish RSI divergence, Close < EMA50, RSI < 45"
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

# Entry point
print("[Agent Started] Running ETH trade signal now...")
run_agent()
