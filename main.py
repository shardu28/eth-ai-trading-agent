import os
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from indicators import generate_signal, get_latest_equity
from sentiment import get_latest_sentiment  # <- ensure this function exists in sentiment.py

# --- Config ---
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
TO_EMAIL = "your_target_email@example.com"  # <-- set recipient

TRADEBOOK_FILE = "tradebook.csv"


# --- Helpers for Equity/Tradebook ---
def load_equity():
    # 🔑 Use same helper as indicators.py to keep consistency
    return get_latest_equity(tradebook_csv=TRADEBOOK_FILE, default_equity=100.0)


def update_tradebook(trade_decision, ind_signal, sentiment_score, equity):
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "side": trade_decision["side"],
        "entry": trade_decision["entry"],
        "tp": trade_decision["tp"],
        "sl": trade_decision["sl"],
        "size": trade_decision["size"],
        "sentiment_score": sentiment_score,
        "signal_source": ind_signal["signal"],
        "status": "open" if trade_decision["side"] in ["buy", "sell"] else "no_trade",
        "equity_after": equity,
    }
    df = pd.DataFrame([row])
    if not os.path.exists(TRADEBOOK_FILE):
        df.to_csv(TRADEBOOK_FILE, index=False)
    else:
        df.to_csv(TRADEBOOK_FILE, mode="a", header=False, index=False)


# --- Decision Matrix Logic ---
def decide_trade(ind_signal, sentiment_score):
    signal_side = ind_signal["signal"]
    entry, tp, sl, size = ind_signal["entry"], ind_signal["take_profit"], ind_signal["stop_loss"], ind_signal["size"]

    decision = {"side": "no_trade", "entry": None, "tp": None, "sl": None, "size": 0, "reason": ""}

    if signal_side == "neutral" or entry is None:
        decision["reason"] = "Indicator returned neutral"
        return decision

    strong_buy = sentiment_score > 0.25
    weak_buy = 0 < sentiment_score <= 0.25
    strong_sell = sentiment_score < -0.25
    weak_sell = -0.25 <= sentiment_score < 0

    if (signal_side == "buy" and strong_buy) or (signal_side == "sell" and strong_sell):
        decision.update({"side": signal_side, "entry": entry, "tp": tp, "sl": sl, "size": size, "reason": "Full agreement"})
        return decision

    if (signal_side == "buy" and weak_buy) or (signal_side == "sell" and weak_sell):
        adj_sl = entry - 1.0 * ind_signal["atr"] if signal_side == "buy" else entry + 1.0 * ind_signal["atr"]
        adj_tp = entry + 1.5 * ind_signal["atr"] if signal_side == "buy" else entry - 1.5 * ind_signal["atr"]
        adj_size = size * 0.5
        decision.update({"side": signal_side, "entry": entry, "tp": adj_tp, "sl": adj_sl, "size": adj_size, "reason": "Soft conflict — reduced risk"})
        return decision

    if (signal_side == "buy" and sentiment_score < -0.1) or (signal_side == "sell" and sentiment_score > 0.1):
        decision["reason"] = f"Hard conflict — indicator={signal_side}, sentiment={sentiment_score:.2f}"
        return decision

    decision["reason"] = "No clear alignment"
    return decision


# --- Email Sending ---
def send_email_report(trade_decision, ind_signal, sentiment_score, equity):
    subject = f"ETH Trading Signal Report - {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}"
    body = f"""
    ---- Trading Signal Report ----
    Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}

    Indicator Signal: {ind_signal['signal']}
    Entry: {ind_signal['entry']}
    TP: {ind_signal['take_profit']}
    SL: {ind_signal['stop_loss']}
    Size: {ind_signal['size']:.4f if ind_signal['size'] else 0}

    Sentiment Score: {sentiment_score:.4f}

    Final Decision: {trade_decision['side']}
    Reason: {trade_decision['reason']}

    Final Entry: {trade_decision['entry']}
    Final TP: {trade_decision['tp']}
    Final SL: {trade_decision['sl']}
    Final Size: {trade_decision['size']}

    Equity After Trade: {equity:.2f}

    --------------------------------
    """

    msg = MIMEMultipart()
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            server.sendmail(EMAIL_ADDRESS, TO_EMAIL, msg.as_string())
        print("✅ Email sent successfully")
    except Exception as e:
        print(f"❌ Email sending failed: {e}")


# --- Main Run ---
if __name__ == "__main__":
    print("🚀 Running main.py...")

    equity = load_equity()

    # 🔑 No more hardcoded indicator params; use defaults from indicators.py
    ind_signal = generate_signal(candles_csv="candles.csv")

    sentiment_score = get_latest_sentiment("sentiment.csv")

    trade_decision = decide_trade(ind_signal, sentiment_score)
    print("📊 Final Decision:", trade_decision)

    if trade_decision["side"] in ["buy", "sell"]:
        update_tradebook(trade_decision, ind_signal, sentiment_score, equity)

    send_email_report(trade_decision, ind_signal, sentiment_score, equity)
