# main.py
import os
import pandas as pd
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from indicators import generate_signal
from sentiment import get_latest_sentiment  # <- ensure this function exists in sentiment.py


# --- Config ---
EQUITY = 1000
ADX_THRESH = 25
ATR_MULT_SL = 1.5
ATR_MULT_TP = 2.5
AVG_WINDOW = 5
VP_WINDOW = 50
RVI_PERIOD = 10
RISK_FRACTION = 0.01

EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
TO_EMAIL = "your_target_email@example.com"  # <-- set recipient


# --- Decision Matrix Logic ---
def decide_trade(ind_signal, sentiment_score):
    """
    Compare indicator signal with sentiment score and return final decision.
    """
    signal_side = ind_signal["signal"]
    entry, tp, sl, size = ind_signal["entry"], ind_signal["take_profit"], ind_signal["stop_loss"], ind_signal["size"]

    decision = {"side": "no_trade", "entry": None, "tp": None, "sl": None, "size": 0, "reason": ""}

    # No signal from indicators
    if signal_side == "neutral" or entry is None:
        decision["reason"] = "Indicator returned neutral"
        return decision

    # Sentiment thresholds
    strong_buy = sentiment_score > 0.25
    weak_buy = 0 < sentiment_score <= 0.25
    strong_sell = sentiment_score < -0.25
    weak_sell = -0.25 <= sentiment_score < 0

    # --- Full Agreement ---
    if (signal_side == "buy" and strong_buy) or (signal_side == "sell" and strong_sell):
        decision.update({"side": signal_side, "entry": entry, "tp": tp, "sl": sl, "size": size, "reason": "Full agreement"})
        return decision

    # --- Soft Conflict (same direction but weak sentiment) ---
    if (signal_side == "buy" and weak_buy) or (signal_side == "sell" and weak_sell):
        adj_sl = entry - 1.0 * ind_signal["atr"] if signal_side == "buy" else entry + 1.0 * ind_signal["atr"]
        adj_tp = entry + 1.5 * ind_signal["atr"] if signal_side == "buy" else entry - 1.5 * ind_signal["atr"]
        adj_size = size * 0.5

        decision.update({"side": signal_side, "entry": entry, "tp": adj_tp, "sl": adj_sl, "size": adj_size, "reason": "Soft conflict — reduced risk"})
        return decision

    # --- Hard Conflict (opposite direction) ---
    if (signal_side == "buy" and sentiment_score < -0.1) or (signal_side == "sell" and sentiment_score > 0.1):
        decision["reason"] = f"Hard conflict — indicator={signal_side}, sentiment={sentiment_score:.2f}"
        return decision

    # Default case
    decision["reason"] = "No clear alignment"
    return decision


# --- Email Sending ---
def send_email_report(trade_decision, ind_signal, sentiment_score):
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

    # Get indicator signal
    ind_signal = generate_signal(
        candles_csv="candles.csv",
        atr_mult_sl=ATR_MULT_SL, atr_mult_tp=ATR_MULT_TP,
        adx_thresh=ADX_THRESH, avg_window=AVG_WINDOW,
        vp_window=VP_WINDOW, rvi_period=RVI_PERIOD,
        risk_fraction=RISK_FRACTION
    )

    # Get sentiment score (latest)
    sentiment_score = get_latest_sentiment("sentiment.csv")

    # Decision
    trade_decision = decide_trade(ind_signal, sentiment_score)
    print("📊 Final Decision:", trade_decision)

    # Save to daily report
    report_file = "daily_report.csv"
    row = {
        "timestamp": datetime.utcnow().isoformat(),
        "indicator_signal": ind_signal["signal"],
        "sentiment_score": sentiment_score,
        "final_decision": trade_decision["side"],
        "reason": trade_decision["reason"],
        "entry": trade_decision["entry"],
        "tp": trade_decision["tp"],
        "sl": trade_decision["sl"],
        "size": trade_decision["size"],
    }
    df = pd.DataFrame([row])
    if not os.path.exists(report_file):
        df.to_csv(report_file, index=False)
    else:
        df.to_csv(report_file, mode="a", header=False, index=False)

    # Email report
    send_email_report(trade_decision, ind_signal, sentiment_score)
