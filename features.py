# features.py

def ema_cross_signal(closes, ema9, ema21, vwma3):
    i = len(closes) - 1
    if None in (closes[i], ema9[i], ema21[i], vwma3[i]):
        return None
    bull = closes[i] > ema9[i] > ema21[i] and closes[i] > vwma3[i]
    bear = closes[i] < ema9[i] < ema21[i] and closes[i] < vwma3[i]
    if bull:
        return 1
    if bear:
        return -1
    return 0
