# indicators.py

def ema(series, length):
    if not series or length <= 0 or len(series) < length:
        return [None] * len(series)
    k = 2 / (length + 1)
    out = [None] * len(series)
    sma = sum(series[:length]) / length
    out[length - 1] = sma
    prev = sma
    for i in range(length, len(series)):
        prev = series[i] * k + prev * (1 - k)
        out[i] = prev
    return out

def vwma(closes, volumes, length):
    out = [None] * len(closes)
    for i in range(len(closes)):
        if i + 1 < length:
            continue
        wsum = 0.0
        vsum = 0.0
        for j in range(i - length + 1, i + 1):
            wsum += closes[j] * volumes[j]
            vsum += volumes[j]
        out[i] = (wsum / vsum) if vsum else None
    return out

def roc(series, n):
    out = [None] * len(series)
    for i in range(len(series)):
        if i < n:
            continue
        prev = series[i - n]
        out[i] = (series[i] - prev) / prev if prev else None
    return out
