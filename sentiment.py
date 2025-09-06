# sentiment.py

# Funding sentiment
def funding_signal(funding_rates):
    if not funding_rates:
        return 0
    avg = sum(funding_rates) / len(funding_rates)
    slope = funding_rates[-1] - funding_rates[0] if len(funding_rates) > 1 else 0
    if avg > 0 and slope > 0:
        return -1   # positive funding often pressures longs
    elif avg < 0 and slope < 0:
        return 1
    return 0

# Momentum sentiment
def momentum_signal(roc3_last, ema9_series):
    e1 = ema9_series[-1]
    e5 = next((x for x in reversed(ema9_series[:-1]) if x is not None), None)
    if e1 is None or e5 is None or roc3_last is None:
        return 0
    slope = e1 - e5
    up = roc3_last > 0 and slope > 0
    down = roc3_last < 0 and slope < 0
    return 1 if up else (-1 if down else 0)
