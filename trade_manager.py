# trade_manager.py
import math

class TradeManager:
    def __init__(self):
        self.state = {"active": False}

    def can_issue(self):
        return not self.state.get("active", False)

    def open(self, side, entry):
        sl_pct = 0.0025   # 0.25%
        tp_pct = sl_pct   # 1:1 RR
        tp = entry * (1 + tp_pct) if side == "buy" else entry * (1 - tp_pct)
        sl = entry * (1 - sl_pct) if side == "buy" else entry * (1 + sl_pct)
        self.state.update({"active": True, "side": side, "entry": entry, "tp": tp, "sl": sl})
        return tp, sl

    def check_close(self, last_price):
        if not self.state.get("active"):
            return None
        side = self.state["side"]
        tp = self.state["tp"]
        sl = self.state["sl"]
        hit = None
        if side == "buy":
            if last_price >= tp:
                hit = "tp"
            elif last_price <= sl:
                hit = "sl"
        else:
            if last_price <= tp:
                hit = "tp"
            elif last_price >= sl:
                hit = "sl"
        if hit:
            self.reset()
        return hit

    def reset(self):
        self.state = {"active": False}
