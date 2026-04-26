import json
import math
from typing import Dict, List

# --- Datamodel imports (provided by the platform) ---
from datamodel import OrderDepth, TradingState, Order

"""
New Strategy v3 — With trend filter to prevent loading into adverse moves.
Only trades underlyings + deep ITM options (BS pricing).
"""

import json
import math
from typing import Dict, List

from datamodel import OrderDepth, TradingState, Order


# ──────────────────────────────── Black‑Scholes ────────────────────────────────

def _erf(x: float) -> float:
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1 / (1 + 0.3275911 * x)
    poly = t * (0.254829592
                + t * (-0.284496736
                       + t * (1.421413741
                              + t * (-1.453152027
                                     + t * 1.061405429))))
    return sign * (1 - poly * math.exp(-x * x))


def _ncdf(x: float) -> float:
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def bs_call_price(S: float, K: float, T: float, sigma: float = 0.0122) -> float:
    intrinsic = max(S - K, 0)
    if T <= 0 or S <= 0:
        return intrinsic
    try:
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * _ncdf(d1) - K * _ncdf(d2)
    except (ValueError, ZeroDivisionError):
        return intrinsic


# ──────────────────────────────── Strategy ────────────────────────────────────

class Trader:
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300,
        "VEV_5200": 300, "VEV_5300": 300,
        "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    ALLOWED_OPTIONS = {"VEV_4000", "VEV_4500"}
    IV = 0.0122

    FAST_ALPHA = 0.12          # fast EMA for fair value
    SLOW_ALPHA = 0.02          # slow EMA for trend filter

    TICKS_PER_DAY = 10_000
    TTE_START = 5.0

    def __init__(self):
        self._state = {}

    def _mid(self, od: OrderDepth) -> float | None:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2
        if od.buy_orders:
            return float(max(od.buy_orders))
        if od.sell_orders:
            return float(min(od.sell_orders))
        return None

    def _ema(self, key: str, value: float, alpha: float) -> float:
        prev = self._state.get(key, value)
        new = alpha * value + (1 - alpha) * prev
        self._state[key] = new
        return new

    @staticmethod
    def _pos(state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    # ──────────────────── Market‑making with trend‑aware limits ────────────────

    def _make_orders(self, product: str, fair: float, spread: float,
                     state: TradingState, vol_frac: float = 0.25,
                     trend_up: bool = True) -> List[Order]:
        """
        Post passive bid/ask around fair, and aggressively take orders.
        Trend filter: if trend is down, max long position = 0 (only allow short).
                      if trend is up,   max short position = 0 (only allow long).
        """
        od = state.order_depths[product]
        pos = self._pos(state, product)
        lim = self.LIMITS[product]
        orders: List[Order] = []

        # Adjust effective position limits based on trend
        if trend_up:
            max_short = 0          # no net short allowed
            max_long = lim
        else:
            max_short = lim        # can be fully short
            max_long = 0           # no net long allowed

        # Aggressive takes – only if within trend‑allowed bounds
        if od.sell_orders:
            ask = min(od.sell_orders)
            if ask < fair - 0.9 * spread:
                # buying -> can only do if pos < max_long
                vol = min(-od.sell_orders[ask], max_long - pos)
                if vol > 0:
                    orders.append(Order(product, ask, vol))
                    pos += vol

        if od.buy_orders:
            bid = max(od.buy_orders)
            if bid > fair + 0.9 * spread:
                # selling -> can only do if pos > -max_short (i.e. pos > -max_short)
                max_sell = pos + max_short   # maximum we can sell to stay within limit
                vol = min(od.buy_orders[bid], max_sell)
                if vol > 0:
                    orders.append(Order(product, bid, -vol))
                    pos -= vol

        # Passive market‑making – only quote on allowed side(s)
        effective_lim = max_short if max_short > 0 else max_long
        inv_ratio = pos / effective_lim if effective_lim != 0 else 0.0
        base_qty = max(1, int(effective_lim * vol_frac))

        # Buy quantity only if we are allowed to hold long
        if max_long > 0:
            buy_qty = max(1, int(base_qty * (1 - inv_ratio)))
            buy_price = math.floor(fair - spread)
            if pos < max_long and buy_qty > 0:
                orders.append(Order(product, buy_price, min(buy_qty, max_long - pos)))
        # Sell quantity only if we are allowed to hold short
        if max_short > 0:
            sell_qty = max(1, int(base_qty * (1 + inv_ratio)))
            sell_price = math.ceil(fair + spread)
            # pos can be negative, max_short is positive limit
            if pos > -max_short and sell_qty > 0:
                orders.append(Order(product, sell_price, -min(sell_qty, pos + max_short)))

        return orders

    def run(self, state: TradingState):
        if state.traderData:
            try:
                self._state = json.loads(state.traderData)
            except Exception:
                self._state = {}

        orders: Dict[str, List[Order]] = {}

        # ── 1. VELVETFRUIT_EXTRACT EMA and trend ──────────────────────────
        ve_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            ve_mid = self._mid(state.order_depths["VELVETFRUIT_EXTRACT"])
        ve_fair = ve_mid if ve_mid else self._state.get("ve_fair", 5255.0)

        ve_ema_fast = self._ema("ve_fair_fast", ve_fair, self.FAST_ALPHA)
        ve_ema_slow = self._ema("ve_fair_slow", ve_fair, self.SLOW_ALPHA)
        trend_up = ve_ema_fast >= ve_ema_slow

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._make_orders(
                "VELVETFRUIT_EXTRACT", ve_ema_fast, spread=3.0,
                state=state, vol_frac=0.25, trend_up=trend_up
            )

        # ── 2. HYDROGEL_PACK (use its own trend, but simple same as VE) ──
        hp_mid = None
        if "HYDROGEL_PACK" in state.order_depths:
            hp_mid = self._mid(state.order_depths["HYDROGEL_PACK"])
        hp_fair = hp_mid if hp_mid else self._state.get("hp_fair", 9990.0)
        # For HP, we just use the same trend as VE (or could have its own)
        hp_ema = self._ema("hp_fair", hp_fair, self.FAST_ALPHA)

        if "HYDROGEL_PACK" in state.order_depths:
            orders["HYDROGEL_PACK"] = self._make_orders(
                "HYDROGEL_PACK", hp_ema, spread=5.0,
                state=state, vol_frac=0.25, trend_up=trend_up
            )

        # ── 3. Deep‑ITM options ───────────────────────────────────────────
        tte = max(0.01, self.TTE_START - state.timestamp / self.TICKS_PER_DAY)
        spot = ve_ema_fast   # use fast EMA for pricing

        for product in self.ALLOWED_OPTIONS:
            if product not in state.order_depths:
                continue

            od = state.order_depths[product]
            pos = self._pos(state, product)
            lim = min(self.LIMITS[product], 60)

            K = float(product.split("_")[1])
            fair_price = bs_call_price(spot, K, tte, self.IV)

            # Apply same trend‑aware limits as underlying
            if trend_up:
                max_long = lim
                max_short = 0
            else:
                max_long = 0
                max_short = lim

            opt_orders: List[Order] = []

            # Take orders (respecting trend limits)
            if od.sell_orders:
                ask = min(od.sell_orders)
                if ask < fair_price - 1.5:
                    vol = min(-od.sell_orders[ask], max_long - pos)
                    if vol > 0:
                        opt_orders.append(Order(product, ask, vol))
                        pos += vol
            if od.buy_orders:
                bid = max(od.buy_orders)
                if bid > fair_price + 1.5:
                    max_sell = pos + max_short
                    vol = min(od.buy_orders[bid], max_sell)
                    if vol > 0:
                        opt_orders.append(Order(product, bid, -vol))
                        pos -= vol

            # Passive quotes (only on allowed sides)
            effective_lim = max_short if max_short > 0 else max_long
            inv_ratio = pos / effective_lim if effective_lim else 0
            base_vol = max(1, int(effective_lim * 0.15))

            if max_long > 0:
                buy_vol = max(1, int(base_vol * (1 - inv_ratio)))
                bid_p = math.floor(fair_price - 2.0)
                if pos < max_long and buy_vol > 0:
                    opt_orders.append(Order(product, bid_p, min(buy_vol, max_long - pos)))
            if max_short > 0:
                sell_vol = max(1, int(base_vol * (1 + inv_ratio)))
                ask_p = math.ceil(fair_price + 2.0)
                if pos > -max_short and sell_vol > 0:
                    opt_orders.append(Order(product, ask_p, -min(sell_vol, pos + max_short)))

            if opt_orders:
                orders[product] = opt_orders

        trader_data = json.dumps(self._state)
        return orders, 0, trader_data