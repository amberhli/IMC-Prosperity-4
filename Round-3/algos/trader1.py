"""
New Strategy v2 — Market-make the underlyings + deep ITM options only (BS pricing)
"""

import json
import math
from typing import Dict, List

from datamodel import OrderDepth, TradingState, Order


# ──────────────────────────────── Black‑Scholes ────────────────────────────────

def _erf(x: float) -> float:
    """Error function approximation (Abramowitz & Stegun)."""
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
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + _erf(x / math.sqrt(2)))


def bs_call_price(S: float, K: float, T: float, sigma: float = 0.0122) -> float:
    """European call price with zero interest rate, T in days."""
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
    # Exchange position limits (we use stricter internal caps)
    LIMITS: Dict[str, int] = {
        "HYDROGEL_PACK":       200,
        "VELVETFRUIT_EXTRACT": 200,
        "VEV_4000": 300, "VEV_4500": 300,
        "VEV_5000": 300, "VEV_5100": 300,
        "VEV_5200": 300, "VEV_5300": 300,
        "VEV_5400": 300, "VEV_5500": 300,
        "VEV_6000": 300, "VEV_6500": 300,
    }

    # Only trade these two deep in-the-money options
    ALLOWED_OPTIONS = {"VEV_4000", "VEV_4500"}

    # Implied volatility per sqrt(day)
    IV = 0.0122

    # EMA smoothing coefficients
    FAST_ALPHA = 0.12
    SLOW_ALPHA = 0.03

    TICKS_PER_DAY = 10_000
    TTE_START = 5.0          # days to expiry at the start of this round

    def __init__(self):
        self._state = {}

    # ──────────────────────────── Utilities ──────────────────────────────────

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

    # ──────────────────── Market‑making with aggressive takes ────────────────

    def _make_orders(self, product: str, fair: float, spread: float,
                     state: TradingState, vol_frac: float = 0.25) -> List[Order]:
        """
        Post passive bid/ask around fair, and aggressively take any orders
        that cross our fair value by more than 0.9 * spread.
        """
        od = state.order_depths[product]
        pos = self._pos(state, product)
        lim = self.LIMITS[product]
        orders: List[Order] = []

        # Aggressive takes
        if od.sell_orders:
            ask = min(od.sell_orders)
            if ask < fair - 0.9 * spread:
                vol = min(-od.sell_orders[ask], lim - pos)
                if vol > 0:
                    orders.append(Order(product, ask, vol))
                    pos += vol

        if od.buy_orders:
            bid = max(od.buy_orders)
            if bid > fair + 0.9 * spread:
                vol = min(od.buy_orders[bid], lim + pos)
                if vol > 0:
                    orders.append(Order(product, bid, -vol))
                    pos -= vol

        # Passive market‑making (skewed by inventory)
        inv_ratio = pos / lim if lim != 0 else 0.0
        base_qty = max(1, int(lim * vol_frac))

        buy_qty  = max(1, int(base_qty * (1 - inv_ratio)))
        sell_qty = max(1, int(base_qty * (1 + inv_ratio)))

        bid_price = math.floor(fair - spread)
        ask_price = math.ceil(fair + spread)

        if pos < lim and buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        if pos > -lim and sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))

        return orders

    # ──────────────────────────────── Main ───────────────────────────────────

    def run(self, state: TradingState):
        # Restore persistent state
        if state.traderData:
            try:
                self._state = json.loads(state.traderData)
            except Exception:
                self._state = {}

        orders: Dict[str, List[Order]] = {}

        # ── 1. VELVETFRUIT_EXTRACT ────────────────────────────────────────
        ve_mid = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            ve_mid = self._mid(state.order_depths["VELVETFRUIT_EXTRACT"])
        ve_fair = ve_mid if ve_mid else self._state.get("ve_fair", 5255.0)
        ve_ema = self._ema("ve_fair", ve_fair, self.FAST_ALPHA)

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._make_orders(
                "VELVETFRUIT_EXTRACT", ve_ema, spread=3.0, state=state, vol_frac=0.25
            )

        # ── 2. HYDROGEL_PACK ──────────────────────────────────────────────
        hp_mid = None
        if "HYDROGEL_PACK" in state.order_depths:
            hp_mid = self._mid(state.order_depths["HYDROGEL_PACK"])
        hp_fair = hp_mid if hp_mid else self._state.get("hp_fair", 9990.0)
        hp_ema = self._ema("hp_fair", hp_fair, self.FAST_ALPHA)

        if "HYDROGEL_PACK" in state.order_depths:
            orders["HYDROGEL_PACK"] = self._make_orders(
                "HYDROGEL_PACK", hp_ema, spread=5.0, state=state, vol_frac=0.25
            )

        # ── 3. Deep‑ITM options ───────────────────────────────────────────
        tte = max(0.01, self.TTE_START - state.timestamp / self.TICKS_PER_DAY)
        spot = ve_ema

        for product in self.ALLOWED_OPTIONS:
            if product not in state.order_depths:
                continue

            od = state.order_depths[product]
            pos = self._pos(state, product)
            lim = min(self.LIMITS[product], 60)        # internal hard cap ±60

            K = float(product.split("_")[1])           # extract strike from name

            # Fair value from Black-Scholes
            fair_price = bs_call_price(spot, K, tte, self.IV)

            opt_orders: List[Order] = []

            # Take mispriced orders
            if od.sell_orders:
                ask = min(od.sell_orders)
                if ask < fair_price - 1.5:
                    vol = min(-od.sell_orders[ask], lim - pos)
                    if vol > 0:
                        opt_orders.append(Order(product, ask, vol))
                        pos += vol

            if od.buy_orders:
                bid = max(od.buy_orders)
                if bid > fair_price + 1.5:
                    vol = min(od.buy_orders[bid], lim + pos)
                    if vol > 0:
                        opt_orders.append(Order(product, bid, -vol))
                        pos -= vol

            # Post passive orders
            inv_ratio = pos / lim if lim else 0
            base_vol = max(1, int(lim * 0.15))
            buy_vol  = max(1, int(base_vol * (1 - inv_ratio)))
            sell_vol = max(1, int(base_vol * (1 + inv_ratio)))

            bid_p = math.floor(fair_price - 2.0)
            ask_p = math.ceil(fair_price + 2.0)

            if pos < lim and buy_vol > 0:
                opt_orders.append(Order(product, bid_p, buy_vol))
            if pos > -lim and sell_vol > 0:
                opt_orders.append(Order(product, ask_p, -sell_vol))

            if opt_orders:
                orders[product] = opt_orders

        # Persist state
        trader_data = json.dumps(self._state)
        return orders, 0, trader_data