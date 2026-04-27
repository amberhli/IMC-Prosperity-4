"""
Strategy v4 — Fixes for drawdown issues identified in v2:

  FIX 1: Raw mid used for aggressive takes (not the lagging EMA).
          EMA is only used to set passive quote prices.

  FIX 2: Hard internal position cap on VELVETFRUIT_EXTRACT (SPOT_CAP).
          Prevents the strategy from accumulating 130–176 unit longs.

  FIX 3: Combined delta cap (DELTA_CAP) across spot + deep-ITM options.
          Each deep-ITM option has delta ≈ 1, so they count 1:1 toward
          the cap together with the spot position.

  FIX 4: Trend-filter guard on aggressive takes.
          If EMA has drifted more than TREND_GUARD ticks above/below the
          raw mid, the market is trending and we suppress aggressive takes
          in the direction that adds to an already-stale position.
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
    # Exchange position limits
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
    FAST_ALPHA = 0.12
    SLOW_ALPHA = 0.03

    TICKS_PER_DAY = 10_000
    TTE_START     = 5.0

    # ── FIX 2: internal position cap for VELVETFRUIT_EXTRACT ──────────────────
    # v2 allowed up to the full exchange limit of 200, leading to 130–176 unit
    # longs during downtrends.  Cap at 60 to keep directional risk manageable.
    SPOT_CAP = 60

    # ── FIX 3: combined delta cap across spot + deep-ITM options ─────────────
    # Deep-ITM calls have delta ≈ 1, so a long in VEV_4000 or VEV_4500 is
    # almost identical to holding another unit of spot.  Cap total exposure.
    DELTA_CAP = 80

    # ── FIX 4: trend-filter threshold (in price ticks) ───────────────────────
    # If EMA deviates more than this from the raw mid, the market is trending.
    # In a downtrend (EMA > raw_mid + TREND_GUARD) we suppress buys.
    # In an uptrend (EMA < raw_mid - TREND_GUARD) we suppress sells.
    TREND_GUARD = 4.0

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
        new  = alpha * value + (1 - alpha) * prev
        self._state[key] = new
        return new

    @staticmethod
    def _pos(state: TradingState, product: str) -> int:
        return state.position.get(product, 0)

    # ──────────────────── Market‑making with aggressive takes ─────────────────

    def _make_orders(
        self,
        product:    str,
        raw_mid:    float,          # FIX 1: raw mid for aggressive-take logic
        ema_fair:   float,          # EMA-smoothed fair for passive quotes
        spread:     float,
        state:      TradingState,
        vol_frac:   float = 0.25,
        pos_cap:    int   = None,   # FIX 2: optional tighter cap
    ) -> List[Order]:
        """
        Aggressive takes are gated on raw_mid (not the lagging EMA) and on
        the trend filter.  Passive quotes are still centred on ema_fair so
        they remain smooth.
        """
        od  = state.order_depths[product]
        pos = self._pos(state, product)
        lim = pos_cap if pos_cap is not None else self.LIMITS[product]
        orders: List[Order] = []

        # ── FIX 4: trend filter ───────────────────────────────────────────────
        drift = ema_fair - raw_mid          # positive → EMA above market (downtrend lag)
        trending_down = drift >  self.TREND_GUARD   # EMA inflated → suppress buys
        trending_up   = drift < -self.TREND_GUARD   # EMA deflated → suppress sells

        # ── FIX 1: aggressive takes use raw_mid, not ema_fair ─────────────────
        if od.sell_orders and not trending_down:
            ask = min(od.sell_orders)
            if ask < raw_mid - 0.9 * spread:
                vol = min(-od.sell_orders[ask], lim - pos)
                if vol > 0:
                    orders.append(Order(product, ask, vol))
                    pos += vol

        if od.buy_orders and not trending_up:
            bid = max(od.buy_orders)
            if bid > raw_mid + 0.9 * spread:
                vol = min(od.buy_orders[bid], lim + pos)
                if vol > 0:
                    orders.append(Order(product, bid, -vol))
                    pos -= vol

        # ── Passive market-making (skewed by inventory, centred on EMA) ───────
        inv_ratio = pos / lim if lim != 0 else 0.0
        base_qty  = max(1, int(lim * vol_frac))

        buy_qty  = max(1, int(base_qty * (1 - inv_ratio)))
        sell_qty = max(1, int(base_qty * (1 + inv_ratio)))

        bid_price = math.floor(ema_fair - spread)
        ask_price = math.ceil (ema_fair + spread)

        if pos < lim and buy_qty > 0:
            orders.append(Order(product, bid_price,  buy_qty))
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

        # ── 1. VELVETFRUIT_EXTRACT ────────────────────────────────────────────
        ve_raw = None
        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            ve_raw = self._mid(state.order_depths["VELVETFRUIT_EXTRACT"])

        ve_fair_raw = ve_raw if ve_raw is not None else self._state.get("ve_fair", 5255.0)
        ve_ema      = self._ema("ve_fair", ve_fair_raw, self.FAST_ALPHA)

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            orders["VELVETFRUIT_EXTRACT"] = self._make_orders(
                product  = "VELVETFRUIT_EXTRACT",
                raw_mid  = ve_fair_raw,   # FIX 1
                ema_fair = ve_ema,
                spread   = 3.0,
                state    = state,
                vol_frac = 0.25,
                pos_cap  = self.SPOT_CAP, # FIX 2
            )

        # ── 2. HYDROGEL_PACK ──────────────────────────────────────────────────
        hp_raw = None
        if "HYDROGEL_PACK" in state.order_depths:
            hp_raw = self._mid(state.order_depths["HYDROGEL_PACK"])

        hp_fair_raw = hp_raw if hp_raw is not None else self._state.get("hp_fair", 9990.0)
        hp_ema      = self._ema("hp_fair", hp_fair_raw, self.FAST_ALPHA)

        if "HYDROGEL_PACK" in state.order_depths:
            orders["HYDROGEL_PACK"] = self._make_orders(
                product  = "HYDROGEL_PACK",
                raw_mid  = hp_fair_raw,
                ema_fair = hp_ema,
                spread   = 5.0,
                state    = state,
                vol_frac = 0.25,
            )

        # ── 3. Deep‑ITM options ───────────────────────────────────────────────
        tte  = max(0.01, self.TTE_START - state.timestamp / self.TICKS_PER_DAY)
        spot = ve_ema   # BS pricing still uses smoothed spot (appropriate)

        # FIX 3: compute current spot position for delta-cap bookkeeping
        spot_pos = self._pos(state, "VELVETFRUIT_EXTRACT")

        for product in self.ALLOWED_OPTIONS:
            if product not in state.order_depths:
                continue

            od      = state.order_depths[product]
            opt_pos = self._pos(state, product)

            # FIX 3: remaining delta budget for this option
            # Each deep-ITM option contributes ~1 delta unit.
            # Budget = DELTA_CAP - |current net delta|
            net_delta   = spot_pos + sum(
                self._pos(state, p) for p in self.ALLOWED_OPTIONS
                if p in state.order_depths
            )
            delta_room_long  = self.DELTA_CAP - net_delta    # how many more longs we can add
            delta_room_short = self.DELTA_CAP + net_delta    # how many more shorts we can add

            lim_raw = min(self.LIMITS[product], 60)          # per-option hard cap (unchanged)

            # Effective buy/sell limits considering both per-option cap and delta cap
            eff_buy_lim  = min(lim_raw - opt_pos, max(0, delta_room_long))
            eff_sell_lim = min(lim_raw + opt_pos, max(0, delta_room_short))

            K          = float(product.split("_")[1])
            fair_price = bs_call_price(spot, K, tte, self.IV)

            # FIX 4: same trend filter applies to options (they move with spot)
            drift          = ve_ema - ve_fair_raw if ve_fair_raw is not None else 0.0
            trending_down  = drift >  self.TREND_GUARD
            trending_up    = drift < -self.TREND_GUARD

            opt_orders: List[Order] = []

            # FIX 1 & 4: use raw option mid for aggressive take threshold;
            #             suppress directional takes during trending markets.
            opt_raw = self._mid(od)
            take_ref = opt_raw if opt_raw is not None else fair_price

            if od.sell_orders and not trending_down:
                ask = min(od.sell_orders)
                if ask < take_ref - 1.5:
                    vol = min(-od.sell_orders[ask], eff_buy_lim)
                    if vol > 0:
                        opt_orders.append(Order(product, ask, vol))
                        opt_pos      += vol
                        eff_buy_lim  -= vol

            if od.buy_orders and not trending_up:
                bid = max(od.buy_orders)
                if bid > take_ref + 1.5:
                    vol = min(od.buy_orders[bid], eff_sell_lim)
                    if vol > 0:
                        opt_orders.append(Order(product, bid, -vol))
                        opt_pos      -= vol
                        eff_sell_lim -= vol

            # Passive quotes — still centred on BS fair value
            inv_ratio = opt_pos / lim_raw if lim_raw else 0
            base_vol  = max(1, int(lim_raw * 0.15))
            buy_vol   = max(0, int(base_vol * (1 - inv_ratio)))
            sell_vol  = max(0, int(base_vol * (1 + inv_ratio)))

            # Also gate passive quote sizes by remaining delta budget
            buy_vol  = min(buy_vol,  max(0, eff_buy_lim))
            sell_vol = min(sell_vol, max(0, eff_sell_lim))

            bid_p = math.floor(fair_price - 2.0)
            ask_p = math.ceil (fair_price + 2.0)

            if opt_pos < lim_raw and buy_vol > 0:
                opt_orders.append(Order(product, bid_p,  buy_vol))
            if opt_pos > -lim_raw and sell_vol > 0:
                opt_orders.append(Order(product, ask_p, -sell_vol))

            if opt_orders:
                orders[product] = opt_orders

        # Persist state
        trader_data = json.dumps(self._state)
        return orders, 0, trader_data