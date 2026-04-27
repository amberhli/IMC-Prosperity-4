# trader.py  — successful algo logic ported faithfully
#
# Key changes vs the previous trader.py:
#   1. IV = 0.0122  (successful algo's calibrated constant; replaces ~0.307 surface)
#   2. TTE in *days* clamped to [0.05, 5.0]  (not years / 365)
#   3. Day detection via ROUND4_OPEN_SIGNATURES + rollover, plus observations fallback
#   4. VFE strategy replaced with _vfe_orders logic (gap-based taker + timer skew)
#   5. Flow state gains mark67_timer and bearish_timer (Mark 67 / Mark 49 on VFE)
#   6. Option strategy uses the successful algo's exact spread / cap / unwind logic
#   7. Hydrogel unchanged (was already correct in trader.py)

from typing import Dict, List, Tuple, Optional
import json
import math

from datamodel import (
    Order,
    OrderDepth,
    TradingState,
    Symbol,
)

# ---------------------------------------------------------------------------
# Position limits  (unchanged from successful algo)
# ---------------------------------------------------------------------------

POSITION_LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK":        200,
    "VELVETFRUIT_EXTRACT":  200,
    "VEV_4000":  300,
    "VEV_4500":  300,
    "VEV_5000":  300,
    "VEV_5100":  300,
    "VEV_5200":  300,
    "VEV_5300":  300,
    "VEV_5400":  300,
    "VEV_5500":  300,
    "VEV_6000":  300,
    "VEV_6500":  300,
}

STRIKES: Dict[str, int] = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

ACTIVE_OPTION_STRIKES = (
    "VEV_5000",
    "VEV_5100",
    "VEV_5200",
    "VEV_5300",
    "VEV_5400",
    "VEV_5500",
)

# ---------------------------------------------------------------------------
# Calibration constants  — taken directly from successful algo
# ---------------------------------------------------------------------------

IV = 0.0122          # single implied-vol constant used for all options

ALPHA_FAST = 0.10    # EMA decay for VFE mid-price

# Day-detection via opening market signatures
ROUND4_OPEN_SIGNATURES = {
    1: {
        "VELVETFRUIT_EXTRACT": 5245.0,
        "VEV_5200": 95.5,
        "VEV_5300": 47.0,
        "VEV_5400": 16.5,
        "VEV_5500": 7.5,
    },
    2: {
        "VELVETFRUIT_EXTRACT": 5267.5,
        "VEV_5200": 104.0,
        "VEV_5300": 53.0,
        "VEV_5400": 17.0,
        "VEV_5500": 6.5,
    },
    3: {
        "VELVETFRUIT_EXTRACT": 5295.5,
        "VEV_5200": 119.5,
        "VEV_5300": 58.0,
        "VEV_5400": 20.5,
        "VEV_5500": 7.0,
    },
}

# ---------------------------------------------------------------------------
# HYDROGEL constants  (unchanged)
# ---------------------------------------------------------------------------

HYDRO_ANCHOR        = 9995
HYDRO_TAKE_BUY      = 18
HYDRO_TAKE_SELL     = 20
HYDRO_PASSIVE_HALF  = 8
HYDRO_CLIP          = 18
HYDRO_PASSIVE_SIZE  = 24

# ---------------------------------------------------------------------------
# VELVETFRUIT market-making constants
# ---------------------------------------------------------------------------

VFE_PASSIVE_SIZE    = 20
VFE_HEDGE_CLIP      = 20
VFE_RESERVATION_K   = 0.04   # reservation-price skew per unit of gap
VFE_MAKE_HALF       = 2.0    # half-spread for passive VFE quotes
VFE_FLOW_TICKS      = 5      # timer duration for Mark 67 / Mark 49 signals

# ---------------------------------------------------------------------------
# Option per-product caps and clip sizes
# ---------------------------------------------------------------------------

OPTION_BASE_CAPS: Dict[str, int] = {
    "VEV_5000": 300,
    "VEV_5100": 300,
    "VEV_5200": 300,
    "VEV_5300": 255,
    "VEV_5400": 300,
    "VEV_5500": 300,
}
OPTION_CLIPS: Dict[str, int] = {
    "VEV_5000": 18,
    "VEV_5100": 16,
    "VEV_5200": 16,
    "VEV_5300": 14,
    "VEV_5400": 16,
    "VEV_5500": 18,
}

# ---------------------------------------------------------------------------
# Pure-Python erf / normal CDF  (from successful algo, no scipy dependency)
# ---------------------------------------------------------------------------

def _erf(x: float) -> float:
    sign = 1.0 if x >= 0 else -1.0
    x = abs(x)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (
        0.254829592
        + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429)))
    )
    return sign * (1.0 - poly * math.exp(-x * x))


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + _erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float = IV) -> float:
    """Black-Scholes call price.  T is in *days* (successful algo convention)."""
    intrinsic = max(S - K, 0.0)
    if T <= 0.0 or S <= 0.0:
        return intrinsic
    try:
        sq = sigma * math.sqrt(T)
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / sq
        return S * _ncdf(d1) - K * _ncdf(d1 - sq)
    except Exception:
        return intrinsic


def bs_delta(S: float, K: float, T: float, sigma: float = IV) -> float:
    """Black-Scholes call delta.  T is in *days*."""
    if T <= 0.0:
        return 1.0 if S > K else 0.0
    try:
        d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
        return _ncdf(d1)
    except Exception:
        return 0.5


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

# ---------------------------------------------------------------------------
# LOB helpers  (kept from trader.py for convenience)
# ---------------------------------------------------------------------------

def get_best_bid(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    if not depth.buy_orders:
        return None, None
    price = max(depth.buy_orders.keys())
    return price, depth.buy_orders[price]


def get_best_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    if not depth.sell_orders:
        return None, None
    price = min(depth.sell_orders.keys())
    return price, depth.sell_orders[price]


def get_mid_price(depth: OrderDepth) -> Optional[float]:
    bid, _ = get_best_bid(depth)
    ask, _ = get_best_ask(depth)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return None

# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------

class Trader:

    # ------------------------------------------------------------------
    # run() — entry point called each tick
    # ------------------------------------------------------------------

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        # 1. Rehydrate persistent state
        if state.traderData:
            try:
                self._state: Dict = json.loads(state.traderData)
            except Exception:
                self._state = {}
        else:
            self._state = {}

        # 2. Update flow signals (Mark 67, Mark 49, Mark 01, Mark 22, Mark 14)
        self._update_flow_state(state)

        # 3. Compute VFE EMA and TTE  (successful algo _ema + _estimate_tte)
        ve_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if ve_depth is not None:
            ve_mid = get_mid_price(ve_depth) or 5250.0
            ve_ema = self._ema("ve_ema", ve_mid, ALPHA_FAST)
        else:
            ve_ema = self._state.get("ve_ema", 5250.0)

        tte = self._estimate_tte(state, ve_ema)

        # 4. Compute net option delta for VFE hedge target
        option_delta = 0.0
        for product in ACTIVE_OPTION_STRIKES:
            pos = state.position.get(product, 0)
            option_delta += pos * bs_delta(ve_ema, STRIKES[product], tte, IV)

        target_vfe = int(round(clamp(-option_delta, -200.0, 200.0)))

        # 5. Build orders
        result: Dict[str, List[Order]] = {}

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = self._vfe_orders(state, ve_ema, target_vfe)

        hydro_orders = self._trade_hydrogel(state)
        if hydro_orders:
            result["HYDROGEL_PACK"] = hydro_orders

        for product in ACTIVE_OPTION_STRIKES:
            if product in state.order_depths:
                result[product] = self._option_orders(state, product, ve_ema, tte)

        # 6. Persist state and return
        serialized = json.dumps(self._state)
        return result, 0, serialized

    # ------------------------------------------------------------------
    # EMA helper
    # ------------------------------------------------------------------

    def _ema(self, key: str, value: float, alpha: float) -> float:
        prev    = self._state.get(key, value)
        updated = alpha * value + (1.0 - alpha) * prev
        self._state[key] = updated
        return updated

    # ------------------------------------------------------------------
    # Day detection  (signature-based + rollover, with observations fallback)
    # ------------------------------------------------------------------

    def _detect_round4_day(self, state: TradingState, spot: float) -> int:
        """Match current market midprices against known day-1/2/3 open signatures."""
        observed = {"VELVETFRUIT_EXTRACT": spot}
        for product in ("VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"):
            od = state.order_depths.get(product)
            if od is None:
                continue
            mid = get_mid_price(od)
            if mid is not None:
                observed[product] = mid

        best_day = 1
        best_err = float("inf")
        for day, ref in ROUND4_OPEN_SIGNATURES.items():
            err   = 0.0
            count = 0
            for product, value in ref.items():
                if product in observed:
                    err   += abs(observed[product] - value)
                    count += 1
            if count == 0:
                continue
            err /= count
            if err < best_err:
                best_err = err
                best_day = day
        return best_day

    def _estimate_tte(self, state: TradingState, spot: float) -> float:
        """
        Return TTE in *days*, clamped to [0.05, 5.0].

        Priority:
          1. state.observations["day"] if present (ground truth).
          2. Signature-based _detect_round4_day() on first tick of the day.
          3. Rollover counter (timestamp wraps back to ~0).
        """
        # --- Try observations first ---
        obs = getattr(state, "observations", None)
        if obs is not None:
            day_val = None
            if hasattr(obs, "__getitem__"):
                day_val = obs.get("day", None)
            elif hasattr(obs, "plainValueObservations"):
                day_val = obs.plainValueObservations.get("day", None)
            if day_val is not None:
                try:
                    current_day = int(day_val)
                    self._state["current_day"] = current_day
                    self._state["prev_ts"]     = state.timestamp
                    tte = 5.0 - float(current_day) - (state.timestamp / 1_000_000.0)
                    return clamp(tte, 0.05, 5.0)
                except (TypeError, ValueError):
                    pass

        # --- Fallback: signature + rollover (exact logic from successful algo) ---
        prev_ts     = self._state.get("prev_ts")
        current_day = self._state.get("current_day")

        if current_day is None:
            current_day = self._detect_round4_day(state, spot)
        elif prev_ts is not None and state.timestamp < prev_ts:
            # Timestamp rolled back → new day
            current_day += 1
        elif prev_ts is None and state.timestamp == 0:
            current_day = self._detect_round4_day(state, spot)

        current_day = int(clamp(current_day, 1, 3))
        self._state["current_day"] = current_day
        self._state["prev_ts"]     = state.timestamp

        tte = 5.0 - float(current_day) - (state.timestamp / 1_000_000.0)
        return clamp(tte, 0.05, 5.0)

    # ------------------------------------------------------------------
    # Flow state update  (successful algo _update_flow_state, exact logic)
    # ------------------------------------------------------------------

    def _update_flow_state(self, state: TradingState) -> None:
        """
        Track informed participant activity across all products.

        VFE:
          mark67_timer  — set when Mark 67 buys or Mark 49 sells (bullish signal).
          bearish_timer — set when Mark 67 sells or Mark 49 buys (bearish signal).

        HYDROGEL:
          hydro_sell_pause — Mark 14 buys large → pause our sell side.
          hydro_buy_pause  — Mark 14 sells large → pause our buy side.

        Options (ATM VEVs):
          option_flow[product] — +qty/5 when Mark 01 buys / Mark 22 sells;
                                  −qty/5 when Mark 01 sells / Mark 22 buys.
          Decays by 0.70 each tick; clamped to [−4, +4].
        """
        # Decay option flow scores
        option_flow = self._state.get("option_flow", {})
        option_flow = {k: v * 0.70 for k, v in option_flow.items()}

        # Decrement timers
        mark67_timer     = max(0, int(self._state.get("mark67_timer", 0))     - 1)
        bearish_timer    = max(0, int(self._state.get("bearish_timer", 0))    - 1)
        hydro_buy_pause  = max(0, int(self._state.get("hydro_buy_pause", 0))  - 1)
        hydro_sell_pause = max(0, int(self._state.get("hydro_sell_pause", 0)) - 1)

        # VFE flow — Mark 67 and Mark 49
        for trade in state.market_trades.get("VELVETFRUIT_EXTRACT", []):
            qty    = int(getattr(trade, "quantity", 0))
            buyer  = getattr(trade, "buyer",  "")
            seller = getattr(trade, "seller", "")

            if buyer == "Mark 67" and qty >= 6:
                mark67_timer = VFE_FLOW_TICKS
            elif seller == "Mark 67" and qty >= 6:
                bearish_timer = VFE_FLOW_TICKS

            if buyer == "Mark 49" and qty >= 8:
                bearish_timer = VFE_FLOW_TICKS
            elif seller == "Mark 49" and qty >= 8:
                mark67_timer = max(mark67_timer, 3)

        # HYDROGEL flow — Mark 14
        for trade in state.market_trades.get("HYDROGEL_PACK", []):
            qty    = int(getattr(trade, "quantity", 0))
            buyer  = getattr(trade, "buyer",  "")
            seller = getattr(trade, "seller", "")

            if buyer == "Mark 14" and qty >= 4:
                hydro_sell_pause = 3
            elif seller == "Mark 14" and qty >= 4:
                hydro_buy_pause  = 3

        # Option flow — Mark 01 and Mark 22
        for product in ACTIVE_OPTION_STRIKES:
            score = float(option_flow.get(product, 0.0))
            for trade in state.market_trades.get(product, []):
                qty    = int(getattr(trade, "quantity", 0))
                buyer  = getattr(trade, "buyer",  "")
                seller = getattr(trade, "seller", "")

                if buyer == "Mark 01" or seller == "Mark 22":
                    score += qty / 5.0
                elif seller == "Mark 01" or buyer == "Mark 22":
                    score -= qty / 5.0

            option_flow[product] = clamp(score, -4.0, 4.0)

        self._state["option_flow"]      = option_flow
        self._state["mark67_timer"]     = mark67_timer
        self._state["bearish_timer"]    = bearish_timer
        self._state["hydro_buy_pause"]  = hydro_buy_pause
        self._state["hydro_sell_pause"] = hydro_sell_pause

    # ------------------------------------------------------------------
    # HYDROGEL  (successful algo _trade_hydrogel, exact logic)
    # ------------------------------------------------------------------

    def _trade_hydrogel(self, state: TradingState) -> List[Order]:
        if "HYDROGEL_PACK" not in state.order_depths:
            return []

        pos   = state.position.get("HYDROGEL_PACK", 0)
        lim   = POSITION_LIMITS["HYDROGEL_PACK"]
        depth = state.order_depths["HYDROGEL_PACK"]

        hydro_buy_pause  = int(self._state.get("hydro_buy_pause",  0))
        hydro_sell_pause = int(self._state.get("hydro_sell_pause", 0))

        best_bid = max(depth.buy_orders.keys())  if depth.buy_orders  else HYDRO_ANCHOR - 1
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else HYDRO_ANCHOR + 1

        orders: List[Order] = []

        # 1. Aggressive taker clips
        if hydro_buy_pause == 0 and best_ask <= HYDRO_ANCHOR - HYDRO_TAKE_BUY:
            take_vol = min(HYDRO_CLIP, lim - pos, -depth.sell_orders.get(best_ask, 0))
            if take_vol > 0:
                orders.append(Order("HYDROGEL_PACK", best_ask, take_vol))
                pos += take_vol

        if hydro_sell_pause == 0 and best_bid >= HYDRO_ANCHOR + HYDRO_TAKE_SELL:
            take_vol = min(HYDRO_CLIP, pos + lim, depth.buy_orders.get(best_bid, 0))
            if take_vol > 0:
                orders.append(Order("HYDROGEL_PACK", best_bid, -take_vol))
                pos -= take_vol

        # 2. Passive maker clips with inventory skew
        inv_ratio = pos / max(lim, 1)
        bid_price = math.floor(HYDRO_ANCHOR - HYDRO_PASSIVE_HALF - (inv_ratio * 3.0))
        ask_price = math.ceil (HYDRO_ANCHOR + HYDRO_PASSIVE_HALF - (inv_ratio * 3.0))

        buy_size  = int(HYDRO_PASSIVE_SIZE * (1.0 - inv_ratio))
        sell_size = int(HYDRO_PASSIVE_SIZE * (1.0 + inv_ratio))

        buy_size  = min(buy_size,  lim - pos)
        sell_size = min(sell_size, pos + lim)

        if buy_size > 0 and hydro_buy_pause == 0:
            safe_bid = min(bid_price, best_ask - 1)
            orders.append(Order("HYDROGEL_PACK", safe_bid, buy_size))

        if sell_size > 0 and hydro_sell_pause == 0:
            safe_ask = max(ask_price, best_bid + 1)
            orders.append(Order("HYDROGEL_PACK", safe_ask, -sell_size))

        return orders

    # ------------------------------------------------------------------
    # VELVETFRUIT  (successful algo _vfe_orders, exact logic)
    # ------------------------------------------------------------------

    def _vfe_orders(
        self,
        state: TradingState,
        fair: float,
        target_pos: int,
    ) -> List[Order]:
        """
        Gap-based taker to close the delta gap quickly, then passive quotes
        around reservation price = fair − K * gap, skewed by Mark 67 / bearish
        timers.
        """
        product = "VELVETFRUIT_EXTRACT"
        od      = state.order_depths.get(product)
        if od is None or not od.buy_orders or not od.sell_orders:
            return []

        pos, lim = state.position.get(product, 0), POSITION_LIMITS[product]
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        mark67_timer  = int(self._state.get("mark67_timer",  0))
        bearish_timer = int(self._state.get("bearish_timer", 0))

        gap         = target_pos - pos
        reservation = fair - VFE_RESERVATION_K * gap
        orders: List[Order] = []

        # Aggressive taker to close large gap
        if gap > 15 and best_ask <= reservation + 1.0:
            qty = min(VFE_HEDGE_CLIP, gap, lim - pos, -od.sell_orders[best_ask])
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                pos += qty
                gap  = target_pos - pos

        if gap < -15 and best_bid >= reservation - 1.0:
            qty = min(VFE_HEDGE_CLIP, -gap, lim + pos, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                pos -= qty
                gap  = target_pos - pos

        # Passive quotes around reservation
        bid_quote = math.floor(reservation - VFE_MAKE_HALF)
        ask_quote = math.ceil (reservation + VFE_MAKE_HALF)

        if mark67_timer > 0:
            bid_quote += 1
            ask_quote += 1
        elif bearish_timer > 0:
            bid_quote -= 1
            ask_quote -= 1

        # Never cross the existing book
        bid_quote = min(best_ask - 1, bid_quote)
        ask_quote = max(best_bid + 1, ask_quote)

        buy_qty  = min(VFE_PASSIVE_SIZE, lim - pos)
        sell_qty = min(VFE_PASSIVE_SIZE, lim + pos)

        if buy_qty  > 0 and gap > -40 and bid_quote > 0:
            orders.append(Order(product, bid_quote,  buy_qty))
        if sell_qty > 0 and gap <  40 and ask_quote > 0:
            orders.append(Order(product, ask_quote, -sell_qty))

        return orders

    # ------------------------------------------------------------------
    # Options  (successful algo _option_orders, exact logic)
    # ------------------------------------------------------------------

    def _option_orders(
        self,
        state: TradingState,
        product: str,
        spot: float,
        tte: float,
    ) -> List[Order]:
        od = state.order_depths.get(product)
        if od is None or not od.buy_orders or not od.sell_orders:
            return []

        pos, _ = state.position.get(product, 0), POSITION_LIMITS[product]
        pos     = state.position.get(product, 0)
        strike  = STRIKES[product]
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        # Fair value with flow adjustment
        fair         = bs_call(spot, strike, tte, IV)
        strike_flow  = float(self._state.get("option_flow", {}).get(product, 0.0))
        fair        += 0.8 * strike_flow

        intrinsic    = max(spot - strike, 0.0)
        time_value   = max(0.5, fair - intrinsic)
        base_half    = max(1.0, time_value * 0.04)

        delta        = bs_delta(spot, strike, tte, IV)
        delta_score  = max(0.0, 1.0 - abs(delta - 0.5) / 0.25)
        sell_half    = base_half * (1.0 + 0.45 * delta_score)

        cap  = OPTION_BASE_CAPS[product]
        clip = OPTION_CLIPS[product]

        # Tighten cap near ATM (high vega = more delta risk per contract)
        if delta_score > 0.0:
            cap = min(cap, int(OPTION_BASE_CAPS[product] * (1.0 - 0.25 * delta_score)))

        # Product-specific bull adjustments
        if product == "VEV_5200":
            if int(self._state.get("mark67_timer", 0)) > 0:
                sell_half += 0.6
                fair      += 0.3
                cap        = int(cap * 0.92)
        elif product == "VEV_5300":
            if int(self._state.get("mark67_timer", 0)) > 0:
                sell_half += 0.2

        orders: List[Order] = []

        # --- Unwind any accidental long positions first ---
        if pos > 0:
            qty = min(pos, clip, od.buy_orders[best_bid])
            if qty > 0 and best_bid >= fair - 0.25 * sell_half:
                orders.append(Order(product, best_bid, -qty))
                pos -= qty
            if pos > 0:
                unwind_ask = max(best_bid + 1, min(best_ask, math.ceil(fair)))
                orders.append(Order(product, unwind_ask, -min(pos, clip)))
            return orders

        remaining_short = max(0, cap + pos)   # pos is ≤ 0 here

        # --- Aggressive SELL: hit bid when it is clearly above FV ---
        if remaining_short > 0 and best_bid >= fair + 0.55 * sell_half:
            qty = min(clip, remaining_short, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                pos             -= qty
                remaining_short -= qty

        # --- Passive ask: post at FV + sell_half ---
        improve   = 1 if strike_flow > 1.25 and (best_ask - best_bid) >= 2 else 0
        ask_target = math.ceil(fair + sell_half)
        ask_quote  = max(
            best_bid + 1,
            min(best_ask - improve if improve else best_ask, ask_target),
        )
        if remaining_short > 0:
            post_qty = min(clip, remaining_short)
            if post_qty > 0:
                orders.append(Order(product, ask_quote, -post_qty))

        # --- Passive cover bid: slowly buy back short ---
        if pos < 0:
            cover_half = max(1.0, base_half * 0.9)
            bid_quote  = min(best_bid, math.floor(fair - cover_half))
            if bid_quote > 0:
                cover_qty = min(
                    max(2, abs(pos) // 12),
                    abs(pos),
                    max(4, clip // 2),
                )
                if cover_qty > 0:
                    orders.append(Order(product, bid_quote, cover_qty))

        return orders