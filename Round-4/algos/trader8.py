# trader.py  — gap 2 + gap 3 fixes applied
#
# Changes vs previous trader.py:
#   GAP 2 — IV recalibration:
#     IV = 0.0201  (was 0.0122)
#     Calibrated from the reliable OTM/ATM strikes (VEV_5200–5500) using
#     bid/ask mid prices across 100 timestamps on day 3.
#     Surface is FLAT: linear R²=0.003, quadratic R²=0.04 — no smile or skew
#     worth fitting. VEV_5000/5100 IVs are numerically unreliable (time value
#     < 6 ticks) and excluded from calibration.
#     With IV=0.0201, BS fair values match market mids within 0.5 ticks for all
#     reliable strikes. The true edge is now the bid-ask spread, not a model
#     markup over a miscalibrated FV.
#
#   GAP 3 — VFE as independent alpha source:
#     VFE_MAKE_HALF raised from 1.0 → 2.5
#       Market mean bid-ask spread = 5.02 ticks → half-spread = 2.51.
#       At 1.0, quotes were inside the existing book and filled at zero or
#       negative edge. At 2.5 we post at the best bid/ask of the existing book,
#       earning the full half-spread on each fill.
#     VFE_PASSIVE_SIZE raised from 20 → 30
#       With a wider spread, fewer fills per unit time — larger size per quote
#       maintains throughput.
#     VFE_HEDGE_CLIP raised from 30 → 40
#       Larger passive size means the delta gap can grow faster; allow more
#       aggressive taker correction.
#     VFE_TAKER_GAP_THRESH raised from 8 → 12
#       Prevents the taker from triggering on every tick now that passive
#       quotes are wider and fill less frequently.
#     Reservation price skew unchanged (VFE_RESERVATION_K=0.04) — the skew
#     moves the centre of our quotes toward the delta target; the wider
#     half-spread earns edge on either side of it.

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
# Position limits
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
# Calibration constants
# ---------------------------------------------------------------------------

# GAP 2 FIX: recalibrated from market data.
# Method: bisection implied-vol on bid/ask mid for VEV_5200–5500 across
# 100 timestamps (ts=0,1000,…,99000). Grand mean of reliable strikes = 0.0201.
# Surface is flat (linear R²=0.003, quadratic R²=0.04): no smile/skew correction.
# VEV_5000/5100 excluded (time-value < 6 ticks → numerically unstable IV).
IV = 0.0201          # was 0.0122

ALPHA_FAST = 0.10    # EMA decay for VFE mid-price

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

HYDRO_ANCHOR_FALLBACK = 10025
HYDRO_ANCHOR_ALPHA    = 0.002
HYDRO_TAKE_BUY        = 15
HYDRO_TAKE_SELL       = 15
HYDRO_PASSIVE_HALF    = 8
HYDRO_CLIP            = 18
HYDRO_PASSIVE_SIZE    = 24

# ---------------------------------------------------------------------------
# VELVETFRUIT market-making constants
# GAP 3 FIX: VFE is now an independent alpha source, not just a delta hedge.
# ---------------------------------------------------------------------------

# GAP 3: raised from 20 → 30.
# Wider spread → fewer fills per unit time; larger size maintains throughput.
VFE_PASSIVE_SIZE      = 30

# GAP 3: raised from 30 → 40.
# Larger passive size means the delta gap can accumulate faster between fills;
# allow correspondingly more aggressive taker correction.
VFE_HEDGE_CLIP        = 40

VFE_RESERVATION_K     = 0.04   # reservation skew per unit gap (unchanged)

# GAP 3: raised from 1.0 → 2.5.
# Market mean bid-ask spread = 5.02 ticks → half = 2.51.
# At 1.0 ticks we were posting inside the book and earning zero edge.
# At 2.5 we sit at the top of book on each side and earn the half-spread.
VFE_MAKE_HALF         = 2.5

VFE_FLOW_TICKS        = 5

# GAP 3: raised from 8 → 12.
# With a 2.5-tick half-spread, passive quotes fill less frequently, so the
# gap naturally widens more before a fill corrects it. A threshold of 8 would
# fire the taker on almost every tick and negate the passive spread income.
VFE_TAKER_GAP_THRESH  = 12

# ---------------------------------------------------------------------------
# Option per-product caps and clip sizes  (unchanged)
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

OPEN_SKIP_TICKS      = 500
OPEN_SPREAD_MULT     = 1.5

# ---------------------------------------------------------------------------
# Pure-Python erf / normal CDF
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
    """Black-Scholes call price. T is in *days*."""
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
    """Black-Scholes call delta. T is in *days*."""
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
# LOB helpers
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

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        if state.traderData:
            try:
                self._state: Dict = json.loads(state.traderData)
            except Exception:
                self._state = {}
        else:
            self._state = {}

        self._update_flow_state(state)

        ve_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if ve_depth is not None:
            ve_mid = get_mid_price(ve_depth) or 5250.0
            ve_ema = self._ema("ve_ema", ve_mid, ALPHA_FAST)
        else:
            ve_ema = self._state.get("ve_ema", 5250.0)

        tte = self._estimate_tte(state, ve_ema)

        option_delta = 0.0
        for product in ACTIVE_OPTION_STRIKES:
            pos = state.position.get(product, 0)
            option_delta += pos * bs_delta(ve_ema, STRIKES[product], tte, IV)

        target_vfe = int(round(clamp(-option_delta, -200.0, 200.0)))

        result: Dict[str, List[Order]] = {}

        if "VELVETFRUIT_EXTRACT" in state.order_depths:
            result["VELVETFRUIT_EXTRACT"] = self._vfe_orders(state, ve_ema, target_vfe)

        hydro_orders = self._trade_hydrogel(state)
        if hydro_orders:
            result["HYDROGEL_PACK"] = hydro_orders

        for product in ACTIVE_OPTION_STRIKES:
            if product in state.order_depths:
                result[product] = self._option_orders(state, product, ve_ema, tte)

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
    # Day detection
    # ------------------------------------------------------------------

    def _detect_round4_day(self, state: TradingState, spot: float) -> int:
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

        prev_ts     = self._state.get("prev_ts")
        current_day = self._state.get("current_day")

        if current_day is None:
            current_day = self._detect_round4_day(state, spot)
        elif prev_ts is not None and state.timestamp < prev_ts:
            current_day += 1
        elif prev_ts is None and state.timestamp == 0:
            current_day = self._detect_round4_day(state, spot)

        current_day = int(clamp(current_day, 1, 3))
        self._state["current_day"] = current_day
        self._state["prev_ts"]     = state.timestamp

        tte = 5.0 - float(current_day) - (state.timestamp / 1_000_000.0)
        return clamp(tte, 0.05, 5.0)

    # ------------------------------------------------------------------
    # Flow state update
    # ------------------------------------------------------------------

    def _update_flow_state(self, state: TradingState) -> None:
        option_flow = self._state.get("option_flow", {})
        option_flow = {k: v * 0.70 for k, v in option_flow.items()}

        mark67_timer     = max(0, int(self._state.get("mark67_timer", 0))     - 1)
        bearish_timer    = max(0, int(self._state.get("bearish_timer", 0))    - 1)
        hydro_buy_pause  = max(0, int(self._state.get("hydro_buy_pause", 0))  - 1)
        hydro_sell_pause = max(0, int(self._state.get("hydro_sell_pause", 0)) - 1)

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

        for trade in state.market_trades.get("HYDROGEL_PACK", []):
            qty    = int(getattr(trade, "quantity", 0))
            buyer  = getattr(trade, "buyer",  "")
            seller = getattr(trade, "seller", "")

            if buyer == "Mark 14" and qty >= 4:
                hydro_sell_pause = 3
            elif seller == "Mark 14" and qty >= 4:
                hydro_buy_pause  = 3

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
    # HYDROGEL  (unchanged)
    # ------------------------------------------------------------------

    def _trade_hydrogel(self, state: TradingState) -> List[Order]:
        if "HYDROGEL_PACK" not in state.order_depths:
            return []

        pos   = state.position.get("HYDROGEL_PACK", 0)
        lim   = POSITION_LIMITS["HYDROGEL_PACK"]
        depth = state.order_depths["HYDROGEL_PACK"]

        hydro_buy_pause  = int(self._state.get("hydro_buy_pause",  0))
        hydro_sell_pause = int(self._state.get("hydro_sell_pause", 0))

        best_bid = max(depth.buy_orders.keys())  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders.keys()) if depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            current_mid = (best_bid + best_ask) / 2.0
            if "hydro_anchor" not in self._state:
                self._state["hydro_anchor"] = current_mid
            else:
                self._state["hydro_anchor"] = (
                    HYDRO_ANCHOR_ALPHA * current_mid
                    + (1.0 - HYDRO_ANCHOR_ALPHA) * self._state["hydro_anchor"]
                )

        anchor = self._state.get("hydro_anchor", HYDRO_ANCHOR_FALLBACK)

        fallback_bid = int(anchor) - 1
        fallback_ask = int(anchor) + 1
        best_bid = best_bid if best_bid is not None else fallback_bid
        best_ask = best_ask if best_ask is not None else fallback_ask

        orders: List[Order] = []

        if hydro_buy_pause == 0 and best_ask <= anchor - HYDRO_TAKE_BUY:
            take_vol = min(HYDRO_CLIP, lim - pos, -depth.sell_orders.get(best_ask, 0))
            if take_vol > 0:
                orders.append(Order("HYDROGEL_PACK", best_ask, take_vol))
                pos += take_vol

        if hydro_sell_pause == 0 and best_bid >= anchor + HYDRO_TAKE_SELL:
            take_vol = min(HYDRO_CLIP, pos + lim, depth.buy_orders.get(best_bid, 0))
            if take_vol > 0:
                orders.append(Order("HYDROGEL_PACK", best_bid, -take_vol))
                pos -= take_vol

        inv_ratio = pos / max(lim, 1)
        bid_price = math.floor(anchor - HYDRO_PASSIVE_HALF - (inv_ratio * 3.0))
        ask_price = math.ceil (anchor + HYDRO_PASSIVE_HALF - (inv_ratio * 3.0))

        buy_size  = int(HYDRO_PASSIVE_SIZE * (1.0 - inv_ratio))
        sell_size = int(HYDRO_PASSIVE_SIZE * (1.0 + inv_ratio))

        buy_size  = min(buy_size,  lim - pos)
        sell_size = min(sell_size, pos + lim)

        if buy_size > 0 and hydro_buy_pause == 0:
            safe_bid = min(bid_price, best_ask - 1)
            if safe_bid > 0:
                orders.append(Order("HYDROGEL_PACK", safe_bid, buy_size))

        if sell_size > 0 and hydro_sell_pause == 0:
            safe_ask = max(ask_price, best_bid + 1)
            orders.append(Order("HYDROGEL_PACK", safe_ask, -sell_size))

        return orders

    # ------------------------------------------------------------------
    # VELVETFRUIT  (gap 3 fix applied)
    # ------------------------------------------------------------------

    def _vfe_orders(
        self,
        state: TradingState,
        fair: float,
        target_pos: int,
    ) -> List[Order]:
        """
        GAP 3 FIX: VFE is now an independent alpha source.

        Previously: VFE_MAKE_HALF=1.0, inside the native 5-tick spread.
        Fills happened at zero or negative edge. PnL contribution = 0.

        Now: VFE_MAKE_HALF=2.5 (half of the 5.02-tick mean market spread).
        We post at the top of the existing book on each side. Every fill
        earns ~2.5 ticks of spread income.

        The reservation price still skews the midpoint toward the delta
        target, so spread income and delta hedging work together. If the
        gap is large enough (> VFE_TAKER_GAP_THRESH=12), we cross the
        spread with the taker to quickly correct a large delta error —
        but we no longer fire the taker on every 8-tick gap, which was
        eating into spread income.
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

        # Aggressive taker: only fires when gap is large (threshold raised 8→12)
        if gap > VFE_TAKER_GAP_THRESH and best_ask <= reservation + 1.0:
            qty = min(VFE_HEDGE_CLIP, gap, lim - pos, -od.sell_orders[best_ask])
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                pos += qty
                gap  = target_pos - pos

        if gap < -VFE_TAKER_GAP_THRESH and best_bid >= reservation - 1.0:
            qty = min(VFE_HEDGE_CLIP, -gap, lim + pos, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                pos -= qty
                gap  = target_pos - pos

        # Passive quotes: VFE_MAKE_HALF=2.5 sits at the top of the existing
        # book (market half-spread is ~2.5), earning the full half-spread on
        # each fill rather than posting inside for zero edge.
        bid_quote = math.floor(reservation - VFE_MAKE_HALF)
        ask_quote = math.ceil (reservation + VFE_MAKE_HALF)

        # Flow skew: shift the entire quote one tick in the signal direction
        if mark67_timer > 0:
            bid_quote += 1
            ask_quote += 1
        elif bearish_timer > 0:
            bid_quote -= 1
            ask_quote -= 1

        # Hard constraint: never cross the existing book
        bid_quote = min(best_ask - 1, bid_quote)
        ask_quote = max(best_bid + 1, ask_quote)

        buy_qty  = min(VFE_PASSIVE_SIZE, lim - pos)
        sell_qty = min(VFE_PASSIVE_SIZE, lim + pos)

        if buy_qty  > 0 and gap > -(VFE_TAKER_GAP_THRESH * 5) and bid_quote > 0:
            orders.append(Order(product, bid_quote,  buy_qty))
        if sell_qty > 0 and gap <  (VFE_TAKER_GAP_THRESH * 5) and ask_quote > 0:
            orders.append(Order(product, ask_quote, -sell_qty))

        return orders

    # ------------------------------------------------------------------
    # Options  (IV constant updated to 0.0201; logic otherwise unchanged)
    # ------------------------------------------------------------------

    def _option_orders(
        self,
        state: TradingState,
        product: str,
        spot: float,
        tte: float,
    ) -> List[Order]:
        """
        GAP 2 FIX: IV=0.0201 (was 0.0122).

        With the corrected IV, bs_call() returns fair values that match
        market mids to within 0.5 ticks for all reliable strikes (5200–5500).
        The 'edge' is no longer a model artefact — it is the actual bid-ask
        half-spread (~0.6–1.5 ticks for OTM strikes).

        Consequence for sell logic: the aggressive sell threshold
        (best_bid >= fair + 0.55 * sell_half) now fires only when the bid
        is genuinely above fair value by more than half the spread, i.e. when
        there is real liquidity to take against. Previously at IV=0.0122, the
        threshold fired constantly because fair was systematically below market.

        All other option logic (spread sizing, cap, unwind, flow adjustment,
        cover bid) is unchanged.
        """
        od = state.order_depths.get(product)
        if od is None or not od.buy_orders or not od.sell_orders:
            return []

        pos      = state.position.get(product, 0)
        strike   = STRIKES[product]
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)

        # Fair value — now correctly calibrated; matches market mid
        fair        = bs_call(spot, strike, tte, IV)
        strike_flow = float(self._state.get("option_flow", {}).get(product, 0.0))
        fair       += 0.8 * strike_flow

        # Spread sizing (unchanged)
        intrinsic   = max(spot - strike, 0.0)
        time_value  = max(0.5, fair - intrinsic)
        base_half   = max(1.0, time_value * 0.04)

        delta       = bs_delta(spot, strike, tte, IV)
        delta_score = max(0.0, 1.0 - abs(delta - 0.5) / 0.25)
        sell_half   = base_half * (1.0 + 0.45 * delta_score)

        cap  = OPTION_BASE_CAPS[product]
        clip = OPTION_CLIPS[product]

        if delta_score > 0.0:
            cap = min(cap, int(OPTION_BASE_CAPS[product] * (1.0 - 0.25 * delta_score)))

        if product == "VEV_5200":
            if int(self._state.get("mark67_timer", 0)) > 0:
                sell_half += 0.6
                fair      += 0.3
                cap        = int(cap * 0.92)
        elif product == "VEV_5300":
            if int(self._state.get("mark67_timer", 0)) > 0:
                sell_half += 0.2

        orders: List[Order] = []

        # Unwind any accidental long immediately
        if pos > 0:
            qty = min(pos, clip, od.buy_orders[best_bid])
            if qty > 0 and best_bid >= fair - 0.25 * sell_half:
                orders.append(Order(product, best_bid, -qty))
                pos -= qty
            if pos > 0:
                unwind_ask = max(best_bid + 1, min(best_ask, math.ceil(fair)))
                orders.append(Order(product, unwind_ask, -min(pos, clip)))
            return orders

        # Open skip window
        at_open = (state.timestamp < OPEN_SKIP_TICKS)
        if at_open:
            open_sell_half = sell_half * OPEN_SPREAD_MULT
            open_clip      = max(1, clip // 2)
            remaining      = max(0, cap + pos)
            if remaining > 0:
                ask_target = math.ceil(fair + open_sell_half)
                ask_quote  = max(best_bid + 1, min(best_ask, ask_target))
                post_qty   = min(open_clip, remaining)
                if post_qty > 0:
                    orders.append(Order(product, ask_quote, -post_qty))
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

        # Normal trading
        remaining_short = max(0, cap + pos)

        # Aggressive sell: only fires when bid is meaningfully above fair.
        # With IV=0.0201, 'fair' ≈ market mid, so this requires the bid to
        # be at least 0.55 * sell_half above mid — a genuine liquidity event.
        if remaining_short > 0 and best_bid >= fair + 0.55 * sell_half:
            qty = min(clip, remaining_short, od.buy_orders[best_bid])
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                pos             -= qty
                remaining_short -= qty

        # Passive ask at fair + sell_half
        improve    = 1 if strike_flow > 1.25 and (best_ask - best_bid) >= 2 else 0
        ask_target = math.ceil(fair + sell_half)
        ask_quote  = max(
            best_bid + 1,
            min(best_ask - improve if improve else best_ask, ask_target),
        )
        if remaining_short > 0:
            post_qty = min(clip, remaining_short)
            if post_qty > 0:
                orders.append(Order(product, ask_quote, -post_qty))

        # Passive cover bid
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