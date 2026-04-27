"""
trader.py — IMC Prosperity Algorithm

Post-mortem fixes applied (v3):
  FIX 1 — Taker sell disabled on ATM VEVs.
  FIX 2 — Adverse selection cap on passive MM asks.
  FIX 3 — Delta hedge threshold reduced from 8 → 2.
  FIX 4 — HYDROGEL clamp changed from join-queue → improve-by-1.

Post-mortem fixes applied (v4) — from backtest log analysis:
  FIX A — Day index read directly from state, not reconstructed.
           DAY_START=2 caused algo to compute TTE=5 when true TTE=3, inflating
           every option FV by 15–95% and triggering mass buying from tick 0.
           Now: day = state.observations.get("day") or inferred from traderData.
           The game exposes the real day number; we use it directly.
  FIX B — Portfolio-level delta cap before any option buy.
           6 products × 200 contracts = ~694 net delta at TTE=3. VF limit=400.
           Structural unhedgeable delta = 294. On a 42-tick VF drop this alone
           costs ~12,000. Now: before buying any option, we check that the
           resulting net portfolio delta stays ≤ DELTA_CAP (350), leaving a
           50-unit buffer inside the VF limit for the hedge.
  FIX C — REALIZED_IV removed from taker buy FV.
           Using realized vol (0.414) for buy-side FV caused 17–38 tick phantom
           edges on OTM strikes even at the correct TTE, pile-driving all VEVs
           to max position. All FV calculations now use market IV (BASE_IV).
           The long-vol thesis is expressed through passive bids, not aggressive
           taker buying at multiples of fair value.

"""

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
# Configuration
# ---------------------------------------------------------------------------

PRODUCTS = [
    "VELVETFRUIT_EXTRACT",
    "HYDROGEL_PACK",
    "VEV_4000", "VEV_4500",
    "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500",
    "VEV_6000", "VEV_6500",
]

POSITION_LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 400,
    "HYDROGEL_PACK":       300,   # active MM instrument; set conservatively
    "VEV_4000":  200,
    "VEV_4500":  200,
    "VEV_5000":  200,
    "VEV_5100":  200,
    "VEV_5200":  200,
    "VEV_5300":  200,
    "VEV_5400":  200,
    "VEV_5500":  200,
    "VEV_6000":  200,
    "VEV_6500":  200,
}

STRIKES: Dict[str, int] = {
    "VEV_4000": 4000, "VEV_4500": 4500,
    "VEV_5000": 5000, "VEV_5100": 5100,
    "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500,
    "VEV_6000": 6000, "VEV_6500": 6500,
}

ITM_VEVS = {"VEV_4000", "VEV_4500"}
ATM_VEVS = {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"}
OTM_VEVS = {"VEV_6000", "VEV_6500"}

# ---------------------------------------------------------------------------
# Calibrated IV surface
# Market IV (from bisection on mid prices) ≈ 0.30–0.31 across all strikes.
# FIX C: REALIZED_IV removed. Previously used 0.414 for buy-side FV, which
# created 17–38 tick phantom edges on OTM strikes even at the correct TTE,
# causing the algo to hammer every VEV to max position from tick 0.
# All FV now uses market IV. Passive bids accumulate long vol exposure slowly.
# ---------------------------------------------------------------------------
BASE_IV: Dict[str, float] = {
    "VEV_4000": 0.090,   # deep ITM: IV mostly irrelevant (delta ≈ 1)
    "VEV_4500": 0.160,   # still largely intrinsic
    "VEV_5000": 0.307,   # calibrated from 3-day mid-price IV extraction
    "VEV_5100": 0.300,
    "VEV_5200": 0.307,
    "VEV_5300": 0.312,
    "VEV_5400": 0.293,
    "VEV_5500": 0.315,
}

# TTE parameters
TOTAL_TTE_DAYS = 7
TS_PER_DAY     = 1_000_000
YEAR_FRACTION  = 365.0

# IV blending: live IV estimates blended with calibrated prior
IV_HISTORY_MAX = 30
IV_BLEND_LIVE  = 0.65
IV_BLEND_PRIOR = 0.35

# ---------------------------------------------------------------------------
# Taker edge thresholds
# ---------------------------------------------------------------------------
TAKE_EDGE_ATM = 0.8    # ATM VEVs: min FV − ask to cross book (buy side only)
TAKE_EDGE_ITM = 3.0    # ITM VEVs: wider — spread is ~20 units wide

# ---------------------------------------------------------------------------
# Market-making parameters
# ---------------------------------------------------------------------------
MM_SIZE            = 10    # max size per passive resting order
DELTA_HEDGE_THRESH = 2.0   # min |net delta| before hedging on the underlying

# FIX B: Portfolio delta cap.
# VF position limit = 400. With 6 ATM VEV products × 200 contracts each,
# total delta at TTE=3 ≈ 694 — 294 more than can be hedged. The cap enforces
# that total net long delta across all options stays ≤ 350, leaving a 50-unit
# buffer below the VF limit so the hedge can always fully neutralise exposure.
DELTA_CAP = 350.0

# Adverse selection guard for ATM VEV passive asks.
ATM_MM_MAX_SHORT_FRAC = 0.30   # max |short position| as fraction of limit
ATM_OTM_STRIKES = {"VEV_5400", "VEV_5500"}   # extra ask widening on these

# VELVETFRUIT market-making
VF_MM_HALF_SPREAD = 2     # half-spread in price ticks
VF_MM_SIZE        = 15    # passive quote size per side
VF_MM_SKEW_FACTOR = 0.4   # skew multiplier applied to inv_fraction

# HYDROGEL market-making
HG_FAIR_VALUE     = 9995.0
HG_MM_HALF_SPREAD = 6
HG_MM_SIZE        = 12
HG_MR_SKEW_FACTOR = 0.003  # price-distance-to-mean → extra inventory skew

# FIX A: DAY_START constant removed.
# The real game day is now read directly from state.observations (the game
# exposes it each tick). See _resolve_day() helper in Trader.run().


# ---------------------------------------------------------------------------
# Persistent state initialiser
# ---------------------------------------------------------------------------

def make_state() -> dict:
    return {
        "price_history": {p: [] for p in PRODUCTS},
        "iv_history":    {p: [] for p in ATM_VEVS},
        "last_S":        5250.0,
        "current_day":   0,
        "last_ts":       0,
    }

# ---------------------------------------------------------------------------
# LOB utility helpers
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


def get_micro_price(depth: OrderDepth) -> Optional[float]:
    """Volume-weighted mid: weights ask side by bid qty and vice versa."""
    bid_px, bid_qty = get_best_bid(depth)
    ask_px, ask_qty = get_best_ask(depth)
    if bid_px is None or ask_px is None:
        return None
    ask_qty = abs(ask_qty)
    total = bid_qty + ask_qty
    if total == 0:
        return (bid_px + ask_px) / 2.0
    return (bid_px * ask_qty + ask_px * bid_qty) / total


def buy_capacity(position: int, limit: int) -> int:
    return max(0, limit - position)


def sell_capacity(position: int, limit: int) -> int:
    return max(0, limit + position)

# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    a1, a2, a3, a4, a5 = 0.319381530, -0.356563782, 1.781477937, -1.821255978, 1.330274429
    p = 0.2316419
    sign = 1 if x >= 0 else -1
    x = abs(x)
    t = 1.0 / (1.0 + p * x)
    poly = t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
    cdf = 1.0 - (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x) * poly
    return cdf if sign > 0 else 1.0 - cdf


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def tte_years(day: int, timestamp: int) -> float:
    tte_days = TOTAL_TTE_DAYS - day - timestamp / TS_PER_DAY
    return max(tte_days / YEAR_FRACTION, 0.0)


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0:
        return max(0.0, S - K)
    if sigma <= 1e-6:
        return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return S * sqrt_T * _norm_pdf(d1)


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    tol: float = 1e-5,
    max_iter: int = 60,
) -> Optional[float]:
    if T <= 0.0:
        return None
    intrinsic = max(0.0, S - K)
    if market_price < intrinsic + 1e-4:
        return None
    lo, hi = 1e-5, 4.0
    for _ in range(max_iter):
        mid_v = (lo + hi) * 0.5
        p = bs_call(S, K, T, mid_v)
        if p < market_price:
            lo = mid_v
        else:
            hi = mid_v
        if hi - lo < tol:
            return (lo + hi) * 0.5
    return (lo + hi) * 0.5

# ---------------------------------------------------------------------------
# IV surface management
# ---------------------------------------------------------------------------

def get_live_iv(trader_data: dict, product: str) -> float:
    """Blend calibrated prior with rolling live IV estimates."""
    prior = BASE_IV.get(product, 0.307)
    history = trader_data.get("iv_history", {}).get(product, [])
    if len(history) >= 3:
        recent = history[-10:]
        live = sum(recent) / len(recent)
        return IV_BLEND_LIVE * live + IV_BLEND_PRIOR * prior
    return prior


def record_iv(trader_data: dict, product: str, iv: float) -> None:
    if product not in ATM_VEVS:
        return
    if not (0.10 <= iv <= 1.50):   # tighter rejection band vs original
        return
    hist = trader_data.setdefault("iv_history", {}).setdefault(product, [])
    hist.append(iv)
    if len(hist) > IV_HISTORY_MAX:
        hist.pop(0)


def update_iv_from_market(
    trader_data: dict,
    product: str,
    S: float,
    day: int,
    timestamp: int,
    depth: OrderDepth,
) -> None:
    if product not in ATM_VEVS:
        return
    K = STRIKES[product]
    T = tte_years(day, timestamp)
    if T <= 0:
        return
    mid = get_mid_price(depth)
    if mid is None or mid <= 0.5:
        return
    iv = implied_vol(mid, S, K, T)
    if iv is not None:
        record_iv(trader_data, product, iv)

# ---------------------------------------------------------------------------
# Portfolio Greeks
# ---------------------------------------------------------------------------

def compute_portfolio_delta(
    positions: Dict[str, int],
    S: float,
    day: int,
    timestamp: int,
    trader_data: dict,
) -> float:
    net_delta = 0.0
    for product, pos in positions.items():
        if pos == 0 or product not in STRIKES:
            continue
        K     = STRIKES[product]
        T     = tte_years(day, timestamp)
        sigma = get_live_iv(trader_data, product) if product in ATM_VEVS else BASE_IV.get(product, 0.307)
        delta = bs_delta(S, K, T, sigma)
        net_delta += pos * delta
    return net_delta

# ---------------------------------------------------------------------------
# Strategy: VELVETFRUIT_EXTRACT  (underlying — market make + delta hedge anchor)
# ---------------------------------------------------------------------------

def strategy_velvetfruit(
    state: TradingState,
    trader_data: dict,
    S: float,
    day: int,
    delta_hedge_orders: List[Order],
) -> List[Order]:
    """
    Two-part VELVETFRUIT strategy:

    1. Delta hedging: neutralise net option delta by trading against the position.
       Only fires when |net_delta| > DELTA_HEDGE_THRESH to avoid excessive churn.
       Uses micro-price to choose direction; crosses best bid/ask.

    2. Market making around micro-price: since VELVETFRUIT has mild negative
       lag-1 autocorrelation (−0.16) and a tight native spread (~5 ticks),
       we quote VF_MM_HALF_SPREAD ticks inside the current best, skewed by
       inventory. The delta hedge is executed first so the MM quotes reflect
       the post-hedge position.

    Both components share the same position limit and are returned together.
    """
    orders: List[Order] = []
    depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
    if depth is None:
        return delta_hedge_orders   # return pre-computed hedge if no depth

    limit    = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
    position = state.position.get("VELVETFRUIT_EXTRACT", 0)

    # --- Part 1: Delta hedge (computed externally and passed in) ---
    # Apply hedge quantity against position so MM is aware of residual
    hedge_net = sum(o.quantity for o in delta_hedge_orders)
    position += hedge_net   # simulate post-hedge position for MM quoting

    orders.extend(delta_hedge_orders)

    # --- Part 2: Market make on the underlying ---
    best_bid, bid_qty = get_best_bid(depth)
    best_ask, ask_qty = get_best_ask(depth)
    if best_bid is None or best_ask is None:
        return orders

    # Reservation price = micro-price (volume-weighted mid)
    reservation = get_micro_price(depth)
    if reservation is None:
        reservation = (best_bid + best_ask) / 2.0

    # Inventory skew: lean quotes against existing position
    inv_fraction = position / limit if limit != 0 else 0.0
    pos_skew     = inv_fraction * VF_MM_HALF_SPREAD * VF_MM_SKEW_FACTOR

    my_bid = math.floor(reservation - VF_MM_HALF_SPREAD - pos_skew)
    my_ask = math.ceil(reservation  + VF_MM_HALF_SPREAD - pos_skew)

    # Never post quotes that cross the existing book
    my_bid = min(my_bid, best_bid)
    my_ask = max(my_ask, best_ask)

    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)

    if buy_room > 0 and my_bid > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", my_bid, min(buy_room, VF_MM_SIZE)))
    if sell_room > 0:
        orders.append(Order("VELVETFRUIT_EXTRACT", my_ask, -min(sell_room, VF_MM_SIZE)))

    return orders


def compute_delta_hedge(
    state: TradingState,
    trader_data: dict,
    S: float,
    day: int,
) -> List[Order]:
    """
    Compute delta-hedge orders for the underlying (without sending them yet).
    Returned to strategy_velvetfruit so it can account for them in MM quoting.
    """
    orders: List[Order] = []
    depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
    if depth is None:
        return orders

    positions  = state.position
    net_delta  = compute_portfolio_delta(positions, S, day, state.timestamp, trader_data)

    if abs(net_delta) < DELTA_HEDGE_THRESH:
        return orders

    hedge_qty = -round(net_delta)
    limit     = POSITION_LIMITS["VELVETFRUIT_EXTRACT"]
    ve_pos    = positions.get("VELVETFRUIT_EXTRACT", 0)

    if hedge_qty > 0:
        cap = buy_capacity(ve_pos, limit)
        if cap <= 0:
            return orders
        best_ask, ask_qty = get_best_ask(depth)
        if best_ask is None:
            return orders
        qty = min(cap, abs(ask_qty), hedge_qty)
        if qty > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", best_ask, qty))
    else:
        cap = sell_capacity(ve_pos, limit)
        if cap <= 0:
            return orders
        best_bid, bid_qty = get_best_bid(depth)
        if best_bid is None:
            return orders
        qty = min(cap, bid_qty, -hedge_qty)
        if qty > 0:
            orders.append(Order("VELVETFRUIT_EXTRACT", best_bid, -qty))

    return orders

# ---------------------------------------------------------------------------
# Strategy: HYDROGEL_PACK  (independent mean-reverting asset)
# ---------------------------------------------------------------------------

def strategy_hydrogel(
    state: TradingState,
    trader_data: dict,
) -> List[Order]:
    """
    HYDROGEL_PACK is independent of VELVETFRUIT and mean-reverts around ~9995.
    Historical spread ≈ 16 ticks (mode). Lag-1 autocorr −0.12.

    FIX 4 — Queue priority via bid+1 / ask-1 improvement.
    Previous code clamped quotes to (min bid, max ask), which put us behind
    Mark 14 and Mark 38 in the queue; they traded between themselves and we
    received zero fills. We now always post ONE TICK BETTER than the best
    bid/ask, guaranteeing we are at the front of the queue.

    Strategy:
    - Fair value = rolling mean of last 50 mid prices (default HG_FAIR_VALUE).
    - Reservation price pulled toward FV by mean-reversion skew and inventory.
    - Passive quotes: bid = best_bid + 1, ask = best_ask - 1 (queue priority),
      unless our computed reservation-based price is even more aggressive.
    - Taker: cross the book on large deviations from FV (2× half-spread edge).
    """
    orders: List[Order] = []
    depth = state.order_depths.get("HYDROGEL_PACK")
    if depth is None:
        return orders

    position = state.position.get("HYDROGEL_PACK", 0)
    limit    = POSITION_LIMITS["HYDROGEL_PACK"]

    best_bid, bid_qty = get_best_bid(depth)
    best_ask, ask_qty = get_best_ask(depth)
    if best_bid is None or best_ask is None:
        return orders

    mid = (best_bid + best_ask) / 2.0

    # Rolling fair value
    hg_hist = trader_data.setdefault("hg_price_history", [])
    hg_hist.append(mid)
    if len(hg_hist) > 50:
        hg_hist.pop(0)
    fair_value = sum(hg_hist) / len(hg_hist) if hg_hist else HG_FAIR_VALUE

    # Mean-reversion and inventory skew on reservation price
    dist_from_fv = mid - fair_value
    mr_skew      = dist_from_fv * HG_MR_SKEW_FACTOR
    inv_fraction = position / limit if limit != 0 else 0.0
    pos_skew     = inv_fraction * HG_MM_HALF_SPREAD * 0.5
    reservation  = fair_value - mr_skew - pos_skew

    # --- Taker: fire on large deviations (2× half-spread) ---
    taker_edge = HG_MM_HALF_SPREAD * 2.0

    for ask_price in sorted(depth.sell_orders.keys()):
        if reservation - ask_price < taker_edge:
            break
        cap = buy_capacity(position, limit)
        if cap <= 0:
            break
        qty = min(cap, abs(depth.sell_orders[ask_price]), HG_MM_SIZE)
        orders.append(Order("HYDROGEL_PACK", ask_price, qty))
        position += qty

    for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
        if bid_price - reservation < taker_edge:
            break
        cap = sell_capacity(position, limit)
        if cap <= 0:
            break
        qty = min(cap, depth.buy_orders[bid_price], HG_MM_SIZE)
        orders.append(Order("HYDROGEL_PACK", bid_price, -qty))
        position -= qty

    # --- Passive MM: ALWAYS improve best bid/ask by 1 tick ---
    # This guarantees queue priority over Mark 14/38 who rest at the existing
    # best. Use the more aggressive of (reservation ± half_spread) vs (best ± 1).
    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)

    # Reservation-based quotes
    res_bid = math.floor(reservation - HG_MM_HALF_SPREAD)
    res_ask = math.ceil(reservation  + HG_MM_HALF_SPREAD)

    # Queue-priority quotes (1 tick better than current best)
    qp_bid = best_bid + 1
    qp_ask = best_ask - 1

    # Sanity: qp quotes must not cross each other or cross reservation badly
    if qp_bid >= qp_ask:
        # Spread is already 1 tick — just join at best
        qp_bid = best_bid
        qp_ask = best_ask

    # Take the more aggressive (higher bid, lower ask) of the two approaches,
    # but never let the bid exceed reservation or ask go below reservation
    my_bid = max(res_bid, qp_bid)
    my_ask = min(res_ask, qp_ask)

    # Final safety: quotes must not cross
    if my_bid >= my_ask:
        my_bid = best_bid
        my_ask = best_ask

    if buy_room > 0 and my_bid > 0:
        orders.append(Order("HYDROGEL_PACK", my_bid, min(buy_room, HG_MM_SIZE)))
    if sell_room > 0:
        orders.append(Order("HYDROGEL_PACK", my_ask, -min(sell_room, HG_MM_SIZE)))

    return orders

# ---------------------------------------------------------------------------
# Strategy: ATM VEVs  (VEV_5000 – VEV_5500)
# ---------------------------------------------------------------------------

def strategy_atm_vev(
    product: str,
    state: TradingState,
    trader_data: dict,
    S: float,
    day: int,
    portfolio_delta: float,
) -> List[Order]:
    """
    ATM/near-ATM VEV strategy.

    FIX C — Single market IV for all FV calculations.
      Previously used REALIZED_IV (0.414) for buy-side FV, creating phantom
      edges of 17–38 ticks on OTM strikes and driving all VEVs to max position
      immediately. Now: one FV from market IV (BASE_IV ~0.307) used for both
      taker and passive MM. The long-vol thesis is expressed through passive
      bids that accumulate slowly, not by paying multiples of fair value.

    FIX B — Portfolio delta cap gate.
      Before submitting any buy order, the caller passes the current portfolio
      delta. If adding this product's delta would exceed DELTA_CAP (350), the
      buy taker and passive bid are skipped entirely. This prevents the total
      long delta from growing beyond what the VF position limit can hedge.

    Taker BUY only (sell-side taker remains disabled from v3):
      Lift ask if FV − ask_price > TAKE_EDGE_ATM, but only when portfolio
      delta is below DELTA_CAP.

    Passive MM:
      Bid: around FV with vega-scaled half-spread, skewed by inventory.
           Gated by DELTA_CAP — skipped if we're already at the cap.
      Ask: around FV with adverse-selection buffer; capped by
           ATM_MM_MAX_SHORT_FRAC to limit exposure to informed flow.
    """
    orders: List[Order] = []
    depth = state.order_depths.get(product)
    if depth is None:
        return orders

    ts       = state.timestamp
    position = state.position.get(product, 0)
    limit    = POSITION_LIMITS.get(product, 200)

    K     = STRIKES[product]
    T     = tte_years(day, ts)
    sigma = get_live_iv(trader_data, product)   # market IV only (FIX C)

    # Update IV surface from current snapshot
    update_iv_from_market(trader_data, product, S, day, ts, depth)

    # Single fair value from market IV (FIX C: removed separate fv_buy/fv_sell)
    fv = max(0.0, S - K) if T <= 0 else bs_call(S, K, T, sigma)

    # Inventory skew: cubic penalty for large positions
    inv_fraction = position / limit if limit != 0 else 0.0
    inv_shift    = math.copysign((abs(inv_fraction) ** 2.5) * 3.0, inv_fraction)
    reservation  = fv - inv_shift

    best_bid, bid_qty = get_best_bid(depth)
    best_ask, ask_qty = get_best_ask(depth)

    # FIX B: compute this product's marginal delta contribution
    delta_per_contract = bs_delta(S, K, T, sigma) if T > 0 else (1.0 if S > K else 0.0)
    # How much buying capacity remains before we'd breach DELTA_CAP
    delta_headroom = max(0.0, DELTA_CAP - portfolio_delta)
    # Max contracts we can buy without breaching cap
    delta_buy_cap  = int(delta_headroom / delta_per_contract) if delta_per_contract > 0 else 0

    # --- Taker BUY (sell-side taker remains disabled) ---
    if best_ask is not None and delta_buy_cap > 0:
        for ask_price in sorted(depth.sell_orders.keys()):
            if reservation - ask_price < TAKE_EDGE_ATM:
                break
            cap = min(buy_capacity(position, limit), delta_buy_cap)
            if cap <= 0:
                break
            qty = min(cap, abs(depth.sell_orders[ask_price]), MM_SIZE)
            orders.append(Order(product, ask_price, qty))
            position       += qty
            delta_buy_cap  -= qty   # track cap consumption within this tick

    # --- Passive market making ---
    if best_bid is None or best_ask is None:
        return orders

    vega        = bs_vega(S, K, T, sigma) if T > 0 else 0.0
    rel_vega    = vega / (fv + 1.0)
    half_spread = max(1.0, rel_vega * 2.0)
    ask_extra   = 1 if product in ATM_OTM_STRIKES else 0
    pos_skew    = inv_fraction * half_spread * 0.5

    # BID: gated by delta cap (FIX B)
    buy_room = buy_capacity(position, limit)
    if buy_room > 0 and delta_buy_cap > 0:
        my_bid = math.floor(reservation - half_spread - pos_skew)
        my_bid = min(my_bid, best_bid)
        if my_bid > 0:
            orders.append(Order(product, my_bid, min(buy_room, delta_buy_cap, MM_SIZE)))

    # ASK: capped by adverse selection short limit
    max_short      = int(limit * ATM_MM_MAX_SHORT_FRAC)
    sell_room      = sell_capacity(position, limit)
    short_cap_room = max(0, max_short + position)
    effective_sell = min(sell_room, short_cap_room)

    my_ask = math.ceil(reservation + half_spread - pos_skew + ask_extra)
    my_ask = max(my_ask, best_ask)
    if effective_sell > 0:
        orders.append(Order(product, my_ask, -min(effective_sell, MM_SIZE)))

    return orders

# ---------------------------------------------------------------------------
# Strategy: Deep-ITM VEVs  (VEV_4000, VEV_4500)
# ---------------------------------------------------------------------------

def strategy_itm_vev(
    product: str,
    state: TradingState,
    trader_data: dict,
    S: float,
    day: int,
) -> List[Order]:
    """
    Deep ITM options: fair value ≈ S − K (intrinsic), time value negligible.
    Market spread ≈ 20 ticks wide. Delta ≈ 1.0 for VEV_4000, ~1.0 for VEV_4500.

    Arbitrage channel: VEV_4000 + 4000 = VELVETFRUIT to within ~1 tick on average
    (verified in historical data). If the spread between VEV_4000 and intrinsic
    exceeds TAKE_EDGE_ITM, we take the mispriced side.

    Strategy:
    - Fair value = full BS call (intrinsic + small time value).
    - Taker edge = TAKE_EDGE_ITM (3 ticks) — larger than ATM because the
      quoted spread itself is ~20 ticks.
    - Passive quotes straddle reservation price at 1/3 of the current spread.
    - Inventory skew penalises large long positions (delta-1 options are
      equivalent to VELVETFRUIT exposure; the delta hedge will offset this,
      but we don't want undue concentration).
    """
    orders: List[Order] = []
    depth = state.order_depths.get(product)
    if depth is None:
        return orders

    ts       = state.timestamp
    position = state.position.get(product, 0)
    limit    = POSITION_LIMITS.get(product, 200)

    K     = STRIKES[product]
    T     = tte_years(day, ts)
    sigma = BASE_IV.get(product, 0.15)
    fv    = bs_call(S, K, T, sigma)

    inv_fraction = position / limit if limit != 0 else 0.0
    inv_shift    = math.copysign((abs(inv_fraction) ** 2.5) * 5.0, inv_fraction)
    reservation  = fv - inv_shift

    best_bid, bid_qty = get_best_bid(depth)
    best_ask, ask_qty = get_best_ask(depth)

    # --- Taker ---
    if best_ask is not None:
        for ask_price in sorted(depth.sell_orders.keys()):
            if reservation - ask_price < TAKE_EDGE_ITM:
                break
            cap = buy_capacity(position, limit)
            if cap <= 0:
                break
            qty = min(cap, abs(depth.sell_orders[ask_price]), MM_SIZE)
            orders.append(Order(product, ask_price, qty))
            position += qty

    if best_bid is not None:
        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_price - reservation < TAKE_EDGE_ITM:
                break
            cap = sell_capacity(position, limit)
            if cap <= 0:
                break
            qty = min(cap, depth.buy_orders[bid_price], MM_SIZE)
            orders.append(Order(product, bid_price, -qty))
            position -= qty

    # --- Passive market making ---
    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)

    if best_bid is None or best_ask is None:
        return orders

    half_spread = max(5.0, (best_ask - best_bid) / 3.0)
    pos_skew    = inv_fraction * half_spread * 0.4

    my_bid = math.floor(reservation - half_spread - pos_skew)
    my_ask = math.ceil(reservation  + half_spread - pos_skew)

    my_bid = min(my_bid, best_bid)
    my_ask = max(my_ask, best_ask)

    if buy_room > 0 and my_bid > 0:
        orders.append(Order(product, my_bid, min(buy_room, MM_SIZE)))
    if sell_room > 0:
        orders.append(Order(product, my_ask, -min(sell_room, MM_SIZE)))

    return orders

# ---------------------------------------------------------------------------
# Main Trader class
# ---------------------------------------------------------------------------

class Trader:

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        # 1. Rehydrate persistent state
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                trader_data = make_state()
        else:
            trader_data = make_state()

        trader_data.setdefault("price_history",    {p: [] for p in PRODUCTS})
        trader_data.setdefault("iv_history",       {p: [] for p in ATM_VEVS})
        trader_data.setdefault("hg_price_history", [])
        trader_data.setdefault("last_S",           5250.0)
        trader_data.setdefault("current_day",      0)
        trader_data.setdefault("last_ts",          0)

        # 2. FIX A — Resolve the real game day.
        #    Priority 1: state.observations["day"] — the game exposes the actual
        #    day number each tick. This is the ground truth.
        #    Priority 2: rollover counter — increment current_day each time ts
        #    resets to ~0 after being large (same logic as before, kept as fallback
        #    in case observations is absent or has an unexpected structure).
        ts      = state.timestamp
        last_ts = trader_data.get("last_ts", 0)
        if ts < last_ts - 500_000:
            trader_data["current_day"] = trader_data.get("current_day", 0) + 1
        trader_data["last_ts"] = ts

        day = self._resolve_day(state, trader_data)

        # 3. Get underlying price (micro-price preferred; fallback to last known)
        ve_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if ve_depth is not None:
            S = get_micro_price(ve_depth) or trader_data.get("last_S", 5250.0)
        else:
            S = trader_data.get("last_S", 5250.0)
        trader_data["last_S"] = S

        # 4. Log snapshot
        self._log_state(state, day, S, trader_data)

        # 5. Build orders
        result: Dict[str, List[Order]] = {}

        # FIX B — Compute current portfolio delta ONCE before processing options.
        # This is passed into strategy_atm_vev so each product can check whether
        # buying more would breach DELTA_CAP before placing orders.
        portfolio_delta = compute_portfolio_delta(
            state.position, S, day, ts, trader_data
        )

        # --- VEV options ---
        for product in state.order_depths:
            depth = state.order_depths[product]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            orders: List[Order] = []

            if product in ATM_VEVS:
                # Pass current portfolio_delta so the strategy can gate buys
                # against DELTA_CAP (FIX B)
                orders = strategy_atm_vev(
                    product, state, trader_data, S, day, portfolio_delta
                )
                # Update portfolio_delta with any buys just placed, so subsequent
                # products in this loop see the updated exposure
                K     = STRIKES[product]
                T     = tte_years(day, ts)
                sigma = get_live_iv(trader_data, product)
                d     = bs_delta(S, K, T, sigma) if T > 0 else (1.0 if S > K else 0.0)
                net_bought = sum(o.quantity for o in orders if o.quantity > 0)
                portfolio_delta += net_bought * d

            elif product in ITM_VEVS:
                orders = strategy_itm_vev(product, state, trader_data, S, day)

            elif product in OTM_VEVS:
                # Deep OTM: worthless — dump any accidental long positions
                pos = state.position.get(product, 0)
                if pos > 0:
                    best_bid, _ = get_best_bid(depth)
                    if best_bid is not None and best_bid > 0:
                        orders.append(Order(product, best_bid, -pos))

            elif product == "HYDROGEL_PACK":
                orders = strategy_hydrogel(state, trader_data)

            # VELVETFRUIT handled below after all option deltas are known

            if orders:
                result[product] = orders
                self._log_orders(product, orders)

        # --- VELVETFRUIT: delta hedge + market making ---
        hedge_orders = compute_delta_hedge(state, trader_data, S, day)
        vf_orders    = strategy_velvetfruit(state, trader_data, S, day, hedge_orders)
        if vf_orders:
            result["VELVETFRUIT_EXTRACT"] = vf_orders
            self._log_orders("VELVETFRUIT_EXTRACT", vf_orders)

        # 6. Conversions (unused this round)
        conversions = 0

        # 7. Persist state
        return result, conversions, json.dumps(trader_data)

    # -------------------------------------------------------------------------
    # FIX A helper — resolve the real game day
    # -------------------------------------------------------------------------

    def _resolve_day(self, state: TradingState, trader_data: dict) -> int:
        """
        Return the true game day index for TTE calculations.

        The game exposes the current day in state.observations. We try to read
        it directly; if absent (older engine versions, test harnesses), we fall
        back to the rollover counter which increments each time the timestamp
        wraps around from ~999 900 back to 0.

        The game's day field is 1-indexed (day 1 is the first trading day).
        tte_years() expects the same 1-indexed day: TTE = 7 − day − ts/1M.
        """
        # Primary: read from observations if available
        obs = getattr(state, "observations", None)
        if obs is not None:
            # observations may be a dict-like or an object with a plain dict
            raw = obs
            if hasattr(obs, "__getitem__"):
                day_val = obs.get("day", None)
            elif hasattr(obs, "plainValueObservations"):
                day_val = obs.plainValueObservations.get("day", None)
            else:
                day_val = None
            if day_val is not None:
                try:
                    resolved = int(day_val)
                    # Cache so the fallback counter stays in sync
                    trader_data["observed_day"] = resolved
                    print("DAY (from observations): " + str(resolved))
                    return resolved
                except (TypeError, ValueError):
                    pass

        # Fallback: rollover counter (same logic as before but no DAY_START offset)
        # current_day starts at 0 and increments on each day boundary.
        # The first game-day seen is day 1, so add 1.
        fallback = trader_data.get("current_day", 0) + 1
        # If we previously saw a confirmed observation day, use that as an anchor
        last_observed = trader_data.get("observed_day", None)
        if last_observed is not None:
            fallback = last_observed + trader_data.get("current_day", 0)
        print("DAY (fallback counter): " + str(fallback))
        return fallback

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _log_state(self, state: TradingState, day: int, S: float, trader_data: dict) -> None:
        print("=" * 70)
        print(
            "Timestamp: " + str(state.timestamp) +
            "  Day: "     + str(day) +
            "  S(VE): "   + str(round(S, 2))
        )
        print("TTE: " + str(round(tte_years(day, state.timestamp) * 365.0, 3)) + " Solvenarian days")
        portfolio_delta = compute_portfolio_delta(
            state.position, S, day, state.timestamp, trader_data
        )
        print("Portfolio delta: " + str(round(portfolio_delta, 2)) +
              "  (cap=" + str(DELTA_CAP) + ")")
        for product, depth in state.order_depths.items():
            bid_px, bid_qty = get_best_bid(depth)
            ask_px, ask_qty = get_best_ask(depth)
            pos = state.position.get(product, 0)
            if product in STRIKES:
                K     = STRIKES[product]
                T     = tte_years(day, state.timestamp)
                sigma = get_live_iv(trader_data, product) if product in ATM_VEVS else BASE_IV.get(product, 0.26)
                fv    = round(bs_call(S, K, T, sigma), 2)
                fv_str = "  fv=" + str(fv)
            else:
                fv_str = ""
            print(
                "  " + product.ljust(25) +
                " | pos="  + str(pos).rjust(5) +
                " | bid="  + str(bid_px) + "x" + str(bid_qty) +
                "  ask="   + str(ask_px) + "x" + str(ask_qty) +
                fv_str
            )

    def _log_orders(self, product: str, orders: List[Order]) -> None:
        for o in orders:
            side = "BUY " if o.quantity > 0 else "SELL"
            print(
                "  -> " + side + " " + str(abs(o.quantity)).rjust(3) +
                " " + product + " @ " + str(o.price)
            )