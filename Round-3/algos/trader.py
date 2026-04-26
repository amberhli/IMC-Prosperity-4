"""
trader.py — IMC Prosperity Algorithm
=====================================
Round strategy:
  VELVETFRUIT_EXTRACT    → underlying; tracked for BS pricing + used for delta hedging!!!
  VEV_4000 / VEV_4500    → deep-ITM calls; fair value ≈ intrinsic (S - K);
                            strategy_itm_vev: taker on intrinsic mispricing (could we market make here too?)
  VEV_5000 – VEV_5500    → near-ATM liquid calls; BS-priced with calibrated IV surface;
                            strategy_atm_vev: taker + market maker around BS fair value
  VEV_6000 / VEV_6500    → deep-OTM, floor-priced at 0 — no edge, skip

Key calibrated parameters (from historical data):
  Underlying mean ~5250, range 5198–5300, std ~15.6
  ATM implied vol surface:
    VEV_5000: σ = 0.260    VEV_5100: σ = 0.260
    VEV_5200: σ = 0.262    VEV_5300: σ = 0.265
    VEV_5400: σ = 0.250    VEV_5500: σ = 0.272
  TTE = 7 Solvenarian days at ts=0 day=0 (day index 0 = problem day 1)
  1 Solvenarian day ≡ 1/365 year for BS

Allowed imports: pandas, numpy, statistics, math, typing, jsonpickle
"""

from typing import Dict, List, Tuple, Optional
import json
import statistics
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

# All tradeable symbols
PRODUCTS = [
    "VELVETFRUIT_EXTRACT",
    "VEV_4000", "VEV_4500",
    "VEV_5000", "VEV_5100", "VEV_5200",
    "VEV_5300", "VEV_5400", "VEV_5500",
    "VEV_6000", "VEV_6500",
]

POSITION_LIMITS: Dict[str, int] = {
    "VELVETFRUIT_EXTRACT": 400,
    "VEV_4000": 200,
    "VEV_4500": 200,
    "VEV_5000": 200,
    "VEV_5100": 200,
    "VEV_5200": 200,
    "VEV_5300": 200,
    "VEV_5400": 200,
    "VEV_5500": 200,
    "VEV_6000": 200,
    "VEV_6500": 200,
}

# Strike prices for each VEV
STRIKES: Dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}

# Deep-ITM: fair value ≈ intrinsic, small time value edge
ITM_VEVS = {"VEV_4000", "VEV_4500"}

# ATM/near-ATM: liquid, BS-priced, actively traded
ATM_VEVS = {"VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"}

# Deep-OTM: floor-priced, no edge
OTM_VEVS = {"VEV_6000", "VEV_6500"}

# Calibrated IV surface (from 3-day historical fit)
# These are stable — live IV updating blends on prior
BASE_IV: Dict[str, float] = {
    "VEV_4000": 0.090,  # effectively intrinsic; IV only matters for tiny time value
    "VEV_4500": 0.160,
    "VEV_5000": 0.260,
    "VEV_5100": 0.260,
    "VEV_5200": 0.262,
    "VEV_5300": 0.265,
    "VEV_5400": 0.250,
    "VEV_5500": 0.272,
}

# TTE parameters
# Timestamp 0 on day 0 (day index) = 7 Solvenarian days to expiry
# Each day runs 0..999,900 in steps of 100 = 1,000,000 timestamp units per day
TOTAL_TTE_DAYS = 7
TS_PER_DAY = 1_000_000
YEAR_FRACTION = 365.0

# How many recent IV samples to blend (exponential-ish blend)
IV_HISTORY_MAX = 30
IV_BLEND_LIVE = 0.65     # weight on live IV estimate vs calibrated prior
IV_BLEND_PRIOR = 0.35

# Taker edge thresholds: minimum (FV - ask) or (bid - FV) to cross the book
# Calibrated: spread is 1-6 units, IV noise ~0.004 → ~0.9 price noise at VEV_5300
TAKE_EDGE_MIN = 0.5   # absolute price units

# Market-making parameters
MM_HALF_SPREAD_BASE = 1.0  # minimum half-spread in price units when quoting
MM_SIZE = 5                # max passive quote size
DELTA_HEDGE_THRESHOLD = 5.0   # min net delta imbalance before hedging on underlying
MAX_HISTORY = 100


# ---------------------------------------------------------------------------
# Persistent state initialiser
# ---------------------------------------------------------------------------

def make_state() -> dict:
    return {
        "price_history": {p: [] for p in PRODUCTS},
        # Rolling IV estimates from observed trade prices
        "iv_history": {p: [] for p in ATM_VEVS},
        # Running underlying price for Greeks
        "last_S": 5250.0,
        # Current day index (updated from timestamp progression)
        "current_day": 0,
    }


# ---------------------------------------------------------------------------
# LOB utility helpers  (same interface as the template)
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
    return limit - position


def sell_capacity(position: int, limit: int) -> int:
    return limit + position


# ---------------------------------------------------------------------------
# Black-Scholes
# ---------------------------------------------------------------------------

def _norm_cdf(x: float) -> float:
    """Standard normal CDF (Abramowitz & Stegun approx, error < 7.5e-8)."""
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
    """Time to expiry in years from current (day, timestamp)."""
    tte_days = TOTAL_TTE_DAYS - day - timestamp / TS_PER_DAY
    return max(tte_days / YEAR_FRACTION, 0.0)


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes European call price."""
    if T <= 0.0:
        return max(0.0, S - K)
    if sigma <= 1e-6:
        return max(0.0, S - K)
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """BS delta = dC/dS."""
    if T <= 0.0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)


def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    """BS vega = dC/dsigma (per unit of sigma)."""
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
    """
    Bisection implied volatility.
    Returns None if price is below intrinsic or doesn't converge.
    """
    if T <= 0.0:
        return None
    intrinsic = max(0.0, S - K)
    if market_price < intrinsic + 1e-4:
        return None
    lo, hi = 1e-5, 4.0
    for _ in range(max_iter):
        mid = (lo + hi) * 0.5
        p = bs_call(S, K, T, mid)
        if p < market_price:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            return (lo + hi) * 0.5
    return (lo + hi) * 0.5


# ---------------------------------------------------------------------------
# IV surface management
# ---------------------------------------------------------------------------

def get_live_iv(trader_data: dict, product: str) -> float:
    """
    Best IV estimate for product.
    Blends calibrated prior with rolling average of live estimates.
    """
    prior = BASE_IV.get(product, 0.26)
    history = trader_data.get("iv_history", {}).get(product, [])
    if len(history) >= 3:
        live = sum(history[-10:]) / len(history[-10:])
        return IV_BLEND_LIVE * live + IV_BLEND_PRIOR * prior
    return prior


def record_iv(trader_data: dict, product: str, iv: float) -> None:
    """Push a new IV sample into the rolling history."""
    if product not in ATM_VEVS:
        return
    if not (0.05 <= iv <= 2.0):
        return  # reject outliers
    hist = trader_data.setdefault("iv_history", {}).setdefault(product, [])
    hist.append(iv)
    if len(hist) > IV_HISTORY_MAX:
        hist.pop(0)


def update_iv_from_market_orders(
    trader_data: dict,
    product: str,
    S: float,
    day: int,
    timestamp: int,
    depth: OrderDepth,
) -> None:
    """
    Infer IV from the current best bid AND best ask mid.
    Only called for ATM VEVs.
    """
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
    """Net delta across all VEV positions (in units of the underlying)."""
    net_delta = 0.0
    for product, pos in positions.items():
        if pos == 0 or product not in STRIKES:
            continue
        K = STRIKES[product]
        T = tte_years(day, timestamp)
        sigma = get_live_iv(trader_data, product) if product in ATM_VEVS else BASE_IV.get(product, 0.26)
        delta = bs_delta(S, K, T, sigma)
        net_delta += pos * delta
    return net_delta


# ---------------------------------------------------------------------------
# Strategy: ATM VEVs  (VEV_5000 – VEV_5500)
# ---------------------------------------------------------------------------

def strategy_atm_vev(
    product: str,
    state: TradingState,
    trader_data: dict,
    S: float,
    day: int,
) -> List[Order]:
    """
    Black-Scholes taker + passive market maker for near-ATM VEVs.

    Taker logic:
      - Compute BS fair value with live IV estimate.
      - If FV > best_ask + TAKE_EDGE_MIN → buy (market is selling cheap).
      - If FV < best_bid - TAKE_EDGE_MIN → sell (market is buying dear).

    Market-maker logic:
      - Post passive limit orders around FV with spread scaled by vega.
      - Skew quotes to lean against existing inventory.

    Inventory management:
      - Non-linear position skew on the reservation price to stay within limits.
      - Reduce size near POSITION_LIMITS.
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
    sigma = get_live_iv(trader_data, product)

    # Update IV from the current snapshot
    update_iv_from_market_orders(trader_data, product, S, day, ts, depth)

    if T <= 0:
        # Expiry: only exercise if deeply ITM
        fv = max(0.0, S - K)
    else:
        fv = bs_call(S, K, T, sigma)

    # --- Inventory skew: shift reservation price to discourage building more ---
    inv_fraction = position / limit if limit != 0 else 0.0
    # Cubic skew: small effect at low inventory, steep near limits
    inv_shift    = math.copysign((abs(inv_fraction) ** 2.5) * 3.0, inv_fraction)
    reservation  = fv - inv_shift

    best_bid, bid_qty = get_best_bid(depth)
    best_ask, ask_qty = get_best_ask(depth)

    # --- Taker: cross the book when clearly mispriced ---
    if best_ask is not None:
        for ask_price in sorted(depth.sell_orders.keys()):
            edge = reservation - ask_price
            if edge < TAKE_EDGE_MIN:
                break
            cap = buy_capacity(position, limit)
            if cap <= 0:
                break
            qty = min(cap, abs(depth.sell_orders[ask_price]), MM_SIZE)
            orders.append(Order(product, ask_price, qty))
            position += qty

    if best_bid is not None:
        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            edge = bid_price - reservation
            if edge < TAKE_EDGE_MIN:
                break
            cap = sell_capacity(position, limit)
            if cap <= 0:
                break
            qty = min(cap, depth.buy_orders[bid_price], MM_SIZE)
            orders.append(Order(product, bid_price, -qty))
            position -= qty

    # --- Passive market making around reservation price ---
    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)

    if best_bid is None or best_ask is None:
        return orders

    # Scale half-spread by vega: wider quotes on high-vega options
    # (more uncertainty → need more edge to bear inventory risk)
    vega       = bs_vega(S, K, T, sigma) if T > 0 else 0.0
    # Vega per unit vol; normalize by FV to get relative uncertainty
    rel_vega   = vega / (fv + 1.0)
    half_spread = max(MM_HALF_SPREAD_BASE, rel_vega * 2.0)

    # Skew: lean against inventory — if long, post cheaper ask, higher bid discouraged
    pos_skew    = inv_fraction * half_spread * 0.5

    my_bid = math.floor(reservation - half_spread - pos_skew)
    my_ask = math.ceil(reservation + half_spread - pos_skew)

    # Sanity: quotes must not cross the market
    my_bid = min(my_bid, best_bid + 1)       # don't lift existing bids
    my_ask = max(my_ask, best_ask - 1)       # don't hit existing asks

    if buy_room > 0 and my_bid > 0:
        orders.append(Order(product, my_bid, buy_room))
    if sell_room > 0:
        orders.append(Order(product, my_ask, -sell_room))

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
    Deep ITM options: fair value ≈ S - K (intrinsic), time value ≈ 0.
    The market quotes bid ~= intrinsic - 10, ask ~= intrinsic + 10.
    Mid price tracks the underlying almost perfectly.

    Strategy:
    - Fair value = intrinsic + small BS time value.
    - Taker: cross ask if ask < FV - edge, cross bid if bid > FV + edge.
    - Post passive quotes around FV.

    Since the bid-ask straddles intrinsic symmetrically (~±10 units),
    the only time to take is when the underlying moves enough that intrinsic
    clearly outside the quoted prices.
    """
    orders: List[Order] = []
    depth = state.order_depths.get(product)
    if depth is None:
        return orders

    ts       = state.timestamp
    position = state.position.get(product, 0)
    limit    = POSITION_LIMITS.get(product, 200)

    K      = STRIKES[product]
    T      = tte_years(day, ts)
    sigma  = BASE_IV.get(product, 0.15)
    fv     = bs_call(S, K, T, sigma)

    inv_fraction = position / limit if limit != 0 else 0.0
    inv_shift    = math.copysign((abs(inv_fraction) ** 2.5) * 5.0, inv_fraction)
    reservation  = fv - inv_shift

    best_bid, bid_qty = get_best_bid(depth)
    best_ask, ask_qty = get_best_ask(depth)

    # For deep ITM, use a larger edge threshold because the spread is wider (~20 units)
    itm_edge = 2.0

    if best_ask is not None:
        for ask_price in sorted(depth.sell_orders.keys()):
            if reservation - ask_price < itm_edge:
                break
            cap = buy_capacity(position, limit)
            if cap <= 0:
                break
            qty = min(cap, abs(depth.sell_orders[ask_price]), MM_SIZE)
            orders.append(Order(product, ask_price, qty))
            position += qty

    if best_bid is not None:
        for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
            if bid_price - reservation < itm_edge:
                break
            cap = sell_capacity(position, limit)
            if cap <= 0:
                break
            qty = min(cap, depth.buy_orders[bid_price], MM_SIZE)
            orders.append(Order(product, bid_price, -qty))
            position -= qty

    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)

    if best_bid is None or best_ask is None:
        return orders

    # Passive quotes: straddle the reservation price inside the wide spread
    half_spread = max(5.0, (best_ask - best_bid) / 3.0)
    pos_skew    = inv_fraction * half_spread * 0.4
    my_bid      = math.floor(reservation - half_spread - pos_skew)
    my_ask      = math.ceil(reservation + half_spread - pos_skew)
    my_bid      = min(my_bid, best_bid + 1)
    my_ask      = max(my_ask, best_ask - 1)

    if buy_room > 0 and my_bid > 0:
        orders.append(Order(product, my_bid, buy_room))
    if sell_room > 0:
        orders.append(Order(product, my_ask, -sell_room))

    return orders


# ---------------------------------------------------------------------------
# Strategy: Delta hedge via VELVETFRUIT_EXTRACT
# ---------------------------------------------------------------------------

def strategy_delta_hedge(
    state: TradingState,
    trader_data: dict,
    S: float,
    day: int,
) -> List[Order]:
    """
    Hedge net delta of the VEV portfolio by trading the underlying.

    Net delta of a long call position is positive (we profit when S rises),
    so we hedge by selling VELVETFRUIT_EXTRACT.  We only hedge if the
    imbalance exceeds DELTA_HEDGE_THRESHOLD to avoid churning.

    Uses micro-price for direction to avoid adverse selection on the hedge.
    """
    orders: List[Order] = []
    depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
    if depth is None:
        return orders

    positions = state.position
    net_delta = compute_portfolio_delta(positions, S, day, state.timestamp, trader_data)

    if abs(net_delta) < DELTA_HEDGE_THRESHOLD:
        return orders

    # Hedge quantity = -(net_delta) in shares of underlying
    hedge_qty = -round(net_delta)
    limit = POSITION_LIMITS.get("VELVETFRUIT_EXTRACT", 400)
    ve_pos = positions.get("VELVETFRUIT_EXTRACT", 0)

    if hedge_qty > 0:
        # Buy underlying (we are net short delta)
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
        # Sell underlying (we are net long delta — most common case)
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

        # Ensure required sub-dicts always exist
        trader_data.setdefault("price_history", {p: [] for p in PRODUCTS})
        trader_data.setdefault("iv_history", {p: [] for p in ATM_VEVS})
        trader_data.setdefault("last_S", 5250.0)
        trader_data.setdefault("current_day", 0)

        # 3. Get underlying price
        ve_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if ve_depth is not None:
            S = get_micro_price(ve_depth) or trader_data.get("last_S", 5250.0)
        else:
            S = trader_data.get("last_S", 5250.0)
        trader_data["last_S"] = S
        

        # 2. Determine current day index from timestamp progression
        #    Timestamps reset to 0 each day; detect rollover by seeing a small ts
        #    after having seen a large one.
        ts = state.timestamp
        last_ts = trader_data.get("last_ts", 0)
        if ts < last_ts - 500_000:
            trader_data["current_day"] = trader_data.get("current_day", 0) + 1
        trader_data["last_ts"] = ts
        # day = trader_data["current_day"]

        
        trader_data.setdefault("day_offset", None)
        if trader_data["day_offset"] is None:
            # Back-solve absolute day from ATM option prices at session open.
            # Try each candidate day offset; pick the one whose implied vol
            # is closest to the calibrated prior.
            S_init = trader_data.get("last_S", 5250.0)
            best_offset, best_err = 0, float("inf")
            for product in ["VEV_5200", "VEV_5100", "VEV_5300"]:
                depth = state.order_depths.get(product)
                if depth is None:
                    continue
                mid = get_mid_price(depth)
                if mid is None:
                    continue
                K = STRIKES[product]
                sigma_prior = BASE_IV[product]
                for d in range(TOTAL_TTE_DAYS):
                    T = tte_years(d, state.timestamp)
                    if T <= 0:
                        continue
                    iv = implied_vol(mid, S_init, K, T)
                    if iv is None:
                        continue
                    err = abs(iv - sigma_prior)
                    if err < best_err:
                        best_err, best_offset = err, d
                break  # one liquid product is enough
            trader_data["day_offset"] = best_offset

        day = trader_data["current_day"] + trader_data["day_offset"]

        # 4. Log snapshot
        self._log_state(state, day, S)

        # 5. Build orders
        result: Dict[str, List[Order]] = {}

        # Process VEV options first (to know our option positions for delta hedge)
        for product in state.order_depths:
            depth = state.order_depths[product]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            orders: List[Order] = []

            if product in ATM_VEVS:
                orders = strategy_atm_vev(product, state, trader_data, S, day)

            elif product in ITM_VEVS:
                orders = strategy_itm_vev(product, state, trader_data, S, day)

            elif product in OTM_VEVS:
                # Deep OTM: worthless, floor-priced — never buy, sell if somehow long
                pos = state.position.get(product, 0)
                if pos > 0:
                    best_bid, bid_qty = get_best_bid(depth)
                    if best_bid is not None and best_bid > 0:
                        orders.append(Order(product, best_bid, -pos))

            # VELVETFRUIT_EXTRACT handled separately below (delta hedge)

            if orders:
                result[product] = orders
                self._log_orders(product, orders)

        # 6. Delta hedge on the underlying
        hedge_orders = strategy_delta_hedge(state, trader_data, S, day)
        if hedge_orders:
            result["VELVETFRUIT_EXTRACT"] = hedge_orders
            self._log_orders("VELVETFRUIT_EXTRACT", hedge_orders)

        # 7. Conversions (unused in this round)
        conversions = 0

        # 8. Persist state
        return result, conversions, json.dumps(trader_data)

    # -------------------------------------------------------------------------
    # Logging helpers
    # -------------------------------------------------------------------------

    def _log_state(self, state: TradingState, day: int, S: float) -> None:
        print("=" * 70)
        print("Timestamp: " + str(state.timestamp) + "  Day: " + str(day) + "  S(VE): " + str(round(S, 2)))
        print("TTE: " + str(round(tte_years(day, state.timestamp) * 365.0, 3)) + " Solvenarian days")
        for product, depth in state.order_depths.items():
            bid_px, bid_qty = get_best_bid(depth)
            ask_px, ask_qty = get_best_ask(depth)
            pos = state.position.get(product, 0)
            if product in STRIKES:
                K     = STRIKES[product]
                T     = tte_years(day, state.timestamp)
                sigma = BASE_IV.get(product, 0.26)
                fv    = round(bs_call(S, K, T, sigma), 2)
                fv_str = "  fv=" + str(fv)
            else:
                fv_str = ""
            print(
                "  " + product.ljust(25) +
                " | pos=" + str(pos).rjust(5) +
                " | bid=" + str(bid_px) + "x" + str(bid_qty) +
                "  ask=" + str(ask_px) + "x" + str(ask_qty) +
                fv_str
            )

    def _log_orders(self, product: str, orders: List[Order]) -> None:
        for o in orders:
            side = "BUY " if o.quantity > 0 else "SELL"
            print("  -> " + side + " " + str(abs(o.quantity)).rjust(3) +
                  " " + product + " @ " + str(o.price))