"""
trader.py — IMC Prosperity Algorithm
=====================================
Round strategy:
  ASH_COATED_OSMIUM      → strategy_take_quotes  (fair value = 10,000)
  INTARIAN_PEPPER_ROOT   → strategy_linear_trend  (strong upward drift, stay max long)

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
# Round configuration
# ---------------------------------------------------------------------------
PRODUCTS = [
    "ASH_COATED_OSMIUM",
    "INTARIAN_PEPPER_ROOT",
]

POSITION_LIMITS: Dict[str, int] = {
    "ASH_COATED_OSMIUM": 80,
    "INTARIAN_PEPPER_ROOT": 80,
}

MAX_HISTORY = 200   # longer window to fit the trend well


# ---------------------------------------------------------------------------
# Persistent state
# ---------------------------------------------------------------------------

def make_state() -> dict:
    return {
        "price_history": {p: [] for p in PRODUCTS},
        "timestamp_history": {p: [] for p in PRODUCTS},
        # Linear regression state for trend strategy
        "trend": {
            "n": 0,
            "sum_x": 0.0,
            "sum_y": 0.0,
            "sum_xx": 0.0,
            "sum_xy": 0.0,
            "slope": None,
            "intercept": None,
        },
    }


def state_update_price(trader_data: dict, product: str, mid: float, ts: int) -> None:
    hist = trader_data["price_history"].setdefault(product, [])
    tsh  = trader_data["timestamp_history"].setdefault(product, [])
    hist.append(mid)
    tsh.append(ts)
    if len(hist) > MAX_HISTORY:
        hist.pop(0)
        tsh.pop(0)


def state_rolling_mean(trader_data: dict, product: str) -> Optional[float]:
    hist = trader_data["price_history"].get(product, [])
    return statistics.mean(hist) if len(hist) >= 2 else None


def state_rolling_std(trader_data: dict, product: str) -> Optional[float]:
    hist = trader_data["price_history"].get(product, [])
    return statistics.stdev(hist) if len(hist) >= 3 else None


# ---------------------------------------------------------------------------
# Helper utilities
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


def get_obi(depth: OrderDepth) -> float:
    bid_px, bid_qty = get_best_bid(depth)
    ask_px, ask_qty = get_best_ask(depth)
    if bid_px is None or ask_px is None:
        return 0.0
    ask_qty = abs(ask_qty)
    total = bid_qty + ask_qty
    if total == 0:
        return 0.0
    return (bid_qty - ask_qty) / total


def buy_capacity(position: int, limit: int) -> int:
    return limit - position


def sell_capacity(position: int, limit: int) -> int:
    return limit + position


# ---------------------------------------------------------------------------
# Online linear regression (Welford-style, O(1) per update)
# ---------------------------------------------------------------------------

def update_linreg(trend: dict, x: float, y: float) -> None:
    """Update running OLS sums with new (x, y) observation."""
    trend["n"]      += 1
    trend["sum_x"]  += x
    trend["sum_y"]  += y
    trend["sum_xx"] += x * x
    trend["sum_xy"] += x * y

    n      = trend["n"]
    sum_x  = trend["sum_x"]
    sum_y  = trend["sum_y"]
    sum_xx = trend["sum_xx"]
    sum_xy = trend["sum_xy"]

    denom = n * sum_xx - sum_x * sum_x
    if abs(denom) < 1e-9:
        return
    slope     = (n * sum_xy - sum_x * sum_y) / denom
    intercept = (sum_y - slope * sum_x) / n
    trend["slope"]     = slope
    trend["intercept"] = intercept


def predict_linreg(trend: dict, x: float) -> Optional[float]:
    if trend["slope"] is None:
        return None
    return trend["slope"] * x + trend["intercept"]


# ---------------------------------------------------------------------------
# Strategy: TAKER — known fair value (ASH_COATED_OSMIUM)
# ---------------------------------------------------------------------------

def strategy_take_quotes(
    product: str,
    state: TradingState,
    trader_data: dict,
    fair_value: float,
) -> List[Order]:
    """
    Takes mispriced quotes relative to a known fair value.
    Uses non-linear inventory skew to stay safe near position limits.
    """
    orders: List[Order] = []
    depth = state.order_depths.get(product)
    if depth is None:
        return orders

    position = state.position.get(product, 0)
    limit    = POSITION_LIMITS.get(product, 20)

    inv_fraction  = position / limit if limit != 0 else 0
    inv_shift     = math.copysign((abs(inv_fraction) ** 3) * 2.5, inv_fraction)
    reservation   = fair_value - inv_shift

    for ask_price in sorted(depth.sell_orders.keys()):
        if ask_price >= reservation:
            break
        cap = buy_capacity(position, limit)
        if cap <= 0:
            break
        qty = min(cap, abs(depth.sell_orders[ask_price]))
        orders.append(Order(product, ask_price, qty))
        position += qty

    for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
        if bid_price <= reservation:
            break
        cap = sell_capacity(position, limit)
        if cap <= 0:
            break
        qty = min(cap, depth.buy_orders[bid_price])
        orders.append(Order(product, bid_price, -qty))
        position -= qty

    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)
    bid_px, _ = get_best_bid(depth)
    ask_px, _ = get_best_ask(depth)

    if bid_px is not None and ask_px is not None:
        my_bid = min(bid_px + 1, math.floor(reservation - 1))
        my_ask = max(ask_px - 1, math.ceil(reservation + 1))
        if buy_room > 0:
            orders.append(Order(product, my_bid, buy_room))
        if sell_room > 0:
            orders.append(Order(product, my_ask, -sell_room))

    return orders


# ---------------------------------------------------------------------------
# Strategy: LINEAR TREND — for INTARIAN_PEPPER_ROOT
# ---------------------------------------------------------------------------

def strategy_linear_trend(
    product: str,
    state: TradingState,
    trader_data: dict,
) -> List[Order]:
    """
    Designed for instruments with a near-perfect upward linear price trend
    and a volatile (but mean-reverting) bid-ask spread.

    Core idea
    ---------
    The price rises predictably over time.  The best strategy is to stay
    as close to the maximum long position as possible at all times, while
    using the spread oscillation to improve entry price slightly.

    Mechanics
    ---------
    1. Fit an online OLS regression of mid-price vs timestamp.
    2. Project the fair value ONE spread-width ahead to get a "trend fair value".
    3. Use spread timing: prefer limit orders when the current spread is wide
       (market makers are far apart → we get filled at a better price once
       spread snaps back).  Fall back to market orders when spread is narrow.
    4. Always post a resting bid at best_bid+1 for any remaining capacity so
       we soak up any sell-side liquidity the moment it appears.
    5. Never willingly sell — only sell if we somehow end up above the limit
       (shouldn't happen) or if the position somehow goes negative due to
       forced fills (defensive only).

    Parameters (tuned to observed data, not overfit)
    -------------------------------------------------
    SPREAD_WIDE_THRESHOLD : we consider the spread "wide" if it exceeds this
                            fraction of the current mid price.  At wide spread,
                            prefer limit orders.
    LOOKAHEAD_TICKS       : how many timestamp units ahead we project fair value.
                            Set to ~half the mean spread-snap period observed.
    SELL_DEFENCE_Z        : only sell defensively if price is this many σ ABOVE
                            the trend line (protects against regime change).
    """
    SPREAD_WIDE_THRESHOLD = 0.0012   # ~12 bps; spread of 15 on mid of 11500 ≈ 13 bps
    LOOKAHEAD_TICKS       = 5_000    # ~1/600 of the 3M-tick window
    SELL_DEFENCE_Z        = 4.0      # very conservative — almost never sells

    orders: List[Order] = []
    depth = state.order_depths.get(product)
    if depth is None:
        return orders

    position = state.position.get(product, 0)
    limit    = POSITION_LIMITS.get(product, 20)

    mid = get_mid_price(depth)
    if mid is None:
        return orders

    ts  = state.timestamp
    bid_px, bid_qty = get_best_bid(depth)
    ask_px, ask_qty = get_best_ask(depth)
    if bid_px is None or ask_px is None:
        return orders

    spread      = ask_px - bid_px
    spread_wide = (spread / mid) > SPREAD_WIDE_THRESHOLD

    # --- Update online regression ---
    trend = trader_data.setdefault("trend", {
        "n": 0, "sum_x": 0.0, "sum_y": 0.0,
        "sum_xx": 0.0, "sum_xy": 0.0,
        "slope": None, "intercept": None,
    })
    update_linreg(trend, float(ts), mid)
    state_update_price(trader_data, product, mid, ts)

    # --- Trend fair value projection ---
    trend_fair = predict_linreg(trend, float(ts) + LOOKAHEAD_TICKS)
    if trend_fair is None:
        # Not enough data yet: just buy everything we can at the ask
        cap = buy_capacity(position, limit)
        if cap > 0:
            orders.append(Order(product, ask_px, cap))
        return orders

    # --- Defensive sell: only if price is absurdly above trend ---
    std = state_rolling_std(trader_data, product)
    if std and std > 1e-6:
        z_above = (mid - trend_fair) / std
        if z_above > SELL_DEFENCE_Z:
            cap = sell_capacity(position, limit)
            if cap > 0:
                # Sell at best bid (take liquidity)
                orders.append(Order(product, bid_px, -min(cap, 10)))
            return orders   # don't buy when dramatically overbought vs trend

    # --- Primary logic: maximise long position ---
    cap = buy_capacity(position, limit)
    if cap <= 0:
        # Already max long — just hold, nothing to do
        return orders

    if spread_wide:
        # Spread is fat: post a limit order just inside the spread.
        # We'll get filled when a seller crosses us, at a better price
        # than the current ask.  Don't chase — the trend will come to us.
        limit_bid = bid_px + 1          # one tick above best bid
        limit_bid = min(limit_bid, math.floor(trend_fair))  # never overpay vs trend
        if limit_bid < ask_px:          # sanity: must still be inside spread
            orders.append(Order(product, limit_bid, cap))
        else:
            # Spread is actually narrow now despite the flag — just take the ask
            orders.append(Order(product, ask_px, cap))
    else:
        # Spread is tight: aggressively take the ask.
        # Slippage is minimal and we want to be long ASAP.
        orders.append(Order(product, ask_px, cap))

    return orders


# ---------------------------------------------------------------------------
# Strategy: OBI market making (fallback for unknown products)
# ---------------------------------------------------------------------------

def strategy_obi_market_make(
    product: str,
    state: TradingState,
    trader_data: dict,
) -> List[Order]:
    OBI_WEIGHT  = 1.5
    SPREAD      = 1
    TAKE_BUFFER = 0.5

    orders: List[Order] = []
    depth = state.order_depths.get(product)
    if depth is None:
        return orders

    position = state.position.get(product, 0)
    limit    = POSITION_LIMITS.get(product, 20)

    micro = get_micro_price(depth)
    if micro is None:
        return orders

    obi = get_obi(depth)
    inv_fraction = position / limit if limit != 0 else 0
    inv_shift    = math.copysign((abs(inv_fraction) ** 2) * 4.0, inv_fraction)
    target_price = micro + (obi * OBI_WEIGHT) - inv_shift

    for ask_price in sorted(depth.sell_orders.keys()):
        if ask_price >= target_price - TAKE_BUFFER:
            break
        cap = buy_capacity(position, limit)
        if cap <= 0:
            break
        qty = min(cap, abs(depth.sell_orders[ask_price]))
        orders.append(Order(product, ask_price, qty))
        position += qty

    for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
        if bid_price <= target_price + TAKE_BUFFER:
            break
        cap = sell_capacity(position, limit)
        if cap <= 0:
            break
        qty = min(cap, depth.buy_orders[bid_price])
        orders.append(Order(product, bid_price, -qty))
        position -= qty

    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)
    bid_px, bid_qty = get_best_bid(depth)
    ask_px, ask_qty = get_best_ask(depth)

    if bid_px is not None and ask_px is not None:
        ideal_bid = math.floor(target_price - SPREAD)
        ideal_ask = math.ceil(target_price + SPREAD)
        safe_bid_min_vol = 5 if obi < -0.4 else 2
        safe_ask_min_vol = 5 if obi >  0.4 else 2
        safe_bid = bid_px if bid_qty > safe_bid_min_vol else bid_px - 1
        safe_ask = ask_px if abs(ask_qty) > safe_ask_min_vol else ask_px + 1
        my_bid = min(safe_bid + 1, ideal_bid, ask_px - 1)
        my_ask = max(safe_ask - 1, ideal_ask, bid_px + 1)
        if buy_room > 0:
            orders.append(Order(product, my_bid, buy_room))
        if sell_room > 0:
            orders.append(Order(product, my_ask, -sell_room))

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

        # Ensure trend sub-dict always exists
        trader_data.setdefault("trend", {
            "n": 0, "sum_x": 0.0, "sum_y": 0.0,
            "sum_xx": 0.0, "sum_xy": 0.0,
            "slope": None, "intercept": None,
        })

        # 2. Log snapshot
        self._log_state(state)

        # 3. Build orders
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            depth = state.order_depths[product]
            if not depth.buy_orders or not depth.sell_orders:
                continue

            orders: List[Order] = []

            if product == "ASH_COATED_OSMIUM":
                orders += strategy_take_quotes(
                    product, state, trader_data, fair_value=10_000
                )

            elif product == "INTARIAN_PEPPER_ROOT":
                orders += strategy_linear_trend(product, state, trader_data)

            else:
                orders += strategy_obi_market_make(product, state, trader_data)

            if orders:
                result[product] = orders
                self._log_orders(product, orders)

        # 4. Conversions
        conversions = 0

        # 5. Persist state
        return result, conversions, json.dumps(trader_data)

    def _log_state(self, state: TradingState) -> None:
        print("=" * 60)
        print("Timestamp: " + str(state.timestamp))
        for product, depth in state.order_depths.items():
            bid_px, bid_qty = get_best_bid(depth)
            ask_px, ask_qty = get_best_ask(depth)
            pos = state.position.get(product, 0)
            mid = get_mid_price(depth)
            print(
                "  " + product.ljust(25) +
                " | pos=" + str(pos).rjust(4) +
                " | bid=" + str(bid_px) + "x" + str(bid_qty) +
                "  ask=" + str(ask_px) + "x" + str(ask_qty) +
                "  mid=" + str(mid)
            )

    def _log_orders(self, product: str, orders: List[Order]) -> None:
        for o in orders:
            side = "BUY " if o.quantity > 0 else "SELL"
            print("  -> " + side + " " + str(abs(o.quantity)).rjust(3) +
                  " " + product + " @ " + str(o.price))