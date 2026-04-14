"""
trader.py — IMC Prosperity Algorithm Template
==============================================
Allowed imports (official): pandas, numpy, statistics, math, typing, jsonpickle
Position limits and product names change every round — update POSITION_LIMITS
and PRODUCTS at the start of each round.

Submission requirements
-----------------------
  • Class must be named exactly  Trader
  • run() must return a 3-tuple: (result, conversions, traderData)
      result      : Dict[str, List[Order]]  — orders per product
      conversions : int                     — # of units to convert via external market
      traderData  : str                     — arbitrary string; persists to next iteration
  • Execution budget: < 900 ms per run() call
  • No I/O (no open(), no os, no requests, no subprocess)
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

MAX_HISTORY = 100   # rolling window length for price history


# ---------------------------------------------------------------------------
# Persistent state — plain dict, serialised with json.dumps/loads
# Add keys here as your strategies need more cross-iteration memory.
# ---------------------------------------------------------------------------

def make_state() -> dict:
    """Factory for a fresh persistent state dict."""
    return {
        # price_history[product] = list of recent mid-prices
        "price_history": {p: [] for p in PRODUCTS},
        # Extend here for round 2+, e.g.:
        # "ema": {p: None for p in PRODUCTS},
        # "vwap_history": {p: [] for p in PRODUCTS},
    }

def state_update_price(trader_data: dict, product: str, mid: float) -> None:
    hist = trader_data["price_history"].setdefault(product, [])
    hist.append(mid)
    if len(hist) > MAX_HISTORY:
        hist.pop(0)

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
    """Returns (price, quantity) of the best bid, or (None, None)."""
    if not depth.buy_orders:
        return None, None
    price = max(depth.buy_orders.keys())
    return price, depth.buy_orders[price]


def get_best_ask(depth: OrderDepth) -> Tuple[Optional[int], Optional[int]]:
    """Returns (price, quantity) of the best ask, or (None, None).
    Note: ask quantities in order_depths are NEGATIVE by Prosperity convention."""
    if not depth.sell_orders:
        return None, None
    price = min(depth.sell_orders.keys())
    return price, depth.sell_orders[price]   # quantity will be negative


def get_mid_price(depth: OrderDepth) -> Optional[float]:
    bid, _ = get_best_bid(depth)
    ask, _ = get_best_ask(depth)
    if bid is not None and ask is not None:
        return (bid + ask) / 2.0
    return None


def get_micro_price(depth: OrderDepth) -> Optional[float]:
    """Volume-weighted mid — weights bid/ask by the opposing side's volume.
    More accurate than simple mid when the book is imbalanced."""
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
    """Order Book Imbalance: ranges -1 (all ask pressure) to +1 (all bid pressure)."""
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
    """Units we can still buy before hitting the long limit."""
    return limit - position


def sell_capacity(position: int, limit: int) -> int:
    """Units we can still sell before hitting the short limit (returned positive)."""
    return limit + position


# ---------------------------------------------------------------------------
# Strategy: TAKER (when fair value is known, take mispriced quotes)
# ---------------------------------------------------------------------------

def strategy_take_quotes(
    product: str,
    state: TradingState,
    trader_data: dict,
    fair_value: float,
) -> List[Order]:
    """
    Aggressively takes any mispriced quotes from the order book.
    Best for products with a stable, known fair value.
    Buys everything priced below fair value, sells everything priced above it.
    Uses non-linear inventory skew to stay within position limits safely.
    """
    orders: List[Order] = []
    depth = state.order_depths.get(product)
    if depth is None:
        return orders

    position = state.position.get(product, 0)
    limit    = POSITION_LIMITS.get(product, 20)

    # Non-linear inventory skew
    inv_fraction = position / limit if limit != 0 else 0
    inv_shift = math.copysign((abs(inv_fraction) ** 3) * 2.5, inv_fraction)
    reservation_price = fair_value - inv_shift

    # Take cheap asks (buy below reservation price)
    for ask_price in sorted(depth.sell_orders.keys()):
        if ask_price >= reservation_price:
            break
        cap = buy_capacity(position, limit)
        if cap <= 0:
            break
        qty = min(cap, abs(depth.sell_orders[ask_price]))
        orders.append(Order(product, ask_price, qty))
        position += qty

    # Take rich bids (sell above reservation price)
    for bid_price in sorted(depth.buy_orders.keys(), reverse=True):
        if bid_price <= reservation_price:
            break
        cap = sell_capacity(position, limit)
        if cap <= 0:
            break
        qty = min(cap, depth.buy_orders[bid_price])
        orders.append(Order(product, bid_price, -qty))
        position -= qty

    # Squeeze out better bids and asks than the bots for the rest of our position capacity
    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)
    bid_px, _ = get_best_bid(depth)
    ask_px, _ = get_best_ask(depth)

    if bid_px is not None and ask_px is not None:
        ideal_bid = math.floor(reservation_price - 1)
        ideal_ask = math.ceil(reservation_price + 1)
        my_bid = min(bid_px + 1, ideal_bid)
        my_ask = max(ask_px - 1, ideal_ask)
        if buy_room > 0:
            orders.append(Order(product, my_bid, buy_room))
        if sell_room > 0:
            orders.append(Order(product, my_ask, -sell_room))

    return orders


# ---------------------------------------------------------------------------
# Strategy: OBI + micro-price market making (for trending/unknown products)
# ---------------------------------------------------------------------------

def strategy_obi_market_make(
    product: str,
    state: TradingState,
    trader_data: dict,
) -> List[Order]:
    """
    Uses Order Book Imbalance and micro-price as a dynamic fair value signal.
    Includes inventory penalty to skew quotes and avoid getting stuck long/short.
    Good default for products where you don't know the true fair value.
    """
    OBI_WEIGHT  = 1.5   # how strongly OBI shifts your fair value estimate
    SPREAD      = 1     # ticks to add/subtract around fair value for resting quotes
    TAKE_BUFFER = 0.5   # cross the spread if mispricing > this amount

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
    inv_shift = math.copysign((abs(inv_fraction) ** 2) * 4.0, inv_fraction)

    # Dynamic fair value: micro-price adjusted for flow and inventory
    target_price = micro + (obi * OBI_WEIGHT) - inv_shift

    # Phase 1: Take liquidity where the book is clearly mispriced
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

    # Phase 2: Post resting quotes with OBI-based dust filtering
    buy_room  = buy_capacity(position, limit)
    sell_room = sell_capacity(position, limit)
    bid_px, bid_qty = get_best_bid(depth)
    ask_px, ask_qty = get_best_ask(depth)

    if bid_px is not None and ask_px is not None:
        ideal_bid = math.floor(target_price - SPREAD)
        ideal_ask = math.ceil(target_price + SPREAD)

        # Filter thin quotes — don't improve price into a dust wall
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
# Strategy: Mean reversion (for products that oscillate around a stable mean)
# ---------------------------------------------------------------------------

def strategy_mean_reversion(
    product: str,
    state: TradingState,
    trader_data: dict,
    z_threshold: float = 1.5,
    order_size: int = 8,
) -> List[Order]:
    """
    Trades when price deviates more than z_threshold standard deviations
    from its rolling mean.  Needs ~3 iterations to warm up.
    """
    orders: List[Order] = []
    depth = state.order_depths.get(product)
    if depth is None:
        return orders

    mid = get_mid_price(depth)
    if mid is None:
        return orders

    state_update_price(trader_data, product, mid)

    mean = state_rolling_mean(trader_data, product)
    std  = state_rolling_std(trader_data, product)
    if mean is None or std is None or std < 1e-6:
        return orders   # not enough history yet

    z = (mid - mean) / std
    position = state.position.get(product, 0)
    limit    = POSITION_LIMITS.get(product, 20)

    bid_px, _ = get_best_bid(depth)
    ask_px, _ = get_best_ask(depth)

    # Price too high -> sell aggressively at the best bid
    if z > z_threshold and bid_px is not None:
        cap = min(order_size, sell_capacity(position, limit))
        if cap > 0:
            orders.append(Order(product, bid_px, -cap))

    # Price too low -> buy aggressively at the best ask
    elif z < -z_threshold and ask_px is not None:
        cap = min(order_size, buy_capacity(position, limit))
        if cap > 0:
            orders.append(Order(product, ask_px, cap))

    return orders


# ---------------------------------------------------------------------------
# Main Trader class
# ---------------------------------------------------------------------------
class Trader:
    """
    IMC Prosperity Trader.

    The exchange calls run() once per iteration.
    Instance variables don't persist between calls, use traderData for memory.
    """

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        """
        Returns
        -------
        result      : Dict[symbol, List[Order]]  — your orders this iteration
        conversions : int   -- units to convert via external market (0 if unused)
        traderData  : str   -- JSON string; passed back verbatim next iteration
        """

        # 1. Rehydrate persistent state from last iteration
        trader_data: dict
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                trader_data = make_state()
        else:
            trader_data = make_state()

        # 2. Log market snapshot (captured in the platform log file)
        self._log_state(state)

        # 3. Build orders -- one strategy per product
        result: Dict[str, List[Order]] = {}

        for product in state.order_depths:
            depth: OrderDepth = state.order_depths[product]

            # Guard: skip empty books (prevents max()/min() crashes)
            if not depth.buy_orders or not depth.sell_orders:
                continue

            orders: List[Order] = []

            if product == "ASH_COATED_OSMIUM":
                # Known fair value = 10,000; take all mispriced quotes
                orders += strategy_take_quotes(
                    product, state, trader_data, fair_value=10_000
                )

            elif product == "INTARIAN_PEPPER_ROOT":
                # Unknown fair value; use OBI + micro-price market making
                orders += strategy_obi_market_make(product, state, trader_data)

            else:
                # Fallback for new products: passive OBI market making
                orders += strategy_obi_market_make(product, state, trader_data)

            if orders:
                result[product] = orders
                self._log_orders(product, orders)

        # 4. Conversions
        conversions = 0

        # 5. Serialise state back to traderData
        serialized_state = json.dumps(trader_data)

        return result, conversions, serialized_state

    def _log_state(self, state: TradingState) -> None:
        print("=" * 60)
        print("Timestamp: " + str(state.timestamp))
        for product, depth in state.order_depths.items():
            bid_px, bid_qty = get_best_bid(depth)
            ask_px, ask_qty = get_best_ask(depth)
            pos = state.position.get(product, 0)
            mid = get_mid_price(depth)
            print(
                "  " + product.ljust(25) + " | pos=" + str(pos).rjust(4) +
                " | bid=" + str(bid_px) + "x" + str(bid_qty) +
                "  ask=" + str(ask_px) + "x" + str(ask_qty) +
                "  mid=" + str(mid)
            )

    def _log_orders(self, product: str, orders: List[Order]) -> None:
        for o in orders:
            side = "BUY " if o.quantity > 0 else "SELL"
            print("  -> " + side + " " + str(abs(o.quantity)).rjust(3) + " " + product + " @ " + str(o.price))
