"""
trader.py — IMC Prosperity Round 5 (Combined)
==============================================
Signal architecture  : winning algo (z-score residuals, sticky state,
                       basket NAV, regression, lead-lag, stress gate)
Execution layer      : our algo (OBI micro-price taker + passive quotes)

Key differences from the winning algo alone:
  - Execution uses OBI-adjusted micro-price, not just best_bid+1
  - Aggressively takes clearly mispriced liquidity before posting passives
  - Market stress gating retained from winning algo

Key differences from our algo alone:
  - Fair value = live rolling residual z-score, not static offset
  - Sticky direction state prevents signal thrashing
  - Pebbles basket NAV signal
  - Regression fair value for Translators and Panels
  - Lead-lag: Robot Laundry -> Robot Mopping
  - Market stress gate scales size and disables MM
"""

from typing import Dict, List, Tuple, Optional
from datamodel import Order, OrderDepth, TradingState
import json
import math


LIMIT        = 10
HISTORY_LIM  = 96
PEBBLES_NAV  = 50_000.0
SQRT2        = math.sqrt(2.0)

SNACKPACK  = ["SNACKPACK_CHOCOLATE","SNACKPACK_PISTACHIO","SNACKPACK_RASPBERRY",
              "SNACKPACK_STRAWBERRY","SNACKPACK_VANILLA"]
PEBBLES    = ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"]
SLEEP_POD  = ["SLEEP_POD_COTTON","SLEEP_POD_LAMB_WOOL","SLEEP_POD_NYLON",
              "SLEEP_POD_POLYESTER","SLEEP_POD_SUEDE"]
GALAXY     = ["GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_PLANETARY_RINGS",
              "GALAXY_SOUNDS_SOLAR_WINDS","GALAXY_SOUNDS_DARK_MATTER",
              "GALAXY_SOUNDS_SOLAR_FLAMES"]
ROBOTS     = ["ROBOT_DISHES","ROBOT_IRONING","ROBOT_LAUNDRY","ROBOT_MOPPING","ROBOT_VACUUMING"]
MICROCHIP  = ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_RECTANGLE",
              "MICROCHIP_SQUARE","MICROCHIP_TRIANGLE"]
UV_VISOR   = ["UV_VISOR_AMBER","UV_VISOR_MAGENTA","UV_VISOR_ORANGE","UV_VISOR_RED","UV_VISOR_YELLOW"]
OXYGEN     = ["OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_EVENING_BREATH","OXYGEN_SHAKE_GARLIC",
              "OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_MORNING_BREATH"]
PANEL      = ["PANEL_1X2","PANEL_1X4","PANEL_2X2","PANEL_2X4","PANEL_4X4"]
TRANSLATOR = ["TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST",
              "TRANSLATOR_SPACE_GRAY","TRANSLATOR_VOID_BLUE"]

ALL_PRODUCTS = SNACKPACK + PEBBLES + SLEEP_POD + GALAXY + ROBOTS + MICROCHIP + UV_VISOR + OXYGEN + PANEL + TRANSLATOR

STRESS_PRODUCTS = [
    "SNACKPACK_RASPBERRY","SNACKPACK_STRAWBERRY","SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA",
    "PEBBLES_XL","PEBBLES_M","MICROCHIP_OVAL","PANEL_1X2",
    "ROBOT_LAUNDRY","TRANSLATOR_SPACE_GRAY","OXYGEN_SHAKE_EVENING_BREATH",
]

SNACK_PAIRS = [
    ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",    1.30, 0.35, 0.40),
    ("SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY", 1.25, 0.35, 0.45),
]

GROUP_RESIDUAL = {
    # Sleep Pods — all 5 variants now covered
    "SLEEP_POD_COTTON":    (SLEEP_POD, 32, 1.75, 0.50, 2),
    "SLEEP_POD_LAMB_WOOL": (SLEEP_POD, 32, 1.70, 0.50, 2),  # was losing -5329 unsignaled
    "SLEEP_POD_NYLON":     (SLEEP_POD, 32, 1.70, 0.50, 2),
    "SLEEP_POD_POLYESTER": (SLEEP_POD, 32, 1.80, 0.50, 2),
    "SLEEP_POD_SUEDE":     (SLEEP_POD, 32, 1.70, 0.50, 2),
    # Oxygen Shakes — all 5
    "OXYGEN_SHAKE_GARLIC":          (OXYGEN, 26, 1.70, 0.35, 2),
    "OXYGEN_SHAKE_MORNING_BREATH":  (OXYGEN, 26, 1.65, 0.40, 2),  # was losing -3299
    "OXYGEN_SHAKE_MINT":            (OXYGEN, 26, 1.65, 0.40, 2),  # was losing -2294
    "OXYGEN_SHAKE_CHOCOLATE":       (OXYGEN, 26, 1.65, 0.40, 2),
    "OXYGEN_SHAKE_EVENING_BREATH":  (OXYGEN, 26, 1.65, 0.40, 2),
    # Microchips
    "MICROCHIP_SQUARE":     (MICROCHIP, 30, 1.60, 0.40, 2),
    "MICROCHIP_OVAL":       (MICROCHIP, 30, 1.60, 0.40, 2),
    "MICROCHIP_CIRCLE":     (MICROCHIP, 30, 1.60, 0.40, 2),
    "MICROCHIP_RECTANGLE":  (MICROCHIP, 30, 1.60, 0.40, 2),
    "MICROCHIP_TRIANGLE":   (MICROCHIP, 30, 1.60, 0.40, 2),
    # Pebbles — XL/XS only (others handled by basket NAV)
    "PEBBLES_XL": (PEBBLES, 30, 1.50, 0.35, 2),
    "PEBBLES_XS": (PEBBLES, 30, 1.50, 0.35, 2),
    "PEBBLES_S":  (PEBBLES, 30, 1.50, 0.35, 2),
    "PEBBLES_L":  (PEBBLES, 30, 1.50, 0.35, 2),
    "PEBBLES_M":  (PEBBLES, 30, 1.50, 0.35, 2),
    # UV Visors — all 5
    "UV_VISOR_AMBER":   (UV_VISOR, 30, 1.60, 0.40, 2),
    "UV_VISOR_MAGENTA": (UV_VISOR, 30, 1.55, 0.40, 2),
    "UV_VISOR_ORANGE":  (UV_VISOR, 30, 1.55, 0.40, 2),
    "UV_VISOR_RED":     (UV_VISOR, 30, 1.55, 0.40, 2),
    "UV_VISOR_YELLOW":  (UV_VISOR, 30, 1.55, 0.40, 2),
    # Robots — all 5
    "ROBOT_MOPPING":    (ROBOTS, 28, 1.55, 0.40, 2),
    "ROBOT_IRONING":    (ROBOTS, 28, 1.55, 0.40, 2),
    "ROBOT_DISHES":     (ROBOTS, 28, 1.55, 0.40, 2),  # was losing -3950 unsignaled
    "ROBOT_LAUNDRY":    (ROBOTS, 28, 1.55, 0.40, 2),
    "ROBOT_VACUUMING":  (ROBOTS, 28, 1.55, 0.40, 2),
    # Galaxy Sounds — all 5
    "GALAXY_SOUNDS_BLACK_HOLES":      (GALAXY, 30, 1.65, 0.40, 2),
    "GALAXY_SOUNDS_DARK_MATTER":      (GALAXY, 30, 1.65, 0.40, 2),  # was losing -3897
    "GALAXY_SOUNDS_PLANETARY_RINGS":  (GALAXY, 30, 1.65, 0.40, 2),  # was losing -7251
    "GALAXY_SOUNDS_SOLAR_FLAMES":     (GALAXY, 30, 1.65, 0.40, 2),
    "GALAXY_SOUNDS_SOLAR_WINDS":      (GALAXY, 30, 1.65, 0.40, 2),
    # Panels — all 5
    "PANEL_1X2": (PANEL, 30, 1.65, 0.40, 2),  # was losing -2440
    "PANEL_1X4": (PANEL, 30, 1.65, 0.40, 2),  # was losing -2177
    "PANEL_2X2": (PANEL, 30, 1.65, 0.40, 2),  # was losing -2345
    "PANEL_2X4": (PANEL, 30, 1.65, 0.40, 2),
    "PANEL_4X4": (PANEL, 30, 1.65, 0.40, 2),  # also in REGRESSION — regression takes priority
    # Translators — all 5
    "TRANSLATOR_ASTRO_BLACK":      (TRANSLATOR, 30, 1.65, 0.40, 2),
    "TRANSLATOR_ECLIPSE_CHARCOAL": (TRANSLATOR, 30, 1.60, 0.40, 2),
    "TRANSLATOR_GRAPHITE_MIST":    (TRANSLATOR, 30, 1.60, 0.40, 2),
    "TRANSLATOR_SPACE_GRAY":       (TRANSLATOR, 30, 1.65, 0.40, 2),  # was losing -5254
    "TRANSLATOR_VOID_BLUE":        (TRANSLATOR, 30, 1.60, 0.40, 2),
}

REGRESSION = {
    "TRANSLATOR_ASTRO_BLACK": {
        "peers":     ["TRANSLATOR_ECLIPSE_CHARCOAL","TRANSLATOR_GRAPHITE_MIST",
                      "TRANSLATOR_SPACE_GRAY","TRANSLATOR_VOID_BLUE"],
        "beta":      [0.214, -0.099, -0.178, -0.694],
        "intercept": 17502.45,
        "window": 36, "entry": 1.75, "exit": 0.45, "size": 2,
    },
    "PANEL_4X4": {
        "peers":     ["PANEL_1X2","PANEL_1X4","PANEL_2X2","PANEL_2X4"],
        "beta":      [-0.064, -0.079, -0.500, -0.442],
        "intercept": 20965.55,
        "window": 36, "entry": 1.70, "exit": 0.45, "size": 2,
    },
}

LAG_RULES = {
    "ROBOT_LAUNDRY": {
        "follower": "ROBOT_MOPPING", "relation_sign": 1,
        "move_z": 1.55, "gap_z": 1.00, "flow_threshold": 6, "size": 1,
    },
}

OBI_WEIGHT  = 1.5
SPREAD_HALF = 2
# Taker is disabled by default — only enabled for products where we have
# extremely high-confidence fair value (e.g. clear book crossing).
# Set > 0 to re-enable aggressive taking; keep at 0 to match winning algo discipline.
TAKE_BUFFER = 0       # ticks of edge required to take liquidity (0 = passive only)
MM_SPREAD   = 2       # half-spread for resting passive quotes around fair value


def make_state() -> dict:
    return {"history": {}, "sticky": {}}


# ---------------------------------------------------------------------------
# Book helpers
# ---------------------------------------------------------------------------

def _best_bid(d: OrderDepth):
    if not d.buy_orders: return None, None
    px = max(d.buy_orders); return px, d.buy_orders[px]

def _best_ask(d: OrderDepth):
    if not d.sell_orders: return None, None
    px = min(d.sell_orders); return px, d.sell_orders[px]

def _mid(d: OrderDepth):
    b, _ = _best_bid(d); a, _ = _best_ask(d)
    return (b + a) / 2.0 if b is not None and a is not None else None

def _micro(d: OrderDepth):
    b_px, b_qty = _best_bid(d); a_px, a_qty = _best_ask(d)
    if b_px is None or a_px is None: return None
    a_qty = abs(a_qty); total = b_qty + a_qty
    if total == 0: return (b_px + a_px) / 2.0
    return (b_px * a_qty + a_px * b_qty) / total

def _obi(d: OrderDepth):
    b_px, b_qty = _best_bid(d); a_px, a_qty = _best_ask(d)
    if b_px is None or a_px is None: return 0.0
    a_qty = abs(a_qty); total = b_qty + a_qty
    return (b_qty - a_qty) / total if total else 0.0

def _buy_cap(pos):  return LIMIT - pos
def _sell_cap(pos): return LIMIT + pos

def _inv_skew(pos, strength=3.0):
    frac = pos / LIMIT if LIMIT else 0.0
    return math.copysign((abs(frac) ** 2) * strength, frac)


# ---------------------------------------------------------------------------
# Stats helpers
# ---------------------------------------------------------------------------

def _mean(arr): return sum(arr) / len(arr)

def _std(arr):
    mu = _mean(arr)
    var = sum((x - mu) ** 2 for x in arr) / len(arr)
    return math.sqrt(var)

def _zscore(series, window):
    if len(series) < window: return None
    arr = series[-window:]
    sigma = _std(arr)
    return (arr[-1] - _mean(arr)) / sigma if sigma > 1e-9 else None

def _last_move_z(series, window):
    if len(series) < window + 1: return None
    arr = series[-(window+1):]
    rets = [arr[i+1] - arr[i] for i in range(len(arr)-1)]
    if len(rets) < 2: return None
    sigma = _std(rets[:-1])
    return rets[-1] / sigma if sigma > 1e-9 else None

def _group_residual_z(history, product, group, window):
    if any(len(history.get(p, [])) < window for p in group): return None
    others = [p for p in group if p != product]
    res = [history[product][i] - _mean([history[p][i] for p in others])
           for i in range(-window, 0)]
    sigma = _std(res)
    return (res[-1] - _mean(res)) / sigma if sigma > 1e-9 else None

def _regression_residual_z(history, product, peers, beta, intercept, window):
    needed = [product] + peers
    if any(len(history.get(p, [])) < window for p in needed): return None
    res = [history[product][i] - (intercept + sum(b * history[p][i] for b, p in zip(beta, peers)))
           for i in range(-window, 0)]
    sigma = _std(res)
    return (res[-1] - _mean(res)) / sigma if sigma > 1e-9 else None

def _sticky_dir(prev, score, entry, exit_):
    if score is None: return prev
    if score > entry:  return 1
    if score < -entry: return -1
    if abs(score) < exit_: return 0
    return prev

def _symmetric_dir(prev, signal, entry, exit_):
    if signal is None: return prev
    if signal > entry:  return -1
    if signal < -entry: return 1
    if abs(signal) < exit_: return 0
    return prev

def _threshold_dir(prev, score, entry, exit_):
    if score > entry:  return 1
    if score < -entry: return -1
    if abs(score) < exit_: return 0
    return prev

def _pair_signal(z_a, z_b):
    if z_a is None or z_b is None: return None
    return (z_a - z_b) / SQRT2

def _relative_pair_dir(prev, signal, z_a, z_b, entry, exit_, leg_min):
    if signal is None or z_a is None or z_b is None: return prev
    if signal > entry and z_a > leg_min and z_b < -leg_min: return -1
    if signal < -entry and z_a < -leg_min and z_b > leg_min: return 1
    if abs(signal) < exit_: return 0
    return prev


# ---------------------------------------------------------------------------
# Market stress
# ---------------------------------------------------------------------------

def _market_stress(history):
    scores = [abs(z) for p in STRESS_PRODUCTS
              if (z := _last_move_z(history.get(p, []), 20)) is not None]
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Signals
# ---------------------------------------------------------------------------

def _snack_targets(history, sticky):
    state = sticky.setdefault("snack", {})
    targets = {}
    for prod_a, prod_b, entry, exit_, leg_min in SNACK_PAIRS:
        key   = f"{prod_a}|{prod_b}"
        z_a   = _zscore(history.get(prod_a, []), 36)
        z_b   = _zscore(history.get(prod_b, []), 36)
        sig   = _pair_signal(z_a, z_b)
        dirn  = _relative_pair_dir(state.get(key, 0), sig, z_a, z_b, entry, exit_, leg_min)
        state[key] = dirn
        targets[prod_a] = dirn * LIMIT
        targets[prod_b] = -dirn * LIMIT

    rs_dir  = state.get("SNACKPACK_RASPBERRY|SNACKPACK_STRAWBERRY", 0)
    pist_z  = _zscore(history.get("SNACKPACK_PISTACHIO", []), 36)
    straw_z = _zscore(history.get("SNACKPACK_STRAWBERRY", []), 36)
    pist_t  = 0
    if rs_dir != 0 and pist_z is not None and straw_z is not None:
        if abs(pist_z) > 0.35 and pist_z * straw_z > 0:
            pist_t = int(math.copysign(5, -rs_dir))
    targets["SNACKPACK_PISTACHIO"] = pist_t
    return targets


def _pebbles_targets(history, mids, sticky):
    if not all(p in mids for p in PEBBLES): return {}
    state = sticky.setdefault("pebbles", {"sum": 0, "xl": 0})
    targets = {p: 0 for p in PEBBLES}

    total   = sum(mids[p] for p in PEBBLES)
    sum_dir = _threshold_dir(state.get("sum", 0), PEBBLES_NAV - total, 5.0, 1.5)
    state["sum"] = sum_dir
    for p in PEBBLES: targets[p] += sum_dir

    window = 40
    if all(len(history.get(p, [])) >= window for p in PEBBLES):
        series = []
        for i in range(-window, 0):
            try:
                val = math.log(max(history["PEBBLES_XL"][i], 1e-9))
                for other in ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L"]:
                    val -= 0.25 * math.log(max(history[other][i], 1e-9))
                series.append(val)
            except Exception:
                pass
        if series:
            xl_z   = _zscore(series, len(series))
            xl_dir = _symmetric_dir(state.get("xl", 0), xl_z, 1.10, 0.30)
            state["xl"] = xl_dir
            targets["PEBBLES_XL"] += xl_dir * 4
            for p in ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L"]:
                targets[p] += -xl_dir

    return {p: max(-LIMIT, min(LIMIT, v)) for p, v in targets.items()}


def _group_residual_targets(history, sticky):
    state = sticky.setdefault("group_res", {})
    targets = {}
    for product, (group, window, entry, exit_, size) in GROUP_RESIDUAL.items():
        z    = _group_residual_z(history, product, group, window)
        dirn = _sticky_dir(state.get(product, 0), -z if z is not None else None, entry, exit_)
        state[product] = dirn
        targets[product] = dirn * size
    return targets


def _regression_targets(history, sticky):
    state = sticky.setdefault("regression", {})
    targets = {}
    for product, cfg in REGRESSION.items():
        z    = _regression_residual_z(history, product, cfg["peers"],
                                       cfg["beta"], cfg["intercept"], cfg["window"])
        dirn = _sticky_dir(state.get(product, 0), -z if z is not None else None,
                            cfg["entry"], cfg["exit"])
        state[product] = dirn
        targets[product] = dirn * cfg["size"]
    return targets


def _lag_nudges(state_trading, history, mids):
    nudges = {}
    for leader, cfg in LAG_RULES.items():
        follower = cfg["follower"]
        if leader not in mids or follower not in mids: continue
        lz = _last_move_z(history.get(leader, []), 20)
        fz = _last_move_z(history.get(follower, []), 20)
        if lz is None or fz is None: continue
        gap = cfg["relation_sign"] * lz - fz
        if abs(lz) < cfg["move_z"] or abs(gap) < cfg["gap_z"]: continue

        depth = state_trading.order_depths.get(leader)
        mid_l = mids.get(leader)
        trades = []
        if hasattr(state_trading, "market_trades") and state_trading.market_trades:
            trades = state_trading.market_trades.get(leader, [])
        flow = 0.0
        if trades and mid_l is not None and depth:
            b_px = max(depth.buy_orders) if depth.buy_orders else None
            a_px = min(depth.sell_orders) if depth.sell_orders else None
            for t in trades:
                price = float(t.price); qty = float(t.quantity)
                if a_px is not None and price >= a_px: flow += qty
                elif b_px is not None and price <= b_px: flow -= qty
                elif price > mid_l: flow += qty
                elif price < mid_l: flow -= qty

        flow_sign = (1 if flow > 0 else -1) if abs(flow) >= cfg["flow_threshold"] else 0
        leader_sign = 1 if lz > 0 else -1
        if flow_sign != 0 and flow_sign != leader_sign: continue

        dirn = 1 if gap > 0 else -1
        nudges[follower] = nudges.get(follower, 0) + dirn * cfg["size"]
    return nudges


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _passive_toward_target(product, target, pos, depth, size_scale=1.0, clip=2):
    """
    Passive-only execution matching the winning algo's style.
    Posts one limit order one tick inside the spread toward the target.
    clip=2 limits how many units we chase per tick to avoid over-filling
    into a fast-moving market. Only fills when the market comes to us.
    """
    need = target - pos
    if need == 0: return []
    b_px, _ = _best_bid(depth); a_px, _ = _best_ask(depth)
    if b_px is None or a_px is None: return []
    qty = max(1, min(abs(need), max(1, int(clip * size_scale))))
    qty = min(qty, _buy_cap(pos) if need > 0 else _sell_cap(pos))
    if qty <= 0: return []
    if need > 0:
        price = b_px + 1 if b_px + 1 < a_px else b_px
        return [Order(product, int(price), qty)]
    price = a_px - 1 if a_px - 1 > b_px else a_px
    return [Order(product, int(price), -qty)]


def _passive_mm(product, state, fair_value, size_scale=1.0):
    """
    Passive market-make around a live fair value.
    Posts bid and ask one tick inside the existing spread.
    No taking — only rests. Skips if spread < 4 ticks.
    """
    orders = []
    depth  = state.order_depths.get(product)
    if depth is None or not depth.buy_orders or not depth.sell_orders: return orders
    pos  = state.position.get(product, 0)
    b_px, _ = _best_bid(depth); a_px, _ = _best_ask(depth)
    if b_px is None or a_px is None: return orders
    if a_px - b_px < 4: return orders
    ob   = _obi(depth)
    skew = _inv_skew(pos, strength=3.0)
    fv   = fair_value + ob * OBI_WEIGHT - skew
    ideal_bid = math.floor(fv) - MM_SPREAD
    ideal_ask = math.ceil(fv)  + MM_SPREAD
    my_bid = max(min(b_px + 1, ideal_bid), b_px)
    my_ask = min(max(a_px - 1, ideal_ask), a_px)
    if my_bid >= my_ask: return orders
    bc = int(_buy_cap(pos)  * size_scale)
    sc = int(_sell_cap(pos) * size_scale)
    if bc > 0: orders.append(Order(product, my_bid, bc))
    if sc > 0: orders.append(Order(product, my_ask, -sc))
    return orders


# ---------------------------------------------------------------------------
# Trader
# ---------------------------------------------------------------------------

class Trader:

    def run(self, state: TradingState):
        td = make_state()
        if state.traderData:
            try:
                td = json.loads(state.traderData)
                td.setdefault("history", {}); td.setdefault("sticky", {})
            except Exception:
                td = make_state()

        history = td["history"]
        sticky  = td["sticky"]

        mids = {}
        for product, depth in state.order_depths.items():
            m = _mid(depth)
            if m is None: continue
            mids[product] = m
            series = history.get(product, [])
            series.append(m)
            history[product] = series[-HISTORY_LIM:]

        stress      = _market_stress(history)
        allow_mm    = stress < 1.35
        other_scale = 1.0
        if stress > 2.00:   other_scale = 0.35
        elif stress > 1.45: other_scale = 0.65

        structural = {}
        structural.update(_snack_targets(history, sticky))
        structural.update(_pebbles_targets(history, mids, sticky))

        auxiliary = {}
        # Regression runs first — more precise signal for TRANSLATOR_ASTRO_BLACK and PANEL_4X4.
        # group_residual fills in the rest; it won't overwrite regression results since
        # the merge loop below uses setdefault semantics (first writer wins per product).
        auxiliary.update(_regression_targets(history, sticky))
        auxiliary.update(_group_residual_targets(history, sticky))

        lag = _lag_nudges(state, history, mids)

        final: Dict[str, int] = {}
        for p, t in structural.items():
            final[p] = max(-LIMIT, min(LIMIT, t))
        regression_products = set(REGRESSION.keys())
        for p, t in auxiliary.items():
            if p not in final:
                final[p] = max(-LIMIT, min(LIMIT, int(round(t * other_scale))))
            elif p in regression_products:
                # Regression signal overrides any existing auxiliary entry
                final[p] = max(-LIMIT, min(LIMIT, int(round(t * other_scale))))
        for p, nudge in lag.items():
            if p in structural: continue
            base = final.get(p, 0)
            final[p] = max(-LIMIT, min(LIMIT, base + nudge))

        result: Dict[str, List[Order]] = {}

        for product, depth in state.order_depths.items():
            if not depth.buy_orders or not depth.sell_orders: continue
            pos = state.position.get(product, 0)
            orders = []

            if product in final:
                target = final[product]
                # Passive-only toward target: matches winning algo discipline.
                # clip=2 means we chase at most 2 units per tick so we don't
                # lunge into a trend; the sticky signal holds across many ticks.
                orders = _passive_toward_target(
                    product, target, pos, depth,
                    size_scale=other_scale, clip=2
                )
            elif pos != 0:
                # No signal: unwind to zero passively regardless of stress level.
                # Passive MM for unsignaled products caused -30k in losses by
                # accumulating inventory into trends. Default is now flat.
                orders = _passive_toward_target(
                    product, 0, pos, depth, size_scale=other_scale, clip=1
                )

            if orders:
                result[product] = orders

        trader_data = json.dumps(td, separators=(",", ":"))
        if len(trader_data) > 49000:
            for p in history: history[p] = history[p][-72:]
            trader_data = json.dumps(td, separators=(",", ":"))

        print(f"t={state.timestamp} stress={stress:.2f} signals={len(final)}")
        return result, 0, trader_data