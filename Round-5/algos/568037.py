from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math
import numpy as np


class Trader:
    LIMIT = 10
    HISTORY_LIMIT = 96
    PEBBLES_TARGET = 50000.0
    SQRT2 = math.sqrt(2.0)

    SNACKPACK = [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_VANILLA",
    ]
    PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
    SLEEP_POD = [
        "SLEEP_POD_COTTON",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_SUEDE",
    ]
    GALAXY = [
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_WINDS",
    ]

    MR_CONFIG = {
        "OXYGEN_SHAKE_CHOCOLATE": {"window": 26, "entry": 1.70, "exit": 0.35, "size": 2},
    }

    LAG_RULES = {
        "ROBOT_LAUNDRY": {
            "follower": "ROBOT_MOPPING",
            "relation_sign": 1,
            "move_z": 1.55,
            "gap_z": 1.00,
            "flow_threshold": 6,
            "size": 1,
        },
    }

    PACK_NAV_PRODUCTS = {
        "SLEEP_POD_COTTON": {"group": SLEEP_POD, "window": 32, "entry": 1.75, "exit": 0.50, "size": 2},
        "SLEEP_POD_NYLON": {"group": SLEEP_POD, "window": 32, "entry": 1.70, "exit": 0.50, "size": 2},
        "SLEEP_POD_POLYESTER": {"group": SLEEP_POD, "window": 32, "entry": 1.80, "exit": 0.50, "size": 2},
    }

    REGRESSION_PRODUCTS = {
        "TRANSLATOR_ASTRO_BLACK": {
            "peers": [
                "TRANSLATOR_ECLIPSE_CHARCOAL",
                "TRANSLATOR_GRAPHITE_MIST",
                "TRANSLATOR_SPACE_GRAY",
                "TRANSLATOR_VOID_BLUE",
            ],
            "beta": [0.214, -0.099, -0.178, -0.694],
            "intercept": 17502.45,
            "window": 36,
            "entry": 1.75,
            "exit": 0.45,
            "size": 2,
        },
        "PANEL_4X4": {
            "peers": ["PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4"],
            "beta": [-0.064, -0.079, -0.500, -0.442],
            "intercept": 20965.55,
            "window": 36,
            "entry": 1.70,
            "exit": 0.45,
            "size": 2,
        },
    }

    MM_PRODUCTS = {
        "MICROCHIP_SQUARE",
        "MICROCHIP_TRIANGLE",
        "TRANSLATOR_GRAPHITE_MIST",
        "UV_VISOR_ORANGE",
    }

    STRESS_PRODUCTS = [
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_VANILLA",
        "PEBBLES_XL",
        "PEBBLES_M",
        "MICROCHIP_OVAL",
        "PANEL_1X2",
        "ROBOT_LAUNDRY",
        "TRANSLATOR_SPACE_GRAY",
        "OXYGEN_SHAKE_EVENING_BREATH",
    ]

    ALL_PRODUCTS = (
        SNACKPACK
        + PEBBLES
        + SLEEP_POD
        + GALAXY
        + [
            "MICROCHIP_CIRCLE",
            "MICROCHIP_OVAL",
            "MICROCHIP_RECTANGLE",
            "MICROCHIP_SQUARE",
            "MICROCHIP_TRIANGLE",
            "OXYGEN_SHAKE_CHOCOLATE",
            "OXYGEN_SHAKE_EVENING_BREATH",
            "OXYGEN_SHAKE_GARLIC",
            "OXYGEN_SHAKE_MINT",
            "OXYGEN_SHAKE_MORNING_BREATH",
            "PANEL_1X2",
            "PANEL_1X4",
            "PANEL_2X2",
            "PANEL_2X4",
            "PANEL_4X4",
            "ROBOT_DISHES",
            "ROBOT_IRONING",
            "ROBOT_LAUNDRY",
            "ROBOT_MOPPING",
            "ROBOT_VACUUMING",
            "TRANSLATOR_ASTRO_BLACK",
            "TRANSLATOR_ECLIPSE_CHARCOAL",
            "TRANSLATOR_GRAPHITE_MIST",
            "TRANSLATOR_SPACE_GRAY",
            "TRANSLATOR_VOID_BLUE",
            "UV_VISOR_AMBER",
            "UV_VISOR_MAGENTA",
            "UV_VISOR_ORANGE",
            "UV_VISOR_RED",
            "UV_VISOR_YELLOW",
        ]
    )

    def run(self, state: TradingState):
        memory = self._load_state(state.traderData)
        history: Dict[str, List[float]] = memory["history"]
        sticky = memory["sticky"]
        prev_mids: Dict[str, float] = memory["prev_mids"]

        mids: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is None:
                continue
            mids[product] = mid
            series = history.get(product, [])
            series.append(mid)
            history[product] = series[-self.HISTORY_LIMIT :]

        market_stress = self._market_stress(history)
        allow_mm = market_stress < 1.35
        if market_stress > 2.00:
            other_scale = 0.35
        elif market_stress > 1.45:
            other_scale = 0.65
        else:
            other_scale = 1.0

        structural_targets = {}
        structural_targets.update(self._snackpack_targets(history, sticky))
        structural_targets.update(self._pebbles_targets(history, mids, sticky))

        auxiliary_targets = {}
        auxiliary_targets.update(self._pack_nav_targets(history, sticky))
        auxiliary_targets.update(self._regression_targets(history, sticky))
        auxiliary_targets.update(self._mean_reversion_targets(history, sticky))
        auxiliary_targets.update(self._galaxy_targets(history, sticky))

        lag_targets = self._lagger_targets(state, history, mids, sticky)

        final_targets: Dict[str, int] = {}
        for product, target in structural_targets.items():
            final_targets[product] = int(np.clip(target, -self.LIMIT, self.LIMIT))

        for product, target in auxiliary_targets.items():
            scaled = int(np.clip(round(target * other_scale), -self.LIMIT, self.LIMIT))
            if product not in final_targets:
                final_targets[product] = scaled

        for product, nudge in lag_targets.items():
            if product in structural_targets:
                continue
            base = final_targets.get(product, 0)
            final_targets[product] = int(np.clip(base + nudge, -self.LIMIT, self.LIMIT))

        result: Dict[str, List[Order]] = {}
        for product, depth in state.order_depths.items():
            pos = state.position.get(product, 0)
            orders: List[Order] = []

            if product in final_targets:
                orders = self._passive_toward_target(product, final_targets[product], pos, depth)
            elif product in self.MM_PRODUCTS and allow_mm:
                orders = self._safety_mm(product, pos, depth, clip=1 if market_stress > 1.20 else 2)
            elif pos != 0:
                orders = self._passive_toward_target(product, 0, pos, depth)

            if orders:
                result[product] = orders

        memory["prev_mids"] = mids
        trader_data = json.dumps(memory, separators=(",", ":"))
        if len(trader_data) > 49000:
            for product in history:
                history[product] = history[product][-72:]
            trader_data = json.dumps(memory, separators=(",", ":"))

        return result, 0, trader_data

    def _snackpack_targets(self, history: Dict[str, List[float]], sticky: dict) -> Dict[str, int]:
        state = sticky.setdefault("snack", {"cv": 0, "rs": 0})
        targets: Dict[str, int] = {}

        choc_z = self._zscore(history.get("SNACKPACK_CHOCOLATE", []), 36)
        van_z = self._zscore(history.get("SNACKPACK_VANILLA", []), 36)
        cv_signal = self._pair_signal(choc_z, van_z)
        cv_dir = self._relative_pair_direction(
            previous=state.get("cv", 0),
            signal=cv_signal,
            z_a=choc_z,
            z_b=van_z,
            entry=1.30,
            exit=0.35,
            leg_min=0.40,
        )
        state["cv"] = cv_dir
        targets["SNACKPACK_CHOCOLATE"] = cv_dir * 10
        targets["SNACKPACK_VANILLA"] = -cv_dir * 10

        rasp_z = self._zscore(history.get("SNACKPACK_RASPBERRY", []), 36)
        straw_z = self._zscore(history.get("SNACKPACK_STRAWBERRY", []), 36)
        rs_signal = self._pair_signal(rasp_z, straw_z)
        rs_dir = self._relative_pair_direction(
            previous=state.get("rs", 0),
            signal=rs_signal,
            z_a=rasp_z,
            z_b=straw_z,
            entry=1.25,
            exit=0.35,
            leg_min=0.45,
        )
        state["rs"] = rs_dir
        targets["SNACKPACK_RASPBERRY"] = rs_dir * 10
        targets["SNACKPACK_STRAWBERRY"] = -rs_dir * 10

        pist_target = 0
        pist_z = self._zscore(history.get("SNACKPACK_PISTACHIO", []), 36)
        if rs_dir != 0 and pist_z is not None and straw_z is not None:
            if abs(pist_z) > 0.35 and pist_z * straw_z > 0:
                pist_target = int(math.copysign(5, -rs_dir))
        targets["SNACKPACK_PISTACHIO"] = pist_target
        return targets

    def _pebbles_targets(self, history: Dict[str, List[float]], mids: Dict[str, float], sticky: dict) -> Dict[str, int]:
        if not all(product in mids for product in self.PEBBLES):
            return {}

        state = sticky.setdefault("pebbles", {"sum": 0, "xl": 0})
        targets = {product: 0 for product in self.PEBBLES}

        total = sum(mids[product] for product in self.PEBBLES)
        gap = self.PEBBLES_TARGET - total
        sum_dir = self._threshold_direction(
            previous=state.get("sum", 0),
            score=gap,
            entry=5.0,
            exit=1.5,
        )
        state["sum"] = sum_dir
        for product in self.PEBBLES:
            targets[product] += sum_dir

        xl_signal = self._spread_z(
            history,
            [
                ("PEBBLES_XL", 1.0),
                ("PEBBLES_XS", -0.25),
                ("PEBBLES_S", -0.25),
                ("PEBBLES_M", -0.25),
                ("PEBBLES_L", -0.25),
            ],
            40,
        )
        xl_dir = self._symmetric_direction(
            previous=state.get("xl", 0),
            signal=xl_signal,
            entry=1.10,
            exit=0.30,
        )
        state["xl"] = xl_dir
        targets["PEBBLES_XL"] += xl_dir * 4
        for product in ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L"]:
            targets[product] += -xl_dir

        return {product: int(np.clip(target, -self.LIMIT, self.LIMIT)) for product, target in targets.items()}

    def _pack_nav_targets(self, history: Dict[str, List[float]], sticky: dict) -> Dict[str, int]:
        state = sticky.setdefault("pack_nav", {})
        targets: Dict[str, int] = {}

        for product, config in self.PACK_NAV_PRODUCTS.items():
            score = self._group_residual_z(history, product, config["group"], config["window"])
            prev = state.get(product, 0)
            direction = self._sticky_direction(prev, -score if score is not None else None, config["entry"], config["exit"])
            state[product] = direction
            targets[product] = direction * config["size"]
        return targets

    def _mean_reversion_targets(self, history: Dict[str, List[float]], sticky: dict) -> Dict[str, int]:
        state = sticky.setdefault("mr", {})
        targets: Dict[str, int] = {}

        for product, config in self.MR_CONFIG.items():
            z_value = self._zscore(history.get(product, []), config["window"])
            prev = state.get(product, 0)
            direction = self._sticky_direction(prev, -z_value if z_value is not None else None, config["entry"], config["exit"])
            state[product] = direction
            targets[product] = direction * config["size"]
        return targets

    def _regression_targets(self, history: Dict[str, List[float]], sticky: dict) -> Dict[str, int]:
        state = sticky.setdefault("regression", {})
        targets: Dict[str, int] = {}

        for product, config in self.REGRESSION_PRODUCTS.items():
            z_value = self._regression_residual_z(
                history,
                product,
                config["peers"],
                config["beta"],
                config["intercept"],
                config["window"],
            )
            prev = state.get(product, 0)
            direction = self._sticky_direction(prev, -z_value if z_value is not None else None, config["entry"], config["exit"])
            state[product] = direction
            targets[product] = direction * config["size"]
        return targets

    def _galaxy_targets(self, history: Dict[str, List[float]], sticky: dict) -> Dict[str, int]:
        return {}

    def _lagger_targets(
        self,
        state: TradingState,
        history: Dict[str, List[float]],
        mids: Dict[str, float],
        sticky: dict,
    ) -> Dict[str, int]:
        active: Dict[str, int] = {}

        for leader, config in self.LAG_RULES.items():
            follower = config["follower"]
            if leader not in mids or follower not in mids:
                continue

            leader_move = self._last_move_z(history.get(leader, []), 20)
            follower_move = self._last_move_z(history.get(follower, []), 20)
            if leader_move is None or follower_move is None:
                continue

            expected_follower = config["relation_sign"] * leader_move
            relative_gap = expected_follower - follower_move

            leader_flow = self._signed_trade_flow(
                state.market_trades.get(leader, []) if hasattr(state, "market_trades") else [],
                mids.get(leader),
                state.order_depths.get(leader),
            )
            flow_sign = np.sign(leader_flow) if abs(leader_flow) >= config["flow_threshold"] else 0

            if abs(leader_move) < config["move_z"] or abs(relative_gap) < config["gap_z"]:
                continue

            leader_sign = 1 if leader_move > 0 else -1
            if flow_sign != 0 and flow_sign != leader_sign:
                continue

            signal_dir = 1 if relative_gap > 0 else -1
            active[follower] = active.get(follower, 0) + signal_dir * config["size"]

        return active

    def _pair_signal(self, z_a: Optional[float], z_b: Optional[float]) -> Optional[float]:
        if z_a is None or z_b is None:
            return None
        return float((z_a - z_b) / self.SQRT2)

    def _relative_pair_direction(
        self,
        previous: int,
        signal: Optional[float],
        z_a: Optional[float],
        z_b: Optional[float],
        entry: float,
        exit: float,
        leg_min: float,
    ) -> int:
        if signal is None or z_a is None or z_b is None:
            return previous
        if signal > entry and z_a > leg_min and z_b < -leg_min:
            return -1
        if signal < -entry and z_a < -leg_min and z_b > leg_min:
            return 1
        if abs(signal) < exit:
            return 0
        return previous

    def _threshold_direction(self, previous: int, score: float, entry: float, exit: float) -> int:
        if score > entry:
            return 1
        if score < -entry:
            return -1
        if abs(score) < exit:
            return 0
        return previous

    def _symmetric_direction(self, previous: int, signal: Optional[float], entry: float, exit: float) -> int:
        if signal is None:
            return previous
        if signal > entry:
            return -1
        if signal < -entry:
            return 1
        if abs(signal) < exit:
            return 0
        return previous

    def _group_vol_ratio(self, history: Dict[str, List[float]], products: List[str], short: int = 8, long: int = 36) -> float:
        ratios = []
        for product in products:
            series = history.get(product, [])
            if len(series) < long + 1:
                continue
            arr = np.array(series[-(long + 1) :], dtype=float)
            ret = np.abs(np.diff(arr))
            long_mean = ret.mean()
            if long_mean < 1e-9:
                continue
            ratios.append(ret[-short:].mean() / long_mean)
        return float(np.mean(ratios)) if ratios else 1.0

    def _market_stress(self, history: Dict[str, List[float]]) -> float:
        scores = []
        for product in self.STRESS_PRODUCTS:
            score = self._last_move_z(history.get(product, []), 20)
            if score is not None:
                scores.append(abs(score))
        return float(np.mean(scores)) if scores else 0.0

    def _spread_z(
        self,
        history: Dict[str, List[float]],
        legs: List[Tuple[str, float]],
        window: int,
    ) -> Optional[float]:
        if any(len(history.get(product, [])) < window for product, _ in legs):
            return None
        series = []
        for offset in range(-window, 0):
            value = 0.0
            for product, weight in legs:
                value += weight * math.log(history[product][offset])
            series.append(value)
        arr = np.array(series, dtype=float)
        sigma = arr.std()
        if sigma < 1e-9:
            return None
        return float((arr[-1] - arr.mean()) / sigma)

    def _group_residual_z(
        self,
        history: Dict[str, List[float]],
        product: str,
        group: List[str],
        window: int,
    ) -> Optional[float]:
        if any(len(history.get(name, [])) < window for name in group):
            return None
        others = [name for name in group if name != product]
        residuals = []
        for offset in range(-window, 0):
            group_mean = float(np.mean([history[name][offset] for name in others]))
            residuals.append(history[product][offset] - group_mean)
        arr = np.array(residuals, dtype=float)
        sigma = arr.std()
        if sigma < 1e-9:
            return None
        return float((arr[-1] - arr.mean()) / sigma)

    def _regression_residual_z(
        self,
        history: Dict[str, List[float]],
        product: str,
        peers: List[str],
        beta: List[float],
        intercept: float,
        window: int,
    ) -> Optional[float]:
        if len(beta) != len(peers):
            return None
        needed = [product] + peers
        if any(len(history.get(name, [])) < window for name in needed):
            return None
        residuals = []
        for offset in range(-window, 0):
            fair_value = intercept
            for coeff, peer in zip(beta, peers):
                fair_value += coeff * history[peer][offset]
            residuals.append(history[product][offset] - fair_value)
        arr = np.array(residuals, dtype=float)
        sigma = arr.std()
        if sigma < 1e-9:
            return None
        return float((arr[-1] - arr.mean()) / sigma)

    def _group_trend_score(
        self,
        history: Dict[str, List[float]],
        products: List[str],
        short_window: int,
        long_window: int,
    ) -> Optional[float]:
        if any(len(history.get(product, [])) < long_window for product in products):
            return None
        pack = []
        for offset in range(-long_window, 0):
            pack.append(float(np.mean([history[product][offset] for product in products])))
        return self._trend_score(pack, short_window, long_window)

    def _trend_score(self, series: List[float], short_window: int, long_window: int) -> Optional[float]:
        if len(series) < long_window:
            return None
        arr = np.array(series[-long_window:], dtype=float)
        ret_sigma = np.diff(arr).std()
        if ret_sigma < 1e-9:
            return None
        short_mean = arr[-short_window:].mean()
        long_mean = arr.mean()
        return float((short_mean - long_mean) / ret_sigma)

    def _last_move_z(self, series: List[float], window: int) -> Optional[float]:
        if len(series) < window + 1:
            return None
        arr = np.array(series[-(window + 1) :], dtype=float)
        ret = np.diff(arr)
        sigma = ret[:-1].std()
        if sigma < 1e-9:
            return None
        return float(ret[-1] / sigma)

    def _zscore(self, series: List[float], window: int) -> Optional[float]:
        if len(series) < window:
            return None
        arr = np.array(series[-window:], dtype=float)
        sigma = arr.std()
        if sigma < 1e-9:
            return None
        return float((arr[-1] - arr.mean()) / sigma)

    def _sticky_direction(self, previous: int, score: Optional[float], entry: float, exit: float) -> int:
        if score is None:
            return previous
        if score > entry:
            return 1
        if score < -entry:
            return -1
        if abs(score) < exit:
            return 0
        return previous

    def _signed_trade_flow(self, trades: List, mid: Optional[float], depth: Optional[OrderDepth]) -> float:
        if not trades or mid is None:
            return 0.0
        best_bid = max(depth.buy_orders) if depth and depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth and depth.sell_orders else None
        signed = 0.0
        for trade in trades:
            price = float(trade.price)
            qty = float(trade.quantity)
            if best_ask is not None and price >= best_ask:
                signed += qty
            elif best_bid is not None and price <= best_bid:
                signed -= qty
            elif price > mid:
                signed += qty
            elif price < mid:
                signed -= qty
        return signed

    def _passive_toward_target(self, product: str, target: int, position: int, depth: OrderDepth) -> List[Order]:
        need = target - position
        if need == 0:
            return []
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return []

        if need > 0:
            price = best_bid + 1 if best_bid + 1 < best_ask else best_bid
            return [Order(product, int(price), need)]
        price = best_ask - 1 if best_ask - 1 > best_bid else best_ask
        return [Order(product, int(price), need)]

    def _safety_mm(self, product: str, position: int, depth: OrderDepth, clip: int) -> List[Order]:
        best_bid = max(depth.buy_orders) if depth.buy_orders else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is None or best_ask is None:
            return []
        if best_ask - best_bid < 4:
            return []

        skew = int(position / 3)
        bid_price = best_bid + 1 - skew
        ask_price = best_ask - 1 - skew
        if bid_price >= ask_price:
            bid_price = best_bid
            ask_price = best_ask

        orders: List[Order] = []
        buy_qty = min(clip, self.LIMIT - position)
        sell_qty = min(clip, self.LIMIT + position)
        if buy_qty > 0:
            orders.append(Order(product, int(bid_price), buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, int(ask_price), -sell_qty))
        return orders

    def _mid(self, depth: OrderDepth) -> Optional[float]:
        if not depth.buy_orders or not depth.sell_orders:
            return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

    def _load_state(self, trader_data: str) -> dict:
        empty = {"history": {}, "sticky": {}, "prev_mids": {}}
        if not trader_data:
            return empty
        try:
            payload = json.loads(trader_data)
        except Exception:
            return empty
        if not isinstance(payload, dict):
            return empty
        payload.setdefault("history", {})
        payload.setdefault("sticky", {})
        payload.setdefault("prev_mids", {})
        return payload