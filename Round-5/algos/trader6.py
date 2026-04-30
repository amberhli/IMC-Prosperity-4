"""
trader.py — IMC Prosperity Round 5 (v5 — backtest-driven patch)
================================================================
Patch notes vs v4 (22,024 profit):

FIX 1 — MICROCHIP_SQUARE removed from MM_PRODUCTS
  Root cause: MICROCHIP_SQUARE ticks with |move| > 15 on 52.6% of ticks
  (mean abs return 19.5 vs MICROCHIP_CIRCLE's 6.7). The MM accumulated
  long inventory during the ts=35000-44600 run-up (+596 pts) and then
  bled ~1,555 PnL as the price trended back down. Pure spread income
  cannot compensate for that inventory risk. Removed from MM. Now only
  traded via residual/sniper signals if triggered (currently none active
  for SQUARE, so it stays flat — losing only the -65 starting residual).

FIX 2 — Galaxy residual entry thresholds raised 1.65 → 2.10
  Root cause: GALAXY_SOUNDS_BLACK_HOLES entered too early, drew down
  -380 before recovering to +507 final. GALAXY_SOUNDS_SOLAR_FLAMES
  drew down -500 before recovering to +263. Higher z-score entry gives
  more confirmation before committing inventory. Also reduced size 2→1
  to limit max drawdown during adverse moves.

FIX 3 — OXYGEN_SHAKE_CHOCOLATE MR warmup guard
  Root cause: The mean-reversion signal triggered at ts≈7500 when the
  window (26 bars) had barely filled. Mid fell from 9324→9038 (trough
  -462 PnL). Added WARMUP_GUARD = 40 bars before any MR trade fires,
  allowing the rolling stats to stabilize. Window raised 26→32 for
  better z-score estimation. Entry threshold raised 1.70→1.90.

FIX 4 — Snackpack end-of-game unwind via EOG_FLAT_THRESHOLD
  Root cause: SNACKPACK_VANILLA held qty=8 at end of game. Pair signal
  was still active (z hadn't crossed exit threshold) but with zero time
  remaining, open inventory is pure mark-to-market risk. Added EOG
  (end-of-game) unwind: when ts >= EOG_FLAT_THRESHOLD (95,000), all
  snackpack & pebbles positions unwind aggressively to flat regardless
  of signal state.

FIX 5 — TRANSLATOR_ECLIPSE_CHARCOAL removed from MM_PRODUCTS
  Root cause: TRANSLATOR_ECLIPSE_CHARCOAL is simultaneously a MM product
  AND a SNIPER follower for TRANSLATOR_SPACE_GRAY. The SNIPER pushes it
  into directional positions (up to ±4 with risk_scale) while MM also
  tries to quote both sides. This caused the -850 trough at ts=37800
  during a trending TRANSLATOR move. Removing it from MM lets SNIPER
  manage it cleanly. TRANSLATOR_GRAPHITE_MIST and TRANSLATOR_VOID_BLUE
  (pure MM, no sniper overlap) remain.

FIX 6 — move_z guard in _imbalance_mm tightened 2.4 → 1.8
  The |z| > 2.4 guard was too permissive for volatile MM products.
  MICROCHIP_TRIANGLE (lost 471 from peak) and UV_VISOR_MAGENTA (lost 460)
  both showed large-move inventory bleed. Tightening to 1.8 means MM
  backs off sooner during fast-moving markets, reducing inventory risk
  at the cost of some spread income during volatile but mean-reverting ticks.
"""

from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional, Tuple
import json
import math
import numpy as np


class Trader:
    LIMIT         = 10
    HISTORY_LIMIT = 96
    PEBBLES_NAV   = 50_000.0
    SQRT2         = math.sqrt(2.0)

    # End-of-game unwind: aggressively flatten snackpack/pebbles after this ts
    EOG_FLAT_THRESHOLD = 95_000

    # ── Product groups ────────────────────────────────────────────────────
    SNACKPACK  = ["SNACKPACK_CHOCOLATE","SNACKPACK_PISTACHIO","SNACKPACK_RASPBERRY",
                  "SNACKPACK_STRAWBERRY","SNACKPACK_VANILLA"]
    PEBBLES    = ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L","PEBBLES_XL"]
    SLEEP_POD  = ["SLEEP_POD_COTTON","SLEEP_POD_LAMB_WOOL","SLEEP_POD_NYLON",
                  "SLEEP_POD_POLYESTER","SLEEP_POD_SUEDE"]
    GALAXY     = ["GALAXY_SOUNDS_BLACK_HOLES","GALAXY_SOUNDS_PLANETARY_RINGS",
                  "GALAXY_SOUNDS_SOLAR_WINDS","GALAXY_SOUNDS_DARK_MATTER",
                  "GALAXY_SOUNDS_SOLAR_FLAMES"]
    MICROCHIP  = ["MICROCHIP_CIRCLE","MICROCHIP_OVAL","MICROCHIP_RECTANGLE",
                  "MICROCHIP_SQUARE","MICROCHIP_TRIANGLE"]
    PANEL      = ["PANEL_1X2","PANEL_1X4","PANEL_2X2","PANEL_2X4","PANEL_4X4"]
    TRANSLATOR = ["TRANSLATOR_ASTRO_BLACK","TRANSLATOR_ECLIPSE_CHARCOAL",
                  "TRANSLATOR_GRAPHITE_MIST","TRANSLATOR_SPACE_GRAY","TRANSLATOR_VOID_BLUE"]
    UV         = ["UV_VISOR_AMBER","UV_VISOR_MAGENTA","UV_VISOR_ORANGE",
                  "UV_VISOR_RED","UV_VISOR_YELLOW"]
    SHAKE      = ["OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_EVENING_BREATH","OXYGEN_SHAKE_GARLIC",
                  "OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_MORNING_BREATH"]
    ROBOT      = ["ROBOT_DISHES","ROBOT_IRONING","ROBOT_LAUNDRY","ROBOT_MOPPING","ROBOT_VACUUMING"]

    CATEGORY_MAP = {
        **{p: "SNACKPACK"  for p in SNACKPACK},
        **{p: "PEBBLES"    for p in PEBBLES},
        **{p: "SLEEP"      for p in SLEEP_POD},
        **{p: "GALAXY"     for p in GALAXY},
        **{p: "MICROCHIP"  for p in MICROCHIP},
        **{p: "PANEL"      for p in PANEL},
        **{p: "TRANSLATOR" for p in TRANSLATOR},
        **{p: "UV"         for p in UV},
        **{p: "SHAKE"      for p in SHAKE},
        **{p: "ROBOT"      for p in ROBOT},
    }
    CATEGORY_PRODUCTS = {
        "SNACKPACK": SNACKPACK, "PEBBLES": PEBBLES, "SLEEP": SLEEP_POD,
        "GALAXY": GALAXY, "MICROCHIP": MICROCHIP, "PANEL": PANEL,
        "TRANSLATOR": TRANSLATOR, "UV": UV, "SHAKE": SHAKE, "ROBOT": ROBOT,
    }
    FRAGMENTED_CATEGORIES = {"SNACKPACK", "PEBBLES"}
    COHESIVE_CATEGORIES   = {"MICROCHIP", "TRANSLATOR", "UV"}

    # ── MM products ────────────────────────────────────────────────────────
    # FIX 1: Removed MICROCHIP_SQUARE (52.6% ticks with |move|>15; bleeds ~1555 PnL)
    # FIX 5: Removed TRANSLATOR_ECLIPSE_CHARCOAL (conflicts with SNIPER role; -850 trough)
    MM_PRODUCTS = {
        "MICROCHIP_CIRCLE",
        "MICROCHIP_TRIANGLE",
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_VOID_BLUE",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_MAGENTA",
        "PANEL_2X4",
        "ROBOT_VACUUMING",
    }

    # ── Sniper / advisor rules ────────────────────────────────────────────
    SNIPER_RULES = {
        "MICROCHIP_OVAL":         {"follower": "MICROCHIP_CIRCLE",           "sign":  1, "threshold": 3.8, "size": 2},
        "PANEL_1X2":              {"follower": "PANEL_2X4",                  "sign":  1, "threshold": 4.8, "size": 1},
        "TRANSLATOR_SPACE_GRAY":  {"follower": "TRANSLATOR_ECLIPSE_CHARCOAL","sign": -1, "threshold": 3.8, "size": 2},
    }
    UV_LEADER     = "UV_VISOR_YELLOW"
    UV_FOLLOWERS  = ["UV_VISOR_RED", "UV_VISOR_MAGENTA"]
    UV_THRESHOLD  = 5.0
    UV_SIZE       = 2

    # ── Lag rule ──────────────────────────────────────────────────────────
    LAG_RULES = {
        "ROBOT_LAUNDRY": {
            "follower": "ROBOT_MOPPING", "relation_sign": 1,
            "move_z": 1.55, "gap_z": 1.00, "flow_threshold": 6, "size": 1,
        },
    }

    # ── Pack NAV ──────────────────────────────────────────────────────────
    PACK_NAV_PRODUCTS = {
        "SLEEP_POD_COTTON":    {"group": SLEEP_POD, "window": 28, "entry": 1.75, "exit": 0.50, "size": 2},
        "SLEEP_POD_NYLON":     {"group": SLEEP_POD, "window": 28, "entry": 1.70, "exit": 0.50, "size": 2},
        "SLEEP_POD_POLYESTER": {"group": SLEEP_POD, "window": 28, "entry": 1.80, "exit": 0.50, "size": 2},
        "SLEEP_POD_SUEDE":     {"group": SLEEP_POD, "window": 28, "entry": 1.70, "exit": 0.50, "size": 2},
    }

    # ── Regression ────────────────────────────────────────────────────────
    REGRESSION_PRODUCTS = {
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

    # ── Mean reversion ────────────────────────────────────────────────────
    # FIX 3: window 26→32, entry 1.70→1.90, added WARMUP_GUARD per product
    MR_CONFIG = {
        "OXYGEN_SHAKE_CHOCOLATE": {
            "window": 32, "entry": 1.90, "exit": 0.35, "size": 2,
            "warmup": 40,  # bars to observe before first trade
        },
    }

    # ── Stress monitor ────────────────────────────────────────────────────
    STRESS_PRODUCTS = [
        "SNACKPACK_RASPBERRY","SNACKPACK_STRAWBERRY","SNACKPACK_CHOCOLATE","SNACKPACK_VANILLA",
        "PEBBLES_XL","PEBBLES_M","MICROCHIP_OVAL","PANEL_1X2",
        "ROBOT_LAUNDRY","TRANSLATOR_SPACE_GRAY","OXYGEN_SHAKE_EVENING_BREATH",
    ]

    # ── Galaxy residual ───────────────────────────────────────────────────
    # FIX 2: entry 1.65→2.10, size 2→1 to reduce trough drawdown
    GALAXY_RESIDUAL = {
        "GALAXY_SOUNDS_BLACK_HOLES":  {"group": GALAXY, "window": 20, "entry": 2.10, "exit": 0.45, "size": 1},
        "GALAXY_SOUNDS_SOLAR_FLAMES": {"group": GALAXY, "window": 20, "entry": 2.10, "exit": 0.45, "size": 1},
    }

    SNACK_WINDOW = 24

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN RUN LOOP
    # ═══════════════════════════════════════════════════════════════════════

    def run(self, state: TradingState):
        memory = self._load_state(state.traderData)
        history:   Dict[str, List[float]] = memory["history"]
        sticky:    dict                    = memory["sticky"]
        prev_mids: Dict[str, float]        = memory["prev_mids"]

        # Track tick count per product for warmup guards
        tick_counts: Dict[str, int] = memory.setdefault("tick_counts", {})

        # Current timestamp for EOG logic
        current_ts = state.timestamp

        # Update history
        mids: Dict[str, float] = {}
        for product, depth in state.order_depths.items():
            mid = self._mid(depth)
            if mid is None: continue
            mids[product] = mid
            series = history.get(product, [])
            series.append(mid)
            history[product] = series[-self.HISTORY_LIMIT:]
            tick_counts[product] = tick_counts.get(product, 0) + 1

        # Market regime
        market_stress = self._market_stress(history)
        allow_mm      = market_stress < 1.35
        category_corr = {
            "MICROCHIP":   self._group_avg_corr(history, self.MICROCHIP,   24),
            "TRANSLATOR":  self._group_avg_corr(history, self.TRANSLATOR,  24),
            "UV":          self._group_avg_corr(history, self.UV,          24),
        }
        other_scale = 1.0
        if market_stress > 2.00:   other_scale = 0.35
        elif market_stress > 1.45: other_scale = 0.65

        # End-of-game flag: flatten structural positions aggressively
        eog = (current_ts >= self.EOG_FLAT_THRESHOLD)

        # ── Signals ──────────────────────────────────────────────────────
        structural = {}
        if eog:
            # FIX 4: Force all snackpack/pebbles to flat at EOG
            for p in self.SNACKPACK + self.PEBBLES:
                structural[p] = 0
        else:
            structural.update(self._snackpack_targets(history, sticky))
            structural.update(self._pebbles_targets(history, mids, sticky))

        auxiliary = {}
        auxiliary.update(self._pack_nav_targets(history, sticky))
        auxiliary.update(self._regression_targets(history, sticky))
        auxiliary.update(self._mean_reversion_targets(history, sticky, tick_counts))
        auxiliary.update(self._galaxy_residual_targets(history, sticky))

        lag_nudges     = self._lagger_targets(state, history, mids)
        advisor_nudges = self._advisor_nudges(history, mids, prev_mids)

        # ── Merge targets ─────────────────────────────────────────────────
        final: Dict[str, int] = {}
        for p, t in structural.items():
            final[p] = int(np.clip(t, -self.LIMIT, self.LIMIT))
        for p, t in auxiliary.items():
            scaled = int(np.clip(round(t * other_scale), -self.LIMIT, self.LIMIT))
            if p not in final:
                final[p] = scaled
        for p, nudge in lag_nudges.items():
            if p in structural: continue
            base = final.get(p, 0)
            final[p] = int(np.clip(base + nudge, -self.LIMIT, self.LIMIT))
        for p, nudge in advisor_nudges.items():
            if p in structural: continue
            base = final.get(p, 0)
            final[p] = int(np.clip(base + nudge, -self.LIMIT, self.LIMIT))

        # ── Execution ────────────────────────────────────────────────────
        result: Dict[str, List[Order]] = {}
        for product, depth in state.order_depths.items():
            pos    = state.position.get(product, 0)
            target = final.get(product, 0)
            orders: List[Order] = []

            if product in final and (target != 0 or (eog and pos != 0)):
                # EOG: be more aggressive unwinding (cross the spread if needed)
                if eog and target == 0 and pos != 0:
                    orders = self._aggressive_toward_flat(product, pos, depth)
                else:
                    orders = self._passive_toward_target(product, target, pos, depth)

            elif product in self.MM_PRODUCTS and allow_mm:
                mm_clip = 1 if market_stress > 1.20 else 2
                widen   = 0
                cat     = self.CATEGORY_MAP.get(product)
                corr    = category_corr.get(cat) if cat in category_corr else None
                if corr is not None and corr > 0.02:
                    mm_clip = 1
                    widen   = 1
                orders = self._imbalance_mm(
                    product, pos, depth,
                    history.get(product, []),
                    base_clip  = mm_clip,
                    widen      = widen,
                    bias_score = self._mm_bias_score(product, history),
                )

            elif pos != 0:
                orders = self._passive_toward_target(product, 0, pos, depth)

            if orders:
                result[product] = orders

        memory["prev_mids"]   = mids
        memory["tick_counts"] = tick_counts
        trader_data = json.dumps(memory, separators=(",", ":"))
        if len(trader_data) > 49000:
            for p in history: history[p] = history[p][-72:]
            trader_data = json.dumps(memory, separators=(",", ":"))

        return result, 0, trader_data

    # ═══════════════════════════════════════════════════════════════════════
    # SIGNAL FUNCTIONS
    # ═══════════════════════════════════════════════════════════════════════

    def _snackpack_targets(self, history, sticky):
        state      = sticky.setdefault("snack", {"cv": 0, "rs": 0})
        targets    = {}
        risk_scale = self._risk_scale_group(history, self.SNACKPACK)

        for (a, b, key, entry, leg_min) in [
            ("SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA",    "cv", 1.30, 0.40),
            ("SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY", "rs", 1.25, 0.45),
        ]:
            z_a    = self._zscore(history.get(a, []), self.SNACK_WINDOW)
            z_b    = self._zscore(history.get(b, []), self.SNACK_WINDOW)
            sig    = self._pair_signal(z_a, z_b)
            dirn   = self._relative_pair_direction(state.get(key, 0), sig, z_a, z_b, entry, 0.35, leg_min)
            state[key] = dirn
            strength   = self._signal_multiplier(sig, entry)
            size       = int(np.clip(round(8 * risk_scale * strength), 4, 10))
            targets[a] = dirn * size
            targets[b] = -dirn * size

        rs_dir  = state.get("rs", 0)
        pist_z  = self._zscore(history.get("SNACKPACK_PISTACHIO", []), self.SNACK_WINDOW)
        straw_z = self._zscore(history.get("SNACKPACK_STRAWBERRY", []), self.SNACK_WINDOW)
        pist_t  = 0
        if rs_dir != 0 and pist_z is not None and straw_z is not None:
            rs_size = abs(targets.get("SNACKPACK_RASPBERRY", 8))
            if abs(pist_z) > 0.35 and pist_z * straw_z > 0:
                pist_t = int(math.copysign(max(2, rs_size // 2), -rs_dir))
        targets["SNACKPACK_PISTACHIO"] = pist_t
        return targets

    def _pebbles_targets(self, history, mids, sticky):
        if not all(p in mids for p in self.PEBBLES): return {}
        state      = sticky.setdefault("pebbles", {"xl": 0})
        targets    = {p: 0 for p in self.PEBBLES}
        risk_scale = self._risk_scale_group(history, self.PEBBLES)

        gap       = self.PEBBLES_NAV - sum(mids[p] for p in self.PEBBLES)
        xl_signal = self._spread_z(history,
            [("PEBBLES_XL",1.0),("PEBBLES_XS",-0.25),("PEBBLES_S",-0.25),
             ("PEBBLES_M",-0.25),("PEBBLES_L",-0.25)], 40)
        xl_dir    = self._symmetric_direction(state.get("xl", 0), xl_signal, 1.10, 0.30)
        state["xl"] = xl_dir

        other_scores = [(self._zscore(history.get(p, []), 36), p)
                        for p in ["PEBBLES_XS","PEBBLES_S","PEBBLES_M","PEBBLES_L"]
                        if self._zscore(history.get(p, []), 36) is not None]

        if xl_dir != 0 and len(other_scores) >= 2:
            hedge_count = 3 if abs(gap) > 4.5 else 2
            chosen = [p for _, p in (sorted(other_scores)[:hedge_count] if xl_dir < 0
                                     else sorted(other_scores, reverse=True)[:hedge_count])]
            strength   = self._signal_multiplier(xl_signal, 1.10)
            xl_size    = int(np.clip(round(4 * risk_scale * strength), 2, 6))
            hedge_size = int(np.clip(round(2 * risk_scale), 1, 3))
            if (xl_dir < 0 and gap > 3.0) or (xl_dir > 0 and gap < -3.0):
                xl_size    = min(6, xl_size + 1)
                hedge_size = min(3, hedge_size + 1)
            targets["PEBBLES_XL"] += xl_dir * xl_size
            for p in chosen: targets[p] += -xl_dir * hedge_size

        return {p: int(np.clip(t, -self.LIMIT, self.LIMIT)) for p, t in targets.items()}

    def _pack_nav_targets(self, history, sticky):
        state   = sticky.setdefault("pack_nav", {})
        targets = {}
        for product, cfg in self.PACK_NAV_PRODUCTS.items():
            score  = self._group_residual_z(history, product, cfg["group"], cfg["window"])
            prev   = state.get(product, 0)
            dirn   = self._sticky_direction(prev, -score if score is not None else None, cfg["entry"], cfg["exit"])
            state[product] = dirn
            strength   = self._signal_multiplier(score, cfg["entry"])
            risk_scale = self._risk_scale_product(history, product)
            size       = int(np.clip(round(cfg["size"] * strength * risk_scale), 1, 4))
            targets[product] = dirn * size
        return targets

    def _regression_targets(self, history, sticky):
        state   = sticky.setdefault("regression", {})
        targets = {}
        for product, cfg in self.REGRESSION_PRODUCTS.items():
            z    = self._regression_residual_z(history, product, cfg["peers"],
                                                cfg["beta"], cfg["intercept"], cfg["window"])
            prev = state.get(product, 0)
            dirn = self._sticky_direction(prev, -z if z is not None else None, cfg["entry"], cfg["exit"])
            state[product] = dirn
            strength   = self._signal_multiplier(z, cfg["entry"])
            risk_scale = self._risk_scale_product(history, product)
            size       = int(np.clip(round(cfg["size"] * strength * risk_scale), 1, 4))
            targets[product] = dirn * size
        return targets

    def _mean_reversion_targets(self, history, sticky, tick_counts):
        # FIX 3: warmup guard — don't trade until we have enough bars
        state   = sticky.setdefault("mr", {})
        targets = {}
        for product, cfg in self.MR_CONFIG.items():
            # Check warmup
            ticks = tick_counts.get(product, 0)
            warmup = cfg.get("warmup", 0)
            series = history.get(product, [])
            if ticks < warmup or len(series) < cfg["window"]:
                targets[product] = 0
                continue
            z    = self._zscore(series, cfg["window"])
            prev = state.get(product, 0)
            dirn = self._sticky_direction(prev, -z if z is not None else None, cfg["entry"], cfg["exit"])
            state[product] = dirn
            strength   = self._signal_multiplier(z, cfg["entry"])
            risk_scale = self._risk_scale_product(history, product)
            size       = int(np.clip(round(cfg["size"] * strength * risk_scale), 1, 4))
            targets[product] = dirn * size
        return targets

    def _galaxy_residual_targets(self, history, sticky):
        # FIX 2: higher entry threshold + smaller size
        state   = sticky.setdefault("galaxy_res", {})
        targets = {}
        for product, cfg in self.GALAXY_RESIDUAL.items():
            z    = self._group_residual_z(history, product, cfg["group"], cfg["window"])
            prev = state.get(product, 0)
            dirn = self._sticky_direction(prev, -z if z is not None else None, cfg["entry"], cfg["exit"])
            state[product] = dirn
            risk_scale = self._risk_scale_product(history, product)
            size       = int(np.clip(round(cfg["size"] * risk_scale), 1, 2))
            targets[product] = dirn * size
        return targets

    def _lagger_targets(self, state_trading, history, mids):
        nudges = {}
        for leader, cfg in self.LAG_RULES.items():
            follower = cfg["follower"]
            if leader not in mids or follower not in mids: continue
            lz = self._last_move_z(history.get(leader, []), 20)
            fz = self._last_move_z(history.get(follower, []), 20)
            if lz is None or fz is None: continue
            gap = cfg["relation_sign"] * lz - fz
            if abs(lz) < cfg["move_z"] or abs(gap) < cfg["gap_z"]: continue
            depth = state_trading.order_depths.get(leader)
            mid_l = mids.get(leader)
            trades = []
            if hasattr(state_trading, "market_trades") and state_trading.market_trades:
                trades = state_trading.market_trades.get(leader, [])
            flow = self._signed_trade_flow(trades, mid_l, depth)
            flow_sign = int(np.sign(flow)) if abs(flow) >= cfg["flow_threshold"] else 0
            leader_sign = 1 if lz > 0 else -1
            if flow_sign != 0 and flow_sign != leader_sign: continue
            dirn  = 1 if gap > 0 else -1
            strength = max(1.0, min(1.8, abs(gap) / cfg["gap_z"]))
            size  = int(np.clip(round(cfg["size"] * strength * self._risk_scale_product(history, follower)), 1, 3))
            nudges[follower] = nudges.get(follower, 0) + dirn * size
        return nudges

    def _advisor_nudges(self, history, mids, prev_mids):
        nudges: Dict[str, int] = {}

        for leader, cfg in self.SNIPER_RULES.items():
            follower = cfg["follower"]
            if not all(p in mids and p in prev_mids for p in [leader, follower]): continue
            group        = self.CATEGORY_PRODUCTS[self.CATEGORY_MAP[leader]]
            leader_abn   = self._abnormal_move_z(history, leader,   group, 20)
            follower_abn = self._abnormal_move_z(history, follower, group, 20)
            if leader_abn is None or follower_abn is None: continue
            if abs(leader_abn) < cfg["threshold"]: continue

            event_vote    = float(np.sign(leader_abn) * cfg["sign"] * min(1.5, abs(leader_abn) / cfg["threshold"]))
            trend_vote    = self._trend_vote(history.get(follower, []))
            reversion_vote= self._reversion_vote(history.get(follower, []))
            if np.sign(event_vote) == np.sign(reversion_vote) and abs(reversion_vote) > 0.8: continue
            if follower_abn * event_vote > 1.25: continue

            total_vote = 0.60 * event_vote + 0.25 * trend_vote + 0.15 * reversion_vote
            if abs(total_vote) < 0.85: continue

            risk_scale = self._risk_scale_product(history, follower)
            size = int(np.clip(round(cfg["size"] * min(1.8, abs(total_vote)) * risk_scale), 1, 4))
            nudges[follower] = nudges.get(follower, 0) + int(np.sign(total_vote)) * size

        if self.UV_LEADER in mids and self.UV_LEADER in prev_mids:
            leader_abn = self._abnormal_move_z(history, self.UV_LEADER, self.UV, 20)
            if leader_abn is not None and abs(leader_abn) >= self.UV_THRESHOLD:
                expected_sign = 1 if leader_abn > 0 else -1
                for follower in self.UV_FOLLOWERS:
                    if follower not in mids or follower not in prev_mids: continue
                    follower_abn = self._abnormal_move_z(history, follower, self.UV, 20)
                    if follower_abn is None or follower_abn * expected_sign > 1.0: continue
                    trend_vote = self._trend_vote(history.get(follower, []))
                    total_vote = 0.70 * expected_sign + 0.30 * trend_vote
                    risk_scale = self._risk_scale_product(history, follower)
                    size = int(np.clip(round(self.UV_SIZE * min(1.6, abs(total_vote)) * risk_scale), 1, 3))
                    nudges[follower] = nudges.get(follower, 0) + int(np.sign(total_vote)) * size

        return nudges

    # ═══════════════════════════════════════════════════════════════════════
    # EXECUTION
    # ═══════════════════════════════════════════════════════════════════════

    def _passive_toward_target(self, product, target, position, depth):
        need = target - position
        if need == 0: return []
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is None or best_ask is None: return []
        if need > 0:
            price = best_bid + 1 if best_bid + 1 < best_ask else best_bid
            return [Order(product, int(price), need)]
        price = best_ask - 1 if best_ask - 1 > best_bid else best_ask
        return [Order(product, int(price), need)]

    def _aggressive_toward_flat(self, product, position, depth):
        """
        FIX 4: EOG unwind — cross the spread to guarantee execution.
        Used at end-of-game when we must close structural positions.
        """
        if position == 0: return []
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is None or best_ask is None: return []
        if position > 0:
            # We're long, need to sell — hit the bid
            return [Order(product, int(best_bid), -position)]
        else:
            # We're short, need to buy — lift the ask
            return [Order(product, int(best_ask), -position)]

    def _imbalance_mm(self, product, position, depth, history, base_clip, widen=0, bias_score=0.0):
        """
        Book-imbalance-aware market making.
        FIX 6: move_z guard tightened 2.4 → 1.8 to back off sooner in volatile markets.
        """
        best_bid = max(depth.buy_orders)  if depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if best_bid is None or best_ask is None: return []
        spread = best_ask - best_bid
        if spread < 4: return []

        move_z = self._last_move_z(history, 18)
        # FIX 6: tighter guard (was 2.4)
        if move_z is not None and abs(move_z) > 1.8: return []

        bid_qty = abs(depth.buy_orders.get(best_bid, 0))
        ask_qty = abs(depth.sell_orders.get(best_ask, 0))
        if bid_qty + ask_qty <= 0: return []

        mid       = (best_bid + best_ask) / 2.0
        micro     = (best_ask * bid_qty + best_bid * ask_qty) / (bid_qty + ask_qty)
        imbalance = (bid_qty - ask_qty) / (bid_qty + ask_qty)

        bias = 0.0
        if micro > mid + 0.12 * spread or imbalance > 0.35:  bias =  1.0
        elif micro < mid - 0.12 * spread or imbalance < -0.35: bias = -1.0
        bias = float(np.clip(0.65 * bias + 0.35 * bias_score, -1.0, 1.0))

        inventory_skew = int(round(position / 4))
        bid_price = best_bid + 1 if best_bid + 1 < best_ask else best_bid
        ask_price = best_ask - 1 if best_ask - 1 > best_bid else best_ask
        if widen:
            bid_price = max(best_bid, bid_price - widen)
            ask_price = min(best_ask, ask_price + widen)
        bid_price -= max(0, inventory_skew)
        ask_price -= min(0, inventory_skew)
        if bid_price >= ask_price:
            bid_price = best_bid; ask_price = best_ask

        buy_clip = base_clip; sell_clip = base_clip
        if bias > 0.15:
            buy_clip += 1; sell_clip = max(1, sell_clip - 1)
        elif bias < -0.15:
            sell_clip += 1; buy_clip = max(1, buy_clip - 1)

        orders = []
        bq = min(buy_clip,  self.LIMIT - position)
        sq = min(sell_clip, self.LIMIT + position)
        if bq > 0: orders.append(Order(product, int(bid_price),  bq))
        if sq > 0: orders.append(Order(product, int(ask_price), -sq))
        return orders

    # ═══════════════════════════════════════════════════════════════════════
    # SIGNAL HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _mm_bias_score(self, product, history):
        series    = history.get(product, [])
        category  = self.CATEGORY_MAP.get(product)
        trend     = self._trend_vote(series)
        reversion = self._reversion_vote(series)
        if category in self.COHESIVE_CATEGORIES:
            return float(np.clip(0.65 * trend + 0.35 * reversion, -1.0, 1.0))
        if category in self.FRAGMENTED_CATEGORIES:
            return float(np.clip(0.20 * trend + 0.80 * reversion, -1.0, 1.0))
        return float(np.clip(0.45 * trend + 0.55 * reversion, -1.0, 1.0))

    def _trend_vote(self, series):
        score = self._ema_diff_score(series, fast=6, slow=18)
        return 0.0 if score is None else float(np.clip(score / 1.5, -1.0, 1.0))

    def _reversion_vote(self, series):
        rsi_score = self._rsi_vote(series, window=14)
        z         = self._zscore(series, 24)
        total     = 0.0
        if rsi_score is not None: total += rsi_score
        if z is not None:         total += float(np.clip(-z / 2.5, -1.0, 1.0))
        return float(np.clip(total / 2.0, -1.0, 1.0))

    def _signal_multiplier(self, signal, entry):
        if signal is None or entry <= 0: return 1.0
        return float(np.clip(abs(signal) / entry, 0.75, 1.75))

    def _risk_scale_product(self, history, product):
        series = history.get(product, [])
        if len(series) < 37: return 1.0
        arr        = np.array(series[-37:], dtype=float)
        ret        = np.abs(np.diff(arr))
        long_mean  = ret[:-8].mean() if len(ret) > 8 else ret.mean()
        short_mean = ret[-8:].mean()
        if long_mean < 1e-9: return 1.0
        return float(np.clip(1.15 / max(0.7, short_mean / long_mean), 0.55, 1.45))

    def _risk_scale_group(self, history, products):
        scales = [self._risk_scale_product(history, p) for p in products if p in history]
        return float(np.mean(scales)) if scales else 1.0

    def _pair_signal(self, z_a, z_b):
        if z_a is None or z_b is None: return None
        return float((z_a - z_b) / self.SQRT2)

    def _relative_pair_direction(self, previous, signal, z_a, z_b, entry, exit_, leg_min):
        if signal is None or z_a is None or z_b is None: return previous
        if signal > entry  and z_a > leg_min  and z_b < -leg_min: return -1
        if signal < -entry and z_a < -leg_min and z_b > leg_min:  return  1
        if abs(signal) < exit_: return 0
        return previous

    def _symmetric_direction(self, previous, signal, entry, exit_):
        if signal is None: return previous
        if signal > entry:  return -1
        if signal < -entry: return  1
        if abs(signal) < exit_: return 0
        return previous

    def _sticky_direction(self, previous, score, entry, exit_):
        if score is None: return previous
        if score > entry:  return  1
        if score < -entry: return -1
        if abs(score) < exit_: return 0
        return previous

    def _abnormal_move_z(self, history, product, group, window):
        product_move = self._last_move_z(history.get(product, []), window)
        if product_move is None: return None
        peers      = [n for n in group if n != product]
        peer_moves = [self._last_move_z(history.get(p, []), window)
                      for p in peers if self._last_move_z(history.get(p, []), window) is not None]
        if not peer_moves: return product_move
        return float(product_move - np.mean(peer_moves))

    # ── Statistical primitives ───────────────────────────────────────────

    def _zscore(self, series, window):
        if len(series) < window: return None
        arr   = np.array(series[-window:], dtype=float)
        sigma = arr.std()
        return float((arr[-1] - arr.mean()) / sigma) if sigma > 1e-9 else None

    def _last_move_z(self, series, window):
        if len(series) < window + 1: return None
        arr   = np.array(series[-(window+1):], dtype=float)
        ret   = np.diff(arr)
        sigma = ret[:-1].std()
        return float(ret[-1] / sigma) if sigma > 1e-9 else None

    def _group_residual_z(self, history, product, group, window):
        if any(len(history.get(n, [])) < window for n in group): return None
        others    = [n for n in group if n != product]
        residuals = [history[product][i] - float(np.mean([history[p][i] for p in others]))
                     for i in range(-window, 0)]
        arr   = np.array(residuals, dtype=float)
        sigma = arr.std()
        return float((arr[-1] - arr.mean()) / sigma) if sigma > 1e-9 else None

    def _regression_residual_z(self, history, product, peers, beta, intercept, window):
        needed = [product] + peers
        if any(len(history.get(n, [])) < window for n in needed): return None
        residuals = [history[product][i] - (intercept + sum(b * history[p][i] for b, p in zip(beta, peers)))
                     for i in range(-window, 0)]
        arr   = np.array(residuals, dtype=float)
        sigma = arr.std()
        return float((arr[-1] - arr.mean()) / sigma) if sigma > 1e-9 else None

    def _spread_z(self, history, legs, window):
        if any(len(history.get(p, [])) < window for p, _ in legs): return None
        series = [sum(w * math.log(history[p][i]) for p, w in legs) for i in range(-window, 0)]
        arr    = np.array(series, dtype=float)
        sigma  = arr.std()
        return float((arr[-1] - arr.mean()) / sigma) if sigma > 1e-9 else None

    def _group_avg_corr(self, history, products, window):
        rets = []
        for p in products:
            series = history.get(p, [])
            if len(series) < window + 1: return None
            rets.append(np.diff(np.array(series[-(window+1):], dtype=float)))
        values = [float(np.corrcoef(rets[i], rets[j])[0, 1])
                  for i in range(len(rets)) for j in range(i+1, len(rets))
                  if rets[i].std() > 1e-9 and rets[j].std() > 1e-9]
        return float(np.mean(values)) if values else None

    def _market_stress(self, history):
        scores = [abs(self._last_move_z(history.get(p, []), 20))
                  for p in self.STRESS_PRODUCTS
                  if self._last_move_z(history.get(p, []), 20) is not None]
        return float(np.mean(scores)) if scores else 0.0

    def _ema_diff_score(self, series, fast, slow):
        if len(series) < slow + 5: return None
        arr        = np.array(series[-(slow+10):], dtype=float)
        fa, sa     = 2.0/(fast+1), 2.0/(slow+1)
        ema_f = ema_s = arr[0]
        diffs  = []
        for v in arr:
            ema_f = fa*v + (1-fa)*ema_f
            ema_s = sa*v + (1-sa)*ema_s
            diffs.append(ema_f - ema_s)
        diff_arr = np.array(diffs[-slow:], dtype=float)
        sigma    = np.diff(arr).std()
        return float(diff_arr[-1] / sigma) if sigma > 1e-9 else None

    def _rsi_vote(self, series, window):
        if len(series) < window + 1: return None
        arr   = np.array(series[-(window+1):], dtype=float)
        delta = np.diff(arr)
        avg_up   = np.maximum(delta, 0).mean()
        avg_down = np.maximum(-delta, 0).mean()
        if avg_up < 1e-9 and avg_down < 1e-9: return 0.0
        rsi = 100.0 if avg_down < 1e-9 else 100.0 - 100.0 / (1.0 + avg_up/avg_down)
        return float(np.clip((50.0 - rsi) / 25.0, -1.0, 1.0))

    def _signed_trade_flow(self, trades, mid, depth):
        if not trades or mid is None: return 0.0
        best_bid = max(depth.buy_orders)  if depth and depth.buy_orders  else None
        best_ask = min(depth.sell_orders) if depth and depth.sell_orders else None
        signed   = 0.0
        for t in trades:
            price, qty = float(t.price), float(t.quantity)
            if best_ask is not None and price >= best_ask:   signed += qty
            elif best_bid is not None and price <= best_bid: signed -= qty
            elif price > mid: signed += qty
            elif price < mid: signed -= qty
        return signed

    def _mid(self, depth):
        if not depth.buy_orders or not depth.sell_orders: return None
        return (max(depth.buy_orders) + min(depth.sell_orders)) / 2.0

    def _load_state(self, trader_data):
        empty = {"history": {}, "sticky": {}, "prev_mids": {}, "tick_counts": {}}
        if not trader_data: return empty
        try:
            payload = json.loads(trader_data)
        except Exception:
            return empty
        if not isinstance(payload, dict): return empty
        payload.setdefault("history", {})
        payload.setdefault("sticky", {})
        payload.setdefault("prev_mids", {})
        payload.setdefault("tick_counts", {})
        return payload