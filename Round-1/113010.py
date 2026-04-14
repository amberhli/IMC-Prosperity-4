import json
import math
from typing import List, Dict, Any
from datamodel import OrderDepth, TradingState, Order, Symbol

class Trader:
    
    def __init__(self):
        # Centralized Configuration for Round 1
        self.POSITION_LIMITS = {
            'ASH_COATED_OSMIUM': 20,
            'INTARIAN_PEPPER_ROOT': 20
        }

    def get_position(self, state: TradingState, product: Symbol) -> int:
        """Safely fetch current inventory."""
        return state.position.get(product, 0)

    def trade_osmium(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
        """Non-Linear Defensive Market Making Strategy"""
        orders: List[Order] = []
        product = 'ASH_COATED_OSMIUM'
        limit = self.POSITION_LIMITS[product]
        fair_value = 10000
        
        # SAFEGUARD: Ignore empty order books to prevent max() crashes
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return orders
        
        # Safely get best prices
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        
        # NON-LINEAR SKEW: Keeps us highly liquid in the center, and ultra-safe at the edges.
        inventory_fraction = current_position / limit
        inventory_shift = math.copysign((abs(inventory_fraction) ** 3) * 2.5, inventory_fraction)
        reservation_price = fair_value - inventory_shift
        
        # --- PHASE 1: TAKER ---
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < reservation_price:
                buy_vol = min(limit - current_position, abs(vol))
                if buy_vol > 0:
                    orders.append(Order(product, ask, buy_vol))
                    current_position += buy_vol
                    
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > reservation_price:
                sell_vol = max(-limit - current_position, -vol)
                if sell_vol < 0:
                    orders.append(Order(product, bid, sell_vol))
                    current_position += sell_vol

        # --- PHASE 2: MAKER ---
        buy_room = limit - current_position
        sell_room = -limit - current_position
        
        ideal_bid = math.floor(reservation_price - 1)
        ideal_ask = math.ceil(reservation_price + 1)
        
        my_bid = min(best_bid + 1, ideal_bid)
        my_ask = max(best_ask - 1, ideal_ask)

        if buy_room > 0:
            orders.append(Order(product, my_bid, buy_room))
        if sell_room < 0:
            orders.append(Order(product, my_ask, sell_room))
            
        return orders

    def trade_pepper(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
        """Anti-Spoofing + ML Ensemble Strategy"""
        orders: List[Order] = []
        product = 'INTARIAN_PEPPER_ROOT'
        limit = self.POSITION_LIMITS[product]
        
        # SAFEGUARD: Ignore empty order books to prevent max() crashes
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return orders
        
        # 1. Raw Level 1 Bests (These might be 1-volume bait traps!)
        raw_best_bid = max(order_depth.buy_orders.keys())
        raw_best_ask = min(order_depth.sell_orders.keys())
        
        # =================================================================
        # 2. DEEP BOOK FILTER: Find the "True" Market Prices
        # We ignore any price levels with less than 3 volume.
        # =================================================================
        DUST_THRESHOLD = 3
        
        true_best_bid, true_best_bid_vol = raw_best_bid, order_depth.buy_orders[raw_best_bid]
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if vol >= DUST_THRESHOLD:
                true_best_bid, true_best_bid_vol = bid, vol
                break
                
        true_best_ask, true_best_ask_vol = raw_best_ask, abs(order_depth.sell_orders[raw_best_ask])
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if abs(vol) >= DUST_THRESHOLD:
                true_best_ask, true_best_ask_vol = ask, abs(vol)
                break

        # =================================================================
        # 3. TRUE PRICING MATH: Un-manipulatable by spoofing
        # =================================================================
        true_mid_price = (true_best_bid + true_best_ask) / 2.0
        true_total_vol = true_best_bid_vol + true_best_ask_vol
        
        # Feature 1: True Zero-Lag Micro-Price
        micro_price = (true_best_bid * true_best_ask_vol + true_best_ask * true_best_bid_vol) / true_total_vol if true_total_vol > 0 else true_mid_price
        
        # Feature 2: True OBI (Order Book Imbalance)
        obi = (true_best_bid_vol - true_best_ask_vol) / true_total_vol if true_total_vol > 0 else 0
        obi_weight = 1.5 
        
        # Feature 3: Non-Linear Inventory Penalty
        inv_fraction = current_position / limit
        inv_shift = math.copysign((abs(inv_fraction) ** 2) * 4.0, inv_fraction) 
        
        # The Final Un-manipulated Target Price
        target_price = micro_price + (obi * obi_weight) - inv_shift
        
        # =================================================================
        # 4. PHASE 1: TAKER (Eat the bait if it's profitable!)
        # =================================================================
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < target_price - 0.5:
                buy_vol = min(limit - current_position, abs(vol))
                if buy_vol > 0:
                    orders.append(Order(product, ask, buy_vol))
                    current_position += buy_vol
                    
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > target_price + 0.5:
                sell_vol = max(-limit - current_position, -vol)
                if sell_vol < 0:
                    orders.append(Order(product, bid, sell_vol))
                    current_position += sell_vol

        # =================================================================
        # 5. PHASE 2: MAKER (Safe Quoting)
        # =================================================================
        buy_room = limit - current_position
        sell_room = -limit - current_position
        
        ideal_bid = math.floor(target_price - 1)
        ideal_ask = math.ceil(target_price + 1)
        
        # ML-Driven Dust Filtering (Using true volume!)
        safe_bid_vol_req = 5 if obi < -0.4 else 2
        safe_ask_vol_req = 5 if obi > 0.4 else 2
        
        safe_best_bid = true_best_bid if true_best_bid_vol > safe_bid_vol_req else true_best_bid - 1
        safe_best_ask = true_best_ask if true_best_ask_vol > safe_ask_vol_req else true_best_ask + 1
        
        # CRITICAL SAFEGUARD: 
        # min(..., raw_best_ask - 1) ensures our maker orders NEVER cross the 
        # raw spread, which would trigger an accidental market order fee.
        my_bid = min(safe_best_bid + 1, ideal_bid, raw_best_ask - 1)
        my_ask = max(safe_best_ask - 1, ideal_ask, raw_best_bid + 1)

        if buy_room > 0:
            orders.append(Order(product, my_bid, buy_room))
        if sell_room < 0:
            orders.append(Order(product, my_ask, sell_room))
            
        return orders

    def run(self, state: TradingState):
        """Main routing function. Keeps execution clean and manageable."""
        result = {}
        
        # --- STATE MEMORY DECOMPRESSION ---
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                pass
                
        # --- STRATEGY ROUTING ---
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            current_pos = self.get_position(state, product)
            
            # Route to the correct strategy
            if product == 'ASH_COATED_OSMIUM':
                result[product] = self.trade_osmium(order_depth, current_pos)
                
            elif product == 'INTARIAN_PEPPER_ROOT':
                result[product] = self.trade_pepper(order_depth, current_pos)

        # --- STATE MEMORY COMPRESSION ---
        serialized_state = json.dumps(trader_data)
        conversions = 0
        
        return result, conversions, serialized_state