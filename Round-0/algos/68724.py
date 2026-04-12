import json
import math
from typing import List, Dict, Any
from datamodel import OrderDepth, TradingState, Order, Symbol

class Trader:
    
    def __init__(self):
        # Centralized Configuration
        self.POSITION_LIMITS = {
            'EMERALDS': 20,
            'TOMATOES': 20
        }

    def get_position(self, state: TradingState, product: Symbol) -> int:
        """Safely fetch current inventory."""
        return state.position.get(product, 0)

    def trade_emeralds(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
        """Non-Linear Defensive Market Making Strategy"""
        orders: List[Order] = []
        product = 'EMERALDS'
        limit = self.POSITION_LIMITS[product]
        fair_value = 10000
        
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

    def trade_tomatoes(self, order_depth: OrderDepth, current_position: int) -> List[Order]:
        """Simulated ML Ensemble Strategy (Micro-Price + Continuous OBI + Skew)"""
        orders: List[Order] = []
        product = 'TOMATOES'
        limit = self.POSITION_LIMITS[product]
        
        best_bid = max(order_depth.buy_orders.keys())
        best_bid_vol = order_depth.buy_orders[best_bid]
        best_ask = min(order_depth.sell_orders.keys())
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        mid_price = (best_bid + best_ask) / 2.0
        
        # Feature 1: Zero-Lag Micro-Price
        total_vol = best_bid_vol + best_ask_vol
        micro_price = (best_bid * best_ask_vol + best_ask * best_bid_vol) / total_vol if total_vol > 0 else mid_price
        
        # Feature 2: OBI (Order Book Imbalance)
        obi = (best_bid_vol - best_ask_vol) / total_vol if total_vol > 0 else 0
        obi_weight = 1.5 
        
        # Feature 3: Non-Linear Inventory Penalty
        inv_fraction = current_position / limit
        inv_shift = math.copysign((abs(inv_fraction) ** 2) * 4.0, inv_fraction) 
        
        # The Final "ML" Fair Value Formula
        target_price = micro_price + (obi * obi_weight) - inv_shift
        
        # --- PHASE 1: TAKER ---
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

        # --- PHASE 2: MAKER ---
        buy_room = limit - current_position
        sell_room = -limit - current_position
        
        ideal_bid = math.floor(target_price - 1)
        ideal_ask = math.ceil(target_price + 1)
        
        # ML-Driven Dust Filtering
        safe_bid_vol_req = 5 if obi < -0.4 else 2
        safe_ask_vol_req = 5 if obi > 0.4 else 2
        
        safe_best_bid = best_bid if best_bid_vol > safe_bid_vol_req else best_bid - 1
        safe_best_ask = best_ask if best_ask_vol > safe_ask_vol_req else best_ask + 1
        
        my_bid = min(safe_best_bid + 1, ideal_bid, best_ask - 1)
        my_ask = max(safe_best_ask - 1, ideal_ask, best_bid + 1)

        if buy_room > 0:
            orders.append(Order(product, my_bid, buy_room))
        if sell_room < 0:
            orders.append(Order(product, my_ask, sell_room))
            
        return orders

    def run(self, state: TradingState):
        """Main routing function. Keeps execution clean and manageable."""
        result = {}
        
        # --- STATE MEMORY DECOMPRESSION ---
        # (Ready for Round 2/3 when you need to store SMA/EMA/Z-scores)
        trader_data = {}
        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                pass
                
        # --- STRATEGY ROUTING ---
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            
            # Skip empty order books to prevent crash
            if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                continue

            current_pos = self.get_position(state, product)
            
            # Route to the correct strategy
            if product == 'EMERALDS':
                result[product] = self.trade_emeralds(order_depth, current_pos)
                
            elif product == 'TOMATOES':
                result[product] = self.trade_tomatoes(order_depth, current_pos)

        # --- STATE MEMORY COMPRESSION ---
        serialized_state = json.dumps(trader_data)
        conversions = 0
        
        return result, conversions, serialized_state