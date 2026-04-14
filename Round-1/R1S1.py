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

    def trade_pepper(self, order_depth: OrderDepth, current_position: int, memory: dict) -> List[Order]:
        """Credit Spread Strategy: Slow Intention + Dynamic Edge + Toxic Flow Protection"""
        orders: List[Order] = []
        product = 'INTARIAN_PEPPER_ROOT'
        limit = self.POSITION_LIMITS[product]
        
        # SAFEGUARD: Ignore empty order books to prevent max() crashes
        if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
            return orders
        
        raw_best_bid = max(order_depth.buy_orders.keys())
        raw_best_ask = min(order_depth.sell_orders.keys())
        
        # =================================================================
        # 1. DEEP BOOK FILTER: Find the "True" Market Spread
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

        true_mid_price = (true_best_bid + true_best_ask) / 2.0
        true_total_vol = true_best_bid_vol + true_best_ask_vol
        true_spread = true_best_ask - true_best_bid

        # =================================================================
        # 2. SLOW INTENTION MATH (The Credit View)
        # =================================================================
        # Calculate instantaneous imbalance
        current_obi = (true_best_bid_vol - true_best_ask_vol) / true_total_vol if true_total_vol > 0 else 0
        
        # Load the previous tick's sentiment from memory
        prev_obi_ema = memory.get('pepper_obi_ema', 0.0)
        
        # ALPHA = 0.1: It takes 10-20 ticks for sentiment to fully build. "No drama."
        obi_ema = (current_obi * 0.1) + (prev_obi_ema * 0.9)
        memory['pepper_obi_ema'] = obi_ema  # Save for next tick
        
        # Feature 1: Zero-Lag Micro-Price
        micro_price = (true_best_bid * true_best_ask_vol + true_best_ask * true_best_bid_vol) / true_total_vol if true_total_vol > 0 else true_mid_price
        
        # Feature 2: Violent Inventory Skew
        # Credit is illiquid. If we are stuck with max inventory, we must aggressively dump it.
        # We use a cubic function so it punishes heavily only when close to the limit.
        inv_fraction = current_position / limit
        inv_shift = math.copysign((abs(inv_fraction) ** 3) * 4.5, inv_fraction) 
        
        # The Final Slow-Moving Target Price
        target_price = micro_price + (obi_ema * 2.5) - inv_shift
        
        # =================================================================
        # 3. PHASE 1: TAKER (Only take massive mispricings)
        # =================================================================
        for ask, vol in sorted(order_depth.sell_orders.items()):
            if ask < target_price - 1.0:
                buy_vol = min(limit - current_position, abs(vol))
                if buy_vol > 0:
                    orders.append(Order(product, ask, buy_vol))
                    current_position += buy_vol
                    
        for bid, vol in sorted(order_depth.buy_orders.items(), reverse=True):
            if bid > target_price + 1.0:
                sell_vol = max(-limit - current_position, -vol)
                if sell_vol < 0:
                    orders.append(Order(product, bid, sell_vol))
                    current_position += sell_vol

        # =================================================================
        # 4. PHASE 2: MAKER (Risk-Managed Quoting)
        # =================================================================
        buy_room = limit - current_position
        sell_room = -limit - current_position
        
        # THE SPREAD IS THE SIGNAL: If market spread widens, our edge dynamically widens.
        base_edge = 1.0
        dynamic_edge = base_edge + max(0, (true_spread - 2) * 0.5)
        
        ideal_bid = math.floor(target_price - dynamic_edge)
        ideal_ask = math.ceil(target_price + dynamic_edge)
        
        # --- DRAWDOWN KILLER: TOXIC FLOW PROTECTION ---
        # If the slow-building intention is clearly leaning heavily in one direction, 
        # we completely shut off our quotes on the opposing side. Don't fight the trend.
        if obi_ema < -0.45:
            buy_room = 0  # Intention is heavily selling. Do not bid.
        if obi_ema > 0.45:
            sell_room = 0 # Intention is heavily buying. Do not ask.
        
        safe_best_bid = true_best_bid if true_best_bid_vol > 3 else true_best_bid - 1
        safe_best_ask = true_best_ask if true_best_ask_vol > 3 else true_best_ask + 1
        
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
