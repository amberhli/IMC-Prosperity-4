import json
import pandas as pd
import io
import matplotlib.pyplot as plt
import sys
import re
import numpy as np

def analyze_log(log_filename):
    print(f"\n" + "="*50)
    print(f"  QUANTITATIVE BACKTEST ANALYSIS: {log_filename}")
    print(f"  " + "="*48 + "\n")
    
    try:
        with open(log_filename, 'r') as f:
            content = f.read()
            data = json.loads(content)
    except Exception as e:
        print(f"Error loading log file: {e}")
        return

    # 1. Parse the Activities Log (CSV data)
    csv_data = data.get('activitiesLog', '')
    if not csv_data:
        print("No activitiesLog found in the file.")
        return
        
    df = pd.read_csv(io.StringIO(csv_data), sep=';')
    
    # 2. Dynamically Identify All Traded Products
    unique_products = df['product'].unique()
    
    # 3. Extract Trades via Regex 
    buy_matches = re.findall(r'"buyer"\s*:\s*"SUBMISSION".*?"symbol"\s*:\s*"([A-Z_]+)"', content)
    sell_matches = re.findall(r'"seller"\s*:\s*"SUBMISSION".*?"symbol"\s*:\s*"([A-Z_]+)"', content)
    
    trade_counts = {}
    for prod in unique_products:
        trade_counts[prod] = {
            'buys': buy_matches.count(prod),
            'sells': sell_matches.count(prod)
        }

    # 4. Analyze PnL Tick-by-Tick
    plt.figure(figsize=(14, 8))
    
    # Generate a dynamic color palette based on how many products exist
    colors = plt.cm.tab10.colors 
    total_combined_pnl = 0
    
    for idx, product in enumerate(unique_products):
        pdf = df[df['product'] == product].copy()
        if pdf.empty:
            continue
            
        # Calculate PnL changes tick-to-tick
        pdf['pnl_change'] = pdf['profit_and_loss'].diff().fillna(0)
        
        # Calculate Drawdowns
        pdf['cummax'] = pdf['profit_and_loss'].cummax()
        pdf['drawdown'] = pdf['profit_and_loss'] - pdf['cummax']
        max_drawdown = pdf['drawdown'].min()
        
        # Win Rate calculation
        winning_ticks = len(pdf[pdf['pnl_change'] > 0])
        losing_ticks = len(pdf[pdf['pnl_change'] < 0])
        total_active_ticks = winning_ticks + losing_ticks
        win_rate = (winning_ticks / total_active_ticks * 100) if total_active_ticks > 0 else 0
        
        # Micro-Sharpe Ratio
        mean_return = pdf['pnl_change'].mean()
        std_return = pdf['pnl_change'].std()
        sharpe_proxy = (mean_return / std_return) if std_return > 0 else 0
        
        # Get final PnL
        final_pnl = pdf['profit_and_loss'].iloc[-1]
        total_combined_pnl += final_pnl
        
        # Print Stats Dashboard
        buys = trade_counts[product]['buys']
        sells = trade_counts[product]['sells']
        
        print(f"[{product} STATS]")
        print(f"  Final PnL:        {final_pnl:.2f} SeaShells")
        print(f"  Max Drawdown:     {max_drawdown:.2f} SeaShells")
        print(f"  Trades Executed:  {buys} Buys | {sells} Sells (Total: {buys + sells})")
        print(f"  Tick Win Rate:    {win_rate:.1f}% ({winning_ticks} W / {losing_ticks} L)")
        print(f"  Micro-Sharpe:     {sharpe_proxy:.4f}\n")
        
        # Assign a distinct color to this product
        color = colors[idx % len(colors)]
        
        # Plot PnL
        plt.plot(pdf['timestamp'].values, pdf['profit_and_loss'].values, label=f'{product} PnL', color=color, linewidth=2.5)
        
        # Fill the drawdown area to visualize risk
        plt.fill_between(pdf['timestamp'].values, pdf['profit_and_loss'].values, pdf['cummax'].values, color=color, alpha=0.1)

    print("==================================================")
    print(f"  TOTAL COMBINED PNL: {total_combined_pnl:.2f} SeaShells")
    print("==================================================\n")

    # 5. Finalize the Graph
    plt.title(f'Quantitative PnL & Risk Analysis - {log_filename}', fontsize=18, fontweight='bold')
    plt.xlabel('Timestamp (Ticks)', fontsize=12)
    plt.ylabel('Cumulative Profit & Loss (SeaShells)', fontsize=12)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    
    # Position legend dynamically to avoid covering data
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <your_log_file.json>")
    else:
        analyze_log(sys.argv[1])
