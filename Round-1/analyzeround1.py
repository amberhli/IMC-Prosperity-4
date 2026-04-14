import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pathlib import Path

# 1. Setup the path to your ROUND 1 folder
home_dir = Path.home()
# UPDATE THIS to match your actual folder name (e.g., "ROUND_1")
data_folder = Path.cwd() / "data"

# Define the exact file paths for all 3 days
prices_d2_path = data_folder / "prices_round_1_day_-2.csv"
prices_d1_path = data_folder / "prices_round_1_day_-1.csv"
prices_d0_path = data_folder / "prices_round_1_day_0.csv"

print(f"Attempting to read data from: {data_folder}...\n")

# 2. Load the prices data
try:
    prices_d2 = pd.read_csv(prices_d2_path, sep=';')
    prices_d1 = pd.read_csv(prices_d1_path, sep=';')
    prices_d0 = pd.read_csv(prices_d0_path, sep=';')
    print("All 3 files loaded successfully!\n")
except FileNotFoundError as e:
    print(f"Error: Could not find the file. {e}")
    print("Please double-check that the folder and file names are correct.")
    exit()

# 3. Combine the datasets (Stitching 3 days continuously)
# Each day has 1,000,000 ticks, so we offset them sequentially
prices_d1['timestamp'] = prices_d1['timestamp'] + 1000000
prices_d0['timestamp'] = prices_d0['timestamp'] + 2000000
prices = pd.concat([prices_d2, prices_d1, prices_d0], ignore_index=True)

# 4. Separate the new assets
osmium = prices[prices['product'] == 'ASH_COATED_OSMIUM'].copy()
pepper = prices[prices['product'] == 'INTARIAN_PEPPER_ROOT'].copy()

# Filter out any ticks where the exchange sent empty mid_prices (0.0)
osmium = osmium[osmium['mid_price'] > 0]
pepper = pepper[pepper['mid_price'] > 0]

# 5. Print basic statistical summaries
print("--- ASH COATED OSMIUM STATS ---")
print(osmium['mid_price'].describe())
print("\n--- INTARIAN PEPPER ROOT STATS ---")
print(pepper['mid_price'].describe())

# 6. Plotting Ash Coated Osmium (Mean Reverting)
plt.figure(figsize=(14, 6))
plt.plot(osmium['timestamp'].values, osmium['mid_price'].values, label='Ash Coated Osmium', color='blue')
plt.title('Ash Coated Osmium Mid Price Over Time (Days -2, -1, 0)')
plt.xlabel('Timestamp (Continuous)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 7. Plotting Intarian Pepper Root (Trending / Volatile)
plt.figure(figsize=(14, 6))
plt.plot(pepper['timestamp'].values, pepper['mid_price'].values, label='Intarian Pepper Root', color='red')
plt.title('Intarian Pepper Root Mid Price Over Time (Days -2, -1, 0)')
plt.xlabel('Timestamp (Continuous)')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 8. Analyze Bid-Ask Spread for Pepper Root
# We use dropna() because occasionally the order book is entirely empty on one side
pepper['spread'] = (pepper['ask_price_1'] - pepper['bid_price_1'])
pepper_spreads = pepper.dropna(subset=['spread'])

plt.figure(figsize=(14, 6))
plt.plot(pepper_spreads['timestamp'].values, pepper_spreads['spread'].values, label='Pepper Root Spread', color='orange', alpha=0.7)
plt.title('Intarian Pepper Root Bid-Ask Spread Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Spread')
plt.legend()
plt.grid(True)
plt.show()
