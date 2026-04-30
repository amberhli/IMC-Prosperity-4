"""
Trading Data Analysis Script
=============================
Place this script in the same directory as your CSV files:
  prices_round_5_day_2.csv, prices_round_5_day_3.csv, prices_round_5_day_4.csv
  trades_round_5_day_2.csv,  trades_round_5_day_3.csv,  trades_round_5_day_4.csv

NOTE: CSVs use semicolon (;) as delimiter.

Run: python analyze_trading_data.py
Output: analysis_output.txt (paste this back to Claude)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ── Config ──────────────────────────────────────────────────────────────────
DAYS  = [2, 3, 4]
ROUND = 5
SEP   = ";"           # CSV delimiter

PRODUCT_GROUPS = {
    "Galaxy Sounds": [
        "GALAXY_SOUNDS_BLACK_HOLES",
        "GALAXY_SOUNDS_DARK_MATTER",
        "GALAXY_SOUNDS_PLANETARY_RINGS",
        "GALAXY_SOUNDS_SOLAR_FLAMES",
        "GALAXY_SOUNDS_SOLAR_WINDS",
    ],
    "Sleeping Pods": [
        "SLEEP_POD_COTTON",
        "SLEEP_POD_LAMB_WOOL",
        "SLEEP_POD_NYLON",
        "SLEEP_POD_POLYESTER",
        "SLEEP_POD_SUEDE",
    ],
    "Microchips": [
        "MICROCHIP_CIRCLE",
        "MICROCHIP_OVAL",
        "MICROCHIP_RECTANGLE",
        "MICROCHIP_SQUARE",
        "MICROCHIP_TRIANGLE",
    ],
    "Pebbles": [
        "PEBBLES_XS",
        "PEBBLES_S",
        "PEBBLES_M",
        "PEBBLES_L",
        "PEBBLES_XL",
    ],
    "Domestic Robotics": [
        "ROBOT_DISHES",
        "ROBOT_IRONING",
        "ROBOT_LAUNDRY",
        "ROBOT_MOPPING",
        "ROBOT_VACUUMING",
    ],
    "UV-Visors": [
        "UV_VISOR_AMBER",
        "UV_VISOR_MAGENTA",
        "UV_VISOR_ORANGE",
        "UV_VISOR_RED",
        "UV_VISOR_YELLOW",
    ],
    "Translators": [
        "TRANSLATOR_ASTRO_BLACK",
        "TRANSLATOR_ECLIPSE_CHARCOAL",
        "TRANSLATOR_GRAPHITE_MIST",
        "TRANSLATOR_SPACE_GRAY",
        "TRANSLATOR_VOID_BLUE",
    ],
    "Panels": [
        "PANEL_1X2",
        "PANEL_1X4",
        "PANEL_2X2",
        "PANEL_2X4",
        "PANEL_4X4",
    ],
    "Oxygen Shakes": [
        "OXYGEN_SHAKE_CHOCOLATE",
        "OXYGEN_SHAKE_EVENING_BREATH",
        "OXYGEN_SHAKE_GARLIC",
        "OXYGEN_SHAKE_MINT",
        "OXYGEN_SHAKE_MORNING_BREATH",
    ],
    "Snack Packs": [
        "SNACKPACK_CHOCOLATE",
        "SNACKPACK_PISTACHIO",
        "SNACKPACK_RASPBERRY",
        "SNACKPACK_STRAWBERRY",
        "SNACKPACK_VANILLA",
    ],
}

MID_PRICE_BASE = 10_000

out_lines = []

def log(*args):
    line = " ".join(str(a) for a in args)
    print(line)
    out_lines.append(line)

def section(title):
    log()
    log("=" * 70)
    log(f"  {title}")
    log("=" * 70)


# ── Load data ────────────────────────────────────────────────────────────────
def load_csvs():
    prices_frames, trades_frames = [], []
    for day in DAYS:
        pfile = f"prices_round_{ROUND}_day_{day}.csv"
        tfile = f"trades_round_{ROUND}_day_{day}.csv"
        if not Path(pfile).exists():
            log(f"[WARN] {pfile} not found – skipping")
            continue
        pdf = pd.read_csv(pfile, sep=SEP)
        pdf["day"] = day
        prices_frames.append(pdf)

        if Path(tfile).exists():
            tdf = pd.read_csv(tfile, sep=SEP)
            tdf["day"] = day
            trades_frames.append(tdf)
        else:
            log(f"[WARN] {tfile} not found")

    prices = pd.concat(prices_frames, ignore_index=True) if prices_frames else pd.DataFrame()
    trades = pd.concat(trades_frames, ignore_index=True) if trades_frames else pd.DataFrame()
    return prices, trades


# ── Mid-price analysis ───────────────────────────────────────────────────────
def analyze_prices(prices):
    section("MID-PRICE STATISTICS PER PRODUCT")

    if "mid_price" not in prices.columns:
        prices["mid_price"] = (prices["bid_price_1"] + prices["ask_price_1"]) / 2

    stats = prices.groupby("product")["mid_price"].agg(
        mean="mean", std="std", min="min", max="max",
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    )
    stats["range"]      = stats["max"] - stats["min"]
    stats["vol_pct"]    = (stats["std"] / MID_PRICE_BASE * 100).round(4)
    log(stats.round(4).to_string())

    section("BID-ASK SPREAD (ASK1 - BID1) PER PRODUCT")
    prices["spread"] = prices["ask_price_1"] - prices["bid_price_1"]
    spread_stats = prices.groupby("product")["spread"].agg(
        mean="mean", std="std", min="min", max="max")
    log(spread_stats.round(4).to_string())

    section("ORDER BOOK DEPTH (MEAN VOLUME AT EACH LEVEL)")
    depth_rows = []
    for side in ["bid", "ask"]:
        for lvl in [1, 2, 3]:
            vcol = f"{side}_volume_{lvl}"
            if vcol in prices.columns:
                grp = prices.dropna(subset=[vcol]).groupby("product")[vcol].mean().round(2)
                grp.name = vcol
                depth_rows.append(grp)
    if depth_rows:
        depth_df = pd.concat(depth_rows, axis=1)
        log(depth_df.to_string())

    section("MEAN REVERSION: DEVIATION FROM BASE (10000) + LAG-1 AUTOCORRELATION")
    prices["dev"] = prices["mid_price"] - MID_PRICE_BASE
    dev_stats = prices.groupby("product")["dev"].agg(mean="mean", std="std")
    dev_stats["autocorr_lag1"] = (
        prices.sort_values(["product", "day", "timestamp"])
              .groupby("product")["dev"]
              .apply(lambda x: x.autocorr(1))
    )
    log(dev_stats.round(4).to_string())

    section("PRICE TREND BY DAY (MEAN MID PER PRODUCT)")
    day_trend = prices.groupby(["product", "day"])["mid_price"].mean().unstack("day")
    log(day_trend.round(2).to_string())

    section("INTRADAY PRICE VOLATILITY (MEAN STD ACROSS 100-TICK BLOCKS)")
    if "timestamp" in prices.columns:
        prices["t_block"] = (prices["timestamp"] // 100) * 100
        intra = (prices.groupby(["product", "day", "t_block"])["mid_price"]
                       .std()
                       .groupby("product")
                       .agg(mean_block_std="mean", max_block_std="max"))
        log(intra.round(4).to_string())

    return prices


# ── Trade analysis ───────────────────────────────────────────────────────────
def analyze_trades(trades):
    section("TRADE PRICE STATISTICS PER PRODUCT")
    if trades.empty:
        log("[INFO] No trade data loaded.")
        return trades

    trade_stats = trades.groupby("symbol")["price"].agg(
        mean="mean", std="std", min="min", max="max", count="count")
    trade_stats["dev_from_base"] = (trade_stats["mean"] - MID_PRICE_BASE).round(2)
    log(trade_stats.round(2).to_string())

    section("TRADE VOLUME PER PRODUCT (TOTAL / MEAN / MAX PER TRADE)")
    vol_stats = trades.groupby("symbol")["quantity"].agg(
        total="sum", mean="mean", max="max")
    log(vol_stats.to_string())

    section("TRADE PRICE DEVIATION FROM BASE (10000)")
    trades["dev"] = trades["price"] - MID_PRICE_BASE
    log(trades.groupby("symbol")["dev"].agg(
        mean="mean", std="std", min="min", max="max").round(2).to_string())

    section("TRADE PRICE DEVIATION BY DAY (MEAN PER PRODUCT)")
    day_price = trades.groupby(["symbol", "day"])["price"].mean().unstack("day")
    log(day_price.round(2).to_string())

    section("TRADE ACTIVITY TIMELINE (TOTAL QUANTITY BY TIMESTAMP)")
    if "timestamp" in trades.columns:
        ts_activity = (trades.groupby(["symbol", "timestamp"])["quantity"]
                             .sum().unstack("timestamp").fillna(0).astype(int))
        log(ts_activity.to_string())

    return trades


# ── Cross-product correlation ────────────────────────────────────────────────
def correlations(prices):
    section("CROSS-PRODUCT MID-PRICE CORRELATIONS")
    if "timestamp" not in prices.columns:
        log("[SKIP] No timestamp column.")
        return

    pivot = prices.pivot_table(
        index=["day", "timestamp"], columns="product",
        values="mid_price", aggfunc="first")
    corr = pivot.corr().round(3)
    log(corr.to_string())

    log()
    log("  High-correlation pairs (|r| > 0.85) — potential pairs-trading candidates:")
    found = False
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            r = corr.iloc[i, j]
            if abs(r) > 0.85:
                log(f"    {cols[i]}  <->  {cols[j]}  r={r:.3f}")
                found = True
    if not found:
        log("    None above threshold.")


# ── Within-group spread analysis ─────────────────────────────────────────────
def within_group_spreads(prices):
    section("WITHIN-GROUP PRICE SPREADS (variant vs. group mean at each tick)")
    for group, products in PRODUCT_GROUPS.items():
        subset = prices[prices["product"].isin(products)].copy()
        if subset.empty:
            continue
        group_mean = subset.groupby(["day", "timestamp"])["mid_price"].mean()
        subset = subset.join(group_mean.rename("group_mean"), on=["day", "timestamp"])
        subset["vs_group"] = subset["mid_price"] - subset["group_mean"]
        spread = subset.groupby("product")["vs_group"].agg(
            mean="mean", std="std", min="min", max="max").round(3)
        log(f"\n  [{group}]")
        log(spread.to_string())


# ── Group-level summary ──────────────────────────────────────────────────────
def group_summary(prices):
    section("PRODUCT GROUP SUMMARY")
    for group, products in PRODUCT_GROUPS.items():
        subset = prices[prices["product"].isin(products)]
        if subset.empty:
            continue
        spread_mean = (subset["ask_price_1"] - subset["bid_price_1"]).mean()
        log(f"\n  [{group}]  products_found={subset['product'].nunique()}/5")
        log(f"    mid_price  mean={subset['mid_price'].mean():.2f}"
            f"  std={subset['mid_price'].std():.4f}"
            f"  range=[{subset['mid_price'].min():.0f}, {subset['mid_price'].max():.0f}]")
        log(f"    spread     mean={spread_mean:.2f}")


# ── Anomaly / arbitrage detection ────────────────────────────────────────────
def anomaly_check(prices, trades):
    section("POTENTIAL ARBITRAGE / ANOMALY SIGNALS")

    crossed = prices[prices["bid_price_1"] >= prices["ask_price_1"]]
    log(f"  Crossed-book rows (bid1 >= ask1): {len(crossed)}")
    if not crossed.empty:
        log(crossed[["day","timestamp","product","bid_price_1","ask_price_1"]].to_string())

    if not trades.empty and "timestamp" in trades.columns:
        merged = trades.merge(
            prices[["day","timestamp","product","bid_price_1","ask_price_1"]],
            left_on=["day","timestamp","symbol"],
            right_on=["day","timestamp","product"], how="left")
        outside = merged[
            (merged["price"] < merged["bid_price_1"]) |
            (merged["price"] > merged["ask_price_1"])]
        log(f"\n  Trades priced outside quoted spread: {len(outside)}")
        if not outside.empty:
            log(outside[["day","timestamp","symbol","price",
                         "bid_price_1","ask_price_1"]].head(30).to_string())

    threshold = 200
    outliers = prices[abs(prices["mid_price"] - MID_PRICE_BASE) > threshold]
    log(f"\n  Mid-price outliers (|dev| > {threshold}): {len(outliers)}")
    if not outliers.empty:
        log(outliers[["day","timestamp","product","mid_price"]].to_string())


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    log("Trading Data Analysis")
    log(f"  Round {ROUND}  |  Days {DAYS}  |  Separator: '{SEP}'")
    log(f"  Mid-price anchor: {MID_PRICE_BASE}")

    prices, trades = load_csvs()
    if prices.empty:
        log("[ERROR] No price data loaded. Check filenames and directory.")
        sys.exit(1)

    log(f"\n  Prices rows : {len(prices):,}")
    log(f"  Trades rows : {len(trades):,}")
    log(f"  Products    : {prices['product'].nunique()}")
    log(f"  Product list: {sorted(prices['product'].unique())}")

    prices = analyze_prices(prices)
    trades = analyze_trades(trades)
    correlations(prices)
    within_group_spreads(prices)
    group_summary(prices)
    anomaly_check(prices, trades)

    out_path = "analysis_output.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines))
    print(f"\n[DONE] Full output saved to: {out_path}")
    print("Paste analysis_output.txt back to Claude to design your trading algorithm.")

if __name__ == "__main__":
    main()