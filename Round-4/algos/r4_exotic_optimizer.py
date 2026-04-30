import math
from dataclasses import dataclass

import numpy as np


CONTRACT_SIZE = 3000
ANNUAL_VOL = 2.51
STEPS_PER_DAY = 4
TRADING_DAYS_PER_YEAR = 252
SPOT = 50.0

TWO_WEEK_DAYS = 10
THREE_WEEK_DAYS = 15
TWO_WEEK_STEPS = TWO_WEEK_DAYS * STEPS_PER_DAY
THREE_WEEK_STEPS = THREE_WEEK_DAYS * STEPS_PER_DAY


@dataclass(frozen=True)
class Instrument:
    name: str
    bid: float
    ask: float
    volume: int
    kind: str
    strike: float | None = None
    expiry_steps: int | None = None
    barrier: float | None = None
    binary_payout: float | None = None


INSTRUMENTS = [
    Instrument("AC", 49.975, 50.025, 200, "spot"),
    Instrument("AC_50_P", 12.00, 12.05, 50, "put", strike=50.0, expiry_steps=THREE_WEEK_STEPS),
    Instrument("AC_50_C", 12.00, 12.05, 50, "call", strike=50.0, expiry_steps=THREE_WEEK_STEPS),
    Instrument("AC_35_P", 4.33, 4.35, 50, "put", strike=35.0, expiry_steps=THREE_WEEK_STEPS),
    Instrument("AC_40_P", 6.50, 6.55, 50, "put", strike=40.0, expiry_steps=THREE_WEEK_STEPS),
    Instrument("AC_45_P", 9.05, 9.10, 50, "put", strike=45.0, expiry_steps=THREE_WEEK_STEPS),
    Instrument("AC_60_C", 8.80, 8.85, 50, "call", strike=60.0, expiry_steps=THREE_WEEK_STEPS),
    Instrument("AC_50_P_2", 9.70, 9.75, 50, "put", strike=50.0, expiry_steps=TWO_WEEK_STEPS),
    Instrument("AC_50_C_2", 9.70, 9.75, 50, "call", strike=50.0, expiry_steps=TWO_WEEK_STEPS),
    Instrument("AC_50_CO", 22.20, 22.30, 50, "chooser", strike=50.0, expiry_steps=THREE_WEEK_STEPS),
    Instrument("AC_40_BP", 5.00, 5.10, 50, "binary_put", strike=40.0, expiry_steps=THREE_WEEK_STEPS, binary_payout=10.0),
    Instrument("AC_45_KO", 0.15, 0.175, 500, "ko_put", strike=45.0, expiry_steps=THREE_WEEK_STEPS, barrier=35.0),
]


def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call_value(spot: float, strike: float, sigma: float, steps: int) -> float:
    time_years = steps / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)
    vol_sqrt_t = sigma * math.sqrt(time_years)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * time_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(spot * norm_cdf(d1) - strike * norm_cdf(d2))


def bs_put_value(spot: float, strike: float, sigma: float, steps: int) -> float:
    call = bs_call_value(spot, strike, sigma, steps)
    return float(call - spot + strike)


def bs_cash_or_nothing_put_value(
    spot: float,
    strike: float,
    sigma: float,
    steps: int,
    payout: float,
) -> float:
    time_years = steps / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)
    vol_sqrt_t = sigma * math.sqrt(time_years)
    d1 = (math.log(spot / strike) + 0.5 * sigma * sigma * time_years) / vol_sqrt_t
    d2 = d1 - vol_sqrt_t
    return float(payout * norm_cdf(-d2))


def chooser_payoff(paths: np.ndarray, strike: float) -> np.ndarray:
    choice_prices = paths[:, TWO_WEEK_STEPS]
    final_prices = paths[:, THREE_WEEK_STEPS]
    is_call = choice_prices >= strike
    call_payoff = np.maximum(final_prices - strike, 0.0)
    put_payoff = np.maximum(strike - final_prices, 0.0)
    return np.where(is_call, call_payoff, put_payoff)


def ko_put_payoff(paths: np.ndarray, strike: float, barrier: float) -> np.ndarray:
    final_put = np.maximum(strike - paths[:, THREE_WEEK_STEPS], 0.0)
    knocked_out = np.any(paths[:, 1 : THREE_WEEK_STEPS + 1] < barrier, axis=1)
    return np.where(knocked_out, 0.0, final_put)


def simulate_path_dependent_values(
    n_paths: int = 2_000_000,
    batch_size: int = 200_000,
    seed: int = 7,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)
    drift = -0.5 * ANNUAL_VOL * ANNUAL_VOL * dt
    diffusion = ANNUAL_VOL * math.sqrt(dt)

    chooser_sum = 0.0
    ko_sum = 0.0
    done = 0

    while done < n_paths:
        this_batch = min(batch_size, n_paths - done)
        half = (this_batch + 1) // 2
        z = rng.standard_normal((half, THREE_WEEK_STEPS))
        shocks = np.vstack([z, -z])[:this_batch]
        log_returns = drift + diffusion * shocks
        log_paths = np.cumsum(log_returns, axis=1)
        paths = np.concatenate(
            [np.full((this_batch, 1), SPOT), SPOT * np.exp(log_paths)],
            axis=1,
        )

        chooser_sum += chooser_payoff(paths, 50.0).sum()
        ko_sum += ko_put_payoff(paths, 45.0, 35.0).sum()
        done += this_batch

    return {
        "AC_50_CO": chooser_sum / n_paths,
        "AC_45_KO": ko_sum / n_paths,
    }


def fair_value(instrument: Instrument, path_dependent_values: dict[str, float]) -> float:
    if instrument.kind == "spot":
        return SPOT
    if instrument.kind == "call":
        return bs_call_value(SPOT, instrument.strike, ANNUAL_VOL, instrument.expiry_steps)
    if instrument.kind == "put":
        return bs_put_value(SPOT, instrument.strike, ANNUAL_VOL, instrument.expiry_steps)
    if instrument.kind == "binary_put":
        return bs_cash_or_nothing_put_value(
            SPOT,
            instrument.strike,
            ANNUAL_VOL,
            instrument.expiry_steps,
            instrument.binary_payout,
        )
    if instrument.kind in {"chooser", "ko_put"}:
        return path_dependent_values[instrument.name]
    raise ValueError(f"Unsupported kind: {instrument.kind}")


def best_trade(instrument: Instrument, fair: float) -> tuple[str, int, float]:
    buy_edge = fair - instrument.ask
    sell_edge = instrument.bid - fair
    if buy_edge <= 0 and sell_edge <= 0:
        return ("hold", 0, 0.0)
    if buy_edge >= sell_edge:
        expected_pnl = buy_edge * CONTRACT_SIZE * instrument.volume
        return ("buy", instrument.volume, expected_pnl)
    expected_pnl = sell_edge * CONTRACT_SIZE * instrument.volume
    return ("sell", instrument.volume, expected_pnl)


def main() -> None:
    path_dependent_values = simulate_path_dependent_values()

    total_expected_pnl = 0.0
    print("Instrument    FairValue   Bid      Ask      EdgeSide  Volume   ExpPnL")
    print("-" * 72)
    for instrument in INSTRUMENTS:
        fair = fair_value(instrument, path_dependent_values)
        side, volume, expected_pnl = best_trade(instrument, fair)
        total_expected_pnl += expected_pnl
        print(
            f"{instrument.name:12} "
            f"{fair:9.5f} "
            f"{instrument.bid:8.3f} "
            f"{instrument.ask:8.3f} "
            f"{side:8} "
            f"{volume:6d} "
            f"{expected_pnl:9.2f}"
        )

    print("-" * 72)
    print(f"Total expected PnL at max edge sizes: {total_expected_pnl:.2f}")


if __name__ == "__main__":
    main()
