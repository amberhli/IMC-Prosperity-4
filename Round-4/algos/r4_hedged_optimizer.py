import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from r4_exotic_optimizer import (
    ANNUAL_VOL,
    CONTRACT_SIZE,
    INSTRUMENTS,
    SPOT,
    STEPS_PER_DAY,
    THREE_WEEK_STEPS,
    TRADING_DAYS_PER_YEAR,
    TWO_WEEK_STEPS,
    chooser_payoff,
    ko_put_payoff,
)


@dataclass
class PortfolioSummary:
    name: str
    positions: np.ndarray
    expected_pnl: float
    pnl_std: float
    pnl_p05: float
    pnl_p01: float
    pnl_min: float


def simulate_paths(n_paths: int = 300_000, seed: int = 17) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dt = 1.0 / (TRADING_DAYS_PER_YEAR * STEPS_PER_DAY)
    drift = -0.5 * ANNUAL_VOL * ANNUAL_VOL * dt
    diffusion = ANNUAL_VOL * math.sqrt(dt)

    half = (n_paths + 1) // 2
    z = rng.standard_normal((half, THREE_WEEK_STEPS))
    shocks = np.vstack([z, -z])[:n_paths]
    log_returns = drift + diffusion * shocks
    log_paths = np.cumsum(log_returns, axis=1)
    return np.concatenate(
        [np.full((n_paths, 1), SPOT), SPOT * np.exp(log_paths)],
        axis=1,
    )


def payoff_matrix(paths: np.ndarray) -> pd.DataFrame:
    final_2w = paths[:, TWO_WEEK_STEPS]
    final_3w = paths[:, THREE_WEEK_STEPS]
    data: dict[str, np.ndarray] = {
        "AC": final_3w,
        "AC_50_P": np.maximum(50.0 - final_3w, 0.0),
        "AC_50_C": np.maximum(final_3w - 50.0, 0.0),
        "AC_35_P": np.maximum(35.0 - final_3w, 0.0),
        "AC_40_P": np.maximum(40.0 - final_3w, 0.0),
        "AC_45_P": np.maximum(45.0 - final_3w, 0.0),
        "AC_60_C": np.maximum(final_3w - 60.0, 0.0),
        "AC_50_P_2": np.maximum(50.0 - final_2w, 0.0),
        "AC_50_C_2": np.maximum(final_2w - 50.0, 0.0),
        "AC_50_CO": chooser_payoff(paths, 50.0),
        "AC_40_BP": np.where(final_3w < 40.0, 10.0, 0.0),
        "AC_45_KO": ko_put_payoff(paths, 45.0, 35.0),
    }
    return pd.DataFrame(data, columns=[inst.name for inst in INSTRUMENTS])


def unit_pnl_matrix(payoffs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    payoff_np = payoffs.to_numpy()
    long_units = np.column_stack(
        [(payoff_np[:, idx] - inst.ask) * CONTRACT_SIZE for idx, inst in enumerate(INSTRUMENTS)]
    )
    short_units = np.column_stack(
        [(inst.bid - payoff_np[:, idx]) * CONTRACT_SIZE for idx, inst in enumerate(INSTRUMENTS)]
    )
    unit_matrix = np.column_stack([long_units, short_units])
    upper_bounds = np.array([inst.volume for inst in INSTRUMENTS] * 2, dtype=float)
    return unit_matrix, long_units, short_units


def holdings_to_positions(holdings: np.ndarray) -> np.ndarray:
    n = len(INSTRUMENTS)
    return holdings[:n] - holdings[n:]


def portfolio_stats(unit_matrix: np.ndarray, holdings: np.ndarray, name: str) -> PortfolioSummary:
    total_pnl = unit_matrix @ holdings
    return PortfolioSummary(
        name=name,
        positions=holdings_to_positions(holdings),
        expected_pnl=float(total_pnl.mean()),
        pnl_std=float(total_pnl.std(ddof=1)),
        pnl_p05=float(np.quantile(total_pnl, 0.05)),
        pnl_p01=float(np.quantile(total_pnl, 0.01)),
        pnl_min=float(total_pnl.min()),
    )


def edge_extreme_holdings(payoffs: pd.DataFrame) -> np.ndarray:
    means = payoffs.mean().to_numpy()
    holdings = np.zeros(len(INSTRUMENTS) * 2)
    for idx, inst in enumerate(INSTRUMENTS):
        long_edge = means[idx] - inst.ask
        short_edge = inst.bid - means[idx]
        if max(long_edge, short_edge) <= 0:
            continue
        if long_edge >= short_edge:
            holdings[idx] = inst.volume
        else:
            holdings[idx + len(INSTRUMENTS)] = inst.volume
    return holdings


def optimize_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    upper_bounds: np.ndarray,
    risk_aversion: float,
    start: np.ndarray | None = None,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> np.ndarray:
    if start is None:
        w = np.zeros_like(mu)
    else:
        w = np.clip(start.astype(float).copy(), 0.0, upper_bounds)

    if risk_aversion <= 0:
        return np.where(mu > 0.0, upper_bounds, 0.0)

    diag = np.diag(sigma)
    for _ in range(max_iter):
        max_change = 0.0
        sigma_w = sigma @ w
        for j in range(len(w)):
            if diag[j] <= 0:
                new_w = upper_bounds[j] if mu[j] > 0 else 0.0
            else:
                numer = mu[j] - risk_aversion * (sigma_w[j] - diag[j] * w[j])
                denom = risk_aversion * diag[j]
                new_w = np.clip(numer / denom, 0.0, upper_bounds[j])
            delta = new_w - w[j]
            if delta != 0.0:
                w[j] = new_w
                sigma_w += sigma[:, j] * delta
                max_change = max(max_change, abs(delta))
        if max_change < tol:
            break
    return w


def round_holdings(raw: np.ndarray, upper_bounds: np.ndarray) -> np.ndarray:
    rounded = np.round(raw).astype(float)
    return np.clip(rounded, 0.0, upper_bounds)


def greedy_local_improvement(
    unit_matrix: np.ndarray,
    holdings: np.ndarray,
    upper_bounds: np.ndarray,
    risk_aversion: float,
    max_passes: int = 5,
) -> np.ndarray:
    current = holdings.copy()
    mu = unit_matrix.mean(axis=0)
    sigma = np.cov(unit_matrix, rowvar=False, ddof=1)
    best = mean_variance_objective(mu, sigma, current, risk_aversion)
    for _ in range(max_passes):
        improved = False
        for j in range(len(current)):
            for step in (-1.0, 1.0):
                candidate = current.copy()
                candidate[j] = np.clip(candidate[j] + step, 0.0, upper_bounds[j])
                value = mean_variance_objective(mu, sigma, candidate, risk_aversion)
                if value > best:
                    current = candidate
                    best = value
                    improved = True
        if not improved:
            break
    return current


def mean_variance_objective(mu: np.ndarray, sigma: np.ndarray, holdings: np.ndarray, risk_aversion: float) -> float:
    return float(holdings @ mu - 0.5 * risk_aversion * holdings @ sigma @ holdings)


def delta_estimates(payoffs: pd.DataFrame, bump: float = 0.25) -> pd.Series:
    base_paths = simulate_paths(n_paths=120_000, seed=23)
    up_paths = base_paths.copy()
    down_paths = base_paths.copy()
    scale_up = (SPOT + bump) / SPOT
    scale_down = (SPOT - bump) / SPOT
    up_paths[:, :] *= scale_up
    down_paths[:, :] *= scale_down
    up_payoffs = payoff_matrix(up_paths).mean()
    down_payoffs = payoff_matrix(down_paths).mean()
    return (up_payoffs - down_payoffs) / (2.0 * bump)


def format_positions(positions: np.ndarray) -> pd.DataFrame:
    rows = []
    for inst, pos in zip(INSTRUMENTS, positions):
        side = "hold"
        volume = 0
        if pos > 0:
            side = "buy"
            volume = int(pos)
        elif pos < 0:
            side = "sell"
            volume = int(-pos)
        rows.append({"Instrument": inst.name, "Side": side, "Volume": volume})
    return pd.DataFrame(rows)


def main() -> None:
    payoffs = payoff_matrix(simulate_paths())
    unit_matrix, _, _ = unit_pnl_matrix(payoffs)
    mu = unit_matrix.mean(axis=0)
    sigma = np.cov(unit_matrix, rowvar=False, ddof=1)
    upper_bounds = np.array([inst.volume for inst in INSTRUMENTS] * 2, dtype=float)

    pure_edge_holdings = edge_extreme_holdings(payoffs)
    pure_edge = portfolio_stats(unit_matrix, pure_edge_holdings, "pure_edge")

    candidate_summaries: list[PortfolioSummary] = [pure_edge]
    candidate_positions: dict[str, np.ndarray] = {"pure_edge": pure_edge.positions}

    for risk_aversion in [1e-8, 2e-8, 5e-8, 1e-7, 2e-7, 5e-7, 1e-6, 2e-6]:
        raw = optimize_mean_variance(mu, sigma, upper_bounds, risk_aversion, start=pure_edge_holdings)
        rounded = round_holdings(raw, upper_bounds)
        improved = greedy_local_improvement(unit_matrix, rounded, upper_bounds, risk_aversion)
        name = f"lambda={risk_aversion:.0e}"
        summary = portfolio_stats(unit_matrix, improved, name)
        candidate_summaries.append(summary)
        candidate_positions[name] = summary.positions

    summary_df = pd.DataFrame(
        {
            "name": [s.name for s in candidate_summaries],
            "expected_pnl": [s.expected_pnl for s in candidate_summaries],
            "pnl_std": [s.pnl_std for s in candidate_summaries],
            "p05": [s.pnl_p05 for s in candidate_summaries],
            "p01": [s.pnl_p01 for s in candidate_summaries],
            "min": [s.pnl_min for s in candidate_summaries],
        }
    ).sort_values(["expected_pnl", "p05"], ascending=[False, False])

    pure_edge_expected = pure_edge.expected_pnl
    feasible = [
        s
        for s in candidate_summaries
        if s.expected_pnl >= 0.9 * pure_edge_expected and s.pnl_std <= 0.8 * pure_edge.pnl_std
    ]
    balanced = min(
        feasible if feasible else candidate_summaries,
        key=lambda s: (s.pnl_std, -s.expected_pnl),
    )

    deltas = delta_estimates(payoffs)
    exposure_rows = []
    for summary in [pure_edge, balanced]:
        exposure_rows.append(
            {
                "portfolio": summary.name,
                "delta": float(np.dot(summary.positions, deltas.to_numpy())),
                "expected_pnl": summary.expected_pnl,
                "pnl_std": summary.pnl_std,
                "p05": summary.pnl_p05,
                "p01": summary.pnl_p01,
            }
        )

    print("Efficient frontier candidates")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print()
    print(f"Chosen balanced portfolio: {balanced.name}")
    print(format_positions(candidate_positions[balanced.name]).to_string(index=False))
    print()
    print("Portfolio risk comparison")
    print(pd.DataFrame(exposure_rows).to_string(index=False, float_format=lambda x: f"{x:,.2f}"))


if __name__ == "__main__":
    main()
