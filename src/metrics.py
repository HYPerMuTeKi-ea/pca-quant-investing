"""Performance and position diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import rolling_drifted_turnover


def target_weight_turnover(weights: pd.DataFrame) -> pd.Series:
    """Half the absolute target-weight change between consecutive dates."""

    return 0.5 * weights.diff().abs().sum(axis=1, min_count=1)


def annualised_volatility(portfolio_returns: pd.Series, periods_per_year: int = 252) -> float:
    return float(portfolio_returns.std(ddof=1) * np.sqrt(periods_per_year))


def terminal_wealth(portfolio_returns: pd.Series) -> float:
    return float((1.0 + portfolio_returns).cumprod().iloc[-1])


def cagr(portfolio_returns: pd.Series, periods_per_year: int = 252) -> float:
    wealth = terminal_wealth(portfolio_returns)
    years = len(portfolio_returns) / periods_per_year
    return float(wealth ** (1.0 / years) - 1.0)


def portfolio_summary(
    portfolio_returns: pd.Series,
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Return the summary statistics used in the report."""

    drifted = rolling_drifted_turnover(weights, returns)
    target = target_weight_turnover(weights).dropna()
    return {
        "annualised_realised_volatility": annualised_volatility(portfolio_returns, periods_per_year),
        "cagr": cagr(portfolio_returns, periods_per_year),
        "terminal_wealth": terminal_wealth(portfolio_returns),
        "mean_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "mean_max_abs_weight": float(weights.abs().max(axis=1).mean()),
        "mean_target_weight_turnover": float(target.mean()) if len(target) else 0.0,
        "mean_drifted_turnover": float(drifted.mean()) if len(drifted) else 0.0,
    }
