"""Rolling one-step-ahead portfolio backtests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .covariance import pca_covariance, sample_covariance
from .portfolio import equal_weights, minimum_variance_weights


def _validate_returns(returns: pd.DataFrame, window: int) -> None:
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame")
    if returns.shape[1] < 1:
        raise ValueError("returns must contain at least one asset")
    if window < 2 or window >= len(returns):
        raise ValueError("window must be at least 2 and smaller than the number of rows")
    if not returns.index.is_monotonic_increasing:
        raise ValueError("returns index must be sorted in increasing order")
    if returns.isna().any().any() or not np.isfinite(returns.to_numpy(dtype=float)).all():
        raise ValueError("returns must contain only finite values")


def rolling_equal_weight(returns: pd.DataFrame, window: int = 252) -> tuple[pd.Series, pd.DataFrame]:
    """Evaluate a fixed equal-weight portfolio one step after each window."""

    _validate_returns(returns, window)
    weights = equal_weights(returns.shape[1])
    dates = returns.index[window:]
    portfolio_returns = pd.Series(
        [float(weights @ returns.iloc[t].to_numpy(dtype=float)) for t in range(window, len(returns))],
        index=dates,
        name="equal_weight_return",
    )
    weights_frame = pd.DataFrame(np.tile(weights, (len(dates), 1)), index=dates, columns=returns.columns)
    return portfolio_returns, weights_frame


def rolling_min_variance(
    returns: pd.DataFrame,
    window: int = 252,
    covariance: str = "sample",
    k: int | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Run the rolling sample- or PCA-covariance minimum-variance strategy."""

    _validate_returns(returns, window)
    if covariance not in {"sample", "pca"}:
        raise ValueError("covariance must be 'sample' or 'pca'")
    if covariance == "pca" and k is None:
        raise ValueError("k is required for PCA covariance")

    return_values: list[float] = []
    weights_values: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []

    for t in range(window, len(returns)):
        history = returns.iloc[t - window : t]
        cov = sample_covariance(history)
        if covariance == "pca":
            cov = pca_covariance(cov, int(k))
        target = minimum_variance_weights(cov)
        next_return = returns.iloc[t].to_numpy(dtype=float)
        return_values.append(float(target @ next_return))
        weights_values.append(target)
        dates.append(returns.index[t])

    return (
        pd.Series(return_values, index=dates, name=f"{covariance}_minvar_return"),
        pd.DataFrame(weights_values, index=dates, columns=returns.columns),
    )


def one_step_drifted_turnover(
    previous_weights: np.ndarray,
    asset_returns: np.ndarray,
    target_weights: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Return post-return drifted weights and turnover needed to hit a target."""

    previous = np.asarray(previous_weights, dtype=float)
    asset = np.asarray(asset_returns, dtype=float)
    target = np.asarray(target_weights, dtype=float)
    if not (previous.shape == asset.shape == target.shape):
        raise ValueError("weight and return vectors must have the same shape")
    portfolio_return = float(previous @ asset)
    denominator = 1.0 + portfolio_return
    if abs(denominator) < 1e-14:
        raise ZeroDivisionError("portfolio value became numerically zero")
    drifted = previous * (1.0 + asset) / denominator
    turnover = float(0.5 * np.abs(drifted - target).sum())
    return drifted, turnover


def rolling_drifted_turnover(weights: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Compute turnover after one day of asset-price drift."""

    if len(weights) < 2:
        return pd.Series(dtype=float, name="drifted_turnover")
    if not weights.index.isin(returns.index).all():
        raise ValueError("all weight dates must exist in returns")

    values: list[float] = []
    dates: list[pd.Timestamp] = []
    for t in range(1, len(weights)):
        previous_date = weights.index[t - 1]
        _, turnover = one_step_drifted_turnover(
            weights.iloc[t - 1].to_numpy(dtype=float),
            returns.loc[previous_date].to_numpy(dtype=float),
            weights.iloc[t].to_numpy(dtype=float),
        )
        values.append(turnover)
        dates.append(weights.index[t])
    return pd.Series(values, index=dates, name="drifted_turnover")
