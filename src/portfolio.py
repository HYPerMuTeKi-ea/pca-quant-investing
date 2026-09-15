"""Portfolio-weight construction."""

from __future__ import annotations

import numpy as np


def equal_weights(n_assets: int) -> np.ndarray:
    """Return an equal-weight vector whose entries sum to one."""

    if n_assets < 1:
        raise ValueError("n_assets must be positive")
    return np.full(n_assets, 1.0 / n_assets)


def minimum_variance_weights(covariance: np.ndarray) -> np.ndarray:
    """Return unconstrained fully-invested minimum-variance weights.

    The only constraint is ``sum(weights) == 1``. Negative weights are
    allowed, matching the completed exploratory backtest.
    """

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    ones = np.ones(cov.shape[0])
    try:
        direction = np.linalg.solve(cov, ones)
    except np.linalg.LinAlgError:
        direction = np.linalg.pinv(cov) @ ones
    normaliser = float(ones @ direction)
    if not np.isfinite(normaliser) or abs(normaliser) < 1e-14:
        raise np.linalg.LinAlgError("covariance matrix cannot be normalised")
    return direction / normaliser
