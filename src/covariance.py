"""Covariance estimators used by the portfolio backtest."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sample_covariance(returns: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Return the usual sample covariance matrix for asset returns."""

    values = returns.to_numpy(dtype=float) if isinstance(returns, pd.DataFrame) else np.asarray(returns, dtype=float)
    if values.ndim != 2:
        raise ValueError("returns must be a two-dimensional array")
    if values.shape[0] < 2:
        raise ValueError("at least two observations are required")
    return np.cov(values, rowvar=False, ddof=1)


def pca_covariance(covariance: np.ndarray, k: int) -> np.ndarray:
    """Build the PCA-regularised covariance estimator used in the project.

    The largest ``k`` eigenvalues/eigenvectors are retained. All remaining
    directions receive the mean of the discarded eigenvalues. This preserves
    residual variance while reducing the effect of noisy small eigenvalues.
    """

    cov = np.asarray(covariance, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("covariance must be a square matrix")
    n_assets = cov.shape[0]
    if not 1 <= int(k) <= n_assets:
        raise ValueError(f"k must be between 1 and {n_assets}")

    # Numerical noise can make a theoretically symmetric matrix appear
    # slightly asymmetric; use the symmetric part for a stable eigendecomposition.
    cov = (cov + cov.T) / 2.0
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    retained_values = eigenvalues[:k]
    retained_vectors = eigenvectors[:, :k]
    factor_covariance = retained_vectors @ np.diag(retained_values) @ retained_vectors.T

    residual_variance = float(np.mean(eigenvalues[k:])) if k < n_assets else 0.0
    residual_projection = np.eye(n_assets) - retained_vectors @ retained_vectors.T
    regularised = factor_covariance + residual_variance * residual_projection
    return (regularised + regularised.T) / 2.0
