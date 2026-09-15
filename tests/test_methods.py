import numpy as np
import pandas as pd

from src.backtest import rolling_equal_weight, rolling_min_variance
from src.covariance import pca_covariance
from src.portfolio import minimum_variance_weights


def synthetic_returns(rows: int = 40, assets: int = 4) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    values = rng.normal(0.0002, 0.01, size=(rows, assets))
    dates = pd.bdate_range("2020-01-01", periods=rows)
    return pd.DataFrame(values, index=dates, columns=[f"A{i}" for i in range(assets)])


def test_pca_covariance_is_symmetric_and_finite():
    returns = synthetic_returns()
    covariance = returns.cov().to_numpy()
    regularised = pca_covariance(covariance, k=2)
    assert regularised.shape == covariance.shape
    assert np.allclose(regularised, regularised.T)
    assert np.isfinite(regularised).all()


def test_minimum_variance_weights_sum_to_one():
    returns = synthetic_returns()
    weights = minimum_variance_weights(returns.cov().to_numpy())
    assert np.isclose(weights.sum(), 1.0)


def test_rolling_outputs_have_common_out_of_sample_length():
    returns = synthetic_returns(rows=40)
    equal_returns, equal_weights = rolling_equal_weight(returns, window=10)
    pca_returns, pca_weights = rolling_min_variance(returns, window=10, covariance="pca", k=2)
    assert len(equal_returns) == len(returns) - 10
    assert equal_weights.shape == pca_weights.shape
    assert equal_returns.index.equals(pca_returns.index)
