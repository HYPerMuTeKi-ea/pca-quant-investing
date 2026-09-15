# Notebooks

I developed the analysis through the notebooks in the following order:

1. `00_environment_check.ipynb` — check the numerical Python environment.
2. `01_download_data.ipynb` — download Tiingo adjusted-close prices and create `data/processed/returns.csv`.
3. `02_data_audit.ipynb` — inspect missing values, date order, duplicate dates, extreme returns, and zero-return runs.
4. `03_static_covariance.ipynb` — inspect the first 252-day covariance, correlation, eigenvalues, and explained variance.
5. `04_rolling_backtest_exploration.ipynb` — the original exploratory backtest covering equal weight, sample MinVar, PCA MinVar, sensitivity to `k`, drifted turnover, rolling volatility, and cumulative wealth.

The notebook paths assume that the notebooks are run from this directory. After the processed returns file has been created, I can also rerun the cleaned implementation with `python scripts/run_backtest.py` from the repository root.
