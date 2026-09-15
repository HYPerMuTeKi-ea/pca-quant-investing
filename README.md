# Rolling PCA Covariance Estimation and Minimum-Variance Portfolios

I carried out this project as a summer quantitative study. I wanted to test whether a PCA-based covariance estimate could make a minimum-variance portfolio more stable out of sample.

The main question was:

> For the same stock universe, training window and optimisation objective, does replacing part of the sample covariance estimate with a PCA-based estimate reduce realised risk or make the portfolio weights more stable?

## What I did

I used adjusted-close prices from Tiingo for 30 US stocks. The data cover 4 January 2010 to 31 July 2026. After calculating simple daily returns, I used a rolling 252-trading-day window to estimate the covariance matrix and form the next day's portfolio.

I compared three portfolios:

- an equal-weight portfolio;
- a sample-covariance minimum-variance portfolio;
- a PCA-covariance minimum-variance portfolio.

The minimum-variance portfolios have weights that sum to one. Short positions are allowed and there is no gross-exposure limit. For the PCA estimator, I kept the first `k` eigen-directions and replaced the remaining eigenvalues with their average. I tested `k = 5, 10, 15, 20`, using `k = 10` for the main comparison.

## Main result

The realised volatility of PCA MinVar with `k = 10` was very close to that of the sample-covariance MinVar portfolio in the saved backtest:

| Portfolio | Annualised volatility | CAGR | Mean target turnover | Mean drifted turnover |
|---|---:|---:|---:|---:|
| Equal Weight | 16.76% | 18.19% | 0.00% | 0.44% |
| Sample MinVar | 13.47% | 12.81% | 4.80% | 4.98% |
| PCA MinVar (`k=10`) | 13.45% | 13.62% | 4.30% | 4.50% |

In this sample, PCA did not show a large reduction in realised volatility. Its clearer difference was in the portfolio holdings: average turnover, gross exposure and maximum individual position were lower than for the sample-covariance portfolio. I therefore interpret the result as partial support for the original hypothesis rather than evidence that PCA is always better.

The backtest is idealised. It does not include commissions, bid-ask spreads, slippage, financing costs or stock-borrow costs. The stock universe is fixed, so selection and survivorship bias are also possible. The results are descriptive and should not be read as a trading recommendation.

## Repository contents

```text
pca-quant-investing/
├── README.md
├── requirements.txt
├── .gitignore
├── config.py
├── data/
│   ├── raw/                  # downloaded data, not committed
│   ├── processed/            # returns.csv after the data notebook is run
│   └── README.md
├── docs/
│   ├── README.md
│   ├── PCA_quant_project_written_summary_English.docx
│   └── PCA_quant_project_written_summary_English.pdf
├── notebooks/
│   ├── README.md
│   ├── 00_environment_check.ipynb
│   ├── 01_download_data.ipynb
│   ├── 02_data_audit.ipynb
│   ├── 03_static_covariance.ipynb
│   └── 04_rolling_backtest_exploration.ipynb
├── results/
│   ├── README.md
│   ├── figures/
│   └── tables/
├── src/
│   ├── backtest.py
│   ├── covariance.py
│   ├── metrics.py
│   └── portfolio.py
├── scripts/
│   └── run_backtest.py
└── tests/
    └── test_methods.py
```

The notebooks show the analysis in the order in which I developed it. The `src/` folder contains a cleaned version of the repeated calculations, and the saved tables and figures give a compact record of the completed run.

## Reproduce the analysis

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

I used the Tiingo API for the prices. Set the API key in the environment before running the download notebook; do not put it in a notebook or commit it to GitHub.

```bash
export TIINGO_API_KEY="your-key"   # Windows PowerShell: $env:TIINGO_API_KEY="your-key"
```

I ran the notebooks in this order:

1. `00_environment_check.ipynb`
2. `01_download_data.ipynb`
3. `02_data_audit.ipynb`
4. `03_static_covariance.ipynb`
5. `04_rolling_backtest_exploration.ipynb`

After `data/processed/returns.csv` has been created, the cleaned pipeline can also be run with:

```bash
python scripts/run_backtest.py
```

The raw vendor data are not included in this repository. This keeps the repository small and avoids committing credentials or files that should be downloaded from the source again.

## Possible extension

A natural extension of this project would be to include reasonable transaction costs and test whether the difference between the two minimum-variance methods remains after costs.
