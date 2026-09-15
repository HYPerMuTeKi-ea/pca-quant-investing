#!/usr/bin/env python3
"""Run the cleaned rolling portfolio pipeline and write tables/figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (  # noqa: E402
    ANNUALISATION_FACTOR,
    PCA_K_VALUES,
    PRIMARY_PCA_K,
    RESULTS_DIR,
    ROLLING_VOL_WINDOW,
    RETURNS_FILE,
    TRAIN_WINDOW,
)
from src.backtest import rolling_equal_weight, rolling_min_variance  # noqa: E402
from src.metrics import portfolio_summary  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--returns", type=Path, default=RETURNS_FILE)
    parser.add_argument("--window", type=int, default=TRAIN_WINDOW)
    parser.add_argument("--k-values", type=int, nargs="+", default=list(PCA_K_VALUES))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def load_returns(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Returns file not found: {path}\n"
            "Run notebooks/01_download_data.ipynb first, or pass --returns PATH."
        )
    returns = pd.read_csv(path, index_col=0, parse_dates=True)
    returns = returns.apply(pd.to_numeric, errors="raise")
    if returns.isna().any().any():
        raise ValueError("The returns file contains missing values")
    return returns.sort_index()


def main() -> None:
    args = parse_args()
    returns = load_returns(args.returns)
    if any(k < 1 or k > returns.shape[1] for k in args.k_values):
        raise ValueError("each k must be between 1 and the number of assets")

    output_dir = args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    equal_returns, equal_weights = rolling_equal_weight(returns, args.window)
    sample_returns, sample_weights = rolling_min_variance(returns, args.window, covariance="sample")
    pca_results: dict[int, tuple[pd.Series, pd.DataFrame]] = {
        k: rolling_min_variance(returns, args.window, covariance="pca", k=k)
        for k in args.k_values
    }

    summary_rows = {
        "Equal Weight": portfolio_summary(equal_returns, equal_weights, returns, ANNUALISATION_FACTOR),
        "Sample MinVar": portfolio_summary(sample_returns, sample_weights, returns, ANNUALISATION_FACTOR),
    }
    primary_k = PRIMARY_PCA_K if PRIMARY_PCA_K in pca_results else args.k_values[0]
    primary_returns, primary_weights = pca_results[primary_k]
    summary_rows[f"PCA MinVar (k={primary_k})"] = portfolio_summary(
        primary_returns, primary_weights, returns, ANNUALISATION_FACTOR
    )
    pd.DataFrame(summary_rows).T.to_csv(tables_dir / "portfolio_summary.csv")

    sensitivity_rows = {}
    for k, (pca_returns, pca_weights) in pca_results.items():
        stats = portfolio_summary(pca_returns, pca_weights, returns, ANNUALISATION_FACTOR)
        sensitivity_rows[k] = {
            "annualised_realised_volatility": stats["annualised_realised_volatility"],
            "mean_gross_exposure": stats["mean_gross_exposure"],
            "mean_max_abs_weight": stats["mean_max_abs_weight"],
            "mean_target_weight_turnover": stats["mean_target_weight_turnover"],
        }
    sensitivity = pd.DataFrame(sensitivity_rows).T
    sensitivity.index.name = "k"
    sensitivity.to_csv(tables_dir / "pca_sensitivity.csv")

    # Rolling volatility figure.
    series_by_name = {
        "Equal Weight": equal_returns,
        "Sample MinVar": sample_returns,
        f"PCA MinVar (k={primary_k})": primary_returns,
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, series in series_by_name.items():
        ax.plot(series.index, series.rolling(ROLLING_VOL_WINDOW).std() * (ANNUALISATION_FACTOR**0.5), label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Annualised realised volatility")
    ax.set_title(f"{ROLLING_VOL_WINDOW}-Day Rolling Realised Volatility")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "rolling_realised_volatility_63d.png", dpi=160)
    plt.close(fig)

    # Cumulative wealth figure.
    fig, ax = plt.subplots(figsize=(12, 6))
    for name, series in series_by_name.items():
        ax.plot(series.index, (1.0 + series).cumprod(), label=name)
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative wealth")
    ax.set_title("Cumulative Wealth of $1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / "cumulative_wealth.png", dpi=160)
    plt.close(fig)

    print(f"Wrote results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
