# Data

I used adjusted-close prices downloaded from Tiingo's daily prices endpoint for the 30 tickers below:

`AAPL`, `AMGN`, `AMZN`, `AXP`, `BA`, `CAT`, `CRM`, `CSCO`, `CVX`, `DIS`, `GOOGL`, `GS`, `HD`, `HON`, `IBM`, `JNJ`, `JPM`, `KO`, `MCD`, `MMM`, `MRK`, `MSFT`, `NKE`, `NVDA`, `PG`, `SHW`, `TRV`, `UNH`, `V`, `WMT`.

The price period is 2010-01-04 to 2026-07-31. The data-download notebook creates:

- `data/raw/djia30_tiingo_adjclose_2010_2026.csv`
- `data/processed/returns.csv`

I did not include these files in version control. Set `TIINGO_API_KEY` in the shell before running `notebooks/01_download_data.ipynb`. I never put the key in a notebook, a committed configuration file, or the repository README.
