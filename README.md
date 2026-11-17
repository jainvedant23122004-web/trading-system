# Nifty IT pre-market direction research

This Python 3.11+ project estimates Nifty IT's intraday open-to-close percentage return from information available before the Indian market opens. Version 1 covers only Nifty IT. It is **experimental research software**, not guaranteed trading advice.

## Data contract

Place a processed CSV at `data/processed/market_data.csv` with one row per Indian trading date. Its columns are `date` (YYYY-MM-DD), the seven features below, and `nifty_it_ret`. All returns are percentage values, so `0.25` means `0.25%`.

| Feature | Meaning for Indian trading date t |
| --- | --- |
| `nasdaq_ret` | Nasdaq return from the previous completed US session |
| `sp500_ret` | S&P 500 return from the previous completed US session |
| `dow_ret` | Dow Jones return from the previous completed US session |
| `vix_ret` | VIX percentage change from the previous completed US session |
| `gift_ret` | GIFT Nifty movement observed before the Indian open |
| `usdinr_ret` | USD/INR overnight or pre-market change observed before the Indian open |
| `sector_prev_ret` | Nifty IT open-to-close return from the previous Indian session |

The target `nifty_it_ret` is `((Nifty IT close - Nifty IT open) / Nifty IT open) * 100` for date t. The loader validates columns, dates, and numeric values; sorts dates; and reports rows dropped for missing or non-finite values. It cannot verify observation timestamps or source provenance. The data producer must ensure every feature was actually known before date t's Indian open. Using same-day closing values or a US session that finished after the relevant Indian open would cause look-ahead bias and make evaluation misleading.

The original example is saved as `data/processed/market_data_synthetic.csv`. It is **synthetic mock data** for checking the code path; its metrics are not real market performance. The active `market_data.csv` is generated from downloaded historical data and is ignored by Git.

## Model and evaluation

The model is a scikit-learn `Pipeline(StandardScaler, Ridge)`. Scaling puts the inputs on comparable scales so Ridge's penalty treats them fairly. Ridge handles correlated global indices and reduces unstable coefficients. Candidate alpha values `0.01`, `0.1`, `1`, `10`, and `100` are compared using forward time-series cross-validation on the training period only. The first 80% of dates are training data; the final 20% are held out. Chronological splitting is necessary because random shuffling would let later conditions influence an earlier prediction.

Training prints the selected alpha, intercept, and coefficients ordered by absolute size. Standardized coefficients provide an approximate comparison of relative feature influence, not causal effects. Holdout output includes MAE, RMSE, R², overall directional accuracy, and directional accuracy plus average realized returns for predictions with absolute value at least `0.10%`, `0.20%`, `0.30%`, and `0.50%`. Empty threshold groups print N/A.

## Historical Data Pipeline

Run the stages separately so raw acquisition can cover a broad history while modeling can later select its own periods:

```powershell
python main.py download-data
python main.py build-dataset
python main.py train
```

`download-data` fetches daily OHLC from Yahoo via yfinance and saves one CSV per asset in `data/raw/`. It reuses valid raw files unless `--refresh` is passed. The default acquisition start is `2015-01-01` and the end is open-ended. Override them with `python main.py download-data --refresh --start 2020-01-01 --end 2025-01-01`; Yahoo treats the end as exclusive. The **acquisition date range is independent of the train/test date range**. Training still uses a chronological 80/20 split of complete processed observations. Cached files are reused as-is, so use `--refresh` when you need newer data.

| Raw file | Yahoo ticker | Derived model value |
| --- | --- | --- |
| `nasdaq.csv` | `^IXIC` | prior completed US session close-to-close `nasdaq_ret` |
| `sp500.csv` | `^GSPC` | prior completed US session close-to-close `sp500_ret` |
| `dow.csv` | `^DJI` | prior completed US session close-to-close `dow_ret` |
| `vix.csv` | `^VIX` | prior completed US session close-to-close `vix_ret` |
| `usdinr.csv` | `INR=X` | latest earlier daily close-to-close `usdinr_ret` |
| `nifty_it.csv` | `^CNXIT` | current Indian open-to-close target and prior Indian session return |

US and Indian trading dates cannot be merged directly: an Indian open on date t happens before that calendar day's US close. Each Indian row therefore takes the latest US source date **strictly earlier** than t. The same backward match naturally handles US and Indian holidays without filling returns with zero. Nifty IT's `sector_prev_ret` is a one-session shift on Indian trading dates. `data/processed/market_data_audit.csv` records the matched source dates so they can be checked for look-ahead bias. The builder checks dates, numeric values, missing data, target lag, and source-date order before writing the model CSV.

The USD/INR value is a **temporary daily close approximation**, not an observed pre-open FX quote. This pipeline conservatively uses a source date strictly before the Indian date because a same-day daily close is not known at 09:15 IST. Replace this data-source stage later with timestamped pre-open FX data. Yahoo daily histories can contain missing or revised prices and do not certify the exact time a quote became available.

GIFT Nifty has no reliable integrated source. You may provide `data/raw/gift_nifty.csv` with `date,gift_ret` rows known before each Indian open. An optional timezone-aware `observed_at` column is checked against 09:15 IST; date-only CSVs require an external provenance check. Without that file, `gift_ret` remains missing for every row. `build-dataset` produces a partial dataset and prints missing counts; `build-dataset --require-complete` fails before replacing the CSV if any value is missing. The seven-feature model cannot train on a dataset with no GIFT values. To exercise the existing model on mock data, run `python main.py train --data data/processed/market_data_synthetic.csv`. That output is only a software demonstration.

## Run

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py train --data data/processed/market_data_synthetic.csv
python main.py predict
python -m pytest -q
```

On macOS or Linux, activate with `source .venv/bin/activate`. `train` reads a processed CSV and saves the full pipeline at `models/nifty_it_ridge.joblib`. `predict` loads that file and runs **synthetic example inputs**, showing a predicted percentage return and the preliminary Strong Bullish, Bullish, Neutral, Bearish, or Strong Bearish classification. For your own pre-market inputs, call `src.predict.predict_return` with a mapping containing exactly the seven feature names. Only load model artifacts you trust: joblib uses pickle serialization.

## Limits and next steps

The mock data cannot establish predictive value. Real data will need verified release and observation timestamps, exchange calendars, holiday and daylight-saving alignment, and checks for data revisions. A single chronological holdout can be sensitive to the chosen market period. Future work should add audited historical data, walk-forward backtesting, a simple baseline, cost and slippage analysis, model stability checks, and calibration of the classification thresholds before any trading use.
