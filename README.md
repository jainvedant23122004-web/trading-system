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

The committed CSV is **synthetic mock data** for checking the code path. Its metrics are not real market performance. Replace it with correctly timestamped historical data for research.

## Model and evaluation

The model is a scikit-learn `Pipeline(StandardScaler, Ridge)`. Scaling puts the inputs on comparable scales so Ridge's penalty treats them fairly. Ridge handles correlated global indices and reduces unstable coefficients. Candidate alpha values `0.01`, `0.1`, `1`, `10`, and `100` are compared using forward time-series cross-validation on the training period only. The first 80% of dates are training data; the final 20% are held out. Chronological splitting is necessary because random shuffling would let later conditions influence an earlier prediction.

Training prints the selected alpha, intercept, and coefficients ordered by absolute size. Standardized coefficients provide an approximate comparison of relative feature influence, not causal effects. Holdout output includes MAE, RMSE, R², overall directional accuracy, and directional accuracy plus average realized returns for predictions with absolute value at least `0.10%`, `0.20%`, `0.30%`, and `0.50%`. Empty threshold groups print N/A.

## Run

From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py train
python main.py predict
python -m pytest -q
```

On macOS or Linux, activate with `source .venv/bin/activate`. `train` reads the processed CSV and saves the full pipeline at `models/nifty_it_ridge.joblib`. `predict` loads that file and runs **synthetic example inputs**, showing a predicted percentage return and the preliminary Strong Bullish, Bullish, Neutral, Bearish, or Strong Bearish classification. For your own pre-market inputs, call `src.predict.predict_return` with a mapping containing exactly the seven feature names. Only load model artifacts you trust: joblib uses pickle serialization.

## Limits and next steps

The mock data cannot establish predictive value. Real data will need verified release and observation timestamps, exchange calendars, holiday and daylight-saving alignment, and checks for data revisions. A single chronological holdout can be sensitive to the chosen market period. Future work should add audited historical data, walk-forward backtesting, a simple baseline, cost and slippage analysis, model stability checks, and calibration of the classification thresholds before any trading use.
