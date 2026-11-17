# NIFTY IT Market Forecasting System

A Python research prototype that estimates NIFTY IT intraday returns using global equities, volatility, currency movements, and previous-session sector performance. The historical workflow is functional; predictive usefulness has not been established.

## Architecture and stack

Yahoo daily OHLC ? data validation ? return features and date alignment ? chronological split ? StandardScaler + Ridge / GridSearchCV ? holdout evaluation ? saved-model manual prediction.

Python 3.11+, pandas, NumPy, scikit-learn, joblib, yfinance, matplotlib, pytest. `requirements.txt` declares dependencies; `requirements-lock.txt` records the installed versions used for this milestone. The parent directory's requirements file is an older environment snapshot; use this project's files.

## Feature and target contract

All returns use percent units: `0.25` means 0.25%, not 25%.

| Column | Definition |
|---|---|
| nasdaq_ret | 100 ? (source close / previous source close ? 1), Nasdaq `^IXIC` |
| sp500_ret | Same calculation, S&P 500 `^GSPC` |
| dow_ret | Same calculation, Dow Jones `^DJI` |
| vix_ret | Same calculation, VIX `^VIX`; percentage change, not level |
| usdinr_ret | Same calculation, USD/INR `INR=X`, with conservative lag and quality exclusions |
| sector_prev_ret | Previous Indian session's NIFTY IT open-to-close return |
| nifty_it_ret | Target: **100 ? (current close ? current open) / current open**, `^CNXIT` |

Exactly six predictors are required. GIFT Nifty is excluded from acquisition, dataset building, completeness checks, training, and prediction. Its unused legacy loader and the original seven-feature synthetic example remain for historical reference.

US features use the latest source date strictly before the Indian target date. This avoids using that day's later US close and accommodates different source holidays through backward matching. Prior-session sector returns are shifted before filtering incomplete model rows. Returns never substitute zero for missing data.

### Availability and FX policy

The intended decision deadline is **09:10 IST**. Prior-date US sessions are assumed complete by this deadline; downloaded CSVs contain dates rather than verified observation/release timestamps. This is a historical date-alignment proxy, **not full point-in-time validation**. Vendor revisions and exact bar availability are uncertified.

The local USD/INR file has **117 rows** whose open/close/high/low bounds are inconsistent. The builder excludes a return if either of its two endpoints has inconsistent OHLC. It preserves missing returns rather than bridging across rejected rows. FX source dates must be strictly earlier than Indian date minus one calendar day (at least two calendar dates old), adding a conservative buffer because Yahoo FX daily bar timing is not certified. This is a delayed daily FX proxy, not a pre-open FX quote. Source ages above 7 calendar days for US data or 10 for FX are rejected as missing. These fixed engineering policies were not selected using holdout scores.

A better timestamp-certified FX source remains future work. Alignment audit dates are saved in `data/processed/market_data_audit.csv`. Missing warm-up/quality-excluded rows remain in the processed CSV and are explicitly removed by the loader. `build-dataset --require-complete` rejects any missing values, including warm-up rows; use the ordinary build for this snapshot.

## Verified historical experiment

The existing raw snapshot was reused without refreshing or selecting a favorable period. Raw NIFTY IT: **2,889 sessions, 2015-01-01?2026-09-18**. After feature warm-up and quality exclusions: **2,671 usable observations, 2015-01-06?2026-09-18**. There are 218 excluded model rows, including 217 missing FX feature rows.

- Training: **2,136 rows**, 2015-01-06?2024-06-11.
- Untouched chronological holdout: **535 rows**, 2024-06-12?2026-09-18.
- Five expanding TimeSeriesSplit folds on training only; GridSearchCV minimizes MSE across alpha 0.01, 0.1, 1, 10, 100.
- Selected alpha: **100**. StandardScaler is fitted inside each CV pipeline. Final model uses training rows only, not the holdout.

| Model | MAE (pp) | RMSE (pp) | R? | Direction accuracy |
|---|---:|---:|---:|---:|
| Ridge | 0.879579 | 1.174003 | -0.000795 | 53.46% |
| Zero return | 0.883178 | 1.174397 | -0.001466 | 0.00% |
| Training mean | 0.880068 | 1.173572 | -0.000061 | 53.64% |
| Previous session | 1.314821 | 1.725682 | -1.162361 | 49.16% |
| Majority direction | 1.063250 | 1.397363 | -0.417833 | 53.64% |

Every comparison uses the same 535 dates. Majority direction is learned from training sign frequencies (negative here); its constant prediction magnitude is training mean absolute return. This supplies regression metrics for a primarily directional benchmark. Zero predicts neutral, so its 0% sign accuracy is not a useful binary-direction benchmark.

Training Ridge: MAE **0.799472 pp**, RMSE **1.092416 pp**, R? **0.004527**, direction accuracy **54.54%**. Test prediction/actual Pearson correlation: **0.019805** (descriptive; no significance claim).

Ridge misses the 55% research goal and positive holdout R?, and does not beat the training mean on RMSE or direction. Slight MAE improvements do not demonstrate a forecasting edge. Three consecutive holdout segments have directional accuracies **48.04%, 56.18%, 56.18%**; this is descriptive stability analysis, not independent walk-forward retraining. The final holdout was not used to tune features, alpha, or classification thresholds. Future iteration should reserve new unseen data.

## Reproduce

Run from this directory. The existing workspace virtual environment is `..\.venv`; a fresh clone may create its own environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt
# Only needed if raw CSVs are absent; downloading later can change results:
python main.py download-data --start 2015-01-01 --end 2026-09-20
python main.py build-dataset
python main.py train --charts
python main.py evaluate
python main.py predict --demo
python -m pytest -q -p no:cacheprovider
```

The downloader reuses existing valid files; `--refresh` replaces caches. Reproducing the exact table requires the recorded raw hashes, not merely requesting the same dates from a revision-prone provider. Raw CSVs and model binaries are ignored by Git and remain local. The active processed CSV is already tracked despite its ignore pattern; this milestone does not change Git tracking.

Supply your own six values for manual inference:

```powershell
python main.py predict --nasdaq-ret 0.35 --sp500-ret 0.22 --dow-ret 0.10 --vix-ret -0.40 --usdinr-ret 0.05 --sector-prev-ret -0.12
```

This uses the historical model with manually supplied inputs; it does not acquire today's observations or certify their cutoff. Supply all six flags or `--demo`, without mixing them. Missing, extra API keys, nonnumeric, and nonfinite inputs are rejected. Preliminary classification boundaries remain unchanged: strong bullish >0.50%, bullish >0.20%, strong bearish <-0.50%, bearish <-0.20%, neutral otherwise. These labels are not validated trade instructions.

`train` and `evaluate` accept `--data`, `--model`, and `--results` overrides. Evaluation verifies recorded dataset/model hashes before reusing the recorded holdout and does not refit. The historical training command refuses filenames labeled synthetic. Custom input provenance remains the caller's responsibility.

## Artifacts and tests

- `models/nifty_it_ridge_historical.joblib`: fitted six-feature pipeline. Original `models/nifty_it_ridge.joblib` is preserved as a synthetic artifact.
- `results/historical/metrics.json`: training/test metrics, directional accuracy, correlation, baselines, three test-period diagnostics.
- `baseline_comparison.csv`, `test_predictions.csv`: identical-date benchmark table and actual/predicted holdout returns.
- `model_metadata.json`: feature/target contract, split dates/counts, UTC training timestamp, selected alpha, CV scores, dataset/raw/model/source-code hashes, environment versions and Git state.
- `coefficients.csv`: standardized feature coefficients; associations, not causal attribution.
- `holdout_diagnostics.png`: actual/predicted returns and cumulative directional accuracy (`--charts`).

**40 tests passed** during milestone verification. Coverage includes six-feature/GIFT exclusion, target/lag/date alignment, FX filtering and staleness, loader completeness, chronological splitting, Ridge fitting, train-only scaling, serialization, input validation, metric calculations, and hash-checked repeat evaluation. Temporary test observations are synthetic fixtures; the documented production experiment uses the existing historical market files.

## Limitations and roadmap

This is a research prototype. A validated live 09:10 forecast pipeline, broker connection, and real execution are not implemented. No profitability is claimed. Historical bar provenance, revisions, and FX timing remain limitations. Complete-case exclusion can affect the research sample. The snapshot ends in September 2026 and is not current.

Next: certified timestamps and improved FX data; genuine walk-forward validation and uncertainty estimates; instrument-specific buy/sell/hold and abstention rules; cost/slippage-aware backtesting with position sizing and risk controls; monitored paper trading. Broker integration is outside this milestone.

## Resume description

- Engineered a six-feature NIFTY IT forecasting dataset from 10+ years of global market history, applying calendar alignment, previous-session lagging, and FX quality exclusions to obtain 2,671 usable observations.
- Built a reproducible StandardScaler?Ridge pipeline with chronological 80/20 validation and five-fold time-series hyperparameter tuning; persisted fitted models and experiment provenance.
- Evaluated 535 unseen sessions against four benchmarks using MAE, RMSE, R?, correlation, and directional accuracy; achieved 53.46% direction accuracy and documented limitations through saved predictions, metrics, and a reusable inference CLI.
