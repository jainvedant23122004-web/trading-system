"""Project paths and training settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "market_data.csv"
AUDIT_PATH = PROJECT_ROOT / "data" / "processed" / "market_data_audit.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_PATH = PROJECT_ROOT / "models" / "nifty_it_ridge_historical.joblib"
RESULTS_DIR = PROJECT_ROOT / "results" / "historical"
DATA_START_DATE = "2015-01-01"
DATA_END_DATE: str | None = None
YAHOO_TICKERS = {
    "nasdaq": "^IXIC",
    "sp500": "^GSPC",
    "dow": "^DJI",
    "vix": "^VIX",
    "usdinr": "INR=X",
    "nifty_it": "^CNXIT",
}
TRAIN_FRACTION = 0.8
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
