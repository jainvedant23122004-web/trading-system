"""Project paths and training settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "market_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "nifty_it_ridge.joblib"
TRAIN_FRACTION = 0.8
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
