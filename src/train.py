"""Chronological training and model reporting."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import ALPHAS, MODEL_PATH, TRAIN_FRACTION
from .features import FEATURES, TARGET


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ordered observations into an 80% train and 20% test period."""
    if len(frame) < 4:
        raise ValueError("At least 4 complete observations are needed")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Data must be sorted by date before splitting")
    cut = int(len(frame) * TRAIN_FRACTION)
    return frame.iloc[:cut].copy(), frame.iloc[cut:].copy()


def train_model(train_data: pd.DataFrame) -> Pipeline:
    """Select alpha with forward time-series folds entirely within training data."""
    pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge())])
    folds = TimeSeriesSplit(n_splits=min(5, len(train_data) - 1))
    search = GridSearchCV(
        pipeline,
        {"ridge__alpha": ALPHAS},
        cv=folds,
        scoring="neg_mean_squared_error",
    )
    search.fit(train_data[FEATURES], train_data[TARGET])
    return search.best_estimator_


def save_model(model: Pipeline, path: str | Path = MODEL_PATH) -> None:
    """Save the complete fitted preprocessing and Ridge pipeline."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, destination)


def print_training_report(model: Pipeline, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    """Display split boundaries and standardized model coefficients."""
    for label, data in (("Training", train_data), ("Test", test_data)):
        print(f"{label} start date: {data['date'].iloc[0].date()}")
        print(f"{label} end date: {data['date'].iloc[-1].date()}")
        print(f"{label} rows: {len(data)}")

    ridge = model.named_steps["ridge"]
    print(f"Selected alpha: {ridge.alpha:g}")
    print(f"Intercept: {ridge.intercept_:+.4f}")
    print("Feature               Coefficient")
    coefficients = sorted(zip(FEATURES, ridge.coef_), key=lambda pair: abs(pair[1]), reverse=True)
    for feature, coefficient in coefficients:
        print(f"{feature:<21} {coefficient:+.4f}")
    print("Standardized feature coefficients give an approximate comparison of relative influence.")
