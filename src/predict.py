"""Prediction interface for seven pre-market feature values."""

from pathlib import Path
from collections.abc import Mapping

import joblib
import numpy as np
import pandas as pd

from .config import MODEL_PATH
from .features import FEATURES


def classify_prediction(prediction: float) -> str:
    """Classify a predicted percentage return using preliminary thresholds."""
    if prediction > 0.50:
        return "Strong Bullish"
    if prediction > 0.20:
        return "Bullish"
    if prediction < -0.50:
        return "Strong Bearish"
    if prediction < -0.20:
        return "Bearish"
    return "Neutral"


def predict_return(values: Mapping[str, float], path: str | Path = MODEL_PATH) -> tuple[float, str]:
    """Load a saved pipeline and predict from exactly seven named inputs."""
    if set(values) != set(FEATURES):
        raise ValueError(f"Prediction requires exactly these features: {', '.join(FEATURES)}")
    frame = pd.DataFrame([{feature: float(values[feature]) for feature in FEATURES}])
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Prediction inputs must be finite numbers")
    model = joblib.load(path)
    prediction = float(model.predict(frame)[0])
    return prediction, classify_prediction(prediction)
