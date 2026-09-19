"""Holdout regression and directional metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from .features import FEATURES, TARGET

THRESHOLDS = (0.10, 0.20, 0.30, 0.50)


def evaluate_model(model: Pipeline, test_data: pd.DataFrame) -> dict[str, float]:
    """Print and return holdout metrics; threshold values use percentage points."""
    actual = test_data[TARGET].to_numpy()
    predicted = model.predict(test_data[FEATURES])
    metrics = {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "r2": float(r2_score(actual, predicted)) if len(actual) >= 2 else float("nan"),
        "directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
    }
    print(f"MAE: {metrics['mae']:.4f} percentage points")
    print(f"RMSE: {metrics['rmse']:.4f} percentage points")
    print(f"R2 (R-squared): {metrics['r2']:.4f}")
    print(f"Directional accuracy: {metrics['directional_accuracy']:.1%}")

    for threshold in THRESHOLDS:
        selected = np.abs(predicted) >= threshold
        count = int(selected.sum())
        print(f"|prediction| >= {threshold:.2f}%: {count} prediction{'s' if count != 1 else ''}")
        if count:
            accuracy = np.mean(np.sign(predicted[selected]) == np.sign(actual[selected]))
            print(f"  Direction accuracy: {accuracy:.1%}")
            print(f"  Average actual return: {np.mean(actual[selected]):+.4f}%")
            print(f"  Average absolute actual return: {np.mean(np.abs(actual[selected])):.4f}%")
        else:
            print("  Direction accuracy: N/A")
            print("  Average actual return: N/A")
            print("  Average absolute actual return: N/A")
    return metrics
