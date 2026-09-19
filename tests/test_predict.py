"""Classification boundary tests."""

import pytest

from src.predict import classify_prediction


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.51, "Strong Bullish"),
        (0.50, "Bullish"),
        (0.21, "Bullish"),
        (0.20, "Neutral"),
        (0.0, "Neutral"),
        (-0.20, "Neutral"),
        (-0.21, "Bearish"),
        (-0.50, "Bearish"),
        (-0.51, "Strong Bearish"),
    ],
)
def test_classification_boundaries(value: float, expected: str) -> None:
    """Strict classification inequalities match the specification."""
    assert classify_prediction(value) == expected
