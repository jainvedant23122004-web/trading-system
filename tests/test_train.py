"""Time ordering tests for the train and test split."""

import pandas as pd
import pytest

from src.train import chronological_split


def test_chronological_split_keeps_last_fifth_for_test() -> None:
    """An ordered ten-row dataset splits into eight then two rows."""
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=10)})
    train, test = chronological_split(frame)
    assert len(train) == 8
    assert len(test) == 2
    assert train["date"].max() < test["date"].min()


def test_chronological_split_rejects_unsorted_data() -> None:
    """Callers must sort before splitting to prevent accidental leakage."""
    frame = pd.DataFrame({"date": pd.date_range("2025-01-01", periods=5)[::-1]})
    with pytest.raises(ValueError, match="sorted"):
        chronological_split(frame)
