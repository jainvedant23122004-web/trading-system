"""Validate the complete internal dataset before writing model columns."""

import numpy as np
import pandas as pd

from ..features import FEATURES, TARGET

SOURCE_FEATURES = ("nasdaq", "sp500", "dow", "vix", "usdinr")


def validate_dataset(frame: pd.DataFrame) -> dict[str, int]:
    """Check schema, chronology, source dates, and previous-session returns."""
    required = ["date", *FEATURES, TARGET, *(f"{name}_source_date" for name in SOURCE_FEATURES)]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {', '.join(sorted(missing))}")
    if frame.empty:
        raise ValueError("Dataset has no Indian trading sessions")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Dataset dates must be sorted ascending")
    if frame["date"].duplicated().any():
        raise ValueError("Dataset has duplicate Indian trading dates")
    for column in [*FEATURES, TARGET]:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Dataset column {column} must be numeric")
        if np.isinf(frame[column].to_numpy(dtype=float)).any():
            raise ValueError(f"Dataset column {column} contains infinity")
    expected_previous = frame[TARGET].shift(1)
    actual_previous = frame["sector_prev_ret"]
    if not np.allclose(actual_previous, expected_previous, equal_nan=True):
        raise ValueError("sector_prev_ret does not equal the previous Indian session's nifty_it_ret")
    for name in SOURCE_FEATURES:
        source_date = f"{name}_source_date"
        present = frame[source_date].notna()
        if (frame.loc[present, source_date] >= frame.loc[present, "date"]).any():
            raise ValueError(f"{name} uses a same-day or future daily close")
        feature = f"{name}_ret"
        if frame.loc[~present, feature].notna().any():
            raise ValueError(f"{feature} has values without a source date")
    return {column: int(frame[column].isna().sum()) for column in [*FEATURES, TARGET]}


def print_dataset_report(frame: pd.DataFrame, missing: dict[str, int]) -> None:
    """Print scope and per-column missing counts."""
    print(f"Indian trading dates: {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(f"Total rows: {len(frame)}")
    for column, count in missing.items():
        print(f"Missing {column}: {count}")
