"""Load and validate pre-aligned market observations."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .config import DATA_PATH
from .features import FEATURES, TARGET


def load_market_data(path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Return complete observations sorted by date, rejecting ambiguous dates.

    The CSV producer is responsible for aligning every feature to information
    available before that row's Indian market open.
    """
    frame = pd.read_csv(path)
    required = ["date", *FEATURES, TARGET]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

    frame = frame.loc[:, required].copy()
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if frame["date"].isna().any():
        raise ValueError("Missing dates are not allowed")
    if frame["date"].duplicated().any():
        raise ValueError("Duplicate dates are not allowed")

    for column in [*FEATURES, TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    missing_rows = frame[[*FEATURES, TARGET]].isna().any(axis=1)
    if missing_rows.any():
        warnings.warn(
            f"Dropped {int(missing_rows.sum())} rows with missing or non-finite values",
            stacklevel=2,
        )
        frame = frame.loc[~missing_rows]

    return frame.sort_values("date").reset_index(drop=True)
