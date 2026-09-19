"""Validation tests for the processed CSV contract."""

import pandas as pd
import pytest
from pathlib import Path
from collections.abc import Iterator

from src.data_loader import load_market_data
from src.features import FEATURES, TARGET


def valid_frame() -> pd.DataFrame:
    """Create two intentionally out-of-order complete records."""
    records = []
    for date in ("2025-01-02", "2025-01-01"):
        records.append({"date": date, **dict.fromkeys(FEATURES, 0.1), TARGET: 0.2})
    return pd.DataFrame(records)


@pytest.fixture
def csv_path() -> Iterator[Path]:
    """Use a workspace-local file where restricted runners permit writes."""
    path = Path(__file__).parent / "loader_test_data.csv"
    yield path
    path.unlink(missing_ok=True)


def test_loader_sorts_and_drops_missing_row(csv_path: Path) -> None:
    """Only complete rows remain and the earliest date comes first."""
    frame = valid_frame()
    frame.loc[0, "gift_ret"] = None
    frame.to_csv(csv_path, index=False)
    with pytest.warns(UserWarning, match="Dropped 1 rows"):
        loaded = load_market_data(csv_path)
    assert len(loaded) == 1
    assert loaded["date"].iloc[0] == pd.Timestamp("2025-01-01")


def test_loader_rejects_missing_column(csv_path: Path) -> None:
    """Absent required input columns produce a readable error."""
    valid_frame().drop(columns="vix_ret").to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="vix_ret"):
        load_market_data(csv_path)


def test_loader_rejects_duplicate_dates(csv_path: Path) -> None:
    """Duplicate target dates cannot silently enter the split."""
    frame = valid_frame()
    frame.loc[1, "date"] = frame.loc[0, "date"]
    frame.to_csv(csv_path, index=False)
    with pytest.raises(ValueError, match="Duplicate dates"):
        load_market_data(csv_path)
