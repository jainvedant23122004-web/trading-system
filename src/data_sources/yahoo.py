"""Yahoo daily OHLC data acquisition and raw-file validation."""

from pathlib import Path

import pandas as pd

RAW_COLUMNS = ("date", "open", "high", "low", "close")


def normalize_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a Yahoo history frame to unique, ordered daily OHLC rows."""
    if frame.empty:
        raise ValueError("Yahoo returned no rows")
    data = frame.copy().reset_index()
    data.columns = [str(column).strip().lower() for column in data.columns]
    if "datetime" in data.columns and "date" not in data.columns:
        data = data.rename(columns={"datetime": "date"})
    missing = set(RAW_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Yahoo OHLC is missing columns: {', '.join(sorted(missing))}")
    return validate_raw_ohlc(data.loc[:, list(RAW_COLUMNS)])


def validate_raw_ohlc(frame: pd.DataFrame) -> pd.DataFrame:
    """Check OHLC values and canonicalize dates for cached or new raw data."""
    missing = set(RAW_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Raw OHLC is missing columns: {', '.join(sorted(missing))}")
    data = frame.loc[:, list(RAW_COLUMNS)].copy()
    data["date"] = pd.to_datetime(data["date"], errors="raise").dt.tz_localize(None).dt.normalize()
    for column in RAW_COLUMNS[1:]:
        data[column] = pd.to_numeric(data[column], errors="raise")
    if data[list(RAW_COLUMNS[1:])].isna().any().any():
        raise ValueError("Raw OHLC has missing prices")
    if not (data[list(RAW_COLUMNS[1:])] > 0).all().all():
        raise ValueError("Raw OHLC prices must be positive")
    if data.empty:
        raise ValueError("Raw OHLC has no rows")
    return data.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def download_yahoo_ohlc(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Download unadjusted daily OHLC; Yahoo's end date is exclusive."""
    import yfinance as yf

    history = yf.Ticker(ticker).history(
        start=start, end=end, interval="1d", auto_adjust=False, actions=False
    )
    return normalize_ohlc(history)


def load_raw_ohlc(path: str | Path) -> pd.DataFrame:
    """Read a previously downloaded raw OHLC CSV."""
    return validate_raw_ohlc(pd.read_csv(path))
