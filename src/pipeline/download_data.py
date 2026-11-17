"""Download all configured Yahoo series with valid-cache reuse."""

from pathlib import Path

from ..config import DATA_END_DATE, DATA_START_DATE, RAW_DIR, YAHOO_TICKERS
from ..data_sources.yahoo import download_yahoo_ohlc, load_raw_ohlc


def download_all(
    start: str = DATA_START_DATE,
    end: str | None = DATA_END_DATE,
    raw_dir: str | Path = RAW_DIR,
    refresh: bool = False,
) -> dict[str, str]:
    """Save each asset's OHLC separately and report failures after all attempts."""
    directory = Path(raw_dir)
    directory.mkdir(parents=True, exist_ok=True)
    status: dict[str, str] = {}
    failures: list[str] = []
    for name, ticker in YAHOO_TICKERS.items():
        path = directory / f"{name}.csv"
        try:
            if path.exists() and not refresh:
                data = load_raw_ohlc(path)
                status[name] = f"cached {len(data)} rows ({data['date'].min().date()} to {data['date'].max().date()})"
            else:
                data = download_yahoo_ohlc(ticker, start, end)
                data.to_csv(path, index=False)
                status[name] = f"downloaded {len(data)} rows ({data['date'].min().date()} to {data['date'].max().date()})"
        except Exception as exc:
            status[name] = f"UNAVAILABLE ({ticker}): {exc}"
            failures.append(name)
    for name, message in status.items():
        print(f"{name}: {message}")
    if failures:
        raise RuntimeError(f"Could not acquire valid raw data for: {', '.join(failures)}")
    return status
