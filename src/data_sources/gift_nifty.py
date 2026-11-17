"""Optional, externally sourced GIFT Nifty pre-open return data."""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from ..config import RAW_DIR


def load_gift_nifty_data(path: str | Path = RAW_DIR / "gift_nifty.csv") -> pd.DataFrame:
    """Load dated GIFT returns, or an empty frame if no source is installed.

    CSV authors must provide values observed before 09:15 IST on each date.
    An optional observed_at timezone-aware timestamp is checked when present.
    """
    source = Path(path)
    if not source.exists():
        warnings.warn(f"GIFT Nifty file absent: {source}; gift_ret remains missing", stacklevel=2)
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "gift_ret": pd.Series(dtype=float)})
    frame = pd.read_csv(source)
    missing = {"date", "gift_ret"} - set(frame.columns)
    if missing:
        raise ValueError(f"GIFT data missing columns: {', '.join(sorted(missing))}")
    frame["date"] = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if frame["date"].duplicated().any():
        raise ValueError("GIFT data has duplicate dates")
    frame["gift_ret"] = pd.to_numeric(frame["gift_ret"], errors="raise")
    if not np.isfinite(frame["gift_ret"]).all():
        raise ValueError("GIFT returns must be finite")
    if "observed_at" in frame:
        if not frame["observed_at"].astype(str).str.contains(r"(?:Z|[+-]\d{2}:\d{2})$", case=False).all():
            raise ValueError("GIFT observed_at timestamps must include a timezone offset")
        observed = pd.to_datetime(frame["observed_at"], utc=True, errors="raise").dt.tz_convert("Asia/Kolkata")
        cutoff = frame["date"].dt.tz_localize("Asia/Kolkata") + pd.Timedelta(hours=9, minutes=15)
        if not (observed < cutoff).all():
            raise ValueError("GIFT observations must precede the Indian 09:15 open")
    return frame.loc[:, ["date", "gift_ret"]].sort_values("date").reset_index(drop=True)
