"""Percentage-point return calculations on raw daily series."""

import numpy as np
import pandas as pd


def close_to_close_returns(frame: pd.DataFrame, output_name: str) -> pd.DataFrame:
    """Calculate daily close-to-close returns within one sorted source series."""
    ordered = frame.sort_values("date").copy()
    ordered[output_name] = (ordered["close"] / ordered["close"].shift(1) - 1) * 100
    return ordered.loc[:, ["date", output_name]]


def nifty_it_returns(frame: pd.DataFrame) -> pd.DataFrame:
    """Create current target and the prior Indian session's target."""
    ordered = frame.sort_values("date").copy()
    if ordered["date"].duplicated().any():
        raise ValueError("Nifty IT has duplicate trading dates")
    if not np.isfinite(ordered[["open", "close"]].to_numpy()).all() or (ordered["open"] <= 0).any():
        raise ValueError("Nifty IT open and close prices must be finite and open positive")
    ordered["nifty_it_ret"] = ((ordered["close"] - ordered["open"]) / ordered["open"]) * 100
    ordered["sector_prev_ret"] = ordered["nifty_it_ret"].shift(1)
    return ordered.loc[:, ["date", "sector_prev_ret", "nifty_it_ret"]].reset_index(drop=True)
