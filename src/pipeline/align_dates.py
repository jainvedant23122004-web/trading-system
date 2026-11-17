"""Backward source-date matching without same-day close leakage."""

import pandas as pd


def align_prior_observation(
    indian_dates: pd.DataFrame, source_returns: pd.DataFrame, feature: str, source_name: str,
    min_calendar_lag_days: int = 1,
) -> pd.DataFrame:
    """Match each Indian date to the last source session strictly before it."""
    source_date = f"{source_name}_source_date"
    source = source_returns.loc[:, ["date", feature]].rename(columns={"date": source_date})
    if source[source_date].duplicated().any():
        raise ValueError(f"{source_name} source dates are duplicated")
    if min_calendar_lag_days < 1:
        raise ValueError("Source lag must be at least one calendar day")
    left = indian_dates.copy()
    left["_match_date"] = left["date"] - pd.Timedelta(days=min_calendar_lag_days - 1)
    result = pd.merge_asof(
        left.sort_values("_match_date"),
        source.sort_values(source_date),
        left_on="_match_date",
        right_on=source_date,
        direction="backward",
        allow_exact_matches=False,
    )
    ages = (result["date"] - result[source_date]).dt.days
    result.loc[ages > (10 if source_name == "usdinr" else 7), feature] = float("nan")
    return result.drop(columns="_match_date")
