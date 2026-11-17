"""Backward source-date matching without same-day close leakage."""

import pandas as pd


def align_prior_observation(
    indian_dates: pd.DataFrame, source_returns: pd.DataFrame, feature: str, source_name: str
) -> pd.DataFrame:
    """Match each Indian date to the last source session strictly before it."""
    source_date = f"{source_name}_source_date"
    source = source_returns.loc[:, ["date", feature]].rename(columns={"date": source_date})
    if source[source_date].duplicated().any():
        raise ValueError(f"{source_name} source dates are duplicated")
    return pd.merge_asof(
        indian_dates.sort_values("date"),
        source.sort_values(source_date),
        left_on="date",
        right_on=source_date,
        direction="backward",
        allow_exact_matches=False,
    )
