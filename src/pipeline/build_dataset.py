"""Build the model CSV and a source-date audit CSV from raw files."""

from pathlib import Path
import warnings

import pandas as pd

from ..config import AUDIT_PATH, DATA_PATH, RAW_DIR
from ..data_sources.gift_nifty import load_gift_nifty_data
from ..data_sources.yahoo import load_raw_ohlc
from ..features import FEATURES, TARGET
from .align_dates import align_prior_observation
from .calculate_returns import close_to_close_returns, nifty_it_returns
from .validate_dataset import SOURCE_FEATURES, print_dataset_report, validate_dataset


def build_dataset(
    raw_dir: str | Path = RAW_DIR,
    output_path: str | Path = DATA_PATH,
    audit_path: str | Path = AUDIT_PATH,
    require_complete: bool = False,
) -> pd.DataFrame:
    """Build chronological Indian sessions; preserve missing GIFT unless required.

    Validation completes before any processed CSV is replaced.
    """
    raw = Path(raw_dir)
    nifty = nifty_it_returns(load_raw_ohlc(raw / "nifty_it.csv"))
    combined = nifty
    for name in SOURCE_FEATURES:
        returns = close_to_close_returns(load_raw_ohlc(raw / f"{name}.csv"), f"{name}_ret")
        combined = align_prior_observation(combined, returns, f"{name}_ret", name)
    gift = load_gift_nifty_data(raw / "gift_nifty.csv")
    combined = combined.merge(gift, on="date", how="left", validate="one_to_one")
    combined = combined.sort_values("date").reset_index(drop=True)
    missing = validate_dataset(combined)
    print_dataset_report(combined, missing)
    if require_complete and any(missing.values()):
        raise ValueError("Complete features required; missing values found. Add GIFT data or use a partial build.")
    if missing["gift_ret"]:
        warnings.warn("Processed dataset has missing GIFT values; training will drop those rows", stacklevel=2)

    output = Path(output_path)
    audit = Path(audit_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit.parent.mkdir(parents=True, exist_ok=True)
    model_columns = ["date", *FEATURES, TARGET]
    audit_columns = ["date", *(f"{name}_source_date" for name in SOURCE_FEATURES)]
    combined.loc[:, model_columns].to_csv(output, index=False)
    combined.loc[:, audit_columns].to_csv(audit, index=False)
    print(f"Wrote model dataset: {output}")
    print(f"Wrote source-date audit: {audit}")
    return combined.loc[:, model_columns]
