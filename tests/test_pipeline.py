"""Synthetic tests for acquisition-independent data preparation."""

import numpy as np
import pandas as pd
import pytest

from src.data_sources.gift_nifty import load_gift_nifty_data
from src.data_sources.yahoo import normalize_ohlc, validate_raw_ohlc
from src.pipeline.align_dates import align_prior_observation
from src.pipeline.build_dataset import build_dataset
from src.pipeline.calculate_returns import close_to_close_returns, nifty_it_returns
from src.pipeline.validate_dataset import validate_dataset


def test_close_to_close_returns_use_percentage_points() -> None:
    """Daily source returns use the preceding source close."""
    source = pd.DataFrame({"date": pd.to_datetime(["2026-09-17", "2026-09-18"]), "close": [100, 105]})
    result = close_to_close_returns(source, "nasdaq_ret")
    assert np.isnan(result.loc[0, "nasdaq_ret"])
    assert result.loc[1, "nasdaq_ret"] == pytest.approx(5.0)


def test_nifty_target_and_previous_session_shift() -> None:
    """The previous feature is the prior Indian session's open-to-close return."""
    source = pd.DataFrame({
        "date": pd.to_datetime(["2026-09-21", "2026-09-17", "2026-09-18"]),
        "open": [100, 100, 100], "high": [103, 102, 101], "low": [99, 99, 98], "close": [102, 101, 99],
    })
    result = nifty_it_returns(source)
    assert result["nifty_it_ret"].tolist() == pytest.approx([1.0, -1.0, 2.0])
    assert np.isnan(result.loc[0, "sector_prev_ret"])
    assert result["sector_prev_ret"].iloc[1:].tolist() == pytest.approx([1.0, -1.0])


def test_us_alignment_is_strictly_prior_and_handles_holiday() -> None:
    """Sep 18 India must use Sep 17 US, while Sep 21 uses Sep 18 US."""
    india = pd.DataFrame({"date": pd.to_datetime(["2026-09-18", "2026-09-21"])})
    us = pd.DataFrame({"date": pd.to_datetime(["2026-09-17", "2026-09-18", "2026-09-21"]),
                       "nasdaq_ret": [1.0, 2.0, 999.0]})
    result = align_prior_observation(india, us, "nasdaq_ret", "nasdaq")
    assert result["nasdaq_ret"].tolist() == [1.0, 2.0]
    assert result["nasdaq_source_date"].tolist() == list(pd.to_datetime(["2026-09-17", "2026-09-18"]))


def test_future_only_source_produces_missing_feature() -> None:
    """No source date on or after the Indian date can fill that row."""
    india = pd.DataFrame({"date": pd.to_datetime(["2026-09-18"])})
    us = pd.DataFrame({"date": pd.to_datetime(["2026-09-18", "2026-09-21"]),
                       "nasdaq_ret": [2.0, 3.0]})
    result = align_prior_observation(india, us, "nasdaq_ret", "nasdaq")
    assert pd.isna(result.loc[0, "nasdaq_ret"])
    assert pd.isna(result.loc[0, "nasdaq_source_date"])


def test_raw_normalization_sorts_and_removes_duplicates() -> None:
    """Yahoo-style OHLC headers become canonical ordered rows."""
    frame = pd.DataFrame({"Date": pd.to_datetime(["2026-09-18", "2026-09-17", "2026-09-18"]),
                          "Open": [100, 100, 101], "High": [102, 102, 103],
                          "Low": [99, 99, 100], "Close": [101, 101, 102]})
    result = normalize_ohlc(frame.set_index("Date"))
    assert len(result) == 2
    assert result["date"].is_monotonic_increasing
    assert result.loc[1, "close"] == 102


def test_raw_missing_column_is_rejected() -> None:
    """Incomplete downloaded OHLC cannot enter preprocessing."""
    with pytest.raises(ValueError, match="high"):
        validate_raw_ohlc(pd.DataFrame({"date": ["2026-09-18"], "open": [1], "low": [1], "close": [1]}))


def test_gift_missing_file_keeps_unavailable() -> None:
    """An absent GIFT source returns no fabricated observations."""
    with pytest.warns(UserWarning, match="gift_ret remains missing"):
        result = load_gift_nifty_data("nonexistent_gift_test_file.csv")
    assert result.empty
    assert result.columns.tolist() == ["date", "gift_ret"]


def test_gift_rejects_observation_after_open() -> None:
    """A supplied timestamp cannot put a late GIFT quote into pre-open inputs."""
    from pathlib import Path

    path = Path(__file__).parent / "gift_test_data.csv"
    try:
        pd.DataFrame({"date": ["2026-09-18"], "gift_ret": [0.2],
                      "observed_at": ["2026-09-18T09:16:00+05:30"]}).to_csv(path, index=False)
        with pytest.raises(ValueError, match="precede"):
            load_gift_nifty_data(path)
    finally:
        path.unlink(missing_ok=True)


def _internal_frame() -> pd.DataFrame:
    """Create a minimal valid internal frame with source-date audit columns."""
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-09-18", "2026-09-21"]),
                          "nifty_it_ret": [1.0, -0.5], "sector_prev_ret": [np.nan, 1.0],
                          "gift_ret": [np.nan, np.nan]})
    for name in ("nasdaq", "sp500", "dow", "vix", "usdinr"):
        frame[f"{name}_ret"] = [0.1, 0.2]
        frame[f"{name}_source_date"] = pd.to_datetime(["2026-09-17", "2026-09-18"])
    return frame


def test_dataset_validation_rejects_duplicate_and_missing_columns() -> None:
    """Duplicate Indian dates and missing schema fields fail visibly."""
    frame = _internal_frame()
    with pytest.raises(ValueError, match="missing columns"):
        validate_dataset(frame.drop(columns="vix_ret"))
    frame.loc[1, "date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_dataset(frame)


def test_dataset_validation_rejects_lookahead_and_bad_shift() -> None:
    """The audit catches same-day source closes and target leakage via the shift."""
    frame = _internal_frame()
    frame.loc[0, "nasdaq_source_date"] = frame.loc[0, "date"]
    with pytest.raises(ValueError, match="same-day or future"):
        validate_dataset(frame)
    frame = _internal_frame()
    frame.loc[1, "sector_prev_ret"] = frame.loc[1, "nifty_it_ret"]
    with pytest.raises(ValueError, match="previous Indian session"):
        validate_dataset(frame)


def test_dataset_validation_reports_missing_values() -> None:
    """Legacy GIFT does not affect completeness; first-session lag is missing."""
    missing = validate_dataset(_internal_frame())
    assert "gift_ret" not in missing
    assert missing["sector_prev_ret"] == 1


def test_build_dataset_outputs_ordered_model_columns(monkeypatch) -> None:
    """The full offline builder emits the eight ordered model columns."""
    from pathlib import Path
    import src.pipeline.build_dataset as builder

    nifty = pd.DataFrame({"date": pd.to_datetime(["2026-09-21", "2026-09-17", "2026-09-18"]),
                          "open": [100, 100, 100], "high": [103, 102, 101], "low": [99, 99, 98], "close": [102, 101, 99]})
    source = pd.DataFrame({"date": pd.to_datetime(["2026-09-16", "2026-09-17", "2026-09-18"]),
                           "open": [100, 101, 102], "high": [100, 101, 102], "low": [100, 101, 102], "close": [100, 101, 102]})
    monkeypatch.setattr(builder, "load_raw_ohlc", lambda path: nifty if Path(path).stem == "nifty_it" else source)
    output = Path(__file__).parent / "pipeline_model_test.csv"
    audit = Path(__file__).parent / "pipeline_audit_test.csv"
    try:
        with pytest.warns(UserWarning, match="USD/INR"):
            result = build_dataset(raw_dir="unused", output_path=output, audit_path=audit)
        assert result["date"].is_monotonic_increasing
        assert result.columns.tolist() == ["date", "nasdaq_ret", "sp500_ret", "dow_ret", "vix_ret",
                                            "usdinr_ret", "sector_prev_ret", "nifty_it_ret"]
        assert pd.read_csv(output).columns.tolist() == result.columns.tolist()
        assert "nasdaq_source_date" in pd.read_csv(audit).columns
    finally:
        output.unlink(missing_ok=True)
        audit.unlink(missing_ok=True)
