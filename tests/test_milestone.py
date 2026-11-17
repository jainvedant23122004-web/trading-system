"""Contract, data availability, training, evaluation, and persistence regressions."""
from io import StringIO
import json
import numpy as np
import pandas as pd
import pytest
from src.features import FEATURES, TARGET
from src.data_loader import load_market_data
from src.data_sources.yahoo import validate_raw_ohlc
from src.pipeline.align_dates import align_prior_observation
from src.pipeline.build_dataset import build_dataset
from src.evaluate import regression_metrics
from src.predict import predict_return
from src.train import chronological_split, train_model, save_model
from src.research import run_training, evaluate_saved


@pytest.fixture
def observations():
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.normal(size=(60, 6)), columns=FEATURES)
    frame["date"] = pd.bdate_range("2020-01-01", periods=60)
    frame[TARGET] = frame.nasdaq_ret * 0.2 + rng.normal(0, 0.1, 60)
    return frame


def test_contract_excludes_gift():
    assert FEATURES == ["nasdaq_ret", "sp500_ret", "dow_ret", "vix_ret", "usdinr_ret", "sector_prev_ret"]


def test_unused_gift_does_not_remove_rows(observations):
    observations["gift_ret"] = np.nan
    assert len(load_market_data(StringIO(observations.to_csv(index=False)))) == 60


def test_fx_extra_lag_and_stale_source():
    dates = pd.DataFrame({"date": pd.to_datetime(["2026-09-18", "2026-10-18"])})
    source = pd.DataFrame({"date": pd.to_datetime(["2026-09-16", "2026-09-17"]), "usdinr_ret": [0.1, 999.]})
    result = align_prior_observation(dates, source, "usdinr_ret", "usdinr", 2)
    assert result.usdinr_ret.iloc[0] == 0.1
    assert pd.isna(result.usdinr_ret.iloc[1])


def test_bad_fx_endpoints_not_bridged(tmp_path):
    dates = pd.date_range("2026-09-01", periods=6)
    raw = pd.DataFrame({"date": dates, "open": 100., "high": 102., "low": 98., "close": 101.})
    for name in ("nasdaq", "sp500", "dow", "vix", "nifty_it", "usdinr"):
        prices = raw.copy()
        if name == "usdinr":
            prices.loc[1, "close"] = 999.
        prices.to_csv(tmp_path / f"{name}.csv", index=False)
    with pytest.warns(UserWarning, match="1 inconsistent"):
        result = build_dataset(tmp_path, tmp_path / "model.csv", tmp_path / "audit.csv")
    assert "gift_ret" not in result
    assert result.usdinr_ret.iloc[3:5].isna().all()
    assert result.usdinr_ret.iloc[5] == 0


def test_training_scaler_and_serialization(observations, tmp_path):
    train, test = chronological_split(observations)
    model = train_model(train)
    assert np.allclose(model.named_steps["scaler"].mean_, train[FEATURES].mean())
    assert model.named_steps["ridge"].alpha in (0.01, 0.1, 1, 10, 100)
    path = tmp_path / "model.joblib"
    save_model(model, path)
    values = test[FEATURES].iloc[0].to_dict()
    actual, _ = predict_return(values, path)
    assert actual == pytest.approx(model.predict(test[FEATURES].iloc[:1])[0])


@pytest.mark.parametrize("kind", ["missing", "extra", "nan", "inf", "text", "none"])
def test_prediction_invalid_inputs(kind):
    values = dict.fromkeys(FEATURES, 0.1)
    if kind == "missing":
        values.pop("nasdaq_ret")
    elif kind == "extra":
        values["gift_ret"] = 0.2
    else:
        values["nasdaq_ret"] = {"nan": np.nan, "inf": np.inf, "text": "bad", "none": None}[kind]
    with pytest.raises(ValueError):
        predict_return(values, "must_not_be_loaded.joblib")


def test_metric_values_and_constant_correlation():
    m = regression_metrics([1., -1.], [0.5, -0.5])
    assert m["mae"] == 0.5
    assert m["rmse"] == 0.5
    assert m["r2"] == 0.75
    assert m["directional_accuracy"] == 1
    assert regression_metrics([1, 2, 3], [0, 0, 0])["correlation"] is None


def test_experiment_reproducible_baselines(observations, tmp_path):
    data, model_path, results = tmp_path / "observations.csv", tmp_path / "model.joblib", tmp_path / "results"
    observations.to_csv(data, index=False)
    model, metadata, report = run_training(data, model_path, results)
    predicted = pd.read_csv(results / "test_predictions.csv")
    train, test = chronological_split(load_market_data(data))
    assert predicted.date.tolist() == test.date.dt.strftime("%Y-%m-%d").tolist()
    assert np.allclose(predicted.training_mean, train[TARGET].mean())
    assert len(pd.read_csv(results / "baseline_comparison.csv")) == 5
    assert metadata["train"]["rows"] == 48 and metadata["test"]["rows"] == 12
    assert evaluate_saved(data, model_path, results) == report
    data.write_text(data.read_text() + "\n")
    with pytest.raises(ValueError, match="differs"):
        evaluate_saved(data, model_path, results)


def test_raw_infinity_rejected():
    frame = pd.DataFrame({"date": ["2020-01-01"], "open": [1], "high": [np.inf], "low": [1], "close": [1]})
    with pytest.raises(ValueError, match="finite"):
        validate_raw_ohlc(frame)
