"""Reproducible historical experiment and saved-model holdout evaluation."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess

import joblib
import numpy as np
import pandas as pd

from .config import DATA_PATH, MODEL_PATH, RESULTS_DIR, RAW_DIR, PROJECT_ROOT
from .data_loader import load_market_data
from .evaluate import regression_metrics
from .features import FEATURES, TARGET
from .train import chronological_split, train_model, save_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def evaluate_experiment(model, train, test, output_dir):
    """Baseline parameters use training data only; all predictions share dates."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    majority = float(np.sign(train[TARGET]).mode().iloc[0])
    # A constant signed training magnitude makes the direction benchmark evaluable
    # for regression too, without choosing its magnitude on the holdout.
    majority_value = majority * float(train[TARGET].abs().mean())
    predictions = pd.DataFrame({"date": test["date"].to_numpy(), "actual": test[TARGET].to_numpy(),
        "ridge": model.predict(test[FEATURES]), "zero": 0.0,
        "training_mean": float(train[TARGET].mean()),
        "previous_session": test["sector_prev_ret"].to_numpy(),
        "majority_direction": majority_value})
    rows = [{"model": name, **regression_metrics(predictions.actual, predictions[name])}
            for name in predictions.columns[2:]]
    table = pd.DataFrame(rows)
    table.to_csv(output / "baseline_comparison.csv", index=False)
    predictions.to_csv(output / "test_predictions.csv", index=False)
    report = {"train": regression_metrics(train[TARGET], model.predict(train[FEATURES])),
              "test": rows[0], "baselines": rows[1:], "majority_direction": majority,
              "majority_prediction": majority_value,
              "test_periods": []}
    for indices in np.array_split(np.arange(len(test)), min(3, len(test))):
        part = predictions.iloc[indices]
        report["test_periods"].append({"start": str(part.date.iloc[0].date()),
            "end": str(part.date.iloc[-1].date()), "rows": len(part),
            **regression_metrics(part.actual, part.ridge)})
    json_write(output / "metrics.json", report)
    pd.DataFrame({"feature": FEATURES, "standardized_coefficient": model.named_steps["ridge"].coef_}).to_csv(
        output / "coefficients.csv", index=False)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return report


def run_training(data_path=DATA_PATH, model_path=MODEL_PATH, output_dir=RESULTS_DIR):
    data_path, output_dir = Path(data_path).resolve(), Path(output_dir)
    if "synthetic" in data_path.name.lower():
        raise ValueError("Historical training requires real observations; synthetic inputs are excluded")
    frame = load_market_data(data_path)
    train, test = chronological_split(frame)
    model = train_model(train)
    report = evaluate_experiment(model, train, test, output_dir)
    save_model(model, model_path)
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    def scope(part):
        return {"start": str(part.date.iloc[0].date()), "end": str(part.date.iloc[-1].date()), "rows": len(part)}
    source_hashes = {p.name: sha256(p) for p in sorted(RAW_DIR.glob("*.csv")) if p.stem in ("nasdaq", "sp500", "dow", "vix", "usdinr", "nifty_it")}
    metadata = {"data_kind": "historical", "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path), "dataset_sha256": sha256(data_path), "raw_sha256": source_hashes,
        "model_sha256": sha256(model_path), "features": FEATURES,
        "target": "100 * (NIFTY IT close - open) / open; percent units",
        "usable_rows": len(frame), "train": scope(train), "test": scope(test),
        "selected_alpha": float(model.named_steps["ridge"].alpha), "cv": model.cv_report_,
        "git_revision": revision, "working_tree_dirty": dirty,
        "source_code_sha256": {str(p.relative_to(PROJECT_ROOT)): sha256(p) for p in [PROJECT_ROOT / "main.py", *sorted(PROJECT_ROOT.glob("src/**/*.py"))]},
        "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in
            ("numpy", "pandas", "scikit-learn", "joblib", "yfinance")},
        "raw_rows": len(pd.read_csv(data_path)),
        "excluded_rows": len(pd.read_csv(data_path)) - len(frame),
        "availability_policy": "US source date < Indian date; FX source date < Indian date minus one calendar day; inconsistent FX endpoints excluded; max source age US 7 / FX 10 days",
        "limitations": "Date-based historical proxy, not verified point-in-time 09:10 IST data. FX vendor bar timezone and revisions uncertified. Not a live forecast or backtest."}
    json_write(output_dir / "model_metadata.json", metadata)
    print(f"Saved historical model: {model_path}")
    return model, metadata, report


def evaluate_saved(data_path=DATA_PATH, model_path=MODEL_PATH, output_dir=RESULTS_DIR):
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "model_metadata.json").read_text(encoding="utf-8"))
    if sha256(data_path) != metadata["dataset_sha256"] or sha256(model_path) != metadata["model_sha256"]:
        raise ValueError("Dataset or model differs from the recorded experiment; train a new experiment")
    train, test = chronological_split(load_market_data(data_path))
    return evaluate_experiment(joblib.load(model_path), train, test, output_dir)


def create_charts(output_dir=RESULTS_DIR):
    """Plot saved holdout predictions without fitting or selecting a model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output_dir = Path(output_dir)
    frame = pd.read_csv(output_dir / "test_predictions.csv", parse_dates=["date"])
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(frame.date, frame.actual, label="Actual", alpha=0.65, linewidth=0.8)
    axes[0].plot(frame.date, frame.ridge, label="Ridge", linewidth=0.9)
    axes[0].set_ylabel("Intraday return (%)")
    axes[0].legend()
    for name in ("ridge", "majority_direction"):
        correct = (np.sign(frame[name]) == np.sign(frame.actual)).astype(float)
        axes[1].plot(frame.date, correct.expanding().mean() * 100, label=name)
    axes[1].set_ylabel("Cumulative accuracy (%)")
    axes[1].legend()
    fig.suptitle("Historical NIFTY IT holdout; research predictions, not trading returns")
    fig.tight_layout()
    fig.savefig(output_dir / "holdout_diagnostics.png", dpi=150)
    plt.close(fig)
