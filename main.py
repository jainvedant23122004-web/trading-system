"""Command-line entry point for acquisition, preprocessing, and modeling."""

import argparse

from src.config import DATA_END_DATE, DATA_PATH, DATA_START_DATE, MODEL_PATH, RESULTS_DIR
from src.features import FEATURES
from src.research import run_training, evaluate_saved, create_charts
from src.pipeline.build_dataset import build_dataset
from src.pipeline.download_data import download_all
from src.predict import predict_return


def main() -> None:
    """Run training or a clearly synthetic input demonstration."""
    parser = argparse.ArgumentParser(description="Experimental Nifty IT Ridge model")
    commands = parser.add_subparsers(dest="command", required=True)
    train_command = commands.add_parser("train", help="Train and evaluate on a complete processed CSV")
    train_command.add_argument("--data", default=DATA_PATH, help="Processed CSV path")
    train_command.add_argument("--model", default=MODEL_PATH)
    train_command.add_argument("--results", default=RESULTS_DIR)
    train_command.add_argument("--charts", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Re-evaluate the recorded historical holdout")
    evaluate.add_argument("--data", default=DATA_PATH)
    evaluate.add_argument("--model", default=MODEL_PATH)
    evaluate.add_argument("--results", default=RESULTS_DIR)
    predict = commands.add_parser("predict", help="Predict from manually supplied features; no live data acquisition")
    predict.add_argument("--model", default=MODEL_PATH)
    predict.add_argument("--demo", action="store_true", help="Use illustrative manual inputs")
    for feature in FEATURES:
        predict.add_argument("--" + feature.replace("_", "-"), dest=feature, type=float)
    download = commands.add_parser("download-data", help="Download or reuse six Yahoo daily series")
    download.add_argument("--refresh", action="store_true", help="Replace cached raw series")
    download.add_argument("--start", default=DATA_START_DATE, help="Acquisition start date YYYY-MM-DD")
    download.add_argument("--end", default=DATA_END_DATE, help="Exclusive acquisition end date YYYY-MM-DD")
    build = commands.add_parser("build-dataset", help="Build from existing raw CSV files")
    build.add_argument("--require-complete", action="store_true", help="Fail if any feature is missing")
    args = parser.parse_args()

    if args.command == "download-data":
        download_all(start=args.start, end=args.end, refresh=args.refresh)
    elif args.command == "build-dataset":
        build_dataset(require_complete=args.require_complete)
    elif args.command == "train":
        run_training(args.data, args.model, args.results)
        if args.charts:
            create_charts(args.results)
    elif args.command == "evaluate":
        evaluate_saved(args.data, args.model, args.results)
    else:
        example_values = {
            "nasdaq_ret": 0.35,
            "sp500_ret": 0.22,
            "dow_ret": 0.10,
            "vix_ret": -0.40,
            "usdinr_ret": 0.05,
            "sector_prev_ret": -0.12,
        }
        supplied = {feature: getattr(args, feature) for feature in FEATURES}
        if args.demo and any(value is not None for value in supplied.values()):
            parser.error("Use --demo or all six manual features, without mixing them")
        if not args.demo and any(value is None for value in supplied.values()):
            parser.error("Supply all six feature flags, or use --demo")
        print("Manual-input demonstration; not an automatically acquired current-day forecast.")
        try:
            prediction, classification = predict_return(example_values if args.demo else supplied, args.model)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        print(f"NIFTY IT Prediction\nPredicted intraday return: {prediction:+.2f}%")
        print(f"Market outlook: {classification}\nModel: Ridge Regression")


if __name__ == "__main__":
    main()
