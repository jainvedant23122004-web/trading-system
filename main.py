"""Command-line entry point for acquisition, preprocessing, and modeling."""

import argparse

from src.config import DATA_END_DATE, DATA_PATH, DATA_START_DATE, MODEL_PATH
from src.data_loader import load_market_data
from src.evaluate import evaluate_model
from src.pipeline.build_dataset import build_dataset
from src.pipeline.download_data import download_all
from src.predict import predict_return
from src.train import chronological_split, print_training_report, save_model, train_model


def main() -> None:
    """Run training or a clearly synthetic input demonstration."""
    parser = argparse.ArgumentParser(description="Experimental Nifty IT Ridge model")
    commands = parser.add_subparsers(dest="command", required=True)
    train_command = commands.add_parser("train", help="Train and evaluate on a complete processed CSV")
    train_command.add_argument("--data", default=DATA_PATH, help="Processed CSV path")
    commands.add_parser("predict", help="Demonstrate a saved-model prediction")
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
        data = load_market_data(args.data)
        train_data, test_data = chronological_split(data)
        model = train_model(train_data)
        print_training_report(model, train_data, test_data)
        print("\nHoldout evaluation:")
        evaluate_model(model, test_data)
        save_model(model, MODEL_PATH)
        print(f"Saved model: {MODEL_PATH}")
    else:
        example_values = {
            "nasdaq_ret": 0.35,
            "sp500_ret": 0.22,
            "dow_ret": 0.10,
            "vix_ret": -0.40,
            "gift_ret": 0.18,
            "usdinr_ret": 0.05,
            "sector_prev_ret": -0.12,
        }
        print("Demonstration with synthetic feature values; not a live market forecast.")
        prediction, classification = predict_return(example_values, MODEL_PATH)
        print(f"Predicted return: {prediction:+.2f}%")
        print(f"Market bias: {classification}")


if __name__ == "__main__":
    main()
