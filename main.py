"""Command-line entry point for model training and a mock prediction."""

import argparse

from src.config import DATA_PATH, MODEL_PATH
from src.data_loader import load_market_data
from src.evaluate import evaluate_model
from src.predict import predict_return
from src.train import chronological_split, print_training_report, save_model, train_model


def main() -> None:
    """Run training or a clearly synthetic input demonstration."""
    parser = argparse.ArgumentParser(description="Experimental Nifty IT Ridge model")
    parser.add_argument("command", choices=("train", "predict"))
    args = parser.parse_args()

    if args.command == "train":
        data = load_market_data(DATA_PATH)
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
