"""Command-line entry point for the Kaggle Santander experiment."""

from __future__ import annotations

import argparse
import json

from .data import prepare
from .experiment import run_experiment
from .loader import load_santander


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle Santander product ranking experiment")
    parser.add_argument(
        "--data-dir", default="data/raw", help="directory containing Kaggle train_ver2.csv"
    )
    parser.add_argument("--max-clients", type=int, default=1000)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="artifacts")
    args = parser.parse_args()
    panel = load_santander(args.data_dir, max_clients=args.max_clients)
    data = prepare(panel, max_events=args.max_events)
    report = run_experiment(
        data, epochs=args.epochs, batch_size=args.batch_size, output_dir=args.output_dir
    )
    print(
        json.dumps(
            {
                "baseline_test": report["baseline"]["test"],
                "transformer_test": report["transformer"]["test"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
