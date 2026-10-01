"""Command-line entry point for the MBD-mini experiment."""

from __future__ import annotations

import argparse
import json

from .data import prepare
from .experiment import run_experiment
from .loader import load_mbd_mini
from .synthetic import make_synthetic_banking_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict next-month banking product uptake from transaction history"
    )
    parser.add_argument(
        "--data-dir", default="data/raw", help="directory with extracted MBD-mini archives"
    )
    parser.add_argument("--max-clients", type=int, default=1000)
    parser.add_argument("--max-events", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output-dir", default="artifacts")
    parser.add_argument(
        "--split-strategy",
        choices=["time", "client"],
        default="time",
        help="future-month backtest (time) or unseen-client evaluation (client)",
    )
    parser.add_argument("--demo", action="store_true", help="use fast synthetic data")
    parser.add_argument("--demo-clients", type=int, default=300)
    parser.add_argument(
        "--no-transformer", action="store_true", help="run only the three tabular models"
    )
    args = parser.parse_args()
    if args.demo:
        trx, targets, splits = make_synthetic_banking_data(clients=args.demo_clients)
        dataset_name = "synthetic demo (generated locally; no real customers)"
    else:
        trx, targets, splits = load_mbd_mini(args.data_dir, max_clients=args.max_clients)
        dataset_name = "ai-lab/MBD-mini"
    data = prepare(trx, targets, splits, max_events=args.max_events)
    report = run_experiment(
        data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        split_strategy=args.split_strategy,
        include_transformer=not args.no_transformer,
        dataset_name=dataset_name,
    )
    summary = {
        name: {
            "macro_average_precision": result["test"]["macro_average_precision"],
            "hit_at_1_among_buyers": result["test"]["hit_at_1_among_buyers"],
            "recall_at_2_among_buyers": result["test"]["recall_at_2_among_buyers"],
        }
        for name, result in report["models"].items()
    }
    print(json.dumps({"split": report["split"], "test": summary}, indent=2))


if __name__ == "__main__":
    main()
