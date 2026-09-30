"""Local, read-only helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def available_runs(root: str | Path) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        return []
    folders = [root, *(path for path in root.iterdir() if path.is_dir())]
    folders = [path for path in folders if (path / "metrics.json").is_file()]
    return sorted(folders, key=lambda path: (path / "metrics.json").stat().st_mtime, reverse=True)


def load_run(folder: str | Path) -> tuple[dict, pd.DataFrame | None]:
    folder = Path(folder)
    report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    predictions_path = folder / "predictions.parquet"
    predictions = pd.read_parquet(predictions_path) if predictions_path.is_file() else None
    return report, predictions


def product_metrics(report: dict, split: str) -> pd.DataFrame:
    baseline = report["baseline"][split]["per_product"]
    transformer = report["transformer"][split]["per_product"]
    return pd.DataFrame(
        [
            {
                "product": product.replace("_", " ").title(),
                "prevalence": values["prevalence"],
                "baseline_ap": values["average_precision"],
                "transformer_ap": transformer[product]["average_precision"],
                "baseline_auc": values["roc_auc"],
                "transformer_auc": transformer[product]["roc_auc"],
            }
            for product, values in baseline.items()
        ]
    )
