"""Local, read-only helpers for the Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import precision_recall_curve


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


def model_reports(report: dict) -> dict:
    if "models" in report:
        return report["models"]
    models = {"gradient_boosting": report["baseline"]}
    if report.get("transformer") is not None:
        models["transformer"] = report["transformer"]
    return models


def model_labels(report: dict) -> dict[str, str]:
    defaults = {
        "logistic_regression": "Logistic Regression",
        "random_forest": "Random Forest",
        "gradient_boosting": "Gradient Boosting",
        "transformer": "Transaction Transformer",
    }
    return {
        name: report.get("model_labels", {}).get(name, defaults[name])
        for name in model_reports(report)
    }


def product_metrics(report: dict, split: str) -> pd.DataFrame:
    models = model_reports(report)
    labels = model_labels(report)
    rows = []
    for model_name, model_report in models.items():
        for product, values in model_report[split]["per_product"].items():
            rows.append(
                {
                    "product": product.replace("_", " ").title(),
                    "model_key": model_name,
                    "model": labels[model_name],
                    "prevalence": values["prevalence"],
                    "average_precision": values["average_precision"],
                    "roc_auc": values["roc_auc"],
                    "top_10pct_lift": values.get("lift", {}).get("top_10pct"),
                }
            )
    return pd.DataFrame(rows)


def precision_recall_data(
    predictions: pd.DataFrame, split: str, product: str, model_names: list[str]
) -> pd.DataFrame:
    frame = predictions[predictions["split"] == split]
    actual = frame[f"actual_{product}"].to_numpy()
    rows = []
    for model_name in model_names:
        column = f"{model_name}_{product}"
        if column not in frame or len(set(actual)) < 2:
            continue
        precision, recall, _ = precision_recall_curve(actual, frame[column].to_numpy())
        rows.extend(
            {"recall": float(x), "precision": float(y), "model_key": model_name}
            for x, y in zip(recall, precision, strict=True)
        )
    return pd.DataFrame(rows)


def load_feature_importance(folder: str | Path) -> pd.DataFrame | None:
    path = Path(folder) / "feature_importance.csv"
    return pd.read_csv(path) if path.is_file() else None
