"""Create static, recruiter-friendly charts from an experiment run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .dashboard_data import model_labels, model_reports, product_metrics


def create_charts(run_dir: str | Path, output_dir: str | Path | None = None) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on an optional extra
        raise RuntimeError('Install chart dependencies with: pip install -e ".[visuals]"') from exc

    run_dir = Path(run_dir)
    output = Path(output_dir) if output_dir else run_dir / "charts"
    output.mkdir(parents=True, exist_ok=True)
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    labels = model_labels(report)
    models = model_reports(report)
    metrics = product_metrics(report, "test")
    created = []

    summary = pd.Series(
        {
            labels[name]: values["test"].get("macro_average_precision")
            for name, values in models.items()
        }
    ).dropna()
    axis = summary.sort_values().plot.barh(color="#2f6fed", figsize=(8, 4))
    axis.set_title("Test macro average precision")
    axis.set_xlabel("Average precision")
    axis.set_ylabel("")
    axis.figure.tight_layout()
    path = output / "model_comparison.png"
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)
    created.append(path)

    pivot = metrics.pivot(index="product", columns="model", values="average_precision")
    axis = pivot.plot.bar(figsize=(10, 5))
    axis.set_title("Average precision by anonymized product")
    axis.set_xlabel("")
    axis.set_ylabel("Average precision")
    axis.tick_params(axis="x", rotation=0)
    axis.figure.tight_layout()
    path = output / "product_average_precision.png"
    axis.figure.savefig(path, dpi=160)
    plt.close(axis.figure)
    created.append(path)

    importance_path = run_dir / "feature_importance.csv"
    if importance_path.is_file():
        importance = pd.read_csv(importance_path)
        aggregate = (
            importance.groupby("feature")["random_forest_importance"]
            .mean()
            .nlargest(10)
            .sort_values()
        )
        axis = aggregate.plot.barh(color="#00a884", figsize=(8, 5))
        axis.set_title("Top tabular features across products")
        axis.set_xlabel("Mean Random Forest importance")
        axis.set_ylabel("")
        axis.figure.tight_layout()
        path = output / "feature_importance.png"
        axis.figure.savefig(path, dpi=160)
        plt.close(axis.figure)
        created.append(path)
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description="Create charts from saved experiment artifacts")
    parser.add_argument("--run-dir", default="artifacts/demo")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    for path in create_charts(args.run_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
