import json

import pandas as pd

from bankrec.dashboard_data import (
    available_runs,
    load_run,
    precision_recall_data,
    product_metrics,
)


def test_discovers_and_reads_local_runs(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    report = {"sampled_clients": 10}
    (run / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    expected = pd.DataFrame({"example_id": [0], "split": ["test"]})
    expected.to_parquet(run / "predictions.parquet", index=False)

    assert available_runs(tmp_path) == [run]
    actual_report, actual_predictions = load_run(run)
    assert actual_report == report
    pd.testing.assert_frame_equal(actual_predictions, expected)


def test_old_run_without_predictions_is_supported(tmp_path):
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    report, predictions = load_run(tmp_path)
    assert report == {}
    assert predictions is None


def test_product_comparison_table():
    product = {
        "prevalence": 0.01,
        "average_precision": 0.02,
        "roc_auc": 0.6,
        "lift": {"top_10pct": 2.0},
    }
    report = {
        "baseline": {"test": {"per_product": {"product_1": product}}},
        "transformer": {
            "test": {
                "per_product": {
                    "product_1": {"prevalence": 0.01, "average_precision": 0.03, "roc_auc": 0.7}
                }
            }
        },
    }
    table = product_metrics(report, "test")
    assert table["model"].tolist() == ["Gradient Boosting", "Transaction Transformer"]
    assert table["average_precision"].tolist() == [0.02, 0.03]
    assert table.iloc[0]["top_10pct_lift"] == 2.0


def test_precision_recall_data_uses_saved_scores():
    predictions = pd.DataFrame(
        {
            "split": ["test"] * 4,
            "actual_product_1": [0, 1, 0, 1],
            "logistic_regression_product_1": [0.1, 0.8, 0.2, 0.7],
        }
    )
    curve = precision_recall_data(predictions, "test", "product_1", ["logistic_regression"])
    assert not curve.empty
    assert curve["precision"].between(0, 1).all()
    assert curve["recall"].between(0, 1).all()
