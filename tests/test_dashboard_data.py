import json

import pandas as pd

from bankrec.dashboard_data import available_runs, load_run, product_metrics


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
    product = {"prevalence": 0.01, "average_precision": 0.02, "roc_auc": 0.6}
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
    assert table.iloc[0]["baseline_ap"] == 0.02
    assert table.iloc[0]["transformer_ap"] == 0.03
