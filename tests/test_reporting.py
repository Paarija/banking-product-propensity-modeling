import json

import pandas as pd
import pytest

from bankrec.reporting import create_charts


def test_create_charts(tmp_path):
    pytest.importorskip("matplotlib")
    product_metrics = {
        f"product_{index}": {
            "prevalence": 0.1,
            "average_precision": 0.2 + index / 100,
            "roc_auc": 0.7,
            "lift": {"top_10pct": 2.0},
        }
        for index in range(1, 5)
    }
    evaluation = {
        "macro_average_precision": 0.22,
        "per_product": product_metrics,
        "examples": 10,
        "positive_clients": 2,
        "hit_at_1_among_buyers": 0.5,
        "recall_at_2_among_buyers": 1.0,
    }
    report = {"models": {"logistic_regression": {"test": evaluation}}}
    (tmp_path / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    pd.DataFrame(
        {
            "product": ["product_1"],
            "feature": ["transaction_count"],
            "random_forest_importance": [0.5],
        }
    ).to_csv(tmp_path / "feature_importance.csv", index=False)

    created = create_charts(tmp_path)
    assert len(created) == 3
    assert all(path.is_file() for path in created)
