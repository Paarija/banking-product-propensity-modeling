import pandas as pd

from bankrec.data import prepare
from bankrec.experiment import run_experiment


def test_end_to_end_smoke(tiny_mbd, tmp_path):
    data = prepare(*tiny_mbd, max_events=3)
    report = run_experiment(data, epochs=1, batch_size=4, output_dir=tmp_path)
    assert report["baseline"]["test"]["examples"] == 2
    assert report["transformer"]["test"]["examples"] == 2
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "transformer.pt").is_file()
    predictions = pd.read_parquet(tmp_path / "predictions.parquet")
    assert len(predictions) == 4
    assert set(predictions["split"]) == {"validation", "test"}
    assert "client_id" not in predictions.columns
    assert predictions["actual_product_1"].isin([0, 1]).all()
