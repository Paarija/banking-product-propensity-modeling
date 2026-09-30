from bankrec.data import prepare
from bankrec.experiment import run_experiment


def test_end_to_end_smoke(tiny_santander, tmp_path):
    data = prepare(tiny_santander, max_events=8)
    report = run_experiment(data, epochs=1, batch_size=8, output_dir=tmp_path)
    assert report["dataset"] == "Kaggle Santander Product Recommendation"
    assert report["baseline"]["test"]["examples"] == 8
    assert report["transformer"]["test"]["examples"] == 8
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "transformer.pt").is_file()
