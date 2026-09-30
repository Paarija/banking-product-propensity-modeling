import pandas as pd
import pytest

from bankrec.data import PRODUCT_COLUMNS, prepare
from bankrec.loader import load_santander


def test_monthly_additions_and_time_split(tiny_santander):
    data = prepare(tiny_santander, max_events=8)
    assert data.labels.shape == (32, 24)
    assert data.folds.tolist()[:4] == [0, 0, 3, 4]
    assert data.labels[0, 0] == 1  # Client 0 adds product 0 after January.
    assert data.labels[1, 0] == 0  # Already owned: not a new addition.
    assert data.owned_products[1, 0]
    assert data.attention_mask[0].sum() == 1  # No products owned in January.
    assert data.attention_mask[1].sum() == 2


def test_gaps_are_not_next_month(tiny_santander):
    june = tiny_santander[tiny_santander.fecha_dato == "2015-05-28"].copy()
    june["fecha_dato"] = "2015-06-28"
    panel = pd.concat([tiny_santander, june])
    panel = panel[panel.fecha_dato != "2015-03-28"]
    data = prepare(panel)
    assert len(data.labels) == 24
    assert set(data.months) == {"2015-01", "2015-04", "2015-05"}


def test_duplicate_customer_month_rejected(tiny_santander):
    panel = pd.concat([tiny_santander, tiny_santander.iloc[:1]])
    with pytest.raises(ValueError, match="duplicate"):
        prepare(panel)


def test_invalid_product_rejected(tiny_santander):
    tiny_santander.loc[0, PRODUCT_COLUMNS[0]] = 2
    with pytest.raises(ValueError, match="binary"):
        prepare(tiny_santander)


def test_kaggle_csv_loader(tiny_santander, tmp_path):
    tiny_santander.to_csv(tmp_path / "train_ver2.csv", index=False)
    loaded = load_santander(tmp_path, max_clients=8)
    assert len(loaded) == len(tiny_santander)
    assert loaded.ncodpers.nunique() == 8
