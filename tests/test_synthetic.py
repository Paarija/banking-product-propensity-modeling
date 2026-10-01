import numpy as np
import pandas as pd

from bankrec.data import prepare
from bankrec.splitting import make_split
from bankrec.synthetic import make_synthetic_banking_data


def test_synthetic_data_is_deterministic_and_valid():
    first = make_synthetic_banking_data(clients=50, months=4, seed=7)
    second = make_synthetic_banking_data(clients=50, months=4, seed=7)
    for left, right in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(left, right)
    transactions, targets, splits = first
    assert len(targets) == 200
    assert targets.filter(like="target_").isin([0, 1]).all().all()
    assert transactions["client_id"].nunique() == 50
    assert set(splits["fold"]) == set(range(5))


def test_time_and_client_splits_answer_different_questions():
    data = prepare(*make_synthetic_banking_data(clients=50, months=4), max_events=8)
    time_split = make_split(data, "time")
    client_split = make_split(data, "client")

    months = pd.to_datetime(np.asarray(data.months))
    assert months[time_split.train].max() < months[time_split.validation].min()
    assert months[time_split.validation].max() < months[time_split.test].min()

    clients = np.asarray(data.client_ids)
    assert not set(clients[client_split.train]) & set(clients[client_split.validation])
    assert not set(clients[client_split.train]) & set(clients[client_split.test])
