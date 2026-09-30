import pandas as pd
import pytest


@pytest.fixture
def tiny_mbd():
    clients = [f"client-{i}" for i in range(10)]
    splits = pd.DataFrame({"client_id": clients, "fold": [0, 1, 2, 3, 4] * 2})
    targets = pd.DataFrame(
        {
            "client_id": clients,
            "mon": ["2022-02-28"] * 10,
            "target_1": [0, 1] * 5,
            "target_2": [1, 0] * 5,
            "target_3": [0, 0, 1, 0, 0] * 2,
            "target_4": [0, 1, 0, 0, 0] * 2,
        }
    )
    transactions = pd.DataFrame(
        [
            {"client_id": cid, "event_time": when, "event_type": i % 3, "amount": i + 1}
            for i, cid in enumerate(clients)
            for when in ("2022-02-10", "2022-02-28", "2022-03-01")
        ]
    )
    return transactions, targets, splits
