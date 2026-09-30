import pandas as pd
import pytest

from bankrec.data import prepare


def test_future_transactions_are_excluded(tiny_mbd):
    data = prepare(*tiny_mbd, max_events=4)
    assert data.attention_mask.sum(axis=1).tolist() == [3] * 10  # CLS + Feb 10 + Feb 28
    assert data.numeric.shape == (10, 5, 2)
    assert data.labels.shape == (10, 4)


def test_empty_history_is_valid(tiny_mbd):
    trx, targets, splits = tiny_mbd
    trx = trx[trx.client_id != "client-0"]
    data = prepare(trx, targets, splits, max_events=2)
    assert data.attention_mask[0].tolist() == [1, 0, 0]
    assert data.baseline_features.iloc[0].transaction_count == 0


def test_bad_targets_are_rejected(tiny_mbd):
    trx, targets, splits = tiny_mbd
    targets = targets.copy()
    targets.loc[0, "target_1"] = 2
    with pytest.raises(ValueError, match="binary"):
        prepare(trx, targets, splits)


def test_duplicate_client_fold_is_rejected(tiny_mbd):
    trx, targets, splits = tiny_mbd
    with pytest.raises(ValueError, match="one fold"):
        prepare(trx, targets, pd.concat([splits, splits.iloc[:1]]))
