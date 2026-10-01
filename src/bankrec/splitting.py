"""Explicit evaluation splits for future-period and unseen-client questions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import PreparedData


@dataclass(frozen=True)
class EvaluationSplit:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    description: str


def make_split(data: PreparedData, strategy: str) -> EvaluationSplit:
    if strategy == "client":
        train = ~np.isin(data.folds, [3, 4])
        validation = data.folds == 3
        test = data.folds == 4
        description = "client-disjoint: folds 0-2 train, 3 validation, 4 test"
        train_clients = set(np.asarray(data.client_ids)[train])
        held_out_clients = set(np.asarray(data.client_ids)[validation | test])
        if train_clients & held_out_clients:
            raise ValueError("client leakage across client-disjoint splits")
    elif strategy == "time":
        periods = pd.to_datetime(pd.Series(data.months), errors="raise").dt.to_period("M")
        unique_periods = sorted(periods.unique())
        if len(unique_periods) < 3:
            raise ValueError("time split needs at least three reporting months")
        validation_period, test_period = unique_periods[-2:]
        train = (periods < validation_period).to_numpy()
        validation = (periods == validation_period).to_numpy()
        test = (periods == test_period).to_numpy()
        description = (
            f"time-based: through {unique_periods[-3]} train, "
            f"{validation_period} validation, {test_period} test"
        )
    else:
        raise ValueError("split strategy must be 'time' or 'client'")
    if not train.any() or not validation.any() or not test.any():
        raise ValueError("train, validation, and test must all contain examples")
    return EvaluationSplit(train, validation, test, description)
