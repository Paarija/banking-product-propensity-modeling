"""Data contracts and leakage-aware history construction for MBD-mini."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

TARGET_COLUMNS = tuple(f"target_{i}" for i in range(1, 5))
TRANSACTION_COLUMNS = ("client_id", "event_time", "event_type", "amount")
TARGET_REQUIRED = ("client_id", "mon", *TARGET_COLUMNS)
SPLIT_REQUIRED = ("client_id", "fold")


@dataclass(frozen=True)
class PreparedData:
    event_types: np.ndarray  # [customers, sequence_length + CLS]
    numeric: np.ndarray  # [customers, sequence_length + CLS, 2]
    attention_mask: np.ndarray  # [customers, sequence_length + CLS]
    labels: np.ndarray  # [customers, 4]
    folds: np.ndarray  # [customers]
    client_ids: tuple[str, ...]
    months: tuple[str, ...]
    baseline_features: pd.DataFrame
    vocab_size: int


def _require(frame: pd.DataFrame, fields: tuple[str, ...], name: str) -> None:
    missing = set(fields) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing columns: {', '.join(sorted(missing))}")


def read_frame(path: str | Path) -> pd.DataFrame:
    """Read one CSV or Parquet table; downloaded dataset files are never committed."""
    path = Path(path)
    if path.is_dir() or path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Expected .csv or .parquet: {path}")


def prepare(
    transactions: pd.DataFrame,
    targets: pd.DataFrame,
    splits: pd.DataFrame,
    *,
    max_events: int = 64,
) -> PreparedData:
    """Use only events strictly before each target month; split by client ID."""
    if max_events < 1:
        raise ValueError("max_events must be positive")
    _require(transactions, TRANSACTION_COLUMNS, "transactions")
    _require(targets, TARGET_REQUIRED, "targets")
    _require(splits, SPLIT_REQUIRED, "splits")

    trx = transactions.loc[:, TRANSACTION_COLUMNS].copy()
    trx["client_id"] = trx["client_id"].astype(str)
    trx["event_time"] = pd.to_datetime(trx["event_time"], errors="raise", utc=True)
    trx["event_type"] = pd.to_numeric(trx["event_type"], errors="raise").astype(int)
    trx["amount"] = pd.to_numeric(trx["amount"], errors="raise").astype(float)
    if not np.isfinite(trx["amount"]).all():
        raise ValueError("transaction amounts must be finite")

    target = targets.loc[:, TARGET_REQUIRED].copy()
    target["client_id"] = target["client_id"].astype(str)
    # MBD's `mon` is the reporting date (month-end); labels cover the following month.
    # Include transactions on the reporting day, but none after it.
    target["cutoff"] = pd.to_datetime(target["mon"], errors="raise", utc=True) + np.timedelta64(
        1, "D"
    )
    if target.duplicated(["client_id", "cutoff"]).any():
        raise ValueError("duplicate client-month target rows")
    for col in TARGET_COLUMNS:
        target[col] = pd.to_numeric(target[col], errors="raise").astype(int)
        if not target[col].isin([0, 1]).all():
            raise ValueError(f"{col} must be binary")

    split = splits.loc[:, SPLIT_REQUIRED].copy()
    split["client_id"] = split["client_id"].astype(str)
    if split["client_id"].duplicated().any():
        raise ValueError("one fold per client is required")
    target = target.merge(split, on="client_id", how="left", validate="many_to_one")
    if target["fold"].isna().any():
        raise ValueError("every target client needs a fold")
    target = target.sort_values(["client_id", "cutoff"], kind="stable").reset_index(drop=True)
    trx = trx.sort_values(["client_id", "event_time"], kind="stable")

    # A local vocabulary is safe because event-type codes are input categories, never labels.
    categories = sorted(trx["event_type"].unique())
    event_to_id = {value: i + 2 for i, value in enumerate(categories)}
    grouped = {cid: group for cid, group in trx.groupby("client_id", sort=False)}
    n, width = len(target), max_events + 1
    event_types = np.zeros((n, width), dtype=np.int64)
    numeric = np.zeros((n, width, 2), dtype=np.float32)
    masks = np.zeros((n, width), dtype=np.int64)
    features = []
    event_types[:, 0] = 1  # CLS; zero is padding.
    masks[:, 0] = 1

    for i, row in enumerate(target.itertuples(index=False)):
        history = grouped.get(row.client_id)
        if history is None:
            history = trx.iloc[:0]
        history = history.loc[history["event_time"] < row.cutoff].tail(max_events)
        count = len(history)
        if count:
            event_types[i, 1 : count + 1] = [event_to_id[x] for x in history["event_type"]]
            amounts = history["amount"].to_numpy(dtype=np.float64)
            days_ago = (row.cutoff - history["event_time"]).dt.total_seconds().to_numpy() / 86400
            numeric[i, 1 : count + 1, 0] = np.sign(amounts) * np.log1p(np.abs(amounts)) / 10
            numeric[i, 1 : count + 1, 1] = np.minimum(days_ago, 365) / 365
            masks[i, 1 : count + 1] = 1
        else:
            amounts = np.array([], dtype=np.float64)
            days_ago = np.array([], dtype=np.float64)
        features.append(
            {
                "transaction_count": count,
                "total_amount": float(amounts.sum()),
                "mean_amount": float(amounts.mean()) if count else 0.0,
                "std_amount": float(amounts.std()) if count else 0.0,
                "days_since_last": float(days_ago[-1]) if count else 365.0,
                "unique_event_types": int(history["event_type"].nunique()),
            }
        )

    return PreparedData(
        event_types=event_types,
        numeric=numeric,
        attention_mask=masks,
        labels=target.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float32),
        folds=target["fold"].to_numpy(dtype=np.int64),
        client_ids=tuple(target["client_id"].tolist()),
        months=tuple(target["mon"].astype(str).tolist()),
        baseline_features=pd.DataFrame(features),
        vocab_size=len(event_to_id) + 2,
    )
