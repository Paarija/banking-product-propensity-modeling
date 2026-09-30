"""Read only a sampled set of MBD-mini clients from partitioned Parquet files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def load_mbd_mini(root: str | Path, *, max_clients: int = 5000, seed: int = 42):
    root = Path(root)
    target_dir, split_dir, trx_dir = (
        root / "targets",
        root / "client_split",
        root / "detail" / "trx",
    )
    missing = [str(path) for path in (target_dir, split_dir, trx_dir) if not path.is_dir()]
    if missing:
        raise FileNotFoundError("Extract MBD-mini archives first; missing: " + ", ".join(missing))
    if max_clients < 1:
        raise ValueError("max_clients must be positive")

    targets = pd.read_parquet(target_dir)
    splits = pd.read_parquet(split_dir)
    # Sample clients, never rows or positives, so all months stay together and prevalence is not engineered.
    clients = np.sort(targets["client_id"].unique())
    if len(clients) > max_clients:
        clients = np.random.default_rng(seed).choice(clients, size=max_clients, replace=False)
    selected = set(clients)
    targets = targets[targets["client_id"].isin(selected)].copy()
    splits = splits[splits["client_id"].isin(selected)].copy()

    dataset = ds.dataset(trx_dir, format="parquet", partitioning="hive")
    table = dataset.to_table(
        columns=["client_id", "event_time", "event_type", "amount"],
        filter=ds.field("client_id").isin(list(selected)),
    )
    return table.to_pandas(), targets, splits
