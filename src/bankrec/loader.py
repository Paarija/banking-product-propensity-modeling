"""Read a client sample from the Kaggle Santander competition CSV in chunks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import PRODUCT_COLUMNS


def load_santander(root: str | Path, *, max_clients: int = 1000, seed: int = 42) -> pd.DataFrame:
    root = Path(root)
    candidates = [
        root / "train_ver2.csv",
        root / "train.csv",
        root / "train_ver2.csv.zip",
        root / "train.csv.zip",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(
            "Download the Santander Product Recommendation competition data from Kaggle "
            "after accepting its rules; put train_ver2.csv (or train.csv) in " + str(root)
        )
    if max_clients < 1:
        raise ValueError("max_clients must be positive")
    header = pd.read_csv(path, nrows=0).columns
    required = {"ncodpers", "fecha_dato", *PRODUCT_COLUMNS}
    missing = required - set(header)
    if missing:
        raise ValueError(f"Santander CSV missing columns: {', '.join(sorted(missing))}")
    clients = set()
    for chunk in pd.read_csv(path, usecols=["ncodpers"], dtype=str, chunksize=200_000):
        clients.update(chunk["ncodpers"].dropna().tolist())
    clients = np.array(sorted(clients))
    if len(clients) > max_clients:
        clients = np.random.default_rng(seed).choice(clients, max_clients, replace=False)
    selected = set(clients)
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=["ncodpers", "fecha_dato", *PRODUCT_COLUMNS],
        dtype=str,
        chunksize=200_000,
        low_memory=False,
    ):
        subset = chunk[chunk["ncodpers"].isin(selected)]
        if not subset.empty:
            chunks.append(subset)
    if not chunks:
        raise ValueError("no sampled customers found in Santander CSV")
    return pd.concat(chunks, ignore_index=True)
