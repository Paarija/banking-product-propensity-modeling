"""Deterministic synthetic banking histories for a fast, public demo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import TARGET_COLUMNS


def make_synthetic_banking_data(
    *, clients: int = 300, months: int = 8, seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create learnable but non-real transaction histories and next-month labels."""
    if clients < 50 or months < 4:
        raise ValueError("demo needs at least 50 clients and four months")
    rng = np.random.default_rng(seed)
    client_ids = np.array([f"demo-{i:04d}" for i in range(clients)])
    folds = np.arange(clients) % 5
    rng.shuffle(folds)
    affinity = rng.normal(size=(clients, 4))
    activity = rng.lognormal(mean=1.4, sigma=0.45, size=clients)
    start = pd.Period("2023-01", freq="M")
    transactions: list[dict] = []
    targets: list[dict] = []

    for client_index, client_id in enumerate(client_ids):
        running_count = np.zeros(4, dtype=int)
        running_amount = np.zeros(4, dtype=float)
        for month_index in range(months):
            period = start + month_index
            events = max(1, int(rng.poisson(activity[client_index])))
            event_types = rng.choice(8, size=events, replace=True)
            amounts = rng.lognormal(
                mean=3.3 + 0.08 * affinity[client_index, 0], sigma=0.8, size=events
            )
            signs = rng.choice([-1, 1], size=events, p=[0.68, 0.32])
            for event_number, (event_type, amount, sign) in enumerate(
                zip(event_types, amounts, signs, strict=True), start=1
            ):
                event_time = period.start_time + pd.Timedelta(
                    days=int(rng.integers(0, 27)), hours=event_number % 20
                )
                transactions.append(
                    {
                        "client_id": client_id,
                        "event_time": event_time,
                        "event_type": int(event_type),
                        "amount": float(sign * amount),
                    }
                )
                product_bucket = int(event_type) % 4
                running_count[product_bucket] += 1
                running_amount[product_bucket] += amount

            # Outcomes use only history available by this reporting month.
            logits = np.array(
                [
                    -4.0 + 0.24 * running_count[0] + 0.35 * affinity[client_index, 0],
                    -4.3 + 0.20 * running_count[1] + 0.30 * affinity[client_index, 1],
                    -4.1 + 0.18 * running_count[2] + 0.0007 * running_amount[2],
                    -4.2 + 0.22 * running_count[3] + 0.30 * affinity[client_index, 3],
                ]
            )
            probabilities = 1 / (1 + np.exp(-np.clip(logits, -8, 8)))
            labels = rng.binomial(1, probabilities)
            targets.append(
                {
                    "client_id": client_id,
                    "mon": str(period.end_time.date()),
                    **{column: int(labels[i]) for i, column in enumerate(TARGET_COLUMNS)},
                }
            )
    return (
        pd.DataFrame(transactions),
        pd.DataFrame(targets),
        pd.DataFrame({"client_id": client_ids, "fold": folds}),
    )
