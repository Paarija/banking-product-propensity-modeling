"""Leakage-aware monthly product-history examples for Kaggle Santander."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRODUCT_COLUMNS = (
    "ind_ahor_fin_ult1",
    "ind_aval_fin_ult1",
    "ind_cco_fin_ult1",
    "ind_cder_fin_ult1",
    "ind_cno_fin_ult1",
    "ind_ctju_fin_ult1",
    "ind_ctma_fin_ult1",
    "ind_ctop_fin_ult1",
    "ind_ctpp_fin_ult1",
    "ind_deco_fin_ult1",
    "ind_deme_fin_ult1",
    "ind_dela_fin_ult1",
    "ind_ecue_fin_ult1",
    "ind_fond_fin_ult1",
    "ind_hip_fin_ult1",
    "ind_plan_fin_ult1",
    "ind_pres_fin_ult1",
    "ind_reca_fin_ult1",
    "ind_tjcr_fin_ult1",
    "ind_valo_fin_ult1",
    "ind_viv_fin_ult1",
    "ind_nomina_ult1",
    "ind_nom_pens_ult1",
    "ind_recibo_ult1",
)


@dataclass(frozen=True)
class PreparedData:
    event_types: np.ndarray
    numeric: np.ndarray
    attention_mask: np.ndarray
    labels: np.ndarray
    folds: np.ndarray  # 0 train, 3 validation, 4 test; time-based, not client-disjoint
    client_ids: tuple[str, ...]
    months: tuple[str, ...]
    baseline_features: pd.DataFrame
    vocab_size: int
    product_names: tuple[str, ...]
    owned_products: np.ndarray


def prepare(panel: pd.DataFrame, *, max_events: int = 64) -> PreparedData:
    """Predict additions next month from prior monthly product ownership only."""
    if max_events < 1:
        raise ValueError("max_events must be positive")
    required = {"ncodpers", "fecha_dato", *PRODUCT_COLUMNS}
    missing = required - set(panel.columns)
    if missing:
        raise ValueError(f"Santander panel missing columns: {', '.join(sorted(missing))}")
    frame = panel.loc[:, ["ncodpers", "fecha_dato", *PRODUCT_COLUMNS]].copy()
    frame["ncodpers"] = frame["ncodpers"].astype(str)
    frame["month"] = pd.to_datetime(frame["fecha_dato"], errors="raise").dt.to_period("M")
    if frame.duplicated(["ncodpers", "month"]).any():
        raise ValueError("duplicate customer-month rows")
    for col in PRODUCT_COLUMNS:
        values = pd.to_numeric(frame[col], errors="coerce").fillna(0)
        if not values.isin([0, 1]).all():
            raise ValueError(f"{col} must be binary or missing")
        frame[col] = values.astype(np.int8)
    frame = frame.sort_values(["ncodpers", "month"], kind="stable")

    examples = []
    for cid, history in frame.groupby("ncodpers", sort=False):
        months = history["month"].tolist()
        holdings = history.loc[:, PRODUCT_COLUMNS].to_numpy(dtype=np.int8)
        for idx in range(len(history) - 1):
            if months[idx] + 1 != months[idx + 1]:
                continue  # A gap is not a next-month outcome.
            owned = holdings[idx].astype(bool)
            added = (holdings[idx + 1].astype(bool) & ~owned).astype(np.float32)
            events = [
                (p + 2, idx - old_idx)
                for old_idx in range(idx + 1)
                for p in np.flatnonzero(holdings[old_idx])
            ][-max_events:]
            examples.append((cid, str(months[idx]), owned, added, events, idx + 1))
    if not examples:
        raise ValueError("no consecutive customer-month pairs in Santander data")
    target_months = sorted({month for _, month, *_ in examples})
    if len(target_months) < 3:
        raise ValueError("need at least three target months for train/validation/test")
    split_month = {target_months[-2]: 3, target_months[-1]: 4}
    n, width = len(examples), max_events + 1
    types = np.zeros((n, width), dtype=np.int64)
    numeric = np.zeros((n, width, 2), dtype=np.float32)
    mask = np.zeros((n, width), dtype=np.int64)
    types[:, 0] = 1  # CLS; zero is padding.
    mask[:, 0] = 1
    features = []
    for row, (_, _, owned, _, events, months_seen) in enumerate(examples):
        for pos, (product, age) in enumerate(events, start=1):
            types[row, pos] = product
            numeric[row, pos, 0] = min(age, 24) / 24
            numeric[row, pos, 1] = 1.0
            mask[row, pos] = 1
        features.append(
            {
                **{name: int(owned[i]) for i, name in enumerate(PRODUCT_COLUMNS)},
                "months_seen": months_seen,
                "current_product_count": int(owned.sum()),
                "history_event_count": len(events),
            }
        )
    return PreparedData(
        event_types=types,
        numeric=numeric,
        attention_mask=mask,
        labels=np.stack([item[3] for item in examples]),
        folds=np.array([split_month.get(item[1], 0) for item in examples]),
        client_ids=tuple(item[0] for item in examples),
        months=tuple(item[1] for item in examples),
        baseline_features=pd.DataFrame(features),
        vocab_size=len(PRODUCT_COLUMNS) + 2,
        product_names=PRODUCT_COLUMNS,
        owned_products=np.stack([item[2] for item in examples]),
    )
