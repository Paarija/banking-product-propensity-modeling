"""Ranking and rare-outcome metrics with explicit undefined-metric handling."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(
    y_true: np.ndarray,
    scores: np.ndarray,
    product_names: tuple[str, ...],
    owned_products: np.ndarray,
) -> dict:
    if (
        y_true.shape != scores.shape
        or y_true.shape != owned_products.shape
        or y_true.ndim != 2
        or y_true.shape[1] != len(product_names)
    ):
        raise ValueError("labels, scores, ownership and product names must match")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    per_product = {}
    for i, name in enumerate(product_names):
        eligible_for_product = ~owned_products[:, i]
        actual = y_true[eligible_for_product, i]
        predicted = scores[eligible_for_product, i]
        distinct = np.unique(actual)
        per_product[name] = {
            "eligible_clients": len(actual),
            "prevalence": float(actual.mean()) if len(actual) else None,
            "average_precision": float(average_precision_score(actual, predicted))
            if len(distinct) == 2
            else None,
            "roc_auc": float(roc_auc_score(actual, predicted)) if len(distinct) == 2 else None,
        }
    eligible = y_true.sum(axis=1) > 0
    # Existing products cannot count as newly added recommendations.
    eligible_scores = np.where(owned_products[eligible], -np.inf, scores[eligible])
    ranked = np.argsort(-eligible_scores, axis=1)
    labels = y_true[eligible]
    hit_at_1 = float(np.mean(labels[np.arange(len(labels)), ranked[:, 0]])) if len(labels) else None
    recall_at_2 = (
        float(
            np.mean(
                np.take_along_axis(labels, ranked[:, :2], axis=1).sum(axis=1) / labels.sum(axis=1)
            )
        )
        if len(labels)
        else None
    )
    masked = np.where(owned_products, -np.inf, scores)
    top_k = np.argsort(-masked, axis=1)[:, : min(7, len(product_names))]
    map_at_7 = []
    for actual, prediction in zip(y_true, top_k, strict=True):
        hits = actual[prediction]
        count = int(actual.sum())
        if count == 0:
            map_at_7.append(0.0)
            continue
        precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        map_at_7.append(float((precision * hits).sum() / min(count, 7)))
    return {
        "examples": len(y_true),
        "positive_clients": int(eligible.sum()),
        "hit_at_1_among_buyers": hit_at_1,
        "recall_at_2_among_buyers": recall_at_2,
        "map_at_7_all_clients": float(np.mean(map_at_7)) if map_at_7 else None,
        "per_product": per_product,
    }
