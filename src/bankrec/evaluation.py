"""Ranking and rare-outcome metrics with explicit undefined-metric handling."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate(y_true: np.ndarray, scores: np.ndarray) -> dict:
    if y_true.shape != scores.shape or y_true.ndim != 2 or y_true.shape[1] != 4:
        raise ValueError("expected matching [n, 4] labels and scores")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    per_product = {}
    average_precisions = []
    for i in range(4):
        actual = y_true[:, i]
        distinct = np.unique(actual)
        average_precision = (
            float(average_precision_score(actual, scores[:, i])) if len(distinct) == 2 else None
        )
        if average_precision is not None:
            average_precisions.append(average_precision)
        prevalence = float(actual.mean()) if len(actual) else None
        lift = {}
        for fraction in (0.01, 0.05, 0.10):
            count = max(1, int(np.ceil(len(actual) * fraction)))
            top_rate = (
                float(actual[np.argsort(-scores[:, i])[:count]].mean()) if len(actual) else None
            )
            lift[f"top_{int(fraction * 100)}pct"] = (
                float(top_rate / prevalence) if prevalence and top_rate is not None else None
            )
        per_product[f"product_{i + 1}"] = {
            "prevalence": prevalence,
            "average_precision": average_precision,
            "roc_auc": float(roc_auc_score(actual, scores[:, i])) if len(distinct) == 2 else None,
            "lift": lift,
        }
    eligible = y_true.sum(axis=1) > 0
    ranked = np.argsort(-scores[eligible], axis=1)
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
    return {
        "examples": len(y_true),
        "positive_clients": int(eligible.sum()),
        "macro_average_precision": float(np.mean(average_precisions))
        if average_precisions
        else None,
        "hit_at_1_among_buyers": hit_at_1,
        "recall_at_2_among_buyers": recall_at_2,
        "per_product": per_product,
    }
