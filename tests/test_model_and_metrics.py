import numpy as np
import torch

from bankrec.data import prepare
from bankrec.evaluation import evaluate
from bankrec.model import TransactionBert


def test_transformer_forward(tiny_mbd):
    data = prepare(*tiny_mbd, max_events=4)
    model = TransactionBert(data.vocab_size, data.event_types.shape[1], hidden_size=32)
    logits = model(
        torch.as_tensor(data.event_types),
        torch.as_tensor(data.numeric),
        torch.as_tensor(data.attention_mask),
    )
    assert logits.shape == (10, 4)
    assert torch.isfinite(logits).all()


def test_undefined_auc_is_reported_as_none():
    labels = np.zeros((2, 4))
    scores = np.full((2, 4), 0.1)
    report = evaluate(labels, scores)
    assert report["per_product"]["product_1"]["roc_auc"] is None
    assert report["positive_clients"] == 0


def test_ranking_metrics():
    labels = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    scores = np.array([[0.9, 0.1, 0.2, 0.0], [0.1, 0.7, 0.2, 0.0]])
    report = evaluate(labels, scores)
    assert report["hit_at_1_among_buyers"] == 1.0
    assert report["recall_at_2_among_buyers"] == 1.0
