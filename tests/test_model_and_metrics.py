import numpy as np
import torch

from bankrec.data import prepare
from bankrec.evaluation import evaluate
from bankrec.model import ProductHistoryTransformer


def test_transformer_forward(tiny_santander):
    data = prepare(tiny_santander, max_events=8)
    model = ProductHistoryTransformer(data.vocab_size, 9, 24, hidden_size=32)
    logits = model(
        torch.as_tensor(data.event_types),
        torch.as_tensor(data.numeric),
        torch.as_tensor(data.attention_mask),
    )
    assert logits.shape == (32, 24)
    assert torch.isfinite(logits).all()


def test_owned_products_masked_from_rankings():
    labels = np.array([[0, 1, 0], [0, 0, 0]])
    scores = np.array([[0.99, 0.8, 0.1], [0.5, 0.1, 0.0]])
    owned = np.array([[True, False, False], [False, False, False]])
    report = evaluate(labels, scores, ("a", "b", "c"), owned)
    assert report["hit_at_1_among_buyers"] == 1.0
    assert report["map_at_7_all_clients"] == 0.5
    assert report["per_product"]["a"]["roc_auc"] is None
