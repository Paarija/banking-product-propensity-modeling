"""Train a simple tabular baseline and transaction transformer on the same folds."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import PreparedData
from .evaluation import evaluate
from .model import TransactionBert


def _predict_baseline(features, labels, train_mask, eval_mask):
    predictions = np.empty((int(eval_mask.sum()), 4), dtype=np.float32)
    for i in range(4):
        outcome = labels[train_mask, i].astype(int)
        if len(np.unique(outcome)) < 2:
            predictions[:, i] = float(outcome.mean())
            continue
        model = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=12, random_state=42)
        model.fit(features.loc[train_mask], outcome)
        predictions[:, i] = model.predict_proba(features.loc[eval_mask])[:, 1]
    return predictions


def run_experiment(
    data: PreparedData,
    *,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    seed: int = 42,
    output_dir: str | Path = "artifacts",
) -> dict:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_mask = ~np.isin(data.folds, [3, 4])
    val_mask = data.folds == 3
    test_mask = data.folds == 4
    if not train_mask.any() or not val_mask.any() or not test_mask.any():
        raise ValueError("need MBD folds 0-2 for training, 3 for validation, and 4 for test")
    if len(
        set(np.asarray(data.client_ids)[train_mask]) & set(np.asarray(data.client_ids)[test_mask])
    ):
        raise ValueError("client leakage between train and test")

    baseline_val = _predict_baseline(data.baseline_features, data.labels, train_mask, val_mask)
    baseline_test = _predict_baseline(data.baseline_features, data.labels, train_mask, test_mask)
    # The baseline is intentionally trained on folds 0-2 only; test is never used for tuning.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransactionBert(data.vocab_size, data.event_types.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    labels_train = data.labels[train_mask]
    positive = labels_train.sum(axis=0)
    negative = len(labels_train) - positive
    pos_weight = np.clip(negative / np.maximum(positive, 1), 1, 50)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(pos_weight, dtype=torch.float32, device=device)
    )

    def loader(mask, shuffle=False):
        return DataLoader(
            TensorDataset(
                torch.as_tensor(data.event_types[mask], dtype=torch.long),
                torch.as_tensor(data.numeric[mask], dtype=torch.float32),
                torch.as_tensor(data.attention_mask[mask], dtype=torch.long),
                torch.as_tensor(data.labels[mask], dtype=torch.float32),
            ),
            batch_size=batch_size,
            shuffle=shuffle,
        )

    train_loader, val_loader, test_loader = (
        loader(train_mask, True),
        loader(val_mask),
        loader(test_mask),
    )
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for types, numeric, mask, labels in train_loader:
            types, numeric, mask, labels = (x.to(device) for x in (types, numeric, mask, labels))
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(types, numeric, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})

    def predict(batches):
        model.eval()
        results = []
        with torch.no_grad():
            for types, numeric, mask, _ in batches:
                logits = model(types.to(device), numeric.to(device), mask.to(device))
                results.append(torch.sigmoid(logits).cpu().numpy())
        return np.concatenate(results)

    transformer_val = predict(val_loader)
    transformer_test = predict(test_loader)
    report = {
        "dataset": "ai-lab/MBD-mini",
        "split": "client-disjoint: folds 0-2 train, 3 validation, 4 test",
        "sampled_clients": len(set(data.client_ids)),
        "history": history,
        "baseline": {
            "validation": evaluate(data.labels[val_mask], baseline_val),
            "test": evaluate(data.labels[test_mask], baseline_test),
        },
        "transformer": {
            "validation": evaluate(data.labels[val_mask], transformer_val),
            "test": evaluate(data.labels[test_mask], transformer_test),
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "vocab_size": data.vocab_size,
            "sequence_length": data.event_types.shape[1],
        },
        output / "transformer.pt",
    )
    return report
