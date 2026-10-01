"""Compare transparent tabular models with a transaction-sequence transformer."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import PreparedData
from .evaluation import evaluate
from .model import TransactionBert
from .splitting import EvaluationSplit, make_split

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "transformer": "Transaction Transformer",
}


def _balanced_weights(outcome: np.ndarray) -> np.ndarray:
    positive = max(int(outcome.sum()), 1)
    negative = max(len(outcome) - positive, 1)
    return np.where(outcome == 1, len(outcome) / (2 * positive), len(outcome) / (2 * negative))


def _fit_tabular_models(
    features: pd.DataFrame,
    labels: np.ndarray,
    split: EvaluationSplit,
    *,
    seed: int,
) -> tuple[dict[str, dict[str, np.ndarray]], pd.DataFrame]:
    train_x = features.loc[split.train]
    eval_sets = {
        "validation": features.loc[split.validation],
        "test": features.loc[split.test],
    }
    names = ("logistic_regression", "random_forest", "gradient_boosting")
    predictions = {
        name: {
            split_name: np.empty((len(frame), labels.shape[1]), dtype=np.float32)
            for split_name, frame in eval_sets.items()
        }
        for name in names
    }
    importance_rows: list[dict] = []

    for product_index in range(labels.shape[1]):
        outcome = labels[split.train, product_index].astype(int)
        if len(np.unique(outcome)) < 2:
            for model_predictions in predictions.values():
                for split_name in eval_sets:
                    model_predictions[split_name][:, product_index] = float(outcome.mean())
            continue
        models = {
            "logistic_regression": make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, solver="liblinear", random_state=seed
                ),
            ),
            "random_forest": RandomForestClassifier(
                n_estimators=120,
                max_depth=10,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            ),
            "gradient_boosting": HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=15, learning_rate=0.08, random_state=seed
            ),
        }
        for model_name, model in models.items():
            fit_kwargs = (
                {"sample_weight": _balanced_weights(outcome)}
                if model_name == "gradient_boosting"
                else {}
            )
            model.fit(train_x, outcome, **fit_kwargs)
            for split_name, frame in eval_sets.items():
                predictions[model_name][split_name][:, product_index] = model.predict_proba(frame)[
                    :, 1
                ]

        logistic = models["logistic_regression"].named_steps["logisticregression"]
        forest = models["random_forest"]
        for feature_index, feature in enumerate(features.columns):
            importance_rows.append(
                {
                    "product": f"product_{product_index + 1}",
                    "feature": feature,
                    "logistic_abs_coefficient": float(abs(logistic.coef_[0, feature_index])),
                    "random_forest_importance": float(forest.feature_importances_[feature_index]),
                }
            )
    return predictions, pd.DataFrame(importance_rows)


def _train_transformer(
    data: PreparedData,
    split: EvaluationSplit,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[TransactionBert, dict[str, np.ndarray], list[dict]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransactionBert(data.vocab_size, data.event_types.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    labels_train = data.labels[split.train]
    positive = labels_train.sum(axis=0)
    negative = len(labels_train) - positive
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.as_tensor(
            np.clip(negative / np.maximum(positive, 1), 1, 50),
            dtype=torch.float32,
            device=device,
        )
    )

    def loader(mask: np.ndarray, shuffle: bool = False) -> DataLoader:
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

    train_loader = loader(split.train, True)
    eval_loaders = {"validation": loader(split.validation), "test": loader(split.test)}
    history = []
    for epoch in range(epochs):
        model.train()
        losses = []
        for types, numeric, mask, labels in train_loader:
            types, numeric, mask, labels = (
                item.to(device) for item in (types, numeric, mask, labels)
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(types, numeric, mask), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(float(loss.item()))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses))})

    predictions = {}
    model.eval()
    with torch.no_grad():
        for split_name, batches in eval_loaders.items():
            results = []
            for types, numeric, mask, _ in batches:
                logits = model(types.to(device), numeric.to(device), mask.to(device))
                results.append(torch.sigmoid(logits).cpu().numpy())
            predictions[split_name] = np.concatenate(results)
    return model, predictions, history


def run_experiment(
    data: PreparedData,
    *,
    epochs: int = 3,
    batch_size: int = 64,
    learning_rate: float = 2e-4,
    seed: int = 42,
    output_dir: str | Path = "artifacts",
    split_strategy: str = "client",
    include_transformer: bool = True,
    dataset_name: str = "ai-lab/MBD-mini",
) -> dict:
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    split = make_split(data, split_strategy)
    tabular_predictions, feature_importance = _fit_tabular_models(
        data.baseline_features, data.labels, split, seed=seed
    )
    all_predictions = dict(tabular_predictions)
    history: list[dict] = []
    transformer_model = None
    if include_transformer:
        transformer_model, transformer_predictions, history = _train_transformer(
            data,
            split,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        all_predictions["transformer"] = transformer_predictions

    split_masks = {"validation": split.validation, "test": split.test}
    model_reports = {
        model_name: {
            split_name: evaluate(data.labels[mask], model_predictions[split_name])
            for split_name, mask in split_masks.items()
        }
        for model_name, model_predictions in all_predictions.items()
    }
    report = {
        "dataset": dataset_name,
        "split_strategy": split_strategy,
        "split": split.description,
        "sampled_clients": len(set(data.client_ids)),
        "feature_count": data.baseline_features.shape[1],
        "model_labels": {name: MODEL_LABELS[name] for name in model_reports},
        "history": history,
        "models": model_reports,
        "baseline": model_reports["gradient_boosting"],
        "transformer": model_reports.get("transformer"),
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    feature_importance.to_csv(output / "feature_importance.csv", index=False)

    prediction_rows = []
    for split_name, mask in split_masks.items():
        frame = pd.DataFrame(
            {
                "split": split_name,
                "reporting_month": np.asarray(data.months)[mask],
                "history_event_count": data.attention_mask[mask].sum(axis=1) - 1,
            }
        )
        for feature in data.baseline_features.columns:
            frame[f"feature_{feature}"] = data.baseline_features.loc[mask, feature].to_numpy()
        for product_index in range(data.labels.shape[1]):
            product = f"product_{product_index + 1}"
            frame[f"actual_{product}"] = data.labels[mask, product_index].astype(np.int8)
            for model_name, model_predictions in all_predictions.items():
                frame[f"{model_name}_{product}"] = model_predictions[split_name][:, product_index]
        prediction_rows.append(frame)
    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions.insert(0, "example_id", np.arange(len(predictions)))
    predictions.to_parquet(output / "predictions.parquet", index=False)

    if transformer_model is not None:
        torch.save(
            {
                "state_dict": transformer_model.state_dict(),
                "vocab_size": data.vocab_size,
                "sequence_length": data.event_types.shape[1],
            },
            output / "transformer.pt",
        )
    return report
