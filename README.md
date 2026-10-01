# Banking Product Propensity Modeling

> Given a customer's transaction history, predict which anonymized banking product they may adopt next month.

This end-to-end data-science project turns historical transactions into a ranked product list. It starts with interpretable tabular models, then tests whether a transformer that reads the transaction sequence adds measurable value.

**Python · Pandas · scikit-learn · PyTorch · Hugging Face Transformers · Streamlit · PyArrow · Matplotlib · Pytest**

The repository is designed to be understood and run without a large download: the quick start generates deterministic fictional customers locally. A separate workflow reproduces the experiment on public, anonymized MBD-mini data.

## What is demonstrated

- Leakage-aware transaction histories and behavioral feature engineering
- Logistic Regression, Random Forest, and Gradient Boosting baselines
- A BERT-style encoder for ordered transaction type, amount, and recency
- Chronological backtesting for future-month performance
- Client-disjoint evaluation for unseen-customer generalization
- Rare-outcome evaluation with average precision, precision-recall curves, recall@k, and top-k lift
- Global tabular feature importance and an interactive Streamlit dashboard
- Tested command-line workflows, synthetic demo data, and reproducible artifacts

The transformer uses the Hugging Face `BertModel` architecture but is initialized from scratch for structured transaction sequences. It is **not** a pretrained English-language model, and it is presented as an experiment rather than assumed to be the best approach.

## Modeling workflow

```text
Transactions up to reporting date
              │
              ├── Tabular behavior features ──> LR / Random Forest / Gradient Boosting
              │
              └── Ordered event sequence ─────> Transaction Transformer
                                                   │
                                                   v
                                   Four next-month propensity scores
                                                   │
                                                   v
                                  Ranked anonymized product shortlist
```

For each customer-month, the pipeline excludes transactions after the reporting cutoff. The default evaluation uses earlier months for training, the second-latest month for validation, and the latest month for testing. This mimics the direction in which a real model would be used.

## Fast local demo

Python 3.10+ is required. From Windows Command Prompt:

```cmd
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dashboard,dev,visuals]"
.venv\Scripts\python.exe -m bankrec.cli --demo --split-strategy time --epochs 1 --output-dir artifacts\demo
.venv\Scripts\python.exe -m bankrec.reporting --run-dir artifacts\demo
.venv\Scripts\python.exe -m streamlit run src\bankrec\dashboard.py --server.address 127.0.0.1
```

Open the local URL printed by Streamlit, usually `http://127.0.0.1:8501`. The demo contains no real customer records and needs no network download.

The dashboard provides:

- Outcome prevalence and transaction-history coverage
- Side-by-side model metrics
- Product-level average precision and top-decile lift
- Precision-recall curves
- Global tabular feature importance
- Held-out customer-month examples with ranked propensity scores
- Clear validation and deployment limitations

## End-to-end notebook

[`notebooks/01_end_to_end_demo.ipynb`](notebooks/01_end_to_end_demo.ipynb) explains data loading, exploratory analysis, imbalance, feature engineering, chronological validation, baseline models, the transformer experiment, evaluation, visual results, and business interpretation.

To run it:

```cmd
.venv\Scripts\python.exe -m pip install -e ".[notebook]"
.venv\Scripts\python.exe -m jupyter lab notebooks\01_end_to_end_demo.ipynb
```

## Run on MBD-mini

The real-data workflow uses [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini), a public 10%-client subset of the [Multimodal Banking Dataset](https://arxiv.org/abs/2409.17587). The repository contains code only; downloaded data and generated model artifacts are Git-ignored.

```cmd
.venv\Scripts\bankrec-download.exe --data-dir data\raw --workers 8
.venv\Scripts\bankrec.exe --data-dir data\raw --max-clients 5000 --max-events 32 --epochs 2 --split-strategy time --output-dir artifacts\mbd-time-5000
```

The downloader verifies SHA-256 digests and extracts only the required transaction, target, and split files. `detail.tar.gz` is roughly 2 GB. MBD-mini is listed as CC BY 4.0; do not republish raw records or use this experimental system for individual decision-making.

## Validation choices

| Strategy | Training | Validation and test | Question answered |
| --- | --- | --- | --- |
| `time` (default) | Earlier reporting months | Two latest months | Does the model generalize to a future period? |
| `client` | Client folds 0–2 | Client folds 3 and 4 | Does it generalize to customers absent from training? |

These designs answer different questions. Time-based validation is emphasized for model selection because a random row split could leak future behavior. The saved report records the exact design used.

## Metrics and interpretation

- **Average precision (AP):** primary classification metric for rare product uptake; read it beside prevalence.
- **Precision-recall curve:** shows the trade-off between finding more positives and making fewer incorrect selections.
- **Top-10% lift:** positive rate among the highest-scored 10% divided by the overall positive rate.
- **Hit@1 / recall@2:** product-ranking metrics calculated only among customer-months with recorded uptake.
- **Feature importance:** global associations from Random Forest and absolute standardized Logistic Regression coefficients.

Accuracy is intentionally not featured because predicting “no uptake” for nearly everyone can look accurate on an imbalanced dataset while being useless.

## Tests and project structure

```cmd
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m ruff check src tests
```

```text
src/bankrec/       data preparation, models, evaluation, CLI, dashboard
notebooks/         guided end-to-end analysis
tests/             unit and integration tests
docs/              recorded experiment results and limitations
artifacts/         local metrics, predictions, weights, and charts (ignored)
```

## Responsible-use limitations

- This predicts historical product uptake; it does not estimate whether an offer would cause uptake.
- The four products and transaction types are anonymized, so no real product strategy is inferred.
- Propensity outputs are ranking scores, not automatically calibrated probabilities.
- Synthetic-demo results prove that the workflow runs, not that it performs on a bank population.
- Production use would require calibration, uncertainty analysis, fairness and privacy review, drift monitoring, secure data controls, and human-approved policy.

See [`docs/results.md`](docs/results.md) for a recorded MBD-mini benchmark and its caveats.

## Resume summary

> Built a leakage-aware banking product propensity pipeline from transaction histories; compared Logistic Regression, Random Forest, and Gradient Boosting with a Hugging Face transaction transformer using chronological validation, average precision, recall@k, and lift, then delivered an interactive Streamlit evaluation dashboard.

## Source and credit

Mollaev et al., *Multimodal Banking Dataset: Understanding Client Needs through Event Sequences*, KDD 2025. The MBD-mini dataset card lists a CC BY 4.0 license.
