# Banking Product Propensity Modeling

An experimental model that ranks four bank products for a client using past anonymized
transactions. The central question is whether a transaction-sequence transformer improves
next-month product ranking over a simple tabular model. **This is a research/portfolio
experiment, not a bank decisioning system or evidence of causal offer uplift.**

**Status:** implementation, tests, and a small real-data experiment are complete.
See [the results and limitations](docs/results.md) before quoting any metric.

## What the project does

1. Loads [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini), a 10%-client subset of
   the [Multimodal Banking Dataset](https://arxiv.org/abs/2409.17587) of real anonymized
   banking activity.
2. For each client and reporting month, takes only transactions on or before the reporting
   date; the four labels describe product uptake in the following month.
3. Compares a scikit-learn gradient-boosting baseline on transaction summaries with a small
   Hugging Face `BertModel` encoder of transaction type, signed log amount, and event recency.
   **The BERT architecture is initialized randomly and trained from scratch. It is not an
   English-language pretrained model.**
4. Reports per-product prevalence, average precision, ROC-AUC, and product-ranking metrics
   on client-disjoint folds. The measured comparison is in [Results](docs/results.md).

## Tech stack

- **Language and data:** Python, Pandas, NumPy, and PyArrow for loading Parquet
  files and building time-aware transaction histories.
- **Modeling:** PyTorch and Hugging Face Transformers for a BERT-style transaction
  encoder trained from scratch; scikit-learn histogram gradient boosting as the
  tabular baseline.
- **Evaluation:** scikit-learn average precision and ROC-AUC, plus custom
  product-ranking metrics on client-disjoint folds.
- **Reproducibility:** Requests and Truststore for verified dataset downloads,
  Pytest for tests, and Ruff for linting and formatting.

## Reproduce

Python 3.10+ is required. A GPU helps but is not required for a small sample.

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
```

Download `client_split.tar.gz`, `targets.tar.gz`, and `detail.tar.gz` from the
[MBD-mini dataset files](https://huggingface.co/datasets/ai-lab/MBD-mini/tree/main)
with the resumable downloader. It verifies each archive's SHA-256 digest and extracts
only the files this project uses. The `detail.tar.gz` archive is roughly 2 GB:

```bash
bankrec-download --data-dir data/raw --workers 8
```

The resulting directories are:

```text
data/raw/client_split/
data/raw/targets/
data/raw/detail/trx/
```

The project only extracts the transaction portion of `detail`; dialogue and geolocation
are not used. Data and model artifacts are excluded from Git. MBD-mini is public and
listed as CC BY 4.0, but anonymization is not a guarantee of zero privacy risk; do
not republish raw records or use them for individual decision-making.

```bash
bankrec --data-dir data/raw --max-clients 1000 --max-events 32 --epochs 2
python -m pytest -q
```

Metrics are written to `artifacts/metrics.json`. Increase `--max-clients` only after a small
run succeeds. Client sampling is random with a fixed seed and does not inspect outcomes.
The report records the sample size and prevalence, because rare purchases make small-sample
scores unstable.

## Evaluation and limitations

- Folds 0–2 train, fold 3 validates, and fold 4 tests. A client cannot cross these splits;
  multiple monthly rows from one client stay together. This measures generalization to new
  clients, **not** a strictly later calendar period.
- The published targets are extremely imbalanced. ROC-AUC alone is inadequate; inspect
  per-product average precision and prevalence. Ranking metrics are calculated among
  clients who actually purchased at least one of the four products.
- This pipeline uses transaction activity only. Product labels are anonymized, so it does
  not invent their real names or imply a specific offer strategy.
- Historical purchase prediction is not treatment-effect estimation. It does not show
  whether recommending a product would cause uptake, and it is not suitable for deployment
  without privacy, fairness, calibration, and business-policy review.
- MBD-mini's reduced client and time coverage is useful for development, but findings need
  confirmation on the full benchmark before broad claims.

## Source and credit

Mollaev et al., *Multimodal Banking Dataset: Understanding Client Needs through Event
Sequences*, KDD 2025. The [dataset card](https://huggingface.co/datasets/ai-lab/MBD-mini)
lists a CC BY 4.0 license. This repository contains code, not redistributed bank records.
