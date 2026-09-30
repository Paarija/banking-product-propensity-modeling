# Banking Product Recommendation from Monthly Histories

An experimental, time-aware recommender for the [Kaggle Santander Product
Recommendation competition](https://www.kaggle.com/competitions/santander-product-recommendation/data).
It asks whether a small BERT-style transformer over a customer's monthly product
ownership history can rank **newly added** products better than a tabular
gradient-boosting baseline. This is a portfolio experiment, not a deployed banking
system or evidence that a recommendation causes a purchase.

**Status:** the code and synthetic-fixture tests are complete. No real-data benchmark
is reported yet because Kaggle requires a signed-in user to accept the competition
rules before the files can be downloaded. Do not quote any performance number for
this project until that experiment has been run.

## Dataset and method

The Kaggle data contains monthly customer records and 24 product-ownership flags.
These are **not individual transactions**. For a customer observed in consecutive
months, a product is labeled positive if it was absent in the current month and
present in the next month. We use only information through the current month as
input. Monthly owned-product tokens, with their age in months, feed a small
Hugging Face `BertModel` encoder. Its weights are initialized randomly and trained
on this data; this does **not** use a pretrained language model or RAG. A
scikit-learn `HistGradientBoostingClassifier` baseline uses current ownership and
simple history counts.

The latest available target month is held out for test, the preceding target
month for validation, and earlier months for training. Customers can occur in
multiple periods; this is a **temporal** test, not a client-disjoint test. Product
scores are masked at ranking time for products already owned. The report includes
per-product prevalence, average precision and ROC-AUC, buyer-only hit@1 and
recall@2, and MAP@7 across all customers (zero for customers with no additions).

## Reproduce

1. Sign into Kaggle and accept the competition rules at the
   [data page](https://www.kaggle.com/competitions/santander-product-recommendation/data).
   Download and extract `train_ver2.csv` (or `train.csv`) into `data/raw/`. The
   competition's test file does not have outcome labels and is not used here.
   Do not redistribute the data; it remains excluded from Git.
2. Install Python 3.10+ dependencies and run the experiment:

```bash
python -m venv .venv
python -m pip install -e ".[dev]"
bankrec --data-dir data/raw --max-clients 1000 --max-events 32 --epochs 2
python -m pytest -q
```

`bankrec` samples customer IDs with seed 42, then reads those customers' complete
monthly records in CSV chunks. Results and weights are written under `artifacts/`
and excluded from Git. A 1,000-customer run is only a pipeline check; rare product
uptake can make its per-product scores unstable. Increase sample size for a
credible comparison. No Kaggle credentials are stored in this repository.

## Limitations

Santander states that the competition sample does not contain real Santander Spain
customers and is not representative of that customer base. Product-uptake
prediction is not causal uplift. The model needs external validation, calibration,
fairness review and business-policy checks before any practical use. The data is
from a 2016 competition, so its patterns may not reflect current banking behavior.
