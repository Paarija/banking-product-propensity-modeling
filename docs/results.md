# MBD-mini experiment: fixed 5,000-client sample

This is an exploratory, reproducible comparison—not a production result, causal
offer-uplift estimate, or full-dataset benchmark. The data is the publicly
available [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini). No raw bank
records or model weights are committed to this repository.

## Setup

- Sample 5,000 client IDs uniformly with NumPy seed 42; retain all available
  client-month rows for each sampled client. Sampling does not use the labels.
- Client-disjoint folds 0–2 train, 3 validation, 4 test. This tests new-client
  generalization, **not** a future-calendar-month forecast.
- Up to 32 prior transactions per client-month; BERT-style encoder initialized
  randomly and trained for 2 epochs, batch size 64. It is not a pretrained
  language model. Baseline: per-product histogram gradient boosting on six
  transaction-history summary features.
- Command: `bankrec --data-dir data/raw --max-clients 5000 --max-events 32 --epochs 2 --batch-size 64 --output-dir artifacts/benchmark-5000`

## Observed results

The test fold contains 12,420 client-month rows; 124 rows have at least one
positive product label. Buyer-only metrics condition on these 124 rows, so they
must not be interpreted as all-customer accuracy.

| Metric | Tabular baseline | Transaction transformer |
| --- | ---: | ---: |
| Validation hit@1 among buyer-months (139 rows) | 39.6% | 46.8% |
| Test hit@1 among buyer-months (124 rows) | 41.1% | 52.4% |
| Test recall@2 among buyer-months | 75.4% | 77.0% |

Average precision (AP) is shown alongside the test-fold positive prevalence;
random ranking would have expected AP near that prevalence.

| Anonymized product | Test prevalence | Baseline AP | Transformer AP |
| --- | ---: | ---: | ---: |
| Product 1 | 0.523% | 0.0159 | 0.0160 |
| Product 2 | 0.048% | 0.0024 | 0.0141 |
| Product 3 | 0.258% | 0.0083 | 0.0087 |
| Product 4 | 0.233% | 0.0043 | 0.0095 |

The transformer beat this baseline on these metrics in **this one run**. Product
2 has only six positive test rows, so its apparent AP gain is especially
unstable. More seeds, a larger sample, uncertainty intervals, and calibration
checks are needed before claiming a robust improvement. Both models use only
transaction history; the labels are anonymized, so no specific banking product
or marketing action is inferred.
