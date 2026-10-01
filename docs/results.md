# MBD-mini future-month benchmark: 5,000 clients

This is an exploratory portfolio benchmark, not a production result, causal offer-uplift estimate, or full-dataset study. It uses the public [MBD-mini](https://huggingface.co/datasets/ai-lab/MBD-mini) dataset. No raw records, prediction files, or model weights are committed.

## Experimental setup

- Uniformly sample 5,000 client IDs with NumPy seed 42 without inspecting outcomes.
- Build each customer-month history using only transactions available by its reporting cutoff.
- Use 62 tabular features covering activity, amount, recency, credit share, and transaction-type counts.
- Train on reporting months through November 2022, validate on December 2022, and test on January 2023.
- Compare Logistic Regression, Random Forest, and histogram Gradient Boosting with a BERT-style transaction encoder trained from scratch for two epochs.
- Use at most 32 prior events per customer-month and a transformer batch size of 64.

Reproduction command:

```cmd
bankrec --data-dir data\raw --max-clients 5000 --max-events 32 --epochs 2 --batch-size 64 --split-strategy time --output-dir artifacts\mbd-time-5000
```

## Test results

The January 2023 test set contains 5,000 customer-month rows. Only 32 rows have at least one positive product label, so buyer-only ranking metrics and product-level scores are highly uncertain.

| Model | Macro AP | Hit@1 among 32 buyer-months | Recall@2 among buyer-months |
| --- | ---: | ---: | ---: |
| Logistic Regression | **0.0235** | 21.9% | 67.2% |
| Random Forest | 0.0065 | **43.8%** | **75.0%** |
| Gradient Boosting | 0.0040 | 28.1% | 68.8% |
| Transaction Transformer | 0.0122 | 28.1% | 54.7% |

There is no single winner across every objective. Logistic Regression has the highest macro average precision, while Random Forest leads the buyer-only ranking metrics. The transformer exceeds both tree models on macro AP but does not beat Logistic Regression.

Average precision should be read beside the extremely low base rate:

| Product | Prevalence | Logistic AP | Random Forest AP | Gradient Boosting AP | Transformer AP |
| --- | ---: | ---: | ---: | ---: | ---: |
| Product 1 | 0.220% | 0.0039 | 0.0036 | 0.0037 | **0.0040** |
| Product 2 | 0.060% | **0.0345** | 0.0056 | 0.0008 | 0.0122 |
| Product 3 | 0.140% | 0.0039 | 0.0088 | 0.0036 | **0.0209** |
| Product 4 | 0.240% | **0.0519** | 0.0077 | 0.0081 | 0.0116 |

## Interpretation

The result supports starting with traditional data science rather than assuming a deeper model will win. Logistic Regression captures useful signal from engineered behavior features and is easier to inspect. The transformer appears useful for Product 3, suggesting event order may matter for some outcomes, but the number of positives is too small for a robust claim.

Before drawing broader conclusions, the experiment would need more clients, repeated seeds, confidence intervals, probability calibration, subgroup analysis, and a longer rolling-window backtest. Historical uptake prediction also does not prove that recommending a product would cause adoption.
