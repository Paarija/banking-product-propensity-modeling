"""Business-focused Streamlit view of local experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from bankrec.dashboard_data import (
    available_runs,
    load_feature_importance,
    load_run,
    model_labels,
    model_reports,
    precision_recall_data,
    product_metrics,
)

PRODUCTS = tuple(f"product_{i}" for i in range(1, 5))
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def _product_label(product: str) -> str:
    return product.replace("_", " ").title()


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def _score_column(frame: pd.DataFrame, model_name: str, product: str) -> str | None:
    preferred = f"{model_name}_{product}"
    legacy = f"baseline_{product}" if model_name == "gradient_boosting" else preferred
    return preferred if preferred in frame else legacy if legacy in frame else None


def _model_summary(report: dict, split: str) -> pd.DataFrame:
    labels = model_labels(report)
    return pd.DataFrame(
        [
            {
                "Model": labels[name],
                "Macro average precision": result[split].get("macro_average_precision"),
                "Buyer-only hit@1": result[split]["hit_at_1_among_buyers"],
                "Buyer-only recall@2": result[split]["recall_at_2_among_buyers"],
            }
            for name, result in model_reports(report).items()
        ]
    )


def main() -> None:
    st.set_page_config(page_title="Banking Product Propensity Modeling", layout="wide")
    st.title("Which product might a customer adopt next month?")
    st.write(
        "This project uses past transaction behavior to rank four anonymized banking "
        "products, then asks whether a transaction-sequence transformer improves over "
        "standard tabular models."
    )
    st.info(
        "Portfolio experiment only. Scores describe historical product uptake; they are "
        "not calibrated offer probabilities or a real banking decision system."
    )

    root = Path(st.sidebar.text_input("Local artifacts folder", str(DEFAULT_ARTIFACTS)))
    runs = available_runs(root)
    if not runs:
        st.warning("No results found. Generate the fast synthetic demo first:")
        st.code(
            "python -m bankrec.cli --demo --split-strategy time "
            "--epochs 1 --output-dir artifacts/demo",
            language="text",
        )
        st.stop()
    folder = st.sidebar.selectbox(
        "Experiment",
        runs,
        format_func=lambda path: str(path.relative_to(root)) if path != root else "default",
    )
    split_name = st.sidebar.radio("Evaluation split", ["test", "validation"])
    try:
        report, predictions = load_run(folder)
        models = model_reports(report)
        labels = model_labels(report)
        metrics = product_metrics(report, split_name)
        importance = load_feature_importance(folder)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not read this run: {exc}")
        st.stop()

    overview, evaluation, explorer, limitations = st.tabs(
        ["Business overview", "Model evaluation", "Example explorer", "Limitations"]
    )

    with overview:
        st.subheader("Experiment at a glance")
        col1, col2, col3 = st.columns(3)
        col1.metric("Sampled customers", f"{report['sampled_clients']:,}")
        reference = next(iter(models.values()))[split_name]
        col2.metric("Held-out customer-months", f"{reference['examples']:,}")
        col3.metric("Rows with product uptake", f"{reference['positive_clients']:,}")
        st.write(f"**Dataset:** {report['dataset']}")
        st.write(f"**Validation design:** {report['split']}")
        if report.get("split_strategy") == "time":
            st.success(
                "Future-month backtest: model development uses earlier reporting months, "
                "while the latest month is held out for testing."
            )
        else:
            st.warning(
                "Unseen-customer test: customers do not cross folds, but this is not a "
                "future-calendar backtest."
            )

        st.subheader("How rare is each outcome?")
        prevalence = (
            metrics.drop_duplicates("product")
            .set_index("product")[["prevalence"]]
            .rename(columns={"prevalence": "Positive prevalence"})
        )
        st.bar_chart(prevalence)
        st.caption(
            "Rare outcomes make accuracy misleading. The project emphasizes average "
            "precision, precision-recall curves, recall@k, and lift."
        )

        if predictions is not None:
            held_out = predictions[predictions["split"] == split_name]
            if not held_out.empty:
                st.subheader("Transaction-history coverage")
                counts = held_out["history_event_count"].clip(upper=64)
                bins = pd.cut(counts, bins=[-1, 4, 9, 19, 31, 64], include_lowest=True)
                distribution = bins.value_counts(sort=False).rename("Customer-months")
                distribution.index = distribution.index.astype(str)
                st.bar_chart(distribution)
                st.caption("Counts are capped by the configured sequence length.")

    with evaluation:
        st.subheader("Traditional models first; transformer as an experiment")
        summary = _model_summary(report, split_name)
        st.dataframe(summary, hide_index=True, width="stretch")
        st.bar_chart(summary.set_index("Model")[["Macro average precision"]])

        selected_product = st.selectbox(
            "Product for detailed evaluation", PRODUCTS, format_func=_product_label
        )
        product_rows = metrics[metrics["product"] == _product_label(selected_product)].copy()
        st.subheader("Average precision and top-decile lift")
        left, right = st.columns(2)
        left.bar_chart(product_rows.set_index("model")[["average_precision"]])
        right.bar_chart(product_rows.set_index("model")[["top_10pct_lift"]])
        st.caption(
            "Lift compares the positive rate in the top-scored 10% with the overall "
            "positive rate. A lift of 2 means twice the base rate."
        )

        if predictions is not None:
            curve = precision_recall_data(predictions, split_name, selected_product, list(models))
            if not curve.empty:
                curve["Model"] = curve["model_key"].map(labels)
                st.subheader("Precision-recall curve")
                st.line_chart(curve, x="recall", y="precision", color="Model")

        if importance is not None and not importance.empty:
            st.subheader("What drives the tabular models?")
            importance_product = st.selectbox(
                "Feature-importance product", PRODUCTS, format_func=_product_label
            )
            product_importance = importance[importance["product"] == importance_product].copy()
            measure = st.radio(
                "Importance view",
                ["random_forest_importance", "logistic_abs_coefficient"],
                format_func=lambda value: (
                    "Random Forest importance"
                    if value == "random_forest_importance"
                    else "Absolute standardized Logistic Regression coefficient"
                ),
                horizontal=True,
            )
            top_features = product_importance.nlargest(10, measure).set_index("feature")[[measure]]
            st.bar_chart(top_features)
            st.caption(
                "These are global associations, not causal effects or explanations for an "
                "individual customer."
            )

    with explorer:
        st.subheader("Inspect a held-out historical example")
        if predictions is None:
            st.info("Run a current experiment to save example-level predictions.")
        else:
            examples = predictions[predictions["split"] == split_name].copy()
            actual_columns = [f"actual_{product}" for product in PRODUCTS]
            buyer_only = st.checkbox("Show only examples with product uptake", value=True)
            if buyer_only:
                examples = examples[examples[actual_columns].sum(axis=1) > 0]
            if examples.empty:
                st.info("No examples match this filter.")
            else:
                examples = examples.head(500)
                months = examples.set_index("example_id")["reporting_month"].to_dict()
                selected = st.selectbox(
                    "Customer-month example",
                    examples["example_id"].tolist(),
                    format_func=lambda item: f"Example {item} · {months[item]}",
                )
                row = examples.loc[examples["example_id"] == selected].iloc[0]
                summary_columns = {
                    "feature_transaction_count": "Transactions",
                    "feature_total_absolute_amount": "Total absolute amount",
                    "feature_credit_share": "Credit share",
                    "feature_recent_30d_count": "Transactions in last 30 days",
                    "feature_days_since_last": "Days since latest transaction",
                    "feature_unique_event_types": "Unique event types",
                }
                available_summary = {
                    label: row[column]
                    for column, label in summary_columns.items()
                    if column in row.index
                }
                if available_summary:
                    st.write("**Transaction summary**")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Feature": available_summary.keys(),
                                "Value": available_summary.values(),
                            }
                        ),
                        hide_index=True,
                        width="stretch",
                    )
                actual = [_product_label(p) for p in PRODUCTS if row[f"actual_{p}"]]
                st.write(
                    f"**Observed next-month uptake:** {', '.join(actual) if actual else 'None'}"
                )

                ranking_rows = []
                for model_name in models:
                    for product in PRODUCTS:
                        column = _score_column(predictions, model_name, product)
                        if column:
                            ranking_rows.append(
                                {
                                    "Model": labels[model_name],
                                    "Product": _product_label(product),
                                    "Propensity score": float(row[column]),
                                }
                            )
                ranking = pd.DataFrame(ranking_rows).sort_values(
                    ["Model", "Propensity score"], ascending=[True, False]
                )
                st.dataframe(ranking, hide_index=True, width="stretch")
                selected_model = st.selectbox("Chart model", list(labels), format_func=labels.get)
                chart = ranking[ranking["Model"] == labels[selected_model]].set_index("Product")
                st.bar_chart(chart[["Propensity score"]])
                st.caption(
                    "A propensity score is the model's ranking score, not calibrated confidence. "
                    "The product names and example IDs are anonymized."
                )

    with limitations:
        st.subheader("What this project does—and does not—show")
        st.markdown(
            """
- It predicts recorded product uptake; it does not estimate whether making an offer causes uptake.
- MBD-mini products and transaction types are anonymized, so no real product strategy is inferred.
- Rare positives make small-sample metrics unstable; average precision should be read beside prevalence.
- A time split measures future-period performance; a client split measures generalization to unseen customers. They answer different questions.
- Deployment would require calibration, fairness and privacy review, monitoring, and human-approved business rules.
"""
        )


if __name__ == "__main__":
    main()
