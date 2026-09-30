"""Read-only Streamlit view of local MBD-mini experiment outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from bankrec.dashboard_data import available_runs, load_run, product_metrics

PRODUCTS = tuple(f"product_{i}" for i in range(1, 5))
DEFAULT_ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"


def _percentage(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def main() -> None:
    st.set_page_config(page_title="Banking Product Propensity Modeling", layout="wide")
    st.title("Banking Product Propensity Modeling")
    st.caption(
        "MBD-mini transaction histories · four anonymized products · client-disjoint evaluation"
    )
    st.info(
        "Research/portfolio experiment only. Scores describe historical product uptake, "
        "not the effect of making an offer or a recommendation for a real customer."
    )

    root = Path(st.sidebar.text_input("Local artifacts folder", str(DEFAULT_ARTIFACTS)))
    runs = available_runs(root)
    if not runs:
        st.warning("No experiment results found. Run the model first to create metrics.json.")
        st.code(
            "python -m bankrec.cli --data-dir data/raw --max-clients 1000 "
            "--max-events 32 --epochs 2 --output-dir artifacts/my-run",
            language="text",
        )
        st.stop()
    folder = st.sidebar.selectbox(
        "Experiment",
        runs,
        format_func=lambda path: str(path.relative_to(root)) if path != root else "default",
    )
    split = st.sidebar.radio("Evaluation split", ["test", "validation"])
    try:
        report, predictions = load_run(folder)
        baseline = report["baseline"][split]
        transformer = report["transformer"][split]
        metrics = product_metrics(report, split)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"Could not read this run: {exc}")
        st.stop()

    st.subheader("Model comparison")
    st.write(f"**Sampled clients:** {report['sampled_clients']:,} · **Split:** {split.title()}")
    st.caption(
        f"{baseline['examples']:,} client-month examples; "
        f"{baseline['positive_clients']:,} contain at least one product uptake. "
        "Ranking metrics below are calculated only on those buyer-month examples."
    )
    left, middle, right = st.columns(3)
    left.metric(
        "Buyer-only hit@1 · Transformer",
        _percentage(transformer["hit_at_1_among_buyers"]),
        delta=(
            f"{(transformer['hit_at_1_among_buyers'] - baseline['hit_at_1_among_buyers']):+.1%} "
            "vs baseline"
            if transformer["hit_at_1_among_buyers"] is not None
            and baseline["hit_at_1_among_buyers"] is not None
            else None
        ),
    )
    middle.metric("Buyer-only hit@1 · Baseline", _percentage(baseline["hit_at_1_among_buyers"]))
    right.metric(
        "Buyer-only recall@2 · Transformer",
        _percentage(transformer["recall_at_2_among_buyers"]),
    )

    st.subheader("Average precision by product")
    st.caption(
        "Higher is better. Product uptake is rare, so compare average precision "
        "with each product's prevalence below."
    )
    chart = metrics.set_index("product")[["baseline_ap", "transformer_ap"]]
    st.bar_chart(chart)
    st.dataframe(
        metrics.rename(
            columns={
                "product": "Product",
                "prevalence": "Positive prevalence",
                "baseline_ap": "Baseline AP",
                "transformer_ap": "Transformer AP",
                "baseline_auc": "Baseline ROC-AUC",
                "transformer_auc": "Transformer ROC-AUC",
            }
        ),
        hide_index=True,
        width="stretch",
    )

    if report.get("history"):
        st.subheader("Training loss")
        st.line_chart(pd.DataFrame(report["history"]).set_index("epoch"))

    st.subheader("Explore a held-out example")
    if predictions is None:
        st.info(
            "This run predates the dashboard and has no saved example-level predictions. "
            "Run the model again with the current code to enable this section."
        )
        return
    examples = predictions[predictions["split"] == split].copy()
    if examples.empty:
        st.info("No examples were saved for this split.")
        return
    buyer_only = st.checkbox("Show only rows with product uptake", value=True)
    if buyer_only:
        actual_cols = [f"actual_{product}" for product in PRODUCTS]
        examples = examples[examples[actual_cols].sum(axis=1) > 0]
    if examples.empty:
        st.info("No buyer-month examples in this split. Turn off the filter to view all rows.")
        return
    if len(examples) > 500:
        st.caption("Showing the first 500 examples in this view.")
        examples = examples.head(500)
    identifiers = examples["example_id"].tolist()
    months_by_id = examples.set_index("example_id")["reporting_month"].to_dict()
    selected = st.selectbox(
        "Historical client-month example",
        identifiers,
        format_func=lambda item: f"Example {item} · {months_by_id[item]}",
    )
    row = examples.loc[examples["example_id"] == selected].iloc[0]
    actual = [product.replace("_", " ").title() for product in PRODUCTS if row[f"actual_{product}"]]
    st.write(f"**Reporting month:** {row['reporting_month']}")
    st.write(f"**Transactions used:** {int(row['history_event_count'])}")
    st.write(f"**Products taken up next month:** {', '.join(actual) if actual else 'None'}")
    prediction_view = pd.DataFrame(
        [
            {
                "Product": product.replace("_", " ").title(),
                "Actual uptake": "Yes" if row[f"actual_{product}"] else "No",
                "Baseline score": float(row[f"baseline_{product}"]),
                "Transformer score": float(row[f"transformer_{product}"]),
            }
            for product in PRODUCTS
        ]
    )
    st.bar_chart(prediction_view.set_index("Product")[["Baseline score", "Transformer score"]])
    st.dataframe(prediction_view, hide_index=True, width="stretch")
    st.caption(
        "These are saved scores for a held-out historical example, not predictions "
        "for a new customer. Scores have not been calibrated for decision-making."
    )


if __name__ == "__main__":
    main()
