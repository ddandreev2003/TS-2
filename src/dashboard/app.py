"""Streamlit dashboard for all ML pipeline stages."""

from __future__ import annotations

import sys
from pathlib import Path

# Streamlit is often started without PYTHONPATH=.; ensure project root is importable.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.dashboard import data_loaders as dl

st.set_page_config(page_title="TS-2 Liquidity Forecast", layout="wide")

PAGES = {
    "Overview": "overview",
    "EDA": "eda",
    "Feature Selection": "features",
    "Models": "models",
    "Forecasts": "forecasts",
    "Drift & Anomalies": "drift",
    "Retrain": "retrain",
}


def _show_plots(group: str, title: str) -> None:
    st.subheader(title)
    plots = dl.list_plot_files(group)
    if not plots:
        st.warning(f"No plots found for `{group}`. Run `python -m pipelines.report_pipeline`.")
        return
    cols = st.columns(2)
    for i, plot_path in enumerate(plots):
        with cols[i % 2]:
            st.image(str(plot_path), use_container_width=True)


def page_overview() -> None:
    st.title("Pipeline Overview")
    metrics = dl.load_metrics()
    eval_data = dl.load_eval()
    manifest = dl.load_manifest()
    cal = dl.load_calibration()
    drift = dl.load_drift_status()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train rows", metrics.get("n_train", "—"))
    c2.metric("Test rows", metrics.get("n_test", "—"))
    c3.metric("Plots generated", manifest.get("n_plots", 0))
    c4.metric("Drift status", drift.get("status", "unknown"))

    st.markdown("### Best model")
    best = manifest.get("best_model") or "—"
    st.write(f"**{best}**")

    st.markdown("### Feature selection")
    st.write(f"Selected method: **{metrics.get('feature_selection_method', '—')}**")

    st.markdown("### Calibration policy")
    policy = cal.get("policy", {})
    st.write(policy.get("rationale", "No calibration study yet."))
    st.write(f"Recommended cadence: **{policy.get('recommended_cadence_days', '—')} days**")

    ranking = dl.model_ranking_df()
    if not ranking.empty:
        st.markdown("### Model ranking (Balance MAE)")
        st.dataframe(ranking, use_container_width=True)


def page_eda() -> None:
    st.title("Exploratory Data Analysis")
    _show_plots("eda", "EDA plots (notebook parity)")


def page_features() -> None:
    st.title("Feature Selection")
    metrics = dl.load_metrics()
    ranking = metrics.get("feature_selection_ranking", [])
    if ranking:
        st.dataframe(pd.DataFrame(ranking), use_container_width=True)
    _show_plots("features", "Stability & correlation plots")


def page_models() -> None:
    st.title("Model Comparison")
    _show_plots("models", "MAE and business cost charts")
    ranking = dl.model_ranking_df()
    if not ranking.empty:
        st.dataframe(ranking, use_container_width=True)


def page_forecasts() -> None:
    st.title("Hold-out Forecasts")
    holdout = dl.load_holdout()
    models = list(holdout.get("models", {}).keys())
    if not models:
        st.warning("No hold-out predictions. Run report pipeline.")
        _show_plots("models", "Forecast grid")
        return

    selected = st.selectbox("Model", models)
    payload = holdout["models"][selected]
    df = pd.DataFrame(
        {
            "date": holdout.get("dates", list(range(len(holdout.get("y_true", []))))),
            "y_true": holdout.get("y_true", []),
            "y_pred": payload.get("y_pred", []),
        }
    )
    st.line_chart(df.set_index("date")[["y_true", "y_pred"]])
    st.dataframe(df.tail(20), use_container_width=True)
    _show_plots("models", "Forecast grid & per-model charts")


def page_drift() -> None:
    st.title("Drift & Anomalies")
    drift = dl.load_drift_status()
    st.json(drift)
    alerts_dir = Path("drift_alerts")
    if alerts_dir.exists():
        alerts = sorted(alerts_dir.glob("alert_*.json"))
        st.write(f"Saved alerts: {len(alerts)}")
    _show_plots("drift", "Residual & control charts")


def page_retrain() -> None:
    st.title("Retrain History")
    events = dl.retrain_events()
    if not events:
        st.info("No retrain events yet.")
        return
    for event in reversed(events):
        with st.expander(f"{event.get('completed_at', 'event')} — {event.get('reason', '')}"):
            st.json(event)


def main() -> None:
    st.sidebar.title("TS-2 Dashboard")
    page = st.sidebar.radio("Stage", list(PAGES.keys()))
    {
        "Overview": page_overview,
        "EDA": page_eda,
        "Feature Selection": page_features,
        "Models": page_models,
        "Forecasts": page_forecasts,
        "Drift & Anomalies": page_drift,
        "Retrain": page_retrain,
    }[page]()


if __name__ == "__main__":
    main()
