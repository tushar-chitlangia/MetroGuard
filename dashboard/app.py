from pathlib import Path
import json
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

st.set_page_config(
    page_title="MetroGuard | Compressor Monitoring",
    page_icon="🚇",
    layout="wide",
)

st.markdown("""
<style>
    .block-container {padding-top: 1.6rem; padding-bottom: 2.2rem; max-width: 1500px;}
  [data-testid="stMetric"] {background: #202733 !important; color: #f4f7fb !important; padding: 14px 16px; border-radius: 10px; border: 1px solid #394354;} [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {color: #cbd5e1 !important; opacity: 1 !important;} [data-testid="stMetricValue"], [data-testid="stMetricValue"] * {color: #ffffff !important; opacity: 1 !important;} 
    .small-note {font-size: 0.88rem; color: #536174;}
    .status-warning {background:#fff5e8; color:#74400b; padding:12px 14px; border-radius:8px; border:1px solid #f5d7ad;}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_csv(name):
    return pd.read_csv(ARTIFACT_DIR / name, parse_dates=["timestamp"] if name.endswith("predictions.csv") else None)


def get_alert_episodes(df, alert_col="isolation_forest_alert"):
    alerts = df.loc[df[alert_col].eq(1), ["timestamp", alert_col]].copy()
    if alerts.empty:
        return pd.DataFrame(columns=["Start", "End", "Minutes"])
    alerts = alerts.sort_values("timestamp")
    new_episode = alerts["timestamp"].diff().gt(pd.Timedelta(minutes=1))
    alerts["episode"] = new_episode.cumsum()
    episodes = alerts.groupby("episode").agg(
        Start=("timestamp", "min"), End=("timestamp", "max"), Minutes=("timestamp", "size")
    ).reset_index(drop=True)
    return episodes.sort_values("Start", ascending=False)


required = ["test_predictions.csv", "validation_predictions.csv", "model_metrics.csv", "validation_metrics.csv"]
missing = [name for name in required if not (ARTIFACT_DIR / name).exists()]
if missing:
    st.error("Project output files are missing: " + ", ".join(missing))
    st.info("Extract the provided project package without changing the folder structure, then run the app again.")
    st.stop()

test_df = load_csv("test_predictions.csv")
val_df = load_csv("validation_predictions.csv")
test_metrics = load_csv("model_metrics.csv")
val_metrics = load_csv("validation_metrics.csv")

st.title("MetroGuard")
st.caption("Metro train air-compressor condition monitoring · MetroPT-3 historical evaluation")

st.markdown(
    '<div class="status-warning"><b>Evaluation note:</b> At the threshold selected using June validation data, '
    'the saved July test results show no alerts during the assumed two-hour warning window for the reported incident. '
    'This is an experimental anomaly-monitoring prototype, not a validated failure-prediction or safety system.</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("View settings")
    period = st.radio("Evaluation period", ["July test", "June validation"], index=0)
    active_df = test_df if period == "July test" else val_df
    metrics_df = test_metrics if period == "July test" else val_metrics
    st.divider()
    st.write("**Data source**")
    st.caption("UCI MetroPT-3 dataset. The dashboard displays saved outputs; it does not retrain the model.")

model_row = metrics_df.loc[metrics_df["Model"].eq("Isolation Forest")]
if model_row.empty:
    st.error("Isolation Forest metrics were not found in the saved metrics file.")
    st.stop()
model_row = model_row.iloc[0]
threshold = float(model_row["Threshold"])
alerts = active_df["isolation_forest_alert"].astype(int)
labels = active_df["assumed_warning_label"].astype(int)
warning_minutes = int(labels.sum())
detected_warning_minutes = int(((alerts == 1) & (labels == 1)).sum())
false_alert_minutes = int(((alerts == 1) & (labels == 0)).sum())

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Minutes analysed", f"{len(active_df):,}")
c2.metric("ML alert minutes", f"{int(alerts.sum()):,}")
c3.metric("Assumed warning minutes", f"{warning_minutes:,}")
c4.metric("Warning minutes flagged", f"{detected_warning_minutes:,}")
c5.metric("Alerts outside assumed warning window", f"{false_alert_minutes:,}")

st.markdown("### Model evaluation")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Precision", f"{float(model_row['Precision']):.3f}")
m2.metric("Recall", f"{float(model_row['Recall']):.3f}")
m3.metric("F1-score", f"{float(model_row['F1-score']):.3f}")
m4.metric("PR-AUC", f"{float(model_row['PR-AUC']):.4f}")
st.caption(f"Fixed threshold from validation: {threshold:.6f}. Warning labels are an assumed two-hour window before a reported incident, not verified labels for every minute.")

summary_tab, trend_tab, alerts_tab, comparison_tab = st.tabs(
    ["Overview", "Sensor trends", "Alert timeline", "Model comparison"]
)

with summary_tab:
    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Anomaly score over time")
        chart_df = active_df[["timestamp", "isolation_forest_score", "assumed_warning_label", "isolation_forest_alert"]].copy()
        if len(chart_df) > 12000:
            chart_df = chart_df.iloc[::5].copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=chart_df["timestamp"], y=chart_df["isolation_forest_score"], mode="lines", name="Anomaly score", line=dict(width=1.4, color="#2878b5")))
        fig.add_hline(y=threshold, line_dash="dash", line_color="#cc6b31", annotation_text="Validation threshold")
        flagged = chart_df[chart_df["isolation_forest_alert"].eq(1)]
        if not flagged.empty:
            fig.add_trace(go.Scatter(x=flagged["timestamp"], y=flagged["isolation_forest_score"], mode="markers", name="ML alert", marker=dict(color="#cc6b31", size=5)))
        warn = chart_df[chart_df["assumed_warning_label"].eq(1)]
        if not warn.empty:
            fig.add_trace(go.Scatter(x=warn["timestamp"], y=warn["isolation_forest_score"], mode="markers", name="Assumed warning window", marker=dict(color="#8b3fa8", size=5, symbol="x")))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02), xaxis_title="Time", yaxis_title="Score")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Baseline vs ML")
        comp = metrics_df[["Model", "Precision", "Recall", "F1-score", "PR-AUC", "False positives per 1000 normal minutes"]].copy()
        st.dataframe(comp.round(4), use_container_width=True, hide_index=True)
        st.markdown("**How to read this**")
        st.write("The threshold is fixed from validation data. July is treated as held-out test data and is not used to tune the threshold.")
        st.write("A flagged minute means the sensor pattern crossed the anomaly threshold. It does not prove that a compressor failure occurred.")

with trend_tab:
    st.subheader("Sensor signals")
    sensor_cols = [c for c in ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"] if c in active_df.columns]
    selected = st.selectbox("Sensor", sensor_cols, index=sensor_cols.index("Oil_temperature") if "Oil_temperature" in sensor_cols else 0)
    date_min = active_df["timestamp"].min().date()
    date_max = active_df["timestamp"].max().date()
    date_range = st.date_input("Time range", value=(date_min, date_max), min_value=date_min, max_value=date_max)
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = date_min, date_max
    view = active_df.loc[
        active_df["timestamp"].between(pd.Timestamp(start_date), pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1))
    ].copy()
    if len(view) > 15000:
        view = view.iloc[::max(1, len(view)//15000)].copy()
    fig = px.line(view, x="timestamp", y=selected, title=f"{selected} over time")
    fig.update_traces(line_width=1.4, line_color="#2878b5")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=50, b=10), xaxis_title="Time", yaxis_title=selected)
    st.plotly_chart(fig, use_container_width=True)
    if "isolation_forest_alert" in view:
        flagged = view[view["isolation_forest_alert"].eq(1)]
        if not flagged.empty:
            st.caption(f"Orange alert markers are summarized in the Alert timeline tab. {len(flagged):,} alert minutes fall in this time range.")

with alerts_tab:
    st.subheader("Alert episodes")
    episodes = get_alert_episodes(active_df)
    if episodes.empty:
        st.info("No Isolation Forest alert minutes were saved for this period at the selected threshold.")
    else:
        st.write(f"Consecutive alert minutes are grouped into episodes. {len(episodes):,} episodes were found.")
        st.dataframe(episodes.head(100).style.format({"Minutes": "{:,.0f}"}), use_container_width=True, hide_index=True)
        st.download_button("Download alert episodes (CSV)", episodes.to_csv(index=False).encode("utf-8"), file_name=f"{period.lower().replace(' ', '_')}_alert_episodes.csv", mime="text/csv")
    st.subheader("Assumed warning-window check")
    st.write(f"Assumed warning minutes flagged by the model: **{detected_warning_minutes:,} / {warning_minutes:,}**.")
    if warning_minutes and detected_warning_minutes == 0:
        st.warning("No assumed warning minutes were flagged at this fixed threshold. Do not describe this run as successful early-failure prediction.")

with comparison_tab:
    st.subheader("Statistical baseline and Isolation Forest")
    compare = metrics_df.set_index("Model")[["Precision", "Recall", "F1-score", "PR-AUC", "False positives per 1000 normal minutes"]]
    st.dataframe(compare.round(4), use_container_width=True)
    fig = px.bar(metrics_df, x="Model", y="PR-AUC", title="Precision–recall average precision", text_auto=".4f")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=50, b=10), yaxis_title="Average precision")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("**Important limitation:** The dataset provides four reported incident periods, not verified labels for every timestamp. The two-hour warning windows are a project assumption; the July test period contains one incident, so the result does not establish general failure-prediction performance.")

st.divider()
st.caption("MetroGuard · AUCML ESE prototype · Historical sensor analysis only; not a live railway monitoring or safety system.")
