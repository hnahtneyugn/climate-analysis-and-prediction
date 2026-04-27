"""
Page — Moisture Divergence Analysis
Parquet files expected:
  daily.parquet       — columns: time, avg_vimdf, year, month
  monthly.parquet     — columns: time, avg_vimdf
  climatology.parquet — columns: month, avg_vimdf
  anomaly.parquet     — columns: time, avg_vimdf, year, month
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Moisture Divergence", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "moisture_divergence"

st.title("Moisture Divergence Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")
df_anom    = pd.read_parquet(f"{DATA_DIR}/anomaly.parquet")

for df in (df_daily, df_monthly, df_anom):
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

if "year" not in df_daily.columns:
    df_daily["year"]  = df_daily["time"].dt.year
if "month" not in df_daily.columns:
    df_daily["month"] = df_daily["time"].dt.month

if "year" not in df_anom.columns:
    df_anom["year"]  = df_anom["time"].dt.year
if "month" not in df_anom.columns:
    df_anom["month"] = df_anom["time"].dt.month

st.markdown("---")


# Section 1 — Heatmap dị thường trong độ hội tụ ẩm

st.subheader("1. Heatmap dị thường trong độ hội tụ ẩm")
st.markdown(
    "Monthly moisture convergence anomaly over 45 years. "
    "Darker summer months in recent decades suggest strengthening convergence."
)

monthly_anom = df_anom.groupby(["year", "month"])["avg_vimdf"].mean().reset_index()
pivot = monthly_anom.pivot(index="year", columns="month", values="avg_vimdf")

fig1 = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=MONTH_LABELS,
    y=pivot.index,
    colorscale="RdBu_r",
    zmid=0,
    colorbar=dict(title="Dị thường hội tụ ẩm")
))
fig1.update_layout(
    title="Heatmap dị thường trong độ hội tụ ẩm",
    xaxis_title="Tháng", yaxis_title="Năm",
    height=600
)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Xu hướng lâu dài

st.subheader("2. Xu hướng lâu dài của độ hội tụ ẩm trung bình năm")
st.markdown(
    "Annual mean moisture convergence with linear regression trend. "
    "A declining slope indicates drying tendency in recent years."
)

annual_mc = df_daily.groupby("year")["avg_vimdf"].mean().reset_index()

slope, intercept, r_val, p_val, _ = stats.linregress(annual_mc["year"], annual_mc["avg_vimdf"])
trend_line = intercept + slope * annual_mc["year"]

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=annual_mc["year"], y=annual_mc["avg_vimdf"],
    mode="markers", marker=dict(size=7, color="#185FA5", opacity=0.65),
    name="Trung bình năm"
))
fig2.add_trace(go.Scatter(
    x=annual_mc["year"], y=trend_line,
    mode="lines", line=dict(color="red", width=2),
    name=f"Xu hướng: {slope*10:.4f}/decade (p={p_val:.3f})"
))
fig2.update_layout(
    title="Xu hướng lâu dài của độ hội tụ ẩm trung bình năm",
    xaxis_title="Năm",
    yaxis_title="Trung bình độ hội tụ ẩm (avg_vimdf)",
    hovermode="x unified", height=450
)
st.plotly_chart(fig2, width="stretch")
