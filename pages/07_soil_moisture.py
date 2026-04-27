"""
Page — Soil Moisture Analysis
Parquet files: daily.parquet, monthly.parquet, climatology.parquet, anomaly.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import linregress
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Soil Moisture", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

sw_vars    = ["swvl1", "swvl2", "swvl3", "swvl4"]
depth_lbls = ["7 cm", "28 cm", "100 cm", "289 cm"]
depths     = [0.035, 0.175, 0.640, 1.945]
thicknesses = np.array([0.07, 0.21, 0.72, 1.89])

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "soil_moisture"

st.title("Soil Moisture Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")
df_anom    = pd.read_parquet(f"{DATA_DIR}/anomaly.parquet")

for df in [df_daily, df_monthly, df_anom]:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

if "time" not in df_daily.columns:
    df_daily = df_daily.reset_index().rename(columns={"index": "time"})
    df_daily["time"] = pd.to_datetime(df_daily["time"])

st.markdown("---")


# Section 1 — Climatology by layer
st.subheader("1. Chỉ số độ ẩm của đất trên từng tầng qua 45 năm")

month_col = "month" if "month" in df_clim.columns else df_clim.index.name
clim_vals = df_clim.reset_index() if "month" not in df_clim.columns else df_clim

fig1 = go.Figure()
colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
for v, lbl, col in zip(sw_vars, depth_lbls, colors):
    fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim_vals[v].values,
        mode="lines+markers", line=dict(color=col, width=2), name=lbl))
fig1.update_layout(title="Chỉ số độ ẩm của đất trong 45 năm (Trung bình của các tháng)",
    xaxis_title="Tháng", yaxis_title="Độ ẩm của đất (m³/m³)", height=500)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Heatmap depth vs month
st.subheader("2. Heatmap về độ ẩm của đất")

clim_matrix = clim_vals[sw_vars].values.T

fig2 = go.Figure(data=go.Heatmap(
    z=clim_matrix, x=MONTH_LABELS, y=depth_lbls,
    colorscale="YlGnBu", zmin=0.05, zmax=0.45,
    colorbar=dict(title="Độ ẩm (m³/m³)")
))
fig2.update_yaxes(autorange="reversed")
fig2.update_layout(title="Heatmap tính độ ẩm đất theo độ sâu và thời gian",
    xaxis_title="Tháng", yaxis_title="Độ sâu", height=400)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Annual column water trend
st.subheader("3. Tổng lượng nước trung bình trong đất qua các năm")

d1, d2, d3, d4 = 0.07, 0.21, 0.72, 1.89
df_daily["daily_col"] = (df_daily["swvl1"]*d1 + df_daily["swvl2"]*d2 +
                         df_daily["swvl3"]*d3 + df_daily["swvl4"]*d4)

df_annual = df_daily.resample("YE", on="time")["daily_col"].mean().reset_index()
df_annual["year"] = df_annual["time"].dt.year
slope, intercept, r_value, p_value, _ = linregress(df_annual["year"], df_annual["daily_col"])
df_annual["trend"] = intercept + slope * df_annual["year"]

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=df_annual["year"], y=df_annual["daily_col"],
    mode="lines+markers", marker=dict(symbol="circle"), line=dict(color="navy"),
    name="Lượng nước trung bình năm"))
fig3.add_trace(go.Scatter(x=df_annual["year"], y=df_annual["trend"],
    mode="lines", line=dict(color="red", dash="dash", width=2),
    name=f"Xu hướng ({slope*1000:.4f} mm/year, p={p_value:.3f})"))

trend_label = "Dấu hiệu dài hạn: Khô hạn" if slope < 0 else "Dấu hiệu dài hạn: Ướt"
fig3.add_annotation(x=0.02, y=0.05, xref="paper", yref="paper",
    text=trend_label, showarrow=False,
    font=dict(color="red" if slope < 0 else "blue", size=13))
fig3.update_layout(title="Xu hướng của tổng lượng nước / độ ẩm trong đất (0-289cm)",
    xaxis_title="Năm", yaxis_title="Lượng nước tương đương (meters)", height=500)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Anomaly heatmaps by layer
st.subheader("4. Heatmap đo dị thường trong độ ẩm đất theo từng tầng")

df_anom["time"] = pd.to_datetime(df_anom["time"])
df_anom_monthly = df_anom.resample("MS", on="time").mean().reset_index()
df_anom_monthly["year"]  = df_anom_monthly["time"].dt.year
df_anom_monthly["month"] = df_anom_monthly["time"].dt.month

layers = sw_vars
max_anom = df_anom_monthly[layers].abs().max().max()
titles_layer = [
    "Tầng 1: 0-7 cm (Bề mặt)", "Tầng 2: 7-28 cm (Vùng rễ cây)",
    "Tầng 3: 28-100 cm (Tầng dưới mặt đất)", "Tầng 4: 100-289 cm (Tầng đất sâu)"
]

fig4 = make_subplots(rows=2, cols=2, subplot_titles=titles_layer)
positions = [(1,1),(1,2),(2,1),(2,2)]

for (r, c), layer in zip(positions, layers):
    pivot = df_anom_monthly.pivot(index="year", columns="month", values=layer)
    fig4.add_trace(go.Heatmap(
        z=pivot.values, x=MONTH_LABELS, y=pivot.index,
        colorscale="RdBu", zmid=0, zmin=-max_anom, zmax=max_anom,
        showscale=(r==2 and c==2),
        colorbar=dict(title="Dị thường (m³/m³)") if r==2 and c==2 else None
    ), row=r, col=c)

fig4.update_layout(height=800, title="Dị thường trong độ ẩm đất (Trung bình tháng)")
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Cross-correlation lag
st.subheader("5. Độ trễ lan truyền ẩm giữa các tầng đất")

pairs  = [("swvl1","swvl2"), ("swvl2","swvl3"), ("swvl3","swvl4")]
labels = ["Tầng 1 → 2 (0.07m to 0.28m)",
          "Tầng 2 → 3 (0.28m to 1.00m)",
          "Tầng 3 → 4 (1.00m to 2.89m)"]
max_lag = 90
lags = range(max_lag + 1)
colors_pairs = ["#3498db", "#e67e22", "#2ecc71"]

fig5 = go.Figure()
for (p1, p2), label, color in zip(pairs, labels, colors_pairs):
    corrs = [df_daily[p1].corr(df_daily[p2].shift(-lag)) for lag in lags]
    peak_lag = int(np.argmax(corrs))
    peak_val = float(np.max(corrs))
    fig5.add_trace(go.Scatter(x=list(lags), y=corrs,
        mode="lines", line=dict(color=color, width=2.5),
        name=f"{label} (Lâu nhất: {peak_lag} ngày)"))
    fig5.add_vline(x=peak_lag, line=dict(color=color, dash="dash", width=1), opacity=0.3)

fig5.update_layout(title="Độ trễ trong việc truyền độ ẩm giữa các tầng đất",
    xaxis_title="Thời gian trễ (Ngày)", yaxis_title="Hệ số tương quan Pearson (r)", height=500)
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — SSI
st.subheader("6. Chỉ số độ ẩm đất chuẩn hoá (SSI)")

weights = [0.07, 0.21, 0.72, 1.89]
df_monthly_ssi = df_monthly.copy()
if "time" in df_monthly_ssi.columns:
    df_monthly_ssi["time"] = pd.to_datetime(df_monthly_ssi["time"])
else:
    df_monthly_ssi = df_monthly_ssi.reset_index()
    df_monthly_ssi["time"] = pd.to_datetime(df_monthly_ssi["time"])

df_monthly_ssi["col_sm"] = sum(df_monthly_ssi[v]*w for v, w in zip(sw_vars, weights))
df_monthly_ssi["month"]  = df_monthly_ssi["time"].dt.month
clim_ssi = df_monthly_ssi.groupby("month")["col_sm"].agg(["mean", "std"]).reset_index()
clim_ssi.columns = ["month", "climatology_mean", "climatology_std"]
df_ssi = df_monthly_ssi.merge(clim_ssi, on="month")
df_ssi["ssi"] = (df_ssi["col_sm"] - df_ssi["climatology_mean"]) / df_ssi["climatology_std"]
df_ssi = df_ssi.sort_values("time")

colors_ssi = ["red" if v < 0 else "blue" for v in df_ssi["ssi"]]

fig6 = go.Figure()
fig6.add_trace(go.Bar(x=df_ssi["time"], y=df_ssi["ssi"],
    marker_color=colors_ssi, opacity=0.8, name="SSI", width=31*24*3600*1000))
fig6.add_hline(y=0, line=dict(color="black", width=0.8))
fig6.add_hline(y=1.5, line=dict(color="blue", dash="dash"), opacity=0.3,
    annotation_text="Rất ẩm ướt (>1.5)", annotation_position="top right")
fig6.add_hline(y=-1.5, line=dict(color="red", dash="dash"), opacity=0.3,
    annotation_text="Khô hạn mạnh (<-1.5)", annotation_position="bottom right")
fig6.update_layout(title="Chỉ số độ ẩm đất được chuẩn hoá (SSI) | 1980-2024",
    xaxis_title="Năm", yaxis_title="Độ lệch chuẩn (σ)", height=550)
st.plotly_chart(fig6, width="stretch")
