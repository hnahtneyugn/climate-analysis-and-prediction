"""
Page — Air & Surface Temperature Analysis
Parquet files: daily.parquet, monthly.parquet, climatology.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Air & Surface Temperature", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "air_n_surface_temp"

st.title("Air & Surface Temperature Analysis")

df_daily     = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly   = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
climatology  = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

df_daily.index   = pd.to_datetime(df_daily.index)
df_monthly.index = pd.to_datetime(df_monthly.index)

if "DTR" not in df_daily.columns:
    df_daily["DTR"] = df_daily["tmax"] - df_daily["tmin"]
if "Decade" not in df_daily.columns:
    df_daily["Decade"] = (df_daily.index.year // 10) * 10

st.success("Data loaded", icon="✅")
st.markdown("---")


# Section 1 — Annual trend
st.subheader("1. Xu hướng thay đổi nhiệt độ hàng năm")

df_yearly = df_daily[["t2m", "tskin"]].resample("YS").mean()
df_yearly["Year"] = df_yearly.index.year

slope_t2m, int_t2m, _, _, _ = stats.linregress(df_yearly["Year"], df_yearly["t2m"])
slope_sk,  int_sk,  _, _, _ = stats.linregress(df_yearly["Year"], df_yearly["tskin"])

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=df_yearly["Year"], y=df_yearly["t2m"],
    mode="markers", marker=dict(color="blue", size=6, opacity=0.5), name="Nhiệt độ không khí (2m)"))
fig1.add_trace(go.Scatter(x=df_yearly["Year"],
    y=int_t2m + slope_t2m * df_yearly["Year"],
    mode="lines", line=dict(color="blue", dash="dash"), name=f"T2m trend ({slope_t2m*10:.3f}°C/decade)"))
fig1.add_trace(go.Scatter(x=df_yearly["Year"], y=df_yearly["tskin"],
    mode="markers", marker=dict(color="red", size=6, opacity=0.5), name="Nhiệt độ bề mặt đất"))
fig1.add_trace(go.Scatter(x=df_yearly["Year"],
    y=int_sk + slope_sk * df_yearly["Year"],
    mode="lines", line=dict(color="red", dash="dash"), name=f"Tskin trend ({slope_sk*10:.3f}°C/decade)"))
fig1.update_layout(title="Xu hướng thay đổi nhiệt độ hàng năm tại Miền Trung (1980-2024)",
    xaxis_title="Năm", yaxis_title="Nhiệt độ (°C)", height=500, hovermode="x unified")
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Seasonal climatology
st.subheader("2. Chu kỳ mùa trung bình (Climatology)")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=climatology["t2m"],
    mode="lines+markers", marker=dict(symbol="circle"), line=dict(color="blue", width=2.5),
    name="Nhiệt độ không khí (2m)"))
fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=climatology["tskin"],
    mode="lines+markers", marker=dict(symbol="square"), line=dict(color="red", width=2.5),
    name="Nhiệt độ bề mặt đất",
    fill="tonexty", fillcolor="rgba(128,128,128,0.2)"))
fig2.update_layout(title="Chu kỳ mùa trung bình (Dữ liệu 45 năm)",
    xaxis_title="Tháng", yaxis_title="Nhiệt độ trung bình (°C)", height=500)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Monthly anomaly heatmap
st.subheader("3. Ma trận dị thường nhiệt độ hàng tháng")

pivot_anom = df_monthly.copy()
pivot_anom["Year"] = pivot_anom.index.year
pivot_anom["Month"] = pivot_anom.index.month
pivot_table = pivot_anom.pivot(index="Year", columns="Month", values="t2m_anom")

fig3 = go.Figure(data=go.Heatmap(
    z=pivot_table.values,
    x=MONTH_LABELS,
    y=pivot_table.index,
    colorscale="RdBu_r",
    zmid=0, zmin=-2, zmax=2,
    colorbar=dict(title="Dị thường (°C)")
))
fig3.update_layout(title="Ma trận dị thường nhiệt độ hàng tháng qua các năm",
    xaxis_title="Tháng", yaxis_title="Năm", height=600)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Hot days frequency
st.subheader("4. Tần suất các ngày nắng nóng cực đoan")

hot_days = (df_daily["tmax"] > 32).resample("YS").sum()
rolling5 = hot_days.rolling(5).mean()

fig4 = go.Figure()
fig4.add_trace(go.Bar(x=hot_days.index.year, y=hot_days.values,
    marker_color="orange", opacity=0.7, name="Số ngày > 32°C"))
fig4.add_trace(go.Scatter(x=hot_days.index.year, y=rolling5.values,
    mode="lines", line=dict(color="red", width=2), name="Trung bình trượt 5 năm"))
fig4.update_layout(title="Tần suất số ngày nắng nóng cực đoan hàng năm",
    xaxis_title="Năm", yaxis_title="Số ngày", height=500, hovermode="x unified")
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — DTR trend
st.subheader("5. Biến động biên độ nhiệt ngày (DTR)")

dtr_yearly = df_daily["DTR"].resample("YS").mean()
x_dtr = dtr_yearly.index.year
y_dtr = dtr_yearly.values
slope_dtr, int_dtr, _, _, _ = stats.linregress(x_dtr, y_dtr)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=x_dtr, y=y_dtr,
    mode="lines+markers", marker=dict(symbol="circle", size=5), line=dict(color="purple"),
    name="DTR trung bình năm"))
fig5.add_trace(go.Scatter(x=x_dtr, y=int_dtr + slope_dtr * x_dtr,
    mode="lines", line=dict(color="black", dash="dash"), name=f"Trend ({slope_dtr*10:.3f}°C/decade)"))
fig5.update_layout(title="Biến động biên độ nhiệt ngày trung bình (Tmax - Tmin)",
    xaxis_title="Năm", yaxis_title="Biên độ nhiệt (°C)", height=500)
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — Decade distribution
st.subheader("6. Sự dịch chuyển phân phối nhiệt độ qua các thập kỷ")

df_daily["Decade_Label"] = df_daily["Decade"].astype(str) + "s"
decades = sorted(df_daily["Decade_Label"].unique())

fig6 = go.Figure()
for decade in decades:
    subset = df_daily[df_daily["Decade_Label"] == decade]["t2m"]
    fig6.add_trace(go.Box(y=subset, name=decade, boxmean=True))
fig6.update_layout(title="Sự dịch chuyển phân phối nhiệt độ qua các thập kỷ",
    xaxis_title="Thập kỷ", yaxis_title="Nhiệt độ không khí hàng ngày (°C)", height=500)
st.plotly_chart(fig6, width="stretch")
st.markdown("---")


# Section 7 — Scatter T2m vs Tskin
st.subheader("7. Tương quan vật lý giữa bề mặt đất và khí quyển")

sample = df_daily.sample(min(5000, len(df_daily)), random_state=42)
fig7 = go.Figure()
fig7.add_trace(go.Scatter(
    x=sample["t2m"], y=sample["tskin"],
    mode="markers",
    marker=dict(color=sample.index.month, colorscale="Spectral_r", opacity=0.4, size=4,
                colorbar=dict(title="Tháng")),
    name="Daily data"
))
lim = [df_daily[["t2m", "tskin"]].min().min(), df_daily[["t2m", "tskin"]].max().max()]
fig7.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
    line=dict(color="black", dash="dash"), name="y = x"))
fig7.update_layout(title="Tương quan giữa Nhiệt độ bề mặt đất và Nhiệt độ không khí",
    xaxis_title="Nhiệt độ không khí (2m) (°C)", yaxis_title="Nhiệt độ bề mặt đất (°C)", height=600)
st.plotly_chart(fig7, width="stretch")
