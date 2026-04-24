"""
Page — Dewpoint & Humidity Analysis
Parquet files: daily.parquet, monthly.parquet, yearly.parquet, anomaly.parquet, climatology.parquet, hourly.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Dewpoint & Humidity", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "dewpts_n_humid"

st.title("Dewpoint & Humidity Analysis")

df_daily     = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly   = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_yearly    = pd.read_parquet(f"{DATA_DIR}/yearly.parquet")
df_anomaly   = pd.read_parquet(f"{DATA_DIR}/anomaly.parquet")
climatology  = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")
df_hourly    = pd.read_parquet(f"{DATA_DIR}/hourly.parquet")

df_daily.index   = pd.to_datetime(df_daily.index)
df_monthly.index = pd.to_datetime(df_monthly.index)
df_yearly.index  = pd.to_datetime(df_yearly.index)
df_hourly.index  = pd.to_datetime(df_hourly.index)

st.success("Data loaded", icon="✅")
st.markdown("---")


# Section 1 — Seasonal climatology
st.subheader("1. Khí hậu học theo mùa: Nhiệt độ điểm sương vs Nhiệt độ không khí")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=climatology["t2m"],
    mode="lines+markers", marker=dict(symbol="circle"), line=dict(color="red", width=2.5),
    name="Nhiệt độ không khí (T2m)"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=climatology["d2m"],
    mode="lines+markers", marker=dict(symbol="square"), line=dict(color="blue", width=2.5),
    fill="tonexty", fillcolor="rgba(128,128,128,0.2)",
    name="Nhiệt độ điểm sương (Td)"))
fig1.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
    marker=dict(size=10, color="rgba(128,128,128,0.4)", symbol="square"),
    name="Chênh lệch điểm sương (DPD)"))
fig1.update_layout(title="Khí hậu học theo mùa: Nhiệt độ điểm sương vs Nhiệt độ không khí",
    xaxis_title="Tháng", yaxis_title="Nhiệt độ (°C)", height=500)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — DPD seasonal and trend
st.subheader("2. Chu kỳ mùa và xu hướng của chênh lệch điểm sương (DPD)")

x_yr = df_yearly.index.year.values
y_dpd = df_yearly["dpd"].values
slope_dpd, int_dpd, _, _, _ = stats.linregress(x_yr, y_dpd)

fig2 = make_subplots(rows=1, cols=2,
    subplot_titles=("Chu kỳ mùa của DPD", "Xu hướng dài hạn của DPD"))
fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=climatology["dpd"],
    mode="lines+markers", marker=dict(symbol="diamond"), line=dict(color="purple", width=2),
    showlegend=False), row=1, col=1)
fig2.add_trace(go.Scatter(x=x_yr, y=y_dpd,
    mode="lines", line=dict(color="purple", width=1.5), opacity=0.6,
    showlegend=False), row=1, col=2)
fig2.add_trace(go.Scatter(x=x_yr, y=int_dpd + slope_dpd * x_yr,
    mode="lines", line=dict(color="black", dash="dash", width=2.5),
    name=f"Xu hướng: {slope_dpd*10:.3f}°C/thập kỷ"), row=1, col=2)
fig2.update_xaxes(title_text="Tháng", row=1, col=1)
fig2.update_yaxes(title_text="DPD (°C)", row=1, col=1)
fig2.update_xaxes(title_text="Năm", row=1, col=2)
fig2.update_yaxes(title_text="DPD trung bình năm (°C)", row=1, col=2)
fig2.update_layout(height=500)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Long-term trend Td vs T2m
st.subheader("3. Xu hướng dài hạn của Nhiệt độ điểm sương (so sánh với T2m)")

slope_d, int_d, _, _, _ = stats.linregress(x_yr, df_yearly["d2m"].values)
slope_t, int_t, _, _, _ = stats.linregress(x_yr, df_yearly["t2m"].values)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x_yr, y=df_yearly["d2m"].values,
    mode="lines", line=dict(color="blue", width=1.5), opacity=0.6, name="Td"))
fig3.add_trace(go.Scatter(x=x_yr, y=int_d + slope_d * x_yr,
    mode="lines", line=dict(color="blue", dash="dash", width=2.5),
    name=f"Td Trend: +{slope_d*10:.3f}°C/thập kỷ"))
fig3.add_trace(go.Scatter(x=x_yr, y=df_yearly["t2m"].values,
    mode="lines", line=dict(color="red", width=1.5), opacity=0.3, name="T2m"))
fig3.add_trace(go.Scatter(x=x_yr, y=int_t + slope_t * x_yr,
    mode="lines", line=dict(color="red", dash="dot", width=2.5),
    name=f"T2m Trend: +{slope_t*10:.3f}°C/thập kỷ"))
fig3.update_layout(title="Xu hướng dài hạn của Nhiệt độ điểm sương (so sánh với T2m)",
    xaxis_title="Năm", yaxis_title="Nhiệt độ trung bình năm (°C)", height=500)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Anomaly heatmap
st.subheader("4. Ma trận dị thường Nhiệt độ điểm sương")

pivot = df_anomaly.pivot(index="year", columns="month", values="d2m")
fig4 = go.Figure(data=go.Heatmap(
    z=pivot.values, x=MONTH_LABELS, y=pivot.index,
    colorscale="BrBG", zmid=0,
    colorbar=dict(title="Dị thường Td (°C)")
))
fig4.update_layout(title="Ma trận dị thường Nhiệt độ điểm sương hàng tháng qua các năm",
    xaxis_title="Tháng", yaxis_title="Năm", height=600)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Scatter T2m vs Td
st.subheader("5. Tương quan Nhiệt độ điểm sương và Nhiệt độ không khí")

sample = df_daily.sample(min(5000, len(df_daily)), random_state=42)
fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=sample["t2m"], y=sample["d2m"], mode="markers",
    marker=dict(color=sample.index.month, colorscale="Twilight", opacity=0.5, size=4,
                colorbar=dict(title="Tháng")),
    name="Daily data"
))
lim = [df_daily[["t2m", "d2m"]].min().min(), df_daily[["t2m", "d2m"]].max().max()]
fig5.add_trace(go.Scatter(x=lim, y=lim, mode="lines",
    line=dict(color="black", dash="dash"), name="Đường bão hòa (RH=100%)"))
fig5.update_layout(title="Tương quan Nhiệt độ điểm sương và Nhiệt độ không khí",
    xaxis_title="Nhiệt độ không khí (T2m) (°C)", yaxis_title="Nhiệt độ điểm sương (Td) (°C)",
    height=600)
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — High RH days
st.subheader("6. Tần suất số ngày độ ẩm cao cực đoan")

high_rh = (df_daily["rh"] > 85).resample("YS").sum()
rolling5 = high_rh.rolling(5).mean()

fig6 = go.Figure()
fig6.add_trace(go.Bar(x=high_rh.index.year, y=high_rh.values,
    marker_color="teal", opacity=0.6, name="Số ngày có RH > 85%"))
fig6.add_trace(go.Scatter(x=high_rh.index.year, y=rolling5.values,
    mode="lines", line=dict(color="darkorange", width=2.5), name="Trung bình trượt 5 năm"))
fig6.update_layout(title="Tần suất số ngày độ ẩm cao cực đoan hàng năm",
    xaxis_title="Năm", yaxis_title="Số ngày", height=500, hovermode="x unified")
st.plotly_chart(fig6, width="stretch")
st.markdown("---")


# Section 7 — Diurnal cycle by season
st.subheader("7. Chu kỳ ngày đêm của Nhiệt độ điểm sương theo mùa")

season_map = {12:"DJF",1:"DJF",2:"DJF",3:"MAM",4:"MAM",5:"MAM",
              6:"JJA",7:"JJA",8:"JJA",9:"SON",10:"SON",11:"SON"}
season_vi  = {"DJF":"Đông","MAM":"Xuân","JJA":"Hè","SON":"Thu"}
colors_s   = {"DJF":"blue","MAM":"green","JJA":"red","SON":"orange"}

df_hourly["season"] = df_hourly.index.month.map(season_map)
df_hourly["hour"]   = df_hourly.index.hour
hourly_season = df_hourly.groupby(["season", "hour"])["d2m"].mean().unstack(level=0)

fig7 = go.Figure()
for s in ["DJF", "MAM", "JJA", "SON"]:
    if s in hourly_season.columns:
        fig7.add_trace(go.Scatter(x=hourly_season.index, y=hourly_season[s],
            mode="lines", line=dict(color=colors_s[s], width=2.5), name=season_vi[s]))
fig7.update_layout(title="Chu kỳ ngày đêm của Nhiệt độ điểm sương theo mùa",
    xaxis_title="Giờ trong ngày (UTC)", yaxis_title="Nhiệt độ điểm sương trung bình (°C)",
    xaxis=dict(tickvals=list(range(0, 24))), height=500)
st.plotly_chart(fig7, width="stretch")
