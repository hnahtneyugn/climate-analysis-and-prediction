"""
Page — Runoff & Drainage Analysis
Parquet files: daily.parquet, monthly.parquet, climatology.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Runoff & Drainage", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "runoff_drainage"

st.title("Runoff & Drainage Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

for df in [df_daily, df_monthly]:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        df.reset_index(inplace=True)
        df["time"] = pd.to_datetime(df["time"])

if "month" not in df_daily.columns:
    df_daily["month"] = df_daily["time"].dt.month
if "year" not in df_daily.columns:
    df_daily["year"] = df_daily["time"].dt.year

df_clim = df_clim.reset_index() if "month" not in df_clim.columns else df_clim

st.success("Data loaded", icon="✅")
st.markdown("---")


# Section 1 — Monthly climatology
st.subheader("1. Phân tích tốc độ dòng chảy bề mặt theo tháng")

months = df_clim["month"].values if "month" in df_clim.columns else range(1, 13)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=df_clim["tr"],
    mode="lines", line=dict(color="#2C2C2A", width=2), name="Tốc độ dòng chảy bề mặt (TR)"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=df_clim["sr"],
    mode="lines", line=dict(color="#D85A30", width=1.6), name="Nước chảy trên bề mặt (SR)"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=df_clim["ssr"],
    mode="lines", line=dict(color="#185FA5", width=1.6), name="Nước thấm xuống đất (SSR)",
    fill="tonexty", fillcolor="rgba(136,135,128,0.15)"))
fig1.update_layout(title="Phân tích tốc độ dòng chảy bề mặt theo tháng",
    xaxis_title="Tháng", yaxis_title="Tốc độ dòng chảy (mm/ngày)", height=500)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Monthly stackplot
st.subheader("2. Xu hướng phân chia dòng chảy")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=df_monthly["time"], y=df_monthly["sr"],
    mode="lines", line=dict(color="#D85A30", width=0),
    fill="tozeroy", fillcolor="#D85A30", name="Nước chảy trên bề mặt (SR)"))
fig2.add_trace(go.Scatter(x=df_monthly["time"], y=df_monthly["sr"] + df_monthly["ssr"],
    mode="lines", line=dict(color="#185FA5", width=0),
    fill="tonexty", fillcolor="#185FA5", name="Nước thấm xuống đất (SSR)"))
fig2.update_layout(title="Xu hướng phân chia dòng chảy (1980-2024)",
    xaxis_title="Năm", yaxis_title="Tốc độ dòng chảy (mm/ngày)", height=450)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Surface fraction trend
st.subheader("3. Xu hướng lâu dài của tỉ lệ nước bề mặt")

annual_sf = df_daily.groupby("year")["sf"].mean()
years_sf  = annual_sf.index.values
slope_sf, intercept_sf, r_sf, p_sf, _ = stats.linregress(years_sf, annual_sf.values)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=years_sf, y=annual_sf.values,
    mode="lines+markers", marker=dict(symbol="circle", size=5),
    line=dict(color="#D85A30", width=1), opacity=0.7, name="Tỉ lệ nước bề mặt"))
fig3.add_trace(go.Scatter(x=years_sf, y=intercept_sf + slope_sf * years_sf,
    mode="lines", line=dict(color="black", dash="dash", width=2),
    name=f"Đường tuyến tính (dốc: {slope_sf:.5f}/năm)"))
fig3.add_annotation(x=0.02, y=0.95, xref="paper", yref="paper",
    text=f"R² = {r_sf**2:.3f}<br>p-value = {p_sf:.4f}", showarrow=False,
    bgcolor="white", bordercolor="gray", font=dict(size=11))
trend_msg = ("Xu hướng tăng dần: Nguy cơ lũ lụt gia tăng" if slope_sf > 0
             else "Xu hướng giảm dần: Tăng khả năng thấm nước của đất")
fig3.add_annotation(x=0.98, y=0.05, xref="paper", yref="paper",
    text=trend_msg, showarrow=False,
    font=dict(color="#D85A30" if slope_sf > 0 else "#185FA5", size=11))
fig3.update_layout(title="Xu hướng lâu dài của tỉ lệ nước bề mặt (SF)",
    xaxis_title="Năm", yaxis_title="Tỉ lệ nước bề mặt", height=500)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Anomaly heatmap
st.subheader("4. Heatmap dị thường tốc độ dòng chảy bề mặt")

pivot_anom = df_daily.pivot_table(index="year", columns="month", values="anom_tr", aggfunc="mean")
fig4 = go.Figure(data=go.Heatmap(
    z=pivot_anom.values, x=MONTH_LABELS, y=pivot_anom.index,
    colorscale="BrBG", zmid=0,
    colorbar=dict(title="Dị thường (mm/ngày)")
))
fig4.update_layout(title="Heatmap dị thường tốc độ dòng chảy bề mặt (1980-2024)",
    xaxis_title="Tháng", yaxis_title="Năm", height=600)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Extreme frequency
st.subheader("5. Dấu hiệu tần suất dòng chảy cực đoan")

q95_sr  = df_daily["sr"].quantile(0.95)
q95_ssr = df_daily["ssr"].quantile(0.95)
df_daily["is_extreme_sr"]  = df_daily["sr"]  > q95_sr
df_daily["is_extreme_ssr"] = df_daily["ssr"] > q95_ssr

annual_ext = df_daily.groupby("year").agg(
    SR_count=("is_extreme_sr", "sum"), SSR_count=("is_extreme_ssr", "sum")
)
annual_ext["SR_rolling"]  = annual_ext["SR_count"].rolling(5, center=True).mean()
annual_ext["SSR_rolling"] = annual_ext["SSR_count"].rolling(5, center=True).mean()

fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True,
    subplot_titles=("Tần suất dòng chảy bề mặt cực đoan (Dấu hiệu lũ quét)",
                    "Tần suất dòng chảy ngầm cực đoan (Dấu hiệu bổ sung nhiều nước ngầm)"))

for row, col_cnt, col_roll, color, thresh_val in [
    (1, "SR_count", "SR_rolling", "#D85A30", q95_sr),
    (2, "SSR_count", "SSR_rolling", "#185FA5", q95_ssr)
]:
    fig5.add_trace(go.Bar(x=annual_ext.index, y=annual_ext[col_cnt],
        marker_color=color, opacity=0.4, showlegend=False), row=row, col=1)
    fig5.add_trace(go.Scatter(x=annual_ext.index, y=annual_ext[col_roll],
        mode="lines", line=dict(color=color, width=2.5), showlegend=False), row=row, col=1)
    fig5.add_hline(y=annual_ext[col_cnt].mean(), line=dict(color="black", dash="dot"), opacity=0.5,
        row=row, col=1)
    fig5.add_annotation(row=row, col=1, x=0.98, y=0.95, xref="paper" if row==1 else "x2",
        yref="paper" if row==1 else "y2",
        text=f"P95: {thresh_val:.2f} mm/ngày", showarrow=False,
        bgcolor="white", bordercolor="gray", font=dict(size=9))

fig5.update_layout(height=700, hovermode="x unified")
fig5.update_yaxes(title_text="Số lượng ngày mỗi năm", row=1, col=1)
fig5.update_yaxes(title_text="Số lượng ngày mỗi năm", row=2, col=1)
fig5.update_xaxes(title_text="Năm", row=2, col=1)
fig5.update_xaxes(
    type="linear",
    range=[annual_ext.index.min()-1, annual_ext.index.max()+1]
)
st.plotly_chart(fig5, width="stretch")
