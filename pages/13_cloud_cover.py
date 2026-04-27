"""
Page — Cloud Cover Analysis
Parquet files: daily.parquet, monthly.parquet, yearly.parquet, climatology.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Cloud Cover", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "cloud_cover"

st.title("Cloud Cover Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_yearly  = pd.read_parquet(f"{DATA_DIR}/yearly.parquet")
clim       = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

df_daily.index   = pd.to_datetime(df_daily.index)
df_monthly.index = pd.to_datetime(df_monthly.index)
df_yearly.index  = pd.to_datetime(df_yearly.index)

st.markdown("---")


# Section 1 — Cloud regime seasonal
st.subheader("1. Khí hậu học mây theo mùa (Cloud Regime)")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["tcc"],
    mode="lines", line=dict(color="#2C2C2A", width=2.5), name="Tổng lượng mây"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["lcc"],
    mode="lines", line=dict(color="#185FA5", width=1.8), name="Mây tầng thấp"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["hcc"],
    mode="lines", line=dict(color="#888780", dash="dash", width=1.8), name="Mây tầng cao"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["mcc"],
    mode="lines", line=dict(color="#7F77DD", dash="dot", width=1.2), name="Mây tầng giữa"))
fig1.update_layout(title="Khí hậu học mây theo mùa (Cloud Regime)",
    xaxis_title="Tháng", yaxis_title="Tỷ lệ che phủ (0-1)",
    yaxis=dict(range=[0, 1]), height=500)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — LHD dominance index
st.subheader("2. Chỉ số thống trị và xu hướng thay đổi cấu trúc")

x_yr = df_yearly.index.year.values
y_lhd = df_yearly["lhd"].values
slope_lhd, intercept_lhd, _, p_lhd, _ = stats.linregress(x_yr, y_lhd)

fig2 = make_subplots(rows=1, cols=2,
    subplot_titles=("Chu kỳ mùa chỉ số thống trị (LCC-HCC)",
                    f"Xu hướng thống trị: {slope_lhd*10:+.4f}/decade (p={p_lhd:.4f})"))

lhd_vals = clim["lhd"].values
fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=lhd_vals,
    mode="lines", line=dict(color="#2C2C2A", width=2), showlegend=False), row=1, col=1)
fig2.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=1)

# Fill regions
pos_y = np.where(lhd_vals > 0, lhd_vals, 0)
neg_y = np.where(lhd_vals < 0, lhd_vals, 0)
fig2.add_trace(go.Scatter(x=MONTH_LABELS + MONTH_LABELS[::-1],
    y=list(pos_y) + [0]*12, fill="toself",
    fillcolor="rgba(24,95,165,0.15)", line=dict(color="rgba(0,0,0,0)"),
    name="LCC Dominant"), row=1, col=1)
fig2.add_trace(go.Scatter(x=MONTH_LABELS + MONTH_LABELS[::-1],
    y=list(neg_y) + [0]*12, fill="toself",
    fillcolor="rgba(136,135,128,0.15)", line=dict(color="rgba(0,0,0,0)"),
    name="HCC Dominant"), row=1, col=1)

fig2.add_trace(go.Scatter(x=x_yr, y=y_lhd, mode="markers",
    marker=dict(color="#2C2C2A", size=8, opacity=0.6), showlegend=False), row=1, col=2)
fig2.add_trace(go.Scatter(x=x_yr, y=intercept_lhd + slope_lhd * x_yr,
    mode="lines", line=dict(color="#D85A30", width=2), showlegend=False), row=1, col=2)
fig2.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=2)

fig2.update_layout(height=500)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — TCC anomaly heatmap
st.subheader("3. Ma trận dị thường tổng lượng mây")

pivot = df_monthly.pivot(index="year", columns="month", values="tcc_anom")
fig3 = go.Figure(data=go.Heatmap(
    z=pivot.values, x=MONTH_LABELS, y=pivot.index,
    colorscale="RdYlBu", zmid=0,
    colorbar=dict(title="Dị thường TCC")
))
fig3.update_layout(title="Ma trận dị thường tổng lượng mây hàng tháng",
    xaxis_title="Tháng", yaxis_title="Năm", height=600)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Cloud structure stackplot
st.subheader("4. Biến động thành phần cấu trúc mây (1980-2024)")

fig4 = go.Figure()
fig4.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly["lcc"],
    mode="lines", line=dict(color="#185FA5", width=0),
    fill="tozeroy", fillcolor="#185FA5", opacity=0.8, name="Mây tầng thấp"))
fig4.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly["lcc"] + df_monthly["mcc"],
    mode="lines", line=dict(color="#7F77DD", width=0),
    fill="tonexty", fillcolor="#7F77DD", opacity=0.8, name="Mây tầng giữa"))
fig4.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly["lcc"] + df_monthly["mcc"] + df_monthly["hcc"],
    mode="lines", line=dict(color="#B4B2A9", width=0),
    fill="tonexty", fillcolor="#B4B2A9", opacity=0.8, name="Mây tầng cao"))
fig4.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly["tcc"],
    mode="lines", line=dict(color="#2C2C2A", width=1), name="Tổng lượng mây"))
fig4.update_layout(title="Biến động thành phần cấu trúc mây (1980-2024)",
    yaxis=dict(range=[0, 1.1]), height=450)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Clear / overcast day frequency
st.subheader("5. Tần suất các ngày quang mây và u ám")

overcast = (df_daily["tcc"] > 0.9).resample("1YE").sum()
clear    = (df_daily["tcc"] < 0.1).resample("1YE").sum()

fig5 = make_subplots(rows=2, cols=1, shared_xaxes=True,
    subplot_titles=["Ngày quang mây (TCC < 0.1)", "Ngày u ám (TCC > 0.9)"])

for row, (data, color) in enumerate([(clear, "#BA7517"), (overcast, "#185FA5")], 1):
    sl, _, _, p_cc, _ = stats.linregress(data.index.year, data.values)
    fig5.add_trace(go.Bar(x=data.index.year, y=data.values,
        marker_color=color, opacity=0.7, showlegend=False), row=row, col=1)
    fig5.update_yaxes(title_text=f"Số ngày/năm | Trend: {sl*10:+.1f} d/decade (p={p_cc:.3f})",
        row=row, col=1)

fig5.update_layout(height=600, hovermode="x unified")
fig5.update_xaxes(title_text="Năm", row=2, col=1)
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — Violin plot by decade
st.subheader("6. So sánh phân phối mây qua các thập kỷ")

decades = {"1980s": ("1980","1989"), "1990s": ("1990","1999"),
           "2000s": ("2000","2009"), "2010s": ("2010","2019"), "2020s": ("2020","2024")}

fig6 = make_subplots(rows=1, cols=3,
    subplot_titles=["Mây tầng thấp", "Mây tầng cao", "L-H Dominance"])

for col_idx, (var, color) in enumerate([("lcc","#185FA5"),("hcc","#888780"),("lhd","#2C2C2A")], 1):
    for d_label, (start, end) in decades.items():
        vals = df_daily.loc[start:end, var].dropna().values
        fig6.add_trace(go.Violin(y=vals, name=d_label, line_color=color,
            showlegend=(col_idx == 1), legendgroup=d_label,
            box_visible=True, meanline_visible=True), row=1, col=col_idx)
    if var == "lhd":
        fig6.add_hline(y=0, line=dict(color="gray", dash="dash"), row=1, col=col_idx)

fig6.update_layout(height=550, violingap=0.3, violinmode="group",
    title="So sánh phân phối mây qua các thập kỷ")
st.plotly_chart(fig6, width="stretch")
