"""
Page — Vegetation Structure Analysis
Parquet files: daily.parquet, climatology.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Vegetation Structure", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "vegetation_structure"

st.title("Vegetation Structure Analysis")

df_daily = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
clim     = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

df_daily.index = pd.to_datetime(df_daily.index)
clim = clim.reset_index() if clim.index.name == "month" else clim

if "month" not in clim.columns:
    clim.insert(0, "month", range(1, 13))

if "ratio" not in clim.columns:
    clim["ratio"] = clim["leaf_area_index_high_vegetation"] / clim["leaf_area_index_low_vegetation"]

st.success("Data loaded", icon="✅")
st.markdown("---")


# Section 1 — Pie chart land cover
st.subheader("1. Thành phần lớp phủ bề mặt trung bình")

avg_hvc = clim["high_vegetation_cover"].mean()
avg_lvc = clim["low_vegetation_cover"].mean()
bare    = max(0, 1 - (avg_hvc + avg_lvc))

fig1 = go.Figure(data=go.Pie(
    labels=["Thực vật cao", "Thực vật thấp", "Khác/Đất trống"],
    values=[avg_hvc, avg_lvc, bare],
    marker=dict(colors=["#1b7837", "#7fbf7b", "#e0e0e0"]),
    pull=[0.05, 0.05, 0.05],
    textinfo="label+percent"
))
fig1.update_layout(title="Thành phần lớp phủ bề mặt trung bình", height=500)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Raw LAI seasonal
st.subheader("2. Chu kỳ mùa của chỉ số diện tích lá (Raw LAI)")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["leaf_area_index_high_vegetation"],
    mode="lines+markers", marker=dict(symbol="circle"), line=dict(color="#1b7837", width=2),
    name="LAI Tầng cao"))
fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["leaf_area_index_low_vegetation"],
    mode="lines+markers", marker=dict(symbol="square"), line=dict(color="#7fbf7b", width=2),
    name="LAI Tầng thấp"))
fig2.update_layout(title="So sánh chu kỳ mùa của chỉ số diện tích lá (Raw LAI)",
    xaxis_title="Tháng", yaxis_title="m²/m²", height=500)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Effective LAI stackplot
st.subheader("3. Tổng diện tích lá hiệu dụng theo mùa")

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["eff_lai_high"],
    mode="lines", line=dict(color="#1b7837", width=0),
    fill="tozeroy", fillcolor="#1b7837", opacity=0.8, name="Effective LAI (Cao)"))
fig3.add_trace(go.Scatter(x=MONTH_LABELS,
    y=clim["eff_lai_high"] + clim["eff_lai_low"],
    mode="lines", line=dict(color="#a6dba0", width=0),
    fill="tonexty", fillcolor="#a6dba0", opacity=0.8, name="Effective LAI (Thấp)"))
fig3.update_layout(title="Tổng diện tích lá hiệu dụng theo mùa (Đã tính trọng số che phủ)",
    xaxis_title="Tháng", yaxis_title="m²/m²", height=500)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — LAI heatmap
st.subheader("4. Heatmap cường độ xanh theo tháng")

heatmap_data = clim[["leaf_area_index_high_vegetation", "leaf_area_index_low_vegetation"]].values.T

fig4 = go.Figure(data=go.Heatmap(
    z=heatmap_data, x=MONTH_LABELS, y=["LAI Cao", "LAI Thấp"],
    colorscale="YlGn",
    text=np.round(heatmap_data, 2).astype(str),
    texttemplate="%{text}",
    colorbar=dict(title="m²/m²")
))
fig4.update_layout(title="Heatmap cường độ xanh của các tầng thực vật theo tháng",
    xaxis_title="Tháng", height=350)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — LAI ratio bar
st.subheader("5. Tỷ lệ đóng góp LAI (Tầng cao / Tầng thấp)")

fig5 = go.Figure()
fig5.add_trace(go.Bar(x=MONTH_LABELS, y=clim["ratio"],
    marker_color="#5aae61", name="Ratio"))
fig5.add_hline(y=1, line=dict(color="red", dash="dash"), opacity=0.5)
fig5.update_layout(title="Tỷ lệ đóng góp LAI (Tầng cao / Tầng thấp)",
    xaxis_title="Tháng", yaxis_title="Tỷ số", height=450)
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — Phase plot
st.subheader("6. Không gian trạng thái thực vật (Phase Plot)")

fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=clim["leaf_area_index_low_vegetation"],
    y=clim["leaf_area_index_high_vegetation"],
    mode="markers+text",
    marker=dict(color=clim["month"] if "month" in clim.columns else list(range(1,13)),
                colorscale="Viridis", size=12,
                colorbar=dict(title="Tháng")),
    text=[str(m) for m in (clim["month"].tolist() if "month" in clim.columns else range(1,13))],
    textposition="top center"
))
fig6.update_layout(title="Không gian trạng thái thực vật (High vs Low LAI Phase Plot)",
    xaxis_title="LAI Tầng thấp", yaxis_title="LAI Tầng cao", height=600)
st.plotly_chart(fig6, width="stretch")
st.markdown("---")


# Section 7 — Correlation low vs high
st.subheader("7. Tương quan cấu trúc giữa các tầng thực vật")

x_lai = clim["leaf_area_index_low_vegetation"].values
y_lai = clim["leaf_area_index_high_vegetation"].values
slope_lai, int_lai, r_lai, _, _ = stats.linregress(x_lai, y_lai)
x_line = np.linspace(x_lai.min(), x_lai.max(), 100)

fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=x_lai, y=y_lai, mode="markers",
    marker=dict(color="darkgreen", size=10), name="Tháng"))
fig7.add_trace(go.Scatter(x=x_line, y=int_lai + slope_lai * x_line,
    mode="lines", line=dict(color="darkgreen"), name=f"Hồi quy (R²={r_lai**2:.2f})"))
fig7.update_layout(title="Tương quan cấu trúc giữa tầng thấp và tầng cao",
    xaxis_title="LAI Tầng thấp", yaxis_title="LAI Tầng cao", height=500)
st.plotly_chart(fig7, width="stretch")
