"""
Page — Column Water Vapour & Clouds Analysis
Parquet files: daily.parquet, monthly.parquet, yearly.parquet, climatology.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Water Vapour & Clouds", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "col_water_vapour_n_clouds"

st.title("Column Water Vapour & Clouds Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_yearly  = pd.read_parquet(f"{DATA_DIR}/yearly.parquet")
clim       = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

df_daily.index   = pd.to_datetime(df_daily.index)
df_monthly.index = pd.to_datetime(df_monthly.index)
df_yearly.index  = pd.to_datetime(df_yearly.index)

st.markdown("---")


# Section 1 — Seasonal climatology
st.subheader("1. Khí hậu học mùa: Hơi nước và Thành phần mây")
st.markdown(
    "Phân tích sự thay đổi của Tổng lượng hơi nước cột (TCWV) cùng các thành phần nước lỏng (CLW) và băng (CIW) trong mây. "
    "**TCWV** phản ánh tổng lượng ẩm có trong một cột khí quyển, là nguồn cung cấp chính cho các quá trình gây mưa. "
    "Thường thì lượng ẩm này sẽ đạt đỉnh vào các tháng mùa hè và mùa mưa do nhiệt độ cao thúc đẩy quá trình bốc hơi."
)

fig1 = make_subplots(specs=[[{"secondary_y": True}]])
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["wvap"],
    mode="lines", line=dict(color="blue", width=3), name="TCWV (kg/m²)"),
    secondary_y=False)
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["clw"],
    mode="lines", line=dict(color="cyan", dash="dash"), name="CLW (kg/m²)"),
    secondary_y=True)
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["ciw"],
    mode="lines", line=dict(color="indigo", dash="dot"), name="CIW (kg/m²)"),
    secondary_y=True)
fig1.update_yaxes(title_text="Hơi nước cột (kg/m²)", secondary_y=False, color="blue")
fig1.update_yaxes(title_text="Nước/Băng mây (kg/m²)", secondary_y=True, color="indigo")
fig1.update_layout(title="Khí hậu học mùa: Hơi nước và Thành phần mây",
    xaxis_title="Tháng", height=500)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Ice fraction seasonal
st.subheader("2. Chu kỳ mùa của tỷ lệ băng trong mây (Ice Fraction)")
st.markdown(
    "Chỉ số này thể hiện tỷ lệ giữa băng và tổng lượng nước trong mây (CIW / [CLW + CIW]). "
    "Tỷ lệ băng cao phản ánh sự hiện diện của các loại mây tầng cao (như mây ti) và các quá trình vi vật lý tạo mưa trong môi trường lạnh."
)

fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["phase"],
    mode="lines+markers", marker=dict(symbol="diamond"), line=dict(color="darkcyan", width=2)))
fig2.update_layout(title="Chu kỳ mùa của tỷ lệ băng trong mây",
    xaxis_title="Tháng", yaxis_title="Tỷ lệ băng (CIW / (CLW + CIW))", height=450)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — TCWV long-term trend
st.subheader("3. Xu hướng dài hạn của Tổng hơi nước cột (TCWV)")
st.markdown(
    "Theo dõi xu hướng biến đổi của tổng lượng hơi nước trong khí quyển từ năm 1980 đến nay. "
    "Theo lý thuyết vật lý khí quyển (Clausius-Clapeyron), khi bầu khí quyển nóng lên, khả năng giữ hơi nước của nó tăng lên. "
    "Xu hướng tăng của TCWV là một minh chứng rõ rệt cho hiện tượng nóng lên toàn cầu."
)

x = df_yearly.index.year.values
y = df_yearly["wvap"].values
slope, intercept, _, p, _ = stats.linregress(x, y)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color="blue", width=1.5),
    opacity=0.5, name="TCWV năm"))
fig3.add_trace(go.Scatter(x=x, y=intercept + slope * x,
    mode="lines", line=dict(color="red", dash="dash", width=2),
    name=f"Trend: {slope*10:.3f} kg/m²/thập kỷ (p={p:.4f})"))
fig3.update_layout(title="Xu hướng dài hạn của Tổng hơi nước cột",
    xaxis_title="Năm", yaxis_title="TCWV (kg/m²)", height=500)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — TCWV anomaly heatmap
st.subheader("4. Ma trận dị thường Hơi nước cột")
st.markdown(
    "Heatmap này nhận diện các năm và tháng có lượng hơi nước trong khí quyển cao bất thường (màu xanh dương) "
    "hoặc thấp bất thường (màu đỏ). Lượng hơi nước dồi dào thường là tiền đề cho các năm có lượng mưa cực đoan và bão mạnh."
)

pivot = df_monthly.pivot(index="year", columns="month", values="wvap_anom")
fig4 = go.Figure(data=go.Heatmap(
    z=pivot.values, x=MONTH_LABELS, y=pivot.index,
    colorscale="RdYlBu", zmid=0,
    colorbar=dict(title="Dị thường TCWV (kg/m²)")
))
fig4.update_layout(title="Ma trận dị thường Hơi nước cột",
    xaxis_title="Tháng", yaxis_title="Năm", height=600)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Cloud phase stackplot
st.subheader("5. Phân chia pha mây theo thời gian")
st.markdown(
    "Biểu đồ chồng thể hiện sự biến động của hàm lượng nước lỏng và băng trong mây qua các thập kỷ. "
    "Sự thay đổi cấu trúc này ảnh hưởng trực tiếp đến khả năng phản xạ bức xạ mặt trời và giữ nhiệt của mây, "
    "đóng vai trò quan trọng trong hệ thống phản hồi khí hậu."
)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly["clw"],
    mode="lines", line=dict(color="#4A90D9", width=0),
    fill="tozeroy", fillcolor="#4A90D9", name="CLW (Nước lỏng)"))
fig5.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly["clw"] + df_monthly["ciw"],
    mode="lines", line=dict(color="#B0C4DE", width=0),
    fill="tonexty", fillcolor="#B0C4DE", name="CIW (Băng)"))
fig5.update_layout(title="Phân chia pha mây: Nước lỏng và Băng theo thời gian",
    xaxis_title="Năm", yaxis_title="kg/m²", height=450)
st.plotly_chart(fig5, width="stretch")