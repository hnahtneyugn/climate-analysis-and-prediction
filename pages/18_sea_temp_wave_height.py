"""
Page — Sea Surface Temperature & Wave Height Analysis
Parquet files expected:
  daily.parquet       — columns: time, sst, swh, anom_sst, anom_swh, year, month
  monthly.parquet     — columns: time, sst, swh
  climatology.parquet — columns: month, sst, swh
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Sea Temp & Wave Height", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "sea_temp_wave_height"

st.title("Sea Surface Temperature & Wave Height Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

for df in (df_daily, df_monthly):
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])

if "year" not in df_daily.columns:
    df_daily["year"]  = df_daily["time"].dt.year
if "month" not in df_daily.columns:
    df_daily["month"] = df_daily["time"].dt.month

st.markdown("---")


# Section 1 — SST và SWH theo tháng

st.subheader("1. Nhiệt độ nước biển và độ cao sóng biển qua các tháng")
st.markdown(
    "Chu kỳ khí hậu hàng tháng của nhiệt độ mặt nước biển (SST - màu đỏ) và độ cao sóng (SWH - màu xanh) kèm theo dải độ lệch chuẩn ±1. "
    "Gió mùa Đông Bắc thường làm giảm nhiệt độ biển trong khi làm tăng độ cao sóng; ngược lại gió mùa Tây Nam thường đi kèm với nhiệt độ biển cao hơn."
)

monthly_std = df_daily.groupby("month")[["sst", "swh"]].std().reset_index()

fig1 = make_subplots(specs=[[{"secondary_y": True}]])

fig1.add_trace(go.Scatter(
    x=df_clim["month"], y=df_clim["sst"],
    name="SST mean", mode="lines",
    line=dict(color="red", width=2)
), secondary_y=False)

fig1.add_trace(go.Scatter(
    x=pd.concat([df_clim["month"], df_clim["month"][::-1]]),
    y=pd.concat([
        df_clim["sst"] + monthly_std["sst"],
        (df_clim["sst"] - monthly_std["sst"])[::-1]
    ]),
    fill="toself", fillcolor="rgba(255,0,0,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="SST ±1 std"
), secondary_y=False)

fig1.add_trace(go.Scatter(
    x=df_clim["month"], y=df_clim["swh"],
    name="SWH mean", mode="lines",
    line=dict(color="blue", width=2)
), secondary_y=True)

fig1.add_trace(go.Scatter(
    x=pd.concat([df_clim["month"], df_clim["month"][::-1]]),
    y=pd.concat([
        df_clim["swh"] + monthly_std["swh"],
        (df_clim["swh"] - monthly_std["swh"])[::-1]
    ]),
    fill="toself", fillcolor="rgba(0,0,255,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="SWH ±1 std"
), secondary_y=True)

fig1.update_xaxes(title_text="Tháng", tickvals=list(range(1, 13)), ticktext=MONTH_LABELS)
fig1.update_yaxes(title_text="Nhiệt độ bề mặt nước biển (°C)", secondary_y=False,
                  color="red", tickfont=dict(color="red"))
fig1.update_yaxes(title_text="Độ cao sóng biển (m)", secondary_y=True,
                  color="blue", tickfont=dict(color="blue"))
fig1.update_layout(
    title="Chỉ số khí hậu hàng tháng trong suốt 45 năm",
    hovermode="x unified", height=500
)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Xu hướng dài hạn SST

st.subheader("2. Xu hướng dài hạn của nhiệt độ mặt nước biển")
st.markdown(
    "Biến động nhiệt độ mặt nước biển trung bình năm cùng với đường xu hướng tuyến tính. "
    "Tín hiệu của biến đổi khí hậu được thể hiện rõ ràng qua tốc độ nóng lên của đại dương qua từng thập kỷ."
)

df_annual_sst = df_daily.groupby("year")["sst"].mean().reset_index()
slope, intercept, r_val, p_val, _ = stats.linregress(df_annual_sst["year"], df_annual_sst["sst"])
trend_per_decade = slope * 10

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_annual_sst["year"], y=df_annual_sst["sst"],
    mode="markers", marker=dict(color="black", size=7, opacity=0.7),
    name="Trung bình SST năm"
))
fig2.add_trace(go.Scatter(
    x=df_annual_sst["year"],
    y=intercept + slope * df_annual_sst["year"],
    mode="lines", line=dict(color="red", width=2),
    name=f"Xu hướng: {trend_per_decade:.3f} °C/decade"
))
fig2.update_layout(
    title=f"Xu hướng dài hạn của nhiệt độ mặt nước biển: {trend_per_decade:.3f} °C mỗi thập kỉ",
    xaxis_title="Năm",
    yaxis_title="Trung bình SST năm (°C)",
    hovermode="x unified", height=450
)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Heatmap dị thường SST

st.subheader("3. Heatmap dị thường nhiệt độ mặt nước biển")
st.markdown(
    "Ma trận dị thường nhiệt độ biển: Sắc xanh chiếm ưu thế trước năm 2000 và sắc đỏ tăng cường mạnh mẽ sau năm 2010 "
    "— một tín hiệu nóng lên rõ rệt đồng nhất với biến đổi khí hậu toàn cầu. Các năm có hiện tượng El Niño/La Niña mạnh được đánh dấu trên biểu đồ."
)

heatmap_data = df_daily.pivot_table(index="year", columns="month", values="anom_sst", aggfunc="mean")

fig3 = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=MONTH_LABELS,
    y=heatmap_data.index,
    colorscale="RdBu_r",
    zmid=0,
    colorbar=dict(title="Dị thường SST (°C)")
))

enso_annotations = []
enso_events = {
    "El Niño": [1982, 1997, 2015],
    "La Niña": [1988, 1999, 2010],
}
for label, years_list in enso_events.items():
    for yr in years_list:
        if yr in heatmap_data.index:
            enso_annotations.append(dict(
                x=11.7, y=yr,
                text=f"← {label}",
                showarrow=False,
                font=dict(size=9, color="black"),
                xref="x", yref="y"
            ))

fig3.update_layout(
    title="Heatmap dị thường nhiệt độ mặt nước biển",
    xaxis_title="Tháng", yaxis_title="Năm",
    annotations=enso_annotations,
    height=700
)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Tần suất và cường độ Marine Heat Waves

st.subheader("4. Xu hướng về tần suất và cường độ của các đợt nước biển nóng bất thường")
st.markdown(
    "Thống kê các đợt sóng nhiệt đại dương (Marine Heat Waves - MHW), được xác định khi nhiệt độ biển vượt ngưỡng bách phân vị thứ 90 theo tháng. "
    "Biểu đồ thể hiện sự gia tăng về cả tần suất (số ngày) và cường độ trung bình của các đợt nóng bất thường này."
)

mhw_thresh = df_daily.groupby("month")["sst"].transform(lambda x: x.quantile(0.90))
df_daily["is_mhw"] = df_daily["sst"] > mhw_thresh

mhw_stats = df_daily.groupby("year").agg(
    mhw_days=("is_mhw", "sum"),
    mhw_intensity=("anom_sst", lambda x: x[df_daily.loc[x.index, "is_mhw"]].mean())
).reset_index()

rolling_mhw = mhw_stats["mhw_days"].rolling(5).mean()

fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=[
                         "Số ngày nước biển nóng bất thường trong năm",
                         "Cường độ trung bình (°C dị thường)"
                     ])

fig4.add_trace(go.Bar(
    x=mhw_stats["year"], y=mhw_stats["mhw_days"],
    name="Số ngày MHW", marker_color="orange", opacity=0.6
), row=1, col=1)
fig4.add_trace(go.Scatter(
    x=mhw_stats["year"], y=rolling_mhw,
    name="Trung bình trượt 5 năm",
    mode="lines", line=dict(color="red", width=2)
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=mhw_stats["year"], y=mhw_stats["mhw_intensity"],
    mode="lines+markers",
    marker=dict(size=5, color="darkred"),
    line=dict(color="darkred", width=1.5),
    name="Cường độ MHW", showlegend=True
), row=2, col=1)

fig4.update_yaxes(title_text="Số ngày", row=1, col=1)
fig4.update_yaxes(title_text="°C dị thường", row=2, col=1)
fig4.update_xaxes(title_text="Năm", row=2, col=1)
fig4.update_layout(
    title="Xu hướng về tần suất và cường độ của các đợt nước biển nóng bất thường",
    hovermode="x unified", height=600
)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Tần suất sóng cực đoan

st.subheader("5. Tần suất sóng cực đoan và mùa bão")
st.markdown(
    "Thống kê số lượng ngày trong năm có độ cao sóng cực đoan (vượt ngưỡng bách phân vị thứ 95). "
    "Đường trung bình trượt 5 năm giúp nhận diện xu hướng biến đổi của năng lượng sóng biển, thường liên quan chặt chẽ đến hoạt động của bão và gió mùa."
)

swh_p95 = df_daily["swh"].quantile(0.95)
df_daily["is_extreme_wave"] = df_daily["swh"] > swh_p95

wave_stats = df_daily.groupby("year").agg(
    extreme_days=("is_extreme_wave", "sum")
).reset_index()

rolling_wave = wave_stats["extreme_days"].rolling(5).mean()

fig5 = go.Figure()
fig5.add_trace(go.Bar(
    x=wave_stats["year"], y=wave_stats["extreme_days"],
    name="Số ngày có sóng cực đoan",
    marker_color="skyblue", opacity=0.8
))
fig5.add_trace(go.Scatter(
    x=wave_stats["year"], y=rolling_wave,
    name="Trung bình trượt 5 năm",
    mode="lines", line=dict(color="blue", width=2)
))
fig5.update_layout(
    title="Tần suất sóng cực đoan",
    xaxis_title="Năm",
    yaxis_title="Số ngày trong năm",
    hovermode="x unified", height=450
)
st.plotly_chart(fig5, width="stretch")