"""
Page — Solar Radiation Analysis
Parquet files: daily.parquet, monthly.parquet, climatology.parquet, anomaly.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Solar Radiation", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "solar_radiation"

st.title("Solar Radiation Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")
df_anom    = pd.read_parquet(f"{DATA_DIR}/anomaly.parquet")

for df in [df_daily, df_monthly, df_anom]:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        df.reset_index(inplace=True)
        df["time"] = pd.to_datetime(df["time"])

rad_col = [c for c in df_daily.columns if c != "time" and c != "number"][0]

if "year" not in df_daily.columns:
    df_daily["year"]  = df_daily["time"].dt.year
if "month" not in df_anom.columns:
    df_anom["month"] = df_anom["time"].dt.month
if "year" not in df_anom.columns:
    df_anom["year"]  = df_anom["time"].dt.year

st.markdown("---")


# Section 1 — Annual trend + piecewise
st.subheader("1. Xu hướng bức xạ trung bình năm (Tín hiệu tối đi / sáng lên)")
st.markdown(
    "Phân tích sự thay đổi dài hạn của lượng bức xạ mặt trời chiếu xuống bề mặt đất. "
    "Sự so sánh giữa mô hình tuyến tính và đa đoạn (piecewise) giúp xác định các giai đoạn 'tối đi' hoặc 'sáng lên' toàn cầu "
    "do tác động của sol khí (aerosol) và biến đổi độ che phủ mây qua các thập kỷ tại khu vực."
)

annual_rad = df_daily.resample("YE", on="time").mean().reset_index()
annual_rad["year"] = annual_rad["time"].dt.year
years = annual_rad["year"].values
vals  = annual_rad[rad_col].values
n = len(vals)

slope, intercept, _, p, se = stats.linregress(years, vals)
fit_linear = intercept + slope * years

x_mean = np.mean(years)
x_sum_sq = np.sum((years - x_mean)**2)
ci95 = 1.96 * se * np.sqrt(1/n + (years - x_mean)**2 / x_sum_sq)

mask_early = years <= 1990
mask_late  = years > 1990
s1, i1, _, p1, _ = stats.linregress(years[mask_early], vals[mask_early])
s2, i2, _, p2, _ = stats.linregress(years[mask_late],  vals[mask_late])

rss_lin = np.sum((vals - fit_linear)**2)
aic_lin = n * np.log(rss_lin/n) + 2*2
fit_pw = np.where(mask_early, i1 + s1*years, i2 + s2*years)
rss_pw = np.sum((vals - fit_pw)**2)
aic_pw = n * np.log(rss_pw/n) + 2*4
better = "Piecewise" if aic_pw < aic_lin else "Linear"

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=years, y=vals, mode="markers",
    marker=dict(color="#BA7517", size=8, opacity=0.6), name="Trung bình năm"))
fig1.add_trace(go.Scatter(x=years, y=fit_linear,
    mode="lines", line=dict(color="#412402", dash="dash", width=2),
    name="Xu hướng tuyến tính"))
fig1.add_trace(go.Scatter(
    x=np.concatenate([years, years[::-1]]),
    y=np.concatenate([fit_linear + ci95, (fit_linear - ci95)[::-1]]),
    fill="toself", fillcolor="rgba(186,117,23,0.1)", line=dict(color="rgba(0,0,0,0)"),
    name="CI 95%"))
fig1.add_trace(go.Scatter(x=years[mask_early], y=fit_pw[mask_early],
    mode="lines", line=dict(color="red", width=2.5),
    name=f"1980-1990: {s1*10:+.2f} W/m²/thập kỉ"))
fig1.add_trace(go.Scatter(x=years[mask_late], y=fit_pw[mask_late],
    mode="lines", line=dict(color="green", width=2.5),
    name=f"1991-2024: {s2*10:+.2f} W/m²/thập kỉ"))
fig1.add_vline(x=1990, line=dict(color="grey", dash="dot"), opacity=0.5)
fig1.update_layout(
    title=(f"Phân tích xu hướng bức xạ Mặt Trời trên bề mặt<br>"
           f"Tổng thể: {slope*10:+.3f} W/m²/thập kỉ (p={p:.3f}) | "
           f"AIC: Linear={aic_lin:.1f}, Piecewise={aic_pw:.1f} ({better} tốt hơn)"),
    xaxis_title="Năm", yaxis_title="Bức xạ Mặt Trời xuống bề mặt (W/m²)", height=550)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Anomaly heatmap
st.subheader("2. Heatmap dị thường trong bức xạ Mặt Trời")
st.markdown(
    "Nhận diện các giai đoạn có lượng bức xạ mặt trời cao bất thường (**màu đỏ**) hoặc thấp bất thường (**màu xanh**) "
    "so với trung bình khí hậu. Những biến động này thường liên quan mật thiết đến sự thay đổi của độ che phủ mây và độ trong suốt của khí quyển qua từng năm."
)

monthly_anom = df_anom.groupby(["year", "month"])[rad_col].mean().reset_index()
heatmap_data = monthly_anom.pivot(index="year", columns="month", values=rad_col)
vmax = float(np.abs(heatmap_data.values).max()) * 0.8

fig2 = go.Figure(data=go.Heatmap(
    z=heatmap_data.values, x=MONTH_LABELS, y=heatmap_data.index,
    colorscale="RdBu_r", zmid=0, zmin=-vmax, zmax=vmax,
    colorbar=dict(title="Dị thường bức xạ (W/m²)")
))
fig2.update_layout(title="Heatmap dị thường bức xạ Mặt Trời trên bề mặt",
    xaxis_title="Tháng", yaxis_title="Năm", height=600)
st.plotly_chart(fig2, width="stretch")