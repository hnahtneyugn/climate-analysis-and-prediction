"""
Page — Surface Heat Fluxes Analysis
Parquet files: daily.parquet, monthly.parquet, climatology.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Surface Heat Fluxes", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "surface_heat_fluxes"

st.title("Surface Heat Fluxes Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

for df in [df_daily, df_monthly]:
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
    else:
        df.reset_index(inplace=True)
        df["time"] = pd.to_datetime(df["time"])

df_clim = df_clim.reset_index() if "month" not in df_clim.columns else df_clim

if "month" not in df_daily.columns:
    df_daily["month"] = df_daily["time"].dt.month
if "year" not in df_daily.columns:
    df_daily["year"] = df_daily["time"].dt.year

st.markdown("---")


# Section 1 — Seasonal energy balance
st.subheader("1. Chu kì mùa của hệ năng lượng cân bằng")

clim_shf = df_clim["shf"]
clim_lhf = df_clim["lhf"]
clim_rad = clim_shf + clim_lhf + 0.05 * (clim_shf + clim_lhf)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim_rad,
    mode="lines", line=dict(color="#BA7517", dash="dash", width=2), name="Bức xạ thuần (Rn)"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim_shf,
    mode="lines", line=dict(color="#D85A30", width=2.5), name="Lượng nhiệt thấy được (H)"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim_lhf,
    mode="lines", line=dict(color="#185FA5", width=2.5), name="Lượng nhiệt ẩn (LE)"))

lhf_arr = clim_lhf.values
shf_arr = clim_shf.values
wet_mask = lhf_arr > shf_arr
for start, end in zip(np.where(np.diff(wet_mask.astype(int)) != 0)[0],
                      np.where(np.diff(wet_mask.astype(int)) != 0)[0][1:] + 1 if True else []):
    pass

fig1.add_trace(go.Scatter(
    x=MONTH_LABELS + MONTH_LABELS[::-1],
    y=list(np.where(lhf_arr > shf_arr, lhf_arr, shf_arr)) +
      list(np.where(lhf_arr > shf_arr, shf_arr, lhf_arr))[::-1],
    fill="toself", fillcolor="rgba(24,95,165,0.15)", line=dict(color="rgba(0,0,0,0)"),
    name="LE > H (Wet)"))

fig1.add_trace(go.Scatter(
    x=MONTH_LABELS, y=(clim_shf + clim_lhf).values,
    mode="lines", line=dict(color="gray", width=0),
    fill="tonexty", fillcolor="rgba(128,128,128,0.2)", name="Dòng nhiệt đi vào đất (G)"))

fig1.update_layout(title="Chu kì mùa của hệ năng lượng cân bằng",
    xaxis_title="Tháng", yaxis_title="Dòng chảy năng lượng (W/m²)", height=550)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Bowen ratio seasonal and trend
st.subheader("2. Chu kì và xu hướng dài hạn của tỉ lệ Bowen")

if "anom_br" in df_daily.columns:
    daily_br_ts = df_daily.set_index("time")["anom_br"]
else:
    daily_br_ts = df_daily.set_index("time")["shf"] / df_daily.set_index("time")["lhf"]

annual_br = daily_br_ts.resample("YE").mean()
years_br  = annual_br.index.year.values
br_vals   = annual_br.values
slope_br, intercept_br, _, p_br, _ = stats.linregress(years_br, br_vals)

fig2 = make_subplots(rows=1, cols=2,
    subplot_titles=("Chu kỳ theo mùa của tỉ lệ Bowen",
                    f"Xu hướng: {slope_br*10:+.4f}/decade (p={p_br:.3f})"))

fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=df_clim["br"],
    mode="lines", line=dict(color="#2C2C2A", width=2), showlegend=False), row=1, col=1)
fig2.add_hline(y=1.0, line=dict(color="gray", dash="dash"), row=1, col=1)

fig2.add_trace(go.Scatter(x=years_br, y=br_vals, mode="markers",
    marker=dict(color="#2C2C2A", size=8, opacity=0.8), showlegend=False), row=1, col=2)
fig2.add_trace(go.Scatter(x=years_br, y=intercept_br + slope_br * years_br,
    mode="lines", line=dict(color="#D85A30", width=1.5), showlegend=False), row=1, col=2)

fig2.update_xaxes(title_text="Tháng", row=1, col=1)
fig2.update_yaxes(title_text="Tỉ lệ Bowen β", row=1, col=1)
fig2.update_xaxes(title_text="Năm", row=1, col=2)
fig2.update_yaxes(title_text="Tỉ lệ Bowen trung bình năm", row=1, col=2)
fig2.update_layout(height=500)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Bowen anomaly heatmap
st.subheader("3. Heatmap dị thường tỉ lệ Bowen")

df_monthly["year"]  = df_monthly["time"].dt.year
df_monthly["month"] = df_monthly["time"].dt.month
df_anom_br = df_monthly.merge(df_clim[["month", "br"]], on="month", suffixes=("", "_clim"))
df_anom_br["br_anomaly"] = df_anom_br["br"] - df_anom_br["br_clim"]
pivot_br = df_anom_br.pivot(index="year", columns="month", values="br_anomaly")

fig3 = go.Figure(data=go.Heatmap(
    z=pivot_br.values, x=MONTH_LABELS, y=pivot_br.index,
    colorscale="RdBu_r", zmid=0,
    colorbar=dict(title="Dị thường Bowen")
))
fig3.update_layout(title="Heatmap dị thường tỉ lệ Bowen (1980-2024)",
    xaxis_title="Tháng", yaxis_title="Năm", height=600)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Bowen seasonal boxplot by decade
st.subheader("4. Xu hướng chuyển dịch theo mùa của tỉ lệ Bowen qua các thập kỉ")

df_plot = df_daily.copy()
df_plot["br"] = df_plot["shf"] / df_plot["lhf"].replace(0, np.nan)
df_plot["decade"] = (df_plot["year"] // 10) * 10
df_plot["decade_label"] = df_plot["decade"].astype(str) + "s"

def get_season(month):
    if month in [12, 1, 2]: return "DJF (Mùa khô / Đông)"
    if month in [3, 4, 5]:  return "MAM (Mùa chuyển tiếp 1)"
    if month in [6, 7, 8]:  return "JJA (Mùa mưa / Hạ)"
    return "SON (Mùa chuyển tiếp 2)"

df_plot["season"] = df_plot["month"].apply(get_season)
df_clean = df_plot[df_plot["br"].between(0, 5)].copy()
season_order = ["DJF (Mùa khô / Đông)", "MAM (Mùa chuyển tiếp 1)",
                "JJA (Mùa mưa / Hạ)", "SON (Mùa chuyển tiếp 2)"]
decades = sorted(df_clean["decade_label"].unique())

fig4 = make_subplots(rows=1, cols=4, subplot_titles=season_order, shared_yaxes=True)
colors_dec = ["#fee5d9", "#fc9272", "#de2d26", "#a50f15", "#67000d"]

for col_idx, season in enumerate(season_order, 1):
    s_data = df_clean[df_clean["season"] == season]
    for d_idx, decade in enumerate(decades):
        vals = s_data[s_data["decade_label"] == decade]["br"].dropna()
        fig4.add_trace(go.Box(y=vals, name=decade, boxmean=True,
            marker_color=colors_dec[d_idx % len(colors_dec)],
            showlegend=(col_idx == 1)), row=1, col=col_idx)
    fig4.update_xaxes(showticklabels=False, row=1, col=col_idx)

fig4.update_yaxes(title_text="Tỉ lệ Bowen (H/LE)", row=1, col=1)
fig4.update_layout(height=600, title="Xu hướng chuyển dịch theo mùa của tỉ lệ Bowen xuyên suốt các thập kỷ",
    boxmode="group")
st.plotly_chart(fig4, width="stretch")
