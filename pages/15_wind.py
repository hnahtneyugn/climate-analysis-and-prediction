"""
Page — Wind Analysis
Parquet files expected:
  daily.parquet       — columns: time, ws10, ws100, wd10, wd100, wd10_circ, wd100_circ,
                                 wpd10, wpd100, alpha, anom_ws10, anom_ws100, month, year, decade
  monthly.parquet     — columns: time, ws10, ws100, wpd100
  climatology.parquet — columns: month, ws10, ws100, alpha, wpd100
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.stats import weibull_min, linregress, circmean
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Wind Analysis", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "wind"

st.title("Wind Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

if "time" in df_daily.columns:
    df_daily["time"] = pd.to_datetime(df_daily["time"])
if "time" in df_monthly.columns:
    df_monthly["time"] = pd.to_datetime(df_monthly["time"])

if "year" not in df_daily.columns:
    df_daily["year"]  = df_daily["time"].dt.year
if "month" not in df_daily.columns:
    df_daily["month"] = df_daily["time"].dt.month
if "decade" not in df_daily.columns:
    df_daily["decade"] = (df_daily["year"] // 10) * 10

st.success("Data loaded", icon="✅")
st.markdown("---")


# Section 1 — Hướng gió và tốc độ gió theo mùa (Polar bar charts)

st.subheader("1. Biểu đồ hướng gió và tốc độ gió theo mùa")
st.markdown(
    "Polar bar charts showing wind direction frequency weighted by speed bins, "
    "for 10m and 100m, split by season."
)

season_months_map = {
    "DJF (Đông)":  [12, 1, 2],
    "MAM (Xuân)":  [3, 4, 5],
    "JJA (Hạ)":   [6, 7, 8],
    "SON (Thu)":   [9, 10, 11],
}
speed_bins = [0, 2, 4, 6, 8, 10, 14, 20]
dir_bins   = np.arange(0, 360, 22.5)
heights    = [("10m", "ws10", "wd10_circ"), ("100m", "ws100", "wd100_circ")]

bin_colors = [
    "#313695", "#4575b4", "#74add1", "#abd9e9",
    "#fee090", "#fdae61", "#f46d43", "#a50026"
]

fig1 = make_subplots(
    rows=2, cols=4,
    subplot_titles=[
        f"{h[0]} — {s}" for h in heights for s in season_months_map
    ],
    specs=[[{"type": "polar"}] * 4 for _ in range(2)]
)

for h_idx, (hlabel, ws_col, wd_col) in enumerate(heights):
    for s_idx, (s_name, s_months) in enumerate(season_months_map.items()):
        sub = df_daily[df_daily["month"].isin(s_months)]
        row = h_idx + 1
        col = s_idx + 1

        for b_idx in range(len(speed_bins) - 1):
            lo, hi = speed_bins[b_idx], speed_bins[b_idx + 1]
            mask = (sub[ws_col] >= lo) & (sub[ws_col] < hi)
            wd_sub = sub.loc[mask, wd_col]

            counts, _ = np.histogram(wd_sub, bins=np.append(dir_bins, 360))
            total = len(sub) + 1e-9
            r_vals = counts / total * 100

            fig1.add_trace(go.Barpolar(
                r=r_vals,
                theta=dir_bins,
                name=f"{lo}-{hi} m/s",
                marker_color=bin_colors[b_idx],
                showlegend=(h_idx == 0 and s_idx == 0),
                legendgroup=f"bin{b_idx}"
            ), row=row, col=col)

fig1.update_layout(
    title="Phân bố hướng gió và tốc độ gió theo mùa: 10m vs 100m",
    height=700,
    polar=dict(angularaxis=dict(direction="clockwise", rotation=90)),
)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Chỉ số tốc độ gió theo tháng

st.subheader("2. Chỉ số khí hậu về tốc độ gió qua các tháng")
st.markdown(
    "Monthly climatological wind speed at 10m and 100m with ±1 std bands. "
    "Secondary axis shows the speed-up ratio ws100/ws10."
)

clim_stats = df_daily.groupby("month").agg(
    ws10_mean=("ws10", "mean"), ws10_std=("ws10", "std"),
    ws100_mean=("ws100", "mean"), ws100_std=("ws100", "std")
).reset_index()
clim_stats["ratio"] = clim_stats["ws100_mean"] / clim_stats["ws10_mean"]

fig2 = make_subplots(specs=[[{"secondary_y": True}]])

fig2.add_trace(go.Scatter(
    x=clim_stats["month"], y=clim_stats["ws100_mean"],
    name="Tốc độ gió 100m", mode="lines+markers",
    line=dict(color="tomato", width=2),
    marker=dict(symbol="circle")
), secondary_y=False)

fig2.add_trace(go.Scatter(
    x=pd.concat([clim_stats["month"], clim_stats["month"][::-1]]),
    y=pd.concat([
        clim_stats["ws100_mean"] + clim_stats["ws100_std"],
        (clim_stats["ws100_mean"] - clim_stats["ws100_std"])[::-1]
    ]),
    fill="toself", fillcolor="rgba(255,99,71,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="100m ±1 std", showlegend=True
), secondary_y=False)

fig2.add_trace(go.Scatter(
    x=clim_stats["month"], y=clim_stats["ws10_mean"],
    name="Tốc độ gió 10m", mode="lines+markers",
    line=dict(color="dodgerblue", width=2),
    marker=dict(symbol="square")
), secondary_y=False)

fig2.add_trace(go.Scatter(
    x=pd.concat([clim_stats["month"], clim_stats["month"][::-1]]),
    y=pd.concat([
        clim_stats["ws10_mean"] + clim_stats["ws10_std"],
        (clim_stats["ws10_mean"] - clim_stats["ws10_std"])[::-1]
    ]),
    fill="toself", fillcolor="rgba(30,144,255,0.15)",
    line=dict(color="rgba(255,255,255,0)"),
    name="10m ±1 std", showlegend=True
), secondary_y=False)

fig2.add_trace(go.Scatter(
    x=clim_stats["month"], y=clim_stats["ratio"],
    name="Speed-up ratio (ws100/ws10)", mode="lines",
    line=dict(color="black", dash="dash", width=1.5),
    opacity=0.6
), secondary_y=True)

fig2.update_xaxes(title_text="Tháng", tickvals=list(range(1, 13)), ticktext=MONTH_LABELS)
fig2.update_yaxes(title_text="Tốc độ gió (m/s)", secondary_y=False)
fig2.update_yaxes(title_text="Speed-up ratio", secondary_y=True)
fig2.update_layout(
    title="Chỉ số tốc độ gió ở độ cao: 10m vs 100m",
    height=500, hovermode="x unified"
)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Xu hướng dài hạn tốc độ gió và Alpha

st.subheader("3. Xu hướng dài hạn của tốc độ gió và chỉ số Alpha")
st.markdown(
    "Annual mean wind speed at 10m/100m and wind shear exponent (α) with trend lines."
)

annual_df = df_daily.groupby("year").agg(
    ws10=("ws10", "mean"),
    ws100=("ws100", "mean"),
    alpha=("alpha", "mean")
).reset_index()

s10, i10, _, p10, _ = linregress(annual_df["year"], annual_df["ws10"])
s100, i100, _, p100, _ = linregress(annual_df["year"], annual_df["ws100"])
sa, ia, _, pa, _ = linregress(annual_df["year"], annual_df["alpha"])

fig3 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=["Tốc độ gió trung bình năm", "Chỉ số Alpha trung bình năm"])

for col_name, color, slope, intercept, pval, name in [
    ("ws10",  "dodgerblue", s10,  i10,  p10,  "10m"),
    ("ws100", "tomato",  s100, i100, p100, "100m"),
]:
    fig3.add_trace(go.Scatter(
        x=annual_df["year"], y=annual_df[col_name],
        mode="lines+markers", marker=dict(size=4),
        line=dict(color=color, width=1), opacity=0.5,
        showlegend=False
    ), row=1, col=1)
    fig3.add_trace(go.Scatter(
        x=annual_df["year"],
        y=intercept + slope * annual_df["year"],
        mode="lines", line=dict(color=color, width=2),
        name=f"{name} Trend: {slope*10:.3f} m/s/decade (p={pval:.3f})"
    ), row=1, col=1)

fig3.add_trace(go.Scatter(
    x=annual_df["year"], y=annual_df["alpha"],
    mode="lines+markers", marker=dict(size=4),
    line=dict(color="black", width=1), opacity=0.5,
    showlegend=False
), row=2, col=1)
fig3.add_trace(go.Scatter(
    x=annual_df["year"],
    y=ia + sa * annual_df["year"],
    mode="lines", line=dict(color="black", width=2),
    name=f"Alpha Trend: {sa*10:.4f}/decade (p={pa:.3f})"
), row=2, col=1)

fig3.update_yaxes(title_text="Tốc độ gió (m/s)", row=1, col=1)
fig3.update_yaxes(title_text="Alpha", row=2, col=1)
fig3.update_xaxes(title_text="Năm", row=2, col=1)
fig3.update_layout(height=600, hovermode="x")
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Mật độ công suất gió và tần suất vượt ngưỡng

st.subheader("4. Xu hướng mật độ công suất gió tại 100m và tần suất gió vượt ngưỡng")
st.markdown(
    "Annual wind power density at 100m and fraction of days above/below the 3 m/s operating threshold."
)

df_daily["is_high"] = (df_daily["ws100"] >= 3).astype(int)
df_daily["is_low"]  = (df_daily["ws100"] <  3).astype(int)

annual_wpd = df_daily.groupby("year").agg(
    wpd100=("wpd100", "mean"),
    high_pct=("is_high", "mean"),
    low_pct=("is_low",  "mean")
).reset_index()
annual_wpd["high_pct"] *= 100
annual_wpd["low_pct"]  *= 100

sw, iw, _, pw, _ = linregress(annual_wpd["year"], annual_wpd["wpd100"])
total_change_pct = ((sw * (annual_wpd["year"].max() - annual_wpd["year"].min()))
                    / annual_wpd["wpd100"].mean()) * 100

fig4 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=[
                         "Mật độ công suất gió hàng năm tại 100m",
                         "Tần suất vượt ngưỡng vận hành"
                     ])

fig4.add_trace(go.Scatter(
    x=annual_wpd["year"], y=annual_wpd["wpd100"],
    mode="lines+markers", marker=dict(size=5),
    line=dict(color="forestgreen", width=1), opacity=0.6,
    name="WPD trung bình năm", showlegend=True
), row=1, col=1)
fig4.add_trace(go.Scatter(
    x=annual_wpd["year"],
    y=iw + sw * annual_wpd["year"],
    mode="lines", line=dict(color="red", dash="dash", width=2),
    name=f"Xu hướng (p={pw:.3f}), {total_change_pct:+.1f}% tổng chu kì"
), row=1, col=1)

for col_key, color, label, sym in [
    ("high_pct", "darkblue",   "Ngày ≥ 3 m/s", "square"),
    ("low_pct",  "darkorange", "Ngày < 3 m/s",  "triangle-down"),
]:
    s, i, _, p, _ = linregress(annual_wpd["year"], annual_wpd[col_key])
    fig4.add_trace(go.Scatter(
        x=annual_wpd["year"], y=annual_wpd[col_key],
        mode="lines+markers", marker=dict(size=5, symbol=sym),
        line=dict(color=color, width=1.5),
        name=label
    ), row=2, col=1)
    fig4.add_trace(go.Scatter(
        x=annual_wpd["year"],
        y=i + s * annual_wpd["year"],
        mode="lines", line=dict(color=color, dash="dot", width=1),
        showlegend=False
    ), row=2, col=1)

fig4.update_yaxes(title_text="Mật độ công suất gió (W/m²)", row=1, col=1)
fig4.update_yaxes(title_text="% Số ngày trong năm", row=2, col=1)
fig4.update_xaxes(title_text="Năm", row=2, col=1)
fig4.update_layout(height=600, hovermode="x")
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Heatmap dị thường tốc độ gió 10m

st.subheader("5. Heatmap dị thường tốc độ gió trong các năm tại độ cao 10m")
st.markdown("Red = faster than climatology; Blue = slower.")

heatmap_data = df_daily.pivot_table(index="year", columns="month", values="anom_ws10", aggfunc="mean")

fig5 = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=MONTH_LABELS,
    y=heatmap_data.index,
    colorscale="RdBu_r",
    zmid=0,
    colorbar=dict(title="Dị thường (m/s)")
))
fig5.update_layout(
    title="Heatmap dị thường tốc độ gió trong các năm",
    xaxis_title="Tháng", yaxis_title="Năm",
    height=600
)
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — Độ nhất quán hướng gió mùa Tây Nam

st.subheader("6. Độ nhất quán của hướng gió mùa Tây Nam trong mùa hạ")
st.markdown("Fraction of JJA days with wind direction within ±45° of the dominant direction.")

monsoon_data = df_daily[df_daily["month"].isin([6, 7, 8])].copy()
angles_rad = np.deg2rad(monsoon_data["wd10_circ"])
dom_dir = np.rad2deg(circmean(angles_rad))

def is_in_monsoon_stream(wd, target):
    diff = np.abs(wd - target)
    return (diff <= 45) | (diff >= 315)

monsoon_data["consistent"] = is_in_monsoon_stream(monsoon_data["wd10_circ"], dom_dir)
monsoon_consistency = monsoon_data.groupby("year")["consistent"].mean()

z = np.polyfit(monsoon_consistency.index, monsoon_consistency.values, 1)
p_trend = np.poly1d(z)

fig6 = go.Figure()
fig6.add_trace(go.Scatter(
    x=monsoon_consistency.index, y=monsoon_consistency.values,
    mode="lines+markers", marker=dict(size=6, color="teal"),
    line=dict(color="teal", width=1.5),
    name="Tỉ lệ nhất quán"
))
fig6.add_trace(go.Scatter(
    x=monsoon_consistency.index,
    y=p_trend(monsoon_consistency.index),
    mode="lines", line=dict(color="red", dash="dash", width=1.5),
    name="Xu hướng"
))
fig6.update_layout(
    title=f"Độ nhất quán hướng gió mùa Tây Nam (±45° từ hướng {dom_dir:.0f}°)",
    xaxis_title="Năm",
    yaxis_title="Tỉ lệ số ngày nhất quán",
    hovermode="x unified", height=450
)
st.plotly_chart(fig6, width="stretch")
st.markdown("---")


# Section 7 — Chu kỳ và tần suất của chỉ số Alpha

st.subheader("7. Biểu đồ chu kỳ và tần suất của chỉ số Alpha")
st.markdown("Left: monthly climatology of Alpha. Right: annual count of extreme shear days.")

alpha_clim = df_daily.groupby("month")["alpha"].agg(["mean", "std"]).reset_index()
high_shear = df_daily[df_daily["alpha"] > 0.4].groupby("year").size()
low_shear  = df_daily[df_daily["alpha"] < 0.1].groupby("year").size()
years_all  = sorted(df_daily["year"].unique())

fig7 = make_subplots(rows=1, cols=2,
                     subplot_titles=["Chu kỳ theo tháng của chỉ số Alpha",
                                     "Tần suất có sự đứt gãy gió cực đoan"])

fig7.add_trace(go.Scatter(
    x=alpha_clim["month"], y=alpha_clim["mean"],
    mode="lines+markers", line=dict(color="purple", width=2),
    name="Alpha mean", showlegend=False
), row=1, col=1)
fig7.add_trace(go.Scatter(
    x=pd.concat([alpha_clim["month"], alpha_clim["month"][::-1]]),
    y=pd.concat([
        alpha_clim["mean"] + alpha_clim["std"],
        (alpha_clim["mean"] - alpha_clim["std"])[::-1]
    ]),
    fill="toself", fillcolor="rgba(128,0,128,0.2)",
    line=dict(color="rgba(255,255,255,0)"),
    name="±1 std", showlegend=False
), row=1, col=1)

fig7.add_trace(go.Bar(
    x=[y - 0.2 for y in years_all],
    y=[high_shear.get(y, 0) for y in years_all],
    name="Strong Shear (>0.4)", marker_color="darkred", width=0.4
), row=1, col=2)
fig7.add_trace(go.Bar(
    x=[y + 0.2 for y in years_all],
    y=[low_shear.get(y, 0) for y in years_all],
    name="Well-Mixed (<0.1)", marker_color="skyblue", width=0.4
), row=1, col=2)

fig7.update_xaxes(title_text="Tháng", tickvals=list(range(1, 13)), ticktext=MONTH_LABELS, row=1, col=1)
fig7.update_xaxes(title_text="Năm", row=1, col=2)
fig7.update_yaxes(title_text="Alpha", row=1, col=1)
fig7.update_yaxes(title_text="Số ngày", row=1, col=2)
fig7.update_layout(height=500, barmode="overlay")
st.plotly_chart(fig7, width="stretch")
st.markdown("---")


# Section 8 — Weibull theo thập kỉ

st.subheader("8. Biểu đồ Weibull theo thập kỷ về tốc độ gió ở độ cao 100m")
st.markdown("Fitted Weibull PDF per decade — stable shape confirms consistent wind resource.")

decades = sorted(df_daily["decade"].unique())
colors_wb = [f"rgba({r},{g},{b},0.9)" for r, g, b in [
    (49, 54, 149), (69, 117, 180), (116, 173, 209), (253, 174, 97), (215, 48, 39)
]]
x_range = np.linspace(0, 25, 200)

fig8 = go.Figure()
for i, decade in enumerate(decades):
    data = df_daily[df_daily["decade"] == decade]["ws100"].dropna()
    shape, loc, scale = weibull_min.fit(data, floc=0)
    pdf = weibull_min.pdf(x_range, shape, loc, scale)
    fig8.add_trace(go.Scatter(
        x=x_range, y=pdf,
        mode="lines", line=dict(color=colors_wb[i % len(colors_wb)], width=2.5),
        name=f"{decade}s (k={shape:.2f}, c={scale:.2f} m/s)"
    ))

fig8.update_layout(
    title="Sự dịch chuyển của phân bố Weibull theo thập kỷ (100m)",
    xaxis_title="Tốc độ gió (m/s)",
    yaxis_title="Mật độ xác suất",
    xaxis=dict(range=[0, 20]),
    hovermode="x unified", height=500
)
st.plotly_chart(fig8, width="stretch")
