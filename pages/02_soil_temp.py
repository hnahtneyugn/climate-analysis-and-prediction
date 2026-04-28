"""
Page — Soil Temperature Analysis
Parquet files: daily.parquet, monthly.parquet, yearly.parquet, anomaly.parquet, climatology.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Soil Temperature", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

soil_vars   = ["soil_level_1", "soil_level_2", "soil_level_3", "soil_level_4"]
layer_names = ["7cm", "28cm", "100cm", "289cm"]
depths      = [0.035, 0.175, 0.640, 1.945]
COLORS      = ["#d73027", "#fc8d59", "#91bfdb", "#4575b4"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "soil_temp"

st.title("Soil Temperature Analysis")

df_daily     = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly   = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_yearly    = pd.read_parquet(f"{DATA_DIR}/yearly.parquet")
df_anomaly   = pd.read_parquet(f"{DATA_DIR}/anomaly.parquet")
climatology  = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

df_daily.index   = pd.to_datetime(df_daily.index)
df_monthly.index = pd.to_datetime(df_monthly.index)
df_yearly.index  = pd.to_datetime(df_yearly.index)

st.markdown("---")


# Section 1 — Heatmap soil heat penetration
st.subheader("1. Sự lan truyền sóng nhiệt theo độ sâu đất & Dấu hiệu nóng lên")
st.markdown(
    "Biểu đồ Heatmap thể hiện sự thâm nhập của nhiệt lượng từ bề mặt xuống các tầng đất sâu theo từng tháng. "
    "Việc so sánh giữa thập kỷ 1980 (giữa) và giai đoạn gần đây (phải) giúp làm rõ sự gia tăng nhiệt độ tích tụ "
    "trong lòng đất qua hơn 40 năm."
)

clim_80s    = df_monthly.loc["1980":"1989"].groupby(df_monthly.loc["1980":"1989"].index.month).mean()
clim_recent = df_monthly.loc["2015":"2024"].groupby(df_monthly.loc["2015":"2024"].index.month).mean()

clim_vals_all    = climatology[soil_vars].values.T
clim_vals_80s    = clim_80s[soil_vars].values.T
clim_vals_recent = clim_recent[soil_vars].values.T
vmin = float(np.min([clim_vals_80s, clim_vals_recent]))
vmax = float(np.max([clim_vals_80s, clim_vals_recent]))

fig1 = make_subplots(rows=1, cols=3,
    subplot_titles=("Trung bình 45 năm", "Thập kỷ 1980-1989", "Thập kỷ 2015-2024"))

for col_idx, (data, title) in enumerate(zip(
        [clim_vals_all, clim_vals_80s, clim_vals_recent],
        ["45 năm", "1980s", "2020s"]), 1):
    fig1.add_trace(go.Heatmap(
        z=data, x=MONTH_LABELS, y=layer_names,
        colorscale="RdYlBu_r", zmin=vmin, zmax=vmax,
        showscale=(col_idx == 3),
        colorbar=dict(title="°C", x=1.02) if col_idx == 3 else None
    ), row=1, col=col_idx)

fig1.update_layout(height=400, title="Nhiệt độ đất theo độ sâu và tháng")
fig1.update_yaxes(autorange="reversed")
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Seasonal cycle all layers
st.subheader("2. Chu kỳ mùa nhiệt độ đất: Độ trễ pha và suy giảm biên độ")
st.markdown(
    "Quan sát chu kỳ mùa ở các tầng đất khác nhau. Các tầng đất càng sâu thì nhiệt độ càng ổn định (biên độ dao động thấp) "
    "và thời gian đạt nhiệt độ cực đại sẽ **trễ hơn** đáng kể so với bề mặt do khả năng dẫn nhiệt của đất có giới hạn."
)

fig2 = go.Figure()
for var, color, name in zip(soil_vars, COLORS, layer_names):
    fig2.add_trace(go.Scatter(x=MONTH_LABELS, y=climatology[var],
        mode="lines+markers", line=dict(color=color, width=2.5), name=name))
fig2.update_layout(title="Chu kỳ mùa nhiệt độ đất: Độ trễ pha và suy giảm biên độ",
    xaxis_title="Tháng", yaxis_title="Nhiệt độ trung bình (°C)", height=500)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Long-term warming trend per layer
st.subheader("3. Xu hướng nóng lên dài hạn tại các tầng đất")
st.markdown(
    "Theo dõi xu hướng nóng lên của đất từ năm 1980 đến nay. Đường nét đứt thể hiện tốc độ gia tăng nhiệt độ trên mỗi thập kỷ. "
    "Mặc dù nằm sâu dưới lòng đất, các tầng đất 100cm và 289cm vẫn đang nóng lên một cách rõ rệt."
)

fig3 = go.Figure()
x_vals = df_yearly.index.year.values
for var, color, name in zip(soil_vars, COLORS, layer_names):
    y_vals = df_yearly[var].values
    slope, intercept, _, _, _ = stats.linregress(x_vals, y_vals)
    fig3.add_trace(go.Scatter(x=x_vals, y=y_vals,
        mode="lines", line=dict(color=color, width=1.5), opacity=0.6,
        showlegend=False))
    fig3.add_trace(go.Scatter(x=x_vals, y=intercept + slope * x_vals,
        mode="lines", line=dict(color=color, dash="dash", width=2.5),
        name=f"{name}: +{slope*10:.3f}°C/thập kỷ"))
fig3.update_layout(title="Xu hướng nóng lên dài hạn tại các tầng đất",
    xaxis_title="Năm", yaxis_title="Nhiệt độ đất trung bình năm (°C)", height=500)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Anomaly heatmaps
st.subheader("4. Ma trận dị thường nhiệt độ đất: Sự thẩm thấu của các năm cực đoan")
st.markdown(
    "Các ma trận dị thường thể hiện những giai đoạn đất nóng hơn (đỏ) hoặc lạnh hơn (xanh) trung bình lịch sử. "
    "Có thể thấy rõ sự 'thẩm thấu' của các đợt nóng kỷ lục từ tầng mặt (7cm) lan dần xuống các tầng sâu bên dưới."
)

vmax_anom = float(df_anomaly[soil_vars].abs().max().max())
fig4 = make_subplots(rows=2, cols=2,
    subplot_titles=[f"Dị thường tầng {n}" for n in layer_names])

positions = [(1,1),(1,2),(2,1),(2,2)]
for (r, c), var in zip(positions, soil_vars):
    pivot = df_anomaly.pivot(index="year", columns="month", values=var)
    fig4.add_trace(go.Heatmap(
        z=pivot.values, x=MONTH_LABELS, y=pivot.index,
        colorscale="RdBu_r", zmid=0, zmin=-vmax_anom, zmax=vmax_anom,
        showscale=(r==2 and c==2),
        colorbar=dict(title="Dị thường (°C)") if r==2 and c==2 else None
    ), row=r, col=c)

fig4.update_layout(height=800, title="Ma trận dị thường nhiệt độ đất theo tầng")
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Thermal lag
st.subheader("5. Lượng hóa Độ trễ pha nhiệt (Thermal Lag) theo Độ sâu")
st.markdown(
    "Phân tích mối quan hệ giữa độ sâu và thời điểm nhiệt độ đạt đỉnh trong năm. "
    "Đường xu hướng cho thấy nhiệt độ đất mất khoảng **hơn 1 tháng** để truyền sóng nhiệt xuống mỗi mét độ sâu."
)

peak_months = climatology[soil_vars].idxmax().values.astype(float)
slope_lag, int_lag, r_lag, _, _ = stats.linregress(depths, peak_months)
x_plot = np.linspace(0, 2.5, 100)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=depths, y=peak_months, mode="markers",
    marker=dict(color="darkred", size=10), name="Tháng đỉnh"))
fig5.add_trace(go.Scatter(x=x_plot, y=int_lag + slope_lag * x_plot,
    mode="lines", line=dict(color="black", dash="dash"),
    name=f"Trend: +{slope_lag:.1f} tháng/mét (R²={r_lag**2:.2f})"))
fig5.update_layout(title="Độ trễ pha nhiệt theo độ sâu đất",
    xaxis_title="Độ sâu (mét)", yaxis_title="Tháng đạt nhiệt độ đỉnh", height=500,
    yaxis=dict(tickvals=list(range(1,13)), ticktext=MONTH_LABELS))
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — Cross-correlation
st.subheader("6. Tương quan và Cross-Correlation giữa các tầng đất")
st.markdown(
    "**Bên trái:** Tương quan tức thời cho thấy tầng mặt liên kết rất chặt chẽ với các tầng kế cận. "
    "**Bên phải:** Phân tích Cross-correlation cho thấy sự ảnh hưởng từ tầng 7cm mất một khoảng thời gian (ngày) để tác động mạnh nhất đến tầng 289cm."
)

fig6 = make_subplots(rows=1, cols=2,
    subplot_titles=("Tương quan Pearson (Độ trễ = 0)", "Cross-Correlation: Level 1 vs Level 4"))

corr_matrix = df_daily[soil_vars].corr()
fig6.add_trace(go.Heatmap(
    z=corr_matrix.values, x=layer_names, y=layer_names,
    colorscale="magma", zmin=0, zmax=1,
    text=corr_matrix.round(3).values.astype(str),
    texttemplate="%{text}",
    colorbar=dict(x=0.45)
), row=1, col=1)

lags = np.arange(0, 120)
corrs = [df_daily["soil_level_1"].corr(df_daily["soil_level_4"].shift(-lag)) for lag in lags]
max_lag = int(lags[np.argmax(corrs)])
max_corr = float(np.max(corrs))
fig6.add_trace(go.Scatter(x=lags, y=corrs, mode="lines", line=dict(color="teal", width=2),
    name="Cross-correlation"), row=1, col=2)
fig6.add_vline(x=max_lag, line=dict(color="red", dash="dash"),
    annotation_text=f"Max lag: {max_lag}d (r={max_corr:.2f})",
    annotation_position="top right", row=1, col=2)

fig6.update_layout(height=450, showlegend=False)
fig6.update_xaxes(title_text="Độ trễ thời gian (Ngày)", row=1, col=2)
fig6.update_yaxes(title_text="Hệ số tương quan (r)", row=1, col=2)
st.plotly_chart(fig6, width="stretch")
st.markdown("---")


# Section 7 — Soil Level 1 vs T2m
st.subheader("7. So sánh tương tác: Nhiệt độ bề mặt đất (Level 1) vs Nhiệt độ không khí")
st.markdown(
    "So sánh nhiệt độ giữa môi trường đất và không khí xung quanh. Phần tô màu đỏ chỉ ra những năm "
    "mà bề mặt đất hấp thụ và giữ nhiệt cao hơn đáng kể so với nhiệt độ không khí, ảnh hưởng đến quá trình trao đổi năng lượng bề mặt."
)

fig7 = go.Figure()
fig7.add_trace(go.Scatter(x=df_yearly.index.year, y=df_yearly["t2m"],
    mode="lines", line=dict(color="gray", dash="dash", width=2), name="Nhiệt độ không khí (T2m)"))
fig7.add_trace(go.Scatter(x=df_yearly.index.year, y=df_yearly["soil_level_1"],
    mode="lines", line=dict(color="#d73027", width=2.5), name="Nhiệt độ mặt đất (Level 1)"))

# Fill where soil > t2m
mask_warm = df_yearly["soil_level_1"] > df_yearly["t2m"]
fig7.add_trace(go.Scatter(
    x=np.concatenate([df_yearly.index.year, df_yearly.index.year[::-1]]),
    y=np.concatenate([
        np.where(mask_warm, df_yearly["soil_level_1"], df_yearly["t2m"]),
        np.where(mask_warm, df_yearly["t2m"], df_yearly["soil_level_1"])[::-1]
    ]),
    fill="toself", fillcolor="rgba(255,0,0,0.1)", line=dict(color="rgba(0,0,0,0)"),
    name="Đất nóng hơn Không khí"
))

fig7.update_layout(title="Nhiệt độ bề mặt đất (Level 1) vs Nhiệt độ không khí",
    xaxis_title="Năm", yaxis_title="Nhiệt độ trung bình năm (°C)", height=500)
st.plotly_chart(fig7, width="stretch")