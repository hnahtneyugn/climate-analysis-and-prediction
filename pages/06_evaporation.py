"""
Page 1 — Evaporation Analysis
Mirrors all 6 sections from 06_evaporation.ipynb.

Expected parquet files (exported at the end of the notebook):
  • daily.parquet       — columns: time, evap, pevap, ef, def, evap_anom, ef_anom
  • monthly.parquet     — columns: time, evap, pevap, ef
  • climatology.parquet — columns: month, evap, pevap, ef
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="Evaporation", page_icon="🌫️", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_season(month: int) -> str:
    if month in (12, 1, 2):  return "DJF (Winter/Dry)"
    if month in (3,  4, 5):  return "MAM (Spring)"
    if month in (6,  7, 8):  return "JJA (Summer/Wet)"
    return "SON (Autumn)"


def add_derived_cols(df_daily: pd.DataFrame) -> pd.DataFrame:
    """Attach season / decade / stress_ratio / season_type columns."""
    df = df_daily.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["season"]      = df["time"].dt.month.map(get_season)
    df["decade"]      = (df["time"].dt.year // 10) * 10
    df["stress_ratio"] = df["pevap"] / (df["evap"] + 1e-6)
    df["season_type"] = df["time"].dt.month.apply(
        lambda x: "Mùa mưa" if 5 <= x <= 10 else "Mùa khô"
    )
    return df


# ── Data paths ───────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "evaporation"

# ── Load ─────────────────────────────────────────────────────────────────────
st.title("🌫️ Evaporation Analysis")

df_daily   = add_derived_cols(pd.read_parquet(f"{DATA_DIR}/daily.parquet"))
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

# Normalise column names produced by xr.Dataset.to_dataframe()
for df in (df_monthly, df_clim):
    df.reset_index(inplace=True, drop=False)

if "time" in df_monthly.columns:
    df_monthly["time"] = pd.to_datetime(df_monthly["time"])

df_daily_idx = df_daily.set_index("time")

st.markdown("---")



# Section 1 — Lượng bốc hơi thực tế và tiềm năng

st.subheader("1. Lượng bốc hơi thực tế và lượng bốc hơi tiềm năng")
st.markdown(
    "Lượng bốc hơi thực tế và tiềm năng trung bình theo tháng trong suốt 45 năm. "
    "Vùng được tô màu đại diện cho **thâm hụt bốc hơi** — khoảng cách giữa hai đường càng lớn "
    "càng cho thấy sự mất cân bằng độ ẩm nghiêm trọng trên bề mặt đất."
)

fig1 = go.Figure()

# Plot Actual Evaporation
fig1.add_trace(go.Scatter(
    x=MONTH_LABELS, y=df_clim["evap"],
    name="Lượng bốc hơi thực tế",
    mode="lines",
    line=dict(color="#2B1DB1", width=2)
))

# Plot Potential Evaporation with fill area to Actual
fig1.add_trace(go.Scatter(
    x=MONTH_LABELS, y=df_clim["pevap"],
    name="Lượng bốc hơi tiềm năng",
    mode="lines",
    line=dict(color="#D85A30", width=2),
    fill='tonexty',
    fillcolor='rgba(39, 13, 4, 0.15)'
))

# Dummy trace for "Thâm hụt bốc hơi" legend entry
fig1.add_trace(go.Scatter(
    x=[None], y=[None], mode='markers',
    marker=dict(size=10, color='rgba(39, 13, 4, 0.15)', symbol='square'),
    name="Thâm hụt bốc hơi"
))

fig1.update_layout(
    title="Lượng bốc hơi trung bình theo tháng và mức thâm hụt bốc hơi",
    xaxis_title="Tháng",
    yaxis_title="mm/ngày",
    hovermode="x unified",
    height=500
)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")



# Section 2 — Chu kỳ mùa và xu hướng dài hạn của EF

st.subheader("2. Chu kỳ mùa và xu hướng dài hạn của hệ số bốc hơi (EF)")
st.markdown(
    "**Bên trái:** Chu kỳ khí hậu của hệ số EF theo từng tháng (tỷ lệ giữa bốc hơi thực tế và tiềm năng).  \n"
    "**Bên phải:** Giá trị EF trung bình hàng năm cùng đường xu hướng tuyến tính — xu hướng giảm là dấu hiệu "
    "cho thấy bề mặt đất đang trở nên khô hạn hơn trong dài hạn."
)

annual_ef = df_daily_idx["ef"].resample("YE").mean()
years_ef  = annual_ef.index.year
ef_vals   = annual_ef.values
slope_ef, intercept_ef, r_ef, p_ef, _ = stats.linregress(years_ef, ef_vals)
trend_y_ef = slope_ef * years_ef + intercept_ef

fig2 = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("A. Chu kỳ về tỉ lệ EF qua các tháng hàng năm", 
                    "B. Dấu hiệu đất đai khô hạn về lâu dài"),
    column_widths=[0.45, 0.55]
)

# Left panel
fig2.add_trace(go.Scatter(
    x=MONTH_LABELS, y=df_clim["ef"], 
    mode="lines+markers", 
    line=dict(color="#2E7D32", width=2.5), 
    name="EF Tháng",
    showlegend=False
), row=1, col=1)

# Right panel
fig2.add_trace(go.Scatter(
    x=years_ef, y=ef_vals, 
    mode="markers", 
    marker=dict(size=8, color="#1B5E20", opacity=0.7), 
    name="EF Hàng năm",
    showlegend=False
), row=1, col=2)

fig2.add_trace(go.Scatter(
    x=years_ef, y=trend_y_ef, 
    mode="lines", 
    line=dict(color="#D32F2F", width=2), 
    name=f"Trend: {slope_ef:.5f}/year  (p={p_ef:.3f})"
), row=1, col=2)

fig2.update_layout(height=500, hovermode="x")
fig2.update_yaxes(title_text="EF (Bốc hơi thực tế / Bốc hơi tiềm năng)", range=[0, 1], row=1, col=1)
fig2.update_xaxes(title_text="Tháng", row=1, col=1)
fig2.update_yaxes(title_text="Chỉ số EF trung bình hàng năm", row=1, col=2)
fig2.update_xaxes(title_text="Năm", row=1, col=2)
st.plotly_chart(fig2, width="stretch")
st.markdown("---")



# Section 3 — Heatmap thâm hụt bốc hơi

st.subheader("3. Heatmap mức độ thâm hụt bốc hơi qua các tháng trong các năm")
st.markdown(
    "Dị thường thâm hụt bốc hơi hàng tháng (DEF = PET − ET). "
    "**Màu đỏ** thể hiện khí quyển đang thiếu hụt nước (water-stressed) nhiều hơn mức trung bình; "
    "**Màu xanh** thể hiện mức độ thâm hụt ít hơn."
)

df_monthly_def = df_daily_idx["def"].resample("MS").mean().to_frame()
df_monthly_def["month"] = df_monthly_def.index.month
df_monthly_def["year"]  = df_monthly_def.index.year
def_clim_map = df_monthly_def.groupby("month")["def"].mean()
df_monthly_def["def_anom"] = df_monthly_def.apply(
    lambda r: r["def"] - def_clim_map[r["month"]], axis=1
)
heatmap_data = df_monthly_def.pivot(index="year", columns="month", values="def_anom")

fig3 = go.Figure(data=go.Heatmap(
    z=heatmap_data.values,
    x=MONTH_LABELS,
    y=heatmap_data.index,
    colorscale="RdBu_r",
    zmid=0,
    colorbar=dict(title="Dị thường thâm hụt<br>(mm/ngày)")
))

fig3.update_layout(
    title="Heatmap dị thường trong sự thâm hụt bốc hơi<br><i>Màu đỏ nghĩa là khí quyển đang thiếu nước nhiều hơn bình thường</i>",
    xaxis_title="Tháng",
    yaxis_title="Năm",
    height=600
)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")



# Section 4 — Xu hướng lâu dài (Actual vs Potential)

st.subheader("4. Xu hướng lâu dài — Bốc hơi thực tế và tiềm năng")
st.markdown(
    "Giá trị trung bình hàng năm của bốc hơi thực tế (E) và bốc hơi tiềm năng (Ep) "
    "với các đường xu hướng tuyến tính — các tín hiệu phân kỳ cho thấy nhu cầu độ ẩm đang tăng."
)

df_annual = df_daily_idx[["evap", "pevap"]].resample("YE").mean()
years_ann  = df_annual.index.year

slope_act, intercept_act, _, p_act, _ = stats.linregress(years_ann, df_annual["evap"])
trend_y_act = slope_act * years_ann + intercept_act

slope_pot, intercept_pot, _, p_pot, _ = stats.linregress(years_ann, df_annual["pevap"])
trend_y_pot = slope_pot * years_ann + intercept_pot

fig4 = make_subplots(
    rows=1, cols=2, shared_xaxes=True,
    subplot_titles=("Bốc hơi thực tế (E)<br><i>(Lượng nước thực sự bay hơi khỏi bề mặt)</i>", 
                    "Bốc hơi tiềm năng (Ep)<br><i>(Nhu cầu của bầu khí quyển)</i>")
)

# Actual Evaporation Panel
fig4.add_trace(go.Scatter(
    x=years_ann, y=df_annual["evap"], 
    mode="markers", 
    marker=dict(color="#1976D2", opacity=0.6, size=8), 
    showlegend=False
), row=1, col=1)

fig4.add_trace(go.Scatter(
    x=years_ann, y=trend_y_act, 
    mode="lines", 
    line=dict(color="darkblue", width=2), 
    name=f"Trend: {slope_act*10:.4f} mm/ngày/thập kỉ<br>p={p_act:.3f}"
), row=1, col=1)

# Potential Evaporation Panel
fig4.add_trace(go.Scatter(
    x=years_ann, y=df_annual["pevap"], 
    mode="markers", 
    marker=dict(color="#E64A19", opacity=0.6, size=8), 
    showlegend=False
), row=1, col=2)

fig4.add_trace(go.Scatter(
    x=years_ann, y=trend_y_pot, 
    mode="lines", 
    line=dict(color="darkred", width=2), 
    name=f"Trend: {slope_pot*10:.4f} mm/ngày/thập kỉ<br>p={p_pot:.3f}"
), row=1, col=2)

fig4.update_layout(
    title="Sự phân kì giữa thực tế từ mặt đất và nhu cầu của bầu trời",
    height=500, hovermode="x"
)
fig4.update_xaxes(title_text="Năm", row=1, col=1)
fig4.update_xaxes(title_text="Năm", row=1, col=2)
fig4.update_yaxes(title_text="mm/ngày", row=1, col=1)
fig4.update_yaxes(title_text="mm/ngày", row=1, col=2)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")



# Section 5 — Mức độ căng thẳng của nước (Water Stress)

st.subheader("5. Mức độ căng thẳng của nước (Water Stress Ratio)")
st.markdown(
    "Chỉ số căng thẳng (Stress ratio) = Ep / E. Giá trị xấp xỉ 1 cho thấy độ ẩm bề mặt đầy đủ; "
    "giá trị càng lớn hơn 1 (>> 1) biểu thị tình trạng hạn hán nghiêm trọng."
)

clim_stress  = df_daily.groupby(df_daily["time"].dt.month)["stress_ratio"].mean()
annual_stress = df_daily_idx["stress_ratio"].resample("YE").mean()

slope_s, intercept_s, _, p_s, _ = stats.linregress(annual_stress.index.year, annual_stress.values)
trend_y_s = slope_s * annual_stress.index.year + intercept_s

fig5 = make_subplots(
    rows=1, cols=2, 
    subplot_titles=("A. Mức độ căng thẳng của nước<br><i>(Bốc hơi tiềm năng / Bốc hơi thực tế)</i>", 
                    "B. Xu hướng lâu dài")
)

fig5.add_trace(go.Scatter(
    x=MONTH_LABELS, y=clim_stress, 
    mode="lines+markers", 
    line=dict(color="purple", width=2.5), 
    showlegend=False
), row=1, col=1)

fig5.add_trace(go.Scatter(
    x=annual_stress.index.year, y=annual_stress.values, 
    mode="markers", 
    marker=dict(color="purple", opacity=0.6, size=8), 
    showlegend=False
), row=1, col=2)

fig5.add_trace(go.Scatter(
    x=annual_stress.index.year, y=trend_y_s, 
    mode="lines", 
    line=dict(color="purple", width=2), 
    name=f"Trend: {slope_s*10:.3f}/thập kỉ<br>p={p_s:.3f}"
), row=1, col=2)

fig5.update_layout(height=500, hovermode="x")
fig5.update_xaxes(title_text="Tháng", row=1, col=1)
fig5.update_yaxes(title_text="Mức độ căng thẳng", row=1, col=1)
fig5.update_xaxes(title_text="Năm", row=1, col=2)
st.plotly_chart(fig5, width="stretch")
st.markdown("---")



# Section 6 — So sánh phân bố theo từng thập kỉ (Violin)

st.subheader("6. So sánh sự phân bố theo từng thập kỉ")
st.markdown(
    "Biểu đồ Violin của lượng bốc hơi thực tế theo từng thập kỷ, được chia theo mùa mưa "
    "(tháng 5-10) và mùa khô (tháng 11-4). Phần 'đuôi' phía trên của biểu đồ thu hẹp dần "
    "trong mùa khô cho thấy khả năng bốc hơi của bề mặt đang suy giảm theo thời gian."
)

fig6 = go.Figure()

seasons = ["Mùa mưa", "Mùa khô"]
sides = ["negative", "positive"]
colors = ["#1f77b4", "#ff7f0e"]

for season, side, color in zip(seasons, sides, colors):
    df_season = df_daily[df_daily["season_type"] == season]
    fig6.add_trace(go.Violin(
        x=df_season["decade"].astype(str),
        y=df_season["evap"],
        legendgroup=season,
        scalegroup=season,
        name=season,
        side=side,
        line_color=color,
        box_visible=True,
        meanline_visible=True
    ))

fig6.update_layout(
    title="Sự dịch chuyển trong phân bố bốc hơi theo từng thập kỉ",
    xaxis_title="Thập kỉ",
    yaxis_title="Bốc hơi thực tế (mm/ngày)",
    violingap=0,
    violinmode='overlay',
    height=600
)
st.plotly_chart(fig6, width="stretch")