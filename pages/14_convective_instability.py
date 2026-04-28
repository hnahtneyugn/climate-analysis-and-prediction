"""
Page — Convective Instability Analysis
Parquet files expected:
  daily.parquet       — columns: cape, ki, tt, cii, severe, month, season, decade
  monthly.parquet     — columns: cape, ki, tt, cii, month, year, cii_anom, cape_anom
  yearly.parquet      — columns: cape, ki, tt, cii, severe
  climatology.parquet — columns: cape, ki, tt, cii  (index = month 1-12)
  raw.parquet         — columns: cape, ki, tt       (index = hourly timestamps)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Convective Instability", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "convective_instability"

st.title("Convective Instability Analysis")

df_daily   = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_yearly  = pd.read_parquet(f"{DATA_DIR}/yearly.parquet")
df_clim    = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")
df_raw     = pd.read_parquet(f"{DATA_DIR}/raw.parquet")

df_daily.index   = pd.to_datetime(df_daily.index)
df_monthly.index = pd.to_datetime(df_monthly.index)
df_yearly.index  = pd.to_datetime(df_yearly.index)
df_raw.index     = pd.to_datetime(df_raw.index)

st.markdown("---")


def minmax(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-9)


# Section 1 — Chu kỳ mùa của các chỉ số đối lưu

st.subheader("1. Chu kỳ mùa của các chỉ số đối lưu")
st.markdown(
    "Chu kỳ mùa đã chuẩn hóa (min-max) của các chỉ số CAPE, K-index và Total Totals. "
    "Sự đồng bộ giữa các chỉ số này đánh dấu giai đoạn chuyển tiếp của mùa gió mùa tại khu vực."
)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=MONTH_LABELS, y=minmax(df_clim["cape"]),
    name="CAPE (max)", line=dict(color="#D85A30", width=2)
))
fig1.add_trace(go.Scatter(
    x=MONTH_LABELS, y=minmax(df_clim["ki"]),
    name="K-index", line=dict(color="#185FA5", width=2)
))
fig1.add_trace(go.Scatter(
    x=MONTH_LABELS, y=minmax(df_clim["tt"]),
    name="Total Totals", line=dict(color="#7F77DD", width=2, dash="dash")
))
fig1.update_layout(
    title="Chu kỳ mùa của các chỉ số đối lưu (Chuẩn hóa 0-1)",
    xaxis_title="Tháng", yaxis_title="Chỉ số chuẩn hóa",
    hovermode="x unified", height=450
)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Xu hướng dài hạn

st.subheader("2. Xu hướng dài hạn của các chỉ số bất ổn định")
st.markdown(
    "Giá trị trung bình năm của các chỉ số CII, CAPE, K-index và Total Totals cùng với đường xu hướng tuyến tính. "
    "Sự gia tăng của các chỉ số này phản ánh bầu khí quyển đang trở nên bất ổn định hơn theo thời gian."
)

metrics = [
    ("cii",  "Composite CII",   "#2C2C2A"),
    ("cape", "CAPE (J/kg)",     "#D85A30"),
    ("ki",   "K-index",         "#185FA5"),
    ("tt",   "Total Totals",    "#7F77DD"),
]

fig2 = make_subplots(rows=2, cols=2,
                     subplot_titles=[m[1] for m in metrics])

for idx, (col, label, color) in enumerate(metrics):
    row, col_pos = divmod(idx, 2)
    row += 1; col_pos += 1

    y = df_yearly[col].values
    x = df_yearly.index.year.values
    slope, intercept, _, p, _ = stats.linregress(x, y)

    fig2.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=6, color=color, opacity=0.6),
        showlegend=False
    ), row=row, col=col_pos)

    fig2.add_trace(go.Scatter(
        x=x, y=intercept + slope * x,
        mode="lines", line=dict(color="red", width=1.5),
        name=f"Trend {label}: {slope*10:+.3f}/decade (p={p:.3f})"
    ), row=row, col=col_pos)

fig2.update_layout(height=600, hovermode="x")
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Heatmap CII anomaly

st.subheader("3. Ma trận dị thường chỉ số đối lưu tổng hợp (CII)")
st.markdown(
    "Nhận diện các tháng có độ bất ổn định cao bất thường (**màu đỏ**) hoặc thấp bất thường (**màu xanh**) so với trung bình lịch sử. "
    "Chỉ số CII cao thường đi kèm với các hiện tượng thời tiết cực đoan như dông sét mạnh."
)

pivot = df_monthly.pivot(index="year", columns="month", values="cii_anom")

fig3 = go.Figure(data=go.Heatmap(
    z=pivot.values,
    x=MONTH_LABELS,
    y=pivot.index,
    colorscale="RdBu_r",
    zmid=0,
    colorbar=dict(title="Dị thường CII")
))
fig3.update_layout(
    title="Ma trận dị thường chỉ số đối lưu tổng hợp (CII)",
    xaxis_title="Tháng", yaxis_title="Năm",
    height=600
)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Tần suất ngày đối lưu mạnh

st.subheader("4. Tần suất ngày đối lưu mạnh")
st.markdown(
    "Thống kê số lượng ngày trong năm có cường độ đối lưu mạnh (CII > 1). "
    "Đường trung bình trượt 5 năm giúp quan sát rõ hơn xu hướng biến động của các hiện tượng đối lưu cực đoan qua các thập kỷ."
)

annual_sev = df_daily["severe"].resample("YS").sum()
x_sev = annual_sev.index.year
y_sev = annual_sev.values
slope_s, _, _, p_s, _ = stats.linregress(x_sev, y_sev)
rolling_mean = pd.Series(y_sev).rolling(5, center=True).mean()

fig4 = go.Figure()
fig4.add_trace(go.Bar(
    x=x_sev, y=y_sev,
    name="Số ngày/năm", marker_color="#D85A30", opacity=0.6
))
fig4.add_trace(go.Scatter(
    x=x_sev, y=rolling_mean,
    name="Trung bình trượt 5 năm",
    line=dict(color="black", width=2)
))
fig4.update_layout(
    title=f"Tần suất ngày đối lưu mạnh — Trend: {slope_s*10:+.1f} d/decade (p={p_s:.3f})",
    xaxis_title="Năm", yaxis_title="Số ngày",
    hovermode="x unified", height=450
)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Cấu trúc nhiệt động lực học theo mùa (Hexbin → 2D histogram)

st.subheader("5. Cấu trúc nhiệt động lực học theo mùa")
st.markdown(
    "Biểu đồ mật độ 2D giữa chỉ số CAPE và K-index, phân chia theo các mùa khí tượng. "
    "Sự tập trung của các điểm dữ liệu giúp nhận diện 'vùng đặc trưng' của các trạng thái bất ổn định khí quyển trong từng mùa."
)

season_map = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3: "MAM", 4: "MAM", 5: "MAM",
    6: "JJA", 7: "JJA", 8: "JJA",
    9: "SON", 10: "SON", 11: "SON",
}
season_vn = {"DJF": "Đông", "MAM": "Xuân", "JJA": "Hè", "SON": "Thu"}

df_daily_copy = df_daily.copy()
df_daily_copy["season"] = df_daily_copy.index.month.map(season_map)

fig5 = make_subplots(rows=2, cols=2,
                     subplot_titles=[f"Mùa {season_vn[s]}" for s in ["DJF", "MAM", "JJA", "SON"]])

for idx, season in enumerate(["DJF", "MAM", "JJA", "SON"]):
    row, col_pos = divmod(idx, 2)
    row += 1; col_pos += 1
    sub = df_daily_copy[df_daily_copy["season"] == season]

    fig5.add_trace(go.Histogram2dContour(
        x=sub["cape"], y=sub["ki"],
        colorscale="YlOrRd",
        showscale=(idx == 3),
        contours=dict(coloring="heatmap"),
        name=season
    ), row=row, col=col_pos)

    fig5.update_xaxes(title_text="CAPE (J/kg)", row=row, col=col_pos)
    fig5.update_yaxes(title_text="K-index", row=row, col=col_pos)

fig5.update_layout(height=700, title="Cấu trúc nhiệt động lực học theo mùa (CAPE vs K-index)")
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — Phân tách đóng góp vào xu hướng CAPE

st.subheader("6. Phân tách đóng góp vào xu hướng CAPE")
st.markdown(
    "Phân tích ảnh hưởng của sự thay đổi nhiệt độ (T2m) và điểm sương (Td) đến biến động của CAPE. "
    "Biểu đồ cho thấy yếu tố nào (nhiệt hay ẩm) đóng vai trò chủ đạo trong việc gia tăng năng lượng đối lưu tại khu vực."
)

try:
    from sklearn.linear_model import LinearRegression
    df_t2m = pd.read_csv(DATA_DIR.parent / "df_monthly_t2m.csv", index_col=0, parse_dates=True)
    df_td  = pd.read_csv(DATA_DIR.parent / "df_monthly_td.csv",  index_col=0, parse_dates=True)

    df_reg = pd.DataFrame({
        "cape_anom": df_monthly["cape_anom"],
        "t2m_anom":  df_t2m["t2m_anom"],
        "td_anom":   df_td["td_anom"],
    }).dropna()

    model = LinearRegression().fit(df_reg[["t2m_anom", "td_anom"]], df_reg["cape_anom"])
    r2 = model.score(df_reg[["t2m_anom", "td_anom"]], df_reg["cape_anom"])

    fig6 = go.Figure(go.Bar(
        x=["Nhiệt độ (T2m)", "Điểm sương (Td)"],
        y=model.coef_,
        marker_color=["#D85A30", "#185FA5"]
    ))
    fig6.update_layout(
        title=f"Phân tách đóng góp vào xu hướng CAPE (R²={r2:.2f})",
        yaxis_title="Hệ số hồi quy (J/kg per K)",
        height=400
    )
    st.plotly_chart(fig6, width="stretch")
except Exception:
    st.info("Mục 6 tạm ẩn — Không tìm thấy tệp dữ liệu bổ trợ.")

st.markdown("---")


# Section 7 — Chu kỳ ngày đêm của CAPE theo mùa

st.subheader("7. Chu kỳ ngày đêm của CAPE theo mùa")
st.markdown(
    "Phân tích sự biến thiên của năng lượng đối lưu trong 24 giờ của một ngày. "
    "Đỉnh của CAPE thường xuất hiện vào thời điểm nhiệt độ bề mặt cao nhất, cung cấp năng lượng cho các cơn dông chiều tối."
)

season_colors = {"DJF": "#185FA5", "MAM": "#639922", "JJA": "#D85A30", "SON": "#BA7517"}
season_months_map = {"DJF": [12,1,2], "MAM": [3,4,5], "JJA": [6,7,8], "SON": [9,10,11]}

fig7 = go.Figure()

for s_name, s_months in season_months_map.items():
    subset = df_raw[df_raw.index.month.isin(s_months)]
    diurnal = subset["cape"].groupby(subset.index.hour).mean()
    fig7.add_trace(go.Scatter(
        x=diurnal.index, y=diurnal.values,
        name=season_vn[s_name],
        line=dict(color=season_colors[s_name], width=2)
    ))

fig7.update_layout(
    title="Chu kỳ ngày đêm của CAPE theo mùa",
    xaxis_title="Giờ (UTC)", yaxis_title="CAPE Trung bình (J/kg)",
    xaxis=dict(tickmode="linear", tick0=0, dtick=3),
    hovermode="x unified", height=450
)
st.plotly_chart(fig7, width="stretch")
st.markdown("---")


# Section 8 — Phân tích đuôi phân phối (Extreme Tail)

st.subheader("8. Phân tích đuôi phân phối và độ lệch (Extreme Tail)")
st.markdown(
    "**Phía trên:** Xu hướng của các giá trị CAPE cực đại (p99).  \n"
    "**Phía dưới:** Tỷ lệ p99/p50 giúp đánh giá liệu các hiện tượng cực đoan đang gia tăng nhanh hơn so với mức trung bình hay không."
)

p99 = df_daily["cape"].resample("YS").quantile(0.99)
p50 = df_daily["cape"].resample("YS").quantile(0.50)
skew_ratio = p99 / (p50 + 1e-9)
years_tail = p99.index.year

sl1, ic1, _, p1, _ = stats.linregress(years_tail, p99.values)
sl2, ic2, _, p2, _ = stats.linregress(years_tail, skew_ratio.values)

fig8 = make_subplots(rows=2, cols=1, shared_xaxes=True,
                     subplot_titles=[
                         f"Xu hướng p99 CAPE: {sl1*10:+.1f} J/kg/thập kỷ (p={p1:.3f})",
                         f"Tỷ lệ lệch p99/p50 — Trend: {sl2*10:+.4f}/thập kỷ"
                     ])

fig8.add_trace(go.Scatter(
    x=years_tail, y=p99.values, mode="markers",
    marker=dict(color="#D85A30"), name="p99 CAPE", showlegend=False
), row=1, col=1)
fig8.add_trace(go.Scatter(
    x=years_tail, y=ic1 + sl1 * years_tail,
    mode="lines", line=dict(color="black", dash="dash"), showlegend=False
), row=1, col=1)

fig8.add_trace(go.Scatter(
    x=years_tail, y=skew_ratio.values, mode="markers",
    marker=dict(color="#7F77DD"), name="p99/p50", showlegend=False
), row=2, col=1)
fig8.add_trace(go.Scatter(
    x=years_tail, y=ic2 + sl2 * years_tail,
    mode="lines", line=dict(color="black", dash="dash"), showlegend=False
), row=2, col=1)

fig8.update_xaxes(title_text="Năm", row=2, col=1)
fig8.update_yaxes(title_text="CAPE (J/kg)", row=1, col=1)
fig8.update_yaxes(title_text="Tỷ lệ p99/p50", row=2, col=1)
fig8.update_layout(height=600, hovermode="x")
st.plotly_chart(fig8, width="stretch")