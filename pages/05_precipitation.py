"""
Page — Precipitation Analysis
Parquet files: daily.parquet, monthly.parquet, yearly_sum.parquet, yearly_mean.parquet, climatology.parquet
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

st.set_page_config(page_title="Precipitation", layout="wide")

MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "precipitation"

st.title("Precipitation Analysis")

df_daily       = pd.read_parquet(f"{DATA_DIR}/daily.parquet")
df_monthly     = pd.read_parquet(f"{DATA_DIR}/monthly.parquet")
df_yearly_sum  = pd.read_parquet(f"{DATA_DIR}/yearly_sum.parquet")
df_yearly_mean = pd.read_parquet(f"{DATA_DIR}/yearly_mean.parquet")
clim           = pd.read_parquet(f"{DATA_DIR}/climatology.parquet")

df_daily.index       = pd.to_datetime(df_daily.index)
df_monthly.index     = pd.to_datetime(df_monthly.index)
df_yearly_sum.index  = pd.to_datetime(df_yearly_sum.index)
df_yearly_mean.index = pd.to_datetime(df_yearly_mean.index)

if "decade" not in df_daily.columns:
    df_daily["decade"] = (df_daily.index.year // 10) * 10

st.markdown("---")


# Section 1 — Seasonal climatology
st.subheader("1. Khí hậu học lượng mưa theo mùa")
st.markdown(
    "Phân tích lượng mưa trung bình hàng ngày theo tháng, được chia thành **Mưa tầng** (mưa diện rộng, kéo dài) và **Mưa đối lưu** (mưa rào mạnh, cục bộ). "
    "Sự phân hóa này giúp hiểu rõ cơ chế gây mưa tại Miền Trung, đặc biệt là sự đóng góp của bão và gió mùa."
)

fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["tp"],
    mode="lines", line=dict(color="black", width=2.5), name="Tổng lượng mưa (Total)"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["sp"],
    mode="lines", line=dict(color="#85B7EB", width=2, dash="dash"), name="Mưa tầng (Stratiform)"))
fig1.add_trace(go.Scatter(x=MONTH_LABELS, y=clim["cp"],
    mode="lines", line=dict(color="#D85A30", width=2, dash="dot"), name="Mưa đối lưu (Convective)"))
fig1.update_layout(title="Khí hậu học lượng mưa theo mùa",
    xaxis_title="Tháng", yaxis_title="Lượng mưa trung bình (mm/ngày)", height=500)
st.plotly_chart(fig1, width="stretch")
st.markdown("---")


# Section 2 — Annual total trend
st.subheader("2. Biến động lượng mưa tổng cộng hàng năm")
st.markdown(
    "Biểu đồ thể hiện tổng lượng mưa tích lũy qua từng năm. Đường trung bình trượt 5 năm và đường xu hướng tuyến tính "
    "giúp xác định liệu khu vực đang có xu hướng trở nên ẩm ướt hơn hay khô hạn hơn trong dài hạn."
)

x_years = df_yearly_sum.index.year
y_total = df_yearly_sum["tp"]
slope, intercept, _, _, _ = stats.linregress(x_years, y_total)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=x_years, y=y_total, marker_color="lightgreen", opacity=0.5, name="Lượng mưa năm"))
fig2.add_trace(go.Scatter(x=x_years, y=y_total.rolling(5, center=True).mean(),
    mode="lines", line=dict(color="green", width=2), name="Trung bình trượt 5 năm"))
fig2.add_trace(go.Scatter(x=x_years, y=intercept + slope * x_years,
    mode="lines", line=dict(color="red", dash="dash", width=2),
    name=f"Xu hướng: {slope*10:.1f} mm/thập kỷ"))
fig2.update_layout(title="Biến động lượng mưa tổng cộng hàng năm",
    xaxis_title="Năm", yaxis_title="Tổng lượng mưa (mm/năm)", height=500, hovermode="x unified")
st.plotly_chart(fig2, width="stretch")
st.markdown("---")


# Section 3 — Convective fraction trend
st.subheader("3. Xu hướng tỷ lệ mưa đối lưu (Convective Fraction)")
st.markdown(
    "Tỷ lệ mưa đối lưu (CF) phản ánh tính chất cực đoan của mưa. Xu hướng tăng của chỉ số này thường đi kèm với "
    "sự gia tăng các trận mưa rào cường độ mạnh trong thời gian ngắn, gây nguy cơ lũ quét và ngập úng đô thị cao hơn."
)

x_yr_m = df_yearly_mean.index.year
y_cf   = df_yearly_mean["cf"]
slope_cf, int_cf, _, p_cf, _ = stats.linregress(x_yr_m, y_cf)

fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x_yr_m, y=y_cf, mode="markers",
    marker=dict(color="steelblue", size=7, opacity=0.5), name="CF năm"))
fig3.add_trace(go.Scatter(x=x_yr_m, y=int_cf + slope_cf * x_yr_m,
    mode="lines", line=dict(color="red", dash="dash", width=2),
    name=f"Trend: {slope_cf*10:.4f}/thập kỷ (p={p_cf:.4f})"))
fig3.update_layout(title=f"Xu hướng tỷ lệ mưa đối lưu",
    xaxis_title="Năm", yaxis_title="Tỷ lệ mưa đối lưu (0-1)", height=500)
st.plotly_chart(fig3, width="stretch")
st.markdown("---")


# Section 4 — Monthly anomaly heatmap
st.subheader("4. Ma trận dị thường lượng mưa hàng tháng")
st.markdown(
    "Heatmap cho thấy sự sai lệch lượng mưa so với trung bình khí hậu. **Màu xanh dương** chỉ các tháng mưa nhiều bất thường (bão, lụt), "
    "trong khi **Màu nâu** chỉ các giai đoạn khô hạn nghiêm trọng."
)

pivot = df_monthly.pivot(index="year", columns="month", values="tp_anom")
fig4 = go.Figure(data=go.Heatmap(
    z=pivot.values, x=MONTH_LABELS, y=pivot.index,
    colorscale="BrBG", zmid=0,
    colorbar=dict(title="Dị thường (mm/ngày)")
))
fig4.update_layout(title="Ma trận dị thường lượng mưa hàng tháng",
    xaxis_title="Tháng", yaxis_title="Năm", height=600)
st.plotly_chart(fig4, width="stretch")
st.markdown("---")


# Section 5 — Convective vs stratiform stackplot
st.subheader("5. Phân hóa tính chất mưa: Đối lưu vs Tầng")
st.markdown(
    "Biểu đồ chồng minh họa sự đóng góp tương đối giữa mưa đối lưu và mưa tầng qua từng tháng trong chuỗi dữ liệu lịch sử. "
    "Giúp nhận diện sự thay đổi trong cấu trúc các hệ thống mây gây mưa qua các thập kỷ."
)

fig5 = go.Figure()
fig5.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly["cp"],
    mode="lines", line=dict(color="#D85A30", width=0),
    fill="tozeroy", fillcolor="#D85A30", name="Mưa đối lưu"))
fig5.add_trace(go.Scatter(x=df_monthly.index, y=df_monthly["cp"] + df_monthly["sp"],
    mode="lines", line=dict(color="#85B7EB", width=0),
    fill="tonexty", fillcolor="#85B7EB", name="Mưa tầng"))
fig5.update_layout(title="Phân hóa tính chất mưa: Đối lưu vs Tầng",
    xaxis_title="Năm", yaxis_title="mm/ngày", height=450)
st.plotly_chart(fig5, width="stretch")
st.markdown("---")


# Section 6 — Extreme rainfall days
st.subheader("6. Tần suất các ngày mưa cực đoan theo ngưỡng")
st.markdown(
    "Thống kê số lượng ngày trong năm có lượng mưa vượt các ngưỡng cường độ khác nhau (từ nhẹ đến rất to). "
    "Sự gia tăng số ngày mưa 'To' và 'Rất to' là chỉ dấu quan trọng cho sự biến động bất thường của thời tiết."
)

thresholds = {"Nhẹ (>1mm)": 1, "Vừa (>10mm)": 10, "To (>25mm)": 25, "Rất to (>40mm)": 40}
fig6 = make_subplots(rows=2, cols=2,
    subplot_titles=list(thresholds.keys()))

positions = [(1,1),(1,2),(2,1),(2,2)]
for (r, c), (label, thresh) in zip(positions, thresholds.items()):
    counts = (df_daily["tp"] > thresh).resample("YS").sum()
    fig6.add_trace(go.Bar(x=counts.index.year, y=counts.values,
        marker_color="steelblue", showlegend=False), row=r, col=c)

fig6.update_layout(height=700, title="Tần suất các ngày mưa cực đoan theo ngưỡng")
st.plotly_chart(fig6, width="stretch")
st.markdown("---")


# Section 7 — ECDF by decade
st.subheader("7. Sự dịch chuyển phân phối lượng mưa theo thập kỷ (ECDF)")
st.markdown(
    "Hàm phân phối tích lũy thực nghiệm (ECDF) cho thấy xác suất xuất hiện của các mức cường độ mưa khác nhau. "
    "Sự dịch chuyển của các đường cong qua từng thập kỷ giúp xác định xem các trận mưa lớn đang trở nên phổ biến hơn hay không."
)

fig7 = go.Figure()
colors_decade = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
decades = sorted(df_daily["decade"].unique())

for i, decade in enumerate(decades):
    subset = df_daily[df_daily["decade"] == decade]["tp"]
    vals = np.sort(subset[subset > 1].values)
    ecdf = np.arange(1, len(vals)+1) / len(vals)
    color = colors_decade[i % len(colors_decade)]
    fig7.add_trace(go.Scatter(x=vals, y=ecdf, mode="lines",
        line=dict(color=color), name=f"{decade}s"))

fig7.update_layout(title="Sự dịch chuyển phân phối lượng mưa theo thập kỷ (ECDF)",
    xaxis_title="Lượng mưa (mm/ngày) - Thang Log",
    yaxis_title="Xác suất tích lũy", xaxis_type="log", height=500)
st.plotly_chart(fig7, width="stretch")
st.markdown("---")


# Section 8 — Dry spell
st.subheader("8. Độ dài đợt khô hạn nhất trong năm (Dry Spell Length)")
st.markdown(
    "Theo dõi số ngày liên tiếp tối đa trong năm có lượng mưa dưới 1mm. "
    "Độ dài đợt khô hạn là một chỉ số sống còn đối với quản lý tài nguyên nước và lập kế hoạch sản xuất nông nghiệp."
)

def get_max_dry_spell(series):
    is_dry = series < 1.0
    groups = (is_dry != is_dry.shift()).cumsum()
    dry_groups = groups[is_dry]
    if dry_groups.empty:
        return 0
    return int(dry_groups.value_counts().max())

annual_dry = df_daily["tp"].groupby(pd.Grouper(freq="YS")).apply(get_max_dry_spell)

fig8 = go.Figure()
fig8.add_trace(go.Scatter(x=annual_dry.index.year, y=annual_dry.values,
    mode="lines+markers", marker=dict(symbol="circle", size=6),
    line=dict(color="brown"), name="Max dry spell"))
fig8.update_layout(title="Độ dài đợt khô hạn nhất trong năm (Dry Spell Length)",
    xaxis_title="Năm", yaxis_title="Số ngày liên tiếp mưa < 1mm", height=500)
st.plotly_chart(fig8, width="stretch")