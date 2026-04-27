import streamlit as st

# 1. Cấu hình chung cho toàn bộ App
st.set_page_config(page_title="Climate Analysis Dashboard", page_icon="🌍", layout="wide")

# 2. Định nghĩa các trang và Mapping tên hiển thị (Sidebar)
# Bạn có thể gom nhóm để sidebar gọn gàng hơn
pages = {
    "Tổng quan": [
        st.Page("app.py", title="Trang chủ", icon="🏠"),
    ],
    "Nhiệt độ & Độ ẩm": [
        st.Page("pages/01_air_n_surface_temp.py", title="Nhiệt độ Không khí & Bề mặt", icon="🌡️"),
        st.Page("pages/02_soil_temp.py", title="Nhiệt độ Đất", icon="🌱"),
        st.Page("pages/03_dewpts_n_humid.py", title="Điểm sương & Độ ẩm", icon="💧"),
    ],
    "Thủy văn & Mây": [
        st.Page("pages/05_precipitation.py", title="Lượng mưa", icon="🌧️"),
        st.Page("pages/06_evaporation.py", title="Bốc hơi", icon="💨"),
        st.Page("pages/07_soil_moisture.py", title="Độ ẩm Đất", icon="🏜️"),
        st.Page("pages/04_col_water_vapour_n_clouds.py", title="Hơi nước & Mây", icon="☁️"),
        st.Page("pages/13_cloud_cover.py", title="Độ che phủ Mây", icon="⛅"),
    ],
    "Năng lượng & Gió": [
        st.Page("pages/09_solar_radiation.py", title="Bức xạ Mặt trời", icon="☀️"),
        st.Page("pages/10_surface_heat_fluxes.py", title="Dòng nhiệt Bề mặt", icon="🔥"),
        st.Page("pages/15_wind.py", title="Phân tích Gió", icon="🌬️"),
    ],
    "Thông số khác": [
        st.Page("pages/08_runoff_drainage.py", title="Dòng chảy & Thoát nước", icon="🌊"),
        st.Page("pages/11_vegetation_structure.py", title="Cấu trúc Thảm thực vật", icon="🌲"),
        st.Page("pages/14_convective_instability.py", title="Bất ổn định Đối lưu", icon="⚡"),
        st.Page("pages/17_moisture_divergence.py", title="Phân kỳ Độ ẩm", icon="🌀"),
        st.Page("pages/18_sea_temp_wave_height.py", title="Nhiệt độ Biển & Sóng", icon="⚓"),
    ]
}

# 3. Khởi tạo Navigation
pg = st.navigation(pages)
pg.run()

# 4. Nội dung hiển thị riêng cho trang chủ (File app.py hiện tại)
# Kiểm tra nếu trang hiện tại đang là app.py (Trang chủ)
import os
current_file = pg.title
if current_file == "Trang chủ":
    st.title("🌍 Climate Analysis Dashboard")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📌 Giới thiệu dự án")
        st.write("""
        Hệ thống phân tích dữ liệu khí hậu khu vực Miền Trung Việt Nam giai đoạn 1980 - 2024.
        Dashboard này giúp theo dõi các biến đổi quan trọng về nhiệt độ, lượng mưa, bức xạ và các chỉ số thủy văn phức tạp khác.
        """)
        
        st.info("💡 **Hướng dẫn:** Hãy sử dụng menu bên trái để khám phá từng nhóm chỉ số khí hậu cụ thể.")
    
    with col2:
        st.header("📊 Thông tin dữ liệu")
        st.write("- **Nguồn:** ERA5 / NASA")
        st.write("- **Khu vực:** Miền Trung VN")
        st.write("- **Tần suất:** Hàng tháng/năm")

    st.divider()
    st.caption("Ứng dụng được phát triển bằng Streamlit & Python")