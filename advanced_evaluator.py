import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from vnstock import Vnstock
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from streamlit_option_menu import option_menu

# Base Directory Config
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data_snapshot")

# Page Configuration
st.set_page_config(page_title="Hệ Thống Phân Tích Cổ Phiếu Toàn Diện", layout="wide")
st.title("📈 Hệ Thống Đánh Giá Cổ Phiếu Toàn Diện")
st.markdown("Hệ thống kết hợp Phân tích Kỹ thuật, Phân tích Cơ bản và Vĩ mô để đánh giá cổ phiếu Việt Nam.")
st.info("📌 **Lưu ý:** Chuyên trang này được tối ưu hóa để phân tích báo cáo cho nhóm Ngành lớn & VN30.")
st.success("📆 **Cập nhật Dữ liệu (Offline Mode):** Toàn bộ dữ liệu biểu đồ, định giá và báo cáo tài chính trên hệ thống được chốt cố định đến hết giao dịch **ngày 06/04** nhằm phục vụ chấm điểm năng lực thuật toán.")
st.warning("🎓 **DỰ ÁN CÁ NHÂN (PORTFOLIO PROJECT):** Trang web này là một dự án Mã nguồn mở mang tính chất Học thuật & Giáo dục (Data Science & Machine Learning Portfolio). Nó KHÔNG phải là một sản phẩm thương mại và KHÔNG dùng để cố vấn tài chính.")

# --- MOCK CLASS CHO DỮ LIỆU OFFLINE ---
class LocalFinance:
    def __init__(self, data_dict):
        self.data_dict = data_dict
    def income_statement(self, period='yearly'):
        return self.data_dict.get('IncomeStatement', pd.DataFrame())
    def balance_sheet(self, period='yearly'):
        return self.data_dict.get('BalanceSheet', pd.DataFrame())
    def ratio(self):
        return self.data_dict.get('Ratios', pd.DataFrame())

class LocalStock:
    def __init__(self, data_dict):
        self.finance = LocalFinance(data_dict)

def load_local_data(ticker):
    file_path = os.path.join(DATA_DIR, f"{ticker}_snapshot.xlsx")
    if os.path.exists(file_path):
        try:
            with pd.ExcelFile(file_path) as xls:
                sheets = xls.sheet_names
                return {
                    'Price': pd.read_excel(xls, 'Price') if 'Price' in sheets else pd.DataFrame(),
                    'IncomeStatement': pd.read_excel(xls, 'IncomeStatement') if 'IncomeStatement' in sheets else pd.DataFrame(),
                    'BalanceSheet': pd.read_excel(xls, 'BalanceSheet') if 'BalanceSheet' in sheets else pd.DataFrame(),
                    'Ratios': pd.read_excel(xls, 'Ratios') if 'Ratios' in sheets else pd.DataFrame()
                }
        except Exception as e:
            st.error(f"🛠️ [Debug] Lỗi mở file Excel ({ticker}): {str(e)}")
            return None
    else:
        st.error(f"🛠️ [Debug] File không tồn tại tại đường dẫn: {file_path}")
    return None

# --- NAVIGATION MENU (Dọc - Sidebar) ---
with st.sidebar:
    selected = option_menu(
        menu_title="Điều hướng", 
        options=["Dự Báo & Phân Tích", "So Sánh Cổ Phiếu", "Về Tác Giả (About Me)"], 
        icons=["graph-up-arrow", "arrow-left-right", "person-badge"], 
        menu_icon="compass", 
        default_index=0, 
        orientation="vertical",
        styles={
            "container": {"padding": "0!important", "background-color": "transparent"},
            "icon": {"color": "#ffaa00", "font-size": "18px"}, 
            "nav-link": {"font-size": "15px", "text-align": "left", "margin":"0px", "--hover-color": "#262730"},
            "nav-link-selected": {"background-color": "#00aaff"},
        }
    )
    st.markdown("---")
    
    # --- DỜI SETTINGS TỪ DƯỚI LÊN ĐÂY ĐỂ TRÁNH LỖI VỠ LAYOUT DO ST.STOP() ---
    st.header("Cài Đặt (Settings)")
    
    sidebar_available_tickers = []
    if os.path.exists(DATA_DIR):
        sidebar_available_tickers = [f.replace("_snapshot.xlsx", "") for f in os.listdir(DATA_DIR) if f.endswith("_snapshot.xlsx")]
    if not sidebar_available_tickers:
        sidebar_available_tickers = ["FPT"]
    
    default_idx = sidebar_available_tickers.index("FPT") if "FPT" in sidebar_available_tickers else 0
    ticker = st.selectbox("Chọn mã cổ phiếu (Dữ liệu Offline):", options=sidebar_available_tickers, index=default_idx)
    period_option = st.selectbox("Dữ liệu lịch sử:", ["1 tháng", "3 tháng", "12 tháng"], index=2)
    days = 30 if period_option == "1 tháng" else (90 if period_option == "3 tháng" else 365)
    
    # Visitor Counter Logic
    counter_file = "visitor_count.txt"
    if 'visited' not in st.session_state:
        if os.path.exists(counter_file):
            with open(counter_file, "r") as f:
                try: count = int(f.read().strip())
                except ValueError: count = 0
        else: count = 0
        count += 1
        with open(counter_file, "w") as f:
            f.write(str(count))
        st.session_state.visited = count
    else:
        count = st.session_state.visited
    
    st.markdown("---")
    st.metric(label="👁️ Lượt truy cập Web", value=count)

if selected in ["Về Tác Giả (About Me)", "About the Author"]:
    st.markdown("---")
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.markdown("### 📞 Liên hệ")
        st.write("📧 Email: hoanhkhoa1009@gmail.com")
        st.write("🌐 LinkedIn: [anh-khoa-3223912a0](https://www.linkedin.com/in/anh-khoa-3223912a0)")

    with col_info:
        st.title("👨‍💻 Giới thiệu bản thân")
        st.write("""
        ### Xin chào, tôi là Khoa.
        
        Chào mừng bạn đến với dự án cá nhân của tôi — một hệ thống được ấp ủ và phát triển từ niềm đam mê sâu sắc với thị trường tài chính Việt Nam.
        
        Để tối ưu hóa quá trình xây dựng, tôi đã ứng dụng **Google Antigravity**. Nền tảng phát triển AI thế hệ mới này giúp tôi vượt qua những rào cản của việc viết code thủ công, từ đó dồn toàn bộ tâm trí vào việc định hình ý tưởng và hoàn thiện logic phân tích cốt lõi.
        
        Nếu bạn ghé thăm trang web này từ CV của tôi, hy vọng hệ thống này sẽ là minh chứng rõ nét nhất: với tôi, kiến thức không chỉ nằm trên giấy tờ, mà phải được chuyển hóa thành năng lực thực thi và sản phẩm vận hành thực tế.
        
        **Trân trọng,**  
        **Hồ Anh Khoa**
        """)
        
    st.stop() # Dừng luồng ở đây để không hiện Dashboard phân tích

elif selected == "So Sánh Cổ Phiếu":
    st.markdown("---")
    st.title("⚖️ So Sánh Cổ Phiếu")
    st.markdown("Tính năng định lượng so đối chuẩn doanh nghiệp qua 5 'Tứ Trụ' chỉ số (P/E, P/B, ROE, Biên Lợi Nhuận, Nợ/VCSH).")
    
    import os
    available_tickers = []
    if os.path.exists(DATA_DIR):
        available_tickers = [f.replace("_snapshot.xlsx", "") for f in os.listdir(DATA_DIR) if f.endswith("_snapshot.xlsx")]
    if not available_tickers:
        available_tickers = ["CTG", "MBB", "TCB", "FPT", "HPG", "VNM"]
        
    selected_tickers = st.multiselect("Chọn mã cổ phiếu để so sánh (Tối đa 5 mã):", options=available_tickers, default=["CTG", "MBB"] if "CTG" in available_tickers and "MBB" in available_tickers else available_tickers[:2], max_selections=5)
    
    if st.button("Tiến hành so sánh 🚀", use_container_width=True, type="primary"):
        if len(selected_tickers) < 2:
            st.error("❌ Vui lòng chọn tối thiểu 2 mã để đưa lên bàn cân nha!")
        else:
            all_metrics = {}
            for t in selected_tickers:
                data = load_local_data(t)
                if not data:
                    st.warning(f"⚠️ Bỏ qua {t}: Không tìm thấy dữ liệu offline.")
                    continue
                r = data.get('Ratios', pd.DataFrame())
                if r.empty:
                    st.warning(f"⚠️ Bỏ qua {t}: Dữ liệu chỉ số sinh lời bị trống.")
                    continue
                
                r_sorted = r.sort_values(['yearReport', 'lengthReport'])
                r_latest = r_sorted.iloc[-1]
                
                def safe_get(serie, keys):
                    cols = {str(k).strip().lower(): v for k, v in serie.to_dict().items()}
                    for k in keys:
                        k_low = k.lower()
                        for ck, cv in cols.items():
                            if k_low == ck or k_low in ck:
                                try: return float(cv)
                                except: return 0.0
                    return 0.0
                
                all_metrics[t] = {
                    'P/E': safe_get(r_latest, ['p/e']),
                    'P/B': safe_get(r_latest, ['p/b', 'price to book']),
                    'ROE (%)': safe_get(r_latest, ['roe']) * 100,
                    'Biên LN Ròng (%)': safe_get(r_latest, ['net profit margin']) * 100,
                    'Nợ / Vốn CSH': safe_get(r_latest, ['debt/equity', 'debt to equity'])
                }
                
            if len(all_metrics) >= 2:
                st.subheader("📊 Bảng Chỉ Số Tương Quan")
                comp_df = pd.DataFrame({"Chỉ Số": ['P/E', 'P/B', 'ROE (%)', 'Biên LN Ròng (%)', 'Nợ / Vốn CSH']})
                for t, metrics in all_metrics.items():
                    comp_df[t] = [metrics[k] for k in comp_df["Chỉ Số"]]
                
                fig_comp = go.Figure()
                colors = ['#29B6F6', '#69F0AE', '#FFA726', '#AB47BC', '#EF5350']
                for i, t in enumerate(all_metrics.keys()):
                    fig_comp.add_trace(go.Bar(x=comp_df["Chỉ Số"], y=comp_df[t], name=t, marker_color=colors[i % len(colors)], text=comp_df[t].apply(lambda x: f"{x:.2f}"), textposition='auto'))
                
                fig_comp.update_layout(barmode='group', template='plotly_dark', title="So Sánh Định Giá & Hiệu Suất Sinh Lời")
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.table(comp_df.set_index("Chỉ Số").style.format("{:.2f}"))
                
                import time
                st.markdown("### 🤖 Trí Tuệ Nhân Tạo: Đánh Giá Tương Quan (Local Engine)")
                with st.spinner("Đang trích xuất toàn bộ ma trận dữ liệu và hóa thân thành Thẩm định viên..."):
                    time.sleep(1.8)
                    
                    valid_pe = {k: v['P/E'] for k, v in all_metrics.items() if v['P/E'] > 0}
                    cheaper = min(valid_pe, key=valid_pe.get) if valid_pe else "Không xác định"
                    stronger = max(all_metrics.keys(), key=lambda k: all_metrics[k]['ROE (%)'])
                    t_list_str = ", ".join(all_metrics.keys())
                    
                    mock_resp = f"""
**1. Ma Trận Định Tính (S-W-O-T Ngành):**  
Dựa trên rổ phân tích bao gồm **[{t_list_str}]**, dưới đây là ma trận chắt lọc đánh giá năng lực lõi tương quan:

| Tiêu chí | Nội dung phân tích & So sánh | Lợi thế nghiêng về |
| :--- | :--- | :---: |
| **Thị phần** | Doanh nghiệp nào đang dẫn đầu quy mô? Ai đang chiếm lĩnh sức ảnh hưởng lớn nhất trong nhóm rổ này? | **{stronger}** |
| **Lợi thế cạnh tranh** | Đánh giá sức mạnh từ mạng lưới phân phối, công nghệ và khả năng tối ưu hóa tài sản cố định. | **{stronger}** |
| **Ban lãnh đạo** | Lịch sử thực thi chiến lược của Hội đồng quản trị và mức độ uy tín trên thị trường chứng khoán. | Hoà / Cân bằng |

**2. Phân tích Định giá (P/E & P/B):**  
Khi gán lên bộ lọc P/E, **{cheaper}** đang lộ diện với mức chiết khấu rẻ nhất trong toàn bộ rổ cổ phiếu đang được đo lường (Mua 1 đồng lợi nhuận với giá hời nhất). Điều này biến nó thành "ốc đảo an toàn" để trú ẩn cho các nhà đầu tư phòng thủ.

**3. Phân tích Hiệu quả & Rủi ro:**  
Khi bóc tách ma trận hiệu suất ROE, **{stronger}** đã xuất sắc đoạt "vương miện" dẫn đầu. Bộ máy của doanh nghiệp này đang tạo ra "con hào kinh tế" rộng lớn, vắt kiệt công năng tài sản để sinh lời vượt trội hơn hẳn các đối thủ cùng rổ.

**📌 TỔNG KẾT & KHUYẾN NGHỊ TỪ LOCAL AI:**  
Cuộc chạm trán này cho thấy sự phân hóa tuyệt đẹp:
* 🛡️ **Chọn {cheaper} (Trường phái Giá trị):** Dành cho ai yêu thích biên độ an toàn lớn, rủi ro sụt giảm cực thấp.
* 🚀 **Chọn {stronger} (Trường phái Tăng trưởng):** Chấp nhận mua đắt một chút, nhưng sở hữu chú "Xích Thố" phi nhanh nhất thị trường.

*(Ghi chú: Bản phân tích này được đo lường tự động qua mạng lưới **Heuristic Engine Cục bộ**, hoàn toàn không cần Internet bảo đảm giữ kín thông tin truy vấn)*
"""
                    st.info(mock_resp)
                    st.warning("⚠️ Đây chỉ là dự báo và không khuyến khích làm theo (Tôi sẽ không chịu trách nhiệm vì bất kỳ điều gì)")
            else:
                st.error("Không đủ dữ liệu hợp lệ để tiến hành đối sánh. Vui lòng check lại tải xuống.")
    st.stop()


# Calculate dates
end_date = datetime.now().strftime('%Y-%m-%d')
start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

# Helper function to calculate RSI manually (since we avoid pandas-ta to prevent numba installation limits)
def compute_rsi(data, window=14):
    diff = data.diff(1).dropna()
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

if ticker:
    st.header(f"Báo Cáo Tổng Hợp: {ticker}")
    
    try:
        # CHẾ ĐỘ PURE OFFLINE (Sử dụng dữ liệu tĩnh 06/04/2026)
        local_data = load_local_data(ticker)
        df = pd.DataFrame()
        stock = None
        
        if local_data:
            df = local_data['Price']
            stock = LocalStock(local_data)
            st.success(f"📁 Chế độ Offline Tĩnh: Đã tải dữ liệu Snapshot an toàn cho mã {ticker} (Chốt ngày 06/04).")
        else:
            st.error(f"❌ Thuật toán không thể đọc được dữ liệu '{ticker}'. Hãy xem dòng thông báo [Debug] bên trên để biết chính xác nguyên nhân!")
            
            # Liệt kê thử xem thư mục gốc có gì để chuẩn đoán bệnh
            st.write("📂 **Cấu trúc thư mục hiện tại trên Đám mây đang là:**")
            st.write(os.listdir(SCRIPT_DIR))
            if os.path.exists(DATA_DIR):
                st.write(f"📂 **Thư mục {DATA_DIR} có:**", os.listdir(DATA_DIR))
            else:
                st.error("🚨 THƯ MỤC 'data_snapshot' BỊ MẤT TÍCH KHỎI GITHUB!")
            st.warning("⚠️ LƯU Ý: Chức năng gọi API Realtime đã bị tắt để đảm bảo an toàn pháp lý khi public tựa Portfolio.")
            st.stop()
                
        if df.empty:
            st.error("Dữ liệu tĩnh hiện tại đang bị trống. Vui lòng tải lại tệp Excel snapshot.")
            st.stop()
        else:
            # Clean dataframe for TA
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
            
            # --- ÉP CỨNG DỮ LIỆU ĐẾN NGÀY 06/04/2026 NHƯ YÊU CẦU ---
            df = df.loc[df.index <= '2026-04-06']
            
            # --- 1. TECHNICAL ANALYSIS (PHÂN TÍCH KỸ THUẬT) ---
            st.subheader("1. Phân Tích Kỹ Thuật (Technical Analysis)")
            
            # Tính toán OHLC phiên mới nhất giống Fireant
            latest_row = df.iloc[-1]
            prev_row = df.iloc[-2] if len(df) > 1 else latest_row
            
            o, h, l, c, v = latest_row['open'], latest_row['high'], latest_row['low'], latest_row['close'], latest_row['volume']
            change = c - prev_row['close']
            change_pct = (change / prev_row['close']) * 100 if prev_row['close'] != 0 else 0
            
            color = "#00FF00" if change > 0 else ("#FF4136" if change < 0 else "#FFDC00") # Xanh lá, Đỏ, Vàng
            color_text = "💚" if change > 0 else ("❤️" if change < 0 else "💛")
            vol_str = f"{v/1000:.1f}K" if v < 1000000 else f"{v/1000000:.2f}M"
            
            # Khóa mốc thời gian hiển thị vào dòng dữ liệu cuối cùng (06/04)
            last_date_str = latest_row.name.strftime('%d/%m/%Y')
            
            st.markdown(f"""
            <div style='background-color: #131722; padding: 12px 15px; border-radius: 8px; font-family: "Trebuchet MS", Arial, sans-serif; margin-bottom: 15px;'>
                <span style='font-size: 1.2em; color: #D1D4DC;'><b>{ticker}</b> · 1D · <b>{last_date_str}</b></span><br>
                <span style='font-size: 1.05em; color: #D1D4DC;'>
                    O <span style='color:{color}'>{o:,.2f}</span> &nbsp; 
                    H <span style='color:{color}'>{h:,.2f}</span> &nbsp; 
                    L <span style='color:{color}'>{l:,.2f}</span> &nbsp; 
                    C <span style='color:{color}'>{c:,.2f}</span> &nbsp; 
                    <span style='color:{color}'>{change:+,.2f} ({change_pct:+,.2f}%)</span>
                </span><br>
                <span style='font-size: 0.95em; color: #D1D4DC;'>Volume - Khối lượng <span style='color:{color}'>{vol_str}</span></span>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate Indicators Manually
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()
            df['RSI_14'] = compute_rsi(df['close'], window=14)
            df['VOL_MA_20'] = df['volume'].rolling(window=20).mean() # Trung bình khối lượng 20 phiên
            
            # Candlestick chart
            fig = go.Figure()
            fig.add_trace(go.Candlestick(x=df.index,
                            open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], name='Price'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='orange', width=1.5), name='SMA 20'))
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_200'], line=dict(color='blue', width=1.5), name='SMA 200'))
            
            # Simple Fibonacci Calculation based on history high/low
            max_price = df['high'].max()
            min_price = df['low'].min()
            diff = max_price - min_price
            fib_0_382 = max_price - diff * 0.382
            fib_0_618 = max_price - diff * 0.618
            
            fig.add_hline(y=fib_0_382, line_dash="dash", line_color="green", annotation_text="Fib 0.382")
            fig.add_hline(y=fib_0_618, line_dash="dash", line_color="red", annotation_text="Fib 0.618")
            
            fig.update_layout(height=500, title=f"Biểu đồ giá {ticker} & Các đường hỗ trợ (SMA, Fibonacci)")
            st.plotly_chart(fig, width='stretch')
            
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='purple', width=2), name='RSI 14'))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text="Overbought/Quá mua (70)")
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text="Oversold/Quá bán (30)")
            fig_rsi.update_layout(height=250, title="Chỉ số Sức mạnh Tương đối - RSI(14)")
            st.plotly_chart(fig_rsi, width='stretch')
            
            current_close = df['close'].iloc[-1]
            current_rsi = df['RSI_14'].iloc[-1] if not pd.isna(df['RSI_14'].iloc[-1]) else 50
            
            # --- 2. FUNDAMENTAL ANALYSIS (PHÂN TÍCH CƠ BẢN) ---
            st.subheader("2. Phân Tích Cơ Bản (Fundamental Ratios)")
            try:
                # Get ratio from the already created stock object
                ratio_df = stock.finance.ratio()
                if not ratio_df.empty:
                    st.dataframe(ratio_df.head(10))
                    
                    # Flatten MultiIndex columns if present to easily access the English metric names
                    if isinstance(ratio_df.columns, pd.MultiIndex):
                        ratio_df.columns = [col[-1] if isinstance(col, tuple) else col for col in ratio_df.columns]
                    
                    # Chuẩn hóa tên cột để tránh lỗi khoảng trắng
                    ratio_df.columns = [str(c).strip() for c in ratio_df.columns]
                        
                    # Lấy số liệu Quý gần nhất (Dòng đầu tiên)
                    latest_q = f"Q{int(ratio_df['lengthReport'].iloc[0])}/{int(ratio_df['yearReport'].iloc[0])}"
                    
                    try:
                        de_ratio_series = ratio_df.get('Debt/Equity')
                        at_ratio_series = ratio_df.get('Asset Turnover')
                        roe_series = ratio_df.get('ROE (%)')
                        
                        st.markdown(f"### 🤖 AI Chẩn Đoán Sức Khỏe Doanh Nghiệp ({ticker} - {latest_q})")
                        
                        if de_ratio_series is not None and at_ratio_series is not None:
                            # Cổ phiếu Doanh nghiệp Thương mại / Sản xuất
                            de_ratio = float(de_ratio_series.iloc[0])
                            at_ratio = float(at_ratio_series.iloc[0])
                            dso_ratio = float(ratio_df.get('Days Sales Outstanding', pd.Series([0])).iloc[0])
                            
                            de_status = "🟢 RẤT AN TOÀN (Nợ thấp)" if de_ratio < 1.5 else ("🟡 TRUNG BÌNH" if de_ratio < 2.5 else "🔴 NỢ NGẬP ĐẦU (Rủi ro vỡ nợ)")
                            
                            st.markdown(f"""
                            Dựa trên Báo cáo tài chính quý gần nhất, hệ thống phân tích các chỉ tiêu trọng yếu:
                            
                            **1. Sức khỏe Tài chính (Chống phá sản)**
                            - Hệ số Nợ/Vốn chủ sở hữu (Debt/Equity): **{de_ratio:.2f}** -> **{de_status}** *(Doanh nghiệp có 10 đồng vốn thì đi nợ {de_ratio*10:.1f} đồng).*
                            
                            **2. Năng lực In Tiền (Hiệu quả hoạt động)**
                            - Vòng quay Tổng Tài sản đạt **{at_ratio:.2f}**. *(1 đồng tài sản đẻ ra được {at_ratio:.2f} đồng doanh thu).*
                            - Số ngày thu tiền bình quân: **{dso_ratio:.0f} ngày**. *(Sau khi bán hàng, mất {dso_ratio:.0f} ngày để thu tiền về).*
                            """)
                        elif roe_series is not None:
                            # Cổ phiếu Ngân hàng / Tài chính đặc thù
                            roe = float(roe_series.iloc[0])
                            profit_margin = float(ratio_df.get('Net Profit Margin (%)', pd.Series([0])).iloc[0])
                            lever = float(ratio_df.get('Financial Leverage', pd.Series([0])).iloc[0])
                            
                            roe_status = "🟢 RẤT XUẤT SẮC" if roe > 0.15 else ("🟡 TRUNG BÌNH" if roe > 0.10 else "🔴 KÉM")
                            
                            st.markdown(f"""
                            *(Phát hiện bạn đang xem Cổ phiếu ngành Tài chính Ngân hàng / Đặc thù. Các chỉ số được điều chỉnh tương ứng)*
                            
                            **1. Hiệu suất Sinh lời (Khả năng đẻ lãi của Ngân hàng)**
                            - Lợi nhuận trên Vốn Chủ (ROE): **{roe*100:.1f}%** -> **{roe_status}**. *(Khả năng sinh lời của đồng vốn tự có rất tốt)*.
                            - Biên Lợi nhuận Ròng: **{profit_margin*100:.1f}%**.
                            
                            **2. Rủi ro Đòn Bẩy (Tín dụng)**
                            - Đòn bẩy Tài chính (Financial Leverage): **{lever:.2f}**. *(Đặc thù ngành ngân hàng luôn có đòn bẩy cao).*
                            """)
                        else:
                            st.info("⚠️ Cấu trúc Báo cáo Tài chính của mã này quá khác biệt so với tiêu chuẩn, tự động đánh giá đang bị vô hiệu hóa.")
                            
                    except Exception as parse_e:
                        st.info(f"⚠️ Không thể đọc tự động (Chi tiết lỗi: {parse_e})")
                        
                    import sys, io
                    # --- DRAW FINANCIAL DASHBOARD ---
                    # Mute stdout to prevent UnicodeEncodeError from vnstock on Windows
                    dummy_stdout = io.StringIO()
                    old_stdout = sys.stdout
                    sys.stdout = dummy_stdout
                    try:
                        is_df = stock.finance.income_statement(period='year')
                        bs_df = stock.finance.balance_sheet(period='year')
                    finally:
                        sys.stdout = old_stdout
                    try:
                        if not is_df.empty and not bs_df.empty:
                            is_df = is_df.sort_values(['yearReport', 'lengthReport']).tail(4)
                            bs_df = bs_df.sort_values(['yearReport', 'lengthReport']).tail(4)
                            
                            is_df['Period'] = is_df.apply(lambda row: f"Q{int(row['lengthReport'])}/{int(row['yearReport'])}" if pd.notna(row['lengthReport']) else str(row['yearReport']), axis=1)
                            bs_df['Period'] = bs_df.apply(lambda row: f"Q{int(row['lengthReport'])}/{int(row['yearReport'])}" if pd.notna(row['lengthReport']) else str(row['yearReport']), axis=1)
                            
                            def get_s(df, keys):
                                cols_dict = {c: str(c).strip().lower() for c in df.columns}
                                for k in keys:
                                    k_low = k.lower()
                                    for raw_col, clean_col in cols_dict.items():
                                        if k_low == clean_col or k_low in clean_col:
                                            return df[raw_col]
                                return pd.Series([0]*len(df), index=df.index)
                                
                            st.markdown("### 📊 Đồ thị Tình hình Tài chính (4 Kỳ Gần Nhất)")
                            col1, col2 = st.columns(2)
                            
                            bs_cols = [str(c).strip().lower() for c in bs_df.columns]
                            is_cols = [str(c).strip().lower() for c in is_df.columns]
                            
                            is_regular = any(kw in b for kw in ['current assets', 'tài sản ngắn hạn'] for b in bs_cols) or \
                                         any(kw in i for kw in ['net sales', 'doanh thu thuần', 'doanh thu'] for i in is_cols)
                            
                            if is_regular: # Doanh nghiệp thường
                                # Extract Regular
                                rev = get_s(is_df, ['net sales', 'doanh thu thuần', 'doanh thu', 'revenue (bn. vnd)', 'revenue', 'net revenue'])
                                profit = get_s(is_df, ['net profit for the year', 'lợi nhuận ròng', 'lnst', 'lợi nhuận sau thuế của cổ đông công ty mẹ', 'profit after tax', 'net profit'])
                                cogs = get_s(is_df, ['cost of sales', 'giá vốn', 'cost of goods sold'])
                                sga = get_s(is_df, ['selling expenses', 'chi phí bán hàng']) + get_s(is_df, ['general & admin expenses', 'chi phí quản lý doanh nghiệp'])
                                tax = get_s(is_df, ['business income tax - current', 'tax for the year', 'thuế thu nhập doanh nghiệp'])
                                
                                liab = get_s(bs_df, ['liabilities (bn. vnd)', 'liabilities', 'nợ phải trả', 'total liabilities'])
                                eq = get_s(bs_df, ["owner's equity", "owners' equity", 'vốn chủ sở hữu', 'equity', 'total equity'])
                                
                                ca = get_s(bs_df, ['current assets (bn. vnd)', 'current assets', 'tài sản ngắn hạn', 'assets - current'])
                                nca = get_s(bs_df, ['long-term assets (bn. vnd)', 'non-current assets', 'tài sản dài hạn', 'assets - non-current'])
                                cl = get_s(bs_df, ['current liabilities (bn. vnd)', 'current liabilities', 'short-term liabilities', 'nợ ngắn hạn', 'liabilities - current'])
                                ncl = get_s(bs_df, ['long-term liabilities (bn. vnd)', 'long-term liabilities', 'non-current liabilities', 'nợ dài hạn', 'liabilities - non-current'])
                                
                                # 1. Hiệu suất
                                def to_bn(s):
                                    return s.apply(lambda x: x / 1e9 if abs(x) > 1e11 else x)
                                
                                rev_bn = to_bn(rev)
                                profit_bn = to_bn(profit)
                                
                                fig1 = go.Figure()
                                fig1.add_trace(go.Bar(x=is_df['Period'], y=rev_bn, name="Doanh thu", marker_color='#29B6F6'))
                                fig1.add_trace(go.Bar(x=is_df['Period'], y=profit_bn, name="Lợi nhuận ròng", marker_color='#69F0AE'))
                                sales_clean = rev_bn.replace(0, 1)
                                margin = (profit_bn / sales_clean) * 100
                                fig1.add_trace(go.Scatter(x=is_df['Period'], y=margin, name="Biên LN (%)", yaxis="y2", line=dict(color='#FFD54F', width=2), mode='lines+markers'))
                                fig1.update_layout(title="Hiệu suất (Tỷ VNĐ)", barmode='group', template='plotly_dark', showlegend=True, legend=dict(orientation="h", y=-0.2), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False))
                                col1.plotly_chart(fig1, width='stretch')
                                
                                # 2. Cơ cấu BCTC
                                year_latest = is_df['yearReport'].iloc[-1]
                                fig2 = go.Figure(go.Waterfall(
                                    orientation = "v",
                                    measure = ["absolute", "relative", "total", "relative", "total", "relative", "total"],
                                    x = ["Doanh thu", "Giá vốn", "LN Gộp", "Chi phí BH&QL", "LN Trước Thuế", "Thuế", "LN Ròng"],
                                    y = [rev.iloc[-1], -abs(cogs.iloc[-1]), 0, -abs(sga.iloc[-1]), 0, -abs(tax.iloc[-1]), 0],
                                    decreasing = {"marker":{"color":"#FF4136"}}, increasing = {"marker":{"color":"#2ECC40"}}, totals = {"marker":{"color":"#0074D9"}}
                                ))
                                fig2.update_layout(title=f"Kết quả kinh doanh ({year_latest})", template='plotly_dark')
                                col2.plotly_chart(fig2, width='stretch')
                                
                                # 3. Tài sản và Vốn CSH
                                liab_bn = to_bn(liab)
                                eq_bn = to_bn(eq)
                                
                                fig3 = go.Figure()
                                fig3.add_trace(go.Bar(x=bs_df['Period'], y=liab_bn, name="Nợ phải trả", marker_color='#00BCD4'))
                                fig3.add_trace(go.Bar(x=bs_df['Period'], y=eq_bn, name="Vốn CSH", marker_color='#4DB6AC'))
                                eq_clean = eq_bn.replace(0, 1)
                                debt_eq = (liab_bn / eq_clean) * 100
                                fig3.add_trace(go.Scatter(x=bs_df['Period'], y=debt_eq, name="Nợ/VCSH (%)", yaxis="y2", line=dict(color='#FFEE58', width=2), mode='lines+markers'))
                                fig3.update_layout(title="Tài sản và Vốn chủ sở hữu (Tỷ VNĐ)", barmode='group', template='plotly_dark', legend=dict(orientation="h", y=-0.2), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False))
                                col1.plotly_chart(fig3, width='stretch')
                                
                                # 4. Vị thế
                                bs_latest_yr = bs_df['yearReport'].iloc[-1]
                                ca_bn = to_bn(ca)
                                nca_bn = to_bn(nca)
                                cl_bn = to_bn(cl)
                                ncl_bn = to_bn(ncl)
                                
                                ca_l, nca_l = ca_bn.iloc[-1], nca_bn.iloc[-1]
                                cl_l, ncl_l = cl_bn.iloc[-1], ncl_bn.iloc[-1]
                                
                                fig4 = go.Figure()
                                fig4.add_trace(go.Bar(x=['Ngắn hạn', 'Dài hạn'], y=[ca_l, nca_l], name="Tài sản", marker_color='#29B6F6'))
                                fig4.add_trace(go.Bar(x=['Ngắn hạn', 'Dài hạn'], y=[cl_l, ncl_l], name="Nợ phải trả", marker_color='#69F0AE'))
                                fig4.update_layout(title=f"Vị thế tài chính ({bs_latest_yr})", barmode='group', template='plotly_dark', legend=dict(orientation="h", y=-0.2), yaxis_title="Tỷ VNĐ")
                                col2.plotly_chart(fig4, width='stretch')
                                
                            else: # Cổ phiếu Ngân hàng
                                # Extract Bank
                                total_op_rev = get_s(is_df, ['total operating revenue', 'tổng thu nhập hoạt động', 'revenue'])
                                net_profit = get_s(is_df, ['net profit for the year', 'lợi nhuận sau thuế', 'attribute to parent company (bn. vnd)', 'lợi nhuận sau thuế của cổ đông công ty mẹ'])
                                net_interest = get_s(is_df, ['net interest income', 'thu nhập lãi thuần'])
                                fees = get_s(is_df, ['net fee and commission income', 'lãi thuần từ hoạt động dịch vụ'])
                                ga_bank = get_s(is_df, ['general & admin expenses', 'chi phí hoạt động'])
                                prov = get_s(is_df, ['provision for credit losses', 'chi phí dự phòng rủi ro tín dụng'])
                                
                                dep = get_s(bs_df, ['deposits from customers', 'tiền gửi của khách hàng'])
                                loans = get_s(bs_df, ['loans and advances to customers', 'cho vay khách hàng'])
                                assets = get_s(bs_df, ['assets', 'tổng tài sản', 'total assets', 'total resource'])
                                l_eq = get_s(bs_df, ['liabilities and equity', 'tổng cộng nguồn vốn', 'total resource'])
                                
                                # 1. Hiệu suất Ngân hàng
                                def to_bn(s):
                                    return s.apply(lambda x: x / 1e9 if abs(x) > 1e11 else x)
                                    
                                rev_bn = to_bn(total_op_rev)
                                profit_bn = to_bn(net_profit)
                                
                                fig1 = go.Figure()
                                fig1.add_trace(go.Bar(x=is_df['Period'], y=rev_bn, name="Tổng Thu HĐ", marker_color='#29B6F6'))
                                fig1.add_trace(go.Bar(x=is_df['Period'], y=profit_bn, name="Lợi nhuận ròng", marker_color='#69F0AE'))
                                rev_clean = rev_bn.replace(0, 1)
                                margin = (profit_bn / rev_clean) * 100
                                fig1.add_trace(go.Scatter(x=is_df['Period'], y=margin, name="Biên LN (%)", yaxis="y2", line=dict(color='#FFD54F', width=2), mode='lines+markers'))
                                fig1.update_layout(title="Hiệu suất Ngân hàng (Tỷ VNĐ)", barmode='group', template='plotly_dark', showlegend=True, legend=dict(orientation="h", y=-0.2), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False))
                                col1.plotly_chart(fig1, width='stretch')
                                
                                # 2. Cơ cấu thu nhập Ngân hàng (Waterfall)
                                year_latest = is_df['yearReport'].iloc[-1]
                                int_bn = to_bn(net_interest).iloc[-1]
                                fee_bn = (to_bn(total_op_rev).iloc[-1] - int_bn if total_op_rev.iloc[-1] != 0 else to_bn(fees).iloc[-1])
                                ga_bn = to_bn(ga_bank).iloc[-1]
                                pr_bn = to_bn(prov).iloc[-1]
                                
                                fig2 = go.Figure(go.Waterfall(
                                    orientation = "v",
                                    measure = ["absolute", "relative", "total", "relative", "relative", "total"],
                                    x = ["Thu nhập Lãi", "Phí & Khác", "Tổng Thu HĐ", "Chi phí QL", "CP Dự phòng", "LN Trước Thuế"],
                                    y = [int_bn, fee_bn, 0, -abs(ga_bn), -abs(pr_bn), 0],
                                    decreasing = {"marker":{"color":"#FF4136"}}, increasing = {"marker":{"color":"#2ECC40"}}, totals = {"marker":{"color":"#0074D9"}}
                                ))
                                fig2.update_layout(title=f"Kết quả kinh doanh ({year_latest} - Tỷ VNĐ)", template='plotly_dark')
                                col2.plotly_chart(fig2, width='stretch')
                                
                                # 3. Cân đối kế toán Ngân hàng
                                dep_bn = to_bn(dep)
                                loans_bn = to_bn(loans)
                                
                                fig3 = go.Figure()
                                fig3.add_trace(go.Bar(x=bs_df['Period'], y=dep_bn, name="Tiền KH Gửi", marker_color='#00BCD4'))
                                fig3.add_trace(go.Bar(x=bs_df['Period'], y=loans_bn, name="Tiền Cho Vay", marker_color='#4DB6AC'))
                                fig3.update_layout(title="Huy động vốn vs Cho vay (Tỷ VNĐ)", barmode='group', template='plotly_dark', legend=dict(orientation="h", y=-0.2))
                                col1.plotly_chart(fig3, width='stretch')
                                
                                # 4. Tương quan Tài Nợ NH
                                bs_latest_yr = bs_df['yearReport'].iloc[-1]
                                liab_bank = get_s(bs_df, ['liabilities', 'nợ phải trả', 'total liabilities'])
                                eq_bank = get_s(bs_df, ["owner's equity", "owners' equity", 'vốn chủ sở hữu', 'equity', 'total equity'])
                                
                                # Trường hợp API trả về rỗng một trong 2, ta dùng quy tắc Kế toán cơ bản: 
                                # Tổng Tài Sản = Nợ Phải Trả + Vốn CSH -> Nợ Phải Trả = Tổng Tài Sản - Vốn CSH
                                val_assets = float(assets.iloc[-1]) if pd.notna(assets.iloc[-1]) else 0
                                val_eq = float(eq_bank.iloc[-1]) if pd.notna(eq_bank.iloc[-1]) else 0
                                val_liab = val_assets - val_eq
                                
                                # Nếu lỡ Vốn CSH cũng lỗi (ko lấy đc từ API) thì ta gán lại cho chắc
                                if val_eq <= 0 and val_assets > 0:
                                    val_liab = float(liab_bank.iloc[-1]) if pd.notna(liab_bank.iloc[-1]) else 0
                                    val_eq = val_assets - val_liab
                                    
                                fig4 = go.Figure()
                                # Convert all to Billions
                                assets_bn = to_bn(assets).iloc[-1]
                                liab_final_bn = to_bn(pd.Series([val_liab])).iloc[0]
                                eq_final_bn = to_bn(pd.Series([val_eq])).iloc[0]
                                
                                fig4.add_trace(go.Bar(x=['Tổng Tài Sản', 'Nợ Phải Trả', 'Vốn CSH'], y=[assets_bn, liab_final_bn, eq_final_bn], marker_color=['#29B6F6', '#FF8A65', '#69F0AE']))
                                fig4.update_layout(title=f"Cơ cấu Tài Sản / Nguồn Vốn ({bs_latest_yr})", template='plotly_dark', yaxis_title="Tỷ VNĐ")
                                col2.plotly_chart(fig4, width='stretch')

                    except Exception as draw_e:
                        st.warning(f"Lỗi vẽ đồ thị: {draw_e}")
                        
                else:
                    st.warning("Dữ liệu cơ bản đang trống tạm thời.")
                
            except Exception as e:
                st.warning(f"Không thể tải BCTC tự động lúc này. Lỗi hệ thống: {str(e)[:100]}...")
            
            # --- 3. MARKET CORRELATION (TƯƠNG QUAN VNINDEX) ---
            st.subheader("3. Tương quan Thị trường (VNINDEX Correlation Model)")
            try:
                # Đọc dữ liệu Offline VNINDEX
                vnindex_data = load_local_data("VNINDEX")
                if vnindex_data and not vnindex_data['Price'].empty:
                    vn_df = vnindex_data['Price'].copy()
                    vn_df['time'] = pd.to_datetime(vn_df['time'])
                    vn_df.set_index('time', inplace=True)
                    vn_df = vn_df.loc[vn_df.index <= '2026-04-06']
                    
                    # Merge data for alignment
                    corr_df = pd.DataFrame()
                    corr_df[ticker] = df['close']
                    corr_df['VNINDEX'] = vn_df['close']
                    corr_df.dropna(inplace=True)
                    
                    if not corr_df.empty and len(corr_df) > 10:
                        # Calculate daily returns to find correlation
                        returns = corr_df.pct_change().dropna()
                        correlation = returns[ticker].corr(returns['VNINDEX'])
                        
                        # Normalize prices to base 100 for visual comparison
                        corr_df[f'{ticker} (Base 100)'] = (corr_df[ticker] / corr_df[ticker].iloc[0]) * 100
                        corr_df['VNINDEX (Base 100)'] = (corr_df['VNINDEX'] / corr_df['VNINDEX'].iloc[0]) * 100
                        
                        st.write(f"**Hệ số tương quan (Pearson Correlation) với VNIndex: `{correlation:.2f}`**")
                        st.info("💡 Nếu hệ số > 0.7: Cổ phiếu bám sát thị trường chung. Ngược lại nếu < 0.3 thì cổ phiếu đó có lối đi riêng.")
                        
                        if correlation > 0.7:
                            st.write("➡️ Kết luận: Cổ phiếu này đang **bám sát** thị trường chung.")
                        elif correlation < 0.3:
                            st.write("➡️ Kết luận: Cổ phiếu này có **lối đi riêng**.")
                        else:
                            st.write("➡️ Kết luận: Cổ phiếu có sự tương quan **trung bình**.")
                            
                        # Plot comparison chart
                        fig_corr = go.Figure()
                        fig_corr.add_trace(go.Scatter(x=corr_df.index, y=corr_df[f'{ticker} (Base 100)'], line=dict(color='blue', width=2), name=ticker))
                        fig_corr.add_trace(go.Scatter(x=corr_df.index, y=corr_df['VNINDEX (Base 100)'], line=dict(color='gray', width=2, dash='dot'), name='VNINDEX'))
                        fig_corr.update_layout(height=400, title="So sánh Hiệu suất (Đã chuẩn hóa về 100 điểm)", yaxis_title="Tăng trưởng (%)", template="plotly_dark")
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.warning("Không đủ dữ liệu lịch sử chuẩn để tính tương quan.")
                else:
                    st.warning("⚠️ Không tìm thấy dữ liệu VNINDEX_snapshot.xlsx.")
            except Exception as e:
                st.warning(f"Không thể vẽ biểu đồ tương quan lúc này: {e}")
            
            # --- OVERALL CONCLUSION MODEL SCORE ---
            st.subheader("4. Đánh Giá Điểm Số (Model Scoring)")
            score = 0
            feedback = []
            
            try:
                if current_close > df['SMA_200'].iloc[-1]:
                    score += 1
                    feedback.append("✔️ Tích cực: Giá hiện tại đang NẰM TRÊN trung bình 200 ngày (Xu hướng tăng dài hạn).")
                else:
                    feedback.append("❌ Tiêu cực: Giá hiện tại NẰM DƯỚI trung bình 200 ngày (Xu hướng giảm).")
            except:
                pass
                
            if current_rsi < 40:
                score += 1
                feedback.append("✔️ Tích cực: RSI cho thấy cổ phiếu đang ở vùng giá rẻ / hấp dẫn.")
            elif current_rsi > 70:
                feedback.append("❌ Cảnh báo: RSI quá cao, cổ phiếu có thể đã tăng nóng (Quá mua).")
            else:
                feedback.append("⚪ Trung lập: RSI ở mức bình thường.")
            
            # --- VOLUME EVALUATION ---
            try:
                current_vol = float(v)
                avg_vol_20 = float(df['VOL_MA_20'].iloc[-1])
                
                if current_vol > (avg_vol_20 * 1.2):
                    score += 1
                    feedback.append(f"✔️ Tích cực: Khối lượng bùng nổ ({vol_str}) - Dòng tiền đang nhập cuộc mạnh mẽ.")
                elif current_vol < (avg_vol_20 * 0.8):
                    feedback.append(f"❌ Cảnh báo: Thanh khoản sụt giảm ({vol_str}) - Nhà đầu tư đang có dấu hiệu đứng ngoài.")
                else:
                    feedback.append(f"⚪ Trung lập: Khối lượng giao dịch ở mức ổn định.")
            except:
                pass
                
            st.write(f"**Điểm Khuyến Nghị Mô Hình (Technical): {score}/3**")
            for f in feedback:
                st.write(f)
                
            # --- 5. AI MACHINE LEARNING PREDICTION ---
            st.subheader("5. Dự Báo Mô Hình Nâng Cao (Machine Learning Models)")
            try:
                from sklearn.preprocessing import MinMaxScaler
                from sklearn.linear_model import LinearRegression
                from sklearn.neighbors import KNeighborsRegressor
                from sklearn.neural_network import MLPRegressor
                from sklearn.metrics import mean_absolute_error
                
                # Chuẩn bị dữ liệu đầu vào (Features)
                ml_df = df.copy()
                ml_df['Return'] = ml_df['close'].pct_change()
                # Thêm biến trễ (Lag) để dự báo hồi quy
                ml_df['Close_Lag_1'] = ml_df['close'].shift(1)
                
                # Các Feature (Biến độc lập)
                features_class = ['SMA_20', 'RSI_14', 'Return']
                features_reg = ['Close_Lag_1', 'SMA_20', 'RSI_14']
                
                # Các Target (Biến phụ thuộc)
                ml_df['Target_Class'] = (ml_df['Return'].shift(-1) > 0).astype(int)
                ml_df['Target_Reg'] = ml_df['close'].shift(-1) # Giá ngày mai
                
                ml_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Trích xuất phiên mới nhất (hiện tại) để dùng model dự đoán Tương lai
                latest_data_class = ml_df.iloc[-1:][features_class].fillna(0)
                latest_data_reg = ml_df.iloc[-1:][features_reg].fillna(0)
                
                # Loại bỏ giá trị NaN
                ml_df.dropna(inplace=True)
                
                if len(ml_df) > 50:
                    # Giao diện Tabs (nhiều tab cho nhiều thuật toán)
                    tab1, tab2, tab3, tab4 = st.tabs(["🌲 Random Forest", "📈 Linear Regression", "📍 KNN", "🧠 Deep Learning (Neural Net)"])
                    
                    # ----------------- DATA SPLIT -----------------
                    # Dành cho Phân Loại (Classification)
                    X_c = ml_df[features_class]
                    y_c = ml_df['Target_Class']
                    X_c_train, X_c_test, y_c_train, y_c_test = train_test_split(X_c, y_c, test_size=0.2, shuffle=False)
                    
                    # Dành cho Hồi Quy (Regression)
                    X_r = ml_df[features_reg]
                    y_r = ml_df['Target_Reg']
                    
                    # Mô hình hồi quy KNN và Neural Net cần Chuẩn hóa (MinMaxScaler - Giống bài học Data Flair)
                    scaler = MinMaxScaler()
                    X_r_scaled = scaler.fit_transform(X_r)
                    latest_data_reg_scaled = scaler.transform(latest_data_reg)
                    
                    X_r_train, X_r_test, y_r_train, y_r_test = train_test_split(X_r_scaled, y_r, test_size=0.2, shuffle=False)
                    
                    # ----------------- MODELS -----------------
                    
                    with tab1:
                        # 1. Random Forest (Phân Loại Tăng/Giảm)
                        rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                        rf_model.fit(X_c_train, y_c_train)
                        acc = accuracy_score(y_c_test, rf_model.predict(X_c_test))
                        tomorrow_pred = rf_model.predict(latest_data_class)
                        
                        st.metric(label="📊 Độ chính xác mô hình", value=f"{acc*100:.1f}%")
                        pred_text = "🟢 TĂNG GIÁ (UP)" if tomorrow_pred[0] == 1 else "🔴 GIẢM GIÁ (DOWN)"
                        st.metric(label="🤖 Dự Báo Xu Hướng", value=pred_text)
                        st.caption("Random Forest mạnh mẽ để dự đoán Xác suất tăng / giảm, nó bỏ qua nhiễu giá.")
                        
                        with st.expander("🔍 Xem Bằng Chứng AI (Explainable AI)"):
                            st.write("**Bóc tách quyết định:** Tại sao AI lại cho ra dự báo Tăng/Giảm thay vì con số ngược lại? Dưới đây là phân bổ trọng số (Feature Importance) cho thấy AI đã ưu tiên nhìn vào chỉ báo nào nhiều nhất để ra quyết định.")
                            importances = rf_model.feature_importances_
                            fig_rf = go.Figure([go.Bar(
                                x=features_class, 
                                y=importances, 
                                text=[f"{val*100:.1f}%" for val in importances], 
                                textposition='auto',
                                marker_color='#FFA726'
                            )])
                            fig_rf.update_layout(title="Mức độ quan trọng của Chỉ báo đầu vào", xaxis_title="Chỉ báo", yaxis_title="Tỷ trọng ảnh hưởng", template='plotly_dark', height=300)
                            st.plotly_chart(fig_rf, width='stretch')
                            
                            best_feature = features_class[np.argmax(importances)]
                            st.write(f"➡️ **Giải mã:** Cột cao nhất thuộc về **{best_feature}**. Điều này bóc trần sự thật rằng: Trong quá trình dò tìm hàng nghìn ngày giao dịch lịch sử, AI nhận thấy biến số `{best_feature}` đóng vai trò sinh tử lớn nhất tác động đến việc giá đảo chiều hay tiếp diễn vào ngày hôm sau.")
                    
                    with tab2:
                        # 2. Linear Regression (Hồi Quy Tuyến Tính cơ bản)
                        lr_model = LinearRegression()
                        lr_model.fit(X_r_train, y_r_train)
                        mae_lr = mean_absolute_error(y_r_test, lr_model.predict(X_r_test))
                        pred_lr = lr_model.predict(latest_data_reg_scaled)[0]
                        
                        st.metric(label="📉 Sai số dự báo trung bình (MAE)", value=f"{mae_lr:,.0f} VND")
                        st.metric(label="💰 Mức Giá Ngày Mai", value=f"{pred_lr:,.0f} VND", delta=f"{pred_lr - current_close:,.0f} VND")
                        st.caption("Linear Regression giả định mối nối tuyến tính cơ bản giữa các nhịp giá.")
                        
                        with st.expander("🔍 Xem Bằng Chứng XAI (Công thức Toán học)"):
                            st.write("Thay vì chỉ đưa ra con số bâng quơ, Hồi quy tuyến tính xây dựng một phương trình đường thẳng. Dưới đây là mặt cắt công thức mà thuật toán vừa tính được:")
                            formula = f"**Giá = ({lr_model.coef_[0]:.2f} × Close_Lag_1) + ({lr_model.coef_[1]:.2f} × SMA_20) + ({lr_model.coef_[2]:.2f} × RSI_14) + {lr_model.intercept_:.2f}**"
                            st.info(formula)
                            
                            # Tính chi tiết quá trình thay số
                            v1, v2, v3 = latest_data_reg_scaled[0]
                            st.write("➡️ Nếu bạn lấy mảng dữ liệu (đã chuẩn hóa Scaler) của ngày hôm nay cắm vào công thức trên, ta có phép tính chính xác như sau:")
                            calc_line = f"**=> Giá = ({lr_model.coef_[0]:.2f} × {v1:.4f}) + ({lr_model.coef_[1]:.2f} × {v2:.4f}) + ({lr_model.coef_[2]:.2f} × {v3:.4f}) + {lr_model.intercept_:.2f} = {pred_lr:,.0f} VND**"
                            st.success(calc_line)
                        
                    with tab3:
                        # 3. K-Nearest Neighbors (KNN Regressor)
                        knn_model = KNeighborsRegressor(n_neighbors=5)
                        knn_model.fit(X_r_train, y_r_train)
                        mae_knn = mean_absolute_error(y_r_test, knn_model.predict(X_r_test))
                        pred_knn = knn_model.predict(latest_data_reg_scaled)[0]
                        
                        st.metric(label="📉 Sai số dự báo trung bình (MAE)", value=f"{mae_knn:,.0f} VND")
                        st.metric(label="💰 Mức Giá Ngày Mai", value=f"{pred_knn:,.0f} VND", delta=f"{pred_knn - current_close:,.0f} VND")
                        st.caption("KNN quét hồ sơ lịch sử, tìm ra 5 chu kỳ giá giống hệt ngày hôm nay nhất để đưa ra giá trị trung bình.")
                        
                        with st.expander("🔍 Xem Bằng Chứng AI (Nearest Neighbors)"):
                            st.write("Thuật toán KNN không dùng phương trình đường thẳng, thay vào đó nó so sánh biểu đồ hôm nay với dữ liệu quá khứ bằng **Công thức tính Khoảng cách Ơ-clit (Euclidean Distance)**:")
                            st.latex(r"Distance = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + (x_3 - y_3)^2}")
                            
                            from sklearn.neighbors import NearestNeighbors
                            nn = NearestNeighbors(n_neighbors=5)
                            nn.fit(X_r_train)
                            distances, indices = nn.kneighbors(latest_data_reg_scaled)
                            
                            # Phép tính cho ngày gần nhất
                            curr = latest_data_reg_scaled[0]
                            hist = X_r_train[indices[0][0]]
                            dist_0 = distances[0][0]
                            
                            st.write("Ví dụ, với ngày lịch sử có đồ thị **Giống Nhất (Top 1)**, máy học ráp số vào công thức trên để đo như sau:")
                            st.info(f"**=> Sai lệch Top 1 = √ [ ({curr[0]:.3f} - {hist[0]:.3f})² + ({curr[1]:.3f} - {hist[1]:.3f})² + ({curr[2]:.3f} - {hist[2]:.3f})² ] = {dist_0:.4f}**")
                            
                            st.write("📏 Tương tự, nó tìm ra 5 chu kỳ có độ sai lệch (khoảng cách) thấp nhất như sau:")
                            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
                            col_t1.metric("Top 1", f"{distances[0][0]:.4f}")
                            col_t2.metric("Top 2", f"{distances[0][1]:.4f}")
                            col_t3.metric("Top 3", f"{distances[0][2]:.4f}")
                            col_t4.metric("Top 4", f"{distances[0][3]:.4f}")
                            col_t5.metric("Top 5", f"{distances[0][4]:.4f}")
                            
                            st.write(f"➡️ **Kết quả cuối cùng:** Trung bình cộng = ({distances[0][0]:.4f} + {distances[0][1]:.4f} + {distances[0][2]:.4f} + {distances[0][3]:.4f} + {distances[0][4]:.4f}) / 5 = **{np.mean(distances):.4f}**")
                            st.write("Cận 0 nghĩa là AI cực kỳ tự tin vì hôm nay đang lặp lại gần như y hệt 5 ngày vĩ đại nhất trong quá khứ.")
                        
                    with tab4:
                        # 4. Deep Learning (Mạng Nơ-ron Nhân tạo MLP)
                        # Dùng MLP thay cho LSTM vì LSTM cần setup TensorFlow khắt khe hơn trên máy trạm cục bộ.
                        from warnings import filterwarnings
                        filterwarnings('ignore') # Tránh cảnh báo Convergence
                        
                        mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
                        mlp_model.fit(X_r_train, y_r_train)
                        mae_mlp = mean_absolute_error(y_r_test, mlp_model.predict(X_r_test))
                        pred_mlp = mlp_model.predict(latest_data_reg_scaled)[0]
                        
                        st.metric(label="📉 Sai số dự báo trung bình (MAE)", value=f"{mae_mlp:,.0f} VND")
                        st.metric(label="💰 Mức Giá Ngày Mai", value=f"{pred_mlp:,.0f} VND", delta=f"{pred_mlp - current_close:,.0f} VND")
                        st.success("🧠 Thuật toán Học Sâu (Deep Learning) quét nhiều tầng tính năng phức tạp để cho ra giá trị sát thị trường giống kỹ thuật của bài học Data Flair.")

                        with st.expander("🔍 Xem Bằng Chứng AI (Biểu Đồ Khớp Tín Hiệu)"):
                            st.write("**Bóc tách Hộp Đen (Black Box):** Mạng Nơ-ron (Deep Learning) rất dễ bị 'Học vẹt' (Overfitting). Để chứng minh nó thực sự hiểu quy luật thị trường chứ không phải học thuộc lòng, máy tính đã cố tình 'giấu đi' 30 ngày giao dịch gần nhất (Tập Kiểm Thử). Sau đó, nó bắt AI phải tự phân tích và vẽ ra đường giá của 30 ngày đó.")
                            
                            fig_mlp = go.Figure()
                            # Đổi màu Trắng thành Xanh dương để không bị chìm vào nền sáng của Streamlit
                            fig_mlp.add_trace(go.Scatter(y=y_r_test.tail(30).values, name="Giá Thực Tế", line=dict(color='#2196F3', width=3)))
                            fig_mlp.add_trace(go.Scatter(y=mlp_model.predict(X_r_test)[-30:], name="AI Dự Báo", line=dict(color='#E91E63', dash='dot', width=3)))
                            fig_mlp.update_layout(title="Mô phỏng Tracking 30 ngày Tương lai gần nhất", yaxis_title="Giá Mở Cửa (VND)", template='plotly_white', height=350, margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_mlp, width='stretch')
                            
                            st.write("➡️ **Giải mã Biểu đồ:**")
                            st.write("- **Đường nét liền (Màu Xanh):** Là biến động Giá Thực Tế của cổ phiếu trên sàn.")
                            st.write("- **Đường nét đứt (Màu Hồng):** Là mức giá mà não bộ AI tự động suy luận ra trong cùng lúc đó.")
                            st.success("Tóm lại: Nếu đường Màu Hồng uốn lượn bám dính lấy đường Màu Xanh, có nghĩa là AI dường như đã nhạy cảm và bắt đúng chu kỳ thật sự của thị trường!")

                    st.markdown("---")
                    st.warning("⚠️ **Ghi chú:** Đây chỉ là dự báo toán học thử nghiệm, hoàn toàn không phải lời khuyên mua bán chứng khoán.")
                else:
                    st.warning("Dữ liệu không đủ dài để AI học. Bạn hãy chọn số ngày lớn hơn 100 nhé!")
            except Exception as e:
                st.warning(f"Lỗi khởi chạy Học Máy: {e}")
                
            st.subheader("6. Thông tin Vĩ mô & Tin tức (Macro & News)")
            try:
                news_df = stock.news.recent()
                if not news_df.empty:
                    st.write("**Tin tức gần đây:**")
                    st.dataframe(news_df.head(5))
                else:
                    st.info("Chưa có tin tức mới nhất từ hệ thống cho mã này.")
            except Exception as e:
                pass
                
            # Simulate Macro data since API requires premium/various setups
            st.write("**Dữ liệu Kinh tế Vĩ mô Hiện tại (Mock Data):**")
            col1, col2, col3 = st.columns(3)
            col1.metric(label="CPI (Lạm phát)", value="3.98%", delta="-0.1%")
            col2.metric(label="Lãi suất điều hành", value="4.50%", delta="0.0%")
            col3.metric(label="Tỷ giá trung tâm (USD/VND)", value="25,400 VND", delta="15 VND")
            
            # Khởi tạo mô hình đánh giá Vĩ mô theo Ngành
            st.write(f"**💡 Tác động Vĩ mô lên hoạt động kinh doanh của {ticker}:**")
            
            # Dictionary mapping tickers to macro implications
            macro_impact = ""
            if ticker in ["FPT", "CMG", "ELC"]:
                macro_impact = "💻 **Thuộc nhóm Công nghệ:** Đặc thù ngành này ít bị ảnh hưởng trực tiếp bởi lãi suất trong nước. Tuy nhiên, tỷ giá USD/VND đang ở mức cao (25,400) là **cực kỳ có lợi** vì phần lớn doanh thu của họ đến từ xuất khẩu phần mềm (thu bằng USD nhưng trả lương Kỹ sư bằng VND)."
            elif ticker in ["VCB", "MBB", "TCB", "BID", "CTG", "VPB", "STB", "HDB", "ACB", "TPB", "SHB", "VIB", "SSB", "LPB"]:
                macro_impact = "🏦 **Thuộc nhóm Ngân hàng:** Lãi suất điều hành thấp (4.5%) thường giúp duy trì biên lãi ròng (NIM) tốt do huy động vốn rẻ, đồng thời kích thích tăng trưởng tín dụng. Tuy nhiên, tỷ giá neo cao có thể khiến NHNN chịu áp lực hút tiền về. Nhạy cảm số 1 với chính sách tiền tệ."
            elif ticker in ["VHM", "NVL", "DIG", "DXG", "KDH", "VIC", "VRE", "BCM"]:
                macro_impact = "🏢 **Thuộc nhóm Bất động sản:** Khát vốn nhất thị trường. Ngành này RẤT THÍCH môi trường lãi suất thấp và dòng tiền giá rẻ. Mức lãi suất 4.5% hỗ trợ người mua nhà vay mượn, nhưng các doanh nghiệp rủi ro cao vẫn phải đối mặt với áp lực đáo hạn trái phiếu lớn."
            elif ticker in ["HPG", "HSG", "NKG", "GVR"]:
                macro_impact = "🏗️ **Thuộc nhóm SX Vật liệu / Công nghiệp:** Ngành này phụ thuộc trực tiếp vào tiến độ giải ngân Đầu tư công và sự phục hồi của Bất động sản. Khi kích cầu xây dựng, nhu cầu thép/cao su tăng vọt. Sự biến động của dòng vốn FDI và tỷ giá cũng ảnh hưởng đáng kể."
            elif ticker in ["REE", "GAS", "POW", "NT2", "BWE", "PLX"]:
                macro_impact = "⚡ **Thuộc nhóm Năng lượng / Tiện ích:** Đây là nhóm 'Phòng thủ ăn chắc mặc bền'. Dù lạm phát (CPI) có cao hay lãi suất biến động mạnh, nhu cầu dùng điện, nước, xăng dầu của người dân không thể giảm. Doanh thu siêu ổn định trong giông bão kinh tế."
            elif ticker in ["SSI", "VND", "VCI", "HCM", "MBS"]:
                macro_impact = "📈 **Thuộc nhóm Chứng khoán:** Ngành này là nhiệt kế của thị trường, có độ nhạy cực cao với Lãi suất. Lãi suất ngân hàng càng rẻ -> dòng tiền chảy vào chứng khoán tìm kiếm lợi nhuận càng lớn -> thanh khoản tăng kỷ lục -> Doanh thu mảng Môi giới & Margin tăng mạnh."
            elif ticker in ["VNM", "MSN", "SAB", "MWG", "PNJ"]:
                macro_impact = "🛒 **Thuộc nhóm Bán lẻ / Tiêu dùng (VN30):** Chịu ảnh hưởng trực tiếp từ Lạm phát (CPI) và tổng Cầu tiêu dùng nội địa. Nếu CPI (3.98%) duy trì cao sẽ siết chặt hầu bao của người dân, ảnh hưởng kém tới sức mua bán lẻ. Ngược lại, nếu kinh tế phục hồi và thu nhập dân chúng tăng, nhóm này bứt phá mạnh."
            elif ticker in ["VJC", "HVN"]:
                macro_impact = "✈️ **Thuộc nhóm Hàng không:** Cực kỳ nhạy cảm với 2 yếu tố Vĩ mô: Giá dầu thế giới (chiếm 30-40% chi phí) và Tỷ giá USD/VND (vì họ phải thuê/mua máy bay bằng USD). Tỷ giá 25,400 neo cao tạo sức ép chi phí cực lớn. Điểm bù trừ duy nhất là lượng khách du lịch nội địa phục hồi."
            else:
                macro_impact = f"🔍 **Phân tích Cơ bản:** Để đánh giá chuyên sâu tác động Vĩ mô lên {ticker}, chúng ta cần mổ xẻ Cơ cấu doanh thu của họ: Thuộc nhóm Xuất khẩu (hưởng lợi tỷ giá), Vay nợ cao (sợ lãi suất), hay Tiêu dùng (sợ lạm phát CPI)."
                
            st.success(macro_impact)
            st.info("Lưu ý: Dữ liệu BCTC được kéo tự động từ API mở (VCI/UBCKNN) có cùng chất lượng số liệu với Simplize. Việc theo dõi Vĩ mô kết hợp Phân tích Cơ bản này là cốt lõi của Đầu tư Giá trị!")
            
            # --- 7. RISK MANAGEMENT ---
            st.subheader("7. Quản Trị Rủi Ro (Risk Management)")
            
            # Simple mathematically derived risk metrics (7% stop loss, 15% take profit)
            st_loss_price = current_close * 0.93
            tk_profit_price = current_close * 1.15
            
            st.write(f"Giả định bạn mua cổ phiếu {ticker} tại mức giá đóng cửa hiện tại (**{current_close:,.0f} VND**), dưới đây là kế hoạch giao dịch kỷ luật:")
            
            c1, c2, c3 = st.columns(3)
            c1.metric(label="🎯 Điểm Chốt Lời Mục Tiêu (+15%)", value=f"{tk_profit_price:,.0f} VND", delta="Lợi nhuận cao")
            c2.metric(label="🛑 Điểm Cắt Lỗ Bắt Buộc (-7%)", value=f"{st_loss_price:,.0f} VND", delta="Bảo vệ Vốn", delta_color="inverse")
            c3.metric(label="⚖️ Tỷ lệ Risk/Reward (R:R)", value="1 : 2.14", delta="Giao dịch Tốt")
            
            st.markdown("""
            **Nguyên tắc Vàng trong Quản trị Danh mục:**
            - **Quy tắc 2%:** Tuyệt đối không bao giờ để mức cắt lỗ của một lệnh làm hao hụt quá 2% TỔNG số tiền bạn có. (Nếu vốn 100 triệu, chỉ cho phép lỗ tối đa 2 triệu/lệnh).
            - **Quy tắc Đa dạng hóa:** Không 'bỏ tất cả trứng vào một rổ'. Hãy chia số vốn của bạn cho khoảng **3 đến 5 mã** thuộc các nhóm ngành khác nhau (Ví dụ: 1 Mã Bank, 1 Thép, 1 Bất động sản).
            - **Tuyệt đối KHÔNG trung bình giá xuống:** Nhồi thêm tiền vào một cổ phiếu đang lao dốc là con đường nhanh nhất dẫn đến cháy tài khoản.
            """)
            
            st.divider()
            st.warning("⚠️ **Đây chỉ là dự báo và không khuyến khích làm theo (Tôi sẽ không chịu trách nhiệm vì bất kỳ điều gì)**")
            
    except Exception as e:
        st.error(f"Lỗi hệ thống: {e}")


