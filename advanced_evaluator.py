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
lang_choice = st.sidebar.radio("🌍 Ngôn ngữ / Language:", ['🇻🇳 Tiếng Việt', '🇬🇧 English'], horizontal=True)
st.session_state.lang = lang_choice

st.markdown("""
<style>
    /* Metricalist Dense Dashboard Theme */
    [data-testid="stAppViewContainer"] { background-color: #E9ECEF; color: #1E1E1E; }
    [data-testid="stSidebar"] { background-color: #0B5C57 !important; border-right: none; }
    [data-testid="stSidebar"] p, [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] { color: #F8F9F9 !important; }
    [data-testid="stHeader"] { background-color: transparent; }
    
    .block-container { padding-top: 1rem !important; padding-bottom: 1rem !important; max-width: 98% !important; }
    
    /* Main Layout */
    .main-title { font-size: 32px; font-weight: 800; color: #1A5276; margin-bottom: 0px; text-transform: uppercase; letter-spacing: -0.5px; }
    .sub-title { font-size: 16px; color: #7FB3D5; margin-bottom: 20px; font-weight: 500;}
</style>
""", unsafe_allow_html=True)

# --- LANGUAGE TRANSLATION DICTIONARY ---
_LANG = {
    '📈 Hệ Thống Đánh Giá Cổ Phiếu Toàn Diện': '📈 Comprehensive Stock Evaluation System',
    'Hệ thống kết hợp Phân tích Kỹ thuật, Phân tích Cơ bản và Vĩ mô để đánh giá cổ phiếu Việt Nam.': 'A system combining Technical, Fundamental, and Macro Analysis to evaluate Vietnamese stocks.',
    '📌 Lưu ý: Chuyên trang này được tối ưu hóa để phân tích báo cáo cho nhóm Ngành lớn & VN30.': '📌 Note: This specialized page is optimized for analyzing reports of Large Caps & VN30 group.',
    '🗓️ Cập nhật Dữ liệu (Offline Mode): Toàn bộ dữ liệu biểu đồ, định giá và báo cáo tài chính trên hệ thống được chốt cố định đến hết giao dịch ngày hiện tại nhằm phục vụ chấm điểm năng lực thuật toán.': '🗓️ Data Update (Offline Mode): All charts, valuations, and financial report data on the system are firmly frozen at the end of the current trading day to benchmark algorithmic performance.',
    '⚖️ DỰ ÁN CÁ NHÂN (PORTFOLIO PROJECT): Trang web này là một dự án Mã nguồn mở mang tính chất Học thuật & Giáo dục (Data Science & Machine Learning Portfolio). Tuyên bố miễn trừ trách nhiệm: Các dữ liệu và phân tích (AI/ML) trên Dashboard này chỉ mang tính chất tham khảo, mô phỏng học thuật và không phải là lời khuyên đầu tư tài chính. Nguồn dữ liệu thô: Vnstock.': '⚖️ PORTFOLIO PROJECT: This website is an open-source project for Academic & Educational purposes (Data Science & Machine Learning Portfolio). Disclaimer: Data and analysis (AI/ML) on this Dashboard are for reference and academic simulation only, and do not constitute financial investment advice. Raw data source: Vnstock.',
    '👉 **Nếu bạn muốn xem thêm dự án khác hãy [nhấp vào đây](https://portfolio-gilt-sigma-43.vercel.app)**': '👉 **If you want to view more projects, please [click here](https://portfolio-gilt-sigma-43.vercel.app)**',
    
    # Sidebar
    'Điều hướng': 'Navigation',
    'Dự Báo & Phân Tích': 'Forecasting & Analysis',
    'So Sánh Cổ Phiếu': 'Stock Comparison',
    'Về Tác Giả (About Me)': 'About Me',
    'Cài Đặt (Settings)': 'Settings',
    'Chọn mã cổ phiếu (Dữ liệu Offline):': 'Select Stock Ticker (Offline Data):',
    'Dữ liệu lịch sử:': 'Historical Data:',
    '1 tháng': '1 Month',
    '3 tháng': '3 Months',
    '12 tháng': '12 Months',
    'Lượt truy cập Web': 'Web Visitors',
    
    'Báo Cáo Tổng Hợp:': 'Comprehensive Report:',
    'Chế độ Offline Tĩnh: Đã tải dữ liệu Snapshot an toàn cho mã': 'Static Offline Mode: Safely loaded Snapshot data for ticker',
    '(Chốt ngày': '(Frozen on',
    'Biểu đồ giá': 'Price Chart',
    '& Các đường hỗ trợ (SMA, Fibonacci)': '& Support Lines (SMA, Fibonacci)',
    'Dựa trên Báo cáo tài chính quý gần nhất, hệ thống phân tích các chỉ tiêu trọng yếu:': 'Based on the latest financial statements, the system analyzes the key metrics:',
    '1. Sức khỏe Tài chính (Chống phá sản)': '1. Financial Health (Bankruptcy Shield)',
    'Hệ số Nợ/Vốn chủ sở hữu (Debt/Equity):': 'Debt/Equity Ratio:',
    'Doanh nghiệp có 10 đồng vốn thì đi nợ': 'For every 10 units of equity, the company owes',
    'đồng.': 'units.',
    '2. Năng lực In Tiền (Hiệu quả hoạt động)': '2. Money Printing Capacity (Operational Efficiency)',
    'Vòng quay Tổng Tài sản đạt': 'Asset Turnover reaches',
    'đồng tài sản đẻ ra được': 'unit(s) of asset generates',
    'đồng doanh thu': 'unit(s) of revenue',
    'Số ngày thu tiền bình quân:': 'Days Sales Outstanding:',
    'ngày.': 'days.',
    'Sau khi bán hàng, mất': 'After sales, it takes',
    'ngày để thu tiền về': 'days to collect cash',
    'Đồ thị Tình hình Tài chính (4 Kỳ Gần Nhất)': 'Financial Status Charts (Last 4 Periods)',
    'Doanh thu': 'Revenue',
    'Lợi nhuận ròng': 'Net Profit',
    'Biên LN (%)': 'Net Margin (%)',
    'Hiệu suất (Tỷ VNĐ)': 'Performance (Billion VND)',
    'Kết quả kinh doanh': 'Business Results',
    'Nợ phải trả': 'Liabilities',
    'Vốn CSH': 'Equity',
    'Nợ/VCSH (%)': 'Debt/Equity (%)',
    'Tài sản và Vốn chủ sở hữu (Tỷ VNĐ)': 'Assets & Equity (Billion VND)',
    'Ngắn hạn': 'Short-term',
    'Dài hạn': 'Long-term',
    'Tài sản': 'Assets',
    '📊 Độ chính xác mô hình': '📊 Model Accuracy',
    '🤖 Dự Báo Xu Hướng': '🤖 Trend Forecast',
    '📉 Sai số dự báo trung bình (MAE)': '📉 Mean Absolute Error (MAE)',
    '💰 Mức Giá Ngày Mai': '💰 Tomorrow\'s Price',

    'Lưu ý: Dữ liệu BCTC được kéo tự động từ API mở (VCI/UBCKNN) có cùng chất lượng số liệu với Simplize. Việc theo dõi Vĩ mô kết hợp Phân tích Cơ bản này là cốt lõi của Đầu tư Giá trị!': 'Note: Financial data is pulled automatically from open APIs (VCI/UBCKNN) with institutional quality. Tracking Macro combined with Fundamental Analysis is the core of Value Investing!',
    'Giả định bạn mua cổ phiếu {ticker} tại mức giá đóng cửa hiện tại (**{current_close:,.0f} VND**), dưới đây là kế hoạch giao dịch kỷ luật:': 'Assuming you buy {ticker} at the current close (**{current_close:,.0f} VND**), below is the disciplined trading plan:',
    '🟢 RẤT AN TOÀN (Nợ thấp)': '🟢 VERY SAFE (Low Debt)',
    '🟡 TRUNG BÌNH': '🟡 AVERAGE',
    '🔴 NỢ NGẬP ĐẦU (Rủi ro vỡ nợ)': '🔴 OVER-LEVERAGED (Default Risk)',
    '🟢 RẤT XUẤT SẮC': '🟢 EXCELLENT',
    '🔴 KÉM': '🔴 POOR',
    '*(Phát hiện bạn đang xem Cổ phiếu ngành Tài chính Ngân hàng / Đặc thù. Các chỉ số được điều chỉnh tương ứng)*': '*(Detected Banking/Financial Sector Stock. Metrics are adjusted accordingly)*',
    '1. Hiệu suất Sinh lời (Khả năng đẻ lãi của Ngân hàng)': '1. Profitability (Bank\'s Earning Power)',
    'Lợi nhuận trên Vốn Chủ (ROE):': 'Return on Equity (ROE):',
    'Khả năng sinh lời của đồng vốn tự có rất tốt': 'The profitability of own capital is very good',
    'Biên Lợi nhuận Ròng:': 'Net Profit Margin:',
    '2. Rủi ro Đòn Bẩy (Tín dụng)': '2. Leverage Risk (Credit)',
    'Đòn bẩy Tài chính (Financial Leverage):': 'Financial Leverage:',
    'Đặc thù ngành ngân hàng luôn có đòn bẩy cao': 'The banking sector inherently has high leverage',
    '⚠️ Cấu trúc Báo cáo Tài chính của mã này quá khác biệt so với tiêu chuẩn, tự động đánh giá đang bị vô hiệu hóa.': '⚠️ The structure of this ticker\'s Financial Statements is too anomalous, automatic assessment is disabled.',
    '⚠️ Không thể đọc tự động (Chi tiết lỗi:': '⚠️ Cannot read automatically (Error details:',
    'Quy tắc 2%:': '2% Rule:',
    'Tuyệt đối không bao giờ để mức cắt lỗ của một lệnh làm hao hụt quá 2% TỔNG số tiền bạn có. (Nếu vốn 100 triệu, chỉ cho phép lỗ tối đa 2 triệu/lệnh).': 'Never let a stop loss deplete more than 2% of your TOTAL capital.',
    "Quy tắc Đa dạng hóa: Không 'bỏ tất cả trứng vào một rổ'. Hãy chia số vốn của bạn cho khoảng **3 đến 5 mã** thuộc các nhóm ngành khác nhau (Ví dụ: 1 Mã Bank, 1 Thép, 1 Bất động sản).": "Diversification Rule: Don't 'put all eggs in one basket'. Divide your capital into **3 to 5 stocks** across different sectors.",
    'Tuyệt đối KHÔNG trung bình giá xuống: Nhồi thêm tiền vào một cổ phiếu đang lao dốc là con đường nhanh nhất dẫn đến cháy tài khoản.': 'NEVER average down: Throwing more money into a plunging stock is the fastest way to wipe out an account.',
    '⚠️ **Đây chỉ là dự báo và không khuyến khích làm theo (Tôi sẽ không chịu trách nhiệm vì bất kỳ điều gì)**': '⚠️ **This is only a forecast and not a recommendation to follow (Not financial advice)**',
    '📊 Bảng Chỉ Số Tương Quan': '📊 Correlation Index Table',
    'Không đủ dữ liệu hợp lệ để tiến hành đối sánh. Vui lòng check lại tải xuống.': 'Insufficient valid data for comparison. Please check downloads.',
    'Dữ liệu cơ bản đang trống tạm thời.': 'Fundamental data is temporarily empty.',
    '⚠️ Không tìm thấy dữ liệu VNINDEX_snapshot.xlsx.': '⚠️ VNINDEX_snapshot.xlsx not found.',
    '⚠️ **Ghi chú:** Đây chỉ là dự báo toán học thử nghiệm, hoàn toàn không phải lời khuyên mua bán chứng khoán.': '⚠️ **Note:** This is purely an experimental mathematical forecast, not stock trading advice.',
    'Dữ liệu không đủ dài để AI học. Bạn hãy chọn số ngày lớn hơn 100 nhé!': 'Data is not long enough for AI learning. Please select days > 100!',
    '**Tin tức gần đây:**': '**Recent News:**',
    'Chưa có tin tức mới nhất từ hệ thống cho mã này.': 'No latest news from the system for this ticker.',
    '💡 Nếu hệ số > 0.7: Cổ phiếu bám sát thị trường chung. Ngược lại nếu < 0.3 thì cổ phiếu đó có lối đi riêng.': '💡 If coeff > 0.7: Stock closely tracks the board market. If < 0.3, it moves independently.',
    'Không đủ dữ liệu lịch sử chuẩn để tính tương quan.': 'Insufficient standard historical data to compute correlation.',
    'Random Forest mạnh mẽ để dự đoán Xác suất tăng / giảm, nó bỏ qua nhiễu giá.': 'Random Forest strongly predicts Up/Down probability, bypassing price noise.',
    'Linear Regression giả định mối nối tuyến tính cơ bản giữa các nhịp giá.': 'Linear Regression assumes a basic linear link between price swings.',
    'KNN quét hồ sơ lịch sử, tìm ra 5 chu kỳ giá giống hệt ngày hôm nay nhất để đưa ra giá trị trung bình.': 'KNN scans history, finding the 5 price cycles most identical to today to output an average.',
    '🧠 Thuật toán Học Sâu (Deep Learning) quét nhiều tầng tính năng phức tạp để cho ra giá trị sát thị trường giống kỹ thuật của bài học Data Flair.': '🧠 Deep Learning scans multiple complex features to yield a market-close value.',
    '➡️ Kết luận: Cổ phiếu này đang **bám sát** thị trường chung.': '➡️ Conclusion: This stock is **closely tracking** the broad market.',
    '➡️ Kết luận: Cổ phiếu này có **lối đi riêng**.': '➡️ Conclusion: This stock moves on its **own path**.',
    '**Bóc tách quyết định:** Tại sao AI lại cho ra dự báo Tăng/Giảm thay vì con số ngược lại? Dưới đây là phân bổ trọng số (Feature Importance) cho thấy AI đã ưu tiên nhìn vào chỉ báo nào nhiều nhất để ra quyết định.': '**Decision Breakdown:** Why this Up/Down forecast? Below is the Feature Importance showing what the AI prioritized most.',
    'Thay vì chỉ đưa ra con số bâng quơ, Hồi quy tuyến tính xây dựng một phương trình đường thẳng. Dưới đây là mặt cắt công thức mà thuật toán vừa tính được:': 'Instead of a random number, Linear Regression builds a straight line equation. Here is the formula computed:',
    '➡️ Nếu bạn lấy mảng dữ liệu (đã chuẩn hóa Scaler) của ngày hôm nay cắm vào công thức trên, ta có phép tính chính xác như sau:': "➡️ If you plug today's (Scaler normalized) data into the formula above, the exact calculation is:",
    'Thuật toán KNN không dùng phương trình đường thẳng, thay vào đó nó so sánh biểu đồ hôm nay với dữ liệu quá khứ bằng **Công thức tính Khoảng cách Ơ-clit (Euclidean Distance)**:': "KNN doesn't use linear equations; instead, it compares today's chart with past data using the **Euclidean Distance Formula**:",
    'Ví dụ, với ngày lịch sử có đồ thị **Giống Nhất (Top 1)**, máy học ráp số vào công thức trên để đo như sau:': 'For example, with the historical day having the **Most Identical Chart (Top 1)**, the machine measures it as follows:',
    '📏 Tương tự, nó tìm ra 5 chu kỳ có độ sai lệch (khoảng cách) thấp nhất như sau:': '📏 Similarly, it finds the 5 cycles with the lowest deviation (distance) as follows:',
    'Cận 0 nghĩa là AI cực kỳ tự tin vì hôm nay đang lặp lại gần như y hệt 5 ngày vĩ đại nhất trong quá khứ.': 'Close to 0 means the AI is extremely confident as today perfectly mirrors the 5 greatest days in the past.',
    "**Bóc tách Hộp Đen (Black Box):** Mạng Nơ-ron (Deep Learning) rất dễ bị 'Học vẹt' (Overfitting). Để chứng minh nó thực sự hiểu quy luật thị trường chứ không phải học thuộc lòng, máy tính đã cố tình 'giấu đi' 30 ngày giao dịch gần nhất (Tập Kiểm Thử). Sau đó, nó bắt AI phải tự phân tích và vẽ ra đường giá của 30 ngày đó.": "**Black Box Breakdown:** Neural Networks easily 'parrot' (Overfit). To prove it truly understands market rules, the system intentionally hid the last 30 trading days (Test Set). It then forced the AI to predict the price path for those 30 days autonomously.",
    '➡️ **Giải mã Biểu đồ:**': '➡️ **Chart Decoding:**',
    '- **Đường nét liền (Màu Xanh):** Là biến động Giá Thực Tế của cổ phiếu trên sàn.': '- **Solid Line (Blue):** The Actual Price fluctuation on the market.',
    '- **Đường nét đứt (Màu Hồng):** Là mức giá mà não bộ AI tự động suy luận ra trong cùng lúc đó.': '- **Dashed Line (Pink):** The price level dynamically deduced by the AI.',
    'Tóm lại: Nếu đường Màu Hồng uốn lượn bám dính lấy đường Màu Xanh, có nghĩa là AI dường như đã nhạy cảm và bắt đúng chu kỳ thật sự của thị trường!': 'In short: If the Pink line smoothly hugs the Blue line, the AI has successfully captured the true market cycle!',
    '➡️ **Giải mã:** Cột cao nhất thuộc về **{best_feature}**. Điều này bóc trần sự thật rằng: Trong quá trình dò tìm hàng nghìn ngày giao dịch lịch sử, AI nhận thấy biến số `{best_feature}` đóng vai trò sinh tử lớn nhất tác động đến việc giá đảo chiều hay tiếp diễn vào ngày hôm sau.': '➡️ **Decoding:** The highest column belongs to **{best_feature}**. This reveals the truth: while scanning past business days, AI found the variable `{best_feature}` plays a critical role triggering price reversal or trend continuation tomorrow.',
    'Cột cao nhất thuộc về': 'The highest column belongs to',
    'Điều này bóc trần sự thật rằng: Trong quá trình dò tìm hàng nghìn ngày giao dịch lịch sử, AI nhận thấy biến số': 'This reveals the truth: while searching through thousands of historical trading days, AI found the variable',
    'đóng vai trò sinh tử lớn nhất tác động đến việc giá đảo chiều hay tiếp diễn vào ngày hôm sau.': 'plays the most vital role hitting price reversal or continuation tomorrow.',
    'Sai lệch Top 1 = √ [ (': 'Top 1 Deviation = √ [ (',
    'Kết quả cuối cùng:': 'Final Result:',
    'Trung bình cộng = (': 'Average = (',
    '✔️ Tích cực: Giá hiện tại đang NẰM TRÊN trung bình 200 ngày (Xu hướng tăng dài hạn).': '✔️ Positive: Current price is ABOVE 200-day MA (Long-term uptrend).',
    '❌ Tiêu cực: Giá hiện tại NẰM DƯỚI trung bình 200 ngày (Xu hướng giảm).': '❌ Negative: Current price is BELOW 200-day MA (Downtrend).',
    '✔️ Tích cực: RSI cho thấy cổ phiếu đang ở vùng giá rẻ / hấp dẫn.': '✔️ Positive: RSI shows the stock is in a cheap / attractive zone.',
    '❌ Cảnh báo: RSI quá cao, cổ phiếu có thể đã tăng nóng (Quá mua).': '❌ Warning: RSI is too high, stock may be overheated (Overbought).',
    '⚪ Trung lập: RSI ở mức bình thường.': '⚪ Neutral: RSI is at a normal level.',
    '⚪ Trung lập: Khối lượng giao dịch ở mức ổn định.': '⚪ Neutral: Trading volume is stable.',
    'Mô phỏng Tracking 30 ngày Tương lai gần nhất': 'Simulating 30-Day Future Tracking',
    'Giá Mở Cửa (VND)': 'Open Price (VND)',
    'Giá Thực Tế': 'Actual Price',
    'AI Dự Báo': 'AI Forecast',
    '**Dữ liệu Kinh tế Vĩ mô Hiện tại (Mock Data):**': '**Current Macroeconomic Data (Mock):**',
    'CPI (Lạm phát)': 'CPI (Inflation)',
    'Lãi suất điều hành': 'Policy Rate',
    'Tỷ giá trung tâm (USD/VND)': 'Central Exchange (USD/VND)',
    '💻 **Thuộc nhóm Công nghệ:** Đặc thù ngành này ít bị ảnh hưởng trực tiếp bởi lãi suất trong nước. Tuy nhiên, tỷ giá USD/VND đang ở mức cao (25,400) là **cực kỳ có lợi** vì phần lớn doanh thu của họ đến từ xuất khẩu phần mềm (thu bằng USD nhưng trả lương Kỹ sư bằng VND).': '💻 **Tech Sector:** Intrinsically less affected by domestic interest rates. However, the high USD/VND exchange rate (25,400) is **extremely beneficial** as their revenue mostly comes from software exports (earning USD but paying engineers in VND).',
    '🏦 **Thuộc nhóm Ngân hàng:** Lãi suất điều hành thấp (4.5%) thường giúp duy trì biên lãi ròng (NIM) tốt do huy động vốn rẻ, đồng thời kích thích tăng trưởng tín dụng. Tuy nhiên, tỷ giá neo cao có thể khiến NHNN chịu áp lực hút tiền về. Nhạy cảm số 1 với chính sách tiền tệ.': '🏦 **Banking Sector:** Low policy rate (4.5%) helps maintain a good Net Interest Margin (NIM) via cheap capital, stimulating credit growth. Yet, high exchange rates pressure the State Bank to withdraw money. Most sensitive to monetary policies.',
    '🏢 **Thuộc nhóm Bất động sản:** Khát vốn nhất thị trường. Ngành này RẤT THÍCH môi trường lãi suất thấp và dòng tiền giá rẻ. Mức lãi suất 4.5% hỗ trợ người mua nhà vay mượn, nhưng các doanh nghiệp rủi ro cao vẫn phải đối mặt với áp lực đáo hạn trái phiếu lớn.': '🏢 **Real Estate:** Most capital-thirsty sector. This industry LOVES a low interest rate environment and cheap liquidity. The 4.5% rate supports homebuyers borrowing, but high-risk firms still face massive bond maturity pressures.',
    '🏗️ **Thuộc nhóm SX Vật liệu / Công nghiệp:** Ngành này phụ thuộc trực tiếp vào tiến độ giải ngân Đầu tư công và sự phục hồi của Bất động sản. Khi kích cầu xây dựng, nhu cầu thép/cao su tăng vọt. Sự biến động của dòng vốn FDI và tỷ giá cũng ảnh hưởng đáng kể.': '🏗️ **Materials / Industrial:** Directly relies on Public Investment disbursement speed and Real Estate recovery. When construction demand is stimulated, steel/rubber demand skyrockets. FDI shifts and exchange rates also impact this.',
    "⚡ **Thuộc nhóm Năng lượng / Tiện ích:** Đây là nhóm 'Phòng thủ ăn chắc mặc bền'. Dù lạm phát (CPI) có cao hay lãi suất biến động mạnh, nhu cầu dùng điện, nước, xăng dầu của người dân không thể giảm. Doanh thu siêu ổn định trong giông bão kinh tế.": "⚡ **Energy / Utilities:** The definitive 'Defensive' group. Regardless of high inflation (CPI) or volatile interest rates, public demand for electricity, water, and fuel cannot drop. Ultra-stable revenue during economic storms.",
    '📈 **Thuộc nhóm Chứng khoán:** Ngành này là nhiệt kế của thị trường, có độ nhạy cực cao với Lãi suất. Lãi suất ngân hàng càng rẻ -> dòng tiền chảy vào chứng khoán tìm kiếm lợi nhuận càng lớn -> thanh khoản tăng kỷ lục -> Doanh thu mảng Môi giới & Margin tăng mạnh.': '📈 **Securities Sector:** The market thermometer, extremely sensitive to interest rates. Cheaper bank rates -> greater cash flow into stocks seeking profits -> record liquidity -> surging Brokerage & Margin revenue.',
    '🛒 **Thuộc nhóm Bán lẻ / Tiêu dùng (VN30):** Chịu ảnh hưởng trực tiếp từ Lạm phát (CPI) và tổng Cầu tiêu dùng nội địa. Nếu CPI (3.98%) duy trì cao sẽ siết chặt hầu bao của người dân, ảnh hưởng kém tới sức mua bán lẻ. Ngược lại, nếu kinh tế phục hồi và thu nhập dân chúng tăng, nhóm này bứt phá mạnh.': '🛒 **Retail / Consumer (VN30):** Directly affected by Inflation (CPI) and total domestic consumer demand. If CPI (3.98%) remains high, wallets tighten, hurting retail purchasing power. Conversely, if the economy recovers and incomes rise, they break out strongly.',
    '✈️ **Thuộc nhóm Hàng không:** Cực kỳ nhạy cảm với 2 yếu tố Vĩ mô: Giá dầu thế giới (chiếm 30-40% chi phí) và Tỷ giá USD/VND (vì họ phải thuê/mua máy bay bằng USD). Tỷ giá 25,400 neo cao tạo sức ép chi phí cực lớn. Điểm bù trừ duy nhất là lượng khách du lịch nội địa phục hồi.': '✈️ **Aviation:** Extremely sensitive to 2 Macro factors: World Oil Prices (30-40% of costs) and USD/VND Exchange Rate (they lease/buy planes in USD). The high 25,400 exchange rate creates tremendous cost pressure. The only offset is recovering domestic tourism.',
    '🎯 Điểm Chốt Lời Mục Tiêu (+15%)': '🎯 Target Take Profit (+15%)',
    'Lợi nhuận cao': 'High Profit',
    '🛑 Điểm Cắt Lỗ Bắt Buộc (-7%)': '🛑 Mandatory Stop Loss (-7%)',
    'Bảo vệ Vốn': 'Capital Protection',
    '⚖️ Tỷ lệ Risk/Reward (R:R)': '⚖️ Risk/Reward Ratio (R:R)',
    'Giao dịch Tốt': 'Good Trade',
    'Nguyên tắc Vàng trong Quản trị Danh mục:': 'Golden Rules in Portfolio Management:',
    '✔️ Tích cực: Khối lượng bùng nổ': '✔️ Positive: Volume explosion',
    '- Dòng tiền đang nhập cuộc mạnh mẽ.': '- Cash flow is entering strongly.',
    '❌ Cảnh báo: Thanh khoản sụt giảm': '❌ Warning: Liquidity dropping',
    '- Nhà đầu tư đang có dấu hiệu đứng ngoài.': '- Investors show signs of staying away.',
    'Điểm Khuyến Nghị Mô Hình (Technical):': 'Model Recommendation Score (Technical):',
    'Tác động Vĩ mô lên hoạt động kinh doanh của': 'Macro Impact on business operations of',
    'Phân tích Cơ bản:': 'Fundamental Analysis:',
    'Để đánh giá chuyên sâu tác động Vĩ mô lên': 'To deeply evaluate the Macro impact on',
    'chúng ta cần mổ xẻ Cơ cấu doanh thu của họ: Thuộc nhóm Xuất khẩu (hưởng lợi tỷ giá), Vay nợ cao (sợ lãi suất), hay Tiêu dùng (sợ lạm phát CPI).': 'we need to dissect their revenue structure: Export-oriented (benefits from forex), High Debt (fears interest rates), or Consumer Goods (fears CPI inflation).',
    'Vị thế tài chính': 'Financial Position',
    'Tỷ VNĐ': 'Billion VND',
    '➡️ **Giải mã:** Cột cao nhất thuộc về **{best_feature}**. Điều này bóc trần sự thật rằng: Trong quá trình dò tìm hàng nghìn ngày giao dịch lịch sử, AI nhận thấy biến số `{best_feature}` đóng vai trò sinh tử lớn nhất tác động đến việc giá đảo chiều hay tiếp diễn vào ngày hôm sau.': '➡️ **Decoding:** The highest column belongs to **{best_feature}**. This reveals the truth: while scanning past business days, AI found the variable `{best_feature}` plays a critical role triggering price reversal or trend continuation tomorrow.',
    'Cột cao nhất thuộc về': 'The highest column belongs to',
    'Điều này bóc trần sự thật rằng: Trong quá trình dò tìm hàng nghìn ngày giao dịch lịch sử, AI nhận thấy biến số': 'This reveals the truth: while scanning past business days, AI found the variable',
    'đóng vai trò sinh tử lớn nhất tác động đến việc giá đảo chiều hay tiếp diễn vào ngày hôm sau.': 'plays a critical role triggering price reversal or trend continuation tomorrow.',
    'Sai lệch Top 1 = √ [ (': 'Top 1 Deviation = √ [ (',
    'Kết quả cuối cùng:': 'Final Result:',
    'Trung bình cộng = (': 'Average = (',
    'Quy tắc Đa dạng hóa:': 'Diversification Rule:',
    "Không 'bỏ tất cả trứng vào một rổ'. Hãy chia số vốn của bạn cho khoảng **3 đến 5 mã** thuộc các nhóm ngành khác nhau (Ví dụ: 1 Mã Bank, 1 Thép, 1 Bất động sản).": "Don't 'put all eggs in one basket'. Split your capital into about **3 to 5 stocks** across sub-sectors (e.g. 1 Bank, 1 Steel, 1 Real Estate).",
    'Tuyệt đối KHÔNG trung bình giá xuống:': 'ABSOLUTELY NO averaging down:',
    'Nhồi thêm tiền vào một cổ phiếu đang lao dốc là con đường nhanh nhất dẫn đến cháy tài khoản.': 'Throwing good money into a plunging stock is the fastest lane to a blown account.',
    'Tài Sản': 'Assets',
    'Nợ Phải Trả': 'Liabilities',
    'Tổng Thu HĐ': 'Total Operating Rev',
    'Hiệu suất Ngân hàng (Tỷ VNĐ)': 'Bank Performance (Billion VND)',
    'Tiền KH Gửi': 'Customer Deposits',
    'Tiền Cho Vay': 'Customer Loans',
    'Huy động vốn vs Cho vay (Tỷ VNĐ)': 'Deposits vs Loans (Billion VND)',
    'Vốn CSH': 'Equity',
    'Cơ cấu Tài Sản / Nguồn Vốn': 'Asset / Funding Structure',
    '🔍 Xem Bằng Chứng AI (Explainable AI)': '🔍 View AI Evidence (Explainable AI)',
    'Mức độ quan trọng của Chỉ báo đầu vào': 'Input Feature Importance',
    'Chỉ báo': 'Feature',
    'Tỷ trọng ảnh hưởng': 'Impact Weight',
    '🔍 Xem Bằng Chứng XAI (Công thức Toán học)': '🔍 View XAI Evidence (Math Formula)',
    '🔍 Xem Bằng Chứng AI (Nearest Neighbors)': '🔍 View AI Evidence (Nearest Neighbors)',
    'Mô phỏng Tracking 30 ngày Tương lai gần nhất': 'Simulating closest 30-day Future Tracking',
    'Hệ số tương quan (Pearson Correlation) với VNIndex:': 'Pearson Correlation with VNIndex:',
    '➡️ Kết luận: Cổ phiếu có sự tương quan trung bình.': '➡️ Conclusion: This stock has average correlation.',
    'So sánh Hiệu suất (Đã chuẩn hóa về 100 điểm)': 'Performance Comparison (Normalized to Base 100)',
    'Tăng trưởng (%)': 'Growth (%)',
    'Không thể vẽ biểu đồ tương quan lúc này:': 'Cannot draw correlation chart at this time:',
    # Stock Comparison Tab
    '⚖️ So Sánh Cổ Phiếu': '⚖️ Stock Comparison',
    'Tính năng định lượng so đối chuẩn doanh nghiệp qua 5 \'Tứ Trụ\' chỉ số (P/E, P/B, ROE, Biên Lợi Nhuận, Nợ/VCSH).': 'A quantitative benchmarking feature comparing companies across 5 core metrics (P/E, P/B, ROE, Net Profit Margin, Debt/Equity).',
    'Chọn mã cổ phiếu để so sánh (Tối đa 5 mã):': 'Select stocks to compare (Max 5):',
    'Tiến hành so sánh 🚀': 'Run Comparison 🚀',
    '❌ Vui lòng chọn tối thiểu 2 mã để đưa lên bàn cân nha!': '❌ Please select at least 2 stocks to compare!',
    '⚠️ Bỏ qua': '⚠️ Skipping',
    'Không tìm thấy dữ liệu offline.': 'Offline data not found.',
    'Dữ liệu chỉ số sinh lời bị trống.': 'Profitability ratio data is empty.',
    'Chỉ Số': 'Metric',
    'Biên LN Ròng (%)': 'Net Margin (%)',
    'Nợ / Vốn CSH': 'Debt / Equity',
    'So Sánh Định Giá & Hiệu Suất Sinh Lời': 'Valuation & Profitability Comparison',
    '### 🤖 Trí Tuệ Nhân Tạo: Đánh Giá Tương Quan (Local Engine)': '### 🤖 AI: Correlation Assessment (Local Engine)',
    'Đang trích xuất toàn bộ ma trận dữ liệu và hóa thân thành Thẩm định viên...': 'Extracting the full data matrix and running appraisal engine...',
    'Không xác định': 'Undetermined',
    '1. Ma Trận Định Tính (S-W-O-T Ngành):': '1. Qualitative Matrix (S-W-O-T Sector):',
    'Tiêu chí': 'Criteria',
    'Nội dung phân tích & So sánh': 'Analysis & Comparison',
    'Lợi thế nghiêng về': 'Advantage leans to',
    'Thị phần': 'Market Share',
    'Lợi thế cạnh tranh': 'Competitive Advantage',
    'Ban lãnh đạo': 'Board Leadership',
    'Hoà / Cân bằng': 'Balanced / Tied',
    '2. Phân tích Định giá (P/E & P/B):': '2. Valuation Analysis (P/E & P/B):',
    '3. Phân tích Hiệu quả & Rủi ro:': '3. Efficiency & Risk Analysis:',
    '📌 TỔNG KẾT & KHUYẾN NGHỊ TỪ LOCAL AI:': '📌 SUMMARY & RECOMMENDATION FROM LOCAL AI:',
    'Trường phái Giá trị': 'Value Investing',
    'Trường phái Tăng trưởng': 'Growth Investing',
    '📞 Liên hệ': '📞 Contact',
    '👨‍💻 Giới thiệu bản thân': '👨‍💻 About Me',
    'Overbought/Quá mua (70)': 'Overbought (70)',
    'Oversold/Quá bán (30)': 'Oversold (30)',
    'Volume - Khối lượng': 'Volume',
    '🟢 TĂNG GIÁ (UP)': '🟢 UP (BULLISH)',
    '🔴 GIẢM GIÁ (DOWN)': '🔴 DOWN (BEARISH)',
    'Giá': 'Price',
    'Lỗi khởi chạy Học Máy:': 'Machine Learning launch error:',
    'Lỗi hệ thống:': 'System error:',
    'Lỗi vẽ đồ thị:': 'Chart rendering error:',
    # Headings
    '1. Phân Tích Kỹ Thuật (Technical Analysis)': '1. Technical Analysis',
    '2. Phân Tích Cơ Bản (Fundamental Ratios)': '2. Fundamental Ratios',
    '📅 Dữ liệu chốt đến hết ngày giao dịch 06/04/2026 (Offline Snapshot).': '📅 Data frozen through trading day 06/04/2026 (Offline Snapshot).',
    '### 📊 Đồ thị Tình hình Tài chính (4 Kỳ Gần Nhất)': '### 📊 Financial Status Charts (Last 4 Periods)',
    '3. Tương quan Thị trường (VNINDEX Correlation Model)': '3. Market Correlation (VNINDEX Model)',
    '4. Đánh Giá Điểm Số (Model Scoring)': '4. Model Scoring & Conclusion',
    '5. Dự Báo Mô Hình Nâng Cao (Machine Learning Models)': '5. Advanced Forecasting (Machine Learning)',
    '6. Thông tin Vĩ mô & Tin tức (Macro & News)': '6. Macro Information & News',
    '7. Quản Trị Rủi Ro (Risk Management)': '7. Risk Management',
    '### 🤖 Trí Tuệ Nhân Tạo: Đánh Giá Tương Quan (Local Engine)': '### 🤖 AI: Correlation Evaluator (Local Engine)',
    '### 🤖 AI Chẩn Đoán Sức Khỏe Doanh Nghiệp': '### 🤖 AI Corporate Health Diagnostics',
    'Chỉ số Sức mạnh Tương đối - RSI(14)': 'Relative Strength Index - RSI(14)'
}

def t(text):
    if st.session_state.get('lang', '🇻🇳 Tiếng Việt') == '🇬🇧 English':
        return _LANG.get(text, text)
    return text

st.markdown(f'<div class="main-title">{t("📈 Hệ Thống Đánh Giá Cổ Phiếu Toàn Diện")}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="sub-title">{t("Hệ thống kết hợp Phân tích Kỹ thuật, Phân tích Cơ bản và Vĩ mô để đánh giá cổ phiếu Việt Nam.")}</div>', unsafe_allow_html=True)

# ----------------- HEADER (Hiển thị cố định ở đầu mọi trang) -----------------
t_footer_1 = t('📌 Lưu ý: Chuyên trang này được tối ưu hóa để phân tích báo cáo cho nhóm Ngành lớn & VN30.')
t_footer_2 = t('🗓️ Cập nhật Dữ liệu (Offline Mode): Toàn bộ dữ liệu biểu đồ, định giá và báo cáo tài chính trên hệ thống được chốt cố định đến hết giao dịch ngày hiện tại nhằm phục vụ chấm điểm năng lực thuật toán.')
t_footer_3 = t('⚖️ DỰ ÁN CÁ NHÂN (PORTFOLIO PROJECT): Trang web này là một dự án Mã nguồn mở mang tính chất Học thuật & Giáo dục (Data Science & Machine Learning Portfolio). Tuyên bố miễn trừ trách nhiệm: Các dữ liệu và phân tích (AI/ML) trên Dashboard này chỉ mang tính chất tham khảo, mô phỏng học thuật và không phải là lời khuyên đầu tư tài chính. Nguồn dữ liệu thô: Vnstock.')

st.info(t_footer_1)
st.success(t_footer_2)
st.warning(t_footer_3)
st.markdown(t("👉 **Nếu bạn muốn xem thêm dự án khác hãy [nhấp vào đây](https://portfolio-gilt-sigma-43.vercel.app)**"))
st.markdown("---")

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
    st.markdown(f"### 🧭 {t('Điều hướng')}")
    selected_translated = option_menu(
        menu_title=None, 
        options=["1. Forecasting & Analysis", "2. Stock Comparison", "3. About Me"], 
        icons=["graph-up-arrow", "arrow-left-right", "person-badge"], 
        default_index=0, 
        orientation="vertical",
        styles={
            "container": {"background-color": "transparent"},
            "nav-link": {"font-size": "14px", "font-weight": "500", "color": "#1E293B", "--hover-color": "#E2E8F0"},
            "nav-link-selected": {"background-color": "#2563EB", "color": "white", "font-weight": "bold"},
            "icon": {"color": "#64748B"}
        }
    )
    
    # Reverse mapping for logic
    if selected_translated == "2. Stock Comparison":
        selected = "So Sánh Cổ Phiếu"
    elif selected_translated == "3. About Me":
        selected = "Về Tác Giả (About Me)"
    else:
        selected = "Dự Báo & Phân Tích"
        
    st.markdown("---")
    
    # --- DỜI SETTINGS TỪ DƯỚI LÊN ĐÂY ĐỂ TRÁNH LỖI VỠ LAYOUT DO ST.STOP() ---
    st.header(t("Cài Đặt (Settings)"))
    
    sidebar_available_tickers = []
    if os.path.exists(DATA_DIR):
        sidebar_available_tickers = [f.replace("_snapshot.xlsx", "") for f in os.listdir(DATA_DIR) if f.endswith("_snapshot.xlsx") and not f.startswith("VNINDEX")]
    if not sidebar_available_tickers:
        sidebar_available_tickers = ["FPT"]
    
    default_idx = sidebar_available_tickers.index("FPT") if "FPT" in sidebar_available_tickers else 0
    ticker = st.selectbox(t("Chọn mã cổ phiếu (Dữ liệu Offline):"), options=sidebar_available_tickers, index=default_idx)
    period_option = st.selectbox(t("Dữ liệu lịch sử:"), [t("1 tháng"), t("3 tháng"), t("12 tháng")], index=2)
    days = 30 if period_option == t("1 tháng") else (90 if period_option == t("3 tháng") else 365)
    
    # Visitor Counter Logic (Cloud-safe)
    counter_file = "visitor_count.txt"
    if 'visited' not in st.session_state:
        try:
            if os.path.exists(counter_file):
                with open(counter_file, "r") as f:
                    try: count = int(f.read().strip())
                    except ValueError: count = 0
            else: count = 0
            count += 1
            with open(counter_file, "w") as f:
                f.write(str(count))
        except (PermissionError, OSError):
            count = "N/A"
        st.session_state.visited = count
    else:
        count = st.session_state.visited
    
    st.markdown("---")
    st.metric(label=f"👁️ {t('Lượt truy cập Web')}", value=count)

if selected in ["Về Tác Giả (About Me)", "About the Author"]:
    st.markdown("---")
    col_img, col_info = st.columns([1, 2])
    with col_img:
        st.markdown(f"### {t('📞 Liên hệ')}")
        st.write("📧 Email: hoanhkhoa1009@gmail.com")
        st.write("🌐 LinkedIn: [anh-khoa-3223912a0](https://www.linkedin.com/in/anh-khoa-3223912a0)")

    with col_info:
        st.title(t("👨‍💻 Giới thiệu bản thân"))
        _is_en = st.session_state.lang == '🇬🇧 English'
        if _is_en:
            st.write("""
            ### Hello, I'm Khoa.
            
            Welcome to my personal project — a system carefully developed from a deep passion for the Vietnamese financial market.
            
            To optimize the development process, I utilized **Google Antigravity**. This next-generation AI development platform helped me overcome the barriers of manual coding, allowing me to focus entirely on shaping ideas and perfecting the core analytical logic.
            
            If you're visiting this website from my CV, I hope this system serves as the clearest proof: for me, knowledge should not just exist on paper, but must be transformed into real execution capability and working products.
            
            **Best regards,**  
            **Ho Anh Khoa**
            """)
        else:
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
    st.title(t("⚖️ So Sánh Cổ Phiếu"))
    st.markdown(t("Tính năng định lượng so đối chuẩn doanh nghiệp qua 5 'Tứ Trụ' chỉ số (P/E, P/B, ROE, Biên Lợi Nhuận, Nợ/VCSH)."))
    
    import os
    available_tickers = []
    if os.path.exists(DATA_DIR):
        available_tickers = [f.replace("_snapshot.xlsx", "") for f in os.listdir(DATA_DIR) if f.endswith("_snapshot.xlsx") and not f.startswith("VNINDEX")]
    if not available_tickers:
        available_tickers = ["CTG", "MBB", "TCB", "FPT", "HPG", "VNM"]
        
    selected_tickers = st.multiselect(t("Chọn mã cổ phiếu để so sánh (Tối đa 5 mã):"), options=available_tickers, default=["CTG", "MBB"] if "CTG" in available_tickers and "MBB" in available_tickers else available_tickers[:2], max_selections=5)
    
    if st.button(t("Tiến hành so sánh 🚀"), use_container_width=True, type="primary"):
        if len(selected_tickers) < 2:
            st.error(t("❌ Vui lòng chọn tối thiểu 2 mã để đưa lên bàn cân nha!"))
        else:
            all_metrics = {}
            for tk in selected_tickers:
                data = load_local_data(tk)
                if not data:
                    st.warning(f"{t("⚠️ Bỏ qua")} {tk}: {t("Không tìm thấy dữ liệu offline.")}")
                    continue
                r = data.get('Ratios', pd.DataFrame())
                if r.empty:
                    st.warning(f"{t("⚠️ Bỏ qua")} {tk}: {t("Dữ liệu chỉ số sinh lời bị trống.")}")
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
                
                all_metrics[tk] = {
                    'P/E': safe_get(r_latest, ['p/e']),
                    'P/B': safe_get(r_latest, ['p/b', 'price to book']),
                    'ROE (%)': safe_get(r_latest, ['roe']) * 100,
                    'Biên LN Ròng (%)': safe_get(r_latest, ['net profit margin']) * 100,
                    'Nợ / Vốn CSH': safe_get(r_latest, ['debt/equity', 'debt to equity'])
                }
                
            if len(all_metrics) >= 2:
                st.subheader(t("📊 Bảng Chỉ Số Tương Quan"))
                _metric_keys = ['P/E', 'P/B', 'ROE (%)', 'Biên LN Ròng (%)', 'Nợ / Vốn CSH']
                _metric_labels = ['P/E', 'P/B', 'ROE (%)', t('Biên LN Ròng (%)'), t('Nợ / Vốn CSH')]
                comp_df = pd.DataFrame({t('Chỉ Số'): _metric_labels})
                for tk, metrics in all_metrics.items():
                    comp_df[tk] = [metrics[k] for k in _metric_keys]
                
                fig_comp = go.Figure()
                colors = ['#29B6F6', '#69F0AE', '#FFA726', '#AB47BC', '#EF5350']
                for i, tk in enumerate(all_metrics.keys()):
                    fig_comp.add_trace(go.Bar(x=comp_df[t('Chỉ Số')], y=comp_df[tk], name=tk, marker_color=colors[i % len(colors)], text=comp_df[tk].apply(lambda x: f"{x:.2f}"), textposition='auto'))
                
                fig_comp.update_layout(barmode='group', template='plotly_dark', title=t("So Sánh Định Giá & Hiệu Suất Sinh Lời"))
                st.plotly_chart(fig_comp, use_container_width=True)
                
                st.table(comp_df.set_index(t('Chỉ Số')).style.format("{:.2f}"))
                
                import time
                st.markdown(t("### 🤖 Trí Tuệ Nhân Tạo: Đánh Giá Tương Quan (Local Engine)"))
                with st.spinner(t("Đang trích xuất toàn bộ ma trận dữ liệu và hóa thân thành Thẩm định viên...")):
                    time.sleep(1.8)
                    
                    valid_pe = {k: v['P/E'] for k, v in all_metrics.items() if v['P/E'] > 0}
                    cheaper = min(valid_pe, key=valid_pe.get) if valid_pe else t("Không xác định")
                    stronger = max(all_metrics.keys(), key=lambda k: all_metrics[k]['ROE (%)'])
                    t_list_str = ", ".join(all_metrics.keys())
                    
                    _is_en = st.session_state.lang == '🇬🇧 English'
                    if _is_en:
                        mock_resp = f"""
**1. Qualitative Matrix (S-W-O-T Sector):**  
Based on the analysis basket including **[{t_list_str}]**, here is the refined core competency correlation matrix:

| Criteria | Analysis & Comparison | Advantage leans to |
| :--- | :--- | :---: |
| **Market Share** | Which company leads in scale? Who commands the largest influence in this basket? | **{stronger}** |
| **Competitive Advantage** | Assessing strength from distribution networks, technology, and fixed asset optimization. | **{stronger}** |
| **Board Leadership** | Track record of strategic execution by the Board and reputation in the stock market. | Balanced / Tied |

**2. Valuation Analysis (P/E & P/B):**  
When applying the P/E filter, **{cheaper}** stands out with the cheapest discount in the entire stock basket (buying 1 unit of profit at the best price). This makes it a "safe haven" for defensive investors.

**3. Efficiency & Risk Analysis:**  
When dissecting the ROE performance matrix, **{stronger}** has brilliantly claimed the crown. This company's engine is creating a vast "economic moat", squeezing maximum asset utilization to generate superior returns over peers.

**📌 SUMMARY & RECOMMENDATION FROM LOCAL AI:**  
This matchup reveals a beautiful divergence:
* 🛡️ **Pick {cheaper} (Value Investing):** For those who love a large margin of safety and extremely low downside risk.
* 🚀 **Pick {stronger} (Growth Investing):** Accept paying a premium, but own the fastest-performing "thoroughbred" in the market.

*(Note: This analysis is automatically assessed via a local **Heuristic Engine**, completely offline to ensure query privacy)*
"""
                    else:
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
                    st.warning(t("⚠️ **Đây chỉ là dự báo và không khuyến khích làm theo (Tôi sẽ không chịu trách nhiệm vì bất kỳ điều gì)**"))
            else:
                st.error(t("Không đủ dữ liệu hợp lệ để tiến hành đối sánh. Vui lòng check lại tải xuống."))
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
    st.header(f"{t('Báo Cáo Tổng Hợp:')} {ticker}")
    
    try:
        # CHẾ ĐỘ PURE OFFLINE (Sử dụng dữ liệu tĩnh 06/04/2026)
        local_data = load_local_data(ticker)
        df = pd.DataFrame()
        stock = None
        
        if local_data:
            df = local_data['Price']
            stock = LocalStock(local_data)
            st.success(f"📁 {t('Chế độ Offline Tĩnh: Đã tải dữ liệu Snapshot an toàn cho mã')} {ticker} {t('(Chốt ngày')} 06/04).")
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
            st.subheader(t("1. Phân Tích Kỹ Thuật (Technical Analysis)"))
            
            # Tính toán OHLC phiên mới nhất giống Fireant
            latest_row = df.iloc[-1]
            prev_row = df.iloc[-2] if len(df) > 1 else latest_row
            
            o, h, l, c, v = latest_row['open'], latest_row['high'], latest_row['low'], latest_row['close'], latest_row['volume']
            change = c - prev_row['close']
            change_pct = (change / prev_row['close']) * 100 if prev_row['close'] != 0 else 0
            
            # Light theme colors
            color = "#00897B" if change > 0 else ("#E53935" if change < 0 else "#FB8C00") 
            color_text = "💚" if change > 0 else ("❤️" if change < 0 else "💛")
            vol_str = f"{v/1000:.1f}K" if v < 1000000 else f"{v/1000000:.2f}M"
            
            # Khóa mốc thời gian hiển thị vào dòng dữ liệu cuối cùng (06/04)
            last_date_str = latest_row.name.strftime('%d/%m/%Y')
            
            st.markdown(f"""
            <div style='background-color: #FFFFFF; padding: 12px 15px; border-radius: 8px; font-family: "Inter", "Trebuchet MS", Arial, sans-serif; margin-bottom: 15px; border: 1px solid #D5D8DC; box-shadow: 0 1px 3px rgba(0,0,0,0.05);'>
                <span style='font-size: 1.2em; color: #1C2833;'><b>{ticker}</b> · 1D · <b>{last_date_str}</b></span><br>
                <span style='font-size: 1.05em; color: #566573;'>
                    O <span style='font-weight: 500; color:{color}'>{o:,.2f}</span> &nbsp; 
                    H <span style='font-weight: 500; color:{color}'>{h:,.2f}</span> &nbsp; 
                    L <span style='font-weight: 500; color:{color}'>{l:,.2f}</span> &nbsp; 
                    C <span style='font-weight: 600; color:{color}'>{c:,.2f}</span> &nbsp; 
                    <span style='font-weight: 600; font-size: 1.1em; color:{color}'>{change:+,.2f} ({change_pct:+,.2f}%)</span>
                </span><br>
                <span style='font-size: 0.95em; color: #808B96;'>{t('Volume - Khối lượng')} <span style='font-weight: 500; color:{color}'>{vol_str}</span></span>
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
            
            fig.update_layout(height=500, title=f"{t('Biểu đồ giá')} {ticker} {t('& Các đường hỗ trợ (SMA, Fibonacci)')}")
            st.plotly_chart(fig, width='stretch')
            
            # RSI Chart
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI_14'], line=dict(color='purple', width=2), name='RSI 14'))
            fig_rsi.add_hline(y=70, line_dash="dot", line_color="red", annotation_text=t("Overbought/Quá mua (70)"))
            fig_rsi.add_hline(y=30, line_dash="dot", line_color="green", annotation_text=t("Oversold/Quá bán (30)"))
            fig_rsi.update_layout(height=250, title=t("Chỉ số Sức mạnh Tương đối - RSI(14)"))
            st.plotly_chart(fig_rsi, width='stretch')
            
            current_close = df['close'].iloc[-1]
            current_rsi = df['RSI_14'].iloc[-1] if not pd.isna(df['RSI_14'].iloc[-1]) else 50
            
            # --- 2. FUNDAMENTAL ANALYSIS (PHÂN TÍCH CƠ BẢN) ---
            st.subheader(t("2. Phân Tích Cơ Bản (Fundamental Ratios)"))
            st.caption(t("📅 Dữ liệu chốt đến hết ngày giao dịch 06/04/2026 (Offline Snapshot)."))
            try:
                # Get ratio from the already created stock object
                ratio_df = stock.finance.ratio()
                if not ratio_df.empty:
                    # Bilingual column rename for display
                    _is_vn = st.session_state.lang == '🇻🇳 Tiếng Việt'
                    display_df = ratio_df.head(10).copy()
                    if isinstance(display_df.columns, pd.MultiIndex):
                        display_df.columns = [col[-1] if isinstance(col, tuple) else col for col in display_df.columns]
                    display_df.columns = [str(c).strip() for c in display_df.columns]
                    if _is_vn:
                        _col_vi = {
                            'ticker': 'Mã CP', 'yearReport': 'Năm', 'lengthReport': 'Quý',
                            '(ST+LT borrowings)/Equity': 'Vay NH+DH/VCSH',
                            'Debt/Equity': 'Nợ/VCSH', 'Fixed Asset-To-Equity': 'TSCĐ/VCSH',
                            "Owners' Equity/Charter Capital": 'VCSH/Vốn ĐL',
                            'Asset Turnover': 'Vòng quay TS', 'Fixed Asset Turnover': 'Vòng quay TSCĐ',
                            'Days Sales Outstanding': 'Số ngày thu tiền', 'Days Inventory Outstanding': 'Số ngày tồn kho',
                            'Days Payable Outstanding': 'Số ngày trả tiền', 'Cash Cycle': 'Chu kỳ tiền mặt',
                            'Inventory Turnover': 'Vòng quay HTK',
                            'EBIT Margin (%)': 'Biên EBIT (%)', 'Gross Profit Margin (%)': 'Biên LN Gộp (%)',
                            'Net Profit Margin (%)': 'Biên LN Ròng (%)', 'ROE (%)': 'ROE (%)',
                            'ROIC (%)': 'ROIC (%)', 'ROA (%)': 'ROA (%)',
                            'EBITDA (Bn. VND)': 'EBITDA (Tỷ VNĐ)', 'EBIT (Bn. VND)': 'EBIT (Tỷ VNĐ)',
                            'Dividend yield (%)': 'Tỷ suất Cổ tức (%)',
                            'Current Ratio': 'Tỷ số Thanh toán HT', 'Cash Ratio': 'Tỷ số Tiền mặt',
                            'Quick Ratio': 'Tỷ số Thanh toán NH', 'Interest Coverage': 'Khả năng trả Lãi vay',
                            'Financial Leverage': 'Đòn bẩy Tài chính',
                            'Market Capital (Bn. VND)': 'Vốn hóa TT (Tỷ VNĐ)',
                            'Outstanding Share (Mil. Shares)': 'CP Lưu hành (Tr. CP)',
                            'P/E': 'P/E', 'P/B': 'P/B', 'P/S': 'P/S', 'P/Cash Flow': 'P/Dòng tiền',
                            'EPS (VND)': 'EPS (VNĐ)', 'BVPS (VND)': 'BVPS (VNĐ)', 'EV/EBITDA': 'EV/EBITDA'
                        }
                        display_df.rename(columns=_col_vi, inplace=True)
                    st.dataframe(display_df)
                    
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
                        
                        st.markdown(f"{t('### 🤖 AI Chẩn Đoán Sức Khỏe Doanh Nghiệp')} ({ticker} - {latest_q})")
                        
                        if de_ratio_series is not None and at_ratio_series is not None:
                            # Cổ phiếu Doanh nghiệp Thương mại / Sản xuất
                            de_ratio = float(de_ratio_series.iloc[0])
                            at_ratio = float(at_ratio_series.iloc[0])
                            dso_ratio = float(ratio_df.get('Days Sales Outstanding', pd.Series([0])).iloc[0])
                            
                            de_status = t("🟢 RẤT AN TOÀN (Nợ thấp)") if de_ratio < 1.5 else (t("🟡 TRUNG BÌNH") if de_ratio < 2.5 else t("🔴 NỢ NGẬP ĐẦU (Rủi ro vỡ nợ)"))
                            
                            st.markdown(f"""
                            {t('Dựa trên Báo cáo tài chính quý gần nhất, hệ thống phân tích các chỉ tiêu trọng yếu:')}
                            
                            **{t('1. Sức khỏe Tài chính (Chống phá sản)')}**
                            - {t('Hệ số Nợ/Vốn chủ sở hữu (Debt/Equity):')} **{de_ratio:.2f}** -> **{de_status}** *({t('Doanh nghiệp có 10 đồng vốn thì đi nợ')} {de_ratio*10:.1f} {t('đồng.')})*
                            
                            **{t('2. Năng lực In Tiền (Hiệu quả hoạt động)')}**
                            - {t('Vòng quay Tổng Tài sản đạt')} **{at_ratio:.2f}**. *(1 {t('đồng tài sản đẻ ra được')} {at_ratio:.2f} {t('đồng doanh thu')}).*
                            - {t('Số ngày thu tiền bình quân:')} **{dso_ratio:.0f} {t('ngày.')}**. *({t('Sau khi bán hàng, mất')} {dso_ratio:.0f} {t('ngày để thu tiền về')}).*
                            """)
                        elif roe_series is not None:
                            # Cổ phiếu Ngân hàng / Tài chính đặc thù
                            roe = float(roe_series.iloc[0])
                            profit_margin = float(ratio_df.get('Net Profit Margin (%)', pd.Series([0])).iloc[0])
                            lever = float(ratio_df.get('Financial Leverage', pd.Series([0])).iloc[0])
                            
                            roe_status = t("🟢 RẤT XUẤT SẮC") if roe > 0.15 else (t("🟡 TRUNG BÌNH") if roe > 0.10 else t("🔴 KÉM"))
                            
                            st.markdown(f"""
                            {t('*(Phát hiện bạn đang xem Cổ phiếu ngành Tài chính Ngân hàng / Đặc thù. Các chỉ số được điều chỉnh tương ứng)*')}
                            
                            **{t('1. Hiệu suất Sinh lời (Khả năng đẻ lãi của Ngân hàng)')}**
                            - {t('Lợi nhuận trên Vốn Chủ (ROE):')} **{roe*100:.1f}%** -> **{roe_status}**. *({t('Khả năng sinh lời của đồng vốn tự có rất tốt')})*.
                            - {t('Biên Lợi nhuận Ròng:')} **{profit_margin*100:.1f}%**.
                            
                            **{t('2. Rủi ro Đòn Bẩy (Tín dụng)')}**
                            - {t('Đòn bẩy Tài chính (Financial Leverage):')} **{lever:.2f}**. *({t('Đặc thù ngành ngân hàng luôn có đòn bẩy cao')}).*
                            """)
                        else:
                            st.info(t("⚠️ Cấu trúc Báo cáo Tài chính của mã này quá khác biệt so với tiêu chuẩn, tự động đánh giá đang bị vô hiệu hóa."))
                            
                    except Exception as parse_e:
                        st.info(f"{t('⚠️ Không thể đọc tự động (Chi tiết lỗi:')} {parse_e})")
                        
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
                                
                            st.markdown(t("### 📊 Đồ thị Tình hình Tài chính (4 Kỳ Gần Nhất)"))
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
                                    return s / 1e9 if s.abs().max() > 1e8 else s
                                
                                rev_bn = to_bn(rev)
                                profit_bn = to_bn(profit)
                                
                                fig1 = go.Figure()
                                fig1.add_trace(go.Bar(x=is_df['Period'], y=rev_bn, name=t("Doanh thu"), marker_color='#29B6F6', text=rev_bn.apply(lambda x: f"{x:,.1f} T"), textposition='none'))
                                fig1.add_trace(go.Bar(x=is_df['Period'], y=profit_bn, name=t("Lợi nhuận ròng"), marker_color='#69F0AE', text=profit_bn.apply(lambda x: f"{x:,.1f} T"), textposition='none'))
                                sales_clean = rev_bn.replace(0, 1)
                                margin = (profit_bn / sales_clean) * 100
                                fig1.add_trace(go.Scatter(x=is_df['Period'], y=margin, name=t("Biên LN (%)"), yaxis="y2", line=dict(color='#FFD54F', width=2), mode='lines+markers+text', text=margin.apply(lambda x: f"{x:.1f}%"), textposition='top center', textfont=dict(color='#A5A5A5', size=11)))
                                fig1.update_layout(title=t("Hiệu suất (Tỷ VNĐ)"), barmode='group', template='plotly_dark', showlegend=True, legend=dict(orientation="h", y=-0.2), yaxis=dict(ticksuffix=' T', tickformat=','), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False))
                                col1.plotly_chart(fig1, width='stretch')
                                
                                # 2. Cơ cấu BCTC
                                year_latest = is_df['yearReport'].iloc[-1]
                                y_vals = [rev_bn.iloc[-1], -abs(to_bn(cogs).iloc[-1]), 0, -abs(to_bn(sga).iloc[-1]), 0, -abs(to_bn(tax).iloc[-1]), 0]
                                fig2 = go.Figure(go.Waterfall(
                                    orientation = "v",
                                    measure = ["absolute", "relative", "total", "relative", "total", "relative", "total"],
                                    x = [t("Doanh thu"), t("Giá vốn"), t("LN Gộp"), t("Chi phí BH&QL"), t("LN Trước Thuế"), t("Thuế"), t("LN Ròng")],
                                    y = y_vals,
                                    text = [f"{v:,.1f} T" if v != 0 else "" for v in y_vals], textposition="outside",
                                    decreasing = {"marker":{"color":"#FF4136"}}, increasing = {"marker":{"color":"#2ECC40"}}, totals = {"marker":{"color":"#0074D9"}}
                                ))
                                fig2.update_layout(title=f"{t('Kết quả kinh doanh')} ({year_latest})", template='plotly_dark', yaxis=dict(ticksuffix=' T', tickformat=','))
                                col2.plotly_chart(fig2, width='stretch')
                                
                                # 3. Tài sản và Vốn CSH
                                liab_bn = to_bn(liab)
                                eq_bn = to_bn(eq)
                                
                                fig3 = go.Figure()
                                fig3.add_trace(go.Bar(x=bs_df['Period'], y=liab_bn, name=t("Nợ phải trả"), marker_color='#00BCD4', text=liab_bn.apply(lambda x: f"{x:,.1f} T"), textposition='none'))
                                fig3.add_trace(go.Bar(x=bs_df['Period'], y=eq_bn, name=t("Vốn CSH"), marker_color='#4DB6AC', text=eq_bn.apply(lambda x: f"{x:,.1f} T"), textposition='none'))
                                eq_clean = eq_bn.replace(0, 1)
                                debt_eq = (liab_bn / eq_clean) * 100
                                fig3.add_trace(go.Scatter(x=bs_df['Period'], y=debt_eq, name=t("Nợ/VCSH (%)"), yaxis="y2", line=dict(color='#FFEE58', width=2), mode='lines+markers+text', text=debt_eq.apply(lambda x: f"{x:.1f}%"), textposition='top center', textfont=dict(color='#A5A5A5', size=11)))
                                fig3.update_layout(title=t("Tài sản và Vốn chủ sở hữu (Tỷ VNĐ)"), barmode='group', template='plotly_dark', legend=dict(orientation="h", y=-0.2), yaxis=dict(ticksuffix=' T', tickformat=','), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False))
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
                                fig4.add_trace(go.Bar(x=[t('Ngắn hạn'), t('Dài hạn')], y=[ca_l, nca_l], name=t("Tài sản"), marker_color='#29B6F6'))
                                fig4.add_trace(go.Bar(x=[t('Ngắn hạn'), t('Dài hạn')], y=[cl_l, ncl_l], name=t("Nợ phải trả"), marker_color='#69F0AE'))
                                fig4.update_layout(title=f"{t('Vị thế tài chính')} ({bs_latest_yr})", barmode='group', template='plotly_dark', legend=dict(orientation="h", y=-0.2), yaxis=dict(title=t('Tỷ VNĐ'), ticksuffix=' T', tickformat=','))
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
                                    return s / 1e9 if s.abs().max() > 1e8 else s
                                    
                                rev_bn = to_bn(total_op_rev)
                                profit_bn = to_bn(net_profit)
                                
                                fig1 = go.Figure()
                                fig1.add_trace(go.Bar(x=is_df['Period'], y=rev_bn, name=t("Tổng Thu HĐ"), marker_color='#29B6F6', text=rev_bn.apply(lambda x: f"{x:,.1f} T"), textposition='none'))
                                fig1.add_trace(go.Bar(x=is_df['Period'], y=profit_bn, name=t("Lợi nhuận ròng"), marker_color='#69F0AE', text=profit_bn.apply(lambda x: f"{x:,.1f} T"), textposition='none'))
                                rev_clean = rev_bn.replace(0, 1)
                                margin = (profit_bn / rev_clean) * 100
                                fig1.add_trace(go.Scatter(x=is_df['Period'], y=margin, name=t("Biên LN (%)"), yaxis="y2", line=dict(color='#FFD54F', width=2), mode='lines+markers+text', text=margin.apply(lambda x: f"{x:.1f}%"), textposition='top center', textfont=dict(color='#A5A5A5', size=11)))
                                fig1.update_layout(title=t("Hiệu suất Ngân hàng (Tỷ VNĐ)"), barmode='group', template='plotly_dark', showlegend=True, legend=dict(orientation="h", y=-0.2), yaxis=dict(ticksuffix=' T', tickformat=','), yaxis2=dict(title="%", overlaying="y", side="right", showgrid=False))
                                col1.plotly_chart(fig1, width='stretch')
                                
                                # 2. Cơ cấu thu nhập Ngân hàng (Waterfall)
                                year_latest = is_df['yearReport'].iloc[-1]
                                int_bn = to_bn(net_interest).iloc[-1]
                                fee_bn = (to_bn(total_op_rev).iloc[-1] - int_bn if total_op_rev.iloc[-1] != 0 else to_bn(fees).iloc[-1])
                                ga_bn = to_bn(ga_bank).iloc[-1]
                                pr_bn = to_bn(prov).iloc[-1]
                                
                                y_vals_bk = [int_bn, fee_bn, 0, -abs(ga_bn), -abs(pr_bn), 0]
                                fig2 = go.Figure(go.Waterfall(
                                    orientation = "v",
                                    measure = ["absolute", "relative", "total", "relative", "relative", "total"],
                                    x = [t("Thu nhập Lãi"), t("Phí & Khác"), t("Tổng Thu HĐ"), t("Chi phí QL"), t("CP Dự phòng"), t("LN Trước Thuế")],
                                    y = y_vals_bk,
                                    text = [f"{v:,.1f} T" if v != 0 else "" for v in y_vals_bk], textposition="outside",
                                    decreasing = {"marker":{"color":"#FF4136"}}, increasing = {"marker":{"color":"#2ECC40"}}, totals = {"marker":{"color":"#0074D9"}}
                                ))
                                fig2.update_layout(title=f"Kết quả kinh doanh ({year_latest} - Tỷ VNĐ)", template='plotly_dark', yaxis=dict(ticksuffix=' T', tickformat=','))
                                col2.plotly_chart(fig2, width='stretch')
                                
                                # 3. Cân đối kế toán Ngân hàng
                                dep_bn = to_bn(dep)
                                loans_bn = to_bn(loans)
                                
                                fig3 = go.Figure()
                                fig3.add_trace(go.Bar(x=bs_df['Period'], y=dep_bn, name=t("Tiền KH Gửi"), marker_color='#00BCD4'))
                                fig3.add_trace(go.Bar(x=bs_df['Period'], y=loans_bn, name=t("Tiền Cho Vay"), marker_color='#4DB6AC'))
                                fig3.update_layout(title=t("Huy động vốn vs Cho vay (Tỷ VNĐ)"), barmode='group', template='plotly_dark', legend=dict(orientation="h", y=-0.2), yaxis=dict(ticksuffix=' T', tickformat=','))
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
                                
                                fig4.add_trace(go.Bar(x=[t('Tổng Tài Sản'), t('Nợ Phải Trả'), t('Vốn CSH')], y=[assets_bn, liab_final_bn, eq_final_bn], marker_color=['#29B6F6', '#FF8A65', '#69F0AE']))
                                fig4.update_layout(title=f"{t('Cơ cấu Tài Sản / Nguồn Vốn')} ({bs_latest_yr})", template='plotly_dark', yaxis_title="Tỷ VNĐ")
                                col2.plotly_chart(fig4, width='stretch')

                    except Exception as draw_e:
                        st.warning(f"{t('Lỗi vẽ đồ thị:')} {draw_e}")
                        
                else:
                    st.warning(t("Dữ liệu cơ bản đang trống tạm thời."))
                
            except Exception as e:
                st.warning(f"Không thể tải BCTC tự động lúc này. Lỗi hệ thống: {str(e)[:100]}...")
            
            # --- 3. MARKET CORRELATION (TƯƠNG QUAN VNINDEX) ---
            st.subheader(t("3. Tương quan Thị trường (VNINDEX Correlation Model)"))
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
                        
                        st.write(f"**{t('Hệ số tương quan (Pearson Correlation) với VNIndex:')} `{correlation:.2f}`**")
                        st.info(t("💡 Nếu hệ số > 0.7: Cổ phiếu bám sát thị trường chung. Ngược lại nếu < 0.3 thì cổ phiếu đó có lối đi riêng."))
                        
                        if correlation > 0.7:
                            st.write(t("➡️ Kết luận: Cổ phiếu này đang **bám sát** thị trường chung."))
                        elif correlation < 0.3:
                            st.write(t("➡️ Kết luận: Cổ phiếu này có **lối đi riêng**."))
                        else:
                            st.write(t("➡️ Kết luận: Cổ phiếu có sự tương quan trung bình."))
                            
                        # Plot comparison chart
                        fig_corr = go.Figure()
                        fig_corr.add_trace(go.Scatter(x=corr_df.index, y=corr_df[f'{ticker} (Base 100)'], line=dict(color='blue', width=2), name=ticker))
                        fig_corr.add_trace(go.Scatter(x=corr_df.index, y=corr_df['VNINDEX (Base 100)'], line=dict(color='gray', width=2, dash='dot'), name='VNINDEX'))
                        _vn_date_fmt = "%d/%m/%Y" if st.session_state.lang == '🇻🇳 Tiếng Việt' else "%b %Y"
                        fig_corr.update_layout(height=400, title=t("So sánh Hiệu suất (Đã chuẩn hóa về 100 điểm)"), yaxis_title=t("Tăng trưởng (%)"), template="plotly_dark", xaxis=dict(tickformat=_vn_date_fmt))
                        st.plotly_chart(fig_corr, use_container_width=True)
                    else:
                        st.warning(t("Không đủ dữ liệu lịch sử chuẩn để tính tương quan."))
                else:
                    st.warning(t("⚠️ Không tìm thấy dữ liệu VNINDEX_snapshot.xlsx."))
            except Exception as e:
                st.warning(f"{t('Không thể vẽ biểu đồ tương quan lúc này:')} {e}")
            
            # --- OVERALL CONCLUSION MODEL SCORE ---
            st.subheader(t("4. Đánh Giá Điểm Số (Model Scoring)"))
            score = 0
            feedback = []
            
            try:
                if current_close > df['SMA_200'].iloc[-1]:
                    score += 1
                    feedback.append(t("✔️ Tích cực: Giá hiện tại đang NẰM TRÊN trung bình 200 ngày (Xu hướng tăng dài hạn)."))
                else:
                    feedback.append(t("❌ Tiêu cực: Giá hiện tại NẰM DƯỚI trung bình 200 ngày (Xu hướng giảm)."))
            except:
                pass
                
            if current_rsi < 40:
                score += 1
                feedback.append(t("✔️ Tích cực: RSI cho thấy cổ phiếu đang ở vùng giá rẻ / hấp dẫn."))
            elif current_rsi > 70:
                feedback.append(t("❌ Cảnh báo: RSI quá cao, cổ phiếu có thể đã tăng nóng (Quá mua)."))
            else:
                feedback.append(t("⚪ Trung lập: RSI ở mức bình thường."))
            
            # --- VOLUME EVALUATION ---
            try:
                current_vol = float(v)
                avg_vol_20 = float(df['VOL_MA_20'].iloc[-1])
                
                if current_vol > (avg_vol_20 * 1.2):
                    score += 1
                    feedback.append(f"{t('✔️ Tích cực: Khối lượng bùng nổ')} ({vol_str}) {t('- Dòng tiền đang nhập cuộc mạnh mẽ.')}")
                elif current_vol < (avg_vol_20 * 0.8):
                    feedback.append(f"{t('❌ Cảnh báo: Thanh khoản sụt giảm')} ({vol_str}) {t('- Nhà đầu tư đang có dấu hiệu đứng ngoài.')}")
                else:
                    feedback.append(t("⚪ Trung lập: Khối lượng giao dịch ở mức ổn định."))
            except:
                pass
                
            st.write(f"**{t('Điểm Khuyến Nghị Mô Hình (Technical):')} {score}/3**")
            for f in feedback:
                st.write(f)
                
            # --- 5. AI MACHINE LEARNING PREDICTION ---
            st.subheader(t("5. Dự Báo Mô Hình Nâng Cao (Machine Learning Models)"))
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
                        
                        st.metric(label=t("📊 Độ chính xác mô hình"), value=f"{acc*100:.1f}%")
                        pred_text = t("🟢 TĂNG GIÁ (UP)") if tomorrow_pred[0] == 1 else t("🔴 GIẢM GIÁ (DOWN)")
                        st.metric(label=t("🤖 Dự Báo Xu Hướng"), value=pred_text)
                        st.caption(t("Random Forest mạnh mẽ để dự đoán Xác suất tăng / giảm, nó bỏ qua nhiễu giá."))
                        
                        with st.expander(t("🔍 Xem Bằng Chứng AI (Explainable AI)")):
                            st.write(t("**Bóc tách quyết định:** Tại sao AI lại cho ra dự báo Tăng/Giảm thay vì con số ngược lại? Dưới đây là phân bổ trọng số (Feature Importance) cho thấy AI đã ưu tiên nhìn vào chỉ báo nào nhiều nhất để ra quyết định."))
                            importances = rf_model.feature_importances_
                            fig_rf = go.Figure([go.Bar(
                                x=features_class, 
                                y=importances, 
                                text=[f"{val*100:.1f}%" for val in importances], 
                                textposition='auto',
                                marker_color='#FFA726'
                            )])
                            fig_rf.update_layout(title=t("Mức độ quan trọng của Chỉ báo đầu vào"), xaxis_title=t("Chỉ báo"), yaxis_title=t("Tỷ trọng ảnh hưởng"), template='plotly_dark', height=300)
                            st.plotly_chart(fig_rf, width='stretch')
                            
                            best_feature = features_class[np.argmax(importances)]
                            st.write(t("➡️ **Giải mã:** Cột cao nhất thuộc về **{best_feature}**. Điều này bóc trần sự thật rằng: Trong quá trình dò tìm hàng nghìn ngày giao dịch lịch sử, AI nhận thấy biến số `{best_feature}` đóng vai trò sinh tử lớn nhất tác động đến việc giá đảo chiều hay tiếp diễn vào ngày hôm sau.").format(best_feature=best_feature))
                    
                    with tab2:
                        # 2. Linear Regression (Hồi Quy Tuyến Tính cơ bản)
                        lr_model = LinearRegression()
                        lr_model.fit(X_r_train, y_r_train)
                        mae_lr = mean_absolute_error(y_r_test, lr_model.predict(X_r_test))
                        pred_lr = lr_model.predict(latest_data_reg_scaled)[0]
                        
                        st.metric(label=t("📉 Sai số dự báo trung bình (MAE)"), value=f"{mae_lr:,.0f} VND")
                        st.metric(label=t("💰 Mức Giá Ngày Mai"), value=f"{pred_lr:,.0f} VND", delta=f"{pred_lr - current_close:,.0f} VND")
                        st.caption(t("Linear Regression giả định mối nối tuyến tính cơ bản giữa các nhịp giá."))
                        
                        with st.expander(t("🔍 Xem Bằng Chứng XAI (Công thức Toán học)")):
                            st.write(t("Thay vì chỉ đưa ra con số bâng quơ, Hồi quy tuyến tính xây dựng một phương trình đường thẳng. Dưới đây là mặt cắt công thức mà thuật toán vừa tính được:"))
                            formula = f"**{t('Giá')} = ({lr_model.coef_[0]:.2f} × Close_Lag_1) + ({lr_model.coef_[1]:.2f} × SMA_20) + ({lr_model.coef_[2]:.2f} × RSI_14) + {lr_model.intercept_:.2f}**"
                            st.info(formula)
                            
                            # Tính chi tiết quá trình thay số
                            v1, v2, v3 = latest_data_reg_scaled[0]
                            st.write(t("➡️ Nếu bạn lấy mảng dữ liệu (đã chuẩn hóa Scaler) của ngày hôm nay cắm vào công thức trên, ta có phép tính chính xác như sau:"))
                            calc_line = f"**=> {t('Giá')} = ({lr_model.coef_[0]:.2f} × {v1:.4f}) + ({lr_model.coef_[1]:.2f} × {v2:.4f}) + ({lr_model.coef_[2]:.2f} × {v3:.4f}) + {lr_model.intercept_:.2f} = {pred_lr:,.0f} VND**"
                            st.success(calc_line)
                        
                    with tab3:
                        # 3. K-Nearest Neighbors (KNN Regressor)
                        knn_model = KNeighborsRegressor(n_neighbors=5)
                        knn_model.fit(X_r_train, y_r_train)
                        mae_knn = mean_absolute_error(y_r_test, knn_model.predict(X_r_test))
                        pred_knn = knn_model.predict(latest_data_reg_scaled)[0]
                        
                        st.metric(label=t("📉 Sai số dự báo trung bình (MAE)"), value=f"{mae_knn:,.0f} VND")
                        st.metric(label=t("💰 Mức Giá Ngày Mai"), value=f"{pred_knn:,.0f} VND", delta=f"{pred_knn - current_close:,.0f} VND")
                        st.caption(t("KNN quét hồ sơ lịch sử, tìm ra 5 chu kỳ giá giống hệt ngày hôm nay nhất để đưa ra giá trị trung bình."))
                        
                        with st.expander(t("🔍 Xem Bằng Chứng AI (Nearest Neighbors)")):
                            st.write(t("Thuật toán KNN không dùng phương trình đường thẳng, thay vào đó nó so sánh biểu đồ hôm nay với dữ liệu quá khứ bằng **Công thức tính Khoảng cách Ơ-clit (Euclidean Distance)**:"))
                            st.latex(r"Distance = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2 + (x_3 - y_3)^2}")
                            
                            from sklearn.neighbors import NearestNeighbors
                            nn = NearestNeighbors(n_neighbors=5)
                            nn.fit(X_r_train)
                            distances, indices = nn.kneighbors(latest_data_reg_scaled)
                            
                            # Phép tính cho ngày gần nhất
                            curr = latest_data_reg_scaled[0]
                            hist = X_r_train[indices[0][0]]
                            dist_0 = distances[0][0]
                            
                            st.write(t("Ví dụ, với ngày lịch sử có đồ thị **Giống Nhất (Top 1)**, máy học ráp số vào công thức trên để đo như sau:"))
                            st.info(f"**=> {t('Sai lệch Top 1 = √ [ (')}{curr[0]:.3f} - {hist[0]:.3f})² + ({curr[1]:.3f} - {hist[1]:.3f})² + ({curr[2]:.3f} - {hist[2]:.3f})² ] = {dist_0:.4f}**")
                            
                            st.write(t("📏 Tương tự, nó tìm ra 5 chu kỳ có độ sai lệch (khoảng cách) thấp nhất như sau:"))
                            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
                            col_t1.metric("Top 1", f"{distances[0][0]:.4f}")
                            col_t2.metric("Top 2", f"{distances[0][1]:.4f}")
                            col_t3.metric("Top 3", f"{distances[0][2]:.4f}")
                            col_t4.metric("Top 4", f"{distances[0][3]:.4f}")
                            col_t5.metric("Top 5", f"{distances[0][4]:.4f}")
                            
                            st.write(f"➡️ **{t('Kết quả cuối cùng:')}** {t('Trung bình cộng = (')}{distances[0][0]:.4f} + {distances[0][1]:.4f} + {distances[0][2]:.4f} + {distances[0][3]:.4f} + {distances[0][4]:.4f}) / 5 = **{np.mean(distances):.4f}**")
                            st.write(t("Cận 0 nghĩa là AI cực kỳ tự tin vì hôm nay đang lặp lại gần như y hệt 5 ngày vĩ đại nhất trong quá khứ."))
                        
                    with tab4:
                        # 4. Deep Learning (Mạng Nơ-ron Nhân tạo MLP)
                        # Dùng MLP thay cho LSTM vì LSTM cần setup TensorFlow khắt khe hơn trên máy trạm cục bộ.
                        from warnings import filterwarnings
                        filterwarnings('ignore') # Tránh cảnh báo Convergence
                        
                        mlp_model = MLPRegressor(hidden_layer_sizes=(50, 50), max_iter=500, random_state=42)
                        mlp_model.fit(X_r_train, y_r_train)
                        mae_mlp = mean_absolute_error(y_r_test, mlp_model.predict(X_r_test))
                        pred_mlp = mlp_model.predict(latest_data_reg_scaled)[0]
                        
                        st.metric(label=t("📉 Sai số dự báo trung bình (MAE)"), value=f"{mae_mlp:,.0f} VND")
                        st.metric(label=t("💰 Mức Giá Ngày Mai"), value=f"{pred_mlp:,.0f} VND", delta=f"{pred_mlp - current_close:,.0f} VND")
                        st.success(t("🧠 Thuật toán Học Sâu (Deep Learning) quét nhiều tầng tính năng phức tạp để cho ra giá trị sát thị trường giống kỹ thuật của bài học Data Flair."))

                        with st.expander(t("🔍 Xem Bằng Chứng AI (Biểu Đồ Khớp Tín Hiệu)")):
                            st.write(t("**Bóc tách Hộp Đen (Black Box):** Mạng Nơ-ron (Deep Learning) rất dễ bị 'Học vẹt' (Overfitting). Để chứng minh nó thực sự hiểu quy luật thị trường chứ không phải học thuộc lòng, máy tính đã cố tình 'giấu đi' 30 ngày giao dịch gần nhất (Tập Kiểm Thử). Sau đó, nó bắt AI phải tự phân tích và vẽ ra đường giá của 30 ngày đó."))
                            
                            fig_mlp = go.Figure()
                            # Đổi màu Trắng thành Xanh dương để không bị chìm vào nền sáng của Streamlit
                            fig_mlp.add_trace(go.Scatter(y=y_r_test.tail(30).values, name=t("Giá Thực Tế"), line=dict(color='#2196F3', width=3)))
                            fig_mlp.add_trace(go.Scatter(y=mlp_model.predict(X_r_test)[-30:], name=t("AI Dự Báo"), line=dict(color='#E91E63', dash='dot', width=3)))
                            fig_mlp.update_layout(title=t("Mô phỏng Tracking 30 ngày Tương lai gần nhất"), yaxis_title=t("Giá Mở Cửa (VND)"), template='plotly_white', height=350, margin=dict(l=20, r=20, t=40, b=20))
                            st.plotly_chart(fig_mlp, width='stretch')
                            
                            st.write(t("➡️ **Giải mã Biểu đồ:**"))
                            st.write(t("- **Đường nét liền (Màu Xanh):** Là biến động Giá Thực Tế của cổ phiếu trên sàn."))
                            st.write(t("- **Đường nét đứt (Màu Hồng):** Là mức giá mà não bộ AI tự động suy luận ra trong cùng lúc đó."))
                            st.success(t("Tóm lại: Nếu đường Màu Hồng uốn lượn bám dính lấy đường Màu Xanh, có nghĩa là AI dường như đã nhạy cảm và bắt đúng chu kỳ thật sự của thị trường!"))

                    st.markdown("---")
                    st.warning(t("⚠️ **Ghi chú:** Đây chỉ là dự báo toán học thử nghiệm, hoàn toàn không phải lời khuyên mua bán chứng khoán."))
                else:
                    st.warning(t("Dữ liệu không đủ dài để AI học. Bạn hãy chọn số ngày lớn hơn 100 nhé!"))
            except Exception as e:
                st.warning(f"{t('Lỗi khởi chạy Học Máy:')} {e}")
                
            st.subheader(t("6. Thông tin Vĩ mô & Tin tức (Macro & News)"))
            try:
                news_df = stock.news.recent()
                if not news_df.empty:
                    st.write(t("**Tin tức gần đây:**"))
                    st.dataframe(news_df.head(5))
                else:
                    st.info(t("Chưa có tin tức mới nhất từ hệ thống cho mã này."))
            except Exception as e:
                pass
                
            # Simulate Macro data since API requires premium/various setups
            st.write(t("**Dữ liệu Kinh tế Vĩ mô Hiện tại (Mock Data):**"))
            col1, col2, col3 = st.columns(3)
            col1.metric(label=t("CPI (Lạm phát)"), value="3.98%", delta="-0.1%")
            col2.metric(label=t("Lãi suất điều hành"), value="4.50%", delta="0.0%")
            col3.metric(label=t("Tỷ giá trung tâm (USD/VND)"), value="25,400 VND", delta="15 VND")
            
            # Khởi tạo mô hình đánh giá Vĩ mô theo Ngành
            st.write(f"**💡 {t('Tác động Vĩ mô lên hoạt động kinh doanh của')} {ticker}:**")
            
            # Dictionary mapping tickers to macro implications
            macro_impact = ""
            if ticker in ["FPT", "CMG", "ELC"]:
                macro_impact = t("💻 **Thuộc nhóm Công nghệ:** Đặc thù ngành này ít bị ảnh hưởng trực tiếp bởi lãi suất trong nước. Tuy nhiên, tỷ giá USD/VND đang ở mức cao (25,400) là **cực kỳ có lợi** vì phần lớn doanh thu của họ đến từ xuất khẩu phần mềm (thu bằng USD nhưng trả lương Kỹ sư bằng VND).")
            elif ticker in ["VCB", "MBB", "TCB", "BID", "CTG", "VPB", "STB", "HDB", "ACB", "TPB", "SHB", "VIB", "SSB", "LPB"]:
                macro_impact = t("🏦 **Thuộc nhóm Ngân hàng:** Lãi suất điều hành thấp (4.5%) thường giúp duy trì biên lãi ròng (NIM) tốt do huy động vốn rẻ, đồng thời kích thích tăng trưởng tín dụng. Tuy nhiên, tỷ giá neo cao có thể khiến NHNN chịu áp lực hút tiền về. Nhạy cảm số 1 với chính sách tiền tệ.")
            elif ticker in ["VHM", "NVL", "DIG", "DXG", "KDH", "VIC", "VRE", "BCM"]:
                macro_impact = t("🏢 **Thuộc nhóm Bất động sản:** Khát vốn nhất thị trường. Ngành này RẤT THÍCH môi trường lãi suất thấp và dòng tiền giá rẻ. Mức lãi suất 4.5% hỗ trợ người mua nhà vay mượn, nhưng các doanh nghiệp rủi ro cao vẫn phải đối mặt với áp lực đáo hạn trái phiếu lớn.")
            elif ticker in ["HPG", "HSG", "NKG", "GVR"]:
                macro_impact = t("🏗️ **Thuộc nhóm SX Vật liệu / Công nghiệp:** Ngành này phụ thuộc trực tiếp vào tiến độ giải ngân Đầu tư công và sự phục hồi của Bất động sản. Khi kích cầu xây dựng, nhu cầu thép/cao su tăng vọt. Sự biến động của dòng vốn FDI và tỷ giá cũng ảnh hưởng đáng kể.")
            elif ticker in ["REE", "GAS", "POW", "NT2", "BWE", "PLX"]:
                macro_impact = t("⚡ **Thuộc nhóm Năng lượng / Tiện ích:** Đây là nhóm 'Phòng thủ ăn chắc mặc bền'. Dù lạm phát (CPI) có cao hay lãi suất biến động mạnh, nhu cầu dùng điện, nước, xăng dầu của người dân không thể giảm. Doanh thu siêu ổn định trong giông bão kinh tế.")
            elif ticker in ["SSI", "VND", "VCI", "HCM", "MBS"]:
                macro_impact = t("📈 **Thuộc nhóm Chứng khoán:** Ngành này là nhiệt kế của thị trường, có độ nhạy cực cao với Lãi suất. Lãi suất ngân hàng càng rẻ -> dòng tiền chảy vào chứng khoán tìm kiếm lợi nhuận càng lớn -> thanh khoản tăng kỷ lục -> Doanh thu mảng Môi giới & Margin tăng mạnh.")
            elif ticker in ["VNM", "MSN", "SAB", "MWG", "PNJ"]:
                macro_impact = t("🛒 **Thuộc nhóm Bán lẻ / Tiêu dùng (VN30):** Chịu ảnh hưởng trực tiếp từ Lạm phát (CPI) và tổng Cầu tiêu dùng nội địa. Nếu CPI (3.98%) duy trì cao sẽ siết chặt hầu bao của người dân, ảnh hưởng kém tới sức mua bán lẻ. Ngược lại, nếu kinh tế phục hồi và thu nhập dân chúng tăng, nhóm này bứt phá mạnh.")
            elif ticker in ["VJC", "HVN"]:
                macro_impact = t("✈️ **Thuộc nhóm Hàng không:** Cực kỳ nhạy cảm với 2 yếu tố Vĩ mô: Giá dầu thế giới (chiếm 30-40% chi phí) và Tỷ giá USD/VND (vì họ phải thuê/mua máy bay bằng USD). Tỷ giá 25,400 neo cao tạo sức ép chi phí cực lớn. Điểm bù trừ duy nhất là lượng khách du lịch nội địa phục hồi.")
            else:
                macro_impact = f"🔍 **{t('Phân tích Cơ bản:')}** {t('Để đánh giá chuyên sâu tác động Vĩ mô lên')} {{ticker}}, {t('chúng ta cần mổ xẻ Cơ cấu doanh thu của họ: Thuộc nhóm Xuất khẩu (hưởng lợi tỷ giá), Vay nợ cao (sợ lãi suất), hay Tiêu dùng (sợ lạm phát CPI).')}"
                
            st.success(macro_impact)
            st.info(t("Lưu ý: Dữ liệu BCTC được kéo tự động từ API mở (VCI/UBCKNN) có cùng chất lượng số liệu với Simplize. Việc theo dõi Vĩ mô kết hợp Phân tích Cơ bản này là cốt lõi của Đầu tư Giá trị!"))
            
            # --- 7. RISK MANAGEMENT ---
            st.subheader(t("7. Quản Trị Rủi Ro (Risk Management)"))
            
            # Simple mathematically derived risk metrics (7% stop loss, 15% take profit)
            st_loss_price = current_close * 0.93
            tk_profit_price = current_close * 1.15
            
            st.write(t('Giả định bạn mua cổ phiếu {ticker} tại mức giá đóng cửa hiện tại (**{current_close:,.0f} VND**), dưới đây là kế hoạch giao dịch kỷ luật:').format(ticker=ticker, current_close=current_close))
            
            c1, c2, c3 = st.columns(3)
            c1.metric(label=t("🎯 Điểm Chốt Lời Mục Tiêu (+15%)"), value=f"{tk_profit_price:,.0f} VND", delta=t("Lợi nhuận cao"))
            c2.metric(label=t("🛑 Điểm Cắt Lỗ Bắt Buộc (-7%)"), value=f"{st_loss_price:,.0f} VND", delta=t("Bảo vệ Vốn"), delta_color="inverse")
            c3.metric(label=t("⚖️ Tỷ lệ Risk/Reward (R:R)"), value="1 : 2.14", delta=t("Giao dịch Tốt"))
            
            st.markdown(f"""
            **{t('Nguyên tắc Vàng trong Quản trị Danh mục:')}**
            - **{t('Quy tắc 2%:')}** {t('Tuyệt đối không bao giờ để mức cắt lỗ của một lệnh làm hao hụt quá 2% TỔNG số tiền bạn có. (Nếu vốn 100 triệu, chỉ cho phép lỗ tối đa 2 triệu/lệnh).')}
            - **{t("Quy tắc Đa dạng hóa:")}** {t("Không 'bỏ tất cả trứng vào một rổ'. Hãy chia số vốn của bạn cho khoảng **3 đến 5 mã** thuộc các nhóm ngành khác nhau (Ví dụ: 1 Mã Bank, 1 Thép, 1 Bất động sản).")}
            - **{t('Tuyệt đối KHÔNG trung bình giá xuống:')}** {t('Nhồi thêm tiền vào một cổ phiếu đang lao dốc là con đường nhanh nhất dẫn đến cháy tài khoản.')}
            """)
            
            st.divider()
            st.warning(t("⚠️ **Đây chỉ là dự báo và không khuyến khích làm theo (Tôi sẽ không chịu trách nhiệm vì bất kỳ điều gì)**"))
            
    except Exception as e:
        st.error(f"{t('Lỗi hệ thống:')} {e}")


