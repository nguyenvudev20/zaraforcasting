# main.py - Ứng dụng Streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(layout="wide")
st.title("📈 Zara Sales Forecasting App")

# 1. Load dữ liệu
st.subheader("1. Tải và tiền xử lý dữ liệu")
df = pd.read_csv("https://raw.githubusercontent.com/nguyenvudev20/zaraforcasting/refs/heads/main/zara.csv", delimiter=";")

df = df.dropna(subset=['name', 'description'])
df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
df['Promotion'] = df['Promotion'].str.strip().str.lower()
df['Seasonal'] = df['Seasonal'].str.strip().str.lower()
df['Product Category'] = df['Product Category'].str.strip().str.title()
df['section'] = df['section'].str.strip().str.upper()
df['price_category'] = pd.qcut(df['price'], q=3, labels=['low', 'medium', 'high'])
df['scrape_month'] = df['scraped_at'].dt.month
df['scrape_dayofweek'] = df['scraped_at'].dt.dayofweek

st.write("✅ Dữ liệu đã được tiền xử lý. Dưới đây là vài dòng đầu:")
st.dataframe(df.head())

# 2. Trực quan hóa dữ liệu
st.subheader("2. Trực quan hóa dữ liệu")

# Phân phối giá
fig1, ax1 = plt.subplots()
sns.histplot(df['price'], bins=20, kde=True, ax=ax1)
ax1.set_title("Phân phối giá sản phẩm")
st.pyplot(fig1)

# Sản phẩm khuyến mãi
fig2, ax2 = plt.subplots()
df['Promotion'].value_counts().plot(kind='bar', ax=ax2)
ax2.set_title("Sản phẩm có/không khuyến mãi")
st.pyplot(fig2)

# Doanh số theo mức giá
fig3, ax3 = plt.subplots()
df.groupby('price_category')['Sales Volume'].mean().plot(kind='bar', color='orange', ax=ax3)
ax3.set_title("Doanh số trung bình theo mức giá")
st.pyplot(fig3)

# Sản phẩm theo ngày thu thập
fig4, ax4 = plt.subplots()
df['scraped_at'].dt.date.value_counts().sort_index().plot(marker='o', ax=ax4)
ax4.set_title("Sản phẩm theo ngày thu thập")
ax4.set_xlabel("Ngày")
ax4.set_ylabel("Số lượng")
st.pyplot(fig4)

# Heatmap
fig5, ax5 = plt.subplots(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm", ax=ax5)
ax5.set_title("Ma trận tương quan đặc trưng số")
st.pyplot(fig5)

# 3. Huấn luyện mô hình
st.subheader("3. Huấn luyện mô hình dự đoán doanh số")

features = ['price', 'Promotion', 'Product Position', 'Seasonal', 'section', 'price_category', 'scrape_month', 'scrape_dayofweek']
X = pd.get_dummies(df[features], drop_first=True)
y = np.log1p(df['Sales Volume'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Dự đoán & đánh giá
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

st.write(f"📉 **Mean Squared Error (MSE):** `{mse:,.2f}`")
st.write(f"📈 **R-squared Score (R²):** `{r2:.4f}`")

# Biểu đồ thực tế vs dự đoán
fig6, ax6 = plt.subplots()
ax6.scatter(y_true, y_pred, alpha=0.6)
ax6.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
ax6.set_xlabel("Giá trị thực tế")
ax6.set_ylabel("Giá trị dự đoán")
ax6.set_title("Doanh số: Thực tế vs Dự đoán")
ax6.grid(True)
st.pyplot(fig6)

# 4. Lưu mô hình
joblib.dump(model, "model.pkl")
# Tải mô hình đã huấn luyện
model = joblib.load("model.pkl")

with st.form("input_form"):
    price = st.number_input("💵 Giá sản phẩm (USD)", value=99.99)
    promo = st.selectbox("🎯 Có khuyến mãi không?", ["yes", "no"])
    position = st.selectbox("📍 Vị trí sản phẩm", ["Aisle", "End-cap", "Feature"])
    seasonal = st.selectbox("🌦 Có phải sản phẩm theo mùa không?", ["yes", "no"])
    section = st.selectbox("🧍 Mục sản phẩm", ["MAN", "WOMAN"])
    price_cat = st.selectbox("💰 Phân loại mức giá", ["low", "medium", "high"])
    month = st.slider("📆 Tháng thu thập dữ liệu", 1, 12, 2)
    dow = st.slider("📅 Thứ trong tuần (0=Thứ 2)", 0, 6, 1)
    submit = st.form_submit_button("Dự đoán")

if submit:
    input_data = {
        'price': price,
        'scrape_month': month,
        'scrape_dayofweek': dow,
        'Promotion_yes': 1 if promo == 'yes' else 0,
        'Product Position_End-cap': 1 if position == 'End-cap' else 0,
        'Product Position_Feature': 1 if position == 'Feature' else 0,
        'Seasonal_yes': 1 if seasonal == 'yes' else 0,
        'section_WOMAN': 1 if section == 'WOMAN' else 0,
        'price_category_low': 1 if price_cat == 'low' else 0,
        'price_category_medium': 1 if price_cat == 'medium' else 0,
    }

    X_input = pd.DataFrame([input_data])
    y_pred_log = model.predict(X_input)
    y_pred = np.expm1(y_pred_log)

    st.success(f"✅ Dự đoán doanh số: **{int(y_pred[0]):,} sản phẩm**")
