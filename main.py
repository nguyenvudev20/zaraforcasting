# --- 1. Thư viện cần thiết ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings("ignore")

# --- 2. Tải và xử lý dữ liệu ---
df = pd.read_csv("https://raw.githubusercontent.com/nguyenvudev20/zaraforcasting/refs/heads/main/zara.csv", delimiter=';')

# Tiền xử lý
df = df.dropna(subset=['name', 'description'])
df['scraped_at'] = pd.to_datetime(df['scraped_at'], errors='coerce')
df['Promotion'] = df['Promotion'].str.strip().str.lower()
df['Seasonal'] = df['Seasonal'].str.strip().str.lower()
df['Product Category'] = df['Product Category'].str.strip().str.title()
df['section'] = df['section'].str.strip().str.upper()
df['price_category'] = pd.qcut(df['price'], q=3, labels=['low', 'medium', 'high'])

# Feature Engineering
df['scrape_month'] = df['scraped_at'].dt.month
df['scrape_dayofweek'] = df['scraped_at'].dt.dayofweek

# --- 3. Trực quan hóa ---
plt.figure(figsize=(8, 5))
sns.histplot(df['price'], bins=20, kde=True)
plt.title("Phân phối giá sản phẩm")
plt.xlabel("Giá (USD)")
plt.ylabel("Số lượng")
plt.grid(True)
plt.show()

plt.figure(figsize=(6, 4))
df['Promotion'].value_counts().plot(kind='bar', color='skyblue')
plt.title("Số lượng sản phẩm có/không khuyến mãi")
plt.ylabel("Số sản phẩm")
plt.show()

plt.figure(figsize=(6, 4))
df.groupby('price_category')['Sales Volume'].mean().plot(kind='bar', color='orange')
plt.title("Doanh số trung bình theo mức giá")
plt.ylabel("Sales Volume trung bình")
plt.show()

plt.figure(figsize=(8, 4))
df['scraped_at'].dt.date.value_counts().sort_index().plot(marker='o')
plt.title("Số lượng sản phẩm theo ngày thu thập")
plt.xlabel("Ngày")
plt.ylabel("Số sản phẩm")
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
plt.title("Ma trận tương quan đặc trưng số")
plt.show()

# --- 4. Chuẩn bị dữ liệu huấn luyện ---
features = ['price', 'Promotion', 'Product Position', 'Seasonal', 'section',
            'price_category', 'scrape_month', 'scrape_dayofweek']
X = pd.get_dummies(df[features], drop_first=True)
y = np.log1p(df['Sales Volume'])

# --- 5. Train/Test split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 6. Huấn luyện mô hình XGBoost ---
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# --- 7. Dự đoán và đánh giá ---
y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)

mse = mean_squared_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print(f"📉 Mean Squared Error: {mse:.2f}")
print(f"📈 R-squared Score: {r2:.4f}")

# --- 8. Biểu đồ kết quả ---
plt.figure(figsize=(8, 5))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
plt.xlabel("Giá trị thực tế")
plt.ylabel("Giá trị dự đoán")
plt.title("So sánh doanh số thực tế và dự đoán")
plt.grid(True)
plt.show()
