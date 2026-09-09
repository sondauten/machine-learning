import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Đọc dữ liệu
data = pd.read_csv("house_prices.csv")

# 2. Chọn đặc trưng (X) và giá nhà (y)
X = data[["area_m2", "bedrooms", "distance_km"]]
y = data["price_billion"]

# 3. Chia dữ liệu để kiểm tra mô hình
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Tạo và huấn luyện mô hình hồi quy tuyến tính
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Đánh giá
predicted = model.predict(X_test)
print("MAE:", round(mean_absolute_error(y_test, predicted), 3), "tỷ")
print("R² :", round(r2_score(y_test, predicted), 3))

# 6. Xem công thức mà mô hình học được
print("\nCông thức:")
print(
    f"Giá = {model.coef_[0]:.4f} * diện_tích "
    f"+ {model.coef_[1]:.4f} * số_phòng "
    f"+ {model.coef_[2]:.4f} * khoảng_cách "
    f"+ {model.intercept_:.4f}"
)

# 7. Nhập thông tin căn nhà muốn dự đoán
area = float(input("\nDiện tích (m²): "))
bedrooms = int(input("Số phòng ngủ: "))
distance = float(input("Khoảng cách tới trung tâm (km): "))

new_house = pd.DataFrame([{
    "area_m2": area,
    "bedrooms": bedrooms,
    "distance_km": distance
}])

prediction = model.predict(new_house)[0]

print(f"\n>>> Giá nhà dự đoán: {prediction:.2f} tỷ VNĐ")
