import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score

# ----------------------------------------------------------------------
# 0. Cấu hình chung
# ----------------------------------------------------------------------
np.random.seed(42)
plt.rcParams["figure.figsize"] = (8, 5)


# ----------------------------------------------------------------------
# 1. Tạo dữ liệu giả
# ----------------------------------------------------------------------
def generate_data(n_samples=30, noise_std=0.15):
    """Tạo dữ liệu phi tuyến y = sin(x) + nhiễu, số mẫu nhỏ để dễ overfit."""
    X = np.sort(np.random.uniform(-3, 3, n_samples))
    y_true = np.sin(X)
    y = y_true + np.random.normal(0, noise_std, size=X.shape)
    return X, y


X, y = generate_data(n_samples=30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Reshape cho sklearn (cần input 2D)
X_train_2d = X_train.reshape(-1, 1)
X_test_2d = X_test.reshape(-1, 1)

print(f"Số mẫu train: {len(X_train)}, số mẫu test: {len(X_test)}")


# ----------------------------------------------------------------------
# Hàm tiện ích: fit + đánh giá + vẽ
# ----------------------------------------------------------------------
def fit_poly_model(degree, alpha=0.0):
    """Fit polynomial regression (LinearRegression nếu alpha=0, ngược lại Ridge).
    Dùng StandardScaler sau PolynomialFeatures để tránh mất ổn định số học
    khi degree cao (các cột x^k có độ lớn rất chênh lệch)."""
    poly = PolynomialFeatures(degree=degree)
    Xtr_poly_raw = poly.fit_transform(X_train_2d)
    Xte_poly_raw = poly.transform(X_test_2d)

    scaler = StandardScaler()
    Xtr_poly = scaler.fit_transform(Xtr_poly_raw)
    Xte_poly = scaler.transform(Xte_poly_raw)

    if alpha == 0.0:
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)

    model.fit(Xtr_poly, y_train)

    train_pred = model.predict(Xtr_poly)
    test_pred = model.predict(Xte_poly)

    train_mse = mean_squared_error(y_train, train_pred)
    test_mse = mean_squared_error(y_test, test_pred)
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    return (model, poly, scaler), train_mse, test_mse, train_r2, test_r2


def plot_fit(model_tuple, title, filename):
    """Vẽ dữ liệu train/test + đường fit của mô hình."""
    model, poly, scaler = model_tuple
    x_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
    x_plot_poly = scaler.transform(poly.transform(x_plot))
    y_plot = model.predict(x_plot_poly)

    plt.figure()
    plt.scatter(X_train, y_train, color="royalblue", label="Train data", zorder=3)
    plt.scatter(X_test, y_test, color="orange", label="Test data", zorder=3)
    plt.plot(x_plot, np.sin(x_plot), "g--", alpha=0.6, label="True function sin(x)")
    plt.plot(x_plot, y_plot, "r-", linewidth=2, label="Model fit")
    plt.ylim(-3, 3)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close()
    print(f"Đã lưu hình: {filename}")


# ----------------------------------------------------------------------
# 2. Baseline: bậc 1 (thường sẽ UNDERFIT vì dữ liệu phi tuyến)
# ----------------------------------------------------------------------
print("\n=== Bước 2: Baseline (degree=1) ===")
mt1, tr_mse1, te_mse1, tr_r2_1, te_r2_1 = fit_poly_model(degree=1)
print(f"Train MSE: {tr_mse1:.4f} | Test MSE: {te_mse1:.4f}")
print(f"Train R2 : {tr_r2_1:.4f} | Test R2 : {te_r2_1:.4f}")
plot_fit(mt1, "Baseline - Degree 1 (Underfit)", "01_baseline_underfit.png")


# ----------------------------------------------------------------------
# 3. Gây OVERFITTING: bậc cao, không regularize
# ----------------------------------------------------------------------
print("\n=== Bước 3: Overfitting (degree=18, no regularization) ===")
OVERFIT_DEGREE = 18
mt_over, tr_mse_o, te_mse_o, tr_r2_o, te_r2_o = fit_poly_model(
    degree=OVERFIT_DEGREE, alpha=0.0
)
print(f"Train MSE: {tr_mse_o:.4f} | Test MSE: {te_mse_o:.4f}")
print(f"Train R2 : {tr_r2_o:.4f} | Test R2 : {te_r2_o:.4f}")
print(">> Nhận xét: Train MSE rất thấp / R2 rất cao nhưng Test MSE cao hơn nhiều")
print(">> => Mô hình học thuộc nhiễu của tập train, không tổng quát hóa được.")
plot_fit(
    mt_over,
    f"Overfit - Degree {OVERFIT_DEGREE} (No Regularization)",
    "02_overfit.png",
)


# ----------------------------------------------------------------------
# 4. Trực quan hóa overfitting bằng Learning Curve theo degree
# ----------------------------------------------------------------------
print("\n=== Bước 4: Learning curve theo degree (1 -> 20) ===")
degrees = list(range(1, 21))
train_mses, test_mses = [], []

for d in degrees:
    _, tr_mse, te_mse, _, _ = fit_poly_model(degree=d, alpha=0.0)
    train_mses.append(tr_mse)
    test_mses.append(te_mse)

plt.figure()
plt.plot(degrees, train_mses, "o-", label="Train MSE")
plt.plot(degrees, test_mses, "o-", label="Test MSE")
plt.yscale("log")
plt.xlabel("Polynomial Degree")
plt.ylabel("MSE (log scale)")
plt.title("Learning Curve: Train vs Test MSE theo Degree")
plt.legend()
plt.tight_layout()
plt.savefig("03_learning_curve.png", dpi=120)
plt.close()
print("Đã lưu hình: 03_learning_curve.png")
print(">> Nhận xét: Degree càng tăng, Train MSE càng giảm nhưng Test MSE")
print(">> giảm rồi tăng trở lại (hình chữ U) -> dấu hiệu overfitting rõ ràng.")


# ----------------------------------------------------------------------
# 5. Sửa thành GOOD FIT
#    Cách 1: Chọn degree tối ưu bằng cross-validation
#    Cách 2: Regularization (Ridge) trên degree cao
# ----------------------------------------------------------------------
print("\n=== Bước 5a: Chọn degree tối ưu bằng Cross-Validation ===")
from sklearn.pipeline import make_pipeline

cv_scores = []
for d in degrees:
    pipe = make_pipeline(
        PolynomialFeatures(degree=d), StandardScaler(), LinearRegression()
    )
    scores = cross_val_score(
        pipe, X_train_2d, y_train,
        scoring="neg_mean_squared_error", cv=5
    )
    cv_scores.append(-scores.mean())

best_degree = degrees[int(np.argmin(cv_scores))]
print(f"Degree tối ưu theo CV: {best_degree}")

mt_best, tr_mse_b, te_mse_b, tr_r2_b, te_r2_b = fit_poly_model(
    degree=best_degree, alpha=0.0
)
print(f"Train MSE: {tr_mse_b:.4f} | Test MSE: {te_mse_b:.4f}")
print(f"Train R2 : {tr_r2_b:.4f} | Test R2 : {te_r2_b:.4f}")
plot_fit(
    mt_best,
    f"Good Fit - Degree tối ưu ({best_degree}) qua Cross-Validation",
    "04_goodfit_cv_degree.png",
)

print("\n=== Bước 5b: Regularization (Ridge) trên degree cao ===")
RIDGE_ALPHA = 5.0
mt_ridge, tr_mse_r, te_mse_r, tr_r2_r, te_r2_r = fit_poly_model(
    degree=OVERFIT_DEGREE, alpha=RIDGE_ALPHA
)
print(f"Train MSE: {tr_mse_r:.4f} | Test MSE: {te_mse_r:.4f}")
print(f"Train R2 : {tr_r2_r:.4f} | Test R2 : {te_r2_r:.4f}")
plot_fit(
    mt_ridge,
    f"Good Fit - Degree {OVERFIT_DEGREE} + Ridge (alpha={RIDGE_ALPHA})",
    "05_goodfit_ridge.png",
)


# ----------------------------------------------------------------------
# 6. Bảng so sánh tổng kết
# ----------------------------------------------------------------------
print("\n=== Bước 6: Bảng so sánh tổng kết ===")
results = [
    ("Underfit (degree=1)", tr_mse1, te_mse1, tr_r2_1, te_r2_1),
    (f"Overfit (degree={OVERFIT_DEGREE}, no reg.)", tr_mse_o, te_mse_o, tr_r2_o, te_r2_o),
    (f"Good fit (degree={best_degree}, CV)", tr_mse_b, te_mse_b, tr_r2_b, te_r2_b),
    (f"Good fit (degree={OVERFIT_DEGREE}, Ridge)", tr_mse_r, te_mse_r, tr_r2_r, te_r2_r),
]

header = f"{'Model':<32}{'Train MSE':>12}{'Test MSE':>12}{'Train R2':>12}{'Test R2':>12}"
print(header)
print("-" * len(header))
for name, tr_mse, te_mse, tr_r2, te_r2 in results:
    print(f"{name:<32}{tr_mse:>12.4f}{te_mse:>12.4f}{tr_r2:>12.4f}{te_r2:>12.4f}")

# Biểu đồ so sánh 3 đường fit trên cùng 1 hình
plt.figure()
x_plot = np.linspace(-3.5, 3.5, 300).reshape(-1, 1)
plt.scatter(X_train, y_train, color="royalblue", label="Train data", zorder=3, alpha=0.7)
plt.scatter(X_test, y_test, color="orange", label="Test data", zorder=3, alpha=0.7)
plt.plot(x_plot, np.sin(x_plot), "g--", alpha=0.5, label="True function")

for model_tuple, label, style in [
    (mt1, "Underfit (deg=1)", "b:"),
    (mt_over, f"Overfit (deg={OVERFIT_DEGREE})", "r-"),
    (mt_best, f"Good fit (deg={best_degree}, CV)", "m-"),
]:
    model, poly, scaler = model_tuple
    y_plot = model.predict(scaler.transform(poly.transform(x_plot)))
    plt.plot(x_plot, y_plot, style, linewidth=2, label=label)

plt.ylim(-3, 3)
plt.title("So sánh: Underfit vs Overfit vs Good Fit")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()
plt.savefig("06_comparison.png", dpi=120)
plt.close()
print("\nĐã lưu hình: 06_comparison.png")
print("\nHoàn tất! Tất cả hình ảnh (.png) đã được lưu trong thư mục hiện tại.")