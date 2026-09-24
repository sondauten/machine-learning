

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# ============================================================
# Lop Perceptron (giong bai 3.29, nhung nam san trong file nay)
# ============================================================
class Perceptron:
    def __init__(self, learning_rate: float = 1.0, max_iters: int = 1000,
                 add_bias: bool = True, random_state: int = None):
        self.eta = learning_rate
        self.max_iters = max_iters
        self.add_bias = add_bias
        self.random_state = random_state
        self.w = None
        self.n_iters_run_ = 0
        self.converged_ = False

    def _add_bias_col(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([X, ones])

    def _misclassified_indices(self, X, y):
        scores = X @ self.w
        y_pred = np.where(scores >= 0, 1.0, -1.0)
        return np.where(y_pred != y)[0]

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.add_bias:
            X = self._add_bias_col(X)

        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        self.w = rng.randn(n_features) * 0.01

        t = 0
        while t < self.max_iters:
            mis_idx = self._misclassified_indices(X, y)
            if len(mis_idx) == 0:
                break

            i = rng.choice(mis_idx)
            xi, yi = X[i], y[i]
            self.w = self.w + self.eta * yi * xi
            t += 1

        self.n_iters_run_ = t
        self.converged_ = len(self._misclassified_indices(X, y)) == 0
        return self

    def _decision_scores(self, X):
        X = np.asarray(X, dtype=float)
        if self.add_bias:
            X = self._add_bias_col(X)
        return X @ self.w

    def predict(self, X):
        scores = self._decision_scores(X)
        return np.where(scores >= 0, 1, -1)


# ============================================================
# Ap dung vao bai toan phan lop nhi phan
# ============================================================
def main():
    # 1) Doc du lieu
    data = load_breast_cancer()
    X, y_raw = data.data, data.target  # y goc: 0 = malignant, 1 = benign

    # Perceptron can nhan {-1, +1} -> chuyen doi
    y = np.where(y_raw == 1, 1, -1)

    print(f"So mau: {X.shape[0]}, so dac trung: {X.shape[1]}")
    print(f"Phan bo lop: benign(+1)={ (y==1).sum() }, malignant(-1)={ (y==-1).sum() }")

    # 2) Chia tap train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # 3) Chuan hoa du lieu (quan trong voi Perceptron vi thuat toan nhay
    #    voi thang do cua dac trung)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4) Huan luyen Perceptron
    model = Perceptron(learning_rate=1.0, max_iters=2000, random_state=42)
    model.fit(X_train_scaled, y_train)

    print(f"\nDa hoi tu (khong con diem loi tren tap train): {model.converged_}")
    print(f"So vong lap (t) da chay: {model.n_iters_run_}")

    # 5) Du bao va danh gia
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred, labels=[-1, 1])

    print("\n=== KET QUA DANH GIA TREN TAP TEST ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-score : {f1:.4f}")
    print("\nMa tran nham lan (hang=thuc te, cot=du bao), thu tu lop [-1, +1]:")
    print(cm)


if __name__ == "__main__":
    main()
