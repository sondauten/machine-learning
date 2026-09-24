

import numpy as np


class Perceptron:
    def __init__(self, learning_rate: float = 1.0, max_iters: int = 1000,
                 add_bias: bool = True, random_state: int = None):
        """
        learning_rate : eta trong quy tac cap nhat w = w + eta*y_i*x_i.
                         Mac dinh = 1.0, dung nhu slide (w = w + y_i*x_i).
        max_iters     : so vong lap toi da (de tranh lap vo han neu du lieu
                         khong linearly separable).
        add_bias      : True neu can tu dong them thanh phan bias (=1) vao x,
                         dung nhu quy uoc x = (1, x1, x2, ..., xd) trong slide.
        random_state  : seed de tai lap ket qua khi khoi tao w0 va khi chon
                         ngau nhien diem bi phan lop loi.
        """
        self.eta = learning_rate
        self.max_iters = max_iters
        self.add_bias = add_bias
        self.random_state = random_state
        self.w = None
        self.n_iters_run_ = 0          # so vong lap (t) da chay
        self.w_history_ = []           # luu lai w qua tung buoc (de minh hoa/ve hinh)

    def _add_bias_col(self, X):
        ones = np.ones((X.shape[0], 1))
        return np.hstack([X, ones])

    def _misclassified_indices(self, X, y):
        """Tra ve chi so cac diem dang bi phan lop sai: sgn(w^T x_i) != y_i."""
        scores = X @ self.w
        y_pred = np.where(scores >= 0, 1.0, -1.0)
        return np.where(y_pred != y)[0]

    def fit(self, X, y):
        """
        X: mang (n_samples, n_features)
        y: mang nhan, gia tri trong {-1, +1}
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        if self.add_bias:
            X = self._add_bias_col(X)

        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Buoc 1: t = 0, chon ngau nhien w0
        self.w = rng.randn(n_features) * 0.01
        self.w_history_ = [self.w.copy()]

        t = 0
        while t < self.max_iters:
            # Buoc 2: kiem tra con diem nao bi phan lop loi khong
            mis_idx = self._misclassified_indices(X, y)
            if len(mis_idx) == 0:
                break  # khong con loi -> dung thuat toan

            # Buoc 3: chon MOT diem bi phan lop loi (ngau nhien) va cap nhat
            i = rng.choice(mis_idx)
            xi, yi = X[i], y[i]
            self.w = self.w + self.eta * yi * xi
            self.w_history_.append(self.w.copy())

            # Buoc 4: t = t + 1, quay lai Buoc 2 (vong while)
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
        """Tra ve nhan du bao trong {-1, +1} cho moi dong cua X (label(x)=sgn(w^T x))."""
        scores = self._decision_scores(X)
        return np.where(scores >= 0, 1, -1)


if __name__ == "__main__":
    # Vi du nho de kiem tra nhanh: bai toan phan tach tuyen tinh don gian
    X_demo = np.array([
        [2, 3], [1, 1], [2, 1],      # lop -1
        [6, 7], [7, 8], [8, 6],      # lop +1
    ])
    y_demo = np.array([-1, -1, -1, 1, 1, 1])

    model = Perceptron(learning_rate=1.0, max_iters=1000, random_state=42)
    model.fit(X_demo, y_demo)

    print("Da hoi tu:", model.converged_)
    print("So vong lap (t) da chay:", model.n_iters_run_)
    print("Trong so cuoi cung (w, bias):", model.w)
    print("Du bao tren tap huan luyen:", model.predict(X_demo))
    print("Nhan thuc te:               ", y_demo)
