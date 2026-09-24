

import numpy as np


def sgn(z):
    """Ham dau: tra ve +1 neu z >= 0, nguoc lai -1 (dung theo dinh nghia
    label(x) = sgn(w^T x) trong slide Perceptron)."""
    return 1 if z >= 0 else -1


def main():
    w = np.array([1, 2, -10])
    x = np.array([3, 4, 1])
    y_true = -1

    # 1) Tinh w^T x
    score = np.dot(w, x)
    print(f"w = {w.tolist()}")
    print(f"x = {x.tolist()}")
    print(f"1) w^T x = (1)(3) + (2)(4) + (-10)(1) = {score}")

    # 2) Nhan du bao
    y_pred = sgn(score)
    print(f"2) Nhan du bao: sgn({score}) = {y_pred}")

    # 3) Kiem tra phan lop sai
    print(f"3) Nhan thuc te y = {y_true}")
    if y_pred != y_true:
        print(f"   -> y_pred ({y_pred}) != y_true ({y_true})")
        print(f"   -> DIEM DU LIEU BI PHAN LOP SAI (misclassified)")
    else:
        print(f"   -> y_pred ({y_pred}) == y_true ({y_true})")
        print(f"   -> Diem du lieu duoc phan lop dung")


if __name__ == "__main__":
    main()
