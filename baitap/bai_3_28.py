

import numpy as np


def sgn(z):
    """Ham dau: tra ve +1 neu z >= 0, nguoc lai -1."""
    return 1 if z >= 0 else -1


def main():
    w = np.array([-2, 1, 0])
    x = np.array([2, 3, 1])
    y = 1

    print(f"w = {w.tolist()}")
    print(f"x = {x.tolist()}")
    print(f"y (nhan thuc te) = {y}")
    print()

    # 1) Kiem tra phan lop sai
    score = np.dot(w, x)
    y_pred = sgn(score)
    print(f"1) w^T x = (-2)(2) + (1)(3) + (0)(1) = {score}")
    print(f"   Nhan du bao: sgn({score}) = {y_pred}")

    is_misclassified = (y_pred != y)
    if is_misclassified:
        print(f"   -> y_pred ({y_pred}) != y ({y}) => BI PHAN LOP SAI")
    else:
        print(f"   -> Duoc phan lop dung, khong can cap nhat")
        return

    # 2) Cap nhat theo quy tac Perceptron: w_new = w + y*x
    print()
    w_new = w + y * x
    print(f"2) Cap nhat: w_new = w + y*x = {w.tolist()} + {y}*{x.tolist()} = {w_new.tolist()}")

    # 3) Tinh lai w^T x sau cap nhat
    print()
    score_new = np.dot(w_new, x)
    print(f"3) w_new^T x = {w_new.tolist()} . {x.tolist()} = {score_new}")
    y_pred_new = sgn(score_new)
    print(f"   Nhan du bao moi: sgn({score_new}) = {y_pred_new}")
    if y_pred_new == y:
        print(f"   -> Sau cap nhat, diem duoc phan lop DUNG (khop voi y={y})")
    else:
        print(f"   -> Sau cap nhat, diem VAN con bi phan lop sai")


if __name__ == "__main__":
    main()
