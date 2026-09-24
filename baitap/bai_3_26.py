

import numpy as np


def gradient_descent(grad_func, cost_func, x0, eta, max_iters=1000, tol=1e-3):
    x = [x0]
    it = 0
    for it in range(max_iters):
        x_new = x[-1] - eta * grad_func(x[-1])
        x.append(x_new)
        if abs(grad_func(x_new)) < tol:
            break
    return x, it + 1


# ============================================================
# Bai 3.26: f(x) = x^2 - 4x + 5
# ============================================================
def grad_326(x):
    return 2 * x - 4


def cost_326(x):
    return x ** 2 - 4 * x + 5


if __name__ == "__main__":
    print("=" * 60)
    print("BAI 3.26: f(x) = x^2 - 4x + 5, x0 = 5, eta = 0.2")
    print("=" * 60)
    x_hist, n_it = gradient_descent(grad_326, cost_326, x0=5, eta=0.2, max_iters=4, tol=0)
    for t, xt in enumerate(x_hist):
        print(f"  x{t} = {xt:.4f}, f(x{t}) = {cost_326(xt):.4f}")
    print(f"  -> Nghiem hoi tu ve x* = 2, f(x*) = 1 (gia tri nho nhat)")
