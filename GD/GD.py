import numpy as np

# ===== Bài 1: f(x) = x^2 - 2 =====
def grad1(x):
    return 2*x          # f'(x) = 2x

def cost1(x):
    return x**2 - 2

# ===== Bài 2: g(x) = (1/3)x^3 - x =====
def grad2(x):
    return x**2 - 1      # g'(x) = x^2 - 1

def cost2(x):
    return (1/3)*x**3 - x

# ===== Hàm Gradient Descent tổng quát =====
def myGD(grad_func, x0, eta, iterations=100, tol=1e-3):
    x = [x0]
    for it in range(iterations):
        x_new = x[-1] - eta * grad_func(x[-1])
        if abs(grad_func(x_new)) < tol:
            x.append(x_new)
            break
        x.append(x_new)
    return (x, it)

# ---- Chạy thử ----
x1, it1 = myGD(grad1, -5, 0.1)
x2, it2 = myGD(grad1, 5, 0.1)
print('Bài 1: x1 = %f, cost = %f, sau %d vòng lặp' % (x1[-1], cost1(x1[-1]), it1))
print('Bài 1: x2 = %f, cost = %f, sau %d vòng lặp' % (x2[-1], cost1(x2[-1]), it2))

x3, it3 = myGD(grad2, -5, 0.1)
x4, it4 = myGD(grad2, 5, 0.1)
print('Bài 2: x3 = %f, cost = %f, sau %d vòng lặp' % (x3[-1], cost2(x3[-1]), it3))
print('Bài 2: x4 = %f, cost = %f, sau %d vòng lặp' % (x4[-1], cost2(x4[-1]), it4))