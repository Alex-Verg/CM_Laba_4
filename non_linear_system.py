import math
import numpy as np


f1 = lambda y: 0.879 - math.cos(y - 1.309)
f2 = lambda x: (math.sin(x + 0.48) - 1.019) / 1.369


f3 = lambda x, y: math.tan(x*y + 0.48) - x**2
f4 = lambda x, y: 0.539 * x **2 + 1.369 * y ** 2 - 1
df3_dx = lambda x, y: y / ((math.cos(x * y + 0.48)) ** 2) - 2 * x
df3_dy = lambda x, y: x / ((math.cos(x * y + 0.48)) ** 2)
df4_dx = lambda x, y: 1.138 * x
df4_dy = lambda x, y: 2.738 * y
F = lambda x: np.array([f3(x[0], x[1]), f4(x[0], x[1])])
dF = lambda x: np.array([[df3_dx(x[0], x[1]), df3_dy(x[0], x[1])], [df4_dx(x[0], x[1]), df4_dy(x[0], x[1])]])


def simple_iteration(x0, y0):
    x = f1(y0)
    y = f2(x0)
    while np.linalg.norm([x0 - x, y0 - y]) > 10e-5:
        x0 = x
        y0 = y
        x = f1(y0)
        y = f2(x0)
        print("%.6f, %.6f" % (x, y))


def newtons_method(x0, y0):
    x0 = np.array([x0, y0])
    x = x0 - np.linalg.inv(dF(x0)) @ (F_norm := F(x0))
    while np.linalg.norm(F_norm) > 10e-5 and np.linalg.norm(x) > 10e-5:
        x0 = x
        x = x0 - np.linalg.inv(dF(x0)) @ (F_norm := F(x0))
        print("%.6f, %.6f" % (x[0], x[1]))


if __name__ == "__main__":
    #simple_iteration(0.52, 0)
    newtons_method(1.2, 0.4)
