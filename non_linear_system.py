import math
import numpy as np


f1 = lambda y: 0.879 - math.cos(y - 1.309)
f2 = lambda x: (math.sin(x + 0.48) - 1.019) / 1.369


f3 = lambda x, y: math.tan(x*y + 0.48) - x**2
f4 = lambda x, y: 0.539 * x **2 + 1.369 * y ** 2 - 1
df3_dx = lambda x, y: y / ((math.cos(x * y + 0.48)) ** 2) - 2 * x
df3_dy = lambda x, y: x / ((math.cos(x * y + 0.48)) ** 2)
df4_dx = lambda x, y: 1.078 * x
df4_dy = lambda x, y: 2.738 * y
F = lambda x: np.array([f3(x[0], x[1]), f4(x[0], x[1])])
dF = lambda x: np.array([[df3_dx(x[0], x[1]), df3_dy(x[0], x[1])], [df4_dx(x[0], x[1]), df4_dy(x[0], x[1])]])


def simple_iteration(x0, y0):
    x = f1(y0)
    y = f2(x0)
    it = 1
    Fk = np.array([f1(y) - x, f2(x) - y])
    print(f"{it: >2d} iteration: (x, y) = ({x: .6f}, {y: .6f}), \n\t\t\t(f1, f2) = ({Fk[0]: .6f}, {Fk[1]: .6f}), \n\t\t\t\t |F| = {np.linalg.norm(Fk): .6f}")
    while np.linalg.norm([x0 - x, y0 - y]) > 1e-5 or np.linalg.norm(Fk) > 1e-5:
        it += 1
        x0 = x
        y0 = y
        x = f1(y0)
        y = f2(x0)
        Fk = np.array([f1(y) - x, f2(x) - y])
        print(f"{it: >2d} iteration: (x, y) = ({x: .6f}, {y: .6f}), \n\t\t\t(f1, f2) = ({Fk[0]: .6f}, {Fk[1]: .6f}), \n\t\t\t\t |F| = {np.linalg.norm(Fk): .6f}")

    return x, y


def newtons_method(x0, y0):
    x0 = np.array([x0, y0])
    x = x0 - np.linalg.inv(dF(x0)) @ (Fk := F(x0))
    it = 1
    print(f"{it: >2d} iteration: (x, y) = ({x[0]: .6f}, {x[1]: .6f}), \n\t\t\t(f1, f2) = ({Fk[0]: .6f}, {Fk[1]: .6f}), \n\t\t\t\t |F| = {np.linalg.norm(Fk): .6f}")
    while np.linalg.norm(Fk) > 1e-5 or np.linalg.norm(x - x0) > 1e-5:
        it += 1
        x0 = x
        x = x0 - np.linalg.inv(dF(x0)) @ (Fk := F(x0))
        print(f"{it: >2d} iteration: (x, y) = ({x[0]: .6f}, {x[1]: .6f}), \n\t\t\t(f1, f2) = ({Fk[0]: .6f}, {Fk[1]: .6f}), \n\t\t\t\t |F| = {np.linalg.norm(Fk): .6f}")

    return x[0], x[1]


if __name__ == "__main__":
    print("Simple iteration method for 1st system:")
    x, y = simple_iteration(0.7, -0.1)
    print(f"Solve for 1st system: x ={x: .6f}, y ={y: .6f}.")
    print("\nNewton's method for 2nd system:")
    x = list()
    print("1st solve:")
    x.append(newtons_method(1.2, 0.4))
    print("\n2nd solve:")
    x.append(newtons_method(0.5, -0.8))
    print("\n3rd solve:")
    x.append(newtons_method(-1.2, -0.4))
    print("\n4th solve:")
    x.append(newtons_method(-0.5, 0.8))
    print(f"Solve for 2st system: x ={x[0][0]: .6f}, y ={x[0][1]: .6f}")
    for i in range(1, 4):
        print(f"\t\t\t\t\t  x ={x[i][0]: .6f}, y ={x[i][1]: .6f}")
