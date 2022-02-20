import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

plt.rcParams['legend.fontsize'] = 10


def Runge():
    h = Tend / n
    U = np.zeros((n + 1, 2))
    U[0] = Ustart

    f = lambda u: np.array([u[1], 2 * k / (3 * m) * (L - u[0]) + g / 3 - 0.25 * u[1]])

    for i in range(n):
        K1 = f(U[i])
        K2 = f(U[i] + K1 * h / 2)
        K3 = f(U[i] + K2 * h / 2)
        K4 = f(U[i] + K3 * h)
        U[i + 1] = U[i] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6

    return U.T

#
# def f():
#     w = sqrt(2 * k / (3 * m))
#     B = Ustart[0] - L - m * g * sin(a) / k
#     A = Ustart[0] / w
#     x = A * sin(w * T) + B * cos(w * T) + L + m * g / k * sin(a)
#     theta = (x - Ustart[0]) / R
#     return np.array([x, theta])


if __name__ == "__main__":
    L, k, m, R, g, a = 0.3, 200, 5, 0.05, 9.81, pi / 6
    Tend, n = 10, 1000

    Ustart = [0.1, 0]
    T = np.linspace(0, Tend, n + 1)

    x, v = Runge()
    theta = (x - Ustart[0]) / R

    X = x * cos(a) + R * cos(a + theta)
    Y = -x * sin(a) - R * sin(a + theta) + 0.5

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')

    ax.plot(X, T, Y)
    # plt.plot(T,x)
    plt.grid()
    plt.show()
