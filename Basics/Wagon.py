import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

plt.rcParams['legend.fontsize'] = 10


def Runge():
    h = Tend / n
    U = np.zeros((n + 1, 4))
    U[0] = Ustart

    def f(u):
        th, om, x, v = u
        c1 = m * M * L / (2 * (M + m)) * cos(th)
        dth = om
        dom = -(c1 * sin(th) * om * om + sin(th) * M * g + M * L / 2 * sin(th) * cos(th) * om * om) / (
                M * L / 6 + c1 * cos(th) + M * L / 2 * sin(th) * sin(th)) - 0.1 * om
        dx = v
        dv = -M * L / (2 * (M + m)) * (dom * cos(th) - om * om * sin(th))
        return np.array([dth, dom, dx, dv])

    for i in range(n):
        K1 = f(U[i])
        K2 = f(U[i] + K1 * h / 2)
        K3 = f(U[i] + K2 * h / 2)
        K4 = f(U[i] + K3 * h)
        U[i + 1] = U[i] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6

    return U.T


if __name__ == "__main__":
    theta_0 = pi - 0.01
    omega_0 = 0
    x_0 = 0
    v_0 = 1

    L, m, M, g = 2, 1000, 10, 9.81
    Tend, n = 10, 1000

    Ustart = [theta_0, omega_0, x_0, v_0]
    T = np.linspace(0, Tend, n + 1)

    theta, omega, position, speed = Runge()
    X = position + L / 2 * sin(theta)
    Y = -L / 2 * cos(theta)

    fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True)
    ax.plot(T, Y)
    ax.grid(ls=':')

    # fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    # ax = fig.add_subplot(projection='3d')
    # ax.plot(X,T,Y)

    plt.show()
