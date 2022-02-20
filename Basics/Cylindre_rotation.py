import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

plt.rcParams['legend.fontsize'] = 10


def Runge(u_zero, n):
    T = np.linspace(Tstart, Tend, n + 1)
    h = (Tend - Tstart) / n
    U = np.zeros((n + 1, 2))
    U[0] = u_zero

    def f(u):
        theta, omega = u
        dth = omega
        dom = -2 * g / (3 * R) * sin(theta) - 0.25 * omega
        return np.array([dth, dom])

    for i in range(n):
        K1 = f(U[i])
        K2 = f(U[i] + K1 * h / 2)
        K3 = f(U[i] + K2 * h / 2)
        K4 = f(U[i] + K3 * h)
        U[i + 1] = U[i] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6

    return T, U


if __name__ == "__main__":
    g = 9.81
    R = 0.1

    t0 = pi - 0.01
    Ustart = [t0, 0]
    Tstart, Tend = 0., 7.

    T_sol, U_sol = Runge([t0, 0], 10000)
    X = -2 * 0.1 * sin(U_sol[:, 0])
    Z = -2 * 0.1 * cos(U_sol[:, 0])

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')
    # ax = fig.gca(projection='3d')

    # plt.plot(T,X,'brown')

    ax.plot(X, T_sol, Z)
    plt.show()
