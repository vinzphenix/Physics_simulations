import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi

plt.rcParams['legend.fontsize'] = 10


def Runge():
    h = Tend / n
    U = np.zeros((n + 1, 2))
    U[0] = Ustart

    f = lambda u: np.array([u[1], (-cos(u[0]) * u[1] ** 2 - g / R * cos(u[0])) / (3.5 + 2 * sin(u[0])) - 0.1 * u[1]])

    for i in range(n):
        K1 = f(U[i])
        K2 = f(U[i] + K1 * h / 2)
        K3 = f(U[i] + K2 * h / 2)
        K4 = f(U[i] + K3 * h)
        U[i + 1] = U[i] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6

    return U.T


if __name__ == "__main__":
    phi_0 = pi / 2 + 0.1
    omega_0 = -3

    g = 9.81
    R, m = 0.3, 1000

    Tend = 20
    n = 4000
    Ustart = [phi_0, omega_0]
    T = np.linspace(0., Tend, n + 1)

    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')

    phi, omega = Runge()
    x = -R * (phi - phi_0)

    X = x + R / 2 * cos(phi)
    Z = R / 2 * sin(phi)

    ax.plot(X, T, Z)
    # plt.plot(T,phi)
    # plt.plot(phi,omega)

    # plt.grid()
    plt.show()
