import numpy as np
import matplotlib.pyplot as plt

from numpy import sin, cos, sqrt, pi, amax, zeros, abs
from scipy.integrate import odeint


# mode = "basic"
mode = "stream"

lw = 2
n_simu = 50

g = 9.81  # [m/sf]  -  accélaration de pesanteur
L = 1.  # [m]     -  longueur 1er pendule
D = 1.
M = 1.


def f(u, _):
    phi, om = u
    return np.array([om, - g / L * sin(phi) - D / M * om])


fig, ax = plt.subplots(1, 1, figsize=(12, 6), constrained_layout=True)
ax.set_xlabel(r'$\varphi \;\rm [rad]$', fontsize=16)
ax.set_ylabel(r'$\omega \;\rm [rad/s]$', fontsize=16)
# ax.set_xlim(0, 3 * pi)
ax.grid(ls=':')
cmap = plt.get_cmap('jet')

if mode == "basic":
    dt = 3e-3  # [s]    -  pas de temps
    Tend = 20  # [s]    -  fin de la simulation
    n = int(1050 * Tend)
    t = np.linspace(0, Tend, n)
    sol = zeros((n, 2, n_simu))
    j, k, l, m = 0, 0, 0, 0
    for i in range(n_simu):
        if i <= 20:
            U0 = np.array([0, (5 + 23.415 * sqrt(j / n_simu))])
            j += 1
        elif i <= 41:
            U0 = np.array([2 * pi, (5 + 23.415 * sqrt(k / n_simu))])
            k += 1
        elif i < 45:
            U0 = np.array([0, (5 + 23.415 * sqrt((21 + l) / n_simu))])
            l += 1
        else:
            U0 = np.array([6 * pi, -(5 + 23.415 * sqrt((21 + m) / n_simu))])
            m += 1
        sol[:, :, i] = odeint(f, U0, t)
    colors = [cmap(i) for i in np.linspace(0, 1, n_simu)]
    y_Max = amax(sol[:, 1, :])
    for i in range(len(sol[0, 0, :])):
        y_max = amax(abs(sol[:, 1, i]))
        ax.plot(sol[:, 0, i], sol[:, 1, i], color=cmap((y_max / y_Max) ** 2))

else:
    nx, ny = 50, 50
    x, y = np.linspace(-2 * pi, 2 * pi, nx), np.linspace(-3 * sqrt(g / L), 3 * sqrt(g / L), ny)
    X, Y = np.meshgrid(x, y)
    U, V = f(np.array([X, Y]), 0)
    ax.streamplot(X, Y, U, V, color=L / 2 * Y ** 2 + g * (1 - cos(X)), density=(3, 2), cmap=cmap)

plt.show()
