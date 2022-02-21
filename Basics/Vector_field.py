import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, pi, hypot

plt.rcParams['image.cmap'] = 'jet'


def first_field():
    x = np.linspace(0, 2 * pi, 40)
    y = np.linspace(0, 2 * pi, 40)
    X, Y = np.meshgrid(x, y)
    U, V = sin(X), (sin(Y)) ** 2

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111)  # ,aspect='equal')
    # q = ax.quiver(X, Y, U, V, hypot(U,V), clim=[0,sqrt(2)])

    q = ax.quiver(X, Y, U, V, hypot(U, V))
    cb = plt.colorbar(q)
    cb.outline.set_visible(False)

    plt.show()


def pendulum_field():

    L = 1
    y1 = np.linspace(-2.0, 8.0, 30)
    y2 = np.linspace(-25.0, 25.0, 30)
    Y1, Y2 = np.meshgrid(y1, y2)

    U = Y2
    V = -9.81 / L * sin(Y1) - 0.5 * Y2

    fig, ax = plt.subplots(1, 1, figsize=(10, 6), constrained_layout=True)

    Q = ax.quiver(Y1, Y2, U / hypot(U, V), V / hypot(U, V), hypot(U, V))
    cb = plt.colorbar(Q)
    cb.outline.set_visible(False)

    plt.xlabel(r'$\theta$')
    plt.ylabel(r'$\.\theta$')

    plt.show()


if __name__ == "__main__":
    # first_field()
    pendulum_field()
