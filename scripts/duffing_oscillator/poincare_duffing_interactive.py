import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from numpy import random as rd
from numpy import cos, pi, array, zeros, fmod

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


def runge(dt, U0, nSteps):
    def f(u, tau):
        u1, u2 = u
        f1 = - alpha * u1 - beta * u1 ** 3 - gamma * u2 + A * cos(w * tau)
        return array([u2, f1])

    point_list = zeros((nPoints, 2))
    index = 0

    for i in range(nSteps - 1):

        t = dt * i

        K1 = f(U0, t)
        K2 = f(U0 + K1 * dt / 2, t)
        K3 = f(U0 + K2 * dt / 2, t)
        K4 = f(U0 + K3 * dt, t)
        U1 = U0 + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6

        x_1, v_1 = U0
        x_2, v_2 = U1
        t_1, t_2 = t - dt, t
        t_1, t_2 = fmod(array([t_1 * w, t_2 * w]) - phi, 2 * pi)

        if (t_2 < t_1) and (100. < t):
            # if (sin(t_1*w) * sin(t_2*w) <= 0.) and (cos(t_1*w) > 0.) and (t > 75.):

            # t_1, t_2 = remainder(array([t_1 * w, t_2 * w]) + pi, 2 * pi) - pi
            coef = t_1 / (t_1 - t_2)
            point_x = x_1 * (1 - coef) + x_2 * coef
            point_y = v_1 * (1 - coef) + v_2 * coef

            point_list[index] = point_x, point_y
            index += 1
            if index == nPoints:
                return index, point_list

        U0 = U1

    print("Not enough", index, nPoints, nSteps)
    return index, point_list


def sampleRandom(nb, list_U0, x_lim=(-1, 1), y_lim=(-1, 1)):
    L_x, L_y = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
    m_x, m_y = (x_lim[1] + x_lim[0]) / 2, (y_lim[1] + y_lim[0]) / 2

    for i in range(nb):
        x0, dx0 = L_x * (rd.rand() - 0.5) + m_x, 4 * (rd.rand() - 0.5) + m_y
        U0 = np.array([x0, dx0])
        nbPoints, points = runge(delta_t, U0.copy(), nMax)
        line, = ax.plot(points[:nbPoints, 0], points[:nbPoints, 1], 'o', markersize=0.5, color='black')

        lines.append(line)
        list_U0.append(U0.copy())


def on_click(event, list_U0):
    if event.inaxes != ax:
        return
    if event.button == 1:
        x0, v0 = event.xdata, event.ydata
        U0 = np.array([x0, v0])
        if U0 is None:
            return
        print("{:.5f}, {:.5f}".format(U0[0], U0[1]))
        list_U0.append(U0.copy())

        nbPoints, points = runge(delta_t, U0.copy(), nMax)

        # line, = ax.plot(points[:nbPoints, 0], points[:nbPoints, 1], color=cmap())

        line, = ax.plot(points[:nbPoints, 0], points[:nbPoints, 1], 'o', markersize=0.5, color='black')
        lines.append(line)

    elif event.button == 2:
        while len(lines) > 0:
            line = lines.pop()
            line.remove()
        list_U0.clear()

    elif event.button == 3:
        if len(lines) > 0:
            line = lines.pop()
            line.remove()
            list_U0.pop()
    fig1.canvas.draw()


def on_touch(event, list_U0):
    if event.key == 'r':
        for i in range(5):
            sampleRandom(1, list_U0, x_lim=x_Lim, y_lim=y_Lim)
            fig1.canvas.draw()


if __name__ == "__main__":
    #alpha, beta, gamma, A, w = 1.0, 5.0, 0.02, 8.0, 0.5
    #alpha, beta, gamma, A, w = -1.0, 1.0, 0.15, 0.3, 1.0
    alpha, beta, gamma, A, w = -1.0, 1.0, 0.1, 0.35, 1.4

    phi = 0.0 * pi

    nPoints = 100
    nMax = 350000
    delta_t = 0.01

    x_Lim = (-1.8, 1.8)
    y_Lim = (-3.5, 3.5)

    fig1, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=False)
    ax.set_xlabel(r'$x$', size=12)
    ax.set_ylabel(r'$v$', rotation=0, size=12)
    #ax.set_xlim([-1.8, 1.8])
    #ax.set_ylim([-3.0, 3.0])
    ax.tick_params(axis='both', which='major', labelsize=8)

    ax.set_title(r"$\ddot{x} + \gamma \dot{x} + \alpha x + \beta x^3 \,=\, A \cos{(\omega \, t)}$")
    ax.text(0.025, 0.95, r'$\alpha = {:.2f} $'.format(alpha), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.025, 0.90, r'$\beta  = {:.2f} $'.format(beta), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.025, 0.85, r'$\gamma = {:.2f} $'.format(gamma), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.150, 0.95, r'$A = {:.2f} $'.format(A), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.150, 0.90, r'$\omega = {:.2f} $'.format(w), fontsize=10, wrap=True, transform=ax.transAxes)

    ax.text(0.801, 0.895, r"$\varphi = {:.2f} \;\pi$".format(phi/pi), fontsize=12, wrap=True, transform=ax.transAxes)

    initConditions = []
    lines = []

    cid_1 = fig1.canvas.mpl_connect('button_press_event', lambda event: on_click(event, initConditions))
    cid_2 = fig1.canvas.mpl_connect('key_press_event', lambda event: on_touch(event, initConditions))

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.075, right=0.95)
    plt.show(block=True)
