import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from numpy import random as rd
from numpy import sin, pi, array, zeros, remainder, fmod

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


def runge(dt, U0, nSteps):
    def f(u, time):
        th1 = phi1_0 + w * time
        th2, dth2 = u

        f1 = - w * w * sin(th2 - th1) / l - sin(th2) / l

        return array([dth2, f1])

    point_list = zeros((nPoints, 2))
    index = 0

    for i in range(nSteps - 1):

        t = dt * i

        K1 = f(U0, t)
        K2 = f(U0 + K1 * dt / 2, t)
        K3 = f(U0 + K2 * dt / 2, t)
        K4 = f(U0 + K3 * dt, t)
        U1 = U0 + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6

        phi_1, om_1 = U0
        phi_2, om_2 = U1
        t_1, t_2 = t - dt, t
        t_1, t_2 = fmod(phi1_0 + array([t_1 * w, t_2 * w]) - angle, 2 * pi)

        if (t_2 < t_1) and (0. < t):
            # if (sin(t_1*w) * sin(t_2*w) <= 0.) and (cos(t_1*w) > 0.) and (t > 75.):

            phi_1, phi_2 = remainder(array([phi_1, phi_2]) + pi, 2 * pi) - pi
            coef = t_1 / (t_1 - t_2)
            point_x = phi_1 * (1 - coef) + phi_2 * coef
            point_y = om_1 * (1 - coef) + om_2 * coef

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
        for i in range(15):
            sampleRandom(1, list_U0, x_lim=x_Lim, y_lim=y_Lim)
            fig1.canvas.draw()


if __name__ == "__main__":
    l = 4.2
    phi1_0, w = 0.0, 1.0 * 1./l

    angle = 0.0 * pi

    interactive = True
    nPoints = 150
    nMax = 200000
    delta_t = 0.05
    # Tend = 2000              # [s]    -  fin de la simulation

    x_Lim = (1.0, 1.6)
    y_Lim = (-2.0, 2.0)

    fig1, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=False)
    ax.set_xlabel(r'$x$', size=12)
    ax.set_ylabel(r'$v$', rotation=0, size=12)
    # ax.set_xlim(x_Lim)
    # ax.set_ylim(y_Lim)
    ax.tick_params(axis='both', which='major', labelsize=8)

    ax.set_title(r"$\lambda \ddot{\varphi_2} + \omega_1^2 \;\sin{(\varphi_2 - \omega_1 \tau)} - \sin{(\varphi_2)} = 0$")
    ax.text(0.025, 0.95, r'$\lambda = {:.2f} $'.format(l), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.025, 0.90, r'$\varphi_0  = {:.2f} $'.format(phi1_0), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.150, 0.95, r'$\omega = {:.2f} $'.format(w), fontsize=10, wrap=True, transform=ax.transAxes)

    ax.text(0.800, 0.895, r"$\phi = {:.2f} \;\pi$".format(angle/pi), fontsize=12, wrap=True, transform=ax.transAxes)

    initConditions = []
    lines = []

    if interactive:
        cid_1 = fig1.canvas.mpl_connect('button_press_event', lambda event: on_click(event, initConditions))
        cid_2 = fig1.canvas.mpl_connect('key_press_event', lambda event: on_touch(event, initConditions))
    else:
        sampleRandom(2, initConditions, x_lim=x_Lim, y_lim=y_Lim)

    plt.subplots_adjust(bottom=0.1, top=0.9, left=0.075, right=0.95)
    plt.show(block=True)
