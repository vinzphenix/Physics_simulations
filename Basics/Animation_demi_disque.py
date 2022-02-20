"""
===========================
Animation Semi-Circle
===========================

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import patches
from numpy import sin, cos, radians, pi, degrees


def Runge():
    h = (Tend - Tstart) / n
    U = np.zeros((n + 1, 4))
    U[0] = Ustart

    def f(u):
        th_, om_, y_, v_ = u

        dth = om_
        dom = (om_ * om_ * (
                (M * M * a * sin(th_) * (R - a * cos(th_))) / (2 * m + M) - M * a * R * sin(th_)) - M * g * a * sin(
            th_)) / (I + M * (a * a + R * R) - 2 * a * M * R * cos(th_) - M * M * (R - a * cos(th_)) ** 2 / (
                M + 2 * m)) - 0.5 * om_

        dy = v_
        dv = (M * dom * (a * cos(th_) - R) - om_ * om_ * M * a * sin(th_)) / (2 * m + M)

        return np.array([dth, dom, dy, dv])

    for i in range(n):
        K1 = f(U[i])
        K2 = f(U[i] + K1 * h / 2)
        K3 = f(U[i] + K2 * h / 2)
        K4 = f(U[i] + K3 * h)
        U[i + 1] = U[i] + h * (K1 + 2 * K2 + 2 * K3 + K4) / 6

    # th = U[:, 0]
    # w = U[:, 1]
    # y = U[:, 2]
    # v = U[:, 3]
    # return np.array([th, w, y, v])
    return U.T


if __name__ == "__main__":
    g = 9.81  # acceleration due to gravity, in m/s²
    R = 0.3  # rayon demi-disque
    a = 4 * R / (3 * pi)  # distance AG
    M = 10.  # masse demi-disque
    d = 0.1  # épaisseur plaque
    m = 5  # masse planche = masse support
    b = 0.1  # rayon petit cercle
    I = M * (R * R / 2 + a * a)

    # initial state
    th1 = radians(-60)  # angle initial demi-disque
    w1 = 4  # vitesse angulaire initiale pendule
    y1 = 0  # position initiale planche
    v1 = -0.4  # vitesse initiale planche
    Ustart = [th1, w1, y1, v1]

    # create a time np.array from 0..100 sampled at 0.05 second steps
    Tstart, Tend, dt = 0., 15., 0.03
    n = round(Tend / dt)
    t = np.linspace(Tstart, Tend, n)

    # Solve equations différentielles
    th, om, y, v = Runge()
    x = y + R * (th - th1)
    phi = y / b

    xc1, yc1 = -2 * R, -R - 2 * d - b
    xc2, yc2 = 5 * R, yc1

    x1, y1 = xc1 + b * cos(phi), yc1 - b * sin(phi)  # position d'un point sur cercle phi
    x2, y2 = xc2 + b * cos(phi), yc2 - b * sin(phi)  # position d'un point sur cercle phi'

    fig, ax = plt.subplots(1, 1, figsize=(16, 7), constrained_layout=True)
    ax.axis([-5 * R, 10 * R, 1.5 * (-R - 2 * d - 2 * d), 3 * R])
    ax.set_aspect("equal", "datalim")
    ax.grid(False)
    ax.axis('off')

    line1, = ax.plot([], [], 'o-', lw=2, color='grey')
    line2, = ax.plot([], [], 'o-', lw=2, color='grey')
    vline3, = ax.plot([], [], '-', lw=2, color='grey')
    vline4, = ax.plot([], [], '-', lw=2, color='grey')
    vline5, = ax.plot([], [], '-', lw=2, color='grey')
    vline6, = ax.plot([], [], '-', lw=2, color='grey')

    time_template = r'time = %.1f s'
    time_text = ax.text(0.05, 2 * R, '', transform=ax.transAxes, fontsize=15)

    circ1 = plt.Circle((xc1, yc1), radius=b, facecolor='lightgrey', lw=4, edgecolor='none')
    circ2 = plt.Circle((xc2, yc2), radius=b, facecolor='lightgrey', lw=4, edgecolor='none')
    sem_circ = patches.Wedge((x[0], 0), R, theta1=-180 - degrees(th[0]), theta2=-degrees(th[0]), width=None, color='k')
    rect = patches.Rectangle((y[0] - 10, -R - 2 * d), 20, 2 * d, alpha=0.25, edgecolor='none')

    ax.add_patch(circ1)
    ax.add_patch(circ2)


    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        vline3.set_data([], [])
        vline4.set_data([], [])
        vline5.set_data([], [])
        vline6.set_data([], [])

        sem_circ.set(center=(x[0], 0), theta1=-180 - degrees(th[0]), theta2=-degrees(th[0]))

        ax.add_patch(sem_circ)
        ax.add_patch(rect)
        ax.add_patch(circ1)
        ax.add_patch(circ2)

        time_text.set_text('')

        return line1, line2, vline3, vline4, vline5, vline6, time_text, sem_circ


    def animate(i):

        sem_circ.set(center=(x[i], 0), theta1=-180 - degrees(th[i]), theta2=-degrees(th[i]))
        ax.add_patch(sem_circ)
        ax.add_patch(rect)

        thisx1, thisy1 = [xc1, x1[i]], [yc1, y1[i]]
        thisx2, thisy2 = [xc2, x2[i]], [yc2, y2[i]]
        thisx3, thisy3 = [y[i] - 8 * R, y[i] - 8 * R], [-R - 2 * d, -R]
        thisx4 = [y[i] - 4 * R, y[i] - 4 * R]
        thisx5 = [y[i], y[i]]
        thisx6 = [y[i] + 4 * R, y[i] + 4 * R]

        line1.set_data(thisx1, thisy1)
        line2.set_data(thisx2, thisy2)
        vline3.set_data(thisx3, thisy3)
        vline4.set_data(thisx4, thisy3)
        vline5.set_data(thisx5, thisy3)
        vline6.set_data(thisx6, thisy3)

        time_text.set_text(time_template % (i * dt))

        return line1, line2, vline3, vline4, vline5, vline6, time_text, sem_circ, rect


    ani = animation.FuncAnimation(fig, animate, len(t), interval=10, blit=True, init_func=init)
    plt.show()
