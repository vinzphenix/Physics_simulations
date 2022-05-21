# Vincent Degrooff - 2022 - EPL
# Emphasize the difference between quasiperiodic
# and periodic trajectories using a Poincare section

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, sqrt
import matplotlib.animation as animation
from potential_torus_3D import set_axes_equal

ftSz1, ftSz2 = 20, 13
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'


def check_cross(th_prev, th_next):
    if np.fmod(th_prev, 2 * pi) > np.fmod(th_next, 2 * pi):
        return int(th_next / (2 * pi)) + 1
    else:
        return 0


def init():
    ax.view_init(elev_max, -45)
    line.set_data([], [])
    marker.set_data([], [])
    poincare.set_data([], [])
    return ax, line, marker, poincare


def animate(idx):
    i = idx * oversample
    # st = max(0, i - 100)
    st = 0
    elev = elev_min + (elev_max - elev_min) / 2. * (1. + cos(pi * i / nFrames))
    azim = ax.azim + delta_azim / nFrames
    ax.view_init(elev, azim)

    marker.set_data(x[i], y[i])
    marker.set_3d_properties(z[i])
    line.set_data(x[st:i+1], y[st:i+1])
    line.set_3d_properties(z[st:i+1])

    res = check_cross(th1[i-oversample], th1[i])
    if i == 0 or res > 0:
        poincare.set_data(crossings[:res, 0], crossings[:res, 1])
        poincare.set_3d_properties(crossings[:res, 2])

    return ax, line, marker, poincare


def setup_ax(this_ax, alpha1=0.25, alpha2=1.00):

    s = r"$\frac{\omega_1}{\omega_2} = \frac{1}{\sqrt{2}}$"
    ax.text2D(0.85, 0.90, s, transform=ax.transAxes, fontsize=ftSz1)

    n_sample, k = 100, 1.5
    x_ = np.linspace(R - k * r, R + k * r, n_sample)
    z_ = np.linspace(- k * r, k * r, n_sample)
    xx, zz = np.meshgrid(x_, z_)
    yy = np.zeros_like(xx)
    this_ax.plot_surface(xx, yy, zz, alpha=alpha1, color='C1', zorder=100)

    th_ = np.linspace(0, 2 * pi, n_sample)
    phi_ = np.linspace(0, 2 * pi, n_sample)
    th, phi = np.meshgrid(th_, phi_)

    k = 1.
    x_torus = (R + k * r * cos(th)) * cos(phi)
    y_torus = (R + k * r * cos(th)) * sin(phi)
    z_torus = k * r * sin(th)
    this_ax.plot_wireframe(x_torus, y_torus, z_torus, rcount=n_sample, ccount=n_sample,
                      linewidth=0.1, alpha=alpha2, zorder=2, color='grey')
    return


if __name__ == "__main__":
    R, r = 4, 2
    w1, w2 = 2, 2 * sqrt(2)
    T = 20 * (2 * pi)

    bool_anim = 1
    slowdown = 0.1
    oversample = 5
    t_anim, fps = slowdown * T, 30
    nFrames = int(t_anim * fps)
    elev_min, elev_max, delta_azim = 40., 40., 0  # view options

    fig = plt.figure(figsize=(10, 7), constrained_layout=True)
    ax = fig.add_subplot(projection='3d')

    t = np.linspace(0, T, oversample * nFrames + 1)
    th1 = w1 * t
    th2 = w2 * t
    x = (R + r * cos(th2)) * cos(th1)
    y = (R + r * cos(th2)) * sin(th1)
    z = r * sin(th2)

    t_cross = 2 * np.arange(w1 * T / (2 * pi)) * pi / w1
    crossings = np.c_[R+r*cos(t_cross * w2), np.zeros_like(t_cross), r * sin(t_cross * w2)]

    if bool_anim:
        setup_ax(ax)
        line, = ax.plot([], [], [], '-', color='C0', lw=1.5, alpha=0.50, zorder=1)
        marker, = ax.plot([], [], [], marker='o', linestyle='', color='C0', zorder=1)
        poincare, = ax.plot([], [], [], marker='o', linestyle='', color='C2', markersize=5)
        set_axes_equal(ax)

        # animate(0)
        # fig.savefig("./animations/fig_torus_periodic.png", format='png', bbox_inches='tight')

        anim = animation.FuncAnimation(fig, animate, nFrames + 1, interval=1000 / fps, blit=True, init_func=init, repeat=False)
        # anim = animation.FuncAnimation(fig, animate, nFrames, interval=1000. / fps, repeat=False)  # remove blit for save
        # anim.save('./animations/{:s}.mp4'.format("poincare_torus"), writer="ffmpeg", dpi=150, fps=fps)

    else:
        ax.view_init(elev_max, -60)
        ax.plot(x, y, z, '-', color='C0', lw=1.5, alpha=0.50, zorder=200)
        setup_ax(ax, 0.5, 1.)
        ax.plot(crossings[:, 0], crossings[:, 1], crossings[:, 2], marker='o', linestyle='', color='C2', markersize=5, zorder=4)
        set_axes_equal(ax)
        # fig.savefig("./figures/poincare_quasi_new.svg", format='svg', bbox_inches='tight')

    plt.show()
