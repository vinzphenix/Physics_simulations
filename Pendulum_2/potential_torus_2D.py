# Vincent Degrooff - 2022 - EPL
# Representation of the potential energy of the double pendulum on the torus - 2D

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, sqrt
import matplotlib.animation as animation

ftSz1, ftSz2 = 20, 13
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'


def compute_level_set(c, mu_l):
    th_ext = np.arccos(1 - c / mu_l) if c < 2 * mu_l else pi
    th_int = np.arccos(1 - (c - 2) / mu_l) if c > 2. else 0.

    th_1d = np.r_[np.linspace(-th_ext, -th_int, (n_th + 1) // 2), np.linspace(th_int, th_ext, n_th // 2)]

    c_phi = 1 - c + mu_l * (1 - cos(th_1d))
    c_phi[c_phi > 1] = 1  # keeps |.| below 1
    c_phi[c_phi < -1] = -1
    s_phi = sqrt(1 - c_phi * c_phi)
    return th_1d, c_phi, s_phi


def init():
    s.set_offsets(np.array([[], []]).T)
    time_text.set_text(time_template.format(param[0]))
    for line1, line2 in zip(lines1, lines2):
        line1.set_data([], [])
        line2.set_data([], [])
    return tuple([*lines1, *lines2, time_text, s])


def animate(i):
    time_text.set_text(time_template.format(param[i]))

    c_list = np.linspace(0 + eps, 2 * (1 + param[i]) - eps, N)
    pot_eq = (1 - cos(ph_eq)) + param[i] * (1 - cos(th_eq))
    s.set_offsets(np.array([y_eq, -x_eq]).T)
    s.set_color(c=cmap(pot_eq / pot_max))

    for line1, line2, c_ in zip(lines1, lines2, c_list):
        th, cos_phi, sin_phi = compute_level_set(c_, param[i])
        x = (R + r * cos(th)) * cos_phi
        y = (R + r * cos(th)) * sin_phi
        line1.set_color(cmap(c_ / pot_max))
        line2.set_color(cmap(c_ / pot_max))
        line1.set_data(y, -x)
        line2.set_data(-y, -x)
    return tuple([*lines1, *lines2, time_text, s])


def setup_ax(this_ax, ft_size=15, x_pos=0.85):
    template = r"$\mu \lambda = ${:.2f}"
    text = ax.text(x_pos, 0.925, "", transform=this_ax.transAxes, fontsize=ft_size)
    lns1 = [ax.plot([], [])[0] for _ in range(N)]
    lns2 = [ax.plot([], [])[0] for _ in range(N)]
    s_ = ax.scatter([], [], s=50, alpha=1.)
    this_ax.plot((R - r) * np.cos(th_circ), (R - r) * np.sin(th_circ), ls=':', color='grey')
    this_ax.plot((R + r) * np.cos(th_circ), (R + r) * np.sin(th_circ), ls=':', color='grey')
    this_ax.axis([-scale * (R + r), scale * (R + r), -scale * (R + r), scale * (R + r)])
    this_ax.set_aspect('equal', 'datalim')
    this_ax.grid(ls=':')
    return template, text, lns1, lns2, s_


if __name__ == "__main__":

    # pot_energy = (1 - cos(phi)) + mu * l * (1 - cos(th))  # = c
    #  cos(phi) = 1 - c + mu * l * (1 - cos(th))

    multiple_plots = 1
    N, n_th, eps = 20, 500, 0.01
    R, r = 5, 2.
    scale = 1.25
    cmap = plt.get_cmap('jet')

    th_circ = np.linspace(-pi, pi, 200)
    th_eq, ph_eq = np.array([0, 0, pi, pi]), np.array([0, pi, 0, pi])
    x_eq = (R + r * cos(th_eq)) * cos(ph_eq)
    y_eq = (R + r * cos(th_eq)) * sin(ph_eq)
    z_eq = r * sin(th_eq)

    if not multiple_plots:
        fig, ax = plt.subplots(1, 1, figsize=(11, 9), constrained_layout=True)

        t_anim, fps = 15, 25
        nFrames = int(t_anim * fps)
        linspace = np.linspace(0.1, 1., nFrames)
        param = linspace ** 2 * 1.25
        pot_max = 2 * (1 + np.amax(param))

        time_template, time_text, lines1, lines2, s = setup_ax(ax)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=100))
        cbar = fig.colorbar(sm, ax=ax, aspect=40)
        cbar.ax.set_ylabel("Energy in %", fontsize=ftSz2)

        anim = animation.FuncAnimation(fig, animate, nFrames, interval=1000 / fps, blit=True, init_func=init, repeat=False)
        # anim = animation.FuncAnimation(fig, animate, nFrames, interval=1000. / fps)
        # anim.save('./Animations/{:s}.mp4'.format("potential_param_2"), writer="ffmpeg", dpi=200, fps=fps)

    else:
        fig, axs = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
        param = np.array([0.05, 0.25, 0.5, 1.])
        pot_max = 2 * (1 + np.amax(param))

        for idx, ax in enumerate(axs.flatten()):
            time_template, time_text, lines1, lines2, s = setup_ax(ax, ftSz2, 0.785)
            animate(idx)

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=pot_max))
        cbar = fig.colorbar(sm, ax=axs.ravel().tolist(), aspect=40)
        cbar.ax.set_ylabel("Energy [/]", fontsize=ftSz2)
        fig.savefig("./Figures/potential_torus_2D.svg", format='svg', bbox_inches='tight')

    plt.show()
