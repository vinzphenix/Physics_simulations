# Vincent Degrooff - 2022 - EPL
# Representation of the potential energy of the double pendulum on the torus - 3D

import numpy as np
import matplotlib.pyplot as plt
from numpy import sin, cos, pi, sqrt, abs
import matplotlib.animation as animation

ftSz1, ftSz2 = 15, 13
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'
idx_anim = -1


def set_axes_equal(this_ax):
    """Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = this_ax.get_xlim3d()
    y_limits = this_ax.get_ylim3d()
    z_limits = this_ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.4 * max([x_range, y_range, z_range])

    this_ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    this_ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    this_ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def compute_level_set(c):
    if c < 2 * mu * l:
        th_lim = np.arccos(1 - c / (mu * l))  # 2 >=  c / (mu * l)
        th_1d = np.linspace(-th_lim, th_lim, n_th)
    elif c <= 2.:
        th_1d = np.linspace(-pi, pi, n_th)
    else:
        th_lim = np.arccos(1 - (c - 2) / (mu * l))
        th_1d = np.r_[np.linspace(-pi, -th_lim, (n_th + 1) // 2), np.linspace(th_lim, pi, n_th // 2)]

    c_phi = 1 - c + mu * l * (1 - cos(th_1d))
    c_phi[1 - c_phi * c_phi < 0.] = np.round(c_phi[1 - c_phi * c_phi < 0.])  # keeps |.| below 1
    s_phi = sqrt(1 - c_phi * c_phi)
    return th_1d, c_phi, s_phi


def set_line_data(this_line_a, this_line_b, x_, y_, z_, c):
    idx = (n_th + 1) // 2
    this_line_a.set_data(x_[:idx], y_[:idx])
    this_line_a.set_3d_properties(z_[:idx])
    this_line_b.set_data(x_[idx:], y_[idx:])
    this_line_b.set_3d_properties(z_[idx:])
    this_line_a.set_color(cmap(c / c_list[-1]))
    this_line_b.set_color(cmap(c / c_list[-1]))


def init():
    time_text.set_text(time_template.format(0))
    line1a.set_data([], [])
    line1b.set_data([], [])
    line2a.set_data([], [])
    line2b.set_data([], [])
    for l1a, l1b, l2a, l2b in zip(lines1a, lines1b, lines2a, lines2b):
        l1a.set_data([], [])
        l1b.set_data([], [])
        l2a.set_data([], [])
        l2b.set_data([], [])
    return tuple([ax, line1a, line1b, line2a, line2b, *lines1a, *lines1b, *lines2a, *lines2b, time_text])


def animate(j):
    """
    Modifies the figure at every frame of the animation
    """
    elev = elev_min + (elev_max - elev_min) / 2. * (1. + cos(pi * j / nFrames))
    azim = ax.azim + delta_azim / nFrames
    ax.view_init(elev, azim)

    time_text.set_text(time_template.format(100 * c_list[j] / c_list[-1]))

    th_1d, c_phi, s_phi = compute_level_set(c_list[j])
    x_i = (R + r * cos(th_1d)) * c_phi
    y_i_p = (R + r * cos(th_1d)) * s_phi
    y_i_m = (R + r * cos(th_1d)) * -s_phi
    z_i = r * sin(th_1d)

    set_line_data(line1a, line1b, x_i, y_i_m, z_i, c_list[j])
    set_line_data(line2a, line2b, x_i, y_i_p, z_i, c_list[j])

    if j % (nFrames // (N - 1)) == 0:
        global idx_anim
        if idx_anim >= 0:
            set_line_data(lines1a[idx_anim], lines1b[idx_anim], x_i, y_i_m, z_i, c_list[j])
            set_line_data(lines2a[idx_anim], lines2b[idx_anim], x_i, y_i_p, z_i, c_list[j])
        idx_anim += 1

    return tuple([ax, line1a, line1b, line2a, line2b, *lines1a, *lines1b, *lines2a, *lines2b, time_text])


def plot_this_level(this_ax, j, this_lw=1., this_a=1.):
    idx = (n_th + 1) // 2
    th_1d, c_phi, s_phi = compute_level_set(c_list[j])
    x_i = (R + r * cos(th_1d)) * c_phi
    y_i_p = (R + r * cos(th_1d)) * s_phi
    y_i_m = (R + r * cos(th_1d)) * -s_phi
    z_i = r * sin(th_1d)
    this_ax.plot(x_i[:idx], y_i_m[:idx], z_i[:idx], color=cmap(c_list[j] / c_list[-1]), lw=this_lw, alpha=this_a)
    this_ax.plot(x_i[:idx], y_i_p[:idx], z_i[:idx], color=cmap(c_list[j] / c_list[-1]), lw=this_lw, alpha=this_a)
    this_ax.plot(x_i[idx:], y_i_m[idx:], z_i[idx:], color=cmap(c_list[j] / c_list[-1]), lw=this_lw, alpha=this_a)
    this_ax.plot(x_i[idx:], y_i_p[idx:], z_i[idx:], color=cmap(c_list[j] / c_list[-1]), lw=this_lw, alpha=this_a)

    return


if __name__ == "__main__":
    elev_min, elev_max, delta_azim = 28., 35., 45  # view options
    t_anim, fps = 20., 25
    nFrames = int(t_anim * fps)

    N, n_th, n_phi = 20, 300, 300
    R, r, eps = 4, 2., 0.01
    mu, l = 0.5, 1.

    bool_anim = True
    w, h = (11, 8) if bool_anim else (11, 6)

    fig = plt.figure(figsize=(w, h), constrained_layout=bool_anim)
    ax = fig.add_subplot(projection='3d')

    th_ = np.linspace(0, 2 * pi, n_th)
    phi_ = np.linspace(0, 2 * pi, n_phi)
    th, phi = np.meshgrid(th_, phi_)

    pot_energy = (1 - cos(phi)) + mu * l * (1 - cos(th))  # = c
    # #  cos(phi) = 1 - c + mu * l * (1 - cos(th))
    #

    x = (R + 0.99*r * cos(th)) * cos(phi)
    y = (R + 0.99*r * cos(th)) * sin(phi)
    z = r * 0.99*sin(th)

    cmap_name = 'jet'
    cmap = plt.get_cmap(cmap_name)
    scalarMap = plt.cm.ScalarMappable(cmap=cmap_name)
    color = scalarMap.to_rgba(pot_energy)
    ax.plot_surface(x, y, z, linewidth=0.0, facecolors=color, alpha=0.10, zorder=0)

    th_eq, ph_eq = np.array([0, 0, pi, pi]), np.array([0, pi, 0, pi])
    x_eq = (R + r * cos(th_eq)) * cos(ph_eq)
    y_eq = (R + r * cos(th_eq)) * sin(ph_eq)
    z_eq = r * sin(th_eq)
    pot_eq = (1 - cos(ph_eq)) + mu * l * (1 - cos(th_eq))
    ax.scatter(x_eq, y_eq, z_eq, s=100, c=cmap(pot_eq/np.amax(pot_energy)), alpha=1)

    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])
    set_axes_equal(ax)

    if bool_anim:

        ax.view_init(elev_max, 100)
        line1a, = ax.plot([], [], [], '-', color='C1', lw=4, zorder=2)
        line1b, = ax.plot([], [], [], '-', color='C1', lw=4, zorder=2)
        line2a, = ax.plot([], [], [], '-', color='C1', lw=4, zorder=2)
        line2b, = ax.plot([], [], [], '-', color='C1', lw=4, zorder=2)

        color, alpha, lw = 'C0', 0.5, 1.
        lines1a = [ax.plot([], [], [], '-', lw=lw, color=color, alpha=alpha)[0] for _ in range(N)]
        lines1b = [ax.plot([], [], [], '-', lw=lw, color=color, alpha=alpha)[0] for _ in range(N)]
        lines2a = [ax.plot([], [], [], '-', lw=lw, color=color, alpha=alpha)[0] for _ in range(N)]
        lines2b = [ax.plot([], [], [], '-', lw=lw, color=color, alpha=alpha)[0] for _ in range(N)]

        time_template = r"$U =$ {:2.0f} % $U_{{max}}$"
        time_text = ax.text2D(0.75, 0.90, "", transform=ax.transAxes, fontsize=15)
        c_list = np.linspace(0 + eps, 2 * (1 + mu * l) - eps, nFrames)

        fig.suptitle(r"Potential energy of the double pendulum on a torus ($\varphi_1, \varphi_2$)", fontsize=ftSz1)

        anim = animation.FuncAnimation(fig, animate, nFrames, interval=1000 / fps, blit=True, init_func=init, repeat=False)
        # anim = animation.FuncAnimation(fig, animate, nFrames, interval=1000. / fps, repeat=False)  # remove blit for save
        # anim.save('./Animations/{:s}.mp4'.format("potential_1"), writer="ffmpeg", dpi=150, fps=fps)
        # anim.save('./Animations/{:s}.html'.format("potential_1"), writer="html", fps=fps)

    else:
        ax.view_init(48, 120)
        ax.set_zlim3d([-r * 1.75, r * 1.75])
        ax.axis("off")

        tot_nb = 20
        selected = [0, 5, 10, 17]
        c_list = np.linspace(0 + eps, 2 * (1 + mu * l) - eps, tot_nb)
        for i, _ in enumerate(c_list):
            if i in selected:
                plot_this_level(ax, i, this_lw=5, this_a=1.)
            else:
                plot_this_level(ax, i, this_lw=1., this_a=0.5)

        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        # fig.savefig("./Figures/potential_torus_3D.svg", format='svg', bbox_inches="tight")

    plt.show()
