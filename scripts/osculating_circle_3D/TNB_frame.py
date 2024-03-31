import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
from numpy import sin, cos, pi
from tqdm import tqdm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 16, 14


def compute_vectors():
    dr = np.gradient(r, h, edge_order=1, axis=1)
    d2r = np.gradient(dr, h, edge_order=1, axis=1)
    d3r = np.gradient(d2r, h, edge_order=1, axis=1)

    ds = np.linalg.norm(dr, axis=0)
    T = dr / ds

    dT = np.gradient(T, h, edge_order=1, axis=1)
    dT_norm = np.linalg.norm(dT, axis=0)
    N = dT / dT_norm

    B = np.cross(T, N, axis=0)

    cross_dr_d2r = np.linalg.norm(np.cross(dr, d2r, axis=0), axis=0)
    k = cross_dr_d2r / np.power(ds, 3)
    tau = np.sum(np.cross(dr, d2r, axis=0) * d3r, axis=0) / np.power(cross_dr_d2r, 2)

    return T[:, s:-s], N[:, s:-s], B[:, s:-s], k[s:-s], tau[s:-s]


def animate(args):
    def update(i):
        i *= skip
        elev = elev_min + (elev_max - elev_min) / 2. * (1. + cos(pi * i / n))
        azim = ax.azim - delta_azim / nFrames
        ax.view_init(elev, azim)

        for arrow, r_ in zip([arrow_T, arrow_N, arrow_B], [r_T, r_N, r_B]):
            arrow.set_data([r[0, i], r_[0, i]], [r[1, i], r_[1, i]])
            arrow.set_3d_properties([r[2, i], r_[2, i]])

        dot_p.set_data([r[0, i]], [r[1, i]])
        dot_p.set_3d_properties([r[2, i]])
        dot_k.set_data(t[i], k[i])
        dot_t.set_data(t[i], tau[i])
        time_text.set_text(time_template.format(t[i]))

        # cx = center[0, i] + 1./k[i] * cos(theta) * T[0, i] + 1./k[i] * sin(theta) * N[0, i]
        # cy = center[1, i] + 1./k[i] * cos(theta) * T[1, i] + 1./k[i] * sin(theta) * N[1, i]
        # cz = center[2, i] + 1./k[i] * cos(theta) * T[2, i] + 1./k[i] * sin(theta) * N[2, i]
        # circle.set_data(cx, cy)
        # circle.set_3d_properties(cz)

        circle[0].remove()
        circle[0] = ax.plot_surface(cx[i], cy[i], cz[i], color='C1', linewidth=0., alpha=tsp, shade=shade, zorder=0)

        return arrow_T, arrow_N, arrow_B, dot_p, dot_k, dot_t, time_text, circle

    T, N, B, k, tau = args
    r_T, r_N, r_B = r + scale_arrow * T, r + scale_arrow * N, r + scale_arrow * B

    fig = plt.figure(figsize=(14., 8.))
    ax = fig.add_subplot(1, 3, (1, 2), projection='3d')
    ax2 = fig.add_subplot(236)
    ax1 = fig.add_subplot(233, sharex=ax2)
    ax1.set_ylabel(r"$\kappa$", fontsize=ftSz2)
    ax2.set_ylabel(r"$\tau$", fontsize=ftSz2)
    ax2.set_xlabel(r"$t$", fontsize=ftSz2)
    plt.setp(ax1.get_xticklabels(), visible=False)
    for this_ax in [ax1, ax2]:
        this_ax.grid(lw=0.5, alpha=0.5)

    ax.plot(r[0], r[1], r[2], color="grey")
    ax1.plot(t, k, color='C1')
    ax2.plot(t, tau, color='C2')

    dot_p, = ax.plot(r[0, 0], r[1, 0], r[2, 0], "o", color='black', ms=1.5 * ms)
    dot_k, = ax1.plot([], [], "o", color='C1', ms=6)
    dot_t, = ax2.plot([], [], "o", color='C2', ms=6)

    arrow_B, = ax.plot([r[0][0], r_B[0][0]], [r[1][0], r_B[1][0]], [r[2][0], r_B[2][0]],
                       marker="o", markevery=[-1], ms=ms, lw=lw, color='C2', zorder=5)
    arrow_N, = ax.plot([r[0][0], r_N[0][0]], [r[1][0], r_N[1][0]], [r[2][0], r_N[2][0]],
                       marker="o", markevery=[-1], ms=ms, lw=lw, color='C1', zorder=7)
    arrow_T, = ax.plot([r[0][0], r_T[0][0]], [r[1][0], r_T[1][0]], [r[2][0], r_T[2][0]],
                       marker="o", markevery=[-1], ms=ms, lw=lw, color='C0', zorder=6)

    time_template = r"$t = {:.2f}$"
    time_text = ax.text2D(0.85, 0.85, "", transform=ax.transAxes, fontsize=ftSz2)
    ax.view_init(elev_max, azim_init)

    # ax.set_aspect('equal')
    # Create cubic bounding box to simulate equal aspect ratio
    fig.tight_layout()
    max_range = (np.amax(r, axis=1) - np.amin(r, axis=1)).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (np.amax(r[0]) + np.amin(r[0]))
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (np.amax(r[1]) + np.amin(r[1]))
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (np.amax(r[2]) + np.amin(r[2]))
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w', alpha=0.)

    ax.set_autoscale_on(False)

    # n_theta = 100
    # theta = np.linspace(0., 2. * pi, n_theta)
    # center = r + (1. / k) * N
    # circle, = ax.plot([], [], [], color="C1", lw=2, alpha=0.5)

    n_xi, n_theta = 2, 50
    xi, theta = np.linspace(0., 1., n_xi), np.linspace(0., 2. * pi, n_theta)
    xi, theta = np.meshgrid(xi, theta)
    xi, theta = xi[np.newaxis, :, :], theta[np.newaxis, :, :]  # axes: time, xi, eta
    center = (r + (1. / k) * N)[:, :, np.newaxis, np.newaxis]  # axes: direction, time, xi, eta
    kc, Tc, Nc = k[:, np.newaxis, np.newaxis], T[:, :, np.newaxis, np.newaxis], N[:, :, np.newaxis, np.newaxis]

    cx = center[0] + xi/kc * cos(theta) * Tc[0] + xi/kc * sin(theta) * Nc[0]
    cy = center[1] + xi/kc * cos(theta) * Tc[1] + xi/kc * sin(theta) * Nc[1]
    cz = center[2] + xi/kc * cos(theta) * Tc[2] + xi/kc * sin(theta) * Nc[2]
    circle = [ax.plot_surface(cx[0], cy[0], cz[0], color='C1', linewidth=0., alpha=tsp, shade=shade, zorder=0)]

    nFrames = n // skip + 1
    # noinspection PyTypeChecker
    anim = FuncAnimation(fig, update, tqdm(range(nFrames)), interval=20)
    if save == "html":
        anim.save(f'./frenet_{name}.html', fps=30)
    elif save == "gif":
        # noinspection PyTypeChecker
        anim.save(f'./frenet_{name}.gif', writer=PillowWriter(fps=25), dpi=100)
    elif save == "mp4":
        anim.save(f'./frenet_{name}.mp4', writer=FFMpegWriter(fps=25))
    elif save == "snapshot":
        update(int(n // 2))
        fig.savefig(f"./frenet_{name}.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()

    return


def xyz():
    if name == "homemade":
        x = sin(t) + 2 * sin(3 * t)
        y = 5 * sin(2 * t)
        z = 4 * cos(t) - cos(5 * t)

    elif name == "tennis":
        a, b = 1., 1./3
        x = a * cos(t) + b * cos(3 * t)
        y = a * sin(t) - b * sin(3 * t)
        z = (2 * np.sqrt(a * b)) * sin(2 * t)

    elif name == "horopter":
        a, b = 1., 1.
        # x = a * (1. + cos(t))
        # y = b * np.tan(t / 2.)
        # z = a * sin(t)
        x = 2. * a / (1. + t * t)
        y = b * t
        z = t * x

    elif name == "viviani":
        # Viviani
        x = cos(t) ** 2
        y = cos(t) * sin(t)
        z = sin(t)

    elif name == "papus":
        alpha = pi / 4.
        x = sin(alpha) * t * cos(t)
        y = sin(alpha) * t * sin(t)
        z = cos(alpha) * t

    else:
        print("Unknown name")
        raise ValueError

    return np.c_[x, y, z].T


if __name__ == "__main__":
    # ffmpeg -ss 0.0 -t 9.5 -i input.mp4 -f gif output.gif
    save, name = "none", "viviani"
    lw, ms, scale_arrow, tsp, shade = 3., 8., 2./3., 0.4, False  # line options
    elev_min, elev_max, azim_init, delta_azim = 10., 10., 100, -60  # view options

    n, skip = 2000, 4
    h = 2 * np.pi / n
    s = 3  # extra values at the beginning and the end to safely differentiate (3rd derivative needed for torsion)
    t0, tf = -pi, pi
    t = np.linspace(t0 - s * h, tf + s * h, n + 1 + 2 * s)

    r = xyz()
    res = compute_vectors()
    t, r = t[s:-s], r[:, s:-s]
    animate(res)
