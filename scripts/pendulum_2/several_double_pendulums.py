import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import physicsim.pendulum_2 as pendulum2

from matplotlib.animation import FuncAnimation, PillowWriter
from tqdm import tqdm
from scripts.pendulum_2.run_pendulum2 import load_configuration

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


def animate_pendulums(t, positions, figsize=(8., 6.), save=''):

    x1 = positions[:, 0, :]
    y1 = positions[:, 1, :]
    x2 = positions[:, 2, :]
    y2 = positions[:, 3, :]
    n_pdl, nt = x1.shape

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x_min, x_max = min(np.amin(x1), np.amin(x2)), max(np.amax(x1), np.amax(x2))
    y_min, y_max = min(np.amin(y1), np.amin(y2)), max(np.amax(y1), np.amax(y2))
    dx, dy = x_max - x_min, y_max - y_min
    sector = patches.Wedge((x_min, y_max), dx/20, theta1=90, theta2=90, color='lightgrey')
    
    x_min, x_max = x_min - 0.1 * dx, x_max + 0.1 * dx
    y_min, y_max = y_min - 0.1 * dy, y_max + 0.1 * dy
    tmp, = ax.plot([x_min, x_max], [y_min, y_max])
    ax.set_aspect("equal", "datalim")
    tmp.remove()

    ax.grid(ls=':')
    time_template = r'$t = %.2f s$'
    time_text = ax.text(0.8, 0.93, '1', fontsize=15, wrap=True, transform=ax.transAxes)
    ax.plot([0, 0], [0, 0], 'o', color='black')

    cmap = plt.get_cmap('magma_r')
    lines = []
    mrkrs = []

    for index in range(n_pdl):
        c = cmap((index + 1) / n_pdl)
        lines.append(ax.plot([], [], '-', color=c, alpha=1.0)[0])
        mrkrs.append(ax.plot([], [], ls="", marker='o', ms=5, color=c)[0])

    def init():
        for line, marker in zip(lines, mrkrs):
            line.set_data([], [])
            marker.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return tuple(lines) + tuple(mrkrs) + (time_text, sector)

    def update(idx):
        for j, (line, marker) in enumerate(zip(lines, mrkrs)):
            line.set_data(
                [0., x1[j, idx], x2[j, idx]], 
                [0., y1[j, idx], y2[j, idx]]
            )
            marker.set_data(
                [x1[j, idx], x2[j, idx]], 
                [y1[j, idx], y2[j, idx]]
            )
        time_text.set_text(time_template % (t[idx]))
        sector.set_theta1(90 - 360 * t[idx] / t[-1])
        ax.add_patch(sector)
        return tuple(lines) + tuple(mrkrs) + (time_text, sector)

    anim = FuncAnimation(
        fig, update, nt, init_func=init, 
        interval=5, blit=True, repeat_delay=3000
    )
    fig.tight_layout()

    if save == "mp4":
        # anim.save('Pendule_multiple_double_5.html', fps=30)
        anim.save('./animations/{:s}.mp4'.format("multiple_pendulum_new"), writer="ffmpeg", dpi=150, fps=30)
    elif save == "gif":
        anim.save('./animations/{:s}.gif'.format("multiple_pendulum"), writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 20.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./figures/{:s}.svg".format("multiple_pendulum"), format="svg", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":

    setup = {"t_sim": 10., "fps": 30, "slowdown": 1., "oversample": 5}
    params = {"g": 9.81, "l1": 1.0, "l2": 1.0, "m1": 1.0, "m2": 1.0}
    initials = {
        "phi1": np.radians(150.), "om1": 0., 
        "phi2": np.radians(150.), "om2": 0.
    }

    # params, initials = load_configuration(5)

    sim = pendulum2.DoublePendulum(setup, params, initials)

    n_sim = 40
    n_steps = sim.n_frames

    phi1_range = np.radians(np.linspace(150, 150.1, n_sim))
    phi2_range = np.radians(np.linspace(150, 150.1, n_sim))
    # phi1_range = np.linspace(initials["phi1"], initials["phi1"] + np.pi/180, n_sim)
    # phi2_range = np.linspace(initials["phi2"], initials["phi2"] + np.pi/180, n_sim)
    positions = np.zeros((n_sim, 4, 1+n_steps))  # x1, y1, x2, y2

    for i in tqdm(range(n_sim)):    
        initials['phi1'] = phi1_range[i]
        initials['phi2'] = phi2_range[i]
        sim = pendulum2.DoublePendulum(setup, params, initials)
        sim.solve_ode(verbose=False)
        kinematics = sim.kinematics
        # x1, y1, v1, x2, y2, v2, ac2, vx2, vy2, acx2, acy2
        positions[i, :, :] = kinematics[[0, 1, 3, 4], :]

    animate_pendulums(sim.t, positions, save="")
