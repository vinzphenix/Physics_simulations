import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import physicsim.pendulum_1 as pendulum_1

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, pi
from scipy.integrate import odeint
from timeit import default_timer as timer
from tqdm import tqdm

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


def animate_pendulums(t, x, y, figsize=(8., 6.), save=''):

    n_pdl, nt = x.shape
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    x_min, x_max = np.min(x), np.max(x)
    y_min, y_max = np.min(y), np.max(y)
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

    cmap = plt.get_cmap('jet')
    lines = []
    markers = []

    for index in range(n_pdl):
        c = cmap((index + 1) / n_pdl)
        lines.append(ax.plot([], [], '-', color=c, alpha=0.2)[0])
        markers.append(ax.plot([], [], ls="", marker='.', ms=5, color=c)[0])

    def init():
        for line, marker in zip(lines, markers):
            line.set_data([], [])
            marker.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return tuple(lines) + tuple(markers) + (time_text, sector)

    def animate(idx):
        for j, (line, marker) in enumerate(zip(lines, markers)):
            line.set_data([0., x[j, idx]], [0., y[j, idx]])
            marker.set_data([x[j, idx]], [y[j, idx]])
        time_text.set_text(time_template % (t[idx]))
        sector.set_theta1(90 - 360 * t[idx] / t[-1])
        ax.add_patch(sector)
        return tuple(lines) + tuple(markers) + (time_text, sector)

    anim = FuncAnimation(
        fig, animate, nt, init_func=init, 
        interval=5, blit=True, repeat_delay=3000
    )
    fig.tight_layout()

    if save == "save":
        anim.save('Pendule_multiple_1.html', fps=20)
    else:
        plt.show()


if __name__ == "__main__":

    setup = {"t_sim": 60., "fps": 30, "slowdown": 4, "oversample": 10}
    params = {"g": 9.81, "l": 0.5, "D": 0.01, "m": 1.}
    initials = {"phi": np.radians(170.), "om": 0.}

    sim = pendulum_1.Pendulum(setup, params, initials)

    n_sim = 100
    n_frames = sim.n_frames

    l_range = np.linspace(0.5, 1.0, n_sim)
    x = np.zeros((n_sim, 1+n_frames))
    y = np.zeros((n_sim, 1+n_frames))

    for i in tqdm(range(n_sim)):    
        params['l'] = l_range[i]
        sim = pendulum_1.Pendulum(setup, params, initials)
        sim.solve_ode(verbose=False)
        phi, om = sim.series
        x[i, :] = +sim.l * sin(phi)
        y[i, :] = -sim.l * (cos(phi))

    animate_pendulums(sim.t, x, y)
