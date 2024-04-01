import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection
from numpy import sin, cos
from scipy.integrate import odeint
from time import perf_counter
from tqdm import tqdm

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

def display_paths(
        var_x, var_y, var_c, sweep, 
        label, color='inferno', 
        parameters=None, lw=1.,
        var_case=1, save=None
    ):

    n_sim, n_steps = var_x.shape

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    x_min, x_max = np.amin(var_x), np.amax(var_x)
    y_min, y_max = np.amin(var_y), np.amax(var_y)
    L_X, L_Y = x_max - x_min, y_max - y_min

    if var_case == 1:
        x_m, y_m = x_min - 0.1 * L_X, y_min - 0.1 * L_Y
        x_M, y_M = x_max + 0.1 * L_X, y_max + 0.1 * L_Y
        tmp, = ax.plot([x_m, x_M], [y_m, y_M])
        ax.set_aspect("equal", "datalim")
        tmp.remove()
    elif var_case == 2:
        x_m, y_m = x_min - 0.4 * L_X, y_min - 0.05 * L_Y
        x_M, y_M = x_max + 0.4 * L_X, y_max + 0.05 * L_Y
        ax.axis([x_m, x_M, y_m, y_M])
    elif var_case == 3:
        ax.axis([-15, 2.5, -3.9, 3.9])
    elif var_case == 4:
        ax.axis([-20, 20, -4.25, 4.25])

    cmap = plt.get_cmap(color)
    norm = plt.Normalize(np.amin(var_c[0, :]), np.amax(var_c[0, :]))

    line = LineCollection([], cmap=cmap, norm=norm, lw=lw)
    line.set_array(var_c[0, :])
    ax.add_collection(line)

    plt.axis('off')
    plt.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=1.00, wspace=None, hspace=None)

    mu_template = label + r'$ = %7.4f$'
    mu_text = ax.text(0.05, 0.95, '', fontsize=15, transform=ax.transAxes)
    # s_center = x_max - 0.1*L_X, y_min + 0.1*L_Y
    # patches.Wedge(s_center, L_X / 20, theta1=90, theta2=90, color='lightgrey')
    progress = plt.Rectangle(
        (0.05, 0.9), 0.15, 0.025, edgecolor='lightgrey',
        facecolor='lightgrey', transform=ax.transAxes
    )
    progress_outline = plt.Rectangle(
        (0.05, 0.9), 0.15, 0.025, edgecolor='lightgrey', 
        facecolor='black', transform=ax.transAxes
    )
    ax.add_patch(progress_outline)

    if parameters is not None:
        for i, param in enumerate(parameters):
            x_pos = 0.275 + (i%4) * 0.18
            y_pos = 0.95 - (i//4) * 0.04
            ax.text(x_pos, y_pos, param, fontsize=11, transform=ax.transAxes)

    def init():
        line.set_segments([])
        mu_text.set_text(mu_template % (sweep[0]))
        progress.set_width(0.)
        return line, mu_text, progress

    def animate(j):

        c_min, c_max = np.amin(var_c[j, :]), np.amax(var_c[j, :])
        points = np.array([var_x[j], var_y[j]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line.set_segments(segments)
        line.set_norm(plt.Normalize(c_min, c_max))
        line.set_array(var_c[j, :])

        mu_text.set_text(mu_template % (sweep[j]))
        progress.set_width((j + 1) / n_sim * 0.15)
        ax.add_patch(progress)

        return line, mu_text, progress

    ani = FuncAnimation(
        fig, animate, n_sim, interval=100,
        blit=True, init_func=init, repeat_delay=1000)

    if save is None:
        plt.show()
    else:
        ani.save(save, fps=20)

    return