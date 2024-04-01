import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os

from matplotlib.collections import LineCollection
from utils.icon import draw_icon

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

directoryName = "./Figures/"
# inferno, jet, viridis, magma, Blues


def see_path(
        vars_x, vars_y, vars_c, 
        figsize=(10., 6.), var_case=2,
        lws=1., colors=None, shifts=None, pad=None,
        displayedInfo=None, icon_name=None, sim=None,
        name='Figure_1', save=None, 
    ):

    n_plots = len(vars_x)

    if isinstance(vars_x, np.ndarray):
        vars_x = [vars_x]
    
    if isinstance(vars_y, np.ndarray):
        vars_y = [vars_y]

    if isinstance(vars_c, np.ndarray):
        vars_c = [vars_c]

    if shifts is None:
        shifts = [(0, 0)] * n_plots
    elif isinstance(shifts, tuple):
        shifts = [shifts] * n_plots

    if colors is None:
        colors = ['inferno'] * n_plots
    elif isinstance(colors, str):
        colors = [colors] * n_plots

    if isinstance(lws, float):
        lws = [lws] * n_plots

    # w = sqrt(NP dx / dy)
    # h = sqrt(NP dy / dx)

    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    infoPosition, infoHeight, infoSize = 0.975, 0.5, 11
    fig.canvas.set_window_title(name)

    x_min, x_max, y_min, y_max = np.inf, -np.inf, np.inf, -np.inf
    for var_x, var_y, var_c in zip(vars_x, vars_y, vars_c):
        var_c[:] = np.nan_to_num(var_c)
        x_min, x_max = min(x_min, np.nanmin(var_x)), max(x_max, np.nanmax(var_x))
        y_min, y_max = min(y_min, np.nanmin(var_y)), max(y_max, np.nanmax(var_y))
    L_X, L_Y = x_max - x_min, y_max - y_min

    pad_x1, pad_x2, pad_y1, pad_y2 = None, None, None, None
    if var_case == 1:
        pad_x, pad_y = 0.25, 0.1
        ax.set_aspect('equal', "datalim")
    elif var_case == 2:
        pad_x, pad_y = 0.15, 0.15
    elif var_case == 3:
        pad_x1, pad_x2, pad_y1, pad_y2 = 0.1, 0.1, 0.15, 0.6
    
    if pad is not None:
        if len(pad) == 2:
            pad_x, pad_y = pad
        elif len(pad) == 4:
            pad_x1, pad_x2, pad_y1, pad_y2 = pad
        
    if pad_x1 is None:
        pad_x1, pad_x2, pad_y1, pad_y2 = pad_x, pad_x, pad_y, pad_y
        
    x_m, y_m = x_min - pad_x1 * L_X, y_min - pad_y1 * L_Y
    x_M, y_M = x_max + pad_x2 * L_X, y_max + pad_y2 * L_Y
    ax.set_xlim(x_m, x_M)
    ax.set_ylim(y_m, y_M)
    # print(x_min, x_max, y_min, y_max)
    # print(x_m, x_M, y_m, y_M)

    for var_x, var_y, var_c, lw, c, shift in zip(vars_x, vars_y, vars_c, lws, colors, shifts):
        cmap = plt.get_cmap(c)
        delta_c = np.nanmax(var_c) - np.nanmin(var_c)
        norm = plt.Normalize(var_c.min() - delta_c * shift[0], var_c.max() + delta_c * shift[1])
        points = np.array(np.array([var_x, var_y])).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        line = LineCollection(segments, cmap=cmap, norm=norm, lw=lw)
        line.set_array(var_c)
        ax.add_collection(line)

    plt.axis('off')
    plt.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=1.00, wspace=None, hspace=None)

    if displayedInfo is not None:
        textbox = ax.text(
            infoPosition, infoHeight, "\n".join(displayedInfo), 
            weight="light", va='center', ha='right', transform=ax.transAxes, 
            fontdict={'family': 'monospace', 'size': infoSize}
        )
        textbox.set_bbox(dict(facecolor='white', edgecolor=None, alpha=0.25))

    if icon_name is not None and sim is not None:
        draw_icon(ax, icon_name, sim, 0.20, 0.20, 0.04)

    # if save == "save" or save == "erase":
    #     new_number = 0
    #     for filename in os.listdir(directoryName):
    #         if not filename.startswith('figure'):
    #             continue
    #         current_nb = int(filename[7:len(filename) - 4])
    #         if current_nb > new_number:
    #             new_number = current_nb
    #     path = directoryName + 'figure_{}'.format(new_number + int(save == "save")) + ".png"
    #     print(f'number = {new_number}')

    if save is not None and save != "no" and save != "":
        path = directoryName + save
        print("saved at ", path)
        plt.savefig(path, transparent=False)
        plt.close(fig)

    plt.show()
    return
