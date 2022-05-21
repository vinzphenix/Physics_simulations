import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.collections import LineCollection
from numpy import array, amax, concatenate, amin

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

directoryName = "C:/Users/vince/PycharmProjects/Project_Perso/Simulations/figures/"
#directoryName = "C:/Users/vince/Pictures/Iphone - Synchronisation/Simulations/"

# inferno, jet, viridis, magma, Blues


def countDigits(a):
    if a < 0.:
        return max(2, 2 + int(np.log10(np.abs(a))))
    elif a > 0.:
        return max(1, 1 + int(np.log10(np.abs(a))))
    else:
        return 1


def see_path_1(lw, variables, colorarray, color='jet', name='Figure_1', shift=(0, 0),
               var_case=2, bar=False, save="no", displayedInfo=""):
    plt.style.use('dark_background')

    if save == "save" or save == "erase":
        fig = plt.figure(figsize=(16, 9))
        infoPosition, infoHeight, infoSize = 0.875, 0.25, 13
    else:
        fig = plt.figure(figsize=(16 * 0.65, 9 * 0.65))
        infoPosition, infoHeight, infoSize = 0.875, 0.15, 11
    fig.canvas.set_window_title(name)

    L_X, L_Y = amax(variables[0]) - amin(variables[0]), amax(variables[1]) - amin(variables[1])
    L_V = amax(colorarray) - amin(colorarray)
    if var_case == 1:
        x_m, y_m = amin(variables[0]) - 0.25 * L_X, amin(variables[1]) - 0.1 * L_Y
        x_M, y_M = amax(variables[0]) + 0.25 * L_X, amax(variables[1]) + 0.1 * L_Y
        ax = fig.add_subplot(111, xlim=(x_m, x_M), ylim=(y_m, y_M), aspect='equal')
    elif var_case == 2:
        x_m, y_m = amin(variables[0]) - 0.20 * L_X, amin(variables[1]) - 0.15 * L_Y
        x_M, y_M = amax(variables[0]) + 0.20 * L_X, amax(variables[1]) + 0.15 * L_Y
        ax = fig.add_subplot(111, xlim=(x_m, x_M), ylim=(y_m, y_M))
    elif var_case == 3:
        x_m, y_m = amin(variables[0]) - 0.1 * L_X, amin(variables[1]) - 0.15 * L_Y
        x_M, y_M = amax(variables[0]) + 0.1 * L_X, amax(variables[1]) + 0.6 * L_Y
        ax = fig.add_subplot(111, xlim=(x_m, x_M), ylim=(y_m, y_M))
        fig.set_size_inches(6, 6 * 9 / 5)
    elif var_case == 4:
        ax = fig.add_subplot(111, xlim=(-30, 30), ylim=(-4, 4))
    else:
        print("Choose a var_case <= 4 !!!")
        return

    cmap = plt.get_cmap(color)
    norm = plt.Normalize(colorarray.min() - L_V * shift[0], colorarray.max() + L_V * shift[1])

    points = array(variables).T.reshape(-1, 1, 2)
    segments = concatenate([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap=cmap, norm=norm, lw=lw)
    line.set_array(colorarray)
    ax.add_collection(line)

    if bar:
        cbar = fig.colorbar(line)
        cbar.ax.set_ylabel('vitesse du pendule [m/s]')
        plt.subplots_adjust(left=0.15, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)
        ax.grid(ls=":")
    else:
        plt.axis('off')
        plt.subplots_adjust(left=0.00, right=1.00, bottom=0.00, top=1.00, wspace=None, hspace=None)

    if displayedInfo != "":
        textbox = ax.text(infoPosition, infoHeight, "\n".join(displayedInfo), weight="light",
                          transform=ax.transAxes, fontdict={'family': 'monospace', 'size': infoSize})
        textbox.set_bbox(dict(facecolor='white', edgecolor=None, alpha=0.25))

    if save == "save" or save == "erase":  # or text=='yes':
        number = 0
        for filename in os.listdir(directoryName):
            if filename.startswith('Figure') and int(filename[7:len(filename) - 4]) > number:
                number = int(filename[7:len(filename) - 4])
        path = directoryName + 'Figure_{}'.format(number + int(save == "save")) + ".png"
        print('number = {}'.format(number))
        plt.savefig(path, bbox_inches="tight")

    plt.show()


def see_path(lw, variables, colorarrays, colorList=('Inferno',),
             shifts=None, var_case=1, save="no", displayedInfo=""):
    if shifts is None:
        shifts = [(0, 0), (0, 0)]
    n = len(variables)
    if len(colorList) != n:
        colorList = colorList[0] * n

    plt.style.use('dark_background')
    if save == "save" or save == "erase":
        fig = plt.figure(figsize=(16, 9))
        infoHeight, infoSize = 0.3, 13
    else:
        fig = plt.figure(figsize=(16 * 0.65, 9 * 0.65))
        infoHeight, infoSize = 0.25, 11

    x_min, x_max, y_min, y_max = float('inf'), -float('inf'), float('inf'), -float('inf')
    for x, y in variables:
        x_min, y_min = min(x_min, amin(x)), min(y_min, amin(y))
        x_max, y_max = max(x_max, amax(x)), max(y_max, amax(y))
    L_X, L_Y = x_max - x_min, y_max - y_min

    if var_case == 1:
        x_m, y_m = x_min - 0.2 * L_X, y_min - 0.2 * L_Y
        x_M, y_M = x_max + 0.2 * L_X, y_max + 0.2 * L_Y
        aspect = 'equal'
        infoPosition = 1.025
    else:
        x_m, y_m = x_min - 0.65 * L_X, y_min - 0.1 * L_Y
        x_M, y_M = x_max + 0.65 * L_X, y_max + 0.1 * L_Y
        aspect = 'auto'
        infoPosition = 0.875

    ax = fig.add_subplot(111, xlim=(x_m, x_M), ylim=(y_m, y_M), aspect=aspect)

    for variable, colorarray, color, shift in zip(variables, colorarrays, colorList, shifts):
        cmap, L_V = plt.get_cmap(color), amax(colorarray) - amin(colorarray)
        # norm = plt.Normalize(0, colorarray.max() * 1.05)
        norm = plt.Normalize(colorarray.min() - L_V * shift[0], colorarray.max() + L_V * shift[1])

        points = array(variable).T.reshape(-1, 1, 2)
        segments = concatenate([points[:-1], points[1:]], axis=1)
        line = LineCollection(segments, cmap=cmap, norm=norm, lw=lw)
        line.set_array(colorarray)
        ax.add_collection(line)

    plt.axis('off')
    plt.subplots_adjust(left=0.05, right=1.00, bottom=0.05, top=0.95, wspace=None, hspace=None)

    if displayedInfo != "":
        textbox = ax.text(infoPosition, infoHeight, "\n".join(displayedInfo), weight="light",
                          transform=ax.transAxes, fontdict={'family': 'monospace', 'size': infoSize})
        textbox.set_bbox(dict(facecolor='white', edgecolor=None, alpha=0.25))

    if save == "save" or save == "erase":  # or text=='yes':
        number = 0
        for filename in os.listdir(directoryName):
            if filename.startswith('Figure') and int(filename[7:len(filename) - 4]) > number:
                number = int(filename[7:len(filename) - 4])

        path = directoryName + 'Figure_{}'.format(number + int(save == "save"))
        print('number = {}'.format(number))
        plt.savefig(path, bbox_inches="tight")  # , dpi=500, transparent=True)

    plt.show()
