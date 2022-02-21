# Vincent Degrooff - 2022 - EPL
# Plot multiple Poincare sections with points stored in the "Data" directory

import os
import numpy as np
import matplotlib.pyplot as plt

ftSz1, ftSz2, ftSz3 = 18, 16, 13
plt.rcParams["text.usetex"] = True
plt.rcParams['font.family'] = 'monospace'


def load_file(s, number):
    filename = f"./Data/coordinates_{s}{number:d}.txt"
    if not os.path.exists(filename):
        filename = f"./Data/coordinates_{s}{number-1:d}.txt"
        # raise FileNotFoundError

    with open(filename, "r") as txt_file:
        E, L, MU, mode = [float(x) for x in txt_file.readline().strip().split(" ")[1:]]
        mode = int(mode)

    print(f"{E:7.4f}, {L:5.2f}, {MU:5.2f}, {mode}")
    return np.loadtxt(filename, skiprows=1), E, mode, L, MU


def plot_series(series, file_numbers, custom=False, save=False):
    m = 1
    a = [1., 1., 1., 1., 1., 1.]
    k = [1, 1, 1, 1, 1, 1]
    ms = [0.25, 0.25, 0.35, 0.25, 0.35, 0.15]

    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)

    for i, ax in enumerate(axs.flatten()):
        res, energy, m, l, mu = load_file(series, file_numbers[i])

        if custom:
            ax.plot(res[::k[i], 0], res[::k[i], 1], '.', markersize=ms[i], color='black', alpha=a[i], rasterized=True)
        else:
            ax.plot(res[:, 0], res[:, 1], '.', markersize=0.25, color='black', alpha=1., rasterized=True)

        if series == "":
            x_pos = 0.80
            ax.text(x_pos, 0.92, r"$E={:.2f}$".format(energy), fontsize=11, transform=ax.transAxes)
            ax.text(x_pos, 0.84, r"$\lambda={:.2f}$".format(l), fontsize=11, transform=ax.transAxes)
            ax.text(x_pos, 0.76, r"$\mu={:.2f}$".format(mu), fontsize=11, transform=ax.transAxes)
        else:
            x_pos = 0.73 if energy < 10. else 0.70
            ax.text(x_pos, 0.90, r"$E={:.2f}$".format(energy), fontsize=14, transform=ax.transAxes)

    for ax in axs[:, 0]:
        ax.set_ylabel(r"$\dot \varphi_{:d}$".format(m), fontsize=ftSz2)
    for ax in axs[-1, :]:
        ax.set_xlabel(r"$\varphi_{:d}$".format(m), fontsize=ftSz2)

    if save:
        fig.savefig(f"./Figures/sections/sections_new.svg", format='svg', bbox_inches='tight', dpi=300)
    plt.show()
    return


def plot_section(this_series, this_nb, save=False):
    res, energy, m, l, mu = load_file(this_series, this_nb)

    fig, ax = plt.subplots(1, 1, figsize=(8, 7), constrained_layout=True)

    ax.plot(res[:, 0], res[:, 1], '.', markersize=0.5, color='black', alpha=1., rasterized=True)
    ax.axis([0.82, 1.48, -1.87, -0.72])

    x_pos = 0.02
    ax.text(x_pos, 0.96, r"$E={:.2f}$".format(energy), fontsize=ftSz2, transform=ax.transAxes)
    ax.text(x_pos, 0.90, r"$\lambda={:.2f}$".format(l), fontsize=ftSz2, transform=ax.transAxes)
    ax.text(x_pos, 0.84, r"$\mu={:.2f}$".format(mu), fontsize=ftSz2, transform=ax.transAxes)

    ax.set_xlabel(r"$\varphi_{:d}$".format(m), fontsize=ftSz2)
    ax.set_ylabel(r"$\dot \varphi_{:d}$".format(m), fontsize=ftSz2)

    if save:
        fig.savefig(f"./Figures/sections/alone_new.svg", format='svg', bbox_inches='tight', dpi=300)
    plt.show()
    return


if __name__ == "__main__":

    # plot_series('a', [1, 4, 5, 6, 7, 9])  # L = 1.  MU = 0.1
    # plot_series('b', [1, 2, 3, 4, 5, 8])  # L = 1.  MU = 0.9
    # plot_series('c', [1, 3, 4, 6, 7, 9])  # L = 3.  MU = 0.5
    # plot_series('d', [1, 2, 3, 4, 5, 8])  # L = 1/3 MU = 0.5
    # plot_series('e', [2, 3, 4, 5, 6, 8])  # L = 1 MU = 0.5
    # plot_series('', [1, 4, 5, 6, 7, 8])  # VARIOUS

    b = False
    plt.rcParams["text.usetex"] = b
    plot_section('b', 4, save=b)
