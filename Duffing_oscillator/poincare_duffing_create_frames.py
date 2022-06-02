import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

plt.rcParams['font.family'] = 'serif'
plt.rcParams['mathtext.rm'] = 'serif'

# plt.rcParams['text.usetex'] = True
plt.rcParams['mathtext.fontset'] = 'cm'

ftSz1, ftSz2, ftSz3 = 20, 12, 10


if __name__ == "__main__":

    sim_nb = 1
    show_axis = True
    filename = f"samples_{sim_nb:d}.txt"
    file = open(filename, 'r')

    line = next(file)
    alpha, beta, gamma, A, w, delta_t = [float(x) for x in line.split()]
    line = next(file)
    nFigs, nRuns, nPoints = [int(x) for x in line.split()]

    df = pd.read_csv(file, delimiter=" ", names=["fig", "x", "v"], index_col=False)

    x_lim = (-1.5, 1.5)
    y_lim = (-0.5, 1.0)
    L_x, L_y = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
    m_x, m_y = (x_lim[1] + x_lim[0]) / 2, (y_lim[1] + y_lim[0]) / 2

    fig, ax = plt.subplots(1, 1, figsize=(8., 5.), constrained_layout=False)

    # ax.axis([-1., 1., -3.5, 3.5])
    ax.axis([-1.6, 1.6, -1.2, 1.2])
    # ax.axis([-1.6, 1.6, -2.2, 2.2])

    if show_axis:
        ax.set_xlabel(r'$x$', size=ftSz2)
        ax.set_ylabel(r'$\dot x$', rotation=0, size=ftSz2)
        ax.set_title(r"$\ddot{x} + \gamma \dot{x} + \alpha x + \beta x^3 \,=\, A \cos{(\omega \, t)}$")
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.text(0.025, 0.95, r'$\alpha = {:.2f} $'.format(alpha), fontsize=ftSz3, transform=ax.transAxes)
        ax.text(0.025, 0.90, r'$\beta  = {:.2f} $'.format(beta), fontsize=ftSz3, transform=ax.transAxes)
        ax.text(0.025, 0.85, r'$\gamma = {:.2f} $'.format(gamma), fontsize=ftSz3, transform=ax.transAxes)
        ax.text(0.150, 0.95, r'$A = {:.2f} $'.format(A), fontsize=ftSz3, transform=ax.transAxes)
        ax.text(0.150, 0.90, r'$\omega = {:.2f} $'.format(w), fontsize=ftSz3, transform=ax.transAxes)
    else:
        ax.axis("off")

    phi_template = r"$\varphi = {:.2f} \;\pi$" if show_axis else None
    phi_text = ax.text(0.86, 0.91, "", fontsize=ftSz2, transform=ax.transAxes) if show_axis else None

    dots, = ax.plot([], [], 'o', markersize=0.5, color='black')
    # fig.subplots_adjust(bottom=0.1, top=0.9, left=0.075, right=0.95)
    fig.tight_layout()

    for i in tqdm(range(nFigs)):

        if show_axis:
            phi_text.set_text(phi_template.format(2 * i / nFigs))

        df_selected = df[df["fig"] == i]
        line_x = np.array(df_selected["x"])
        line_v = np.array(df_selected["v"])
        dots.set_data(line_x, line_v)

        str_additional = "" if show_axis else "_no_axis"
        fig.savefig(f"./frames/case_{sim_nb:d}{str_additional:s}/frame_{i:05d}", transparent=not show_axis)
