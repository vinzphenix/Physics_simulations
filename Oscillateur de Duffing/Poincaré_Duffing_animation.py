import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

if __name__ == "__main__":

    filename = "samples_3.txt"
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

    for i in range(nFigs):
        print("figure {:d}".format(i))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=False)
        ax.set_xlabel(r'$x$', size=12)
        ax.set_ylabel(r'$v$', rotation=0, size=12)
        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-3.5, 3.5])
        ax.tick_params(axis='both', which='major', labelsize=8)

        ax.set_title(r"$\ddot{x} + \gamma \dot{x} + \alpha x + \beta x^3 \,=\, A \cos{(\omega \, t)}$")
        ax.text(0.025, 0.95, r'$\alpha = {:.2f} $'.format(alpha), fontsize=10, wrap=True, transform=ax.transAxes)
        ax.text(0.025, 0.90, r'$\beta  = {:.2f} $'.format(beta), fontsize=10, wrap=True, transform=ax.transAxes)
        ax.text(0.025, 0.85, r'$\gamma = {:.2f} $'.format(gamma), fontsize=10, wrap=True, transform=ax.transAxes)
        ax.text(0.150, 0.95, r'$A = {:.2f} $'.format(A), fontsize=10, wrap=True, transform=ax.transAxes)
        ax.text(0.150, 0.90, r'$\omega = {:.2f} $'.format(w), fontsize=10, wrap=True, transform=ax.transAxes)

        ax.text(0.785, 0.895, r"$\varphi = {:.3f} \;\pi$".format(2 * i / nFigs), fontsize=12, wrap=True, transform=ax.transAxes)

        df_selected = df[df["fig"] == i]
        line_x = np.array(df_selected["x"])
        line_v = np.array(df_selected["v"])

        ax.plot(line_x, line_v, 'o', markersize=0.5, color='black')
        fig.subplots_adjust(bottom=0.1, top=0.9, left=0.075, right=0.95)
        fig.savefig("./Duffing_3/frame_{:04d}".format(i))
        plt.close()
