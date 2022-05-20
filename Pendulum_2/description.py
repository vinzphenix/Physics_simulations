# Vincent Degrooff - 2022 - EPL
# Representation of the double pendulum system

import matplotlib.pyplot as plt
import numpy as np
from numpy import sin, cos, pi
from matplotlib.patches import Arc

ftSz1, ftSz2 = 22, 19
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'


def draw_one_frame(phi1, phi2, l1, l2):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), constrained_layout=True)

    x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
    x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

    ax.plot([0., x1], [0., y1], '-', lw=2, color='C1', alpha=0.8)
    ax.plot([x1, x2], [y1, y2], '-', lw=2, color='C2', alpha=0.8)

    dy1, dy2 = 1. * y1, 1. * (y2 - y1)
    ax.plot([0., 0.], [0., dy1], ':', lw=2, color='grey', alpha=0.8)
    ax.plot([x1, x1], [y1, y1 + dy1], ':', lw=2, color='grey', alpha=0.8)

    ax.plot([0.], [0.], 'o', markersize=15, color='grey')
    ax.plot([x1], [y1], 'o', markersize=15, color='C1')
    ax.plot([x2], [y2], 'o', markersize=15, color='C2')

    x_pos, y_pos, eps = -0.2, -1., 0.025
    plt.arrow(x_pos, y_pos, 0.2, 0., head_length=0.05, head_width=0.035, color='black', width=0.01)
    plt.arrow(x_pos, y_pos, 0., 0.2, head_length=0.05, head_width=0.035, color='black', width=0.01)
    ax.text(x_pos + 0.2, y_pos + 1.35 * eps, r"$x$", size=ftSz2)
    ax.text(x_pos + eps, y_pos + 0.2, r"$y$", size=ftSz2)

    ax.text(0. + eps, 0. + eps, r"$O$", size=ftSz1, color='grey')
    ax.text(x1 + 2*eps, y1 + 2*eps, r"$P_1$", size=ftSz1, color='C1')
    ax.text(x2 + 2*eps, y2 + 2*eps, r"$P_2$", size=ftSz1, color='C2')
    ax.text(x1 - 5*eps, y1 - 3*eps, r"$m_1$", size=ftSz2, color='C1')
    ax.text(x2 - 6*eps, y2 - eps, r"$m_2$", size=ftSz2, color='C2')
    ax.text((0. + x1) / 2. + eps, (0. + y1) / 2. + eps, r"$\ell_1$", size=ftSz2, color='C1')
    ax.text(0.45 * x1 + 0.55 * x2 + eps, 0.45 * y1 + 0.55 * y2 + eps, r"$\ell_2$", size=ftSz2, color='C2')

    dist = 0.5
    arc1 = Arc((0., 0.), dist * l1, dist * l1, 0., 270, 270 + np.degrees(phi1), lw=2, color='C1', alpha=0.8)
    arc2 = Arc((x1, y1), dist * l2, dist * l2, 0., 270, 270 + np.degrees(phi2), lw=2, color='C2', alpha=0.8)
    ax.add_patch(arc1)
    ax.add_patch(arc2)

    angl, dist = 0.35, 0.7 * dist
    x_pos, y_pos = dist * l1 * np.cos(-pi / 2. + phi1 * angl), dist * l1 * np.sin(-pi / 2 + phi1 * angl)
    ax.text(x_pos, y_pos, r"$\varphi_1$", size=ftSz2, color='C1')
    angl = 0.28
    x_pos, y_pos = dist * l2 * np.cos(-pi / 2. + phi2 * angl), dist * l2 * np.sin(-pi / 2 + phi2 * angl)
    ax.text(x1 + x_pos, y1 + y_pos, r"$\varphi_2$", size=ftSz2, color='C2')

    ax.set_aspect('equal', 'datalim')
    ax.axis('off')
    # fig.savefig("./Figures/description.svg", format='svg', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    draw_one_frame(np.pi/3, np.pi/6, 0.66, 1.)
