# Vincent Degrooff - 2022 - EPL
# Eigenvalues of the Jacobian of the double pendulum system at its 4 equilibria

import numpy as np
import matplotlib.pyplot as plt
from numpy import sqrt, sin, cos, pi
from matplotlib.ticker import FormatStrFormatter

ftSz1, ftSz2, ftSz3 = 18, 17, 14
plt.rcParams["text.usetex"] = False
plt.rcParams['font.family'] = 'serif'


def compute_eigenvalues(eq, l, m):
    l, m = l.astype(complex), m.astype(complex)
    sq = sqrt(l * l + 4 * l * m - 2 * l + 1)

    if eq == 0:
        eig_1 = +sqrt(l * (1 - m) * (+sq - l - 1.))
        eig_2 = -sqrt(l * (1 - m) * (+sq - l - 1.))
        eig_3 = +sqrt(l * (1 - m) * (-sq - l - 1.))
        eig_4 = -sqrt(l * (1 - m) * (-sq - l - 1.))
    elif eq == 1:
        eig_1 = +sqrt(l * (1 - m) * (+sq - l + 1.))
        eig_2 = -sqrt(l * (1 - m) * (+sq - l + 1.))
        eig_3 = +sqrt(l * (1 - m) * (-sq - l + 1.))
        eig_4 = -sqrt(l * (1 - m) * (-sq - l + 1.))
    elif eq == 2:
        eig_1 = +sqrt(l * (1 - m) * (+sq + l - 1.))
        eig_2 = -sqrt(l * (1 - m) * (+sq + l - 1.))
        eig_3 = +sqrt(l * (1 - m) * (-sq + l - 1.))
        eig_4 = -sqrt(l * (1 - m) * (-sq + l - 1.))
    elif eq == 3:
        eig_1 = +sqrt(l * (1 - m) * (+sq + l + 1.))
        eig_2 = -sqrt(l * (1 - m) * (+sq + l + 1.))
        eig_3 = +sqrt(l * (1 - m) * (-sq + l + 1.))
        eig_4 = -sqrt(l * (1 - m) * (-sq + l + 1.))
    else:
        raise ValueError

    return np.array([eig_1, eig_2, eig_3, eig_4]) / (sqrt(2.) * l * (1. - m))


if __name__ == "__main__":

    n_lambda, N = 30, 30

    fig, axs = plt.subplots(2, 4, figsize=(12.5, 7), constrained_layout=True, sharey='row', sharex='row')
    lin_sym = np.linspace(-5., 5., n_lambda)

    l_fixed = 2.
    param_l = np.ones(n_lambda) * l_fixed  # sqrt(1 + lin_sym ** 2) - lin_sym
    param_mu = np.linspace(0.05, 0.95, N)  # np.ones(N) * 0.5

    titles = [r'$(\varphi_1=0, \varphi_2=0)$ $E_1=0$', r'$(\varphi_1=0, \varphi_2=\pi)$ $E_2=2\mu\lambda$',
              r'$(\varphi_1=\pi, \varphi_2=0)$ $E_3=2$',
              r'$(\varphi_1\!=\!\pi, \varphi_2\!=\!\pi)$ $E_4\!=\!2\!+\!2 \mu \lambda$']

    phi_list = [(0., 0.), (0, pi), (pi, 0), (pi, pi)]
    for i, ax in enumerate(axs[0, :]):
        l1, l2 = 1 / (1 + l_fixed), l_fixed / (1 + l_fixed)
        phi1, phi2 = phi_list[i]
        x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
        x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)
        ax.plot([0., x1], [0., y1], 'o', ls='-', lw=5, ms=5, color='C1', alpha=0.5, markevery=[-1])
        ax.plot([x1, x2], [y1, y2], 'o', ls='-', lw=2, ms=5, color='C2', alpha=1., markevery=[-1])
        ax.plot([0.], [0.], 'o', lw=2, color='black', alpha=1., zorder=0)
        ax.axis([-1., 1., -1.1, 1.1])
        ax.axis('off')

    for i, ax in enumerate(axs[1, :]):
        eigs = compute_eigenvalues(i, param_l, param_mu)
        for j, eig in enumerate(eigs):
            length = max(abs(eig.real[-1] - eig.real[0]), abs(eig.imag[-1] - eig.imag[0]))
            ax.plot(eig.real, eig.imag, '.', ls='-', label=r'$\alpha_{:d}$'.format(j), lw=1., color='C'+str(j))
            ax.arrow(eig.real[N // 2], eig.imag[N // 2],
                     eig.real[N // 2 + 1] - eig.real[N // 2], eig.imag[N // 2 + 1] - eig.imag[N // 2],
                     shape='full', color='C' + str(j), head_length=.4, head_width=0.75, lw=0.)

        limits = ax.get_ylim()
        ax.vlines(0., limits[0], limits[1], ls='-', lw=3, color='lightgrey', zorder=0)
        ax.set_ylim(limits)
        ax.grid(ls=':')
        ax.legend(fontsize=ftSz3)
        ax.set_title(titles[i], fontsize=ftSz2)
        ax.set_xlabel(r'$\Re(\alpha)$', fontsize=ftSz2)

    axs[1, 0].set_ylabel(r'$\Im(\alpha)$', fontsize=ftSz2)
    axs[1, 0].yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
    template = r"Equilibria of the system for $\alpha={:.2f}$ and $\mu \in [{:.2f}, {:.2f}]$"
    # fig.suptitle(template.format(l_fixed, param_mu[0], param_mu[-1]), fontsize=ftSz1)

    # fig.savefig("./Figures/eigenvalues_latex_{:.2f}.svg".format(l_fixed), format='svg', bbox_inches='tight')
    plt.show()
