import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from numpy import pi, radians, sin, cos, amin, amax
from scipy.integrate import odeint

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14


########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81  # [m/s²]  -  accélaration de pesanteur
L = 100  # [m]     -  longueur du pendule
h = L + 1  # [m]     -  hauteur de l'accroche du pendule
phi = 80  # [°]     -  latitude
W = 100.  # [tr/j] -  vitesse angulaire terrestre

# alpha_0 = 20     # [°]     -  inclinaison pendule % verticale
# d_alpha = 0      # [°/s]   -  vitesse angulaire de alpha
# beta_0  = 90     # [°]     -  beta = 0 <==> direction nord-sud
# d_beta  =  180   # [°/s]   -  vitesse angulaire de beta

alpha_0 = 80  # [°]     -  inclinaison pendule % verticale
d_alpha = 0  # [°/s]   -  vitesse angulaire de alpha
beta_0 = 0  # [°]     -  beta = 0 <==> direction nord-sud
d_beta = 0  # [°/s]   -  vitesse angulaire de beta

dt = 0.1  # [s]     -  pas de temps
Tend = 120.  # [s]     -  fin de la simulation

# fps = int(1 / dt)  # animation en temps réel
fps = 10
# animation en accéléré

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####
w = 2 * pi / 86400 * W
phi = radians(phi)
U0 = radians(np.array([alpha_0, d_alpha, beta_0, d_beta]))
u0 = U0[:2]

t = np.arange(0, Tend + 0.5 * dt, dt)
n = len(t)


def f_foucault(u, _):
    a, da, b, db = u
    sa, ca, sb, cb = sin(a), cos(a), sin(b), cos(b)
    f1 = sa * ca * db * db - g / L * sa + 2 * w * db * (sin(phi) * sa * ca + cos(phi) * sa * sa * cb)
    f2 = (-2 * ca * da * db - 2 * w * da * (sin(phi) * ca + cos(phi) * sa * cb)) / sa
    return np.array([da, f1, db, f2])


sol1 = odeint(f_foucault, U0, t)
alpha, d_alpha = sol1[:, 0], sol1[:, 1]
beta, d_beta = sol1[:, 2], sol1[:, 3]

x = L * sin(alpha) * cos(beta)
y = L * sin(alpha) * sin(beta)
z = h - L * cos(alpha)
x_min, x_max = amin(x), amax(x)
y_min, y_max = amin(y), amax(y)
z_min, z_max = amin(z), amax(z)
print('   ====== Position initiale ======    \n   x = {:.2f} \t  y = {:.2f} \t  z = {:.2f}'.format(x[0], y[0], z[0]))


########################################################################################################

#####     ================      Création de la figure      ================      #####
def see_animation(save=""):
    ratio = 1 if save == "snapshot" else max(1, n // (int(Tend * fps)))
    plt.rcParams['text.usetex'] = (save == "snapshot")

    fig = plt.figure(figsize=(14., 7.))
    M = L * sin(radians(min(90, alpha_0))) * 1.1
    ax1 = fig.add_subplot(121, xlim=(-M, M), ylim=(-M, M))
    ax2 = fig.add_subplot(243, xlim=(-M, M), ylim=(-0.1*M, 2*M))
    ax3 = fig.add_subplot(247, xlim=(-M, M), ylim=(-0.1*M, 2*M))
    ax1.set_aspect("equal", "datalim")
    ax2.set_aspect("equal", "datalim")
    ax3.set_aspect("equal", "datalim")

    ax_a = fig.add_subplot(244)
    ax_b = fig.add_subplot(248)
    ax_a.set_xlabel(r"$t$", fontsize=ftSz2)
    ax_b.set_xlabel(r"$t$", fontsize=ftSz2)
    ax_a.set_ylabel(r"$\alpha$", fontsize=ftSz2)
    ax_b.set_ylabel(r"$\beta$", fontsize=ftSz2)
    ax_a.xaxis.set_ticklabels([])
    ax_b.xaxis.set_ticklabels([])

    axs = [ax1, ax2, ax3, ax_a, ax_b]
    for ax in axs:
        ax.grid(ls=':')

    ax1.text(0.5, 0.96, 'N', fontsize=15, fontweight='bold', ha='center', wrap=True, transform=ax1.transAxes)  # Nord
    ax1.text(0.96, 0.5, 'E', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Est
    ax1.text(0.01, 0.5, 'O', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Ouest
    ax1.text(0.5, 0.01, 'S', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Sud

    ax2.text(0.93, 0.5, 'E', fontsize=15, fontweight='bold', wrap=True, transform=ax2.transAxes)  # Est
    ax2.text(0.01, 0.5, 'O', fontsize=15, fontweight='bold', wrap=True, transform=ax2.transAxes)  # Ouest

    ax3.text(0.93, 0.5, 'S', fontsize=15, fontweight='bold', wrap=True, transform=ax3.transAxes)  # Sud
    ax3.text(0.01, 0.5, 'N', fontsize=15, fontweight='bold', wrap=True, transform=ax3.transAxes)  # Nord

    ax1.text(0.05, 0.92, r'$\Omega = {:.0f} \; tr/j$'.format(W), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.05, 0.87, r'$L = {:.0f} \; m$'.format(L), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.05, 0.82, r'$\phi = {:.0f} \; °$'.format(180 * phi / pi), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.75, 0.87, r'$\alpha_0 = {:.2f} \; °$'.format(alpha_0), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.75, 0.82, r'$\beta_0 = {:.2f} \; °$'.format(beta_0), fontsize=ftSz3, transform=ax1.transAxes)

    time_template = r'$t \;\:\: = {:.2f} \; s$' if save == "snapshot" else r'$t \;\:\: = \mathtt{{{:.2f}}} \; s$'
    time_text = ax1.text(0.75, 0.92, "", fontsize=ftSz3, transform=ax1.transAxes)

    beta_wrapped = np.fmod(beta, 2 * pi)
    positions = np.where(np.diff(beta_wrapped) < 0.)[0] + 1
    t_with_nan = np.insert(t, positions, np.nan)
    beta_with_nan = np.insert(beta_wrapped, positions, np.nan)
    # alpha_with_nan, beta_with_nan = alpha, beta
    period = t[positions[0]] if len(positions) > 0 else Tend

    ax_a.plot(t, alpha, color='C0')
    ax_b.plot(t_with_nan, beta_with_nan, color='C1')

    ########################################################################################################

    #####     ================      Animation      ================      #####

    line1, = ax1.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax2.plot([], [], 'o-', lw=2, color='C2')
    line3, = ax3.plot([], [], 'o-', lw=2, color='C2')
    line4, = ax1.plot([], [], '-', lw=1, color='grey')
    cursor_a, = ax_a.plot([], [], 'o', markersize=5, color='C0')
    cursor_b, = ax_b.plot([], [], 'o', markersize=5, color='C0')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        time_text.set_text('')
        cursor_a.set_data([], [])
        cursor_b.set_data([], [])
        return line1, line2, line3, line4, time_text, cursor_a, cursor_b, ax_a, ax_b

    def update(i):
        i *= ratio
        start = max((i - 1500, 0))

        thisx0, thisx1, thisx2 = [0, y[i]], [0, y[i]], [0, x[i]]
        thisy0, thisy1, thisy2 = [0, -x[i]], [h, z[i]], [h, z[i]]

        line1.set_data(thisx0, thisy0)
        line2.set_data(thisx1, thisy1)
        line3.set_data(thisx2, thisy2)
        line4.set_data([y[start:i + 1]], [-x[start:i + 1]])
        cursor_a.set_data(t[i], alpha[i])
        cursor_b.set_data(t[i], beta_wrapped[i])
        start = max(0., t[i] - period / 2.)
        time_text.set_text(time_template.format(i * dt))
        for ax_ in [ax_a, ax_b]:
            ax_.set_xlim([start, start + period])

        return line1, line2, line3, line4, time_text, cursor_a, cursor_b, ax_a, ax_b

    fig.tight_layout()
    ani = anim.FuncAnimation(fig, update, n // ratio, interval=5, blit=True, init_func=init, repeat_delay=3000)

    if save == "save":
        ani.save('Foucault_2.html', fps=fps)
    if save == "snapshot":
        update(int(100. * n / Tend))
        fig.savefig("./pendulum_foucault.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


see_animation(save="")
