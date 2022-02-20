import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, radians
from timeit import default_timer as timer
from Utils.Fixed_Path import see_path_1

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81  # accélération de pesanteur
L = 0.25  # longueur du pendule
m = 1.00  # Masse du pendule
D = 0.00  # Coefficient de frottements

phi_0 = 100  # angle initial    -  [°]
om_0 = 0.0  # vitesse initiale -  [°/s]

Tend = 10  # [s]    -  fin de la simulation

n, fps = int(500 * Tend), 30
ratio = n // (int(Tend * fps))
dt = Tend / n

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####
t = np.linspace(0, Tend, n)
U0 = radians(np.array([phi_0, om_0]))


def Runge(u_init):
    U = np.zeros((n, 2))
    U[0] = u_init
    
    f = lambda u: np.array([u[1], - g / L * sin(u[0]) - D / m * u[1]])

    for i in range(n - 1):
        K1 = f(U[i])
        K2 = f(U[i] + K1 * dt / 2)
        K3 = f(U[i] + K2 * dt / 2)
        K4 = f(U[i] + K3 * dt)
        U[i + 1] = U[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6

    return np.array([U[:, 0], U[:, 1]])


tic = timer()
phi, om = Runge(U0)
print("      Elapsed time : %f seconds" % (timer() - tic))

x, vx = L * sin(phi), L * om * cos(phi)
y, vy = -L * cos(phi), L * om * sin(phi)


########################################################################################################

#####     ================      Animation du Système      ================      #####

def see_animation(save=False):

    #####     ================      Création de la figure      ================      #####

    fig, axs = plt.subplots(1, 2, figsize=(10, 5.))

    ax = axs[0]
    ax.axis([-L * 1.1, L * 1.1, -1.1 * L, 1.1 * L])
    ax.set_aspect("equal")
    ax.grid(ls=':')

    ax2 = axs[1]
    ax2.grid(ls=':')
    ax2.set_xlabel('phi [rad]')
    ax2.set_ylabel('omega [rad/s]')

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], '-', lw=1, color='grey')
    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')

    time_template = r'$t = %.1fs $'
    time_text = ax.text(0.45, 0.94, '', fontsize=15, transform=ax.transAxes)
    sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.05, 0.95, r'$L  = {:.2f} m$'.format(L), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.05, 0.90, r'$m  = {:.2f} kg$'.format(m), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.75, 0.95, r'$\varphi_1  = {:.2f} $'.format(phi_0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.75, 0.90, r'$\omega_1  = {:.2f} $'.format(om_0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax2.plot(phi, om, color='C1', label='pendule inertie')

    # ax2.legend()

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        phase1.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line1, line2, phase1, time_text, sector

    def animate(i):
        i *= ratio
        start = max(0, i - 100)

        thisx = [0, x[i]]
        thisy = [0, y[i]]

        line1.set_data(thisx, thisy)
        line2.set_data([x[start:i + 1]], [y[start:i + 1]])
        phase1.set_data(phi[i], om[i])

        time_text.set_text(time_template % (t[i + ratio - 1]))
        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(sector)

        return line1, line2, phase1, time_text, sector

    anim = FuncAnimation(fig, animate, n // ratio, interval=5, blit=True, init_func=init, repeat_delay=3000)
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save:
        anim.save('Pendule_simple_1', fps=30)
    else:
        plt.show()


see_animation()
#see_path(1,array([x,y]),hypot(vx,vy))
