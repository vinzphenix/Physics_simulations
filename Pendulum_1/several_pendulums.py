import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, radians, pi, amax, abs
from scipy.integrate import odeint
from timeit import default_timer as timer

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g   = 9.81              # [m/s²]  -  accélaration de pesanteur
l   = 0.50              # [m]     -  longueur petit pendule
L   = 1.00              # [m]     -  longueur grand pendule
n_p = 30                # [/]     -  nombre de pendules
D   = 0.1              # [kg/s]  -  coefficient de frottement linéaire
m   = 1                 # [kg]    -  masse pendule

phi0 = 170.0             # [°]     -  angle 1er pendule
om0  = 0.00             # [rad/s] -  v angulaire 1er pendule en résonance

Tend = 30                # [s]    -  fin de la simulation
n = int(400 * Tend)
fps = 20
ratio = n // (int(Tend * fps))

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####
phi0 = radians(phi0)
t = np.linspace(0, Tend, n)

long, long2 = np.linspace(l, L, n_p), np.linspace(l, L, 2 * n_p)
U0 = np.c_[np.linspace(phi0, phi0, n_p), np.linspace(om0, om0, n_p)]

f = lambda u, _, li: np.array([u[1], - g / li * sin(u[0]) - D / m * u[1]])
phi, om = np.zeros((n_p, n)), np.zeros((n_p, n))

tic = timer()
for i in range(n_p):
    sol = odeint(f, U0[i], t, args=(long[i],))
    phi[i], om[i] = sol[:, 0], sol[:, 1]

print("      Elapsed time : %f seconds" % (timer() - tic))

x = sin(phi) * long[:, None]
y = -cos(phi) * long[:, None]

y_max = amax(y) + 0.1 * amax(abs(y))


########################################################################################################

#####     ================      Animation du Système      ================      #####

def see_animation(save):

    #####     ================      Création de la figure      ================      #####

    w, h = 8, 6
    fig, ax = plt.subplots(1, 1, figsize=(w, h), constrained_layout=True)

    if phi0 >= pi / 2:
        ax.axis([-L * 1.1 * w / h, L * 1.1 * w / h, -L * 1.1, L * 1.1])
        ax.set_aspect("equal")
        sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')
    elif phi0 < pi / 2:
        ax.axis([-1.1 * L * sin(phi0) * w / h, 1.1 * L * sin(phi0) * w / h, -2.2 * L * sin(phi0), 0])
        ax.set_aspect("equal")
        sector = patches.Wedge((L * sin(phi0), -2 * L * sin(phi0)), L / 15, theta1=90, theta2=90, color='lightgrey')

    ax.grid(ls=':')
    # line, = ax.plot([], [], lw=2)
    time_template = r'$t = %.2f s$'
    time_text = ax.text(0.8, 0.93, '1', fontsize=15, wrap=True, transform=ax.transAxes)

    cmap = plt.get_cmap('jet')
    lines = []

    for index in range(n_p):
        # = ax1.plot([], [], 'o-', lw=1, color=cmap((index + 1) / n_p))[0]
        lines.append(ax.plot([], [], marker='o', ms=5, color=cmap((index + 1) / n_p))[0])

    #####     ================      Animation      ================      #####

    def init():
        for line in lines:
            line.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return tuple(lines) + (time_text, sector)

    def animate(idx):
        idx *= ratio

        for j, line in enumerate(lines):
            # line.set_data([0, x[j][2 * j]], [0, y[j][2 * j]])
            # line.set_data([x[idx][2 * j]], [y[idx][2 * j]])
            line.set_data([x[j][idx]], [y[j][idx]])

        time_text.set_text(time_template % (t[idx + ratio - 1]))
        sector.set_theta1(90 - 360 * t[idx + ratio - 1] / Tend)
        ax.add_patch(sector)

        return tuple(lines) + (time_text, sector)

    anim = FuncAnimation(fig, animate, n//ratio, init_func=init, interval=5, blit=True, repeat_delay=3000)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.02, top=0.92, wspace=None, hspace=None)

    if save == "save":
        anim.save('Pendule_multiple_1.html', fps=20)
    else:
        plt.show()


see_animation(save="")
