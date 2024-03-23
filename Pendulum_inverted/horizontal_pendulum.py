import sys
import os

# Add the root directory of your project to sys.path
current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import sin, cos, sqrt, pi, amax, amin, zeros
from scipy.integrate import odeint
from timeit import default_timer as timer
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14
#inferno, viridis, jet, magma

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81  # [m/s²]  -  accélaration de pesanteur
l = 0.40  # [m]     -  longueur pendule
mp = 10.00  # [kg]    -  masse pendule
mb = 6.00  # [kg]    -  masse bloc
F = 81  # [N]     -  intensité force
w = 30.  # [rad]   -  pulsation force # sqrt(g/l)

x_0 = 0  # [m]     -  position du bloc
dx_0 = 0  # [m/s]   -  vitesse du bloc
phi_0 = 0.6 * pi  # [rad]   -  angle pendule
om_0 = 0  # [rad/s] -  v angulaire pendule

Tend = 30.  # [s]    -  fin de la simulation
n = int(500 * Tend)
fps = 30
ratio = n // (int(Tend * fps))


# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 2, 50, 2 * pi * 7, 0, 0, pi / 4, 0)
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 10, 1000, 10 * g / l, 0, 0, pi / 3, 0)
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 10, 100, 3 * sqrt(g / l), 0, 0, pi, pi)  # THIS WAY
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 10.0, 5.00, -80, 10 * sqrt(g / l), 0, 0, pi, 0)

# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.4, 10.0, sqrt(g / l), 80, 4 * sqrt(g / l ** 2), 0, 0, 0.39 * pi, 0)
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.4, 1.00, 10.0, 50, sqrt(g / (0.4) ** 2), 0, 0, pi / 4, 0)  # [om,cos(phi)]
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.4, 10.0, 10.0, 150, sqrt(g / l) * sqrt(g / l ** 2), 0, 0, 2 * pi / 3, 0)
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.4, 10.0 / 4, 10.0, 50, sqrt(g / l ** 2), 0, 0, 3 * pi / 4, 0)

# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 2.5, 10, 50, 2.5*sqrt(g), 0, 0, 3*pi/4, 0
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.5, 1e-3, 10, 1000, 20*g, 0, 0, pi/3, 0

# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.3, 10.0, sqrt(g / 0.3), 80, 4 * sqrt(g / 0.3 ** 2), 0, 0, 0.39 * pi, 0)
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (4., 10.0, sqrt(g / 4.), 80, 4 * sqrt(g / 4. ** 2), 0, 0, 0.39 * pi, 0)
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.4, 10.0, 10.0/4, 50, sqrt(g / l ** 2), 0, 0, 3 * pi / 4, 0)


# THESE LOOK THE SAME
l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 2, 250, 2 * pi * 6.52, 0, 0, pi / 4, 0)  # gif
#l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 2, 500, 2 * pi * 8.51, 0, 0, pi / 4, 0)
#l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 2, 599, 2*pi*10, 0, 0, pi / 4, 0)
#l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 2, 802.95, 2*pi*10, 0, 0, pi / 4, 0)
#l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 2, 750, 2 * pi * 10.1, 0, 0, pi / 4, 0)
#l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = (0.5, 0.001, 2, 1000, 2 * pi * 12., 0, 0, pi / 4, 0)

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

U0 = np.array([x_0, dx_0, phi_0, om_0])
M = mp + mb


def solver(u, t_):
    x_, dx_, phi_, om_ = u
    s, c = sin(phi_), cos(phi_)
    f1 = (F * cos(w * t_) + mp * (g * s * c - l * om_ * om_ * s)) / (M - mp * c ** 2)
    f2 = 1 / l * (f1 * c + g * s)
    return np.array([dx_, f1, om_, f2])


t = np.linspace(0, Tend, n)  # ODEINT

tic = timer()
sol = odeint(solver, U0, t)
print("      Elapsed time : %f seconds" % (timer() - tic))


x, dx  = np.array(sol[:, 0]), np.array(sol[:, 1])  # ODEINT
phi, om = np.array(sol[:, 2]), np.array(sol[:, 3])

#t = sol.t                             # Solve_ivp
#x   = sol.y[0,:] ; dx = sol.y[1,:]
#phi = sol.y[2,:] ; om = sol.y[3,:]

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

xb, yb = x, zeros(n+1)
xp, yp = x - l*sin(phi), l*cos(phi)

x_min, x_max = amin(xb), amax(xb)
d = abs((x_min-l) - (x_max+l))

vxp = dx - l*om*cos(phi)
vyp = -l*om*sin(phi)
vp  = sqrt(vxp*vxp+vyp*vyp)

########################################################################################################

#####     ================      Animation du Système      ================      #####


def see_animation(variables=(phi, om), save=""):
    global ratio, n
    ratio = 1 if save == "snapshot" else ratio
    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

    #####     ================      Création de la figure      ================      #####

    fig = plt.figure(figsize=(15., 6.5))
    ax = fig.add_subplot(121, xlim=(x_min - 1.2 * l, x_max + 1.2 * l), ylim=(-d / 2, d / 2), aspect='equal')
    ax2 = fig.add_subplot(122)
    ax2.set_xlabel(r'$\varphi \rm \; [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \rm \; [rad/s]$', fontsize=ftSz2)
    ax.grid(ls=':')
    ax2.grid(ls=':')

    line1, = ax.plot([], [], 'o-', lw=2, color='C2')
    line2, = ax.plot([], [], '-', lw=1, color='grey')
    rect = plt.Rectangle((xb[0] - l, -0.1 * l), 2 * l, 0.2 * l, color='C1')
    ax.add_patch(rect)

    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.48, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge(((x_max + 1.2 * l) * 0.9, -d / 2 * 0.87), d / 20, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.02, 0.96, r'$l  \: \: \: = {:.3f} \: kg$'.format(l), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$m  = {:.3f} \: kg$'.format(mp), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.88, r'$M  = {:.3f} \: kg$'.format(mb), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.22, 0.96, r'$F   = {:.2f} \: N$'.format(F), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.22, 0.92, r'$\omega  = {:.2f} \: rad/s$'.format(w), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax.text(0.70, 0.96, r'$x_0 = {:.2f} $'.format(x_0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.70, 0.92, r'$v_0 = {:.2f} $'.format(dx_0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.84, 0.96, r'$\varphi_0  = {:.2f} $'.format(180 / pi * phi_0), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)
    ax.text(0.84, 0.92, r'$\dot{{\varphi}}_0  = {:.2f} $'.format(180 / pi * om_0), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)

    ax2.plot(variables[0], variables[1], color='C2')

    ########################################################################################################

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        rect.set_bounds(xb[0] - l, -0.1 * l, 2 * l, 0.2 * l)
        phase1.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line1, line2, rect, phase1, time_text, sector

    def update(i):
        i *= ratio
        start = max(0, i - 20000)
        thisx, thisy = [xb[i], xp[i]], [yb[i], yp[i]]

        line1.set_data(thisx, thisy)
        line2.set_data([xp[start:i + 1]], [yp[start:i + 1]])
        rect.set_x(xb[i] - l)

        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(sector)

        phase1.set_data(variables[0][i], variables[1][i])

        return line1, line2, rect, phase1, time_text, sector

    n //= 16 if save == "gif" else 1
    anim = FuncAnimation(fig, update, n//ratio, interval=20, repeat_delay=3000, blit=True, init_func=init)
    # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)
    fig.tight_layout()

    if save == "save":
        anim.save('Horizontal_Inverted_Pendulum_2.html', fps=30)
    elif save == "gif":
        anim.save('./pendulum_horizontal.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        update(int(12. * n / Tend))
        fig.savefig("./pendulum_horizontal.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


params1 = np.array([l, mp, mb])
params2 = np.array([g, F, w])
params3 = np.array([x_0, dx_0, np.degrees(phi_0), om_0])

dcm1, dcm2, dcm3 = 3, 3, 3
fmt1, fmt2, fmt3 = countDigits(amax(params1)) + 1 + dcm1, 1 + 1 + dcm2, 1 + 1 + dcm3
for val in params2:
    fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)
for val in params3:
    fmt3 = max(fmt3, countDigits(val) + 1 + dcm3)

parameters = [
    r"Axe x : $x_2$",
    r"Axe y : $y_2$",
    r"Axe c : $v_2$",
    "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
    r"$l \quad$ = {:>{width}.{dcm}f} $\rm m$".format(l, width=fmt1, dcm=dcm1),
    r"$m_p$ = {:>{width}.{dcm}f} $\rm kg$".format(mp, width=fmt1, dcm=dcm1),
    r"$m_b$ = {:>{width}.{dcm}f} $\rm kg$".format(mb, width=fmt1, dcm=dcm1),
    "",
    r"$g\,$ = {:>{width}.{dcm}f} $\rm m/s^2$".format(g, width=fmt2, dcm=dcm2),
    r"$F$ = {:>{width}.{dcm}f} $\rm N$".format(F, width=fmt2, dcm=dcm2),
    r"$w$ = {:>{width}.{dcm}f} $\rm rad/s$".format(w, width=fmt2, dcm=dcm2),
    "",
    r"$x \;\;$ = {:>{width}.{dcm}f} $\rm m$".format(x_0, width=fmt3, dcm=dcm3),
    r"$dx$ = {:>{width}.{dcm}f} $\rm m/s$".format(dx_0, width=fmt3, dcm=dcm3),
    r"$\varphi \,\,\,$ = {:>{width}.{dcm}f} $\rm deg$".format(np.degrees(phi_0), width=fmt3, dcm=dcm3),
    r"$\omega \,\,\,$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om_0, width=fmt3, dcm=dcm3)
]
parameters[0] = r"Axe x : $x_p^2$"
parameters[1] = r"Axe y : $y_p$"
parameters[2] = r"Axe c : $v_p$"

#see_path_1(1., np.array([xp, yp]), vp, color='jet', var_case=1, name='0', shift=(0., 0.), save="no", displayedInfo=parameters)

"""
see_path_1(1., np.array([xp, yp]), vp, color='jet', var_case=1, name='0', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([phi, om]), vp, color='jet', var_case=2, name='1', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([phi, x]), vp, color='viridis', var_case=2, name='2', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([phi, dx]), vp, color='viridis', var_case=2, name='3', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([om, x]), vp, color='viridis', var_case=2, name='4', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([om, dx]), vp, color='viridis', var_case=2, name='5', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([x, dx]), vp, color='viridis', var_case=2, name='6', shift=(0., 0.), save="no", displayedInfo=parameters)
"""

see_animation(save="")
