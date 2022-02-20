import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, sqrt, pi, amax, amin, zeros
from scipy.integrate import odeint
from timeit import default_timer as timer
from Utils.Fixed_Path import countDigits, see_path_1, see_path

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g  = 9.810               # [m/s²]  -  accélaration de pesanteur
l  = 0.500               # [m]     -  longueur pendule
mp = 0.001               # [kg]    -  masse pendule
mb = 1.000               # [kg]    -  masse bloc
F  = 260.0               # [N]     -  intensité force
w  = 80.00               # [rad]   -  pulsation force

x_0   = 0.00             # [m]     -  position du bloc
dx_0  = 0.00             # [m/s]   -  vitesse du bloc
phi_0 = pi/4             # [rad]   -  angle 1er pendule
om_0  = 0.00             # [rad/s] -  v angulaire 1er pendule

Tend = 20  # [s]    -  fin de la simulation
n = int(500 * Tend)
fps = 30
ratio = n // (int(Tend * fps))

# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.5, 1e-3, 10.0, 1e4, 10 * g / l, 0, 0, 0.99 * pi / 4 * pi / 2, 0
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.5, 1e-3, 10.0, 1e4, 10 * g / l, 0, 0, 0.1, 0
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 0.001, 5, 100, 3 * sqrt(g / l), 0, 0, pi, pi
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 0.001, 6, 15 * sqrt(g / l), 3 * sqrt(g / l), 0, 0, pi, pi
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 0.001, 6, 100, 3 * sqrt(g / 0.4), 0, 0, pi, pi
#
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 0.001, 6, 100, 3 * sqrt(g / 0.5), 0, 0, pi, pi
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 0.001, 6., 150, 0.613 * g / 0.4, 0, 0, pi, 1.25 * pi
#
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 1.00, 10.0, 200, 4 * sqrt(g / (0.5) ** 2), 0, 0, pi / 4, 0  # [om,cos(phi)]
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 10.0, 10.0, 150, sqrt(g / l) * sqrt(g / l ** 2), 0, 0, 2 * pi / 3, 0
# l, mp, mb, F, w, x_0, dx_0, phi_0, om_0 = 0.4, 10.0 / 4, 10.0, 50, sqrt(g / l ** 2), 0, 0, 3 * pi / 4, 0

########################################################################################################T

#####     ================      Résolution de l'équation différentielle      ================      #####

U0  = np.array([x_0, dx_0, phi_0, om_0])
M = mp+mb


def solver(u, t_):
    x_, dx_, phi_, om_ = u
    s, c = sin(phi_), cos(phi_)
    f1 = (-g * mp * s * s + F * cos(w * t_) - mp * l * om_ * om_ * c) / (M - mp * s * s)
    f2 = s / l * (g - f1)
    return np.array([dx_, f1, om_, f2])


t = np.linspace(0, Tend, n)  # ODEINT

tic = timer()
sol = odeint(solver, U0, t)
print("      Elapsed time : %f seconds" % (timer() - tic))

x, dx, phi, om = sol.T

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

xb, yb = zeros(n + 1), -x
xp, yp = -l * sin(phi), -x + l * cos(phi)

y_min, y_max = amin(yb), amax(yb)
d = abs((y_min - 1.1 * l) - (y_max + 1.1 * l))

vxp = -l * om * cos(phi)
vyp = -dx - l * om * sin(phi)
vp = sqrt(vxp * vxp + vyp * vyp)

########################################################################################################

#####     ================      Animation du Système      ================      #####


def see_animation(variables=(phi, om), save=False):

    #####     ================      Création de la figure      ================      #####

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(121, xlim=(-d / 2, d / 2), ylim=(y_min - 1.2 * l, y_max + 1.2 * l), aspect='equal')
    ax2 = fig.add_subplot(122, xlabel='phi [rad]', ylabel='omega [rad/s]')
    ax.grid(ls=':')
    ax2.grid(ls=':')

    line1, = ax.plot([], [], 'o-', lw=2, color='C2')
    line2, = ax.plot([], [], '-', lw=1, color='grey')
    rect = plt.Rectangle((-l, yb[0] - 0.2 * l), 2 * l, 0.2 * l, color='C1')
    ax.add_patch(rect)

    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')

    time_template = r'$t = %.1f s$'
    time_text = ax.text(0.48, 0.94, '', fontsize=15, transform=ax.transAxes)
    sector = patches.Wedge((d / 2 * 0.85, (y_min - 1.2 * l) * 0.85), d / 20, theta1=90, theta2=90, color='lightgrey')

    ftsz = 11
    ax.text(0.02, 0.96, r'$l  \: \: \: = {:.2f} \: kg$'.format(l), fontsize=ftsz, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$m  = {:.2f} \: kg$'.format(mp), fontsize=ftsz, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.88, r'$M  = {:.2f} \: kg$'.format(mb), fontsize=ftsz, wrap=True, transform=ax.transAxes)
    ax.text(0.19, 0.96, r'$F   = {:.2f} \: N$'.format(F), fontsize=ftsz, wrap=True, transform=ax.transAxes)
    ax.text(0.19, 0.92, r'$\omega  = {:.2f} \: rad/s$'.format(w), fontsize=ftsz, wrap=True, transform=ax.transAxes)

    ax.text(0.74, 0.96, r'$x = {:.2f} $'.format(x_0), fontsize=ftsz, wrap=True, transform=ax.transAxes)
    ax.text(0.74, 0.92, r'$v = {:.2f} $'.format(dx_0), fontsize=ftsz, wrap=True, transform=ax.transAxes)
    ax.text(0.86, 0.96, r'$\varphi  = {:.2f} $'.format(180 / pi * phi_0), fontsize=ftsz, wrap=True,
            transform=ax.transAxes)
    ax.text(0.86, 0.92, r'$\.\varphi  = {:.2f} $'.format(om_0), fontsize=ftsz, wrap=True,
            transform=ax.transAxes)

    ax2.plot(variables[0], variables[1], color='C1')

    ########################################################################################################

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        rect.set_bounds(-l, yb[0] - 0.2 * l, 2 * l, 0.2 * l)
        phase1.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line1, line2, rect, phase1, time_text, sector

    def animate(i):
        i *= ratio
        start = max(0, i-50000)
        thisx = [xb[i], xp[i]]
        thisy = [yb[i], yp[i]]

        line1.set_data(thisx, thisy)
        line2.set_data([xp[start:i + 1]], [yp[start:i + 1]])
        rect.set_y(yb[i] - 0.2 * l)

        time_text.set_text(time_template % (t[i]))
        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(sector)

        phase1.set_data(variables[0][i], variables[1][i])

        return line1, line2, rect, phase1, time_text, sector

    anim = FuncAnimation(fig, animate, n // ratio, interval=5, repeat_delay=3000, init_func=init, blit=True)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save:
        anim.save('Vertical_Inverted_Pendulum_3.html', fps=30)
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
parameters[0] = r"Axe x : $x_p$"
parameters[1] = r"Axe y : $y_p$"
parameters[2] = r"Axe c : $v_p$"

# see_path_1(1.5, np.array([xp, yp]), vp, color='jet', var_case=1, name='0', shift=(0., 0.), save="no", displayedInfo=parameters)
"""see_path_1(1, np.array([phi, om]), vp, color='jet', var_case=2, name='1', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([phi, x]), vp, color='viridis', var_case=2, name='2', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([phi, dx]), vp, color='viridis', var_case=2, name='3', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([om, x]), vp, color='viridis', var_case=2, name='4', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([om, dx]), vp, color='viridis', var_case=2, name='5', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, np.array([x, dx]), vp, color='viridis', var_case=2, name='6', shift=(0., 0.), save="no", displayedInfo=parameters)

see_path_1(1, np.array([om, cos(phi+pi/2)]), vp, color='inferno', var_case=2, name='7', shift=(0., 0.), save="no", displayedInfo=parameters)
#"""

see_animation()
