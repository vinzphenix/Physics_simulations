import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, radians, array, sqrt, degrees, amax, amin, abs
from scipy.integrate import odeint
from timeit import default_timer as timer
from Utils.Fixed_Path import countDigits, see_path_1

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

# ATTENTION : convention angles relatifs

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81  # [m/s²]  -  accélération de pesanteur
l1 = 0.1  # [m]     -  longueur 1er pendule
l2 = 0.4  # [m]     -  longueur 2d pendule
D = 0.0  # [kg/s]  -  coefficient de frottement linéaire
m = 1  # [kg]  -  masse du 2d pendule

phi1_0 = 0  # [°]     -  angle 1er pendule
phi2_0 = 0  # [°]     -  angle 2em pendule
om1 = -0.333 * sqrt(g / 0.4)  # [rad/s] -  v angulaire 1er pendule
om2_0 = 0  # [rad/s] -  v angulaire 2em pendule

Tend = 60.  # [s]    -  fin de la simulation
n, fps = int(200 * Tend), 30
ratio = n // (int(Tend * fps))

phi1_0, phi2_0 = radians(phi1_0), radians(phi2_0)

# phi1_0, phi2_0, om1, om2_0, l1, l2, m, D = 0, 0, sqrt(g / 0.4), 0., 0.1, 0.4, 1., 0
# liste = [1/2, 2/3, 3/5, 9/17, 17/37]
# om1 *= liste[1]

# phi1_0, phi2_0, om1, om2_0, l1, l2, m, D = 0, 0, 0.5*sqrt(g / 0.1), 0., 0.1, 0.1, 1., 0
# phi1_0, phi2_0, om1, om2_0, l1, l2, m, D = 0, 0, 0.4*sqrt(g / 0.1), 0., 0.1, 0.1, 1., 0
# phi1_0, phi2_0, om1, om2_0, l1, l2, m, D = 0, 0, 0.4*sqrt(g / (0.2/sqrt(2))), 0., 0.2, 0.2/sqrt(2), 1., 0

# phi1_0, phi2_0, om1, om2_0, l1, l2, m, D = 0   , 0    , sqrt(g/0.1) , 0      , 0.1     , 0.1    , 1   , 0

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

U0 = array([phi2_0, om2_0 - om1])
t = np.linspace(0, Tend, n)


def f(u, t_):
    phi, om = u
    # f1 = -D / m * (om + om1) - D * l1 * om1 / (m * l2) * cos(phi) - l1 * om1 * om1 / l2 * sin(phi) - g / l2 * sin(
    #    phi1_0 + om1 * _ + phi)
    f1 = -l1 * om1 * om1 / l2 * sin(phi) - g / l2 * sin(phi1_0 + om1 * t_ + phi)
    return array([om, f1])


tic = timer()
sol1 = odeint(f, U0, t)
print("      Elapsed time : %f seconds" % (timer() - tic))

phi1, om1 = phi1_0 + om1 * t, np.linspace(om1, om1, n)
phi2, om2 = array(sol1[:, 0]), array(sol1[:, 1])

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

x1, y1 = l1 * sin(phi1), -l1 * cos(phi1)
x2, y2 = x1 + l2 * sin(phi1 + phi2), y1 - l2 * cos(phi1 + phi2)

vx2 = l1 * cos(phi1) * om1 + l2 * cos(phi1 + phi2) * (om1 + om2)
vy2 = l1 * sin(phi1) * om1 + l2 * sin(phi1 + phi2) * (om1 + om2)
v2 = sqrt(vx2 * vx2 + vy2 * vy2)

acx2 = -l1 * sin(phi1) * om1**2 - l2 * sin(phi1 + phi2) * (om1 + om2)**2 + l2 * cos(phi1 + phi2) * (-l1 * om1 * om1 / l2 * sin(phi2) - g / l2 * sin(phi1_0 + om1 * t + phi2))
acy2 = +l1 * cos(phi1) * om1**2 + l2 * cos(phi1 + phi2) * (om1 + om2)**2 + l2 * sin(phi1 + phi2) * (-l1 * om1 * om1 / l2 * sin(phi2) - g / l2 * sin(phi1_0 + om1 * t + phi2))
ac2 = sqrt(acx2**2 + acy2**2)

x_max = 1.05 * amax(abs(x2))
y_min, y_max = amin(y2), amax(y2)
y_min, y_max = y_min - 0.05 * abs(y_min), y_max + 0.05 * abs(y_max)

vx_max, vy_max = amax(vx2), amax(vy2)

T = (m * g * cos(phi1) - D * l1 * om1 * sin(phi2) + m * (l1 * om1 * om1 * cos(phi2) + l2 * (om1 + om2) ** 2))
L, T_max = l1 + l2, amax(T)
phim, dphim = amax(abs(phi1 + phi2)), amax(om1 + om2)


########################################################################################################

#####     ================      Animation du Système      ================      #####


def see_animation(save=False):
    #####     ================      Création de la figure      ================      #####

    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(121, xlim=(-L * 1.1, L * 1.1), ylim=(-1.1 * L, L * 1.1), aspect='equal')
    ax.grid(ls=':')
    ax2 = fig.add_subplot(122)
    ax2.grid(ls=':')
    ax2.set_xlabel(r'$\varphi \; \rm [rad]$', fontsize=12)
    ax2.set_ylabel(r'$\omega \; \rm [rad/s]$', fontsize=12)

    time_template = r'$t = %.2f \rm s$'
    time_text = ax.text(0.38, 0.94, '1', fontsize=15, wrap=True, transform=ax.transAxes)
    sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')

    fontsize = 11
    ax.text(0.02, 0.96, r'$l_1 = {:.2f} \: \rm m$'.format(l1), fontsize=fontsize, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$l_2 = {:.2f} \: \rm m$'.format(l2), fontsize=fontsize, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.96, r'$m  = {:.2f} \: \rm kg$'.format(m), fontsize=fontsize, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.92, r'$D  = {:.2f} \: \rm kg/s$'.format(D), fontsize=fontsize, wrap=True, transform=ax.transAxes)

    ax.text(0.56, 0.96, r'$\varphi_1  = {:.2f} \;\rm deg$'.format(degrees(phi1_0)), fontsize=fontsize, wrap=True,
            transform=ax.transAxes)
    ax.text(0.56, 0.92, r'$\varphi_2  = {:.2f} \;\rm deg$'.format(degrees(phi2_0)), fontsize=fontsize, wrap=True,
            transform=ax.transAxes)
    ax.text(0.76, 0.96, r'$\omega_1  = {:.2f} \;\rm rad/s$'.format(degrees(om1[0])), fontsize=fontsize, wrap=True, transform=ax.transAxes)
    ax.text(0.76, 0.92, r'$\omega_2  = {:.2f} \;\rm rad/s$'.format(om2_0), fontsize=fontsize, wrap=True, transform=ax.transAxes)

    ax2.plot(phi1 + phi2, om1 + om2, color='C2')

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], 'o-', lw=2, color='C2')
    line3, = ax.plot([], [], '-', lw=1, color='grey')
    rect = plt.Rectangle((L * 1.0, 0), L * 0.05, T[0] / T_max * L)
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C0')

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        time_text.set_text('')
        rect.set_bounds(L * 1.05, 0, L * 0.05, T[0] / T_max * L)
        phase2.set_data([], [])
        sector.set_theta1(90)
        return line1, line2, line3, time_text, rect, phase2, sector

    def animate(i):
        i *= ratio
        start = max((i - 50000, 0))

        thisx1, thisx2 = [0, x1[i]], [x1[i], x2[i]]
        thisy1, thisy2 = [0, y1[i]], [y1[i], y2[i]]

        line1.set_data(thisx1, thisy1)
        line2.set_data(thisx2, thisy2)
        line3.set_data(x2[start:i + 1], y2[start:i + 1])

        time_text.set_text(time_template % (t[i + ratio - 1]))

        rect.set_bounds(L * 1.0, 0, L * 0.05, T[i] / T_max * L)
        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(rect)
        ax.add_patch(sector)

        phase2.set_data(phi1[i] + phi2[i], om1[i] + om2[i])

        return line1, line2, line3, time_text, rect, phase2, sector

    ani = FuncAnimation(fig, animate, n // ratio,
                        interval=5, blit=True,
                        init_func=init, repeat_delay=5000)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save:
        ani.save('Pendule_entraine_4.html', fps=30)
    else:
        plt.show()


params1 = array([l1, l2])
params2 = array([degrees(phi1_0), degrees(phi2_0), om1[0], om2_0])
dcm1, dcm2 = 3, 4
fmt1, fmt2 = countDigits(amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
for val2 in params2:
    fmt2 = max(fmt2, countDigits(val2) + 1 + dcm2)

parameters = [
    r"Axe x : $x_2$",
    r"Axe y : $y_2$",
    r"Axe c : $v_2$",
    "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
    r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(l1, width=fmt1, dcm=dcm1),
    r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(l2, width=fmt1, dcm=dcm1),
    "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(g), "",
    r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(phi1_0), width=fmt2, dcm=dcm2),
    r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(phi2_0), width=fmt2, dcm=dcm2),
    r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om1[0], width=fmt2, dcm=dcm2),
    r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om2_0, width=fmt2, dcm=dcm2)
]

#see_path_1(1, array([x2, y2]), v2, color='Blues', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)

parameters[0] = r"Axe x : $\varphi_1 + \varphi_2$"
parameters[1] = r"Axe y : $\omega_1 + \omega_2$"
parameters[2] = r"Axe c : $v_2 \; \sqrt{a_2}$"
#see_path_1(1, array([phi1+phi2, (om1+om2)]), -sqrt(ac2)*v2, color='inferno', var_case=2, shift=(-0., 0.), save="save", displayedInfo=parameters)

see_animation()
