import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, linspace, array, sqrt, amax, degrees
from scipy.integrate import odeint
from timeit import default_timer as timer
from Utils.Fixed_Path import countDigits, see_path_1, see_path

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g   = 9.81             # [m/s²]  -  accélaration de pesanteur
l1  = 0.2               # [m]     -  longueur 1er pendule
l2  = 0.2               # [m]     -  longueur 2em pendule
l3  = 0.2               # [m]     -  longueur 3em pendule
m1  = 0.1               # [kg]    -  masse 1er pendule
m2  = 0.1               # [kg]    -  masse 2em pendule
m3  = 0.1               # [kg]    -  masse 3em pendule


phi1_0 = -0.20813379             # [°]     -  angle 1er pendule
phi2_0  = -0.47019033            # [°]     -  angle 2e pendule
phi3_0 = 0.80253405              # [°]     -  angle 2e pendule
om1_0  = -4.0363589              # [rad/s] -  v angulaire 1er pendule
om2_0 = 4.42470966               # [rad/s] -  v angulaire 2em pendule
om3_0  = 8.3046730               # [rad/s] -  v angulaire 2em pendule

Tend = 7.  # [s]    -  fin de la simulation
n = int(1050 * Tend)
fps = 30
ratio = n // (int(Tend * fps))

# phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = -0.06113, 0.42713, 2.01926, 0, 0, 0
# m1, m2, m3, l1, l2, l3 = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1

# phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = -0.20813379, -0.47019033, 0.80253405, -4.0363589, 4.42470966, 8.3046730
# m1, m2, m3, l1, l2, l3 = 0.1, 0.1, 0.1, 0.15, 0.1, 0.1

# phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = -0.22395671, 0.47832902, 0.22100014, -1.47138911, 1.29229544, -0.27559337
# m1, m2, m3, l1, l2, l3 = 0.1, 0.2, 0.1, 0.15, 0.2, 0.3

# phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = -0.78539816, 0.79865905, 0.72867705, 0.74762606, 2.56473963, -2.05903234
# m1, m2, m3, l1, l2, l3 = 0.35, 0.2, 0.3, 0.3, 0.2, 0.25

# phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = 1.30564176, 1.87626915, 1.13990186, 0.75140557, 1.65979939, -2.31442362
# m1, m2, m3, l1, l2, l3 = 0.35, 0.2, 0.3, 0.3, 0.2, 0.25

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

L, M, m = l1 + l2 + l3, m1 + m2 + m3, m2 + m3
#phi1_0, phi2_0, phi3_0 = radians([phi1_0, phi2_0, phi3_0])

U0 = array([phi1_0, om1_0, phi2_0, om2_0, phi3_0, om3_0])


def f(u, _):
    th1, w1, th2, w2, th3, w3 = u

    C31, S31 = cos(th3 - th1), sin(th3 - th1)
    C32, S32 = cos(th3 - th2), sin(th3 - th2)
    C21, S21 = cos(th2 - th1), sin(th2 - th1)

    Num1 = (m3 * C31 * C32 - m * C21) / (m - m3 * C32 * C32) * (l1 * w1 * w1 * (
            -m * S21 + m3 * S31 * C32) + m3 * l2 * w2 * w2 * S32 * C32 + m3 * l3 * w3 * w3 * S32 - m * g * sin(
        th2) + m3 * g * sin(th3) * C32)
    Num2 = m * l2 * w2 * w2 * S21 + m3 * C31 * (
        g * sin(th3) + l2 * w2 * w2 * S32 + l1 * w1 * w1 * S31) + m3 * l3 * w3 * w3 * S31 - M * g * sin(th1)
    Den = l1 * (M + (m * C21 - m3 * C31 * C32) * (-m * C21 + m3 * C31 * C32) / (m - m3 * C32 * C32) - m3 * C31 * C31)

    f1 = (Num1 + Num2) / Den
    f2 = (f1 * (-l1 * m * C21 + m3 * l1 * C31 * C32) + l1 * w1 * w1 * (
            -m * S21 + m3 * S31 * C32) + m3 * l2 * w2 * w2 * S32 * C32 + m3 * l3 * w3 * w3 * S32 - m * g * sin(
        th2) + m3 * g * sin(th3) * C32) / (l2 * (m - m3 * C32 * C32))
    f3 = 1 / l3 * (-g * sin(th3) - l2 * w2 * w2 * S32 - l2 * f2 * C32 - l1 * w1 * w1 * S31 - l1 * f1 * C31)

    return array([w1, f1, w2, f2, w3, f3])


t = linspace(0, Tend, n)  # ODEINT

tic = timer()
sol = odeint(f, U0, t)
print("      Elapsed time : %f seconds" % (timer() - tic))

phi1, om1, phi2, om2, phi3, om3 = sol.T

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

x1, y1 = 0. + l1 * sin(phi1), 0. - l1 * cos(phi1)
x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)
x3, y3 = x2 + l3 * sin(phi3), y2 - l3 * cos(phi3)

vx2 = -l1 * om1 * sin(phi1) - l2 * om2 * sin(phi2)
vy2 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2)
v2 = sqrt(vx2 * vx2 + vy2 * vy2)

vx3 = -l1 * om1 * sin(phi1) - l2 * om2 * sin(phi2) - l3 * om3 * sin(phi3)
vy3 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2) + l3 * om3 * cos(phi3)
v3 = sqrt(vx3 * vx3 + vy3 * vy3)


########################################################################################################

#####     ================      Animation du Système      ================      #####

def see_animation(save=False):

    #####     ================      Création de la figure      ================      #####

    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)

    ax = axs[0]
    ax.axis([-1.1 * L, 1.1 * L, -1.1 * L, 1.1 * L])
    ax.set_aspect("equal")

    ax2 = axs[1]
    ax2.set_xlabel('phi [rad]')
    ax2.set_ylabel('omega [rad/s]')

    ax.grid(ls=':')
    ax2.grid(ls=':')

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], 'o-', lw=2, color='C2')
    line3, = ax.plot([], [], 'o-', lw=2, color='C3')
    line4, = ax.plot([], [], '-', lw=1, color='grey')
    line5, = ax.plot([], [], '-', lw=1, color='lightgrey')

    sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')

    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase3, = ax2.plot([], [], marker='o', ms=8, color='C0')

    time_template = r'$t = %.1f s$'
    time_text = ax.text(0.42, 0.94, '', fontsize=15, transform=ax.transAxes)

    ax.text(0.02, 0.96, r'$l_1  = {:.2f} \: \rm m$'.format(l1), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$l_2  = {:.2f} \: \rm m$'.format(l2), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.88, r'$l_3  = {:.2f} \: \rm m$'.format(l3), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.96, r'$m_1  = {:.2f} \: \rm kg$'.format(m1), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.92, r'$m_2  = {:.2f} \: \rm kg$'.format(m2), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.88, r'$m_3  = {:.2f} \: \rm kg$'.format(m3), fontsize=12, wrap=True, transform=ax.transAxes)

    ax.text(0.62, 0.96, r'$\varphi_1  = {:.2f}$'.format(phi1_0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.62, 0.92, r'$\varphi_2  = {:.2f}$'.format(phi2_0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.62, 0.88, r'$\varphi_3  = {:.2f}$'.format(phi3_0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.82, 0.96, r'$\omega_1  = {:.2f}$'.format(om1_0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.82, 0.92, r'$\omega_2  = {:.2f}$'.format(om2_0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.82, 0.88, r'$\omega_3  = {:.2f}$'.format(om3_0), fontsize=12, wrap=True, transform=ax.transAxes)

    ax2.plot(phi1, om1, color='C1')
    ax2.plot(phi2, om2, color='C2')
    ax2.plot(phi3, om3, color='C3')

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])
        sector.set_theta1(90)
        phase1.set_data([], [])
        phase2.set_data([], [])
        phase3.set_data([], [])
        time_text.set_text('')
        return line1, line2, line3, line4, line5, phase1, time_text, phase2, phase3, sector

    def animate(i):
        i *= ratio
        start = max(0, i - 10000)
        thisx, thisx2, thisx3 = [0, x1[i]], [x1[i], x2[i]], [x2[i], x3[i]]
        thisy, thisy2, thisy3 = [0, y1[i]], [y1[i], y2[i]], [y2[i], y3[i]]

        line1.set_data(thisx, thisy)
        line2.set_data(thisx2, thisy2)
        line3.set_data(thisx3, thisy3)
        line4.set_data([x3[start:i + 1]], [y3[start:i + 1]])
        line5.set_data([x2[start:i + 1]], [y2[start:i + 1]])

        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(sector)
        time_text.set_text(time_template % (t[i + ratio - 1]))

        phase1.set_data(phi1[i], om1[i])
        phase2.set_data(phi2[i], om2[i])
        phase3.set_data(phi3[i], om3[i])

        return line1, line2, line3, line4, line5, phase1, time_text, phase2, phase3, sector

    anim = FuncAnimation(fig, animate, n // ratio,
                         interval=5, blit=True, init_func=init, repeat_delay=3000)

    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save:
        anim.save('triple_pendulum_2.html', fps=30)
    else:
        plt.show()


params1 = array([l1, l2, l3, m1, m2, m3])
params2 = array([np.degrees(phi1_0), np.degrees(phi2_0), np.degrees(phi3_0), om1_0, om2_0, om3_0])
dcm1, dcm2 = 3, 4
fmt1, fmt2 = countDigits(amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
for val in params2:
    fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

parameters = [
    r"Axe x : $x_3$",
    r"Axe y : $y_3$",
    r"Axe c : $v_3$",
    "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
    r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(l1, width=fmt1, dcm=dcm1),
    r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(l2, width=fmt1, dcm=dcm1),
    r"$l_3 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(l3, width=fmt1, dcm=dcm1),
    r"$m_1$ = {:>{width}.{dcm}f} $\rm kg$".format(m1, width=fmt1, dcm=dcm1),
    r"$m_2$ = {:>{width}.{dcm}f} $\rm kg$".format(m2, width=fmt1, dcm=dcm1),
    r"$m_3$ = {:>{width}.{dcm}f} $\rm kg$".format(m3, width=fmt1, dcm=dcm1),
    "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(g), "",
    r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(phi1_0), width=fmt2, dcm=dcm2),
    r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(phi2_0), width=fmt2, dcm=dcm2),
    r"$\varphi_3$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(phi3_0), width=fmt2, dcm=dcm2),
    r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om1_0, width=fmt2, dcm=dcm2),
    r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om2_0, width=fmt2, dcm=dcm2),
    r"$\omega_3$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om3_0, width=fmt2, dcm=dcm2)
]

# see_path_1(1, array([x2, y2]), v2, color='jet', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, array([x3, y3]), v3, color='viridis', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)

see_animation()
