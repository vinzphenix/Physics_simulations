import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, linspace, array, sqrt, degrees, amax, amin, radians
from scipy.integrate import odeint
from timeit import default_timer as timer

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

# ATTENTION : convention angles ABSOLUS

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81              # [m/s²]  -  accélaration de pesanteur
l1, l2 = 0.1, 2.0    # [m]     -  longueur des pendules

Tend, fps = 20, 30              # [s]    -  fin de la simulation
n = int(750 * Tend)
ratio = n//(int(Tend*fps))

liste = [1/2, 2/3, 3/5, 9/17, 17/37]

phi2_0, om2_0 = 0.0, -1.0
phi1_0, om1 = 0.0, 5.0 * sqrt(g/l2)

#om1 = 5.0 * sqrt(g/l2)

#phi1_0, phi2_0, om1, om2_0, l1, l2, m, D = (0   , 0    , sqrt(g/0.4) , 0      , 0.1     , 0.4    , 1   ,0)
#phi1_0, phi2_0, om1, om2_0, l1, l2, m, D = (0   , 0    , sqrt(g/0.1) , 0      , 0.1     , 0.1    , 1   , 0)
#om1 *= liste[1]

w = sqrt(g / l1)
l = l2 / l1
phi1_0, phi2_0 = radians([phi1_0, phi2_0])
om1, om2_0 = om1 / w, om2_0 / w

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

U0 = array([phi2_0, om2_0])
t = linspace(0, Tend*w, n)


def f(u, time):
    th1 = phi1_0 + om1 * time
    th2, dth2 = u

    f1 = - om1 * om1 * sin(th2 - th1) / l - sin(th2) / l

    return array([dth2, f1])


tic = timer()
sol = odeint(f, U0, t)
print("\tElapsed time : %f seconds" % (timer() - tic))

phi1, om1 = phi1_0 + om1 * t, linspace(om1,  om1, n)
phi2, om2 = sol.T

t /= w
om1, om2 = om1 * w, om2 * w
U0[1] *= w

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

x1, y1 = l1 * sin(phi1), -l1 * cos(phi1)
x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

vx2 = l1 * cos(phi1) * om1 + l2 * cos(phi2) * om2
vy2 = l1 * sin(phi1) * om1 + l2 * sin(phi2) * om2
v2 = sqrt(vx2 * vx2 + vy2 * vy2)

x_max = 1.05 * amax(abs(x2))
y_min, y_max = amin(y2), amax(y2)
y_min, y_max = y_min - 0.05 * abs(y_min), y_max + 0.05 * abs(y_max)

vx_max, vy_max = amax(vx2), amax(vy2)

T = g * cos(phi1) + l1 * om1 * om1 * cos(phi2 - phi1) + l2 * om2 ** 2
L, T_max = l1 + l2, amax(T)
phim, dphim = amax(abs(phi2)), amax(om2)

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
    time_text = ax.text(0.42, 0.94, '1', fontsize=15, wrap=True, transform=ax.transAxes)
    sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.02, 0.96, r'$l_1 = {:.2f} \: \rm m$'.format(l1), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$l_2 = {:.2f} \: \rm m$'.format(l2), fontsize=12, wrap=True, transform=ax.transAxes)

    ax.text(0.64, 0.96, r'$\varphi_1  = {:.2f}$'.format(degrees(phi1_0)), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.64, 0.92, r'$\varphi_2  = {:.2f}$'.format(degrees(phi2_0)), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.96, r'$\omega_1  = {:.2f}$'.format(degrees(om1[0])), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.92, r'$\omega_2  = {:.2f}$'.format(degrees(om2_0)), fontsize=12, wrap=True, transform=ax.transAxes)

    ax2.plot(phi2, om2, color='C2')

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], 'o-', lw=2, color='C2')
    line3, = ax.plot([], [], '-', lw=1, color='grey')
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C0')
    rect = plt.Rectangle((L * 1.0, 0), L * 0.05, T[0] / T_max * L)

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        time_text.set_text('')
        rect.set_bounds(L*1.05, 0, L*0.05, T[0]/T_max*L)
        phase2.set_data([], [])
        sector.set_theta1(90)
        return line1, line2, line3, time_text, rect, phase2, sector

    def animate(i):
        i *= ratio
        start = max((i-50000, 0))

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

        phase2.set_data(phi2[i], om2[i])

        return line1, line2, line3, time_text, rect, phase2, sector

    ani = FuncAnimation(fig, animate, n // ratio,
                        interval=5, blit=True,
                        init_func=init, repeat_delay=5000)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save:
        ani.save('Pendule_entraine_4.html', fps=30)
    else:
        plt.show()


def __main__():
    """ Execute this file here """


if __name__ == "__main__":
    see_animation()

    #from Utils.Fixed_Path import see_path_1
    #see_path_1(1, array([x2, y2]), v2)
