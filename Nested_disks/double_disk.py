import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import sin, cos, hypot, radians, linspace, array, sqrt, degrees, amax, amin, zeros
from scipy.integrate import odeint
from timeit import default_timer as timer
from Utils.Fixed_Path import see_path_1

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14

########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81  # acceleration due to gravity, in m/s²
R = 0.50  # Rayon anneau [m]
M = 1.00  # Masse anneau [kg]
a = 0.20  # Rayon disque [m]
m = 1.00  # Masse disque [kg]

c = 2 * M + 3 / 2 * m
# m/c = 2 / (4M/m + 3)
s = R - a
k, k1 = -0.00, 0.00

th0 = -80.00  # angle initial disque
om0 = 0 * sqrt(g / R)  # vitesse angulaire initiale disque
x0 = 0.00  # position initiale anneau
v0 = 0.50  # vitesse initiale anneau

phi1_0 = 0
phi2_0 = 0

Tend = 10
n = int(300 * Tend)
fps = 30
ratio = n // (int(Tend * fps))

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

U0 = array([radians(th0), om0, x0, v0])
t = linspace(0, Tend, n)


def f(u, _):
    # th, om, x, v = u

    # cxp = k + k1 * (1 + sin(u[0]))
    # ctc = m * s * cos(u[0])
    # ctp = -k1 * s * (1 + sin(u[0]))
    # ctpp = m * s * (0.5 + sin(u[0]))
    #f1 = ((m * s / c) * (0.5 + sin(u[0)) * (u[3] * cxp + u[1 * u[1 * ctc + u[1 * ctp) + u[3] * (-k1 * (s + 1)) + u[1 * (
    #        k1 * s * (1 + s)) + s * m * g * cos(u[0)) / (m * s * s - m * s / c * (0.5 + sin(u[0)) * ctpp)
    #f2 = 1 / c * (u[3] * cxp + u[1 * u[1 * ctc + u[1 * ctp + f1 * ctpp)

    f1 = (g * cos(u[0]) / (R - a) + m * cos(u[0]) * (0.5 + sin(u[0])) * u[1] * u[1] / c) / (
            1.5 - m * (0.5 + sin(u[0])) ** 2 / c)
    f2 = m * (R - a) * cos(u[0]) * u[1] * u[1] / c + m * (R - a) * (0.5 + sin(u[0])) * f1 / c

    return array([u[1], f1, u[3], f2])


tic = timer()
sol = odeint(f, U0, t)
print("      Elapsed time : %f seconds" % (timer() - tic))

th = array(sol[:, 0])
om = array(sol[:, 1])
x = array(sol[:, 2])
v = array(sol[:, 3])

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

phi1 = phi1_0 + (x - x0) / R
phi2 = phi2_0 + (R * (phi1 - phi1_0) + (a - R) * (th - th0)) / a

xc1 = x
yc1 = zeros(n)  # position du centre de l'anneau
xc2, yc2 = x + (R - a) * cos(th), -(R - a) * sin(th)  # position du centre du disque
x1, y1 = xc1 + R * cos(phi1), yc1 - R * sin(phi1)  # position d'un point sur l'anneau
x2, y2 = xc2 + a * cos(phi2), yc2 - a * sin(phi2)  # position d'un point sur le disque

vx2 = v + (a - R) * sin(th) * om - a * sin(phi2) * ((v + (a - R) * om) / a)
vy2 = (a - R) * cos(th) * om - a * cos(phi2) * ((v + (a - R) * om) / a)
v2 = hypot(vx2, vy2)

xmin, xmax = amin(xc1) - 1.5 * R, amax(xc1) + 1.5 * R
ymin, ymax = -1.5 * R, 3 * R


########################################################################################################

#####     ================      Animation du Système      ================      #####

def see_animation(save=""):
    global ratio, n
    ratio = 1 if save == "snapshot" else ratio
    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

    #####     ================      Création de la figure      ================      #####

    fig = plt.figure(figsize=(13, 8))
    ax = fig.add_subplot(211, autoscale_on=False, xlim=(xmin, xmax), ylim=(ymin, ymax))
    ax.set_aspect("equal", "datalim")
    ax2 = fig.add_subplot(223)
    ax2.grid(ls=':')
    ax.grid(ls=':')
    ax3 = fig.add_subplot(224)
    ax3.grid(ls=':')
    ax2.set_xlabel(r'$\theta \; \rm [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$v \; \rm [m/s]$', fontsize=ftSz2)
    ax3.set_xlabel(r'$t \; \rm [s]$', fontsize=ftSz2)

    line1, = ax.plot([], [], 'o-', lw=2, color='orange')
    line2, = ax.plot([], [], 'o-', lw=2, color='grey')
    line3, = ax.plot([], [], 'o-', lw=2, color='black')
    phase21, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase31, = ax3.plot([], [], marker='o', ms=8, color='C0')
    ax.hlines(-R, xmin - 5 * R, xmax + 5 * R, color='black', linewidth=1)

    circ1 = patches.Circle((xc1[0], yc1[0]), radius=R, facecolor='None', edgecolor='black', lw=2)
    circ2 = patches.Circle((xc2[0], yc2[0]), radius=a, facecolor='lightgrey', edgecolor='None')

    fig.tight_layout()
    xmin_, xmax_ = ax.get_xlim()
    L_X = xmax_ - xmin_
    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.45, 0.9, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((xmax_ - L_X * 0.04, ymax - 0.04 * L_X),
                           L_X * 0.025, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.02, 0.92, r'$R  = {:.2f} $'.format(R), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.84, r'$r  = {:.2f} $'.format(a), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.10, 0.92, r'$M  = {:.2f} $'.format(M), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.10, 0.84, r'$m  = {:.2f} $'.format(m), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.70, 0.92, r'$\theta_0  = {:.2f} $'.format(th0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.70, 0.84, r'$\omega_0  = {:.2f} $'.format(degrees(om0)), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.92, r'$x_0  = {:.2f} $'.format(x0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.84, r'$v_0  = {:.2f} $'.format(v0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax2.plot(th, v, color='C1')
    # ax2.plot(om,v,color='C1')
    ax3.plot(t, v, color='C2')

    # ax2.plot(t,th,label='theta %time')
    # ax2.plot(t,om,label='omega % time')
    # ax2.plot(t,x,label='position % time')
    # ax2.plot(t,v,label='speed % time')
    # ax2.plot(th,v,label='speed in function of theta')

    #####     ================      Animation      ================      #####

    def init():
        circ1.center = (xc1[0], yc1[0])
        circ2.center = (xc2[0], yc2[0])
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        phase21.set_data([], [])
        phase31.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line1, line2, line3, phase21, phase31, time_text, circ1, circ2, sector

    def update(i):
        i *= ratio

        thisx0, thisx1, thisx2 = [xc1[i], xc2[i]], [xc1[i], x1[i]], [xc2[i], x2[i]]
        thisy0, thisy1, thisy2 = [yc1[i], yc2[i]], [yc1[i], y1[i]], [yc2[i], y2[i]]

        circ1.center = (xc1[i], yc1[i])
        circ2.center = (xc2[i], yc2[i])
        ax.add_patch(circ1)
        ax.add_patch(circ2)

        line1.set_data(thisx0, thisy0)
        line2.set_data(thisx1, thisy1)
        line3.set_data(thisx2, thisy2)
        phase21.set_data(th[i], v[i])
        phase31.set_data(t[i], v[i])
        time_text.set_text(time_template.format(t[i + ratio - 1]))
        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(sector)

        return line1, line2, line3, phase21, phase31, time_text, circ1, circ2, sector

    n //= 2 if save == "gif" else 1
    anim = FuncAnimation(fig, update, n // ratio, interval=33, blit=True, init_func=init, repeat_delay=3000)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save == "save":
        anim.save('double_disk_3.html', fps=30)
    elif save == "gif":
        anim.save('./disks.gif', writer=PillowWriter(fps=fps))
    elif save == "snapshot":
        update(int(6.6 * n / Tend))
        fig.savefig("./disks.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


# see_path_1(2, array([th, v]), v2, var_case=2, bar=False, save=False)

see_animation(save="")
