import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, hypot, radians, array, amax
from timeit import default_timer as timer
from scipy.integrate import odeint
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14

#########################################################################################################

#####     ===========      Paramètres  de la simulation / Conditions initiales     ===========      #####

Tend = 60.35  # [s]    -  fin de la simulation

g = 9.81  # accélération de pesanteur

l = 1.00  # longueur  - [m]
m = 1.00  # masse     - [kg]
k = 37.535  # coef de raideur - [N/m]
D = 0.0  # coef frottement (couple pur)

r0 = 0.75  # m * g / k
v0 = 0.00
th0 = 10.00  # angle initial    -  [°]
om0 = 0.00  # vitesse initiale -  [°/s]

n = int(200 * Tend)
fps = 30
ratio = n // (int(Tend * fps))
dt = Tend / n

st = 0
#l, m, k, th0, om0, r0, v0 = 0.7, 1.00, 199, 18, 0, m * g / k, -7


########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

t = np.linspace(0, Tend, n)
U0 = array([radians(th0), radians(om0), r0, v0])


def f(u, _):
    th_, om_, r_, v_ = u
    d2th_ = -2 * v_ * om_ / (r_ + l) - g * sin(th_) / (r_ + l) - D / m * (r_ + l) * om_
    d2r_ = (r_ + l) * om_ * om_ + g * cos(th_) - k / m * r_ - D / m * v_
    return array([om_, d2th_, v_, d2r_])


tic = timer()
sol = odeint(f, U0, t)
print("\tElapsed time : %f seconds" % (timer() - tic))

th = array(sol[st:, 0])
om = array(sol[st:, 1])
r = array(sol[st:, 2])
v = array(sol[st:, 3])

x =   (r + l) * sin(th)
y = - (r + l) * cos(th)

vx =   v * sin(th) + (r + l) * cos(th) * om
vy = - v * cos(th) + (r + l) * sin(th) * om
speed = hypot(vx, vy)
vx_max = amax(vx)
vy_max = amax(vy)

d2r = (r + l) * om * om + g * cos(th) - k / m * r
d2th = -2 * v * om / (r + l) - g * sin(th) / (r + l)
ddx = + (d2r - (r + l) * om * om) * sin(th) + (2 * v * om + (r + l) * d2th) * cos(th)
ddy = - (d2r - (r + l) * om * om) * cos(th) + (2 * v * om + (r + l) * d2th) * sin(th)
acc = hypot(ddx, ddy)

########################################################################################################

#####     ================      Animation du Système      ================      #####


def see_animation(save="", phaseSpace=0):
    global ratio
    if save == "snapshot":
        ratio = 1
        plt.rcParams['text.usetex'] = True

    #####     ================      Création de la figure      ================      #####

    max_x = 1.1 * amax(r + l)

    fig = plt.figure(figsize=(13., 6.))

    ax = fig.add_subplot(121, xlim=(-max_x, max_x), ylim=(-max_x, max_x), aspect='equal')
    ax.grid(ls=':')
    ax2 = fig.add_subplot(122)
    ax2.grid(ls=':')

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], 'o-', lw=4, color='grey', alpha=0.3)
    line3, = ax.plot([], [], '-', lw=1, color='grey', alpha=0.8)

    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C1')

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.79, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((0.875*l, 0.85*l), 0.04*l, theta1=90, theta2=90, color='lightgrey', transform=ax.transAxes)

    ax.text(0.02, 0.86, r'$k  \,\, = {:.2f} \,N/m$'.format(k), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.82, r'$\ell \:\; = {:.2f} \,m$'.format(l), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.78, r'$m  = {:.2f} \,kg$'.format(m), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax.text(0.02, 0.96, r'$r         = {:.2f} $'.format(r0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$v         = {:.2f} $'.format(v0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.17, 0.96, r'$\vartheta = {:.2f} $'.format(th0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.17, 0.92, r'$\omega    = {:.2f} $'.format(om0), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    # 87 90 93 96
    if phaseSpace == 0:
        ax2.plot(th, om, color='C0', label=r'$\vartheta \; :\; \omega$')
        ax2.plot(r - m * g / k, v, color='C1', label=r'$r \;\, : \; v$')
        ax2.legend(fontsize=ftSz3)
    elif phaseSpace == 1:
        ax2.plot(x, y, color='C0', label='Trajectoire')
        ax2.set_aspect('equal')
    else:
        ax2.plot(th, (r - m * g / k), color='C1', label=r'$\vartheta / r$')
        ax2.plot(th, v, color='C2', label=r'$\vartheta / v$')
        ax2.plot(om, (r - m * g / k), color='C3', label=r'$\omega / r$')
        ax2.plot(om, v, color='C4', label=r'$\omega / v$')
        ax2.plot(r, v, color='C5', label=r'$r / v$')
        ax2.legend(fontsize=ftSz3)

    #####     ================      Animation      ================      #####

    def init():

        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        phase1.set_data([], [])
        phase2.set_data([], [])

        time_text.set_text('')
        sector.set_theta1(90)

        liste = [line1, line2, line3, phase1, phase2, time_text, sector]

        return tuple(liste)

    def update(i):

        i *= ratio
        start = max(0, i - 10000)

        line1.set_data([0, x[i]], [0, y[i]])
        line2.set_data([0, (l + m * g / k) * sin(th[i])], [0, -(l + m * g / k) * cos(th[i])])
        line3.set_data([x[start:i + 1]], [y[start:i + 1]])

        if phaseSpace == 0:
            phase1.set_data(th[i], om[i])
            phase2.set_data(r[i] - m * g / k, v[i])
        elif phaseSpace == 1:
            phase1.set_data(x[i], y[i])
        else:
            phase1.set_data(th[i], r[i] - m * g / k)
            phase2.set_data(om[i], v[i])

        time_text.set_text(time_template.format(t[i + ratio - 1]))
        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(sector)

        liste = [line1, line2, line3, phase1, phase2, time_text, sector]

        return tuple(liste)

    anim = FuncAnimation(fig, update, n // ratio, interval=10, blit=True, init_func=init, repeat_delay=3000)
    fig.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save == "save":
        anim.save('Pendule_Elastique_1', fps=30)
    if save == "snapshot":
        update(int(8. * n / Tend))
        fig.savefig("./pendulum_elastic.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


params1 = array([l, m, k])
params2 = array([th0, om0, r0, v0])

dcm1, dcm2 = 3, 3
fmt1, fmt2 = countDigits(amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
for val in params2:
    fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

parameters = [
    r"Axe x : $x$",
    r"Axe y : $y$",
    r"Axe c : $speed$",
    "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
    r"$l\;\;$ = {:>{width}.{dcm}f} $\rm m$".format(l, width=fmt1, dcm=dcm1),
    r"$m$ = {:>{width}.{dcm}f} $\rm kg$".format(m, width=fmt1, dcm=dcm1),
    r"$k\:\:$ = {:>{width}.{dcm}f} $\rm N/m$".format(k, width=fmt1, dcm=dcm1),
    "", r"$g\,$ = {:.2f} $\rm m/s^2$".format(g), "",
    r"$\vartheta $ = {:>{width}.{dcm}f} $\rm deg$".format(th0, width=fmt2, dcm=dcm2),
    r"$\omega$ = {:>{width}.{dcm}f} $\rm deg/s$".format(om0, width=fmt2, dcm=dcm2),
    r"$r\,$ = {:>{width}.{dcm}f} $\rm m$".format(r0, width=fmt2, dcm=dcm2),
    r"$v\,$ = {:>{width}.{dcm}f} $\rm m/s$".format(v0, width=fmt2, dcm=dcm2)
]
parameters[0] = r"Axe x : $x$"
parameters[1] = r"Axe y : $y$"
parameters[2] = r"Axe c : $acc$"

# see_path_1(1., array([x, y]), -acc, color='inferno', var_case=1, name='0', shift=(0., 0.), save="save", displayedInfo=parameters)
"""see_path_1(1, array([th, om]), speed, color='inferno', var_case=2, name='1', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, array([th, r]), speed, color='viridis', var_case=2, name='2', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, array([th, v]), speed, color='viridis', var_case=2, name='3', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, array([om, r]), speed, color='viridis', var_case=2, name='4', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, array([om, v]), speed, color='viridis', var_case=2, name='5', shift=(0., 0.), save="no", displayedInfo=parameters)
see_path_1(1, array([r, v]), speed, color='viridis', var_case=2, name='6', shift=(0., 0.), save="no", displayedInfo=parameters)
#"""

#see_path_1(1,array([x,y]), hypot(vx,vy), 'inferno', shift=0, var_case=1, save=False)
#see_path_1(1,array([x,y]), hypot(ddx,ddy), 'inferno_r', shift=0.1, var_case=1, save=False)

#see_path_1(1, array([th, om]), hypot(r, v), 'inferno', var_case=2, save=False)
# see_path_2(1,array([phi2,sin(om2)]), array([phi1,sin(om1)]), hypot(v2x,v2y), hypot(vx,vy), 'inferno','viridis', var_case=2, save=False)

see_animation(save="", phaseSpace=0)
