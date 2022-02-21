import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from numpy import sin, cos, hypot, radians, linspace, array, degrees, amax, amin, ones, mean
from scipy.integrate import odeint
from time import perf_counter
from Utils.Fixed_Path import countDigits, see_path_1, see_path

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'


########################################################################################################

#####     ================      Paramètres  de la simulation      ================      ####

g = 9.81                  # accélération de pesanteur
M = 3.51                   # grosse masse
m = 1.0                   # petite masse

r0  = 1                  # position initiale  -  [m]
dr0 = 0.0                  # vitesse initiale   -  [m/s]
th0 = 150                  # angle initial      -  [°]
om0 = 0.0                  # vitesse angulaire  -  [rad/s]

#Tend = 25.7 + (13. + 3.11)*5
Tend = 500.
fps = 30
n = int(500 * Tend)
ratio = n // (int(Tend * fps))

lw, savefig = 1, "no"
th0 = radians(th0)

########################################################################################################

#####     ================      Paramètres  intéressants      ================      ####

#r0, th0, dr0, om0, M = 1, 1.4, 0, 0, 1.527
#r0, th0, dr0, om0, M = 0.25, 0.35, 0, 2.83, 3
#r0, th0, dr0, om0, M = 1, 0, pi, 1, 4.35
#r0, th0, dr0, om0, M = 0.5, 0, pi, 1, 1.95

#r0, th0, dr0, om0, M = 1., 3 * pi / 4, 0, 0, 5.6
#r0, th0, dr0, om0, M = 1., 5 * pi / 6, 0, 0, 7.244

#r0, th0, dr0, om0, M = 1., 5 * pi / 6, 0, 0, 6.01401
#r0, th0, dr0, om0, M = 1., 5 * pi / 6, 0, 0, 16
#r0, th0, dr0, om0, M = 1., 3 * pi / 4, 0, 0, 16
#r0, th0, dr0, om0, M = 1., 3 * pi / 4, 0, 0, 20.75

M_list = array(
    [1.133, 1.172, 1.278, 1.337, 1.555, 1.655, 1.67, 1.904, 2.165, 2.394,
     2.8121, 2.945, 3.125, 3.52, 3.867, 4.1775, 4.745, 5.475, 5.68, 6.014,
     6.806, 7.244, 7.49, 7.7, 8.182370012, 8.182370165, 16, 19, 21, 24])

#r0, th0, dr0, om0, M = 1., pi / 2., 0, 0, M_list[0]
#r0, th0, dr0, om0, M = 1., 5 * pi / 6., 0, 0, 4.80  #M_list[16]

#M = 5.569 # 6.1187 # 6.114 # 4.126 # 4.233 # 1.565 # 2.9452 # 2.021  # good or not ?


########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####
t = linspace(0, Tend, n)
U0 = array([r0, dr0, th0, om0])


def f(u, _):
    r_, dr_, th_, om_ = u
    f1 = (m * r_ * om_ * om_ + g * (-M + m * cos(th_))) / (M + m)
    f2 = -1 / r_ * (2 * dr_ * om_ + g * sin(th_))
    return array([dr_, f1, om_, f2])


tic = perf_counter()
sol = odeint(f, U0, t)
print("\tElapsed time : %.3f seconds" % (perf_counter() - tic))
print("\tMasse        : %.3f " % M)

r, dr = array(sol[:, 0]), array(sol[:, 1])
th, om = array(sol[:, 2]), array(sol[:, 3])

########################################################################################################

#####     ================      Ecriture des Positions / Vitesses      ================      #####

x2, y2 = r*sin(th), -r*cos(th)

L_X, L_Y = amax(x2) - amin(x2), amax(y2) - amin(y2)
x_m, y_m = amin(x2) - 0.2*L_X, amin(y2) - 0.2*L_Y
x_M, y_M = amax(x2) + 0.2*L_X, amax(y2) + 0.2*L_Y

d  = mean(r)
L = max(r) + d*1.2

x1, y1 = -d * ones(n), r + d - L

vx =  dr*sin(th) + r*om*cos(th)
vy = -dr*cos(th) + r*om*sin(th)
v = hypot(vx, vy)

ddr = (m * r * om * om + g * (-M + m * cos(th))) / (M + m)
dom = -1 / r * (2 * dr * om + g * sin(th))
acx = ddr * sin(th) + dr * cos(th) * om - r * om * om * sin(th) + (r * dom + dr * om) * cos(th)
acy = -ddr * cos(th) + dr * sin(th) * om + r * om * om * cos(th) + (r * dom + dr * om) * sin(th)
a = hypot(acx, acy)

x_m = min(-d*1.1, x_m)
if abs(amin(y2)) < L_X:
    y_m = x_m
    y_M = x_M

########################################################################################################

#####     ================      Animation du Système      ================      #####


def see_animation(save=False):

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(121, autoscale_on=False, xlim=(x_m, x_M), ylim=(y_m, y_M), aspect='equal')
    ax2 = fig.add_subplot(122)
    ax.grid(ls=':')
    ax2.grid(ls=':')
    ax2.set_xlabel(r'$\theta \: \rm [rad]$')
    ax2.set_ylabel(r'$\omega \: \rm [m/s]$')

    line,  = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], '-', lw=1, color='grey')
    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C1')

    time_template = r'$t = %.1fs$'
    time_text = ax.text(0.42, 0.94, '', fontsize=15, transform=ax.transAxes)
    sector = patches.Wedge((x_M - L_X/10, x_m + L_X/10), L_X/20, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.04, 0.94, r'$\mu  = {:.2f}$'.format(M/m), fontsize=15, wrap=True, transform=ax.transAxes)

    ax.text(0.72, 0.96, r'$r  = {:.2f} $'.format(r0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.72, 0.92, r'$dr  = {:.2f} $'.format(dr0), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.87, 0.96, r'$\theta  = {:.2f} $'.format(degrees(th0)), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.87, 0.92, r'$\omega  = {:.2f} $'.format(degrees(om0)), fontsize=12, wrap=True, transform=ax.transAxes)

    ax2.plot(th, om, color='C0')
    ax2.plot(r, dr, color='C1')

    #####     ================      Animation      ================      #####

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        phase1.set_data([], [])
        phase2.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line, line2, phase1, phase2, time_text, sector

    def animate(i):
        i *= ratio
        start = max(0, i - 250000)
        thisx = [x2[i], 0, x1[i], x1[i]]
        thisy = [y2[i], 0, 0, y1[i]]

        line.set_data(thisx, thisy)
        line2.set_data([x2[start:i + 1]], [y2[start:i + 1]])
        phase1.set_data(th[i], om[i])
        phase2.set_data(r[i], dr[i])

        time_text.set_text(time_template % (t[i+ratio-1]))
        sector.set_theta1(90-360*t[i+ratio-1]/Tend)
        ax.add_patch(sector)

        return line, line2, phase1, phase2, time_text, sector

    anim = FuncAnimation(fig, animate, n // ratio,
                         interval=5, blit=True,
                         init_func=init, repeat_delay=5000)
    plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)

    if save:
        anim.save('Atwood_System_1.html', fps=30)
    else:
        plt.show()


params2 = array([r0, dr0, np.degrees(th0), om0])
dcm1, dcm2 = 5, 3
fmt2 = 1 + 1 + dcm2
for val in params2:
    fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

parameters = [
    r"Axe x : $x_2$",
    r"Axe y : $y_2$",
    r"Axe c : $v_2$", "",
    r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
    r"$\mu$ = {:.{dcm}f}".format(M/m, dcm=dcm1),
    "", r"$g$ = {:.2f} $\rm m/s^2$".format(g), "",
    r"$r \;\,\,$ = {:>{width}.{dcm}f} $\rm m$".format(r0, width=fmt2, dcm=dcm2),
    r"$dr$ = {:>{width}.{dcm}f} $\rm m/s$".format(dr0, width=fmt2, dcm=dcm2),
    r"$\vartheta \;\,$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(th0), width=fmt2, dcm=dcm2),
    r"$\omega \;\,$ = {:>{width}.{dcm}f} $\rm rad/s$".format(om0, width=fmt2, dcm=dcm2)
]

parameters[0] = r"Axe x : $\omega \, * \, r$"
parameters[1] = r"Axe y : $r$"
parameters[2] = r"Axe c : $v^2$"

# see_path_1(lw, array([x2, y2]), v, color='jet', var_case=1, shift=(-0., 0.), save='no', displayedInfo=parameters)
#see_path_1(1, array([th, om]), r, 'inferno', name='th - om', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
#see_path_1(1., array([th, r]), v, 'Blues', name='th - r', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
#see_path_1(1, array([th, dr]), r, 'inferno', name='th - dr', shift=(0.0, 0.), var_case=2, save='no', displayedInfo=parameters)
#see_path_1(1, array([om, r]), v, 'inferno', name='om - r', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
#see_path_1(1., array([om, dr]), r, 'inferno', name='om - dr', shift=(0.1, 0.), var_case=4, save='no', displayedInfo=parameters)
#see_path_1(1., array([r, dr]), r, 'inferno', name='r dr', shift=(0.1, -0.), var_case=2, save='no', displayedInfo=parameters)


"""
see_path_1(1., array([dr*r, th]), v, 'Blues', name='1', shift=(-0., 0.), var_case=2, save='no', displayedInfo=parameters)
see_path_1(1., array([dr, om*th]), -r, 'Blues', name='2', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
see_path_1(1., array([dr, om*dr]), -r, 'Blues', name='3', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
see_path_1(1., array([dr, om*r]), -r, 'Blues', name='4', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
"""

"""parameters[0] = r"Axe x : $\varphi_1 $  -  $\varphi_2$"
parameters[1] = r"Axe y : $\omega_1$  -  $\omega_2$"
parameters[2] = r"Axe c : $\varphi_2$  -  $\varphi_1$"
see_path(1, [array([x2, y2]), array([2*x2, 2*y2])],
         [v, v], ["Blues", "viridis"],
         var_case=2, save="no", displayedInfo=parameters)"""

#see_path_1(1, array([om, dr]), r, 'inferno', name='om - dr', shift=(0., 0.), var_case=4, save='save')

see_animation()
