import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation
from matplotlib.patches import Arrow
from numpy import sin, cos, radians, sqrt, pi, degrees, amax
from timeit import default_timer as timer
from scipy.integrate import solve_ivp
from Utils.Fixed_Path import see_path_1, see_path
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

#########################################################################################################

#####     ===========      Paramètres  de la simulation / Conditions initiales     ===========      #####

Tend = 20  # [s]    -  fin de la simulation

g = 9.81  # accélération de pesanteur

# Pendule L
L = 2.00  # longueur  - [m]
h = 0.06  # épaisseur - [m]
M = 2.0  # masse     - [kg]
th0 = 0.0  # angle initial    -  [°]
om0 = 0.0  # vitesse initiale -  [°/s]
# D = 0.00   # coef frottement (couple pur)

# Pendule 1
l1 = sqrt(2)  # longueur  - [m]
h1 = 0.05  # épaisseur - [m]
m1 = 1.  # masse     - [kg]
phi10 = 30.  # angle initial    -  [°]
om10 = 0.00  # vitesse initiale -  [°/s]
# D1 = 0.00

# Pendule 2
l2 = 1.00  # longueur  - [m]
h2 = 0.05  # épaisseur - [m]
m2 = 0.50  # masse     - [kg]
phi20 = -30.  # angle initial    -  [°]
om20 = 0.00  # vitesse initiale -  [°/s]
# D2 = 0.00

fps = 20
ratio = 10
n = int(ratio * fps * Tend)
method = 'RK45'
# ratio = n // (int(Tend * fps))
# dt = Tend / n

########################################################################################################

#####     ================      Résolution de l'équation différentielle      ================      #####

H = L - h
H1 = (l1 - h1) / 2.
H2 = (l2 - h2) / 2.

d = sqrt(2) * L / 2 * (L - h) / (2 * L - h)

IG0 = M / 6. * (5 * L ** 2 - 5 * L * h + h ** 2) * (L ** 2 - L * h + h ** 2) / (2 * L - h) ** 2
IG1 = m1 / 12. * (l1 ** 2 + h1 ** 2)
IG2 = m2 / 12. * (l2 ** 2 + h2 ** 2)

dd = L / 2. * (L - h) / (2. * L - h)
c0 = 2 * M * dd * dd + H * H * (m1 + m2) + IG0
c1 = m1 * H1 * H1 + IG1
c2 = m2 * H2 * H2 + IG2

A11 = c1 * c2
A12 = -m1 / 2. * H * H1 * c2
A13 = -m2 / 2. * H * H2 * c1
A22_fix = c0 * c2
A22_var = -(m2 / 2. * H * H2) ** 2
A23 = m1 * m2 / 4. * H * H * H1 * H2
A33_fix = c0 * c1
A33_var = -(m1 / 2. * H * H1) ** 2

B11 = m1 / 2. * H * H1
B12 = -m2 / 2. * H * H2
B13 = -sqrt(2) * M * g * dd
B21 = -m1 / 2. * H * H1
B22 = -m1 * H1 * g
B31 = m2 / 2. * H * H2
B32 = -m2 * H2 * g

D1 = c0 * c1 * c2
D2 = -c1 * (m2 / 2. * H * H2) ** 2
D3 = -c2 * (m1 / 2. * H * H1) ** 2

t = np.linspace(0, Tend, n)
# phi10 = 1/9*pi ; phi20 = 1/9*pi

U0 = radians(np.array([th0, phi10, phi20, om0, om10, om20]))


def f(_, u):
    th_, phi1_, phi2_, w, w1, w2 = u
    COS1 = cos(pi / 4. + phi1_ - th_)
    COS2 = cos(pi / 4. + phi2_ - th_)
    SIN1 = sin(pi / 4. + phi1_ - th_)
    SIN2 = sin(pi / 4. + phi2_ - th_)

    A = np.array([
        [A11, A12 * COS1, A13 * SIN2],
        [A12 * COS1, A22_fix + A22_var * SIN2 * SIN2, A23 * COS1 * SIN2],
        [A13 * SIN2, A23 * COS1 * SIN2, A33_fix + A33_var * COS1 * COS1]
    ])

    B = np.array([B11 * w * w1 * SIN1 + B12 * w * w2 * COS2 + B13 * sin(th_),
                  B21 * w * w1 * SIN1 + B22 * sin(phi1_),
                  B31 * w * w2 * COS2 + B32 * sin(phi2_)])

    delta = D1 + D2 * SIN2 * SIN2 + D3 * COS1 * COS1

    # d2th = (m1 * H * H1 * sin(pi / 4 + phi1_ - th_) * w1 * w1 - m2 * H * H2 * cos(pi / 4 + phi2_ - th_) * w2 * w2
    #         - g * (M * d * sin(th_) + m1 * H * sin(th_ - pi / 4) + m2 * H * cos(th_ - pi / 4)) - D * w
    #         + (m1 * H * H1) ** 2 / (2 * c1) * cos(2 * th_ - 2 * phi1_) * w * w
    #         - (m2 * H * H2) ** 2 / (2 * c2) * cos(2 * th_ - 2 * phi2_) * w * w
    #         + m1 * H * H1 / c1 * (m1 * g * sin(phi1_) + D1 * w1) * cos(pi / 4 + phi1_ - th_)
    #         + m2 * H * H2 / c2 * (m2 * g * sin(phi2_) + D2 * w2) * sin(pi / 4 + phi1_ - th_)) / (
    #            (M * d * d + I + (m1 + m2) * H * H - (m1 * H * H1 * cos(pi / 4 + phi1_ - th_)) ** 2 / c1 - (
    #                m2 * H * H2 * sin(pi / 4 + phi2_ - th_)) ** 2 / c2))
    # d2phi1 = -m1 * H1 * H / c1 * (d2th * cos(pi / 4 + phi1_ - th_) + w * w * sin(pi / 4 + phi1_ - th_)) \
    #          - m1 * g * sin(phi1_) / c1 - D1 * w1 / c1
    # d2phi2 = -m2 * H2 * H / c2 * (d2th * sin(pi / 4 + phi2_ - th_) - w * w * cos(pi / 4 + phi2_ - th_)) \
    #          - m2 * g * sin(phi2_) / c2 - D2 * w2 / c2

    # d2th_ = (m1 * H * H1 * sin(pi / 4 + phi1_ - th_) * w1 * w1 - m2 * H * H2 * cos(pi / 4 + phi2_ - th_) * w2 * w2
    #         - g * (M * d * sin(th_) + m1 * H * sin(th_ - pi / 4) + m2 * H * cos(th_ - pi / 4))
    #         + (m1 * H * H1) ** 2 / (2 * c1) * cos(2 * th_ - 2 * phi1_) * w * w
    #         - (m2 * H * H2) ** 2 / (2 * c2) * cos(2 * th_ - 2 * phi2_) * w * w
    #         + m1 * H * H1 / c1 * (m1 * g * sin(phi1_)) * cos(pi / 4 + phi1_ - th_)
    #         + m2 * H * H2 / c2 * (m2 * g * sin(phi2_)) * sin(pi / 4 + phi1_ - th_)) / \
    #        ((M * d * d + I + (m1 + m2) * H * H
    #          - (m1 * H * H1 * cos(pi / 4 + phi1_ - th_)) ** 2 / c1
    #          - (m2 * H * H2 * sin(pi / 4 + phi2_ - th_)) ** 2 / c2)
    #        )
    # d2phi1 = -m1 * H1 * H / c1 * (d2th_ * cos(pi / 4 + phi1_ - th_) + w * w * sin(pi / 4 + phi1_ - th_)) \
    #          - m1 * g * sin(phi1_) / c1
    # d2phi2 = -m2 * H2 * H / c2 * (d2th_ * sin(pi / 4 + phi2_ - th_) - w * w * cos(pi / 4 + phi2_ - th_))\
    #          - m2 * g * sin(phi2_) / c2

    return np.r_[w, w1, w2, np.dot(A, B) / delta]
    # return np.array([w, d2th, w1, d2phi1, w2, d2phi2])
    # ddphi = - d * M * g / I * sin(th) - D * L**2 / I * om


tic = timer()
# sol = odeint(f, U0, t)
sol = solve_ivp(f, [0, Tend], U0, method=method, t_eval=t, atol=1.e-9, rtol=1.e-9)
print("\tElapsed time : %f seconds" % (timer() - tic))

th, phi1, phi2, om, om1, om2 = sol.y

Ax = + L * sin(th - pi / 4) + h / sqrt(2) * cos(th)
Ay = - L * cos(th - pi / 4) + h / sqrt(2) * sin(th)
alpha = pi / 4 + th

Bx = - h / sqrt(2) * cos(th)
By = - h / sqrt(2) * sin(th)
beta = -pi / 4 + th

Q1x = + (L - h) * sin(th - pi / 4)
Q1y = - (L - h) * cos(th - pi / 4)
Q2x = + (L - h) * cos(th - pi / 4)
Q2y = + (L - h) * sin(th - pi / 4)

Cx = Q1x - h1 / sqrt(2) * (sin(phi1 + pi / 4))
Cy = Q1y + h1 / sqrt(2) * (cos(phi1 + pi / 4))
alpha1 = -pi / 2 + phi1

Dx = Q2x - h2 / sqrt(2) * (sin(phi2 + pi / 4))
Dy = Q2y + h2 / sqrt(2) * (cos(phi2 + pi / 4))
alpha2 = -pi / 2 + phi2

# Ax = + L * sqrt(2)/2 * (sin(th) - cos(th)) + h / sqrt(2) * cos(th)
# Ay = - L * sqrt(2)/2 * (cos(th) + sin(th)) + h / sqrt(2) * sin(th)
# Bx = - h / sqrt(2) * cos(th)
# By = - h / sqrt(2) * sin(th)

x = d * sin(th)
y = - d * cos(th)

vx, vy = d * cos(th) * om, d * sin(th) * om
vx_max, vy_max = amax(vx), amax(vy)

v1x = H * om * cos(th - pi / 4) + H1 * om1 * cos(phi1)
v1y = H * om * sin(th - pi / 4) + H1 * om1 * sin(phi1)

v2x = - H * om * sin(th - pi / 4) + H2 * om2 * cos(phi2)
v2y = + H * om * cos(th - pi / 4) + H2 * om2 * sin(phi2)


########################################################################################################

#####     ================      Animation du Système      ================      #####


def see_animation(save=False, arrow_velocity=False, phaseSpace=0):
    #####     ================      Création de la figure      ================      #####

    fig, axs = plt.subplots(1, 2, figsize=(11.2, 6.3))

    ax = axs[0]
    ax.axis([-1.1 * (L + max(l1, l2)), 1.1 * (L + max(l1, l2)), -1.1 * (L + max(l1, l2)), 1.1 * (L + max(l1, l2))])
    ax.set_aspect("equal")
    ax.grid(ls=':')

    ax2 = axs[1]
    ax2.grid(ls=':')
    ax2.set_xlabel(r'$\varphi$ [rad]')
    ax2.set_ylabel(r'$\omega$ [rad/s]')

    rect1 = plt.Rectangle((Ax[0], Ay[0]), L, h, 45 + degrees(th[0]), color='C0')
    rect2 = plt.Rectangle((Bx[0], By[0]), L, h, -45 + degrees(th[0]), color='C0')
    rect3 = plt.Rectangle((Cx[0], Cy[0]), l1, h1, -90 + degrees(phi1[0]), color='C1')
    rect4 = plt.Rectangle((Dx[0], Dy[0]), l2, h2, -90 + degrees(phi2[0]), color='C2')

    point, = ax.plot([], [], marker='o', ms=5, color='black')
    point1, = ax.plot([], [], marker='o', ms=5, color='black')
    point2, = ax.plot([], [], marker='o', ms=5, color='black')

    phase, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase1, = ax2.plot([], [], marker='o', ms=8, color='C1')
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C2')

    if arrow_velocity:
        arrow_s = Arrow(x[0], y[0], vx[0] * L / (2 * vx_max), vy[0] * L / (2 * vy_max), color='C3',
                        edgecolor=None, width=L / 10)

    time_template = r'$t = %.1fs $'
    time_text = ax.text(0.45, 0.94, '', fontsize=15, transform=ax.transAxes)
    sector = patches.Wedge((0.5, 0.85), 0.04, theta1=90, theta2=90, color='lightgrey', transform=ax.transAxes)

    ax.text(0.17, 0.96, r'$L  = {:.2f} \,m$'.format(L), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.17, 0.93, r'$h  = {:.2f} \,m$'.format(h), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.17, 0.90, r'$m  = {:.2f} \,kg$'.format(M), fontsize=10, wrap=True, transform=ax.transAxes)

    ax.text(0.02, 0.96, r'$l_1  = {:.2f} \,m$'.format(l1), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.93, r'$h_1  = {:.2f} \,m$'.format(h1), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.90, r'$m_1  = {:.2f} \,kg$'.format(m1), fontsize=10, wrap=True, transform=ax.transAxes)

    ax.text(0.02, 0.85, r'$l_2  = {:.2f} \,m$'.format(l2), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.82, r'$h_2  = {:.2f} \,m$'.format(h2), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.79, r'$m_2  = {:.2f} \,kg$'.format(m2), fontsize=10, wrap=True, transform=ax.transAxes)

    # ax.text(0.02, 0.84, r'$D  = {:.2f} \,kg/s$'.format(D), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.73, 0.96, r'$\vartheta  = {:.2f} $'.format(th0), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.73, 0.93, r'$\varphi_1  = {:.2f} $'.format(phi10), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.73, 0.90, r'$\varphi_2  = {:.2f} $'.format(phi20), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.88, 0.96, r'$\omega  \;\,  = {:.2f} $'.format(om0), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.88, 0.93, r'$\omega_1  = {:.2f} $'.format(om10), fontsize=10, wrap=True, transform=ax.transAxes)
    ax.text(0.88, 0.90, r'$\omega_2  = {:.2f} $'.format(om20), fontsize=10, wrap=True, transform=ax.transAxes)
    if phaseSpace == 0:
        ax2.plot(th, om, color='C0', label='pendule L')
        ax2.plot(phi1, om1, color='C1', label='masse 1')
        ax2.plot(phi2, om2, color='C2', label='masse 2')
        ax2.legend()
    else:
        ax2.plot(phi1, phi2, color='C1')
        ax2.plot(om1, om2, color='C2')

    #####     ================      Animation      ================      #####

    a_s = None

    def init():
        ax.plot(0, 0, marker='o', ms=8, color='black')
        rect1.set_height(h)
        rect2.set_height(h)
        rect3.set_height(h1)
        rect4.set_height(h2)
        point.set_data([], [])
        point1.set_data([], [])
        point2.set_data([], [])
        phase1.set_data([], [])
        phase2.set_data([], [])
        liste = [rect1, rect2, rect3, rect4, point, point1, point2, phase1, phase2, time_text, sector]

        if phaseSpace == 0:
            phase.set_data([], [])
            liste.append(phase)
        if arrow_velocity:
            nonlocal a_s
            a_s = ax.add_patch(arrow_s)
            liste.append(arrow_s)

        time_text.set_text('')
        sector.set_theta1(90)
        return tuple(liste)

    def animate(i):
        i *= ratio
        rect1_ = plt.Rectangle((Ax[i], Ay[i]), L, h, 45 + degrees(th[i]), color='C0')
        rect2_ = plt.Rectangle((Bx[i], By[i]), L, h, -45 + degrees(th[i]), color='C0')
        rect3_ = plt.Rectangle((Cx[i], Cy[i]), l1, h1, -90 + degrees(phi1[i]), color='C1')
        rect4_ = plt.Rectangle((Dx[i], Dy[i]), l2, h2, -90 + degrees(phi2[i]), color='C2')

        for rect in [rect1_, rect2_, rect3_, rect4_]:
            ax.add_patch(rect)

        point.set_data(0, 0)
        point1.set_data(Q1x[i], Q1y[i])
        point2.set_data(Q2x[i], Q2y[i])

        if phaseSpace == 0:
            phase.set_data(th[i], om[i])
            phase1.set_data(phi1[i], om1[i])
            phase2.set_data(phi2[i], om2[i])
            liste = [rect1_, rect2_, rect3_, rect4_, point, point1, point2, phase, phase1, time_text, sector, phase2]
        else:
            phase1.set_data(phi1[i], phi2[i])
            phase2.set_data(om1[i], om2[i])
            liste = [rect1, rect2, rect3, rect4, point, point1, point2, phase1, time_text, sector, phase2]

        if arrow_velocity:
            nonlocal a_s, arrow_s
            ax.patches.remove(a_s)
            arrow_s = Arrow(x[i], y[i], vx[i] * L / (2 * vx_max), vy[i] * L / (2 * vy_max), color='C3',
                            edgecolor=None, width=L / 10)
            a_s = ax.add_patch(arrow_s)
            liste.append(arrow_s)

        time_text.set_text(time_template % (t[i + ratio - 1]))
        sector.set_theta1(90 - 360 * t[i + ratio - 1] / Tend)
        ax.add_patch(sector)

        return tuple(liste)

    anim = FuncAnimation(fig, animate, n // ratio,
                         interval=1000. / fps, blit=True,
                         init_func=init, repeat_delay=3000)

    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save:
        anim.save('Pendule_L_1', fps=30)
    else:
        plt.show()


dcm1, dcm2 = 2, 3

parameters = [
    r"Axe x : $x_2$",
    r"Axe y : $y_2$",
    r"Axe c : $v_2$",
    "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
    r"$L - \,h\, - \,m\,\:$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(L, h, M, dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
    r"$l_1 - h_1 - m_1$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(l1, h1, m1, dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
    r"$l_2 - h_2 - m_2$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(l2, h2, m2, dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
    "",
    r"$g\,$ = {:.2f} $\rm m/s^2$".format(g),
    "",
    r"$\vartheta \:\;- \omega\:\:$ = {:.{dcmA}f}, {:.{dcmB}f}".format(th0, om0, dcmA=dcm2, dcmB=dcm2),
    r"$\varphi_1 - \omega_1$ = {:.{dcmA}f}, {:.{dcmB}f}".format(phi10, om10, dcmA=dcm2, dcmB=dcm2),
    r"$\varphi_2 - \omega_2$ = {:.{dcmA}f}, {:.{dcmB}f}".format(phi20, om20, dcmA=dcm2, dcmB=dcm2)
]
parameters[0] = r"Axe x : $x_p$"
parameters[1] = r"Axe y : $y_p$"
parameters[2] = r"Axe c : $v_p$"

# see_path_1(1, np.array([th, phi1]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([th, phi2]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([th, om]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([th, om1]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([th, om2]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([phi1, phi2]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([phi1, om]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([phi1, om1]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([phi1, om2]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([phi2, om]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([phi2, om1]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([phi2, om2]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([om, om1]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([om, om2]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
# see_path_1(1, np.array([om1, om2]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
#
# see_path_1(1, np.array([om1, om2]), np.hypot(v2x, v1y), 'magma', var_case=2, save=False)
# see_path_1(1, np.array([phi1, om1 * phi2]), np.hypot(v2x, v1y), 'inferno', var_case=2, save=False)
# see_path(1, [np.array([phi2, sin(om2)]), np.array([phi1, sin(om1)])], [np.hypot(v2x, v2y), np.hypot(vx, vy)],
#             ['inferno', 'viridis'], var_case=2, save=False)
#
# see_path(1, [np.array([phi1, phi2]), np.array([om1, om2])], [np.hypot(v1x, v1y), np.hypot(v2x, v2y)],
#             ['viridis', 'inferno_r'], var_case=2, save="no", displayedInfo=parameters)
#
# see_path(1, [np.array([th, om]), np.array([phi1, om1]), np.array([phi2, om2])],
#             [np.hypot(vx, vy), np.hypot(v1x, v1y), np.hypot(v2x, v2y)],
#             ('viridis', 'inferno_r', 'jet'), var_case=2, save=False)

see_animation(arrow_velocity=True, phaseSpace=0)
