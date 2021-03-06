# ok now, but not able to find parameters for nice trajectories

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Arrow, Rectangle
from numpy import sin, cos, radians, sqrt, pi, degrees, amax
from timeit import default_timer as timer
from scipy.integrate import odeint  # solve_ivp
from Utils.Fixed_Path import see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 11


class Simulation:
    def __init__(self, dic_params, dic_initial, dic_setup):
        self.g = dic_params["g"]
        self.L, self.h, self.M, self.D = dic_params["L"], dic_params["h"], dic_params["M"], dic_params["D"]
        self.l1, self.h1, self.m1, self.D1 = dic_params["l1"], dic_params["h1"], dic_params["m1"], dic_params["D1"]
        self.l2, self.h2, self.m2, self.D2 = dic_params["l2"], dic_params["h2"], dic_params["m2"], dic_params["D2"]

        self.l = self.L - self.h
        L, l, h = self.L, self.l, self.h

        self.d, self.d1, self.d2 = sqrt2 / 2. * L * l / (L + l), (self.l1 - self.h1) / 2., (self.l2 - self.h2) / 2.
        self.IG0 = self.M / 6. * (5. * L ** 2 - 5. * L * h + h ** 2) * (L ** 2 - L * h + h ** 2) / (2. * L - h) ** 2
        self.IG1 = self.m1 / 12. * (self.l1 ** 2 + self.h1 ** 2)
        self.IG2 = self.m2 / 12. * (self.l2 ** 2 + self.h2 ** 2)

        self.th, self.phi1, self.phi2 = dic_initial["th"], dic_initial["phi1"], dic_initial["phi2"]
        self.om, self.om1, self.om2 = dic_initial["om"], dic_initial["om1"], dic_initial["om2"]

        self.t_sim = dic_setup["t_sim"]
        self.fps = dic_setup["fps"]
        self.slowdown = dic_setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = dic_setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.nFrames = int(self.fps * self.t_anim)
        self.nSteps = self.oversample * self.nFrames

        return


def solve_equations_of_motion(sim):
    """
    solve the equations of motion of this system
    :param sim: object of the class Simulation with all parameters needed
    :return: the vectors containing the solutions th, phi... for every time step
    """
    def f(_, u, MAT, RHS):
        th_, phi1_, phi2_, w, w1, w2 = u
        COS1, COS2 = cos(pi / 4. + phi1_ - th_), cos(pi / 4. + phi2_ - th_)
        SIN1, SIN2 = sin(pi / 4. + phi1_ - th_), sin(pi / 4. + phi2_ - th_)

        INV_12, INV_13, INV_23 = INV_12_var * COS1, INV_13_var * SIN2, INV_23_var * SIN2 * COS1
        INV_22, INV_33 = INV_22_fix + INV_22_var * SIN2 * SIN2, INV_33_fix + INV_33_var * COS1 * COS1
        MAT[0, 0], MAT[0, 1], MAT[0, 2] = INV_11_fix, INV_12, INV_13
        MAT[1, 0], MAT[1, 1], MAT[1, 2] = INV_12, INV_22, INV_23
        MAT[2, 0], MAT[2, 1], MAT[2, 2] = INV_13, INV_23, INV_33

        RHS[0] = MASS_D * SIN1 * w1 * w1 + MASS_E * COS2 * w2 * w2 + RHS_11 * sin(th_) + RHS_12 * sin(
                th_ - pi / 4.) + RHS_13 * cos(th_ - pi / 4.) - sim.D * w
        RHS[1] = -MASS_D * SIN1 * w * w + RHS_21 * sin(phi1_) - sim.D1 * w1
        RHS[2] = -MASS_E * COS2 * w * w + RHS_31 * sin(phi2_) - sim.D2 * w2

        delta = MASS_A * MASS_B * MASS_C \
                - MASS_B * MASS_E * MASS_E * SIN2 * SIN2 \
                - MASS_C * MASS_D * MASS_D * COS1 * COS1
        res = np.einsum("ijk,jk->ik", MAT, RHS) / delta

        if _ < 0.:
            return np.vstack((w, w1, w2, res[0], res[1], res[2]))
        else:
            return np.hstack((w, w1, w2, res[0], res[1], res[2]))

    g, M, d, l, m1, m2, d1, d2 = sim.g, sim.M, sim.d, sim.l, sim.m1, sim.m2, sim.d1, sim.d2
    MATRIX = np.empty((3, 3, 1))
    VECTOR = np.empty((3, 1))

    MASS_A = M * d * d + l * l * (m1 + m2) + sim.IG0
    MASS_D = l * m1 * d1
    MASS_E = l * m2 * d2
    MASS_B = m1 * d1 * d1 + sim.IG1
    MASS_C = m2 * d2 * d2 + sim.IG2

    INV_12_var = -MASS_C * MASS_D
    INV_13_var = +MASS_B * MASS_E
    INV_23_var = -MASS_D * MASS_E
    INV_11_fix = +MASS_B * MASS_C
    INV_22_fix = +MASS_A * MASS_C
    INV_22_var = -MASS_E * MASS_E
    INV_33_fix = +MASS_A * MASS_B
    INV_33_var = -MASS_D * MASS_D

    RHS_11, RHS_12, RHS_13 = -g * M * d, -g * m1 * l, -g * m2 * l
    RHS_21, RHS_31 = -g * m1 * d1, -g * m2 * d2

    t = np.linspace(0., sim.t_sim, sim.nSteps + 1)
    # phi10 = 1/9*pi ; phi20 = 1/9*pi
    U0 = radians(np.array([sim.th, sim.phi1, sim.phi2, sim.om, sim.om1, sim.om2]))

    tic = timer()
    sol = odeint(f, U0, t, tfirst=True, atol=1.e-9, rtol=1.e-9, args=(MATRIX, VECTOR))
    # sol = solve_ivp(f, [0, sim.t_sim], U0, method="LSODA", t_eval=t, atol=1.e-12, rtol=1.e-12, args=(MATRIX, VECTOR))
    print(f"\tTime to integrate ODE : {timer() - tic:.2f} seconds")

    # th, phi1, phi2, om, om1, om2 = sol.y
    th, phi1, phi2, om, om1, om2 = sol.T
    # phi2 = np.fmod(np.fmod(phi2, 2 * pi) + 2 * pi + pi, 2 * pi) - pi
    MATRIX = np.empty((3, 3, sim.nSteps + 1))
    VECTOR = np.empty((3, sim.nSteps + 1))
    _, _, _, d2th, d2phi1, d2phi2 = f(-1., (th, phi1, phi2, om, om1, om2), MATRIX, VECTOR)

    return t, th, phi1, phi2, om, om1, om2, d2th, d2phi1, d2phi2


def get_kinematics(sim, time_series):
    t, th, phi1, phi2, om, om1, om2, d2th, d2phi1, d2phi2 = time_series

    x = sim.d * sin(th)
    y = - sim.d * cos(th)
    vx, vy = sim.d * cos(th) * om, sim.d * sin(th) * om
    d2x = -sim.d * sin(th) * om*om + sim.d * cos(th) * d2th
    d2y = sim.d * cos(th) * om*om + sim.d * sin(th) * d2th

    v1x = sim.l * om * cos(th - pi / 4) + sim.h1 * om1 * cos(phi1)
    v1y = sim.l * om * sin(th - pi / 4) + sim.h1 * om1 * sin(phi1)
    v2x = - sim.l * om * sin(th - pi / 4) + sim.h2 * om2 * cos(phi2)
    v2y = + sim.l * om * cos(th - pi / 4) + sim.h2 * om2 * sin(phi2)

    return x, y, vx, vy, d2x, d2y, v1x, v1y, v2x, v2y


def see_animation(sim, time_series, arrow=False, phaseSpace=0, save=""):
    """
    Creates and displays the animation of the system
    :param sim: object of the class Simulation with all parameters needed
    :param time_series: the vectors containing the solutions th, phi... for every time step
    :param arrow: (boolean) display or not the acceleration of the L-shaped pendulum
    :param phaseSpace: (integer) choose the display of the phase space diagram
    :param save: choose the way of saving (or not) the animation
    """
    L, l, l1, l2, h, h1, h2 = sim.L, sim.l, sim.l1, sim.l2, sim.h, sim.h1, sim.h2
    d, d1, d2, M, m1, m2 = sim.d, sim.d1, sim.d2, sim.M, sim.m1, sim.m2
    tFull, thFull, phi1Full, phi2Full, omFull, om1Full, om2Full, d2thFull, d2phi1Full, d2phi2Full = time_series

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]
    t, th, phi1, phi2, om, om1, om2, d2th, d2phi1, d2phi2 = time_series

    Q1x, Q1y = l * sin(th - pi / 4), - l * cos(th - pi / 4)
    Q2x, Q2y = l * cos(th - pi / 4), + l * sin(th - pi / 4)

    Ax, Ay = -h / sqrt2 * cos(th), -h / sqrt2 * sin(th)
    alpha_A = degrees(pi / 4. + th)
    width_A, height_A = h, -L
    Bx, By = Ax, Ay
    alpha_B = degrees(-pi / 4. + th)
    width_B, height_B = h, -l
    Cx, Cy = Q1x - h1 / sqrt2 * cos(phi1 - pi / 4.), Q1y - h1 / sqrt2 * sin(phi1 - pi / 4)
    alpha_C = degrees(phi1)
    width_C, height_C = h1, -l1
    Dx, Dy = Q2x - h2 / sqrt2 * cos(phi2 - pi / 4.), Q2y - h2 / sqrt2 * sin(phi2 - pi / 4)
    alpha_D = degrees(phi2)
    width_D, height_D = h2, -l2

    # E_K = 0.5 * (M * d * d + l * l * (m1 + m2) + sim.IG0) * om * om + 0.5 * (
    #         m1 * d1 * d1 + sim.IG1) * om1 * om1 + 0.5 * (m2 * d2 * d2 + sim.IG2) * om2 * om2 + l * om * (
    #               m1 * d1 * om1 * cos(pi / 4. + phi1 - th) - m2 * d2 * om2 * sin(pi / 4. + phi2 - th))
    # E_U = sim.g * (
    #         M * d * (1 - cos(th)) + m1 * l * (1 - cos(th - pi / 4)) + m2 * l * (sin(th - pi / 4) + 1) + m1 * d1 * (
    #             1 - cos(phi1)) + m2 * d2 * (1 - cos(phi2)))
    # plt.stackplot(t, E_K, E_U)
    # plt.show()

    x, y, vx, vy, d2x, d2y, v1x, v1y, v2x, v2y = get_kinematics(sim, time_series)
    ax_max, ay_max = amax(d2x), amax(d2y)

    #####     ================      Cr??ation de la figure      ================      #####

    fig, axs = plt.subplots(1, 2, figsize=(14., 7.))
    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

    ax = axs[0]
    ax.axis([-1.1 * (L + max(l1, l2)), 1.1 * (L + max(l1, l2)), -1.1 * (L + max(l1, l2)), 1.1 * (L + max(l1, l2))])
    ax.set_aspect("equal")
    ax.grid(ls=':')

    ax2 = axs[1]
    ax2.grid(ls=':')
    ax2.set_xlabel(r'$\varphi \; \rm [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \; \rm [rad \: / \: s]$', fontsize=ftSz2)

    M_max = np.amax(np.array([M, m1, m2]))
    rect1 = Rectangle((Ax[0], Ay[0]), width_A, height_A, alpha_A[0], color='C0', alpha=M_max/M_max, zorder=0)
    rect2 = Rectangle((Bx[0], By[0]), width_B, height_B, alpha_B[0], color='C0', alpha=M_max/M_max, zorder=0)
    rect3 = Rectangle((Cx[0], Cy[0]), width_C, height_C, alpha_C[0], color='C1', alpha=M_max/M_max, zorder=2)
    rect4 = Rectangle((Dx[0], Dy[0]), width_D, height_D, alpha_D[0], color='C2', alpha=M_max/M_max, zorder=4)
    for rect in [rect1, rect2, rect3, rect4]:
        ax.add_patch(rect)

    point, = ax.plot([], [], marker='o', ms=5, color='black', zorder=1)
    point1, = ax.plot([], [], marker='o', ms=5, color='black', zorder=3)
    point2, = ax.plot([], [], marker='o', ms=5, color='black', zorder=5)

    phase, = ax2.plot([], [], marker='o', ms=8, color='C0', zorder=3)
    phase1, = ax2.plot([], [], marker='o', ms=8, color='C1', zorder=2)
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C2', zorder=1)

    if arrow:
        dx_arrow, dy_arrow = d2x * L / (2 * ax_max), d2y * L / (2 * ay_max)
        arrow_s = Arrow(x[0], y[0], dx_arrow[0], dy_arrow[0], color='C3', edgecolor=None, width=L / 10)

    time_template = r'$t \;\:\: = {:.2f} \; s$' if save == "snapshot" else r'$t \;\:\: = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.45, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((0.5, 0.85), 0.04, theta1=90, theta2=90, color='lightgrey', transform=ax.transAxes)

    ax.text(0.17, 0.96, r'$L  = {:.2f} \,m$'.format(L), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.17, 0.93, r'$h  = {:.2f} \,m$'.format(h), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.17, 0.90, r'$m  = {:.2f} \,kg$'.format(M), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax.text(0.02, 0.96, r'$l_1  = {:.2f} \,m$'.format(l1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.93, r'$h_1  = {:.2f} \,m$'.format(h1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.90, r'$m_1  = {:.2f} \,kg$'.format(m1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax.text(0.02, 0.85, r'$l_2  = {:.2f} \,m$'.format(l2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.82, r'$h_2  = {:.2f} \,m$'.format(h2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.79, r'$m_2  = {:.2f} \,kg$'.format(m2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    # ax.text(0.02, 0.84, r'$D  = {:.2f} \,kg/s$'.format(D), fontsize=12, wrap=True, transform=ax.transAxes)
    ax.text(0.73, 0.96, r'$\vartheta  = {:.2f} $'.format(sim.th), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.73, 0.93, r'$\varphi_1  = {:.2f} $'.format(sim.phi1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.73, 0.90, r'$\varphi_2  = {:.2f} $'.format(sim.phi2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.88, 0.96, r'$\omega  \;\,  = {:.2f} $'.format(sim.om), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.88, 0.93, r'$\omega_1  = {:.2f} $'.format(sim.om1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.88, 0.90, r'$\omega_2  = {:.2f} $'.format(sim.om2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    if phaseSpace == 0:
        ax2.plot(thFull, omFull, color='C0', label='pendule L', zorder=3)
        ax2.plot(phi1Full, om1Full, ls='-', marker='', ms=1, color='C1', label='masse 1', zorder=2)
        ax2.plot(phi2Full, om2Full, ls='-', marker='', ms=1, color='C2', label='masse 2', zorder=1)
        ax2.legend(fontsize=0.5 * (ftSz2 + ftSz3))
    else:
        ax2.plot(phi1, phi2, color='C1')
        ax2.plot(om1, om2, color='C2')

    #####     ================      Animation      ================      #####

    a_s = None

    def init():
        point.set_data([], [])
        point1.set_data([], [])
        point2.set_data([], [])
        phase1.set_data([], [])
        phase2.set_data([], [])
        liste = [rect1, rect2, rect3, rect4, point, point1, point2, phase1, phase2, time_text, sector]
        if phaseSpace == 0:
            phase.set_data([], [])
            liste.append(phase)
        if arrow:
            nonlocal a_s
            a_s = ax.add_patch(arrow_s)
            liste.append(arrow_s)
        time_text.set_text('')
        sector.set_theta1(90)
        return tuple(liste)

    def update(i):
        liste = []

        rect1.set(xy=(Ax[i], Ay[i]), angle=alpha_A[i])
        rect2.set(xy=(Bx[i], By[i]), angle=alpha_B[i])
        rect3.set(xy=(Cx[i], Cy[i]), angle=alpha_C[i])
        rect4.set(xy=(Dx[i], Dy[i]), angle=alpha_D[i])
        liste += [rect1, rect2, rect3, rect4]

        point.set_data(0, 0)
        point1.set_data(Q1x[i], Q1y[i])
        point2.set_data(Q2x[i], Q2y[i])
        liste += [point, point1, point2]

        if phaseSpace == 0:
            phase.set_data(th[i], om[i])
            phase1.set_data(phi1[i], om1[i])
            phase2.set_data(phi2[i], om2[i])
            liste += [phase, phase1, phase2]
        else:
            phase1.set_data(phi1[i], phi2[i])
            phase2.set_data(om1[i], om2[i])
            liste += [phase1, phase2]

        if arrow:
            nonlocal a_s, arrow_s
            ax.patches.remove(a_s)
            arrow_s = Arrow(x[i], y[i], dx_arrow[i], dy_arrow[i], color='C3', edgecolor=None, width=L / 10)
            a_s = ax.add_patch(arrow_s)
            liste += [arrow_s]

        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)
        liste += [time_text, sector]

        return tuple(liste)

    sim.nFrames //= 10 if save == "gif" else 1
    anim = FuncAnimation(fig, update, sim.nFrames, interval=1000./sim.fps, blit=True, init_func=init, repeat_delay=3000)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)
    fig.tight_layout()

    if save == "save":
        anim.save('Pendule_L_1', fps=30)
    elif save == "gif":
        # noinspection PyTypeChecker
        anim.save('./pendulum_L.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted, sim.oversample = 12.5, 1
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./pendulum_L.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def display_path(sim, time_series):
    t, th, phi1, phi2, om, om1, om2, d2th, d2phi1, d2phi2 = time_series
    x, y, vx, vy, d2x, d2y, v1x, v1y, v2x, v2y = get_kinematics(sim, time_series)

    dcm1, dcm2 = 2, 3
    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$L - \,h\, - \,m\,\:$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(sim.L, sim.h, sim.M,
                                                                                dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
        r"$l_1 - h_1 - m_1$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(sim.l1, sim.h1, sim.m1,
                                                                            dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
        r"$l_2 - h_2 - m_2$ = {:.{dcmA}f}, {:.{dcmB}f}, {:.{dcmC}f}".format(sim.l2, sim.h2, sim.m2, dcmA=dcm1, dcmB=dcm1, dcmC=dcm1),
        "",
        r"$g\,$ = {:.2f} $\rm m/s^2$".format(sim.g),
        "",
        r"$\vartheta \:\;- \omega\:\:$ = {:.{dcmA}f}, {:.{dcmB}f}".format(sim.th, sim.om, dcmA=dcm2, dcmB=dcm2),
        r"$\varphi_1 - \omega_1$ = {:.{dcmA}f}, {:.{dcmB}f}".format(sim.phi1, sim.om1, dcmA=dcm2, dcmB=dcm2),
        r"$\varphi_2 - \omega_2$ = {:.{dcmA}f}, {:.{dcmB}f}".format(sim.phi2, sim.om2, dcmA=dcm2, dcmB=dcm2)
    ]
    parameters[0] = r"Axe x : $x_p$"
    parameters[1] = r"Axe y : $y_p$"
    parameters[2] = r"Axe c : $v_p$"

    see_path_1(1, np.array([th, phi1]), np.hypot(v2x, v1y), color='jet', shift=(0., 0.), save="no", displayedInfo=parameters)
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

    see_path(1, [np.array([th, om]), np.array([phi1, om1]), np.array([phi2, om2])],
                [np.hypot(vx, vy), np.hypot(v1x, v1y), np.hypot(v2x, v2y)],
                ('viridis', 'inferno_r', 'jet'), var_case=2, save=False)
    return


if __name__ == "__main__":
    sqrt2 = sqrt(2)

    mode = "animation"
    # mode = "path"
    params = {"g": 9.81,
              "L": 1.00, "h": 0.20, "M": 5., "D": 0.,
              "l1": 1.00, "h1": 0.10, "m1": 1., "D1": 0.,
              "l2": 1./sqrt2, "h2": 0.10, "m2": 1./sqrt2, "D2": 0.}
    initial = {"th": 0., "phi1": 175., "phi2": 0.,
               "om": 0., "om1": 0., "om2": 0.}
    setup = {"t_sim": 20., "fps": 50., "slowdown": 0.5, "oversample": 50}

    simulation = Simulation(params, initial, setup)
    solutions = solve_equations_of_motion(simulation)

    if mode == "animation":
        see_animation(simulation, solutions, arrow=False, phaseSpace=0, save="")
    elif mode == "path":
        display_path(simulation, solutions)
