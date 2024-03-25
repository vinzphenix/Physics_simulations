import sys
import os

# Add the root directory of your project to sys.path
current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter, HTMLWriter
from numpy import sin, cos, hypot, radians, array, amax, amin, mean
from scipy.integrate import odeint
from time import perf_counter
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14
lw = 1.


class Atwood_Simulation:
    def __init__(self, dic_params, dic_initial, dic_setup):
        self.g, self.M, self.m = dic_params["g"], dic_params["M"], dic_params["m"]
        self.mu = self.M / self.m

        self.r, self.dr, = dic_initial["r"], dic_initial["dr"]
        self.th, self.om = dic_initial["th"], dic_initial["om"]
        self.thd = np.degrees(self.th)

        self.t_sim, self.fps = dic_setup["t_sim"], dic_setup["fps"]
        self.slowdown = dic_setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = dic_setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.nFrames = int(self.fps * self.t_anim)
        self.nSteps = self.oversample * self.nFrames


def atwood_ode(sim):
    def f(u, _):
        r_, dr_, th_, om_ = u
        f1 = (m * r_ * om_ * om_ + g * (-M + m * cos(th_))) / (M + m)
        f2 = -1 / r_ * (2 * dr_ * om_ + g * sin(th_))
        return array([dr_, f1, om_, f2])

    m, g, M = sim.m, sim.g, sim.M
    t = np.linspace(0, sim.t_sim, sim.nSteps + 1)
    U0 = array([sim.r, sim.dr, sim.th, sim.om])

    tic = perf_counter()
    sol = odeint(f, U0, t)
    print(f"\tElapsed time : {perf_counter() - tic: .3f} seconds")

    r, dr, th, om = sol.T
    return t, r, dr, th, om


def atwood_kinematics(sim, time_series):
    g, M, m = sim.g, sim.M, sim.m
    t, r, dr, th, om = time_series

    d = mean(r)
    L = max(r) + d * 1.2

    x1, y1 = -d * np.ones_like(t), r + d - L
    x2, y2 = r * sin(th), -r * cos(th)

    vx = dr * sin(th) + r * om * cos(th)
    vy = -dr * cos(th) + r * om * sin(th)
    v = hypot(vx, vy)

    ddr = (m * r * om * om + g * (-M + m * cos(th))) / (M + m)
    dom = -1 / r * (2 * dr * om + g * sin(th))
    acx = ddr * sin(th) + dr * cos(th) * om - r * om * om * sin(th) + (r * dom + dr * om) * cos(th)
    acy = -ddr * cos(th) + dr * sin(th) * om + r * om * om * cos(th) + (r * dom + dr * om) * sin(th)
    a = hypot(acx, acy)
    return x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a


def see_animation(sim, time_series, save=""):
    t_full, r_full, dr_full, th_full, om_full = time_series
    _, _, x2_full, y2_full, _, _, _, _, _, _, _, _ = atwood_kinematics(sim, time_series)

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, r, dr, th, om = time_series
    x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = atwood_kinematics(sim, time_series)
    L_X, L_Y = amax(x2) - amin(x2), amax(y2) - amin(y2)
    x_m, y_m = amin(x2) - 0.2 * L_X, amin(y2) - 0.2 * L_Y
    x_M, y_M = amax(x2) + 0.2 * L_X, amax(y2) + 0.2 * L_Y

    d = mean(r)
    x_m = min(-d * 1.1, x_m)
    if abs(amin(y2)) < L_X:
        y_m = x_m
        y_M = x_M

    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
    fig, axs = plt.subplots(1, 2, figsize=(14., 7.))  # , constrained_layout=True)
    ax, ax2 = axs[0], axs[1]

    tmp, = ax.plot([x_m, x_M], [y_m, y_M])
    ax.set_aspect("equal", "datalim")
    tmp.remove()
    ax.grid(ls=':')
    ax2.grid(ls=':')
    ax2.set_xlabel(r'$\vartheta \;\; \rm [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \;\; \rm [rad\,/\,s]$', fontsize=ftSz2)

    line,  = ax.plot([], [], 'o-', lw=2.5, color='C1', zorder=10)
    line2, = ax.plot([], [], '-', lw=1, color='grey')
    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C1', alpha=0.)

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.40, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((x_M - L_X/10, x_m + L_X/10), L_X/20, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.04, 0.94, r'$\mu  = {:.2f}$'.format(sim.mu), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ftSz4 = ftSz3 * 0.9
    ax.text(0.66, 0.96, r'$r  = {:.2f} $'.format(sim.r), fontsize=ftSz4, transform=ax.transAxes)
    ax.text(0.66, 0.92, r'$\dot{{r}}  = {:.2f} $'.format(sim.dr), fontsize=ftSz4, transform=ax.transAxes)
    ax.text(0.80, 0.96, r'$\vartheta  = {:.2f} \:\rm deg$'.format(sim.thd), fontsize=ftSz4, transform=ax.transAxes)
    ax.text(0.80, 0.92, r'$\omega  = {:.2f} \:\rm rad/s$'.format(sim.om), fontsize=ftSz4, transform=ax.transAxes)

    ax2.plot(th_full, om_full, color='C0')
    # ax2.plot(r_full, dr_full, color='C1')

    #####     ================      Animation      ================      #####

    def init():
        line.set_data([], [])
        line2.set_data([], [])
        phase1.set_data([], [])
        phase2.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line, line2, phase1, phase2, time_text, sector

    def update(i):
        start = max(0, i - int(sim.slowdown * sim.fps * 100.))  # display history of last ... seconds
        thisx = [x2[i], 0, x1[i], x1[i]]
        thisy = [y2[i], 0, 0, y1[i]]

        line.set_data(thisx, thisy)
        line2.set_data([x2_full[k * start:k * i + 1]], [y2_full[k * start: k * i + 1]])
        phase1.set_data(th[i], om[i])
        phase2.set_data(r[i], dr[i])

        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)

        return line, line2, phase1, phase2, time_text, sector

    fig.tight_layout()
    k = sim.oversample
    sim.nFrames //= 20 if save == "gif" else 1
    anim = FuncAnimation(fig, update, sim.nFrames + 1, interval=5., blit=True, init_func=init, repeat_delay=5000)
    # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)

    if save == "html":
        anim.save('atwood.html', writer=HTMLWriter(fps=sim.fps))
    elif save == "gif":
        # noinspection PyTypeChecker
        anim.save('./atwood.gif', writer=PillowWriter(fps=sim.fps))
    elif save == "mp4":
        anim.save(f"./atwood.mp4", writer=FFMpegWriter(fps=sim.fps))
    elif save == "snapshot":
        t_wanted, sim.oversample = 20., 1
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./atwood.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def display_path_atwood(sim, time_series):
    g, M, m = sim.g, sim.M, sim.m
    t, r, dr, th, om = time_series
    x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = atwood_kinematics(sim, time_series)

    params2 = array([sim.r, sim.dr, sim.thd, sim.om])
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
        r"$r \;\,\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.r, width=fmt2, dcm=dcm2),
        r"$dr$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.dr, width=fmt2, dcm=dcm2),
        r"$\vartheta \;\,$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.thd, width=fmt2, dcm=dcm2),
        r"$\omega \;\,$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om, width=fmt2, dcm=dcm2)
    ]

    parameters[0] = r"Axe x : $\omega \, * \, r$"
    parameters[1] = r"Axe y : $r$"
    parameters[2] = r"Axe c : $v^2$"

    # see_path_1(lw, array([x2, y2]), v, color='jet', var_case=1, shift=(-0., 0.), save='save', displayedInfo=parameters)
    # see_path_1(1, array([th, om]), r, 'inferno', name='th - om', shift=(0., 0.), var_case=2, save='save', displayedInfo=parameters)
    # see_path_1(1., array([th, r]), v, 'Blues', name='th - r', shift=(0., 0.), var_case=2, save='save', displayedInfo=parameters)
    # see_path_1(1, array([th, dr]), v, 'inferno', name='th - dr', shift=(0.0, 0.), var_case=2, save='save', displayedInfo=parameters)
    # see_path_1(1, array([om, r]), v, 'inferno', name='om - r', shift=(0., 0.), var_case=2, save='save', displayedInfo=parameters)
    # see_path_1(1., array([om, v]), r, 'inferno', name='om - dr', shift=(0.1, 0.), var_case=4, save='no', displayedInfo=parameters)
    # see_path_1(1., array([r, dr]), r, 'inferno', name='r dr', shift=(0.1, -0.), var_case=2, save='save', displayedInfo=parameters)

    """
    see_path_1(1., array([dr*r, th]), v, 'Blues', name='1', shift=(-0., 0.), var_case=2, save='no', displayedInfo=parameters)
    see_path_1(1., array([dr, om*th]), -r, 'Blues', name='2', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
    see_path_1(1., array([dr, om*dr]), -r, 'Blues', name='3', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
    see_path_1(1., array([dr, om*r]), -r, 'Blues', name='4', shift=(0., 0.), var_case=2, save='no', displayedInfo=parameters)
    """

    parameters[0] = r"Axe x : $\varphi_1 $  -  $\varphi_2$"
    parameters[1] = r"Axe y : $\omega_1$  -  $\omega_2$"
    parameters[2] = r"Axe c : $\varphi_2$  -  $\varphi_1$"
    # see_path(1, [array([x2, y2]), array([2*x2, 2*y2])],
    #          [v, v], ["Blues", "viridis"],
    #          var_case=2, save="no", displayedInfo=parameters)

    see_path_1(1, array([om, v]), r, 'inferno', name='om - dr', shift=(0., 0.), var_case=4, save='save')
    return


def load_configuration(i):
    
    g, m = 9.81, 1.0
    r, dr, om = 1., 0., 0.  # default intials

    if 1 <= i <= 40:
        mu_list = [
            1.1185, 1.133, 1.172, 1.173, 1.278,
            1.279, 1.338, 1.548, 1.555, 1.565,
            2.000, 2.010, 2.140, 2.150, 2.165,
            2.380, 2.900, 2.945, 3.000, 3.010,
            3.125, 3.300, 3.410, 3.520, 3.867,
            4.633, 4.734, 4.737, 4.745, 4.9986,
            5.475, 5.680, 6.010, 6.013, 6.014,
            6.114, 6.806, 8.150, 19.00, 46.00,
        ]
        M = mu_list[i - 1]
        phi = 90.
    elif 41 <= i <= 57:
        mu_list = [
            1.173, 1.337, 1.655, 1.904, 2.165,
            2.394, 2.812, 3.125, 3.510, 3.520,
            4.1775, 4.745, 4.80, 6.014, 7.242,
            7.244, 16.00, 
        ]
        M = mu_list[i - 41]
        phi = 150.
    elif i == 58:
        M = 16.0
        phi = 35.0
    elif i == 59:
        M = 3.0
        r, dr, phi, om = 0.25, 0., 20.054, 2.830  # these are degrees !
    elif i == 60:
        M = 1.527
        r, dr, phi, om = 1.0, 0., 1.4 * 180 / np.pi, 0.0
    else:
        raise ValueError("Invalid configuration number")

    params = {"g": g, "m": m, "M": M}
    initials = {"r": r, "th": np.radians(phi), "dr": dr, "om": om}
    return params, initials


if __name__ == "__main__":

    params = {
        "g": 9.81, "m": 1., "M": 4.737
    }

    initial = {
        "r": 1., "th": np.radians(90.), "dr": 0., "om": 0.
    }

    setup = {
        "t_sim": 20.*np.sqrt(1.0), "fps": 30., "slowdown": 1., "oversample": 50
    }


    params, initial = load_configuration(60)

    simulation = Atwood_Simulation(params, initial, setup)
    solutions = atwood_ode(simulation)

    see_animation(simulation, solutions, save="")
    # display_path_atwood(simulation, solutions)
