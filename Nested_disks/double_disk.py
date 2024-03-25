import sys
import os

# Add the root directory of your project to sys.path
current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import sin, cos
from scipy.integrate import odeint
from time import perf_counter
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14


class NestedDisks:
    def __init__(self, params, initials, setup):
        self.g = params["g"]
        self.R, self.M = params["R"], params["M"]
        self.a, self.m = params["a"], params["m"]
        self.k, self.k1 = params["k"], params["k1"]
        self.c = 2 * self.M + 3 / 2 * self.m
        self.s = self.R - self.a

        self.th, self.om = initials["th"], initials["om"]
        self.x, self.dx = initials["x"], initials["dx"]
        self.phi1, self.phi2 = 0., 0.
        self.thd = np.degrees(self.th)

        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames

        return


def f(u, _, sim):
    g, R, M, a, m = sim.g, sim.R, sim.M, sim.a, sim.m
    k, k1, c = sim.k, sim.k1, sim.c
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

    return np.array([u[1], f1, u[3], f2])


def nested_disks_ode(sim):
    t = np.linspace(0., sim.t_sim, sim.n_steps + 1)  # ODEINT
    U0 = np.array([sim.th, sim.om, sim.x, sim.dx])

    start = perf_counter()
    sol = odeint(f, U0, t, args=(sim,))
    end = perf_counter()

    print(f"\tElapsed time : {end - start:.3f} seconds")
    th, om, x, v = sol.T

    return t, th, om, x, v


def nested_disks_kinematics(sim, time_series):
    t, th, om, x, v = time_series
    R, a = sim.R, sim.a

    phi1 = sim.phi1 + (x - sim.x) / R
    phi2 = sim.phi2 + (R * (phi1 - sim.phi1) + (a - R) * (th - sim.th)) / a

    xc1 = x
    yc1 = np.zeros_like(t)  # position du centre de l'anneau
    xc2, yc2 = x + (R - a) * cos(th), -(R - a) * sin(th)  # position du centre du disque
    x1, y1 = xc1 + R * cos(phi1), yc1 - R * sin(phi1)  # position d'un point sur l'anneau
    x2, y2 = xc2 + a * cos(phi2), yc2 - a * sin(phi2)  # position d'un point sur le disque

    vx2 = v + (a - R) * sin(th) * om - a * sin(phi2) * ((v + (a - R) * om) / a)
    vy2 = (a - R) * cos(th) * om - a * cos(phi2) * ((v + (a - R) * om) / a)
    v2 = np.hypot(vx2, vy2)

    return phi1, phi2, xc1, yc1, xc2, yc2, x1, y1, x2, y2, v2


def see_animation(sim, time_series, save=""):

    t_full, th_full, om_full, x_full, dx_full = time_series
    kinematics = nested_disks_kinematics(sim, time_series)

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, th, om, x, dx = time_series
    phi1, phi2, xc1, yc1, xc2, yc2, x1, y1, x2, y2, v2 = nested_disks_kinematics(sim, time_series)

    R, a, M, m = sim.R, sim.a, sim.M, sim.m

    xmin, xmax = np.amin(xc1) - 1.5 * R, np.amax(xc1) + 1.5 * R
    ymin, ymax = -1.5 * R, 3 * R

    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
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
    phase21, = ax2.plot([], [], marker='o', ms=8, color='C1')
    phase31, = ax3.plot([], [], marker='o', ms=8, color='C2')
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

    ax.text(0.02, 0.92, r'$R  = {:.2f} $'.format(sim.R), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.84, r'$r  = {:.2f} $'.format(sim.a), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.10, 0.92, r'$M  = {:.2f} $'.format(sim.M), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.10, 0.84, r'$m  = {:.2f} $'.format(sim.m), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.70, 0.92, r'$\theta_0  = {:.2f} $'.format(sim.thd), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.70, 0.84, r'$\omega_0  = {:.2f} $'.format(sim.om), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.92, r'$x_0  = {:.2f} $'.format(sim.x), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.84, r'$v_0  = {:.2f} $'.format(sim.dx), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax2.plot(th_full, dx_full, color='C1')
    # ax2.plot(om_full, dx_full, color='C1')
    ax3.plot(t_full, dx_full, color='C2')

    # ax2.plot(t_full, th_full, label='theta %time')
    # ax2.plot(t_full, om_full, label='omega % time')
    # ax2.plot(t_full, x_full, label='position % time')
    # ax2.plot(t_full, dx_full, label='speed % time')
    # ax2.plot(th_full, dx_full, label='speed in function of theta')

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
        thisx0, thisx1, thisx2 = [xc1[i], xc2[i]], [xc1[i], x1[i]], [xc2[i], x2[i]]
        thisy0, thisy1, thisy2 = [yc1[i], yc2[i]], [yc1[i], y1[i]], [yc2[i], y2[i]]
        circ1.center = (xc1[i], yc1[i])
        circ2.center = (xc2[i], yc2[i])
        ax.add_patch(circ1)
        ax.add_patch(circ2)
        line1.set_data(thisx0, thisy0)
        line2.set_data(thisx1, thisy1)
        line3.set_data(thisx2, thisy2)
        phase21.set_data(th[i], dx[i])
        phase31.set_data(t[i], dx[i])
        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)

        return line1, line2, line3, phase21, phase31, time_text, circ1, circ2, sector

    anim = FuncAnimation(fig, update, sim.n_frames+1, interval=33, blit=True, init_func=init, repeat_delay=3000)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save == "save":
        anim.save('double_disk_3.html', fps=30)
    elif save == "gif":
        anim.save('./disks.gif', writer=PillowWriter(fps=sim.fps))
    elif save == "snapshot":
        t_wanted = 6.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./disks.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def path_nested_disks(sim, time_series):
    t, th, om, x, dx = time_series
    R, a, M, m = sim.R, sim.a, sim.M, sim.m
    phi1, phi2, xc1, yc1, xc2, yc2, x1, y1, x2, y2, v2 = nested_disks_kinematics(sim, time_series)

    params1 = np.array([sim.R, sim.M, sim.a, sim.m])
    params2 = np.array([sim.thd, sim.om, sim.x, sim.dx])
    dcm1, dcm2 = 3, 4
    fmt1, fmt2 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val in params2:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

    parameters = [
        r"Axe x : $\vartheta$",
        r"Axe y : $\omega$",
        r"Axe c : $\dot x$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$R \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.R, width=fmt1, dcm=dcm1),
        r"$M \;\:\,$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.M, width=fmt1, dcm=dcm1),
        r"$a \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.a, width=fmt1, dcm=dcm1),
        r"$m \;\:\,$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m, width=fmt1, dcm=dcm1),
        "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\vartheta$ = {:>{width}.{dcm}f} $\rm ^\circ$".format(sim.thd, width=fmt2, dcm=dcm2),
        r"$\omega$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om, width=fmt2, dcm=dcm2),
        r"$x$ = {:>{width}.{dcm}f} $\rm m$".format(sim.x, width=fmt2, dcm=dcm2),
        r"$\dot x$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.dx, width=fmt2, dcm=dcm2),
    ]

    see_path_1(2, np.array([t, dx]), dx, var_case=2, bar=False, save=False)
    return


if __name__ == "__main__":

    params = {
        'g': 9.81, 'k': -0.00, 'k1': 0.00,
        'R': 0.50, 'M': 1.00, 'a': 0.20, 'm': 1.00, 
    }

    initials = {
        'th': np.radians(-80.00), 'om': 0 * np.sqrt(params['g'] / params['R']), 
        'x': 0.00, 'dx': 0.50,
    }

    setup = {
        't_sim': 10., 'fps': 30, 'slowdown': 1.0, 'oversample': 10,
    }

    sim = NestedDisks(params, initials, setup)
    time_series = nested_disks_ode(sim)

    see_animation(sim, time_series, save="")
    # path_nested_disks(sim, time_series)
