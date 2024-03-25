import sys
import os

# Add the root directory of your project to sys.path
current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import sin, cos, radians, array, sqrt, degrees, amax, amin, abs
from scipy.integrate import odeint
from time import perf_counter
from Utils.Fixed_Path import countDigits, see_path_1

matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['mathtext.rm'] = 'serif'

# ATTENTION : convention angles relatifs


class DrivenPendulum():
    def __init__(self, params, initials, setup):
        self.g, self.l1, self.l2 = params['g'], params['l1'], params['l2']
        self.D, self.m = params['D'], params['m']

        self.phi1, self.phi2 = initials['phi1'], initials['phi2']
        self.phi1d, self.phi2d = degrees(initials['phi1']), degrees(initials['phi2'])
        self.om1, self.om2 = initials['om1'], initials['om2']

        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames

        return
    

def f(u, t_, sim):
    l1, l2, g = sim.l1, sim.l2, sim.g
    phi, om = u
    # f1 = -sim.D / sim.m * (om + sim.om1) - sim.D * l1 * sim.om1 / (sim.m * l2) * cos(phi) - l1 * sim.om1 * sim.om1 / l2 * sin(phi) - g / l2 * sin(sim.phi1 + sim.om1 * t_ + phi)
    f1 = -l1 * sim.om1 * sim.om1 / l2 * sin(phi) - g / l2 * sin(sim.phi1 + sim.om1 * t_ + phi)
    return array([om, f1])


def driven_pendulum_ode(sim):
    t = np.linspace(0., sim.t_sim, sim.n_steps+1)
    U0 = array([sim.phi2, sim.om2 - sim.om1])

    start = perf_counter()
    sol = odeint(f, U0, t, args=(sim,))
    end = perf_counter()

    print(f"\tElapsed time : {end-start:.3f} seconds")

    phi1, om1 = sim.phi1 + sim.om1 * t, np.ones_like(t) * sim.om1
    phi2, om2 = sol.T

    return t, phi1, om1, phi2, om2


def driven_pendulum_kinematics(sim, time_series):

    l1, l2, g, m, D = sim.l1, sim.l2, sim.g, sim.m, sim.D
    t, phi1, om1, phi2, om2 = time_series

    x1, y1 = l1 * sin(phi1), -l1 * cos(phi1)
    x2, y2 = x1 + l2 * sin(phi1 + phi2), y1 - l2 * cos(phi1 + phi2)

    vx2 = l1 * cos(phi1) * om1 + l2 * cos(phi1 + phi2) * (om1 + om2)
    vy2 = l1 * sin(phi1) * om1 + l2 * sin(phi1 + phi2) * (om1 + om2)
    v2 = sqrt(vx2 * vx2 + vy2 * vy2)

    acx2 = -l1 * sin(phi1) * om1**2 - l2 * sin(phi1 + phi2) * (om1 + om2)**2 + l2 * cos(phi1 + phi2) * (-l1 * om1 * om1 / l2 * sin(phi2) - g / l2 * sin(sim.phi1 + om1 * t + phi2))
    acy2 = +l1 * cos(phi1) * om1**2 + l2 * cos(phi1 + phi2) * (om1 + om2)**2 + l2 * sin(phi1 + phi2) * (-l1 * om1 * om1 / l2 * sin(phi2) - g / l2 * sin(sim.phi1 + om1 * t + phi2))
    ac2 = sqrt(acx2**2 + acy2**2)

    T = (m * g * cos(phi1) - D * l1 * om1 * sin(phi2) + m * (l1 * om1 * om1 * cos(phi2) + l2 * (om1 + om2) ** 2))

    return x1, y1, x2, y2, v2, ac2, T


def see_animation(sim, time_series, save=False):

    t, phi1_full, om1_full, phi2_full, om2_full = time_series
    x1, y1, x2_full, y2_full, v2, ac2, T = driven_pendulum_kinematics(sim, time_series)

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, phi1, om1, phi2, om2 = time_series
    x1, y1, x2, y2, v2, ac2, T = driven_pendulum_kinematics(sim, time_series)

    L, T_max = sim.l1 + sim.l2, amax(T)

    fig, axs = plt.subplots(1, 2, figsize=(14, 8))
    ax, ax2 = axs

    # To make matplotlib understand the aspect ratio
    tmp, = ax.plot([1.1*L, 1.1*L, -1.1*L, -1.1*L], [1.1*L, -1.1*L, -1.1*L, 1.1*L], ls='--')
    ax.set_aspect('equal', adjustable='datalim')
    tmp.remove()
    ax2.set_xlabel(r'$\varphi \; \rm [rad]$', fontsize=12)
    ax2.set_ylabel(r'$\omega \; \rm [rad/s]$', fontsize=12)
    ax.grid(ls=':')
    ax2.grid(ls=':')

    time_template = r'$t = %.2f \rm s$'
    time_text = ax.text(0.38, 0.94, '1', fontsize=15, wrap=True, transform=ax.transAxes)
    sector = patches.Wedge((L, -L), L / 15, theta1=90, theta2=90, color='lightgrey')

    fontsize = 11
    ax.text(0.02, 0.96, r'$l_1 = {:.2f} \: \rm m$'.format(sim.l1), fontsize=fontsize, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$l_2 = {:.2f} \: \rm m$'.format(sim.l2), fontsize=fontsize, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.96, r'$m  = {:.2f} \: \rm kg$'.format(sim.m), fontsize=fontsize, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.92, r'$D  = {:.2f} \: \rm kg/s$'.format(sim.D), fontsize=fontsize, wrap=True, transform=ax.transAxes)

    ax.text(0.56, 0.96, r'$\varphi_1  = {:.2f} \;\rm deg$'.format(sim.phi1d), fontsize=fontsize, wrap=True,
            transform=ax.transAxes)
    ax.text(0.56, 0.92, r'$\varphi_2  = {:.2f} \;\rm deg$'.format(sim.phi2d), fontsize=fontsize, wrap=True,
            transform=ax.transAxes)
    ax.text(0.76, 0.96, r'$\omega_1  = {:.2f} \;\rm rad/s$'.format(sim.om1), fontsize=fontsize, wrap=True, transform=ax.transAxes)
    ax.text(0.76, 0.92, r'$\omega_2  = {:.2f} \;\rm rad/s$'.format(sim.om2), fontsize=fontsize, wrap=True, transform=ax.transAxes)

    ax2.plot(phi1_full + phi2_full, om1_full + om2_full, color='C2')

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

    def update(i):
        start = max((i - 50000, 0))

        thisx1, thisx2 = [0, x1[i]], [x1[i], x2[i]]
        thisy1, thisy2 = [0, y1[i]], [y1[i], y2[i]]

        line1.set_data(thisx1, thisy1)
        line2.set_data(thisx2, thisy2)
        line3.set_data(x2_full[k*start:k*i + 1], y2_full[k*start:k*i + 1])

        time_text.set_text(time_template % (t[i]))

        rect.set_bounds(L * 1.0, 0, L * 0.05, T[i] / T_max * L)
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(rect)
        ax.add_patch(sector)

        phase2.set_data(phi1[i] + phi2[i], om1[i] + om2[i])

        return line1, line2, line3, time_text, rect, phase2, sector

    anim = FuncAnimation(fig, update, sim.n_frames + 1, interval=20, blit=True, init_func=init, repeat_delay=5000)
    fig.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)

    if save == "save":
        anim.save('Pendule_entraine_4.html', fps=30)
    elif save == "gif":
        anim.save('./driven_pendulum.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 20.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./driven_pendulum.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def path_driven_pendulum(sim, time_series):

    t, phi1, om1, phi2, om2 = time_series
    x1, y1, x2, y2, v2, ac2, T = driven_pendulum_kinematics(sim, time_series)

    params1 = array([sim.l1, sim.l2])
    params2 = array([sim.phi1d, sim.phi2d, sim.om1, sim.om2])
    dcm1, dcm2 = 3, 4
    fmt1, fmt2 = countDigits(amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val2 in params2:
        fmt2 = max(fmt2, countDigits(val2) + 1 + dcm2)

    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l1, width=fmt1, dcm=dcm1),
        r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l2, width=fmt1, dcm=dcm1),
        "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi1d, width=fmt2, dcm=dcm2),
        r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi2d, width=fmt2, dcm=dcm2),
        r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om1, width=fmt2, dcm=dcm2),
        r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om2, width=fmt2, dcm=dcm2)
    ]

    see_path_1(1, array([x2, y2]), v2, color='Blues', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)

    parameters[0] = r"Axe x : $\varphi_1 + \varphi_2$"
    parameters[1] = r"Axe y : $\omega_1 + \omega_2$"
    parameters[2] = r"Axe c : $v_2 \; \sqrt{a_2}$"
    see_path_1(1, array([phi1+phi2, (om1+om2)]), -sqrt(ac2)*v2, color='inferno', var_case=2, shift=(-0., 0.), save="no", displayedInfo=parameters)
    
    return


def load_configuration(i):
    # resonnance at om1 = sqrt(g / l2)

    g, D, m = 9.81, 0.0, 1.0
    phi1, phi2, om2 = 0., 0., 0.
    if i == 1:
        l1, l2 = 0.1, 0.1
        om1 = 0.5 * sqrt(g / l2)
    elif i == 2:
        l1, l2 = 0.1, 0.1
        om1 = 0.4 * sqrt(g / 0.1)
    elif i == 3:
        l1, l2 = 0.2, 0.2 / sqrt(2)
        om1 = 0.4 * sqrt(g / l2)
    elif 4 <= i <= 10:
        l1, l2 = 0.1, 0.4
        c = [2/3, 1/2, -0.333, 3/5, 1., 9/17, 17/37]
        om1 = sqrt(g / l2) * c[i-4]
    else:
        raise ValueError("Invalid configuration number.")

    params = {'g': g, 'l1': l1, 'l2': l2, 'D': D, 'm': m}
    initials = {'phi1': phi1, 'phi2': phi2, 'om1': om1, 'om2': om2}
    return params, initials
    

if __name__ == "__main__":

    params = {
        'g': 9.81, 'l1': 0.1, 'l2': 0.4, 
        'D': 0.0, 'm': 1
    }
    
    initials = {
        'phi1': 0, 
        'phi2': 0, 
        'om1': -0.333 * sqrt(params['g'] / 0.4),
        'om2': 0
    }

    setup = {
        't_sim': 60., 'fps': 30, 'slowdown': 1.0, 'oversample': 10
    }

    params, initials = load_configuration(10)

    sim = DrivenPendulum(params, initials, setup)
    time_series = driven_pendulum_ode(sim)

    # see_animation(sim, time_series)
    path_driven_pendulum(sim, time_series)
