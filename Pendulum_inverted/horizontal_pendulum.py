import sys
import os

# Add the root directory of your project to sys.path
current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import sin, cos, sqrt, pi
from scipy.integrate import odeint
from time import perf_counter
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14
#inferno, viridis, jet, magma


class HorizontalPendulum:
    def __init__(self, params, initials, setup):
        self.g, self.l, self.mp = params['g'], params['l'], params['mp']
        self.mb, self.F, self.w = params['mb'], params['F'], params['w']
        self.M = self.mp + self.mb

        self.x, self.dx = initials['x'], initials['dx']
        self.phi, self.om = initials['phi'], initials['om']
        self.phid = np.degrees(self.phi)

        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames

        return


def f(u, t_, sim):
    x_, dx_, phi_, om_ = u
    s, c = sin(phi_), cos(phi_)
    g, mp, mb, l = sim.g, sim.mp, sim.mb, sim.l
    force = sim.F * cos(sim.w * t_)
    f1 = (force + mp * s * (g * c - l * om_ * om_)) / (mb + mp * s * s)
    f2 = (f1 * c + g * s) / l
    return np.array([dx_, f1, om_, f2])


def horizontal_pendulum_ode(sim):

    t = np.linspace(0, sim.t_sim, sim.n_steps + 1)
    U0 = np.array([sim.x, sim.dx, sim.phi, sim.om])

    start = perf_counter()
    sol = odeint(f, U0, t, args=(sim,))
    end = perf_counter()

    print(f"\tElapsed time : {end-start:.3f} seconds")
    x, dx, phi, om = sol.T

    return t, x, dx, phi, om


def horizontal_pendulum_kinematics(sim, time_series):

    t, x, dx, phi, om = time_series
    l = sim.l

    xb, yb = x, np.zeros_like(t)
    xp, yp = x - l*sin(phi), l*cos(phi)

    vxp = dx - l*om*cos(phi)
    vyp = -l*om*sin(phi)
    vp  = sqrt(vxp*vxp+vyp*vyp)

    return xb, yb, xp, yp, vp


def see_animation(sim, time_series, save=""):

    t, x_full, dx_full, phi_full, om_full = time_series
    xb, yb, xp_full, yp_full, vp = horizontal_pendulum_kinematics(sim, time_series)

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, x, dx, phi, om = time_series
    xb, yb, xp, yp, vp = horizontal_pendulum_kinematics(sim, time_series)

    x_min, x_max = np.amin(xb), np.amax(xb)
    d = abs((x_min-sim.l) - (x_max+sim.l))

    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
    fig, axs = plt.subplots(1, 2, figsize=(13., 6.))
    ax, ax2 = axs
    ax2.set_xlabel(r'$\varphi \rm \; [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \rm \; [rad/s]$', fontsize=ftSz2)
    ax.grid(ls=':')
    ax2.grid(ls=':')

    tmp, = ax.plot([x_min - 1.2*sim.l, x_max + 1.2*sim.l], [-d/2, d/2])
    ax.set_aspect("equal", "datalim")
    tmp.remove()

    line1, = ax.plot([], [], 'o-', lw=2, color='C2')
    line2, = ax.plot([], [], '-', lw=1, color='grey')
    rect = plt.Rectangle((xb[0] - sim.l, -0.1 * sim.l), 2 * sim.l, 0.2 * sim.l, color='C1')
    ax.add_patch(rect)

    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.48, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge(((x_max + 1.2 * sim.l) * 0.9, -d / 2 * 0.87), d / 20, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.02, 0.96, r'$l  \: \: \: = {:.3f} \: kg$'.format(sim.l), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$m  = {:.3f} \: kg$'.format(sim.mp), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.88, r'$M  = {:.3f} \: kg$'.format(sim.mb), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.22, 0.96, r'$F   = {:.2f} \: N$'.format(sim.F), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.22, 0.92, r'$\omega  = {:.2f} \: rad/s$'.format(sim.w), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax.text(0.70, 0.96, r'$x_0 = {:.2f} $'.format(sim.x), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.70, 0.92, r'$v_0 = {:.2f} $'.format(sim.dx), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.84, 0.96, r'$\varphi_0  = {:.2f} $'.format(sim.phid), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)
    ax.text(0.84, 0.92, r'$\dot{{\varphi}}_0  = {:.2f} $'.format(sim.om), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)

    ax2.plot(phi_full, om_full, color='C2')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        rect.set_bounds(xb[0] - sim.l, -0.1 * sim.l, 2 * sim.l, 0.2 * sim.l)
        phase1.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line1, line2, rect, phase1, time_text, sector

    def update(i):
        start = max(0, i - 20000)
        thisx, thisy = [xb[i], xp[i]], [yb[i], yp[i]]

        line1.set_data(thisx, thisy)
        line2.set_data([xp_full[k*start:k*i + 1]], [yp_full[k*start:k*i + 1]])
        rect.set_x(xb[i] - sim.l)

        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)

        phase1.set_data(phi[i], om[i])

        return line1, line2, rect, phase1, time_text, sector

    anim = FuncAnimation(fig, update, sim.n_frames+1, interval=20, repeat_delay=3000, blit=True, init_func=init)
    fig.tight_layout()

    if save == "save":
        anim.save('Horizontal_Inverted_Pendulum_2.html', fps=30)
    elif save == "gif":
        anim.save('./pendulum_horizontal.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 20.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./pendulum_horizontal.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def path_horizontal_pendulum(sim, time_series):

    t, x, dx, phi, om = time_series
    xb, yb, xp, yp, vp = horizontal_pendulum_kinematics(sim, time_series)

    params1 = np.array([sim.l, sim.mp, sim.mb])
    params2 = np.array([sim.g, sim.F, sim.w])
    params3 = np.array([sim.x, sim.dx, sim.phid, sim.om])

    dcm1, dcm2, dcm3 = 3, 3, 3
    fmt1, fmt2, fmt3 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2, 1 + 1 + dcm3
    for val in params2:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)
    for val in params3:
        fmt3 = max(fmt3, countDigits(val) + 1 + dcm3)

    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$l \quad$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l, width=fmt1, dcm=dcm1),
        r"$m_p$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.mp, width=fmt1, dcm=dcm1),
        r"$m_b$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.mb, width=fmt1, dcm=dcm1),
        "",
        r"$g\,$ = {:>{width}.{dcm}f} $\rm m/s^2$".format(sim.g, width=fmt2, dcm=dcm2),
        r"$F$ = {:>{width}.{dcm}f} $\rm N$".format(sim.F, width=fmt2, dcm=dcm2),
        r"$w$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.w, width=fmt2, dcm=dcm2),
        "",
        r"$x \;\;$ = {:>{width}.{dcm}f} $\rm m$".format(sim.x, width=fmt3, dcm=dcm3),
        r"$dx$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.dx, width=fmt3, dcm=dcm3),
        r"$\varphi \,\,\,$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phid, width=fmt3, dcm=dcm3),
        r"$\omega \,\,\,$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om, width=fmt3, dcm=dcm3)
    ]
    parameters[0] = r"Axe x : $x_p^2$"
    parameters[1] = r"Axe y : $y_p$"
    parameters[2] = r"Axe c : $v_p$"

    see_path_1(1., np.array([-xp*xp, yp]), vp, color='jet', var_case=1, name='0', shift=(0., 0.), save="no", displayedInfo=parameters)

    # see_path_1(1., np.array([xp, yp]), vp, color='jet', var_case=1, name='0', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([phi, om]), vp, color='jet', var_case=2, name='1', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([phi, x]), vp, color='viridis', var_case=2, name='2', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([phi, dx]), vp, color='viridis', var_case=2, name='3', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([om, x]), vp, color='viridis', var_case=2, name='4', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([om, dx]), vp, color='viridis', var_case=2, name='5', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([x, dx]), vp, color='viridis', var_case=2, name='6', shift=(0., 0.), save="no", displayedInfo=parameters)
        
    return


def load_configuration(i):
    g = 9.81
    if 1 <= i <= 3:
        l, mp, mb, w = 0.5, 0.001, 2, 2 * pi * 10
        x, dx, phi, om = 0, 0, pi / 4, 0
        F_list = [200, 599, 802.95]
        F = F_list[i-1]
    elif i == 4:
        l, mp, mb, F, w = 0.4, 1.00, 10.0, 50, sqrt(g / (0.4) ** 2)
        x, dx, phi, om = 0, 0, pi / 4, 0  # [om,cos(phi)]
    elif i == 5:
        l, mp, mb, F, w = 0.4, 2.5, 10, 50, 2.5 * sqrt(g)
        x, dx, phi, om = 0, 0, 3 * pi / 4, 0
    elif i == 6:
        l, mp, mb, F, w = 0.5, 0.001, 10, 1000, 10 * g / 0.4
        x, dx, phi, om = 0, 0, pi / 3, 0
    elif i == 7:
        l, mp, mb, F, w = 1.0, 1.0, 5.0, 2000, 20 * sqrt(g/1.0)
        x, dx, phi, om = 0, 0, pi / 12, 0
    elif i == 8:
        l, mp, mb, F, w = 0.3, 10.0, sqrt(g / 0.3), 80, 4 * sqrt(g / 0.3 ** 2)
        x, dx, phi, om = 0, 0, 0.39 * pi, 0
    elif i == 9:
        l, mp, mb, F, w = 0.4, 10.0, 2.5, 50, sqrt(g)
        x, dx, phi, om = 0, 0, 3 * pi / 4, 0
    elif i == 10:
        l, mp, mb, F, w = 0.4, 10.0, 2.5, 50, sqrt(g)
        x, dx, phi, om = 0, 0, 0., 0
    elif i == 11:
        l, mp, mb, F, w = 1.0, 1.0, 5.0, 0., 50*sqrt(g)
        x, dx, phi, om = 0, 0, pi, 3.0
    elif i == 12:
        l, mp, mb, F, w = 0.33, 10., 3.0, 80, 40
        x, dx, phi, om = 0, 0, pi/2, 0.0
    elif i == 13:
        l, mp, mb, F, w = 0.4, 10.0, 6.0, 81, 30.
        x, dx, phi, om = 0, 0, 0.6*pi, 0
    elif i == 14:
        l, mp, mb, F, w = 0.4, 2.5, 10, 50, 2.5 * sqrt(g)
        x, dx, phi, om = 0, 0, 3 * pi / 4, 0

    params = {'g': g, 'l': l, 'mp': mp, 'mb': mb, 'F': F, 'w': w}
    print(params)
    initials = {'x': x, 'dx': dx, 'phi': phi, 'om': om}
    return params, initials


if __name__ == "__main__":
    
    params = {
        'g': 9.81, 'l': 0.4, 'mp': 10.0, 
        'mb': 10.0, 'F': 50.0, 'w': 2.5 * sqrt(9.81),
    }

    initials = {
        'x': 0.0, 'dx': 0.0, 'phi': 3 * pi / 4, 'om': 0.0,
    }

    setup = {
        't_sim': 150.0, 'fps': 30, 'slowdown': 1.0, 'oversample': 50,
    }

    params, initials = load_configuration(14)

    sim = HorizontalPendulum(params, initials, setup)
    time_series = horizontal_pendulum_ode(sim)

    # see_animation(sim, time_series, save="")
    path_horizontal_pendulum(sim, time_series)
