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
ftSz1, ftSz2, ftSz3 = 20, 17, 13


class VerticalPendulum:
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
    force = sim.F * cos(sim.w * t_)
    f1 = (-sim.g * sim.mp * s * s + force - sim.mp * sim.l * om_ * om_ * c) / (sim.mb + sim.mp * c * c)
    f2 = s * (sim.g - f1) / sim.l
    return np.array([dx_, f1, om_, f2])


def vertical_pendulum_ode(sim):

    t = np.linspace(0, sim.t_sim, sim.n_steps + 1)
    U0 = np.array([sim.x, sim.dx, sim.phi, sim.om])

    start = perf_counter()
    sol = odeint(f, U0, t, args=(sim,))
    end = perf_counter()

    print(f"\tElapsed time : {end-start:.3f} seconds")
    x, dx, phi, om = sol.T

    return t, x, dx, phi, om


def vertical_pendulum_kinematics(sim, time_series):

    t, x, dx, phi, om = time_series

    xb, yb = np.zeros_like(t), -x
    xp, yp = -sim.l * sin(phi), -x + sim.l * cos(phi)

    vxp = -sim.l * om * cos(phi)
    vyp = -dx - sim.l * om * sin(phi)
    vp = sqrt(vxp * vxp + vyp * vyp)

    return xb, yb, xp, yp, vp


def see_animation(sim, time_series, save=""):

    t, x_full, dx_full, phi_full, om_full = time_series
    xb, yb, xp_full, yp_full, vp = vertical_pendulum_kinematics(sim, time_series)

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, x, dx, phi, om = time_series
    xb, yb, xp, yp, vp = vertical_pendulum_kinematics(sim, time_series)

    y_min, y_max = np.amin(yb), np.amax(yb)
    d = abs((y_min - 1.1 * sim.l) - (y_max + 1.1 * sim.l))

    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")

    fig, axs = plt.subplots(1, 2, figsize=(13., 6.))
    ax, ax2 = axs
    ax2.set_xlabel(r'$\varphi \rm \; [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \rm \; [rad/s]$', fontsize=ftSz2)
    ax.grid(ls=':')
    ax2.grid(ls=':')
    
    tmp, = ax.plot([-1.1*sim.l, 1.1*sim.l], [y_min - 1.1 * sim.l, y_max + 1.1 * sim.l])
    ax.set_aspect("equal", "datalim")
    tmp.remove()

    line1, = ax.plot([], [], 'o-', lw=2, color='C2')
    line2, = ax.plot([], [], '-', lw=1, color='grey')
    rect = plt.Rectangle((-sim.l, yb[0] - 0.1 * sim.l), 2 * sim.l, 0.2 * sim.l, color='C1')
    ax.add_patch(rect)

    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.48, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((d / 2 * 0.85, (y_min - 1.2 * sim.l) * 0.85), d / 20, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.02, 0.96, r'$l  \: \: \: = {:.3f} \: kg$'.format(sim.l), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$m  = {:.3f} \: kg$'.format(sim.mp), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.88, r'$M  = {:.3f} \: kg$'.format(sim.mb), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.21, 0.96, r'$F   = {:.2f} \: N$'.format(sim.F), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.21, 0.92, r'$\omega  = {:.2f} \: rad/s$'.format(sim.w), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax.text(0.74, 0.96, r'$x = {:.2f} $'.format(sim.x), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.74, 0.92, r'$v = {:.2f} $'.format(sim.dx), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.86, 0.96, r'$\varphi  = {:.2f} $'.format(sim.phid), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)
    ax.text(0.86, 0.92, r'$\dot{{\varphi}}  = {:.2f} $'.format(sim.om), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax2.plot(phi_full, om_full, color='C1')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        rect.set_bounds(-sim.l, yb[0] - 0.1 * sim.l, 2 * sim.l, 0.2 * sim.l)
        phase1.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line1, line2, rect, phase1, time_text, sector

    def update(i):
        start = max(0, i-50000)
        thisx = [xb[i], xp[i]]
        thisy = [yb[i], yp[i]]

        line1.set_data(thisx, thisy)
        line2.set_data([xp_full[k*start:k*i + 1]], [yp_full[k*start:k*i + 1]])
        rect.set_y(yb[i] - 0.1 * sim.l)

        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)

        phase1.set_data(phi[i], om[i])

        return line1, line2, rect, phase1, time_text, sector

    anim = FuncAnimation(fig, update, sim.n_frames+1, interval=20, repeat_delay=3000, init_func=init, blit=True)
    fig.tight_layout()

    if save == "save":
        anim.save('Vertical_Inverted_Pendulum_3.html', fps=30)
    elif save == "gif":
        anim.save('./pendulum_vertical.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 15.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./pendulum_vertical.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def path_vertical_pendulum(sim, time_series):

    t, x, dx, phi, om = time_series
    xb, yb, xp, yp, vp = vertical_pendulum_kinematics(sim, time_series)

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
    parameters[0] = r"Axe x : $x_p$"
    parameters[1] = r"Axe y : $y_p$"
    parameters[2] = r"Axe c : $v_p$"

    see_path_1(1.5, np.array([xp, yp]), vp, color='jet', var_case=1, name='0', shift=(0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, np.array([phi, om]), vp, color='jet', var_case=2, name='1', shift=(0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, np.array([phi, x]), vp, color='viridis', var_case=2, name='2', shift=(0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, np.array([phi, dx]), vp, color='viridis', var_case=2, name='3', shift=(0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, np.array([om, x]), vp, color='viridis', var_case=2, name='4', shift=(0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, np.array([om, dx]), vp, color='viridis', var_case=2, name='5', shift=(0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, np.array([x, dx]), vp, color='viridis', var_case=2, name='6', shift=(0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, np.array([om, cos(phi+pi/2)]), vp, color='inferno', var_case=2, name='7', shift=(0., 0.), save="no", displayedInfo=parameters)
    return


def load_configuration(i):
    g = 9.81
    if i == 1:
        l, mp, mb, F, w = 0.5, 1e-3, 10.0, 1e4, 10 * g / 0.5
        x, dx, phi, om = 0, 0, 0.1, 0
    elif i == 2:
        l, mp, mb, F, w = 0.4, 0.001, 5, 100, 3 * sqrt(g / 0.5)
        x, dx, phi, om = 0, 0, pi, pi
    elif i == 3:
        l, mp, mb, F, w = 0.4, 0.001, 6, 15 * sqrt(g / 0.5), 3 * sqrt(g / 0.5)
        x, dx, phi, om = 0, 0, pi, pi
    elif i == 4:
        l, mp, mb, F, w = 0.4, 0.001, 6, 100, 3 * sqrt(g / 0.4)
        x, dx, phi, om = 0, 0, pi, pi
    elif i == 5:
        l, mp, mb, F, w = 0.4, 1.00, 10.0, 200, 4 * sqrt(g / 0.5**2)
        x, dx, phi, om = 0, 0, pi / 4, 0  # [om,cos(phi)]
    elif i == 6:
        l, mp, mb, F, w = 0.4, 0.001, 6., 150, 4.8 * sqrt(g)
        x, dx, phi, om = 0, 0, pi, 1.25 * pi
    elif i == 7:
        l, mp, mb, F, w = 0.4, 0.001, 6, 100, 3 * sqrt(g / 0.5)
        x, dx, phi, om = 0, 0, pi, pi
    else:
        raise ValueError("Invalid configuration number")

    print(l, mp, mb, F, w, x, dx, phi, om)
    params = {'g': g, 'l': l, 'mp': mp, 'mb': mb, 'F': F, 'w': w}
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
        't_sim': 20.0, 'fps': 30, 'slowdown': 2.0, 'oversample': 10,
    }

    params, initials = load_configuration(6)

    sim = VerticalPendulum(params, initials, setup)
    time_series = vertical_pendulum_ode(sim)

    # see_animation(sim, time_series, save="")
    path_vertical_pendulum(sim, time_series)
