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
from numpy import sin, cos, radians
from time import perf_counter
from Utils.Fixed_Path import see_path_1

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 18, 14


class Pendulum:
    def __init__(self, params, initials, setup):
        self.g, self.D = params["g"], params["D"]
        self.m, self.l = params["m"], params["l"]

        self.phi, self.om = initials["phi"], initials["om"]
        self.phid = np.degrees(self.phi)

        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames

        return


def f(u, _, sim):
    return np.array([u[1], - sim.g / sim.l * sin(u[0]) - sim.D / sim.m * u[1]])


def pendulum_ode(sim):
    
    t, dt = np.linspace(0., sim.t_sim, sim.n_steps+1, retstep=True)  # ODEINT

    U = np.zeros((sim.n_steps+1, 2))
    U[0] = np.array([sim.phi, sim.om])

    start = perf_counter()
    for i in range(sim.n_steps):
        K1 = f(U[i], t[i], sim)
        K2 = f(U[i] + K1 * dt / 2, t[i] + dt / 2, sim)
        K3 = f(U[i] + K2 * dt / 2, t[i] + dt / 2, sim)
        K4 = f(U[i] + K3 * dt, t[i] + dt, sim)
        U[i + 1] = U[i] + dt * (K1 + 2 * K2 + 2 * K3 + K4) / 6
    end = perf_counter()

    print(f"\tElapsed time : {end-start:.3f} seconds")
    phi, om = U.T

    return t, phi, om


def pendulum_kinematics(sim, time_series):
    t, phi, om = time_series
    x, vx = sim.l * sin(phi), sim.l * om * cos(phi)
    y, vy = -sim.l * cos(phi), sim.l * om * sin(phi)
    speed = np.hypot(vx, vy)
    return x, y, speed


def see_animation(sim, time_series, save=""):

    t, phi_full, om_ful = time_series
    x_full, y_full, speed_full = pendulum_kinematics(sim, time_series)    
    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, phi, om = time_series
    x, y, speed = pendulum_kinematics(sim, time_series)

    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6.))

    ax = axs[0]
    tmp, = ax.plot([-sim.l * 1.15, sim.l * 1.15], [-1.1 * sim.l, 1.2 * sim.l])
    ax.set_aspect("equal")
    tmp.remove()
    ax.grid(ls=':')

    ax2 = axs[1]
    ax2.grid(ls=':')
    ax2.set_xlabel(r'$\varphi \rm \; [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \rm \; [rad/s]$', fontsize=ftSz2)

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], '-', lw=1, color='grey')
    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.40, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((1 * sim.l, -0.95 * sim.l), sim.l / 10, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.05, 0.95, r'$L  = {:.2f} \; m$'.format(sim.l), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.05, 0.90, r'$m  = {:.2f} \; kg$'.format(sim.m), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.75, 0.95, r'$\varphi_1  = {:.2f} $'.format(sim.phid), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.75, 0.90, r'$\omega_1  = {:.2f} $'.format(sim.om), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax2.plot(phi_full, om_ful, color='C1', label='pendule inertie')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        phase1.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)
        return line1, line2, phase1, time_text, sector

    def update(i):
        start = max(0, i - 100)
        thisx = [0, x[i]]
        thisy = [0, y[i]]
        line1.set_data(thisx, thisy)
        line2.set_data([x_full[k*start:k*i + 1]], [y_full[k*start:k*i + 1]])
        phase1.set_data(phi[i], om[i])
        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)

        return line1, line2, phase1, time_text, sector

    anim = FuncAnimation(fig, update, sim.n_frames+1, interval=20, blit=True, init_func=init, repeat_delay=3000)
    fig.tight_layout()

    if save == "save":
        anim.save('Pendule_simple_1', fps=30)
    elif save == "gif":
        anim.save('./simple_pendulum.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 4.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./simple_pendulum.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    params = {"g": 9.81, "D": 0.1, "m": 1., "l": 1.}
    initials = {"phi": radians(179.), "om": 0.}
    setup = {"t_sim": 10., "fps": 30, "slowdown": 1., "oversample": 1}

    sim = Pendulum(params, initials, setup)
    time_series = pendulum_ode(sim)

    see_animation(sim, time_series)
