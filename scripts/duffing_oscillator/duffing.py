import numpy as np
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter
from numpy import sin, cos
from scipy.integrate import solve_ivp
from time import perf_counter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['mathtext.rm'] = 'serif'
ftSz1, ftSz2, ftSz3 = 20, 16, 13


class Simulation:
    def __init__(self, dic_params, dic_initial, dic_setup):
        # ddy + eta dy + (+/- 1 + eps y^2) * y = cos(sigma t)  # NON dimensional form
        self.kappa, self.gamma = dic_params["kappa"], dic_params["gamma"]
        self.alpha, self.beta = dic_params["alpha"], dic_params["beta"]
        self.omega = dic_params["omega"]

        self.x, self.v = dic_initial["x"], dic_initial["v"]

        self.t_sim, self.fps = 2. * np.pi / self.omega * dic_setup["n_cycles"], dic_setup["fps"]
        self.slowdown = dic_setup["slowdown"]
        self.oversample = dic_setup["oversample"]  # display one frame every ... frames
        self.nFrames = int(self.fps * self.t_sim * self.slowdown)
        self.nSteps = self.oversample * self.nFrames
        self.transient_percentage = dic_setup["end_transient"]


def solve_equations(sim):
    def event_periodic(tau, _):
        return sin(sim.omega * tau / 2.)
        # return sin(sim.sigma * tau / 2.)

    def f(tau, u):
        x_, v_ = u
        # d2x = -sim.eta * v_ - (sim.mode + sim.eps * x_ * x_) * x_ + cos(sim.sigma * tau)
        d2x = -sim.kappa * v_ - (sim.beta + sim.alpha * x_ * x_) * x_ + sim.gamma * cos(sim.omega * tau)
        return np.array([v_, d2x])

    t = np.linspace(0, sim.t_sim, sim.nSteps + 1)
    U0 = np.array([sim.x, sim.v])

    tic = perf_counter()
    sol = solve_ivp(f, [0., sim.t_sim], U0, t_eval=t, method="RK45", rtol=1.e-8, atol=1.e-8, events=event_periodic)
    print(f"\tElapsed time : {perf_counter() - tic: .3f} seconds")

    x, v = sol.y
    t_events, y_events = sol.t_events[0], sol.y_events[0]
    return t, x, v, t_events, y_events


def see_animation(sim, time_series, save=""):
    t_full, x_full, v_full, t_events, y_events = time_series
    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series[:3]):
        time_series[idx] = series[::k]
    t, x, v = time_series[:3]

    def init():
        dot1.set_data([], [])
        dot2.set_data([], [])
        mv_dot.set_data([], [])
        points.set_data([], [])
        time_text.set_text("")
        return dot1, dot2, mv_dot, points, time_text

    def update(i):
        mv_dot.set_data(x[i], v[i])
        index = len(t_events[t_events <= t[i]])
        this_x = [y_events[index-1, 0]] if index > 0 else []
        this_v = [y_events[index-1, 1]] if index > 0 else []
        dot1.set_data(this_x, this_v)
        dot2.set_data(this_x, this_v)
        points.set_data([y_events[:index, 0]], [y_events[:index, 1]])
        time_text.set_text(time_template.format(t[i] * sim.omega / (2. * np.pi)))
        return dot1, dot2, mv_dot, points, time_text

    fig, axs = plt.subplots(1, 2, figsize=(14., 6.), sharex="all", sharey="all")
    ax1, ax2 = axs[0], axs[1]

    t_start = sim.transient_percentage * sim.t_sim
    # start_full, start = np.argmin(np.abs(t_full - t_start)), np.argmin(np.abs(t - t_start))  # don't show transient
    t_events, y_events = t_events[t_events >= t_start], y_events[t_events >= t_start]

    ax1.plot(x_full[t_full >= t_start], v_full[t_full >= t_start], ls='-', color='C0')
    ax2.plot(y_events[:, 0], y_events[:, 1], ls='', marker=".", ms=1., color='black', alpha=0.25)

    mv_dot, = ax1.plot([], [], marker='o', ms=5., color='C1')
    dot1, = ax1.plot([], [], marker='o', ms=8., color='C3')
    dot2, = ax2.plot([], [], marker='o', ms=8., color='C3')
    points, = ax2.plot([], [], ls='', marker='o', ms=1., color='black', alpha=1.)

    ode_text = r"$\ddot{x} + \kappa \dot{x} + \beta x + \alpha x^3 \,=\, \Gamma \cos{(\omega \, t)}$"
    ax2.text(0.50, 0.95, ode_text, fontsize=ftSz2, ha="center", transform=ax2.transAxes)
    time_template = r'$t = \mathtt{{{:05.2f}}} \; \frac{{2\pi}}{{\omega}}$'
    time_text = ax1.text(0.80, 0.94, '', fontsize=ftSz2, transform=ax1.transAxes)

    ax1.text(0.05, 0.92, r'$\beta  = {:.2f} $'.format(sim.beta), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.05, 0.96, r'$\alpha  = {:.2f}$'.format(sim.alpha), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.20, 0.92, r'$\Gamma  = {:.2f}$'.format(sim.gamma), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.20, 0.96, r'$\omega  = {:.2f}$'.format(sim.omega), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.35, 0.96, r'$\kappa = {:.2f} $'.format(sim.kappa), fontsize=ftSz3, transform=ax1.transAxes)

    size_x, size_v = np.amax(x) - np.amin(x), np.amax(v) - np.amin(v)
    ax1.margins(x=size_x / 50., y=size_v / 10.)
    fig.tight_layout()
    anim = FuncAnimation(fig, update, sim.nFrames + 1, interval=5., blit=True, init_func=init, repeat_delay=5000)
    if save == "mp4":
        anim.save(f"./atwood.mp4", writer=FFMpegWriter(fps=sim.fps))
    else:
        plt.show()
    return


if __name__ == "__main__":
    # initial, params = {"x": 1., "v": 0.}, {"eta": 9.81, "mode": 1., "eps": 1., "sigma": 1.}
    # ddy + eta dy + (+/- 1 + eps y^2) * y = cos(sigma t)  # NON dimensional form

    initial, params = {"x": 1., "v": 0.}, {"kappa": 0.3, "beta": -1., "alpha": 1., "gamma": 0.37, "omega": 1.25}
    # initial, params = {"x": 1., "v": 0.}, {"kappa": 0.3, "beta": -1., "alpha": 1., "gamma": 0.4, "omega": 0.5}
    # initial, params = {"x": 1., "v": 0.}, {"kappa": 0.02, "beta": 1., "alpha": 5., "gamma": 8., "omega": 0.5}

    setup = {"n_cycles": 100, "fps": 20., "oversample": 5, "slowdown": 0.2, "end_transient": 0.10}

    simulation = Simulation(params, initial, setup)
    solutions = solve_equations(simulation)
    see_animation(simulation, solutions, save="")
