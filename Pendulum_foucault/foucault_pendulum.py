import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.axis import Tick 
from time import perf_counter
from numpy import pi, radians, sin, cos, amin, amax
from scipy.integrate import odeint

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14


class FoucaultPendulum:
    def __init__(self, params, initials, setup):
        self.g, self.L, self.phi, self.W = params['g'], params['L'], params['phi'], params['W']
        self.h = self.L + 1
        self.w = 2 * pi / 86400 * self.W

        self.a, self.da = initials['alpha'], initials['da']
        self.b, self.db = initials['beta'], initials['db']
        
        self.phid, self.alphad, self.betad = np.degrees([self.phi, self.a, self.b])

        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames
        return


def f(u, _, sim):
    g, L, phi, w = sim.g, sim.L, sim.phi, sim.w
    a, da, b, db = u
    sa, ca, sb, cb = sin(a), cos(a), sin(b), cos(b)
    f1 = sa * ca * db * db - g / L * sa + 2 * w * db * (sin(phi) * sa * ca + cos(phi) * sa * sa * cb)
    f2 = (-2 * ca * da * db - 2 * w * da * (sin(phi) * ca + cos(phi) * sa * cb)) / sa
    return np.array([da, f1, db, f2])


def foucault_ode(sim):
    t = np.linspace(0., sim.t_sim, sim.n_steps+1)  # ODEINT
    U0 = np.array([sim.a, sim.da, sim.b, sim.db])

    start = perf_counter()
    sol = odeint(f, U0, t, args=(sim,))
    end = perf_counter()

    print(f"\tElapsed time : {end-start:.3f} seconds")
    a, da, b, db = sol.T

    return t, a, da, b, db


def foucault_kinematics(sim, time_series):
    t, alpha, dalpha, beta, db = time_series
    L, h = sim.L, sim.h
    x = L * sin(alpha) * cos(beta)
    y = L * sin(alpha) * sin(beta)
    z = h - L * cos(alpha)
    return x, y, z


def see_animation(sim, time_series, save=""):

    t_full, a_full, da_full, b_full, db_full = time_series
    x_full, y_full, z_full = foucault_kinematics(sim, time_series)
    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, alpha, dalpha, beta, dbeta = time_series
    x, y, z = foucault_kinematics(sim, time_series)

    x_min, x_max = amin(x), amax(x)
    y_min, y_max = amin(y), amax(y)
    z_min, z_max = amin(z), amax(z)

    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
    
    fig = plt.figure(figsize=(12., 7.))
    M = sim.L * np.amax(np.sin(alpha)) * 1.1
    ax1 = fig.add_subplot(121, xlim=(-M, M), ylim=(-M, M))
    ax2 = fig.add_subplot(243, xlim=(-M, M), ylim=(-0.1*M, 1.9*M))
    ax3 = fig.add_subplot(247, xlim=(-M, M), ylim=(-0.1*M, 1.9*M), sharex=ax2)
    all_bounds = [[-M, M, -M, M], [-M, M, -0.1*M, 1.9*M], [-M, M, -0.1*M, 1.9*M]]
    for ax, bounds in zip([ax1, ax2, ax3], all_bounds):
        tmp, = ax1.plot([-M, M], [-M, M])
        ax.set_aspect("equal", "datalim")
        tmp.remove()

    ax_a = fig.add_subplot(244)
    ax_b = fig.add_subplot(248, sharex=ax_a)
    # ax_b.set_xlabel(r"$t$", fontsize=ftSz2)
    # ax_a.set_ylabel(r"$\alpha$", fontsize=ftSz2)
    # ax_b.set_ylabel(r"$\beta$", fontsize=ftSz2)
    # if save != "gif":
    #     ax_a.xaxis.set_ticklabels([])
    #     ax_b.xaxis.set_ticklabels([])

    axs = [ax1, ax2, ax3, ax_a, ax_b]
    for ax in axs:
        ax.grid(ls=':')

    ax1.text(0.5, 0.96, 'N', fontsize=15, fontweight='bold', ha='center', wrap=True, transform=ax1.transAxes)  # Nord
    ax1.text(0.96, 0.5, 'E', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Est
    ax1.text(0.01, 0.5, 'O', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Ouest
    ax1.text(0.5, 0.01, 'S', fontsize=15, fontweight='bold', wrap=True, transform=ax1.transAxes)  # Sud

    ax2.text(0.93, 0.5, 'E', fontsize=15, fontweight='bold', wrap=True, transform=ax2.transAxes)  # Est
    ax2.text(0.03, 0.5, 'O', fontsize=15, fontweight='bold', wrap=True, transform=ax2.transAxes)  # Ouest

    ax3.text(0.93, 0.5, 'S', fontsize=15, fontweight='bold', wrap=True, transform=ax3.transAxes)  # Sud
    ax3.text(0.03, 0.5, 'N', fontsize=15, fontweight='bold', wrap=True, transform=ax3.transAxes)  # Nord

    ax1.text(0.05, 0.92, r'$\Omega = {:.0f} \; tr/j$'.format(sim.W), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.05, 0.87, r'$L = {:.0f} \; m$'.format(sim.L), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.05, 0.82, r'$\phi = {:.0f} \; °$'.format(sim.phid), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.70, 0.87, r'$\alpha_0 = {:.2f} \; °$'.format(sim.alphad), fontsize=ftSz3, transform=ax1.transAxes)
    ax1.text(0.70, 0.82, r'$\beta_0 = {:.2f} \; °$'.format(sim.betad), fontsize=ftSz3, transform=ax1.transAxes)

    time_template = r'$t \; = {:.2f} \; s$' if save == "snapshot" else r'$t \;\:\: = \mathtt{{{:.2f}}} \; s$'
    time_text = ax1.text(0.70, 0.92, "", fontsize=ftSz3, transform=ax1.transAxes)

    beta_wrapped = np.remainder(b_full + np.pi, 2 * np.pi) - np.pi
    positions = np.where(np.abs(np.diff(beta_wrapped)) > np.pi)[0] + 1
    t_with_nan = np.insert(t_full, positions, np.nan)
    beta_with_nan = np.insert(beta_wrapped, positions, np.nan)
    period = t_full[positions[0]] if len(positions) > 0 else sim.t_sim

    ax_a.plot(t_full, a_full, color='C0')
    ax_b.plot(t_with_nan, beta_with_nan, color='C1')

    ########################################################################################################

    #####     ================      Animation      ================      #####

    line1, = ax1.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax2.plot([], [], 'o-', lw=2, color='C2')
    line3, = ax3.plot([], [], 'o-', lw=2, color='C2')
    line4, = ax1.plot([], [], '-', lw=1, color='grey')
    cursor_a, = ax_a.plot([], [], 'o', markersize=5, color='C0')
    cursor_b, = ax_b.plot([], [], 'o', markersize=5, color='C1')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        time_text.set_text('')
        cursor_a.set_data([], [])
        cursor_b.set_data([], [])
        return line1, line2, line3, line4, time_text, cursor_a, cursor_b, #ax_a, ax_b

    def update(i):
        start = max((i - 1500, 0))
        thisx0, thisx1, thisx2 = [0, y[i]], [0, y[i]], [0, x[i]]
        thisy0, thisy1, thisy2 = [0, -x[i]], [sim.h, z[i]], [sim.h, z[i]]

        line1.set_data(thisx0, thisy0)
        line2.set_data(thisx1, thisy1)
        line3.set_data(thisx2, thisy2)
        line4.set_data([y_full[k*start:k*i + 1]], [-x_full[k*start:k*i + 1]])
        cursor_a.set_data(t[i], a_full[k*i])
        cursor_b.set_data(t[i], beta_wrapped[k*i])
        time_text.set_text(time_template.format(t[i]))
        # start = max(0., t[i] - period / 2.)
        # for ax_ in [ax_a, ax_b]:
        #     ax_.set_xlim([start, start + period])

        return line1, line2, line3, line4, time_text, cursor_a, cursor_b, #ax_a, ax_b

    anim = FuncAnimation(fig, update, sim.n_frames+1, interval=5, blit=(save != "gif"), init_func=init, repeat_delay=3000)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.18, hspace=0.1, top=0.98, bottom=0.09)

    if save == "save":
        anim.save('Foucault_2.html', fps=sim.fps)
    elif save == "gif":
        anim.save('./foucault_pendulum.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 20.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./foucault_pendulum.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":

    params = {'g': 9.81, 'L': 100, 'phi': np.radians(75), 'W': 1.0}
    initials = {'alpha': np.radians(80), 'da': 0, 'beta': 0, 'db': 0}
    # initials = {'alpha': 20, 'da': 0, 'beta': 90, 'db': 180}

    setup = {'t_sim': 3600, 'fps': 30, 'slowdown': 0.01, 'oversample': 10}

    sim = FoucaultPendulum(params, initials, setup)
    time_series = foucault_ode(sim)
    
    see_animation(sim, time_series, save="")
