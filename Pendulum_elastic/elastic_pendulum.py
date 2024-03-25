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
from numpy import sin, cos, hypot, radians, array, amax
from time import perf_counter
from scipy.integrate import odeint
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14


class ElasticPendulum():
    def __init__(self, params, initials, setup):
        self.g, self.l, self.m, self.k, self.D = params["g"], params["l"], params["m"], params["k"], params["D"]
        self.th, self.om, self.r, self.dr = initials["th"], initials["om"], initials["r"], initials["dr"]
        self.thd, self.omd = np.degrees(self.th), np.degrees(self.om)
       
        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames
        return
    

def f(u, _, sim):
    g, l, m, k, D = sim.g, sim.l, sim.m, sim.k, sim.D
    th_, om_, r_, v_ = u
    d2th_ = -2 * v_ * om_ / (r_ + l) - g * sin(th_) / (r_ + l) - D / m * (r_ + l) * om_
    d2r_ = (r_ + l) * om_ * om_ + g * cos(th_) - k / m * r_ - D / m * v_
    return array([om_, d2th_, v_, d2r_])


def elastic_pendulum_ode(sim):
    t = np.linspace(0, sim.t_sim, sim.n_steps+1)
    U0 = array([sim.th, sim.om, sim.r, sim.dr])

    start = perf_counter()
    sol = odeint(f, U0, t, args=(sim,))
    end = perf_counter()

    print(f"\tElapsed time : {end-start:.3f} seconds")
    th, om, r, dr = sol.T
    
    return t, th, om, r, dr


def elastic_pendulum_kinematics(sim, time_series):
    t, th, om, r, dr = time_series
    g, l, m, k = sim.g, sim.l, sim.m, sim.k

    x =   (r + l) * sin(th)
    y = - (r + l) * cos(th)

    vx =   dr * sin(th) + (r + l) * cos(th) * om
    vy = - dr * cos(th) + (r + l) * sin(th) * om
    speed = hypot(vx, vy)

    d2r = (r + l) * om * om + g * cos(th) - k / m * r
    d2th = -2 * dr * om / (r + l) - g * sin(th) / (r + l)
    ddx = + (d2r - (r + l) * om * om) * sin(th) + (2 * dr * om + (r + l) * d2th) * cos(th)
    ddy = - (d2r - (r + l) * om * om) * cos(th) + (2 * dr * om + (r + l) * d2th) * sin(th)
    acc = hypot(ddx, ddy)

    return x, y, vx, vy, speed, d2r, d2th, ddx, ddy, acc


def see_animation(sim, time_series, save="", phaseSpace=0):

    t, th_full, om_full, r_full, dr_full = time_series
    kinematics = elastic_pendulum_kinematics(sim, time_series)
    x_full, y_full = kinematics[0], kinematics[1]

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, th, om, r, dr = time_series
    x, y, vx, vy, speed, d2r, d2th, ddx, ddy, acc = elastic_pendulum_kinematics(sim, time_series)
    r_rest = sim.m * sim.g / sim.k

    #####     ================      Création de la figure      ================      #####

    max_x = 1.1 * amax(r + sim.l)
    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
    

    fig, axs = plt.subplots(1, 2, figsize=(13., 6.))
    ax, ax2 = axs
    ax.grid(ls=':')
    ax2.grid(ls=':')
    tmp, = ax.plot([-max_x, max_x], [-max_x, max_x], 'k-', lw=1)
    ax.set_aspect('equal', 'datalim')
    tmp.remove()

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], 'o-', lw=4, color='grey', alpha=0.3)
    line3, = ax.plot([], [], '-', lw=1, color='grey', alpha=0.8)

    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C1')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    xm, delta_x = 0.5*(xmin + xmax), xmax - xmin
    ym, delta_y = 0.5*(ymin + ymax), ymax - ymin
    delta = min(delta_x, delta_y)
    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.79, 0.94, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((xmax - 0.1 * delta, ymin + 0.1*delta), 0.04*delta, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.02, 0.86, r'$k  \,\, = {:.2f} \,N/m$'.format(sim.k), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.82, r'$\ell \:\; = {:.2f} \,m$'.format(sim.l), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.78, r'$m  = {:.2f} \,kg$'.format(sim.m), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax.text(0.02, 0.96, r'$r         = {:.2f} $'.format(sim.r), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$\dot r    = {:.2f} $'.format(sim.dr), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.17, 0.96, r'$\vartheta = {:.2f} $'.format(sim.thd), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.17, 0.92, r'$\omega    = {:.2f} $'.format(sim.om), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    # 87 90 93 96
    if phaseSpace == 0:
        ax2.plot(th_full, om_full, color='C0', label=r'$\vartheta \; :\; \omega$')
        ax2.plot(r_full - r_rest, dr_full, color='C1', label=r'$r \;\, : \; \dot r$')
        ax2.legend(fontsize=ftSz3)
    elif phaseSpace == 1:
        ax2.plot(x_full, y_full, color='C0', label='Trajectoire')
        ax2.set_aspect('equal')
    else:
        ax2.plot(th_full, r_full - r_rest, color='C1', label=r'$\vartheta / r$')
        ax2.plot(th_full, dr_full, color='C2', label=r'$\vartheta / \dot r$')
        ax2.plot(om_full, r_full - r_rest, color='C3', label=r'$\omega / r$')
        ax2.plot(om_full, dr_full, color='C4', label=r'$\omega / \dot r$')
        ax2.plot(r_full, dr_full, color='C5', label=r'$r / \dot r$')
        ax2.legend(fontsize=ftSz3)

    #####     ================      Animation      ================      #####

    def init():

        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        phase1.set_data([], [])
        phase2.set_data([], [])

        time_text.set_text('')
        sector.set_theta1(90)

        liste = [line1, line2, line3, phase1, phase2, time_text, sector]

        return tuple(liste)

    def update(i):
        start = max(0, i - 10000)
        line1.set_data([0, x[i]], [0, y[i]])
        line2.set_data([0, (sim.l + r_rest) * sin(th[i])], [0, -(sim.l + r_rest) * cos(th[i])])
        line3.set_data([x_full[k*start:k*i + 1]], [y_full[k*start:k*i + 1]])

        if phaseSpace == 0:
            phase1.set_data(th[i], om[i])
            phase2.set_data(r[i] - r_rest, dr[i])
        elif phaseSpace == 1:
            phase1.set_data(x[i], y[i])
        else:
            phase1.set_data(th[i], r[i] - r_rest)
            phase2.set_data(om[i], dr[i])

        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)

        liste = [line1, line2, line3, phase1, phase2, time_text, sector]
        return tuple(liste)

    anim = FuncAnimation(fig, update, sim.n_frames + 1, interval=10, blit=True, init_func=init, repeat_delay=3000)
    fig.tight_layout()

    if save == "save":
        anim.save('Pendule_Elastique_1', fps=30)
    elif save == "gif":
        anim.save('./elastic_pendulum.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 8.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./elastic_pendulum.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def path_elastic_pendulum(sim, time_series):

    t, th, om, r, dr = time_series
    x, y, vx, vy, speed, d2r, d2th, ddx, ddy, acc = elastic_pendulum_kinematics(sim, time_series)

    params1 = array([sim.l, sim.m, sim.k])
    params2 = array([sim.thd, sim.om, sim.r, sim.dr])

    dcm1, dcm2 = 3, 3
    fmt1, fmt2 = countDigits(amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val in params2:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

    parameters = [
        r"Axe x : $x$",
        r"Axe y : $y$",
        r"Axe c : $speed$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$l\;\;$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l, width=fmt1, dcm=dcm1),
        r"$m$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m, width=fmt1, dcm=dcm1),
        r"$k\:\:$ = {:>{width}.{dcm}f} $\rm N/m$".format(sim.k, width=fmt1, dcm=dcm1),
        "", r"$g\,$ = {:.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\vartheta $ = {:>{width}.{dcm}f} $\rm deg$".format(sim.thd, width=fmt2, dcm=dcm2),
        r"$\omega$ = {:>{width}.{dcm}f} $\rm deg/s$".format(sim.omd, width=fmt2, dcm=dcm2),
        r"$r\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.r, width=fmt2, dcm=dcm2),
        r"$\dot r\,$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.dr, width=fmt2, dcm=dcm2)
    ]
    parameters[0] = r"Axe x : $x$"
    parameters[1] = r"Axe y : $y$"
    parameters[2] = r"Axe c : $acc$"

    see_path_1(1., array([x, y]), -acc, color='inferno', var_case=1, name='0', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, array([th, om]), speed, color='inferno', var_case=2, name='1', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, array([th, r]), speed, color='viridis', var_case=2, name='2', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, array([th, dr]), speed, color='viridis', var_case=2, name='3', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, array([om, r]), speed, color='viridis', var_case=2, name='4', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, array([om, dr]), speed, color='viridis', var_case=2, name='5', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, array([r, dr]), speed, color='viridis', var_case=2, name='6', shift=(0., 0.), save="no", displayedInfo=parameters)

    # see_path_1(1, array([x, y]), hypot(vx, vy), 'inferno', shift=(0., 0.), var_case=1, save=False)
    # see_path_1(1, array([x, y]), hypot(ddx, ddy), 'inferno_r', shift=(0.1, 0.1), var_case=1, save=False)
    # see_path_1(1, array([th, om]), hypot(r, dr), 'inferno', var_case=2, save=False)
    return


def load_configuration(i):

    g = 9.81
    if i == 1:
        l, m, k = 0.7, 1.0, 100.
        th, om, r, dr = 0., 80, m * g / k, -0.6
    elif i == 2:
        l, m, k = 0.7, 1.0, 100.
        th, om, r, dr = 0., 80, 0.120, -0.7
    elif i == 3:
        l, m, k = 0.7, 1.0, 200.
        th, om, r, dr = 2., 0., 0.05, -5.0
    elif 4 <= i <= 8:
        k_list = [206., 271.88, 185.5, 147.5, 101.9]
        l, m, k = 0.5, 1.0, k_list[i-4]
        th, om, r, dr = 0., 150., 0., 1.0
    elif i == 9:
        l, m, k = 0.7, 1.0, 1000.
        th, om, r, dr = 170., 0., 0.01, 10.0
    elif i == 10:
        l, m, k = 0.7, 1.0, 98.8
        th, om, r, dr = 45, 0., 0.5, 0.
    elif i == 11:
        l, m, k = 1.0, 1.0, 101.
        th, om, r, dr = 30., 0., 0.7, 0.
    elif i == 12:
        l, m, k = 1.0, 1.0, 100.2
        th, om, r, dr = 45, 0., 0.7, 0.
    elif i == 13:
        l, m, k = 1.0, 1.0, 30.2
        th, om, r, dr = 10, 0., 0.875, 0.
    elif i == 14:
        l, m, k = 1.0, 1.0, 37.535
        th, om, r, dr = 10, 0., 0.750, 0.

    params = {"g": g, "D": 0.0, "l": l, "m": m, "k": k}
    initials = {"th": np.radians(th), "om": np.radians(om), "r": r, "dr": dr}
    return params, initials


if __name__ == "__main__":
    

    params = {
        "g": 9.81, "D": 0.1, 
        "l": 1., "m": 1., "k": 37.535
    }

    initials = {
        "th": radians(10), "om": 0., 
        "r": 0.75, "dr": 0.
    }

    setup = {
        "t_sim": 20., "fps": 30., 
        "slowdown": 1., "oversample": 10
    }

    params, initials = load_configuration(14)

    sim = ElasticPendulum(params, initials, setup)
    time_series = elastic_pendulum_ode(sim)

    # see_animation(sim, time_series, save="", phaseSpace=0)
    path_elastic_pendulum(sim, time_series)
