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
from numpy import sin, cos, linspace, array, sqrt, amax, degrees
from scipy.integrate import odeint
from time import perf_counter
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 12


class TriplePendulum:
    def __init__(self, params, initials, setup):
        self.g = params["g"]
        self.m1, self.m2, self.m3 = params["m1"], params["m2"], params["m3"]
        self.l1, self.l2, self.l3 = params["l1"], params["l2"], params["l3"]
        self.L = self.l1 + self.l2 + self.l3
        self.M = self.m1 + self.m2 + self.m3
        self.m = self.m2 + self.m3

        self.phi1, self.phi2, self.phi3 = initials["phi1"], initials["phi2"], initials["phi3"]
        self.om1, self.om2, self.om3 = initials["om1"], initials["om2"], initials["om3"]

        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames

        return
    

def f(u, _, sim):
    th1, w1, th2, w2, th3, w3 = u
    l1, l2, l3 = sim.l1, sim.l2, sim.l3
    m1, m2, m3 = sim.m1, sim.m2, sim.m3
    M, m = sim.M, sim.m
    g = sim.g

    C31, S31 = cos(th3 - th1), sin(th3 - th1)
    C32, S32 = cos(th3 - th2), sin(th3 - th2)
    C21, S21 = cos(th2 - th1), sin(th2 - th1)

    Num1 = (m3 * C31 * C32 - m * C21) / (m - m3 * C32 * C32) * (l1 * w1 * w1 * (
            -m * S21 + m3 * S31 * C32) + m3 * l2 * w2 * w2 * S32 * C32 + m3 * l3 * w3 * w3 * S32 - m * g * sin(
        th2) + m3 * g * sin(th3) * C32)
    Num2 = m * l2 * w2 * w2 * S21 + m3 * C31 * (
        g * sin(th3) + l2 * w2 * w2 * S32 + l1 * w1 * w1 * S31) + m3 * l3 * w3 * w3 * S31 - M * g * sin(th1)
    Den = l1 * (M - (m * C21 - m3 * C31 * C32) * (m * C21 - m3 * C31 * C32) / (m - m3 * C32 * C32) - m3 * C31 * C31)

    f1 = (Num1 + Num2) / Den
    f2 = (f1 * (-l1 * m * C21 + m3 * l1 * C31 * C32) + l1 * w1 * w1 * (
            -m * S21 + m3 * S31 * C32) + m3 * l2 * w2 * w2 * S32 * C32 + m3 * l3 * w3 * w3 * S32 - m * g * sin(
        th2) + m3 * g * sin(th3) * C32) / (l2 * (m - m3 * C32 * C32))
    f3 = 1 / l3 * (-g * sin(th3) - l2 * w2 * w2 * S32 - l2 * f2 * C32 - l1 * w1 * w1 * S31 - l1 * f1 * C31)

    return array([w1, f1, w2, f2, w3, f3])


def triple_pendulum_ode(sim):
    
    t = linspace(0., sim.t_sim, sim.n_steps+1)  # ODEINT
    U0 = array([sim.phi1, sim.om1, sim.phi2, sim.om2, sim.phi3, sim.om3])

    start = perf_counter()
    sol = odeint(f, U0, t, args=(sim,))
    end = perf_counter()

    print(f"\tElapsed time : {end-start:.3f} seconds")
    phi1, om1, phi2, om2, phi3, om3 = sol.T

    return t, phi1, om1, phi2, om2, phi3, om3


def triple_pendulum_kinematics(sim, time_series):
    l1, l2, l3 = sim.l1, sim.l2, sim.l3
    t, phi1, om1, phi2, om2, phi3, om3 = time_series

    x1, y1 = 0. + l1 * sin(phi1), 0. - l1 * cos(phi1)
    x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)
    x3, y3 = x2 + l3 * sin(phi3), y2 - l3 * cos(phi3)

    vx2 = -l1 * om1 * sin(phi1) - l2 * om2 * sin(phi2)
    vy2 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2)
    v2 = sqrt(vx2 * vx2 + vy2 * vy2)

    vx3 = -l1 * om1 * sin(phi1) - l2 * om2 * sin(phi2) - l3 * om3 * sin(phi3)
    vy3 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2) + l3 * om3 * cos(phi3)
    v3 = sqrt(vx3 * vx3 + vy3 * vy3)
    
    return x1, y1, x2, y2, x3, y3, vx2, vy2, v2, vx3, vy3, v3


def see_animation(sim, time_series, save=""):
    
    t, phi1_full, om1_full, phi2_full, om2_full, phi3_full, om3_full = time_series
    x1, y1, x2_full, y2_full, x3_full, y3_full, vx2, vy2, v2, vx3, vy3, v3 = triple_pendulum_kinematics(sim, time_series)
    
    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, phi1, om1, phi2, om2, phi3, om3 = time_series
    x1, y1, x2, y2, x3, y3, vx2, vy2, v2, vx3, vy3, v3 = triple_pendulum_kinematics(sim, time_series)

    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.))
    ax, ax2 = axs[0], axs[1]

    xmin, xmax = min(np.amin(x1), np.amin(x2), np.amin(x3)), max(np.amax(x1), np.amax(x2), np.amax(x3))
    ymin, ymax = -sim.L, max(np.amax(y1), np.amax(y2), np.amax(y3), 0.)
    xmax = max(-xmin, xmax)
    xmax = max(xmax, 0.5 * (ymax-ymin))
    ax.axis([-1.15 * xmax, 1.15 * xmax, ymin - 0.1 * xmax, ymin + 2.2 * xmax])
    # ax.axis([-1.1*L, 1.1*L, -1.1*L, 1.1*L])
    ax.set_aspect("equal")
    ax2.set_xlabel(r'$\varphi \rm \;[rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \rm \;[rad/s]$', fontsize=ftSz2)
    ax.grid(ls=':')
    ax2.grid(ls=':')

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], 'o-', lw=2, color='C2')
    line3, = ax.plot([], [], 'o-', lw=2, color='C3')
    line4, = ax.plot([], [], '-', lw=1, color='grey')
    line5, = ax.plot([], [], '-', lw=1, color='lightgrey')

    sector = patches.Wedge((xmax, ymin + 0.1 / 2. * xmax), xmax / 10., theta1=90, theta2=90, color='lightgrey')

    phase1, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase2, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase3, = ax2.plot([], [], marker='o', ms=8, color='C0')

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.50, 0.95, '', fontsize=ftSz2, transform=ax.transAxes, ha="center")

    ax.text(0.02, 0.96, r'$l_1  = {:.2f} \: \rm m$'.format(sim.l1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$l_2  = {:.2f} \: \rm m$'.format(sim.l2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.88, r'$l_3  = {:.2f} \: \rm m$'.format(sim.l3), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.96, r'$m_1  = {:.2f} \: \rm kg$'.format(sim.m1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.92, r'$m_2  = {:.2f} \: \rm kg$'.format(sim.m2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.88, r'$m_3  = {:.2f} \: \rm kg$'.format(sim.m3), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax.text(0.64, 0.96, r'$\varphi_1  = {:.2f}$'.format(sim.phi1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.64, 0.92, r'$\varphi_2  = {:.2f}$'.format(sim.phi2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.64, 0.88, r'$\varphi_3  = {:.2f}$'.format(sim.phi3), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.84, 0.96, r'$\omega_1  = {:.2f}$'.format(sim.om1), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.84, 0.92, r'$\omega_2  = {:.2f}$'.format(sim.om2), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.84, 0.88, r'$\omega_3  = {:.2f}$'.format(sim.om3), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax2.plot(phi1_full, om1_full, color='C1')
    ax2.plot(phi2_full, om2_full, color='C2')
    ax2.plot(phi3_full, om3_full, color='C3')

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        line3.set_data([], [])
        line4.set_data([], [])
        line5.set_data([], [])
        sector.set_theta1(90)
        phase1.set_data([], [])
        phase2.set_data([], [])
        phase3.set_data([], [])
        time_text.set_text('')
        return line1, line2, line3, line4, line5, phase1, time_text, phase2, phase3, sector

    def update(i):
        start = max(0, i - 10000)
        thisx, thisx2, thisx3 = [0, x1[i]], [x1[i], x2[i]], [x2[i], x3[i]]
        thisy, thisy2, thisy3 = [0, y1[i]], [y1[i], y2[i]], [y2[i], y3[i]]

        line1.set_data(thisx, thisy)
        line2.set_data(thisx2, thisy2)
        line3.set_data(thisx3, thisy3)
        line4.set_data([x3_full[k*start:k*i + 1]], [y3_full[k*start:k*i + 1]])
        line5.set_data([x2_full[k*start:k*i + 1]], [y2_full[k*start:k*i + 1]])

        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)
        time_text.set_text(time_template.format(t[i]))

        phase1.set_data(phi1[i], om1[i])
        phase2.set_data(phi2[i], om2[i])
        phase3.set_data(phi3[i], om3[i])

        return line1, line2, line3, line4, line5, phase1, time_text, phase2, phase3, sector

    # n //= 2 if save == "gif" else 1
    anim = FuncAnimation(fig, update, sim.n_frames+1, interval=20, blit=True, init_func=init, repeat_delay=3000)
    fig.tight_layout()

    if save == "save":
        anim.save('triple_pendulum_2.html', fps=30)
    elif save == "gif":
        anim.save('./triple_pendulum.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 20.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./triple_pendulum.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def path_triple_pendulum(sim, time_series):

    t, phi1, om1, phi2, om2, phi3, om3 = time_series
    x1, y1, x2, y2, x3, y3, vx2, vy2, v2, vx3, vy3, v3 = triple_pendulum_kinematics(sim, time_series)

    params1 = array([sim.l1, sim.l2, sim.l3, sim.m1, sim.m2, sim.m3])
    params2 = np.r_[
        np.degrees(np.array([sim.phi1, sim.phi2, sim.phi3])),
        np.array([sim.om1, sim.om2, sim.om3])
    ]
    dcm1, dcm2 = 3, 4
    fmt1, fmt2 = countDigits(amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val in params2:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

    parameters = [
        r"Axe x : $x_3$",
        r"Axe y : $y_3$",
        r"Axe c : $v_3$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l1, width=fmt1, dcm=dcm1),
        r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l2, width=fmt1, dcm=dcm1),
        r"$l_3 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l3, width=fmt1, dcm=dcm1),
        r"$m_1$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m1, width=fmt1, dcm=dcm1),
        r"$m_2$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m2, width=fmt1, dcm=dcm1),
        r"$m_3$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m3, width=fmt1, dcm=dcm1),
        "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(sim.phi1), width=fmt2, dcm=dcm2),
        r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(sim.phi2), width=fmt2, dcm=dcm2),
        r"$\varphi_3$ = {:>{width}.{dcm}f} $\rm deg$".format(degrees(sim.phi3), width=fmt2, dcm=dcm2),
        r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om1, width=fmt2, dcm=dcm2),
        r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om2, width=fmt2, dcm=dcm2),
        r"$\omega_3$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om3, width=fmt2, dcm=dcm2)
    ]

    see_path_1(1, array([x2, y2]), v2, color='jet', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, array([x3, y3]), v3, color='viridis', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)
    return 


def load_configuration(i):
    if i == 0:
        phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = -0.06113, 0.42713, 2.01926, 0, 0, 0
        m1, m2, m3, l1, l2, l3 = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    elif i == 1:
        phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = -0.20813379, -0.47019033, 0.80253405, -4.0363589, 4.42470966, 8.3046730
        m1, m2, m3, l1, l2, l3 = 0.1, 0.1, 0.1, 0.15, 0.1, 0.1
    elif i == 2:
        phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = -0.22395671, 0.47832902, 0.22100014, -1.47138911, 1.29229544, -0.27559337
        m1, m2, m3, l1, l2, l3 = 0.1, 0.2, 0.1, 0.15, 0.2, 0.3
    elif i == 3:
        phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = -0.78539816, 0.79865905, 0.72867705, 0.74762606, 2.56473963, -2.05903234
        m1, m2, m3, l1, l2, l3 = 0.35, 0.2, 0.3, 0.3, 0.2, 0.25
    elif i == 4:
        phi1_0, phi2_0, phi3_0, om1_0, om2_0, om3_0 = 1.30564176, 1.87626915, 1.13990186, 0.75140557, 1.65979939, -2.31442362
        m1, m2, m3, l1, l2, l3 = 0.35, 0.2, 0.3, 0.3, 0.2, 0.25
    else:
        raise ValueError("Invalid configuration number")
    
    params = {'g': 9.81, 'm1': m1, 'm2': m2, 'm3': m3, 'l1': l1, 'l2': l2, 'l3': l3}
    initials = {'phi1': phi1_0, 'phi2': phi2_0, 'phi3': phi3_0, 'om1': om1_0, 'om2': om2_0, 'om3': om3_0}
    return params, initials


if __name__ == "__main__":

    params = {
        'g': 9.81,
        'm1': 0.1, 'm2': 0.1, 'm3': 0.1,
        'l1': 0.2, 'l2': 0.2, 'l3': 0.2
    }

    initials = {
        'phi1': -0.20813379, 'phi2': -0.47019033, 'phi3': 0.80253405,
        'om1': -4.0363589, 'om2': 4.42470966, 'om3': 8.3046730
    }

    setup = {
        't_sim': 7., 'fps': 30, 'slowdown': 3., 'oversample': 10
    }

    params, initials = load_configuration(3)

    sim = TriplePendulum(params, initials, setup)
    time_series = triple_pendulum_ode(sim)

    see_animation(sim, time_series, save="")
    # path_triple_pendulum(sim, time_series)
