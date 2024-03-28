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
from numpy import sin, cos, sqrt, pi, hypot
from scipy.integrate import odeint
from time import perf_counter
from Utils.Fixed_Path import countDigits, see_path_1, see_path
from Utils.dialog_box import ask


class DoublePendulum:
    def __init__(self, params, initials, setup):
        self.g = params.get('g', 9.81)
        if 'l1' in params and 'l2' in params and 'm1' in params and 'm2' in params:
            flag_dim = True
        elif 'lambda' in params and 'mu' in params:
            flag_dim = False
        else:
            raise ValueError("Invalid parameters")

        if flag_dim:
            self.l1, self.l2 = params['l1'], params['l2']
            self.m1, self.m2 = params['m1'], params['m2']
            self.l, self.mu = self.l2 / self.l1, self.m2 / (self.m1 + self.m2)
            self.ref_time = np.sqrt(self.l1 / self.g)
        else:
            self.l, self.mu = params['lambda'], params['mu']
            self.l1, self.l2 = 1., 1. * self.l
            self.m1, self.m2 = 1., 1. * self.mu / (1 - self.mu)
            self.ref_time = 1.0
        
        self.L = self.l1 + self.l2

        self.phi1, self.phi2 = initials['phi1'], initials['phi2']
        self.om1, self.om2 = initials['om1'], initials['om2']
        
        self.om1a, self.om2a = self.om1 * self.ref_time, self.om2 * self.ref_time
        self.phi1d, self.phi2d = np.degrees(self.phi1), np.degrees(self.phi2)
        self.om1d, self.om2d = self.om1a * np.sqrt(self.g / self.l1), self.om2a * np.sqrt(self.g / self.l1)

        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames
        
        return


def dynamics(t, u, l, mu):
    th1, w1, th2, w2 = u[0], u[1], u[2], u[3]
    Cos, Sin = cos(th1 - th2), sin(th1 - th2)
    f1 = (mu * Cos * (sin(th2) - w1 * w1 * Sin) - l * mu * w2 * w2 * Sin - sin(th1)) / (1. - mu * Cos * Cos)
    f2 = Sin * (cos(th1) + w1 * w1 + l * mu * w2 * w2 * Cos) / (l - l * mu * Cos * Cos)
    return np.array([w1, f1, w2, f2])


def double_pendulum_ode(sim):
    
    t = np.linspace(0., sim.t_sim / sim.ref_time, sim.n_steps+1)
    U0 = np.array([sim.phi1, sim.om1a, sim.phi2, sim.om2a])

    start = perf_counter()
    sol = odeint(dynamics, U0, t, args=(sim.l, sim.mu), tfirst=True)
    end = perf_counter()

    print(f"\tElapsed time : {end-start:.3f} seconds")
    phi1, om1, phi2, om2 = sol.T

    return t*sim.ref_time, phi1, om1/sim.ref_time, phi2, om2/sim.ref_time


def double_pendulum_kinematics(sim, time_series, extra=False):
    l1, l2 = sim.l1, sim.l2
    m1, m2 = sim.m1, sim.m2
    g, M = sim.g, m1 + m2
    t, phi1, om1, phi2, om2 = time_series

    x1, y1 = l1 * sin(phi1), -l1 * (cos(phi1))
    x2, y2 = x1 + l2 * sin(phi2), y1 - l2 * cos(phi2)

    vx1, vy1 = l1 * om1 * cos(phi1), l1 * om1 * sin(phi1)
    vx2 = l1 * om1 * cos(phi1) + l2 * om2 * cos(phi2)
    vy2 = l1 * om1 * sin(phi1) + l2 * om2 * sin(phi2)
    v1, v2 = hypot(vx1, vy1), hypot(vx2, vy2)

    _, dom1, _, dom2 = dynamics(0., [phi1, om1, phi2, om2], sim.l, sim.mu)
    acx2 = l1 * dom1 * cos(phi1) - l1 * om1 * om1 * sin(phi1) + l2 * dom2 * cos(phi2) - l2 * om2 * om2 * sin(phi2)
    acy2 = l1 * dom1 * sin(phi1) + l1 * om1 * om1 * cos(phi1) + l2 * dom2 * sin(phi2) + l2 * om2 * om2 * cos(phi2)
    ac2 = hypot(acx2, acy2)

    if extra:
        return x1, y1, v1, x2, y2, v2, ac2, vx2, vy2, acx2, acy2
    else:
        return x1, y1, v1, x2, y2, v2, ac2


def wrapped_angles(time_series):
    new_time_series = [None] * len(time_series)
    time_series[1][:] = np.remainder(time_series[1] + np.pi, 2 * np.pi) - np.pi  # phi1
    time_series[3][:] = np.remainder(time_series[3] + np.pi, 2 * np.pi) - np.pi  # phi2
    # insert np.nan to break the line when the angle goes from -pi to pi
    idxs = np.where(np.abs(np.diff(time_series[1])) > 3)[0] + 1
    for k in range(len(new_time_series)):
        new_time_series[k] = np.insert(time_series[k], idxs, np.nan)
    idxs = np.where(np.abs(np.diff(new_time_series[3])) > 3)[0] + 1
    for k in range(len(new_time_series)):
        new_time_series[k] = np.insert(new_time_series[k], idxs, np.nan)
    return new_time_series


def see_animation(sim, time_series, size=(8.4, 4.8), save="", show_v=False, show_a=False, wrap=False):
    
    t_full, phi1_f, om1_f, phi2_f, om2_f = time_series
    x1, y1, v1, x2_full, y2_full, v2, a2 = double_pendulum_kinematics(sim, time_series)

    if wrap:
        wrapped_time_series = wrapped_angles(time_series)
        _, phi1_plot, om1_plot, phi2_plot, om2_plot = wrapped_time_series
    else:
        _, phi1_plot, om1_plot, phi2_plot, om2_plot = time_series

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]
    
    t, phi1, om1, phi2, om2 = time_series
    x1, y1, v1, x2, y2, v2, a2, vx2, vy2, acx2, acy2 = double_pendulum_kinematics(sim, time_series, extra=True)

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['text.usetex'] = False
    ftSz1, ftSz2, ftSz3 = 20, 17, 12
    scale = size[0] / 14

    fig, axs = plt.subplots(1, 2, figsize=size)
    ax, ax2 = axs
    ax.grid(ls=':')
    ax2.grid(ls=':')
    ax2.set_xlabel(r'$\varphi \; \rm [\:rad\:]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$\omega \;\rm [\:rad/s\:]$', fontsize=ftSz2)
    
    tmp, = ax.plot([-sim.L*1.1, sim.L*1.1], [-1.1*sim.L, 1.1*sim.L])
    ax.set_aspect('equal', 'datalim')
    tmp.remove()

    line1, = ax.plot([], [], 'o-', lw=2, color='C1')
    line2, = ax.plot([], [], 'o-', lw=2, color='C2')
    line3, = ax.plot([], [], '-', lw=1, color='grey')

    phase1, = ax2.plot([], [], marker='o', ms=8 * scale, color='C0')
    phase2, = ax2.plot([], [], marker='o', ms=8 * scale, color='C0')

    time_template = r'$t = %.1fs$'
    time_text = ax.text(0.42, 0.94, '', fontsize=ftSz2 * scale, transform=ax.transAxes)
    sector = patches.Wedge((sim.L, -sim.L), sim.L / 15, theta1=90, theta2=90, color='lightgrey')

    ax.text(0.02, 0.96, r'$l_1  = {:.2f} \: \rm m$'.format(sim.l1), fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.02, 0.92, r'$l_2  = {:.2f} \: \rm m$'.format(sim.l2), fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.96, r'$m_1  = {:.2f} \: \rm kg$'.format(sim.m1), fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.18, 0.92, r'$m_2  = {:.2f} \: \rm kg$'.format(sim.m2), fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)

    ax.text(0.61, 0.96, r'$\varphi_1  = {:.2f} $'.format(sim.phi1d), 
            fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.61, 0.92, r'$\varphi_2  = {:.2f} $'.format(sim.phi2d), 
            fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.81, 0.96, r'$\omega_1  = {:.2f} $'.format(sim.om1), 
            fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)
    ax.text(0.81, 0.92, r'$\omega_2  = {:.2f} $'.format(sim.om2), 
            fontsize=ftSz3 * scale, wrap=True, transform=ax.transAxes)


    v_max, a_max = np.amax(v2), np.amax(a2)
    scale_v, scale_a = 2 * sim.L / (3 * v_max), 2 * sim.L / (3 * a_max)
    if show_v:
        arrow_v = ax.arrow([], [], [], [], color='C3', edgecolor=None, width=sim.L / 50)
    if show_a:
        arrow_a = ax.arrow([], [], [], [], color='C4', edgecolor=None, width=sim.L / 50)

    ax2.plot(phi1_plot, om1_plot, color='C1', ls='-', marker='', markersize=0.1)
    ax2.plot(phi2_plot, om2_plot, color='C2', ls='-', marker='', markersize=0.1)

    #####     ================      Animation      ================      #####

    def init():
        for line in [line1, line2, line3]:
            line.set_data([], [])
        for phase in [phase1, phase2]:
            phase.set_data([], [])
        time_text.set_text('')
        sector.set_theta1(90)

        res = [line1, line2, line3, phase1, time_text, phase2, sector]
        if show_v:
            arrow_v.set_data(x=x2[0], y=y2[0], dx=vx2[0] * scale_v, dy=vy2[0] * scale_v)
            res.append(arrow_v)
        if show_a:
            arrow_a.set_data(x=x2[0], y=y2[0], dx=acx2[0] * scale_a, dy=acy2[0] * scale_a)
            res.append(arrow_a)

        return tuple(res)

    def update(i):
        start = max(0, i - 250000)
        thisx, thisy = [0, x1[i]], [0, y1[i]]
        thisx2, thisy2 = [x1[i], x2[i]], [y1[i], y2[i]]

        line1.set_data(thisx, thisy)
        line2.set_data(thisx2, thisy2)
        line3.set_data([x2_full[k*start:k*i + 1]], [y2_full[k*start:k*i + 1]])
        phase1.set_data(phi1[i], om1[i])
        phase2.set_data(phi2[i], om2[i])

        time_text.set_text(time_template % (t[i]))
        sector.set_theta1(90 - 360 * t[i] / sim.t_sim)
        ax.add_patch(sector)

        res = [line1, line2, line3, phase1, time_text, phase2, sector]
        if show_v:
            arrow_v.set_data(x=x2[i], y=y2[i], dx=vx2[i] * scale_v, dy=vy2[i] * scale_v)
            res.append(arrow_v)
        if show_a:
            arrow_a.set_data(x=x2[i], y=y2[i], dx=acx2[i] * scale_a, dy=acy2[i] * scale_a)
            res.append(arrow_a)

        return tuple(res)

    def onThisClick(event):
        if event.button == 3:
            anim.event_source.stop()
            plt.close(fig)
        return

    fig.canvas.mpl_connect('button_press_event', onThisClick)
    anim = FuncAnimation(fig, update, sim.n_frames+1, interval=20, blit=True, init_func=init, repeat_delay=3000)
    # plt.subplots_adjust(left=0.05, bottom=0.08, right=0.95, top=0.92, wspace=None, hspace=None)
    plt.tight_layout()

    if save == "save":
        anim.save('double_pendulum_1.html', fps=30)
    elif save == "gif":
        anim.save('./animations/trajectory.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 20.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./figures/trajectory.svg", format="svg", bbox_inches="tight")
    else:
        plt.show()


def path_double_pendulum(sim, time_series, wrap=False):

    if wrap:
        time_series = wrapped_angles(time_series)

    t, phi1, om1, phi2, om2 = time_series
    x1, y1, v1, x2, y2, v2, ac2 = double_pendulum_kinematics(sim, time_series)

    params1 = np.array([sim.l1, sim.l2, sim.m1, sim.m2])
    params2 = np.array([sim.phi1d, sim.phi2d, sim.om1, sim.om2])
    dcm1, dcm2 = 3, 4
    fmt1, fmt2 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val in params2:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l1, width=fmt1, dcm=dcm1),
        r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l2, width=fmt1, dcm=dcm1),
        r"$m_1$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m1, width=fmt1, dcm=dcm1),
        r"$m_2$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m2, width=fmt1, dcm=dcm1),
        "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi1d, width=fmt2, dcm=dcm2),
        r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi2d, width=fmt2, dcm=dcm2),
        r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om1, width=fmt2, dcm=dcm2),
        r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om2, width=fmt2, dcm=dcm2),
    ]

    parameters[0] = r"Axe x : $x_2$"
    parameters[1] = r"Axe y : $y_2$"
    parameters[2] = r"Axe c : $v_2$"
    see_path_1(1, np.array([x2, y2]), ac2, color='viridis', var_case=1,  shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1., np.array([phi1, om2]), v1, color='inferno', shift=(0., -0.), var_case=2, save="no", displayedInfo=parameters, name='1')
    
    parameters[0] = r"Axe x : $\varphi_1 $  -  $\varphi_2$"
    parameters[1] = r"Axe y : $\omega_1$  -  $\omega_2$"
    parameters[2] = r"Axe c : $\varphi_2$  -  $\varphi_1$"
    # see_path(1, [np.array([phi1, om1]), np.array([phi2, om2])],
    #         [v1*v2, v1*v2],
    #         #["YlOrRd", "YlGnBu"],
    #         ["inferno", "inferno"],
    #         [(-0., 0.), (-0., 0.)],
    #         var_case=2, save="no", displayedInfo=parameters
    # ) 
    return


def load_configuration(i):
    # init_data = phi1, om1, phi2, om2

    if i == 1:
        l, mu = 0.25, 0.75
        init_data = 0.000, 0.732, -0.017, -0.136

    elif i == 2:
        l, mu = 0.5, 0.1
        init_data = 0.00000, 0.12762, -2.49751, -0.23353

    elif i == 3:
        l, mu = 0.5, 0.5
        init_data = 0.000, 0.901, 0.571, 0.262

    elif i == 4:
        l, mu = 0.8, 0.1
        init_data = 0.00631, 0.03685, 0.00000, 2.85528

    elif 5 <= i <= 8:
        l, mu = 1., 0.05
        u_zero_list = [
            [2 * pi / 9, 0, 2 * pi / 9 * sqrt(2), 0],  # loop
            [0.000, 0.304, -0.209, -3.160],
            [0.000, 0.617, 1.488, -0.791],
            [0.000, 0.588, 1.450, -1.223],
        ]
        init_data = u_zero_list[i - 5]
        # [0.000, 0.642, -0.485, 0.796],  # same loop trajectory, different initial conditions
        # [-0.31876, 0.54539, 0., 1.08226],  
        # [-0.68847079, -0.00158915, -0.9887163, 0.00234737],

    elif i == 9:
        l, mu = 1., 0.2
        init_data = -0.72788, -0.47704, 0.00000, 1.30007

    elif i == 10:
        l, mu = 1., 0.5
        init_data = -0.707, 0.091, 0.000, 0.611

    elif 11 <= i <= 14:
        l, mu = 2, 0.1
        u_zero_list = [
            [0.000, 0.84037, 1.72621, 0.97821],
            [0.000, 0.93306, -1.09668, 0.82402],
            [0.000, 0.92070, -2.35338, 0.14850],
            [0.000, 0.86758, -0.00314, 1.00024],
        ]
        init_data = u_zero_list[i - 11]

    elif i == 15:
        l, mu = 2., 0.3
        init_data = 0.00000, 1.32395, -1.22116, -0.09790

    elif 16 <= i <= 18:
        l, mu = 2., 0.5
        u_zero_list = [
            [0.000, 1.312, 0.840, -0.431],
            [0.000, 1.723, 1.145, -0.588],
            [0.000, 1.865, 0.008, 0.131],
        ]
        init_data = u_zero_list[i - 16]

    elif i == 19:
        l, mu = 2., 2./3.
        init_data = -1.11776, 0.85754, 0.00000, 0.80942

    elif i == 20:
        l, mu = 2., 0.8
        init_data = -0.00811, -1.28206, 0.00000, 0.93208

    elif 21 <= i <= 24:
        l, mu = 3., 1. / 3.
        u_zero_list = [
            [0.000, 1.832, -0.622, -0.550],
            [0.000, 1.644, 0.841, -0.466],
            [0.000, 1.974, 0.832, -0.558],
            [0.00000, 0.27063, -1.89320, -0.84616],
        ]
        init_data = u_zero_list[i - 21]

    elif 25 <= i <= 29:
        l, mu = 3., 0.5
        u_zero_list = [
            [0.00000, 3.18131, -0.99358, -0.65052],
            [0.00000, 3.20265, 2.65215, 0.11353],
            [0.00000, 3.50990, -2.18652, 0.65160],
            [0.00000, 3.08546, 3.14036, 0.05781],
            [0.00000, 5.12663, 2.54620, -0.54259],
        ]
        init_data = u_zero_list[i - 25]

    elif i == 30:
        l, mu = 3., 0.9
        init_data = 0.000, 3.156, 0.004, -1.074

    elif 31 <= i <= 33:
        l, mu = 5., 0.2
        u_zero_list = [
            [0.00000, 0.87736, -1.67453, -0.05226],
            [0.00000, 0.90244, 0.43123, 0.48978],
            [0.00000, 2.67628, 0.01220, -0.30299],
        ]
        init_data = u_zero_list[i - 31]

    elif i == 34:
        l, mu = 1. / 3., 0.5
        init_data = 0.00000, 0.18331, -3.01259, -4.32727
    
    elif i == 35:
        l, mu = 1. / 3., 0.5
        init_data = 0.00000, 0.24415, -2.77591, 0.05822

    elif i == 36:
        l, mu = 1., 0.1
        init_data = -0.70048, 0.92888, 0.00000, 1.96695

    elif i == 37:
        l, mu = 1., 0.9
        init_data = 0.00000, 1.24989, 1.23706, -1.05777
        # init_data = 0.00000, 1.24501, -1.25257, -1.01287  # bit more fuzzy

    elif i == 38:
        l, mu = 1., 1. / (2 + 1.)
        init_data = [0.000, np.sqrt(1/9.81), 0.000, 8.5*np.sqrt(1/9.81)]

    elif i == 39:
        l, mu = 1. / 2., 1. / (1 + 1.75)
        init_data = [0.000, -1.749 * np.sqrt(2/9.81), 0.000, 1.749 * np.sqrt(2/9.81)]

    elif i == 40:
        l, mu = 1., 1. / (1 + 10)
        init_data = [0.000, 3.5 * np.sqrt(1/9.81), 0.000, -4 * np.sqrt(2/9.81)]

    elif i == 41:
        l, mu = 1., 2. / (2. + 1.)  # fuzzy
        # l, mu = 1., 2.057 / (2.057 + 1.)  # sharp
        init_data = [0.000, 3.5 * np.sqrt(1/9.81), 0.000, -4 * np.sqrt(2/9.81)]

    elif i == 42:
        l, mu = 1.810, 1. / (12. + 1.)
        init_data = [0.698, 0.000, 0.987, 0.000]

    elif i == 43:
        l, mu = 1. / 1.1, 1. / (12. + 1.)
        init_data = [2 / 9 * pi, 0., 2 / 9 * pi * sqrt(2), 0.]
    
    elif i == 44:
        l, mu = 1., 1. / (1. + 4.)  # fuzzy
        # l, mu = 1., 1. / (1. + 3.984209)  # sharp
        init_data = [2 / 9 * pi, 0., 2 / 9 * pi * sqrt(2), 0.]

    elif i == 45:
        l, mu = np.sqrt(2), 0.5  # fuzzy
        # l, mu = 1., 0.5  # sharp
        init_data = [2 / 9 * pi, 0., 2 / 9 * pi * sqrt(2), 0.]
    
    elif i == 46:
        l, mu = 1., 0.260
        init_data = [0.00000, 0.76537, 0.00000, -0.76537]

    elif i == 47:
        l, mu = 0.5, 1. / (1. + 1.75)
        init_data = [0.00000, -0.78971, 0.00000, 0.78971]

    elif 48 <= i <= 53:
        l, mu = 1., 0.5
        u_zero_list = [
            [0.58673, 0.00000, 2.85847, 0.00000],
            [0.87511, 0.00000, -2.05532, 0.00000],
            [1.44555, 0.00000, 2.62321, 0.00000],
            [-7./36. * np.pi, 0., 4./75. * np.pi, 0.],
            [-np.pi/6., 0., np.pi/12., 0.],
            [-7./144.*np.pi, 0., 5./18.*np.pi, 0.]
        ]
        init_data = u_zero_list[i - 48]

    elif i == 54:
        l, mu = 1., 1. / (2. + 1.)
        init_data = [0., 3.5/np.sqrt(9.81), 0.00000, -5.657/np.sqrt(9.81)]

    elif i == 55:
        l, mu = 2., 0.5
        init_data = [0.00000, 1.31200, 0.84000, -0.43100]

    elif i == 56:
        l, mu = 2., 2. / (2. + 1.)
        init_data = [2*np.pi/3, -0.54, 0.00000, 0.54]


    params = {'g': 9.81, 'lambda': l, 'mu': mu}
    initials = dict(zip(['phi1', 'om1', 'phi2', 'om2'], init_data))
    return params, initials


if __name__ == "__main__":
    
    params = {
        'l1': 2.00, 'l2': 1.00, 'm1': 1.75, 'm2': 1
    }

    initials = {
        'phi1': 0., 'phi2': 0, 'om1': -1.749, 'om2': 1.749, 
    }

    setup = {
        't_sim':250.0*np.sqrt(9.81), 'fps': 30, 'slowdown': 0.15, 'oversample': 10
    }

    params, initials = load_configuration(5)

    sim = DoublePendulum(params, initials, setup)
    time_series = double_pendulum_ode(sim)

    see_animation(sim, time_series, size=(12., 6.), save="", show_v=False, show_a=False, wrap=False)
    # path_double_pendulum(sim, time_series, wrap=False)
