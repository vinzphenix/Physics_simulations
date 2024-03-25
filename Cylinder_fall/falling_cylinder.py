import sys
import os
current_directory = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_directory, '..'))
sys.path.append(project_root)

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from matplotlib.animation import FuncAnimation, PillowWriter
from numpy import sin, cos, tan, hypot, radians, linspace, array, arange, amax, amin
from scipy.integrate import odeint
from time import perf_counter
from Utils.Fixed_Path import countDigits, see_path_1, see_path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['text.usetex'] = False
ftSz1, ftSz2, ftSz3 = 20, 17, 14


class Cylinder:
    def __init__(self, params, initials, setup):
        self.g, self.alpha = params['g'], params['alpha']
        self.R, self.M, self.L, self.m = params['R'], params['M'], params['L'], params['m']
        self.C, self.D1, self.D2 = params['C'], params['D1'], params['D2']
        
        self.th, self.om = initials['th'], initials['om']
        self.x, self.v = initials['x'], initials['v']

        self.alphad, self.thd = np.degrees(self.alpha), np.degrees(self.th)
        
        self.t_sim, self.fps = setup["t_sim"], setup["fps"]
        self.slowdown = setup["slowdown"]  # . < 1 : faster; . = 1 : real time; 1 < . : slow motion
        self.oversample = setup["oversample"]  # display one frame every ... frames
        self.t_anim = self.slowdown * self.t_sim
        self.n_frames = int(self.fps * self.t_anim)
        self.n_steps = self.oversample * self.n_frames

        return


def f(u, _, sim):
    g, a = sim.g, sim.alpha
    R, M, L, m = sim.R, sim.M, sim.L, sim.m
    C, D1, D2 = sim.C, sim.D1, sim.D2

    dv = (m + M) * g * sin(a) + C / R - (D1 + D2) * u[3] + M * L * u[1] * u[1] * cos(u[0]) + \
         M * sin(u[0]) * (g * cos(a + u[0]) - D2 * (L * u[3] * sin(u[0]) + L * L * u[1]))
    dom = 1 / L * (g * cos(a + u[0]) - D2 * (L * u[3] * sin(u[0]) + L * L * u[1]) + dv * sin(u[0]))

    return array([u[1], dom, u[3], dv])


def cylinder_ode(sim):

    t = linspace(0., sim.t_sim, sim.n_steps+1)
    U0 = array([np.pi/2 - sim.th - sim.alpha, sim.om, sim.x, sim.v])

    start = perf_counter()
    sol = odeint(f, U0, t, args=(sim,))
    end = perf_counter()
    print(f"\tElapsed time : {end-start:.3f} seconds")
    th, om, x, v = sol.T
    
    return t, th, om, x, v


def cylinder_kinematics(sim, time_series):
    t, th, om, x, v = time_series
    a = sim.alpha
    phi = (x - sim.x) / sim.R 

    xc, yc = x * cos(a), -x * sin(a)  # position du centre du cercle
    x1, y1 = xc + sim.R * cos(phi + a), yc - sim.R * sin(phi + a)  # position d'un point sur cercle
    x3, y3 = xc - sim.R * cos(phi + a), yc + sim.R * sin(phi + a)  # position d'un point sur cercle
    x2, y2 = xc + sim.L * cos(th + a), yc - sim.L * sin(th + a)  # position du pendule

    vx2 = +v * cos(a) - sim.L * sin(th + a) * om
    vy2 = -v * sin(a) - sim.L * cos(th + a) * om
    v2 = hypot(vx2, vy2)

    return xc, yc, x1, y1, x2, y2, x3, y3, vx2, vy2, v2



# xmin, xmax = amin(xc) - 4.0 * max(R, L), amax(xc) + 4.0 * max(R, L)
# ymin, ymax = amin(yc) - 1.5 * max(R, L), amax(yc) + 1.5 * max(R, L)
# L_X, L_Y = xmax - xmin, ymax - ymin
# #print('xmin = ', xmin) ; print('xmax = ', xmax)
# #print('ymin = ', ymin) ; print('ymax = ', ymax)

def see_animation(sim, time_series, save=""):
    
    t, th_full, om_full, x_full, v_full = time_series
    kinematics = cylinder_kinematics(sim, time_series)
    xc, yc = kinematics[0], kinematics[1]

    k, time_series = sim.oversample, list(time_series)
    for idx, series in enumerate(time_series):
        time_series[idx] = series[::k]

    t, th, om, x, v = time_series
    xc, yc, x1, y1, x2, y2, x3, y3, vx2, vy2, v2 = cylinder_kinematics(sim, time_series)

    a = sim.alpha
    xmin, xmax = amin(xc) - 4.0 * max(sim.R, sim.L), amax(xc) + 4.0 * max(sim.R, sim.L)
    ymin, ymax = amin(yc) - 1.5 * max(sim.R, sim.L), amax(yc) + 1.5 * max(sim.R, sim.L)
    L_X, L_Y = xmax - xmin, ymax - ymin

    plt.rcParams['text.usetex'] = (save == "snapshot") or (save == "gif")
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(211, xlim=(xmin, xmax), ylim=(ymin, ymax), aspect='equal')
    ax2 = fig.add_subplot(223)
    ax2.grid(ls=':')
    ax.grid(ls=':')
    ax3 = fig.add_subplot(224)
    ax3.grid(ls=':')
    ax2.set_xlabel(r'$\theta \: \rm [rad]$', fontsize=ftSz2)
    ax2.set_ylabel(r'$v \: \rm [m/s]$', fontsize=ftSz2)
    ax3.set_xlabel(r'$\omega \: \rm [rad/s]$', fontsize=ftSz2)  # ; ax3.set_ylabel(r'$v \: \rm [m/s]$')

    line1, = ax.plot([], [], 'o-', lw=2, color='grey')
    line2, = ax.plot([], [], 'o-', lw=2, color='orange')
    phase21, = ax2.plot([], [], marker='o', ms=8, color='C0')
    phase31, = ax3.plot([], [], marker='o', ms=8, color='C0')

    xd = linspace(xmin, xmax, 2)
    ax.plot(xd, yc[0] - sim.R / cos(a) - tan(a) * (xd - xc[0]))

    time_template = r'$t = {:.2f} \; s$' if save == "snapshot" else r'$t = \mathtt{{{:.2f}}} \; s$'
    time_text = ax.text(0.45, 0.88, '', fontsize=ftSz2, transform=ax.transAxes)
    sector = patches.Wedge((xmax - L_X * 0.03, ymax - 0.03 * L_X),
                           L_X * 0.02, theta1=90, theta2=90, color='lightgrey')

    circ = patches.Circle((xc[0], yc[0]), radius=sim.R, edgecolor=None, facecolor='lightgrey', lw=4)

    ax.text(0.02, 0.06, r'$\alpha  = {:.2f} \: \rm [^\circ] $'.format(sim.alphad), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)
    ax.text(0.02, 0.14, r'$\tau  = {:.2f} \: \rm [N \cdot m]$'.format(sim.C), fontsize=ftSz3, wrap=True,
            transform=ax.transAxes)
    ax.text(0.15, 0.06, r'$M  = {:.2f} \: \rm [kg]$'.format(sim.M), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.15, 0.14, r'$m  = {:.2f} \: \rm [kg]$'.format(sim.m), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.28, 0.06, r'$L  = {:.2f} \: \rm [m]$'.format(sim.L), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.28, 0.14, r'$R  = {:.2f} \: \rm [m]$'.format(sim.R), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.72, 0.92, r'$\theta_0  = {:.2f} $'.format(sim.thd), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.72, 0.84, r'$\omega_0  = {:.2f} $'.format(sim.om), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.92, r'$x_0  = {:.2f} $'.format(sim.x), fontsize=ftSz3, wrap=True, transform=ax.transAxes)
    ax.text(0.80, 0.84, r'$v_0  = {:.2f} $'.format(sim.v), fontsize=ftSz3, wrap=True, transform=ax.transAxes)

    ax2.plot(th_full, v_full, color='C1')
    #ax2.plot(om_full, v_full, color='C1')
    ax3.plot(om_full, v_full, color='C2')

    #####     ================      Animation      ================      #####

    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        phase21.set_data([], [])
        phase31.set_data([], [])
        time_text.set_text('')
        circ.center = (xc[0], yc[0])
        sector.set_theta1(90)
        return line1, line2, time_text, circ, sector, phase21, phase31

    def update(i):
        thisx1, thisx2 = [x1[i], xc[i], x3[i]], [xc[i], x2[i]]
        thisy1, thisy2 = [y1[i], yc[i], y3[i]], [yc[i], y2[i]]

        line1.set_data(thisx1, thisy1)
        line2.set_data(thisx2, thisy2)
        phase21.set_data(th[i], v[i])
        phase31.set_data(om[i], v[i])

        circ.center = (xc[i], yc[i])
        ax.add_patch(circ)

        time_text.set_text(time_template.format(t[i]))
        sector.set_theta1(90 - 360*t[i] / sim.t_sim)
        ax.add_patch(sector)

        return line1, line2, time_text, circ, sector, phase21, phase31

    anim = FuncAnimation(fig, update, sim.n_frames+1, interval=20, blit=True, init_func=init, repeat_delay=3000)
    # plt.subplots_adjust(left=0.05, right=0.95, bottom=0.08, top=0.92, wspace=None, hspace=None)
    plt.tight_layout()

    if save == "save":
        anim.save('Cylinder_Fall_2.html', fps=30)
    elif save == "gif":
        anim.save('./cylinder.gif', writer=PillowWriter(fps=20))
    elif save == "snapshot":
        t_wanted = 10.
        t_idx = np.argmin(np.abs(t - t_wanted))
        update(t_idx)
        fig.savefig("./cylinder.svg", format="svg", bbox_inches="tight")
        # plt.show()
    else:
        plt.show()


def path_cylinder(sim, time_series):

    t, th, x, om, v = time_series
    xc, yc, x1, y1, x2, y2, x3, y3, vx, vy, v = cylinder_kinematics(sim, time_series)

    params1 = array([sim.alpha, sim.R, sim.m, sim.M, sim.L])
    params2 = array([sim.th, sim.om, sim.x, sim.v])
    dcm1, dcm2 = 3, 3
    fmt1, fmt2 = 1 + 1 + dcm1, 1 + 1 + dcm2
    for val1 in params1:
        fmt1 = max(fmt1, countDigits(val1) + 1 + dcm1)
    for val2 in params2:
        fmt2 = max(fmt2, countDigits(val2) + 1 + dcm2)

    parameters = [
        r"Axe x : $x \;*\; \omega$",
        r"Axe y : $x \;*\; v$",
        r"Axe c : $\vartheta$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(sim.t_sim), "",
        r"$\alpha_{{slope}} \,\:$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.alpha, width=fmt1, dcm=dcm1),
        r"$r_{{wheel}} \:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.R, width=fmt1, dcm=dcm1),
        r"$m_{{wheel}} $ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m, width=fmt1, dcm=dcm1),
        r"$m_{{pdlm}} \,\,$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.M, width=fmt1, dcm=dcm1),
        r"$L_{{pdlm}} \;\;$ = {:>{width}.{dcm}f} $\rm m$".format(sim.L, width=fmt1, dcm=dcm1),
        "", r"$C_{{wheel}}$ = {:>6.3f} $\rm N \, m$".format(sim.C), "",
        r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\vartheta$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.thd, width=fmt2, dcm=dcm2),
        r"$\omega$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om, width=fmt2, dcm=dcm2),
        r"$x$ = {:>{width}.{dcm}f} $\rm m$".format(sim.x, width=fmt2, dcm=dcm2),
        r"$v$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.v, width=fmt2, dcm=dcm2)
    ]
    parameters[0] = r"Axe x : $\omega$"
    parameters[1] = r"Axe y : $v$"
    parameters[2] = r"Axe c : $\vartheta$"

    see_path_1(
        2, array([om*x, v*x]), th, color='Blues', shift=(0.15, -0.15),
        var_case=2, save="no", displayedInfo=parameters
    )
    return


def load_configuration(i):

    g, D1, D2 = 9.81, 0., 0.
    if i == 1:
        alpha, R, M, L, m, C = 10.0, 0.80, 1.00, 1.00, 9.5, -14.65
        th, om, x, v = 170, 0.00, 0.00, 1.00
    else:
        raise ValueError("Invalid configuration number")

    params = {
        'g': g, 'alpha': np.radians(alpha), 
        'R': R, 'M': M, 'L': L, 'm': m, 
        'C': C, 'D1': D1, 'D2': D2
    }
    initials = {
        'th': np.radians(th), 'om': om, 'x': x, 'v': v
    }
    return params, initials


if __name__ == "__main__":

    params = {
        'g': 9.81, 'alpha': 10.0,
        'R': 0.80, 'M': 1.00, 'L': 1.00, 'm': 9.5,
        'C': -14.65, 'D1': 0.0, 'D2': 0.0
    }
    initials = {
        'th': 170, 'om': 0.00, 'x': 0.00, 'v': 1.00
    }
    setup = {
        "t_sim": 27.9, "fps": 30, "slowdown": 1., "oversample": 10
    }

    params, initials = load_configuration(1)
    
    sim = Cylinder(params, initials, setup)
    time_series = cylinder_ode(sim)

    see_animation(sim, time_series, save="no")
    # path_cylinder(sim, time_series)
