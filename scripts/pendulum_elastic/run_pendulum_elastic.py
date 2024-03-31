import physicsim.elastic_pendulum as el_pendulum
from utils.display import countDigits, see_path_1, see_path
import numpy as np


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


def display(sim):
    t = sim.full_t
    th, om, r, dr = sim.full_series
    x, y, vx, vy, speed, d2r, d2th, ddx, ddy, acc = sim.full_kinematics

    params1 = np.array([sim.l, sim.m, sim.k])
    params2 = np.array([sim.thd, sim.om, sim.r, sim.dr])

    dcm1, dcm2 = 3, 3
    fmt1, fmt2 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
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

    see_path_1(1., np.array([x, y]), -acc, color='inferno', var_case=1, name='0', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([th, om]), speed, color='inferno', var_case=2, name='1', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([th, r]), speed, color='viridis', var_case=2, name='2', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([th, dr]), speed, color='viridis', var_case=2, name='3', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([om, r]), speed, color='viridis', var_case=2, name='4', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([om, dr]), speed, color='viridis', var_case=2, name='5', shift=(0., 0.), save="no", displayedInfo=parameters)
    # see_path_1(1, np.array([r, dr]), speed, color='viridis', var_case=2, name='6', shift=(0., 0.), save="no", displayedInfo=parameters)

    # see_path_1(1, np.array([x, y]), hypot(vx, vy), 'inferno', shift=(0., 0.), var_case=1, save=False)
    # see_path_1(1, np.array([x, y]), hypot(ddx, ddy), 'inferno_r', shift=(0.1, 0.1), var_case=1, save=False)
    # see_path_1(1, np.array([th, om]), hypot(r, dr), 'inferno', var_case=2, save=False)
    return


if __name__ == "__main__":

    params = {
        "g": 9.81, "D": 0.1, 
        "l": 1., "m": 1., "k": 37.535
    }

    initials = {
        "th": np.radians(10), "om": 0., 
        "r": 0.75, "dr": 0.
    }

    setup = {
        "t_sim": 20., "fps": 30., 
        "slowdown": 1., "oversample": 10
    }

    params, initials = load_configuration(14)    
    
    sim = el_pendulum.PendulumElastic(setup, params, initials)
    sim.solve_ode()
    sim.animate(save="no", phaseSpace=0)

    # display(sim)
