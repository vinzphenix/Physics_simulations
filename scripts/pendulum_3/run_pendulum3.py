import physicsim.pendulum_3 as tp_pendulum
import numpy as np
from physicsim.simulation import countDigits
from utils.display import see_path_1, see_path
from numpy import pi, sqrt


def load_configuration(i):
    if i == 1:
        phi1, phi2, phi3, om1, om2, om3 = -0.06113, 0.42713, 2.01926, 0, 0, 0
        m1, m2, m3, l1, l2, l3 = 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    elif i == 2:
        phi1, phi2, phi3, om1, om2, om3 = -0.20813379, -0.47019033, 0.80253405, -4.0363589, 4.42470966, 8.3046730
        m1, m2, m3, l1, l2, l3 = 0.1, 0.1, 0.1, 0.15, 0.1, 0.1
    elif i == 3:
        phi1, phi2, phi3, om1, om2, om3 = -0.78539816, 0.79865905, 0.72867705, 0.74762606, 2.56473963, -2.05903234
        m1, m2, m3, l1, l2, l3 = 0.35, 0.2, 0.3, 0.3, 0.2, 0.25
    elif i == 4:
        phi1, phi2, phi3, om1, om2, om3 = -0.22395671, 0.47832902, 0.22100014, -1.47138911, 1.29229544, -0.27559337
        m1, m2, m3, l1, l2, l3 = 0.1, 0.2, 0.1, 0.15, 0.2, 0.3
    elif i == 5:
        phi1, phi2, phi3, om1, om2, om3 = 1.30564176, 1.87626915, 1.13990186, 0.75140557, 1.65979939, -2.31442362
        m1, m2, m3, l1, l2, l3 = 0.35, 0.2, 0.3, 0.3, 0.2, 0.25
    else:
        raise ValueError("Invalid configuration number")
    
    params = {'g': 9.81, 'm1': m1, 'm2': m2, 'm3': m3, 'l1': l1, 'l2': l2, 'l3': l3}
    initials = {'phi1': phi1, 'phi2': phi2, 'phi3': phi3, 'om1': om1, 'om2': om2, 'om3': om3}
    return params, initials


def display(sim, wrap=False):

    t = sim.full_t
    phi1, om1, phi2, om2, phi3, om3 = sim.full_series
    x1, y1, x2, y2, x3, y3, vx2, vy2, v2, vx3, vy3, v3 = sim.full_kinematics

    params1 = np.array([sim.l1, sim.l2, sim.l3, sim.m1, sim.m2, sim.m3])
    params2 = np.array([sim.phi1d, sim.phi2d, sim.phi3d, sim.om1, sim.om2, sim.om3])
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
        r"$l_3 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l3, width=fmt1, dcm=dcm1),
        r"$m_1$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m1, width=fmt1, dcm=dcm1),
        r"$m_2$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m2, width=fmt1, dcm=dcm1),
        r"$m_3$ = {:>{width}.{dcm}f} $\rm kg$".format(sim.m3, width=fmt1, dcm=dcm1),
        "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi1d, width=fmt2, dcm=dcm2),
        r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi2d, width=fmt2, dcm=dcm2),
        r"$\varphi_3$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi3d, width=fmt2, dcm=dcm2),
        r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om1, width=fmt2, dcm=dcm2),
        r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om2, width=fmt2, dcm=dcm2),
        r"$\omega_3$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om3, width=fmt2, dcm=dcm2),
    ]

    # see_path_1(1, np.array([x2, y2]), v2, color='jet', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)
    see_path_1(1, np.array([x3, y3]), v3, color='viridis', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)
    return

if __name__ == "__main__":

    params = {
        'g': 9.81,
        'm1': 0.1, 'm2': 0.1, 'm3': 0.1,
        'l1': 0.2, 'l2': 0.2, 'l3': 0.2
    }

    initials = {
        'phi1': -0.4, 'phi2': -0.9, 'phi3': -3,
        'om1': 0, 'om2': 0, 'om3': 0
    }

    setup = {
        't_sim': 20., 'fps': 30, 'slowdown': 1., 'oversample': 10
    }

    params, initials = load_configuration(1)

    sim = tp_pendulum.TriplePendulum(setup, params, initials)
    sim.solve_ode()
    sim.animate(figsize=(13., 6.), save="no")

    # display(sim)
