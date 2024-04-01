import physicsim.pendulum_driven as dr_pendulum
import numpy as np
from physicsim.simulation import countDigits
from utils.display import see_path_1, see_path


def load_configuration(i):
    # resonnance at om1 = np.sqrt(g / l2)

    g = 9.81
    phi1, phi2, om2 = 0., 0., 0.
    if i == 1:
        l1, l2 = 0.1, 0.1
        om1 = 0.5 * np.sqrt(g / l2)
    elif i == 2:
        l1, l2 = 0.1, 0.1
        om1 = 0.4 * np.sqrt(g / 0.1)
    elif i == 3:
        l1, l2 = 0.2, 0.2 / np.sqrt(2)
        om1 = 0.4 * np.sqrt(g / l2)
    elif 4 <= i <= 10:
        l1, l2 = 0.1, 0.4
        c = [2/3, 1/2, -0.333, 3/5, 1., 9/17, 17/37]
        om1 = np.sqrt(g / l2) * c[i-4]
    elif i == 11:
        l1, l2 = 1.0, 0.5
        om1 = 0.25 * np.sqrt(g)
        phi2, om2 = -2.32810, 0.06747*np.sqrt(9.81)
    else:
        raise ValueError("Invalid configuration number.")

    params = {'g': g, 'l1': l1, 'l2': l2}
    initials = {'phi1': phi1, 'phi2': phi2, 'om1': om1, 'om2': om2}
    return params, initials


import matplotlib.pyplot as plt
def display(sim, wrap=False):
    
    if wrap:
        t, series, kinematics = sim.get_cut_series(0, 2)  # phi1, phi2
    else:
        t, series, kinematics = sim.full_t, sim.full_series, sim.full_kinematics
    
    phi1, om1, phi2, om2 = series
    x1, y1, x2, y2, v2, ac2, T1, T2 = kinematics

    params1 = np.array([sim.l1, sim.l2])
    params2 = np.array([sim.phi1d, sim.phi2d, sim.om1, sim.om2])
    dcm1, dcm2 = 3, 4
    fmt1, fmt2 = countDigits(np.amax(params1)) + 1 + dcm1, 1 + 1 + dcm2
    for val2 in params2:
        fmt2 = max(fmt2, countDigits(val2) + 1 + dcm2)

    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$",
        "", r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$l_1 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l1, width=fmt1, dcm=dcm1),
        r"$l_2 \;\:\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.l2, width=fmt1, dcm=dcm1),
        "", r"$g$  = {:>5.2f} $\rm m/s^2$".format(sim.g), "",
        r"$\varphi_1$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi1d, width=fmt2, dcm=dcm2),
        r"$\varphi_2$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.phi2d, width=fmt2, dcm=dcm2),
        r"$\omega_1$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om1, width=fmt2, dcm=dcm2),
        r"$\omega_2$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om2, width=fmt2, dcm=dcm2)
    ]

    see_path_1(1, np.array([x2, y2]), ac2, color='Blues', var_case=1, shift=(-0., 0.), save="no", displayedInfo=parameters)

    parameters[0] = r"Axe x : $\varphi_1 + \varphi_2$"
    parameters[1] = r"Axe y : $\omega_1 + \omega_2$"
    parameters[2] = r"Axe c : $v_2 \; \sqrt{a_2}$"
    # see_path_1(
    #     2., 
    #     np.array([phi2, om2]), 
    #     -np.sqrt(ac2)*v2, 
    #     color='inferno', var_case=2, 
    #     shift=(-0., 0.), 
    #     save="no", displayedInfo=parameters
    # )
    
    return


if __name__ == "__main__":

    params = {'g': 9.81, 'l1': 1.0, 'l2': 0.5}
    
    initials = {
        'phi1': np.radians(0.),
        'phi2': np.radians(0.),
        'om1': 0.5 / 2.0 * np.sqrt(9.81),
        'om2': 0.
    }

    setup = {
        't_sim': 2000., 'fps': 30, 'slowdown': 1.0, 'oversample': 10
    }

    params, initials = load_configuration(11)

    sim = dr_pendulum.DrivenPendulum(setup, params, initials)
    sim.solve_ode()
    # sim.animate(save="no", wrap=False)

    display(sim, wrap=True)
