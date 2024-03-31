import physicsim.atwood_pendulum as atwood
from utils.display import countDigits, see_path_1, see_path
import numpy as np

def display(sim):
    
    t = sim.full_t
    r, dr, th, om = sim.full_series
    x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = sim.full_kinematics

    params = np.array([sim.r, sim.dr, sim.thd, sim.om])
    dcm1, dcm2 = 5, 3
    fmt2 = 1 + 1 + dcm2
    for val in params:
        fmt2 = max(fmt2, countDigits(val) + 1 + dcm2)

    parameters = [
        r"Axe x : $x_2$",
        r"Axe y : $y_2$",
        r"Axe c : $v_2$", "",
        r"$\Delta t$ = {:.2f} $\rm s$".format(t[-1]), "",
        r"$\mu$ = {:.{dcm}f}".format(sim.M / sim.m, dcm=dcm1),
        "", r"$g$ = {:.2f} $\rm m/s^2$".format(sim.g), "",
        r"$r \;\,\,$ = {:>{width}.{dcm}f} $\rm m$".format(sim.r, width=fmt2, dcm=dcm2),
        r"$dr$ = {:>{width}.{dcm}f} $\rm m/s$".format(sim.dr, width=fmt2, dcm=dcm2),
        r"$\vartheta \;\,$ = {:>{width}.{dcm}f} $\rm deg$".format(sim.thd, width=fmt2, dcm=dcm2),
        r"$\omega \;\,$ = {:>{width}.{dcm}f} $\rm rad/s$".format(sim.om, width=fmt2, dcm=dcm2)
    ]

    parameters[0] = r"Axe x : $\omega \, * \, r$"
    parameters[1] = r"Axe y : $r$"
    parameters[2] = r"Axe c : $v^2$"

    # see_path_1(lw, np.array([x2, y2]), v, color='jet', var_case=1, shift=(-0., 0.), save='save', displayedInfo=parameters)
    # see_path_1(1, np.array([th, om]), r, 'inferno', name='th - om', shift=(0., 0.), var_case=2, save='save', displayedInfo=parameters)
    # see_path_1(1., np.array([th, r]), v, 'Blues', name='th - r', shift=(0., 0.), var_case=2, save='save', displayedInfo=parameters)
    # see_path_1(1, np.array([th, dr]), v, 'inferno', name='th - dr', shift=(0.0, 0.), var_case=2, save='save', displayedInfo=parameters)
    # see_path_1(1, np.array([om, r]), v, 'inferno', name='om - r', shift=(0., 0.), var_case=2, save='save', displayedInfo=parameters)
    # see_path_1(1., np.array([om, v]), r, 'inferno', name='om - dr', shift=(0.1, 0.), var_case=4, save='no', displayedInfo=parameters)
    # see_path_1(1., np.array([r, dr]), r, 'inferno', name='r dr', shift=(0.1, -0.), var_case=2, save='save', displayedInfo=parameters)

    parameters[0] = r"Axe x : $\varphi_1 $  -  $\varphi_2$"
    parameters[1] = r"Axe y : $\omega_1$  -  $\omega_2$"
    parameters[2] = r"Axe c : $\varphi_2$  -  $\varphi_1$"
    # see_path(1, [np.array([x2, y2]), np.array([2*x2, 2*y2])],
    #          [v, v], ["Blues", "viridis"],
    #          var_case=2, save="no", displayedInfo=parameters)

    see_path_1(1, np.array([om, v]), r, 'inferno', name='om - dr', shift=(0., 0.), var_case=2, save='')
    return


def load_configuration(i):
    
    g, m = 9.81, 1.0
    r, dr, om = 1., 0., 0.  # default intials

    if 1 <= i <= 40:
        mu_list = [
            1.1185, 1.133, 1.172, 1.173, 1.278,
            1.279, 1.338, 1.548, 1.555, 1.565,
            2.000, 2.010, 2.140, 2.150, 2.165,
            2.380, 2.900, 2.945, 3.000, 3.010,
            3.125, 3.300, 3.410, 3.520, 3.867,
            4.633, 4.734, 4.737, 4.745, 4.9986,
            5.475, 5.680, 6.010, 6.013, 6.014,
            6.114, 6.806, 8.150, 19.00, 46.00,
        ]
        M = mu_list[i - 1]
        phi = 90.
    elif 41 <= i <= 57:
        mu_list = [
            1.173, 1.337, 1.655, 1.904, 2.165,
            2.394, 2.812, 3.125, 3.510, 3.520,
            4.1775, 4.745, 4.80, 6.014, 7.242,
            7.244, 16.00, 
        ]
        M = mu_list[i - 41]
        phi = 150.
    elif i == 58:
        M = 16.0
        phi = 35.0
    elif i == 59:
        M = 3.0
        r, dr, phi, om = 0.25, 0., 20.054, 2.830  # these are degrees !
    elif i == 60:
        M = 1.527
        r, dr, phi, om = 1.0, 0., 1.4 * 180 / np.pi, 0.0
    else:
        raise ValueError("Invalid configuration number")

    params = {"g": g, "m": m, "M": M}
    initials = {"r": r, "th": np.radians(phi), "dr": dr, "om": om}
    return params, initials


if __name__ == "__main__":

    params = {
        "g": 9.81, "m": 1., "M": 4.737
    }

    initials = {
        "r": 1., "th": np.radians(90.), "dr": 0., "om": 0.
    }

    setup = {
        "t_sim": 50.*np.sqrt(1.0), "fps": 30., "slowdown": 1., "oversample": 50
    }

    # params, initials = load_configuration(1)

    sim = atwood.Atwood_Pendulum(setup, params, initials)
    sim.solve_ode()
    sim.animate(save="no")

    # display(sim)
