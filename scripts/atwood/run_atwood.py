import physicsim.pendulum_atwood as atwood
from utils.display import see_path
import numpy as np

def display(sim):
    
    t = sim.full_t
    r, dr, th, om = sim.full_series
    x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a = sim.full_kinematics

    parameters = sim.get_parameters()

    parameters[0] = r"Axe x : $\omega \, * \, r$"
    parameters[1] = r"Axe y : $r$"
    parameters[2] = r"Axe c : $v^2$"

    see_path(x2, y2, v, colors='jet', var_case=1, save='', displayedInfo=parameters)

    parameters[0] = r"Axe x : $\varphi_1$  -  $\varphi_2$"
    parameters[1] = r"Axe y : $\omega_1$  -  $\omega_2$"
    parameters[2] = r"Axe c : $\varphi_2$  -  $\varphi_1$"

    see_path(
        th, om, v, 
        colors='inferno', shifts=(0., 0.), var_case=2, 
        name='om - dr', save='', displayedInfo=parameters
    )
    see_path(
        [x2, 2*x2], [y2, 2*y2], [v, v],
        colors=['Blues', 'viridis'], var_case=2,
        save="no", displayedInfo=parameters
    )
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
        "g": 9.81, "m": 1., "M": 3.
    }

    initials = {
        "r": 1., "th": np.radians(150.), "dr": 0., "om": 0.
    }

    setup = {
        "t_sim": 50, "fps": 30., "slowdown": 1.0, "oversample": 50
    }

    params, initials = load_configuration(4)

    sim = atwood.AtwoodPendulum(setup, params, initials)
    sim.solve_ode()
    sim.animate(save="no")

    # display(sim)
