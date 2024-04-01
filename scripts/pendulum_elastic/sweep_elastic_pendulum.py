import numpy as np
from tqdm import tqdm
from physicsim.pendulum_elastic import PendulumElastic
from utils import sweep_parameter


if __name__ == "__main__":

    n_sim = 100
    param_range = np.linspace(28., 34., n_sim)

    setup = {"t_sim": 30., "fps": 30., "slowdown": 1., "oversample": 5}
    initials = {
        "r": 0.875,
        "dr": 0., 
        "th": np.radians(10.), 
        "om": 0.
    }
    params = {
        "g": 9.81, "D": 0.,
        "l": 1., "m": 1., "k": 1.
    }
    sim = PendulumElastic(setup, params, initials)

    var_arrays = np.zeros((n_sim, 3, 1 + sim.n_steps))

    for i in tqdm(range(n_sim)):
        params["k"] = param_range[i]
        
        sim = PendulumElastic(setup, params, initials)
        sim.solve_ode(verbose=False)
        kinematics = sim.full_kinematics
        # x, y, vx, vy, speed, d2r, d2th, ddx, ddy, acc
        var_arrays[i, :, :] = kinematics[[0, 1, 4], :]

    parameters = sim.get_parameters()
    parameters[7] = ""   # remove varying parameter from the list
    idxs = [12, 13, 14, 15, 6, 7, 8]
    parameters = parameters[idxs]

    sweep_parameter.display_paths(
        var_arrays[:, 0], var_arrays[:, 1], var_arrays[:, 2],
        param_range, label=r"$l_2$", color='inferno',
        parameters=parameters,
        lw=1., var_case=1, save=None
    )
