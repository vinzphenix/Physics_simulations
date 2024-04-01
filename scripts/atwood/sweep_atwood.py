import numpy as np
from tqdm import tqdm
from physicsim.pendulum_atwood import AtwoodPendulum
from utils import sweep_parameter

if __name__ == "__main__":

    n_sim = 100
    mu_range = np.linspace(1.125, 1.175, n_sim)

    setup = {"t_sim": 150., "fps": 30., "slowdown": 1., "oversample": 2}
    initials = {"r": 1., "dr": 0., "th": np.radians(150.), "om": 0.}
    params = {"g": 9.81, "m": 1., "M": mu_range[0]}
    sim = AtwoodPendulum(setup, params, initials)
    
    var_arrays = np.zeros((n_sim, 3, 1 + sim.n_steps))

    for i in tqdm(range(n_sim)):
        params = {"g": 9.81, "m": 1., "M": mu_range[i]}
        sim = AtwoodPendulum(setup, params, initials)
        sim.solve_ode(verbose=False)
        kinematics = sim.full_kinematics
        # x1, y1, x2, y2, vx, vy, v, ddr, dom, acx, acy, a
        var_arrays[i, :, :] = kinematics[[2, 3, 6], :]

    parameters = sim.get_parameters()
    parameters[5] = ""   # remove varying parameter from the list
    idxs = [10, 11, 12, 13]
    parameters = parameters[idxs]

    sweep_parameter.display_paths(
        var_arrays[:, 0], var_arrays[:, 1], var_arrays[:, 2],
        mu_range,  label=r"$\mu$", color='inferno',
        parameters=parameters,
        lw=1., var_case=1, save=None
    )
