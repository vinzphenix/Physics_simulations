import numpy as np
from tqdm import tqdm
from physicsim.pendulum_2 import DoublePendulum
from utils import sweep_parameter

if __name__ == "__main__":

    n_sim = 100
    param_range = np.linspace(2.7, 2.9, n_sim)

    setup = {"t_sim": 100., "fps": 30., "slowdown": 1., "oversample": 5}
    initials = {
        "phi1": 2.*np.pi / 9., 
        "phi2": 2.*np.pi / 9. * np.sqrt(2), 
        "om1": 0., 
        "om2": 0.
    }
    params = {
        "g": 9.81, 
        "l1": 1., "l2": 1., 
        "m1": 4., "m2": 1.
    }
    sim = DoublePendulum(setup, params, initials)

    var_arrays = np.zeros((n_sim, 3, 1 + sim.n_steps))

    for i in tqdm(range(n_sim)):
        params["l2"] = param_range[i]
        # params["m2"] = param_range[i]
        # initials["phi2"] = param_range[i]
        
        sim = DoublePendulum(setup, params, initials)
        sim.solve_ode(verbose=False)
        kinematics = sim.full_kinematics
        # x1, y1, v1, x2, y2, v2, ac2, vx2, vy2, acx2, acy2
        var_arrays[i, :, :] = kinematics[[3, 4, 5], :]

    parameters = sim.get_parameters()
    parameters[7] = ""   # remove varying parameter from the list
    idxs = [13, 14, 15, 16, 6, 7, 8, 9]
    parameters = parameters[idxs]

    sweep_parameter.display_paths(
        var_arrays[:, 0], var_arrays[:, 1], var_arrays[:, 2],
        param_range, label=r"$l_2$", color='inferno',
        parameters=parameters,
        lw=1., var_case=1, save=None
    )
